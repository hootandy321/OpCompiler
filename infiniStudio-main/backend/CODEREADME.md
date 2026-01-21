# Backend 模块核心实现文档

Backend 模块是基于 Flask + SocketIO 的 Python Web 服务，为 InfiniStudio 提供完整的 RESTful API 和 WebSocket 实时通信能力。该模块实现了服务器管理、服务部署、SSH 终端、任务调度等核心功能，采用 SQLite 作为持久化存储，通过 Paramiko 实现 SSH 远程控制。

## 1. 模块结构

- **`app.py`**: 单体应用主文件，包含完整的 Flask 路由、WebSocket 事件处理、数据库操作、SSH 连接管理等所有业务逻辑（1602 行）
- **`requirements.txt`**: Python 依赖清单，定义 Flask、SocketIO、Paramiko 等核心库版本

## 2. 核心类与组件

### Flask 应用实例
- **Location**: `app.py:17-20`
- **Primary Function**: WSGI 应用入口，配置跨域策略和 WebSocket 支持
- **Key Members**:
  - `app.config['SECRET_KEY']`: Flask 会话加密密钥（需生产环境替换）
  - `socketio`: SocketIO 实例，支持 CORS 全域访问
- **Lifecycle**: 单例模式，应用启动时初始化，`debug=True` 开发模式，监听 `0.0.0.0:5000`

### 数据库管理模块
- **Location**: `app.py:32-186`
- **Primary Function**: 初始化 SQLite 数据库，创建 7 张核心业务表
- **Schema Design**:
  - `brands`: 硬件加速卡品牌（id, name, logo, sort_order）
  - `accelerator_cards`: 加速卡规格（外键关联 brand_id，存储 fp8_perf, int8_perf 等性能指标）
  - `models`: AI 模型元数据（id, name, logo, parameters）
  - `servers`: 物理服务器信息（id, host_ip, port, username, password, agent_port, status, last_check）
  - `services`: 部署服务（id, model_id, server_ids, deploy_command, deploy_status, deploy_result）
  - `chat_history`: 聊天记录（service_id, role, content, created_at）
  - `tasks`: 计划任务（name, command, server_id, schedule_type, status, result）
- **Migration Strategy**: 使用 `ALTER TABLE` 动态添加新列，捕获 `OperationalError` 实现幂等性
- **Database Path**: `../datasets/infini.db`（相对路径，自动创建目录）

### SSH 连接池管理
- **Location**: `app.py:188-190`
- **Data Structures**:
  - `ssh_connections`: 临时连接池 `{session_id: {ssh, chan, server_id, service_id, persistent}}`
  - `service_ssh_connections`: 持久化连接池 `{service_id: {ssh, chan, server_id, sessions: [session_id, ...]}}`
- **Connection Reuse**: 同一服务的多个 WebSocket 会话共享底层 SSH 连接，每个会话使用独立 channel
- **Lifecycle**: WebSocket 连接建立时创建，断开时清理（持久化连接保留供后续复用）

### Paramiko SSH 客户端封装
- **Location**: `app.py:754-800`
- **Primary Function**: 执行远程 Shell 命令，支持完整登录环境加载
- **Core Methods**:
  - `execute_ssh_command(server, command)`: 执行单条命令并返回结果
- **Implementation Details**:
  - 使用 `paramiko.SSHClient` 建立 SSHv2 连接（默认 30s 超时）
  - 命令转义策略：将 `'` 替换为 `'\''` 防止 shell 注入
  - 执行命令：`bash -l -c 'escaped_command'`（`-l` 加载登录环境，`get_pty=True` 获取伪终端）
  - 返回值：`{success: bool, output: str, error: str, exit_status: int}`
- **Error Handling**: 捕获异常返回 `success=False`，确保 SSH 连接在 `finally` 块中关闭

### 服务代理通信模块
- **Location**: `app.py:659-686`
- **Primary Function**: 通过 HTTP 调用目标服务器上的 Agent 服务（端口 8888）
- **Core Methods**:
  - `call_service_agent(server, endpoint, method, data, timeout)`: 统一接口封装
- **API Pattern**:
  - 部署：`POST http://{host_ip}:8888/service/{service_id}/deploy`
  - 重启：`POST http://{host_ip}:8888/service/{service_id}/restart`
  - 停止：`POST http://{host_ip}:8888/service/{service_id}/stop`
  - 状态：`GET http://{host_ip}:8888/service/{service_id}/status`
- **Timeout Strategy**: 默认 30 秒超时，防止长时间阻塞

### 后台部署线程
- **Location**: `app.py:802-870`
- **Primary Function**: 异步执行多服务器部署任务，更新服务状态
- **Workflow**:
  1. 更新 `deploy_status = 'deploying'`
  2. 遍历所有目标服务器，调用 `call_service_agent()` 部署
  3. 聚合所有服务器结果到 `deploy_results` 数组
  4. 更新 `deploy_status = 'online'`，保存 `deploy_result` JSON
  5. 立即调用 `update_service_status()` 同步真实状态
- **Error Recovery**: 异常时设置 `deploy_status = 'offline'`
- **Concurrency**: 使用 `threading.Thread(..., daemon=True)` 启动守护线程

### 服务状态同步机
- **Location**: `app.py:688-752`
- **Primary Function**: 定期查询 Agent 接口，将服务运行状态映射到数据库
- **Status Mapping**:
  - `agent_status = 'running'` → `deploy_status = 'deployed'`
  - `agent_status = 'stopped'` → `deploy_status = 'online'`（服务器在线但服务未运行）
  - 连接失败 → `deploy_status = 'offline'`
- **State Protection**: 如果当前状态是 `deploying`，跳过更新保持状态
- **Auto-Refresh**: `/api/services` GET 请求自动触发后台线程更新所有服务状态

### WebSocket SSH 终端
- **Location**: `app.py:1431-1597`
- **Primary Function**: 提供基于 WebSocket 的交互式 SSH 终端
- **Event Handlers**:
  - `ssh_connect`: 建立 SSH 连接，支持临时连接和持久化连接复用
  - `ssh_input`: 转发用户输入到远程 channel
  - `ssh_disconnect`: 清理会话资源（持久化连接保留）
  - `disconnect`: 全局断开处理
- **Receive Loop**: `ssh_receive_thread(session_id)` 在独立线程中循环读取 `chan.recv(4096)`，通过 `socketio.emit()` 推送输出
- **Performance**: `recv_ready()` 检查避免阻塞，`time.sleep(0.05)` 降低 CPU 占用
- **Terminal**: 使用 `xterm-256color` 终端类型，支持 ANSI 转义序列

### 服务器资源监控
- **Location**: `app.py:542-630`
- **Primary Function**: 通过 SSH 获取远程服务器 CPU、内存、磁盘使用率
- **Core Methods**:
  - `get_server_resources(server)`: 单服务器监控
  - `ping_host(host)`: ICMP 检测主机在线状态（跨平台兼容 Windows/Linux）
- **Monitoring Commands**:
  - CPU: `grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$3+$4+$5)}'`（备用 `top -bn1`）
  - Memory: `free | grep Mem | awk '{printf "%.1f", $3/$2 * 100}'`
  - Disk: `df -h / | tail -1 | awk '{print $5}' | sed 's/%//'`
- **Error Handling**: 每个指标独立 `try-except`，部分失败不影响其他指标

### 聊天完成代理
- **Location**: `app.py:1192-1272`
- **Primary Function**: 代理 OpenAI 兼容的 `/chat/completions` API 到目标服务器 8000 端口
- **Streaming Support**: 检测 `stream=True` 参数，使用 `Flask Response` 生成器转发 SSE 事件流
- **Default Model**: 请求未指定 `model` 时默认使用 `'jiuge'`
- **Timeout**: 120 秒超时，流式响应使用 `requests.stream=True`
- **Headers**: 设置 `Cache-Control: no-cache`, `Connection: keep-alive`, `X-Accel-Buffering: no`

## 3. API 接口

### 文件上传
```python
POST /api/upload
# 上传图片文件，支持 png/jpg/jpeg/gif/webp
# 返回: {"url": "/api/uploads/{uuid}.ext", "filename": "{uuid}.ext"}

GET /api/uploads/<filename>
# 返回上传的静态文件
```

### 品牌管理
```python
GET /api/brands
# 返回所有品牌，按 sort_order ASC, created_at DESC 排序

POST /api/brands
# 创建品牌，支持 JSON 和 form-data，支持 logo_file 文件上传
# 参数: {name, logo?, logo_file?}

PUT /api/brands/<int:brand_id>
# 更新品牌，同上参数

DELETE /api/brands/<int:brand_id>
# 删除品牌

POST /api/brands/reorder
# 批量更新排序
# 参数: {orders: [{id, sort_order}, ...]}
```

### 加速卡管理
```python
GET /api/brands/<int:brand_id>/accelerators
# 返回指定品牌的所有加速卡

POST /api/brands/<int:brand_id>/accelerators
# 创建加速卡
# 参数: {name, model, memory?, fp8_perf?, int8_perf?, bf16_perf?, fp16_perf?, fp32_perf?, interconnect_bandwidth?}

PUT /api/accelerators/<int:accelerator_id>
# 更新加速卡规格

DELETE /api/accelerators/<int:accelerator_id>
# 删除加速卡
```

### 模型管理
```python
GET /api/models
# 返回所有模型

POST /api/models
# 创建模型，支持 logo_file 上传
# 参数: {name, logo?, logo_file?, parameters?}

PUT /api/models/<int:model_id>
# 更新模型

DELETE /api/models/<int:model_id>
# 删除模型
```

### 服务器管理
```python
GET /api/servers
# 返回所有服务器，JOIN 品牌和加速卡表

POST /api/servers
# 创建服务器
# 参数: {name, brand_id?, model_id?, host_ip, port?, username, password?}

PUT /api/servers/<int:server_id>
# 更新服务器

DELETE /api/servers/<int:server_id>
# 删除服务器

POST /api/servers/<int:server_id>/check
# Ping 检查服务器在线状态，更新数据库

GET /api/servers/<int:server_id>/resources
# 获取单服务器资源使用率

POST /api/servers/resources
# 批量获取所有服务器资源

POST /api/servers/check-all
# 批量检查所有服务器状态
```

### 服务管理
```python
GET /api/services
# 返回所有服务，自动触发后台状态更新

POST /api/services/refresh-status
# 同步刷新所有服务状态，返回最新列表

POST /api/services
# 创建服务，有 deploy_command 时自动触发部署线程
# 参数: {name, model_id, server_ids: [], deploy_command?}

PUT /api/services/<int:service_id>
# 更新服务配置

POST /api/services/<int:service_id>/restart
# 重启服务，调用 Agent 的 /restart 接口

POST /api/services/<int:service_id>/stop
# 停止服务，调用 Agent 的 /stop 接口

GET /api/services/<int:service_id>/deploy-log
# 获取部署日志（deploy_result JSON）

DELETE /api/services/<int:service_id>
# 删除服务
```

### 聊天接口
```python
GET /api/services/<int:service_id>/chat
# 获取聊天历史，按 created_at ASC 排序

POST /api/services/<int:service_id>/chat
# 添加消息
# 参数: {role, content}

DELETE /api/services/<int:service_id>/chat
# 清空聊天历史

POST /api/services/<int:service_id>/chat/completions
# 代理 OpenAI 兼容接口，支持流式响应
# 参数: {model?, messages[], stream?, ...}
```

### 任务管理
```python
GET /api/tasks
# 返回所有任务，JOIN 服务器表

POST /api/tasks
# 创建任务
# 参数: {name, command, server_id, schedule_type}

PUT /api/tasks/<int:task_id>
# 更新任务

DELETE /api/tasks/<int:task_id>
# 删除任务

POST /api/tasks/<int:task_id>/execute
# 立即执行任务，使用 execute_ssh_command()

GET /api/tasks/<int:task_id>/result
# 获取任务执行结果（result JSON）
```

### 统计接口
```python
GET /api/stats
# 返回 {server_count, service_count, online_server_count}
```

### WebSocket 事件
```python
# 客户端 → 服务器
emit('ssh_connect', {server_id, auto_command?, service_id?})
emit('ssh_input', {input: string})
emit('ssh_disconnect')
emit('disconnect')

# 服务器 → 客户端
emit('ssh_connected', {status, reused?})
emit('ssh_output', {data: string})
emit('ssh_error', {error: string})
emit('ssh_disconnected', {status})
```

## 4. 使用示例

```python
# 后端服务启动
# 安装依赖
pip install -r requirements.txt

# 启动开发服务器（自动运行在 0.0.0.0:5000）
python app.py

# 前端通过 API 管理服务器
import requests

# 创建服务器
response = requests.post('http://localhost:5000/api/servers', json={
    'name': 'GPU Server 1',
    'brand_id': 1,
    'model_id': 1,
    'host_ip': '192.168.1.100',
    'username': 'root',
    'password': 'password',
    'port': 22
})
server_id = response.json()['id']

# 部署服务到服务器
service_response = requests.post('http://localhost:5000/api/services', json={
    'name': 'LLM Service',
    'model_id': 1,
    'server_ids': [server_id],
    'deploy_command': 'python /path/to/server.py --port 8000'
})

# 通过 WebSocket 连接 SSH 终端
import socketio_client
sio = socketio_client.Client()
sio.connect('http://localhost:5000')

# 建立 SSH 连接
sio.emit('ssh_connect', {
    'server_id': server_id,
    'auto_command': 'cd /workspace && python main.py'
})

# 监听终端输出
@sio.on('ssh_output')
def on_output(data):
    print(data['data'], end='')

# 发送输入
sio.emit('ssh_input', {'input': 'ls -l\n'})

# 断开连接
sio.emit('ssh_disconnect')
```

## 5. 实现细节

### 内存管理
- **SQLite 连接管理**: 每个请求创建新连接（`get_db()`），请求结束立即关闭，无连接池
- **文件上传**: 使用 `uuid.uuid4()` 生成唯一文件名，避免覆盖，存储在 `../uploads/` 目录
- **WebSocket 会话**: 使用 `request.sid` 作为会话标识，连接信息存储在全局字典 `ssh_connections`

### 并发控制
- **线程模型**: 部署、重启、状态更新使用独立 `threading.Thread`，标记为 `daemon=True`（主线程退出时自动终止）
- **SSH 连接复用**: 同一服务的多个 WebSocket 会话共享一个 SSH 连接，使用独立 channel 隔离
- **竞态条件保护**: 数据库写入无锁保护（SQLite 默认 serializable 隔离级别），SSH 连接字典操作无锁（单线程 GIL 保护）

### 性能优化
- **批量操作**: `/api/servers/resources` 和 `/api/servers/check-all` 支持批量查询，减少往返次数
- **异步状态更新**: 服务列表接口触发后台线程异步更新状态，不阻塞响应
- **流式响应**: 聊天完成接口支持 SSE 流式输出，降低首字延迟（TTFB）
- **接收线程优化**: SSH 终端使用 `time.sleep(0.05)` 降低轮询频率，减少 CPU 占用

### 错误处理
- **数据库迁移**: 使用 `ALTER TABLE` 时捕获 `sqlite3.OperationalError`，实现幂等性
- **SSH 超时**: Paramiko 连接设置 `timeout=30`，命令执行使用 `channel.recv_exit_status()` 等待完成
- **Agent 通信**: `call_service_agent()` 返回 `{'error': string}` 格式，调用方检查 `error` 字段判断失败
- **优雅降级**: 资源监控每个指标独立捕获异常，部分失败不影响其他指标

### 安全性
- **文件上传**: 使用 `secure_filename()` 验证文件名，白名单限制扩展名为 `{png, jpg, jpeg, gif, webp}`
- **命令注入**: `execute_ssh_command()` 转义单引号，使用 `bash -l -c` 执行，避免 shell 注入
- **CORS 配置**: 使用 `flask-cors` 允许跨域请求，生产环境应限制 `cors_allowed_origins`
- **密钥管理**: `app.config['SECRET_KEY']` 硬编码为 `'your-secret-key-here'`，生产环境需替换为随机密钥

### 依赖项
- **Flask 3.0.0**: Web 框架
- **flask-cors 4.0.0**: 跨域支持
- **flask-socketio 5.3.5**: WebSocket 支持
- **python-socketio 5.10.0**: Socket.IO 客户端库
- **eventlet 0.33.3**: 异步网络库（WebSocket 后端）
- **paramiko 3.4.0**: SSHv2 协议实现
- **requests 2.31.0**: HTTP 客户端
- **sqlite3**: Python 内置数据库
- **werkzeug**: Flask 工具库（文件上传）

### 设计模式
- **Repository Pattern**: 数据库操作封装在 `get_db()` 和路由处理函数中
- **Proxy Pattern**: `/chat/completions` 接口代理到后端 Agent 服务
- **Thread Pool Pattern**: 后台任务使用独立线程（非线程池，每次创建新线程）
- **Singleton Pattern**: Flask 应用、SocketIO 实例、SSH 连接池均为全局单例
- **Observer Pattern**: WebSocket 事件驱动，客户端订阅 `ssh_output` 事件接收数据
