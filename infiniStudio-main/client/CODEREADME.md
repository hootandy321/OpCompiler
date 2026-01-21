# `Service Agent` 服务部署代理核心实现文档

这是一个基于 Flask 的轻量级服务部署代理系统，运行在目标服务器上，通过 HTTP REST API 接收远程指令，实现服务的部署、重启、停止和状态监控。核心特性包括进程生命周期管理、优雅关闭机制、跨平台兼容性（Linux/Windows）以及完整的环境变量隔离。

## 1. Module Structure

- **`service_agent.py`**: 核心实现文件，包含 Flask HTTP 服务器、进程管理逻辑和 REST API 端点
- **`requirements.txt`**: Python 依赖声明，仅依赖 Flask 3.0.0
- **`README.md`**: 用户使用文档，包含 API 说明和部署指南

## 2. Core Classes

### `ServiceAgent Application`
- **Location**: `service_agent.py` (lines 35-247)
- **Primary Function**: Flask 应用实例，提供 HTTP 接口用于远程管理服务器上的服务进程。支持通过 REST API 部署、重启、停止和查询服务状态，维护全局服务注册表，处理进程生命周期管理。
- **Key Members**:
  - `services` (dict): 全局服务注册表，结构为 `{service_id: {'process': subprocess.Popen, 'command': str, 'status': str, 'start_time': datetime}}`。存储所有已部署服务的进程对象、启动命令、运行状态和启动时间戳
  - `DEFAULT_PORT` (int): 默认 HTTP 监听端口，值为 8888
  - `app` (Flask): Flask 应用实例，封装所有 HTTP 路由和请求处理逻辑
- **Core Methods**:
  - `get_env_with_path()`: 构建包含标准系统路径的环境变量字典。遍历常见系统二进制路径（`/usr/local/sbin`, `/usr/local/bin`, `/usr/sbin`, `/usr/bin`, `/sbin`, `/bin`），确保这些路径存在于 `PATH` 环境变量中，避免因环境变量缺失导致的命令执行失败
  - `get_service_status(service_id)`: 查询指定服务的运行状态。从全局注册表查找服务，通过 `process.poll()` 检测进程是否存活（返回 `None` 表示运行中），返回包含 `status`、`pid`、`command`、`start_time` 或 `return_code` 的状态字典
  - `health()` (GET `/health`): 健康检查端点。返回固定 JSON 响应 `{'status': 'ok', 'message': 'Service agent is running'}`，用于监控系统检测代理可用性
  - `deploy_service(service_id)` (POST `/service/<service_id>/deploy`): 部署新服务。从 JSON body 提取 `command` 字段，若服务已存在且运行中则先执行优雅停止（`process.terminate()` + 5秒超时等待，失败则 `process.kill()`），随后使用 `subprocess.Popen` 启动新进程。关键参数：`shell=True` 支持复杂 shell 命令，`preexec_fn=os.setsid` 在 Linux/Mac 上创建新进程组，`stdout=subprocess.PIPE, stderr=subprocess.PIPE` 捕获输出流。将进程信息存入 `services` 注册表并返回 PID
  - `restart_service(service_id)` (POST `/service/<service_id>/restart`): 重启服务。支持两种模式：① 提供 `command` 参数则使用新命令重启；② 未提供则复用历史命令。先终止旧进程（`terminate()` + 5秒超时 + `kill()` 强制清理），等待 1 秒确保进程完全退出后，用相同逻辑启动新进程并更新注册表
  - `stop_service(service_id)` (POST `/service/<service_id>/stop`): 停止服务。验证服务存在性后，根据操作系统类型执行不同的终止策略：Linux/Mac 使用 `os.killpg(os.getpgid(process.pid), signal.SIGTERM)` 发送信号到整个进程组（确保子进程也被终止），Windows 使用 `process.terminate()`。实现优雅关闭：先 SIGTERM，等待 5 秒，若超时则 SIGKILL 强制杀死。成功后更新注册表状态为 `stopped`
  - `get_service_status_endpoint(service_id)` (GET `/service/<service_id>/status`): 服务状态查询端点。调用 `get_service_status()` 获取服务状态并返回 JSON 响应，处理异常返回 500 错误
  - `list_services()` (GET `/services`): 全量服务列表端点。遍历 `services` 注册表，对每个服务调用 `get_service_status()` 构建完整状态字典，返回 `{service_id: status_dict}` 的 JSON 映射
- **Lifecycle**: 单进程常驻应用。通过 `python service_agent.py [port]` 启动，解析命令行参数指定监听端口（默认 8888），使用 `app.run(host='0.0.0.0', debug=False)` 启动 Flask 内置服务器。运行期间维护 `services` 全局字典，进程终止时操作系统自动回收所有子进程（未实现显式的清理钩子）

## 3. API Interface

```python
# 核心部署接口
POST /service/<service_id>/deploy
# 接收 JSON body: {"command": "shell命令字符串"}
# 返回: {"status": "deployed", "pid": 12345, "message": "服务部署成功"}
# 或错误: {"error": "错误信息"} (HTTP 400/500)

# 服务重启接口
POST /service/<service_id>/restart
# 可选 JSON body: {"command": "新命令字符串"}（不提供则使用历史命令）
# 返回: {"status": "restarted", "pid": 12345, "message": "服务重启成功"}

# 服务停止接口
POST /service/<service_id>/stop
# 返回: {"status": "stopped", "message": "服务已停止"}

# 服务状态查询接口
GET /service/<service_id>/status
# 返回: {"status": "running|stopped", "pid": 12345, "command": "启动命令", "start_time": "ISO时间戳"}
# 或: {"status": "stopped", "return_code": 退出码, "message": "进程已结束"}

# 全量服务列表接口
GET /services
# 返回: {"service_id_1": {...状态...}, "service_id_2": {...状态...}, ...}

# 健康检查接口
GET /health
# 返回: {"status": "ok", "message": "Service agent is running"}

# 环境变量构建函数（内部辅助）
def get_env_with_path() -> dict
# 返回包含完整系统 PATH 的环境变量字典，用于 subprocess.Popen 的 env 参数
```

## 4. Usage Example

```python
# 示例：部署和管理一个长期运行的服务
import requests
import time

# 配置代理地址
AGENT_URL = "http://target-server:8888"

# 1. 健康检查（验证代理可用）
health = requests.get(f"{AGENT_URL}/health").json()
assert health['status'] == 'ok'

# 2. 部署一个 Python Web 服务
deploy_response = requests.post(
    f"{AGENT_URL}/service/myapp/deploy",
    json={"command": "cd /opt/myapp && python app.py --port 8080"}
).json()
print(f"服务已启动，PID: {deploy_response['pid']}")

# 3. 查询服务状态
status = requests.get(f"{AGENT_URL}/service/myapp/status").json()
print(f"状态: {status['status']}, PID: {status['pid']}, 启动时间: {status['start_time']}")

# 4. 使用新配置重启服务
restart_response = requests.post(
    f"{AGENT_URL}/service/myapp/restart",
    json={"command": "cd /opt/myapp && python app.py --port 8080 --debug"}
).json()
print(f"服务已重启，新 PID: {restart_response['pid']}")

# 5. 停止服务
stop_response = requests.post(f"{AGENT_URL}/service/myapp/stop").json()
print(f"停止结果: {stop_response['status']}")

# 6. 批量查询所有服务
all_services = requests.get(f"{AGENT_URL}/services").json()
for service_id, service_status in all_services.items():
    print(f"{service_id}: {service_status['status']}")
```

## 5. Implementation Details

### 进程管理机制
- **进程隔离**: 使用 `subprocess.Popen` 启动独立进程，`stdout=subprocess.PIPE, stderr=subprocess.PIPE` 捕获标准输出流（当前未实现日志读取，未来可用于实时查看服务输出）
- **进程组管理**: Linux/Mac 平台通过 `preexec_fn=os.setsid` 创建新会话，结合 `os.killpg(os.getpgid(pid), signal.SIGTERM)` 实现进程组级别的信号发送，确保服务派生的所有子进程都能被正确终止
- **优雅关闭策略**: 实现两阶段终止：先 `SIGTERM`（Windows 为 `terminate()`），等待 5 秒让进程清理资源，若超时则 `SIGKILL`（Windows 为 `kill()`）强制杀死。避免数据丢失但保证进程最终退出
- **状态同步**: 通过 `process.poll()` 非阻塞检测进程存活状态，每次状态查询时实时更新注册表中的 `status` 字段，避免僵尸进程信息残留

### 并发与安全性
- **线程安全**: Flask 开发服务器默认单线程处理请求，`services` 字典作为全局共享数据结构未使用锁保护。若使用多线程 WSGI 服务器（如 Gunicorn），需引入 `threading.Lock` 保护 `services` 的读写操作
- **命令注入风险**: `shell=True` 允许执行任意 shell 命令，若 API 未做访问控制且暴露在公网，攻击者可通过 `command` 参数注入恶意命令（如 `rm -rf /`）。建议添加 IP 白名单、身份认证或迁移到 `shell=False` + 参数化命令列表
- **环境变量隔离**: `get_env_with_path()` 确保子进程继承完整系统 PATH，但未隔离其他环境变量。若服务需要特定环境配置（如 Python 虚拟环境），需在 `command` 中显式激活或扩展该函数注入自定义环境变量

### 跨平台兼容性
- **操作系统检测**: 使用 `os.name` 区分 Unix (`'posix'`) 和 Windows (`'nt'`)，在进程创建和终止时采用不同 API（`os.setsid` / `os.killpg` 仅 Unix 有效）
- **路径处理**: 硬编码 Unix 风格路径（`/usr/local/bin` 等），未使用 `os.path.join` 或 `pathlib`，在 Windows 上这些路径不存在但不影响功能（仅用于 PATH 补全）

### 错误处理
- **异常捕获**: 所有 API 端点使用 `try-except` 捕获异常，返回 `{"error": "错误信息"}` 和对应 HTTP 状态码（400 参数错误，404 服务不存在，500 服务器错误）
- **进程泄漏防护**: 部署和重启时先检测旧进程是否存活（`poll()` 为 `None`），确保启动新进程前终止旧进程，避免端口占用和资源泄漏
- **超时处理**: `process.wait(timeout=5)` 配合 `TimeoutExpired` 异常，实现优雅关闭超时后的强制杀死逻辑

### 设计模式
- **Registry 模式**: `services` 字典作为进程注册中心，通过 `service_id` 索引维护全局服务状态，支持多实例管理
- **RESTful API**: 遵循 REST 设计原则，使用 HTTP 方法语义（GET 查询，POST 修改）和资源层级路径（`/service/<id>/action`）
- **单例应用**: Flask `app` 对象作为全局单例，所有路由函数共享 `services` 状态，避免进程间通信开销

### 依赖与部署
- **最小依赖**: 仅依赖 Flask 3.0.0，无第三方数据库或消息队列，部署时只需 `pip install -r requirements.txt`
- **服务化建议**: 文档建议使用 systemd 或 supervisor 管理代理进程，确保代理崩溃后自动重启，并实现开机自启动
- **网络配置**: 监听 `0.0.0.0` 允许所有网络接口访问，需配合防火墙规则（如 `ufw allow 8888`）限制访问来源

### 扩展性考虑
- **日志输出流**: 当前 `stdout/stderr` 被管道捕获但未读取，长期运行可能导致缓冲区阻塞。可扩展为后台线程持续读取并写入日志文件或暴露 `GET /service/<id>/logs` 接口
- **进程监控**: 未实现进程意外退出后的自动重启，可添加后台线程周期性 `poll()` 所有进程，发现退出时根据历史命令自动重启（类似 supervisor 的 `autorestart` 功能）
- **认证授权**: 当前无访问控制，可集成 JWT 或 API Key 验证，在 Flask `before_request` 钩子中拦截未授权请求
