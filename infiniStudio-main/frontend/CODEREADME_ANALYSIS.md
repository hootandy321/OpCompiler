# 📂 目录: frontend 架构全景

## 1. 子系统职责

`frontend` 目录是 **InfiniStudio 的 Web 前端应用**,负责为整个 AI 模型服务管理平台提供用户交互界面。作为 InfiniStudio 的前端子系统,它通过现代 Web 技术栈实现了对 AI 模型部署、服务器管理、服务监控等核心功能的可视化操作。

该前端应用采用 **Vue 3 + Ant Design Vue** 技术栈,提供了响应式的单页面应用(SPA)体验,通过 RESTful API 与后端服务通信,并通过 WebSocket 实现实时 SSH 终端交互。其核心职责包括:
- 提供品牌、模型、服务器、服务、任务的 CRUD 管理界面
- 实时监控服务运行状态和服务器资源使用情况
- 提供 Web SSH 终端进行远程服务器操作
- 支持在线聊天测试已部署的 AI 服务

## 2. 模块导航

* **📂 public**
    * *功能*: 静态资源目录
    * *职责*: 存放 HTML 入口文件和公共静态资源,包含应用的 index.html 模板
    * *文档状态*: 无独立文档

* **📂 src**
    * *功能*: 应用源代码根目录
    * *职责*: 包含应用的核心业务逻辑、组件、视图和路由配置
    * *文档状态*: 无独立文档

* **📂 src/api**
    * *功能*: API 通信层
    * *职责*: 封装所有后端 API 调用,包括品牌管理、加速卡管理、模型管理、服务器管理、服务管理、聊天记录、计划任务、统计信息等 8 个功能模块的 HTTP 请求接口
    * *文档状态*: 无独立文档

* **📂 src/components**
    * *功能*: 可复用组件库
    * *职责*: 提供 SshTerminal 组件,基于 xterm.js 和 Socket.IO 实现的 Web SSH 终端,支持实时连接远程服务器、自动调整终端大小、优雅断开连接等功能
    * *文档状态*: 无独立文档

* **📂 src/router**
    * *功能*: 路由配置管理
    * *职责*: 定义应用的 8 个主要路由页面,包括总览、品牌管理(含详情页)、模型管理、服务器管理、服务管理(含聊天页)、计划任务管理
    * *文档状态*: 无独立文档

* **📂 src/views**
    * *功能*: 页面视图组件
    * *职责*: 包含 8 个主要业务页面组件 - Overview(总览仪表板)、Brands(品牌管理)、BrandDetail(品牌详情与加速卡管理)、Models(模型管理)、Servers(服务器管理)、Services(服务管理)、ServiceChat(服务聊天测试)、Tasks(计划任务管理)
    * *文档状态*: 无独立文档

## 3. 架构逻辑图解

### 3.1 技术栈依赖关系

```
Vue 3.3.4 (核心框架)
  ├── Vue Router 4.2.5 (路由管理)
  ├── Ant Design Vue 4.0.0 (UI 组件库)
  ├── Axios 1.6.0 (HTTP 客户端)
  ├── Socket.IO Client 4.7.2 (WebSocket 通信)
  ├── xterm.js 6.0.0 (终端模拟器)
  └── vuedraggable 4.1.0 (拖拽排序)
```

### 3.2 应用启动流程

```
入口: public/index.html
  ↓
main.js (应用初始化)
  ├─ 创建 Vue 应用实例
  ├─ 配置 Axios (baseURL: /api, timeout: 30s)
  ├─ 注册 Ant Design Vue 全局组件
  ├─ 配置 Vue Router
  └─ 挂载到 #app 元素
  ↓
App.vue (根组件)
  ├─ 渲染固定顶部导航栏 (Logo + 菜单)
  ├─ 渲染左侧侧边栏导航菜单
  └─ 动态加载 <router-view> 内容
  ↓
各页面视图组件 (根据当前路由)
```

### 3.3 模块间数据流

#### A. 页面导航流程
```
用户点击侧边栏菜单
  ↓
router/index.js 路由匹配
  ↓
加载对应 Views 组件
  ├─ / → Overview.vue (总览)
  ├─ /brands → Brands.vue (品牌列表)
  ├─ /brands/:id → BrandDetail.vue (品牌详情)
  ├─ /models → Models.vue (模型管理)
  ├─ /servers → Servers.vue (服务器管理)
  ├─ /services → Services.vue (服务管理)
  ├─ /services/:id/chat → ServiceChat.vue (聊天测试)
  └─ /tasks → Tasks.vue (任务管理)
```

#### B. API 通信流程
```
Views 组件调用 API 方法
  ↓
src/api/index.js (封装 Axios 请求)
  ├─ 构建 HTTP 请求 (GET/POST/PUT/DELETE)
  ├─ 设置请求头 (multipart/form-data 用于文件上传)
  └─ 发送到 /api 端点
  ↓
后端服务器处理 (需要反向代理配置)
  ↓
返回响应数据
  ↓
Views 组件更新 UI 状态
```

#### C. WebSocket 实时通信流程 (SSH 终端)
```
ServiceChat/Servers 页面加载 SshTerminal 组件
  ↓
建立 WebSocket 连接 (socket.io-client 连接到 localhost:5000)
  ↓
发送 ssh_connect 事件 (携带 server_id, 可选 auto_command, service_id)
  ↓
后端 SSH 会话建立
  ↓
双向实时数据流:
  ├─ 用户输入 → ssh_input 事件 → 后端 SSH
  └─ 后端 SSH 输出 → ssh_output 事件 → 终端显示
  ↓
组件卸载或用户断开 → 清理 WebSocket 和终端实例
```

#### D. 功能模块交互关系

```
总览页面 (Overview)
  ├─ 调用 /api/stats 获取统计数据
  ├─ 显示服务器总数、在线服务器、运行服务数、计划任务数
  └─ 提供快速导航到各管理页面

品牌管理 (Brands → BrandDetail)
  ├─ 品牌列表: CRUD 操作 + 拖拽排序
  ├─ 品牌详情: 管理该品牌下的加速卡型号
  └─ 上传品牌 Logo (文件上传 API)

模型管理 (Models)
  ├─ 模型列表: CRUD 操作
  ├─ 关联品牌和加速卡信息
  └─ 上传模型 Logo

服务器管理 (Servers)
  ├─ 服务器列表: CRUD 操作
  ├─ 连接测试: 检查 SSH 连通性
  ├─ 资源监控: 获取 GPU/内存使用情况
  └─ 集成 SshTerminal 组件进行临时 SSH 连接

服务管理 (Services → ServiceChat)
  ├─ 服务列表: 展示所有部署的服务状态
  ├─ 部署服务: 选择模型、服务器、配置参数
  ├─ 服务操作: 启动、停止、重启、查看部署日志
  ├─ 聊天测试: 通过流式 API 测试已部署服务
  └─ 持久化 SSH: 为服务提供长期 SSH 会话

计划任务 (Tasks)
  ├─ 任务列表: CRUD 操作
  ├─ 定时执行: cron 表达式调度
  └─ 执行历史: 查看任务执行结果
```

### 3.4 关键技术特性

1. **响应式设计**: 使用 Ant Design 的 Grid 系统 (Row/Col) 实现移动端适配
2. **实时交互**: 通过 Socket.IO 实现 SSH 终端的实时双向通信
3. **流式响应**: 支持聊天 API 的 Server-Sent Events (SSE) 流式输出
4. **拖拽排序**: 使用 vuedraggable 实现品牌的拖拽重排序
5. **表单处理**: 支持 FormData 文件上传 (Logo 图片)
6. **状态管理**: 各组件独立管理状态,未使用全局状态管理库
7. **错误处理**: 全局捕获 ResizeObserver 警告,防止控制台噪音
8. **主题定制**: 自定义 Ant Design 样式,使用渐变色主题

### 3.5 与后端集成要求

该前端应用需要以下后端支持:
- **RESTful API**: 运行在 `/api` 路径下的 HTTP 服务
- **WebSocket 服务**: 运行在 `localhost:5000` 的 Socket.IO 服务器 (用于 SSH)
- **反向代理**: 开发环境需要 vue.config.js 配置代理,生产环境需要 Nginx 等反向代理
- **CORS 配置**: 允许前端跨域访问后端 API

### 3.6 文件组织结构

```
frontend/
├── public/
│   └── index.html              # HTML 模板
├── src/
│   ├── main.js                 # 应用入口
│   ├── App.vue                 # 根组件 (布局框架)
│   ├── api/
│   │   └── index.js            # API 封装 (140 行,8 个模块)
│   ├── components/
│   │   └── SshTerminal.vue     # SSH 终端组件 (400 行)
│   ├── router/
│   │   └── index.js            # 路由配置 (60 行,8 个路由)
│   └── views/
│       ├── Overview.vue        # 总览页面
│       ├── Brands.vue          # 品牌管理
│       ├── BrandDetail.vue     # 品牌详情
│       ├── Models.vue          # 模型管理
│       ├── Servers.vue         # 服务器管理
│       ├── Services.vue        # 服务管理
│       ├── ServiceChat.vue     # 聊天测试
│       └── Tasks.vue           # 任务管理
├── package.json                # 依赖配置
├── vue.config.js               # Vue CLI 配置
└── babel.config.js             # Babel 配置
```

### 3.7 设计模式与架构特点

- **组件化架构**: 页面组件、业务组件分离,便于复用和维护
- **API 层抽象**: 统一的 API 接口封装,便于后续扩展和测试
- **声明式路由**: 使用 Vue Router 的声明式路由配置
- **响应式数据流**: 利用 Vue 3 Composition API 进行状态管理
- **渐进式增强**: 从基础功能开始,逐步添加拖拽、实时通信等高级特性

---

**总结**: `frontend` 目录是一个结构清晰、功能完整的 Vue 3 单页应用,涵盖了 AI 模型服务管理平台的全部前端功能。其架构体现了现代 Web 应用的最佳实践,通过模块化组织和清晰的职责划分,为后续功能扩展和维护提供了良好的基础。
