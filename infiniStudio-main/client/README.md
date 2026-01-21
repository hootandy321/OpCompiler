# 服务部署客户端代理

这是一个运行在服务器上的Python脚本，提供HTTP接口用于部署、重启、关闭服务。

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
python service_agent.py [端口号]
```

默认端口为8888，可以通过命令行参数指定其他端口。

## API接口

### 健康检查
- `GET /health` - 检查代理是否运行

### 部署服务
- `POST /service/<service_id>/deploy`
  - Body: `{"command": "部署命令"}`
  - 返回: `{"status": "deployed", "pid": 12345, "message": "服务部署成功"}`

### 重启服务
- `POST /service/<service_id>/restart`
  - Body: `{"command": "重启命令"}`
  - 返回: `{"status": "restarted", "pid": 12345, "message": "服务重启成功"}`

### 停止服务
- `POST /service/<service_id>/stop`
  - 返回: `{"status": "stopped", "message": "服务已停止"}`

### 获取服务状态
- `GET /service/<service_id>/status`
  - 返回: `{"status": "running|stopped", "pid": 12345, ...}`

### 列出所有服务
- `GET /services`
  - 返回所有服务的状态

## 注意事项

1. 确保防火墙允许访问指定的端口
2. 建议使用systemd或supervisor等工具将脚本作为服务运行
3. 脚本会维护服务进程，关闭时会优雅地停止进程

