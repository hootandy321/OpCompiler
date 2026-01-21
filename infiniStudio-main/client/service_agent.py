#!/usr/bin/env python3
"""
服务部署客户端代理
运行在服务器上，提供HTTP接口用于部署、重启、关闭服务
"""

from flask import Flask, request, jsonify
import subprocess
import threading
import os
import signal
import time
from datetime import datetime
import json

def get_env_with_path():
    """获取包含PATH的环境变量字典"""
    env = os.environ.copy()
    # 确保PATH包含常见的系统路径
    common_paths = [
        '/usr/local/sbin',
        '/usr/local/bin',
        '/usr/sbin',
        '/usr/bin',
        '/sbin',
        '/bin'
    ]
    current_path = env.get('PATH', '')
    for path in common_paths:
        if path not in current_path:
            current_path = f"{path}:{current_path}"
    env['PATH'] = current_path
    return env

app = Flask(__name__)

# 存储服务进程信息
# {service_id: {'process': subprocess.Popen, 'command': str, 'status': str, 'start_time': datetime}}
services = {}

# 默认端口
DEFAULT_PORT = 8888

def get_service_status(service_id):
    """获取服务状态"""
    if service_id not in services:
        return {'status': 'stopped', 'message': '服务不存在'}
    
    service_info = services[service_id]
    process = service_info.get('process')
    
    if process is None:
        return {'status': 'stopped', 'message': '服务未运行'}
    
    # 检查进程是否还在运行
    if process.poll() is None:
        # 进程正在运行
        return {
            'status': 'running',
            'pid': process.pid,
            'command': service_info.get('command', ''),
            'start_time': service_info.get('start_time', '').isoformat() if service_info.get('start_time') else None
        }
    else:
        # 进程已结束
        return_code = process.returncode
        services[service_id]['process'] = None
        services[service_id]['status'] = 'stopped'
        return {
            'status': 'stopped',
            'return_code': return_code,
            'message': f'进程已结束，退出码: {return_code}'
        }

@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({'status': 'ok', 'message': 'Service agent is running'})

@app.route('/service/<service_id>/deploy', methods=['POST'])
def deploy_service(service_id):
    """部署服务"""
    try:
        data = request.json
        command = data.get('command', '')
        
        if not command:
            return jsonify({'error': '部署命令不能为空'}), 400
        
        # 如果服务已经在运行，先停止
        if service_id in services and services[service_id].get('process'):
            process = services[service_id]['process']
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except:
                    process.kill()
                services[service_id]['process'] = None
        
        # 启动新进程
        # 使用shell=True来支持复杂命令，但要注意安全性
        # 使用环境变量确保命令能找到正确的PATH等环境变量
        env = get_env_with_path()
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            preexec_fn=os.setsid if os.name != 'nt' else None  # 创建新的进程组
        )
        
        services[service_id] = {
            'process': process,
            'command': command,
            'status': 'running',
            'start_time': datetime.now()
        }
        
        return jsonify({
            'status': 'deployed',
            'pid': process.pid,
            'message': '服务部署成功'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/service/<service_id>/restart', methods=['POST'])
def restart_service(service_id):
    """重启服务"""
    try:
        data = request.json
        command = data.get('command', '')
        
        if not command:
            # 如果没有提供命令，使用之前的命令
            if service_id in services:
                command = services[service_id].get('command', '')
            if not command:
                return jsonify({'error': '重启命令不能为空'}), 400
        
        # 先停止服务
        if service_id in services and services[service_id].get('process'):
            process = services[service_id]['process']
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except:
                    process.kill()
        
        # 等待一小段时间确保进程完全停止
        time.sleep(1)
        
        # 启动新进程
        # 使用环境变量确保命令能找到正确的PATH等环境变量
        env = get_env_with_path()
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            preexec_fn=os.setsid if os.name != 'nt' else None
        )
        
        services[service_id] = {
            'process': process,
            'command': command,
            'status': 'running',
            'start_time': datetime.now()
        }
        
        return jsonify({
            'status': 'restarted',
            'pid': process.pid,
            'message': '服务重启成功'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/service/<service_id>/stop', methods=['POST'])
def stop_service(service_id):
    """停止服务"""
    try:
        if service_id not in services:
            return jsonify({'error': '服务不存在'}), 404
        
        process = services[service_id].get('process')
        
        if process is None:
            return jsonify({'status': 'stopped', 'message': '服务未运行'})
        
        if process.poll() is None:
            # 进程正在运行，尝试优雅停止
            try:
                if os.name != 'nt':
                    # Linux/Mac: 发送SIGTERM到进程组
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    # Windows: 终止进程
                    process.terminate()
                
                # 等待进程结束
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # 如果5秒内没有结束，强制杀死
                    if os.name != 'nt':
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
            except Exception as e:
                return jsonify({'error': f'停止服务失败: {str(e)}'}), 500
        
        services[service_id]['process'] = None
        services[service_id]['status'] = 'stopped'
        
        return jsonify({'status': 'stopped', 'message': '服务已停止'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/service/<service_id>/status', methods=['GET'])
def get_service_status_endpoint(service_id):
    """获取服务状态"""
    try:
        status = get_service_status(service_id)
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/services', methods=['GET'])
def list_services():
    """列出所有服务"""
    result = {}
    for service_id, info in services.items():
        status = get_service_status(service_id)
        result[service_id] = status
    return jsonify(result)

if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    print(f'Service agent starting on port {port}...')
    app.run(host='0.0.0.0', port=port, debug=False)

