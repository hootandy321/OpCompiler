from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import sqlite3
import os
import threading
import time
import paramiko
from datetime import datetime
import json
from werkzeug.utils import secure_filename
import uuid
import subprocess
import platform
import requests

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

DATABASE = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'infini.db')
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_db():
    """初始化数据库"""
    os.makedirs(os.path.dirname(DATABASE), exist_ok=True)
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # 品牌表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS brands (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            logo TEXT,
            sort_order INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 为现有表添加sort_order列（如果不存在）
    try:
        cursor.execute('ALTER TABLE brands ADD COLUMN sort_order INTEGER DEFAULT 0')
    except sqlite3.OperationalError:
        # 列已存在，忽略错误
        pass
    
    # 加速卡表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS accelerator_cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            model TEXT NOT NULL,
            memory TEXT,
            fp8_perf TEXT,
            int8_perf TEXT,
            bf16_perf TEXT,
            fp16_perf TEXT,
            fp32_perf TEXT,
            interconnect_bandwidth TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (brand_id) REFERENCES brands(id)
        )
    ''')
    
    # 模型表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            logo TEXT,
            parameters TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 为现有表添加parameters列（如果不存在）
    try:
        cursor.execute('ALTER TABLE models ADD COLUMN parameters TEXT')
    except sqlite3.OperationalError:
        # 列已存在，忽略错误
        pass
    
    # 服务器表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS servers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            brand_id INTEGER,
            model_id INTEGER,
            host_ip TEXT NOT NULL,
            port INTEGER DEFAULT 22,
            username TEXT NOT NULL,
            password TEXT,
            agent_port INTEGER DEFAULT 8888,
            status TEXT DEFAULT 'offline',
            last_check TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (brand_id) REFERENCES brands(id),
            FOREIGN KEY (model_id) REFERENCES accelerator_cards(id)
        )
    ''')
    
    # 为现有表添加agent_port列（如果不存在）
    try:
        cursor.execute('ALTER TABLE servers ADD COLUMN agent_port INTEGER DEFAULT 8888')
    except sqlite3.OperationalError:
        pass
    
    # 服务表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS services (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            model_id INTEGER NOT NULL,
            server_ids TEXT NOT NULL,
            deploy_command TEXT,
            deploy_status TEXT DEFAULT 'pending',
            deploy_result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES models(id)
        )
    ''')
    
    # 为现有表添加deploy_command和deploy_status列（如果不存在）
    try:
        cursor.execute('ALTER TABLE services ADD COLUMN deploy_command TEXT')
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute('ALTER TABLE services ADD COLUMN deploy_status TEXT DEFAULT "pending"')
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute('ALTER TABLE services ADD COLUMN deploy_result TEXT')
    except sqlite3.OperationalError:
        pass
    
    # 聊天记录表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            service_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (service_id) REFERENCES services(id)
        )
    ''')
    
    # 计划任务表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            command TEXT NOT NULL,
            server_id INTEGER NOT NULL,
            schedule_type TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            last_run TIMESTAMP,
            result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (server_id) REFERENCES servers(id)
        )
    ''')
    
    # 为现有表添加result列（如果不存在）
    try:
        cursor.execute('ALTER TABLE tasks ADD COLUMN result TEXT')
    except sqlite3.OperationalError:
        # 列已存在，忽略错误
        pass
    
    conn.commit()
    conn.close()

init_db()

# SSH连接池
ssh_connections = {}  # 临时连接：{session_id: {...}}
service_ssh_connections = {}  # 服务持久化连接：{service_id: {ssh, chan, sessions: [session_id, ...]}}

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# ==================== 文件上传 ====================

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """上传文件接口"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # 生成唯一文件名
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4()}.{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # 返回文件URL
        file_url = f"/api/uploads/{filename}"
        return jsonify({'url': file_url, 'filename': filename}), 200
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/uploads/<filename>')
def uploaded_file(filename):
    """返回上传的文件"""
    return send_from_directory(UPLOAD_FOLDER, filename)

# ==================== 品牌管理 ====================

@app.route('/api/brands', methods=['GET'])
def get_brands():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM brands ORDER BY sort_order ASC, created_at DESC')
    brands = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(brands)

@app.route('/api/brands', methods=['POST'])
def create_brand():
    # 支持JSON和表单数据
    if request.content_type and 'application/json' in request.content_type:
        data = request.json
        logo_url = data.get('logo')
    else:
        data = request.form.to_dict()
        logo_url = data.get('logo')
        # 如果有文件上传
        if 'logo_file' in request.files:
            file = request.files['logo_file']
            if file.filename and allowed_file(file.filename):
                ext = file.filename.rsplit('.', 1)[1].lower()
                filename = f"{uuid.uuid4()}.{ext}"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                logo_url = f"/api/uploads/{filename}"
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO brands (name, logo) VALUES (?, ?)
    ''', (data['name'], logo_url))
    conn.commit()
    brand_id = cursor.lastrowid
    conn.close()
    return jsonify({'id': brand_id, 'message': 'Brand created successfully'}), 201

@app.route('/api/brands/<int:brand_id>', methods=['PUT'])
def update_brand(brand_id):
    # 支持JSON和表单数据
    if request.content_type and 'application/json' in request.content_type:
        data = request.json
        logo_url = data.get('logo')
    else:
        data = request.form.to_dict()
        logo_url = data.get('logo')
        # 如果有文件上传
        if 'logo_file' in request.files:
            file = request.files['logo_file']
            if file.filename and allowed_file(file.filename):
                ext = file.filename.rsplit('.', 1)[1].lower()
                filename = f"{uuid.uuid4()}.{ext}"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                logo_url = f"/api/uploads/{filename}"
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE brands SET name = ?, logo = ? WHERE id = ?
    ''', (data['name'], logo_url, brand_id))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Brand updated successfully'})

@app.route('/api/brands/<int:brand_id>', methods=['DELETE'])
def delete_brand(brand_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM brands WHERE id = ?', (brand_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Brand deleted successfully'})

@app.route('/api/brands/reorder', methods=['POST'])
def reorder_brands():
    """批量更新品牌排序"""
    data = request.json
    brand_orders = data.get('orders', [])  # [{id: 1, sort_order: 0}, {id: 2, sort_order: 1}, ...]
    
    conn = get_db()
    cursor = conn.cursor()
    for item in brand_orders:
        cursor.execute('UPDATE brands SET sort_order = ? WHERE id = ?', (item['sort_order'], item['id']))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Brand order updated successfully'})

# ==================== 加速卡管理 ====================

@app.route('/api/brands/<int:brand_id>/accelerators', methods=['GET'])
def get_accelerators(brand_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM accelerator_cards WHERE brand_id = ? ORDER BY created_at DESC', (brand_id,))
    accelerators = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(accelerators)

@app.route('/api/brands/<int:brand_id>/accelerators', methods=['POST'])
def create_accelerator(brand_id):
    data = request.json
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO accelerator_cards 
        (brand_id, name, model, memory, fp8_perf, int8_perf, bf16_perf, fp16_perf, fp32_perf, interconnect_bandwidth)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (brand_id, data['name'], data['model'], data.get('memory'), 
          data.get('fp8_perf'), data.get('int8_perf'), data.get('bf16_perf'), 
          data.get('fp16_perf'), data.get('fp32_perf'), data.get('interconnect_bandwidth')))
    conn.commit()
    accelerator_id = cursor.lastrowid
    conn.close()
    return jsonify({'id': accelerator_id, 'message': 'Accelerator created successfully'}), 201

@app.route('/api/accelerators/<int:accelerator_id>', methods=['PUT'])
def update_accelerator(accelerator_id):
    data = request.json
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE accelerator_cards 
        SET name = ?, model = ?, memory = ?, fp8_perf = ?, int8_perf = ?, 
            bf16_perf = ?, fp16_perf = ?, fp32_perf = ?, interconnect_bandwidth = ?
        WHERE id = ?
    ''', (data['name'], data['model'], data.get('memory'), data.get('fp8_perf'), 
          data.get('int8_perf'), data.get('bf16_perf'), data.get('fp16_perf'), 
          data.get('fp32_perf'), data.get('interconnect_bandwidth'), accelerator_id))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Accelerator updated successfully'})

@app.route('/api/accelerators/<int:accelerator_id>', methods=['DELETE'])
def delete_accelerator(accelerator_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM accelerator_cards WHERE id = ?', (accelerator_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Accelerator deleted successfully'})

# ==================== 模型管理 ====================

@app.route('/api/models', methods=['GET'])
def get_models():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM models ORDER BY created_at DESC')
    models = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(models)

@app.route('/api/models', methods=['POST'])
def create_model():
    # 支持JSON和表单数据
    if request.content_type and 'application/json' in request.content_type:
        data = request.json
        logo_url = data.get('logo')
    else:
        data = request.form.to_dict()
        logo_url = data.get('logo')
        # 如果有文件上传
        if 'logo_file' in request.files:
            file = request.files['logo_file']
            if file.filename and allowed_file(file.filename):
                ext = file.filename.rsplit('.', 1)[1].lower()
                filename = f"{uuid.uuid4()}.{ext}"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                logo_url = f"/api/uploads/{filename}"
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO models (name, logo, parameters) VALUES (?, ?, ?)', 
                   (data['name'], logo_url, data.get('parameters')))
    conn.commit()
    model_id = cursor.lastrowid
    conn.close()
    return jsonify({'id': model_id, 'message': 'Model created successfully'}), 201

@app.route('/api/models/<int:model_id>', methods=['PUT'])
def update_model(model_id):
    # 支持JSON和表单数据
    if request.content_type and 'application/json' in request.content_type:
        data = request.json
        logo_url = data.get('logo')
    else:
        data = request.form.to_dict()
        logo_url = data.get('logo')
        # 如果有文件上传
        if 'logo_file' in request.files:
            file = request.files['logo_file']
            if file.filename and allowed_file(file.filename):
                ext = file.filename.rsplit('.', 1)[1].lower()
                filename = f"{uuid.uuid4()}.{ext}"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                logo_url = f"/api/uploads/{filename}"
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('UPDATE models SET name = ?, logo = ?, parameters = ? WHERE id = ?', 
                   (data['name'], logo_url, data.get('parameters'), model_id))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Model updated successfully'})

@app.route('/api/models/<int:model_id>', methods=['DELETE'])
def delete_model(model_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM models WHERE id = ?', (model_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Model deleted successfully'})

# ==================== 服务器管理 ====================

@app.route('/api/servers', methods=['GET'])
def get_servers():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT s.*, b.name as brand_name, ac.name as model_name 
        FROM servers s
        LEFT JOIN brands b ON s.brand_id = b.id
        LEFT JOIN accelerator_cards ac ON s.model_id = ac.id
        ORDER BY s.created_at DESC
    ''')
    servers = []
    for row in cursor.fetchall():
        server = dict(row)
        servers.append(server)
    conn.close()
    return jsonify(servers)

@app.route('/api/servers', methods=['POST'])
def create_server():
    data = request.json
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO servers (name, brand_id, model_id, host_ip, port, username, password)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (data['name'], data.get('brand_id'), data.get('model_id'), 
          data['host_ip'], data.get('port', 22), data['username'], data.get('password')))
    conn.commit()
    server_id = cursor.lastrowid
    conn.close()
    return jsonify({'id': server_id, 'message': 'Server created successfully'}), 201

@app.route('/api/servers/<int:server_id>', methods=['PUT'])
def update_server(server_id):
    data = request.json
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE servers 
        SET name = ?, brand_id = ?, model_id = ?, host_ip = ?, port = ?, username = ?, password = ?
        WHERE id = ?
    ''', (data['name'], data.get('brand_id'), data.get('model_id'), data['host_ip'], 
          data.get('port', 22), data['username'], data.get('password'), server_id))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Server updated successfully'})

@app.route('/api/servers/<int:server_id>', methods=['DELETE'])
def delete_server(server_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM servers WHERE id = ?', (server_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Server deleted successfully'})

def ping_host(host):
    """使用ping命令检查主机是否在线"""
    try:
        # 根据操作系统选择ping命令参数
        if platform.system().lower() == 'windows':
            # Windows: -n 1 表示发送1个包, -w 1000 表示超时1000ms
            result = subprocess.run(['ping', '-n', '1', '-w', '1000', host], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
        else:
            # Linux/Mac: -c 1 表示发送1个包, -W 1 表示超时1秒
            result = subprocess.run(['ping', '-c', '1', '-W', '1', host], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
        return result.returncode == 0
    except:
        return False

@app.route('/api/servers/<int:server_id>/check', methods=['POST'])
def check_server(server_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM servers WHERE id = ?', (server_id,))
    server = dict(cursor.fetchone())
    conn.close()
    
    # 使用ping检查主机是否在线
    status = 'online' if ping_host(server['host_ip']) else 'offline'
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('UPDATE servers SET status = ?, last_check = ? WHERE id = ?', 
                   (status, datetime.now(), server_id))
    conn.commit()
    conn.close()
    
    return jsonify({'status': status})

def get_server_resources(server):
    """通过SSH获取服务器资源使用情况"""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(server['host_ip'], port=server['port'], 
                   username=server['username'], password=server.get('password'),
                   timeout=5)
        
        resources = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'disk_usage': 0.0
        }
        
        # 获取CPU使用率
        try:
            # 使用更可靠的方法获取CPU使用率
            stdin, stdout, stderr = ssh.exec_command("grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$3+$4+$5)} END {print usage}'")
            cpu_output = stdout.read().decode().strip()
            if cpu_output:
                resources['cpu_usage'] = round(float(cpu_output), 1)
            else:
                # 备用方法：使用top命令
                stdin, stdout, stderr = ssh.exec_command("top -bn1 | grep 'Cpu(s)' | sed 's/.*, *\\([0-9.]*\\)%* id.*/\\1/' | awk '{print 100 - $1}'")
                cpu_output = stdout.read().decode().strip()
                if cpu_output:
                    resources['cpu_usage'] = round(float(cpu_output), 1)
        except:
            pass
        
        # 获取内存使用率
        try:
            stdin, stdout, stderr = ssh.exec_command("free | grep Mem | awk '{printf \"%.1f\", $3/$2 * 100}'")
            mem_output = stdout.read().decode().strip()
            if mem_output:
                resources['memory_usage'] = round(float(mem_output), 1)
        except:
            pass
        
        # 获取磁盘使用率
        try:
            stdin, stdout, stderr = ssh.exec_command("df -h / | tail -1 | awk '{print $5}' | sed 's/%//'")
            disk_output = stdout.read().decode().strip()
            if disk_output:
                resources['disk_usage'] = round(float(disk_output), 1)
        except:
            pass
        
        ssh.close()
        return resources
    except Exception as e:
        return {
            'cpu_usage': None,
            'memory_usage': None,
            'disk_usage': None,
            'error': str(e)
        }

@app.route('/api/servers/<int:server_id>/resources', methods=['GET'])
def get_server_resources_api(server_id):
    """获取服务器资源使用情况"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM servers WHERE id = ?', (server_id,))
    server = dict(cursor.fetchone())
    conn.close()
    
    resources = get_server_resources(server)
    return jsonify(resources)

@app.route('/api/servers/resources', methods=['POST'])
def get_all_servers_resources():
    """批量获取所有服务器的资源使用情况"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM servers')
    servers = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    results = []
    for server in servers:
        resources = get_server_resources(server)
        results.append({
            'id': server['id'],
            **resources
        })
    
    return jsonify(results)

@app.route('/api/servers/check-all', methods=['POST'])
def check_all_servers():
    """批量检查所有服务器状态"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM servers')
    servers = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    results = []
    for server in servers:
        status = 'online' if ping_host(server['host_ip']) else 'offline'
        
        # 更新数据库
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('UPDATE servers SET status = ?, last_check = ? WHERE id = ?', 
                      (status, datetime.now(), server['id']))
        conn.commit()
        conn.close()
        
        results.append({'id': server['id'], 'status': status})
    
    return jsonify({'results': results})

# ==================== 服务管理 ====================

def call_service_agent(server, endpoint, method='GET', data=None, timeout=30):
    """
    调用服务代理接口
    server: 服务器信息字典
    endpoint: 接口路径，如 '/service/1/deploy'
    method: HTTP方法
    data: 请求数据
    timeout: 超时时间（秒）
    """
    try:
        agent_port = server.get('agent_port', 8888)
        url = f"http://{server['host_ip']}:{agent_port}{endpoint}"
        
        if method == 'GET':
            response = requests.get(url, timeout=timeout)
        elif method == 'POST':
            response = requests.post(url, json=data, timeout=timeout)
        else:
            return {'error': f'Unsupported method: {method}'}
        
        if response.status_code == 200:
            return response.json()
        else:
            return {'error': f'HTTP {response.status_code}: {response.text}'}
    except requests.exceptions.RequestException as e:
        return {'error': f'连接失败: {str(e)}'}
    except Exception as e:
        return {'error': str(e)}

def update_service_status(service_id):
    """更新服务状态：offline、online、deploying、deployed"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM services WHERE id = ?', (service_id,))
        service = dict(cursor.fetchone())
        conn.close()
        
        if not service:
            return
        
        # 如果状态是deploying，不更新（保持部署中状态）
        if service.get('deploy_status') == 'deploying':
            return
        
        server_ids = json.loads(service['server_ids']) if service['server_ids'] else []
        if not server_ids:
            # 没有服务器，状态为offline
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute('UPDATE services SET deploy_status = ? WHERE id = ?', ('offline', service_id))
            conn.commit()
            conn.close()
            return
        
        # 获取第一个服务器的状态（简化处理，实际可以聚合多个服务器状态）
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM servers WHERE id = ?', (server_ids[0],))
        server = dict(cursor.fetchone())
        conn.close()
        
        # 调用客户端接口获取状态
        result = call_service_agent(server, f'/service/{service_id}/status', 'GET', timeout=5)
        
        if 'error' in result:
            # 连接失败，状态为offline
            db_status = 'offline'
        else:
            # 根据状态更新数据库
            agent_status = result.get('status', 'stopped')
            # 映射客户端状态到数据库状态
            # running -> deployed, stopped -> online（服务器在线但服务未运行）
            if agent_status == 'running':
                db_status = 'deployed'
            else:
                db_status = 'online'  # 服务器在线但服务未运行
        
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('UPDATE services SET deploy_status = ? WHERE id = ?', (db_status, service_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f'Error updating service status: {e}')
        # 出错时设为offline
        try:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute('UPDATE services SET deploy_status = ? WHERE id = ?', ('offline', service_id))
            conn.commit()
            conn.close()
        except:
            pass

def execute_ssh_command(server, command):
    """
    执行SSH命令的改进版本
    使用伪终端和完整的环境变量加载
    """
    ssh = None
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(server['host_ip'], port=server['port'], 
                   username=server['username'], password=server.get('password'),
                   timeout=30)
        
        # 转义命令中的单引号，以便在bash -l -c中使用
        # 将 ' 替换为 '\'' (结束引号 + 转义的单引号 + 开始引号)
        escaped_command = command.replace("'", "'\\''")
        
        # 使用bash -l -c来执行命令，这样可以加载完整的登录环境
        # -l 表示登录shell，会加载 ~/.bash_profile, ~/.bashrc 等
        # 同时使用get_pty=True获取伪终端，这对于某些命令很重要
        full_command = f"bash -l -c '{escaped_command}'"
        
        stdin, stdout, stderr = ssh.exec_command(full_command, get_pty=True)
        
        # 等待命令执行完成
        exit_status = stdout.channel.recv_exit_status()
        
        # 读取所有输出
        output = stdout.read().decode('utf-8', errors='ignore')
        error = stderr.read().decode('utf-8', errors='ignore')
        
        return {
            'success': exit_status == 0,
            'output': output,
            'error': error,
            'exit_status': exit_status
        }
    except Exception as e:
        return {
            'success': False,
            'output': '',
            'error': str(e),
            'exit_status': -1
        }
    finally:
        if ssh:
            ssh.close()

def deploy_service_thread(service_id, server_ids, deploy_command):
    """在后台线程中执行部署命令"""
    try:
        # 更新状态为部署中
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('UPDATE services SET deploy_status = ? WHERE id = ?', ('deploying', service_id))
        conn.commit()
        conn.close()
        
        # 获取服务器信息
        conn = get_db()
        cursor = conn.cursor()
        servers = []
        for server_id in server_ids:
            cursor.execute('SELECT * FROM servers WHERE id = ?', (server_id,))
            server = dict(cursor.fetchone())
            servers.append(server)
        conn.close()
        
        # 在每个服务器上调用客户端接口部署
        all_success = True
        deploy_results = []
        for server in servers:
            server_result = {
                'server_id': server['id'],
                'server_name': server['name'],
                'server_ip': server['host_ip'],
                'success': False,
                'output': '',
                'error': ''
            }
            try:
                result = call_service_agent(server, f'/service/{service_id}/deploy', 'POST', {
                    'command': deploy_command
                })
                if 'error' in result:
                    server_result['error'] = result['error']
                    all_success = False
                else:
                    server_result['success'] = True
                    server_result['output'] = result.get('message', '部署成功')
            except Exception as e:
                server_result['error'] = str(e)
                all_success = False
            
            deploy_results.append(server_result)
        
        # 更新部署状态和结果
        conn = get_db()
        cursor = conn.cursor()
        # 部署完成后，状态会根据实际运行情况自动更新为deployed或online
        # 这里先设置为online，后续通过状态检查自动更新
        status = 'online'  # 部署完成，等待状态检查更新
        result_json = json.dumps(deploy_results, ensure_ascii=False)
        cursor.execute('UPDATE services SET deploy_status = ?, deploy_result = ? WHERE id = ?', 
                      (status, result_json, service_id))
        conn.commit()
        conn.close()
        
        # 部署完成后立即检查一次状态
        update_service_status(service_id)
    except Exception as e:
        # 部署失败，设为offline
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('UPDATE services SET deploy_status = ? WHERE id = ?', ('offline', service_id))
        conn.commit()
        conn.close()

@app.route('/api/services', methods=['GET'])
def get_services():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT s.*, m.name as model_name 
        FROM services s
        LEFT JOIN models m ON s.model_id = m.id
        ORDER BY s.created_at DESC
    ''')
    services = []
    for row in cursor.fetchall():
        service = dict(row)
        service['server_ids'] = json.loads(service['server_ids']) if service['server_ids'] else []
        services.append(service)
    conn.close()
    
    # 自动更新所有服务状态（不阻塞响应）
    def update_all_statuses():
        for service in services:
            try:
                update_service_status(service['id'])
            except:
                pass
    
    threading.Thread(target=update_all_statuses, daemon=True).start()
    
    return jsonify(services)

@app.route('/api/services/refresh-status', methods=['POST'])
def refresh_services_status():
    """手动刷新所有服务状态"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT s.*, m.name as model_name 
        FROM services s
        LEFT JOIN models m ON s.model_id = m.id
        ORDER BY s.created_at DESC
    ''')
    services = []
    for row in cursor.fetchall():
        service = dict(row)
        service['server_ids'] = json.loads(service['server_ids']) if service['server_ids'] else []
        services.append(service)
    conn.close()
    
    # 立即更新所有服务状态（同步执行）
    for service in services:
        try:
            update_service_status(service['id'])
        except:
            pass
    
    # 重新获取更新后的服务列表
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT s.*, m.name as model_name 
        FROM services s
        LEFT JOIN models m ON s.model_id = m.id
        ORDER BY s.created_at DESC
    ''')
    updated_services = []
    for row in cursor.fetchall():
        service = dict(row)
        service['server_ids'] = json.loads(service['server_ids']) if service['server_ids'] else []
        updated_services.append(service)
    conn.close()
    
    return jsonify(updated_services)

@app.route('/api/services', methods=['POST'])
def create_service():
    data = request.json
    conn = get_db()
    cursor = conn.cursor()
    server_ids = json.dumps(data.get('server_ids', []))
    deploy_command = data.get('deploy_command', '')
    cursor.execute('''
        INSERT INTO services (name, model_id, server_ids, deploy_command, deploy_status)
        VALUES (?, ?, ?, ?, ?)
    ''', (data['name'], data['model_id'], server_ids, deploy_command, 'offline'))
    conn.commit()
    service_id = cursor.lastrowid
    conn.close()
    
    # 如果有部署命令，调用客户端接口部署
    if deploy_command and data.get('server_ids'):
        threading.Thread(target=deploy_service_thread, args=(service_id, data['server_ids'], deploy_command), daemon=True).start()
    
    return jsonify({'id': service_id, 'message': 'Service created successfully'}), 201

@app.route('/api/services/<int:service_id>', methods=['PUT'])
def update_service(service_id):
    data = request.json
    conn = get_db()
    cursor = conn.cursor()
    server_ids = json.dumps(data.get('server_ids', []))
    deploy_command = data.get('deploy_command', '')
    cursor.execute('''
        UPDATE services SET name = ?, model_id = ?, server_ids = ?, deploy_command = ? WHERE id = ?
    ''', (data['name'], data['model_id'], server_ids, deploy_command, service_id))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Service updated successfully'})

@app.route('/api/services/<int:service_id>/restart', methods=['POST'])
def restart_service(service_id):
    """重启服务"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM services WHERE id = ?', (service_id,))
    service = dict(cursor.fetchone())
    conn.close()
    
    if not service:
        return jsonify({'error': 'Service not found'}), 404
    
    server_ids = json.loads(service['server_ids']) if service['server_ids'] else []
    if not server_ids:
        return jsonify({'error': 'No servers associated with this service'}), 400
    
    # 获取服务器信息
    conn = get_db()
    cursor = conn.cursor()
    servers = []
    for server_id in server_ids:
        cursor.execute('SELECT * FROM servers WHERE id = ?', (server_id,))
        server = dict(cursor.fetchone())
        servers.append(server)
    conn.close()
    
    # 执行重启命令（使用部署命令，但添加重启逻辑）
    # 这里假设重启命令就是在部署命令前添加重启逻辑，或者使用相同的命令
    # 实际使用中，可以添加专门的重启命令字段，这里简化处理使用部署命令
    restart_command = service.get('deploy_command', '')
    if not restart_command:
        return jsonify({'error': 'No deploy command found'}), 400
    
    # 在后台线程中执行重启
    def restart_thread():
        all_success = True
        deploy_results = []
        for server in servers:
            server_result = {
                'server_id': server['id'],
                'server_name': server['name'],
                'server_ip': server['host_ip'],
                'success': False,
                'output': '',
                'error': ''
            }
            try:
                result = call_service_agent(server, f'/service/{service_id}/restart', 'POST', {
                    'command': restart_command
                })
                if 'error' in result:
                    server_result['error'] = result['error']
                    all_success = False
                else:
                    server_result['success'] = True
                    server_result['output'] = result.get('message', '重启成功')
            except Exception as e:
                server_result['error'] = str(e)
                all_success = False
            
            deploy_results.append(server_result)
        
        # 更新部署状态和结果
        conn = get_db()
        cursor = conn.cursor()
        # 重启完成后，状态设为online，错误信息保存在deploy_result中
        status = 'online'  # 重启完成，等待状态检查更新
        result_json = json.dumps(deploy_results, ensure_ascii=False)
        cursor.execute('UPDATE services SET deploy_status = ?, deploy_result = ? WHERE id = ?', 
                      (status, result_json, service_id))
        conn.commit()
        conn.close()
        
        # 重启完成后立即检查一次状态
        update_service_status(service_id)
    
    threading.Thread(target=restart_thread, daemon=True).start()
    
    return jsonify({'message': 'Service restart initiated'})

@app.route('/api/services/<int:service_id>/stop', methods=['POST'])
def stop_service(service_id):
    """停止服务"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM services WHERE id = ?', (service_id,))
    service = dict(cursor.fetchone())
    conn.close()
    
    if not service:
        return jsonify({'error': 'Service not found'}), 404
    
    server_ids = json.loads(service['server_ids']) if service['server_ids'] else []
    if not server_ids:
        return jsonify({'error': 'No servers associated with this service'}), 400
    
    # 获取服务器信息
    conn = get_db()
    cursor = conn.cursor()
    servers = []
    for server_id in server_ids:
        cursor.execute('SELECT * FROM servers WHERE id = ?', (server_id,))
        server = dict(cursor.fetchone())
        servers.append(server)
    conn.close()
    
    # 在后台线程中执行停止
    def stop_thread():
        all_success = True
        stop_results = []
        for server in servers:
            server_result = {
                'server_id': server['id'],
                'server_name': server['name'],
                'server_ip': server['host_ip'],
                'success': False,
                'message': ''
            }
            try:
                result = call_service_agent(server, f'/service/{service_id}/stop', 'POST')
                if 'error' in result:
                    server_result['message'] = result['error']
                    all_success = False
                else:
                    server_result['success'] = True
                    server_result['message'] = result.get('message', '停止成功')
            except Exception as e:
                server_result['message'] = str(e)
                all_success = False
            
            stop_results.append(server_result)
        
        # 更新部署状态
        conn = get_db()
        cursor = conn.cursor()
        status = 'online'  # 停止后，状态设为online（服务器在线但服务未运行）
        cursor.execute('UPDATE services SET deploy_status = ? WHERE id = ?', (status, service_id))
        conn.commit()
        conn.close()
        
        # 停止后立即检查一次状态
        update_service_status(service_id)
    
    threading.Thread(target=stop_thread, daemon=True).start()
    
    return jsonify({'message': 'Service stop initiated'})

@app.route('/api/services/<int:service_id>/deploy-log', methods=['GET'])
def get_deploy_log(service_id):
    """获取服务部署日志"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT deploy_result FROM services WHERE id = ?', (service_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return jsonify({'error': 'Service not found'}), 404
    
    deploy_result = row['deploy_result']
    if deploy_result:
        try:
            results = json.loads(deploy_result)
            return jsonify(results)
        except:
            return jsonify([])
    return jsonify([])

@app.route('/api/services/<int:service_id>', methods=['DELETE'])
def delete_service(service_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM services WHERE id = ?', (service_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Service deleted successfully'})

# ==================== 聊天记录 ====================

@app.route('/api/services/<int:service_id>/chat', methods=['GET'])
def get_chat_history(service_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM chat_history WHERE service_id = ? ORDER BY created_at ASC
    ''', (service_id,))
    messages = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(messages)

@app.route('/api/services/<int:service_id>/chat', methods=['POST'])
def add_chat_message(service_id):
    data = request.json
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chat_history (service_id, role, content)
        VALUES (?, ?, ?)
    ''', (service_id, data['role'], data['content']))
    conn.commit()
    message_id = cursor.lastrowid
    conn.close()
    return jsonify({'id': message_id, 'message': 'Message added successfully'}), 201

@app.route('/api/services/<int:service_id>/chat', methods=['DELETE'])
def clear_chat_history(service_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM chat_history WHERE service_id = ?', (service_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Chat history cleared successfully'})

@app.route('/api/services/<int:service_id>/chat/completions', methods=['POST'])
def chat_completions(service_id):
    """调用大模型API（代理到目标服务器的8000端口）"""
    try:
        # 获取服务信息
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM services WHERE id = ?', (service_id,))
        service = dict(cursor.fetchone())
        conn.close()
        
        if not service:
            return jsonify({'error': 'Service not found'}), 404
        
        # 获取服务器的IP地址
        server_ids = json.loads(service['server_ids']) if service['server_ids'] else []
        if not server_ids:
            return jsonify({'error': 'No servers associated with this service'}), 400
        
        # 获取第一个服务器的信息
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM servers WHERE id = ?', (server_ids[0],))
        server = dict(cursor.fetchone())
        conn.close()
        
        if not server:
            return jsonify({'error': 'Server not found'}), 404
        
        # 构建目标URL
        server_ip = server['host_ip']
        target_url = f"http://{server_ip}:8000/chat/completions"
        
        # 获取请求数据
        request_data = request.json or {}
        
        # 设置默认值
        if 'model' not in request_data:
            request_data['model'] = 'jiuge'
        
        # 转发请求到目标服务器
        response = requests.post(
            target_url,
            json=request_data,
            headers={'Content-Type': 'application/json'},
            timeout=120,
            stream=request_data.get('stream', False)
        )
        
        # 如果是流式响应，需要特殊处理
        if request_data.get('stream', False):
            from flask import Response
            def generate():
                for line in response.iter_lines():
                    if line:
                        # 确保每一行都以 data: 开头（SSE格式）
                        line_str = line.decode('utf-8', errors='ignore')
                        if not line_str.startswith('data: '):
                            yield f'data: {line_str}\n\n'.encode('utf-8')
                        else:
                            yield f'{line_str}\n\n'.encode('utf-8')
            return Response(
                generate(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'  # 禁用nginx缓冲
                }
            )
        else:
            # 非流式响应，直接返回JSON
            if response.status_code == 200:
                return jsonify(response.json())
            else:
                return jsonify({'error': f'API request failed: {response.text}'}), response.status_code
                
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Connection failed: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== 计划任务 ====================

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT t.*, s.name as server_name 
        FROM tasks t
        LEFT JOIN servers s ON t.server_id = s.id
        ORDER BY t.created_at DESC
    ''')
    tasks = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(tasks)

@app.route('/api/tasks', methods=['POST'])
def create_task():
    data = request.json
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO tasks (name, command, server_id, schedule_type)
        VALUES (?, ?, ?, ?)
    ''', (data['name'], data['command'], data['server_id'], data['schedule_type']))
    conn.commit()
    task_id = cursor.lastrowid
    conn.close()
    return jsonify({'id': task_id, 'message': 'Task created successfully'}), 201

@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    data = request.json
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE tasks SET name = ?, command = ?, server_id = ?, schedule_type = ? WHERE id = ?
    ''', (data['name'], data['command'], data['server_id'], data['schedule_type'], task_id))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Task updated successfully'})

@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM tasks WHERE id = ?', (task_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Task deleted successfully'})

@app.route('/api/tasks/<int:task_id>/execute', methods=['POST'])
def execute_task(task_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT t.*, s.* FROM tasks t JOIN servers s ON t.server_id = s.id WHERE t.id = ?', (task_id,))
    task_row = cursor.fetchone()
    if not task_row:
        return jsonify({'error': 'Task not found'}), 404
    
    task = dict(task_row)
    conn.close()
    
    # 更新任务状态
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('UPDATE tasks SET status = ?, last_run = ? WHERE id = ?', 
                   ('executing', datetime.now(), task_id))
    conn.commit()
    conn.close()
    
    # 执行任务（这里简化处理，实际应该使用后台任务队列）
    try:
        result = execute_ssh_command(task, task['command'])
        result_data = {
            'output': result['output'], 
            'error': result['error'], 
            'success': result['success']
        }
        result_json = json.dumps(result_data, ensure_ascii=False)
        
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('UPDATE tasks SET status = ?, result = ? WHERE id = ?', 
                      ('completed', result_json, task_id))
        conn.commit()
        conn.close()
        
        return jsonify(result_data)
    except Exception as e:
        error_message = str(e)
        result_data = {'output': '', 'error': error_message, 'success': False}
        result_json = json.dumps(result_data, ensure_ascii=False)
        
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('UPDATE tasks SET status = ?, result = ? WHERE id = ?', 
                      ('failed', result_json, task_id))
        conn.commit()
        conn.close()
        return jsonify(result_data), 500

@app.route('/api/tasks/<int:task_id>/result', methods=['GET'])
def get_task_result(task_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT result, last_run FROM tasks WHERE id = ?', (task_id,))
    task_row = cursor.fetchone()
    conn.close()
    
    if not task_row:
        return jsonify({'error': 'Task not found'}), 404
    
    result_str = task_row['result']
    last_run = task_row['last_run']
    
    response_data = {
        'last_run': last_run
    }
    
    if result_str:
        try:
            result_data = json.loads(result_str)
            response_data.update(result_data)
            return jsonify(response_data)
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid result data', 'raw': result_str, 'last_run': last_run}), 500
    else:
        response_data['message'] = 'No result available yet'
        return jsonify(response_data)
    return jsonify({'message': 'Task result stored in database'})

# ==================== 统计信息 ====================

@app.route('/api/stats', methods=['GET'])
def get_stats():
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) as count FROM servers')
    server_count = dict(cursor.fetchone())['count']
    
    cursor.execute('SELECT COUNT(*) as count FROM services')
    service_count = dict(cursor.fetchone())['count']
    
    cursor.execute('SELECT COUNT(*) as count FROM servers WHERE status = ?', ('online',))
    online_server_count = dict(cursor.fetchone())['count']
    
    conn.close()
    return jsonify({
        'server_count': server_count,
        'service_count': service_count,
        'online_server_count': online_server_count
    })

# ==================== SSH WebSocket终端 ====================

@socketio.on('ssh_connect')
def handle_ssh_connect(data):
    server_id = data['server_id']
    auto_command = data.get('auto_command', None)  # 可选：自动执行的命令
    service_id = data.get('service_id', None)  # 服务ID，用于持久化连接
    session_id = request.sid
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM servers WHERE id = ?', (server_id,))
    server = dict(cursor.fetchone())
    conn.close()
    
    try:
        # 如果是服务连接，检查是否已有持久化连接
        if service_id and service_id in service_ssh_connections:
            # 复用已有连接
            conn_info = service_ssh_connections[service_id]
            conn_info['sessions'].append(session_id)
            
            # 创建新的channel用于此会话（每个会话需要独立的channel）
            chan = conn_info['ssh'].invoke_shell(term='xterm-256color')
            chan.settimeout(0.1)
            
            ssh_connections[session_id] = {
                'ssh': conn_info['ssh'],
                'chan': chan,
                'server_id': server_id,
                'service_id': service_id,
                'persistent': True
            }
            
            emit('ssh_connected', {'status': 'connected', 'reused': True})
            
            # 启动接收线程
            threading.Thread(target=ssh_receive_thread, args=(session_id,), daemon=True).start()
        else:
            # 创建新连接
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(server['host_ip'], port=server['port'], 
                       username=server['username'], password=server.get('password'))
            
            chan = ssh.invoke_shell(term='xterm-256color')
            chan.settimeout(0.1)
            
            ssh_connections[session_id] = {
                'ssh': ssh,
                'chan': chan,
                'server_id': server_id,
                'service_id': service_id,
                'persistent': service_id is not None
            }
            
            # 如果是服务连接，保存到持久化连接池
            if service_id:
                service_ssh_connections[service_id] = {
                    'ssh': ssh,
                    'chan': chan,  # 主channel
                    'server_id': server_id,
                    'sessions': [session_id]
                }
            
            emit('ssh_connected', {'status': 'connected'})
            
            # 如果有自动执行的命令，等待shell准备就绪后执行
            if auto_command:
                # 等待shell准备就绪（通常需要等待一下）
                time.sleep(0.5)
                chan.send(auto_command + '\r\n')
            
            # 启动接收线程
            threading.Thread(target=ssh_receive_thread, args=(session_id,), daemon=True).start()
    except Exception as e:
        emit('ssh_error', {'error': str(e)})

def ssh_receive_thread(session_id):
    if session_id not in ssh_connections:
        return
    
    chan = ssh_connections[session_id]['chan']
    
    try:
        while session_id in ssh_connections:
            try:
                if chan.recv_ready():
                    data = chan.recv(4096)
                    socketio.emit('ssh_output', {'data': data.decode('utf-8', errors='ignore')}, room=session_id)
                else:
                    time.sleep(0.05)  # 减少延迟以提高响应速度
            except paramiko.ssh_exception.SSHException:
                break
            except Exception as e:
                if session_id in ssh_connections:
                    socketio.emit('ssh_error', {'error': str(e)}, room=session_id)
                break
    except Exception:
        pass
    finally:
        # 清理连接
        if session_id in ssh_connections:
            try:
                ssh_connections[session_id]['chan'].close()
                ssh_connections[session_id]['ssh'].close()
            except:
                pass
            del ssh_connections[session_id]

@socketio.on('ssh_input')
def handle_ssh_input(data):
    session_id = request.sid
    if session_id in ssh_connections:
        try:
            ssh_connections[session_id]['chan'].send(data['input'])
        except:
            pass

@socketio.on('ssh_disconnect')
def handle_ssh_disconnect():
    session_id = request.sid
    if session_id in ssh_connections:
        conn_info = ssh_connections[session_id]
        service_id = conn_info.get('service_id')
        is_persistent = conn_info.get('persistent', False)
        
        try:
            # 只关闭这个会话的channel
            conn_info['chan'].close()
            
            # 如果是持久化连接，只移除session，不断开SSH连接
            if is_persistent and service_id and service_id in service_ssh_connections:
                service_conn = service_ssh_connections[service_id]
                if session_id in service_conn['sessions']:
                    service_conn['sessions'].remove(session_id)
                # 如果所有session都断开了，可以选择保持连接或关闭
                # 这里选择保持连接，以便后续复用
            else:
                # 临时连接，完全关闭
                conn_info['ssh'].close()
        except:
            pass
        del ssh_connections[session_id]
    emit('ssh_disconnected', {'status': 'disconnected'})

@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    if session_id in ssh_connections:
        conn_info = ssh_connections[session_id]
        service_id = conn_info.get('service_id')
        is_persistent = conn_info.get('persistent', False)
        
        try:
            # 只关闭这个会话的channel
            conn_info['chan'].close()
            
            # 如果是持久化连接，只移除session，不断开SSH连接
            if is_persistent and service_id and service_id in service_ssh_connections:
                service_conn = service_ssh_connections[service_id]
                if session_id in service_conn['sessions']:
                    service_conn['sessions'].remove(session_id)
            else:
                # 临时连接，完全关闭
                conn_info['ssh'].close()
        except:
            pass
        del ssh_connections[session_id]

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000, host='0.0.0.0')

