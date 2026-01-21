# Infini 项目环境安装指南

本指南帮助您快速配置完整的Infini开发环境。

## 快速开始

### 方法1: 使用自动化脚本（推荐）

```bash
cd /home/qy/src/Infini
bash setup_infini_env.sh
```

脚本会自动完成：
1. ✅ 创建Python虚拟环境
2. ✅ 配置环境变量（INFINI_ROOT, LD_LIBRARY_PATH）
3. ✅ 安装ninetoothed（九齿编译器）
4. ✅ 安装ntops（九齿算子库）
5. ✅ 安装InfiniCore底层库
6. ✅ 安装InfiniCore C++库
7. ✅ 安装InfiniCore Python包

### 方法2: 手动逐步安装

如果自动脚本失败，可以参考以下步骤手动安装：

## 详细安装步骤

### 前置要求

- **Python**: 3.10+
- **编译器**: gcc-11+ 或 clang-16+
- **构建工具**: [XMake](https://xmake.io/)
- **GPU环境**（可选）: CUDA Toolkit

### 1. 创建虚拟环境

```bash
cd /home/qy/src/Infini
python3 -m venv infini_venv
source infini_venv/bin/activate
```

### 2. 配置环境变量

```bash
export INFINI_ROOT="$HOME/.infini"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$LD_LIBRARY_PATH"
mkdir -p $INFINI_ROOT/{lib,bin,include}
```

**永久保存到 ~/.bashrc（可选）**:
```bash
echo 'export INFINI_ROOT="$HOME/.infini"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 3. 安装 ninetoothed

```bash
cd /home/qy/src/Infini/ninetoothed
pip install --upgrade pip
pip install -e .
```

**验证**:
```python
python -c "import ninetoothed; print(ninetoothed.__version__)"
```

### 4. 安装 ntops

```bash
cd /home/qy/src/Infini/ntops
pip install -e .
```

**验证**:
```python
python -c "import ntops; print('ntops loaded successfully')"
```

### 5. 安装 InfiniCore 底层库

**仅CPU版本**:
```bash
cd /home/qy/src/Infini/InfiniCore
python scripts/install.py
```

**包含GPU支持**:
```bash
cd /home/qy/src/Infini/InfiniCore
python scripts/install.py --nv-gpu=y --cuda=$CUDA_HOME
```

**手动安装（备选）**:
```bash
# 配置
xmake f -cv                    # CPU
xmake f --nv-gpu=y --cuda=$CUDA_HOME -cv  # NVIDIA GPU

# 编译安装
xmake build
xmake install
```

### 6. 安装 InfiniCore C++ 库

```bash
cd /home/qy/src/Infini/InfiniCore
xmake build _infinicore
xmake install _infinicore
```

### 7. 安装 InfiniCore Python 包

```bash
cd /home/qy/src/Infini/InfiniCore
pip install -e .
```

**验证**:
```python
python -c "import infinicore; print('InfiniCore loaded successfully')"
```

## 安装验证

### 快速验证

```bash
# 激活虚拟环境
source /home/qy/src/Infini/infini_venv/bin/activate

# 检查Python包
pip list | grep -E "(ninetoothed|ntops|infinicore)"

# 运行基础测试
cd /home/qy/src/Infini/InfiniCore
python test/infinicore/run.py --cpu
```

### 完整测试

```bash
# 运行所有算子测试
cd /home/qy/src/Infini/InfiniCore
python test/infinicore/run.py --cpu --verbose
```

## 目录结构

安装完成后，您的目录结构应该是：

```
/home/qy/src/Infini/
├── ninetoothed/          # 九齿编译器源码
├── ntops/                # 九齿算子库源码
├── InfiniCore/           # InfiniCore源码
├── infini_venv/          # Python虚拟环境
└── setup_infini_env.sh   # 安装脚本

~/.infini/                # INFINI_ROOT
├── lib/                  # 编译好的库文件
├── bin/                  # 可执行文件
└── include/              # 头文件
```

## 常见问题

### 1. 虚拟环境激活失败

**错误**: `Command not found: activate`

**解决**:
```bash
# 确保使用正确的路径
source /home/qy/src/Infini/infini_venv/bin/activate
```

### 2. CUDA相关错误

**错误**: `CUDA_HOME not set`

**解决**:
```bash
# 查找CUDA路径
which nvcc
export CUDA_HOME=/usr/local/cuda  # 根据实际路径调整
```

### 3. XMake未安装

**解决**:
```bash
# 安装XMake
bash <(curl -L https://xmake.io/shget.sh)
```

### 4. 权限错误

**错误**: `Permission denied when writing to $INFINI_ROOT`

**解决**:
```bash
# 确保目录存在且有写权限
mkdir -p $HOME/.infini
chmod u+w $HOME/.infini
```

### 5. Python包导入失败

**错误**: `ImportError: No module named 'xxx'`

**解决**:
```bash
# 确保虚拟环境已激活
source /home/qy/src/Infini/infini_venv/bin/activate

# 重新安装
pip install -e .
```

## 卸载

如需完全卸载：

```bash
# 1. 停用虚拟环境
deactivate

# 2. 删除虚拟环境
rm -rf /home/qy/src/Infini/infini_venv

# 3. 删除安装文件
rm -rf $HOME/.infini

# 4. 从.bashrc中移除环境变量
# 编辑 ~/.bashrc，删除以下行：
# export INFINI_ROOT="$HOME/.infini"
# export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$LD_LIBRARY_PATH"
```

## 下一步

环境配置完成后，您可以：

1. **运行benchmark测试**:
   ```bash
   cd /home/qy/src/Infini/ninetoothed/tests
   python benchmark_demo.py
   ```

2. **开发九齿算子**:
   - 参考ninetoothed文档
   - 参考ntops示例

3. **使用InfiniCore**:
   - 查看 `InfiniCore/test/` 中的示例
   - 阅读API文档

## 技术支持

- **文档**: 各项目的README.md和README_ANALYSIS.md
- **Issues**: https://github.com/InfiniTensor
- **测试**: 使用 `--help` 参数查看各测试脚本的选项

---

**祝您使用愉快！** 🚀
