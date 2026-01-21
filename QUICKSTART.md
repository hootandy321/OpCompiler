# Infini 环境安装完成！

## ✅ 安装成功

所有组件已成功安装并验证：

### 已安装的组件

| 组件 | 版本 | 状态 |
|------|------|------|
| **ninetoothed** (九齿编译器) | 0.23.0 | ✅ |
| **ntops** (九齿算子库) | 0.1.0 | ✅ |
| **PyTorch** | 2.9.1+cu128 | ✅ |
| **InfiniCore 底层库** | - | ✅ |
| **InfiniCore C++ 库** | - | ✅ |
| **InfiniCore Python包** | - | ✅ |
| **CUDA支持** | 12.8 | ✅ |

### 已安装的库文件

```
~/.infini/lib/
├── libinfinicore_cpp_api.so   (1.9 MB)
├── libinfiniop.so             (455 KB)
├── libinfinirt.so             (14 KB)
└── libinfiniccl.so            (14 KB)
```

## 🚀 快速开始

### 1. 激活环境

**每次使用前需要激活环境**：

```bash
source /home/qy/src/Infini/activate_infini_env.sh
```

或者手动激活：

```bash
# 激活虚拟环境
source /home/qy/src/Infini/infini_venv/bin/activate

# 设置环境变量
export INFINI_ROOT="$HOME/.infini"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$LD_LIBRARY_PATH"
```

### 2. 运行 Benchmark 测试

```bash
# 激活环境
source /home/qy/src/Infini/activate_infini_env.sh

# 进入测试目录
cd /home/qy/src/Infini/ninetoothed/tests

# 运行演示benchmark（不需要GPU）
python benchmark_demo.py

# 查看生成的报告
cat benchmark_reports/demo_report.md
```

### 3. 使用九齿开发算子

```python
import ninetoothed as ntl
from ninetoothed import Tensor, block_size

# 定义分块大小
BLOCK_SIZE = block_size(lower_bound=64, upper_bound=128)

def arrangement(x, y, output):
    x_arranged = x.tile((BLOCK_SIZE,))
    y_arranged = y.tile((BLOCK_SIZE,))
    output_arranged = output.tile((BLOCK_SIZE,))
    return x_arranged, y_arranged, output_arranged

def application(x, y, output):
    output[:] = x + y

tensors = (Tensor(1), Tensor(1), Tensor(1))

# 生成融合kernel
kernel = ntl.make(arrangement, application, tensors)
```

### 4. 使用 ntops 算子库

```python
import ntops.torch as torch_ops
import torch

# 创建张量
x = torch.randn(1024, 1024, device='cuda')
y = torch.randn(1024, 1024, device='cuda')

# 使用ntops的加法算子（融合优化）
z = torch_ops.add(x, y)

# 或者使用融合的矩阵乘法
a = torch.randn(512, 512, device='cuda')
b = torch.randn(512, 512, device='cuda')
c = torch.randn(512, 512, device='cuda')
result = torch_ops.addmm(c, a, b)  # c + a@b
```

## 📂 重要目录

```
/home/qy/src/Infini/
├── ninetoothed/              # 九齿编译器
│   └── tests/
│       └── benchmark_*.py   # Benchmark脚本
├── ntops/                    # 九齿算子库
├── InfiniCore/               # 核心框架
├── infini_venv/              # Python虚拟环境
├── activate_infini_env.sh    # 快速激活脚本
├── setup_infini_env.sh       # 安装脚本
└── SETUP_GUIDE.md            # 详细安装指南
```

## 🔍 验证安装

运行以下命令验证所有组件：

```bash
source /home/qy/src/Infini/activate_infini_env.sh

# 验证Python包
python -c "
import ninetoothed
import ntops
import torch
print('✓ ninetoothed: OK')
print('✓ ntops: OK')
print('✓ PyTorch:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
"

# 验证库文件
ls -lh $INFINI_ROOT/lib/
```

## 📚 学习资源

### 官方文档

- **ninetoothed**: `/home/qy/src/Infini/ninetoothed/README.md`
- **ntops**: `/home/qy/src/Infini/ntops/README_ANALYSIS.md`
- **InfiniCore**: `/home/qy/src/Infini/InfiniCore/README.md`
- **项目架构**: `/home/qy/src/Infini/README_ANALYSIS.md`

### Benchmark相关

- **Benchmark演示**: `/home/qy/src/Infini/ninetoothed/tests/benchmark_demo.py`
- **Benchmark文档**: `/home/qy/src/Infini/ninetoothed/tests/BENCHMARK_README.md`
- **项目总结**: `/home/qy/src/Infini/ninetoothed/tests/BENCHMARK_SUMMARY.md`

## 🛠️ 常用操作

### 更新九齿算子

如果您修改了ninetoothed或ntops的代码：

```bash
source /home/qy/src/Infini/activate_infini_env.sh

# 重新安装（开发者模式）
cd /home/qy/src/Infini/ninetoothed
pip install -e .

cd /home/qy/src/Infini/ntops
pip install -e .
```

### 重新编译InfiniCore

```bash
cd /home/qy/src/Infini/InfiniCore

# 重新编译底层库
python scripts/install.py

# 重新编译C++库
xmake build _infinicore
xmake install _infinicore

# 重新安装Python包
pip install -e .
```

### 运行测试

```bash
# InfiniCore测试
cd /home/qy/src/Infini/InfiniCore
python test/infinicore/run.py --cpu

# ninetoothed测试
cd /home/qy/src/Infini/ninetoothed
pytest tests/

# ntops测试
cd /home/qy/src/Infini/ntops
pytest tests/
```

## 💡 提示

1. **每次打开新终端**，都需要运行 `source /home/qy/src/Infini/activate_infini_env.sh`

2. **退出虚拟环境**：运行 `deactivate`

3. **永久保存环境变量**（可选）：
   ```bash
   echo 'source /home/qy/src/Infini/activate_infini_env.sh' >> ~/.bashrc
   ```

4. **查看已安装的包**：
   ```bash
   pip list | grep -E "(ninetoothed|ntops|torch|infinicore)"
   ```

## 🆘 遇到问题？

### 环境未激活

**错误**: `ModuleNotFoundError: No module named 'ninetoothed'`

**解决**:
```bash
source /home/qy/src/Infini/activate_infini_env.sh
```

### CUDA不可用

**错误**: `CUDA not available`

**解决**: 检查CUDA安装
```bash
nvidia-smi
echo $CUDA_HOME
```

### 库文件找不到

**错误**: `error while loading shared libraries: libinfiniop.so`

**解决**:
```bash
export LD_LIBRARY_PATH="$HOME/.infini/lib:$LD_LIBRARY_PATH"
```

## 🎯 下一步

现在您可以：

1. ✅ **运行benchmark**：测试九齿算子融合性能
2. ✅ **开发算子**：使用九齿编译器编写自定义算子
3. ✅ **使用ntops**：直接使用优化的九齿算子库
4. ✅ **阅读文档**：深入学习各个组件的使用方法

---

**祝您开发愉快！** 🎉

如有问题，请参考各项目的README文档或提交issue。
