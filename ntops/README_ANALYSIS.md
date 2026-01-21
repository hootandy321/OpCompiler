# 📂 `ntops/` 算子库架构全景

## 1. 子系统职责

**ntops** (NineToothed OPS) 是基于 **ninetoothed** 编译器构建的高性能深度学习算子库。它为 Infini 生态系统提供了一系列优化的GPU算子,覆盖从基础数学运算到复杂的Transformer核心算子。

该模块在整个 Infini 架构中的作用:
- **算子实现层**: 将ninetoothed的符号化编译能力封装为可用的算子API
- **PyTorch兼容层**: 提供与PyTorch API兼容的接口,便于无缝集成
- **性能优化层**: 通过自动调优和内存布局优化提供超越原生PyTorch的性能

## 2. 模块导航

### 核心内核实现 (`src/ntops/kernels/`)

**目录结构**:
```
kernels/
├── __init__.py              # 内核导出接口
├── element_wise.py          # 逐元素算子基类
├── reduction.py             # 归约算子
├── add.py, mul.py, sub.py... # 基础算术算子
├── relu.py, gelu.py, silu.py... # 激活函数
├── layer_norm.py, rms_norm.py  # 归一化算子
├── matmul.py, bmm.py, mm.py    # 矩阵乘法
├── softmax.py, sigmoid.py...   # 数学函数
├── scaled_dot_product_attention.py  # 注意力机制
└── rotary_position_embedding.py     # 旋转位置编码
```

**设计模式**:
- **三段式结构**: 每个算子包含 `arrangement`, `application`, `tensors` 三部分
- **premake函数**: 返回 `(arrangement_func, application_func, tensors)` 三元组
- **模块化**: 基础算子可组合形成复杂算子

**核心组件**:

#### 2.1 逐元素算子 (`element_wise.py`)
- **arrangement函数**: 将输入张量展平并分块
  ```python
  def arrangement(*tensors, block_size=None):
      return tuple(
          tensor.flatten().tile((block_size,)) if tensor.ndim != 0 else tensor
          for tensor in tensors
      )
  ```
- **特点**: 适用于所有逐元素操作(加、减、乘、除、激活函数等)

#### 2.2 矩阵乘法 (`mm.py`, `bmm.py`)
- **分块策略**: 使用可配置的 `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`
- **精度变体**: 支持 IEEE float32 和 TF32 两种精度模式
- **融合操作**: `addmm` 实现矩阵乘法加法融合

#### 2.3 注意力机制 (`scaled_dot_product_attention.py`)
- **复杂布局**: 多级分块和广播操作
- **KV缓存**: 支持带KV缓存的推理模式
- **因果掩码**: 支持 `UPPER_LEFT` 和 `LOWER_RIGHT` 两种因果变体
- **参数**:
  - `query`, `key`, `value`: 注意力的三要素
  - `attn_mask`: 注意力掩码
  - `is_causal`: 是否使用因果掩码
  - `scale`: 缩放因子

#### 2.4 归一化算子 (`layer_norm.py`, `rms_norm.py`)
- **LayerNorm**: 完整的层归一化实现,支持可学习参数
- **RMSNorm**: 更简化的根均方归一化
- **优化策略**: 沿归一化维度分块,减少全局内存访问

### PyTorch接口封装 (`src/ntops/torch/`)

**目录结构**:
```
torch/
├── __init__.py       # PyTorch接口导出
├── utils.py          # 工具函数和缓存
├── add.py            # torch.add兼容接口
├── mul.py            # torch.mul兼容接口
├── matmul.py         # torch.matmul兼容接口
└── ...               # 其他算子的PyTorch封装
```

**设计模式**:

#### 统一接口模式
每个算子遵循相同的封装模式:
```python
def op_name(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.op_name.premake, input.ndim)
    kernel(input, other, out)

    return out
```

#### 关键特性
- **API兼容**: 完全兼容PyTorch函数签名
- **自动内存管理**: 自动创建输出张量(如果未提供)
- **内核缓存**: `_cached_make` 确保相同配置只编译一次
- **类型保持**: 输出张量保持输入张量的设备和数据类型

#### 工具函数 (`utils.py`)

**内核缓存机制**:
```python
@functools.cache
def _cached_make(premake, *args, num_warps, num_stages, max_num_configs, **keywords):
    return ninetoothed.make(
        *premake(*args, **keywords),
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=max_num_configs,
    )
```

**全局配置管理**:
- `set_default_num_warps(num_warps)`: 设置默认warp数
- `set_default_num_stages(num_stages)`: 设置默认流水线阶段数
- `set_default_max_num_configs(max_num_configs)`: 设置最大调优配置数

**精度适配**:
```python
def _get_matmul_input_precision():
    if torch.get_float32_matmul_precision() == "highest":
        return ntops.kernels.mm.InputPrecisionVariant.IEEE
    return ntops.kernels.mm.InputPrecisionVariant.TF32
```

### 测试套件 (`tests/`)

**测试覆盖**:
- **基础算术**: `test_add.py`, `test_sub.py`, `test_mul.py`, `test_div.py`
- **比较运算**: `test_eq.py`, `test_lt.py`, `test_gt.py`, `test_le.py`, `test_ge.py`, `test_ne.py`
- **数学函数**: `test_sin.py`, `test_cos.py`, `test_exp.py`, `test_pow.py`, `test_tanh.py`
- **激活函数**: `test_relu.py`, `test_gelu.py`, `test_sigmoid.py`
- **归一化**: `test_layer_norm.py`, `test_rms_norm.py`
- **矩阵运算**: `test_mm.py`, `test_bmm.py`, `test_addmm.py`, `test_matmul.py`
- **特殊函数**: `test_clamp.py`, `test_dropout.py`, `test_softmax.py`
- **高级算子**: `test_scaled_dot_product_attention.py`
- **位运算**: `test_bitwise_and.py`, `test_bitwise_or.py`, `test_bitwise_not.py`
- **特殊值**: `test_isinf.py`, `test_isnan.py`

**测试工具** (`conftest.py`, `skippers.py`, `utils.py`):
- 设备检测(CPU/CUDA)
- 随机种子管理
- 条件跳过(如无GPU时跳过CUDA测试)

## 3. 架构逻辑图解

### 3.1 双层架构设计

```
┌─────────────────────────────────────────────────────┐
│         用户代码(User Code)                         │
│  import ntops.torch as torch_ops                    │
│  result = torch_ops.add(x, y)                       │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│       PyTorch接口层 (torch/)                        │
│  ┌─────────────────────────────────────────────┐   │
│  │  API封装                                    │   │
│  │  • 兼容PyTorch函数签名                      │   │
│  │  • 自动内存管理                              │   │
│  │  • 参数验证                                  │   │
│  └─────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────┐   │
│  │  内核缓存(_cached_make)                     │   │
│  │  • functools.cache缓存编译结果               │   │
│  │  • 全局配置管理                              │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│       内核抽象层 (kernels/)                         │
│  ┌─────────────────────────────────────────────┐   │
│  │  premake函数                                │   │
│  │  返回: (arrangement, application, tensors)   │   │
│  └─────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────┐   │
│  │  arrangement函数                            │   │
│  │  • 定义内存布局(tile/expand/squeeze)         │   │
│  │  • 优化数据访问模式                          │   │
│  └─────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────┐   │
│  │  application函数                            │   │
│  │  • 定义计算逻辑                              │   │
│  │  • 与布局无关的抽象算法                      │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│       ninetoothed编译层                             │
│  ninetoothed.make(arrangement, application, tensors)│
│  • 符号化张量操作                                    │
│  • AST代码生成                                      │
│  • 自动调优                                         │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│       GPU内核执行                                    │
│  Triton编译 → PTX → GPU执行                         │
└─────────────────────────────────────────────────────┘
```

### 3.2 算子实现范式

**标准算子模板**:
```python
# 步骤1: 定义application(计算逻辑)
def application(input, other, output):
    output = input + other  # 符号化操作

# 步骤2: 定义arrangement(内存布局)
def arrangement(*tensors, block_size=None):
    return tuple(
        tensor.flatten().tile((block_size,))
        for tensor in tensors
    )

# 步骤3: 定义premake(工厂函数)
def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )
    return arrangement_, application, tensors

# 步骤4: PyTorch封装
def add(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    kernel = _cached_make(premake, input.ndim)
    kernel(input, other, out)
    return out
```

### 3.3 复杂算子示例:SDPA

**Scaled Dot-Product Attention的内存布局**:

```
输入张量:
  query: (batch, seq_q, num_heads, head_dim)
  key:   (batch, seq_k, num_heads, head_dim)
  value: (batch, seq_k, num_heads, head_dim)

步骤1: 分块
  query_arranged: (batch, num_heads, seq_q, head_dim)
                → (batch, num_heads, seq_q/BLOCK_M, BLOCK_M, head_dim)

  key_arranged:   (batch, num_heads, seq_k, head_dim)
                → (batch, num_heads, seq_k/BLOCK_N, BLOCK_N, head_dim)

  value_arranged: 同key

步骤2: 广播对齐
  query: (batch, num_heads, seq_q/BLOCK_M, BLOCK_M, 1, head_dim)
  key:   (batch, num_heads, 1, BLOCK_N, seq_k/BLOCK_N, head_dim)
  value: (batch, num_heads, 1, BLOCK_N, seq_k/BLOCK_N, head_dim)

步骤3: 应用计算
  for m in range(BLOCK_M):
      for n in range(BLOCK_N):
          attn_score = query[m] @ key[n].T  # (head_dim, head_dim)
          attn_weight = softmax(attn_score * scale)
          output[m] += attn_weight @ value[n]
```

## 4. 算子分类与特性

### 4.1 基础算术算子
| 算子 | 功能 | 特殊参数 |
|------|------|---------|
| `add` | 加法 | `alpha`(缩放因子) |
| `sub` | 减法 | - |
| `mul` | 乘法 | - |
| `div` | 除法 | - |
| `addmm` | 矩阵乘加融合 | `beta`(输出缩放), `alpha`(矩阵缩放) |

### 4.2 比较运算算子
| 算子 | 功能 | 输出类型 |
|------|------|---------|
| `eq` | 等于 | bool |
| `lt`/`le` | 小于/小于等于 | bool |
| `gt`/`ge` | 大于/大于等于 | bool |
| `ne` | 不等于 | bool |

### 4.3 数学函数算子
| 算子 | 函数 | 数值稳定性 |
|------|------|-----------|
| `exp` | 指数 | 需处理溢出 |
| `sin/cos` | 三角函数 | 直接映射libdevice |
| `pow` | 幂运算 | 支持整数和浮点指数 |
| `rsqrt` | 平方根倒数 | 常用于LayerNorm |

### 4.4 激活函数算子
| 算子 | 公式 | 近似模式 |
|------|------|---------|
| `relu` | `max(0, x)` | - |
| `gelu` | `x * Φ(x)` | 支持`"tanh"`近似 |
| `silu` | `x / (1 + e^(-x))` | - |
| `sigmoid` | `1 / (1 + e^(-x))` | - |
| `tanh` | 双曲正切 | 直接映射libdevice |

### 4.5 归一化算子
| 算子 | 归一化维度 | 可学习参数 |
|------|-----------|-----------|
| `layer_norm` | 最后C维 | weight, bias |
| `rms_norm` | 最后C维 | weight |

**LayerNorm计算**:
```python
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True)
output = (x - mean) / sqrt(var + eps) * weight + bias
```

**RMSNorm计算**:
```python
rms = sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
output = (x / rms) * weight
```

### 4.6 矩阵运算算子
| 算子 | 输入形状 | 输出形状 |
|------|---------|---------|
| `mm` | (M, K), (K, N) | (M, N) |
| `bmm` | (B, M, K), (B, K, N) | (B, M, N) |
| `matmul` | 广播 | 广播 |

**优化特性**:
- **分块策略**: 可配置的块大小(BLOCK_SIZE_M/N/K)
- **精度模式**: IEEE float32 或 TF32
- **融合**: `addmm`融合矩阵乘法和加法

### 4.7 特殊算子

#### Dropout (`dropout.py`)
- **训练模式**: 随机mask,按概率置零
- **推理模式**: 恒等映射
- **实现**: 通过 `training` 参数切换

#### Softmax (`softmax.py`)
- **数值稳定**: 减去最大值避免溢出
- **支持多维度**: 沿指定维度归一化

#### Clamp (`clamp.py`)
- **截断**: 将值限制在[min, max]范围内
- **用途**: 激活函数裁剪、梯度裁剪

#### SDPA (`scaled_dot_product_attention.py`)
**特性**:
- **KV缓存**: 支持增量推理
- **因果掩码**: 自回归生成
- **注意力掩码**: 灵活的padding和future masking

#### RoPE (`rotary_position_embedding.py`)
- **旋转位置编码**: 增强Transformer的位置感知
- **融合**: 与注意力计算融合

## 5. 性能优化策略

### 5.1 内存布局优化
- **分块(Tile)**: 将大张量分解为小块,提高缓存利用率
- **向量化**: 利用GPU SIMT架构
- **合并访问**: 确保内存访问合并

### 5.2 计算优化
- **内核融合**: 多个操作融合为单个内核,减少内存访问
- **流水线(Pipeline)**: 隐藏内存延迟
- **自动调优**: 根据硬件自动选择最优配置

### 5.3 编译优化
- **内核缓存**: 相同配置只编译一次
- **符号计算**: 编译时求常量
- **函数内联**: 减少函数调用开销

## 6. 与PyTorch的对比

| 特性 | PyTorch原生 | ntops |
|------|------------|-------|
| **性能** | 通用优化 | 针对特定形状优化 |
| **灵活性** | 完全动态 | 需要固定维度(编译时) |
| **调优** | 手动调整 | 自动调优 |
| **API** | 标准PyTorch API | 完全兼容 |
| **后端** | ATen/CUDA | Triton |

## 7. 使用示例

### 7.1 基础算子
```python
import ntops.torch as torch_ops
import torch

x = torch.randn(1024, device='cuda')
y = torch.randn(1024, device='cuda')

# 使用ntops加速的加法
z = torch_ops.add(x, y)
```

### 7.2 矩阵乘法
```python
A = torch.randn(512, 512, device='cuda')
B = torch.randn(512, 512, device='cuda')

# 自动调优的矩阵乘法
C = torch_ops.bmm(A.unsqueeze(0), B.unsqueeze(0)).squeeze(0)
```

### 7.3 注意力机制
```python
query = torch.randn(2, 8, 128, 64, device='cuda')  # (batch, heads, seq, dim)
key = torch.randn(2, 8, 128, 64, device='cuda')
value = torch.randn(2, 8, 128, 64, device='cuda')

output = torch_ops.scaled_dot_product_attention(
    query, key, value,
    is_causal=True,  # 自回归
    scale=0.125
)
```

### 7.4 配置调优参数
```python
from ntops.torch.utils import set_default_num_warps

# 设置默认配置
set_default_num_warps(8)

# 后续算子使用该配置
result = torch_ops.add(x, y)
```

## 8. 扩展指南

### 8.1 添加新算子

**步骤1**: 在`src/ntops/kernels/`创建内核实现
```python
# new_op.py
def application(input, output):
    output = ...  # 计算逻辑

def arrangement(*tensors, block_size=None):
    return tuple(
        tensor.flatten().tile((block_size,))
        for tensor in tensors
    )

def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))
    return arrangement_, application, tensors
```

**步骤2**: 在`src/ntops/kernels/__init__.py`导出
```python
from ntops.kernels.new_op import new_op
```

**步骤3**: 在`src/ntops/torch/`创建PyTorch接口
```python
# torch/new_op.py
import torch
import ntops
from ntops.torch.utils import _cached_make

def new_op(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    kernel = _cached_make(ntops.kernels.new_op.premake, input.ndim)
    kernel(input, out)
    return out
```

**步骤4**: 在`src/ntops/torch/__init__.py`导出
```python
from ntops.torch.new_op import new_op
```

**步骤5**: 添加测试
```python
# tests/test_new_op.py
def test_new_op():
    input = torch.randn(1024, device='cuda')
    output = ntops.torch.new_op(input)
    expected = torch.new_op(input)
    assert torch.allclose(output, expected)
```

### 8.2 调试算子

**启用详细日志**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**可视化内存布局**:
```python
from ninetoothed import visualization

# 在内核实现中添加
visualization.visualize(tensor)
```

**符号求值**:
```python
from ninetoothed import eval

# 检查符号张量的实际布局
result = eval(tensor, {block_size: 64})
print(result)
```

## 9. 限制与注意事项

### 9.1 编译时限制
- **固定形状**: 张量维度必须在编译时确定
- **首次调用慢**: JIT编译有启动开销(通过缓存缓解)

### 9.2 功能限制
- **不支持动态控制流**: 如`if`条件依赖于运行时值
- **不支持复杂索引**: 如高级索引

### 9.3 性能考量
- **小张量性能差**: 内核启动开销可能超过计算收益
- **建议**: 对小张量使用PyTorch原生操作

## 10. 未来方向

- **更多算子**: 持续添加常用深度学习算子
- **性能优化**: 针对新GPU架构优化
- **混合精度**: 更好的FP8/BF16支持
- **分布式**: 多GPU和节点间通信算子

---

**相关文档**:
- [ninetoothed编译器文档](../ninetoothed/README_ANALYSIS.md)
- [ntops API参考](https://github.com/InfiniTensor/ntops)

**最后更新**: 2025-01-14
