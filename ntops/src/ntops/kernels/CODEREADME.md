# ntops.kernels 核心实现文档

本模块是 ntops (Neural Tensor Operations) 的核心计算内核集合，基于 ninetoothed 框架实现的高性能张量运算库。提供了深度学习工作负载中常见的算子实现，包括矩阵乘法、注意力机制、归一化层、激活函数和逐元素运算。

## 1. 模块结构

- **`__init__.py`**: 模块导出接口，统一暴露所有 38 个内核函数
- **`element_wise.py`**: 逐元素运算的通用内存排布策略
- **`reduction.py`**: 归约运算的通用内存排布策略
- **`scaled_dot_product_attention.py`**: 缩放点积注意力 (SDPA) 实现，支持因果掩码和 KV 缓存
- **`layer_norm.py`**: Layer Normalization 实现，支持可配置归一化维度
- **`rms_norm.py`**: Root Mean Square Normalization 实现
- **`rotary_position_embedding.py`**: 旋转位置编码 (RoPE)，支持交错和非交错模式
- **`softmax.py`**: Softmax 激活函数，使用在线算法计算以保证数值稳定性
- **`gelu.py`**: GELU 激活函数，支持精确和 tanh 近似两种模式
- **`silu.py`**: SiLU (Sigmoid Linear Unit) 激活函数
- **`dropout.py`**: Dropout 正则化，使用随机数生成器
- **`mm.py`**: 矩阵乘法 (Matrix Multiplication) 内核
- **`bmm.py`**: 批量矩阵乘法 (Batched Matrix Multiplication)
- **`addmm.py`**: 矩阵乘加融合操作 (output = input + beta × x @ y)
- **`add.py` / `sub.py` / `mul.py` / `div.py`**: 基础算术运算
- **`pow.py`**: 幂运算，使用 libdevice 实现
- **`relu.py` / `sigmoid.py` / `tanh.py` / `exp.py` / `rsqrt.py`**: 常用激活函数和数学运算
- **`sin.py` / `cos.py`**: 三角函数
- **`abs.py` / `neg.py`**: 绝对值和取负
- **`clamp.py`**: 裁剪操作
- **`eq.py` / `ne.py` / `lt.py` / `le.py` / `gt.py` / `ge.py`**: 比较运算
- **`bitwise_and.py` / `bitwise_or.py` / `bitwise_not.py`**: 位运算
- **`isnan.py` / `isinf.py`**: NaN 和 Inf 检测

## 2. 核心类与数据结构

### Tensor (来自 ninetoothed 框架)
本模块中的所有内核都基于 ninetoothed 的 Tensor 抽象，具有以下特性：
- **ndim**: 张量维度
- **dtype**: 数据类型 (float16, float32, int64 等)
- **shape**: 形状，支持 shape_options 进行编译时优化
- **tile()**: 内存分块操作，核心性能优化手段
- **offsets()**: 获取张量在源张量中的偏移量

### ninetoothed.language (ntl)
编译时的张量运算 DSL：
- **ntl.zeros() / ntl.full()**: 创建常量张量
- **ntl.dot()**: 点积运算，支持 TF32/IEEE 精度模式
- **ntl.exp() / ntl.exp2()**: 指数运算
- **ntl.sqrt() / ntl.rsqrt()**: 平方根和倒数平方根
- **ntl.maximum() / ntl.max() / ntl.sum()**: 归约运算
- **ntl.where()**: 条件选择
- **ntl.cast()**: 类型转换
- **ntl.trans()**: 矩阵转置

### InputPrecisionVariant (mm.py)
矩阵乘法的输入精度模式枚举：
- **TF32**: 使用 TensorFloat-32 精度 (A100/H100 优化)
- **IEEE**: 完整 IEEE 754 浮点精度

### CausalVariant (scaled_dot_product_attention.py)
因果掩码变体：
- **UPPER_LEFT**: 左上三角掩码（标准解码器）
- **LOWER_RIGHT**: 右下三角掩码

## 3. API 接口规范

所有内核遵循统一的三阶段接口模式：

```python
# 阶段 1: arrangement - 内存排布策略
def arrangement(*tensors, block_size=None, **kwargs):
    """
    将输入/输出张量按照 block_size 进行分块和内存重排
    返回排布后的张量元组
    """

# 阶段 2: application - 计算逻辑
def application(*tensors):
    """
    实际的计算内核逻辑
    操作排布后的张量，通常包含 for 循环进行迭代
    """

# 阶段 3: premake - 模板预配置
def premake(ndim, dtype=None, block_size=None, **kwargs):
    """
    创建模板张量和配置
    返回: (arrangement_, application, tensors)
    """
```

### 关键 API 示例

#### Scaled Dot-Product Attention
```python
def arrangement(
    query, key, value, present_key, present_value,
    present_key_slot, present_value_slot, attn_mask,
    is_causal, scale, output,
    with_attn_mask, causal_variant, with_kv_cache,
    block_size_m=None, block_size_n=None
):
    """
    配置注意力内核的内存排布
    block_size_m: Query 序列分块大小
    block_size_n: Key/Value 序列分块大小
    """

def application(
    query, key, value, attn_mask, is_causal,
    scale, output, with_attn_mask, causal_variant
):
    """
    核心 Flash Attention 算法实现
    - 使用在线算法计算 softmax (先累加再归一化)
    - 支持 causal masking
    - O(N²) 复杂度，但通过分块减少内存访问
    """
```

#### Layer Normalization
```python
def application(input, weight, bias, eps, output, num_normalized_elements):
    """
    标准化流程:
    1. 计算均值: mean = sum(input) / N
    2. 计算方差: var = sum((input - mean)²) / N
    3. 标准化: output = (input - mean) / sqrt(var + eps) * weight + bias
    """
```

#### Matrix Multiplication
```python
def application(input, other, output, input_precision):
    """
    累加器循环:
    accumulator = zeros((M, N), dtype=float32)
    for k in range(K):
        accumulator += dot(input[k], other[k], input_precision)
    output = accumulator
    """
```

#### Softmax
```python
def application(input, output):
    """
    在线 softmax 算法 (两遍扫描):
    第一遍: 找出最大值并计算分母
    第二遍: 计算最终 softmax 值
    保证数值稳定性，避免溢出
    """
```

## 4. 使用示例

### 示例 1: 使用逐元素运算
```python
from ninetoothed import Tensor
from ntops.kernels import add

# 配置阶段
arrangement_, application, tensors = add.pmake(
    ndim=2,          # 二维张量
    dtype=ninetoothed.float16,
    block_size=64    # 每个块 64 个元素
)

# 分配张量
input_tensor = Tensor((1024, 1024), dtype=ninetoothed.float16)
other_tensor = Tensor((1024, 1024), dtype=ninetoothed.float16)
output_tensor = Tensor((1024, 1024), dtype=ninetoothed.float16)

# 内存排布
input_arranged, other_arranged, _, _, output_arranged = arrangement_(
    input_tensor, other_tensor, 0.5, output_tensor
)

# 执行计算
application(input_arranged, other_arranged, 0.5, output_arranged)
```

### 示例 2: 使用 Scaled Dot-Product Attention
```python
from ntops.kernels import scaled_dot_product_attention

# 创建注意力内核配置
arrangement_, application, tensors = scaled_dot_product_attention.pmake(
    with_kv_cache=False,
    emb_dim=128,              # 嵌入维度
    is_causal=True,           # 使用因果掩码
    with_attn_mask=False,     # 无额外注意力掩码
    causal_variant=1,         # UPPER_LEFT
    dtype=ninetoothed.float16,
    block_size_m=64,
    block_size_n=64
)

query, key, value, attn_mask, output = tensors[:5]
# 填充数据...
query_arranged, key_arranged, value_arranged, attn_mask_arranged, \
is_causal, scale, output_arranged, with_attn_mask, causal_variant = \
    arrangement_(query, key, value, attn_mask, is_causal, scale, output,
                with_attn_mask, causal_variant)

# 执行 Flash Attention
application(query_arranged, key_arranged, value_arranged,
            attn_mask_arranged, is_causal, scale, output_arranged,
            with_attn_mask, causal_variant)
```

### 示例 3: 使用 Layer Normalization
```python
from ntops.kernels import layer_norm

arrangement_, application, tensors = layer_norm.pmake(
    ndim=3,                      # 三维张量 (batch, seq, hidden)
    normalized_shape=(768,),     # 归一化维度
    dtype=ninetoothed.float16,
    block_size=128
)

input, weight, bias, eps, output, num_normalized = tensors
# input: (batch_size, seq_len, 768)
# weight, bias: (768,)
# eps: 标量 epsilon 值
# num_normalized: constexpr = 768

input_arranged, weight_arranged, bias_arranged, \
eps_arranged, output_arranged, num_normalized_arranged = \
    arrangement_(input, weight, bias, eps, output, num_normalized)

application(input_arranged, weight_arranged, bias_arranged,
            eps_arranged, output_arranged, num_normalized_arranged)
```

## 5. 实现细节

### 内存管理
- **分块策略 (Tiling)**: 所有内核都基于 `block_size` 参数进行内存分块，这是性能优化的核心。默认使用 `ninetoothed.block_size()` 获取硬件最优块大小
- **内存排布**: `arrangement` 阶段通过 `tile()`, `expand()`, `squeeze()` 等操作将张量重排为适合 CUDA 并行访问的模式
- **类型转换**: 某些操作内部使用 float32 累加器（如 LayerNorm, Softmax）以保证精度，最终转换回输出类型

### 并发性
- **数据并行**: 通过 ninetoothed 框架自动处理 CUDA 线程分配，每个 block 处理一个 tile
- **无显式同步**: 依赖 ninetoothed 的隐式同步机制，避免显式 __syncthreads() 调用
- **只读访问**: 计算内核通常不修改输入张量，只写入输出张量

### 性能优化
- **在线算法**: Softmax 使用在线算法 (Welford 算法变体)，避免存储完整的中间矩阵
- **Flash Attention**: SDPA 实现采用分块注意力，通过 O(N²d) 内存减少到 O(Nd)，其中 d 是块大小
- **精度控制**: 矩阵乘法支持 TF32 模式，在 A100/H100 上提升 2-4x 性能
- **向量化**: ninetoothed 自动进行内存访问向化和指令级并行优化

### 错误处理
- **边界检查**: 通过 `offsets()` 和 `source.shape` 进行运行时边界检查，避免越界访问
- **条件填充**: 使用 `ntl.where()` 处理不规则张量形状（如序列长度不一致）
- **常量表达式**: 使用 `constexpr=True` 标记编译时常量，减少运行时开销

### 依赖关系
- **ninetoothed**: 核心张量抽象和代码生成框架
- **ninetoothed.language**: 编译时 DSL，提供 ntl 命名空间下的所有操作
- **标准库**: functools (partial 应用), math (常量), enum (枚举类)

### 设计模式
- **策略模式**: 通过 `premake` 函数支持不同的算法变体（如 GELU 的 tanh 近似 vs 精确实现）
- **模板方法模式**: arrangement → application 的两阶段设计，分离数据布局和计算逻辑
- **工厂模式**: `premake` 作为工厂函数，根据参数创建定制化的内核配置
- **组合模式**: 复杂操作 (如 addmm) 通过组合简单操作 (mm) 实现

### 算法复杂度
- **矩阵乘法**: O(M × N × K)，使用分块优化缓存局部性
- **Scaled dot-product attention**: O(M × N × D)，其中 M 是 query 序列长度，N 是 key 序列长度，D 是嵌入维度
- **Layer/RMS Norm**: O(elements)，两次遍历（计算统计量 + 应用归一化）
- **Softmax**: O(elements × reduce_dim)，两次遍历（找最大值 + 计算 exp）
- **逐元素运算**: O(elements)，单次遍历

### 特殊实现细节
1. **数值稳定性**:
   - Softmax: 使用 max(x) - x 的技巧避免 exp 溢出
   - LayerNorm: 先计算统计量再应用，避免重复计算
   - SiLU: 内部转换为 float32 计算 exp，再转回输入类型

2. **因果掩码**:
   - SDPA 支持两种因果变体，通过比较 query 和 key 的 offsets 实现掩码
   - 掩码在 QK 计算后、softmax 前应用，将被掩码位置设为 -inf

3. **旋转位置编码**:
   - 支持 interleaved 模式（交替存储 sin/cos）和 non-interleaved 模式（分开存储）
   - 使用 2D 分块和 dilation/strides 实现高效的内存访问模式

4. **Dropout**:
   - 使用 `ntl.rand(seed, offsets)` 生成随机数
   - 训练时按概率 p 置零并按 1/(1-p) 缩放，推理时直接传递输入

5. **比较运算**:
   - 所有比较运算返回布尔张量
   - isnan 通过 `x != x` 检测 NaN（IEEE 754 特性）
   - isinf 分别检测 +inf 和 -inf
