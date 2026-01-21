# `ntops.torch` PyTorch Operations Bindings Documentation

该模块实现了 PyTorch 张量操作的高性能 CUDA 内核绑定接口,通过 ninetoothed 框架提供底层内核的自动调优与缓存机制,覆盖了线性代数、激活函数、注意力机制、归一化层等核心深度学习算子。

## 1. 模块结构

- **`__init__.py`**: 模块公共接口导出,暴露全部 39 个张量操作函数
- **`utils.py`**: 内核编译缓存与配置管理,提供全局参数设置(num_warps, num_stages, max_num_configs)和矩阵乘法精度检测
- **`scaled_dot_product_attention.py`**: 缩放点积注意力(SDPA)实现,支持 GQA(分组查询注意力)、KV 缓存、因果掩码
- **`rotary_position_embedding.py`**: 旋转位置编码(RoPE)应用,支持交错与非交错模式
- **`layer_norm.py`**: Layer Normalization 层归一化实现
- **`rms_norm.py`**: RMS Normalization 均方根归一化实现
- **`dropout.py`**: Dropout 随机失活实现,支持训练/推理模式切换
- **`matmul.py`**: 通用矩阵乘法路由(2D/3D 张量分发至 mm/bmm)
- **`bmm.py`**: 批量矩阵乘法(Batch Matrix Multiplication)
- **`mm.py`**: 矩阵乘法(Matrix Multiplication)
- **`addmm.py`**: 矩阵乘法加偏置操作
- **`softmax.py`**: Softmax 激活函数
- **`relu.py`**: ReLU 激活函数
- **`gelu.py`**: GELU 激活函数(支持 tanh/approximate 近似)
- **`silu.py`**: SiLU(Swish)激活函数
- **`sigmoid.py`**: Sigmoid 激活函数
- **`tanh.py`**: Tanh 激活函数
- **`clamp.py`**: 张量值裁剪操作
- **`abs.py`, `neg.py`**: 单目算术运算符
- **`add.py`, `sub.py`, `mul.py`, `div.py`, `pow.py`**: 双目算术运算符
- **`exp.py`, `rsqrt.py`**: 数学函数
- **`sin.py`, `cos.py`**: 三角函数
- **`eq.py`, `ne.py`, `lt.py`, `le.py`, `gt.py`, `ge.py`**: 比较运算符
- **`bitwise_and.py`, `bitwise_or.py`, `bitwise_not.py`**: 位运算符
- **`isnan.py`, `isinf.py`**: 数值检查函数

## 2. 核心类与函数

### `_CachedMakeDefaultConfig` (utils.py)
- **位置**: `utils.py:9-18`
- **主要功能**: 全局内核编译配置的单例类,存储 ninetoothed 内核生成参数
- **关键成员**:
  - `num_warps`: CUDA kernel 的 warp 数量配置(影响线程块调度)
  - `num_stages`: CUDA pipeline 的流水线阶段数(影响指令级并行)
  - `max_num_configs`: 自动调优时生成的最大配置数量
- **生命周期**: 模块加载时创建全局单例 `_cached_make_default_config`,通过 getter/setter 函数访问

### `_cached_make` (utils.py)
- **位置**: `utils.py:45-63`
- **主要功能**: 带 functools.cache 装饰器的内核编译缓存函数,避免重复编译相同配置的内核
- **核心逻辑**:
  1. 检查 `num_warps`, `num_stages`, `max_num_configs` 参数,若为 None 则使用全局配置
  2. 调用 `premake(*args, **keywords)` 生成内核配置元组
  3. 调用 `ninetoothed.make()` 编译并返回优化后的 CUDA 内核
- **性能优化**: 使用 `@functools.cache` 实现参数化缓存,相同参数调用直接返回编译好的内核对象
- **缓存键**: 由 `premake` 函数及其所有参数、三个编译配置参数组成

### `_get_matmul_input_precision` (utils.py)
- **位置**: `utils.py:66-70`
- **主要功能**: 根据 PyTorch 全局设置 `torch.get_float32_matmul_precision()` 确定矩阵乘法输入精度变体
- **返回值**:
  - `ntops.kernels.mm.InputPrecisionVariant.IEEE`: 当设置为 "highest" 时
  - `ntops.kernels.mm.InputPrecisionVariant.TF32`: 默认情况(TensorFloat-32)
- **作用域**: 被所有矩阵乘法相关函数(mm, bmm, addmm)调用

### `scaled_dot_product_attention` (scaled_dot_product_attention.py)
- **位置**: `scaled_dot_product_attention.py:10-108`
- **主要功能**: 实现高效的缩放点积注意力机制,支持 GQA(分组查询注意力)、KV 缓存、因果掩码、自定义 attention mask
- **函数签名**:
  ```python
  def scaled_dot_product_attention(
      query, key, value, attn_mask=None, dropout_p=0, is_causal=False,
      scale=None, enable_gqa=False, causal_variant=None,
      present_key=None, present_value=None, present_key_slot=None, present_value_slot=None
  )
  ```
- **核心算法**:
  1. **输入验证**: 检查 `dropout_p` 必须为 0(未实现 dropout),`attn_mask` 与 `is_causal` 互斥
  2. **GQA 检查**:
     - 非 GQA 模式: `num_heads_q == num_heads_kv`
     - GQA 模式: `num_heads_q % num_heads_kv == 0`(查询头数必须能被键值头数整除)
  3. **掩码处理**:
     - bool 类型掩码转换为 float: `True` → 0, `False` → `-inf`
     - 扩展掩码形状至 `(batch_size, num_heads_q, seq_len_q, seq_len_kv)`
  4. **缩放因子**: 默认 `scale = 1 / sqrt(head_dim)`, head_dim 为 `query.shape[-1]`
  5. **内核选择**: 根据 `with_kv_cache`(是否存在 present_key)选择不同的内核变体
- **内核参数**:
  - 无 KV 缓存: `(query, key, value, attn_mask, is_causal, scale, output, with_attn_mask, causal_variant)`
  - 有 KV 缓存: 额外传入 `(present_key, present_value, present_key_slot, present_value_slot)`
- **复杂度**: O(seq_len_q² × seq_len_kv + seq_len_q × seq_len_kv × head_dim)
- **内存分配**: 使用 `torch.empty_like(query)` 预分配输出张量,避免动态分配

### `rotary_position_embedding` (rotary_position_embedding.py)
- **位置**: `rotary_position_embedding.py:7-29`
- **主要功能**: 应用旋转位置编码(RoPE)到输入张量,支持 4D 张量(batch, seq_len, num_heads, head_dim)
- **函数签名**:
  ```python
  def rotary_position_embedding(input, sin_table, cos_table, interleaved=True, inplace=False)
  ```
- **核心逻辑**:
  1. **输出分配**: `inplace=True` 时直接修改 input,否则创建 `torch.empty_like(input)`
  2. **表扩展**: 将 `sin_table`, `cos_table` 从 `(seq_len, head_dim)` 广播至 `(batch_size, seq_len, num_heads, head_dim)`
  3. **内核编译**: 固定 `num_warps=1`,根据 `input.ndim` 和 `interleaved` 参数生成专用内核
- **参数说明**:
  - `interleaved`: True 表示交替排列(sin/cos 交错),False 表示分块排列
- **实现细节**: 使用 `[None, :, None, :]` 技巧插入 batch 和 heads 维度进行广播

### `layer_norm` (layer_norm.py)
- **位置**: `layer_norm.py:9-33`
- **主要功能**: Layer Normalization 层归一化,对指定维度进行标准化并应用仿射变换
- **函数签名**:
  ```python
  def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5)
  ```
- **核心算法**:
  1. **形状标准化**: 将 `normalized_shape` 转换为 tuple,支持整数输入
  2. **参数处理**:
     - `weight` 默认为全 1 张量(`torch.ones_like`),并通过 `expand_as` 广播至 input 形状
     - `bias` 默认为全 0 张量(`torch.zeros_like`),同样广播
  3. **内核调用**: 传入 `math.prod(normalized_shape)` 作为归一化元素数量
- **数学公式**: `output = (input - mean) / sqrt(var + eps) * weight + bias`
- **内存优化**: weight/bias 通过 `expand_as` 避免实际内存复制(零拷贝视图)

### `rms_norm` (rms_norm.py)
- **位置**: `rms_norm.py:9-31`
- **主要功能**: RMS Normalization 均方根归一化,LLaMA 等 LLM 模型常用归一化层
- **函数签名**:
  ```python
  def rms_norm(input, normalized_shape, weight=None, eps=None)
  ```
- **核心算法**:
  1. **EPS 默认值**: 若 `eps=None`,使用 `torch.finfo(input.dtype).eps`(根据 dtype 获取机器极小值)
  2. **权重处理**: weight 默认为全 1 张量并广播
  3. **内核调用**: 传入 `math.prod(normalized_shape)` 和 `eps` 参数
- **数学公式**: `output = input / sqrt(mean(input²) + eps) * weight`
- **与 LayerNorm 对比**: 不计算均值,无 bias 参数,计算量更少

### `dropout` (dropout.py)
- **位置**: `dropout.py:9-27`
- **主要功能**: 实现 Dropout 随机失活正则化,训练时随机置零,推理时恒等映射
- **函数签名**:
  ```python
  def dropout(input, p=0.5, training=True, inplace=False)
  ```
- **核心逻辑**:
  1. **快速路径**: `not training or p == 0` 时直接返回(input 或 clone)
  2. **随机种子**: 使用 `random.randrange(0, 2**31)` 生成 31 位随机种子传递给 CUDA 内核
  3. **内核调用**: 根据 `input.ndim` 编译对应维度的 dropout 内核
- **随机性保证**: 每次调用生成新随机种子,确保 GPU 端随机数生成的不可预测性
- **就地模式**: `inplace=True` 时直接修改 input,节省内存

### `matmul` (matmul.py)
- **位置**: `matmul.py:4-18`
- **主要功能**: 通用矩阵乘法路由函数,根据张量维度分发至 mm(2D) 或 bmm(3D)
- **函数签名**:
  ```python
  def matmul(input, other, *, out=None)
  ```
- **分发逻辑**:
  1. **维度检查**: 仅支持 2D 和 3D 张量(断言失败抛出异常)
  2. **2D×2D**: 直接调用 `ntops.torch.mm()`
  3. **3D×3D 或混合**: 维度 <3 的张量通过 `unsqueeze(0)` 插入 batch 维度,调用 `ntops.torch.bmm()`
- **设计模式**: Facade 模式,统一接口隐藏底层实现差异

### `bmm` (bmm.py)
- **位置**: `bmm.py:7-18`
- **主要功能**: 批量矩阵乘法,形状为 `(b, m, k) @ (b, k, n) → (b, m, n)`
- **函数签名**:
  ```python
  def bmm(input, mat2, *, out=None)
  ```
- **核心逻辑**:
  1. **输出分配**: `out=None` 时创建形状 `(b, m, n)` 的空张量,dtype 和 device 继承自 input
  2. **内核调用**: 传入 `_get_matmul_input_precision()` 确定浮点精度
- **约束**: batch 维度 b 必须相等,内部维度 k 必须匹配

### `mm` (mm.py)
- **位置**: `mm.py:7-18`
- **主要功能**: 2D 矩阵乘法,形状为 `(m, k) @ (k, n) → (m, n)`
- **函数签名**:
  ```python
  def mm(input, mat2, *, out=None)
  ```
- **核心逻辑**: 与 `bmm` 类似,但处理 2D 张量,输出形状为 `(m, n)`

### `addmm` (addmm.py)
- **位置**: `addmm.py:7-18`
- **主要功能**: 矩阵乘法加偏置操作,计算 `beta * input + alpha * (mat1 @ mat2)`
- **函数签名**:
  ```python
  def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None)
  ```
- **数学公式**: `output = beta * input + alpha * (mat1 @ mat2)`
- **应用场景**: 线性层 `y = xW^T + b` 的融合实现,减少中间张量分配

### `softmax` (softmax.py)
- **位置**: `softmax.py:7-16`
- **主要功能**: Softmax 归一化,沿指定维度计算指数归一化
- **函数签名**:
  ```python
  def softmax(input, dim, dtype=None)
  ```
- **核心逻辑**:
  1. **dtype 处理**: 若 `dtype=None`,输出 dtype 与 input 相同
  2. **内核编译**: 根据 `input.ndim` 和 `dim` 参数生成专用内核(不同维度的 softmax 计算路径不同)
- **数学公式**: `softmax(x_i) = exp(x_i) / Σ(exp(x_j))`

### 激活函数系列(relu, gelu, silu, sigmoid, tanh)
- **统一模式**: 所有激活函数遵循相同的实现模式
- **函数签名示例**:
  ```python
  def relu(input, inplace=False)  # ReLU 支持就地操作
  def gelu(input, approximate="none")  # GELU 支持近似模式
  def silu(input, inplace=False)  # SiLU 支持就地操作
  def sigmoid(input, *, out=None)  # Sigmoid 使用 out 参数
  def tanh(input, *, out=None)  # Tanh 使用 out 参数
  ```
- **内核编译**: 根据 `input.ndim` 生成对应维度的内核
- **GELU 近似模式**: `approximate` 参数可为 "none"(精确)、"tanh"(tanh 近似)
- **就地优化**: ReLU 和 SiLU 支持 `inplace=True` 避免额外内存分配

### 算术运算符系列(abs, neg, add, sub, mul, div, pow)
- **单目运算符**: `abs`, `neg` 接受单个输入,使用 `out` 参数
- **双目运算符**: `add`, `sub`, `mul`, `div`, `pow` 接受两个输入
- **特殊参数**:
  - `add` 和 `sub` 支持 `alpha` 缩放因子(计算 `input + alpha * other`)
  - `div` 支持 `rounding_mode` 参数(取整模式: None, "trunc", "floor")
- **输出分配**: 统一使用 `torch.empty_like(input)` 预分配输出

### 数学函数系列(exp, rsqrt, sin, cos)
- **exp**: 指数函数 `e^x`
- **rsqrt**: 倒数平方根 `1 / sqrt(x)`
- **sin, cos**: 三角函数
- **实现模式**: 所有函数均根据 `input.ndim` 编译专用内核

### 比较运算符系列(eq, ne, lt, le, gt, ge)
- **函数签名**: `eq(input, other, *, out=None)`
- **返回类型**: 布尔张量或与输入相同 dtype 的张量
- **应用**: 用于张量比较、掩码生成、条件过滤

### 位运算符系列(bitwise_and, bitwise_or, bitwise_not)
- **函数签名**: `bitwise_and(input, other, *, out=None)`
- **特殊处理**: `bitwise_not` 根据 `input.dtype == torch.bool` 生成不同内核(bool 类型特殊处理)
- **应用**: 整数张量的位级操作

### 数值检查系列(isnan, isinf)
- **函数签名**: `isnan(input)`, `isinf(input)`(无 out 参数)
- **返回**: 布尔张量,标记 NaN/Inf 位置
- **应用**: 梯度爆炸/消失检测,数值稳定性检查

## 3. API 接口

```python
# 注意力机制核心 API
def scaled_dot_product_attention(
    query: Tensor, key: Tensor, value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    causal_variant: Optional[CausalVariant] = None,
    present_key: Optional[Tensor] = None,
    present_value: Optional[Tensor] = None,
    present_key_slot: Optional[int] = None,
    present_value_slot: Optional[int] = None
) -> Tensor:
    """缩放点积注意力,支持 GQA、KV 缓存、因果掩码"""

# 旋转位置编码
def rotary_position_embedding(
    input: Tensor,
    sin_table: Tensor,
    cos_table: Tensor,
    interleaved: bool = True,
    inplace: bool = False
) -> Tensor:
    """应用 RoPE 位置编码"""

# 归一化层
def layer_norm(
    input: Tensor,
    normalized_shape: Union[int, Tuple[int, ...]],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5
) -> Tensor:
    """Layer Normalization"""

def rms_norm(
    input: Tensor,
    normalized_shape: Union[int, Tuple[int, ...]],
    weight: Optional[Tensor] = None,
    eps: Optional[float] = None
) -> Tensor:
    """RMS Normalization"""

# 矩阵乘法系列
def matmul(input: Tensor, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """通用矩阵乘法(2D/3D 路由)"""

def bmm(input: Tensor, mat2: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """批量矩阵乘法"""

def mm(input: Tensor, mat2: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """2D 矩阵乘法"""

def addmm(input: Tensor, mat1: Tensor, mat2: Tensor, *,
          beta: float = 1, alpha: float = 1, out: Optional[Tensor] = None) -> Tensor:
    """矩阵乘法加偏置: beta * input + alpha * (mat1 @ mat2)"""

# 激活函数
def relu(input: Tensor, inplace: bool = False) -> Tensor:
    """ReLU 激活: max(0, x)"""

def gelu(input: Tensor, approximate: str = "none") -> Tensor:
    """GELU 激活(支持 tanh 近似)"""

def silu(input: Tensor, inplace: bool = False) -> Tensor:
    """SiLU(Swish) 激活: x * sigmoid(x)"""

def softmax(input: Tensor, dim: int, dtype: Optional[torch.dtype] = None) -> Tensor:
    """Softmax 归一化"""

# Dropout
def dropout(input: Tensor, p: float = 0.5, training: bool = True,
            inplace: bool = False) -> Tensor:
    """随机失活(训练时应用,推理时恒等)"""

# 全局配置管理
def set_default_num_warps(num_warps: Optional[int]) -> None:
    """设置默认 CUDA warp 数量"""

def set_default_num_stages(num_stages: Optional[int]) -> None:
    """设置默认 pipeline 流水线阶段数"""

def set_default_max_num_configs(max_num_configs: Optional[int]) -> None:
    """设置默认最大内核配置数量"""
```

## 4. 使用示例

```python
import torch
import ntops.torch

# ========== 基础张量操作 ==========
# 算术运算
a = torch.randn(3, 4, device='cuda')
b = torch.randn(3, 4, device='cuda')
c = ntops.torch.add(a, b, alpha=2.0)  # 计算 a + 2.0 * b
d = ntops.torch.mul(a, b)
e = ntops.torch.abs(a)

# 矩阵乘法
x = torch.randn(2, 3, 4, device='cuda')  # (batch, m, k)
y = torch.randn(2, 4, 5, device='cuda')  # (batch, k, n)
z = ntops.torch.bmm(x, y)  # 输出 (2, 3, 5)

# 通用 matmul 自动路由
m2d_a = torch.randn(3, 4, device='cuda')
m2d_b = torch.randn(4, 5, device='cuda')
result_2d = ntops.torch.matmul(m2d_a, m2d_b)  # 调用 mm

# ========== 归一化层 ==========
# Layer Normalization
input = torch.randn(2, 10, 512, device='cuda')
weight = torch.randn(512, device='cuda')
bias = torch.randn(512, device='cuda')
output = ntops.torch.layer_norm(input, normalized_shape=512, weight=weight, bias=bias)

# RMS Normalization(无 bias)
rms_output = ntops.torch.rms_norm(input, normalized_shape=512, weight=weight, eps=1e-6)

# ========== 激活函数 ==========
# ReLU 就地操作节省内存
relu_out = ntops.torch.relu(input.clone(), inplace=True)

# GELU 使用 tanh 近似加速
gelu_out = ntops.torch.gelu(input, approximate='tanh')

# Softmax 沿维度 1
logits = torch.randn(2, 10, 100, device='cuda')
probs = ntops.torch.softmax(logits, dim=1)

# ========== Dropout ==========
# 训练模式应用 dropout
dropout_out = ntops.torch.dropout(input, p=0.3, training=True)
# 推理模式直接返回输入
inference_out = ntops.torch.dropout(input, p=0.3, training=False)

# ========== 注意力机制 ==========
# 标准 Multi-Head Attention
batch_size, seq_len, num_heads, head_dim = 2, 128, 8, 64
query = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
key = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
value = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')

# 因果注意力(用于 GPT 等 decoder-only 模型)
causal_out = ntops.torch.scaled_dot_product_attention(
    query, key, value,
    is_causal=True,
    scale=head_dim ** -0.5
)

# 自定义 attention mask
attn_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device='cuda')
attn_mask = torch.tril(attn_mask)  # 下三角掩码
masked_out = ntops.torch.scaled_dot_product_attention(
    query, key, value,
    attn_mask=attn_mask
)

# ========== 分组查询注意力(GQA) ==========
# 查询头 32,键值头 8(4 倍分组)
num_heads_q = 32
num_heads_kv = 8
query_gqa = torch.randn(batch_size, num_heads_q, seq_len, head_dim, device='cuda')
key_gqa = torch.randn(batch_size, num_heads_kv, seq_len, head_dim, device='cuda')
value_gqa = torch.randn(batch_size, num_heads_kv, seq_len, head_dim, device='cuda')

gqa_out = ntops.torch.scaled_dot_product_attention(
    query_gqa, key_gqa, value_gqa,
    enable_gqa=True  # 启用 GQA 模式
)

# ========== KV 缓存(用于自回归生成) ==========
max_cache_len = 1024
present_key = torch.empty(batch_size, num_heads, max_cache_len, head_dim, device='cuda')
present_value = torch.empty(batch_size, num_heads, max_cache_len, head_dim, device='cuda')
slot = 0  # 当前写入位置

kv_cache_out = ntops.torch.scaled_dot_product_attention(
    query, key, value,
    present_key=present_key,
    present_value=present_value,
    present_key_slot=slot,
    present_value_slot=slot
)

# ========== 旋转位置编码(RoPE) ==========
# 生成 sin/cos 表
seq_len = 128
head_dim = 64
positions = torch.arange(seq_len, device='cuda')
dim_range = torch.arange(0, head_dim, 2, device='cuda')
freqs = positions[:, None] / (10000 ** (dim_range[None, :] / head_dim))
sin_table = torch.sin(freqs)  # (seq_len, head_dim/2)
cos_table = torch.cos(freqs)  # (seq_len, head_dim/2)

# 应用 RoPE(交错模式)
query_rope = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda')
rotated_query = ntops.torch.rotary_position_embedding(
    query_rope, sin_table, cos_table,
    interleaved=True,  # sin/cos 交错排列
    inplace=False
)

# ========== 全局配置优化 ==========
# 针对大矩阵乘法调优参数
ntops.torch.set_default_num_warps(4)  # 增加 warp 数
ntops.torch.set_default_num_stages(4)  # 增加 pipeline 阶段
ntops.torch.set_default_max_num_configs(32)  # 生成更多候选配置

# ========== 复杂操作示例:Transformer 层 ==========
class TransformerLayer:
    def __init__(self, hidden_dim, num_heads, device='cuda'):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.device = device

        # 权重参数(简化示例)
        self.qkv_weight = torch.randn(hidden_dim, 3 * hidden_dim, device=device)
        self.out_weight = torch.randn(hidden_dim, hidden_dim, device=device)
        self.ffn_weight1 = torch.randn(hidden_dim, 4 * hidden_dim, device=device)
        self.ffn_weight2 = torch.randn(4 * hidden_dim, hidden_dim, device=device)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # QKV 投影
        qkv = ntops.torch.matmul(x, self.qkv_weight)  # (batch, seq_len, 3*hidden_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # 重整为多头格式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 自注意力(因果掩码)
        attn_out = ntops.torch.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            scale=self.head_dim ** -0.5
        )

        # 合并多头
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        # 输出投影
        out = ntops.torch.matmul(attn_out, self.out_weight)

        # FFN: GELU -> 线性
        hidden = ntops.torch.matmul(out, self.ffn_weight1)
        hidden = ntops.torch.gelu(hidden)
        ffn_out = ntops.torch.matmul(hidden, self.ffn_weight2)

        return ffn_out

# 使用 Transformer 层
layer = TransformerLayer(hidden_dim=512, num_heads=8)
input_tensor = torch.randn(2, 128, 512, device='cuda')
output = layer.forward(input_tensor)
```

## 5. 实现细节

### 内存管理策略
- **预分配输出张量**: 所有函数优先使用 `torch.empty_like(input)` 预分配输出,避免内核执行期间动态分配内存
- **零拷贝广播**: weight/bias 参数通过 `expand_as` 广播,创建视图而非实际复制,节省内存带宽
- **就地操作模式**: ReLU, SiLU, Dropout 等函数支持 `inplace=True`,直接修改输入张量,适用于可安全覆盖的场景
- **元张量占位**: `scaled_dot_product_attention` 中无 attn_mask 时使用 `device="meta"` 创建空张量,避免内存占用

### 并发与线程安全
- **随机种子隔离**: Dropout 每次调用生成新的随机种子(`random.randrange(0, 2**31)`),确保多线程/多进程环境下随机性独立
- **全局配置单例**: `_cached_make_default_config` 作为模块级单例,通过 getter/setter 访问,Python GIL 保证配置修改的原子性
- **缓存线程安全**: `@functools.cache` 装饰器内置线程安全机制,多线程并发调用相同配置的内核时只会编译一次

### 性能优化技术
- **内核编译缓存**: `_cached_make` 使用 `@functools.cache` 实现参数化缓存,相同参数直接返回已编译内核,避免重复编译开销(编译耗时约秒级)
- **自动调优配置**: ninetoothed 框架根据 `num_warps`, `num_stages`, `max_num_configs` 生成多个内核变体,在运行时选择最优配置
- **TensorFloat-32 加速**: `_get_matmul_input_precision` 检测 PyTorch 设置,在支持 Ampere GPU(A100)时使用 TF32 加速矩阵乘法(吞吐量提升 8 倍)
- **流水线并行**: `num_stages` 参数控制 CUDA pipeline 深度,隐藏全局内存延迟,提升计算吞吐量
- **Warp 调度**: `num_warps` 参数控制 SM 内的 warp 数量,优化寄存器使用和占用率(occupancy)

### 错误处理机制
- **断言式验证**: 使用 `assert` 检查输入约束(如 dropout_p == 0, 维度匹配),失败时抛出 Python 异常
- **形状推导**: `layer_norm` 和 `rms_norm` 自动推导 `normalized_shape` 的乘积,传递给内核用于归一化计算
- **互斥参数检查**: `scaled_dot_product_attention` 确保 `attn_mask` 与 `is_causal` 不能同时使用,避免语义冲突
- **GQA 头数验证**: 检查 `num_heads_q % num_heads_kv == 0`,确保查询头能均匀分配到键值头

### 依赖关系
- **核心依赖**: `ninetoothed`(内核编译框架)、`ntops.kernels`(底层 CUDA 内核)、`torch`(张量计算)
- **模块级依赖**: 所有算子文件依赖 `ntops.torch.utils` 的 `_cached_make` 函数
- **枚举类型**: `scaled_dot_product_attention` 依赖 `ntops.kernels.scaled_dot_product_attention.CausalVariant`

### 设计模式
- **Facade 模式**: `matmul` 函数作为统一接口,隐藏 `mm` 和 `bmm` 的实现细节
- **Strategy 模式**: `scaled_dot_product_attention` 根据 `with_kv_cache` 选择不同内核策略
- **Template Method 模式**: 所有算子函数遵循相同的模板:参数验证 → 输出分配 → 内核编译 → 内核调用 → 返回结果
- **Singleton 模式**: `_cached_make_default_config` 作为全局配置单例
- **Cache Aside 模式**: `_cached_make` 先查缓存,未命中则调用 ninetoothed.make 编译并缓存

### 特殊优化
- **dtype 特化**: `bitwise_not` 根据 `input.dtype == torch.bool` 生成不同内核(bool 类型使用特殊指令)
- **维度特化**: softmax 等函数根据 `dim` 参数生成专用内核,不同维度的归一化使用不同的线程块组织策略
- **近似模式**: GELU 支持 `approximate="tanh"` 使用 tanh 近似公式,牺牲微小精度换取 2 倍速度提升
- **自动 eps 推导**: `rms_norm` 使用 `torch.finfo(input.dtype).eps` 根据 dtype 自动获取数值稳定的 eps 值(fp16: 1e-5, fp32: 1e-8)

### 算子覆盖范围
该模块实现了 PyTorch 核心算子的 39 个函数,覆盖以下类别:
- **线性代数**: matmul, bmm, mm, addmm(4 个)
- **注意力机制**: scaled_dot_product_attention(1 个)
- **位置编码**: rotary_position_embedding(1 个)
- **归一化**: layer_norm, rms_norm(2 个)
- **激活函数**: relu, gelu, silu, sigmoid, tanh, softmax(6 个)
- **正则化**: dropout(1 个)
- **算术运算**: abs, neg, add, sub, mul, div, pow(7 个)
- **数学函数**: exp, rsqrt, sin, cos(4 个)
- **比较运算**: eq, ne, lt, le, gt, ge(6 个)
- **位运算**: bitwise_and, bitwise_or, bitwise_not(3 个)
- **数值检查**: isnan, isinf(2 个)
- **其他**: clamp(1 个)

总计 39 个函数,提供与 PyTorch 原生 API 兼容的接口,底层通过 ninetoothed 自动生成优化的 CUDA 内核。
