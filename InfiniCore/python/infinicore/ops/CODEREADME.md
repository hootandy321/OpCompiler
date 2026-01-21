# InfiniCore Python Operations API 实现文档

本模块是 InfiniCore 框架的 Python 操作接口层，提供从 Python 到底层 C++/CUDA 实现的绑定。该模块实现了神经网络推理和训练中的核心算子，特别优化了大语言模型（LLM）中的注意力机制和归一化操作。

## 1. 模块结构

- **`__init__.py`**: 包初始化文件（空文件）
- **`add.py`**: 张量加法操作的 Python 绑定
- **`add_rms_norm.py`**: 融合加法与 RMS 归一化操作（性能优化算子）
- **`attention.py`**: 标准注意力机制接口
- **`matmul.py`**: 矩阵乘法操作（支持 alpha 缩放因子）
- **`mul.py`**: 张量逐元素乘法操作
- **`narrow.py`**: 张量切片操作（沿指定维度提取子张量）
- **`paged_attention_prefill.py`**: 分页注意力预填充阶段（处理新 token 的初始注意力计算）
- **`paged_attention.py`**: 分页注意力解码阶段（处理自回归生成的增量注意力）
- **`paged_caching.py`**: 分页缓存写入操作（将 KV 对写入 KV cache）
- **`rearrange.py`**: 张量重排操作
- **`squeeze.py`**: 移除大小为 1 的维度
- **`unsqueeze.py`**: 在指定位置插入大小为 1 的维度

## 2. 核心设计模式

### 统一的 API 接口模式

所有算子函数遵循统一的双模式设计：

1. **函数式模式（Functional）**：当 `out=None` 时，返回新分配的张量
2. **就地操作模式（In-place）**：当 `out` 参数提供时，直接写入输出张量并返回它

```python
def operation(input, other, *, out=None):
    if out is None:
        return Tensor(_infinicore.operation(input._underlying, other._underlying))
    _infinicore.operation_(out._underlying, input._underlying, other._underlying)
    return out
```

### 底层绑定机制

所有函数通过 `_infinicore` 模块（pybind11 绑定的 C++ 扩展）调用底层实现：
- **前缀无下划线函数**：如 `_infinicore.add()`，分配新内存并返回结果
- **后缀下划线函数**：如 `_infinicore.add_()`，就地修改操作（in-place）

### 张量包装策略

所有输入参数通过 `. _underlying` 属性访问底层 C++ 张量对象：
```python
Tensor(_infinicore.operation(input._underlying, other._underlying))
```

## 3. API 接口详解

### 基础算术操作

#### `add(input, other, *, out=None)` → Tensor
逐元素加法操作。

**参数**：
- `input` (Tensor): 左操作数
- `other` (Tensor): 右操作数
- `out` (Tensor | None): 可选输出张量

**返回**：加法结果张量

**底层绑定**：
- `_infinicore.add(input, other)` → 新张量
- `_infinicore.add_(out, input, other)` → 就地写入

#### `mul(input, other, *, out=None)` → Tensor
逐元素乘法操作。

**参数**：
- `input` (Tensor): 左操作数
- `other` (Tensor): 右操作数
- `out` (Tensor | None): 可选输出张量

**返回**：乘法结果张量

**底层绑定**：
- `_infinicore.mul(input, other)` → 新张量
- `_infinicore.mul_(out, input, other)` → 就地写入

#### `matmul(input, other, *, alpha=1.0, out=None)` → Tensor
矩阵乘法操作，支持缩放因子。

**参数**：
- `input` (Tensor): 左矩阵 (shape: `[M, K]` 或批次版本)
- `other` (Tensor): 右矩阵 (shape: `[K, N]` 或批次版本)
- `alpha` (float): 缩放因子，默认 1.0，用于结果乘法：`output = alpha * (input @ other)`
- `out` (Tensor | None): 可选输出张量

**返回**：矩阵乘法结果 (shape: `[M, N]`)

**底层绑定**：
- `_infinicore.matmul(input, other, alpha)` → 新张量
- `_infinicore.matmul_(out, input, other, alpha)` → 就地写入

### 融合算子（性能优化）

#### `add_rms_norm(a, b, weight, epsilon=1e-5, *, out=None)` → (Tensor, Tensor)
融合加法与 RMS 归一化操作，减少内存访问和中间结果存储，优化 Transformer 残差连接层。

**参数**：
- `a` (Tensor): 第一个输入张量（通常是隐藏状态）
- `b` (Tensor): 第二个输入张量（通常是跳跃连接输入）
- `weight` (Tensor): 缩放权重参数
- `epsilon` (float): 数值稳定小常数，默认 1e-5
- `out` (tuple[Tensor, Tensor] | None): 可选输出元组 `(y, residual_out)`

**返回**：元组 `(normalized_result, add_result)`
- `normalized_result`: RMSNorm(a + b) * weight
- `add_result`: a + b（可作为后续层的残差连接）

**数学定义**：
```
sum = a + b
rms = sqrt(mean(sum^2) + epsilon)
normalized = (sum / rms) * weight
```

**底层绑定**：
- `_infinicore.add_rms_norm(a, b, weight, epsilon)` → 返回底层元组
- `_infinicore.add_rms_norm_(y, residual_out, a, b, weight, epsilon)` → 就地写入

**典型使用场景**：Transformer 的 Pre-LN 或 Post-LN 层

```python
# Pre-LN Transformer 层
normalized, residual = add_rms_norm(hidden, input, weight)
hidden = attention(normalized) + mlp(normalized) + residual
```

#### `add_rms_norm_(y, residual_out, a, b, weight, epsilon=1e-5)`
`add_rms_norm` 的就地操作版本，强制要求输出张量。

**参数**：
- `y` (Tensor): 预分配的归一化结果输出张量
- `residual_out` (Tensor): 预分配的加法结果输出张量
- `a`, `b`, `weight`, `epsilon`: 同 `add_rms_norm`

### 张量形状操作

#### `narrow(input: Tensor, dim: int, start: int, length: int)` → Tensor
沿指定维度提取张量的连续切片（类似于 PyTorch 的 `narrow`）。

**参数**：
- `input` (Tensor): 输入张量
- `dim` (int): 目标维度（0-based 索引）
- `start` (int): 切片起始索引
- `length` (int): 切片长度

**返回**：切片后的新张量视图（与原张量共享内存）

**实现细节**：直接调用底层张量对象的 `narrow` 方法
```python
Tensor(input._underlying.narrow(dim, start, length))
```

#### `squeeze(input: Tensor, dim: int)` → Tensor
移除指定位置大小为 1 的维度。

**参数**：
- `input` (Tensor): 输入张量
- `dim` (int): 要移除的维度索引

**返回**：压缩后的新张量视图

**约束**：指定维度的大小必须为 1，否则行为未定义

#### `unsqueeze(input: Tensor, dim: int)` → Tensor
在指定位置插入大小为 1 的维度。

**参数**：
- `input` (Tensor): 输入张量
- `dim` (int): 插入维度的位置（0-based）

**返回**：扩展后的新张量视图

**示例**：
```python
x = tensor.shape([2, 3])  # shape: [2, 3]
y = unsqueeze(x, 0)       # shape: [1, 2, 3]
z = unsqueeze(x, 1)       # shape: [2, 1, 3]
```

### 注意力机制算子

#### `attention(q, k, v, k_cache, v_cache, pos, *, out=None)` → Tensor
标准注意力机制接口，支持 KV cache 和位置编码。

**参数**：
- `q` (Tensor): 查询张量
- `k` (Tensor): 键张量
- `v` (Tensor): 值张量
- `k_cache` (Tensor): 键缓存张量（存储历史键）
- `v_cache` (Tensor): 值缓存张量（存储历史值）
- `pos` (int): 当前位置索引（用于位置编码）
- `out` (Tensor | None): 可选输出张量

**返回**：注意力输出结果

**底层绑定**：
- `_infinicore.attention(q, k, v, k_cache, v_cache, pos)` → 新张量
- `_infinicore.attention_(out, q, k, v, k_cache, v_cache, pos)` → 就地写入

**应用场景**：传统自回归生成中的单步注意力计算

### 分页注意力算子（Paged Attention）

分页注意力算法（源自 vLLM）实现了高效的 KV Cache 管理，通过块表（block table）映射逻辑页到物理页，解决显存碎片问题。

#### `paged_attention_prefill(q, k_cache, v_cache, block_tables, history_lens, cu_seqlens_q, alibi_slopes=None, scale=1.0, *, out=None)` → Tensor
分页注意力的预填充阶段，处理批次中所有序列的初始 token（context encoding）。

**参数**：
- `q` (Tensor): 查询张量，shape `[total_q_tokens, num_heads, head_dim]`
- `k_cache` (Tensor): 键缓存张量，shape `[num_blocks, block_size, num_kv_heads, head_dim]`
- `v_cache` (Tensor): 值缓存张量，shape `[num_blocks, block_size, num_kv_heads, head_dim]`
- `block_tables` (Tensor): 块表，shape `[num_seqs, max_num_blocks_per_seq]`，将逻辑页映射到物理页
- `history_lens` (Tensor): 每个序列的历史长度，shape `[num_seqs]`
- `cu_seqlens_q` (Tensor): 累积序列长度张量（CUDA 内核使用），shape `[num_seqs + 1]`
- `alibi_slopes` (Tensor | None): ALiBi 位置编码斜率，shape `[num_heads]`，默认 None
- `scale` (float): 注意力缩放因子，默认 1.0（通常为 `1.0 / sqrt(head_dim)`）
- `out` (Tensor | None): 可选输出张量

**返回**：预填充阶段的注意力输出

**底层绑定**：
- `_infinicore.paged_attention_prefill(q, k_cache, v_cache, block_tables, history_lens, cu_seqlens_q, alibi_ptr, scale)` → 新张量
- `_infinicore.paged_attention_prefill_(...)` → 就地写入

**特点**：
- 使用 `cu_seqlens_q` 实现批量的变长序列处理
- `alibi_slopes` 支持 ALiBi（Attention with Linear Biases）位置编码
- 适用于推理初期处理长 prompt

#### `paged_attention(q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes=None, scale=1.0, *, out=None)` → Tensor
分页注意力的解码阶段，处理自回归生成的单个新 token。

**参数**：
- `q` (Tensor): 查询张量，shape `[num_seqs, num_heads, head_dim]`
- `k_cache`, `v_cache`: 同 `paged_attention_prefill`
- `block_tables` (Tensor): 块表，shape `[num_seqs, max_num_blocks_per_seq]`
- `cache_lens` (Tensor): 每个序列的缓存长度，shape `[num_seqs]`
- `alibi_slopes` (Tensor | None): ALiBi 位置编码斜率，默认 None
- `scale` (float): 注意力缩放因子，默认 1.0
- `out` (Tensor | None): 可选输出张量

**返回**：解码阶段的注意力输出

**底层绑定**：
- `_infinicore.paged_attention(q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes, scale)` → 新张量
- `_infinicore.paged_attention_(...)` → 就地写入

**特点**：
- 优化的单步生成内核
- 只计算新 token 与历史 cache 的注意力
- 支持多查询注意力（MQA）和分组查询注意力（GQA）

#### `paged_caching(k_cache, v_cache, k, v, slot_mapping)` → (Tensor, Tensor)
将新的键值对写入分页 KV cache。

**参数**：
- `k_cache` (Tensor): 键缓存张量（就地修改）
- `v_cache` (Tensor): 值缓存张量（就地修改）
- `k` (Tensor): 新的键张量
- `v` (Tensor): 新的值张量
- `slot_mapping` (Tensor): 槽位映射张量，将每个 token 映射到 cache 中的物理位置

**返回**：元组 `(k_cache, v_cache)`（返回更新后的 cache 引用）

**底层绑定**：
```python
_infinicore.paged_caching_(k_cache._underlying, v_cache._underlying, k._underlying, v._underlying, slot_mapping._underlying)
```

**实现细节**：
- 使用 `slot_mapping` 将逻辑 token 位置映射到物理 cache 块
- 就地写入 cache，无返回值（通过修改输入张量实现）
- 典型流程：先调用 `paged_caching` 写入 KV，再调用 `paged_attention` 计算

**使用场景示例**：
```python
# 自回归生成步骤
for step in range(max_steps):
    q, k, v = model.forward(last_token)
    paged_caching(k_cache, v_cache, k, v, slot_mapping)
    output = paged_attention(q, k_cache, v_cache, block_tables, cache_lens)
    last_token = output
```

### 其他操作

#### `rearrange(input, *, out=None)` → Tensor
张量重排操作（具体语义由底层实现定义）。

**参数**：
- `input` (Tensor): 输入张量
- `out` (Tensor | None): 可选输出张量

**返回**：重排后的张量

**底层绑定**：
- `_infinicore.rearrange(input)` → 新张量
- `_infinicore.rearrange_(out, input)` → 就地写入

**注意**：此函数的具体重排规则（如转置、置换等）由底层 C++ 实现决定

## 4. 使用示例

### 基础算术操作

```python
from infinicore.tensor import Tensor
from infinicore.ops import add, mul, matmul

# 创建输入张量
a = Tensor.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
b = Tensor.from_numpy(np.array([[5.0, 6.0], [7.0, 8.0]]))

# 函数式模式：返回新张量
c = add(a, b)  # c = a + b

# 就地操作模式：写入预分配张量
output = Tensor.empty_like(a)
add(a, b, out=output)  # 直接写入 output

# 矩阵乘法带缩放
result = matmul(a, b, alpha=0.5)  # result = 0.5 * (a @ b)
```

### Transformer 层融合操作

```python
from infinicore.ops import add_rms_norm

# Pre-LN Transformer 层
hidden = Tensor(...)  # shape: [batch, seq_len, hidden_dim]
input = Tensor(...)   # 残差连接输入
weight = Tensor(...)  # 归一化权重参数

# 融合操作：一次内核调用完成 add + rms_norm
normalized, residual = add_rms_norm(hidden, input, weight, epsilon=1e-5)

# 后续处理：attention 和 mlp
# residual 可直接用于下一层的残差连接
```

### 分页注意力推理流程

```python
from infinicore.ops import paged_attention_prefill, paged_attention, paged_caching

# 1. 预填充阶段：处理 prompt
prompts_q = Tensor(...)  # shape: [total_tokens, num_heads, head_dim]
block_tables = Tensor(...)
history_lens = Tensor(...)
cu_seqlens_q = Tensor(...)

# 预填充注意力
prefill_output = paged_attention_prefill(
    prompts_q, k_cache, v_cache, block_tables, history_lens, cu_seqlens_q,
    scale=1.0 / sqrt(head_dim)
)

# 2. 解码阶段：自回归生成
for step in range(generation_steps):
    # 计算 new token 的 QKV
    q, k, v = model.forward(last_token)

    # 写入 KV cache
    slot_mapping = compute_slot_mapping(...)
    paged_caching(k_cache, v_cache, k, v, slot_mapping)

    # 计算注意力（与历史 cache）
    token_output = paged_attention(
        q, k_cache, v_cache, block_tables, cache_lens,
        scale=1.0 / sqrt(head_dim)
    )

    last_token = token_output
```

### 张量形状操作

```python
from infinicore.ops import narrow, squeeze, unsqueeze

x = Tensor(...)  # shape: [10, 20, 30]

# narrow：切片操作
y = narrow(x, dim=1, start=5, length=10)  # shape: [10, 10, 30]

# squeeze：移除大小为 1 的维度
z = Tensor(...)  # shape: [1, 20, 1, 30]
squeezed = squeeze(z, dim=0)  # shape: [20, 1, 30]
squeezed2 = squeeze(squeezed, dim=1)  # shape: [20, 30]

# unsqueeze：插入维度
unsqueezed = unsqueeze(squeezed2, dim=1)  # shape: [20, 1, 30]
```

## 5. 实现细节

### 内存管理策略

- **零拷贝视图**：`narrow`、`squeeze`、`unsqueeze` 返回与原张量共享内存的视图
- **显式分配**：算术操作的函数式模式由底层 C++ 内核分配新内存
- **就地复用**：所有算子支持 `out` 参数以复用预分配内存，减少分配开销

### 并发与线程安全

- **全局解释器锁（GIL）**：Python 绑定层在调用底层 C/CUDA 内核时会释放 GIL，允许其他 Python 线程运行
- **CUDA 流并发**：底层实现可能支持 CUDA stream 并行（由 C++ 层管理）

### 性能优化技术

1. **算子融合**：`add_rms_norm` 将两个操作融合为单次内核启动，减少内存访问
2. **分页内存管理**：Paged Attention 算法通过块表实现 KV cache 的动态分配，减少显存碎片
3. **ALiBi 支持**：注意力算子原生支持 ALiBi 位置编码，避免额外操作
4. **变长序列批处理**：`paged_attention_prefill` 使用 `cu_seqlens_q` 实现高效的不等长序列批量处理

### 错误处理

- **类型检查**：所有函数期望参数为 `Tensor` 类型，通过 `. _underlying` 访问底层对象
- **空值处理**：`alibi_slopes` 等可选参数在传递给底层前转换为 `None` 或底层指针
- **形状约束**：形状兼容性由底层 C++ 内核验证（可能在运行时抛出异常）

### 依赖关系

- **`infinicore.lib._infinicore`**: pybind11 绑定的 C++ 扩展模块，提供所有底层算子实现
- **`infinicore.tensor.Tensor`**: 张量包装类，维护底层 C++ 张量对象的生命周期

### 设计模式

- **Thin Wrapper Pattern**: Python 函数仅做参数转换和分发，核心逻辑在 C++ 层
- **Optional Output Pattern**: 所有算子通过 `out` 参数支持就地操作，减少内存分配
- **Type Hiding**: 通过 `. _underlying` 隐藏底层 C++ 对象类型，保持 Python API 清洁
