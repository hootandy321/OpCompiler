# PyBind11 Ops 桥接层核心实现文档

本模块是 InfiniCore 框架的 Python-C++ FFI 层,负责将底层 C++ 运算符 API 暴露给 Python 生态系统。通过 pybind11 实现类型安全的绑定层,支持张量操作、注意力机制、归一化、激活函数等深度学习核心算子的 Python 可调用接口。

## 1. 模块结构

- **`add.hpp`**: 张量加法运算的 Python 绑定,支持元素级加和就地加法两种模式
- **`add_rms_norm.hpp`**: 融合加法与 RMS 归一化的复合算子绑定,优化残差连接场景
- **`attention.hpp`**: 标准 KV 缓存注意力机制绑定,支持因果掩码与位置编码
- **`causal_softmax.hpp`**: 因果掩码 Softmax 激活函数绑定,确保自回归生成时的位置约束
- **`embedding.hpp`**: 词嵌入查找表绑定,将 token ID 映射为稠密向量表示
- **`linear.hpp`**: 线性变换层绑定(仿射变换 y=xA^T+b),支持可选偏置项
- **`matmul.hpp`**: 矩阵乘法运算绑定,支持缩放系数 alpha
- **`mul.hpp`**: 逐元素张量乘法绑定,支持广播机制
- **`paged_attention.hpp`**: 分页注意力机制绑定,支持 ALiBi 位置偏置与动态 KV 块管理
- **`paged_attention_prefill.hpp`**: 分页注意力预填充阶段绑定,处理变长序列打包请求
- **`pagic_caching.hpp`**: KV 缓存写入绑定,将计算结果写入分页内存槽位
- **`random_sample.hpp`**: 核采样与 Top-k 采样绑定,实现概率驱动的 token 生成
- **`rearrange.hpp`**: 张量维度重排绑定,支持内存布局转换
- **`rms_norm.hpp`**: RMS 归一化层绑定(Root Mean Square Layer Normalization 变体)
- **`rope.hpp`**: 旋转位置编码绑定,支持 GPT-J 与 GPT-NeoX 两种实现算法
- **`silu.hpp`**: SiLU/Swish 激活函数绑定(x * sigmoid(x))
- **`swiglu.hpp`**: SwiGLU 激活函数绑定(SiLU(x) * gate 的 GLU 变体)

## 2. 核绑定架构

### 绑定函数命名约定
所有绑定函数遵循 `bind_<opname>` 模式,每个头文件提供一个内联函数向 pybind11 模块注册 Python 可调用接口。

### 双模式 API 设计
每个算子暴露两个 Python 函数:
1. **分配模式** `op()`: 分配新张量返回结果(如 `add(a, b)`)
2. **就地模式** `op_()`: 在预分配输出张量上执行操作(如 `add_(out, a, b)`)

这种设计允许 Python 侧在内存预分配与计算便利性之间灵活选择。

### 可选参数包装策略
对于 `bias`, `alibi_slopes` 等可选参数,采用 `pybind11::object` 接收 Python 的 `None`,内部转换为 `std::optional<Tensor>`:
```cpp
std::optional<Tensor> bias_tensor = std::nullopt;
if (!bias.is_none()) {
    bias_tensor = bias.cast<Tensor>();
}
return op::linear(input, weight, bias_tensor);
```

## 3. 关键算子绑定详解

### `add_rms_norm.hpp` - 融合残差归一化
**函数签名**:
```cpp
m.def("add_rms_norm",
      &op::add_rms_norm,
      py::arg("a"),
      py::arg("b"),
      py::arg("weight"),
      py::arg("epsilon") = 1e-5f);

m.def("add_rms_norm_",
      &op::add_rms_norm_,
      py::arg("y"),          // 归一化结果输出
      py::arg("residual_out"), // 加法结果输出(用于残差连接)
      py::arg("a"),
      py::arg("b"),
      py::arg("weight"),
      py::arg("epsilon") = 1e-5f);
```

**返回值**: 就地模式返回二元组 `(normalized_result, add_result)`

**设计意图**: 在 Transformer 前馈网络中,残差连接与层归一化通常成对出现。融合算子减少内存读写:
1. 计算 `a + b` 存入 `residual_out`
2. 对加和执行 RMS 归一化存入 `y`
3. 消除中间张量分配

### `paged_attention.hpp` - 分页注意力
**函数签名**:
```cpp
Tensor py_paged_attention(
    Tensor q,                    // 查询张量 [batch, seq_len, num_heads, head_dim]
    Tensor k_cache,              // 键缓存 [num_blocks, block_size, num_heads, head_dim]
    Tensor v_cache,              // 值缓存 [num_blocks, block_size, num_heads, head_dim]
    Tensor block_tables,         // 块映射表 [batch, max_blocks_per_seq]
    Tensor cache_lens,           // 实际缓存长度 [batch]
    pybind11::object alibi_slopes, // 可选 ALiBi 位置偏置斜率
    float scale                  // 缩放因子(通常为 1/sqrt(head_dim))
);
```

**关键数据结构**:
- **`block_tables`**: 将虚拟序列位置映射到物理内存块,支持非连续 KV 存储
- **`alibi_slopes`**: ALiBi(Attention with Linear Biases)位置编码的注意力偏置乘数,每个注意力头一个斜率值

**应用场景**: 大语言模型推理时的 KV Cache 分块管理,避免固定长度的连续内存浪费。

### `paged_attention_prefill.hpp` - 预填充阶段
**附加参数**:
```cpp
Tensor history_lens,    // 每个序列的历史长度 [batch]
Tensor cu_seqlens_q,    // 累积序列长度索引 [batch+1],用于打包张量的寻址
```

**算法差异**: 预填充阶段需要处理变长输入序列的批量处理,通过 `cu_seqlens_q` 实现打包张量的高效寻址。与解码阶段的单步自回归生成不同,预填充通常一次处理多个 token。

### `rope.hpp` - 旋转位置编码
**枚举类型暴露**:
```cpp
py::enum_<infinicore::nn::RoPE::Algo>(m, "RoPEAlgo")
    .value("GPT_J", infinicore::nn::RoPE::Algo::GPT_J)
    .value("GPT_NEOX", infinicore::nn::RoPE::Algo::GPT_NEOX);
```

**函数签名**:
```cpp
m.def("rope",
      &op::rope,
      py::arg("x"),          // 输入张量 [batch, seq_len, num_heads, head_dim]
      py::arg("pos"),        // 位置索引 [batch, seq_len] 或 [seq_len]
      py::arg("sin_table"),  // 预计算的正弦表
      py::arg("cos_table"),  // 预计算的余弦表
      py::arg("algo"));      // GPT_J 或 GPT_NEOX 算法选择
```

**算法差异**:
- **GPT-J**: 旋转应用于查询和键的最后两个维度
- **GPT-NeoX**: 独立旋转每个注意力头的一半维度

### `random_sample.hpp` - 神经解码采样
**函数签名**:
```cpp
m.def("random_sample",
      &op::random_sample,
      py::arg("logits"),      // 词汇表分布 [vocab_size]
      py::arg("random_val"),  // 随机数值 [0, 1) 标量
      py::arg("topp"),        // nucleus sampling 阈值(如 0.9)
      py::arg("topk"),        // top-k 采样阈值(如 50)
      py::arg("temperature")); // 温度参数(如 0.7)
```

**采样策略**:
1. 温度缩放: `logits = logits / temperature`
2. Top-k 过滤: 保留概率最大的 k 个 token
3. Nucleus(p) 过滤: 保留累积概率达到 p 的最小 token 集
4. Softmax 归一化: 转换为概率分布
5. 采样: 根据 `random_val` 在累积分布中采样

## 4. Python 端使用示例

### 基础算子调用
```python
import infinicore

# 张量加法
result = infinicore.add(tensor_a, tensor_b)

# 就地加法(节省内存分配)
output = infinicore.empty_like(tensor_a)
infinicore.add_(output, tensor_a, tensor_b)

# 矩阵乘法
c = infinicore.matmul(a, b, alpha=1.0)

# 带偏置的线性变换
y = infinicore.linear(x, weight, bias=bias_tensor)  # 有偏置
y = infinicore.linear(x, weight)                    # 无偏置
```

### Transformer 模块组合
```python
# 残差连接 + RMS 归一化
normalized, residual = infinicore.add_rms_norm(
    hidden_states,
    input_tensor,
    weight,
    epsilon=1e-5
)

# 自注意力计算
attn_out = infinicore.attention(
    q=query_proj,
    k=key_proj,
    v=value_proj,
    k_cache=key_cache,
    v_cache=value_cache,
    pos=current_position
)

# 旋转位置编码应用
rotated_q = infinicore.rope(
    x=q,
    pos=position_ids,
    sin_table=sin_cache,
    cos_table=cos_cache,
    algo=infinicore.RoPEAlgo.GPT_NEOX
)
```

### 大模型推理流程
```python
# 分页 KV 缓存写入
infinicore.paged_caching_(
    k_cache=key_cache_blocks,
    v_cache=value_cache_blocks,
    k=new_keys,
    v=new_values,
    slot_mapping=physical_slot_ids
)

# 分页注意力计算
attn_output = infinicore.paged_attention(
    q=query_states,
    k_cache=key_cache_blocks,
    v_cache=value_cache_blocks,
    block_tables=seq_block_mappings,
    cache_lens=seq_actual_lengths,
    alibi_slopes=alibi_bias_tensor,  # 可选
    scale=1.0 / (head_dim ** 0.5)
)

# 核采样生成 token
token_id = infinicore.random_sample(
    logits=logits,
    random_val=random.random(),
    topp=0.9,
    topk=50,
    temperature=0.7
)
```

### 激活函数流水线
```python
# SwiGLU (用于 LLaMA 等模型)
gate = infinicore.linear(x, gate_weight)
up = infinicore.linear(x, up_weight)
activated = infinicore.swiglu(gate, up)

# SiLU 激活
hidden = infinicore.silu(linear_output)

# RMS 归一化
normalized = infinicore.rms_norm(
    x=hidden_states,
    weight=norm_weight,
    epsilon=1e-5
)
```

## 5. 实现细节

### 命名空间结构
```cpp
namespace infinicore {
    namespace ops {
        inline void bind_add(py::module &m);
        inline void bind_linear(py::module &m);
        // ... 其他绑定函数
    }
}
```

所有绑定函数位于 `infinicore::ops` 命名空间,避免符号冲突。通过 `inline` 关键头文件优化,允许多个翻译单元包含同一头文件而不违反 ODR。

### 参数绑定策略
**位置参数**: 使用 `py::arg("name")` 显式命名,提升 Python 端可读性并支持关键字参数调用。

**默认参数**: C++ 默认值直接传递给 Python,如 `py::arg("epsilon") = 1e-5f`。

**文档字符串**: 使用原始字符串字面量 `R"doc(...)doc"` 包含多行文档,自动附加到 Python 函数的 `__doc__` 属性。

### 类型转换机制
- **Tensor 类型**: pybind11 自动将 Python 侧的 `infinicore.Tensor` 转换为 C++ 的 `Tensor` 对象
- **Optional<Tensor>**: 通过 `pybind11::object` 中间层处理 Python 的 `None`
- **枚举类型**: 使用 `py::enum_` 暴露为 Python 的 `enum.Enum`

### 内存管理语义
**分配模式**: C++ 侧创建新 Tensor 对象,pybind11 将所有权转移给 Python(引用计数管理)。

**就地模式**: Python 传入预分配的输出 Tensor,C++ 直接写入其内存缓冲区,零拷贝操作。

### 性能优化考量
1. **减少数据拷贝**: 就地模式允许 Python 侧预分配内存,避免中间张量分配
2. **融合算子**: `add_rms_norm` 等融合操作合并多个内核启动,降低 GPU kernel 启动开销
3. **缓存友好**: `paged_attention` 的块表设计支持非连续内存访问,配合虚拟内存管理

### 错误处理策略
底层 C++ 运算符抛出 C++ 异常时,pybind11 自动转换为 Python 异常:
- `std::runtime_error` → `RuntimeError`
- `std::invalid_argument` → `ValueError`
- 类型不匹配 → `TypeError`

Python 侧可通过标准 `try-except` 捕获:
```python
try:
    result = infinicore.add(a, b)
except RuntimeError as e:
    print(f"Compute error: {e}")
```

### 依赖关系
**上游依赖**:
- `pybind11/pybind11.h`: Python C++ 绑定框架
- `infinicore/ops/*.hpp`: 底层算子实现头文件

**下游影响**:
- Python 包的 `__init__.py` 调用各 `bind_*` 函数注册模块
- 推理引擎(如 InfiniLM)依赖这些 Python API 构建模型计算图

### 设计模式
**外观模式(Facade)**: 为复杂的 C++ 运算符提供简洁的 Python 接口,隐藏底层实现细节。

**策略模式(Strategy)**: 通过 `algo` 参数(如 RoPE)允许运行时选择算法变体。

**适配器模式(Adapter)**: 将 `std::optional<Tensor>` 等 C++ 类型适配为 Python 的 `None` 语义。
