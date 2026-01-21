# InfiniCore Operators 实现文档

本模块实现了 InfiniCore 框架的核心算子集合，涵盖基础数学运算、神经网络层、注意力机制、位置编码等关键操作，采用统一的多设备分发架构支持多种硬件后端（CPU、CUDA、Ascend、Kunlun 等）。

## 1. 模块结构

- **`add.hpp`**: 逐元素加法算子，支持张量加法及运算符重载
- **`add_rms_norm.hpp`**: 融合加法与 RMS 归一化的复合算子，用于 Transformer 残差连接
- **`attention.hpp`**: 标准注意力机制算子，支持 KV 缓存
- **`causal_softmax.hpp`**: 因果掩码 Softmax，用于自回归模型
- **`embedding.hpp`**: 嵌入层查找算子
- **`gemm.hpp`**: 通用矩阵乘法（GEMM）接口，支持 alpha/beta 缩放
- **`linear.hpp`**: 线性变换层（全连接层），包含可选偏置
- **`matmul.hpp`**: 矩阵乘法算子，支持 alpha 缩放因子
- **`mul.hpp`**: 逐元素乘法算子
- **`ones.hpp`**: 全 1 张量生成算子
- **`paged_attention.hpp`**: 分页注意力机制算子，支持 ALiBi 偏置
- **`paged_attention_prefill.hpp`**: 分页注意力预填充阶段算子，处理变长序列批处理
- **`paged_caching.hpp`**: 分页 KV 缓存写入算子
- **`random_sample.hpp`**: 随机采样算子，支持 top-k、top-p 和温度采样
- **`rearrange.hpp`**: 张量重排算子
- **`rms_norm.hpp`**: RMS 归一化算子
- **`rope.hpp`**: 旋转位置编码（RoPE）算子
- **`silu.hpp`**: SiLU 激活函数（Swish）算子
- **`swiglu.hpp`**: SwiGLU 激活函数算子，GLU 变体
- **`common/op.hpp`**: 算子基类和公共定义
- **`common/dispatcher.hpp`**: 设备分发器模板类
- **`common/cache.hpp`**: 算子级 LRU 缓存模板类

## 2. 核心类

### `OpDispatcher<Fn>`
- **Location**: `common/dispatcher.hpp`
- **Primary Function**: 多设备算子分发器，为不同硬件后端注册和查找算子实现函数
- **Key Members**:
  - `table_`: `std::array<Fn, static_cast<size_t>(Device::Type::COUNT>` - 设备类型到函数指针的查找表
- **Core Methods**:
  - `registerDevice(Device::Type device_type, Fn fn, bool override_existing)`: 注册单个设备的算子实现，默认覆盖已存在的实现
  - `registerDevice(std::initializer_list<Device::Type> device_types, Fn fn)`: 批量注册多个设备的算子实现
  - `registerAll(Fn fn)`: 为所有设备类型注册同一实现
  - `lookup(Device::Type device_type)`: O(1) 时间复杂度查找指定设备的算子函数
- **Lifecycle**: 值类型，每个算子类持有静态单例实例

### `OpCache<Key, Value>`
- **Location**: `common/cache.hpp`
- **Primary Function**: 设备感知的 LRU 缓存，用于缓存算子中间结果（如 CUDA kernels、算法特化实现）
- **Key Members**:
  - `caches_`: `std::array<CacheVector, static_cast<size_t>(Device::Type::COUNT)>` - 二维缓存结构，第一维为设备类型，第二维为设备索引
  - `capacity_`: `size_t` - 缓存容量上限
  - `destructor_`: `Destructor` - 自定义资源析构函数
- **Core Methods**:
  - `getCache(Device::Type device_type, size_t device_index)`: 获取指定设备的缓存实例，自动扩容缓存向量
  - `getCache(Device device)`: 便捷方法，从 Device 对象提取类型和索引
  - `setCapacity(size_t capacity)`: 动态调整所有缓存实例的容量
  - `clear()`: 清空所有缓存，正确处理设备上下文切换
- **Implementation Details**:
  - 基于 `infinicore::common::LRUCache` 实现
  - `clear()` 方法在清理前切换到目标设备上下文，清理后恢复原设备
  - 缓存向量按需扩容：当 `device_index >= cache_vector.size()` 时执行 `resize(device_index + 1)`

### `Add`
- **Location**: `add.hpp`
- **Primary Function**: 逐元素加法运算，支持函数式和运算符重载两种调用方式
- **Schema**: `void (*)(Tensor, Tensor, Tensor)` - (output, input_a, input_b)
- **Core Methods**:
  - `execute(Tensor c, Tensor a, Tensor b)`: 调用设备特定实现执行加法
  - `dispatcher()`: 返回静态分发器单例引用
- **Public APIs**:
  - `Tensor add(Tensor a, Tensor b)`: Out-of-place API，自动分配输出张量
  - `void add_(Tensor c, Tensor a, Tensor b)`: In-place API，写入预分配张量
  - `Tensor operator+(Tensor a, Tensor b)`: 运算符重载，提供语法糖

### `AddRMSNorm`
- **Location**: `add_rms_norm.hpp`
- **Primary Function**: 融合加法与 RMS 归一化的复合算子，用于 Transformer 层的残差连接和归一化，返回归一化结果和残差和
- **Schema**: `void (*)(Tensor, Tensor, Tensor, Tensor, Tensor, float)` - (normalized_output, residual_output, input_a, input_b, weight, epsilon)
- **Core Methods**:
  - `execute(Tensor y, Tensor residual_out, Tensor a, Tensor b, Tensor weight, float epsilon)`: 执行融合运算
- **Public APIs**:
  - `std::pair<Tensor, Tensor> add_rms_norm(Tensor a, Tensor b, Tensor weight, float epsilon)`: 返回归一化结果和残差和
  - `void add_rms_norm_(Tensor y, Tensor residual_out, Tensor a, Tensor b, Tensor weight, float epsilon)`: In-place 版本
- **Implementation Details**:
  - 默认 epsilon = 1e-5f
  - 融合设计减少内存访问和 kernel 启动开销

### `Attention`
- **Location**: `attention.hpp`
- **Primary Function**: 标准注意力机制实现，支持 KV 缓存用于推理加速
- **Schema**: `void (*)(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, size_t)` - (output, query, key, value, k_cache, v_cache, position)
- **Core Methods**:
  - `execute(Tensor out, Tensor q, Tensor k, Tensor v, Tensor k_cache, Tensor v_cache, size_t pos)`: 计算注意力输出
- **Public APIs**:
  - `Tensor attention(Tensor q, Tensor k, Tensor v, Tensor k_cache, Tensor v_cache, size_t pos)`
  - `void attention_(Tensor out, Tensor q, Tensor k, Tensor v, Tensor k_cache, Tensor v_cache, size_t pos)`
- **Implementation Details**:
  - `pos` 参数指定当前生成 token 的位置，用于增量式 KV 缓存更新

### `CausalSoftmax`
- **Location**: `causal_softmax.hpp`
- **Primary Function**: 带因果掩码的 Softmax，确保自回归模型中当前位置只能关注之前的位置
- **Schema**: `void (*)(Tensor, Tensor)` - (output, input)
- **Core Methods**:
  - `execute(Tensor output, Tensor input)`: 在最后一个维度上执行因果 softmax
- **Public APIs**:
  - `Tensor causal_softmax(Tensor input)`
  - `void causal_softmax_(Tensor output, Tensor input)`

### `RMSNorm`
- **Location**: `rms_norm.hpp`
- **Primary Function**: RMS 归一化（Root Mean Square Normalization），Layer Norm 的变体，无需中心化操作
- **Schema**: `void (*)(Tensor, Tensor, Tensor, float)` - (output, input, weight, epsilon)
- **Core Methods**:
  - `execute(Tensor y, Tensor x, Tensor weight, float epsilon)`: 执行 RMS 归一化
- **Public APIs**:
  - `Tensor rms_norm(Tensor x, Tensor weight, float epsilon)`: 默认 epsilon=1e-5f
  - `void rms_norm_(Tensor y, Tensor x, Tensor weight, float epsilon)`
- **Mathematical Formula**:
  ```
  y = x / sqrt(mean(x²) + epsilon) * weight
  ```

### `PagedAttention`
- **Location**: `paged_attention.hpp`
- **Primary Function**: 分页注意力机制，用于高效处理变长序列和动态批处理
- **Schema**: `void (*)(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, std::optional<Tensor>, float)` - (output, query, k_cache, v_cache, block_tables, cache_lens, alibi_slopes, scale)
- **Core Methods**:
  - `execute(...)`: 执行分页注意力计算
- **Public APIs**:
  - `Tensor paged_attention(Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, Tensor cache_lens, std::optional<Tensor> alibi_slopes, float scale)`
- **Key Parameters**:
  - `block_tables`: 逻辑块到物理块的映射表，实现非连续 KV 存储的灵活管理
  - `cache_lens`: 每个请求的实际 KV 缓存长度
  - `alibi_slopes`: ALiBi（Attention with Linear Biases）偏置斜率，可选参数
  - `scale`: 缩放因子，通常为 `1/sqrt(head_dim)`

### `PagedAttentionPrefill`
- **Location**: `paged_attention_prefill.hpp`
- **Primary Function**: 分页注意力的预填充阶段，处理 prompt 阶段的批量计算，支持变长序列
- **Schema**: `void (*)(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, std::optional<Tensor>, float)` - (output, query, k_cache, v_cache, block_tables, total_kv_lens, cum_seqlens_q, alibi_slopes, scale)
- **Core Methods**:
  - `execute(...)`: 执行预填充阶段的注意力计算
- **Public APIs**:
  - `Tensor paged_attention_prefill(Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, Tensor total_kv_lens, Tensor cum_seqlens_q, std::optional<Tensor> alibi_slopes, float scale)`
- **Key Parameters**:
  - `total_kv_lens`: 每个请求的完整 KV 序列长度
  - `cum_seqlens_q`: Query 序列的累积长度（前缀和数组），用于在打包张量中定位不同序列的边界
  - 其他参数同 `PagedAttention`
- **Implementation Details**:
  - 使用打包张量格式处理变长序列，减少内存碎片
  - `cum_seqlens_q` 实现高效的数据分割，避免显式循环

### `PagedCaching`
- **Location**: `paged_caching.hpp`
- **Primary Function**: 将新生成的 Key 和 Value 写入分页 KV 缓存
- **Schema**: `void (*)(Tensor, Tensor, Tensor, Tensor, Tensor)` - (k_cache, v_cache, key, value, slot_mapping)
- **Core Methods**:
  - `execute(Tensor k_cache, Tensor v_cache, Tensor k, Tensor v, Tensor slot_mapping)`: 写入缓存
- **Public APIs**:
  - `void paged_caching_(Tensor k_cache, Tensor v_cache, Tensor k, Tensor v, Tensor slot_mapping)`
- **Key Parameters**:
  - `slot_mapping`: 每个 token 应该写入的物理缓存槽位索引

### `RandomSample`
- **Location**: `random_sample.hpp`
- **Primary Function**: 从 logits 分布中进行随机采样，支持多种采样策略
- **Schema**: `void (*)(Tensor, Tensor, float, float, int, float)` - (indices, logits, random_val, topp, topk, temperature)
- **Core Methods**:
  - `execute(Tensor indices, Tensor logits, float random_val, float topp, int topk, float temperature)`: 执行采样
- **Public APIs**:
  - `Tensor random_sample(Tensor logits, float random_val, float topp, int topk, float temperature)`
  - `void random_sample_(Tensor indices, Tensor logits, float random_val, float topp, int topk, float temperature)`
- **Key Parameters**:
  - `random_val`: 随机数 [0, 1)，用于控制采样随机性
  - `topp`: top-p（nucleus）采样阈值，保留累积概率达到 p 的最小 token 集合
  - `topk`: top-k 采样参数，仅保留概率最高的 k 个 token
  - `temperature`: 温度参数，控制分布平滑度（越低越确定性）

### `RoPE`
- **Location**: `rope.hpp`
- **Primary Function**: 应用旋转位置编码到输入张量
- **Schema**: `void (*)(Tensor, const Tensor&, const Tensor&, const Tensor&, const Tensor&, infinicore::nn::RoPE::Algo)` - (output, input, positions, sin_table, cos_table, algorithm)
- **Core Methods**:
  - `execute(Tensor x_out, const Tensor &x, const Tensor &pos, const Tensor &sin_table, const Tensor &cos_cache, infinicore::nn::RoPE::Algo algo)`: 应用 RoPE 变换
- **Public APIs**:
  - `Tensor rope(const Tensor &x, const Tensor &pos, const Tensor &sin_table, const Tensor &cos_table, infinicore::nn::RoPE::Algo algo)`
  - `void rope_(Tensor x_out, const Tensor &x, const Tensor &pos, const Tensor &sin_table, const Tensor &cos_table, infinicore::nn::RoPE::Algo algo)`
- **Supported Algorithms**:
  - `Algo::GPT_J`: 偶数维度应用 sin，奇数维度应用 cos（交错模式）
  - `Algo::GPT_NEOX`: 前半维度应用 sin，后半维度应用 cos（连续模式）
- **Dependencies**: `infinicore::nn::RoPE::Algo` 枚举定义在 `../nn/rope.hpp`

### `Silu`
- **Location**: `silu.hpp`
- **Primary Function**: SiLU（Swish）激活函数：`silu(x) = x * sigmoid(x)`
- **Schema**: `void (*)(Tensor, Tensor)` - (output, input)
- **Core Methods**:
  - `execute(Tensor output, Tensor input)`: 计算 SiLU 激活
- **Public APIs**:
  - `Tensor silu(Tensor input)`
  - `void silu_(Tensor output, Tensor input)`

### `SwiGLU`
- **Location**: `swiglu.hpp`
- **Primary Function**: SwiGLU 激活函数，GLU（Gated Linear Unit）的 Swish 变体
- **Schema**: `void (*)(Tensor, Tensor, Tensor)` - (output, input_a, input_b)
- **Core Methods**:
  - `execute(Tensor c, Tensor a, Tensor b)`: 计算 `SwiGLU(a, b) = (SiLU(a) * b)`
- **Public APIs**:
  - `Tensor swiglu(Tensor a, Tensor b)`
  - `void swiglu_(Tensor c, Tensor a, Tensor b)`
- **Mathematical Formula**:
  ```
  SwiGLU(a, b) = SiLU(a) ⊙ b = (a * σ(a)) ⊙ b
  ```
  其中 σ 是 sigmoid 函数，⊙ 表示逐元素乘法

### `Gemm`
- **Location**: `gemm.hpp`
- **Primary Function**: 通用矩阵乘法（GEneral Matrix Multiply），支持 BLAS 级别的 alpha/beta 缩放
- **Schema**: 通过 `INFINICORE_GRAPH_OP_CLASS` 宏自动生成，签名为 `void (*)(Tensor, Tensor, Tensor, float, float)` - (C, A, B, alpha, beta)
- **Public APIs**:
  - `Tensor gemm(Tensor a, Tensor b, float alpha = 1.0f, float beta = 0.0f)`: 默认计算 `C = alpha * A * B + beta * C`
  - `void gemm_(Tensor c, Tensor a, Tensor b, float alpha, float beta)`
- **Implementation Details**:
  - 使用 `INFINICORE_GRAPH_OP_CLASS` 宏自动生成类定义和分发器
  - 宏定义位于 `../graph/graph.hpp`

### `Linear`
- **Location**: `linear.hpp`
- **Primary Function**: 线性变换层（全连接层），计算 `output = input @ weight.T + bias`
- **Public APIs**:
  - `Tensor linear(Tensor input, Tensor weight, std::optional<Tensor> bias)`: bias 可选
  - `void linear_(Tensor out, Tensor input, Tensor weight, std::optional<Tensor> bias)`
- **Implementation Details**:
  - 使用 `std::optional<Tensor>` 表示可选的 bias 参数
  - 内部可能调用 GEMM 或 MatMul 算子实现

### `MatMul`
- **Location**: `matmul.hpp`
- **Primary Function**: 矩阵乘法算子，支持 alpha 缩放
- **Public APIs**:
  - `Tensor matmul(Tensor a, Tensor b, float alpha = 1.0f)`: 计算 `C = alpha * (A @ B)`
  - `void matmul_(Tensor c, Tensor a, Tensor b, float alpha = 1.0f)`
- **Difference from GEMM**: 不支持 beta 参数，接口更简洁

### `Mul`
- **Location**: `mul.hpp`
- **Primary Function**: 逐元素乘法
- **Schema**: `void (*)(Tensor, Tensor, Tensor)` - (output, input_a, input_b)
- **Core Methods**:
  - `execute(Tensor c, Tensor a, Tensor b)`: 执行逐元素乘法
- **Public APIs**:
  - `Tensor mul(Tensor a, Tensor b)`
  - `void mul_(Tensor c, Tensor a, Tensor b)`

### `Ones`
- **Location**: `ones.hpp`
- **Primary Function**: 生成全 1 张量
- **Schema**: `void (*)(Tensor)` - (output)
- **Core Methods**:
  - `execute(Tensor output)`: 将输出张量所有元素填充为 1
- **Public APIs**:
  - `Tensor ones()`: 自动推断形状并分配
  - `void ones_(Tensor output)`: 填充预分配张量

### `Rearrange`
- **Location**: `rearrange.hpp`
- **Primary Function**: 张量重排（类似于 einops.rearrange），用于维度变换和转置
- **Schema**: `void (*)(Tensor, Tensor)` - (output, input)
- **Core Methods**:
  - `execute(Tensor y, Tensor x)`: 执行重排操作
- **Public APIs**:
  - `Tensor rearrange(Tensor x)`
  - `void rearrange_(Tensor y, Tensor x)`
- **Note**: 具体重排规则可能由张量元数据或额外参数指定

### `embedding`
- **Location**: `embedding.hpp`
- **Primary Function**: 嵌入层查找，将离散索引映射到连续向量空间
- **Public APIs**:
  - `Tensor embedding(Tensor input, Tensor weight)`: input 为索引张量，weight 为嵌入矩阵
  - `void embedding_(Tensor out, Tensor input, Tensor weight)`
- **Implementation Details**:
  - 不使用 `OpDispatcher` 机制，可能是简化实现或内联在其他算子中

## 3. API Interface

```cpp
// 基础数学运算
Tensor add(Tensor a, Tensor b);
void add_(Tensor c, Tensor a, Tensor b);
Tensor operator+(Tensor a, Tensor b);

Tensor mul(Tensor a, Tensor b);
void mul_(Tensor c, Tensor a, Tensor b);

// 矩阵运算
Tensor gemm(Tensor a, Tensor b, float alpha = 1.0f, float beta = 0.0f);
void gemm_(Tensor c, Tensor a, Tensor b, float alpha, float beta);

Tensor matmul(Tensor a, Tensor b, float alpha = 1.0f);
void matmul_(Tensor c, Tensor a, Tensor b, float alpha = 1.0f);

Tensor linear(Tensor input, Tensor weight, std::optional<Tensor> bias);
void linear_(Tensor out, Tensor input, Tensor weight, std::optional<Tensor> bias);

// 归一化
Tensor rms_norm(Tensor x, Tensor weight, float epsilon = 1e-5f);
void rms_norm_(Tensor y, Tensor x, Tensor weight, float epsilon);

std::pair<Tensor, Tensor> add_rms_norm(Tensor a, Tensor b, Tensor weight, float epsilon = 1e-5f);
void add_rms_norm_(Tensor y, Tensor residual_out, Tensor a, Tensor b, Tensor weight, float epsilon);

// 激活函数
Tensor silu(Tensor input);
void silu_(Tensor output, Tensor input);

Tensor swiglu(Tensor a, Tensor b);
void swiglu_(Tensor c, Tensor a, Tensor b);

// 注意力机制
Tensor attention(Tensor q, Tensor k, Tensor v, Tensor k_cache, Tensor v_cache, size_t pos);
void attention_(Tensor out, Tensor q, Tensor k, Tensor v, Tensor k_cache, Tensor v_cache, size_t pos);

Tensor paged_attention(Tensor q, Tensor k_cache, Tensor v_cache,
                      Tensor block_tables, Tensor cache_lens,
                      std::optional<Tensor> alibi_slopes, float scale);
void paged_attention_(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache,
                      Tensor block_tables, Tensor cache_lens,
                      std::optional<Tensor> alibi_slopes, float scale);

Tensor paged_attention_prefill(Tensor q, Tensor k_cache, Tensor v_cache,
                               Tensor block_tables, Tensor total_kv_lens,
                               Tensor cum_seqlens_q,
                               std::optional<Tensor> alibi_slopes, float scale);
void paged_attention_prefill_(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache,
                              Tensor block_tables, Tensor total_kv_lens,
                              Tensor cum_seqlens_q,
                              std::optional<Tensor> alibi_slopes, float scale);

void paged_caching_(Tensor k_cache, Tensor v_cache, Tensor k, Tensor v, Tensor slot_mapping);

// 位置编码
Tensor rope(const Tensor &x, const Tensor &pos,
            const Tensor &sin_table, const Tensor &cos_table,
            infinicore::nn::RoPE::Algo algo);
void rope_(Tensor x_out, const Tensor &x, const Tensor &pos,
           const Tensor &sin_table, const Tensor &cos_table,
           infinicore::nn::RoPE::Algo algo);

// 序列操作
Tensor causal_softmax(Tensor input);
void causal_softmax_(Tensor output, Tensor input);

Tensor random_sample(Tensor logits, float random_val,
                     float topp, int topk, float temperature);
void random_sample_(Tensor indices, Tensor logits, float random_val,
                    float topp, int topk, float temperature);

// 工具函数
Tensor ones();
void ones_(Tensor output);

Tensor rearrange(Tensor x);
void rearrange_(Tensor y, Tensor x);

Tensor embedding(Tensor input, Tensor weight);
void embedding_(Tensor out, Tensor input, Tensor weight);
```

## 4. Usage Example

```cpp
#include "infinicore/ops.hpp"
#include "infinicore/tensor.hpp"

using namespace infinicore;
using namespace infinicore::op;

// 初始化设备和上下文
Device device(Device::Type::NVIDIA, 0);
context::setDevice(device);

// 示例 1: 基础算术运算
Tensor a = ones({2, 3}, DataType::F32, device);
Tensor b = ones({2, 3}, DataType::F32, device);
Tensor c = add(a, b);  // c = a + b

// 示例 2: 线性变换
Tensor input = randn({32, 768}, DataType::F32, device);
Tensor weight = randn({768, 2048}, DataType::F32, device);
Tensor bias = randn({2048}, DataType::F32, device);
Tensor output = linear(input, weight, bias);

// 示例 3: RMS 归一化 + 残差连接
Tensor hidden_state = /* from previous layer */;
Tensor residual = hidden_state;
Tensor normalized, new_residual;
std::tie(normalized, new_residual) = add_rms_norm(hidden_state, residual, weight, 1e-5f);

// 示例 4: SwiGLU 激活
Tensor gate = /* projection */;
Tensor up = /* projection */;
Tensor activated = swiglu(gate, up);

// 示例 5: 应用 RoPE 位置编码
Tensor q = randn({1, 32, 128, 64}, DataType::F32, device);  // [batch, heads, seq_len, head_dim]
Tensor pos = arange(0, 128, 1, DataType::I32, device);     // [seq_len]
infinicore::nn::RoPE rope_layer(64, 2048, 10000.0,
                                 infinicore::nn::RoPE::Algo::GPT_NEOX);
Tensor q_rotated = rope(q, pos, rope_layer.sin_cache, rope_layer.cos_cache,
                        infinicore::nn::RoPE::Algo::GPT_NEOX);

// 示例 6: 分页注意力
Tensor query = /* current query */;
Tensor k_cache = /* physical key cache [num_blocks, block_size, num_heads, head_dim] */;
Tensor v_cache = /* physical value cache */;
Tensor block_tables = /* [batch, max_blocks_per_seq] */;
Tensor cache_lens = /* [batch] */;
Tensor attn_out = paged_attention(query, k_cache, v_cache,
                                  block_tables, cache_lens,
                                  std::nullopt,  // no ALiBi
                                  1.0f / sqrt(64.0f));

// 示例 7: 写入 KV 缓存
Tensor new_key = /* current key */;
Tensor new_value = /* current value */;
Tensor slot_mapping = /* physical slot indices [batch, seq_len] */;
paged_caching_(k_cache, v_cache, new_key, new_value, slot_mapping);

// 示例 8: 随机采样
Tensor logits = /* model output logits [vocab_size] */;
float random_val = 0.42f;  // from random generator
float temperature = 0.8f;
float topp = 0.9f;
int topk = 50;
Tensor token_id = random_sample(logits, random_val, topp, topk, temperature);

// 示例 9: GEMM 矩阵乘法
Tensor A = randn({1024, 768}, DataType::F32, device);
Tensor B = randn({768, 1024}, DataType::F32, device);
Tensor C = gemm(A, B, 1.0f, 0.0f);  // C = 1.0 * A @ B + 0.0 * C
```

## 5. Implementation Details

### 架构设计模式

- **策略模式（Strategy Pattern）**: `OpDispatcher` 为每种设备类型存储不同的算子实现策略，运行时根据 `Device::Type` 动态选择
- **单例模式（Singleton Pattern）**: 每个算子类通过 `static OpDispatcher &dispatcher()` 维护全局唯一的分发器实例
- **模板方法模式（Template Method Pattern）**: 所有算子类统一提供 `execute()` 静态方法，通过 `dispatcher()` 调用设备特定实现

### 设备分发机制

- **查找表实现**: `OpDispatcher` 使用 `std::array<Fn, (size_t)Device::Type::COUNT>` 作为函数指针表，O(1) 查找复杂度
- **设备类型枚举**: 支持 CPU、NVIDIA、CAMBRICON、ASCEND、METAX、MOORE、ILUVATAR、KUNLUN、HYGEN、QY 共 10 种设备类型
- **注册策略**: `registerDevice()` 默认覆盖已存在的实现（`override_existing = true`），支持动态替换算子实现

### 内存管理

- **Out-of-place API**: 函数名无后缀（如 `add`），自动分配输出张量，返回新 Tensor 对象
- **In-place API**: 函数名带下划线后缀（如 `add_`），写入预分配张量，避免额外内存分配
- **智能指针**: Tensor 内部使用 `std::shared_ptr` 管理底层数据，自动引用计数

### 缓存机制

- **LRU 缓存**: `OpCache` 基于 `infinicore::common::LRUCache` 实现，自动淘汰最近最少使用的条目
- **设备隔离**: 每个设备类型和设备索引拥有独立的缓存实例，避免跨设备数据竞争
- **上下文切换**: `OpCache::clear()` 在清理缓存前自动切换到目标设备上下文，清理后恢复原设备
- **按需扩容**: `getCache()` 方法当 `device_index >= cache_vector.size()` 时自动扩容

### 并发安全

- **静态分发器**: 每个算子类的 `dispatcher()` 返回静态单例引用，多线程共享
- **设备上下文**: 通过 `context::setDevice()` 和 `context::getDevice()` 管理每线程设备上下文
- **无锁设计**: 分发器查找表为只读结构，注册操作通常在初始化阶段单线程完成

### 性能优化

- **算子融合**: `AddRMSNorm` 将加法和归一化融合为单个 kernel，减少内存访问和 kernel 启动开销
- **分页注意力**: `PagedAttention` 和 `PagedAttentionPrefill` 支持非连续 KV 存储，提高显存利用率
- **打包张量**: `PagedAttentionPrefill` 使用 `cum_seqlens_q` 前缀和数组处理变长序列，避免显式循环和内存碎片
- **ALiBi 支持**: 分页注意力算子可选 `alibi_slopes` 参数，支持 Attention with Linear Biases

### 错误处理

- **设备查找失败**: `OpDispatcher::lookup()` 使用 `std::array::at()`，越界访问抛出 `std::out_of_range` 异常
- **空可选参数**: `Linear` 和 `PagedAttention` 使用 `std::optional<Tensor>` 表示可选参数，调用前需检查 `has_value()`
- **设备上下文验证**: `OpCache::clear()` 在切换设备前保存当前设备，异常时确保恢复

### 依赖关系

- **Tensor 定义**: `infinicore::Tensor` 定义在 `../tensor.hpp`
- **设备管理**: `infinicore::Device` 和 `Device::Type` 定义在 `../device.hpp`
- **上下文管理**: `infinicore::context::getDevice()` 和 `setDevice()` 定义在 `../context/context.hpp`
- **RoPE 算法**: `infinicore::nn::RoPE::Algo` 枚举定义在 `../nn/rope.hpp`
- **图操作宏**: `INFINICORE_GRAPH_OP_CLASS` 定义在 `../graph/graph.hpp`

### 宏定义

- **`INFINICORE_GRAPH_OP_CLASS`**: 自动生成算子类定义，包含：
  - 构造函数
  - `execute()` 静态方法
  - `dispatcher()` 静态方法
  - `schema` 类型定义
- **使用示例**: `INFINICORE_GRAPH_OP_CLASS(Gemm, Tensor, Tensor, Tensor, float, float)` 生成 `Gemm` 类，接受 5 个模板参数（返回类型和 4 个参数类型）

### 算子实现注册

每个算子的设备特定实现（如 CUDA kernel、CPU 实现）在对应的 backend 文件中通过以下模式注册：

```cpp
// 在 cuda/backend.cpp 中
void cuda_add_impl(Tensor c, Tensor a, Tensor b) {
    // CUDA kernel 实现
}

// 初始化时注册
static bool cuda_add_registered = []() {
    op::Add::dispatcher().registerDevice(
        Device::Type::NVIDIA,
        cuda_add_impl
    );
    return true;
}();
```

### 公共 API 设计

- **一致性**: 所有算子同时提供 out-of-place 和 in-place 两个版本
- **运算符重载**: `Add` 提供 `operator+()`，`Mul` 可扩展提供 `operator*()`
- **默认参数**: 常用参数提供默认值（如 `epsilon = 1e-5f`, `alpha = 1.0f`）
- **返回值优化**: Out-of-place API 返回 Tensor 对象，利用 C++17 返回值优化（RVO）避免拷贝
