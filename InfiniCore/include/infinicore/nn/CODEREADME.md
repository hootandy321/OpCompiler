# `Neural Network (nn)` Core Implementation Documentation

本模块提供了构建深度神经网络的核心组件层，采用了 PyTorch 风格的模块化设计范式。实现了从基础模块管理到具体层操作的完整抽象体系，包括线性层、嵌入层、归一化层以及旋转位置编码等现代 Transformer 架构关键组件。

## 1. Module Structure

- **`module.hpp`**: 定义了 `Module` 基类和宏工具集，提供了层级化参数管理、状态字典持久化、子模块注册等基础设施
- **`parameter.hpp`**: 实现了 `Parameter` 类，继承自 `Tensor` 并增加了张量并行配置元数据
- **`linear.hpp`**: 提供了三种线性层实现：`Linear`（标准）、`ColumnParallelLinear`（列并行）、`RowParallelLinear`（行并行）
- **`embedding.hpp`**: 实现了 `Embedding` 层，用于离散 token 到稠密向量的查表映射
- **`rmsnorm.hpp`**: 实现了 `RMSNorm` 层，用于 LLaMA 等现代语言模型的根均方层归一化
- **`rope.hpp`**: 实现了 `RoPE`（Rotary Position Embedding）层，支持 GPT-J 和 GPT-NeoX 两种算法风格

## 2. Core Classes

### `Module`
- **Location**: `module.hpp`
- **Primary Function**: 所有神经网络模块的抽象基类，提供了层级化参数管理、子模块注册、状态字典序列化等核心功能
- **Key Members**:
  - `device_`: 当前模块所在设备（CPU/CUDA 等）
  - `submodules_`: 子模块的名称到 `shared_ptr<Module>` 的哈希映射
  - `parameters_`: 可学习参数的名称到 `Parameter` 的哈希映射
  - `buffers_`: 不可学习但需持久化的张量（如 BatchNorm 统计量、RoPE 缓存）
- **Core Methods**:
  - `state_dict() const`: 递归收集所有子模块的参数和缓冲区，返回扁平化的 `std::unordered_map<std::string, Parameter>`，键名为点分路径（如 "layer1.weight"）
  - `load_state_dict(const std::unordered_map<std::string, Tensor>&)`: 从状态字典递归加载参数，支持部分加载和容错处理
  - `register_parameter(const std::string&, Parameter)`: 注册可学习参数到 `parameters_`，返回 `Tensor` 视图用于构造函数初始化
  - `register_buffer(const std::string&, Parameter)`: 注册非梯度缓冲区到 `buffers_`（如 RoPE 的 sin/cos 缓存）
  - `add_module<M>(const std::string&, shared_ptr<M>)`: 添加已有子模块到层级结构，使用编译期 `static_assert` 确保 `M` 继承自 `Module`
  - `register_module<M, Args...>(const std::string&, Args&&...)`: 创建并注册新子模块，使用完美转发将参数传递给构造函数
  - `register_modules<M, Args...>(size_t, const std::string&, Args&&...)`: 批量创建同类型子模块，自动命名为 "name.0", "name.1", ...
- **Lifecycle**:
  - 默认构造函数为 `= default`，子类在构造函数中通过宏或手动调用 `register_*` 方法完成初始化
  - 采用组合模式，父子模块通过 `shared_ptr` 形成树状结构，递归算法遍历整棵树进行序列化/反序列化
  - 析构函数自动管理子模块生命周期（通过 `shared_ptr` 引用计数）

### `Parameter`
- **Location**: `parameter.hpp`
- **Primary Function**: 继承自 `Tensor` 的可训练参数包装类，增加了张量并行（Tensor Parallelism, TP）的元数据配置
- **Key Members**:
  - `tp_dim_`: 张量并行沿哪个维度切分（0 表示按行切分，1 表示按列切分）
  - `tp_rank_`: 当前分片在 TP 组中的 rank（例如：2 卡并行中，卡 0 的 rank 为 0）
  - `tp_size_`: TP 组的总卡数
- **Core Methods**:
  - `Parameter(const Tensor&, Size tp_dim, Size tp_rank, Size tp_size)`: 从现有 Tensor 构造参数，并附加 TP 元数据
  - `Parameter(const Shape&, const DataType&, const Device&, Size tp_dim, Size tp_rank, Size tp_size)`: 直接分配未初始化内存
  - `load_blob(const void* data)`: 从原始内存二进制数据加载参数值（用于从磁盘加载权重）
  - `load(const Tensor& tensor)`: 从另一个 Tensor 加载，自动处理 TP 分片逻辑
- **Design**: 零开销抽象，内存布局与 `Tensor` 完全一致，仅增加 3 个 `Size` 字段（通常 8 字节）用于存储 TP 配置

### `BaseLinear`
- **Location**: `linear.hpp`
- **Primary Function**: 线性变换层的抽象基类，实现 `output = input @ weight.T + bias`
- **Key Members**:
  - `weight_`: 形状为 `[out_features, in_features]` 的参数矩阵
  - `bias_`: 形状为 `[out_features]` 的偏置向量（可选，由 `has_bias_` 控制）
  - `in_features_`, `out_features_`: 输入输出特征维度
  - `dtype_`: 参数数据类型（通常为 `F32` 或 `F16`）
- **Core Methods**:
  - `forward(Tensor& input) const`: 计算线性变换，使用 `ops::matmul()` 和 `ops::add()`
  - `forward(Tensor& input, Tensor& residual) const`: 支持残差连接的变体（用于 Transformer），计算 `output = input @ weight.T + bias + residual`
  - `compute_linear(Tensor& input) const`: 内部辅助方法，封装矩阵乘法和可选偏置添加的通用逻辑
- **Weight Layout**: 权重矩阵采用 `[out_features, in_features]` 布局，转置后与输入相乘（即 `weight.T` 参与 matmul），这与 PyTorch 的 `nn.Linear` 一致

### `Linear` (inherits BaseLinear)
- **Location**: `linear.hpp`
- **Primary Function**: 标准线性层，无张量并行
- **Core Methods**:
  - `forward(Tensor& input) const`: 调用 `compute_linear()` 执行标准矩阵乘法
  - `extra_repr() const`: 返回字符串表示（如 "Linear(in_features=768, out_features=3072, bias=True)"）
- **Use Case**: 单卡训练或推理，不涉及模型并行

### `ColumnParallelLinear` (inherits BaseLinear)
- **Location**: `linear.hpp`
- **Primary Function**: 列并行线性层，权重矩阵按**输出维度**切分（每个 rank 拥有部分 `out_features`）
- **Key Members**:
  - `tp_rank_`: 当前 rank（0 到 tp_size_-1）
  - `tp_size_`: 并行组总卡数
- **Parallelism Strategy**:
  - 权重 `weight` 形状为 `[out_features / tp_size, in_features]`（每个 rank 持有列分片）
  - 输入 `input` 在所有 rank 上**完全复制**（identical）
  - 输出 `output` 形状为 `[out_features / tp_size, ...]`（每个 rank 计算部分输出）
  - 后续通常接 `AllGather` 或 `Concat` 操作合并各 rank 的输出
- **Use Case**: Transformer 的第一层线性变换（如 MLP 的 fc1，Attention 的 Q/K/V 投影）

### `RowParallelLinear` (inherits BaseLinear)
- **Location**: `linear.hpp`
- **Primary Function**: 行并行线性层，权重矩阵按**输入维度**切分，并执行 All-Reduce 聚合结果
- **Key Members**:
  - `communicator_`: `infinicclComm_t` 类型的通信句柄，用于 NCCL 集合通信
  - `tp_rank_`, `tp_size_`: 同列并行
- **Parallelism Strategy**:
  - 权重 `weight` 形状为 `[out_features, in_features / tp_size]`（每个 rank 持有行分片）
  - 输入 `input` 需已按输入维度切分（通常由前一个 `ColumnParallelLinear` 产生）
  - 输出计算后通过 `AllReduce` 在所有 rank 上求和，得到完整输出
  - All-Reduce 代价：O(卡数) 的通信复杂度
- **Use Case**: Transformer 的第二层线性变换（如 MLP 的 fc2，Attention 的 out 投影）

### `Embedding`
- **Location**: `embedding.hpp`
- **Primary Function**: 离散索引到稠密向量的查表映射，等价于实现一个大小为 `[num_embeddings, embedding_dim]` 的查找表
- **Key Members**:
  - `weight_`: 形状为 `[num_embeddings, embedding_dim]` 的嵌入矩阵
  - `num_embeddings_`: 词汇表大小（如 10000 表示 10000 个 token）
  - `embedding_dim_`: 每个嵌入的维度（如 768, 1024, 4096）
  - `padding_idx_`: 可选的填充索引（如 0），该位置的嵌入不参与梯度更新
- **Core Methods**:
  - `forward(const Tensor& indices) const`: 使用 `ops::embedding_lookup()` 执行查表操作
    - 输入 `indices` 形状为任意 `(*)`（如 `[batch_size, seq_len]`）
    - 输出形状为 `(*, embedding_dim)`（如 `[batch_size, seq_len, 768]`）
  - `extra_repr() const`: 返回字符串表示（如 "Embedding(num_embeddings=10000, embedding_dim=768)"）
- **Implementation**: 底层通常调用 `Gather` 操作，从 `weight` 矩阵的行中选择对应索引的向量

### `RMSNorm`
- **Location**: `rmsnorm.hpp`
- **Primary Function**: 根均方层归一化，公式为 `y = (x / RMS(x)) * weight`，其中 `RMS(x) = sqrt(mean(x^2) + eps)`
- **Key Members**:
  - `weight_`: 形状为 `[normalized_shape]` 的可学习缩放参数
  - `normalized_shape_`: 要归一化的特征维度大小（如 hidden_size = 4096）
  - `eps_`: 数值稳定性的小常数（默认 1e-6）
- **Core Methods**:
  - `forward(const Tensor& x) const`: 沿最后一个维度计算 RMS 并归一化
    - 输入形状为 `(*, normalized_shape)`
    - 计算步骤：1) 求 `x^2` 的均值 2) 加 `eps` 后开方得到 RMS 3) 除以 RMS 4) 乘以 `weight`
    - 输出形状与输入相同
  - `extra_repr() const`: 返回字符串表示
- **Difference from LayerNorm**: 不减去均值，不使用 bias 参数，计算量减少约 30%，被 LLaMA、Galactica 等模型采用

### `RoPE`
- **Location**: `rope.hpp`
- **Primary Function**: 旋转位置编码（Rotary Position Embedding），通过旋转矩阵将相对位置信息注入 query 和 key 向量
- **Key Members**:
  - `sin_cache_`, `cos_cache_`: 预计算的 sin/cos 查找表，形状为 `[max_seq_len, head_dim / 2]`
  - `head_dim_`: 每个 attention head 的维度（必须是偶数）
  - `max_seq_len_`: 预缓存的最大序列长度
  - `theta_`: 基频参数（默认 10000.0，控制位置编码的周期性）
  - `algo_`: 算法类型枚举（`GPT_J` 或 `GPT_NEOX`）
- **Core Methods**:
  - `RoPE(size_t head_dim, size_t max_seq_len, double theta, Algo algo, ...)`: 构造函数，调用 `initialize_cache()` 预计算 sin/cos 表
  - `initialize_cache()`: 根据公式 `freq = 1 / (theta^(2i/head_dim))` 计算频率，生成 `sin(pos * freq)` 和 `cos(pos * freq)`
  - `forward(const Tensor& x, const Tensor& pos, bool in_place) const`: 应用 RoPE 变换
    - 输入 `x` 形状为 `(..., head_dim)`，`pos` 形状为 `(*)`（位置 IDs）
    - 根据 `algo_` 选择维度重组策略：
      - `GPT_J`: 交替偶数奇数维度（x0, x1, x2, x3 -> x0, x2, x1, x3）
      - `GPT_NEOX`: 前半部分为 sin 分量，后半部分为 cos 分量（x0...xn/2, xn/2...xn -> sin_part, cos_part）
    - 应用 2D 旋转：`[-x1, x0] @ [cos, sin; -sin, cos]`（偶数对独立旋转）
    - 支持 `in_place` 修改或创建新张量
  - `forward(const Tensor& y, const Tensor& x, const Tensor& pos) const`: 三参数版本，强制输出到 `y`
- **Buffer vs Parameter**: `sin_cache` 和 `cos_cache` 通过 `register_buffer()` 注册，不参与梯度更新和 `state_dict` 保存
- **Complexity**: 预计算 O(max_seq_len * head_dim)，前向推理 O(batch * seq_len * head_dim)，无动态内存分配

## 3. API Interface

```cpp
// ============== Module Registration Macros ==============

// Declare a single submodule (in class definition)
INFINICORE_NN_MODULE(Linear, layer1);  // Expands to: std::shared_ptr<Linear> layer1_

// Declare a vector of submodules
INFINICORE_NN_MODULE_VEC(Linear, layers);  // Expands to: std::vector<std::shared_ptr<Linear>> layers_

// Declare a learnable parameter
INFINICORE_NN_PARAMETER(weight);  // Expands to: infinicore::nn::Parameter weight_

// Declare a non-learnable buffer
INFINICORE_NN_BUFFER(cache);  // Expands to: infinicore::nn::Parameter cache_

// ============== Module Initialization Macros (in constructor) ==============

// Initialize a single module
INFINICORE_NN_MODULE_INIT(layer1, 128, 64);  // layer1_ = register_module<Linear>("layer1", 128, 64)

// Initialize a vector of modules (3 Linear layers, each with 32->16 dimensions)
INFINICORE_NN_MODULE_VEC_INIT(layers, 3, Linear, 32, 16);

// Initialize a parameter with shape {out_features, in_features}, F32 dtype, on CUDA device
INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, DataType::F32, Device(DeviceType::CUDA)));

// Initialize a buffer
INFINICORE_NN_BUFFER_INIT(cache, ({max_seq_len, head_dim}, DataType::F32, device));

// ============== Module Public APIs ==============

// Get state dict (recursive, flattened with dot notation)
const std::unordered_map<std::string, Parameter>& state_dict() const;
// Returns: {"layer1.weight": Tensor, "layer1.bias": Tensor, ...}

// Load state dict (supports partial loading, recursive matching)
void load_state_dict(const std::unordered_map<std::string, Tensor>& _state_dict);
// Matches keys like "layer1.weight" to nested modules automatically

// Load a single parameter from blob (binary data from checkpoint file)
void load_parameter_from_blob(const std::string& name, const void* data);
// Used for efficient weight loading without intermediate Tensor allocation

// ============== Linear Layer APIs ==============

// Standard linear layer: output = input @ weight.T + bias
Linear(size_t in_features, size_t out_features, bool bias = true,
       const DataType& dtype = DataType::F32, const Device& device = Device());

// Column parallel linear (weight sharded on output dimension)
ColumnParallelLinear(size_t in_features, size_t out_features, bool bias = true,
                     const DataType& dtype = DataType::F32, const Device& device = Device(),
                     Size tp_rank = 0, Size tp_size = 1);

// Row parallel linear (weight sharded on input dimension, AllReduce output)
RowParallelLinear(size_t in_features, size_t out_features, bool bias = true,
                  const DataType& dtype = DataType::F32, const Device& device = Device(),
                  Size tp_rank = 0, Size tp_size = 1, infinicclComm_t communicator = nullptr);

// Forward pass
Tensor forward(Tensor& input) const;  // Standard: output = input @ weight.T + bias
Tensor forward(Tensor& input, Tensor& residual) const;  // With residual connection

// ============== Embedding Layer APIs ==============

// Create embedding lookup table
Embedding(size_t num_embeddings, size_t embedding_dim,
          std::optional<int64_t> padding_idx = std::nullopt,
          const DataType& dtype = DataType::F32, const Device& device = Device());

// Forward: lookup embeddings for indices
Tensor forward(const Tensor& indices) const;
// Input: indices of shape (*) (e.g., [batch_size, seq_len])
// Output: embeddings of shape (*, embedding_dim) (e.g., [batch_size, seq_len, 768])

// ============== RMSNorm Layer APIs ==============

// Create RMSNorm layer
RMSNorm(size_t normalized_shape, double eps = 1e-6,
        const DataType& dtype = DataType::F32, const Device& device = Device());

// Forward: y = (x / sqrt(mean(x^2) + eps)) * weight
Tensor forward(const Tensor& x) const;
// Input/Output: shape (*, normalized_shape), normalized over last dimension

// ============== RoPE Layer APIs ==============

// Create RoPE layer with pre-computed sin/cos cache
RoPE(size_t head_dim, size_t max_seq_len, double theta = 10000.0,
     Algo algo = Algo::GPT_J,  // or Algo::GPT_NEOX
     const DataType& dtype = DataType::F32, const Device& device = Device());

// Forward: apply rotary position embedding
Tensor forward(const Tensor& x, const Tensor& pos, bool in_place = false) const;
// Input x: shape (..., head_dim), typically [batch, num_heads, seq_len, head_dim]
// Input pos: shape (*), typically [seq_len] or [batch, seq_len]
// Output: rotated tensor with same shape as x

// Three-parameter version (explicit output tensor)
Tensor forward(const Tensor& y, const Tensor& x, const Tensor& pos) const;
// Writes result to y, useful for in-place operations
```

## 4. Usage Example

```cpp
#include <infinicore/nn/module.hpp>
#include <infinicore/nn/linear.hpp>
#include <infinicore/nn/embedding.hpp>
#include <infinicore/nn/rmsnorm.hpp>
#include <infinicore/nn/rope.hpp>

using namespace infinicore;
using namespace infinicore::nn;

// Example 1: Define a Transformer MLP block using macros
class TransformerMLP : public Module {
protected:
    // Declare submodules and parameters
    INFINICORE_NN_MODULE(Linear, fc1);          // Expansion layer: hidden -> 4*hidden
    INFINICORE_NN_MODULE(Linear, fc2);          // Projection layer: 4*hidden -> hidden
    INFINICORE_NN_PARAMETER(gating);            // Optional gating parameter

public:
    TransformerMLP(size_t hidden_size, const Device& device)
        : device_(device) {

        // Initialize submodules
        // fc1: [hidden_size] -> [4 * hidden_size], column-parallel for tensor parallelism
        INFINICORE_NN_MODULE_INIT(fc1, hidden_size, 4 * hidden_size, true, DataType::F16, device);

        // fc2: [4 * hidden_size] -> [hidden_size], row-parallel with AllReduce
        INFINICORE_NN_MODULE_INIT(fc2, 4 * hidden_size, hidden_size, true, DataType::F16, device);

        // Initialize custom parameter
        INFINICORE_NN_PARAMETER_INIT(gating, ({hidden_size}, DataType::F32, device));
    }

    // Forward pass with residual connection
    Tensor forward(Tensor& x) {
        auto hidden = fc1_->forward(x);       // [batch, seq, hidden] -> [batch, seq, 4*hidden]
        auto activated = ops::gelu(hidden);   // GELU activation
        auto output = fc2_->forward(activated); // [batch, seq, 4*hidden] -> [batch, seq, hidden]
        return output;                         // No residual here (added outside)
    }

private:
    Device device_;
};

// Example 2: Build a GPT-style transformer block
class GPTBlock : public Module {
protected:
    INFINICORE_NN_MODULE(RMSNorm, ln1);
    INFINICORE_NN_MODULE(RMSNorm, ln2);
    INFINICORE_NN_MODULE(Embedding, wte);       // Token embeddings
    INFINICORE_NN_MODULE(RoPE, rope);           // Rotary position embedding
    INFINICORE_NN_MODULE(TransformerMLP, mlp);  // MLP sub-block

public:
    GPTBlock(size_t vocab_size, size_t hidden_size, size_t num_heads,
             size_t max_seq_len, const Device& device)
        : device_(device) {

        // Initialize token embeddings
        INFINICORE_NN_MODULE_INIT(wte, vocab_size, hidden_size, std::nullopt, DataType::F16, device);

        // Initialize RMSNorm layers (eps=1e-5)
        INFINICORE_NN_MODULE_INIT(ln1, hidden_size, 1e-5, DataType::F16, device);
        INFINICORE_NN_MODULE_INIT(ln2, hidden_size, 1e-5, DataType::F16, device);

        // Initialize RoPE with GPT-NeoX style, max_seq_len=2048
        INFINICORE_NN_MODULE_INIT(rope, hidden_size / num_heads,  // head_dim
                                  2048,                           // max_seq_len
                                  10000.0,                        // theta
                                  RoPE::Algo::GPT_NEOX,           // algorithm
                                  DataType::F32, device);

        // Initialize MLP sub-block
        INFINICORE_NN_MODULE_INIT(mlp, hidden_size, device);
    }

    Tensor forward(Tensor& input_ids, Tensor& positions) {
        // Lookup token embeddings
        auto hidden = wte_->forward(input_ids);  // [batch, seq_len] -> [batch, seq_len, hidden]

        // Apply pre-normalization (RMSNorm)
        auto normalized = ln1_->forward(hidden);

        // Apply RoPE to queries and keys (in-place)
        // normalized is usually reshaped to [batch, num_heads, seq_len, head_dim] before RoPE
        auto rotated = rope_->forward(normalized, positions, true);  // in-place rotation

        // Process through MLP
        auto mlp_output = mlp_->forward(rotated);

        // Apply second RMSNorm
        auto final = ln2_->forward(mlp_output);

        return final;
    }

private:
    Device device_;
};

// Example 3: Save and load model state
void save_and_load_example() {
    Device device(DeviceType::CUDA, 0);

    // Create a model
    auto model = std::make_shared<GPTBlock>(10000, 768, 12, 2048, device);

    // Forward pass
    auto input_ids = Tensor::from_data({2, 10}, DeviceType::CPU);  // [batch=2, seq_len=10]
    auto positions = Tensor::arange(0, 10, 1, device);              // [seq_len]
    auto output = model->forward(input_ids, positions);

    // Save state dict to file (pseudo-code)
    auto state_dict = model->state_dict();
    // save_to_disk("model_checkpoint.bin", state_dict);

    // Load state dict (e.g., from PyTorch checkpoint)
    std::unordered_map<std::string, Tensor> loaded_state;
    // load_from_disk("model_checkpoint.bin", loaded_state);

    // Load parameters into model (auto-recursive matching)
    model->load_state_dict(loaded_state);

    // Alternative: load a single parameter from raw memory
    std::vector<float> weight_data(768 * 768);
    // ... fill weight_data from file ...
    model->load_parameter_from_blob("wte.weight", weight_data.data());
}

// Example 4: Tensor Parallelism setup
void tensor_parallel_example() {
    int world_size = 4;  // 4 GPUs
    int rank = 1;        // Current GPU rank

    Device device(DeviceType::CUDA, rank);
    infinicclComm_t comm;  // Assume NCCL communicator initialized

    // Column parallel: weight split on output dimension
    auto col_linear = ColumnParallelLinear(
        768,          // in_features
        3072,         // out_features (total)
        true,         // bias
        DataType::F16,
        device,
        rank,         // tp_rank
        world_size    // tp_size (each GPU has 3072/4 = 768 output features)
    );

    // Row parallel: weight split on input dimension
    auto row_linear = RowParallelLinear(
        3072,         // in_features (total)
        768,          // out_features
        true,         // bias
        DataType::F16,
        device,
        rank,         // tp_rank
        world_size,   // tp_size (each GPU has 3072/4 = 768 input features)
        comm          // NCCL communicator for AllReduce
    );

    // Forward through tensor-parallel layers
    auto input = Tensor::randn({2, 10, 768}, device);
    auto hidden = col_linear.forward(input);  // Output: [2, 10, 768] (sharded)
    auto output = row_linear.forward(hidden); // AllReduce happens internally
    // Output: [2, 10, 768] (full, identical on all GPUs)
}
```

## 5. Implementation Details

**Memory Management**:
- 所有参数和缓冲区通过 `Tensor` 类管理底层内存，使用引用计数共享所有权
- 张量并行场景下，`Parameter` 存储分片后的本地权重，All-Reduce 操作在 `RowParallelLinear::forward()` 中透明完成
- `RoPE` 的 sin/cos 缓存通过 `register_buffer()` 注册，使用 `Device` 上的预分配内存，避免推理时重复计算

**Concurrency**:
- 模块本身不是线程安全的，假设在单线程上下文使用（如 Python GIL 保护下的调用）
- 张量并行的集体通信（NCCL AllReduce）在 `RowParallelLinear::forward()` 中同步执行，阻塞直到所有 rank 完成
- 状态字典的序列化/反序列化使用递归遍历，无锁设计（假设构造函数完成后参数不变）

**Performance**:
- **宏带来的零开销**: `INFINICORE_NN_*` 宏在编译期展开为直接的 `register_*` 调用，无虚函数开销
- **预计算策略**: `RoPE` 在构造时预计算 sin/cos 表，推理时仅执行查表和元素级操作，复杂度 O(seq_len * head_dim)
- **算子融合**: `forward()` 方法中的 `ops::matmul()`、`ops::add()`、`ops::gelu()` 可能被后端编译器融合为单个 kernel（如 CUDA Core）
- **张量并行优化**: `ColumnParallelLinear` 避免通信，`RowParallelLinear` 使用 All-Reduce（ring-allreduce 算法，带宽最优）

**Error Handling**:
- `static_assert` 在编译期检查模板参数类型（如 `register_module<M>` 要求 `M` 继承自 `Module`）
- 运行时错误通过抛出异常传播（如维度不匹配、设备不一致、NCCL 通信失败）
- `load_state_dict()` 支持部分加载和容错：找不到的参数会跳过，形状不匹配的参数会报错

**Dependencies**:
- **核心依赖**: `tensor.hpp`（提供 `Tensor` 类和 `ops` 命名空间）、`context/context.hpp`（提供 `Device` 枚举）
- **通信库**: `infiniccl.h`（NCCL 封装，用于张量并行的 All-Reduce）
- **日志库**: `spdlog/spdlog.h`（用于调试日志，非关键路径）
- **标准库**: `<unordered_map>`（状态字典存储）、`<memory>`（`shared_ptr`）、`<type_traits>`（`static_assert` 和类型萃取）

**Design Patterns**:
- **Composite Pattern**: `Module` 作为抽象组件，`Linear`、`Embedding` 等作为叶子节点，`submodules_` 存储子组件
- **Template Method Pattern**: `BaseLinear` 定义通用的 `compute_linear()` 算法骨架，子类重写 `forward()` 处理并行逻辑
- **Factory Pattern**: `register_module<M>()` 作为模板工厂方法，根据类型参数 `M` 创建子模块实例
- **Macro Metaprogramming**: `INFINICORE_NN_*` 宏提供 DSL 风格的声明式接口，减少样板代码
- **Strategy Pattern**: `RoPE::Algo` 枚举封装不同的旋转算法（GPT-J vs GPT-NeoX），运行时切换策略

**PyTorch Compatibility**:
- `Module` API 高度模仿 PyTorch 的 `nn.Module`：`state_dict()`、`load_state_dict()`、`register_parameter()`、`register_buffer()`
- 参数命名约定使用点分路径（如 "layer1.weight"），与 PyTorch checkpoint 格式兼容
- 层语义与 PyTorch 一致：`Linear` 对应 `nn.Linear`、`Embedding` 对应 `nn.Embedding`、`RMSNorm` 对应 `nn.RMSNorm`
- 宏系统模仿 PyTorch 的 `nn.Module` 子类模式，但使用 C++ 模板而非 Python 的动态特性
