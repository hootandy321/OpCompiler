# Llama Model Architecture Implementation

该模块实现了完整的 Llama 大语言模型架构，基于 InfiniCore 深度学习框架构建，支持 Grouped Query Attention (GQA)、Rotary Position Embeddings (RoPE) 和 KV Cache 等先进特性。该实现严格遵循 HuggingFace Transformers 的 Llama 模型结构设计，提供高效的前向推理能力和分布式训练支持。

## 1. 模块结构

- **`llama_config.hpp`**: 定义 `LlamaConfig` 配置结构体，包含模型超参数和验证逻辑
- **`llama_attention.hpp/cpp`**: 实现多头自注意力机制，支持 GQA 和 RoPE 位置编码
- **`llama_mlp.hpp/cpp`**: 实现前馈神经网络（MLP），采用 SwiGLU 激活函数
- **`llama_decoder_layer.hpp/cpp`**: 单个 Transformer 解码器层，包含注意力、MLP 和残差连接
- **`llama_model.hpp/cpp`**: 核心 Transformer 模型（不含 LM head），管理嵌入层、解码器层和 RoPE
- **`llama_for_causal_lm.hpp/cpp`**: 完整的因果语言模型，添加语言建模头输出 logits
- **`llama.hpp`**: 主头文件，统一包含所有组件

## 2. 核心类

### `LlamaConfig`
- **位置**: `llama_config.hpp`
- **主要功能**: 存储和验证 Llama 模型的所有超参数配置
- **关键成员**:
  - `vocab_size` (size_t): 词汇表大小，默认 32000
  - `hidden_size` (size_t): 隐藏层维度，默认 4096
  - `num_hidden_layers` (size_t): 解码器层数，默认 32
  - `num_attention_heads` (size_t): 注意力头数，默认 32
  - `num_key_value_heads` (size_t): KV 头数（用于 GQA），默认 32
  - `max_position_embeddings` (size_t): 最大序列长度，默认 2048
  - `rope_theta` (double): RoPE 基础频率，默认 10000.0
  - `rms_norm_eps` (double): RMSNorm 归一化 epsilon，默认 1e-6
  - `dtype` (infinicore::DataType): 模型参数数据类型，默认 F32
  - `use_cache` (bool): 是否启用 KV 缓存，默认 true
  - `attention_bias` (bool): Q/K/V 投影是否使用 bias，默认 true
- **核心方法**:
  - `kv_dim() const`: 计算 KV 投影维度（用于 GQA），公式为 `hidden_size * num_key_value_heads / num_attention_heads`
  - `validate() const`: 验证配置合法性，检查 hidden_size 能否被 num_attention_heads 整除、num_attention_heads 能否被 num_key_value_heads 整除、head_dim 是否正确

### `LlamaAttention`
- **位置**: `llama_attention.hpp/cpp`
- **主要功能**: 实现多头自注意力机制，支持 Grouped Query Attention（GQA）和 Rotary Position Embeddings（RoPE）
- **关键成员**:
  - `qkv_proj` (QKVParallelLinear): 融合的 Q/K/V 投影层（张量并行优化）
  - `o_proj` (RowParallelLinear): 输出投影层（行并行）
  - `rotary_emb_` (shared_ptr<RoPE>): 共享的旋转位置编码模块
  - `layer_idx_` (size_t): 当前注意力层索引（用于 KV Cache 访问）
  - `num_attention_heads_` (size_t): 每个张量并行秩的注意力头数（已除以 tp_size）
  - `num_key_value_heads_` (size_t): 每个 TP 秩的 KV 头数
  - `head_dim_` (size_t): 每个注意力头的维度，默认 128
  - `kv_dim_` (size_t): KV 投影总维度
  - `rank_info_` (RankInfo): 张量并行秩信息（tp_rank, tp_size, comm）
- **核心方法**:
  - `forward(...)`: 前向传播，计算注意力输出
    - **输入**: `hidden_states` [batch, seq_len, hidden_size], `position_ids` [batch, seq_len] 或 [seq_len]
    - **算法流程**:
      1. 融合 Q/K/V 投影：使用 `QKVParallelLinear::forward_split()` 分割 Q, K, V
      2. 张量重塑：reshape 为 [batch, seq_len, num_heads, head_dim]
      3. 应用 RoPE：对 Q 和 K 应用旋转位置编码（Q 需要 permute 以匹配 RoPE 接口）
      4. KV Cache 管理：支持 StaticKVCache 和 PagedKVCache（待实现）
      5. Grouped Query Attention：将多个 query heads 分组到单个 KV head
      6. 缩放点积注意力：scaling = 1/√head_dim，计算 QK^T 并应用因果 softmax
      7. 输出投影：通过 `o_proj` 投影回 hidden_size 维度
    - **输出**: [batch, seq_len, hidden_size]
    - **时间复杂度**: O(seq_len² × hidden_size)（标准注意力复杂度）
  - `set_rotary_emb(...)`: 设置共享的 RoPE 模块（由父模型注入）
- **生命周期**: 由 `LlamaDecoderLayer` 持有，构造时接收 layer_idx，需要外部调用 `set_rotary_emb()` 注入 RoPE

### `LlamaMLP`
- **位置**: `llama_mlp.hpp/cpp`
- **主要功能**: 实现 SwiGLU 激活的前馈神经网络，采用门控线性单元变体
- **关键成员**:
  - `gate_up_proj` (GateUpParallelLinear): 融合的门控和上投影层（张量并行）
  - `down_proj` (RowParallelLinear): 下投影层（行并行）
  - `hidden_size_` (size_t): 输入/输出维度，默认 4096
  - `intermediate_size_` (size_t): 中间层维度，默认 11008
  - `rank_info_` (RankInfo): 张量并行信息
- **核心方法**:
  - `forward(...)`: 前向传播，计算 MLP 输出
    - **输入**: `hidden_states` [batch, seq_len, hidden_size]
    - **算法**: SwiGLU(x) = down_proj(SiLU(gate_proj(x)) ⊙ up_proj(x))
      1. 融合投影：`gate_up_proj->forward_split()` 分离 gate 和 up
      2. SwiGLU 激活：`op::swiglu(up, gate)` 计算 `gate * sigmoid(gate) * up`
      3. 下投影：`down_proj->forward(intermediate)` 投影回 hidden_size
    - **输出**: [batch, seq_len, hidden_size]
- **设计模式**: 使用 `GateUpParallelLinear` 融合两个线性层以减少 kernel 启动开销

### `LlamaDecoderLayer`
- **位置**: `llama_decoder_layer.hpp/cpp`
- **主要功能**: 单个 Transformer 解码器层，包含预归一化、注意力、残差连接、MLP 和后归一化
- **关键成员**:
  - `input_layernorm` (RMSNorm): 预注意力层归一化（RMSNorm）
  - `post_attention_layernorm` (RMSNorm): 预 MLP 层归一化
  - `self_attn` (LlamaAttention): 自注意力模块
  - `mlp` (LlamaMLP): 前馈神经网络模块
  - `layer_idx_` (size_t): 层索引
  - `rank_info_` (RankInfo): 分布式训练信息
- **核心方法**:
  - `forward(...)`: 前向传播，处理一个解码器层
    - **输入**: `hidden_states` [batch, seq_len, hidden_size]
    - **算法流程**（Pre-LN Transformer）:
      1. 保存残差：`residual = hidden_states`
      2. 预归一化：`normed_states = input_layernorm(hidden_states)`
      3. 自注意力 + 残差：`output = hidden_states + self_attn(normed_states)`
      4. 保存残差：`residual = output`
      5. 预 MLP 归一化：`normed_states = post_attention_layernorm(output)`
      6. MLP + 残差：`output = output + mlp(normed_states)`
    - **输出**: [batch, seq_len, hidden_size]
  - `set_rotary_emb(...)`: 将 RoPE 模块注入到注意力层
- **设计模式**: 采用 Pre-LN（Layer Normalization 前）架构，提升训练稳定性

### `LlamaModel`
- **位置**: `llama_model.hpp/cpp`
- **主要功能**: 核心 Transformer 模型，管理词嵌入、多层解码器、最终归一化和 RoPE
- **关键成员**:
  - `embed_tokens` (Embedding): 词嵌入层 [vocab_size, hidden_size]
  - `layers_` (vector<LlamaDecoderLayer>): 解码器层列表（数量 = num_hidden_layers）
  - `norm` (RMSNorm): 最终层归一化
  - `rotary_emb` (RoPE): 旋转位置编码（GPT-NEOX 风格）
  - `kv_cache_` (shared_ptr<Cache>): KV 缓存（支持 StaticKVCache 和 PagedKVCache）
  - `config_` (LlamaConfig): 模型配置
  - `rank_info_` (RankInfo): 分布式训练信息
- **核心方法**:
  - `forward(...)`: 前向传播，处理输入 token 序列
    - **输入**: `input_ids` [batch, seq_len], `position_ids` [batch, seq_len] 或 [seq_len]
    - **算法流程**:
      1. 词嵌入：`hidden_states = embed_tokens(input_ids)`
      2. 层堆叠：遍历所有 `layers_`，逐层处理 hidden_states
      3. 最后 token 归一化：提取最后一个 token 的 hidden state 并应用 RMSNorm
      4. 返回归一化的表示 [batch, 1, hidden_size]
    - **输出**: [batch, 1, hidden_size]（仅最后一个 token）
    - **时间复杂度**: O(num_layers × seq_len² × hidden_size)
  - `reset_cache(...)`: 重置或初始化 KV Cache
    - **支持类型**:
      - `StaticKVCacheConfig`: 静态 KV Cache（预分配固定大小）
      - `PagedKVCacheConfig`: 分页 KV Cache（PagedAttention，待完善）
    - **初始化参数**: head_dim, num_kv_heads, num_layers, max_position_embeddings, dtype
- **生命周期**: 单例模式，由 `LlamaForCausalLM` 持有，构造时初始化所有子模块并注入 RoPE

### `LlamaForCausalLM`
- **位置**: `llama_for_causal_lm.hpp/cpp`
- **主要功能**: 完整的因果语言模型，在 `LlamaModel` 基础上添加语言建模头
- **关键成员**:
  - `model` (LlamaModel): 基础 Transformer 模型
  - `lm_head` (Linear): 语言建模头 [hidden_size, vocab_size]，输出词汇表 logits
  - `device_` (Device): 模型所在设备
- **核心方法**:
  - `forward(const Input&)`: 前向传播，生成词汇表 logits
    - **输入**: `Input` 结构体（包含 `input_ids`, `position_ids`, `cache_lengths` 等可选字段）
    - **算法流程**:
      1. 提取输入张量：`input_ids`, `position_ids` 等
      2. 基础模型前向：`hidden_states = model->forward(...)`
      3. LM head 投影：`logits = lm_head->forward(hidden_states)`
      4. 返回 `Output` 结构体 `{logits}`
    - **输出**: `Output` 结构体，包含 `logits` [batch, 1, vocab_size]
  - `reset_cache(...)`: 委托给 `model->reset_cache()`
- **继承关系**: 继承自 `InfinilmModel`（基类），实现统一的模型接口

## 3. API 接口

```cpp
namespace infinilm::models::llama {

// 1. 配置结构体
struct LlamaConfig : public InfinilmModel::Config {
    size_t vocab_size = 32000;
    size_t hidden_size = 4096;
    size_t num_hidden_layers = 32;
    size_t num_attention_heads = 32;
    size_t num_key_value_heads = 32;
    size_t head_dim = 128;
    size_t max_position_embeddings = 2048;
    double rope_theta = 10000.0;
    double rms_norm_eps = 1e-6;
    bool use_cache = true;
    bool attention_bias = true;

    size_t kv_dim() const;
    bool validate() const;
};

// 2. 完整的因果语言模型 API
class LlamaForCausalLM : public InfinilmModel {
public:
    // 构造函数
    LlamaForCausalLM(const LlamaConfig &config,
                     const infinicore::Device &device,
                     engine::distributed::RankInfo rank_info = {});

    // 前向传播：生成 logits
    Output forward(const Input &input) const;

    // Cache 管理
    void reset_cache(const cache::CacheConfig *cache_config) override;

    // 访问器
    const LlamaConfig &config() const;
    LlamaModel &model();
};

// 3. 输入输出结构（定义在 InfinilmModel 基类）
struct Input {
    std::optional<infinicore::Tensor> input_ids;        // [batch, seq_len]
    std::optional<infinicore::Tensor> position_ids;     // [batch, seq_len] 或 [seq_len]
    std::optional<infinicore::Tensor> cache_lengths;    // [n_req]
    std::optional<infinicore::Tensor> input_lengths;    // [n_req]
    std::optional<infinicore::Tensor> input_offsets;    // [n_req]
    std::optional<infinicore::Tensor> block_tables;     // PagedAttention
    std::optional<infinicore::Tensor> slot_mapping;     // PagedAttention
};

struct Output {
    infinicore::Tensor logits;  // [batch, 1, vocab_size]
};
}
```

## 4. 使用示例

```cpp
#include "llama_for_causal_lm.hpp"
#include "infinicore/context/context.hpp"

using namespace infinilm::models::llama;

int main() {
    // 1. 初始化设备（GPU/CPU）
    auto device = infinicore::Device::create("cuda:0");

    // 2. 配置模型参数（兼容 Llama-7B）
    LlamaConfig config;
    config.vocab_size = 32000;
    config.hidden_size = 4096;
    config.num_hidden_layers = 32;
    config.num_attention_heads = 32;
    config.num_key_value_heads = 32;
    config.max_position_embeddings = 2048;
    config.dtype = infinicore::DataType::F16;  // 半精度浮点

    // 3. 创建模型实例
    engine::distributed::RankInfo rank_info{0, 1, nullptr};  // 单 GPU
    auto model = std::make_shared<LlamaForCausalLM>(config, device, rank_info);

    // 4. 初始化 KV Cache（用于增量解码）
    cache::StaticKVCacheConfig cache_config{
        .max_batch_size = 1,
        .max_seq_len = 2048
    };
    model->reset_cache(&cache_config);

    // 5. 准备输入张量
    int64_t input_ids_data[] = {1, 2, 3, 4, 5};  // 示例 token IDs
    auto input_ids = infinicore::Tensor::from_vector(
        {1, 5},  // [batch=1, seq_len=5]
        input_ids_data,
        device
    );

    int64_t position_ids_data[] = {0, 1, 2, 3, 4};  // 位置索引
    auto position_ids = infinicore::Tensor::from_vector(
        {5},  // [seq_len]
        position_ids_data,
        device
    );

    // 6. 封装输入并执行前向传播
    InfinilmModel::Input input;
    input.input_ids = input_ids;
    input.position_ids = position_ids;

    auto output = model->forward(input);

    // 7. 获取 logits（预测下一个 token）
    auto logits = output.logits;  // [1, 1, 32000]

    // 8. 应用 softmax 并采样（需自行实现）
    // auto probs = softmax(logits);
    // auto next_token = sample(probs);

    return 0;
}
```

## 5. 实现细节

### 内存管理
- **张量并行策略**: 使用 `QKVParallelLinear` 和 `GateUpParallelLinear` 融合多个线性层，减少 kernel 启动开销和内存访问次数
- **RoPE 共享**: `rotary_emb` 模块在 `LlamaModel` 层创建，通过 `set_rotary_emb()` 注入到所有注意力层，避免重复计算旋转频率
- **KV Cache 复用**: 支持 `StaticKVCache`（连续内存）和 `PagedKVCache`（分页内存），在增量解码时避免重复计算 KV
- **数据类型灵活性**: 支持 FP32、FP16、BF16 等数据类型，通过 `config.dtype` 统一配置

### 并发与分布式
- **张量并行 (Tensor Parallelism, TP)**:
  - 使用 `RowParallelLinear` 和 `ColumnParallelLinear` 分割线性层权重
  - `num_attention_heads` 和 `num_key_value_heads` 在构造时除以 `tp_size`
  - 通过 `rank_info.comm`（MPI/NCCL communicator）进行跨秩通信
  - 约束：`num_key_value_heads % tp_size == 0`
- **层并行 (Pipeline Parallelism)**: 当前未实现，但架构设计支持未来扩展
- **分布式 KV Cache**: `kv_cache_` 通过 `rank_info` 在不同 TP 秩间分割 KV 数据

### 性能优化
- **融合 Kernel**: `QKVParallelLinear::forward_split()` 和 `GateUpParallelLinear::forward_split()` 在单次 kernel 调用中完成多个矩阵乘法
- **因果 Softmask 优化**: 使用 `op::causal_softmax_()` 原地应用因果掩码和 softmax，减少中间张量分配
- **Grouped Query Attention (GQA)**: 通过减少 KV head 数量（`num_key_value_heads < num_attention_heads`）降低内存占用和计算量
  - 内存节省比例：`num_key_value_heads / num_attention_heads`
  - 例如：Llama-2-70B 使用 `num_key_value_heads=8`，`num_attention_heads=64`，节省 87.5% KV 内存
- **RoPE 位置编码**: 相对位置编码，无需训练绝对位置嵌入，支持外推到更长序列（通过调整 `rope_theta`）

### 错误处理
- **配置验证**: `LlamaConfig::validate()` 检查参数合法性（如整除关系），防止运行时错误
- **RoPE 未配置检查**: `LlamaAttention::forward()` 抛出异常如果 `rotary_emb_` 为空
- **Cache 类型动态转换**: 使用 `dynamic_pointer_cast` 安全地检测 `StaticKVCache` 或 `PagedKVCache`
- **PagedAttention 待实现**: 当前对 PagedKVCache 抛出 `"not implemented"` 异常

### 依赖关系
- **InfiniCore 框架**:
  - `infinicore/nn/module.hpp`: 模块基类和注册宏（`INFINICORE_NN_MODULE`）
  - `infinicore/nn/linear.hpp`: 线性层（`Linear`, `RowParallelLinear`, `ColumnParallelLinear`）
  - `infinicore/nn/rope.hpp`: 旋转位置编码
  - `infinicore/nn/rmsnorm.hpp`: RMS 归一化
  - `infinicore/ops.hpp`: 算子库（matmul, softmax, swiglu, add）
  - `infinicore/tensor.hpp`: 张量操作（view, permute, narrow, contiguous）
- **InfiniLM 内部模块**:
  - `layers/fused_linear.hpp`: 自定义融合线性层（`QKVParallelLinear`, `GateUpParallelLinear`）
  - `cache/kv_cache.hpp`: KV Cache 抽象（`Cache`, `StaticKVCache`, `PagedKVCache`）
  - `engine/distributed/distributed.hpp`: 分布式训练基础设施（`RankInfo`）
  - `infinilm_model.hpp`: 基类 `InfinilmModel`，定义统一接口

### 设计模式
- **模块注册模式**: 使用 `INFINICORE_NN_MODULE` 宏自动注册子模块到父模块的 `modules_` 映射
- **依赖注入**: RoPE 模块通过 `set_rotary_emb()` 从外部注入，而非在构造函数中创建
- **工厂模式**: `reset_cache()` 根据 `CacheConfig` 类型动态创建 `StaticKVCache` 或 `PagedKVCache`
- **策略模式**: `Cache` 抽象基类定义接口，具体缓存类型实现不同策略
- **Pre-LN Transformer**: 归一化层放在残差连接之前，提升训练稳定性和推理性能

### 与 HuggingFace Transformers 对齐
- **配置兼容**: `LlamaConfig` 字段命名和默认值与 HuggingFace `LlamaConfig` 完全一致
- **模型结构**:
  - `LlamaModel` 对应 `transformers.LlamaModel`
  - `LlamaForCausalLM` 对应 `transformers.LlamaForCausalLM`
- **数学等价性**:
  - 注意力计算：`softmax(QK^T / √d_k)V`（标准缩放点积注意力）
  - MLP 激活：`SwiGLU(x) = down(SiLU(gate(x)) ⊙ up(x))`
  - 归一化：`RMSNorm(x) = x / √(mean(x²) + ε)`
- **权重加载**: 当前未实现 `from_pretrained()`，但架构支持直接加载 HuggingFace 格式权重（需自行实现权重转换逻辑）
