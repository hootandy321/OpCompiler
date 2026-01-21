# LLM 模型接口层核心实现文档

本模块为 InfiniLM 推理框架的模型抽象层，提供了三种大语言模型（DeepSeek-V3、Jiuge、Jiuge-AWQ）的 C 语言接口定义。这些接口封装了模型权重加载、推理执行、KV Cache 管理、采样等核心功能，支持多协处理器分布式推理和批处理。

## 1. 模块结构

- **`deepseek.h`**: DeepSeek-V3 MoE（混合专家）大语言模型接口，支持专家路由、稀疏激活、量化权重等特性
- **`jiuge.h`**: Jiuge 标准版 LLM 接口，基于 Transformer 架构，支持多头注意力和分组查询注意力（GQA）
- **`jiuge_awq.h`**: Jiuge 量化版接口，使用 AWQ（Activation-aware Weight Quantization）量化技术压缩模型权重

## 2. 核心数据结构

### `DeepSeekV3Meta`
- **位置**: `deepseek.h`
- **功能**: DeepSeek-V3 模型的元数据配置，包含模型维度、数据类型、MoE 参数等
- **关键成员**:
  - `dt_logits`: logits 输出的数据类型
  - `dt_norm`, `dt_quant_weight`, `dt_quant_scale`, `dt_quant_zero`: 归一化和量化相关数据类型
  - `n_sparse_layer`, `n_dense_layer`: 稀疏层和密集层的数量（MoE 架构特有）
  - `d`: 隐藏层维度
  - `nh`: 注意力头数
  - `nkvh`: KV 注意力头数（用于 GQA，nh >= nkvh）
  - `d_rope`: 旋转位置编码维度
  - `d_nope`: 非旋转位置编码维度
  - `r_q`, `r_kv`: Q/KV 的压缩比率（用于 Yarn 扩展）
  - `d_qk`, `d_v`: Query-Key 和 Value 的维度
  - `routed_scale`: 路由缩放因子（MoE 负载均衡参数）
  - `nexperts`: 总专家数量
  - `kexperts`: 每次激活的专家数量（top-k routing）
  - `di`, `di_moe`: FFN 中间维度（普通层和 MoE 层）
  - `epsilon`: LayerNorm/RMSNorm 的 epsilon 参数
  - `rope_theta`: RoPE 的 theta 基数
  - `end_token`: 结束 token ID

### `DeepSeekV3Weights` (不透明类型)
- **位置**: `deepseek.h`
- **功能**: DeepSeek-V3 模型权重的容器，存储所有权重张量的 GPU 协处理器内存指针
- **生命周期**:
  1. 调用 `createDeepSeekV3Weights()` 分配内存并初始化
  2. 通过 `DeepSeekV3WeightLoader` 函数指针逐个加载权重
  3. 传递给 `createDeepSeekV3Model()` 构建推理模型
  4. 模型销毁时自动释放

### `DeepSeekV3WeightLoader`
- **位置**: `deepseek.h`
- **功能**: 函数指针结构体，定义了加载 DeepSeek-V3 各层权重的接口
- **核心方法**:
  - `load_input_embd(weights, cpu_ptr)`: 加载输入 embedding 表 `[dvoc, d]`
  - `load_output_norm(weights, cpu_ptr)`: 加载输出层归一化权重 `[d]`
  - `load_output_embd(weights, cpu_ptr)`: 加载输出 embedding 表（用于语言模型头）`[dvoc, d]`

  **Attention 权重**（每层）:
  - `load_attn_norm(weights, cpu_ptr, layer_id)`: Attention 前的 RMSNorm `[d]`
  - `load_attn_q_a_proj(weights, weight_ptr, scale_ptr, zero_ptr, layer_id)`: Q 的 A 量化投影（低秩分解第一步）`[d, d_qk]`
  - `load_attn_q_a_layernorm(weights, cpu_ptr, layer_id)`: Q A 投影后的 LayerNorm `[d_qk]`
  - `load_attn_q_b_proj(weights, weight_ptr, scale_ptr, zero_ptr, layer_id)`: Q 的 B 量化投影（低秩分解第二步）`[d_qk, nh * dh]`
  - `load_attn_kv_a_proj_with_mqa(weights, weight_ptr, scale_ptr, zero_ptr, layer_id)`: KV 的 A 量化投影（带 MQA 支持）`[d, 2 * d_v]`
  - `load_attn_kv_a_layernorm(weights, cpu_ptr, layer_id)`: KV A 投影后的 LayerNorm `[2 * d_v]`
  - `load_attn_kv_b_proj(weights, weight_ptr, scale_ptr, zero_ptr, layer_id)`: KV 的 B 量化投影 `[2 * d_v, nkvh * dh]`
  - `load_attn_o_proj(weights, weight_ptr, scale_ptr, zero_ptr, layer_id)`: Output 投影量化权重 `[nkvh * dh, d]`

  **MLP 密集层权重**（非 MoE 层）:
  - `load_mlp_norm(weights, cpu_ptr, layer_id)`: MLP 前的 RMSNorm `[d]`
  - `load_mlp_dense(weights, gate_w/s/z, up_w/s/z, down_w/s/z, layer_id)`: 加载标准 FFN 的 gate/up/down 三层量化权重

  **MLP 稀疏层权重**（MoE 层）:
  - `load_mlp_gate_weight(weights, cpu_ptr, layer_id)`: 门控网络的权重 `[d, nexperts]`
  - `load_mlp_gate_bias(weights, cpu_ptr, layer_id)`: 门控网络的 bias `[nexperts]`
  - `load_mlp_shared_experts(weights, gate_w/s/z, up_w/s/z, down_w/s/z, layer_id)`: 加载共享专家的量化权重
  - `load_mlp_experts(weights, gate_w/s/z, up_w/s/z, down_w/s/z, layer_id, expert_id)`: 逐个加载路由专家的量化权重

### `JiugeMeta`
- **位置**: `jiuge.h`
- **功能**: Jiuge 标准模型的元数据配置
- **关键成员**:
  - `nlayer`: Transformer 层数
  - `d`: 隐藏层维度
  - `nh`: Query 注意力头数
  - `nkvh`: KV 注意力头数（GQA 架构，nkvh <= nh）
  - `dh`: 每个注意力头的维度
  - `di`: FFN 中间维度
  - `dctx`: 最大上下文长度
  - `dvoc`: 词汇表大小
  - `epsilon`: RMSNorm epsilon
  - `theta`: RoPE theta 基数
  - `end_token`: 结束 token ID

### `JiugeWeights`
- **位置**: `jiuge.h`
- **功能**: Jiuge 模型的权重结构体，直接包含所有权重指针（用于直接传递模式）
- **关键成员**:
  - `nlayer`: 层数
  - `dt_norm`, `dt_mat`: 归一化和矩阵权重数据类型
  - `transpose_linear_weights`: 权重布局标志（0=W, 非0=W^T，匹配 PyTorch 默认格式）
  - `input_embd`: `[dvoc, d]` 输入 embedding 表
  - `output_norm`: `[d]` 输出层 RMSNorm 权重
  - `output_embd`: `[dvoc, d]` 输出 embedding 表
  - `attn_norm`: `nlayer * [d]` 每层注意力前的 RMSNorm
  - `attn_qkv`: `nlayer * [ndev, (nh + 2 * nkvh) / ndev * dh, d]` 合并的 QKV 投影（支持多卡分布）
  - `attn_qkv_b`: `nlayer * [ndev, (nh + 2 * nkvh) / ndev * dh]` QKV bias
  - `attn_q_norm`, `attn_k_norm`: `nlayer * [dh]` QPN（Query-Key normalization）权重
  - `attn_o`: `nlayer * [ndev, d, nkvh / ndev * dh]` Output 投影（多卡分布）
  - `ffn_norm`: `nlayer * [d]` FFN 前的 RMSNorm
  - `ffn_gate_up`: `nlayer * [ndev, 2 * di / ndev, d]` 合并的 gate+up 投影（SwiGLU 激活）
  - `ffn_down`: `nlayer * [ndev, d, di / ndev]` down 投影

### `JiugeAWQMeta`
- **位置**: `jiuge_awq.h`
- **功能**: Jiuge AWQ 量化版本的元数据配置
- **关键成员**:
  - 继承 `JiugeMeta` 的所有字段（nlayer, d, nh, nkvh, dh, di, dctx, dvoc, epsilon, theta, end_token）
  - `dt_linear_w`: 线性层量化权重数据类型
  - `dt_norm_w`: 归一化层权重数据类型
  - `nbit`: 量化比特数（通常为 4-bit）
  - `quant_group_size`: AWQ 量化分组大小（通常为 128）
  - `has_qkv_bias`: 是否包含 QKV bias（char 布尔值）

### `JiugeModel`, `JiugeAWQModel`, `DeepSeekV3Model` (不透明类型)
- **功能**: 模型推理的执行引擎，封装了计算图和协处理器内核
- **生命周期**:
  1. 调用 `createXxxModel()` 构建模型（初始化计算内核）
  2. 多次调用 `inferBatchXxx()` 或 `forwardBatchXxx()` 执行推理
  3. 调用 `destroyXxxModel()` 释放资源

### `KVCache` (不透明类型，定义在 `cache.h`)
- **功能**: 存储 Transformer 各层的 Key-Value 缓存，支持自回归生成
- **管理接口**:
  - `createKVCache(nlayers, max_len, nkvh, dk, dv, dtype, device, dev_ids, ndev)`: 创建缓存
  - `duplicateKVCache(cache, seq_len)`: 复制缓存（用于批量请求的上下文初始化）
  - `dropKVCache(cache)`: 释放缓存

## 3. API 接口

### DeepSeek-V3 模型接口

```c
// 权重容器创建
DeepSeekV3Weights *
createDeepSeekV3Weights(const DeepSeekV3Meta *meta,
                        infiniDevice_t device,
                        int ndev,
                        const int *dev_ids);
// 根据 meta 配置分配权重内存空间，支持多卡分布

// 权重加载器创建
DeepSeekV3WeightLoader *
createDeepSeekV3WeightLoader();
// 返回函数指针表，用于逐个加载权重到 GPU

// 模型构建
struct DeepSeekV3Model *
createDeepSeekV3Model(const DeepSeekV3Meta *meta,
                      const DeepSeekV3Weights *weights);
// 基于元数据和权重构建推理引擎

// 模型销毁
void destroyDeepSeekV3Model(struct DeepSeekV3Model *model);

// KV Cache 管理
struct DeepSeekV3Cache *
createDeepSeekV3Cache(const struct DeepSeekV3Model *model);

void dropDeepSeekV3Cache(const struct DeepSeekV3Model *model,
                         struct DeepSeekV3Cache *cache);

// 批量推理（带采样）
void inferBatchDeepSeekV3(
    struct DeepSeekV3Model *model,
    const uint32_t *tokens,      // 扁平化的 token 序列 [所有请求的所有 token]
    uint32_t ntok,                // 总 token 数
    const uint32_t *req_lens,     // 每个请求的 token 长度 [nreq]
    uint32_t nreq,                // 请求数量
    const uint32_t *req_pos,      // 每个请求的起始位置 [nreq]
    struct DeepSeekV3Cache **caches, // 每个请求的 KVCache 指针数组 [nreq]
    const float *temperature,     // 采样温度数组 [nreq]，0.0=贪心
    const uint32_t *topk,         // top-k 采样参数 [nreq]，1=贪心
    const float *topp,            // top-p 采样参数 [nreq]
    uint32_t *output              // 输出数组 [nreq]
);
// 执行前向推理 + 核采样，生成下一个 token

// 批量前向传播（无采样）
void forwardBatchDeepSeekV3(
    struct DeepSeekV3Model *model,
    const uint32_t *tokens,
    uint32_t ntok,
    const uint32_t *req_lens,
    uint32_t nreq,
    const uint32_t *req_pos,
    struct DeepSeekV3Cache **caches,
    void *logits                 // 输出 logits，形状为 [nreq, dvoc]
);
// 仅执行前向推理，输出 logits 分布（用于自定义采样或 log-liklihood 计算）
```

### Jiuge 标准模型接口

```c
// 模型构建（直接权重模式）
struct JiugeModel *
createJiugeModel(const JiugeMeta *meta,
                 const JiugeWeights *weights,
                 infiniDevice_t device,
                 int ndev,
                 const int *dev_ids);
// 直接使用 JiugeWeights 结构体构建模型，权重必须在 CPU 内存

// 模型销毁
void destroyJiugeModel(struct JiugeModel *model);

// 批量推理（带采样）
void inferBatchJiuge(
    struct JiugeModel *model,
    const uint32_t *tokens,
    uint32_t ntok,
    const uint32_t *req_lens,
    uint32_t nreq,
    const uint32_t *req_pos,
    struct KVCache **kv_caches,
    const float *temperature,
    const uint32_t *topk,
    const float *topp,
    uint32_t *output
);

// 批量前向传播（无采样）
void forwardBatchJiuge(
    struct JiugeModel *model,
    const uint32_t *tokens,
    uint32_t ntok,
    const uint32_t *req_lens,
    uint32_t nreq,
    const uint32_t *req_pos,
    struct KVCache **kv_caches,
    void *logits
);
```

### Jiuge AWQ 量化模型接口

```c
// 权重容器创建（使用通用 ModelWeights 类型）
struct ModelWeights *
createJiugeAWQWeights(const JiugeAWQMeta *meta,
                      infiniDevice_t device,
                      int ndev,
                      const int *dev_ids);

// 权重加载（通用接口）
void loadModelWeight(struct ModelWeights *weights, const char *name, void *data);
void loadModelWeightDistributed(struct ModelWeights *weights, const char *name,
                                void *data, int *ranks, int nrank);

// 模型构建
struct JiugeAWQModel *
createJiugeAWQModel(const JiugeAWQMeta *meta,
                    const ModelWeights *weights);

// 模型销毁
void destroyJiugeAWQModel(struct JiugeAWQModel *model);

// 批量推理（带采样）
void inferBatchJiugeAWQ(
    struct JiugeAWQModel *model,
    const uint32_t *tokens,
    uint32_t ntok,
    const uint32_t *req_lens,
    uint32_t nreq,
    const uint32_t *req_pos,
    struct KVCache **kv_caches,
    const float *temperature,
    const uint32_t *topk,
    const float *topp,
    uint32_t *output
);

// 批量前向传播（无采样）
void forwardBatchJiugeAWQ(
    struct JiugeAWQModel *model,
    const uint32_t *tokens,
    uint32_t ntok,
    const uint32_t *req_lens,
    uint32_t nreq,
    const uint32_t *req_pos,
    struct KVCache **kv_caches,
    void *logits
);
```

## 4. 使用示例

### 示例 1: DeepSeek-V3 推理流程

```c
#include <infinicore_infer/models/deepseek.h>

// 1. 配置模型元数据
DeepSeekV3Meta meta = {
    .dt_logits = INFINI_FLOAT32,
    .dt_norm = INFINI_FLOAT32,
    .dt_quant_weight = INFINI_UINT4,
    .dt_quant_scale = INFINI_FLOAT32,
    .dt_quant_zero = INFINI_FLOAT32,
    .n_sparse_layer = 60,
    .n_dense_layer = 1,
    .d = 7168,
    .nh = 64,
    .nkvh = 8,  // GQA: 64个Q头对应8个KV头
    .d_rope = 64,
    .d_nope = 7104,
    .r_q = 1,
    .r_kv = 1,
    .d_qk = 512,
    .d_v = 128,
    .routed_scale = 2.5,
    .nexperts = 256,
    .kexperts = 8,
    .di = 18432,
    .di_moe = 2048,
    .dctx = 16384,
    .dvoc = 102400,
    .epsilon = 1e-5f,
    .rope_theta = 10000.0f,
    .end_token = 100001
};

// 2. 创建权重容器（分配 GPU 内存）
int dev_ids[] = {0, 1, 2, 3};
DeepSeekV3Weights *weights = createDeepSeekV3Weights(
    &meta, INFINI_DEVICE_CUDA, 4, dev_ids);

// 3. 获取权重加载器
DeepSeekV3WeightLoader *loader = createDeepSeekV3WeightLoader();

// 4. 加载全局权重
void *cpu_input_embd = load_from_disk("input_embd.bin");
loader->load_input_embd(weights, cpu_input_embd);

// 5. 加载 Attention 权重（逐层）
for (size_t layer = 0; layer < meta.n_sparse_layer + meta.n_dense_layer; layer++) {
    void *q_a_w, *q_a_s, *q_a_z;
    load_quantized_weight(layer, "q_a", &q_a_w, &q_a_s, &q_a_z);
    loader->load_attn_q_a_proj(weights, q_a_w, q_a_s, q_a_z, layer);

    // ... 加载其他权重
}

// 6. 加载 MoE 专家权重
for (size_t layer = 0; layer < meta.n_sparse_layer; layer++) {
    for (size_t expert = 0; expert < meta.nexperts; expert++) {
        void *gate_w, *gate_s, *gate_z;
        void *up_w, *up_s, *up_z;
        void *down_w, *down_s, *down_z;
        load_expert_weight(layer, expert, &gate_w, &gate_s, &gate_z,
                           &up_w, &up_s, &up_z, &down_w, &down_s, &down_z);
        loader->load_mlp_experts(weights, gate_w, gate_s, gate_z,
                                 up_w, up_s, up_z, down_w, down_s, down_z,
                                 layer, expert);
    }
}

// 7. 创建推理模型
DeepSeekV3Model *model = createDeepSeekV3Model(&meta, weights);

// 8. 创建 KV Cache
DeepSeekV3Cache *cache = createDeepSeekV3Cache(model);

// 9. 准备输入
uint32_t tokens[] = {1, 2, 3, 4, 5};  // 输入 token 序列
uint32_t req_lens[] = {5};             // 单个请求，长度为 5
uint32_t req_pos[] = {0};              // 起始位置为 0
float temperature = 0.7f;
uint32_t topk = 40;
float topp = 0.9f;
uint32_t output;

// 10. 执行推理
inferBatchDeepSeekV3(model, tokens, 5, req_lens, 1, req_pos,
                     &cache, &temperature, &topk, &topp, &output);

// 11. 清理资源
dropDeepSeekV3Cache(model, cache);
destroyDeepSeekV3Model(model);
```

### 示例 2: Jiuge 标准模型推理流程

```c
#include <infinicore_infer/models/jiuge.h>

// 1. 配置元数据
JiugeMeta meta = {
    .nlayer = 32,
    .d = 4096,
    .nh = 32,
    .nkvh = 8,  // GQA: 32个Q头对应8个KV头
    .dh = 128,
    .di = 11008,
    .dctx = 8192,
    .dvoc = 128256,
    .epsilon = 1e-5f,
    .theta = 10000.0f,
    .end_token = 2
};

// 2. 准备权重（假设权重已加载到 CPU）
JiugeWeights weights = {
    .nlayer = meta.nlayer,
    .dt_norm = INFINI_FLOAT32,
    .dt_mat = INFINI_FLOAT16,
    .transpose_linear_weights = 1,  // PyTorch 格式（W^T）
    .input_embd = cpu_input_embd_ptr,
    .output_norm = cpu_output_norm_ptr,
    .output_embd = cpu_output_embd_ptr,
    .attn_norm = cpu_attn_norm_ptrs,
    .attn_qkv = cpu_attn_qkv_ptrs,
    // ... 其他权重
};

// 3. 创建模型
int dev_ids[] = {0};
JiugeModel *model = createJiugeModel(&meta, &weights,
                                     INFINI_DEVICE_CUDA, 1, dev_ids);

// 4. 创建 KV Cache
KVCache *cache = createKVCache(meta.nlayer, meta.dctx, meta.nkvh,
                               meta.dh, meta.dh, INFINI_FLOAT16,
                               INFINI_DEVICE_CUDA, dev_ids, 1);

// 5. 批量推理示例（2个请求）
uint32_t tokens[] = {1, 2, 3, 4, 5, 6, 7, 8};  // 请求1: [1,2,3,4], 请求2: [5,6,7,8]
uint32_t req_lens[] = {4, 4};
uint32_t req_pos[] = {0, 0};
float temperatures[] = {0.8f, 0.6f};
uint32_t topks[] = {40, 40};
float topps[] = {0.9f, 0.95f};
uint32_t outputs[2];

inferBatchJiuge(model, tokens, 8, req_lens, 2, req_pos,
                &cache, temperatures, topks, topps, outputs);

// 6. 清理资源
dropKVCache(cache);
destroyJiugeModel(model);
```

### 示例 3: Jiuge AWQ 量化模型推理流程

```c
#include <infinicore_infer/models/jiuge_awq.h>

// 1. 配置量化元数据
JiugeAWQMeta meta = {
    .dt_logits = INFINI_FLOAT32,
    .dt_linear_w = INFINI_UINT4,  // 4-bit 权重量化
    .dt_norm_w = INFINI_FLOAT32,
    .nlayer = 32,
    .d = 4096,
    .nh = 32,
    .nkvh = 8,
    .dh = 128,
    .di = 11008,
    .dctx = 8192,
    .dvoc = 128256,
    .epsilon = 1e-5f,
    .theta = 10000.0f,
    .end_token = 2,
    .nbit = 4,
    .quant_group_size = 128,
    .has_qkv_bias = 1
};

// 2. 创建权重容器
int dev_ids[] = {0};
ModelWeights *weights = createJiugeAWQWeights(&meta, INFINI_DEVICE_CUDA, 1, dev_ids);

// 3. 加载量化权重（使用通用接口）
void *qkv_w, *qkv_s, *qkv_z;
load_quantized_tensor("qkv_weight.bin", "qkv_scale.bin", "qkv_zero.bin",
                      &qkv_w, &qkv_s, &qkv_z);
loadModelWeight(weights, "attn_qkv", qkv_w);
loadModelWeight(weights, "attn_qkv_scale", qkv_s);
loadModelWeight(weights, "attn_qkv_zero", qkv_z);

// ... 加载其他权重

// 4. 创建模型
JiugeAWQModel *model = createJiugeAWQModel(&meta, weights);

// 5. 推理流程与标准模型相同
KVCache *cache = createKVCache(...);
// ... 执行推理
dropKVCache(cache);
destroyJiugeAWQModel(model);
```

## 5. 实现细节

### 内存管理策略
- **权重存储**：所有权重直接存储在协处理器显存（GPU/HBM/NPU）中，通过 `infiniDevice_t` 指定设备类型（CUDA、BANG、KUNLUN 等）
- **多卡分布**：支持张量并行（Tensor Parallelism），通过 `ndev` 和 `dev_ids` 参数指定多卡分布策略
  - Attention QKV、Output、FFN Gate/Up/Down 层沿 `ndev` 维度切分
  - KV Cache 自动分布到多个设备以平衡内存和带宽
- **不透明类型**：`DeepSeekV3Weights`、`DeepSeekV3Model`、`KVCache` 等为不透明指针，内部实现隐藏，通过函数接口访问

### 并发与线程安全
- **无锁设计**：推理 API (`inferBatchXxx`, `forwardBatchXxx`) 本身是无状态的，可以多线程并发调用
- **请求隔离**：每个请求有独立的 KVCache，批处理时多个请求共享计算资源但互不干扰
- **协处理器同步**：内部使用 InfiniRT 运行时管理协处理器任务的同步和流调度

### 性能优化技术
- **量化压缩**：
  - DeepSeek-V3: 使用 4-bit 权重量化（`INFINI_UINT4`），配合 per-channel scale 和 zero_point
  - Jiuge-AWQ: AWQ 量化算法，激活感知的权重量化，`quant_group_size=128` 的分组量化
- **低秩分解**（DeepSeek-V3 特有）：
  - Q 投影分解为 A（`[d, d_qk]`）→ LayerNorm → B（`[d_qk, nh*dh]`）
  - KV 投影分解为 A（`[d, 2*d_v]`）→ LayerNorm → B（`[2*d_v, nkvh*dh]`）
  - 降低参数量，减少计算开销
- **分组查询注意力（GQA）**：
  - Query 头数 `nh` 大于 KV 头数 `nkvh`（例如 32:8）
  - 减少 KV Cache 内存占用，提升推理吞吐
- **混合专家（MoE）**（DeepSeek-V3 特有）：
  - 稀疏层（`n_sparse_layer`）使用 256 个专家，每次激活 top-8（`kexperts=8`）
  - 密集层（`n_dense_layer`）使用标准 FFN
  - 路由网络动态选择专家，`routed_scale` 参数用于负载均衡
- **内核融合**：推理接口内部自动融合以下操作以减少 kernel launch 开销：
  - Gemm + Bias + Activation
  - LayerNorm/RMSNorm
  - RoPE 位置编码
  - 量化/反量化

### 误差处理机制
- **参数校验**：创建模型时校验元数据参数的合法性（如维度匹配、层数非零）
- **显存不足**：权重分配或 KVCache 创建失败时返回 NULL，调用者需检查返回值
- **越界保护**：推理 API 不显式检查 token ID 范围，需调用者保证 tokens 在 `[0, dvoc)` 内
- **多卡一致性**：多设备环境下，InfiniCCL 和 InfiniRT 自动处理通信失败和设备不可用情况

### 依赖关系
- **InfiniRT**: 硬件抽象层，提供统一的设备管理和内存分配接口
- **InfiniOP**: 算子库，提供 MatMul、LayerNorm、RoPE、量化等计算内核
- **InfiniCCL**: 集合通信库，支持多卡 AllReduce、AllGather 等操作
- **外部依赖**: 无，所有接口为纯 C 函数，无第三方库依赖

### 设计模式
- **工厂模式**：`createXxxModel()` 函数作为工厂方法，根据元数据构造模型对象
- **策略模式**：`DeepSeekV3WeightLoader` 函数指针表定义权重加载策略，支持不同加载流程
- **不透明指针模式**：隐藏内部实现细节，暴露清晰的 C API 边界
- **资源获取即初始化（RAII）的 C 变体**：创建函数分配资源，销毁函数释放资源，调用者需手动管理生命周期

### 数据类型约定
- **精度类型**：使用 `infiniDtype_t` 枚举表示（`INFINI_FLOAT32`, `INFINI_FLOAT16`, `INFINI_UINT4` 等）
- **索引类型**：所有长度、维度、ID 使用 `size_t` 无符号 64 位整数
- **Token 类型**：使用 `uint32_t` 表示 token ID，支持最大 4G 词汇表
- **布尔类型**：使用 `char` 表示布尔值（0=假，非0=真），如 `has_qkv_bias`

### 多设备支持
- **设备类型**：通过 `infiniDevice_t` 枚举支持多种硬件后端
  - `INFINI_DEVICE_CUDA`: NVIDIA GPU
  - `INFINI_DEVICE_BANG`: 寒武纪 MLU
  - `INFINI_DEVICE_KUNLUN`: 昆仑芯
  - `INFINI_DEVICE_ASCEND`: 华为昇腾 NPU
  - `INFINI_DEVICE_METAX`: 算能
- **设备编号**：`dev_ids` 数组指定使用的物理设备 ID，支持子集选择（如 4 卡服务器使用 2 卡）
- **内存亲和性**：权重和 KVCache 自动分配到指定设备，推理时无需显式管理设备上下文

### 注意机制细节
- **旋转位置编码（RoPE）**：支持 Yarn 扩展，通过 `d_rope`、`d_nope`、`r_q`、`r_kv` 参数控制位置编码的维度和缩放
- **QPN（Query-Key Normalization）**：Jiuge 模型支持对 Q 和 K 进行额外的 LayerNorm（`attn_q_norm`、`attn_k_norm`），提升训练稳定性
- **ALiBi 替代**：部分模型变体可能使用 ALiBi（Attention with Linear Biases）替代 RoPE，通过 `theta=0` 标识

### 采样策略
- **贪心采样**：`temperature=0.0` 且 `topk=1` 时选择概率最高的 token
- **核采样（Top-p / Nucleus Sampling）**：根据 `topp` 阈值截断累积概率分布
- **Top-k 采样**：根据 `topk` 限制候选 token 数量
- **温度缩放**：温度参数 `temperature` 控制分布平滑度（越高越随机）
- **自定义采样**：使用 `forwardBatchXxx()` 获取原始 logits，自行实现采样策略（如 beam search、重复惩罚等）
