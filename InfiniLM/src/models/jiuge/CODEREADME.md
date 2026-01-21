# `Jiuge` 大语言模型推理实现核心文档

Jiuge 是一个高性能的大语言模型（LLM）推理引擎实现，支持多设备并行推理、KV Cache 优化、RoPE 位置编码、GQA（Grouped Query Attention）等现代 Transformer 架构特性。该模块实现了完整的 Transformer 推理流程，包括注意力机制、前馈神经网络、采样和分布式通信。

## 1. 模块结构

- **`jiuge_weight.hpp`**: 权重加载与张量转换工具，提供从模型权重到推理张量的映射函数，包括 RoPE 位置编码表（sin/cos）的动态生成。
- **`jiuge_impl.hpp`**: 核心数据结构定义，包括设备资源管理、推理状态同步、推理请求封装和模型主体结构。
- **`jiuge.cpp`**: 核心推理引擎实现，包含设备资源初始化、批量推理计算、多线程调度和模型生命周期管理。

## 2. 核心数据结构

### `JiugeDeviceResource`
- **位置**: `jiuge_impl.hpp:15-31`
- **主要功能**: 封装单个推理设备的所有资源，包括权重张量、计算流、通信器和内存池
- **关键成员**:
  - `device`: 设备类型（CPU/CUDA/Kunlun 等）
  - `device_id`: 设备物理 ID
  - `handle`: InfiniOP 算子库句柄
  - `w_in_embd`, `w_out_embd`: 输入/输出嵌入层权重（shape: `[dvoc, d]` 或 `[d, dvoc]`，支持转置）
  - `w_attn_qkv`, `b_attn_qkv`: 注意力 QKV 投影权重及偏置（按层分片，支持张量并行）
  - `w_attn_q_norm`, `w_attn_k_norm`: Q/K 的 RMSNorm 归一化权重（可选，用于 QK-Norm 变体）
  - `w_attn_out`: 注意力输出投影权重
  - `w_ffn_gate_up`, `w_ffn_down`: FFN 门控/上投影和下投影权重
  - `sin_table`, `cos_table`: RoPE 位置编码查找表（shape: `[dctx, dh/2]`，支持 F16/BF16/F32）
  - `stream`: CUDA/设备计算流
  - `comm`: NCCL 通信器（多设备并行时使用）
  - `memory_pool`: 设备内存池（默认 128MB）
- **生命周期**: 在 `createDeviceResource()` 中初始化，在 `launchDevice()` 线程中使用，在 `releaseDeviceResource()` 中释放

### `InferState`
- **位置**: `jiuge_impl.hpp:33-39`
- **主要功能**: 管理单个设备推理线程的同步状态，实现生产者-消费者模式
- **关键成员**:
  - `mtx`: 互斥锁，保护状态变量
  - `cv_load`: 加载完成条件变量（用于初始化同步）
  - `cv_start`: 推理启动条件变量（主线程通知工作线程）
  - `cv_done`: 推理完成条件变量（工作线程通知主线程）
  - `loaded`: 设备资源加载完成标志
  - `proceed`: 推理执行标志（true 表示有新任务）
  - `exit_flag`: 线程退出标志
- **同步流程**:
  1. 主线程等待 `cv_load` 确保所有设备初始化完成
  2. 主线程设置 `proceed=true` 并通过 `cv_start` 通知工作线程
  3. 工作线程执行推理，完成后设置 `proceed=false` 并通过 `cv_done` 通知主线程
  4. 主线程按逆序等待所有设备完成（确保最后一个设备先完成）

### `InferRequest`
- **位置**: `jiuge_impl.hpp:41-53`
- **主要功能**: 封装单次批量推理请求的所有输入输出参数
- **关键成员**:
  - `tokens`: 扁平化的输入 token 序列（所有请求拼接）
  - `ntok`: 总 token 数量
  - `req_lens`: 每个请求的长度数组（shape: `[nreq]`）
  - `nreq`: 请求数量
  - `req_pos`: 每个请求的起始位置（用于增量推理）
  - `kv_caches`: KV Cache 指针数组（shape: `[nreq]`，每个请求独立）
  - `temperature`, `topk`, `topp`: 采样参数（每请求独立）
  - `output`: 输出 token 数组（shape: `[nreq]`，仅推理模式）
  - `logits`: 输出 logits 缓冲区（仅前向传播模式）
- **内存管理**: 所有指针由调用者管理，内部仅引用不释放

### `JiugeModel`
- **位置**: `jiuge_impl.hpp:55-65`
- **主要功能**: 模型主体，管理多设备资源、工作线程和推理请求
- **关键成员**:
  - `meta`: 模型元数据（层数、隐藏层维度、头数等）
  - `device`: 设备类型
  - `dev_ids`: 设备 ID 列表（支持多设备并行）
  - `dev_resources`: 设备资源数组（每设备一个）
  - `states`: 线程同步状态数组（每设备一个）
  - `threads`: 工作线程数组（每设备一个线程）
  - `req`: 当前推理请求（线程间共享）
- **初始化流程**:
  1. 创建设备资源数组和同步状态数组
  2. 如果多设备，初始化 NCCL 通信器
  3. 启动所有工作线程（调用 `launchDevice()`）
  4. 等待所有设备加载完成（`cv_load` 同步）
- **销毁流程**:
  1. 设置所有线程的 `exit_flag=true`
  2. 通知所有线程退出（`cv_start`）
  3. 等待所有线程加入（`join()`）
  4. 释放模型对象

## 3. 核心函数接口

```cpp
// 权重张量转换函数（jiuge_weight.hpp）
std::shared_ptr<Tensor> getInEmbd(JiugeMeta const *meta, JiugeWeights const *w);
// 从权重中提取输入嵌入层张量，shape: [dvoc, d] 或 [d, dvoc]（取决于 transpose_linear_weights）

std::shared_ptr<Tensor> getAttnQKV(JiugeMeta const *meta, JiugeWeights const *w,
                                   size_t layer, size_t idev, size_t ndev);
// 提取第 layer 层的 QKV 投影权重，支持张量并行分片
// 分片策略：将 (nh + 2*nkvh) * dh 维度按设备数均分
// 计算偏移量：offset = idev * ((nkvh * 2 + nh) / ndev * dh) * d * dsize(dt_mat)

std::shared_ptr<Tensor> getSinTable(JiugeMeta const *meta);
// 动态生成 RoPE sin 查找表，shape: [dctx, dh/2]
// 算法：sin(i / theta^(j / (dh/2)))，i 为位置，j 为维度索引
// 支持数据类型：F16/BF16/F32（自动转换）

std::shared_ptr<Tensor> getCosTable(JiugeMeta const *meta);
// 动态生成 RoPE cos 查找表，shape: [dctx, dh/2]
// 算法：cos(i / theta^(j / (dh/2)))

// 设备资源管理（jiuge.cpp）
void createDeviceResource(JiugeDeviceResource *rsrc, const JiugeMeta *meta,
                          const JiugeWeights *weights,
                          infiniDevice_t device, int idev, int ndev,
                          int dev_id, infinicclComm_t comm);
// 初始化单个设备的所有资源
// 1. 设置设备并创建 InfiniOP 句柄和计算流
// 2. 加载所有层权重（支持张量并行分片）
// 3. 生成 RoPE 位置编码表（sin/cos）
// 4. 创建 128MB 内存池
// 5. 同步设备确保所有操作完成

void releaseDeviceResource(JiugeDeviceResource &res);
// 释放设备资源
// 1. 同步设备
// 2. 释放所有权重张量（shared_ptr 自动管理）
// 3. 清空所有向量
// 4. 销毁句柄、流和通信器

// 核心推理函数（jiuge.cpp）
void inferDeviceBatch(const JiugeMeta &meta, JiugeDeviceResource &rsrc,
                      uint32_t idev, uint32_t ndev,
                      const uint32_t *tokens, uint32_t ntok,
                      const uint32_t *req_lens, uint32_t nreq,
                      const uint32_t *req_pos,
                      struct KVCache **kv_caches,
                      const float *temperature, const uint32_t *topk,
                      const float *topp, uint32_t *output, void *last_logits);
// 单设备批量推理实现
// 参数：
//   - meta: 模型元数据
//   - rsrc: 设备资源（包含权重、流、内存池）
//   - idev/ndev: 当前设备索引和总设备数
//   - tokens/ntok: 扁平化 token 序列和总长度
//   - req_lens/nreq: 每个请求的长度和请求数量
//   - req_pos: 每个请求的起始位置
//   - kv_caches: KV Cache 数组
//   - temperature/topk/topp: 采样参数
//   - output: 输出 token 数组（可为 nullptr）
//   - last_logits: 输出 logits 缓冲区（可为 nullptr）
//
// 执行流程：
// 1. 分配推理缓冲区（logits, qkv, gate_up, o_buf, prob_buf, result_buf）
// 2. 准备输入：
//    - 构造批量位置 ID（req_pos + 偏移）
//    - 词嵌入查找（表查找，每个 token 独立）
// 3. 逐层计算：
//    a) Attention:
//       - RMSNorm 归一化
//       - QKV 投影（可选偏置）
//       - QK-Norm（可选，针对 Q/K 分别归一化）
//       - RoPE 位置编码（旋转 q 和 k）
//       - KV Cache 更新（拼接新 k/v）
//       - QK 矩阵乘法（分组查询注意力 GQA）
//       - 因果 softmax
//       - 注意力权重与 V 的乘积
//       - 输出投影 + 残差连接（仅 rank 0）
//       - AllReduce（多设备时）
//    b) FFN:
//       - RMSNorm 归一化
//       - 门控/上投影（SiLU 激活）
//       - 下投影 + 残差连接（仅 rank 0）
//       - AllReduce（多设备时）
// 4. 输出（仅 rank 0）：
//    - 如果 last_logits 非空：计算最终 logits 并拷贝到主机
//    - 如果 output 非空：采样生成 token（随机采样支持 top-k/top-p/温度）
//
// 关键优化：
// - 使用 MemoryPool 避免频繁分配
// - 支持张量模型并行（权重分片 + AllReduce）
// - GQA 优化（多查询注意力，减少 KV Cache 内存）
// - 批量推理（动态形状支持）

// C API 接口（jiuge.cpp）
struct JiugeModel *createJiugeModel(const JiugeMeta *meta,
                                   const JiugeWeights *weights,
                                   infiniDevice_t device,
                                   int ndev, const int *dev_ids);
// 创建 Jiuge 模型实例
// 参数：
//   - meta: 模型元数据（层数、维度、头数等）
//   - weights: 模型权重指针
//   - device: 设备类型（INFINI_DEVICE_CPU/CUDA/KUNLUN 等）
//   - ndev: 设备数量
//   - dev_ids: 设备 ID 数组
// 返回：模型实例指针

void inferBatchJiuge(struct JiugeModel *model,
                    const uint32_t *tokens, uint32_t ntok,
                    const uint32_t *req_lens, uint32_t nreq,
                    const uint32_t *req_pos,
                    struct KVCache **kv_caches,
                    const float *temperature, const uint32_t *topk,
                    const float *topp, uint32_t *output);
// 批量推理并采样生成 token（阻塞调用）
// 流程：
// 1. 设置推理请求参数
// 2. 顺序通知所有设备开始推理
// 3. 逆序等待所有设备完成（确保最后一个设备先完成）

void forwardBatchJiuge(struct JiugeModel *model,
                      const uint32_t *tokens, uint32_t ntok,
                      const uint32_t *req_lens, uint32_t nreq,
                      const uint32_t *req_pos,
                      struct KVCache **kv_caches, void *logits);
// 批量前向传播，输出 logits（不采样）
// 用途：用于需要获取最终 logits 的场景（如 log-likelihood 计算）

void destroyJiugeModel(struct JiugeModel *model);
// 销毁模型实例
// 流程：
// 1. 设置所有线程的 exit_flag
// 2. 通知所有线程退出
// 3. 等待所有线程加入
// 4. 释放模型对象
```

## 4. 使用示例

```cpp
// 示例：创建 Jiuge 模型并执行批量推理

// 1. 准备模型元数据
JiugeMeta meta = {
    .nlayer = 32,           // Transformer 层数
    .d = 4096,              // 隐藏层维度
    .nh = 32,               // 注意力头数
    .nkvh = 4,              // KV 头数（GQA，nh/nkvh = 8 组）
    .dh = 128,              // 每个头的维度
    .di = 11008,            // FFN 中间层维度
    .dvoc = 128256,         // 词汇表大小
    .dctx = 4096,           // 最大上下文长度
    .dt_logits = INFINI_DTYPE_F16,  // logits 数据类型
    .dt_norm = INFINI_DTYPE_F32,    // 归一化权重类型
    .dt_mat = INFINI_DTYPE_F16,     // 矩阵权重类型
    .epsilon = 1e-5f,       // RMSNorm epsilon
    .theta = 10000.0f,      // RoPE theta 基数
};

// 2. 准备模型权重（已从文件加载）
JiugeWeights weights = {
    .input_embd = input_embd_ptr,        // [dvoc, d]
    .output_embd = output_embd_ptr,      // [d, dvoc]
    .output_norm = output_norm_ptr,      // [d]
    .attn_norm = attn_norm_layers,       // [nlayer][d]
    .attn_qkv = attn_qkv_layers,         // [nlayer][d, (nh + 2*nkvh) * dh]
    .attn_qkv_b = attn_qkv_bias_layers,  // [nlayer][(nh + 2*nkvh) * dh]（可选）
    .attn_q_norm = attn_q_norm_layers,   // [nlayer][dh]（可选）
    .attn_k_norm = attn_k_norm_layers,   // [nlayer][dh]（可选）
    .attn_o = attn_o_layers,             // [nlayer][nh * dh, d]
    .ffn_norm = ffn_norm_layers,         // [nlayer][d]
    .ffn_gate_up = ffn_gate_up_layers,   // [nlayer][d, 2 * di]
    .ffn_down = ffn_down_layers,         // [nlayer][di, d]
    .transpose_linear_weights = 0,       // 权重是否转置存储
};

// 3. 创建模型（单设备）
int dev_id = 0;
JiugeModel *model = createJiugeModel(&meta, &weights,
                                     INFINI_DEVICE_CUDA,
                                     1, &dev_id);

// 或创建多设备模型（张量并行）
int dev_ids[4] = {0, 1, 2, 3};
JiugeModel *model = createJiugeModel(&meta, &weights,
                                     INFINI_DEVICE_CUDA,
                                     4, dev_ids);

// 4. 准备推理请求
// 假设有 3 个请求，长度分别为 5, 3, 4 tokens
uint32_t nreq = 3;
uint32_t req_lens[3] = {5, 3, 4};
uint32_t ntok = 5 + 3 + 4;  // 总 token 数
uint32_t tokens[12] = {...};  // 扁平化的 token 序列
uint32_t req_pos[3] = {0, 10, 20};  // 每个请求的起始位置

// 准备 KV Cache（每个请求独立）
struct KVCache **kv_caches = new KVCache *[nreq];
for (uint32_t i = 0; i < nreq; i++) {
    kv_caches[i] = createKVCache(...);  // 外部创建的 KV Cache
}

// 5. 准备采样参数
float temperature[3] = {0.8f, 0.7f, 0.9f};  // 每请求独立
uint32_t topk[3] = {50, 40, 50};
float topp[3] = {0.9f, 0.95f, 0.85f};

// 6. 执行批量推理（采样模式）
uint32_t output[3];  // 输出 token 数组
inferBatchJiuge(model, tokens, ntok, req_lens, nreq, req_pos,
               kv_caches, temperature, topk, topp, output);

// 7. 处理输出
for (uint32_t i = 0; i < nreq; i++) {
    printf("Request %u generated token: %u\n", i, output[i]);
}

// 或执行前向传播（不采样，获取 logits）
void *logits = malloc(dsize(meta.dt_logits) * nreq * meta.dvoc);
forwardBatchJiuge(model, tokens, ntok, req_lens, nreq, req_pos,
                 kv_caches, logits);
// logits 是一个 [nreq, dvoc] 的数组

// 8. 清理资源
destroyJiugeModel(model);
for (uint32_t i = 0; i < nreq; i++) {
    destroyKVCache(kv_caches[i]);
}
delete[] kv_caches;
free(logits);
```

## 5. 实现细节

### 内存管理策略
- **MemoryPool**: 每个设备维护 128MB 的内存池，用于推理过程中的临时缓冲区分配（logits, qkv, gate_up, o_buf 等），避免频繁的设备内存分配/释放
- **权重共享**: 所有权重使用 `std::shared_ptr` 管理，支持多线程安全引用计数
- **RoPE 表预分配**: sin/cos 查找表在初始化时生成并转换为推理数据类型（F16/BF16），避免运行时重复计算

### 并发与同步
- **线程模型**: 每个设备一个工作线程（`launchDevice()`），采用生产者-消费者模式
- **同步原语**:
  - `std::mutex` + `std::condition_variable` 实现线程间同步
  - `InferState` 结构封装同步状态（loaded/proceed/exit_flag）
  - 主线程按逆序等待设备完成（避免最后一个设备成为瓶颈）
- **无锁设计**: 工作线程之间无直接通信，仅通过主线程协调

### 性能优化技术
- **张量模型并行**:
  - QKV 投影按设备数分片：`(nh + 2*nkvh) * dh` 维度切分
  - FFN 中间层分片：`di` 维度切分
  - 注意力输出和 FFN 输出通过 AllReduce 聚合（仅 rank 0 添加残差）
- **GQA（Grouped Query Attention）**:
  - 多个查询头共享一组 KV 头（`ngroup = nh / nkvh`）
  - 减少 KV Cache 内存占用（`nkvh << nh` 时效果显著）
  - 实现：将 Q 重排为 `[nkvh, ngroup, seq_len, dh]`，K/V 为 `[nkvh, total_len, dh]`
- **KV Cache 复用**:
  - 支持增量推理（`req_pos` 记录已处理位置）
  - KV Cache 通过外部管理（`KVCache` 结构），支持跨请求共享
- **算子融合**:
  - SiLU 激活与门控融合（`swiglu(gate, up, gate)`）
  - RMSNorm 融合 epsilon（避免单独的 add 操作）
- **批量推理优化**:
  - 动态形状支持（每个请求长度可不同）
  - 位置 ID 批量构造（`req_pos + offset`）
  - 分片处理（每个请求独立计算 QK、softmax）

### 错误处理与容错
- **宏检查**: 使用 `RUN_INFINI()` 宏检查 InfiniRT/InfiniOP/InfiniNCCL 调用返回值
- **设备同步**: 关键操作后调用 `infinirtDeviceSynchronize()` 确保完成
- **流同步**: 多设备通信前调用 `infinirtStreamSynchronize()` 确保依赖完成
- **数据类型检查**: RoPE 表生成时检查 `dt_logits` 是否为 F16/BF16/F32，否则报错退出

### 依赖关系
- **InfiniCore**: 提供 `Tensor`、`MemoryPool`、`InferenceContext`、`CacheManager` 等基础设施
- **InfiniRT**: 设备管理（`infinirtSetDevice`）、内存操作（`infinirtMemcpyAsync`）、流管理（`infinirtStreamCreate`）
- **InfiniOP**: 算子库（`infiniopCreateHandle`）、基础算子（rmsnorm, linear, rope, causalSoftmax 等）
- **InfiniNCCL**: 多设备通信（`infinicclCommInitAll`, `infinicclAllReduce`）
- **KV Cache**: 外部依赖（`struct KVCache`），由上层管理

### 设计模式
- **RAII**: 使用 `std::shared_ptr` 管理权重张量生命周期
- **Thread-per-device**: 每个设备一个专用线程，避免上下文切换
- **Producer-Consumer**: 主线程生产任务，工作线程消费任务
- **Object Pool**: MemoryPool 实现缓冲区复用
- **Strategy Pattern**: 支持多种设备类型（CPU/CUDA/Kunlun）和通信模式（单机/多机）
- **Template Method**: 推理流程固定（Attention -> FFN -> Output），具体算子可配置

### 关键算法复杂度
- **注意力计算**: O(ntok² × d)（每层 QK 矩阵乘法，softmax 与 V 的乘法）
- **KV Cache 更新**: O(ntok × d)（拼接操作）
- **FFN**: O(ntok × d × di)（两次线性变换）
- **采样**: O(dvoc × nreq)（logits 计算和随机采样）
- **总复杂度**: O(nlayer × (ntok² × d + ntok × d × di) + nreq × dvoc)
