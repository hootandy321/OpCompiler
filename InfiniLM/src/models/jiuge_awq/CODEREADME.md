# JiugeAWQ 模型实现核心文档

JiugeAWQ 是针对九格大模型的 AWQ (Activation-aware Weight Quantization) 量化推理实现，支持多设备并行推理、INT4 权重量化和高效的批量推理。该模块提供了完整的模型权重加载、设备资源管理和推理执行流程。

## 1. 模块结构

- **`jiuge_awq.hpp`**: 核心数据结构定义，包括量化权重、设备资源、推理请求和状态管理
- **`jiuge_awq_weight.cpp`**: 权重加载器实现，负责从权重文件中加载和初始化所有模型参数
- **`jiuge_awq.cpp`**: 推理引擎核心，实现设备资源管理、批量推理和多层 Transformer 前向传播

## 2. 核心数据结构

### `QuantInt4Weight`
- **位置**: `jiuge_awq.hpp:11-13`
- **功能**: 封装 INT4 量化权重的三个核心组件
- **成员变量**:
  - `w`: 量化后的权重张量（INT32 存储，打包格式）
  - `s`: 缩放因子张量（FP16），用于反量化
  - `z`: 零点偏移张量（INT32），用于反量化
- **量化方案**: 使用分组量化（quant_group_size），每组共享一个 scale 和 zero point

### `JiugeAWQDeviceWeight`
- **位置**: `jiuge_awq.hpp:15-20`
- **功能**: 单个设备的完整模型权重容器
- **关键成员**:
  - `w_in_embd`: 词嵌入层权重，形状 [dvoc, d]
  - `w_out_norm`: 输出层 RMSNorm 权重
  - `w_out_embd`: 语言模型头部权重（转置，形状 [d, dvoc]）
  - `sin_table`/`cos_table`: RoPE 旋转位置编码的正弦/余弦表，形状 [dctx, dh/2]
  - `w_attn_norm`: 每层的注意力层前归一化权重（向量）
  - `b_attn_q/k/v`: Q/K/V 投影偏置项（可选）
  - `w_attn_q/k/v/out`: 注意力层的量化投影权重（QKV 和输出投影）
  - `w_ffn_norm`: 每层的 FFN 前归一化权重
  - `w_ffn_gate/up/down`: FFN 的门控、上投影和下投影量化权重
- **生命周期**: 由 `JiugeAWQWeights` 构造时分配，模型销毁时释放

### `JiugeAWQWeights`
- **位置**: `jiuge_awq.hpp:22-33`
- **功能**: 权重加载器，继承自 `infinicore::weights::Loader`
- **核心方法**:
  - `JiugeAWQWeights(meta, device, dev_ids)`: 构造函数，初始化所有设备权重张量并注册到加载器
  - `device_weights()`: 返回所有设备的权重向量引用
- **权重注册流程**:
  1. 遍历每个设备和每一层
  2. 为权重张量分配形状和类型
  3. 通过 `register_weight()` 将张量与权重路径绑定
  4. 支持分分布列（FULL/COLUMN/ROW）以优化多设备加载

### `DeviceResource`
- **位置**: `jiuge_awq.hpp:35-48`
- **功能**: 封装单个设备运行时的所有资源
- **成员变量**:
  - `device`: 设备类型（CUDA/CPU 等）
  - `device_id`: 设备编号
  - `handle`: InfiniOP 算子库句柄
  - `weights`: 该设备对应的模型权重
  - `stream`: CUDA 流或设备执行流
  - `comm`: NCCL 通信句柄（多设备时使用）
  - `memory_pool`: 内存池，默认 128MB 预分配
- **生命周期**:
  - 构造时通过 `createDeviceResource()` 初始化
  - 推理线程中持续使用
  - 模型销毁时通过 `releaseDeviceResource()` 释放

### `InferRequest`
- **位置**: `jiuge_awq.hpp:50-62`
- **功能**: 封装单次批量推理请求的所有输入输出
- **成员变量**:
  - `tokens`: 扁平化的输入 token 序列，长度为 ntok
  - `ntok`: 批量中所有请求的总 token 数
  - `req_lens`: 每个请求的 token 长度数组，长度为 nreq
  - `nreq`: 批量中的请求数量
  - `req_pos`: 每个请求的起始位置（past length）
  - `kv_caches`: KV 缓存指针数组，每个请求一个
  - `temperature`: 采样温度数组
  - `topk`: top-k 采样参数数组
  - `topp`: top-p (nucleus) 采样参数数组
  - `output`: 输出 token 数组（仅采样模式）
  - `logits`: 输出 logits 指针（仅前向传播模式）

### `InferState`
- **位置**: `jiuge_awq.hpp:64-70`
- **功能**: 设备推理线程的同步状态管理
- **成员变量**:
  - `mtx`: 互斥锁，保护状态变量
  - `cv_load`: 加载完成条件变量
  - `cv_start`: 推理启动条件变量
  - `cv_done`: 推理完成条件变量
  - `loaded`: 设备资源是否已加载完成
  - `proceed`: 是否可以开始新一轮推理
  - `exit_flag`: 线程退出标志
- **同步机制**: 使用条件变量实现生产者-消费者模式，主线程通过 `proceed` 触发推理，工作线程推理完成后重置 `proceed` 并通知

### `JiugeAWQModel`
- **位置**: `jiuge_awq.hpp:72-82`
- **功能**: 模型实例的顶级容器，管理所有设备和线程
- **成员变量**:
  - `meta`: 模型元数据（层数、隐藏层维度、头数等）
  - `device`: 设备类型
  - `dev_ids`: 设备 ID 列表
  - `dev_resources`: 每个设备的资源向量
  - `states`: 每个设备的推理状态向量
  - `threads`: 每个设备的推理线程向量
  - `req`: 当前推理请求（共享状态）
- **核心方法**:
  - `JiugeAWQModel(meta, weights)`: 构造函数，初始化所有设备资源并启动推理线程
- **线程模型**: 每个设备一个专属推理线程，通过 `InferState` 同步

## 3. 核心 API 接口

### 模型创建与销毁

```cpp
// 权重加载器创建
ModelWeights *createJiugeAWQWeights(
    const JiugeAWQMeta *meta,      // 模型元数据
    infiniDevice_t device,          // 设备类型
    int ndev,                       // 设备数量
    const int *dev_ids              // 设备 ID 数组
);
// 从权重文件加载所有模型参数到 GPU 内存

// 模型实例创建
JiugeAWQModel *createJiugeAWQModel(
    const JiugeAWQMeta *meta,      // 模型元数据
    const ModelWeights *weights     // 已加载的权重
);
// 初始化所有设备资源并启动推理线程

// 模型销毁
void destroyJiugeAWQModel(struct JiugeAWQModel *model);
// 通知所有线程退出并等待完成，释放资源
```

### 推理执行接口

```cpp
// 批量推理并采样（生成模式）
void inferBatchJiugeAWQ(
    struct JiugeAWQModel *model,    // 模型实例
    const uint32_t *tokens,         // 输入 tokens
    uint32_t ntok,                  // 总 token 数
    const uint32_t *req_lens,       // 每个请求的长度
    uint32_t nreq,                  // 请求数量
    const uint32_t *req_pos,        // 每个请求的起始位置
    struct KVCache **kv_caches,     // KV 缓存数组
    const float *temperature,       // 采样温度
    const uint32_t *topk,          // top-k 采样
    const float *topp,             // top-p 采样
    uint32_t *output                // 输出 tokens
);
// 对批量请求执行完整的前向传播和采样，返回生成的 token

// 批量前向传播（无采样）
void forwardBatchJiugeAWQ(
    struct JiugeAWQModel *model,    // 模型实例
    const uint32_t *tokens,         // 输入 tokens
    uint32_t ntok,                  // 总 token 数
    const uint32_t *req_lens,       // 每个请求的长度
    uint32_t nreq,                  // 请求数量
    const uint32_t *req_pos,        // 每个请求的起始位置
    struct KVCache **kv_caches,     // KV 缓存数组
    void *logits                    // 输出 logits 指针
);
// 执行前向传播并返回最后一个 token 的 logits，用于 speculating decoding
```

## 4. 使用示例

```cpp
// 1. 定义模型元数据
JiugeAWQMeta meta = {
    .nlayer = 32,         // Transformer 层数
    .d = 4096,            // 隐藏层维度
    .nh = 32,             // 注意力头数
    .nkvh = 32,           // KV 头数（可等于 nh）
    .dh = 128,            // 每个头的维度
    .di = 11008,          // FFN 中间层维度
    .dctx = 2048,         // 最大上下文长度
    .dvoc = 32000,        // 词汇表大小
    .dt_logits = INFINI_DTYPE_F16,  // 计算精度
    .dt_norm_w = INFINI_DTYPE_F32,  // 归一化权重类型
    .epsilon = 1e-5,      // RMSNorm epsilon
    .theta = 10000.0,     // RoPE theta
    .has_qkv_bias = true, // 是否有 QKV 偏置
    .nbit = 4,            // 量化位数
    .quant_group_size = 128  // 量化分组大小
};

// 2. 加载权重（假设使用 2 个 GPU）
int dev_ids[] = {0, 1};
ModelWeights *weights = createJiugeAWQWeights(
    &meta, INFINI_DEVICE_CUDA, 2, dev_ids
);
// 内部会从文件加载所有权重到 GPU 0 和 GPU 1

// 3. 创建模型实例
JiugeAWQModel *model = createJiugeAWQModel(&meta, weights);
// 启动 2 个推理线程，每个 GPU 一个

// 4. 准备推理请求
uint32_t tokens[] = {1, 2, 3, 4};  // 2 个请求，各 2 个 tokens
uint32_t req_lens[] = {2, 2};
uint32_t req_pos[] = {0, 0};       // 都是新生成的请求
uint32_t ntok = 4, nreq = 2;

// 假设 KV 缓存已预分配
struct KVCache *kv_caches[2];
kv_caches[0] = allocateKVCache(...);
kv_caches[1] = allocateKVCache(...);

float temperature[] = {0.8f, 0.7f};
uint32_t topk[] = {50, 50};
float topp[] = {0.9f, 0.95f};
uint32_t output[2];

// 5. 执行推理
inferBatchJiugeAWQ(
    model,
    tokens, ntok,
    req_lens, nreq,
    req_pos,
    kv_caches,
    temperature, topk, topp,
    output
);
// 主线程会阻塞等待所有设备完成推理

// output[0] 和 output[1] 现在包含生成的 tokens

// 6. 清理
destroyJiugeAWQModel(model);
// 所有线程退出，资源释放
```

## 5. 实现细节

### 权重量化与反量化

- **量化格式**: INT4 分组量化（Group-wise Quantization）
  - 权重按 `quant_group_size`（默认 128）分组
  - 每组有独立的 scale (FP16) 和 zero point (INT32)
  - 4 个 INT4 权重打包成一个 INT32 存储
  - 量化权重张量形状: `[in_dim, out_dim * nbit / 32]`
  - Scale 张量形状: `[in_dim / quant_group_size, out_dim]`
  - Zero point 张量形状: `[in_dim / quant_group_size, out_dim * nbit / 32]`

- **反量化过程**（在推理时动态进行）:
  ```cpp
  dequant_linear(output, input, qweight, scale, zero_point,
                 alpha, beta, bias, add_bias)
  // 1. 使用 CUDA 核反量化: output = (input @ dequant(qweight, scale, zero_point)) + bias
  // 2. 可选的残差连接: output = alpha * output + beta * residual
  ```

### 多设备并行推理

- **张量并行策略**: 按列分割注意力头和 FFN 中间层
  - 每个设备负责 `nh / ndev` 个注意力头
  - FFN 的 `gate_proj` 和 `up_proj` 输出维度为 `di / ndev`
  - 每层的 `o_proj` 和 `down_proj` 输出需要进行 All-Reduce 汇总

- **通信模式**:
  - 使用 NCCL 进行跨设备 All-Reduce
  - 在 Attention 和 FFN 的输出投影后执行
  - 仅当 `ndev > 1` 时启用通信
  - 每个设备独立的推理线程，通过条件变量同步

### 内存管理

- **内存池策略**:
  - 每个设备预分配 128MB 内存池（`MemoryPool`）
  - 所有中间缓冲区从内存池分配
  - 推理过程中避免频繁的 malloc/free

- **缓冲区复用**:
  - `logits_in`/`logits_out`: 交替作为每层的输入输出
  - `q_buf`/`k_buf`/`v_buf`: 跨层复用的注意力缓冲区
  - `qk_buf`: 动态大小的 QK 矩阵乘积缓冲区，按最大序列长度预分配
  - `rearrange_q_buf`: Q 张量重排缓冲区，用于分组注意力计算

### 注意力机制实现

- **算法**: 分组查询注意力（Grouped Query Attention, GQA）
  - `nh` 个查询头，`nkvh` 个 KV 头（`nkvh <= nh`）
  - 每 `ngroup = nh / nkvh` 个查询头共享一组 KV
  - 减少 KV Cache 内存占用，提升推理吞吐

- **计算流程**（对每个请求）:
  1. **RMSNorm**: `logits_out = rmsnorm(logits_in, w_attn_norm)`
  2. **QKV 投影**: 反量化线性变换得到 Q, K, V
  3. **RoPE 旋转**: 对 Q 和 K 应用旋转位置编码（`rope_v2`）
  4. **KV Cache 更新**: 将新的 K, V 拼接到缓存末尾
  5. **QK 计算**: `QK = Q @ K^T / sqrt(dh)`，形状 `[nh, seq_len, total_len]`
  6. **Causal Softmax**: 应用因果掩码并 softmax
  7. **Attention Value**: `AttnVal = softmax(QK) @ V`
  8. **输出投影**: 反量化线性变换 + 残差连接
  9. **All-Reduce**: 多设备间汇总结果

- **形状变换**:
  - Q: `[ntok, nh, dh]` → permute → `[nkvh, ngroup, seq_len, dh]`（用于 GEMM）
  - K/V Cache: `[max_ctx, nkvh, dh]`
  - QK GEMM: `[nkvh, ngroup*seq_len, total_len]` → reshape → `[nh, seq_len, total_len]`

### FFN 实现

- **结构**: SwiGLU 激活函数
  1. **RMSNorm**: `logits_out = rmsnorm(logits_in, w_ffn_norm)`
  2. **Gate 投影**: `gate = dequant_linear(logits_out, w_ffn_gate)`
  3. **Up 投影**: `up = dequant_linear(logits_out, w_ffn_up)`
  4. **SwiGLU**: `swiglu(gate, up) = gate * swish(up)`
  5. **Down 投影**: `logits_in = dequant_linear(swiglu_out, w_ffn_down)` + 残差
  6. **All-Reduce**: 多设备汇总

### 采样策略

- **支持的采样模式**:
  - **Temperature Scaling**: 控制输出的随机性
  - **Top-k采样**: 仅从概率最高的 k 个 token 中采样
  - **Top-p (Nucleus)采样**: 从累积概率达到 p 的最小集合中采样
  - 使用 `std::random_device` 和 `std::mt19937` 生成均匀随机数

- **采样流程**（仅 Rank 0 执行）:
  1. 对每个请求的最后一个 token 进行 RMSNorm
  2. 线性变换得到 logits: `[nreq, dvoc]`
  3. 调用 `randomSample()` 应用 temperature、top-k、top-p 并采样
  4. 将结果拷贝回 CPU

### RoPE 位置编码

- **实现**: RoPE v2（改进版旋转位置编码）
  - 预计算 sin/cos 表: `[dctx, dh/2]`
  - 对 Q 和 K 的头维度后半部分应用旋转
  - 频率计算公式: `θ_i = θ^(-2i/dh)`，默认 `θ = 10000`
  - 支持任意长度的位置索引

### 线程同步与调度

- **生产者-消费者模式**:
  - **主线程**（生产者）: 准备请求，设置 `req` 结构，通知所有设备开始
  - **设备线程**（消费者）: 等待 `proceed` 信号，执行推理，完成后通知主线程
  - 使用 `std::condition_variable` 和 `std::mutex` 实现

- **初始化同步**:
  - 构造函数中为每个设备启动线程
  - 使用 `cv_load` 等待所有设备资源初始化完成

- **推理同步**:
  - 主线程按顺序（从 ndev-1 到 0）等待每个设备的 `cv_done`
  - 确保所有设备完成当前推理后再返回

- **退出机制**:
  - 销毁模型时设置所有线程的 `exit_flag`
  - 通知 `cv_start` 唤醒线程
  - 线程检查 `exit_flag` 并退出循环

### 性能优化

- **计算优化**:
  - 所有 GEMM 使用 InfiniOP 高度优化内核
  - 反量化与矩阵乘融合为单次算子调用
  - 使用 FP16 混合精度计算（`dt_logits = INFINI_DTYPE_F16`）

- **内存优化**:
  - KV Cache 按需增长，避免预分配最大上下文长度
  - 中间缓冲区复用，减少内存分配次数
  - 权重张量支持按列/行分割，减少冗余存储

- **并行优化**:
  - 多设备张量并行，线性扩展计算能力
  - 异步执行流（`stream`），支持计算与通信重叠
  - 每个设备独立线程，避免锁竞争

### 错误处理

- **宏封装**: 使用 `RUN_INFINI()` 宏检查 InfiniRuntime API 调用
  - 失败时抛出异常并打印错误信息
- **资源清理**: 析构函数中确保所有句柄和流正确销毁
- **线程安全**: 使用互斥锁保护共享状态（`InferState`）

### 依赖项

- **InfiniCore**: 基础张量运算和设备抽象
- **InfiniRuntime**: 设备管理、内存分配、流管理
- **InfiniOP**: 高性能算子库（GEMM、RMSNorm、Softmax、反量化等）
- **InfiniCCL**: 跨设备通信（NCCL 封装）
- **标准库**: `<thread>`, `<mutex>`, `<condition_variable>`, `<random>`

### 设计模式

- **RAII**: `DeviceResource` 的创建和销毁使用 RAII 模式
- **生产者-消费者**: 主线程和设备线程之间的请求分发
- **线程池**: 每个设备一个专属线程，避免线程创建开销
- **策略模式**: 支持 CPU 和 CUDA 等多种设备类型
- **模板方法**: 推理流程固定，具体算子实现可替换
