# Attention 算子核心实现文档

本模块实现了 Transformer 模型中的多头注意力机制（Multi-Head Attention），支持 KV Cache 优化和分组查询注意力（Grouped Query Attention, GQA）。该算子是 InfiniOP 框架中用于 LLM 推理的核心组件之一，提供高效的注意力计算和缓存管理功能。

## 1. 模块结构

- **`attention.h`**: 定义注意力算子的描述符宏 `DESCRIPTOR`，为不同命名空间生成类型安全的 Descriptor 类模板
- **`operator.cc`**: 注意力算子的完整实现，包括描述符创建、工作空间管理、算子执行和资源释放

## 2. 核心类与数据结构

### `InfiniopAttentionDescriptor`
- **位置**: `operator.cc:14-32`
- **主要功能**: 封装注意力计算所需的所有子算子描述符、工作空间布局和计算参数
- **关键成员**:
  - `rearrange_desc_k/v/q/out`: 用于张量重排的描述符，处理 K/V/Q 和输出的内存布局转换
  - `matmul_desc1/2`: 两个矩阵乘法描述符，分别计算 Q*K^T 和 Softmax(QK^T)*V
  - `softmax_desc`: 因果 softmax 描述符，应用注意力掩码和归一化
  - `workspace_size`: 总工作空间大小（字节）
  - `op_workspace_offset`: 子算子工作空间在总工作空间中的偏移量
  - `op_workspace_size`: 子算子使用的最大工作空间大小
  - `q_cont_offset`: Q 张量连续化存储的偏移量（仅在 Q 非连续时使用）
  - `att_score_offset`: 注意力分数张量（QK^T 结果）的存储偏移量
  - `att_val_offset`: 注意力值张量（Softmax 后与 V 相乘结果）的存储偏移量
  - `k_cache_offset/v_cache_offset`: KV Cache 在缓存张量中的写入位置偏移（基于 pos 参数）
  - `qk_alpha`: 缩放因子，值为 `1/sqrt(head_dim)`，用于缩放 QK^T 结果

### `Descriptor` (宏生成类)
- **位置**: `attention.h:7-35`
- **主要功能**: 通过宏 `DESCRIPTOR(NAMESPACE)` 为不同命名空间生成类型安全的描述符包装类
- **关键成员**:
  - `_opaque`: 不透明指针，指向 `InfiniopAttentionDescriptor` 实例
  - `_workspace_size`: 工作空间大小（字节）
- **核心方法**:
  - `workspaceSize()`: 返回所需工作空间大小
  - `create(handle, desc_ptr, y_desc, x_desc)`: 静态工厂方法，创建描述符实例
- **生命周期**: 由宏生成，构造时初始化基类和成员，析构时释放资源

## 3. API 接口

```cpp
// 创建注意力描述符
infiniStatus_t infiniopCreateAttentionDescriptor(
    infiniopHandle_t handle,                           // InfiniOP 句柄，包含设备和上下文信息
    infiniopAttentionDescriptor_t *desc_ptr,           // [输出] 创建的描述符指针
    infiniopTensorDescriptor_t out_desc,               // 输出张量描述符 [seq_len, n_q_head, head_dim]
    infiniopTensorDescriptor_t q_desc,                 // Query 张量描述符 [n_q_head, seq_len, head_dim]
    infiniopTensorDescriptor_t k_desc,                 // Key 张量描述符 [n_kv_head, seq_len, head_dim]
    infiniopTensorDescriptor_t v_desc,                 // Value 张量描述符 [n_kv_head, seq_len, head_dim]
    infiniopTensorDescriptor_t k_cache_desc,           // Key Cache 描述符 [n_kv_head, max_seq_len, head_dim]
    infiniopTensorDescriptor_t v_cache_desc,           // Value Cache 描述符 [n_kv_head, max_seq_len, head_dim]
    size_t pos                                         // 当前序列位置，用于 KV Cache 索引
);
// 返回 INFINI_STATUS_SUCCESS 成功，否则返回错误码（如形状不匹配、步长非法等）

// 获取所需工作空间大小
infiniStatus_t infiniopGetAttentionWorkspaceSize(
    infiniopAttentionDescriptor_t desc,                // 注意力描述符
    size_t *size                                       // [输出] 所需工作空间大小（字节）
);

// 执行注意力计算
infiniStatus_t infiniopAttention(
    infiniopAttentionDescriptor_t desc_,               // 注意力描述符
    void *workspace_,                                  // 工作空间指针
    size_t workspace_size_,                            // 工作空间大小
    void *out,                                        // [输出] 输出张量 [seq_len, n_q_head, head_dim]
    void const *q,                                    // Query 数据 [n_q_head, seq_len, head_dim]
    void const *k,                                    // Key 数据 [n_kv_head, seq_len, head_dim]
    void const *v,                                    // Value 数据 [n_kv_head, seq_len, head_dim]
    void *k_cache,                                    // Key Cache [n_kv_head, max_seq_len, head_dim]
    void *v_cache,                                    // Value Cache [n_kv_head, max_seq_len, head_dim]
    void *stream                                      // 计算流（CUDA 流或其他设备流）
);
// 返回 INFINI_STATUS_SUCCESS 成功，INFINI_STATUS_INSUFFICIENT_WORKSPACE 工作空间不足

// 销毁描述符并释放资源
infiniStatus_t infiniopDestroyAttentionDescriptor(
    infiniopAttentionDescriptor_t desc_                // 要销毁的描述符
);
```

## 4. 使用示例

```cpp
// 示例：在 LLM 推理中使用 Attention 算子（GPT 风格自回归生成）
#include "infiniop/ops/attention.h"

// 假设配置参数
size_t n_q_head = 32;          // Query 头数
size_t n_kv_head = 8;          // KV 头数（GQA：n_q_head / n_kv_head = 4）
size_t head_dim = 128;         // 每个头的维度
size_t seq_len = 1;            // 推理时每次生成一个 token
size_t max_seq_len = 2048;     // 最大序列长度（缓存大小）
size_t pos = 0;                // 当前序列位置（从 0 开始）

// 1. 创建张量描述符
infiniTensorDescriptor_t out_desc, q_desc, k_desc, v_desc;
infiniTensorDescriptor_t k_cache_desc, v_cache_desc;

size_t out_shape[3] = {seq_len, n_q_head, head_dim};
size_t qkv_shape[3] = {n_kv_head, seq_len, head_dim};  // k/v 使用 n_kv_head
size_t q_shape[3] = {n_q_head, seq_len, head_dim};     // q 使用 n_q_head
size_t cache_shape[3] = {n_kv_head, max_seq_len, head_dim};

infiniopCreateTensorDescriptor(&out_desc, 3, out_shape, nullptr, INFINI_DTYPE_FLOAT16);
infiniopCreateTensorDescriptor(&q_desc, 3, q_shape, nullptr, INFINI_DTYPE_FLOAT16);
infiniopCreateTensorDescriptor(&k_desc, 3, qkv_shape, nullptr, INFINI_DTYPE_FLOAT16);
infiniopCreateTensorDescriptor(&v_desc, 3, qkv_shape, nullptr, INFINI_DTYPE_FLOAT16);
infiniopCreateTensorDescriptor(&k_cache_desc, 3, cache_shape, nullptr, INFINI_DTYPE_FLOAT16);
infiniopCreateTensorDescriptor(&v_cache_desc, 3, cache_shape, nullptr, INFINI_DTYPE_FLOAT16);

// 2. 创建注意力描述符
infiniopAttentionDescriptor_t attn_desc;
infiniStatus_t status = infiniopCreateAttentionDescriptor(
    handle, &attn_desc, out_desc, q_desc, k_desc, v_desc,
    k_cache_desc, v_cache_desc, pos
);

// 3. 获取并分配工作空间
size_t workspace_size;
infiniopGetAttentionWorkspaceSize(attn_desc, &workspace_size);
void *workspace = nullptr;
cudaMalloc(&workspace, workspace_size);

// 4. 分配输入/输出/Cache 内存
void *out, *q, *k, *v, *k_cache, *v_cache;
cudaMalloc(&out, seq_len * n_q_head * head_dim * sizeof(half));
cudaMalloc(&q, seq_len * n_q_head * head_dim * sizeof(half));
cudaMalloc(&k, seq_len * n_kv_head * head_dim * sizeof(half));
cudaMalloc(&v, seq_len * n_kv_head * head_dim * sizeof(half));
cudaMalloc(&k_cache, n_kv_head * max_seq_len * head_dim * sizeof(half));
cudaMalloc(&v_cache, n_kv_head * max_seq_len * head_dim * sizeof(half));

// 5. 执行注意力计算（在自回归循环中）
cudaStream_t stream;
cudaStreamCreate(&stream);

for (pos = 0; pos < target_length; ++pos) {
    // 填充 q, k, v（从前层获取）
    // ...

    // 执行注意力
    status = infiniopAttention(
        attn_desc, workspace, workspace_size,
        out, q, k, v, k_cache, v_cache, stream
    );

    // 使用 out 进行后续计算
    // ...
}

// 6. 清理资源
infiniopDestroyAttentionDescriptor(attn_desc);
cudaFree(workspace);
cudaFree(out); cudaFree(q); cudaFree(k); cudaFree(v);
cudaFree(k_cache); cudaFree(v_cache);
cudaStreamDestroy(stream);
```

## 5. 实现细节

### 算法流程

该算子实现了标准的缩放点积注意力（Scaled Dot-Product Attention），支持分组查询注意力（GQA）和 KV Cache 优化。计算流程如下：

1. **KV Cache 更新**:
   - 将当前的 Key 和 Value 张量通过 `infiniopRearrange` 写入对应的 Cache 位置
   - 写入位置由 `pos` 参数决定：`k_cache_offset = pos * k_cache_desc->getByteStrides()[1]`
   - 这样在自回归生成时，每次只需追加新的 KV 对，避免重新计算历史 token

2. **Query 预处理**:
   - 检查 Q 张量是否在维度 0 和 1 上连续（`isContiguous(0, 1)`）
   - 如果不连续，通过 `rearrange_desc_q` 将其重排为连续布局，存储在工作空间中
   - 这确保后续矩阵乘法的高效内存访问

3. **注意力分数计算（MatMul 1）**:
   - 执行 `Q * K^T`，形状变换：
     - Q: `[n_q_head, seq_len, head_dim]` → `[n_kv_head, n_group, seq_len, head_dim]` → `[n_kv_head, n_group*seq_len, head_dim]`
     - K (从 Cache): `[n_kv_head, total_seq_len, head_dim]` → `[n_kv_head, head_dim, total_seq_len]` (转置)
     - 输出 (QK): `[n_kv_head, n_group*seq_len, total_seq_len]`
   - 应用缩放因子 `qk_alpha = 1/sqrt(head_dim)`，防止梯度消失

4. **因果 Softmax**:
   - 对 QK 结果应用因果 softmax（`infiniopCausalSoftmax`）
   - 掩码确保位置 `i` 只能 attend 到位置 `≤ i` 的 token
   - 在最后一个维度（total_seq_len）上归一化

5. **加权聚合（MatMul 2）**:
   - 执行 `Softmax(QK^T) * V`，形状变换：
     - Softmax(QK): `[n_kv_head, n_group*seq_len, total_seq_len]`
     - V (从 Cache): `[n_kv_head, total_seq_len, head_dim]`
     - 输出: `[n_kv_head, n_group*seq_len, head_dim]`

6. **输出重排**:
   - 将结果从 `[n_kv_head, n_group*seq_len, head_dim]` 重排为 `[seq_len, n_q_head, head_dim]`
   - 变换路径：`[n_kv_head, n_group, seq_len, head_dim]` → `[n_q_head, seq_len, head_dim]` → `[seq_len, n_q_head, head_dim]`

### 内存管理策略

- **工作空间布局**:
  - 工作空间分为两部分：`temp_tensors_size`（临时张量存储）和 `op_workspace_size`（子算子工作空间）
  - 临时张量按以下顺序排列：`att_score_offset` → `att_val_offset`（可能复用 `q_cont_offset` 的空间）
  - `q_cont_offset` 和 `att_val_offset` 可能重叠（`max(q_cont_size, att_val_size)`），因为 Q 连续化和注意力值计算不会同时使用该区域
  - 所有偏移量按 256 字节对齐（`alignment = 256`），以优化内存访问性能

- **KV Cache 管理**:
  - Cache 张量形状为 `[n_kv_head, max_seq_len, head_dim]`
  - 通过 `pos` 参数动态计算写入偏移，支持增量式 KV 追加
  - 在推理阶段，Cache 在整个生成过程中持久化，避免重复计算历史 token 的表示

### 性能优化技术

- **分组查询注意力（GQA）**:
  - 通过 `n_group = n_q_head / n_kv_head` 实现多头查询共享 KV
  - 减少内存占用和计算量（从 O(n_q_head * seq_len * head_dim) 降至 O(n_kv_head * seq_len * head_dim)）
  - 适用于大模型推理（如 LLaMA 2/3），在保持性能的同时降低 KV Cache 开销

- **算子融合与复用**:
  - 将重排、矩阵乘法、softmax 等 6 个子算子封装在单个描述符中
  - 工作空间共享：所有子算子复用同一块工作空间，取最大需求（`max(matmul1_ws, matmul2_ws, softmax_ws)`）
  - 减少内核启动开销和内存传输

- **内存对齐**:
  - 所有工作空间偏移和大小按 256 字节对齐
  - 优化 GPU 内存合并访问和缓存行利用率

### 并发与线程安全

- **流式执行**:
  - 所有子算子调用都接受 `stream` 参数，支持 CUDA 流或其他设备流
  - 允许多个注意力操作在不同流上并发执行
  - 内部通过 `CHECK_STATUS` 串行调度子算子，但子算子可能在设备上并行执行

- **无状态设计**:
  - 描述符在创建后是只读的（除了 KV Cache 数据本身）
  - 多个线程可以并发调用 `infiniopAttention`，只要使用不同的流和工作空间

### 错误处理

- **输入验证**:
  - 张量维度检查：所有输入必须是 3D 张量（`ndim() == 3`）
  - 形状一致性检查：输出 QKV 头数、序列长度、头维度必须匹配
  - 内存布局检查：输出必须连续（`isContiguous()`），QKV 的最后一维步长必须为 1
  - 工作空间大小检查：运行时验证 `workspace_size_ >= workspace_size`

- **错误码**:
  - `INFINI_STATUS_BAD_TENSOR_SHAPE`: 张量维度不为 3
  - `INFINI_STATUS_BAD_TENSOR_STRIDES`: 步长不满足要求（输出不连续或最后一维不连续）
  - `INFINI_STATUS_BAD_PARAM`: 形状不匹配（如 head_dim 不一致、seq_len 超过 max_seq_len）
  - `INFINI_STATUS_INSUFFICIENT_WORKSPACE`: 工作空间分配不足

### 依赖关系

- **上游依赖**:
  - `infiniop::Rearrange`: 张量重排算子，用于 Q 连续化、KV 写入 Cache、输出格式转换
  - `infiniop::Gemm`: 通用矩阵乘法，执行 Q*K^T 和 Softmax(QK^T)*V
  - `infiniop::CausalSoftmax`: 因果 softmax，应用注意力掩码和指数归一化
  - `utils::align`: 内存对齐工具函数
  - `TRANSFORM_TENSOR_DESC`: 宏工具，用于张量形状变换（分割、合并、转置）

- **数据类型支持**:
  - 支持任意浮点类型（通过 `infiniSizeOf(dtype)` 动态获取大小）
  - 典型使用 FP16 或 BF16 以减少内存占用和加速计算

### 设计模式

- **组合模式（Composite Pattern）**:
  - `InfiniopAttentionDescriptor` 将多个子算子（Rearrange、Gemm、Softmax）组合为单一的高层算子
  - 对外隐藏内部复杂性，提供简洁的注意力计算接口

- **模板方法模式（Template Method）**:
  - `DESCRIPTOR` 宏为不同命名空间生成结构相同的描述符类
  - 支持多后端实现（如 CUDA、CPU）的接口统一

- **RAII（资源获取即初始化）**:
  - 描述符构造时分配子算子描述符，析构时自动释放
  - 通过 `infiniopDestroyAttentionDescriptor` 确保资源清理

### 复杂度分析

- **时间复杂度**:
  - MatMul 1 (Q*K^T): O(n_q_head * seq_len * total_seq_len * head_dim)
  - Softmax: O(n_q_head * seq_len * total_seq_len)
  - MatMul 2 (Softmax(QK^T)*V): O(n_q_head * seq_len * total_seq_len * head_dim)
  - 总复杂度: O(n_q_head * seq_len * total_seq_len * head_dim)
  - 在推理阶段（seq_len=1）: O(n_q_head * 1 * pos * head_dim) = O(n_q_head * pos * head_dim)

- **空间复杂度**:
  - KV Cache: O(n_kv_head * max_seq_len * head_dim)
  - 工作空间: O(n_q_head * seq_len * total_seq_len) + 子算子临时空间
  - GQA 优化将 Cache 复杂度从 O(n_q_head) 降至 O(n_kv_head)

### 典型应用场景

- **自回归文本生成**:
  - 在 GPT、LLaMA 等解码器模型中，每步生成一个 token
  - KV Cache 避免重新计算历史 token，显著加速推理

- **分组查询注意力（GQA）**:
  - 适用于大语言模型（如 LLaMA 2 70B），通过减少 KV 头数降低内存开销
  - 平衡性能（Multi-Head Attention）和效率（Multi-Query Attention）

- **变长序列处理**:
  - 通过 `pos` 参数灵活处理不同长度的输入序列
  - 支持增量式解码和批处理中的动态序列长度
