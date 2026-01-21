# Paged Attention Prefill NVIDIA 后端实现文档

## 模块概述

本模块实现了 NVIDIA CUDA 后端的分页注意力预填充（Paged Attention Prefill）算子，用于大语言模型推理服务中的高效 KV 缓存管理。该实现支持分块 KV 缓存（Paged KV Cache）、多查询注意力（MQA）、分组查询注意力（GQA）以及 ALiBI 位置偏置，是推理加速的核心组件。

## 1. 模块结构

- **`paged_attention_prefill_nvidia.cuh`**: NVIDIA 后端描述符头文件，通过 `DESCRIPTOR` 宏定义 `op::paged_attention_prefill::nvidia::Descriptor` 类结构
- **`paged_attention_prefill_nvidia.cu`**: NVIDIA CUDA 核心实现文件，包含内核启动函数、描述符生命周期管理和类型派发逻辑

## 2. 核心类与数据结构

### `PagedAttentionPrefillInfo`
- **位置**: `../info.h` (父目录共享)
- **功能**: 分页注意力预填充操作的元数据容器，存储张量形状、步长、配置参数等信息
- **核心成员**:
  - `infiniDtype_t dtype`: 数据类型（FP16/BF16/FP32）
  - `float scale`: 注意力缩放因子（通常为 `1/sqrt(head_size)`）
  - `size_t num_seqs`: 批次中的序列数量
  - `size_t num_heads`: Query 头数量
  - `size_t num_kv_heads`: KV 头数量（支持 GQA/MQA）
  - `size_t head_size`: 每个注意力头的维度
  - `size_t block_size`: 每个 KV 缓存块的 token 数量
  - `size_t max_num_blocks_per_seq`: 每个序列的最大块数
  - `size_t total_q_tokens`: 所有序列的 Query token 总数
  - `ptrdiff_t q_stride, q_head_stride`: Query 张量的跨步
  - `ptrdiff_t kv_block_stride, kv_head_stride`: KV 缓存的跨步
- **工厂方法**: `create()` - 从张量描述符验证并构造 `PagedAttentionPrefillInfo`，检查数据类型一致性、形状合法性、步长有效性

### `op::paged_attention_prefill::nvidia::Descriptor`
- **位置**: `paged_attention_prefill_nvidia.cuh` (宏生成) + `paged_attention_prefill_nvidia.cu`
- **功能**: NVIDIA 后端算子描述符，管理 CUDA 设备上下文和计算参数
- **内部结构**:
  ```cpp
  struct Descriptor::Opaque {
      std::shared_ptr<device::nvidia::Handle::Internal> internal;  // CUDA 设备句柄
  };
  ```
- **核心成员**:
  - `Opaque *_opaque`: CUDA 设备上下文的不透明指针
  - `PagedAttentionPrefillInfo _info`: 算子元数据
  - `size_t _workspace_size`: 工作空间大小（当前为 0）
- **生命周期**:
  - **创建**: `create()` 静态方法构造描述符，调用 `PagedAttentionPrefillInfo::create()` 验证输入
  - **销毁**: `~Descriptor()` 析构函数释放 `_opaque` 内存
  - **计算**: `calculate()` 方法执行内核启动

## 3. 核心内核实现

### `pagedAttentionPrefillKernel<Tdata, Tcompute>`
- **位置**: `../cuda/kernel.cuh` (父目录共享 CUDA 内核)
- **命名空间**: `op::paged_attention_prefill::cuda`
- **功能**: 分页注意力预填充的 CUDA 核心函数，计算单 token 单头的注意力输出
- **函数签名**:
  ```cpp
  template <typename Tdata, typename Tcompute>
  __global__ void pagedAttentionPrefillKernel(
      Tdata *out_, const Tdata *q_, const Tdata *k_cache_, const Tdata *v_cache_,
      const int64_t *block_tables_, const int64_t *total_kv_lens_,
      const int64_t *cum_seq_lens_q_, const float *alibi_slopes_,
      const size_t num_heads, const size_t num_kv_heads, const float scale,
      const size_t max_num_blocks_per_seq, const size_t block_size,
      const ptrdiff_t kv_block_stride, const ptrdiff_t kv_head_stride,
      const ptrdiff_t q_stride, const ptrdiff_t q_head_stride,
      const size_t head_size, const size_t num_seqs);
  ```

- **Grid/Block 配置**:
  - **Grid**: `(total_q_tokens, num_heads)` - 每个线程块处理一个 token 的一个头
  - **Block**: `(head_size,)` - 每个线程处理 head 维度的一个元素

- **算法流程**:

  1. **序列定位** (第 36-48 行):
     ```cpp
     const size_t global_token_idx = blockIdx.x;  // 全局 token 索引
     const size_t head_idx = blockIdx.y;          // 头索引
     const size_t dim_idx = threadIdx.x;          // 维度索引

     // 二分查找当前 token 所属的序列
     size_t seq_idx = find_seq_id(global_token_idx, cum_seq_lens_q_, num_seqs);
     size_t q_token_idx = global_token_idx - cum_seq_lens_q_[seq_idx];
     ```
     - `find_seq_id()` 使用二分查找在 `cum_seq_lens_q` 中定位序列 ID，时间复杂度 O(log `num_seqs`)

  2. **因果掩码计算** (第 48-51 行):
     ```cpp
     const size_t total_kv_len = total_kv_lens_[seq_idx];
     const size_t q_len = cum_seq_lens_q_[seq_idx + 1] - cum_seq_lens_q_[seq_idx];
     const size_t history_len = total_kv_len - q_len;
     const size_t causal_limit = history_len + q_token_idx;  // 因果上限
     ```
     - 计算当前 token 可见的历史 KV token 数量（支持增量预填充）

  3. **多查询/分组查询映射** (第 56-57 行):
     ```cpp
     const size_t num_queries_per_kv = num_heads / num_kv_heads;
     const size_t kv_head_idx = head_idx / num_queries_per_kv;  // KV 头索引
     ```
     - 支持 GQA（`num_kv_heads < num_heads`）和 MQA（`num_kv_heads == 1`）

  4. **注意力分数计算 - 第一遍（找最大值）** (第 62-80 行):
     ```cpp
     Tcompute max_score = -FLT_MAX;
     for (size_t t = 0; t <= causal_limit; ++t) {
         // 分页地址计算
         const size_t b_idx = t / block_size;
         const size_t t_off = t % block_size;
         const ptrdiff_t physical_block_id = block_table[b_idx];
         const Tdata *k_vec = k_cache_ + physical_block_id * kv_block_stride
                              + kv_head_idx * kv_head_stride + t_off * head_size;

         // 点积计算 Q·K^T
         Tcompute score = 0.0f;
         for (size_t d = 0; d < head_size; ++d) {
             score += static_cast<Tcompute>(q_vec[d]) * static_cast<Tcompute>(k_vec[d]);
         }
         score *= static_cast<Tcompute>(scale);

         // ALiBI 偏置
         if (alibi_slope != 0.0f) {
             score += alibi_slope * static_cast<float>(t - causal_limit);
         }

         max_score = fmaxf(max_score, score);
     }
     ```
     - **分页缓存访问**: 通过 `block_tables[seq_idx][b_idx]` 获取物理块 ID
     - **内存访问模式**: 跨步访问 KV 缓存，支持非连续内存布局
     - **数值稳定性**: 第一遍扫描记录最大分数，用于后续 softmax 归一化

  5. **Softmax 归一化 - 第二遍（求指数和）** (第 82-98 行):
     ```cpp
     Tcompute sum_exp = 0.0f;
     for (size_t t = 0; t <= causal_limit; ++t) {
         // 重新计算 score（与第一遍相同）
         sum_exp += expf(static_cast<float>(score - max_score));  // 减去最大值避免溢出
     }
     ```

  6. **加权求和 - 第三遍（计算输出）** (第 100-122 行):
     ```cpp
     Tcompute acc = 0.0f;
     Tcompute inv_sum = 1.0f / (sum_exp + 1e-6f);  // 添加 epsilon 避免除零
     for (size_t t = 0; t <= causal_limit; ++t) {
         // 重新计算 score
         Tcompute prob = expf(static_cast<float>(score - max_score)) * inv_sum;

         const Tdata *v_vec = v_cache_ + physical_block_id * kv_block_stride
                              + kv_head_idx * kv_head_stride + t_off * head_size;
         acc += prob * static_cast<Tcompute>(v_vec[dim_idx]);  // 仅累加当前线程负责的维度
     }
     out_ptr[dim_idx] = static_cast<Tdata>(acc);
     ```
     - **并行策略**: 每个线程负责一个输出维度，减少线程间同步

- **复杂度分析**:
  - **时间**: O(`causal_limit` × `head_size`) - 对每个历史 token 计算点积，共三次遍历
  - **空间**: O(1) - 仅使用寄存器局部变量

### `launchPagedAttentionPrefill<Tdata, Tcompute>`
- **位置**: `paged_attention_prefill_nvidia.cu`
- **功能**: 封装内核启动逻辑，验证参数并配置 Grid/Block
- **函数签名**:
  ```cpp
  template <typename Tdata, typename Tcompute>
  infiniStatus_t launchPagedAttentionPrefill(
      Tdata *out, const Tdata *q, const Tdata *k_cache, const Tdata *v_cache,
      const int64_t *block_tables, const int64_t *seq_lens,
      const int64_t *cum_seq_lens_q, const float *alibi_slopes,
      const size_t num_heads, const size_t num_seqs, const size_t num_kv_heads,
      const float scale, const size_t max_num_blocks_per_seq, const size_t block_size,
      const size_t total_q_tokens, const size_t head_size,
      const ptrdiff_t kv_block_stride, const ptrdiff_t kv_head_stride,
      const ptrdiff_t q_stride, const ptrdiff_t q_head_stride,
      cudaStream_t stream);
  ```
- **实现细节** (第 32-50 行):
  - **参数验证**: 检查 `total_q_tokens` 和 `num_heads` 非零
  - **Grid 配置**: `dim3 grid(total_q_tokens, num_heads)` - 每个 (token, head) 对一个线程块
  - **Block 配置**: `dim3 block(head_size)` - 每个头维度一个线程
  - **内核启动**: 使用 `<<<grid, block, 0, stream>>>` 异步执行
  - **错误处理**: 返回 `INFINI_STATUS_BAD_TENSOR_SHAPE` 或 `INFINI_STATUS_SUCCESS`

## 4. API 接口

### `Descriptor::create()`
```cpp
static infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,                      // InfiniOP 全局句柄
    Descriptor **desc_ptr,                        // [输出] 描述符指针
    infiniopTensorDescriptor_t out_desc,          // 输出张量 [total_tokens, num_heads, head_size]
    infiniopTensorDescriptor_t q_desc,            // Query 张量 [total_tokens, num_heads, head_size]
    infiniopTensorDescriptor_t k_cache_desc,      // K 缓存 [num_blocks, num_kv_heads, block_size, head_size]
    infiniopTensorDescriptor_t v_cache_desc,      // V 缓存 [num_blocks, num_kv_heads, block_size, head_size]
    infiniopTensorDescriptor_t block_tables_desc, // 块表 [num_seqs, max_num_blocks_per_seq]
    infiniopTensorDescriptor_t seq_lens_desc,     // 序列长度 [num_seqs]
    infiniopTensorDescriptor_t cum_seq_lens_q_desc, // Query 累积长度 [num_seqs + 1]
    const std::optional<infiniopTensorDescriptor_t> &alibi_slopes_desc, // [可选] ALiBI 斜率 [num_heads]
    float scale);                                 // 注意力缩放因子
```
- **功能**: 验证输入张量并构造描述符
- **返回值**: `INFINI_STATUS_SUCCESS` 或错误码
- **副作用**: 为 `*desc_ptr` 分配内存，调用方负责析构

### `Descriptor::calculate()`
```cpp
infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,       // 工作空间（当前未使用）
    void *out, const void *q,                     // 输出和 Query 数据
    const void *k_cache, const void *v_cache,     // KV 缓存数据
    const void *block_tables,                     // 块表数据
    const void *seq_lens,                         // 序列长度（当前内核未使用）
    const void *cum_seq_lens_q,                   // Query 累积长度
    const void *alibi_slopes,                     // ALiBI 斜率（可选）
    void *stream_) const;                         // CUDA 流
```
- **功能**: 执行分页注意力预填充计算
- **类型派发**: 根据 `_info.dtype` 选择模板特化:
  - `INFINI_DTYPE_F16` → `<half, float>`
  - `INFINI_DTYPE_BF16` → `<__nv_bfloat16, float>`
  - `INFINI_DTYPE_F32` → `<float, float>`
- **计算类型**: `Tcompute` 始终为 `float`，确保数值精度

## 5. 使用示例

```cpp
#include "infinicore.h"
#include "infiniop/ops/paged_attention_prefill/paged_attention_prefill_nvidia.cuh"

// 1. 创建 InfiniOP 句柄
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_NVIDIA, 0);

// 2. 定义张量形状
const size_t num_seqs = 4;
const size_t num_heads = 32;
const size_t num_kv_heads = 8;  // GQA 配置
const size_t head_size = 128;
const size_t block_size = 16;
const size_t max_num_blocks_per_seq = 128;

// 假设每个序列有不同长度的 queries
std::vector<int64_t> seq_lens = {10, 20, 15, 25};
std::vector<int64_t> cum_seq_lens_q = {0, 10, 30, 45, 70};
const size_t total_q_tokens = 70;

// 3. 创建张量描述符
infiniopTensorDescriptor_t q_desc, k_cache_desc, v_cache_desc, out_desc;
infiniopTensorDescriptor_t block_tables_desc, seq_lens_desc, cum_seq_lens_q_desc;

// Q: [70, 32, 128]
infiniopCreateTensorDescriptor(handle, &q_desc, INFINI_DTYPE_F16, 3,
    std::vector<int64_t>{total_q_tokens, num_heads, head_size}.data());

// K/V Cache: [num_blocks, 8, 16, 128]
infiniopCreateTensorDescriptor(handle, &k_cache_desc, INFINI_DTYPE_F16, 4,
    std::vector<int64_t>{1024, num_kv_heads, block_size, head_size}.data());
infiniopCreateTensorDescriptor(handle, &v_cache_desc, INFINI_DTYPE_F16, 4,
    std::vector<int64_t>{1024, num_kv_heads, block_size, head_size}.data());

// Block Tables: [4, 128]
infiniopCreateTensorDescriptor(handle, &block_tables_desc, INFINI_DTYPE_I64, 2,
    std::vector<int64_t>{num_seqs, max_num_blocks_per_seq}.data());

// Seq Lens: [4]
infiniopCreateTensorDescriptor(handle, &seq_lens_desc, INFINI_DTYPE_I64, 1,
    std::vector<int64_t>{num_seqs}.data());

// Cum Seq Lens Q: [5]
infiniopCreateTensorDescriptor(handle, &cum_seq_lens_q_desc, INFINI_DTYPE_I64, 1,
    std::vector<int64_t>{num_seqs + 1}.data());

// Output: [70, 32, 128]
infiniopCreateTensorDescriptor(handle, &out_desc, INFINI_DTYPE_F16, 3,
    std::vector<int64_t>{total_q_tokens, num_heads, head_size}.data());

// 4. 创建算子描述符
op::paged_attention_prefill::nvidia::Descriptor *prefill_desc;
auto status = op::paged_attention_prefill::nvidia::Descriptor::create(
    handle, &prefill_desc,
    out_desc, q_desc, k_cache_desc, v_cache_desc,
    block_tables_desc, seq_lens_desc, cum_seq_lens_q_desc,
    std::nullopt,  // 不使用 ALiBI
    1.0f / sqrt(head_size));  // scale = 1/sqrt(d_k)

if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 5. 分配 GPU 内存
half *d_q, *d_k_cache, *d_v_cache, *d_out;
int64_t *d_block_tables, *d_seq_lens, *d_cum_seq_lens_q;

cudaMalloc(&d_q, total_q_tokens * num_heads * head_size * sizeof(half));
cudaMalloc(&d_k_cache, 1024 * num_kv_heads * block_size * head_size * sizeof(half));
cudaMalloc(&d_v_cache, 1024 * num_kv_heads * block_size * head_size * sizeof(half));
cudaMalloc(&d_out, total_q_tokens * num_heads * head_size * sizeof(half));
cudaMalloc(&d_block_tables, num_seqs * max_num_blocks_per_seq * sizeof(int64_t));
cudaMalloc(&d_seq_lens, num_seqs * sizeof(int64_t));
cudaMalloc(&d_cum_seq_lens_q, (num_seqs + 1) * sizeof(int64_t));

// 6. 拷贝数据到 GPU
cudaMemcpy(d_q, h_q, ..., cudaMemcpyHostToDevice);
cudaMemcpy(d_k_cache, h_k_cache, ..., cudaMemcpyHostToDevice);
cudaMemcpy(d_v_cache, h_v_cache, ..., cudaMemcpyHostToDevice);
cudaMemcpy(d_block_tables, h_block_tables, ..., cudaMemcpyHostToDevice);
cudaMemcpy(d_seq_lens, seq_lens.data(), ..., cudaMemcpyHostToDevice);
cudaMemcpy(d_cum_seq_lens_q, cum_seq_lens_q.data(), ..., cudaMemcpyHostToDevice);

// 7. 创建 CUDA 流
cudaStream_t stream;
cudaStreamCreate(&stream);

// 8. 执行计算
status = prefill_desc->calculate(
    nullptr, 0,  // 无需工作空间
    d_out, d_q, d_k_cache, d_v_cache,
    d_block_tables, d_seq_lens, d_cum_seq_lens_q,
    nullptr,  // 无 ALiBI
    stream);

if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 9. 同步并拷回结果
cudaStreamSynchronize(stream);
cudaMemcpy(h_out, d_out, ..., cudaMemcpyDeviceToHost);

// 10. 清理
cudaFree(d_q);
cudaFree(d_k_cache);
cudaFree(d_v_cache);
cudaFree(d_out);
cudaFree(d_block_tables);
cudaFree(d_seq_lens);
cudaFree(d_cum_seq_lens_q);
cudaStreamDestroy(stream);
delete prefill_desc;
infiniopDestroyHandle(handle);
```

## 6. 实现细节

### 内存管理
- **零拷贝工作空间**: 当前实现不需要额外工作空间（`_workspace_size = 0`），所有计算在寄存器中完成
- **分页 KV 缓存**: 支持非连续的物理内存布局，通过 `block_tables` 映射逻辑块到物理块
- **多类型支持**: 模板化设计支持 `half`、`__nv_bfloat16`、`float` 数据类型，计算统一使用 `float` 精度

### 并发策略
- **线程层次**:
  - **Grid 级**: 每个 (token, head) 对分配一个线程块，完全并行化
  - **Block 级**: 每个头维度分配一个线程，协作计算单个注意力输出
- **同步点**: 无显式 `__syncthreads()`，每个线程独立处理自己的维度
- **流执行**: 支持异步 CUDA 流，可与其他算子流水线执行

### 性能优化
- **寄存器优化**: 所有中间变量（`max_score`、`sum_exp`、`acc`）存储在寄存器中，避免全局内存访问
- **循环展开**: 编译器自动展开 `head_size` 循环（使用 `-O3` 优化）
- **数值稳定性**:
  - Softmax 使用两遍算法（找最大值 + 指数和），避免浮点溢出
  - 除法添加 epsilon (`1e-6f`) 避免除零
- **内存合并**: KV 缓存访问使用跨步模式，但每个线程访问连续内存（`t_off * head_size + d`）

### 错误处理
- **输入验证**:
  - 数据类型检查：Q、K、V、Out 必须同类型（FP16/BF16/FP32）
  - 形状检查：Q 为 3D、K/V 为 4D、block_tables 为 2D
  - 维度限制：`head_size ≤ 1024`
  - 累积长度一致性：`cum_seq_lens_q[num_seqs] == total_q_tokens`
- **运行时错误**:
  - `total_q_tokens == 0` 或 `num_heads == 0` → `INFINI_STATUS_BAD_TENSOR_SHAPE`
  - 不支持的数据类型 → `INFINI_STATUS_BAD_TENSOR_DTYPE`

### 依赖关系
- **外部依赖**:
  - CUDA Toolkit: `cuda_fp16.h`、`cuda_bf16.h`（半精度支持）
  - 标准库: `float.h`（`FLT_MAX`）、`math.h`（`expf`）
- **内部依赖**:
  - `../../../devices/nvidia/nvidia_common.cuh`: CUDA 设备句柄
  - `../../../devices/nvidia/nvidia_kernel_common.cuh`: 内核通用工具（类型定义、宏）
  - `../paged_attention_prefill.h`: 描述符宏定义
  - `../info.h`: `PagedAttentionPrefillInfo` 类
  - `../cuda/kernel.cuh`: `pagedAttentionPrefillKernel` 内核实现

### 设计模式
- **工厂模式**: `Descriptor::create()` 静态工厂方法，集中验证逻辑
- **策略模式**: 模板特化根据数据类型选择不同内核实现
- **不透明指针模式**: `Descriptor::Opaque` 封装 CUDA 特定状态，隐藏实现细节
- **RAII**: 描述符析构函数自动释放 `_opaque` 内存

### 算法特性
- **因果注意力**: 通过 `causal_limit` 强制每个 token 只能访问自身及之前的 KV
- **增量预填充**: 支持 `history_len > 0`，可与已有 KV 缓存拼接
- **ALiBI (Attention with Linear Biases)**:
  - 通过 `alibi_slopes` 为每个头添加线性位置偏置
  - 偏置公式: `score += alibi_slope * (t - causal_limit)`
  - 支持 `alibi_slopes == nullptr`（无偏置）
- **GQA/MQA 支持**:
  - 通过 `num_queries_per_kv = num_heads / num_kv_heads` 映射多 Query 头到单一 KV 头
  - 减少 KV 缓存内存占用和计算量

## 7. 性能特征

- **计算复杂度**: O(`total_q_tokens` × `num_heads` × `avg_kv_len` × `head_size`)
- **内存复杂度**: O(1) 额外内存 + O(`num_kv_blocks` × `num_kv_heads` × `block_size` × `head_size`) KV 缓存
- **并行度**: `total_q_tokens × num_heads` 个线程块，每块 `head_size` 个线程
- **瓶颈**: 当 `avg_kv_len` 较大时，注意力计算（三次遍历）成为瓶颈；内存带宽限制 KV 缓存访问速度

## 8. 限制与注意事项

1. **head_size 上限**: 当前实现限制 `head_size ≤ 1024`，受 CUDA Block 最大线程数限制
2. **不支持 FlashAttention**: 未使用分块 tiling 技术，无法利用共享内存加速
3. **重复计算**: 三次遍历 KV 缓存（找最大值、求和、加权求和），可优化为单遍缓存分数
4. **内存非合并**: 跨步访问 KV 缓存可能导致缓存行利用率低
5. **无 Tensor Core 加速**: 未使用 WMMA API 优化矩阵乘法
