# Paged Attention NVIDIA 后端核心实现文档

本模块实现了 Paged Attention 算法的 NVIDIA GPU 后端，针对大语言模型推理中的 KV Cache 分页内存管理场景进行优化。通过分块 KV Cache 和动态块表映射，实现高效的可变长度序列注意力计算。

## 1. 模块结构

- **`paged_attention_nvidia.cuh`**: NVIDIA 后端描述符声明文件，通过宏定义生成完整的 Descriptor 类
- **`paged_attention_nvidia.cu`**: NVIDIA 后端实现主文件，包含内核启动逻辑、设备描述符管理和多态调度

## 2. 核心类与结构

### `Descriptor::Opaque` (内部结构体)
- **位置**: `paged_attention_nvidia.cu:27-29`
- **功能**: 封装 NVIDIA 设备特定的内部句柄，提供硬件架构信息
- **关键成员**:
  - `internal`: `std::shared_ptr<device::nvidia::Handle::Internal>` - NVIDIA 设备内部句柄，用于查询硬件能力（如最大线程块大小）
- **生命周期**: 由 `Descriptor::create` 动态分配，在 `Descriptor` 析构时释放

### `op::paged_attention::nvidia::Descriptor` (外部接口类)
- **位置**: 通过 `paged_attention.h` 中的 `DESCRIPTOR(nvidia)` 宏生成
- **功能**: Paged Attention 操作的 NVIDIA 后端描述符，继承自 `InfiniopDescriptor`
- **关键成员**:
  - `_opaque`: `Opaque *` - NVIDIA 设备句柄的封装
  - `_info`: `PagedAttentionInfo` - 包含张量形状、步长、数据类型等元信息
  - `_workspace_size`: `size_t` - 工作空间大小（当前为 0）
- **核心方法**:
  - `create(handle, desc_ptr, ...)` (L35-53): 静态工厂方法，验证输入张量描述符的形状和类型一致性，初始化描述符
    - 调用 `PagedAttentionInfo::create` 进行元数据验证
    - 从 `infiniopHandle_t` 提取 NVIDIA 内部句柄
    - 返回 `INFINI_STATUS_SUCCESS` 或错误码
  - `calculate(workspace, workspace_size, out, q, k_cache, v_cache, ...)` (L95-145): 执行 Paged Attention 计算
    - 根据 `maxThreadsPerBlock()` 动态选择线程块大小（512/1024/4096）
    - 根据 `head_size` 和 `dtype` 进行模板特化，展开为具体的内核实例
    - 使用三层宏嵌套（`SWITCH_HEAD_SIZE` → `LAUNCH_HEADSIZE_BLOCKSIZE`）实现编译期优化

## 3. CUDA 核心内核

### `pagedAttention<Tdata, Tcompute, HEAD_SIZE, NUM_THREADS>` (内核函数)
- **位置**: `paged_attention_nvidia.cu:10-23`
- **功能**: NVIDIA 后端的外层包装内核，将调用转发到 CUDA 通用实现
- **模板参数**:
  - `Tdata`: 数据类型（`half`, `__nv_bfloat16`, `float`）
  - `Tcompute`: 计算类型（固定为 `float`，保证数值稳定性）
  - `HEAD_SIZE`: 注意力头维度（编译期常量，支持 16/32/64/128/256）
  - `NUM_THREADS`: 线程块大小（编译期常量，支持 512/1024/4096）
- **参数列表**:
  ```cpp
  Tdata *out                                    // 输出 [num_seqs, num_heads, head_size]
  const Tdata *q                                // 查询 [num_seqs, num_heads, head_size]
  const Tdata *k_cache                          // 键缓存 [num_blocks, num_kv_heads, block_size, head_size]
  const Tdata *v_cache                          // 值缓存 [num_blocks, num_kv_heads, block_size, head_size]
  const int64_t *block_tables                   // 块表 [num_seqs, max_num_blocks_per_seq]
  const int64_t *seq_lens                       // 序列长度 [num_seqs]
  const float *alibi_slopes                     // ALiBI 偏置斜率 [num_heads] (可选)
  const size_t num_kv_heads                     // KV 头数量（用于 GQA/MQA）
  const float scale                             // 缩放因子（通常为 1/√head_size）
  const size_t max_num_blocks_per_seq           // 每个序列的最大块数
  const size_t block_size                       // 每个块的 token 数量
  const ptrdiff_t q_stride                      // Q 张量序列维度步长
  const ptrdiff_t kv_block_stride               // KV Cache 块维度步长
  const ptrdiff_t kv_head_stride                // KV Cache 头维度步长
  const ptrdiff_t o_stride                      // 输出张量序列维度步长
  ```
- **实现细节**:
  - 直接调用 `op::paged_attention::cuda::pagedAttentionKernel`（位于 `../cuda/kernel.cuh`）
  - NVIDIA 特定逻辑完全由 `Descriptor::calculate` 中的内核启动配置处理

### `pagedAttentionKernel<Tdata, Tcompute, HEAD_SIZE, NUM_THREADS>` (设备函数)
- **位置**: `../cuda/kernel.cuh:10-145`
- **功能**: Paged Attention 的核心 CUDA 设备函数，实现分块注意力计算
- **并行策略**: 二维网格布局（每个序列头一个线程块，每个线程块内 `NUM_THREADS` 个线程）
  - `gridDim.x = num_heads`: X 维度表示注意力头索引
  - `gridDim.y = num_seqs`: Y 维度表示序列索引
  - `threadIdx.x ∈ [0, NUM_THREADS)`: 线程块内的线程索引

#### 算法流程

**阶段 1：设置与查询加载** (L28-56)
1. 计算当前块的 `seq_idx`、`head_idx`，检查序列长度是否为 0（早期退出）
2. 计算对应的 KV 头索引（支持 GQA/MQA 的多头查询与单头 KV）：`kv_head_idx = head_idx / (num_heads / num_kv_heads)`
3. 加载 ALiBI 偏置斜率（如果提供）
4. 获取当前序列的块表指针
5. 从全局内存加载查询向量到共享内存：所有线程协作加载 `HEAD_SIZE` 个元素
6. 同步线程块（`__syncthreads()`）

**阶段 2：计算 QK 点积与查找最大 Logit** (L57-98)
1. 并行遍历序列中的所有 token（步长为 `NUM_THREADS`）
2. 对每个 token：
   - 计算其所在的块索引和块内偏移：`block_idx = token_idx / block_size`
   - 从块表查找物理块号：`physical_block_num = block_table[block_idx]`
   - 计算 K 向量的全局内存指针
   - 计算点积：使用手动循环展开（每 8 个元素展开一次）优化向量化
   - 应用缩放因子：`qk *= scale`
   - 添加 ALiBI 位置偏置（如果启用）：`qk += alibi_slope * (token_idx - seq_len + 1)`
   - 将结果存储到共享内存中的 `logits` 数组
3. 同步线程块
4. 使用 CUB 的 `BlockReduce` 进行归约，找到全局最大 logit 值（数值稳定的关键）
5. 将最大值广播到所有线程（存储到共享内存 `global_qk_max`）

**阶段 3：计算 Softmax** (L100-120)
1. 所有线程并行计算指数：`expf(logits[i] - global_qk_max)`
2. 使用 CUB 的 `BlockReduce` 计算指数和
3. Thread 0 计算归一化因子：`inv_sum = 1.0f / (exp_sum + 1e-6f)`
4. 所有线程并行应用归一化：`logits[i] *= inv_sum`
5. 多次同步确保一致性

**阶段 4：加权聚合值向量** (L122-144)
1. 并行遍历输出向量的每个维度（步长为 `NUM_THREADS`）
2. 对每个维度：
   - 遍历序列中的所有 token（串行循环）
   - 对每个 token，重新查找其物理块和 V 向量指针
   - 累加加权值：`acc += logits[token_idx] * v_cache[h_dim]`
   - 将结果写回到全局内存输出张量

## 4. 并行归约操作

### `op::common_cuda::reduce_op::max<BLOCK_SIZE, Tdata>`
- **位置**: `../../../reduce/cuda/reduce.cuh:49-79`
- **功能**: 在共享内存上查找最大值
- **算法**:
  1. 每个线程遍历数据（步长为 `BLOCK_SIZE`），计算局部最大值
  2. 使用 CUB 的 `BlockReduce::Reduce` 进行线程块级归约
  3. 仅 Thread 0 返回正确结果（需手动广播）

### `op::common_cuda::reduce_op::sum<BLOCK_SIZE, Tdata, Tcompute>`
- **位置**: `../../../reduce/cuda/reduce.cuh:34-47`
- **功能**: 在共享内存上计算和
- **算法**:
  1. 每个线程遍历数据（步长为 `BLOCK_SIZE`），累加局部和
  2. 使用 CUB 的 `BlockReduce::Sum` 进行线程块级归约

## 5. 使用示例

```cpp
#include "infinicore.h"
#include "infiniop/ops/paged_attention/nvidia/paged_attention_nvidia.cuh"

// 创建句柄和描述符
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_NVIDIA, 0);

// 定义张量形状
constexpr size_t num_seqs = 8;
constexpr size_t num_heads = 32;
constexpr size_t num_kv_heads = 8;      // GQA: 4 个查询头共享 1 个 KV 头
constexpr size_t head_size = 128;
constexpr size_t block_size = 16;
constexpr size_t max_num_blocks_per_seq = 128;

// 创建张量描述符
infiniTensorDescriptor_t q_desc, k_cache_desc, v_cache_desc, out_desc;
infiniTensorDescriptor_t block_tables_desc, seq_lens_desc;

// ... 初始化张量描述符 ...

// 创建 Paged Attention 描述符
op::paged_attention::nvidia::Descriptor *paged_attn_desc;
float scale = 1.0f / sqrtf(head_size);
auto status = op::paged_attention::nvidia::Descriptor::create(
    handle, &paged_attn_desc,
    out_desc, q_desc, k_cache_desc, v_cache_desc,
    block_tables_desc, seq_lens_desc,
    std::nullopt,  // 不使用 ALiBI
    scale
);

// 分配 GPU 内存
half *d_q, *d_k_cache, *d_v_cache, *d_out;
int64_t *d_block_tables, *d_seq_lens;
cudaMalloc(&d_q, num_seqs * num_heads * head_size * sizeof(half));
// ... 分配其他张量 ...

// 获取 CUDA Stream
cudaStream_t stream;
cudaStreamCreate(&stream);

// 执行计算
status = paged_attn_desc->calculate(
    nullptr, 0,           // 无需工作空间
    d_out, d_q, d_k_cache, d_v_cache,
    d_block_tables, d_seq_lens, nullptr,  // 无 ALiBI
    stream
);

// 同步与清理
cudaStreamSynchronize(stream);
delete paged_attn_desc;
infiniopDestroyHandle(handle);
```

## 6. 实现细节

### 内存管理
- **共享内存分配策略**: 每个线程块分配 `(HEAD_SIZE + max_num_blocks_per_seq * block_size + 2) * sizeof(float)` 字节
  - `HEAD_SIZE * sizeof(float)`: 存储查询向量（所有线程共享）
  - `max_num_blocks_per_seq * block_size * sizeof(float)`: 存储 logits（长度为序列长度）
  - `2 * sizeof(float)`: 存储全局最大值和归一化因子
- **计算类型分离**: 数据类型 `Tdata`（FP16/BF16/FP32）与计算类型 `Tcompute`（固定为 FP32）分离，确保数值稳定性

### 并发策略
- **线程块布局**: 二维网格 `(num_heads, num_seqs, 1)`，每个线程块独立处理一个序列的一个注意力头
- **线程块大小自适应**: 根据硬件架构 `maxThreadsPerBlock()` 动态选择 512/1024/4096
- **无跨线程块同步**: 算法设计确保线程块之间完全独立，无需全局同步

### 性能优化
- **手动循环展开**: QK 点积计算中每 8 个元素展开一次，充分利用 CUDA 的内存合并访问
- **编译期模板特化**: `HEAD_SIZE` 和 `NUM_THREADS` 作为模板参数，允许编译器进行激进优化
- **归约使用 CUB 库**: 利用 NVIDIA 高度优化的 `BlockReduce` 原语，避免手动实现复杂的 Warp Shuffle
- **零拷贝转发**: NVIDIA 后端内核直接调用 CUDA 通用实现，避免代码重复

### 错误处理
- **早期退出**: 序列长度为 0 的线程块立即返回，避免无效计算
- **形状验证**: `PagedAttentionInfo::create` 检查 `head_size` 必须为 16/32/64/128/256 之一
- **类型检查**: 严格验证所有输入张量的数据类型一致性
- **错误码传播**: 使用 `infiniStatus_t` 枚举返回详细错误信息（如 `INFINI_STATUS_BAD_TENSOR_DTYPE`）

### 依赖关系
- **外部依赖**:
  - CUB 库：`cub/block/block_reduce.cuh` - 提供 BlockReduce 原语
  - CUDA Toolkit：`cuda_fp16.h`, `cuda_bf16.h` - 半精度和 bfloat16 支持
- **内部依赖**:
  - `device::nvidia::Handle::Internal` - NVIDIA 设备句柄管理
  - `op::common_cuda::reduce_op` - 并行归约操作
  - `op::paged_attention::cuda::pagedAttentionKernel` - CUDA 通用内核实现

### 设计模式
- **策略模式 (Strategy Pattern)**: 通过模板特化支持不同的 `HEAD_SIZE` 和 `NUM_THREADS` 组合
- **工厂模式 (Factory Pattern)**: `Descriptor::create` 静态方法封装对象创建逻辑
- **适配器模式 (Adapter Pattern)**: NVIDIA 后端作为薄层适配器，将接口转换为 CUDA 通用实现

### 算法复杂度
- **时间复杂度**: O(`seq_len * head_size`) - 每个注意力头需要遍历所有 token 和所有维度
- **空间复杂度**: O(`seq_len + head_size`) - 共享内存存储 logits 和查询向量
- **并行度**: O(`num_heads * num_seqs * NUM_THREADS`) - 所有序列头独立并行，线程块内 NUM_THREADS 线程协作

### 支持的硬件特性
- **CUDA 架构**: 支持最大线程块大小为 512/1024/4096 的架构
- **数据类型**: FP16、BF16、FP32
- **注意力模式**: 支持 MHA (多头注意力)、GQA (分组查询注意力)、MQA (多查询注意力)
- **位置编码**: 支持 ALiBI (Attention with Linear Biases) 位置偏置
