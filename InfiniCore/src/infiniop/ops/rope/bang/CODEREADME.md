# RoPE BANG 后端实现文档

## 1. 目录概述

本目录实现了 **RoPE (Rotary Position Embedding)** 操作的 **CAMBRICON BANG** 硬件后端，用于寒武纪(MLU)系列 AI 加速卡。RoPE 是大语言模型(LLM)中用于编码位置信息的核心技术。

### 文件清单

- **`rope_bang.h`** - 头文件，定义 BANG 后端的 Descriptor 接口
- **`rope_bang.mlu`** - 主实现文件，包含描述符创建、类型分发和 kernel 启动逻辑
- **`rope_bang_kernel.mlu`** - 核心 kernel 实现，包含 RoPE 计算的设备端代码

## 2. 核心功能描述

RoPE 通过旋转变换将位置信息注入到注意力机制的 Query 和 Key 向量中，支持两种主要的算法变体：

### 2.1 GPT-J 风格 (INFINIOP_ROPE_ALGO_GPT_J)
- **数据布局**: 交错排列 `(x0, x1, x2, x3, ...)`
- **旋转配对**: 相邻元素配对 `(x0, x1), (x2, x3), ...`
- **计算模式**: 偶数位置作为实部，奇数位置作为虚部进行 2D 旋转

### 2.2 GPT-NeoX 风格 (默认)
- **数据布局**: 连续排列 `(x0...xd/2-1, xd/2...xd-1)`
- **旋转配对**: 前半段与后半段配对
- **计算模式**: 前半段作为实部，后半段作为虚部进行 2D 旋转

## 3. 文件详细分析

### 3.1 `rope_bang.h`

```cpp
#ifndef __INFINIOP_ROPE_BANG_H__
#define __INFINIOP_ROPE_BANG_H__

#include "../rope.h"

DESCRIPTOR(bang)

#endif // __INFINIOP_ROPE_BANG_H__
```

**解析**:
- 包含父目录的通用 RoPE 接口 `../rope.h`
- 使用宏 `DESCRIPTOR(bang)` 展开为 `op::rope::bang::Descriptor` 类定义
- 该宏定义了标准的描述符接口（虚析构函数、create、calculate 等方法）

### 3.2 `rope_bang.mlu` - 主接口实现

#### 3.2.1 命名空间与析构函数

```cpp
namespace op::rope::bang {

Descriptor::~Descriptor() = default;
```

#### 3.2.2 Descriptor::create - 描述符创建 (行 9-33)

```cpp
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t pos_desc,
    infiniopTensorDescriptor_t sin_desc,
    infiniopTensorDescriptor_t cos_desc,
    infiniopRoPEAlgo_t algo) {
```

**功能**: 创建 RoPE 操作的描述符对象，验证输入参数并初始化内部状态。

**执行流程**:
1. **类型转换** (行 19): 将通用句柄转换为 BANG 设备句柄
   ```cpp
   auto handle = reinterpret_cast<device::bang::Handle *>(handle_);
   ```

2. **信息提取** (行 21-22): 调用 `RoPEInfo::createRoPEInfo` 提取张量形状、步长、数据类型等元信息
   ```cpp
   auto info = RoPEInfo::createRoPEInfo(y_desc, x_desc, pos_desc, sin_desc, cos_desc, algo);
   CHECK_RESULT(info);
   ```

3. **对象构造** (行 25-30): 创建 Descriptor 实例，存储 RoPE 信息和设备属性
   ```cpp
   *desc_ptr = new Descriptor(
       info.take(),      // RoPEInfo 对象（独占所有权）
       0,                // 保留字段
       nullptr,          // 保留字段
       handle->device,   // 设备类型
       handle->device_id // 设备 ID
   );
   ```

#### 3.2.3 calculateRoPE - 模板化计算函数 (行 35-67)

```cpp
template <typename Tdata, typename Tindex>
infiniStatus_t calculateRoPE(const RoPEInfo &info,
                             Tdata *y,
                             const Tdata *x,
                             const Tindex *pos_ids,
                             const Tdata *sin_table,
                             const Tdata *cos_table,
                             cnrtQueue_t queue) {
```

**功能**: 配置 kernel 启动参数并调用 RoPE kernel。

**关键步骤**:

1. **维度提取** (行 43-45):
   ```cpp
   auto dimx = uint32_t(info.seqlen);   // 序列长度
   auto dimy = uint32_t(info.nhead);    // 注意力头数
   auto table_dim = uint32_t(info.table_dim); // 旋转维度表大小
   ```

2. **Kernel 配置** (行 47-54):
   ```cpp
   cnrtDim3_t k_dim;
   cnrtFunctionType_t k_type;

   k_dim.x = 4;  // 使用 4 个计算集群(Cluster)
   k_dim.y = 1;
   k_dim.z = 1;
   k_type = CNRT_FUNC_TYPE_UNION1;  // Union1 类型任务分发
   ```

3. **Kernel 启动** (行 57-62):
   ```cpp
   ropeKernel<<<k_dim, k_type, queue>>>(
       y, x, pos_ids, sin_table, cos_table,
       dimx, dimy, table_dim,
       info.y_stride_seqlen, info.y_stride_nhead,
       info.x_stride_seqlen, info.x_stride_nhead,
       info.algo);
   ```

4. **同步等待** (行 64):
   ```cpp
   cnrtQueueSync(queue);
   ```

#### 3.2.4 宏定义 - 类型分发器 (行 69-98)

```cpp
#define CALCULATE_ROPE(TDATA, TINDEX)       \
    calculateRoPE(_info,                    \
                  (TDATA *)y,               \
                  (const TDATA *)x,         \
                  (const TINDEX *)pos_ids,  \
                  (const TDATA *)sin_table, \
                  (const TDATA *)cos_table, \
                  (cnrtQueue_t)stream)

#define ROPE_TYPE(TDATA)                        \
    switch (_info.pos_type) {                   \
    case INFINI_DTYPE_U8:                       \
        return CALCULATE_ROPE(TDATA, uint8_t);  \
    case INFINI_DTYPE_U16:                      \
        return CALCULATE_ROPE(TDATA, uint16_t); \
    // ... 所有整数类型
    }
```

**功能**: 通过两层宏实现数据类型和位置 ID 类型的双重分发，避免代码重复。

**类型映射**:
- **数据类型 (TDATA)**: `half` (FP16), `bfloat16_t` (BF16), `float` (FP32)
- **位置类型 (TINDEX)**: `int8_t` ~ `int64_t`, `uint8_t` ~ `uint64_t`

#### 3.2.5 Descriptor::calculate - 公共接口 (行 100-122)

```cpp
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *pos_ids,
    const void *sin_table,
    const void *cos_table,
    void *stream) const {
```

**功能**: 类型安全的入口函数，根据运行时数据类型分发到对应的模板实例。

**分发逻辑** (行 110-119):
```cpp
switch (_info.data_type) {
case INFINI_DTYPE_F16:
    ROPE_TYPE(half);
case INFINI_DTYPE_BF16:
    ROPE_TYPE(bfloat16_t);
case INFINI_DTYPE_F32:
    ROPE_TYPE(float);
default:
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}
```

### 3.3 `rope_bang_kernel.mlu` - 核心计算 Kernel

#### 3.3.1 全局内存声明 (行 4)

```cpp
__nram__ char nram_buffer[NRAM_MAX_SIZE];
```

**说明**:
- `__nram__`: CAMBRICON MLU 的近端 RAM（类似 GPU 的 shared memory），高带宽低延迟
- `NRAM_MAX_SIZE`: 硬件定义的最大 NRAM 大小（通常为 256KB 或更大）
- 使用 `char` 类型以字节为单位进行内存管理

#### 3.3.2 calculateRope - 单次旋转计算 (行 6-56)

```cpp
template <typename Tdata>
__mlu_device__ void calculateRope(
    Tdata *out, const Tdata *in,
    const Tdata *sin_table, const Tdata *cos_table,
    Tdata *sin_cache, Tdata *cos_cache,
    Tdata *x1sin, Tdata *x0cos, Tdata *x0sin, Tdata *x1cos,
    Tdata *input_0, Tdata *input_1, Tdata *input_cache,
    int theta_index, int out_index, int in_index,
    int chunk_size, int half_chunk_size, int data_segsize,
    int src_load_stride, int dst_load_stride, int src_write_stride, int dst_write_stride,
    bool is_gpt_j_style) {
```

**功能**: 执行单个 chunk 的 RoPE 计算，包含数据加载、旋转运算和结果写回。

**内存布局图** (NRAM 分配):
```
+------------------+
| sin_cache        |  half_chunk_size 个元素
+------------------+
| cos_cache        |  half_chunk_size 个元素
+------------------+
| x1sin            |  half_chunk_size 个元素
+------------------+
| x0cos            |  half_chunk_size 个元素
+------------------+
| x0sin            |  half_chunk_size 个元素
+------------------+
| x1cos            |  half_chunk_size 个元素
+------------------+
| input_0          |  half_chunk_size 个元素（实部）
+------------------+
| input_1          |  half_chunk_size 个元素（虚部）
+------------------+
| input_cache      |  chunk_size 个元素（临时存储）
+------------------+
```

**执行步骤**:

1. **加载 sin/cos 表** (行 19-20):
   ```cpp
   __memcpy(sin_cache, sin_table + theta_index, half_chunk_size * sizeof(Tdata), GDRAM2NRAM);
   __memcpy(cos_cache, cos_table + theta_index, half_chunk_size * sizeof(Tdata), GDRAM2NRAM);
   ```

2. **加载输入数据** (行 23):
   ```cpp
   __memcpy(input_cache, in + in_index, chunk_size * sizeof(Tdata), GDRAM2NRAM);
   ```

3. **数据重排与分割** (行 25-34):

   **GPT-J 风格** (行 25-29):
   ```cpp
   if (is_gpt_j_style) {
       // 从交错布局分离出偶数和奇数位置
       __memcpy(input_0, input_cache, data_segsize, NRAM2NRAM, dst_load_stride, src_load_stride, half_chunk_size - 1);
       __memcpy(input_1, input_cache + 1, data_segsize, NRAM2NRAM, dst_load_stride, src_load_stride, half_chunk_size - 1);
   }
   ```
   - `input_0`: 偶数位置元素 `(x0, x2, x4, ...)`
   - `input_1`: 奇数位置元素 `(x1, x3, x5, ...)`
   - 使用 stride 内存访问实现解交错

   **GPT-NeoX 风格** (行 30-33):
   ```cpp
   else {
       __memcpy(input_0, input_cache, half_chunk_size * sizeof(Tdata), NRAM2NRAM);
       __memcpy(input_1, input_cache + half_chunk_size, half_chunk_size * sizeof(Tdata), NRAM2NRAM);
   }
   ```
   - `input_0`: 前半段 `(x0, x1, ..., xd/2-1)`
   - `input_1`: 后半段 `(xd/2, xd/2+1, ..., xd-1)`
   - 直接内存拷贝，无 stride

4. **旋转计算** (行 37-42):
   ```cpp
   __bang_mul(x0cos, input_0, cos_cache, half_chunk_size);  // x0 * cos(θ)
   __bang_mul(x1sin, input_1, sin_cache, half_chunk_size);  // x1 * sin(θ)
   __bang_mul(x0sin, input_0, sin_cache, half_chunk_size);  // x0 * sin(θ)
   __bang_mul(x1cos, input_1, cos_cache, half_chunk_size);  // x1 * cos(θ)
   __bang_sub(input_0, x0cos, x1sin, half_chunk_size);      // x0' = x0*cos - x1*sin
   __bang_add(input_1, x0sin, x1cos, half_chunk_size);      // x1' = x0*sin + x1*cos
   ```

   **数学公式**:
   ```
   [x0']     [cos(θ)  -sin(θ)] [x0]
   [x1']  =  [sin(θ)   cos(θ)] [x1]
   ```

5. **数据重组** (行 44-52):

   **GPT-J 风格** (行 44-47):
   ```cpp
   __memcpy(input_cache, input_0, data_segsize, NRAM2NRAM, dst_write_stride, src_write_stride, half_chunk_size - 1);
   __memcpy(input_cache + 1, input_1, data_segsize, NRAM2NRAM, dst_write_stride, src_write_stride, half_chunk_size - 1);
   ```
   - 将计算结果交错写回 `(x0', x1', x2', x3', ...)`

   **GPT-NeoX 风格** (行 48-51):
   ```cpp
   __memcpy(input_cache, input_0, half_chunk_size * sizeof(Tdata), NRAM2NRAM);
   __memcpy(input_cache + half_chunk_size, input_1, half_chunk_size * sizeof(Tdata), NRAM2NRAM);
   ```

6. **写回结果** (行 55):
   ```cpp
   __memcpy(out + out_index, input_cache, chunk_size * sizeof(Tdata), NRAM2GDRAM);
   ```

#### 3.3.3 ropeKernel - 全局 kernel 函数 (行 58-188)

```cpp
template <typename Tdata, typename Tindex>
__mlu_global__ void ropeKernel(
    Tdata *y,
    const Tdata *x,
    const Tindex *pos_ids,
    const Tdata *sin_table,
    const Tdata *cos_table,
    uint32_t seqlen,
    uint32_t nhead,
    uint32_t table_dim,
    ptrdiff_t y_stride_seqlen,
    ptrdiff_t y_stride_nhead,
    ptrdiff_t x_stride_seqlen,
    ptrdiff_t x_stride_nhead,
    infiniopRoPEAlgo_t algo) {
```

**功能**: 在 MLU 设备端执行的 RoPE 主函数，负责任务分发、内存管理和循环调度。

**关键变量分析**:

1. **算法标识** (行 74):
   ```cpp
   const bool is_gpt_j_style = (algo == INFINIOP_ROPE_ALGO_GPT_J);
   ```

2. **NRAM 容量计算** (行 76-78):
   ```cpp
   const size_t nram_usable = NRAM_MAX_SIZE - (ALIGN_SIZE * 9);
   const size_t max_chunk_elements = nram_usable / (9 * sizeof(Tdata));
   ```
   - `ALIGN_SIZE`: 对齐边界（通常为 512B 或 1KB）
   - 保留 9 个对齐边界用于 9 个缓冲区
   - `max_chunk_elements`: NRAM 可容纳的最大元素数量

3. **位置 ID 缓存决策** (行 81):
   ```cpp
   const bool use_pos_ids_buffer = (seqlen * sizeof(Tindex) <= (nram_usable / 2));
   ```
   - 如果位置 ID 表能装入一半 NRAM，则缓存到 NRAM 以减少 GDRAM 访问
   - 否则直接从 GDRAM 读取

4. **Chunk 大小计算** (行 83-88):
   ```cpp
   int half_chunk_size;
   if (is_gpt_j_style) {
       half_chunk_size = std::min((int)(max_chunk_elements / 2), (int)table_dim);
   } else {
       half_chunk_size = std::min((int)(max_chunk_elements / 2), (int)table_dim);
   }
   ```
   - `half_chunk_size`: 每次处理的旋转对数量
   - 受 NRAM 容量和 `table_dim` 限制

5. **Stride 配置** (行 90-106):

   **GPT-J 模式**:
   ```cpp
   data_segsize = sizeof(Tdata);           // 每次处理 1 个元素
   src_load_stride = 2 * sizeof(Tdata);    // 读取间隔为 2（跳过相邻元素）
   dst_load_stride = 1 * sizeof(Tdata);    // 写入连续存储
   src_write_stride = 1 * sizeof(Tdata);   // 读取连续存储
   dst_write_stride = 2 * sizeof(Tdata);   // 写入间隔为 2（交错写回）
   ```

   **GPT-NeoX 模式**:
   ```cpp
   data_segsize = half_chunk_size * sizeof(Tdata);  // 批量处理
   src_load_stride = 1 * sizeof(Tdata);
   dst_load_stride = 1 * sizeof(Tdata);
   src_write_stride = 1 * sizeof(Tdata);
   dst_write_stride = 1 * sizeof(Tdata);
   ```

6. **任务分发** (行 108-113):
   ```cpp
   const int batch_volume = seqlen * nhead;
   const int remaining_tasks = batch_volume % taskDim;  // taskDim = k_dim.x = 4
   const int base_tasks_per_core = batch_volume / taskDim;
   const int actual_tasks = base_tasks_per_core + (taskId < remaining_tasks ? 1 : 0);
   const int task_start_idx = (taskId < remaining_tasks
       ? taskId * base_tasks_per_core + taskId
       : taskId * base_tasks_per_core + remaining_tasks);
   ```

   **分发策略**: 负载均衡的块分发
   - 前 `remaining_tasks` 个核心多处理 1 个任务
   - 确保所有核心工作负载均衡

   **示例** (`batch_volume = 10`, `taskDim = 4`):
   ```
   Core 0: tasks [0, 1, 2]     (3 tasks)
   Core 1: tasks [3, 4]        (2 tasks)
   Core 2: tasks [5, 6]        (2 tasks)
   Core 3: tasks [7, 8, 9]     (3 tasks)
   ```

7. **NRAM 对齐与初始化** (行 115-124):
   ```cpp
   char *aligned_nram = (char *)(((size_t)nram_buffer + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1));

   Tindex *srcP = nullptr;
   if (use_pos_ids_buffer) {
       srcP = (Tindex *)aligned_nram;
       __memcpy(srcP, pos_ids, seqlen * sizeof(Tindex), GDRAM2NRAM);
       aligned_nram = (char *)(((size_t)srcP + seqlen * sizeof(Tindex) + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1));
   }
   ```
   - 对齐到 `ALIGN_SIZE` 边界以优化内存访问性能
   - 如果缓存位置 ID，则分配空间并更新指针

8. **主处理循环** (行 137-187):

   **外层循环** (行 138-141): 遍历分配给当前核心的所有任务
   ```cpp
   for (int i = task_start_idx; i < task_start_idx + actual_tasks; i++) {
       int seq_idx = i / nhead;   // 序列索引
       int head_idx = i % nhead;  // 注意力头索引
   ```

   **偏移量计算** (行 142-146):
   ```cpp
   int out_offset = seq_idx * y_stride_seqlen + head_idx * y_stride_nhead;
   int in_offset = seq_idx * x_stride_seqlen + head_idx * x_stride_nhead;

   Tindex pos_idx = use_pos_ids_buffer ? srcP[seq_idx] : pos_ids[seq_idx];
   int rot_offset = pos_idx * table_dim;
   ```

   **内层循环** (行 149-186): 分块处理旋转维度
   ```cpp
   int processed = 0;
   while (processed < table_dim) {
       int current_half_chunk = std::min<uint32_t>(half_chunk_size, table_dim - processed);
       int current_chunk_size = 2 * current_half_chunk;
       int theta_offset = rot_offset + processed;

       int dst_offset, src_offset;
       if (is_gpt_j_style) {
           dst_offset = out_offset + processed * 2;
           src_offset = in_offset + processed * 2;
       } else {
           dst_offset = out_offset + processed;
           src_offset = in_offset + processed;
       }
   ```

   **动态 NRAM 分配** (行 163-173):
   ```cpp
   char *chunk_base = aligned_nram;
   sin_cache = (Tdata *)chunk_base;
   cos_cache = sin_cache + current_half_chunk;
   x1sin = cos_cache + current_half_chunk;
   x0cos = x1sin + current_half_chunk;
   x0sin = x0cos + current_half_chunk;
   x1cos = x0sin + current_half_chunk;
   input_0 = x1cos + current_half_chunk;
   input_1 = input_0 + current_half_chunk;
   input_cache = input_1 + current_half_chunk;
   ```
   - 根据当前 chunk 大小动态分配 9 个缓冲区
   - 指针算术确保正确的内存布局

   **调用计算函数** (行 175-183):
   ```cpp
   calculateRope<Tdata>(
       y, x, sin_table, cos_table,
       sin_cache, cos_cache, x1sin, x0cos, x0sin, x1cos,
       input_0, input_1, input_cache,
       theta_offset, dst_offset, src_offset,
       current_chunk_size, current_half_chunk,
       data_segsize,
       src_load_stride, dst_load_stride, src_write_stride, dst_write_stride,
       is_gpt_j_style);
   ```

   **进度更新** (行 185):
   ```cpp
   processed += current_half_chunk;
   ```

## 4. 数据流与算法流程

### 4.1 调用链路图

```
用户代码
  └─> Descriptor::calculate(void *y, void *x, ...)
       ├─> ROPE_TYPE(half) [宏分发]
       │    └─> calculateRoPE<half, int32_t>(RoPEInfo, ...)
       │         └─> ropeKernel<half, int32_t><<<dim, type, queue>>>(...)
       │              └─> calculateRope<half>(...) [设备端]
       │                   ├─> __memcpy(GDRAM2NRAM): 加载 sin/cos 表
       │                   ├─> __memcpy(GDRAM2NRAM): 加载输入数据
       │                   ├─> __memcpy(NRAM2NRAM): 数据重排（GPT-J）
       │                   ├─> __bang_mul/__bang_sub/__bang_add: 旋转计算
       │                   ├─> __memcpy(NRAM2NRAM): 数据重组（GPT-J）
       │                   └─> __memcpy(NRAM2GDRAM): 写回结果
       └─> cnrtQueueSync(queue): 同步等待
```

### 4.2 内存访问模式

**输入布局** (以 GPT-J, `table_dim=8`, `seqlen=4`, `nhead=2` 为例):
```
GDRAM (x):
[seq0_head0] x00 x01 x02 x03 x04 x05 x06 x07
[seq0_head1] x00 x01 x02 x03 x04 x05 x06 x07
[seq1_head0] x00 x01 x02 x03 x04 x05 x06 x07
...
```

**NRAM 处理流程**:
```
1. 加载到 NRAM:
   input_cache: [x00, x01, x02, x03, x04, x05, x06, x07]

2. 分离偶奇:
   input_0: [x00, x02, x04, x06]  (实部)
   input_1: [x01, x03, x05, x07]  (虚部)

3. 旋转计算:
   x0cos = input_0 * cos
   x1sin = input_1 * sin
   x0sin = input_0 * sin
   x1cos = input_1 * cos
   input_0 = x0cos - x1sin
   input_1 = x0sin + x1cos

4. 交错重组:
   input_cache: [x00', x01', x02', x03', x04', x05', x06', x07']

5. 写回 GDRAM:
   [seq0_head0] x00' x01' x02' x03' x04' x05' x06' x07'
```

## 5. 性能优化策略

### 5.1 内存层级优化

1. **NRAM 利用**:
   - 将频繁访问的 sin/cos 表和输入数据缓存到 NRAM
   - 避免 GDRAM（高延迟）的重复访问

2. **分块处理**:
   - 根据 NRAM 容量动态计算 `chunk_size`
   - 大表分多次加载，小表一次性处理

3. **位置 ID 缓存** (行 81, 120-123):
   - 如果 `seqlen * sizeof(Tindex) <= nram_usable / 2`，缓存到 NRAM
   - 否则从 GDRAM 直接读取

### 5.2 并行计算优化

1. **多 Cluster 并行** (行 51, `k_dim.x = 4`):
   - 使用 4 个计算集群并行处理不同的 `(seq, head)` 组合
   - 负载均衡的任务分发算法

2. **向量化指令**:
   - `__bang_mul`: 向量乘法，一次处理多个元素
   - `__bang_add/__bang_sub`: 向量加减法
   - 充分利用 MLU 的 SIMD 宽度（通常 128 或 256 lane）

3. **内存流水线**:
   - NRAM 与 GDRAM 传输与计算重叠（硬件自动调度）
   - 多缓冲区技术（9 个缓冲区）

### 5.3 算法适应性

1. **GPT-J 优化**:
   - Stride 访问模式优化交错布局
   - `data_segsize = sizeof(Tdata)`：单元素步进

2. **GPT-NeoX 优化**:
   - 连续内存访问，带宽利用率高
   - `data_segsize = half_chunk_size * sizeof(Tdata)`：批量处理

### 5.4 对齐优化

```cpp
const size_t nram_usable = NRAM_MAX_SIZE - (ALIGN_SIZE * 9);
char *aligned_nram = (char *)(((size_t)nram_buffer + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1));
```
- 所有缓冲区对齐到 `ALIGN_SIZE` 边界
- 避免 crossing cache line 或内存 bank 冲突

## 6. 类型支持矩阵

### 6.1 数据类型 (Tdata)

| 类型       |枚举值                 | 描述     | 精度  | 用途           |
|-----------|----------------------|----------|-------|---------------|
| `half`    | `INFINI_DTYPE_F16`   | FP16     | 16-bit | LLM 推理（主流）|
| `bfloat16_t` | `INFINI_DTYPE_BF16` | BF16     | 16-bit | 训练场景       |
| `float`   | `INFINI_DTYPE_F32`   | FP32     | 32-bit | 高精度计算     |

### 6.2 位置 ID 类型 (Tindex)

| 类型       |枚举值                 | 描述     | 范围                  |
|-----------|----------------------|----------|----------------------|
| `uint8_t` | `INFINI_DTYPE_U8`    | 无符号 8-bit| 0 ~ 255             |
| `uint16_t`| `INFINI_DTYPE_U16`   | 无符号 16-bit| 0 ~ 65,535         |
| `uint32_t`| `INFINI_DTYPE_U32`   | 无符号 32-bit| 0 ~ 4.29×10^9      |
| `uint64_t`| `INFINI_DTYPE_U64`   | 无符号 64-bit| 0 ~ 1.84×10^19     |
| `int8_t`  | `INFINI_DTYPE_I8`    | 有符号 8-bit| -128 ~ 127         |
| `int16_t` | `INFINI_DTYPE_I16`   | 有符号 16-bit| -32,768 ~ 32,767   |
| `int32_t` | `INFINI_DTYPE_I32`   | 有符号 32-bit| ±2.14×10^9         |
| `int64_t` | `INFINI_DTYPE_I64`   | 有符号 64-bit| ±9.22×10^18        |

### 6.3 张量形状支持

| 张量   | 形状                     | 描述           |
|--------|--------------------------|----------------|
| `x`    | `(batch, seqlen, nhead, table_dim)` | 输入 Query/Key |
| `y`    | `(batch, seqlen, nhead, table_dim)` | 输出旋转后向量 |
| `pos_ids` | `(batch, seqlen)`      | 位置编码索引    |
| `sin_table`| `(max_pos, table_dim/2)`| 正弦表        |
| `cos_table`| `(max_pos, table_dim/2)`| 余弦表        |

## 7. 关键常量与配置

| 常量            | 值/来源        | 描述                        |
|-----------------|---------------|-----------------------------|
| `NRAM_MAX_SIZE` | 硬件定义       | MLU NRAM 容量（如 256KB）    |
| `ALIGN_SIZE`    | 编译时常量      | 内存对齐边界（通常 512B）    |
| `taskDim`       | 4             | 使用的 Cluster 数量          |
| `k_type`        | `CNRT_FUNC_TYPE_UNION1` | 任务分发类型 |

## 8. 错误处理

### 8.1 错误码

| 错误码                           | 触发条件                |
|---------------------------------|------------------------|
| `INFINI_STATUS_BAD_TENSOR_DTYPE`| 不支持的数据类型         |
| `INFINI_STATUS_BAD_PARAM`       | 形状/步长不匹配（来自 RoPEInfo） |

### 8.2 检查机制

1. **编译时类型检查**: 模板实例化确保类型安全
2. **运行时分发**: `switch` 语句覆盖所有支持的类型
3. **RoPEInfo 验证**: 在 `create()` 中通过 `CHECK_RESULT(info)` 捕获错误

## 9. GPT-J vs GPT-NeoX 详细对比

| 特性                | GPT-J                        | GPT-NeoX                    |
|---------------------|------------------------------|-----------------------------|
| **数据布局**         | 交错 `(x0, x1, x2, x3, ...)` | 分段 `(前半, 后半)`         |
| **旋转配对**         | `(x0,x1), (x2,x3), ...`      | `(前半, 后半)`              |
| **内存访问模式**     | Stride 访问（跨步）           | 连续访问                    |
| **data_segsize**    | `sizeof(Tdata)`              | `half_chunk_size * sizeof(Tdata)` |
| **src_load_stride** | `2 * sizeof(Tdata)`          | `1 * sizeof(Tdata)`         |
| **dst_load_stride** | `1 * sizeof(Tdata)`          | `1 * sizeof(Tdata)`         |
| **src_write_stride**| `1 * sizeof(Tdata)`          | `1 * sizeof(Tdata)`         |
| **dst_write_stride**| `2 * sizeof(Tdata)`          | `1 * sizeof(Tdata)`         |
| **带宽利用率**       | 较低（非连续访问）            | 较高（连续访问）             |
| **适用场景**         | 交错内存布局的模型            | 分段内存布局的模型           |

**示例** (`table_dim=4`):

**GPT-J**:
```
输入: [x0, x1, x2, x3]
分离: input_0=[x0,x2], input_1=[x1,x3]
旋转: [x0',x2'] = [x0,x2]⊗[cos0,cos2] - [x1,x3]⊗[sin1,sin3]
     [x1',x3'] = [x0,x2]⊗[sin0,sin2] + [x1,x3]⊗[cos1,cos3]
重组: [x0', x1', x2', x3']
```

**GPT-NeoX**:
```
输入: [x0, x1, x2, x3]
分离: input_0=[x0,x1], input_1=[x2,x3]
旋转: [x0',x1'] = [x0,x1]⊗[cos0,cos1] - [x2,x3]⊗[sin0,sin1]
     [x2',x3'] = [x0,x1]⊗[sin0,sin1] + [x2,x3]⊗[cos0,cos1]
重组: [x0', x1', x2', x3']
```

## 10. 依赖关系总结

### 10.1 外部依赖

1. **上层接口**:
   - `../rope.h`: 通用 RoPE 接口（提供 `DESCRIPTOR` 宏和 `RoPEInfo`）

2. **设备通用代码**:
   - `../../../devices/bang/common_bang.h`: BANG 设备通用定义（Handle、类型）

3. **硬件驱动**:
   - `cnrtQueue_t`, `cnrtDim3_t`, `CNRT_FUNC_TYPE_UNION1`: CNRT 驱动 API
   - `__nram__`, `__mlu_device__`, `__mlu_global__`: MLU 编译器扩展
   - `__memcpy`, `__bang_mul`, `__bang_add`, `__bang_sub`: BANG 指令集

### 10.2 内部依赖

```
rope_bang.h
  └─> ../rope.h
       └─> (上层张量描述符定义)

rope_bang.mlu
  ├─> rope_bang.h (宏展开的 Descriptor 类)
  ├─> rope_bang_kernel.mlu (kernel 函数声明)
  └─> ../../../devices/bang/common_bang.h

rope_bang_kernel.mlu
  ├─> rope_bang.h
  └─> ../../../devices/bang/common_bang.h
```

---

**文档版本**: 1.0
**最后更新**: 2026-01-14
**维护者**: InfiniCore Team
**适用硬件**: CAMBRICON MLU 系列（MLU270, MLU290, MLU370 等）
**编译器**: BANGC (Cambricon C++ Compiler for MLU)
