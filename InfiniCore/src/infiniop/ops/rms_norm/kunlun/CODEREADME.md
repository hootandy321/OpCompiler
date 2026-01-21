# RMS Norm Kunlun Backend Implementation Documentation

RMS Norm 操作的昆仑（Kunlun）XPU 加速实现，利用昆仑芯片的集群架构（cluster-based architecture）进行高效的层归一化计算，支持 FP16、BF16 和 FP32 数据类型的混合精度。

## 1. Module Structure

- **`kernel.h`**: 设备端核心计算 kernel 实现，包含 `rmsnormBlock` 模板函数，使用共享内存和块内归约完成 RMS Norm 计算的并行化
- **`rms_norm_kunlun.h`**: 昆仑后端的描述符声明，通过 `DESCRIPTOR(kunlun)` 宏注册后端接口
- **`rms_norm_kunlun.xpu`**: 主机端实现，包含描述符创建、kernel 启动逻辑 (`launchKernel`) 和计算调度 (`calculate`)

## 2. Core Classes

### `rmsnormBlock` (Device Kernel Function)
- **Location**: `kernel.h`
- **Primary Function**: 在单个 cluster 内执行 RMS Norm 计算的设备端函数，通过协作线程并行完成平方和归约、RMS 计算和加权归一化
- **Template Parameters**:
  - `BLOCK_SIZE`: cluster 内的核心数量（默认 64）
  - `Tcompute`: 计算精度类型（通常为 `float`）
  - `Tdata`: 输入/输出数据类型（`half`/`bfloat16_t`/`float`）
  - `Tweight`: 权重数据类型（`half`/`bfloat16_t`/`float`）
- **Key Parameters**:
  - `__shared_ptr__ Tdata *y`: 输出张量的共享内存指针
  - `__shared_ptr__ const Tdata *x`: 输入张量的共享内存指针
  - `__shared_ptr__ const Tweight *w`: 权重参数的共享内存指针
  - `size_t dim`: 特征维度大小
  - `float epsilon`: 数值稳定性小量
- **Core Methods**:
  - `reduce_op::sumSquared<BLOCK_SIZE>()`: 第 18 行，调用昆仑专用的 reduce 操作计算 Σx²，时间复杂度 O(dim/BLOCK_SIZE + log(BLOCK_SIZE))
  - `rsqrt(ss / Tcompute(dim) + epsilon)`: 第 22 行，由 core 0 执行 RMS = sqrt(Σx²/dim + ε) 的倒数计算
  - `sync_cluster()`: 第 24、32 行，集群级屏障同步，确保所有核心在 RMS 计算完成后再进行后续乘法
  - Element-wise 循环（第 27-31 行）：每个核心按步长 `BLOCK_SIZE` 并行计算 `y[i] = x[i] * w[i] * rms`
- **Lifecycle**: 作为 `__global__` kernel 的内部子程序，由 host 端通过 cluster 并行调用，无独立生命周期

### `rmsnormKernel` (Global Kernel Function)
- **Location**: `rms_norm_kunlun.xpu:10-38`
- **Primary Function**: 全局 kernel 函数，负责将数据从全局内存搬运到共享内存（SM），调用 `rmsnormBlock` 计算，再将结果写回全局内存
- **Template Parameters**: 与 `rmsnormBlock` 相同
- **Key Variables**:
  - `__shared__ Tdata x_sm[SM_SIZE / sizeof(Tdata)]`: 输入数据的共享内存缓冲区，大小由 `SM_SIZE` 决定（通常 48KB）
  - `__shared__ Tweight w_sm[SM_SIZE / sizeof(Tweight)]`: 权重参数的共享内存缓冲区（全局只读，所有 cluster 共享）
  - `__shared__ Tdata y_sm[SM_SIZE / sizeof(Tdata)]`: 输出数据的共享内存缓冲区
- **Core Operations**:
  - `GM2SM_ASYNC`（第 26-27 行）：由 core 0 执行异步内存拷贝，将当前 cluster 对应的行数据和权重从全局内存加载到 SM
  - `rmsnormBlock<BLOCK_SIZE>`（第 32 行）：在共享内存中执行 RMS Norm 计算
  - `SM2GM_ASYNC`（第 35 行）：由 core 0 异步写回结果到全局内存
  - `sync_cluster()`（第 29、37 行）：屏障同步，确保内存拷贝完成后再进行计算
- **Launch Configuration**:
  - Grid: `<<<nhead_, BLOCK_SIZE, stream>>>`，其中 `nhead_` 为 attention head 数量，每个 cluster 处理一个 head
  - Block: `BLOCK_SIZE` 核心（默认 64），协作处理单个 head 的 `dim` 维度

### `Descriptor::Opaque`
- **Location**: `rms_norm_kunlun.xpu:42-44`
- **Primary Function**: 不透明指针结构，封装昆仑设备句柄的内部实现（Pimpl 模式）
- **Key Members**:
  - `std::shared_ptr<device::kunlun::Handle::Internal> internal`: 昆仑设备句柄的智能指针，管理 XPU 上下文和资源
- **Lifecycle**: 在 `Descriptor::create` 中构造，在 `Descriptor::~Descriptor` 中析构，采用 RAII 管理设备资源

### `Descriptor` (Host API)
- **Location**: `rms_norm_kunlun.xpu:40-147`
- **Primary Function**: 昆仑后端的 RMS Norm 操作描述符，继承自基类 `op::rms_norm::Descriptor`，负责 kernel 编译、参数验证和执行调度
- **Inherited Members**:
  - `_info`: `RMSNormInfo` 实例，存储张量形状、步长、数据类型和 epsilon 参数
  - `_workspace_size`: 所需工作空间大小（本实现中为 0）
  - `device`, `device_id`: 设备类型和 ID
- **Core Methods**:
  - `create(handle, desc_ptr, y_desc, x_desc, w_desc, epsilon)`: 第 50-68 行，工厂函数，验证张量描述符一致性（通过 `RMSNormInfo::create`），构造 Descriptor 对象并初始化 `Opaque` 成员
  - `~Descriptor()`: 第 46-48 行，析构函数，释放 `Opaque` 指针
  - `calculate(workspace, workspace_size, y, x, w, stream)`: 第 119-145 行，执行函数，检查工作空间大小后调用 `launchKernel<64>` 启动计算，支持 3D 张量 [batch, nhead, dim] 或 2D 张量 [batch, dim]
- **Error Handling**:
  - `INFINI_STATUS_BAD_TENSOR_DTYPE`: 第 111 行，不支持的数据类型组合
  - `INFINI_STATUS_INSUFFICIENT_WORKSPACE`: 第 125 行，工作空间不足（虽当前实现不使用工作空间）

### `launchKernel<BLOCK_SIZE>` (Template Function)
- **Location**: `rms_norm_kunlun.xpu:70-117`
- **Primary Function**: 模板化的 kernel 启动函数，根据数据类型分发不同的 kernel 实例，处理 batch 维度的循环调度
- **Template Parameters**:
  - `BLOCK_SIZE`: 编译时常量，指定 cluster 内的核心数（默认 64）
- **Key Parameters**:
  - `batch_size, nhead, dim`: 张量形状维度
  - `y, x, w`: 输入/输出/权重指针
  - `stride_y_batch, stride_y_nhead, stride_x_batch, stride_x_nhead`: 各维度的步长（字节偏移量）
  - `atype, wtype`: 数据类型枚举（`INFINI_DTYPE_F16/BF16/F32`）
  - `epsilon`: RMS Norm 的数值稳定性参数
  - `stream`: 昆仑 XPU 流
- **Type Dispatch Logic**: 第 96-112 行，使用 `#define LAUNCH_KERNEL` 宏生成 7 种类型组合的 kernel 实例：
  - FP16+FP16, FP16+BF16, FP16+F32
  - BF16+BF16, BF16+F16, BF16+F32
  - F32+F32
  所有组合均使用 `float` 作为计算精度 `Tcompute`
- **Batch Loop**: 第 88-94 行，外层循环遍历 batch 维度，每次迭代为一个 batch 的所有 head 启动 `nhead_` 个 cluster

## 3. API Interface

```cpp
// Factory function for creating Kunlun RMS Norm descriptor
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,                          // 昆仑设备句柄
    Descriptor **desc_ptr,                            // 输出：构造的描述符指针
    infiniopTensorDescriptor_t y_desc,                // 输出张量描述符
    infiniopTensorDescriptor_t x_desc,                // 输入张量描述符
    infiniopTensorDescriptor_t w_desc,                // 权重张量描述符
    float epsilon);                                   // RMS Norm epsilon 参数
// 返回：INFINI_STATUS_SUCCESS 或错误码

// Calculation function
infiniStatus_t Descriptor::calculate(
    void *workspace,                                  // 工作空间（当前未使用）
    size_t workspace_size,                            // 工作空间大小（必须 >= _workspace_size）
    void *y,                                          // 输出数据指针 [batch, nhead, dim] 或 [batch, dim]
    const void *x,                                    // 输入数据指针
    const void *w,                                    // 权重指针 [dim]
    void *stream) const;                              // 昆仑 XPU 流

// Global kernel function (device-side)
template <unsigned int BLOCK_SIZE, typename Tcompute, typename Tdata, typename Tweight>
__global__ void rmsnormKernel(
    Tdata *y,                  // 输出张量
    int32_t stride_y,          // nhead 维度的步长（元素数）
    const Tdata *x,            // 输入张量
    int32_t stride_x,          // nhead 维度的步长
    const Tweight *w,          // 权重参数（全局只读）
    uint32_t dim,              // 特征维度大小
    float epsilon);            // RMS Norm epsilon

// Device-side block function
template <unsigned int BLOCK_SIZE, typename Tcompute, typename Tdata, typename Tweight>
__device__ void rmsnormBlock(
    __shared_ptr__ Tdata *y,   // 共享内存中的输出指针
    __shared_ptr__ const Tdata *x,     // 共享内存中的输入指针
    __shared_ptr__ const Tweight *w,   // 共享内存中的权重指针
    size_t dim,                // 特征维度大小
    float epsilon);            // RMS Norm epsilon
```

## 4. Usage Example

```cpp
// Example: Using Kunlun RMS Norm Backend
#include "infiniop/handle.h"
#include "infiniop/rms_norm.h"

// 1. Create Kunlun device handle
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_KUNLUN, 0);

// 2. Create tensor descriptors
int64_t shape[] = {batch_size, num_heads, hidden_dim};
int64_t x_strides[] = {num_heads * hidden_dim, hidden_dim, 1};
int64_t w_strides[] = {hidden_dim, 1};  // 1D weight tensor

infiniopTensorDescriptor_t x_desc, w_desc, y_desc;
infiniopCreateTensorDescriptor(&x_desc, INFINI_DTYPE_F16, 3, shape, x_strides);
infiniopCreateTensorDescriptor(&w_desc, INFINI_DTYPE_F32, 1, shape + 2, w_strides);
infiniopCreateTensorDescriptor(&y_desc, INFINI_DTYPE_F16, 3, shape, x_strides);

// 3. Create RMS Norm descriptor
op::rms_norm::kunlun::Descriptor *rms_desc;
auto status = op::rms_norm::kunlun::Descriptor::create(
    handle, &rms_desc, y_desc, x_desc, w_desc, 1e-5f);

// 4. Allocate device memory
half *d_x, *d_y;
float *d_w;
kunlunMalloc(&d_x, batch_size * num_heads * hidden_dim * sizeof(half));
kunlunMalloc(&d_y, batch_size * num_heads * hidden_dim * sizeof(half));
kunlunMalloc(&d_w, hidden_dim * sizeof(float));

// Copy data to device (host-to-device)
kunlunMemcpy(d_x, h_x, size, KUNLUN_MEMCPY_HOST_TO_DEVICE);
kunlunMemcpy(d_w, h_w, hidden_dim * sizeof(float), KUNLUN_MEMCPY_HOST_TO_DEVICE);

// 5. Create stream
kunlunStream_t stream;
kunlunStreamCreate(&stream, 0);

// 6. Execute RMS Norm calculation
status = rms_desc->calculate(nullptr, 0, d_y, d_x, d_w, stream);

// 7. Copy result back to host
kunlunMemcpy(h_y, d_y, size, KUNLUN_MEMCPY_DEVICE_TO_HOST);

// 8. Cleanup
kunlunStreamDestroy(stream);
kunlunFree(d_x);
kunlunFree(d_y);
kunlunFree(d_w);
delete rms_desc;
infiniopDestroyHandle(handle);
```

## 5. Implementation Details

- **Memory Hierarchy Strategy**:
  - **Global Memory (GM)**: 存储完整的输入/输出张量和权重参数，容量大但延迟高
  - **Shared Memory (SM)**: 每个 cluster 独立的 48KB 片上存储，用于缓存当前处理的行数据和权重，延迟低且支持异步传输
  - **Async Transfer**: 使用 `GM2SM_ASYNC`/`SM2GM_ASYNC` 隐藏内存延迟，允许计算与数据搬运重叠（虽然当前实现在同步点后未充分利用重叠）

- **Cluster-based Parallelism**:
  - **Grid Dimension**: `nhead_` 个 cluster，每个 cluster 独立处理一个 attention head 的所有维度，适合 Transformer 的多头并行
  - **Block Dimension**: `BLOCK_SIZE`（64）个核心协作处理单个 head 的 `dim` 维度，通过步长循环实现负载均衡
  - **Synchronization**: `sync_cluster()` 提供 cluster 内的屏障，确保所有核心在 RMS 计算完成后再执行归一化乘法

- **Reduce Algorithm**:
  - 调用 `op::common_kunlun::reduce_op::sumSquared<BLOCK_SIZE>()` 实现高效的并行归约
  - 预期采用树形归约（tree reduction）模式，时间复杂度 O(log(BLOCK_SIZE))，而非线性 O(BLOCK_SIZE)
  - 每个 core 先计算局部平方和，再通过跨核通信累加到 core 0

- **Mixed-Precision Support**:
  - **Storage Type**: 支持 FP16、BF16 和 FP32 的任意组合（输入/输出类型 `atype` 和权重类型 `wtype` 可不同）
  - **Compute Type**: 所有组合均使用 `float` 作为 `Tcompute`，确保 `rsqrt` 和除法的数值精度
  - **Type Safety**: 编译期模板实例化避免运行时分支，7 种有效组合通过 `#define LAUNCH_KERNEL` 宏展开生成专门化代码

- **Numerical Stability**:
  - **Epsilon**: 在 `rsqrt` 前添加 `epsilon`（通常 1e-5）防止除零
  - **Order of Operations**: 先求和再归一化（Σx²/dim + ε），最后计算倒数，符合 RMS Norm 的数学定义

- **Stride Handling**:
  - 支持非连续内存布局，通过 `stride_x`/`stride_y` 参数计算 batch 和 nhead 维度的偏移
  - 外层 batch 循环（第 88 行）每次迭代处理一个 batch 的所有 head，而非将 batch 作为 grid 维度，可能影响 batch 级并行效率

- **Performance Considerations**:
  - **Block Size**: 固定使用 64，与昆仑 XPU 的 cluster 核心数匹配，最大化硬件利用率
  - **Shared Memory Utilization**: 每个数据类型占用 `SM_SIZE / sizeof(T)`，FP16 时可缓存 48KB/2B = 24K 元素，适用于大多数 hidden_dim（通常 4K-8K）
  - **Data Reuse**: 权重 `w` 在每个 cluster 中重复加载（第 27 行），可能优化为预加载到常量缓存或共享权重 SM（需硬件支持）
  - **Batch Loop**: 外层 batch 循环在主机端串行执行，对于大 batch 可改为 grid 维度 `<<<batch_size * nhead_, BLOCK_SIZE>>>` 以提升并行度

- **Dependencies**:
  - `device::kunlun::Handle`: 昆仑设备句柄，管理 XPU 上下文
  - `device::kunlun::kernel`: 提供 `core_id()`, `cluster_id()`, `sync_cluster()` 等 kernel API
  - `op::common_kunlun::reduce_op::sumSquared`: 昆仑专用的归约操作
  - `RMSNormInfo`: 基类提供张量形状验证和步长计算

- **Design Patterns**:
  - **Pimpl Idiom**: `Descriptor::Opaque` 隐藏设备句柄的实现细节
  - **CRTP (Curiously Recurring Template Pattern)**: 通过 `DESCRIPTOR(kunlun)` 宏将 `op::rms_norm::kunlun::Descriptor` 注册到后端分发系统
  - **Template Metaprogramming**: 编译期类型分发避免运行时开销，7 种数据类型组合生成独立 kernel 实例
  - **RAII**: `std::shared_ptr` 管理设备句柄生命周期
