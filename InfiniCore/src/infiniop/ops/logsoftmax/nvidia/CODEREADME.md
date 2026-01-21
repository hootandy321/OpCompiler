# LogSoftmax NVIDIA CUDA 算子核心实现文档

本模块实现了 LogSoftmax 操作的 NVIDIA GPU CUDA 后端，支持多种精度组合（FP32、FP16、BF16）的混合精度计算，针对 NVIDIA GPU 架构进行了优化，使用 CUDA Block Reduce 实现高效的并行归约操作。

## 1. 模块结构

- **`logsoftmax_nvidia.cuh`**: 头文件，定义 NVIDIA 后端的 Descriptor 类接口，使用宏 `DESCRIPTOR(nvidia)` 实例化
- **`logsoftmax_nvidia.cu`**: 核心实现文件，包含算子描述符的创建、核函数调度逻辑和混合精度支持
- **`../cuda/kernel.cuh`**: CUDA 核函数实现，包含设备端算法逻辑
- **`../logsoftmax.h`**: 通用接口定义，使用宏生成后端特定的 Descriptor 类
- **`../info.h`**: LogSoftmaxInfo 数据结构和张量形状验证逻辑

## 2. 核心类与数据结构

### `Descriptor` 类
- **命名空间**: `op::logsoftmax::nvidia`
- **继承**: `InfiniopDescriptor`（基类定义在 `../../operator.h`）
- **主要功能**: NVIDIA CUDA 后端的 LogSoftmax 算子描述符，负责管理 GPU 上下文、张量信息和核函数调度

#### 关键成员变量
- **`Opaque *_opaque`**: 不透明指针，指向 NVIDIA 设备句柄的内部实现（`std::shared_ptr<device::nvidia::Handle::Internal>`）
- **`LogSoftmaxInfo _info`**: 存储张量元数据（形状、步长、数据类型等）
- **`size_t _workspace_size`**: 工作空间大小（当前实现为 0，不需要额外工作空间）

#### 核心方法

**`create()` - 算子描述符创建**
```cpp
static infiniStatus_t create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc);
```
- **功能**: 创建 LogSoftmax NVIDIA 算子描述符
- **参数**:
  - `handle`: Infini 运行时句柄，包含设备和上下文信息
  - `y_desc`: 输出张量描述符
  - `x_desc`: 输入张量描述符
- **实现逻辑**:
  1. 调用 `LogSoftmaxInfo::create()` 验证张量形状和数据类型
  2. 提取 NVIDIA 设备句柄的内部实现（`internal()`）
  3. 实例化 Descriptor 对象，存储设备类型和设备 ID
- **返回**: `INFINI_STATUS_SUCCESS` 或错误码
- **时间复杂度**: O(1)

**`calculate()` - 执行计算**
```cpp
infiniStatus_t calculate(
    void *workspace, size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const;
```
- **功能**: 启动 CUDA 核函数执行 LogSoftmax 计算
- **参数**:
  - `workspace`: 工作空间指针（当前未使用）
  - `y`: 输出张量设备指针
  - `x`: 输入张量设备指针
  - `stream`: CUDA 流指针
- **实现逻辑**:
  1. 转换流指针为 `cudaStream_t`
  2. 根据 GPU 架构查询最大线程数（`maxThreadsPerBlock()`）
  3. 选择合适的 BLOCK_SIZE 模板参数（4096/1024/512）
  4. 调用 `launchKernel<BLOCK_SIZE>()` 启动核函数
- **支持架构**:
  - CUDA_BLOCK_SIZE_1024 (1024 线程/块)
  - CUDA_BLOCK_SIZE_512 (512 线程/块)
  - CUDA_BLOCK_SIZE_4096 (4096 线程/块，适用于现代 GPU)
- **错误处理**: 如果架构不支持，返回 `INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED`

### `LogSoftmaxInfo` 结构体
- **位置**: `../info.h`
- **主要功能**: 存储和验证 LogSoftmax 操作的张量元数据

#### 关键字段
```cpp
infiniDtype_t x_dtype;       // 输入数据类型
infiniDtype_t y_dtype;       // 输出数据类型
size_t batch_size;            // 批次大小（2D: shape[0], 3D: shape[0]*shape[1]）
size_t probs_size;            // 概率维度大小（2D: shape[1], 3D: shape[2]）
size_t ndim;                  // 张量维度数（2 或 3）
size_t seq_len;               // 序列长度（仅 3D 张量使用）
ptrdiff_t y_stride_b;         // 输出批次步长
ptrdiff_t y_stride_p;         // 输出概率步长
ptrdiff_t x_stride_b;         // 输入批次步长
ptrdiff_t x_stride_p;         // 输入概率步长
ptrdiff_t y_stride_0/1/2;     // 输出原始维度步长
ptrdiff_t x_stride_0/1/2;     // 输入原始维度步长
```

#### 静态工厂方法 `create()`
```cpp
static utils::Result<LogSoftmaxInfo> create(
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc);
```
- **验证逻辑**:
  1. 数据类型检查：支持 FP32、FP16、BF16，允许混合精度
  2. 形状检查：输入输出张量形状必须一致
  3. 维度检查：仅支持 2D 或 3D 张量
  4. 步长计算：处理 2D 和 3D 张量的不同内存布局
- **3D 张量特殊处理**:
  - 将前两维展平为批次维度：`batch_size = shape[0] * shape[1]`
  - 保留原始步长用于 GPU 核函数的正确内存访问

### `Opaque` 内部结构
```cpp
struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};
```
- **功能**: 封装 NVIDIA 设备句柄的内部实现
- **生命周期**: 由 Descriptor 构造时分配，析构时释放

## 3. CUDA 核函数实现

### `logSoftmaxKernel()` - 设备端核心算法
- **位置**: `../cuda/kernel.cuh`
- **函数签名**:
```cpp
template <unsigned int BLOCK_SIZE, typename Tdata_out, typename Tdata_in, typename Tcompute>
__device__ void logSoftmaxKernel(
    Tdata_out *y, const Tdata_in *x,
    size_t batch_size, size_t probs_size, size_t ndim, size_t seq_len,
    ptrdiff_t y_stride_b, ptrdiff_t y_stride_p,
    ptrdiff_t x_stride_b, ptrdiff_t x_stride_p,
    ptrdiff_t y_stride_0, ptrdiff_t y_stride_1,
    ptrdiff_t x_stride_0, ptrdiff_t x_stride_1);
```

#### 算法流程（三阶段归约）

**阶段 1: 寻找最大值（数值稳定性）**
```
max_val = max(x[i]) for i in [0, probs_size)
```
- 每个线程处理 `probs_size / BLOCK_SIZE` 个元素
- 使用 CUB `BlockReduce` 进行块内归约
- 初始化为 `-INFINITY` 保证正确性
- 支持 CUDA 12.9+ 的 `cuda::maximum()` 或旧版 `cub::Max()`
- 结果存储到共享内存 `shared_max_val`，同步所有线程

**阶段 2: 计算指数和**
```
sum_exp = Σ exp(x[i] - max_val) for i in [0, probs_size)
```
- 减去最大值避免数值溢出
- 使用类型推导选择 `expf()`（float）或 `exp()`（double）
- 再次使用 BlockReduce 求和
- 结果存储到共享内存 `shared_sum_exp`，同步所有线程

**阶段 3: 计算 LogSoftmax**
```
log_softmax[i] = x[i] - max_val - log(sum_exp)
```
- 使用 `logf()` 或 `log()` 计算对数
- 并行写入所有元素
- 自动处理输出类型转换（如 float → half）

#### 内存访问模式
- **2D 张量**:
  - `offset = batch_idx * stride_b`
  - 简单的线性索引
- **3D 张量**:
  - 将线性批次索引还原为 2D 索引：
    ```cpp
    batch_dim_idx = batch_idx / seq_len;
    seq_dim_idx = batch_idx % seq_len;
    ```
  - 计算原始偏移：`offset = batch_dim_idx * stride_0 + seq_dim_idx * stride_1`
  - 支持非连续内存布局（如转置张量）

### `logSoftmax()` - 全局核函数包装器
```cpp
template <unsigned int BLOCK_SIZE, typename Tdata_out, typename Tdata_in, typename Tcompute>
__global__ void logSoftmax(...);
```
- **功能**: 包装 `logSoftmaxKernel`，作为 CUDA 核函数入口点
- **启动配置**:
  - Grid: `(batch_size, 1, 1)` - 每个批次一个块
  - Block: `(BLOCK_SIZE, 1, 1)` - 可配置线程数

## 4. 混合精度支持

### `launchKernel()` - 模板调度器
```cpp
template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(
    void *y, const void *x,
    infiniDtype_t x_dtype, infiniDtype_t y_dtype,
    size_t batch_size, size_t probs_size, size_t ndim, size_t seq_len,
    ptrdiff_t y_stride_b, ptrdiff_t y_stride_p,
    ptrdiff_t x_stride_b, ptrdiff_t x_stride_p,
    ptrdiff_t y_stride_0, ptrdiff_t y_stride_1,
    ptrdiff_t x_stride_0, ptrdiff_t x_stride_1,
    cudaStream_t stream);
```

#### 支持的精度组合（7种）
1. **FP16 → FP32**: 输入 half，计算 float，输出 float
   ```cpp
   logSoftmax<BLOCK_SIZE, float, half, float><<<grid, BLOCK_SIZE, 0, stream>>>
   ```
2. **FP32 → FP16**: 输入 float，计算 float，输出 half
3. **BF16 → FP32**: 输入 `__nv_bfloat16`，计算 float，输出 float
4. **FP32 → BF16**: 输入 float，计算 float，输出 `__nv_bfloat16`
5. **FP16 → FP16**: 输入 half，计算 float，输出 half
6. **BF16 → BF16**: 输入 `__nv_bfloat16`，计算 float，输出 `__nv_bfloat16`
7. **FP32 → FP32**: 全精度计算

#### 计算类型选择策略
- 所有组合均使用 **float 作为计算类型**（`Tcompute = float`）
- 保证数值精度，避免 FP16/BF16 计算的精度损失
- 自动处理输入/输出的类型转换

## 5. API 使用示例

```cpp
#include "infinicore.h"

// 1. 创建张量描述符（假设输入形状 [batch_size, vocab_size]）
infiniopTensorDescriptor_t x_desc, y_desc;
infiniopCreateTensorDescriptor(handle, &x_desc);
infiniopSetTensorDescriptor(handle, x_desc, 2,
    {batch_size, vocab_size},     // 形状
    {vocab_size, 1});              // 行主序步长

infiniopCreateTensorDescriptor(handle, &y_desc);
infiniopSetTensorDescriptor(handle, y_desc, 2,
    {batch_size, vocab_size},
    {vocab_size, 1});

// 2. 创建 LogSoftmax 算子描述符
op::logsoftmax::nvidia::Descriptor *logsoftmax_desc;
auto status = op::logsoftmax::nvidia::Descriptor::create(
    handle,
    &logsoftmax_desc,
    y_desc,
    x_desc);
if (status != INFINI_STATUS_SUCCESS) {
    // 错误处理
}

// 3. 分配 GPU 内存
void *d_x, *d_y;
cudaMalloc(&d_x, batch_size * vocab_size * sizeof(float));
cudaMalloc(&d_y, batch_size * vocab_size * sizeof(float));

// 4. 复制数据到 GPU
cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

// 5. 执行计算
cudaStream_t stream;
cudaStreamCreate(&stream);
status = logsoftmax_desc->calculate(
    nullptr, 0,    // 无需工作空间
    d_y,           // 输出
    d_x,           // 输入
    stream);       // CUDA 流

// 6. 同步并取回结果
cudaStreamSynchronize(stream);
cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

// 7. 清理资源
delete logsoftmax_desc;
infiniopDestroyTensorDescriptor(x_desc);
infiniopDestroyTensorDescriptor(y_desc);
cudaFree(d_x);
cudaFree(d_y);
cudaStreamDestroy(stream);
```

## 6. 实现细节与优化策略

### 内存管理
- **工作空间**: 当前实现不需要额外工作空间（`workspace_size = 0`）
- **共享内存**: 每个 CUDA 块使用 2 个 `Tcompute` 类型的共享内存变量：
  - `shared_max_val`: 存储最大值
  - `shared_sum_exp`: 存储指数和
- **共享内存总量**: `2 * sizeof(float) = 8 bytes`（与 BLOCK_SIZE 无关）

### 并发与线程安全
- **CUDA 流支持**: 每次调用可指定不同的 CUDA 流，实现并发执行
- **块间独立**: 不同批次（grid.x）之间完全独立，无竞争条件
- **块内同步**: 使用 `__syncthreads()` 确保所有线程在阶段转换前完成
- **无全局内存竞争**: 每个线程写入独立的输出元素

### 性能优化技术
1. **块级归约**: 使用 CUB 库的 `BlockReduce` 原语，比共享内存手动实现更高效
2. **模板特化**: 编译期实例化不同 BLOCK_SIZE 和类型组合，零运行时开销
3. **寄存器缓存**: 将 `shared_max_val` 和 `shared_sum_exp` 缓存到寄存器，减少共享内存访问
4. **循环展开**: 编译器自动展开 `for (int i = tid; i < probs_size; i += BLOCK_SIZE)`
5. **类型专用数学函数**: 使用 `expf()`/`logf()`（float）而非 `exp()`/`log()`（double）

### 算法复杂度
- **时间复杂度**: O(N) per batch，其中 N = `probs_size`
  - 三个阶段均为线性扫描
  - BlockReduce 为 O(log BLOCK_SIZE)，可忽略（BLOCK_SIZE ≤ 4096）
- **空间复杂度**: O(1) 额外空间（仅共享内存）
- **并行度**: `batch_size * BLOCK_SIZE` 个线程同时工作

### 错误处理机制
1. **数据类型验证**: 在 `LogSoftmaxInfo::create()` 中检查，不支持的类型返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
2. **形状验证**: 检查维度数（2D/3D）和形状一致性
3. **架构支持**: 运行时检查 `maxThreadsPerBlock()`，不支持的架构返回错误
4. **CUDA 错误**: 使用 `CHECK_STATUS` 宏传播 CUDA API 错误

### 设计模式
1. **策略模式 (Strategy Pattern)**: 不同 BLOCK_SIZE 和类型组合通过模板实例化
2. **工厂模式 (Factory Pattern)**: `Descriptor::create()` 静态工厂方法
3. **RAII 资源管理**: Descriptor 析构函数自动释放 Opaque 资源
4. **Pimpl 惯用法**: Opaque 结构隐藏 NVIDIA Handle 的内部实现细节

### 依赖关系
- **CUB 库**: NVIDIA CUDA Blocks 原语库（`<cub/block/block_reduce.cuh>`）
- **CUDA 运行时**: `cuda_runtime.h`
- **Infini 核心**: `infinicore.h`, `../../operator.h`
- **张量抽象**: `../../tensor.h`
- **工具库**: `../../../utils.h`（Result<T> 类型）

### 数值稳定性保证
1. **最大值归一化**: 减去最大值避免 `exp()` 溢出
   ```
   exp(x[i] - max_val) ∈ [0, 1]
   ```
2. **对数域计算**: 直接计算 `x - max - log(sum_exp)` 而非 `log(exp(x) / sum_exp)`
3. **高精度计算**: 使用 float 作为计算类型，即使输入/输出为低精度

### GPU 架构适配
- **计算能力 5.0+**: 支持 FP16（`__half`）
- **计算能力 8.0+**: 支持 BF16（`__nv_bfloat16`）
- **动态线程配置**: 根据 `maxThreadsPerBlock()` 选择最优 BLOCK_SIZE
  - 老架构（如 Kepler）：512 线程/块
  - 主流架构（如 Pascal/Turing/Volta）：1024 线程/块
  - 新架构（如 Hopper）：4096 线程/块

### 3D 张量特殊处理
本实现支持 Transformer 模型中常见的 3D 张量 `[batch, seq_len, vocab_size]`：
- **扁平化**: 将前两维展平为 `[batch * seq_len, vocab_size]` 处理
- **索引重建**: 核函数中反向计算原始 2D 索引，支持非连续内存
- **应用场景**: 语言模型输出的 LogSoftmax（如 GPT、BERT）
- **内存效率**: 避免张量复制，直接在原始布局上计算

## 7. 限制与注意事项

1. **仅支持 2D/3D 张量**: 1D 或 4D+ 张量返回错误
2. **输入输出形状必须一致**: 不支持广播
3. **不支持原地操作**: 输入输出指针不能相同
4. **CUDA 版本要求**: 建议使用 CUDA 11.0+（BF16 支持）
5. **批次大小限制**: 受 GPU 全局内存限制，但通常足够大
6. **概率维度限制**: 建议 `probs_size` ≤ 65536（受限于线程寄存器使用）

## 8. 性能参考（理论估算）

假设 GPU 为 NVIDIA A100（计算能力 8.0）：
- **峰值算力**: 19.5 TFLOPS（FP32 Tensor Core）
- **内存带宽**: 2 TB/s
- **典型延迟**:
  - `batch_size=32, vocab_size=50000`: 约 0.1ms
  - `batch_size=1, vocab_size=50000`: 约 0.01ms
- **瓶颈分析**:
  - 小批次（< 8）：内存带宽受限
  - 大批次（> 64）：计算单元饱和
- **优化建议**:
  - 对于大批次，使用 `probs_size` ≥ 1024 以充分利用并行度
  - 对于小批次，合并多个批次到单个核函数调用以提升占用率
