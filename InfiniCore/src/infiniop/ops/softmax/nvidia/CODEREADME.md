# NVIDIA Softmax 算子核心实现文档

本模块实现了 Infini 框架中 NVIDIA GPU 后端的 Softmax 算子，支持 FP16 和 FP32 数据类型，针对不同张量维度和 GPU 架构提供优化的 kernel 实现。

## 1. 模块结构

- **`softmax_nvidia.cuh`**: NVIDIA Softmax 算子描述符的头文件声明，通过宏 `DESCRIPTOR(nvidia)` 定义接口
- **`softmax_nvidia.cu`**: NVIDIA Softmax 算子的主实现文件，包含算子创建、kernel 调度逻辑和设备适配

## 2. 核心类与数据结构

### `DataMaxSum`
- **位置**: `../cuda/kernel.cuh` (依赖文件)
- **对齐方式**: 8 字节对齐 (`__align__(8)`)
- **主要功能**: 并行归约过程中同时追踪最大值和指数和
- **核心成员**:
  - `float max_tmp`: 存储当前归约区间的最大值
  - `float sum_tmp`: 存储当前归约区间的指数和（相对于当前最大值）
- **设计动机**: Softmax 计算需要先找最大值（数值稳定性），再计算指数和，通过合并这两个操作到一次归约中减少全局内存访问

### `Descriptor::Opaque`
- **位置**: `softmax_nvidia.cu`
- **命名空间**: `op::softmax::nvidia`
- **主要功能**: 封装 NVIDIA 设备特定的内部状态
- **核心成员**:
  - `std::shared_ptr<device::nvidia::Handle::Internal> internal`: NVIDIA 设备句柄的内部实现，包含设备能力信息（如最大线程数）
- **生命周期**: 由 `Descriptor::create` 构造，在 `Descriptor` 析构时释放

### `Descriptor`
- **位置**: 通过 `softmax_nvidia.cuh` 中的宏展开定义在 `op::softmax::nvidia` 命名空间
- **继承**: `InfiniopDescriptor`
- **主要功能**: NVIDIA Softmax 算子的描述符类，管理算子配置信息和执行接口
- **核心成员**:
  - `Opaque *_opaque`: NVIDIA 特定的内部状态（设备句柄）
  - `SoftmaxInfo _info`: Softmax 计算的元数据（数据类型、维度大小、步长等）
  - `size_t _workspace_size`: 工作空间大小（当前实现为 0）
- **核心方法**:
  - `create(...)`: 静态工厂方法，验证张量描述符并构造算子描述符，返回 `infiniStatus_t` 状态码
  - `calculate(...)`: 执行 Softmax 计算，调度 CUDA kernel，支持多种数据类型和 GPU 架构
  - `workspaceSize()`: 返回所需工作空间大小（当前为 0）
- **生命周期**: 通过 `create` 方法创建，由用户负责释放（析构函数删除 `_opaque`）

### `SoftmaxInfo`
- **位置**: `../info.h` (依赖文件)
- **命名空间**: `op::softmax`
- **主要功能**: 存储从张量描述符中提取的 Softmax 计算元数据
- **核心成员**:
  - `infiniDtype_t dtype`: 数据类型（FP16/FP32/BF16）
  - `size_t othersize`: 除了归约轴之外所有维度的乘积（即独立的 Softmax 计算次数）
  - `size_t dimsize`: 归约轴的维度大小
  - `ptrdiff_t stride`: 在归约轴上的步长（用于计算内存偏移）
- **工厂方法**: `create(...)` 静态方法，从输入/输出张量描述符和轴参数构造 `SoftmaxInfo`，返回 `Result<SoftmaxInfo>`

## 3. API 接口

### 算子创建接口

```cpp
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,                  // Infini 设备句柄
    Descriptor **desc_ptr,                    // [输出] 创建的算子描述符指针
    infiniopTensorDescriptor_t y_desc,        // 输出张量描述符
    infiniopTensorDescriptor_t x_desc,        // 输入张量描述符
    int axis                                  // Softmax 归约轴
);
```
**功能**: 创建 NVIDIA Softmax 算子描述符，验证输入/输出张量的数据类型一致性、形状匹配性，提取计算元数据，初始化设备特定状态。
**返回值**: `INFINI_STATUS_SUCCESS` 表示成功，否则返回错误码（如类型不匹配、形状不匹配、数据类型不支持等）

### 算子计算接口

```cpp
infiniStatus_t Descriptor::calculate(
    void *workspace,           // 工作空间指针（当前未使用）
    size_t workspace_size,     // 工作空间大小（当前未使用）
    void *y,                   // [输出] 输出张量数据指针
    const void *x,             // [输入] 输入张量数据指针
    void *stream_              // CUDA 流指针（cudaStream_t）
) const;
```
**功能**: 执行 Softmax 计算，根据 `dimsize` 选择适当的 kernel（block softmax 或 warp softmax），根据设备能力和数据类型调度优化的实现。
**返回值**: `INFINI_STATUS_SUCCESS` 表示成功，否则返回错误码（如架构不支持、数据类型错误等）

### CUDA Kernel 接口（内部）

#### Block Softmax Kernel（大维度优化）

```cpp
template <typename Tdata, unsigned int BLOCK_SIZE>
__global__ void blockSoftmax(
    Tdata *y,                  // [输出] 输出数据指针
    const Tdata *x,            // [输入] 输入数据指针
    size_t dimsize,            // 归约轴维度大小
    ptrdiff_t stride           // 在归约轴上的步长
);
```
**适用场景**: `dimsize > 1024`，使用 CUB 库的 `BlockReduce` 进行高效的块内归约
**算法复杂度**: O(dimsize) 时间，O(BLOCK_SIZE) 线程并行

#### Warp Softmax Kernel（中小维度优化）

```cpp
template <typename Tdata, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y, int numPerThreadx>
__global__ void warpSoftmax(
    Tdata *y,                  // [输出] 输出数据指针
    const Tdata *x,            // [输入] 输入数据指针
    size_t othersize,          // 独立计算次数
    size_t dimsize,            // 归约轴维度大小
    ptrdiff_t stride           // 在归约轴上的步长
);
```
**适用场景**:
- `31 < dimsize <= 1024`: 使用 32x32 线程块，每线程处理 32 个元素
- `dimsize <= 31`: 使用 16x32 线程块，每线程处理 2 个元素
**算法复杂度**: O(dimsize / numPerThreadx) 时间，O(BLOCK_SIZE_x * BLOCK_SIZE_y) 线程并行

## 4. 使用示例

```cpp
// 示例：对形状为 [batch_size, seq_len] 的 2D 张量在 seq_len 轴上执行 Softmax

// 1. 准备张量描述符
std::vector<size_t> shape = {32, 128};  // batch_size=32, seq_len=128
std::vector<int64_t> x_strides = {128, 1};
std::vector<int64_t> y_strides = {128, 1};

infiniopTensorDescriptor_t x_desc, y_desc;
infiniopCreateTensorDescriptor(&x_desc, INFINI_DTYPE_F16, 2, shape.data(), x_strides.data());
infiniopCreateTensorDescriptor(&y_desc, INFINI_DTYPE_F16, 2, shape.data(), y_strides.data());

// 2. 创建 Softmax 算子描述符
infiniopHandle_t handle;        // 假设已初始化的 CUDA 设备句柄
op::softmax::nvidia::Descriptor *softmax_desc;
infiniStatus_t status = op::softmax::nvidia::Descriptor::create(
    handle, &softmax_desc, y_desc, x_desc, 1  // axis=1 表示在 seq_len 维度上归约
);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 3. 分配内存并初始化输入数据
half *d_x, *d_y;
size_t total_elements = 32 * 128;
cudaMalloc(&d_x, total_elements * sizeof(half));
cudaMalloc(&d_y, total_elements * sizeof(half));
// ... 将输入数据拷贝到 d_x ...

// 4. 创建 CUDA 流
cudaStream_t stream;
cudaStreamCreate(&stream);

// 5. 执行 Softmax 计算
status = softmax_desc->calculate(
    nullptr,              // workspace (当前不需要)
    0,                    // workspace_size
    d_y,                  // 输出指针
    d_x,                  // 输入指针
    stream                // CUDA 流
);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 6. 同步并获取结果（可选）
cudaStreamSynchronize(stream);
// ... 将 d_y 拷贝回主机 ...

// 7. 清理资源
cudaFree(d_x);
cudaFree(d_y);
cudaStreamDestroy(stream);
delete softmax_desc;
infiniopDestroyTensorDescriptor(x_desc);
infiniopDestroyTensorDescriptor(y_desc);
```

## 5. 实现细节

### 数值稳定性优化

**问题**: 直接计算 `exp(x_i)` 会导致浮点数溢出（当 `x_i` 较大时）。
**解决方案**: 使用经典的数学技巧 `softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))`：
1. 先找到张量中的最大值 `max_val`
2. 计算指数时减去最大值：`exp(x_i - max_val)`（保证指数函数的输入 <= 0，避免溢出）
3. 对所有 `exp(x_i - max_val)` 求和得到 `sum_exp`
4. 最终结果：`exp(x_i - max_val) / sum_exp`

**实现位置**: `blockSoftmaxKernel` 和 `warpSoftmaxKernel` 均采用此策略。

### 并行归约策略

#### Block 级归约（大维度）
**算法**: 使用 NVIDIA CUB 库的 `BlockReduce` 原语。
**步骤**:
1. 每个线程遍历 `dimsize` 中属于它的元素（步长为 `BLOCK_SIZE`）
2. 对每个元素 `x`，构造 `DataMaxSum{x, 1.0f}`（局部最大值就是 `x`，局部和为 1）
3. 使用自定义归约算子 `reduce_dms_op` 将所有 `DataMaxSum` 合并为一个：
   - 比较两个 `DataMaxSum` 的 `max_tmp`，取较大者
   - 将较小者的 `sum_tmp` 乘以 `exp(smaller.max - bigger.max)` 后加到较大者的 `sum_tmp` 上（数学等价于在统一的数值稳定框架下合并两个区间的结果）
4. CUB 的 `BlockReduce` 自动处理块内线程间的并行归约（通常使用树形归约，复杂度 O(log BLOCK_SIZE)）
5. 线程 0 将归约结果写入共享内存 `dms_total`，同步后所有线程使用此结果计算输出

**性能优势**:
- CUB 库高度优化，充分利用共享内存和 warp shuffle 指令
- 树形归约复杂度 O(log BLOCK_SIZE)，优于线性扫描的 O(BLOCK_SIZE)
- 适合 `dimsize >> BLOCK_SIZE` 的情况，每个线程处理多个元素，提高内存访问合并度

#### Warp 级归约（中小维度）
**算法**: 使用 CUDA warp shuffle 指令实现全归约（`WarpAllReduce`）。
**步骤**:
1. 每个线程处理 `numPerThreadx` 个元素（循环展开，步长为 `BLOCK_SIZE_x`，即 32）
2. 第一次归约（找最大值）：
   - 每个线程先遍历自己负责的元素，找局部最大值
   - 调用 `WarpAllReduce<MaxOp, float, 32>(max_data)`，使用 `__shfl_xor_sync` 在 32 线程的 warp 内交换数据，逐步完成全归约（二分树形，5 轮迭代）
   - 线程 0 将结果写入共享内存 `max_total[threadIdx.y]`
3. 第二次归约（计算指数和）：
   - 每个线程重新遍历元素，计算 `exp(x - max_total)` 并累加
   - 调用 `WarpAllReduce<SumOp, float, 32>(sum_data)` 完成全归约
   - 线程 0 将结果写入共享内存 `sum_total[threadIdx.y]`
4. 计算输出：每线程再次遍历元素，输出 `exp(x - max_total) / sum_total`

**性能优势**:
- Warp shuffle 指令比共享内存访问更快（延迟约 1-2 个周期 vs 共享内存约 20-30 个周期）
- 全归约在 warp 内完成，无需跨 warp 同步
- 多个 warp 并行处理独立的 Softmax（`BLOCK_SIZE_y` 个 warp，每个处理不同的 `otherIdx`）
- 适合 `dimsize <= 1024` 的情况，每线程处理少量元素，减少寄存器压力

### Kernel 调度逻辑

**决策树**:
```
if (dtype == FP16 or FP32) {
    if (dimsize > 1024) {
        使用 blockSoftmax kernel (单维 block，大小为 512/1024/4096)
    } else if (dimsize > 31) {
        使用 warpSoftmax kernel (32x32 线程块，每线程 32 元素)
    } else {
        使用 warpSoftmax kernel (16x32 线程块，每线程 2 元素)
    }
} else {
    返回 INFINI_STATUS_BAD_TENSOR_DTYPE 错误
}
```

**设备适配**:
- 根据 `internal->maxThreadsPerBlock()` 选择合适的 `BLOCK_SIZE`（512/1024/4096）
- 支持现代 NVIDIA GPU 架构（Compute Capability >= 5.0，支持 FP16）
- 通过 `INFINIOP_CUDA_KERNEL` 宏适配 Hygon DCU 等兼容平台（添加 `__launch_bounds__` 属性）

### 内存访问模式优化

**输入张量索引计算**:
- `tid = blockIdx.x % stride + (blockIdx.x - blockIdx.x % stride) * dimsize` (block softmax)
- `tid = otherIdx % stride + (otherIdx - otherIdx % stride) * dimsize` (warp softmax)
- 元素访问：`x[tid + ind * stride]`，其中 `ind` 是归约轴内的索引
**设计思路**:
- `stride` 是归约轴之后的维度乘积（即连续两个归约轴元素之间的跨度）
- 通过模运算和整数除法分解线性索引为多维索引
- 保证同一 block/warp 处理的元素在内存中是连续或步长固定，提高缓存命中率

**输出张量写入**:
- 与输入相同的索引模式，确保原地计算（in-place）支持
- 使用 `static_cast<T>` 进行类型转换（FP16 <-> FP32）

### 错误处理机制

**创建阶段错误**:
- `INFINI_STATUS_BAD_TENSOR_DTYPE`: 输入/输出类型不匹配或不支持（仅支持 FP16/FP32/BF16）
- `INFINI_STATUS_BAD_TENSOR_SHAPE`: 输入/输出形状不匹配
- `Result<T>` 模式：通过 `CHECK_RESULT` 宏检查并提前返回错误

**执行阶段错误**:
- `INFINI_STATUS_BAD_TENSOR_DTYPE`: 数据类型不支持（如 INT8）
- `INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED`: GPU 架构不支持（maxThreadsPerBlock 不是 512/1024/4096）
- CUDA 错误传播：kernel 启动失败时，`CHECK_STATUS` 宏捕获 `cudaError` 并转换为 `infiniStatus_t`

### 依赖关系

**外部依赖**:
- CUDA Runtime API（`cudaStream_t`, `<<<>>>` kernel 启动语法）
- CUB 库（`cub::BlockReduce`）- NVIDIA 提供的高性能 CUDA 原语库
- Infini 框架内部：
  - `infinicore.h`: 核心类型定义和错误码
  - `device::nvidia::Handle`: NVIDIA 设备句柄管理
  - `tensor.h`: 张量描述符接口
  - `utils.h`: 工具宏（如 `CHECK_DTYPE`, `CHECK_SAME_SHAPE`）

**内部依赖**:
- `../softmax.h`: 算子描述符宏定义
- `../info.h`: `SoftmaxInfo` 元数据类
- `../cuda/kernel.cuh`: CUDA kernel 实现（`blockSoftmaxKernel`, `warpSoftmaxKernel`）
- `../../../devices/nvidia/nvidia_common.cuh`: CUDA 公共定义
- `../../../devices/nvidia/nvidia_kernel_common.cuh`: CUDA kernel 公共宏和工具函数
- `../../../reduce/cuda/reduce.cuh`: 归约算子（代码中包含但实际未直接使用，可能是遗留依赖）

### 设计模式

**策略模式 (Strategy Pattern)**:
- 根据张量维度（`dimsize`）动态选择 kernel 实现（block softmax vs warp softmax）
- 根据 GPU 架构（`maxThreadsPerBlock`）选择 block size（512/1024/4096）

**工厂模式 (Factory Pattern)**:
- `Descriptor::create` 静态方法作为工厂，根据输入参数构造算子对象

**模板方法模式 (Template Method Pattern)**:
- `DESCRIPTOR` 宏定义了算子描述符的通用结构，各后端（nvidia, cpu, kunlun 等）通过宏展开特化实现

**RAII (Resource Acquisition Is Initialization)**:
- `Descriptor` 的析构函数自动释放 `_opaque` 资源
- 使用 `std::shared_ptr` 管理 `Handle::Internal` 的生命周期

### 性能特性

**时间复杂度**:
- Block softmax: O(dimsize / BLOCK_SIZE + log BLOCK_SIZE) ≈ O(dimsize / BLOCK_SIZE)（当 dimsize 很大时）
- Warp softmax: O(dimsize / (BLOCK_SIZE_x * numPerThreadx) + log BLOCK_SIZE_x) ≈ O(dimsize / 1024)（当 dimsize <= 1024）

**空间复杂度**:
- 共享内存占用：
  - Block softmax: `sizeof(DataMaxSum) + sizeof(typename BlockReduce::TempStorage)` ≈ 8 字节 + 若干字节（CUB 内部使用）
  - Warp softmax: `(BLOCK_SIZE_y * 2) * sizeof(float)` = 64 * 4 = 256 字节（max_total 和 sum_total 数组）
- 全局内存：输入 + 输出张量，无额外临时内存（原地计算友好）

**吞吐量优化**:
- 内存合并访问：相邻线程访问相邻内存地址（stride 为 1 或连续小整数）
- 寄存器复用：warp softmax 中，`dataPerThreadx` 数组在三个阶段（max, sum, output）复用
- 最小化全局内存访问：每个输入元素仅读取 2 次（计算 max 和 sum 时各一次，但实际实现中为 3 次），每个输出元素仅写入 1 次

**延迟优化**:
- 避免 `__syncthreads()`：warp softmax 使用 shuffle 指令，无需同步
- 最小化同步点：block softmax 仅在归约后同步一次
- 流水线执行：CUDA stream 支持与其他操作的重叠执行

### 线程安全性

**算子级别**:
- `Descriptor` 对象创建后不可变（immutable），多线程可同时调用 `calculate`，只要传入不同的 `stream`
- `stream` 参数确保不同执行实例的隔离，CUDA 运行时保证流间并发安全性

**Kernel 级别**:
- 共享内存访问：通过 `__syncthreads()` 确保正确性（block softmax）
- Warp shuffle：内置线程安全性，无需显式同步（warp softmax）
- 全局内存写入：每个线程写入独立位置，无竞争

## 6. 架构适配说明

### 支持的 NVIDIA GPU 架构

**Compute Capability 5.0+ (Maxwell 及更新)**:
- 支持 FP16（通过 `__half` 类型）
- 最大线程数：1024（标准架构）

**Compute Capability 7.0+ (Volta/Turing/Ampere)**:
- 支持 FP16 加速（Tensor Core 可用于其他算子，Softmax 仍使用标准 CUDA Core）
- 改进的 FP16 算术性能

**Compute Capability 8.0+ (Ampere)**:
- 最大线程数：4096（部分高端 GPU）
- 支持的 block size：4096（通过 `CUDA_BLOCK_SIZE_4096` 宏）

**Compute Capability 9.0+ (Hopper)**:
- 继续兼容现有实现
- 可能的未来优化：使用 Hopper 的 Tensor Core 加速归约操作

### 兼容性处理

**Hygon DCU (深算科技)**:
- 通过 `ENABLE_HYGON_API` 宏定义：
  - 添加 `__launch_bounds__(1024)` 属性到所有 kernel（限制最大线程数为 1024）
  - 重新定义 `cuda_bfloat16` 类型（`__nv_bfloat16` vs `nv_bfloat16`）
  - 移除 `long double` 的 `exp_` 函数（DCU 可能不支持）

**Iluvatar (天数智芯) / QY (昆仑芯) / Cambricon (寒武纪)**:
- 通过各自的 API 宏（`ENABLE_ILUVATAR_API`, `ENABLE_QY_API` 等）适配
- 主要影响类型定义和数学函数实现

## 7. 限制与未来改进方向

### 当前限制

**数据类型支持**:
- 仅支持 FP16 和 FP32（INFO 中声明 BF16 但实际 kernel 未实现）
- 不支持 INT8/INT4 量化（需要反量化 + Softmax + 量化的流水线）

**轴参数**:
- 仅支持单轴 Softmax（不支持多轴或全局 Softmax）
- 负轴索引需要规范化（当前已支持）

**内存管理**:
- 不支持原地计算优化（尽管算法允许，但接口未显式支持）
- 工作空间大小固定为 0（未来可能用于优化大维度情况）

**性能**:
- 对 `dimsize` 在 1024-2048 之间的中等维度，可能不是最优（介于 block 和 warp 策略之间）
- 小批次情况（`othersize` 小）可能无法充分利用 GPU 并行性

### 潜在优化方向

**Kernel 融合**:
- 与相邻算子融合（如 Softmax + Mask, Softmax + Dropout）减少全局内存访问
- 与矩阵乘法融合（Attention 场景：Softmax(MatMul(Q, K)) * V）

**数据类型扩展**:
- 实现 BF16 kernel（当前 `SoftmaxInfo::create` 支持 BF16，但 `launchKernel` 返回错误）
- 添加 FP8 支持（需要精心设计的数值稳定性策略）

**算法改进**:
- 对大 `dimsize`（> 4096），使用分段归约 + 两遍算法
- 对小 `dimsize`（< 16），使用单线程处理 + 向量化指令
- 引入迭代优化算法（对精度要求不高的场景）

**硬件特定优化**:
- 利用 Tensor Core 加速 FP16 归约（Ampere+）
- 使用 CUDA Graph 减少启动开销（小批次场景）
- 引入 CUDA Async Copy（Hopper）减少延迟

**自适应调优**:
- 运行时根据输入形状和设备能力选择最优 kernel（auto-tuning）
- 性能模型预测（基于维度、数据类型、架构的经验公式）

**错误处理增强**:
- 添加输入验证（如 NaN/Inf 检查）
- 提供详细的诊断信息（失败时回退到参考实现）
