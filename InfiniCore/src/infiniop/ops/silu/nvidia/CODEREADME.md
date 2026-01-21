# SiLU (Swish) NVIDIA CUDA 算子核心实现文档

本模块实现了 Swish 激活函数（又称 SiLU：Sigmoid Linear Unit）的 NVIDIA GPU CUDA 后端。SiLU 是深度学习中的平滑非线性激活函数，定义为 `SiLU(x) = x * sigmoid(x) = x / (1 + e^(-x))`。该实现基于 Infini 框架的元素操作（Elementwise）基础设施，支持多种数据类型（FP16、BF16、FP32、FP64）的张量运算，并针对 NVIDIA GPU 进行了向量化优化。

## 1. 模块结构

- **`silu_nvidia.cuh`**: CUDA 后端 API 声明头文件，通过宏定义生成 Descriptor 类接口
- **`silu_nvidia.cu`**: CUDA 后端实现文件，包含算子描述符的创建、销毁和计算调度逻辑
- **`../cuda/kernel.cuh`**: SiLU 操作的核心 CUDA 设备函数定义，包含各数据类型的向量化实现

## 2. 核心类与数据结构

### 2.1 `op::silu::nvidia::Descriptor`
- **位置**: `silu_nvidia.cuh` (通过宏展开生成)
- **主要功能**: SiLU 算子的 CUDA 后端描述符，继承自 `InfiniopDescriptor`，管理算子的元数据、设备实现和工作空间需求
- **核心成员**:
  - `_dtype`: `infiniDtype_t`，输出张量的数据类型（支持 BF16、F16、F32、F64）
  - `_info`: `op::elementwise::ElementwiseInfo`，元素操作的元数据（形状、步长、广播信息等）
  - `_device_info`: `std::unique_ptr<op::elementwise::nvidia::DeviceImpl>`，CUDA 设备实现的智能指针
  - `_workspace_size`: `size_t`，设备端计算所需的工作空间大小（存储元数据和输入指针数组）
- **生命周期**:
  - **创建**: 通过静态工厂方法 `Descriptor::create()` 构造，验证输入输出张量形状一致性，初始化元素操作元数据和 CUDA 设备实现
  - **销毁**: 使用默认析构函数，智能指针自动管理 `_device_info` 的生命周期
  - **所有权**: 独占所有权，由调用者负责在不再需要时释放

### 2.2 `op::silu::cuda::SiluOp`
- **位置**: `../cuda/kernel.cuh`
- **主要功能**: CUDA 设备端的 SiLU 操作仿函数（Functor），定义各数据类型的 SiLU 计算逻辑
- **核心成员**:
  - `num_inputs`: `static constexpr size_t = 1`，输入张量的数量（单目运算）
- **核心方法**:
  - `operator()<T>(const T &x)`: CUDA 设备函数，对单个元素 x 执行 SiLU 计算
    - **FP16 向量化 (`half2`)**: 使用 CUDA 内置函数 `__hmul2`、`__h2div`、`__hadd2`、`h2exp`、`__hneg2` 实现 SIMD 向量化运算，一次处理两个 FP16 值
    - **FP16 标量 (`half`)**: 转换为 FP32 计算 `x_f / (1.0f + exp(-x_f))`，再转回 FP16
    - **BF16 标量 (`cuda_bfloat16`)**: 转换为 FP32 计算后转回 BF16
    - **FP32 标量 (`float`)**: 直接计算 `x * (1.0f / (1.0f + exp(-x)))`，使用 `__expf` 内置函数
    - **FP64 标量 (`double`)**: 使用双精度 `exp(-x)` 计算 `x / (1.0 + exp(-x))`
- **实现细节**:
  - 使用 `if constexpr` 编译期类型分发，零运行时开销
  - FP16 向量化版本利用 GPU 的 16 位浮点 SIMD 指令，理论上可达到 2x 吞吐量提升
  - 所有分支都强制内联（`__forceinline__`）以减少函数调用开销

### 2.3 `op::elementwise::ElementwiseInfo`
- **位置**: `../../elementwise/elementwise.h` (基类基础设施)
- **主要功能**: 封装元素操作的元数据，包括张量形状、步长、连续性和广播信息
- **核心成员**:
  - `_meta`: `std::vector<size_t>`，扁平化存储所有元数据（输出形状、步长、所有输入形状、步长、连续性标志、广播标志）
  - `_output_size`: `size_t`，输出张量的元素总数
  - `_ndim`: `size_t`，张量的维度数
  - `_output_contiguous`: `bool`，输出张量是否内存连续
- **内存布局**（按 `_meta` 中的顺序）:
  1. 输出形状 (`ndim` 个 `size_t`)
  2. 输出步长 (`ndim` 个 `ptrdiff_t`)
  3. 所有输入形状 (`input_size * ndim` 个 `size_t`)
  4. 所有输入步长 (`input_size * ndim` 个 `ptrdiff_t`)
  5. 输入连续性标志 (`input_size` 个 `bool`)
  6. 输入广播标志 (`input_size` 个 `bool`)

### 2.4 `op::elementwise::nvidia::DeviceImpl`
- **位置**: `../../elementwise/nvidia/elementwise_nvidia.cuh`
- **主要功能**: CUDA 元素操作的设备端执行引擎，负责内核启动、内存管理和并行调度
- **核心成员**:
  - `_opaque`: `std::shared_ptr<Opaque>`，Pimpl 模式的实现细节指针
- **关键方法**:
  - `calculate<BLOCK_SIZE, Op, Tdata>(...)`: 单一数据类型的计算调度入口
    - 验证工作空间大小
    - 根据 `_dtype` 分发到对应的模板实例化
    - 调用内部 `calculateImpl` 启动 CUDA 内核
  - `calculateImpl<BLOCK_SIZE, N, Op, Tdata>(...)`: 内部计算实现
    - 准备元数据并传输到设备内存
    - 计算网格和块维度
    - 启动 CUDA 内核，可能分多次循环处理大张量
  - `infoToDevice<N>(...)`: 将主机端元数据异步复制到设备内存
    - 使用 `cudaMemcpyAsync` 避免阻塞
    - 在设备工作空间中偏移计算各类元数据的指针位置

## 3. API 接口

### 3.1 算子描述符创建接口

```cpp
namespace op::silu::nvidia {

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,              // [输入] Infini 设备句柄，包含设备和上下文信息
    Descriptor **desc_ptr,                 // [输出] 指向描述符指针的指针，函数会分配并返回新的 Descriptor
    infiniopTensorDescriptor_t out_desc,   // [输入] 输出张量描述符，定义数据类型、形状和布局
    std::vector<infiniopTensorDescriptor_t> input_desc_vec  // [输入] 输入张量描述符向量，SiLU 只有一个输入
);
```
**功能**: 创建 SiLU 算子的 CUDA 描述符，验证输入输出张量形状一致性，初始化元数据和设备实现
**返回值**:
- `INFINI_STATUS_SUCCESS`: 成功创建
- `INFINI_STATUS_BAD_TENSOR_DTYPE`: 数据类型不支持（仅支持 BF16/F16/F32/F64）
- `INFINI_STATUS_BAD_TENSOR_SHAPE`: 输入输出形状不一致

### 3.2 计算执行接口

```cpp
infiniStatus_t Descriptor::calculate(
    void *workspace,                       // [输入] 设备端工作空间指针，用于存储元数据
    size_t workspace_size,                 // [输入] 工作空间大小（字节），必须 >= workspaceSize()
    void *output,                          // [输入/输出] 输出张量的设备内存指针
    std::vector<const void *> inputs,      // [输入] 输入张量的设备内存指针向量
    void *stream                           // [输入] CUDA 流指针，用于异步执行
) const;
```
**功能**: 在 GPU 上执行 SiLU 计算，根据数据类型分发到相应的模板实例化
**返回值**:
- `INFINI_STATUS_SUCCESS`: 计算成功提交到 GPU
- `INFINI_STATUS_INSUFFICIENT_WORKSPACE`: 工作空间不足
- `INFINI_STATUS_BAD_TENSOR_DTYPE`: 遇到不支持的内部数据类型

### 3.3 查询接口

```cpp
size_t workspaceSize() const;
```
**功能**: 返回执行该算子所需的工作空间大小（字节），用于分配设备内存
**计算公式**: `info.getMetaMemSize() + info.getInputSize() * sizeof(void*)`

## 4. 使用示例

```cpp
// 示例：在 NVIDIA GPU 上执行 SiLU 激活函数
#include "infiniop/ops/silu/nvidia/silu_nvidia.cuh"

void example_silu_nvidia() {
    // 1. 初始化 Infini 句柄和 CUDA 设备
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle, INFINI_DEVICE_NVIDIA, 0);

    // 2. 定义张量形状和数据类型
    const std::vector<size_t> shape = {1024, 1024};  // 1M 元素
    infiniDtype_t dtype = INFINI_DTYPE_F16;           // 半精度浮点

    // 3. 创建输入输出张量描述符
    infiniopTensorDescriptor_t input_desc, output_desc;
    infoniopCreateTensorDescriptor(&input_desc, dtype, shape, nullptr);
    infoniopCreateTensorDescriptor(&output_desc, dtype, shape, nullptr);

    // 4. 创建 SiLU 算子描述符
    op::silu::nvidia::Descriptor *silu_desc = nullptr;
    std::vector<infiniopTensorDescriptor_t> inputs = {input_desc};
    auto status = op::silu::nvidia::Descriptor::create(
        handle, &silu_desc, output_desc, inputs);
    if (status != INFINI_STATUS_SUCCESS) {
        // 错误处理
        return;
    }

    // 5. 分配 GPU 内存和工作空间
    const size_t num_elements = 1024 * 1024;
    const size_t bytes = num_elements * sizeof(half);
    half *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    size_t workspace_size = silu_desc->workspaceSize();
    void *d_workspace;
    cudaMalloc(&d_workspace, workspace_size);

    // 6. 准备输入数据（假设已从主机复制到设备）
    // cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // 7. 获取 CUDA 流并执行计算
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    status = silu_desc->calculate(
        d_workspace, workspace_size,
        d_output, {d_input},
        stream);

    // 8. 同步并获取结果
    cudaStreamSynchronize(stream);
    // cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // 9. 清理资源
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_workspace);
    cudaStreamDestroy(stream);
    delete silu_desc;
    infiniopDestroyHandle(handle);
}
```

## 5. 实现细节

### 5.1 内存管理策略
- **设备内存分配**: 工作空间由调用者在主机端预分配，算子描述符仅计算大小不负责分配
- **元数据传输**: 使用 `cudaMemcpyAsync` 异步传输元数据到设备，避免 CPU-GPU 同步阻塞
- **智能指针**: `Descriptor` 持有 `DeviceImpl` 的 `unique_ptr`，确保 RAII 语义
- **零拷贝优化**: 输入输出张量数据直接在设备内存操作，无需主机介入

### 5.2 并发与并行计算
- **线程块大小**: 固定为 256 线程/块 (`BLOCK_SIZE = 256`)，在 CUDA 占用率模型中是较好的平衡点
- **网格尺寸计算**: `gridDims.x = min(CEIL_DIV(output_size, 256), internal->gridSizeX())`，确保不超过设备网格限制
- **步进式内核启动**: 对于超大张量（超过 `grid_size * block_size`），使用 `for` 循环分多次启动内核，每次偏移 `step` 个元素
- **流式执行**: 计算在用户提供的 CUDA 流上异步执行，支持与其它操作并发

### 5.3 性能优化技术
- **向量化 SIMD**: 针对 FP16 类型提供 `half2` 向量化版本，利用 CUDA 的 16 位浮点 SIMD 指令（`__hmul2`、`__h2div` 等），理论上吞吐量翻倍
- **编译期类型分发**: 使用 `if constexpr` 和模板特化，所有类型分支在编译期确定，无运行时分支预测开销
- **强制内联**: 所有设备函数标记 `__forceinline__`，减少函数调用栈开销
- **连续性检测**: 对连续内存张量使用线性索引 `idx`，对非连续张量调用 `indexToOffset` 进行维度到偏移的映射，避免通用路径的性能损失
- **广播优化**: 在编译期通过 `input_broadcasted` 标志跳过不必要的索引计算

### 5.4 错误处理与边界条件
- **形状验证**: 创建时检查 `CHECK_SAME_SHAPE(output_shape, input_shape)`，确保输入输出形状一致
- **数据类型检查**: 宏 `CHECK_DTYPE` 验证数据类型是否在支持列表 {BF16, F16, F32, F64} 中
- **工作空间验证**: 执行时检查 `workspace_size < _workspace_size`，返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
- **空张量处理**: 内部实现中 `if (output_size == 0) return INFINI_STATUS_SUCCESS`，正确处理零元素张量
- **CUDA 错误传播**: 使用 `CHECK_CUDA` 宏检查所有 CUDA API 调用，将错误转换为 `infiniStatus_t`

### 5.5 依赖关系
- **上层依赖**: 依赖 `op::elementwise::nvidia::DeviceImpl` 作为元素操作执行引擎
- **数学库**: 使用 CUDA 内置数学函数 `__expf`（单精度）、`exp`（双精度）、`h2exp`（FP16 向量）
- **张量基础设施**: 依赖 `InfiniopDescriptor`、`infiniopTensorDescriptor_t` 等张量抽象
- **设备抽象**: 依赖 `device::nvidia::Handle` 提供设备属性（最大线程数、网格尺寸限制等）

### 5.6 设计模式应用
- **Pimpl 模式**: `DeviceImpl` 使用 `Opaque` 结构体隐藏实现细节，减少头文件依赖
- **CRTP (奇异递归模板模式)**: `ELEMENTWISE_DESCRIPTOR` 宏生成派生类，提供统一的描述符接口
- **策略模式**: 通过模板参数 `Op`（如 `SiluOp`）注入具体计算逻辑，实现元素操作框架的泛化
- **工厂模式**: `Descriptor::create()` 静态工厂方法封装复杂的初始化逻辑
- **RAII**: 智能指针管理 `DeviceImpl` 生命周期，确保资源正确释放

## 6. 算法复杂度分析

- **时间复杂度**: O(N)，其中 N 为张量元素总数。每个元素执行常数次算术运算（乘法、指数、除法）。
- **空间复杂度**: O(1) 额外空间（不包括输入输出张量）。工作空间仅存储元数据，大小与 N 无关，仅与维度数和输入数量相关。
- **并行复杂度**: 理论上可达到 O(N / P)，其中 P 为 GPU 并行线程数。实际性能受内存带宽、指令吞吐量和 warp 占用率限制。

## 7. 数据类型精度与性能权衡

| 数据类型 | 存储大小 | 计算精度 | 内存带宽 | 典型应用场景 |
|---------|---------|---------|---------|------------|
| FP16    | 2 bytes | 低（易溢出） | 高 | 推理加速、移动端/边缘设备 |
| BF16    | 2 bytes | 中（指数位少） | 高 | 混合精度训练、Transformer 模型 |
| FP32    | 4 bytes | 标准 | 中 | 通用深度学习训练 |
| FP64    | 8 bytes | 高 | 低 | 科学计算、高精度数值模拟 |

**注意**: FP16 向量化 (`half2`) 版本在内存访问上可达到 2x 吞吐量，但数值稳定性较差，建议配合 Loss Scaling 使用。

## 8. 未来优化方向

- **Tensor Core 加速**: 对 FP16/BF16 数据类型，可利用 NVIDIA Tensor Core WMMA API 实现矩阵分块加速
- **融合算子**: 将 SiLU 与相邻的线性层或归一化层融合，减少全局内存访问
- **动态调度**: 根据张量大小自适应选择块大小和网格策略
- **多 CUDA 流**: 对超大张量使用多流并行执行，提升设备利用率
- **Graph Capture**: 支持 CUDA Graph 捕获以减少内核启动开销
