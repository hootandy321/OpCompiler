# Zeros 操作 NVIDIA CUDA 后端实现文档

## 概述

本模块实现了 Infini 框架中 Zeros（全零填充）操作的 NVIDIA GPU CUDA 后端。该模块基于通用逐元素（elementwise）操作框架，提供高性能的张量零值初始化功能，支持 15 种数据类型，包括整数、浮点、布尔和半精度浮点类型。

## 1. 模块结构

- **`zeros_nvidia.cuh`**: 操作符描述符的声明，通过宏定义生成完整的 Descriptor 类接口
- **`zeros_nvidia.cu`**: Zeros 操作符的核心实现，包含描述符创建、计算调度和数据类型分派逻辑
- **`../cuda/kernel.cuh`**: CUDA 设备端操作符实现（`ZerosOp` functor），定义了各数据类型的零值生成逻辑

## 2. 核心类

### `op::zeros::nvidia::Descriptor`
- **位置**: `zeros_nvidia.cuh`（通过 `ELEMENTWISE_DESCRIPTOR` 宏生成）
- **主要功能**: 封装 Zeros 操作的元数据、设备实现和工作空间管理
- **继承关系**: 继承自 `InfiniopDescriptor` 基类
- **核心成员**:
  - `infiniDtype_t _dtype`: 输出张量的数据类型
  - `op::elementwise::ElementwiseInfo _info`: 张量形状、步长、广播等元数据
  - `std::unique_ptr<op::elementwise::nvidia::DeviceImpl> _device_info`: CUDA 设备实现指针
  - `size_t _workspace_size`: 设备端所需工作空间大小（字节数）

**核心方法**:

- **`create(handle, desc_ptr, out_desc, input_desc_vec)`**:
  - **功能**: 静态工厂方法，创建并初始化 Zeros 操作描述符
  - **参数验证**:
    - 检查输出数据类型是否为支持的 15 种类型之一（BYTE, BOOL, I8-I64, U8-U64, F8, F16, F32, F64, BF16）
    - 验证输入和输出张量的形状完全一致（`CHECK_SAME_SHAPE`）
  - **初始化流程**:
    1. 使用 `CREATE_ELEMENTWISE_CUDA_DESCRIPTOR` 宏创建逐元素操作元数据
    2. 调用 `ElementwiseInfo::create()` 提取张量形状、步长、连续性等信息
    3. 计算工作空间大小 = 元数据大小 + 输入指针数组大小
    4. 创建 `DeviceImpl` 实例并包装为智能指针
    5. 构造 `Descriptor` 对象并赋值给输出参数
  - **复杂度**: O(ndim)，主要用于遍历维度复制元数据

- **`calculate(workspace, workspace_size, output, inputs, stream)`**:
  - **功能**: 在 GPU 上执行 Zeros 操作的内核启动
  - **数据类型分派**: 基于编译期常量 `256` 作为 CUDA 块大小，使用 `switch-case` 语句为每种数据类型调用特化的模板实例
  - **支持的数据类型映射**:
    | 枚举值 | 类型标识 | CUDA 类型 |
    |--------|----------|-----------|
    | 1      | BYTE     | uint8_t   |
    | 2      | BOOL     | bool      |
    | 3-6    | I8-I64   | int8_t/int16_t/int32_t/int64_t |
    | 7-10   | U8-U64   | uint8_t/uint16_t/uint32_t/uint64_t |
    | 11     | F8       | cuda_fp8_e4m3 |
    | 12     | F16      | half       |
    | 13     | F32      | float      |
    | 14     | F64      | double     |
    | 19     | BF16     | cuda_bfloat16 |
    | 15-18  | C16-C128 | （未实现，返回 `INFINI_STATUS_NOT_IMPLEMENTED`） |
  - **内核调用签名**: `_device_info->calculate<256, cuda::ZerosOp, T>(_info, workspace, output, inputs, stream)`
  - **错误处理**:
    - 工作空间不足返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
    - 不支持的数据类型返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`

- **`~Descriptor()`**: 默认析构函数，自动管理 `_device_info` 智能指针的生命周期

### `op::zeros::cuda::ZerosOp`
- **位置**: `../cuda/kernel.cuh`
- **主要功能**: CUDA 设备端函数对象（functor），定义各数据类型的零值生成逻辑
- **设计模式**: Functor 模式，通过 `operator()` 重载实现可调用对象
- **核心常量**:
  - `static constexpr size_t num_inputs = 1`: 声明操作符需要 1 个输入张量（尽管 Zeros 操作实际不读取输入）

**核心方法**:

- **`operator()(const T &x) const`**:
  - **功能**: 设备端内联函数，返回类型 `T` 的零值表示
  - **编译期分支**: 使用 `if constexpr` 在编译期根据类型 `T` 选择正确的零值构造
  - **类型特化**:
    - **布尔类型**: `return false`
    - **整数类型 (int8_t/16_t/32_t/64_t, uint8_t/16_t/32_t/64_t)**: `return 0`
    - **FP8 (cuda_fp8_e4m3)**: `return cuda_fp8_e4m3(0.0f)`，调用构造函数从浮点数转换
    - **FP16 (half)**: `return __float2half(0.0f)`，使用 CUDA 内建函数转换
    - **FP32 (float)**: `return 0.0f`
    - **FP64 (double)**: `return 0.0`
    - **BF16 (cuda_bfloat16)**: `return __float2bfloat16(0.0f)`，使用 CUDA 内建函数转换
    - **默认分支**: `return 0.0`（fallback，保持类型安全）
  - **性能**: 所有分支均为编译期常量，零运行时开销
  - **内存访问**: 不读取参数 `x`，仅用于满足统一接口

### `op::elementwise::nvidia::DeviceImpl`
- **位置**: `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/nvidia/elementwise_nvidia.cuh`
- **主要功能**: 封装 CUDA 内核启动逻辑，管理元数据从主机到设备的传输
- **实现模式**: Pimpl（Pointer to Implementation）模式，通过 `std::shared_ptr<Opaque>` 隐藏实现细节
- **生命周期**: 由 `Descriptor` 独占管理，使用 `std::unique_ptr`

**核心内部类 `Opaque`**:

- **成员**:
  - `std::shared_ptr<device::nvidia::Handle::Internal> internal`: CUDA 设备句柄的内部实现

- **关键方法**:
  - **`calculateImpl<BLOCK_SIZE, N, Op, Tdata, Args...>()`**:
    - **功能**: 执行类型统一的逐元素操作
    - **模板参数**:
      - `BLOCK_SIZE`: CUDA 块大小（Zeros 使用 256）
      - `N`: 输入张量数量（Zeros 为 1）
      - `Op`: 操作符类型（`cuda::ZerosOp`）
      - `Tdata`: 数据类型（如 `float`, `int32_t` 等）
    - **实现**: 转发到 `launchElementwiseKernel`，传递类型特化的内核函数指针

  - **`infoToDevice<N>()`**:
    - **功能**: 将逐元素操作的元数据和输入指针数组从主机内存异步复制到设备内存
    - **内存布局**:
      ```
      workspace 开始位置：
      [输入指针数组 (N * sizeof(void*))]
      [输出形状 (ndim * sizeof(size_t))]
      [输出步长 (ndim * sizeof(ptrdiff_t))]
      [所有输入形状 (N * ndim * sizeof(size_t))]
      [所有输入步长 (N * ndim * sizeof(ptrdiff_t))]
      [输入连续性标志 (N * sizeof(bool))]
      [输入广播标志 (N * sizeof(bool))]
      ```
    - **步骤**:
      1. 在设备内存中计算各段的指针偏移
      2. 调用 `cudaMemcpyAsync` 异步复制输入指针数组
      3. 异步复制元数据（形状、步长、标志）
    - **CUDA API**: `cudaMemcpyAsync(..., cudaMemcpyHostToDevice, stream)`
    - **错误处理**: 使用 `CHECK_CUDA` 宏检查复制状态

  - **`launchElementwiseKernel<BLOCK_SIZE, N, KernelFunc, Tout, Args...>()`**:
    - **功能**: 配置 CUDA 网格和块维度，启动逐元素操作内核
    - **网格配置策略**:
      - 块维度: `min(BLOCK_SIZE, maxThreadsPerBlock)`，确保不超过硬件限制
      - 网格维度: `min(CEIL_DIV(output_size, BLOCK_SIZE), gridSizeX)`，支持大张量的分块处理
      - 步长: `gridDims.x * blockDims.x`，用于多次内核启动的索引偏移
    - **多轮启动**: 当 `output_size > gridDims.x * blockDims.x` 时，循环启动多次内核，每次处理 `step` 个元素
    - **内核调用**: `kernel_func<<<gridDims, blockDims, 0, stream>>>(..., offset)`
    - **边界检查**: 如果 `output_size == 0`，直接返回成功

### `op::elementwise::ElementwiseInfo`
- **位置**: `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/elementwise.h`
- **主要功能**: 结构体，存储逐元素操作的元数据（形状、步长、广播、连续性）
- **内存管理**: 手动管理，使用 `std::vector<size_t>` 作为底层存储
- **不可复制**: 仅支持移动构造，禁用拷贝和移动赋值

**核心字段**:
- `std::vector<size_t> _meta`: 打包存储所有元数据的连续内存
- `size_t _output_size`: 输出张量的元素总数
- `size_t _input_size`: 输入张量的数量
- `size_t _ndim`: 张量的维度数
- `bool _output_contiguous`: 输出张量是否在内存中连续

**访问方法**:
- `getMetaMemSize()`: 返回元数据占用字节数
- `getOutputSize()`: 返回输出张量元素数量
- `getInputSize()`: 返回输入张量数量
- `getNdim()`: 返回张量维度数
- `getOutputShape()`: 返回输出形状数组指针
- `getOutputStrides()`: 返回输出步长数组指针
- `getInputShape(index)`: 返回第 `index` 个输入的形状数组指针
- `getInputStrides(index)`: 返回第 `index` 个输入的步长数组指针
- `getInputContiguous()`: 返回输入连续性标志数组指针
- `getInputBroadcasted()`: 返回输入广播标志数组指针

**工厂方法 `create(output_desc, input_descs)`**:
- **功能**: 从张量描述符构造 `ElementwiseInfo`
- **验证**:
  - 输出描述符非空且输入描述符非空
  - 输出张量不能有广播维度（`hasBroadcastDim()`）
- **内存分配**: 计算总元数据大小并分配 `std::vector<size_t>`（向上取整到 `size_t` 边界）
- **数据复制**: 使用 `std::memcpy` 高效复制形状、步长数组
- **广播检测**: `input_broadcasted[i] = !input_contiguous[i] && (desc->ndim() != ndim || desc->hasBroadcastDim())`
- **返回**: `Result<ElementwiseInfo>` 类型的值对象

### CUDA 内核函数

#### `elementwiseKernel<N, Op, Tdata, Args...>`
- **位置**: `elementwise_nvidia.cuh` 第 104-133 行
- **功能**: 执行输入类型统一的逐元素操作内核
- **模板参数**:
  - `N`: 输入张量数量（编译期常量）
  - `Op`: 操作符 functor 类型
  - `Tdata`: 统一的数据类型
  - `Args...`: 额外的内核参数类型
- **内核签名**:
  ```cuda
  __global__ void elementwiseKernel(
      size_t output_size, size_t ndim, bool output_contiguous,
      const bool *__restrict__ input_contiguous,
      const bool *__restrict__ input_broadcasted,
      const size_t *__restrict__ output_shape,
      const size_t *__restrict__ input_shapes,
      const ptrdiff_t *__restrict__ output_strides,
      const ptrdiff_t *__restrict__ input_strides,
      Tdata *output, const void *const *inputs,
      size_t offset, Args... args)
  ```
- **执行逻辑**:
  1. 计算全局线程索引: `idx = blockIdx.x * blockDim.x + threadIdx.x + offset`
  2. 边界检查: `if (idx < output_size)`
  3. 输出索引计算: 调用 `getOutputIndex()`，如果是连续张量则直接使用 `idx`，否则调用 `indexToOffset` 进行多维索引映射
  4. 构建 `InputIndexer` 结构体，封装输入索引计算逻辑
  5. 使用 `unpackInputsAndApply` 和 `std::make_index_sequence<N>{}` 展开编译期索引序列
  6. 调用 `Op{}(typed_inputs[Is.value][indexer(Is.value)]..., args...)` 执行操作并写入输出
- **性能优化**: `__restrict__` 关键字提示编译器指针不重叠，启用更激进的优化

#### `elementwiseKernel<Op, Tout, Tin...>`
- **位置**: `elementwise_nvidia.cuh` 第 156-184 行
- **功能**: 执行支持混合类型的逐元素操作内核（Zeros 操作不使用此版本）
- **差异**: 支持输出和输入使用不同数据类型，使用 `typedInputPtr<Tin>` 为每个输入进行类型转换
- **调用**: `Op{}.template operator()<Tout, Tin...>((typed_inputs[Is.value][indexer(Is.value)])...)`

#### 辅助设备函数

**`typedInputPtr<T>(ptr)`**:
- **功能**: 将 `void*` 无类型指针转换为 `const T*` 类型指针
- **属性**: `__device__ __forceinline__`，确保在设备端执行并强制内联

**`getOutputIndex(idx, is_contiguous, ndim, shape, strides)`**:
- **功能**: 计算输出张量的内存偏移索引
- **优化路径**: 如果 `is_contiguous` 为 true，直接返回 `idx`，避免复杂计算
- **通用路径**: 调用 `device::nvidia::indexToOffset(idx, ndim, shape, strides)` 进行线性索引到多维索引的映射

**`InputIndexer::operator()(input_id)`**:
- **功能**: 计算第 `input_id` 个输入张量的内存偏移
- **优化路径**: 如果 `input_contiguous[input_id]` 为 true，直接返回 `idx`
- **通用路径**: 调用 `device::nvidia::indexToOffset` 处理广播和步长

**`unpackInputsAndApply<F>(f, index_sequence)`**:
- **功能**: 编译期辅助函数，将索引序列展开为可调用对象的参数
- **实现**: 使用 C++14 的 `std::integral_constant` 和折叠表达式在编译期展开变参模板

## 3. API 接口

```cpp
namespace op::zeros::nvidia {

// 描述符类（由宏生成）
class Descriptor final : public InfiniopDescriptor {
public:
    ~Descriptor();

    size_t workspaceSize() const;

    // 创建描述符的静态工厂方法
    static infiniStatus_t create(
        infiniopHandle_t handle,                  // [in] CUDA 设备句柄
        Descriptor **desc_ptr,                    // [out] 输出描述符指针
        infiniopTensorDescriptor_t output_desc,   // [in] 输出张量描述符
        std::vector<infiniopTensorDescriptor_t> input_descs); // [in] 输入张量描述符向量

    // 执行 Zeros 操作
    infiniStatus_t calculate(
        void *workspace,              // [in] 设备工作空间指针
        size_t workspace_size,        // [in] 工作空间大小（字节）
        void *output,                 // [out] 输出张量设备指针
        std::vector<const void *> inputs, // [in] 输入张量设备指针向量（未使用）
        void *stream) const;          // [in] CUDA 流指针
};

} // namespace op::zeros::nvidia
```

## 4. 使用示例

```cpp
#include "infiniop/zeros_nvidia.cuh"

// 1. 创建 CUDA 设备句柄
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_NVIDIA, 0);

// 2. 定义张量形状和数据类型
std::vector<size_t> shape = {1024, 1024};
infiniDtype_t dtype = INFINI_DTYPE_F32;

// 3. 创建输出张量描述符
infiniopTensorDescriptor_t output_desc;
infiniopCreateTensorDescriptor(&output_desc, dtype, shape.data(), shape.size());

// 4. 创建输入张量描述符（Zeros 需要一个占位输入，形状必须与输出一致）
std::vector<infiniopTensorDescriptor_t> input_descs = {output_desc};

// 5. 创建 Zeros 操作描述符
op::zeros::nvidia::Descriptor *zeros_desc = nullptr;
infiniStatus_t status = op::zeros::nvidia::Descriptor::create(
    handle, &zeros_desc, output_desc, input_descs);

if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 6. 分配设备内存
void *d_output;
size_t output_size = output_desc->numel() * sizeof(float);
cudaMalloc(&d_output, output_size);

// 7. 分配工作空间（存储元数据和输入指针）
size_t workspace_size = zeros_desc->workspaceSize();
void *d_workspace;
cudaMalloc(&d_workspace, workspace_size);

// 8. 创建 CUDA 流
cudaStream_t stream;
cudaStreamCreate(&stream);

// 9. 执行 Zeros 操作
std::vector<const void *> inputs = {nullptr}; // 输入未使用，可传入任意指针
status = zeros_desc->calculate(d_workspace, workspace_size, d_output, inputs, stream);

// 10. 同步并检查结果
cudaStreamSynchronize(stream);

// 11. 清理资源
cudaFree(d_output);
cudaFree(d_workspace);
cudaStreamDestroy(stream);
delete zeros_desc;
infiniopDestroyTensorDescriptor(output_desc);
infiniopDestroyHandle(handle);
```

**典型输出结果**:
- 输出张量中所有元素均为 0.0f（float 类型）或对应类型的零值表示

## 5. 实现细节

### 内存管理

- **元数据打包**: `ElementwiseInfo` 使用单个 `std::vector<size_t>` 打包存储所有形状、步长、标志信息，减少内存碎片和分配次数
- **工作空间布局**: 设备端工作空间采用分段布局，依次存储输入指针数组和元数据，通过指针算术计算各段起始地址
- **异步传输**: 所有主机到设备的内存复制使用 `cudaMemcpyAsync`，支持与主机端计算和内核执行的重叠
- **智能指针**: `Descriptor` 使用 `std::unique_ptr` 管理 `DeviceImpl`，`DeviceImpl` 使用 `std::shared_ptr` 管理 `Opaque`，确保异常安全和自动清理

### 并发控制

- **CUDA 流**: 所有内核启动和内存传输都通过用户提供的 `stream` 参数，支持多个操作序列在不同流中并发执行
- **无同步点**: 内核启动后立即返回，不调用 `cudaStreamSynchronize`，允许主机端继续调度工作
- **流隔离**: 不同流中的操作可以并发执行，只要它们访问不同的内存区域

### 性能优化

- **编译期常量**: 块大小（256）和输入数量（1）均为编译期常量，允许编译器完全展开循环和优化寄存器分配
- **类型特化**: 为每种数据类型生成独立的内核实例，避免运行时类型分支
- **零拷贝优化**: `ZerosOp::operator()` 不读取输入参数，编译器可优化掉输入张量的全局内存加载
- **连续路径**: 当输出张量连续时，`getOutputIndex` 直接返回线性索引，避免昂贵的 `indexToOffset` 计算
- **块大小选择**: 使用 256 线程/块，在现代 GPU 上平衡了占用率（occupancy）和块调度开销
- **网格分割**: 对于超大张量（元素数 > gridSizeX * blockSize），循环多次启动内核，每次处理 `gridDims.x * blockDims.x` 个元素，避免网格维度超出硬件限制

### 错误处理

- **状态码**: 使用 `infiniStatus_t` 枚举返回详细错误信息
  - `INFINI_STATUS_SUCCESS`: 操作成功
  - `INFINI_STATUS_BAD_PARAM`: 输入参数为空指针
  - `INFINI_STATUS_BAD_TENSOR_DTYPE`: 不支持的数据类型
  - `INFINI_STATUS_BAD_TENSOR_STRIDES`: 输出张量有广播维度（不允许）
  - `INFINI_STATUS_INSUFFICIENT_WORKSPACE`: 工作空间大小不足
  - `INFINI_STATUS_NOT_IMPLEMENTED`: 复数类型未实现
- **宏封装**: `CHECK_CUDA` 和 `CHECK_STATUS` 宏简化错误检查代码，自动将 CUDA 错误转换为 `infiniStatus_t`
- **早期返回**: 在 `create` 和 `calculate` 方法中使用早期返回模式，失败时立即返回错误码，避免继续执行

### 设计模式

- **CRTP（奇异递归模板模式）的简化**: `ELEMENTWISE_DESCRIPTOR` 宏为每个操作生成命名空间特化的 `Descriptor` 类，避免代码重复
- **Functor 模式**: `ZerosOp` 定义 `operator()` 使其可作为可调用对象传递给模板算法
- **工厂模式**: `Descriptor::create` 静态方法封装复杂的对象构造逻辑，提供清晰的错误处理
- **Pimpl 模式**: `DeviceImpl` 通过 `Opaque` 内部类隐藏实现细节，减少头文件依赖
- **策略模式**: `ElementwiseInfo` 封装不同张量布局（连续/非连续、广播/非广播）的索引计算策略

### 依赖关系

- **内部依赖**:
  - `op::elementwise::ElementwiseInfo`: 逐元素操作元数据
  - `op::elementwise::nvidia::DeviceImpl`: CUDA 设备实现框架
  - `device::nvidia::Handle`: CUDA 设备句柄
  - `device::nvidia::indexToOffset`: 多维索引到内存偏移的映射函数
  - `cuda::ZerosOp`: CUDA 设备端操作符实现
- **外部依赖**:
  - CUDA Runtime API (`cudaMemcpyAsync`, `cudaMalloc`, `cudaStream_t`)
  - CUDA 内建函数 (`__float2half`, `__float2bfloat16`)
  - C++ 标准库 (`std::vector`, `std::unique_ptr`, `std::shared_ptr`)
  - Infini 通用工具 (`utils::Result`, `CHECK_CUDA`, `INFINIOP_CUDA_KERNEL`)

### 广播语义

- **定义**: 广播允许小张量在逐元素操作中自动扩展以匹配大张量的形状
- **实现**: `InputIndexer` 在运行时为每个输入张量计算正确的内存偏移，处理维度大小为 1 的情况
- **限制**: Zeros 操作的输入和输出形状必须完全一致（`CHECK_SAME_SHAPE`），不支持广播
- **元数据**: `input_broadcasted[i]` 标志指示第 `i` 个输入是否需要广播处理

### 类型安全

- **编译期检查**: `static_assert(sizeof...(Tin) == Op::num_inputs)` 确保模板参数数量匹配
- **SFINAE**: 使用 `std::enable_if_t` 约束模板函数重载，区分统一类型和混合类型版本
- **类型推导**: `calculate` 方法根据 `_dtype` 成员变量自动选择正确的模板实例
- **零成本抽象**: 所有类型分支在编译期解析，运行时无虚函数或类型判断开销

### 数据类型支持细节

- **FP8 (cuda_fp8_e4m3)**: 8 位浮点格式，符号位 1 位，指数 4 位，尾数 3 位，用于深度学习量化和推理加速
- **FP16 (half)**: 半精度浮点，IEEE 754 标准，16 位存储
- **BF16 (cuda_bfloat16)**: 脑浮点格式，16 位存储，与 FP32 相同的指数范围（8 位），尾数 7 位
- **FP8/FP16/BF16 零值构造**: 使用 `__float2half` 和 `__float2bfloat16` 内建函数从 32 位浮点数转换，确保位模式正确
- **整数类型**: 所有有符号和无符号整数类型的零值均为 0，使用 `return 0` 自动适应类型宽度

### 内核启动开销分析

- **网格配置**: 使用 `min(CEIL_DIV(output_size, 256), gridSizeX)` 计算网格大小，确保不超过设备限制
- **多轮策略**: 对于超大张量（如形状 {65536, 65536}，元素数 4.29e9），循环多次启动内核，每轮最多处理 `gridSizeX * 256` 个元素
- **偏移传递**: 每轮内核的 `offset` 参数累加 `step`，确保线程索引覆盖整个输出范围
- **开销权衡**: 多轮启动增加了内核启动开销，但避免了超出网格维度限制，在张量尺寸大于 2^27 时必要

### 与其他操作的集成

- **逐元素操作框架**: Zeros 操作完全复用通用的逐元素操作基础设施，仅通过 `ZerosOp` functor 区分语义
- **组合性**: 可与其他逐元素操作（如 Add, Mul）组合形成复杂表达式图
- **图优化**: 在计算图编译阶段，Zeros 操作可被识别为常量初始化，允许编译器进行死代码消除和内存复用优化
