# Moore Elementwise Operations Core Implementation Documentation

该模块实现了针对 Moore（Moore Threads GPU）硬件的逐元素运算（elementwise operations）后端，支持张量广播、非连续内存布局和多种数据类型的高性能并行计算。这是 Infini 框架中 elementwise 操作在 Moore GPU 架构上的具体实现，通过 MUSA（Moore Unified Streaming Architecture）编程模型提供 GPU 加速能力。

## 1. Module Structure

- **`elementwise_moore_api.h`**: 定义公共 API 接口和设备实现类的声明，包含类型擦除的设备句柄和计算调度接口
- **`elementwise_moore.h`**: 核心实现文件，包含 GPU kernel 函数、内存管理、kernel 启动逻辑和索引计算工具

## 2. Core Classes

### `DeviceImpl`
- **Location**: `elementwise_moore_api.h`
- **Primary Function**: 提供类型擦除的设备实现接口，作为用户 API 与底层 GPU 实现之间的桥梁。采用 Pimpl（Pointer to Implementation）模式隐藏实现细节。
- **Key Members**:
  - `struct Opaque`: 前向声明的内部实现结构体，封装所有具体实现细节
  - `std::shared_ptr<Opaque> _opaque`: 指向内部实现的智能指针，管理对象生命周期
- **Core Methods**:
  - `static utils::Result<DeviceImpl *> create(Args &&...args)`: 工厂方法，构造 DeviceImpl 实例并返回 Result 包装的指针
  - `template <uint32_t BLOCK_SIZE, typename Op, typename Tdata, typename... Args> calculate(...)`: 单一数据类型版本的逐元素运算入口，适用于所有输入输出类型相同的场景
  - `template <uint32_t BLOCK_SIZE, typename Op, typename Tout, typename... Tin, typename... Args> calculate(...)`: 多数据类型版本的逐元素运算入口，支持输入输出类型不同，通过 SFINAE 约束输入类型数量与操作定义匹配
- **Lifecycle**: 使用 `std::shared_ptr` 管理 `Opaque` 内部对象，采用 Pimpl 模式确保二进制兼容性和编译期依赖隔离

### `DeviceImpl::Opaque`
- **Location**: `elementwise_moore.h`
- **Primary Function**: 实际执行 GPU kernel 启动和内存管理的核心实现类，负责将高级别的 ElementwiseInfo 转换为设备端可执行的 kernel 参数
- **Key Members**:
  - `std::shared_ptr<device::moore::Handle::Internal> internal`: Moore 设备句柄，封装 MUSA 上下文、流和设备属性
- **Core Methods**:
  - `calculateImpl(...)`: 模板方法，根据数据类型统一性分发到两种不同实现（同质类型 vs 异质类型），实际调用 `launchElementwiseKernel`
  - `launchElementwiseKernel(...)`: 核心启动逻辑，执行以下步骤：
    1. 将主机端元数据（形状、步长、广播标志）异步拷贝到设备端 workspace
    2. 使用 `infoToDevice` 方法在设备内存中布局元数据结构
    3. 计算网格和块维度：`blockDims.x = min(BLOCK_SIZE, maxThreadsPerBlock)`，`gridDims.x = min(ceil(output_size/blockDims.x), gridSizeX)`
    4. 使用循环分步策略处理大型张量（step = gridDims.x * blockDims.x），每次处理一个 grid 的工作量
    5. 启动 CUDA kernel：`kernel_func<<<gridDims, blockDims, 0, stream>>>(...)`
  - `infoToDevice(...)`: 私有辅助方法，将 ElementwiseInfo 的元数据打包到设备端的连续内存布局：
    - 输入指针数组（N 个指针）
    - 输出形状（ndim 个 size_t）
    - 输出步长（ndim 个 ptrdiff_t）
    - 所有输入形状（N * ndim 个 size_t）
    - 所有输入步长（N * ndim 个 ptrdiff_t）
    - 输入连续性标志（N 个 bool）
    - 输入广播标志（N 个 bool）
- **Memory Strategy**: 使用用户提供的 workspace 缓冲区临时存储设备端元数据，避免动态分配。workspace 布局：`[输入指针数组 | 元数据区域]`

### `InputIndexer`
- **Location**: `elementwise_moore.h`
- **Primary Function**: 设备端索引计算函数对象，根据输入张量的连续性和广播属性，将线性输出索引映射到各输入张量的正确内存偏移
- **Key Members**:
  - `size_t idx`: 当前元素的线性索引
  - `size_t ndim`: 张量维度数
  - `const bool *input_contiguous`: 各输入张量的连续性标志数组
  - `const bool *input_broadcasted`: 各输入张量的广播标志数组
  - `const size_t *input_shapes`: 所有输入张量的形状数据（N * ndim）
  - `const ptrdiff_t *input_strides`: 所有输入张量的步长数据（N * ndim）
  - `const ptrdiff_t *output_strides`: 输出张量的步长（用于广播计算）
- **Core Methods**:
  - `__device__ __forceinline__ size_t operator()(size_t input_id) const`: 根据 `input_id` 返回对应输入张量的偏移量。如果输入连续则直接返回 `idx`，否则调用 `device::moore::indexToOffset` 进行多维索引计算
- **Algorithm**: O(ndim) 索引转换，通过模运算和除法从线性索引恢复各维度坐标，再乘以对应步长求和

## 3. API Interface

```cpp
// 创建 elementwise 描述符的宏（用于具体算子的工厂函数）
#define CREATE_ELEMENTWISE_MOORE_DESCRIPTOR(HANDLE, DTYPE, OUT_DESC, INPUT_DESC_VEC)
// 功能：自动化描述符创建流程
// 1. 从张量描述符构造 ElementwiseInfo
// 2. 计算 workspace 大小 = 元数据大小 + 输入指针数组大小
// 3. 创建 Moore DeviceImpl 实例
// 4. 构造并返回 Descriptor 对象

// DeviceImpl 核心计算接口
template <uint32_t BLOCK_SIZE, typename Op, typename Tdata, typename... Args>
infiniStatus_t calculate(
    const op::elementwise::ElementwiseInfo &info,  // 张量元数据（形状、步长、广播信息）
    void *workspace,                                // 设备端临时内存，用于存储元数据
    void *output,                                   // 输出张量的设备指针
    const std::vector<const void *> &inputs,        // 输入张量的设备指针数组
    void *stream,                                   // MUSA 执行流
    Args &&...args);                                // 额外的算子参数（如标量值）

// 类型特化版本（输入输出类型不同）
template <uint32_t BLOCK_SIZE, typename Op, typename Tout, typename... Tin, typename... Args>
infiniStatus_t calculate(..., std::enable_if_t<(sizeof...(Tin) == Op::num_inputs), int> = 0);
```

## 4. Usage Example

```cpp
// 示例：在 Moore GPU 上执行逐元素加法 C = A + B
// 假设已有张量描述符：output_desc, input_a_desc, input_b_desc

// 1. 创建 ElementwiseInfo（自动提取形状、步长、广播信息）
auto info_result = op::elementwise::ElementwiseInfo::create(
    output_desc,
    {input_a_desc, input_b_desc}
);
CHECK_RESULT(info_result);
auto info = info_result.take();

// 2. 创建 Moore 设备实现
auto device_impl_result = op::elementwise::moore::DeviceImpl::create(handle->internal());
CHECK_RESULT(device_impl_result);
auto* device_impl = device_impl_result.take();

// 3. 分配 workspace（存储设备端元数据）
size_t workspace_size = info.getMetaMemSize() + info.getInputSize() * sizeof(void*);
void* workspace;
musaMalloc(&workspace, workspace_size);

// 4. 准备数据指针
std::vector<const void*> inputs = {d_tensor_a, d_tensor_b};  // 设备端指针
void* d_output;  // 设备端输出指针
musaStream_t stream;

// 5. 执行计算（BLOCK_SIZE=1024，Op=Add，Tdata=float）
infiniStatus_t status = device_impl->calculate<1024, AddOp, float>(
    info, workspace, d_output, inputs, stream
);

// 6. 清理资源
musaFree(workspace);
delete device_impl;
```

## 5. Implementation Details

### Memory Management
- **Workspace Layout**: 线性布局的设备内存，分为两个区域：
  1. 输入指针数组：`N * sizeof(void*)` 字节，存储各输入张量的设备地址
  2. 元数据区域：从 `workspace + N*sizeof(void*)` 开始，存储 `ElementwiseInfo` 的所有元数据
- **Host-to-Device Transfer**: 使用 `musaMemcpyAsync` 异步拷贝元数据，通过 stream 实现与计算的潜在重叠
- **Pointer Arithmetic**: 通过偏移量计算在连续 workspace 中定位各类元数据，避免多次小内存拷贝

### Concurrency
- **Kernel Launch**: 使用 CUDA/MUSA 的网格-块（grid-block）并行模型
- **Block Size**: 模板参数 `BLOCK_SIZE`（典型值 512/1024/2048），通过 `std::min(BLOCK_SIZE, maxThreadsPerBlock)` 适应硬件限制
- **Grid Size**: 动态计算 `min(ceil(output_size/blockDims.x), gridSizeX)`，确保不超过硬件网格维度限制
- **Stepping Strategy**: 循环启动 kernel 处理超大型张量：`for (size_t i = 0; i < output_size; i += step)`，每次处理 `gridDims.x * blockDims.x` 个元素

### Performance
- **Divergence Avoidance**: 对于连续张量使用直接索引（`idx`），避免索引计算的线程分化
- **Broadcast Optimization**: 通过预计算 `input_contiguous` 和 `input_broadcasted` 标志，在 kernel 中快速选择索引策略
- **Type Specialization**: 提供同质类型（`Tdata`）和异质类型（`Tout`, `Tin...`）两个模板分支，编译器可生成最优化的类型特化代码
- **Index-to-Offset Algorithm**: O(ndim) 复杂度的多维索引转换，通过反向迭代（`i-- > 0`）从最低维度开始计算，利用整数除法和模运算

### Error Handling
- **Result Type**: 使用 `utils::Result<T>` 包装返回值，区分成功状态和错误码
- **CHECK_RESULT Macro**: 宏展开检查 `Result` 状态，失败时提前返回
- **CHECK_MOORE Macro**: 检查 MUSA API 调用状态，失败时返回对应错误码
- **Static Assertions**: 编译期验证输入类型数量与操作定义的 `num_inputs` 匹配

### Dependencies
- **MUSA Runtime**: `musa_runtime_api.h` 提供核心 API（`musaMemcpyAsync`, kernel 启动语法）
- **Moore Device Handle**: `device::moore::Handle::Internal` 封装设备属性（`maxThreadsPerBlock`, `gridSizeX`）
- **ElementwiseInfo**: 从 `elementwise.h` 导入，存储张量布局和广播信息
- **Type Traits**: 使用 `std::enable_if_t` 和 SFINAE 实现编译期类型分发

### Design Patterns
- **Pimpl (Pointer to Implementation)**: `DeviceImpl` 通过不透明指针隐藏 `Opaque` 实现细节，减少编译依赖
- **Factory Method**: `DeviceImpl::create` 静态工厂方法封装对象构造逻辑
- **Strategy Pattern**: 两种 `calculate` 重载根据类型统一性选择不同的实现策略
- **Template Method**: `launchElementwiseKernel` 定义算法骨架，`infoToDevice` 实现具体步骤
- **Functor**: `InputIndexer` 重载 `operator()` 实现无状态索引计算函数对象

### Kernel Execution Flow
1. **Thread Identification**: `size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset`
2. **Bounds Checking**: `if (idx < output_size)` 确保不越界
3. **Index Calculation**:
   - 输出索引：`out_idx = is_contiguous ? idx : indexToOffset(idx, ndim, shape, strides)`
   - 输入索引：通过 `InputIndexer` 为每个输入计算独立偏移
4. **Operation Execution**: `output[out_idx] = Op{}(inputs[0][idx0], inputs[1][idx1], ...)`
5. **Compile-Time Dispatch**: `unpackInputsAndApply` 使用 `std::make_index_sequence<N>` 展开参数包，生成无分支代码
