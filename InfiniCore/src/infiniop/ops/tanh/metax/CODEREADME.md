# Tanh Operator METAX Backend Implementation Documentation

本模块实现了双曲正切（Tanh）激活函数的 METAX GPU 加速后端，支持华为昇腾（Metax）硬件架构的高性能张量计算。该实现采用逐元素操作框架，通过复用通用的 elementwise 基础设施实现了对 F16、BF16、F32、F64 数据类型的完整支持。

## 1. Module Structure

- **`tanh_metax.h`**: 操作符描述符的 API 定义，通过宏生成统一的 Descriptor 类接口
- **`tanh_metax.maca`**: 核心实现文件，包含算子创建、计算调度和设备端内核调用逻辑

## 2. Core Classes

### `op::tanh::metax::Descriptor`
- **Location**: 通过 `ELEMENTWISE_DESCRIPTOR(tanh, metax)` 宏在 `tanh_metax.h` 中生成
- **Primary Function**: Tanh 操作符的描述符类，封装了算子的元数据、设备实现和执行接口
- **Key Members**:
  - `_dtype`: `infiniDtype_t` - 存储输出张量的数据类型（F16/BF16/F32/F64）
  - `_info`: `op::elementwise::ElementwiseInfo` - 封装输入输出张量的形状、步幅、连续性等元数据
  - `_device_info`: `std::unique_ptr<op::elementwise::metax::DeviceImpl>` - METAX 设备端实现的智能指针
  - `_workspace_size`: `size_t` - 设备端工作空间大小（字节），用于存储元数据和输入指针数组
- **Core Methods**:
  - `create(infiniopHandle_t, Descriptor**, infiniopTensorDescriptor_t, std::vector<infiniopTensorDescriptor_t>)`:
    - **功能**: 静态工厂方法，构造并初始化 Tanh 描述符
    - **算法**: 通过 `CREATE_ELEMENTWISE_METAX_DESCRIPTOR` 宏创建 ElementwiseInfo、计算工作空间大小、初始化 METAX DeviceImpl
    - **参数验证**: 检查数据类型合法性（F16/BF16/F32/F64），验证输入输出张量形状一致性
    - **复杂度**: O(N)，其中 N 为张量维度数，用于遍历和复制元数据
  - `calculate(void *workspace, size_t workspace_size, void *output, std::vector<const void *> inputs, void *stream) const`:
    - **功能**: 执行 Tanh 计算的核心接口
    - **算法**: 根据数据类型分派到对应的模板特化，调用 `_device_info->calculate<256, cuda::TanhOp, T>()`
    - **工作空间**: 传入的 workspace 必须不小于 `_workspace_size`，否则返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
    - **数据流**: host → device（通过 `hcMemcpyAsync`）→ kernel 计算 → device → output
    - **复杂度**: O(E)，其中 E 为输出张量元素总数，每个元素执行一次 tanh 函数
- **Lifecycle**:
  - **构造**: 通过 `create()` 静态方法创建，采用移动语义转移 ElementwiseInfo 和 DeviceImpl 所有权
  - **析构**: 默认析构函数（`= default`），智能指针自动管理资源释放
  - **所有权**: Descriptor 拥有 ElementwiseInfo（值类型）和 DeviceImpl（唯一指针），生命周期由调用者控制

### `op::elementwise::metax::DeviceImpl`
- **Location**: `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/metax/elementwise_metax.h`
- **Primary Function**: METAX 后端设备端实现的通用容器，通过 Pimpl 惯例封装具体实现细节
- **Key Members**:
  - `_opaque`: `std::shared_ptr<Opaque>` - 不透明指针，指向实际的实现对象（Opaque 结构体）
- **Core Methods**:
  - `create(Args &&...args)`: 静态工厂方法，返回 `Result<DeviceImpl *>` 类型
  - `calculate<BLOCK_SIZE, Op, Tdata>(...)`: 当输入输出类型相同时的模板特化
  - `calculate<BLOCK_SIZE, Op, Tout, Tin...>(...)`: 当输入输出类型不同时的模板特化
- **Design Pattern**: Pimpl (Pointer to Implementation) 惯例，隔离设备特定实现与公共接口

### `op::elementwise::metax::DeviceImpl::Opaque`
- **Location**: `elementwise_metax.h` 中的内嵌结构体
- **Primary Function**: 实际的设备实现持有者，管理 METAX 句柄和内核启动逻辑
- **Key Members**:
  - `internal`: `std::shared_ptr<device::metax::Handle::Internal>` - METAX 设备句柄的内部表示
- **Core Methods**:
  - `calculateImpl<BLOCK_SIZE, N, Op, Tdata, Args...>(...)`:
    - **功能**: 调用 `launchElementwiseKernel` 启动 CUDA 内核
    - **模板参数**: BLOCK_SIZE=256（固定线程块大小），N=Op::num_inputs（输入张量数量，Tanh 为 1）
    - **内核函数**: `elementwiseKernel<N, Op, Tdata, Args...>`
  - `infoToDevice<N>(...)`:
    - **功能**: 将主机端的 ElementwiseInfo 元数据复制到设备端工作空间
    - **内存布局**: [输入指针数组 (N*sizeof(void*))] [output_shape (ndim*sizeof(size_t))] [output_strides (ndim*sizeof(ptrdiff_t))] [input_shapes (N*ndim*sizeof(size_t))] [input_strides (N*ndim*sizeof(ptrdiff_t))] [input_contiguous (N*sizeof(bool))] [input_broadcasted (N*sizeof(bool))]
    - **传输**: 使用 `hcMemcpyAsync` 进行异步内存拷贝，方向为 `hcMemcpyHostToDevice`
  - `launchElementwiseKernel<BLOCK_SIZE, N, KernelFunc, Tout, Args...>(...)`:
    - **功能**: 计算网格/块维度并启动 METAX 内核
    - **块维度**: `min(BLOCK_SIZE, maxThreadsPerBlock)`，通常为 256
    - **网格维度**: `min(ceil_div(output_size, blockDims.x), gridSizeX)`，确保不超过设备限制
    - **网格步长**: `gridDims.x * blockDims.x`，用于处理大张量的多轮启动
    - **启动循环**: `for (size_t i = 0; i < output_size; i += step)` 覆盖所有元素
    - **内核调用**: `kernel_func<<<gridDims, blockDims, 0, stream>>>(..., offset=i, ...)`

### `op::tanh::cuda::TanhOp`
- **Location**: `/home/qy/src/Infini/InfiniCore/src/infiniop/ops/tanh/cuda/kernel.cuh`
- **Primary Function**: 设备端函数对象（Functor），定义 Tanh 的元素级计算逻辑
- **Key Members**:
  - `num_inputs`: `static constexpr size_t = 1` - 标识此操作符为单输入操作
- **Core Methods**:
  - `operator()(const T &input) const`:
    - **功能**: 对标量输入执行 tanh 计算，支持多种数据类型
    - **类型分发**:
      - `half2`: 通过 `__half22float2` 转换为 float2，对每个分量调用 `tanh_f32_func`，再通过 `__float22half2_rn` 转回
      - `half`: 转换为 float 计算，通过 `__float2half_rn` 转回（舍入到最近偶数）
      - `cuda_bfloat162`: 分别提取高低半部分，转换为 float 计算，通过 `__floats2bfloat162_rn` 合并
      - `cuda_bfloat16`: 转换为 float 计算，通过 `__float2bfloat16_rn` 转回
      - `float`: 直接调用 `tanh_f32_func`，内部使用 `tanhf` 函数
      - `double`: 直接调用 `std::tanh`
      - **其他**: 回退到 `std::tanh`
    - **性能优化**: 向量化类型（half2/bfloat162）利用 SIMD 指令并行计算两个元素
  - `tanh_f32_func(float x) const`:
    - **功能**: 单精度浮点的 tanh 包装函数
    - **实现**: 调用标准数学库 `tanhf`（METAX 设备端实现）

## 3. API Interface

```cpp
// 创建 Tanh 描述符（工厂方法）
infiniStatus_t op::tanh::metax::Descriptor::create(
    infiniopHandle_t handle,                  // METAX 设备句柄
    Descriptor **desc_ptr,                    // 输出：描述符指针
    infiniopTensorDescriptor_t out_desc,      // 输出张量描述符
    std::vector<infiniopTensorDescriptor_t> input_desc_vec);  // 输入张量描述符列表（大小为1）

// 执行 Tanh 计算
infiniStatus_t op::tanh::metax::Descriptor::calculate(
    void *workspace,              // 设备端工作空间指针
    size_t workspace_size,        // 工作空间大小（字节）
    void *output,                 // 输出张量的设备指针
    std::vector<const void *> inputs,  // 输入张量的设备指针数组（大小为1）
    void *stream) const;          // METAX 流（hcStream_t）

// 查询所需工作空间大小
size_t op::tanh::metax::Descriptor::workspaceSize() const;

// 析构函数
op::tanh::metax::Descriptor::~Descriptor();
```

### 宏接口

```cpp
// 定义在 elementwise.h 中，用于生成 Descriptor 类
#define ELEMENTWISE_DESCRIPTOR(OP, NAMESPACE)

// 定义在 elementwise_metax_api.h 中，用于创建描述符的辅助宏
#define CREATE_ELEMENTWISE_METAX_DESCRIPTOR(HANDLE, DTYPE, OUT_DESC, INPUT_DESC_VEC)
```

## 4. Usage Example

```cpp
#include "tanh_metax.h"
#include <vector>

// 初始化 METAX 设备
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_METAX, 0);

// 准备张量描述符（假设输入形状为 [1024, 1024]）
int64_t shape[] = {1024, 1024};
int64_t strides[] = {1024, 1};
infiniopTensorDescriptor_t input_desc, output_desc;
infiniopCreateTensorDescriptor(&input_desc, INFINI_DTYPE_F32, 2, shape, strides);
infiniopCreateTensorDescriptor(&output_desc, INFINI_DTYPE_F32, 2, shape, strides);

// 创建 Tanh 描述符
op::tanh::metax::Descriptor* tanh_desc = nullptr;
infiniStatus_t status = op::tanh::metax::Descriptor::create(
    handle, &tanh_desc, output_desc, {input_desc});
if (status != INFINI_STATUS_SUCCESS) {
    // 错误处理
}

// 分配设备内存
float* d_input;
float* d_output;
hcMalloc((void**)&d_input, 1024 * 1024 * sizeof(float));
hcMalloc((void**)&d_output, 1024 * 1024 * sizeof(float));

// 分配工作空间（包含元数据 + 输入指针数组）
size_t workspace_size = tanh_desc->workspaceSize();
void* d_workspace;
hcMalloc(&d_workspace, workspace_size);

// 创建流
hcStream_t stream;
hcStreamCreate(&stream);

// 上传输入数据（假设 h_input 为主机数据）
hcMemcpyAsync(d_input, h_input, 1024 * 1024 * sizeof(float), hcMemcpyHostToDevice, stream);

// 执行 Tanh 计算
std::vector<const void*> inputs = {d_input};
status = tanh_desc->calculate(d_workspace, workspace_size, d_output, inputs, stream);

// 下载结果
hcMemcpyAsync(h_output, d_output, 1024 * 1024 * sizeof(float), hcMemcpyDeviceToHost, stream);
hcStreamSynchronize(stream);

// 清理资源
delete tanh_desc;
hcFree(d_input);
hcFree(d_output);
hcFree(d_workspace);
hcStreamDestroy(stream);
infiniopDestroyTensorDescriptor(input_desc);
infiniopDestroyTensorDescriptor(output_desc);
infiniopDestroyHandle(handle);
```

## 5. Implementation Details

### Memory Management
- **工作空间布局**: 紧凑打包的元数据结构，避免多次内存分配。计算为 `info.getMetaMemSize() + info.getInputSize() * sizeof(void*)`，其中 `getMetaMemSize()` 包含所有形状、步幅、连续性和广播标志的存储空间。
- **所有权语义**: Descriptor 使用移动语义接收 ElementwiseInfo 和 DeviceImpl，避免不必要的拷贝。DeviceImpl 采用 Pimpl 惯例，通过 shared_ptr 管理 Opaque 对象生命周期。
- **设备端内存**: 通过 `hcMallocAsync` 分配（由调用者负责），内核执行期间保持有效。
- **异步传输**: 使用 `hcMemcpyAsync` 进行主机到设备的元数据拷贝，允许与计算流水线重叠。

### Concurrency
- **流语义**: 所有操作在用户提供的 `hcStream_t` 流上执行，支持与其它 METAX 操作的并发。
- **线程安全**: Descriptor 对象在创建后不可变（const 成员函数），多个流可安全并发调用 `calculate()`，前提是使用不同的工作空间和输出缓冲区。
- **内核并发**: METAX 内核通过 `<<<grid, block, 0, stream>>>` 启动，允许同一设备上的多个内核在不同流上并发执行。
- **同步点**: 仅在需要主机端访问结果时调用 `hcStreamSynchronize`，否则允许流水线继续执行。

### Performance
- **线程块大小**: 固定为 256（`BLOCK_SIZE`），平衡了寄存器使用、warp 调度效率和占用率。
- **网格循环**: 对于大张量（超过 `gridSizeX * blockSizeX`），使用循环多次启动内核，每次处理一个网格步长的元素。
- **向量化**: TanhOp 的 half2 和 bfloat162 特化利用 SIMD 指令，每个指令处理两个 FP16 元素，理论吞吐量翻倍。
- **连续路径优化**: 对于连续张量，内核使用简单的线性索引（`idx`），避免调用昂贵的 `indexToOffset` 函数。
- **广播支持**: 通过 InputIndexer 灵活处理非连续和广播张量，但需要额外的模运算和除法计算索引。
- **时间复杂度**: O(E)，E 为输出元素数量，每个元素执行一次 tanh 计算。常数因子受数据类型影响（FP16/BF16 需类型转换）。

### Error Handling
- **返回值策略**: 所有 API 返回 `infiniStatus_t` 枚举，包括 `INFINI_STATUS_SUCCESS`、`INFINI_STATUS_BAD_PARAM`、`INFINI_STATUS_BAD_TENSOR_DTYPE`、`INFINI_STATUS_BAD_TENSOR_STRIDES`、`INFINI_STATUS_INSUFFICIENT_WORKSPACE` 等。
- **类型检查**: 在 `create()` 中通过 `CHECK_DTYPE` 宏验证数据类型，拒绝不支持的类型（如 INT32）。
- **形状验证**: 通过 `CHECK_SAME_SHAPE` 宏确保输入输出张量形状一致（逐元素操作的前提）。
- **工作空间验证**: 在 `calculate()` 中检查 `workspace_size < _workspace_size`，防止缓冲区溢出。
- **设备端检查**: 使用 `CHECK_METAX` 宏包装 METAX API 调用（如 `hcMemcpyAsync`），自动转换错误码。
- **结果类型**: `DeviceImpl::create()` 返回 `utils::Result<DeviceImpl *>`，使用值类型封装成功或失败状态，避免异常。

### Dependencies
- **外部库**:
  - `hcBLAS`/`mcBLAS`: METAX BLAS 库（通过 `device::metax::Handle` 间接使用）
  - `hcDNN`/`mcDNN`: METAX 深度学习库（间接依赖）
  - METAX 驱动运行时: `hcMalloc`, `hcMemcpyAsync`, `hcStreamCreate` 等 API
- **内部模块**:
  - `op::elementwise::ElementwiseInfo`: 元数据管理（`/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/elementwise.h`）
  - `op::elementwise::metax::DeviceImpl`: 通用 METAX 逐元素操作实现（`elementwise_metax.h`）
  - `device::metax::Handle`: METAX 设备句柄和属性查询（`/home/qy/src/Infini/InfiniCore/src/infiniop/devices/metax/metax_common.h`）
  - `op::tanh::cuda::TanhOp`: CUDA/METAX 兼容的设备端 Tanh 实现（`/home/qy/src/Infini/InfiniCore/src/infiniop/ops/tanh/cuda/kernel.cuh`）
- **工具函数**:
  - `CEIL_DIV`: 整数向上取整除法（定义在 `/home/qy/src/Infini/InfiniCore/src/utils.h`）
  - `device::metax::indexToOffset`: 将扁平索引转换为内存偏移量（`metax_kernel_common.h`）
  - `CHECK_RESULT`, `CHECK_STATUS`, `CHECK_DTYPE`, `CHECK_SAME_SHAPE`: 错误检查宏（全局工具）

### Design Patterns
- **CRTP (Curiously Recurring Template Pattern)**: `ELEMENTWISE_DESCRIPTOR` 宏通过命名空间和宏展开生成派生类，避免手动重复代码。
- **Pimpl (Pointer to Implementation)**: `DeviceImpl` 通过 `std::shared_ptr<Opaque>` 隐藏实现细节，减少编译依赖和二进制兼容性风险。
- **Strategy Pattern**: `calculate()` 方法根据数据类型（F16/BF16/F32/F64）分派到不同的模板实例化，每个实例化使用特定的类型和计算路径。
- **Template Method Pattern**: `launchElementwiseKernel` 定义内核启动的骨架算法，`calculateImpl` 提供具体内核函数和模板参数。
- **Factory Pattern**: `create()` 静态方法作为工厂，封装复杂的对象构造逻辑（元数据计算、工作空间分配、设备实现初始化）。
- **RAII (Resource Acquisition Is Initialization)**: Descriptor 和 ElementwiseInfo 使用析构函数自动管理内存（虽然 ElementwiseInfo 使用手动管理的 `std::vector<size_t>`，但通过移动语义避免拷贝）。
- **Functor Pattern**: `TanhOp` 重载 `operator()`，可作为模板参数传递给通用内核，实现类型擦除和高阶函数。

### Type System
- **数据类型支持**:
  - `INFINI_DTYPE_F16`: 16 位浮点（half），通过 `__half` 和 CUDA intrinsics 优化
  - `INFINI_DTYPE_BF16`: 16 位脑浮点（bfloat16），通过 `hpcc_bfloat16` 实现
  - `INFINI_DTYPE_F32`: 32 位单精度浮点（float），直接使用 `tanhf`
  - `INFINI_DTYPE_F64`: 64 位双精度浮点（double），使用 `std::tanh`
- **类型别名**:
  - `cuda_bfloat16` → `hpcc_bfloat16`（METAX 兼容层）
  - `cuda_bfloat162` → `hpcc_bfloat162`（向量化 bfloat16）
  - `INFINIOP_METAX_KERNEL` → `__global__ void`（内核函数标记）
- **模板特化**: 根据输入输出类型是否相同，选择不同的 `DeviceImpl::calculate` 特化，避免不必要的类型转换。

### Hardware Considerations
- **METAX 架构适配**:
  - 块大小限制: `internal->maxThreadsPerBlock()` 动态查询设备属性
  - 网格大小限制: `internal->gridSizeX()` 限制 X 维度网格大小
  - Warp 大小: 通过 `internal->warpSize()` 查询（通常为 32），影响分支 divergent 性能
- **内存合并**: 内核使用 `InputIndexer` 计算偏移，对于连续张量保证内存访问合并；对于广播张量可能产生跨步访问。
- **计算吞吐**: FP16/BF16 通过类型转换在 FP32 单元上计算，实际吞吐取决于硬件对 FP16 的原生支持。
- **寄存器压力**: `elementwiseKernel` 模板实例化后，每个线程需要存储输入指针、形状、步幅等局部变量，高占用率可能受限于寄存器数量。

### 浮点语义
- **精度**:
  - FP32: IEEE 754 单精度，约 7 位十进制有效数字
  - FP16: IEEE 754 半精度，约 3-4 位十进制有效数字，指数范围有限
  - BF16: 脑浮点，与 FP32 相同的指数范围（8 位），但尾数仅 7 位
  - FP64: IEEE 754 双精度，约 15-16 位十进制有效数字
- **舍入模式**: 使用 "round to nearest even"（RN 模式），通过 `__float2half_rn`、`__float22half2_rn` 等函数确保符合 IEEE 754 标准。
- **特殊值**: `tanh` 函数对所有有限输入返回 (-1, 1) 范围内的值，对于 `±inf` 返回 `±1`，对于 `NaN` 返回 `NaN`。

### 代码复用与模块化
- **零代码抽象**: `tanh_metax.maca` 仅 60 行代码，通过复用通用的 elementwise 框架和 TanhOp functor，避免重复实现内存管理、内核启动、类型分发等逻辑。
- **宏驱动设计**: `ELEMENTWISE_DESCRIPTOR` 和 `CREATE_ELEMENTWISE_METAX_DESCRIPTOR` 宏消除了数十个逐元素操作符（Relu、Sigmoid、Gelu 等）的样板代码。
- **跨后端共享**: `TanhOp` 定义在 `cuda/` 目录下，但被 METAX 后端复用（通过类型别名 `cuda_bfloat16` → `hpcc_bfloat16`），体现了逻辑与设备的解耦。
- **模板元编程**: 大量使用 `constexpr if`（`if constexpr`）、`std::enable_if_t`、`std::index_sequence` 等技术在编译期生成类型特化代码，避免运行时分支。
