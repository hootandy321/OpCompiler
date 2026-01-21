# Ones MetAX Core Implementation Documentation

本模块实现基于 MetAX 设备（华为昇腾 NPU 架构）的张量全一填充操作，通过统一的逐元素运算框架为 InfiniOP 提供 device-specific 的 ones 算子实现。该模块将 CUDA 版本的 OnesOp 算子适配到 MetAX 硬件平台，支持所有标准数据类型和 FP8/BF16 等低精度格式。

## 1. Module Structure

- **`ones_metax.h`**: 接口头文件，通过 `ELEMENTWISE_DESCRIPTOR` 宏声明 Descriptor 类的公共接口
- **`ones_metax.maca`**: 核心实现文件，包含 Descriptor 类的析构函数、create 工厂方法和 calculate 计算方法的具体实现

## 2. Core Classes

### `op::ones::metax::Descriptor`
- **Location**: `ones_metax.h` (声明), `ones_metax.maca` (实现)
- **Primary Function**: 封装 MetAX 设备上的 ones 操作符，管理张量元数据、设备实现对象和工作空间内存，为上层提供统一的算子创建与执行接口
- **Key Members**:
  - `_dtype: infiniDtype_t`: 输出张量的数据类型，用于在 calculate 方法中进行类型分发
  - `_info: op::elementwise::ElementwiseInfo`: 张量形状、步幅、连续性、广播等元数据，由元素级运算框架统一管理
  - `_device_info: std::unique_ptr<op::elementwise::metax::DeviceImpl>`: MetAX 设备特定的实现对象，包含内核启动逻辑和硬件参数
  - `_workspace_size: size_t`: 执行内核所需的 GPU/NPU 设备内存大小（用于存放输入指针数组和元数据）
- **Core Methods**:
  - `~Descriptor()`: 默认析构函数，自动释放 _device_info 管理的资源
  - `create(infiniopHandle_t handle_, Descriptor **desc_ptr, infiniopTensorDescriptor_t out_desc, std::vector<infiniopTensorDescriptor_t> input_desc_vec)`:
    - **功能**: 工厂方法，从张量描述符构造 Descriptor 对象
    - **算法**: 验证数据类型白名单（15种支持类型） → 验证输入输出形状一致性（CHECK_SAME_SHAPE） → 调用 `CREATE_ELEMENTWISE_METAX_DESCRIPTOR` 宏构造 ElementwiseInfo 和 DeviceImpl → 计算 workspace 大小（元数据 + 输入指针数组）
    - **复杂度**: O(ndim)，其中 ndim 为张量维度数
  - `calculate(void *workspace, size_t workspace_size, void *output, std::vector<const void *> inputs, void *stream) const`:
    - **功能**: 在 MetAX 设备上执行 ones 操作内核
    - **算法**: 工作空间大小检查（workspace_size >= _workspace_size） → 根据 _dtype 进行 15 路类型分支 → 调用 `_device_info->calculate<256, cuda::OnesOp, T>` 模板方法，传入 CUDA 定义的 OnesOp 函数对象
    - **复杂度**: O(output_size/256)，块大小固定为 256 线程
- **Lifecycle**: 由 create 工厂方法构造，用户负责销毁；内部持有 DeviceImpl 的 unique_ptr，析构时自动清理

## 3. API Interface

```cpp
namespace op::ones::metax {

class Descriptor final : public InfiniopDescriptor {
public:
    ~Descriptor();

    // Factory method: creates and initializes Descriptor from tensor descriptors
    // Returns INFINI_STATUS_SUCCESS on success, error code on failure
    static infiniStatus_t create(
        infiniopHandle_t handle_,                    // MetAX device handle
        Descriptor **desc_ptr,                       // [out] Created descriptor pointer
        infiniopTensorDescriptor_t out_desc,         // Output tensor descriptor
        std::vector<infiniopTensorDescriptor_t> input_desc_vec); // Input tensor descriptors (size=1, ignored for ones)

    // Execute ones operation on MetAX device
    // Returns INFINI_STATUS_SUCCESS on success, error code on failure
    infiniStatus_t calculate(
        void *workspace,                 // Device memory workspace (size >= workspaceSize())
        size_t workspace_size,           // Size of workspace in bytes
        void *output,                    // [out] Output tensor device pointer
        std::vector<const void *> inputs, // Input tensor device pointers (unused for ones)
        void *stream) const;             // MetAX stream (hcStream_t)
};
}
```

## 4. Usage Example

```cpp
// Example: Creating a ones tensor of shape {1024, 1024} with FP32 dtype on MetAX device
#include "ones_metax.h"

// Initialize MetAX handle (assuming already created)
infiniopHandle_t metax_handle;
// ... (handle initialization omitted)

// Create tensor descriptors
int64_t shape[] = {1024, 1024};
int64_t strides[] = {1024, 1};  // Contiguous row-major
infiniopTensorDescriptor_t input_desc, output_desc;
infiniopCreateTensorDescriptor(&input_desc, INFINI_DTYPE_F32, 2, shape, strides);
infiniopCreateTensorDescriptor(&output_desc, INFINI_DTYPE_F32, 2, shape, strides);

// Create ones operation descriptor
op::ones::metax::Descriptor* ones_desc = nullptr;
std::vector<infiniopTensorDescriptor_t> inputs = {input_desc};
auto status = op::ones::metax::Descriptor::create(
    metax_handle, &ones_desc, output_desc, inputs);
if (status != INFINI_STATUS_SUCCESS) {
    // Handle error
}

// Allocate device memory
void* d_output = nullptr;
hcMalloc(&d_output, 1024 * 1024 * sizeof(float));
size_t workspace_size = ones_desc->workspaceSize();
void* d_workspace = nullptr;
hcMalloc(&d_workspace, workspace_size);

// Execute ones operation (output tensor will be filled with ones)
hcStream_t stream;
hcStreamCreate(&stream);
std::vector<const void*> input_ptrs = {nullptr};  // Input unused for ones
status = ones_desc->calculate(d_workspace, workspace_size, d_output, input_ptrs, stream);

// Synchronize and cleanup
hcStreamSynchronize(stream);
hcFree(d_output);
hcFree(d_workspace);
delete ones_desc;
infiniopDestroyTensorDescriptor(input_desc);
infiniopDestroyTensorDescriptor(output_desc);
```

## 5. Implementation Details

- **Memory Management**:
  - **Workspace Layout**: 工作空间分为两个区域——首部存储输入指针数组（`input_size * sizeof(void*)`，ones 操作为 1 * sizeof(void*)），剩余部分存储 ElementwiseInfo 的元数据（输出形状、步幅、输入形状、步幅、连续性标志、广播标志等）
  - **设备内存传输**: 通过 `hcMemcpyAsync` 将元数据和输入指针数组异步传输到 NPU，流式执行以隐藏传输延迟。传输顺序为：先复制输入指针数组，再复制元数据到 workspace 的偏移位置
  - **Pimpl 模式**: DeviceImpl 使用 Opaque 指针隐藏实现细节，避免暴露 MetAX 特定的内部 API（如 hcdnn/hcblas 句柄管理）

- **Concurrency**:
  - **异步执行**: 所有内核启动和内存拷贝使用 `hcStream_t` 异步流，支持主机与设备并行操作及多流并发
  - **线程安全**: Descriptor 对象本身不保证线程安全；多线程场景需为每个线程创建独立的 Descriptor 或外部加锁。不同 stream 可以并发执行同一 Descriptor 的 calculate 方法

- **Performance**:
  - **内核配置**: 固定使用 256 线程块大小（`BLOCK_SIZE=256`），网格大小动态计算为 `min(ceil_div(output_size, 256), device_info->gridSizeX())`，确保不超过硬件网格限制
  - **步进循环**: 对于大型张量（output_size > grid*block），使用 for 循环分步启动内核，每次推进 step = gridDims.x * blockDims.x，避免单次内核超出硬件限制
  - **连续性优化**: 对于连续张量（`isOutputContiguous()==true`），使用线性索引直接访问；对于非连续张量，调用 `device::metax::indexToOffset` 进行维度级索引转换，复杂度 O(ndim)
  - **类型特化**: 使用模板特化避免运行时分支，每种数据类型生成独立的内核实例，编译期优化

- **Error Handling**:
  - **类型白名单检查**: `CHECK_DTYPE` 宏验证数据类型是否在支持的 15 种类型列表中（BYTE/BOOL/I8/I16/I32/I64/U8/U16/U32/U64/F8/F16/F32/F64/BF16），复数类型（C16/C32/C64/C128）返回 `INFINI_STATUS_NOT_IMPLEMENTED`
  - **形状一致性验证**: `CHECK_SAME_SHAPE` 确保输入输出张量形状完全一致（逐元素运算不要求自动广播，需用户手动对齐）
  - **工作空间大小验证**: calculate 方法检查 workspace_size >= _workspace_size，否则返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
  - **结果类型传播**: 所有非复数类型返回类型为 1 的字面量或类型转换后的 1（如 `true`, `1`, `1.0f`, `__float2half(1.0f)`, `cuda_fp8_e4m3(1.0f)`, `__float2bfloat16(1.0f)`），确保与输出类型语义一致

- **Dependencies**:
  - **上游依赖**:
    - `op::elementwise::metax::DeviceImpl`: 提供 MetAX 设备特定的内核启动和模板计算接口，封装了元素级运算的通用执行逻辑
    - `op::elementwise::ElementwiseInfo`: 管理张量元数据的通用结构，包含形状、步幅、连续性、广播标志等，内存布局为连续的 size_t 数组
    - `cuda::OnesOp`: CUDA 命名空间下定义的 OnesOp 函数对象（`ones/cuda/kernel.cuh`），通过 `if constexpr` 为每种类型返回类型安全的 1 值
    - `device::metax::Handle`: MetAX 设备句柄，封装 hcblas/hcdnn 库和硬件参数（warp_size, maxThreadsPerBlock, grid_size 等）
    - `ELEMENTWISE_DESCRIPTOR` 宏（`elementwise/elementwise.h`）：自动生成 Descriptor 类的成员变量、构造函数和公共接口声明，消除重复代码
    - `CREATE_ELEMENTWISE_METAX_DESCRIPTOR` 宏（`elementwise/metax/elementwise_metax_api.h`）：封装 create 方法中的通用逻辑（Info 创建、DeviceImpl 创建、workspace 计算），简化算子实现
  - **外部依赖**:
    - MetAX 驱动 API（`hcMalloc`, `hcMemcpyAsync`, `hcStream_t`, `hcStreamCreate`, `hcStreamSynchronize`, `hcFree` 等）
    - CUDA 类型系统（`cuda_fp8_e4m3`, `cuda_bfloat16`, `half` 等），用于类型安全的低精度格式支持
    - C++ 标准库（`std::vector`, `std::unique_ptr`, `std::shared_ptr`, `std::index_sequence`, `std::enable_if_t` 等）

- **Design Patterns**:
  - **工厂模式**: create 静态方法作为构造器，封装对象创建和验证逻辑，确保返回的对象始终处于有效状态
  - **策略模式**: ElementwiseInfo 存储张量布局策略（连续/非连续、广播），DeviceImpl 根据策略选择不同的内核路径（线性索引 vs. 维度索引转换）
  - **模板方法模式**: `elementwiseKernel` 模板函数定义逐元素运算的骨架，OnesOp 作为具体策略传入，实现算术逻辑与执行框架的解耦
  - **类型擦除**: DeviceImpl 使用模板和 SFINAE 重载，支持统一输入类型（Tdata）和混合输入类型（Tout, Tin...）两种调用方式，提高复用性
  - **宏元编程**: ELEMENTWISE_DESCRIPTOR 宏为不同算子（ones, zeros, add, mul 等）生成结构一致的 Descriptor 类，避免代码重复，统一接口规范
  - **CRTP (Curiously Recurring Template Pattern)**: OnesOp 作为无状态函数对象，通过模板参数传递给 elementwiseKernel，实现零开销抽象
