# Zeros MetAX Core Implementation Documentation

本模块实现基于 MetAX 设备（华为昇腾 NPU 架构）的张量零值填充操作，通过统一的逐元素运算框架为 InfiniOP 提供 device-specific 的 zeros 算子实现。该模块将 CUDA 版本的 ZerosOp 算子适配到 MetAX 硬件平台，支持所有标准数据类型和 FP8/BF16 等低精度格式。

## 1. Module Structure

- **`zeros_metax.h`**: 接口头文件，通过 `ELEMENTWISE_DESCRIPTOR` 宏声明 Descriptor 类的公共接口
- **`zeros_metax.maca`**: 核心实现文件，包含 Descriptor 类的析构函数、create 工厂方法和 calculate 计算方法的具体实现

## 2. Core Classes

### `op::zeros::metax::Descriptor`
- **Location**: `zeros_metax.h` (声明), `zeros_metax.maca` (实现)
- **Primary Function**: 封装 MetAX 设备上的 zeros 操作符，管理张量元数据、设备实现对象和工作空间内存，为上层提供统一的算子创建与执行接口
- **Key Members**:
  - `_dtype: infiniDtype_t`: 输出张量的数据类型，用于在 calculate 方法中进行类型分发
  - `_info: op::elementwise::ElementwiseInfo`: 张量形状、步幅、连续性、广播等元数据，由元素级运算框架统一管理
  - `_device_info: std::unique_ptr<op::elementwise::metax::DeviceImpl>`: MetAX 设备特定的实现对象，包含内核启动逻辑和硬件参数
  - `_workspace_size: size_t`: 执行内核所需的 GPU/NPU 设备内存大小（用于存放输入指针数组和元数据）
- **Core Methods**:
  - `~Descriptor()`: 默认析构函数，自动释放 _device_info 管理的资源
  - `create(infiniopHandle_t handle_, Descriptor **desc_ptr, infiniopTensorDescriptor_t out_desc, std::vector<infiniopTensorDescriptor_t> input_desc_vec)`:
    - **功能**: 工厂方法，从张量描述符构造 Descriptor 对象
    - **算法**: 验证数据类型和白名单检查 → 验证输入输出形状一致性 → 调用 `CREATE_ELEMENTWISE_METAX_DESCRIPTOR` 宏构造 ElementwiseInfo 和 DeviceImpl → 计算 workspace 大小（元数据 + 输入指针数组）
    - **复杂度**: O(ndim)，其中 ndim 为张量维度数
  - `calculate(void *workspace, size_t workspace_size, void *output, std::vector<const void *> inputs, void *stream) const`:
    - **功能**: 在 MetAX 设备上执行 zeros 操作内核
    - **算法**: 工作空间大小检查 → 根据 _dtype 进行 15 路类型分支 → 调用 `_device_info->calculate<256, cuda::ZerosOp, T>` 模板方法，传入 CUDA 定义的 ZerosOp 函数对象
    - **复杂度**: O(output_size/256)，块大小固定为 256 线程
- **Lifecycle**: 由 create 工厂方法构造，用户负责销毁；内部持有 DeviceImpl 的 unique_ptr，析构时自动清理

## 3. API Interface

```cpp
namespace op::zeros::metax {

class Descriptor final : public InfiniopDescriptor {
public:
    ~Descriptor();

    // Factory method: creates and initializes Descriptor from tensor descriptors
    // Returns INFINI_STATUS_SUCCESS on success, error code on failure
    static infiniStatus_t create(
        infiniopHandle_t handle_,                    // MetAX device handle
        Descriptor **desc_ptr,                       // [out] Created descriptor pointer
        infiniopTensorDescriptor_t out_desc,         // Output tensor descriptor
        std::vector<infiniopTensorDescriptor_t> input_desc_vec); // Input tensor descriptors (size=1, ignored for zeros)

    // Execute zeros operation on MetAX device
    // Returns INFINI_STATUS_SUCCESS on success, error code on failure
    infiniStatus_t calculate(
        void *workspace,                 // Device memory workspace (size >= workspaceSize())
        size_t workspace_size,           // Size of workspace in bytes
        void *output,                    // [out] Output tensor device pointer
        std::vector<const void *> inputs, // Input tensor device pointers (unused for zeros)
        void *stream) const;             // MetAX stream (hcStream_t)
};
}
```

## 4. Usage Example

```cpp
// Example: Creating a zeros tensor of shape {1024, 1024} with FP32 dtype on MetAX device
#include "zeros_metax.h"

// Initialize MetAX handle (assuming already created)
infiniopHandle_t metax_handle;
// ... (handle initialization omitted)

// Create tensor descriptors
int64_t shape[] = {1024, 1024};
int64_t strides[] = {1024, 1};  // Contiguous row-major
infiniopTensorDescriptor_t input_desc, output_desc;
infiniopCreateTensorDescriptor(&input_desc, INFINI_DTYPE_F32, 2, shape, strides);
infiniopCreateTensorDescriptor(&output_desc, INFINI_DTYPE_F32, 2, shape, strides);

// Create zeros operation descriptor
op::zeros::metax::Descriptor* zeros_desc = nullptr;
std::vector<infiniopTensorDescriptor_t> inputs = {input_desc};
auto status = op::zeros::metax::Descriptor::create(
    metax_handle, &zeros_desc, output_desc, inputs);
if (status != INFINI_STATUS_SUCCESS) {
    // Handle error
}

// Allocate device memory
void* d_output = nullptr;
hcMalloc(&d_output, 1024 * 1024 * sizeof(float));
size_t workspace_size = zeros_desc->workspaceSize();
void* d_workspace = nullptr;
hcMalloc(&d_workspace, workspace_size);

// Execute zeros operation (output tensor will be filled with zeros)
hcStream_t stream;
hcStreamCreate(&stream);
std::vector<const void*> input_ptrs = {nullptr};  // Input unused for zeros
status = zeros_desc->calculate(d_workspace, workspace_size, d_output, input_ptrs, stream);

// Synchronize and cleanup
hcStreamSynchronize(stream);
hcFree(d_output);
hcFree(d_workspace);
delete zeros_desc;
infiniopDestroyTensorDescriptor(input_desc);
infiniopDestroyTensorDescriptor(output_desc);
```

## 5. Implementation Details

- **Memory Management**:
  - **Workspace Layout**: 工作空间分为两个区域——首部存储输入指针数组（`input_size * sizeof(void*)`），剩余部分存储 ElementwiseInfo 的元数据（形状、步幅、连续性、广播标志等）
  - **设备内存传输**: 通过 `hcMemcpyAsync` 将元数据和输入指针数组异步传输到 NPU，流式执行以隐藏传输延迟
  - **Pimpl 模式**: DeviceImpl 使用 Opaque 指针隐藏实现细节，避免暴露 MetAX 特定的内部 API

- **Concurrency**:
  - **异步执行**: 所有内核启动和内存拷贝使用 `hcStream_t` 异步流，支持主机与设备并行操作
  - **线程安全**: Descriptor 对象本身不保证线程安全；多线程场景需为每个线程创建独立的 Descriptor 或外部加锁

- **Performance**:
  - **内核配置**: 固定使用 256 线程块大小（`BLOCK_SIZE=256`），网格大小动态计算为 `min(ceil_div(output_size, 256), device_info->gridSizeX())`
  - **步进循环**: 对于大型张量（output_size > grid*block），使用 for 循环分步启动内核，避免单次内核超出硬件限制
  - **连续性优化**: 对于连续张量，使用线性索引直接访问；对于非连续张量，调用 `device::metax::indexToOffset` 进行维度级索引转换，复杂度 O(ndim)

- **Error Handling**:
  - **类型白名单检查**: `CHECK_DTYPE` 宏验证数据类型是否在支持的 15 种类型列表中（BYTE/BOOL/I8-I64/U8-U64/F8/F16/F32/F64/BF16）
  - **形状一致性验证**: `CHECK_SAME_SHAPE` 确保输入输出张量形状完全一致
  - **工作空间大小验证**: calculate 方法检查 workspace_size >= _workspace_size，否则返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
  - **未实现类型**: 复数类型（C16/C32/C64/C128）返回 `INFINI_STATUS_NOT_IMPLEMENTED`

- **Dependencies**:
  - **上游依赖**:
    - `op::elementwise::metax::DeviceImpl`: 提供 MetAX 设备特定的内核启动和模板计算接口
    - `op::elementwise::ElementwiseInfo`: 管理张量元数据的通用结构
    - `cuda::ZerosOp`: CUDA 命名空间下定义的 ZerosOp 函数对象（`zeros/cuda/kernel.cuh`），通过 `if constexpr` 为每种类型返回零值
    - `device::metax::Handle`: MetAX 设备句柄，封装 hcblas/hcdnn 库和硬件参数
    - `ELEMENTWISE_DESCRIPTOR` 宏（`elementwise/elementwise.h`）：自动生成 Descriptor 类的成员变量、构造函数和公共接口声明
    - `CREATE_ELEMENTWISE_METAX_DESCRIPTOR` 宏（`elementwise/metax/elementwise_metax_api.h`）：封装 create 方法中的通用逻辑（Info 创建、DeviceImpl 创建、workspace 计算）
  - **外部依赖**:
    - MetAX 驱动 API（`hcMalloc`, `hcMemcpyAsync`, `hcStream_t` 等）
    - hpcc/hpcc 库（`__hpcc_fp8_e4m3`, `hpcc_bfloat16` 等）
    - C++ 标准库（`std::vector`, `std::unique_ptr`, `std::shared_ptr`）

- **Design Patterns**:
  - **工厂模式**: create 静态方法作为构造器，封装对象创建和验证逻辑
  - **策略模式**: ElementwiseInfo 存储张量布局策略（连续/非连续、广播），DeviceImpl 根据策略选择不同的内核路径
  - **模板方法模式**: `elementwiseKernel` 模板函数定义逐元素运算的骨架，ZerosOp 作为具体策略传入
  - **类型擦除**: DeviceImpl 使用模板和 SFINAE 重载，支持统一输入类型（Tdata）和混合输入类型（Tout, Tin...）两种调用方式
  - **宏元编程**: ELEMENTWISE_DESCRIPTOR 宏为不同算子（zeros, ones, add 等）生成结构一致的 Descriptor 类，避免代码重复
