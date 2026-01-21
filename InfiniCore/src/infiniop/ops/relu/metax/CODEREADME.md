# ReLU Metax 后端实现文档

本模块实现了 ReLU (Rectified Linear Unit) 激活函数在 Metax 加速卡上的硬件后端，通过 NineToothed 代码生成框架提供高性能的元素级计算支持。

## 1. 模块结构

- **`relu_metax.h`**: Metax 后端的 API 声明文件，定义了 `op::relu::metax::Descriptor` 类
- **`relu_metax.maca`**: Metax 后端的核心实现，包含 descriptor 创建和 kernel 启动逻辑

## 2. 核心类

### `op::relu::metax::Descriptor`
- **位置**: `relu_metax.maca`
- **主要功能**: Metax 设备上的 ReLU 操作描述符，继承自通用的 `InfiniopDescriptor` 基类，管理 ReLU 计算的元数据、工作空间大小和设备实现
- **关键成员**:
  - `_dtype`: `infiniDtype_t` - 输入/输出张量的数据类型 (支持 F16, F32, F64, BF16)
  - `_info`: `op::elementwise::ElementwiseInfo` - 元素级操作的形状、步幅和布局信息
  - `_device_info`: `std::unique_ptr<op::elementwise::metax::DeviceImpl>` - Metax 设备特定实现的 opaque 指针
  - `_workspace_size`: `size_t` - 所需工作空间大小（字节）
  - 基类成员 `device_type`, `device_id`: 设备类型和 ID 标识
- **核心方法**:
  - `create(handle_, desc_ptr, out_desc, input_desc_vec)`: 静态工厂方法，验证输入输出张量的数据类型和形状一致性，构造 `ElementwiseInfo` 对象，创建 `DeviceImpl` 实例，计算所需工作空间大小，并在堆上分配 descriptor 实例
  - `calculate(workspace, workspace_size, output, inputs, stream)`: 执行 ReLU 计算，验证工作空间充足性，从 `ElementwiseInfo` 提取输入/输出张量的形状和步幅，构造 `NineToothedTensor` 对象，调用 `launch_relu` 启动 NineToothed 生成的 kernel，使用固定的 block_size=1024
  - `~Descriptor()`: 析构函数（默认实现）
- **生命周期**: 由工厂方法 `create` 在堆上构造，由外部调用者通过 `infiniopDestroyReluDescriptor` 显式销毁

## 3. API 接口

```cpp
namespace op::relu::metax {

class Descriptor final : public InfiniopDescriptor {
public:
    ~Descriptor();

    // 创建 ReLU descriptor
    static infiniStatus_t create(
        infiniopHandle_t handle_,                    // Metax 设备句柄
        Descriptor **desc_ptr,                       // [输出] 指向新创建 descriptor 的指针
        infiniopTensorDescriptor_t out_desc,         // 输出张量描述符
        std::vector<infiniopTensorDescriptor_t> input_desc_vec  // 输入张量描述符向量 (仅包含 x)
    );
    // 返回: INFINI_STATUS_SUCCESS | INFINI_STATUS_BAD_TENSOR_DTYPE | INFINI_STATUS_BAD_TENSOR_SHAPE

    // 执行 ReLU 计算
    infiniStatus_t calculate(
        void *workspace,                             // 工作空间指针
        size_t workspace_size,                       // 工作空间大小（字节）
        void *output,                                // 输出张量数据指针
        std::vector<const void *> inputs,            // 输入张量数据指针向量 (仅包含 x)
        void *stream                                 // Metax 计算流
    ) const;
    // 返回: INFINI_STATUS_SUCCESS | INFINI_STATUS_INSUFFICIENT_WORKSPACE | INFINI_STATUS_INTERNAL_ERROR | INFINI_STATUS_BAD_TENSOR_DTYPE
};

} // namespace op::relu::metax
```

## 4. 使用示例

```cpp
// 示例: 在 Metax 设备上执行 ReLU 激活操作
#include "infiniop/ops/relu/metax/relu_metax.h"

// 1. 准备张量描述符 (假设已创建)
infiniopTensorDescriptor_t x_desc;  // 输入张量 [N, C, H, W], dtype=INFINI_DTYPE_F16
infiniopTensorDescriptor_t y_desc;  // 输出张量 [N, C, H, W], dtype=INFINI_DTYPE_F16

// 2. 创建 ReLU descriptor
infiniopHandle_t metax_handle;      // Metax 设备句柄 (已初始化)
infiniopReluDescriptor_t relu_desc;
infiniStatus_t status = infiniopCreateReluDescriptor(
    metax_handle,
    &relu_desc,
    y_desc,
    x_desc
);
// 内部调用 op::relu::metax::Descriptor::create()

if (status != INFINI_STATUS_SUCCESS) {
    // 错误处理
}

// 3. 查询所需工作空间
size_t workspace_size;
status = infiniopGetReluWorkspaceSize(relu_desc, &workspace_size);

// 4. 分配工作空间和设备内存
void *workspace = nullptr;
if (workspace_size > 0) {
    workspace = malloc(workspace_size);
}
void *x_d, *y_d;
// 假设已通过 Metax 内存分配 API 分配设备内存
// x_d 指向输入数据, y_d 指向输出数据

// 5. 创建 Metax 计算流
hcStream_t stream;
// 初始化 stream (通过 metax_handle->internal() 访问底层 API)

// 6. 执行 ReLU 计算
status = infiniopRelu(
    relu_desc,
    workspace,
    workspace_size,
    y_d,           // 输出
    x_d,           // 输入
    stream         // Metax 流
);
// 内部调用 relu_desc->calculate(workspace, workspace_size, y_d, {x_d}, stream)
// calculate() 将:
//   - 验证 workspace_size >= _workspace_size
//   - 从 _info 提取 x_shape, x_strides, y_shape, y_strides
//   - 构造 NineToothedTensor{x_d, x_shape, x_strides}
//   - 构造 NineToothedTensor{y_d, y_shape, y_strides}
//   - 调用 launch_relu(stream, x, y, ndim, _dtype, 1024)

// 7. 同步流以等待计算完成
hcStreamSynchronize(stream);

// 8. 清理资源
free(workspace);
infiniopDestroyReluDescriptor(relu_desc);  // 内部调用 delete descriptor
// 释放 x_d, y_d, stream 等资源
```

## 5. 实现细节

- **宏驱动设计**: 通过 `ELEMENTWISE_DESCRIPTOR(relu, metax)` 宏自动生成完整的 `Descriptor` 类定义，该宏定义在 `elementwise.h` 中，展开为包含 `_dtype`, `_info`, `_device_info`, `_workspace_size` 成员和 `create()`, `calculate()` 方法的完整类声明

- **设备抽象层**:
  - 通过 `CREATE_ELEMENTWISE_METAX_DESCRIPTOR` 宏封装 Metax 特定的初始化逻辑，该宏调用 `ElementwiseInfo::create()` 提取张量元数据，计算工作空间大小为 `info.getMetaMemSize() + info.getInputSize() * sizeof(void*)`，创建 `op::elementwise::metax::DeviceImpl` 实例并构造 `Descriptor` 对象
  - `DeviceImpl` 使用 opaque 指针模式 (`std::shared_ptr<Opaque>`) 隐藏 Metax 设备的底层实现细节（hcBLAS/hcDNN 句柄池管理）

- **NineToothed 集成**:
  - 通过 `#include "../../../../../build/ninetoothed/relu.h"` 引入 NineToothed 代码生成框架提供的 kernel 启动函数 `launch_relu()`
  - 在 `calculate()` 中构造 `NineToothedTensor` 对象，封装数据指针、形状数组 (`uint64_t*`) 和步幅数组 (`int64_t*`)
  - 使用固定的 `block_size = 1024` 作为 kernel 启动参数，该值是 Metax 设备的典型线程块大小
  - `launch_relu()` 返回 0 表示成功，非零表示错误，映射为 `INFINI_STATUS_INTERNAL_ERROR`

- **数据类型支持**: 通过 `CHECK_DTYPE` 宏验证，支持四种浮点类型:
  - `INFINI_DTYPE_F16`: 半精度浮点 (16-bit)
  - `INFINI_DTYPE_F32`: 单精度浮点 (32-bit)
  - `INFINI_DTYPE_F64`: 双精度浮点 (64-bit)
  - `INFINI_DTYPE_BF16`: 脑浮点 (16-bit)

- **形状验证**: 通过 `CHECK_SAME_SHAPE(y_shape, x_shape)` 宏确保输入输出张量的形状完全匹配，ReLU 是逐元素操作，不改变张量形状

- **工作空间管理**:
  - 工作空间大小在 `create()` 时计算并存储在 `_workspace_size`
  - 工作空间用于存储 `ElementwiseInfo` 的元数据（形状、步幅等）和输入张量指针数组
  - `calculate()` 在执行前检查 `workspace_size < _workspace_size`，若不足则返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`

- **编译条件控制**:
  - 整个文件在 `#ifdef ENABLE_NINETOOTHED` 保护下，只有启用 NineToothed 框架时才编译此实现
  - 在 `operator.cc` 中通过 `#ifdef ENABLE_METAX_API` 和 `#ifdef ENABLE_NINETOOTHED` 双重条件启用此后端
  - 允许在同一项目中同时编译多个硬件后端 (CPU, NVIDIA, Metax 等)，通过预处理器宏选择

- **错误处理**:
  - `create()` 失败时通过 `CHECK_RESULT` 宏传播错误，返回 `INFINI_STATUS_BAD_TENSOR_DTYPE` 或 `INFINI_STATUS_BAD_TENSOR_SHAPE`
  - `calculate()` 使用 switch-case 枚举所有支持的数据类型，default 分支返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
  - `launch_relu()` 失败时返回 `INFINI_STATUS_INTERNAL_ERROR`

- **内存管理**:
  - `Descriptor` 对象通过 `new` 分配，由调用者通过 `infiniopDestroyReluDescriptor` 中的 `delete` 释放
  - `DeviceImpl` 使用 `std::unique_ptr` 管理底层 opaque 对象的生命周期
  - `ElementwiseInfo` 通过移动语义 (`std::move`) 转移所有权到 `Descriptor`，避免不必要的拷贝

- **设计模式**:
  - **Factory Pattern**: `create()` 静态方法作为工厂，封装复杂的对象构造逻辑
  - **Strategy Pattern**: 通过 `InfiniopDescriptor` 基类和多后端实现 (cpu, nvidia, metax) 实现设备无关的接口
  - **Opaque Pointer Pattern**: `DeviceImpl::Opaque` 隐藏 Metax 特定的实现细节，减少头文件依赖
  - **CRTP (Curiously Recurring Template Pattern)**: 元素级操作通过 `ELEMENTWISE_DESCRIPTOR` 宏复用通用的 descriptor 结构

- **性能考虑**:
  - block_size=1024 针对 Metax 设备的 warp size 和 max threads per block 优化，通常 Metax 设备的 warp size 为 32，max threads per_block 为 1024
  - `ElementwiseInfo` 在 descriptor 创建时预计算形状/步幅信息，避免在每次 `calculate()` 时重复计算
  - `DeviceImpl` 内部使用句柄池 (`Pool<hcblasHandle_t>`, `Pool<hcdnnHandle_t>`) 复用昂贵的库句柄创建开销
