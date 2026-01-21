# Rearrange Operator (Ascend Backend) Core Implementation Documentation

本模块实现了 Infini 框架中 Rearrange (张量重排) 操作的华为昇腾 (Ascend NPU) 后端支持。通过 CANN 神经网络库的 `aclnnInplaceCopy` API 实现张量在不同内存布局间的高效复制，支持任意维度的张量重塑、转置、广播等操作。

## 1. Module Structure

- **`rearrange_ascend.h`**: 声明 Ascend 后端的 Rearrange 描述符类，通过 `DESCRIPTOR(ascend)` 宏展开生成完整的类定义
- **`rearrange_ascend.cc`**: 实现 Rearrange 算子的 Ascend NPU 特定逻辑，包括算子创建、工作空间预分配、核函数调度

## 2. Core Classes

### `Descriptor::Opaque`
- **Location**: `rearrange_ascend.cc:7-18`
- **Primary Function**: 封装 Ascend ACLNN (Ascend Compute Library Neural Network) 特定的不透明资源，包括张量描述符和执行器工作空间
- **Key Members**:
  - `aclnnTensorDescriptor_t dst`: 目标张量的 ACLNN 描述符，包含数据类型、形状、步长等元信息
  - `aclnnTensorDescriptor_t src`: 源张量的 ACLNN 描述符
  - `void *workspace`: ACLNN InplaceCopy 算子所需的设备侧工作空间缓冲区
  - `uint64_t workspace_size`: 工作空间大小（字节），在算子初始化时通过 ACL API 查询获得
- **Lifecycle**:
  - 在 `Descriptor::create()` 中创建，调用 `aclnnInplaceCopyGetWorkspaceSize` 查询并分配工作空间
  - 析构函数自动释放所有资源：删除张量描述符对象，调用 `aclrtFree` 释放设备内存

### `Descriptor`
- **Location**: 通过 `rearrange.h` 中的 `DESCRIPTOR(ascend)` 宏展开定义
- **Primary Function**: 实现 Infini 框架的 Rearrange 算子接口，封装 Ascend NPU 的执行逻辑
- **Key Members**:
  - `Opaque *_opaque`: 指向 Ascend 特定资源的指针
  - `utils::RearrangeMeta _meta`: 通用重排元数据，用于验证张量布局的合法性（实际计算由 ACLNN 处理）
- **Core Methods**:
  - `create(handle_, desc_ptr, y_desc, x_desc)`: 构建函数，验证输入/输出张量的形状、数据类型一致性，调用 `RearrangeMeta::create` 生成布局元数据，创建 ACLNN 张量描述符，预分配工作空间
  - `calculate(y, x, stream)`: 执行函数，设置实际数据地址，调用 `aclnnInplaceCopy` 在 Ascend NPU 上执行张量复制
- **Lifecycle**:
  - 工厂模式：通过静态 `create` 方法实例化
  - RAII 管理：析构时自动清理 Opaque 资源
  - 可重复使用：同一描述符可用于多次 `calculate` 调用（处理不同数据）

## 3. API Interface

```cpp
namespace op::rearrange::ascend {

class Descriptor final : public InfiniopDescriptor {
public:
    // 创建 Rearrange 算子描述符
    // 参数:
    //   handle_: Infini 句柄，封装 Ascend 设备和上下文
    //   desc_ptr: 输出参数，返回创建的描述符指针
    //   y_desc: 目标张量描述符（定义输出布局）
    //   x_desc: 源张量描述符（定义输入布局）
    // 返回: INFINI_STATUS_SUCCESS 或错误码
    static infiniStatus_t create(
        infiniopHandle_t handle_,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc);

    // 执行张量重排操作
    // 参数:
    //   y: 目标张量的设备内存地址
    //   x: 源张量的设备内存地址
    //   stream: Ascend ACL 流（用于异步执行）
    // 返回: INFINI_STATUS_SUCCESS 或错误码
    infiniStatus_t calculate(
        void *y,
        const void *x,
        void *stream) const;

    // 析构函数：释放 ACLNN 资源
    ~Descriptor();
};

} // namespace op::rearrange::ascend
```

## 4. Usage Example

```cpp
// 示例：在 Ascend NPU 上执行张量转置 (NCHW -> NHWC)
#include "rearrange_ascend.h"
#include "infinicore.h"

// 1. 创建 Infini 句柄和 Ascend 设备
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_ASCEND, 0);

// 2. 定义输入张量 (NCHW 格式: [N, C, H, W])
int64_t nchw_shape[] = {32, 64, 224, 224};
int64_t nchw_strides[] = {64 * 224 * 224, 224 * 224, 224, 1};
infiniopTensorDescriptor_t x_desc;
infiniopCreateTensorDescriptor(&x_desc, INFINI_DTYPE_F16, 4, nchw_shape, nchw_strides);

// 3. 定义输出张量 (NHWC 格式: [N, H, W, C])
int64_t nhwc_shape[] = {32, 224, 224, 64};
int64_t nhwc_strides[] = {224 * 224 * 64, 224 * 64, 64, 1};
infiniopTensorDescriptor_t y_desc;
infiniopCreateTensorDescriptor(&y_desc, INFINI_DTYPE_F16, 4, nhwc_shape, nhwc_strides);

// 4. 创建 Rearrange 算子描述符
op::rearrange::ascend::Descriptor *rearrange_op;
auto status = op::rearrange::ascend::Descriptor::create(
    handle, &rearrange_op, y_desc, x_desc);

if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误：形状或步长不兼容
}

// 5. 分配设备内存并初始化输入数据
void *d_x, *d_y;
size_t x_size = 32 * 64 * 224 * 224 * sizeof(uint16_t); // FP16
size_t y_size = 32 * 224 * 224 * 64 * sizeof(uint16_t);
aclrtMalloc(&d_x, x_size, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(&d_y, y_size, ACL_MEM_MALLOC_HUGE_FIRST);
// ... (将数据从主机复制到 d_x)

// 6. 创建 ACL 流
aclrtStream stream;
aclrtCreateStream(&stream);

// 7. 执行重排操作（在 NPU 上异步执行）
status = rearrange_op->calculate(d_y, d_x, stream);

// 8. 同步并获取结果
aclrtSynchronizeStream(stream);
// ... (将 d_y 复制回主机)

// 9. 清理资源
aclrtDestroyStream(stream);
aclrtFree(d_x);
aclrtFree(d_y);
delete rearrange_op;
infiniopDestroyHandle(handle);
```

## 5. Implementation Details

### Memory Management (内存管理)
- **预分配策略**: 工作空间在算子创建时通过 `aclnnInplaceCopyGetWorkspaceSize` 查询，使用 `ACL_MEM_MALLOC_HUGE_FIRST` 标志分配大页内存，减少 TLB 缺失
- **资源封装**: ACLNN 张量描述符 (`aclnnTensorDescriptor`) 封装了 `aclTensor` 对象，自动管理元数据到 CANN 格式的转换
- **RAII 模式**: `Opaque` 结构体的析构函数确保所有 ACL 资源正确释放，避免内存泄漏

### Concurrency (并发机制)
- **流式执行**: `calculate` 方法接受 `aclrtStream` 参数，支持异步执行和多个算子并行
- **执行器复用**: 虽然当前实现每次调用都重新获取工作空间大小和创建执行器（第 86 行），但理论上可通过缓存 `aclOpExecutor` 实现优化（类似 GEMM 算子的 `lookup` 哈希表）
- **线程安全**: 不同流上的操作可并行；同一流上的操作按序执行

### Performance (性能优化)
- **零拷贝优化**: 使用 `aclnnInplaceCopy` 直接在目标张量上操作，避免中间缓冲区
- **元数据验证**: 通过 `utils::RearrangeMeta::create` 验证布局合法性（O(n log n) 维度排序算法），确保 ACLNN 操作能够成功执行
- **大页内存**: 工作空间使用 `ACL_MEM_MALLOC_HUGE_FIRST` 分配，提升内存带宽利用率

### Error Handling (错误处理)
- **输入验证**: 检查数据类型一致性（`CHECK_API_OR`），形状和步长匹配（第 33-38 行）
- **ACL 错误传播**: 使用 `CHECK_ACL` 宏检查所有 ACL API 返回值，自动提取 Ascend 错误消息
- **元数据验证**: `RearrangeMeta::create` 返回 `Result<T>` 类型，通过 `CHECK_RESULT` 检查步长合法性（例如检测 dst_stride == 0 的非法布局）

### Dependencies (依赖关系)
- **外部依赖**:
  - `acl/acl.h`, `acl/acl_rt.h`: Ascend 运行时 API
  - `aclnn/acl_meta.h`: ACLNN 元数据定义
  - `aclnnop/aclnn_copy.h`: ACLNN InplaceCopy 算子接口
- **内部依赖**:
  - `../rearrange.h`: 定义通用的 Rearrange 描述符接口
  - `../../../devices/ascend/common_ascend.h`: 提供 Ascend 工具宏和 `aclnnTensorDescriptor` 封装
  - `../../../utils.h`: 提供 `RearrangeMeta` 通用布局计算逻辑

### Design Patterns (设计模式)
- **桥接模式**: `Descriptor` 类继承自 `InfiniopDescriptor`，通过 `Opaque` 结构体桥接到 Ascend 特定实现
- **工厂方法**: 静态 `create` 方法封装复杂的初始化逻辑
- **PIMPL 惯用法**: `Opaque` 前向声明，隐藏 Ascend 特定类型，减少头文件依赖
- **策略模式**: 相同的 `RearrangeMeta` 元数据可用于不同后端（CPU、CUDA、Ascend），实现算法与硬件解耦

### Key Implementation Nuances (关键实现细节)
1. **执行器生命周期管理**:
   - 第 52-53 行：首次调用 `aclnnInplaceCopyGetWorkspaceSize` 仅获取工作空间大小
   - 第 69 行：立即销毁临时执行器（`aclDestroyAclOpExecutor`），避免资源泄漏
   - 第 86 行：每次 `calculate` 重新创建执行器，设置实际数据地址后调用

2. **地址动态绑定**:
   - 使用 `AclSetTensorAddr` 宏（第 84-85 行）在执行前绑定实际数据地址
   - 支持批量处理：同一描述符可用于不同内存块的重复操作（例如循环中处理多个张量）

3. **步长处理**:
   - ACLNN 要求步长以字节为单位，`aclnnTensorDescriptor` 构造函数自动转换
   - `utils::RearrangeMeta` 验证步长合法性，过滤长度为 1 的维度，合并连续维度优化计算

4. **与 CPU 实现的差异**:
   - CPU 后端使用 OpenMP 并行 `memcpy` 循环（见 `utils::RearrangeMeta::launch`）
   - Ascend 后端完全委托 ACLNN，利用 NPU 专用硬件加速，无需手动并行化
