# CPU Device Backend Core Implementation Documentation

该模块实现了 InfiniOp 框架中 CPU 设备后端的核心基础设施，提供设备句柄管理和通用计算工具函数，支持张量索引计算、填充形状处理等底层操作。

## 1. Module Structure

- **`cpu_handle.h`**: CPU 设备句柄类声明，定义设备创建接口
- **`cpu_handle.cc`**: CPU 设备句柄实现，构造并初始化 CPU 设备实例
- **`common_cpu.h`**: CPU 通用计算工具函数声明，提供张量索引和填充操作接口
- **`common_cpu.cc`**: CPU 通用计算工具函数实现，包含索引转换和形状计算逻辑

## 2. Core Classes

### `device::cpu::Handle`
- **Location**: `cpu_handle.h` / `cpu_handle.cc`
- **Primary Function**: CPU 设备句柄类，继承自基类 `InfiniopHandle`，用于表示和管理 CPU 计算设备实例。该类封装了设备类型标识和设备 ID，作为 CPU 后端所有操作的入口点。
- **Key Members**:
  - 继承 `InfiniopHandle::device`: 设备类型，固定为 `INFINI_DEVICE_CPU`
  - 继承 `InfiniopHandle::device_id`: 设备 ID，CPU 设备固定为 0
- **Core Methods**:
  - `Handle()`: 私有构造函数，初始化基类为 CPU 设备类型（`INFINI_DEVICE_CPU`）和设备 ID 0
  - `static create(InfiniopHandle **handle_ptr, int device_id)`: 工厂方法，动态分配并初始化 CPU 设备句柄实例，返回状态码
- **Lifecycle**:
  - 使用静态工厂模式创建实例，不允许直接构造
  - 通过 `new Handle{}` 在堆上分配内存
  - 生命周期由调用者管理，需配合 `infiniopDestroyHandle` 释放

## 3. API Interface

```cpp
// 设备句柄创建接口
namespace device::cpu {
    class Handle : public InfiniopHandle {
    public:
        // 创建 CPU 设备句柄
        // 参数:
        //   handle_ptr: 输出参数，返回新创建的句柄指针
        //   device_id: 设备 ID（CPU 固定为 0，参数保留用于接口统一）
        // 返回: infiniStatus_t 状态码，成功返回 INFINI_STATUS_SUCCESS
        static infiniStatus_t create(InfiniopHandle **handle_ptr, int device_id);
    };
}

// 张量索引计算接口
namespace op::common_cpu {
    // 将扁平化索引转换为内存偏移量（基于步长）
    // 参数:
    //   flat_index: 扁平化的元素索引（0-based）
    //   ndim: 张量维度数
    //   shape: 各维度大小数组 [dim0, dim1, ..., dimN-1]
    //   strides: 各维度步长数组（字节为单位）
    // 返回: 内存偏移量（字节为单位）
    // 算法: 从最低维度到最高维度迭代，使用模运算和除法提取各维度索引
    size_t indexToOffset(size_t flat_index, size_t ndim,
                        const size_t *shape, const ptrdiff_t *strides);

    // 计算填充后的张量总元素数量
    // 参数:
    //   ndim: 张量维度数
    //   shape: 原始形状数组
    //   pads: 填充量数组（从第 2 维开始应用，长度为 ndim-2）
    // 返回: 填充后的总元素数
    // 填充规则: 前两维（通常为 batch 和 channel）不填充，从第 2 维开始每维两侧各加 pads[i]
    size_t getPaddedSize(size_t ndim, size_t *shape, const size_t *pads);

    // 计算并返回填充后的张量形状
    // 参数:
    //   ndim: 张量维度数
    //   shape: 原始形状数组
    //   pads: 填充量数组
    // 返回: std::vector<size_t> 填充后的形状向量
    // 实现: 先拷贝原始形状，再对第 2 维及之后的维度应用 2*pads[i-2] 的增量
    std::vector<size_t> getPaddedShape(size_t ndim, const size_t *shape,
                                       const size_t *pads);
}
```

## 4. Usage Example

```cpp
#include "infiniop/devices/cpu/cpu_handle.h"
#include "infiniop/devices/cpu/common_cpu.h"

using namespace device::cpu;
using namespace op::common_cpu;

// 示例 1: 创建 CPU 设备句柄
InfiniopHandle* cpu_handle = nullptr;
infiniStatus_t status = Handle::create(&cpu_handle, 0);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
    return;
}

// 使用句柄进行后续操作...

// 销毁句柄（由调用方提供的接口）
infiniopDestroyHandle(cpu_handle);

// 示例 2: 计算张量索引偏移
// 假设有一个形状为 [2, 3, 4] 的张量，步长为 [48, 16, 4]
size_t shape[] = {2, 3, 4};
ptrdiff_t strides[] = {48, 16, 4};  // 假设元素大小为 4 字节

// 计算扁平索引 5 的内存偏移
size_t flat_idx = 5;
size_t offset = indexToOffset(flat_idx, 3, shape, strides);
// 逻辑: 5 % 4 = 1 (dim2), 5/4 = 1; 1 % 3 = 1 (dim1), 1/3 = 0; 0 % 2 = 0 (dim0)
// offset = 1*4 + 1*16 + 0*48 = 20 字节

// 示例 3: 计算填充形状（用于卷积操作）
// 原始形状 [N, H, W, C] = [1, 32, 32, 3]
// 填充量 [padding_h, padding_w] = [1, 1]
size_t shape[] = {1, 32, 32, 3};
size_t pads[] = {1, 1};  // 对应 H 和 W 维度的填充

// 计算填充后的形状
auto padded_shape = getPaddedShape(4, shape, pads);
// 结果: [1, 32+2*1, 32+2*1, 3] = [1, 34, 34, 3]

// 计算填充后的总大小
size_t padded_size = getPaddedSize(4, shape, pads);
// 结果: 1 * 34 * 34 * 3 = 3468

#ifdef ENABLE_OMP
// 示例 4: 结合 OpenMP 并行处理
#pragma omp parallel for
for (int i = 0; i < total_elements; ++i) {
    size_t offset = indexToOffset(i, ndim, shape, strides);
    // 使用 offset 访问数据...
    data_ptr[offset] = process_element(i);
}
#endif
```

## 5. Implementation Details

### 内存管理与数据结构
- **句柄分配**: 使用 C++ `new` 操作符在堆上分配 `Handle` 对象，不使用智能指针，由调用者通过 `infiniopDestroyHandle` 释放
- **形状计算**: `getPaddedShape` 使用 `std::vector<size_t>` 动态存储结果，初始通过 `memcpy` 快速拷贝原始形状（O(ndim) 时间）
- **索引算法**: `indexToOffset` 使用反向迭代（从 `ndim` 递减到 0）实现扁平索引到多维索引的转换，每次迭代处理一个维度

### 并发与线程安全
- **OpenMP 支持**: 通过 `#ifdef ENABLE_OMP` 条件编译包含 OpenMP 头文件，为工具函数的并行化提供基础
- **无状态函数**: `common_cpu` 中的所有工具函数都是纯函数，无副作用，天然线程安全
- **句柄创建**: `Handle::create` 分配新内存，不涉及共享状态，多线程同时调用安全

### 性能优化
- **索引计算复杂度**: `indexToOffset` 时间复杂度为 O(ndim)，空间复杂度 O(1)，使用反向递减循环避免临时数组分配
- **内存拷贝优化**: `getPaddedShape` 使用 `memcpy` 拷贝形状数组，相比逐元素赋值更高效（编译器可向量化）
- **填充计算优化**: `getPaddedSize` 在单次循环中累乘所有维度，避免中间向量分配，O(ndim) 时间复杂度

### 错误处理
- **无参数验证**: 当前实现未对输入参数进行空指针检查或边界验证，依赖调用者保证参数合法性
- **状态返回**: `Handle::create` 固定返回 `INFINI_STATUS_SUCCESS`，不处理 `new` 失败的异常情况
- **整数溢出**: `getPaddedSize` 的累乘操作可能在大形状下溢出，未进行防护检查

### 设计模式
- **工厂方法模式**: `Handle::create` 静态方法封装对象创建，控制实例化过程
- **策略模式**: `Handle` 继承自 `InfiniopHandle`，与其他设备后端（CUDA、Ascend 等）实现统一接口
- **命名空间隔离**: 使用 `device::cpu` 和 `op::common_cpu` 命名空间避免符号冲突

### 依赖关系
- **外部依赖**: OpenMP（可选，通过 `ENABLE_OMP` 宏控制）
- **内部依赖**:
  - `infinicore.h`: 提供设备类型枚举和状态码定义
  - `../../handle.h`: 定义 `InfiniopHandle` 基类结构（包含 `device` 和 `device_id` 字段）
  - `../../../utils.h`: 提供通用工具宏和内联函数（`CEIL_DIV`, `utils::align`）
- **编译系统**: 需支持 C++14 或更高标准（使用 `std::vector`, `memcpy`）

### 填充语义细节
- **维度规则**: 填充仅应用于第 2 维及之后的维度（`i >= 2`），前两维（通常是 batch 和 channels）保持不变
- **填充位置**: 每个维度的两侧对称填充，每侧增加 `pads[i-2]`，总共增加 `2 * pads[i-2]`
- **典型用途**: 卷积神经网络中的 padding 操作，保持特征图空间分辨率或控制输出尺寸
- **零填充假设**: 当前仅计算形状和大小，不执行实际内存填充，调用者需负责数据初始化

### 算法实现细节

**indexToOffset 扁平索引转换算法**:
```
输入: flat_index = 5, shape = [2, 3, 4], strides = [48, 16, 4]
初始化: res = 0

迭代 i = 2 (最低维):
  res += (5 % 4) * 4 = 1 * 4 = 4
  flat_index = 5 / 4 = 1

迭代 i = 1:
  res += (1 % 3) * 16 = 1 * 16 = 16
  flat_index = 1 / 3 = 0

迭代 i = 0:
  res += (0 % 2) * 48 = 0
  flat_index = 0

最终: res = 20 (字节偏移)
对应多维索引: [0, 1, 1]
```

**步长语义**: 步长以字节为单位，表示在该维度上移动一个元素所需的内存偏移量，支持非连续张量和广播操作。
