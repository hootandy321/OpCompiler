# Bang 设备后端核心实现文档

本模块为 Infini 框架提供了 Cambricon MLU (Machine Learning Unit) 硬件后端支持，实现了基于 BangC 编程模型的算子基础设施。该模块封装了 CNRT (Cambricon Neural Network Runtime) 和 CNNL (Cambricon Neural Network Library) 的底层调用，为上层算子提供统一的设备抽象接口。

## 1. 模块结构

- **`bang_handle.h`**: 定义 Bang 设备句柄的抽象接口和 Cambricon 特化句柄类
- **`bang_handle.cc`**: 实现 Bang 设备句柄的生命周期管理、CNNL 句柄池以及类型转换工具函数
- **`bang_kernel_common.h`**: 提供 BangC 设备端核心工具函数，包括索引计算、广播处理和非连续内存复制优化
- **`common_bang.h`**: 定义内部实现类、常量配置和公共辅助函数接口

## 2. 核心类

### `device::bang::Handle`
- **位置**: `bang_handle.h`, `bang_handle.cc`
- **主要功能**: Bang 设备的抽象句柄基类，继承自 `InfiniopHandle`，封装 MLU 设备资源和 CNNL 运行时环境
- **关键成员**:
  - `_internal`: `std::shared_ptr<Internal>` 指针，采用 Pimpl (Pointer to Implementation) 模式隐藏实现细节
- **核心方法**:
  - `Handle(infiniDevice_t device, int device_id)`: 构造函数，初始化设备类型和设备 ID，创建内部实现对象
  - `internal() const`: 返回内部实现对象的共享指针，用于访问设备特定功能
- **生命周期**: 采用共享所有权语义，通过 `std::shared_ptr` 管理 `Internal` 对象，确保句柄复制时共享底层资源

### `Handle::Internal`
- **位置**: `common_bang.h`, `bang_handle.cc`
- **主要功能**: 实现句柄的内部逻辑，管理 CNNL 句柄池和设备拓扑信息
- **关键成员**:
  - `cnnl_handles`: `Pool<cnnlHandle_t>` 类型，CNNL 句柄对象池，用于多线程环境下的句柄复用
  - `_core_per_cluster`: `int` 类型，每个 MLU 集群 (Cluster) 包含的计算核心数量
  - `_cluster_count`: `int` 类型，MLU 设备上的集群总数
- **核心方法**:
  - `Internal(int device_id)`: 构造函数，调用 `cnrtDeviceGetAttribute` 查询设备拓扑属性，获取集群数和每集群核心数
  - `useCnnl(cnrtQueue_t queue, const Fn<cnnlHandle_t> &f) const`: 从句柄池中获取或创建 CNNL 句柄，绑定到指定的 CNRT 队列，执行用户传入的函数 `f`，使用完毕后归还句柄到池中
  - `getCorePerCluster() const`: 返回每个集群的计算核心数，用于并行任务调度
  - `getClusterCount() const`: 返回设备集群总数，用于计算并行度
- **生命周期**: 由 `Handle` 通过 `std::shared_ptr` 管理，实现句柄复制时的资源共享

### `device::bang::cambricon::Handle`
- **位置**: `bang_handle.h`, `bang_handle.cc`
- **主要功能**: Cambricon MLU 设备的具体实现类，继承自 `bang::Handle`
- **核心方法**:
  - `Handle(int device_id)`: 构造函数，调用父类构造函数并传入 `INFINI_DEVICE_CAMBRICON` 设备类型
  - `static create(InfiniopHandle **handle_ptr, int device_id)`: 工厂方法，动态分配 `Handle` 对象并通过输出参数返回句柄指针
- **生命周期**: 由静态工厂方法创建，调用者负责释放内存（返回裸指针）

### `device::bang::kernel::InputIndexer`
- **位置**: `bang_kernel_common.h`
- **主要功能**: 设备端索引计算辅助结构体，用于处理多输入张量的广播和步长（stride）映射
- **关键成员**:
  - `idx`: `size_t` 类型，当前任务的基准索引
  - `ndim`: `size_t` 类型，张量维度数
  - `input_contiguous`: `const bool*` 指针，标记每个输入张量是否内存连续
  - `input_broadcasted`: `const bool*` 指针，标记每个输入张量是否需要广播
  - `input_shapes`: `const size_t*` 指针，所有输入张量的形状数组（按 `ndim` 间隔拼接）
  - `input_strides`: `const ptrdiff_t*` 指针，所有输入张量的步长数组（按 `ndim` 间隔拼接）
  - `output_strides`: `const ptrdiff_t*` 指针，输出张量的步长数组
- **核心方法**:
  - `operator()(size_t input_id, size_t element_idx) const`: 函数调用运算符，计算给定输入张量的元素在输出坐标系中的内存偏移量。如果输入连续，直接返回 `idx + element_idx`；否则调用 `indexToOffset` 进行多维索引转换
- **使用场景**: 在二元/多元算子（如 add, mul）的设备端核函数中，用于处理不同形状张量之间的元素对齐

## 3. API 接口

```cpp
// 设备句柄创建工厂
namespace device::bang::cambricon {
    infiniStatus_t Handle::create(InfiniopHandle **handle_ptr, int device_id);
    // 功能: 创建 Cambricon MLU 设备句柄
    // 参数:
    //   - handle_ptr: 输出参数，返回创建的句柄指针
    //   - device_id: MLU 设备 ID（0 表示第一个设备）
    // 返回值: 成功返回 INFINI_STATUS_SUCCESS
}

// CNNL 句柄访问接口
infiniStatus_t Handle::Internal::useCnnl(cnrtQueue_t queue, const Fn<cnnlHandle_t> &f) const;
// 功能: 获取 CNNL 句柄并执行用户回调函数
// 参数:
//   - queue: CNRT 计算队列，用于绑定 CNNL 句柄
//   - f: 回调函数，接收 CNNL 句柄并执行具体 CNNL 算子调用
// 返回值: 成功返回 INFINI_STATUS_SUCCESS，失败返回对应的错误码

// 数据类型转换工具
cnnlDataType_t getCnnlDtype(infiniDtype_t dt);
// 功能: 将 Infini 框架的通用数据类型转换为 CNNL 特定类型
// 支持的类型: F32, F64, F16, BF16, I8, I32, I64, U8
// 返回值: 对应的 CNNL 数据类型枚举，不支持的类型返回 CNNL_DTYPE_INVALID

// 张量描述符设置工具
infiniStatus_t setCnnlTensor(cnnlTensorDescriptor_t desc, const InfiniopTensorDescriptor *layout);
// 功能: 设置 CNNL 张量描述符（不包含步长信息，仅适用于连续张量）
// 参数:
//   - desc: CNNL 张量描述符句柄（需预先创建）
//   - layout: Infini 张量描述符，包含形状和数据类型信息
// 实现细节: 调用 cnnlSetTensorDescriptor，使用 CNNL_LAYOUT_ARRAY 布局

infiniStatus_t setCnnlTensorEx(cnnlTensorDescriptor_t desc, const InfiniopTensorDescriptor *layout);
// 功能: 设置 CNNL 张量描述符（包含步长信息，支持非连续张量）
// 参数:
//   - desc: CNNL 张量描述符句柄（需预先创建）
//   - layout: Infini 张量描述符，包含形状、步长和数据类型信息
// 实现细节: 调用 cnnlSetTensorDescriptorEx，显式传递维度大小和步长数组
```

## 4. 使用示例

```cpp
// 示例: 使用 Cambricon MLU 设备执行 CNNL 算子
#include "infiniop/devices/bang/bang_handle.h"
#include "cnnl.h"

// 步骤 1: 创建设备句柄
InfiniopHandle* handle = nullptr;
int device_id = 0;  // 使用第一个 MLU 设备
auto status = device::bang::cambricon::Handle::create(&handle, device_id);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 步骤 2: 获取内部实现并查询设备拓扑
auto bang_handle = static_cast<device::bang::Handle*>(handle);
auto internal = bang_handle->internal();
int cluster_count = internal->getClusterCount();      // 例如: 16 个集群
int cores_per_cluster = internal->getCorePerCluster(); // 例如: 4 个核心/集群

// 步骤 3: 创建 CNRT 队列
cnrtQueue_t queue;
cnrtQueueCreate(&queue);

// 步骤 4: 使用 CNNL 句柄执行算子（例如: 矩阵乘法）
status = internal->useCnnl(queue, [](cnnlHandle_t cnnl_handle) {
    // 创建输入/输出张量描述符
    cnnlTensorDescriptor_t input_desc, output_desc;
    cnnlCreateTensorDescriptor(&input_desc);
    cnnlCreateTensorDescriptor(&output_desc);

    // 设置张量描述符
    InfiniopTensorDescriptor input_layout = {/* 形状: [128, 256], dtype: F32 */};
    InfiniopTensorDescriptor output_layout = {/* 形状: [128, 256], dtype: F32 */};
    setCnnlTensor(input_desc, &input_layout);
    setCnnlTensor(output_desc, &output_layout);

    // 执行 CNNL 算子（此处以激活函数为例）
    const float alpha = 1.0f, beta = 0.0f;
    void* input_ptr = /* MLU 设备内存指针 */;
    void* output_ptr = /* MLU 设备内存指针 */;
    cnnlStatus_t ret = cnnlActivationForward(cnnl_handle,
                                             CNNL_ACTIVATION_RELU,
                                             &alpha, input_desc, input_ptr,
                                             &beta, output_desc, output_ptr);

    // 清理描述符
    cnnlDestroyTensorDescriptor(input_desc);
    cnnlDestroyTensorDescriptor(output_desc);

    return (ret == CNNL_STATUS_SUCCESS) ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
});

// 步骤 5: 同步队列并清理资源
cnrtQueueSync(queue);
cnrtQueueDestroy(queue);
delete handle;  // 释放句柄（shared_ptr 自动管理内部资源）
```

## 5. 实现细节

### 内存管理

- **CNNL 句柄池**: 使用 `Pool<cnnlHandle_t>` 类（定义在 `../pool.h`）实现句柄对象池。在多线程环境下，每个线程可以从池中 pop 一个句柄，使用完毕后 push 回池中，避免频繁创建/销毁句柄的开销。首次调用 `useCnnl` 时会通过 `cnnlCreate` 创建新句柄。

- **NRAM (Neural RAM) 限制**: 定义常量 `NRAM_MAX_SIZE = 1024 * 240`（约 240KB），表示 MLU 片上高速缓存的大小。设备端核函数在加载数据时需确保单次拷贝的数据量不超过此限制。

- **内存对齐**: 定义常量 `ALIGN_SIZE = 128` 字节，符合 MLU 硬件对全局内存访问的对齐要求，优化内存带宽利用率。

### 并发控制

- **句柄池线程安全**: CNNL 句柄池的操作（`pop`/`push`）由 `Pool` 类内部实现同步机制（推测使用互斥锁或无锁队列），确保多线程环境下的正确性。

- **CNRT 队列隔离**: 每次调用 `useCnnl` 时，通过 `cnnlSetQueue` 将 CNNL 句柄绑定到用户指定的 CNRT 队列，实现不同任务流的隔离和并行执行。

- **设备拓扑感知**: 通过查询 `cnrtAttrClusterCount` 和 `cnrtAttrMcorePerCluster` 获取设备并行度信息，可用于指导任务调度策略（例如: 根据集群数划分并行任务）。

### 性能优化

- **非连续内存优化**: `bang_kernel_common.h` 中的 `calculateChunkSize` 函数实现了智能分块算法，自动检测张量的连续维度边界。对于部分连续的张量（例如: shape `[4, 3]`, strides `[3, 1]`），算法会识别最后一维是连续的，从而将内存拷贝操作合并为更大的块，减少 `__memcpy_async` 调用次数。时间复杂度 O(ndim)，空间复杂度 O(1)。

- **索引计算优化**: `indexToOffset` 函数采用从高维到低维的迭代算法（类似霍纳法则），通过模运算和除法逐步降维。对于连续张量，该函数会被短路（short-circuit），直接使用线性索引，避免冗余计算。

- **异步内存拷贝**: `nonContiguousMemcpy` 函数模板使用 MLU 的 `__memcpy_async` 内置函数，支持 GDRAM（全局显存）与 NRAM（片上缓存）之间的异步数据传输。在计算核心访问数据的同时，内存控制器可以并行处理下一批次的数据加载/存储，掩盖内存延迟。

- **广播零拷贝**: 对于广播操作（例如: 将向量 `[3]` 广播到矩阵 `[4, 3]`），`InputIndexer` 的设计使得广播维度无需实际复制数据。通过步长为 0 的索引逻辑（隐式在 `indexToOffset` 中处理），多个输出位置可以映射到同一个输入内存地址。

### 错误处理

- **统一错误检查宏**: 定义 `CHECK_BANG(API)` 宏，展开为 `CHECK_INTERNAL(API, CNNL_STATUS_SUCCESS)`。所有 CNRT/CNNL API 调用均通过此宏包装，自动检测返回值，并在失败时记录错误日志和错误码。

- **类型安全转换**: `getCnnlDtype` 函数使用 `switch-case` 枚举所有支持的 Infini 数据类型，对于未知类型返回 `CNNL_DTYPE_INVALID`，避免未定义行为。

- **错误传播**: `useCnnl` 方法接收的用户回调函数返回 `infiniStatus_t`，任何 CNNL 调用失败都会通过 `CHECK_STATUS` 宏立即返回错误码，终止当前操作并传播给上层调用者。

### 依赖关系

- **外部依赖**:
  - `cnnl.h`: Cambricon 神经网络计算库头文件，提供张量操作、激活函数、卷积等算子 API
  - `cnrt.h`: Cambricon 运行时库头文件，提供设备管理、队列管理、内存管理等底层 API
  - `../../../utils.h`: 工具宏定义（`CHECK_INTERNAL`, `CHECK_STATUS`）
  - `../../handle.h`: 设备句柄基类 `InfiniopHandle` 定义
  - `../../tensor.h`: 张量描述符 `InfiniopTensorDescriptor` 定义
  - `../pool.h`: 通用对象池模板类 `Pool<T>` 实现

- **内部模块依赖**: 本模块不依赖其他硬件后端（如 CUDA、ROCm），完全独立实现 Cambricon 特定逻辑。

### 设计模式

- **Pimpl 模式 (Pointer to Implementation)**: `bang::Handle` 类通过 `std::shared_ptr<Internal>` 成员隐藏实现细节，避免在头文件中暴露 CNNL/CNRT 的具体类型，降低编译依赖。

- **RAII (Resource Acquisition Is Initialization)**: 虽然未显式展示析构函数，但 `Internal` 类的 `cnnl_handles` 成员应在其析构时自动释放所有 CNNL 句柄（由 `Pool<cnnlHandle_t>` 的析构函数处理）。

- **工厂方法模式**: `cambricon::Handle::create` 静态方法封装对象创建逻辑，返回抽象基类指针 `InfiniopHandle*`，支持运行时多态。

- **函数对象模式**: `InputIndexer` 重载 `operator()`，使其可作为高阶函数传递，简化设备端索引计算的调用语法。

- **策略模式**: `useCnnl` 方法接受函数参数 `f`，允许调用者自定义 CNNL 句柄的使用逻辑，实现句柄管理策略与业务逻辑的解耦。
