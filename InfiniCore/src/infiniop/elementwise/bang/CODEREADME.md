# Elementwise Operations - BANG Backend Core Implementation Documentation

此模块实现了 InfiniOP 框架中逐元素操作（elementwise operations）的 BANG 硬件后端。BANG 是寒武纪（Cambricon）MLU 设备的编程架构。该模块提供了高性能的逐元素计算内核，支持广播、非连续内存布局和多种数据类型。

## 1. 模块结构

- **`elementwise_bang_api.h`**: 定义 BANG 设备实现的公共 API 接口和宏，提供 `DeviceImpl` 类的声明和操作符创建宏。
- **`elementwise_bang.h`**: 实现 `DeviceImpl` 的核心逻辑，包含元数据传输到设备内存的实现和模板化计算调度。
- **`elementwise_bang_kernel.h`**: 实现 BANG MLU 设备内核代码，包含设备端逐元素计算核心、内存复制优化和内核启动配置。

## 2. 核心类

### `DeviceImpl`
- **位置**: `elementwise_bang_api.h`, `elementwise_bang.h`
- **主要功能**: 封装 BANG 设备上的逐元素操作执行，提供类型安全的模板化接口
- **关键成员**:
  - `_opaque`: `std::shared_ptr<Opaque>` - Pimpl 模式的不透明实现指针，隐藏设备特定细节
  - `Opaque::internal`: `std::shared_ptr<device::bang::Handle::Internal>` - BANG 设备句柄内部实现，包含硬件配置信息

- **核心方法**:
  - `create<Args...>(Args&&... args)`: `Result<DeviceImpl*>` - 工厂方法，创建 `DeviceImpl` 实例，采用完美转发将参数传递给 `Opaque` 构造函数
  - `calculate<Op, Tdata, Args...>(ElementwiseInfo&, void*, void*, const vector<const void*>&, void*, Args&&...)`: `infiniStatus_t` - 执行逐元素计算的主入口，从操作符 Functor 提取输入数量 `N`，分派到 `calculateImpl`

- **生命周期**: 使用 Pimpl（Pointer to Implementation）模式，`DeviceImpl` 持有 `Opaque` 的共享指针，确保资源管理和实现细节的完全隐藏。构造通过 `create()` 工厂方法进行，析构为默认实现。

### `DeviceImpl::Opaque`
- **位置**: `elementwise_bang.h`
- **主要功能**: 实现逐元素计算的设备端逻辑，处理元数据传输和内核启动
- **关键成员**:
  - `internal`: `std::shared_ptr<device::bang::Handle::Internal>` - BANG 设备句柄，提供核心数、集群数等硬件信息

- **核心方法**:
  - `calculateImpl<N, Op, Tdata, Args...>(...)`: `infiniStatus_t` - 核心计算实现，首先检查输出大小是否为零（早期退出），然后调用 `infoToDevice()` 将主机元数据复制到设备，最后调用 `Op::launch()` 启动设备内核，使用 `cnrtQueueSync()` 同步队列
  - `infoToDevice<N>(...)`: `infiniStatus_t` - 将逐元素操作的元数据（输入指针数组、形状、步幅、连续性标志、广播标志）从主机内存传输到设备内存。计算设备内存布局：`workspace` 存储输入指针数组，后续空间存储元数据（输出形状、输出步幅、输入形状、输入步幅、连续性标志、广播标志）。使用 `cnrtMemcpy()` 执行主机到设备的异步复制。

- **生命周期**: 由 `DeviceImpl::create()` 通过 `std::make_shared` 构造，生命周期由 `DeviceImpl` 的共享指针管理

### 内核函数和设备端核心
- **位置**: `elementwise_bang_kernel.h`
- **主要功能**: 在 MLU 设备上执行逐元素计算，优化 NRAM（Neural RAM）使用和内存访问模式

- **核心设备函数**:
  - `launchOp<N, Op, Tdata, Args...>(...)`: `__mlu_device__ void` - 设备端逐元素操作实现。进行 NRAM 内存规划：`nram_usable = NRAM_MAX_SIZE - (ALIGN_SIZE * (N + 1))`，计算最大批次大小 `max_batch = nram_usable / ((N + 1) * sizeof(Tdata))`。分批处理元素：
    1. **输入阶段**: 将输入数据从 GDRAM 复制到 NRAM。对于连续输入使用 `__memcpy_async()` 批量复制，对于非连续输入调用 `nonContiguousMemcpy()` 逐元素复制。调用 `__sync_io()` 同步内存传输。
    2. **计算阶段**: 实例化操作符 `Op op`，调用 `op(output_buffer, input_buffers[0], input_buffers[1], curr_batch, args...)` 执行逐元素计算。调用 `__sync_compute()` 同步计算。
    3. **输出阶段**: 将结果从 NRAM 写回 GDRAM。对于连续输出使用 `__memcpy_async()` 批量复制，对于非连续输出调用 `nonContiguousMemcpy()` 逐元素写入。

  - `elementwiseKernel<N, Op, Tdata, Args...>(...)`: `__mlu_global__ void` - BANG 全局内核入口。计算每个任务的工作负载：`elements_per_task = (output_size + taskDim - 1) / taskDim`，确定当前任务的起始和结束索引。分配 NRAM 缓冲区：`__nram__ Tdata nram_buf[NRAM_MAX_SIZE / sizeof(Tdata)]`。使用 `getOutputIndex()` 计算输出索引，创建 `InputIndexer` 辅助对象用于处理广播和非连续布局。为每个输入计算索引偏移量，然后调用 `launchOp()` 执行实际计算。

  - `launchElementwiseKernelWrapper<N, Op, Tdata, Args...>(...)`: `void` - 内核启动的中间层，确定最优启动配置。从 `internal` 获取硬件信息：`core_per_cluster`（每个集群的核心数）和 `cluster_count`（集群数量）。设置内核启动维度：`dim.x = core_per_cluster`, `dim.y = cluster_count`, `dim.z = 1`。根据问题特征选择内核类型：对于大型连续操作（`output_size > 1MB` 且 `output_contiguous`），使用 `CNRT_FUNC_TYPE_UNION1`；否则使用 `CNRT_FUNC_TYPE_BLOCK`。使用 `<<<dim, func_type, queue>>>` 语法启动内核。

- **辅助类和函数**:
  - `InputIndexer`: 从 `device::bang::kernel` 命名空间导入，用于计算多维数组索引，处理广播和步幅
  - `nonContiguousMemcpy<Tdata>(...)`: 从 `device::bang::kernel` 命名空间导入，处理非连续内存的逐元素复制
  - `getOutputIndex(...)`: 从 `device::bang::kernel` 命名空间导入，计算给定线性索引的输出内存偏移

## 3. API 接口

```cpp
// 创建 BANG 设备实现实例
template <typename... Args>
utils::Result<DeviceImpl *> DeviceImpl::create(Args &&...args);
// 参数通过完美转发传递给 Opaque 构造函数
// 返回包含新 DeviceImpl 实例的 Result 对象

// 执行逐元素操作
template <typename Op, typename Tdata, typename... Args>
infiniStatus_t DeviceImpl::calculate(
    const op::elementwise::ElementwiseInfo &info,  // 操作元数据（形状、步幅等）
    void *workspace,                                // 设备工作空间内存
    void *output,                                   // 输出张量缓冲区
    const std::vector<const void *> &inputs,        // 输入张量指针向量
    void *queue,                                    // BANG 队列（cnrtQueue_t 的 void* 包装）
    Args &&...args);                                // 操作符的额外参数（如标量值）
// 返回 INFINI_STATUS_SUCCESS 或错误代码

// 操作符必须实现的接口（Op Functor 要求）
struct Op {
    static constexpr size_t num_inputs = N;         // 输入张量数量

    template <typename Tdata, typename... Args>
    static void launch(
        size_t output_size,                         // 输出元素总数
        size_t ndim,                                // 维度数量
        bool output_contiguous,                     // 输出是否连续
        const void *input_contiguous,               // 输入连续性标志数组
        const void *input_broadcasted,              // 输入广播标志数组
        const void *output_shape,                   // 输出形状
        const void *input_shapes,                   // 输入形状
        const void *output_strides,                 // 输出步幅
        const void *input_strides,                  // 输入步幅
        void *output,                               // 输出缓冲区
        const void *const *inputs,                  // 输入缓冲区数组
        cnrtQueue_t queue,                          // BANG 队列
        const std::shared_ptr<device::bang::Handle::Internal> &internal, // 设备句柄
        Args... args);                              // 额外参数
};
```

### 宏接口

```cpp
// 创建 BANG 逐元素操作描述符的宏
#define CREATE_ELEMENTWISE_BANG_DESCRIPTOR(HANDLE, DTYPE, OUT_DESC, INPUT_DESC_VEC)
// HANDLE: 设备句柄指针
// DTYPE: 输出数据类型（如 float, int32_t）
// OUT_DESC: 输出张量描述符
// INPUT_DESC_VEC: 输入张量描述符向量
// 展开后创建 ElementwiseInfo、计算工作空间大小、创建 DeviceImpl、构造 Descriptor

// 声明特定操作符内核的宏
#define LAUNCH_ELEMENTWISE_KERNEL(OpName)
// OpName: 操作名称（如 Add, Mul）
// 展开后声明 launch##OpName##Kernel<Tdata, Args...> 函数

// 实现特定操作符内核的宏
#define LAUNCH_ELEMENTWISE_KERNEL_IMPL(OpName, Op)
// OpName: 操作名称
// Op: 操作符 Functor 类型
// 展开后实现 launch##OpName##Kernel，调用 launchElementwiseKernelWrapper

// 实例化特定操作符内核的宏
#define LAUNCH_ELEMENTWISE_KERNEL_INSTANTIATE(OpName, T, ...)
// OpName: 操作名称
// T: 数据类型（如 float, half）
// ...: 额外模板参数
// 展开后显式实例化模板函数以生成目标代码
```

## 4. 使用示例

```cpp
// 示例：在 BANG 设备上执行逐元素加法操作

#include "elementwise_bang_api.h"
#include "elementwise_bang.h"

using namespace op::elementwise::bang;

// 1. 准备设备句柄和操作符
auto handle = getBangDeviceHandle(); // 获取 BANG 设备句柄
using AddOp = some::add::Operator;  // 假设的加法操作符

// 2. 创建张量描述符（假设已准备好）
TensorDescriptor output_desc = {/* ... */};
std::vector<TensorDescriptor> input_descs = {/* 输入1, 输入2 */};

// 3. 使用宏创建操作描述符
Descriptor* desc_ptr = nullptr;
CREATE_ELEMENTWISE_BANG_DESCRIPTOR(handle, float, output_desc, input_descs);

// 4. 准备数据和工作空间
void* d_output = allocateDeviceMemory(output_desc.size());
void* d_input1 = allocateDeviceMemory(input_descs[0].size());
void* d_input2 = allocateDeviceMemory(input_descs[1].size());
void* workspace = allocateDeviceMemory(desc_ptr->workspace_size);
cnrtQueue_t queue = createBangQueue(handle);

// 5. 执行逐元素加法
std::vector<const void*> inputs = {d_input1, d_input2};
infiniStatus_t status = desc_ptr->device_impl->calculate<AddOp, float>(
    desc_ptr->info,      // ElementwiseInfo
    workspace,           // 设备工作空间
    d_output,            // 输出缓冲区
    inputs,              // 输入缓冲区数组
    queue                // BANG 队列
);

// 6. 等待完成并清理
if (status == INFINI_STATUS_SUCCESS) {
    cnrtQueueSync(queue); // 确保操作完成
    // 使用 d_output 中的结果
}

// 内部执行流程（简化）：
// 1. calculateImpl 检查输出大小，调用 infoToDevice
// 2. infoToDevice 复制元数据到设备（输入指针、形状、步幅、标志）
// 3. AddOp::launch 调用 launchElementwiseKernelWrapper
// 4. wrapper 选择内核类型（BLOCK 或 UNION1），计算启动维度
// 5. elementwiseKernel 在 MLU 上启动：
//    - 每个任务计算其工作负载范围
//    - 分配 NRAM 缓冲区
//    - 创建 InputIndexer 处理索引计算
//    - 调用 launchOp：
//      a. 分批从 GDRAM 复制输入到 NRAM
//      b. 在 NRAM 上执行逐元素加法
//      c. 将结果从 NRAM 写回 GDRAM
```

## 5. 实现细节

### 内存管理
- **NRAM 分层策略**: 使用 Neural RAM（MLU 的片上高速内存）作为三级缓存层次结构的顶层。计算最大可用 NRAM：`nram_usable = NRAM_MAX_SIZE - (ALIGN_SIZE * (N + 1))`，其中 `ALIGN_SIZE` 用于内存对齐。计算批次大小：`max_batch = nram_usable / ((N + 1) * sizeof(Tdata))`，确保所有输入和输出缓冲区都能放入 NRAM。
- **内存对齐**: 使用 `ALIGN_SIZE` 对齐 NRAM 缓冲区地址：`(reinterpret_cast<size_t>(nram_buf) + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1)`，确保高效内存访问。
- **异步内存复制**: 使用 `__memcpy_async()` 在 GDRAM 和 NRAM 之间异步复制数据，隐藏内存延迟。调用 `__sync_io()` 显式同步内存传输，确保数据就绪后再执行计算。
- **工作空间布局**: 设备工作空间内存布局：`[输入指针数组 (N * sizeof(void*))] [输出形状 (ndim * sizeof(size_t))] [输出步幅 (ndim * sizeof(ptrdiff_t))] [输入形状 (N * ndim * sizeof(size_t))] [输入步幅 (N * ndim * sizeof(ptrdiff_t))] [连续性标志 (N * sizeof(bool))] [广播标志 (N * sizeof(bool))]`。

### 并发
- **任务并行**: 使用 BANG 的任务并行模型，根据硬件配置（`core_per_cluster`, `cluster_count`）设置启动维度 `dim.x` 和 `dim.y`。每个任务处理输出元素的一个子集：`elements_per_task = (output_size + taskDim - 1) / taskDim`，任务索引通过 `taskId` 内置变量获取。
- **内核类型选择**: 根据问题特征动态选择内核类型。对于大型连续操作（`output_size > 1MB` 且 `output_contiguous`），使用 `CNRT_FUNC_TYPE_UNION1` 以获得更好的资源利用。对于小型或非连续操作，使用 `CNRT_FUNC_TYPE_BLOCK` 以减少启动开销。
- **同步**: 使用 `cnrtQueueSync()` 在主机端同步队列，确保内核执行完成。在设备端使用 `__sync_io()` 和 `__sync_compute()` 分别同步内存传输和计算操作，避免数据竞争。

### 性能
- **批次处理**: 采用批次处理策略，每次处理 `max_batch` 个元素，最大化 NRAM 利用率。减少全局内存访问次数，提高内存带宽利用率。时间复杂度：O(n)，其中 n 是输出元素数量，每个元素仅处理一次。
- **连续内存优化**: 对于连续内存布局使用 `__memcpy_async()` 批量复制，充分利用内存带宽。对于非连续内存回退到 `nonContiguousMemcpy()` 逐元素复制，保证正确性。
- **索引计算优化**: 使用 `InputIndexer` 类缓存广播和步幅信息，避免重复计算。预先计算 `input_indexes` 数组，减少循环内的索引计算开销。
- **硬件感知配置**: 根据硬件特性（核心数、集群数）动态调整并行度。对于大于 1MB 的操作使用 UNION 内核类型，可能启用跨集群的更细粒度并行。

### 错误处理
- **早期退出**: 在 `calculateImpl` 中检查 `output_size == 0`，直接返回 `INFINI_STATUS_SUCCESS`，避免无效的内核启动。
- **状态码传播**: 使用 `CHECK_STATUS` 和 `CNRT_CHECK` 宏检查错误，将 BANG 运行时错误（如 `cnrtMemcpy` 失败）转换为 `infiniStatus_t` 状态码。错误通过返回值传播到调用者，不使用异常。
- **类型安全**: 使用 `static_assert(N == Op::num_inputs, "template N is not equal to Op::num_inputs!")` 在编译时验证模板参数一致性，防止运行时错误。

### 依赖
- **BANG 运行时**: 依赖 Cambricon CNRT（Compute Runtime）API，包括 `cnrtMemcpy`、`cnrtQueueSync`、`cnrtDim3_t`、`cnrtFunctionType_t`。使用 `CNRT_MEM_TRANS_DIR_HOST2DEV`、`GDRAM2NRAM`、`NRAM2GDRAM` 等内存复制方向常量。
- **设备通用工具**: 依赖 `device::bang::common_bang.h` 提供的 BANG 设备句柄和通用宏。依赖 `device::bang::bang_kernel_common.h` 提供的内核辅助函数（`InputIndexer`、`nonContiguousMemcpy`、`getOutputIndex`、`NRAM_MAX_SIZE`、`ALIGN_SIZE`）。
- **上层抽象**: 依赖 `op::elementwise::ElementwiseInfo`（来自 `../elementwise.h`）封装操作元数据。依赖 `infiniStatus_t` 枚举（来自 `../../../utils.h`）表示操作状态。

### 设计模式
- **Pimpl（Pointer to Implementation）**: `DeviceImpl` 使用不透明指针模式隐藏 `Opaque` 实现细节，减少编译依赖和头文件暴露。
- **CRTP（Curiously Recurring Template Pattern）**: 操作符 Functor 通过静态 `launch` 方法提供定制实现，`DeviceImpl::calculate` 在编译时多态分派到具体操作符。
- **策略模式**: 内核类型（`CNRT_FUNC_TYPE_BLOCK` vs `CNRT_FUNC_TYPE_UNION1`）根据运行时特征选择，实现不同的并行策略。
- **工厂模式**: `DeviceImpl::create()` 作为工厂方法封装实例化逻辑，返回 `Result<DeviceImpl*>` 提供错误安全的构造。
- **模板方法模式**: `elementwiseKernel` 定义算法骨架（工作分区、索引计算），`launchOp` 实现具体步骤（内存复制、计算、写回），支持不同的操作符和内存布局。
