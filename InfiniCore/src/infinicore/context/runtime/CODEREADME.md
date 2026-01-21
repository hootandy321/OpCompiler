# Runtime Core Implementation Documentation

Runtime 是 InfiniCore 的核心运行时管理类，负责设备上下文、内存分配、CUDA 流管理、事件计时以及计算图录制等底层基础设施功能。它是连接上层抽象与底层硬件加速库（infinirt 和 infiniop）的桥梁。

## 1. Module Structure

- **`runtime.hpp`**: Runtime 类的接口声明，定义了设备管理、内存分配、数据传输、事件计时和图录制等核心 API
- **`runtime.cc`**: Runtime 类的实现，封装了 infiniRT 和 infiniOP 的调用逻辑

## 2. Core Classes

### `Runtime`
- **Location**: `runtime.hpp`, `runtime.cc`
- **Primary Function**: 管理单个计算设备的运行时环境，提供设备激活、内存管理、异步执行、性能计量和图录制等基础设施服务
- **Key Members**:
  - `device_`: `Device` - 目标计算设备（CPU、CUDA 等）
  - `stream_`: `infinirtStream_t` - 异步执行流，用于并行执行操作
  - `infiniop_handle_`: `infiniopHandle_t` - infiniOP 库的句柄，用于执行算子操作
  - `device_memory_allocator_`: `std::unique_ptr<PinnableBlockAllocator>` - 设备内存分配器，使用 PinnableBlockAllocator 管理设备内存
  - `pinned_host_memory_allocator_`: `std::unique_ptr<MemoryAllocator>` - 锁页主机内存分配器（仅 GPU），使用 DevicePinnedHostAllocator 实现零拷贝优化
  - `graph_manager_`: `std::unique_ptr<graph::GraphManager>` - 计算图管理器，用于录制和优化计算图
- **Core Methods**:
  - `Runtime(Device device)`: 构造函数，初始化设备、流、infiniop 句柄和内存分配器；根据设备类型选择分配器策略（CPU/GPU）
  - `~Runtime()`: 析构函数，按正确顺序释放资源（先释放分配器，再销毁 infiniop 句柄和流）
  - `activate()`: 激活当前运行时所在的设备（调用 `infinirtSetDevice`），确保后续操作在正确设备上执行；返回 this 支持链式调用
  - `allocateMemory(size_t size)`: 从设备内存分配器分配指定大小的内存，返回带有自定义删除器的 `shared_ptr<Memory>`，利用 RAII 自动管理内存生命周期
  - `allocatePinnedHostMemory(size_t size)`: 分配锁页主机内存（仅 GPU），实现 DMA 加速传输；CPU 设备回退到普通内存分配
  - `memcpyH2D/D2H/D2D(...)`: 封装主机到设备、设备到主机、设备到设备的内存拷贝，支持异步（async=true）和同步模式
  - `createEvent()`, `recordEvent()`, `elapsedTime()`: 事件计时方法集，用于性能分析；支持在指定流上记录时间点、计算耗时
  - `syncStream()`, `syncDevice()`: 同步流和设备，确保所有异步操作完成
  - `startGraphRecording()`, `stopGraphRecording()`, `addGraphOperator()`: 计算图录制接口，开启/停止录制并添加算子，用于图优化和加速
- **Lifecycle**:
  - **Construction**: 由 `ContextImpl` 友元类构造，接收设备参数并初始化所有子系统（设备激活、流创建、infiniop 初始化、分配器初始化）
  - **Active State**: 构造后自动激活设备，后续操作在激活设备上执行；支持 `activate()` 方法切换设备
  - **Destruction**: 析构时按依赖顺序释放资源（分配器 → infiniop 句柄 → 流），确保无资源泄漏
  - **Ownership**: 由 `ContextImpl` 独占管理，用户通过 Context 接口间接访问

## 3. API Interface

```cpp
class Runtime {
    // 设备管理
    Device device() const;
    Runtime* activate();  // 返回 this 支持链式调用

    // 流和句柄访问
    infinirtStream_t stream() const;
    infiniopHandle_t infiniopHandle() const;

    // 同步控制
    void syncStream();   // 同步当前流
    void syncDevice();   // 同步整个设备

    // 内存分配（返回 RAII 管理的智能指针）
    std::shared_ptr<Memory> allocateMemory(size_t size);
    std::shared_ptr<Memory> allocatePinnedHostMemory(size_t size);

    // 内存拷贝（支持异步）
    void memcpyH2D(void* dst, const void* src, size_t size, bool async = true);
    void memcpyD2H(void* dst, const void* src, size_t size);
    void memcpyD2D(void* dst, const void* src, size_t size, bool async = true);

    // 事件计时 API
    infinirtEvent_t createEvent();
    infinirtEvent_t createEventWithFlags(uint32_t flags);
    void recordEvent(infinirtEvent_t event, infinirtStream_t stream = nullptr);
    bool queryEvent(infinirtEvent_t event);  // 查询事件是否完成
    void synchronizeEvent(infinirtEvent_t event);
    void destroyEvent(infinirtEvent_t event);
    float elapsedTime(infinirtEvent_t start, infinirtEvent_t end);  // 返回毫秒
    void streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event);

    // 计算图录制 API
    bool isGraphRecording() const;
    void startGraphRecording();
    void addGraphOperator(std::shared_ptr<graph::GraphOperator> op);
    std::shared_ptr<graph::Graph> stopGraphRecording();
};
```

## 4. Usage Example

```cpp
// 示例：使用 Runtime 进行设备操作和性能计量
#include "infinicore/context/runtime.hpp"

using namespace infinicore;

// 1. 创建 Runtime（由 ContextImpl 内部管理）
Device gpu_device(Device::Type::CUDA, 0);  // 第一个 GPU
Runtime runtime(gpu_device);

// 2. 分配内存（RAII 自动管理）
auto device_buffer = runtime.allocateMemory(1024 * 1024);  // 1MB 设备内存
auto pinned_buffer = runtime.allocatePinnedHostMemory(1024);  // 锁页主机内存

// 3. 异步数据传输
float host_data[256];
runtime.memcpyH2D(device_buffer->data(), host_data, sizeof(host_data), async = true);

// 4. 同步等待
runtime.syncStream();

// 5. 性能计量
auto start = runtime.createEvent();
auto end = runtime.createEvent();

runtime.recordEvent(start);
// ... 执行操作 ...
runtime.recordEvent(end);

float elapsed_ms = runtime.elapsedTime(start, end);
printf("Kernel time: %.3f ms\n", elapsed_ms);

runtime.destroyEvent(start);
runtime.destroyEvent(end);

// 6. 计算图录制（用于优化）
runtime.startGraphRecording();
runtime.addGraphOperator(op1);  // 添加算子
runtime.addGraphOperator(op2);
auto optimized_graph = runtime.stopGraphRecording();
optimized_graph->execute();  // 执行优化后的图

// 内存自动释放（无需手动 free）
```

## 5. Implementation Details

- **内存管理策略**:
  - 使用 **PinnableBlockAllocator** 管理设备内存，支持内存固定（pinning）模式用于计算图录制
  - GPU 设备额外使用 **DevicePinnedHostAllocator** 分配锁页主机内存，启用 DMA 零拷贝传输
  - CPU 设备不使用锁页内存（`pinned_host_memory_allocator_` 为 nullptr），分配时回退到普通内存
  - 所有内存分配返回 `shared_ptr<Memory>`，通过 lambda 捕获分配器指针实现自动释放，避免内存泄漏

- **并发与流管理**:
  - 每个 Runtime 拥有独立的 `infinirtStream_t`，支持多流并行执行
  - 异步拷贝（`async=true`）在流上非阻塞执行，提高吞吐量
  - `activate()` 方法调用 `infinirtSetDevice` 确保多 GPU 环境下操作在正确设备上执行
  - 事件与流的依赖关系通过 `streamWaitEvent()` 管理，实现精确的执行顺序控制

- **错误处理**:
  - 所有 infiniRT/infiniOP 调用通过 `INFINICORE_CHECK_ERROR` 宏检查错误码，失败时抛出异常
  - 锁页内存分配失败时（CPU 设备）回退到普通内存并打印警告（`spdlog::warn`）

- **计算图录制机制**:
  - 图录制期间调用 `device_memory_allocator_->set_pin_mode(true)` 固定内存地址，确保图的可重用性
  - `GraphManager` 管理录制的算子序列，`stopGraphRecording()` 返回优化的 `Graph` 对象
  - 录制完成后关闭固定模式（`set_pin_mode(false)`），恢复正常内存分配

- **资源管理**:
  - 析构函数按依赖顺序释放资源：先释放分配器（可能持有设备内存），再销毁 infiniop 句柄和流
  - RAII 设计确保异常安全，即使中途出错也能正确释放已分配资源

- **性能优化**:
  - 锁页主机内存避免 DMA 分页开销，加速 H2D/D2H 传输
  - 异步执行隐藏传输延迟，提高设备利用率
  - 事件计时精度达微秒级（依赖 infiniRT 实现），支持性能剖析

- **设计模式**:
  - **RAII**: 内存分配使用智能指针自动管理生命周期
  - **Friend Class**: `ContextImpl` 作为友元类独占 Runtime 构造和访问权限，封装实现细节
  - **Facade Pattern**: Runtime 封装复杂的 infiniRT/infiniOP 调用，提供简洁的高级 API

- **依赖关系**:
  - **外部库**: 依赖 `infiniRT`（硬件抽象层）和 `infiniOP`（算子库）
  - **内部模块**:
    - `PinnableBlockAllocator`: 设备内存分配
    - `DevicePinnedHostAllocator`: 锁页主机内存分配（GPU）
    - `GraphManager`: 计算图录制和管理
    - `Memory`: 封装内存块的 RAII 类
    - `Device`: 设备类型和索引标识
