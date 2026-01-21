# `infinicore::context::allocators` 内存分配器模块文档

该模块实现了 InfiniCore 框架的多策略内存分配系统，提供从基础主机内存分配到高性能设备内存池的完整解决方案。

## 模块结构 (Module Structure)

- **MemoryAllocator** (`memory_allocator.hpp`)：内存分配器的抽象基类，定义统一接口
- **HostAllocator** (`host_allocator.hpp/cc`)：基于标准 malloc/free 的主机内存分配器
- **DevicePinnedHostAllocator** (`device_pinned_allocator.hpp/cc`)：设备固定内存（页锁定内存）分配器，支持跨设备延迟释放
- **PinnableBlockAllocator** (`pinnable_block_allocator.hpp/cc`)：高性能大小分类内存池分配器，支持固定/图模式切换
- **StreamOrderedAllocator** (`stream_ordered_allocator.hpp/cc`)：流有序异步内存分配器，与 CUDA Stream 同步

## 核心类详解 (Key Classes)

### MemoryAllocator（抽象基类）

- **定义位置**：`context/allocators/memory_allocator.hpp:8-14`
- **主要功能**：定义所有内存分配器的统一接口规范
- **关键成员**：无（纯接口类）
- **核心方法**：
    - `virtual std::byte *allocate(size_t size) = 0`：分配指定大小的内存块
    - `virtual void deallocate(std::byte *ptr) = 0`：释放之前分配的内存块
- **设计模式**：抽象工厂模式，支持运行时多态分配策略

### HostAllocator

- **定义位置**：`context/allocators/host_allocator.hpp:6-13`
- **主要功能**：封装标准 C 库的内存分配函数，提供对 CPU 内存的基本管理
- **关键成员**：无（无状态分配器）
- **核心方法**：
    - `std::byte *allocate(size_t size)`：调用 `std::malloc` 分配主机内存，处理 size=0 边界情况
    - `void deallocate(std::byte *ptr)`：调用 `std::free` 释放内存，包含 nullptr 安全检查
- **实现细节**：
    - 使用 `infinirt.h` 中的 INFINICORE_CHECK_ERROR 宏进行错误检查（虽然 malloc/free 不会失败）
    - 适用于不需要设备交互的纯 CPU 计算场景

### DevicePinnedHostAllocator

- **定义位置**：`context/allocators/device_pinned_allocator.hpp:10-25`
- **主要功能**：分配"页锁定内存"（Pinned Memory），这种内存不会被操作系统换出到磁盘，支持 DMA 高速传输
- **关键成员**：
    - `Device owner_`：拥有此分配器的设备标识
    - `std::queue<std::byte *> gc_queue_`：延迟释放队列（TODO 注明：非线程安全）
- **核心方法**：
    - `std::byte *allocate(size_t size)`：调用 `infinirtMallocHost` 分配固定内存
    - `void deallocate(std::byte *ptr)`：
        - 如果当前设备与 owner_ 相同，立即释放并执行垃圾回收
        - 否则，将指针推入 gc_queue_ 延迟释放
    - `void gc()`：清空垃圾回收队列，逐个调用 `infinirtFreeHost` 释放内存
- **算法逻辑**：**跨设备延迟释放策略**
    - 问题背景：在不同设备上下文中释放固定内存可能导致竞争条件
    - 解决方案：使用队列缓存待释放指针，在下次分配或析构时统一清理
    - 析构函数自动调用 `gc()` 确保内存泄漏不会发生
- **使用场景**：CPU-GPU 数据传输、多设备协同计算

### PinnableBlockAllocator（⭐ 核心组件）

- **定义位置**：`context/allocators/pinnable_block_allocator.hpp:10-48`
- **主要功能**：实现基于**大小分类（Size-Class）**的高性能内存池，支持**固定模式（Pinned Mode）**与**普通模式**的动态切换
- **内部数据结构**：
    - **Block 结构**（行 12-17）：
        - `void *ptr`：设备内存指针
        - `size_t size`：块大小（字节）
        - `bool frozen`：是否被固定（在固定/图模式下分配的块）
        - `bool in_use`：是否正在使用（用于检测双重释放）
    - **SizeClass 结构**（行 20-23）：
        - `size_t block_size`：该类别固定的块大小
        - `std::vector<std::shared_ptr<Block>> free_blocks`：空闲块缓存池
- **关键成员**：
    - `Device device_`：目标设备
    - `bool pinned_mode_`：当前是否处于固定模式（影响新分配块的 `frozen` 标志）
    - `std::vector<SizeClass> size_classes_`：11 个预定义大小等级（见下文）
    - `std::vector<std::shared_ptr<Block>> large_blocks_`：超过最大大小等级的大块列表
    - `std::unordered_map<void *, std::shared_ptr<Block>> all_blocks_`：所有已分配块的索引（用于 O(1) 查找）
    - `std::mutex mutex_`：保证线程安全
- **核心方法**：

    #### `std::byte *allocate(size_t size)`（行 39-96）

    **算法流程**：

    1. **对齐处理**：将 size 向上对齐到 256 字节边界（GPU 内存对齐要求）
    2. **大小分类匹配**：
        - 遍历 `size_classes_` 数组（从小到大）
        - 找到第一个满足 `size <= cls.block_size` 的类别
        - 如果该类别的 `free_blocks` 非空，复用缓存块
        - 否则调用 `infinirtMalloc` 分配新块，标记 `frozen = pinned_mode_`
    3. **大块分配**：
        - 如果超过所有大小等级，搜索 `large_blocks_` 列表
        - 查找第一个满足 `size >= block->size && !block->in_use` 的块
        - 若未找到，分配新的大块
    4. **块记录**：将新块添加到 `all_blocks_` 索引

    **大小等级定义**（行 23-35）：
    ```
    32 KB, 256 KB, 1 MB, 2 MB, 4 MB, 8 MB,
    16 MB, 32 MB, 64 MB, 128 MB, 256 MB
    ```

    #### `void deallocate(std::byte *ptr)`（行 99-126）

    **算法流程**：

    1. 查找 `all_blocks_` 索引定位块
    2. 检查 `in_use` 标志，防止双重释放（抛出异常）
    3. 清除 `in_use` 标志
    4. 如果块属于某个大小等级，添加到对应 `free_blocks` 缓存池

    #### `void trim()`（行 129-153）

    **功能**：释放所有非固定（`!frozen`）且未使用的缓存块，归还内存给 GPU

    #### `void set_pin_mode(bool pinned)`（行 33）
    **功能**：动态切换固定模式，影响后续 `allocate` 分配的块的 `frozen` 属性

- **初始化/生命周期**：
    - 构造函数：初始化 11 个大小等级，所有 free_blocks 为空
    - 析构函数：遍历 `all_blocks_`，逐个调用 `infinirtFree` 释放所有内存

- **算法亮点**：
    1. **Size-Class 分配策略**：减少内存碎片，提高分配效率（类似 jemalloc 的思想）
    2. **Block 复用机制**：释放的块缓存到 free_blocks，避免频繁调用底层 API
    3. **固定模式支持**：`frozen` 标志保护关键内存块不被 `trim()` 误释放
    4. **线程安全**：所有公共方法都使用 `std::lock_guard<std::mutex>` 保护

### StreamOrderedAllocator

- **定义位置**：`context/allocators/stream_ordered_allocator.hpp:8-18`
- **主要功能**：提供与 CUDA Stream 同步的异步内存分配，实现**流有序（Stream-Ordered）**内存管理
- **关键成员**：
    - `Device device_`：目标设备
- **核心方法**：
    - `std::byte *allocate(size_t size)`：调用 `infinirtMallocAsync(ptr, size, stream)`，从当前上下文获取 Stream
    - `void deallocate(std::byte *ptr)`：调用 `infinirtFreeAsync(ptr, stream)`，异步释放内存
- **设计思想**：
    - 异步分配不会阻塞 CPU，允许内存操作与计算重叠
    - 依赖 `context::getStream()` 获取当前活动流
    - 与 CUDA Graph 捕获兼容
- **使用场景**：深度学习推理引擎、需要高吞吐量的流水线式计算

## `infinicore::context::allocators` API

### 选择合适的分配器

| 分配器类型 | 适用场景 | 线程安全 | 性能 |
|-----------|---------|---------|------|
| **HostAllocator** | 纯 CPU 计算，小型缓冲区 | ✅ | ⭐⭐ |
| **DevicePinnedHostAllocator** | CPU-GPU 数据传输 | ❌ | ⭐⭐⭐ |
| **PinnableBlockAllocator** | 高频分配/释放，深度学习推理/训练 | ✅ | ⭐⭐⭐⭐⭐ |
| **StreamOrderedAllocator** | CUDA Graph、流式计算 | ✅ | ⭐⭐⭐⭐ |

### 基础接口（所有分配器通用）

```cpp
#include "memory_allocator.hpp"

// 分配内存
std::byte *ptr = allocator->allocate(1024); // 分配 1KB

// 释放内存
allocator->deallocate(ptr);
```

### PinnableBlockAllocator 扩展接口

```cpp
#include "pinnable_block_allocator.hpp"

// 创建分配器（指定设备）
PinnableBlockAllocator allocator(Device::GPU());

// 切换到固定模式（用于图捕获）
allocator.set_pin_mode(true);

// 修剪未使用的缓存块（归还内存给 GPU）
allocator.trim();
```

## 使用示例 (Usage Example)

### 示例 1：使用 HostAllocator 分配 CPU 内存

```cpp
#include "host_allocator.hpp"

namespace infinicore {

void cpu_computation_example() {
    HostAllocator allocator;

    // 分配 1MB 主机内存
    std::byte *buffer = allocator.allocate(1024 * 1024);

    // 使用缓冲区进行 CPU 计算
    // ...

    // 释放内存
    allocator.deallocate(buffer);
}

} // namespace infinicore
```

### 示例 2：使用 PinnableBlockAllocator 进行高性能推理

```cpp
#include "pinnable_block_allocator.hpp"
#include <vector>

namespace infinicore {

void inference_example() {
    Device gpu = Device::GPU(0); // 使用第 0 号 GPU
    PinnableBlockAllocator allocator(gpu);

    // 正常推理模式：允许缓存复用
    std::vector<std::byte *> tensors;
    for (int i = 0; i < 100; ++i) {
        // 分配 16MB 张量（会命中 size_class 的 16MB 等级）
        std::byte *tensor = allocator.allocate(16 * 1024 * 1024);
        tensors.push_back(tensor);
        // ... 推理计算 ...
        allocator.deallocate(tensor); // 回收到缓存池
    }

    // 切换到 CUDA Graph 捕获模式
    allocator.set_pin_mode(true);

    // 捕获图：所有新分配的块都被标记为 frozen
    std::byte *graph_tensor = allocator.allocate(4 * 1024 * 1024);
    // ... CUDA Graph 捕获逻辑 ...

    // 退出图模式，修剪缓存
    allocator.set_pin_mode(false);
    allocator.trim(); // 释放所有非冻结的缓存块
}

} // namespace infinicore
```

### 示例 3：使用 StreamOrderedAllocator 进行流式分配

```cpp
#include "stream_ordered_allocator.hpp"
#include "../context_impl.hpp" // for context::getStream()

namespace infinicore {

void async_allocation_example() {
    Device gpu = Device::GPU(0);
    StreamOrderedAllocator allocator(gpu);

    // 创建 CUDA Stream
    Stream stream = createStream();

    // 切换到自定义流
    context::setStream(stream);

    // 异步分配（不阻塞 CPU）
    std::byte *buffer = allocator.allocate(1024);

    // 提交异步内核（使用同一流）
    launch_kernel(buffer, 1024, stream);

    // 异步释放（在内核完成后自动释放）
    allocator.deallocate(buffer);
}

} // namespace infinicore
```

## 实现细节 (Implementation Details)

### 1. 内存对齐策略

- **GPU 对齐要求**：`PinnableBlockAllocator::allocate` 将所有请求大小对齐到 256 字节边界（行 46）
- **实现**：`align_up(size, 256) = (size + 255) / 256 * 256`
- **原因**：CUDA 设备内存访问通常需要 256 字节对齐以保证最佳性能

### 2. 双重释放检测

- **PinnableBlockAllocator** 的 `deallocate` 方法会检查 `block->in_use` 标志（行 112-114）
- 如果检测到双重释放，抛出 `std::runtime_error("Double free detected")`
- **HostAllocator** 不进行此检查（直接调用 `std::free`）

### 3. 跨设备内存释放的延迟策略

- **问题**：`DevicePinnedHostAllocator` 在设备 A 上下文中分配的固定内存，可能在设备 B 上下文中释放
- **风险**：立即释放可能导致设备 A 仍在使用该内存时被释放
- **解决方案**：
    - 如果当前设备与 `owner_` 相同，立即释放
    - 否则，推入 `gc_queue_` 延迟释放队列
    - 在下次分配或析构时统一清理队列

### 4. Size-Class 选择算法

- **PinnableBlockAllocator** 的大小等级是预定义的（行 23-35）
- **分配策略**：
    - 找到第一个满足 `size <= cls.block_size` 的等级
    - 可能造成内部碎片（例如请求 33KB 时分配 256KB 的块）
    - 但通过缓存池复用机制，实际碎片影响较小
- **灵感来源**：类似 jemalloc、tcmalloc 等通用分配器的设计

### 5. 线程安全保证

| 分配器 | 线程安全机制 |
|-------|------------|
| **HostAllocator** | 无状态，天然线程安全 |
| **DevicePinnedHostAllocator** | ❌ 非线程安全（代码注释明确标注 TODO） |
| **PinnableBlockAllocator** | ✅ 使用 `std::mutex` 保护所有公共方法 |
| **StreamOrderedAllocator** | 无状态（仅存储 device_），天然线程安全 |

### 6. 固定模式（Pinned Mode）的语义

- **目的**：支持 CUDA Graph 捕获，防止内存块在图重放时被重新分配
- **实现**：
    - 调用 `set_pin_mode(true)` 后，所有新分配的块 `frozen = true`
    - `trim()` 方法会跳过所有 `frozen` 块
    - 已分配的块不会改变 `frozen` 状态（仅影响新分配）
- **典型使用流程**：
    1. 正常推理/训练模式：`set_pin_mode(false)`（默认）
    2. 进入 CUDA Graph 捕获：`set_pin_mode(true)`
    3. 捕获完成后，退出图模式：`set_pin_mode(false)`
    4. 调用 `trim()` 释放缓存

### 7. 错误处理机制

- **统一宏**：`INFINICORE_CHECK_ERROR`（定义在 `utils.hpp`）封装所有 infinirt API 调用
- **失败行为**：如果 infinirt API 返回错误码，宏会抛出异常或终止程序
- **检查点**：
    - `infinirtMalloc` / `infinirtMallocHost` / `infinirtMallocAsync`
    - `infinirtFree` / `infinirtFreeHost` / `infinirtFreeAsync`

### 8. 内存泄漏防护

- **析构函数保障**：所有分配器的析构函数都会释放底层内存
    - `DevicePinnedHostAllocator::~DevicePinnedHostAllocator`：调用 `gc()` 清空队列
    - `PinnableBlockAllocator::~PinnableBlockAllocator`：遍历 `all_blocks_` 逐个释放
- **RAII 语义**：推荐使用智能指针或作用域对象管理分配器生命周期

## 性能特征 (Performance Characteristics)

### 时间复杂度分析

| 操作 | HostAllocator | DevicePinnedHostAllocator | PinnableBlockAllocator | StreamOrderedAllocator |
|-----|--------------|--------------------------|----------------------|----------------------|
| **allocate** | O(1) | O(1) | O(k) / O(n) | O(1) |
| **deallocate** | O(1) | O(1) / O(m) | O(1) | O(1) |
| **trim** | N/A | O(m) | O(n) | N/A |

**符号说明**：
- `k`：大小等级数量（固定为 11）
- `n`：`large_blocks_` + `size_classes_` 的总块数
- `m`：垃圾回收队列中的待释放指针数量

**PinnableBlockAllocator 分配细节**：
- **最佳情况**：请求大小匹配某个 size_class，且 free_blocks 非空 → O(k)
- **最坏情况**：请求超大块，需遍历 large_blocks_ → O(n)
- **平均情况**：由于 size_class 覆盖范围广（32KB~256MB），大部分请求命中 size_class → 接近 O(k)

### 空间复杂度

| 分配器 | 额外开销 |
|-------|---------|
| **HostAllocator** | 0 字节（无状态） |
| **DevicePinnedHostAllocator** | O(m)（gc_queue_ 存储待释放指针） |
| **PinnableBlockAllocator** | O(n)（all_blocks_ 索引 + Block 元数据） |
| **StreamOrderedAllocator** | sizeof(Device) 字节 |

### 推荐使用场景（性能导向）

1. **小规模 CPU 计算** → `HostAllocator`（零开销）
2. **CPU-GPU 数据传输** → `DevicePinnedHostAllocator`（DMA 加速）
3. **深度学习推理/训练** → `PinnableBlockAllocator`（最高吞吐量）
4. **CUDA Graph 应用** → `PinnableBlockAllocator` + `set_pin_mode(true)`
5. **流式管线** → `StreamOrderedAllocator`（异步计算）

## 依赖关系 (Dependencies)

### 内部依赖

```
allocators/
├── memory_allocator.hpp（基础接口）
│   ├── host_allocator.hpp/cc
│   ├── device_pinned_allocator.hpp/cc ──┐
│   ├── pinnable_block_allocator.hpp/cc   ├→ ../context_impl.hpp（设备/流管理）
│   └── stream_ordered_allocator.hpp/cc ──┘
└── ../../utils.hpp（INFINICORE_CHECK_ERROR 宏）
```

### 外部依赖

- **infinirt.h**：硬件抽象层 API
    - `infinirtMalloc` / `infinirtFree`
    - `infinirtMallocHost` / `infinirtFreeHost`
    - `infinirtMallocAsync` / `infinirtFreeAsync`
- **C++ 标准库**：
    - `<memory>`：std::byte, std::shared_ptr
    - `<mutex>`：std::mutex, std::lock_guard
    - `<queue>`：std::queue
    - `<unordered_map>`：std::unordered_map
    - `<vector>`：std::vector
    - `<algorithm>`：std::find_if

## 已知限制与未来方向 (Known Limitations & Future Directions)

### 当前限制

1. **DevicePinnedHostAllocator 非线程安全**：
   - 代码注释（行 23）明确标注 `TODO: this is not thread-safe`
   - `gc_queue_` 未加锁保护，多线程环境下可能导致数据竞争

2. **PinnableBlockAllocator 的固定模式语义不够直观**：
   - `set_pin_mode` 只影响新分配的块，不改变已存在块的状态
   - 可能导致用户误用：期望所有块都被冻结，但实际上只有新分配的块才受保护

3. **无内存池预分配机制**：
   - `PinnableBlockAllocator` 按需分配，首次使用时会有延迟
   - 未来可添加 `warmup(size_t size)` 方法预分配缓存池

### 可能的优化方向

1. **引入 jemalloc-inspired 分裂/合并机制**：
   - 允许大块分裂成小块，小块合并成大块
   - 减少内部碎片

2. **实现 Per-Thread 缓存**：
   - 为 `PinnableBlockAllocator` 添加线程局部缓存
   - 减少锁竞争，提升多线程性能

3. **支持内存池共享**：
   - 允许多个分配器共享同一个底层内存池
   - 减少跨分配器的内存碎片

4. **添加统计信息接口**：
   - 提供 `getStats()` 方法返回分配/释放次数、缓存命中率等
   - 辅助性能调优

---

**文档生成时间**：2026-01-14
**分析范围**：`InfiniCore/src/infinicore/context/allocators/` 所有源文件
**文档版本**：v1.0
