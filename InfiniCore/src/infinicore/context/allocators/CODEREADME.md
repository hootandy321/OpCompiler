# Allocators 模块核心实现文档

该模块实现了 InfiniCore 的内存分配子系统，提供多种内存分配策略以支持不同的设备内存管理需求。模块采用抽象基类设计模式，支持主机内存、设备锁定内存、流有序内存和可固定块内存的统一分配接口。

## 1. 模块结构

- **`memory_allocator.hpp`**: 定义抽象内存分配器接口，提供统一的分配/释放 API
- **`host_allocator.hpp/cc`**: 实现标准主机内存分配器，封装 `malloc/free`
- **`device_pinned_allocator.hpp/cc`**: 实现设备锁定主机内存分配器，支持跨设备延迟释放
- **`stream_ordered_allocator.hpp/cc`**: 实现流有序异步内存分配器，用于 CUDA Stream 上的异步内存操作
- **`pinnable_block_allocator.hpp/cc`**: 实现基于大小类的可固定块内存池分配器，支持内存缓存和固定模式

## 2. 核心类

### `MemoryAllocator`
- **位置**: `memory_allocator.hpp`
- **主要功能**: 定义所有内存分配器的抽象基类接口
- **关键成员**: 无（纯接口类）
- **核心方法**:
  - `virtual std::byte *allocate(size_t size) = 0`: 分配指定大小的内存块，返回字节指针
  - `virtual void deallocate(std::byte *ptr) = 0`: 释放先前分配的内存块
  - `virtual ~MemoryAllocator() = default`: 虚析构函数确保多态正确释放
- **生命周期**: 纯虚基类，不能直接实例化，生命周期由派生类管理
- **设计模式**: Strategy Pattern（策略模式），通过虚函数接口实现多态分配策略

### `HostAllocator`
- **位置**: `host_allocator.hpp/cc`
- **主要功能**: 封装标准 C 库的 `malloc/free`，提供普通主机内存分配
- **关键成员**: 无（无状态分配器）
- **核心方法**:
  - `std::byte *allocate(size_t size)`: 对 `std::malloc` 的封装，零大小请求返回 `nullptr`
  - `void deallocate(std::byte *ptr)`: 对 `std::free` 的封装，处理 `nullptr` 安全释放
- **实现细节**:
  - 使用 `std::malloc` 分配原始内存，返回类型转换为 `std::byte*`
  - 空指针保护：`size == 0` 返回 `nullptr`，`ptr == nullptr` 直接返回
  - 无内存对齐保证（依赖系统默认对齐）
- **生命周期**: 默认构造/析构，无资源管理需求
- **时间复杂度**: O(1) 分配和释放（系统调用复杂度除外）

### `DevicePinnedHostAllocator`
- **位置**: `device_pinned_allocator.hpp/cc`
- **主要功能**: 管理设备锁页主机内存（Pinned Memory），支持跨设备安全释放
- **关键成员**:
  - `Device owner_`: 分配器所属的设备句柄
  - `std::queue<std::byte *> gc_queue_`: 延迟释放队列（非线程安全）
- **核心方法**:
  - `explicit DevicePinnedHostAllocator(Device device)`: 构造函数，记录所属设备
  - `std::byte *allocate(size_t size)`: 通过 `infinirtMallocHost` 分配锁页内存
  - `void deallocate(std::byte *ptr)`: 智能释放策略——当前设备匹配时立即释放，否则加入 GC 队列
  - `void gc()`: 批量释放 GC 队列中的所有待释放内存块
  - `~DevicePinnedHostAllocator()`: 析构时自动调用 `gc()` 清理待释放队列
- **实现细节**:
  - **锁页内存**: 使用 `infinirtMallocHost` 分配，确保内存不会被换页，优化 DMA 传输
  - **跨设备安全**: 释放时检查 `owner_ == context::getDevice()`，避免跨设备访问错误
  - **延迟释放机制**: 非当前设备的内存块加入 `gc_queue_`，等待设备切换回原设备时批量释放
  - **错误检查**: 使用 `INFINICORE_CHECK_ERROR` 宏检查所有 infinirt 调用
  - **线程安全警告**: 代码注释明确标注 `gc_queue_` 不是线程安全的（TODO）
- **设计模式**: RAII（析构时自动清理）+ 延迟清理模式
- **时间复杂度**: allocate O(1), deallocate O(1)（平均情况，gc 时 O(n)）

### `StreamOrderedAllocator`
- **位置**: `stream_ordered_allocator.hpp/cc`
- **主要功能**: 在 CUDA Stream 上执行异步内存分配/释放操作
- **关键成员**:
  - `Device device_`: 关联的设备句柄
- **核心方法**:
  - `explicit StreamOrderedAllocator(Device device)`: 构造函数，保存设备句柄
  - `std::byte *allocate(size_t size)`: 使用 `infinirtMallocAsync` 在当前流上异步分配内存
  - `void deallocate(std::byte *ptr)`: 使用 `infinirtFreeAsync` 在当前流上异步释放内存
- **实现细节**:
  - **异步分配**: 所有操作都通过 `context::getStream()` 获取当前 CUDA Stream
  - **流依赖**: 内存生命周期与 Stream 关联，释放操作会在 Stream 中排队
  - **零检查**: `size == 0` 或 `ptr == nullptr` 时安全返回
  - **无同步保证**: 不等待操作完成，调用者负责同步
- **适用场景**: CUDA Graph 记录、流水线内存操作、重叠计算与内存管理
- **时间复杂度**: O(1) 分配和释放（异步入队）

### `PinnableBlockAllocator`
- **位置**: `pinnable_block_allocator.hpp/cc`
- **主要功能**: 实现基于大小类（Size-Class）的块内存池，支持内存缓存和固定模式切换
- **关键成员**:
  - `Device device_`: 目标设备
  - `bool pinned_mode_`: 固定模式标志，true 时分配的块被冻结不能释放
  - `std::vector<SizeClass> size_classes_`: 大小类数组，包含 11 个预定义大小类（32KB 到 256MB）
  - `std::vector<std::shared_ptr<Block>> large_blocks_`: 大块内存列表（超过最大大小类的请求）
  - `std::unordered_map<void *, std::shared_ptr<Block>> all_blocks_`: 所有已分配块的指针到块映射
  - `std::mutex mutex_`: 互斥锁保证线程安全

#### 内部数据结构
- **`struct Block`**: 单个内存块的元数据
  - `void *ptr`: 设备内存指针
  - `size_t size`: 块大小（字节）
  - `bool frozen`: 冻结标志（在固定/图模式下使用）
  - `bool in_use`: 使用状态标志

- **`struct SizeClass`**: 大小类描述
  - `size_t block_size`: 该类的固定块大小
  - `std::vector<std::shared_ptr<Block>> free_blocks`: 该类的空闲块缓存池

- **核心方法**:
  - `PinnableBlockAllocator(Device device)`: 初始化 11 个大小类（32KB、256KB、1MB、2MB、4MB、8MB、16MB、32MB、64MB、128MB、256MB）
  - `std::byte *allocate(size_t size)`: 智能分配逻辑
    1. **对齐**: 将请求大小向上对齐到 256 字节边界（GPU 内存对齐要求）
    2. **大小类匹配**: 遍历 `size_classes_`，找到第一个满足 `size <= cls.block_size` 的大小类
    3. **缓存复用**: 如果该大小类的 `free_blocks` 非空，弹出并复用
    4. **新块分配**: 如果缓存为空，调用 `infinirtMalloc` 分配新块，标记 `frozen = pinned_mode_`
    5. **大块处理**: 超过最大大小类的请求，在 `large_blocks_` 中查找可复用的非使用块，否则分配新块
  - `void deallocate(std::byte *ptr)`: 块回收逻辑
    1. **查找**: 在 `all_blocks_` 中查找指针，失败则抛出异常
    2. **双重释放检查**: 如果 `block->in_use == false`，抛出 `double free` 异常
    3. **状态更新**: 设置 `block->in_use = false`
    4. **大小类归还**: 如果块属于某个大小类，加入对应 `free_blocks` 缓存池
  - `void set_pin_mode(bool pinned)`: 切换固定模式，新分配的块将根据此模式设置 `frozen` 标志
  - `void trim()`: 释放非冻结的缓存块，回收 GPU 内存
    - 遍历所有大小类的 `free_blocks`，释放 `frozen == false` 的块
    - 遍历 `large_blocks_`，释放非使用且非冻结的块
    - 从 `all_blocks_` 中移除已释放的块
  - `~PinnableBlockAllocator()`: 析构函数
    - 遍历 `all_blocks_`，调用 `infinirtFree` 释放所有块
    - 清空所有容器

- **实现细节**:
  - **内存对齐**: 使用 `align_up(size, 256)` 确保所有分配 256 字节对齐
  - **线程安全**: 所有公共方法使用 `std::lock_guard<std::mutex>` 保护
  - **智能缓存**: 大小类策略减少碎片，空闲块缓存池减少分配/释放开销
  - **固定模式**: `pinned_mode_` 控制块的可释放性，用于 CUDA Graph 等需要内存固定的场景
  - **异常安全**: 无效指针和双重释放都会抛出 `std::runtime_error`
  - **共享所有权**: 使用 `std::shared_ptr<Block>` 管理块元数据，支持 `all_blocks_` 和缓存池的共享引用
- **设计模式**:
  - Object Pool Pattern（对象池模式）：缓存空闲块复用
  - Size-Class Allocation（大小类分配）：类似 jemalloc 的分配策略
  - RAII：析构时自动释放所有资源
- **空间复杂度**: O(n)，n 为已分配块数量
- **时间复杂度**:
  - allocate: O(1) 平均（大小类查找最多 11 次比较）
  - deallocate: O(1) 哈希查找 + O(1) 大小类匹配
  - trim: O(n)，n 为缓存块总数

## 3. API 接口

```cpp
// 抽象基类接口
namespace infinicore {
class MemoryAllocator {
public:
    virtual ~MemoryAllocator() = default;
    virtual std::byte *allocate(size_t size) = 0;
    virtual void deallocate(std::byte *ptr) = 0;
};

// 主机内存分配器
class HostAllocator : public MemoryAllocator {
public:
    HostAllocator() = default;
    std::byte *allocate(size_t size) override;  // 封装 std::malloc
    void deallocate(std::byte *ptr) override;   // 封装 std::free
};

// 设备锁页内存分配器
class DevicePinnedHostAllocator : public MemoryAllocator {
public:
    explicit DevicePinnedHostAllocator(Device device);
    ~DevicePinnedHostAllocator();

    std::byte *allocate(size_t size) override;  // infinirtMallocHost
    void deallocate(std::byte *ptr) override;   // 智能释放或入队
    void gc();                                  // 批量释放待释放队列
};

// 流有序异步分配器
class StreamOrderedAllocator : public MemoryAllocator {
public:
    explicit StreamOrderedAllocator(Device device);
    std::byte *allocate(size_t size) override;  // infinirtMallocAsync
    void deallocate(std::byte *ptr) override;   // infinirtFreeAsync
};

// 可固定块内存池分配器
class PinnableBlockAllocator : public MemoryAllocator {
public:
    explicit PinnableBlockAllocator(Device device);
    ~PinnableBlockAllocator();

    std::byte *allocate(size_t size) override;   // 大小类分配或大块分配
    void deallocate(std::byte *ptr) override;    // 归还到缓存池
    void set_pin_mode(bool pinned);              // 切换固定模式
    void trim();                                 // 释放非冻结缓存块
};
}
```

## 4. 使用示例

```cpp
#include "infinicore/context/allocators/host_allocator.hpp"
#include "infinicore/context/allocators/device_pinned_allocator.hpp"
#include "infinicore/context/allocators/stream_ordered_allocator.hpp"
#include "infinicore/context/allocators/pinnable_block_allocator.hpp"

using namespace infinicore;

// 示例 1: 使用主机内存分配器
void host_allocator_example() {
    HostAllocator allocator;
    std::byte *buffer = allocator.allocate(1024);  // 分配 1KB 主机内存
    // 使用 buffer...
    allocator.deallocate(buffer);  // 释放内存
}

// 示例 2: 使用设备锁页内存分配器
void pinned_allocator_example(Device device) {
    DevicePinnedHostAllocator allocator(device);
    std::byte *pinned_mem = allocator.allocate(4 * 1024 * 1024);  // 分配 4MB 锁页内存
    // DMA 传输优化：可安全用于 infinirtMemcpyH2D
    allocator.deallocate(pinned_mem);  // 智能释放
    allocator.gc();  // 手动触发垃圾回收（可选）
}

// 示例 3: 使用流有序分配器（异步内存操作）
void stream_ordered_example(Device device) {
    StreamOrderedAllocator allocator(device);
    std::byte *async_mem = allocator.allocate(1024);
    // 内存分配在 CUDA Stream 上异步执行
    // 调用者负责同步 Stream
    allocator.deallocate(async_mem);  // 异步释放
    // 注意：释放操作仅入队，需等待 Stream 完成
}

// 示例 4: 使用可固定块内存池分配器
void block_allocator_example(Device device) {
    PinnableBlockAllocator allocator(device);

    // 普通模式：分配和释放会缓存空闲块
    std::byte *block1 = allocator.allocate(512 * 1024);  // 512KB → 分配 1MB 大小类
    allocator.deallocate(block1);                         // 归还到 1MB 大小类缓存池

    // 固定模式：用于 CUDA Graph 记录
    allocator.set_pin_mode(true);  // 启用固定模式
    std::byte *frozen_block = allocator.allocate(2 * 1024 * 1024);  // 2MB，frozen = true
    allocator.deallocate(frozen_block);  // 不会真正释放，保留在缓存中

    // 回收非冻结内存
    allocator.trim();  // 释放 block1（非冻结），保留 frozen_block（冻结）

    // 析构时自动释放所有块
}

// 示例 5: 多态分配器使用
void polymorphic_allocator_example(MemoryAllocator *allocator, Device device) {
    std::byte *buffer = allocator->allocate(1024);
    // 统一接口，运行时选择具体分配策略
    allocator->deallocate(buffer);
}
```

## 5. 实现细节

### 内存管理策略
- **HostAllocator**: 直接封装系统 `malloc/free`，无缓存，适合小规模临时分配
- **DevicePinnedHostAllocator**: 锁页内存优化 DMA 传输，延迟释放机制避免跨设备访问冲突，GC 队列批量释放提高效率
- **StreamOrderedAllocator**: 基于 CUDA Stream 的异步分配，适合流水线操作和 CUDA Graph 记录
- **PinnableBlockAllocator**:
  - **大小类分配**: 预定义 11 个大小类（32KB~256MB，2 倍递增），减少外部碎片
  - **内存池缓存**: 空闲块缓存池避免频繁分配/释放，提升性能
  - **固定模式**: 支持内存冻结（frozen 标志），用于 CUDA Graph 等需要内存布局稳定的场景
  - **显式回收**: `trim()` 方法释放非冻结缓存块，允许用户控制内存占用

### 并发与线程安全
- **HostAllocator**: 无状态，天然线程安全（malloc/free 线程安全由 C 库保证）
- **DevicePinnedHostAllocator**: **非线程安全**，代码注释明确警告 `gc_queue_` 无锁保护
- **StreamOrderedAllocator**: 无状态，线程安全由 infinirt runtime 保证
- **PinnableBlockAllocator**: **完全线程安全**，使用 `std::mutex` 保护所有公共方法
  - `allocate()`: 锁定整个分配过程（查找、分配、插入映射）
  - `deallocate()`: 锁定整个释放过程（查找、状态更新、归还缓存池）
  - `trim()`: 锁定整个清理过程（遍历、释放、移除）
  - `~PinnableBlockAllocator()`: 锁定析构过程

### 性能优化
- **大小类快速查找**: 线性遍历 11 个大小类，O(1) 常数时间（实际最多 11 次比较）
- **内存对齐**: 256 字节对齐满足 GPU 访问要求，提升传输效率
- **缓存池复用**: 空闲块复用避免 infinirt 调用开销
- **批量释放**: `gc_queue_` 和 `trim()` 批量释放减少系统调用次数
- **异步操作**: `StreamOrderedAllocator` 允许内存操作与计算重叠

### 错误处理
- **空指针保护**: 所有分配器检查 `size == 0` 和 `ptr == nullptr`，避免无效操作
- **异常机制**:
  - `PinnableBlockAllocator::deallocate()` 在无效指针时抛出 `std::runtime_error("Pointer not allocated by this allocator")`
  - 双重释放检测：`block->in_use == false` 时抛出 `std::runtime_error("Double free detected")`
- **错误检查宏**: 使用 `INFINICORE_CHECK_ERROR` 包装 infinirt 调用，自动检查返回码

### 依赖关系
- **外部依赖**:
  - `<infinirt.h>`: InfiniRuntime 抽象层，提供跨平台的内存管理 API（`infinirtMalloc`, `infinirtMallocHost`, `infinirtMallocAsync` 等）
  - `<memory>`: C++ 标准库，提供 `std::byte` 和智能指针
  - `<mutex>`: C++ 标准库，提供 `std::mutex` 和 `std::lock_guard`
  - `<queue>`, `<vector>`, `<unordered_map>`: STL 容器
- **内部依赖**:
  - `infinicore/memory.hpp`: 提供 Device 类型定义
  - `context_impl.hpp`: 提供 `context::getDevice()` 和 `context::getStream()` 访问当前上下文
  - `utils.hpp`: 提供 `INFINICORE_CHECK_ERROR` 宏

### 设计模式
- **Strategy Pattern（策略模式）**: `MemoryAllocator` 抽象基类定义统一接口，派生类实现不同分配策略
- **Template Method Pattern（模板方法模式）**: 虚函数接口强制派生类实现 `allocate/deallocate`
- **Object Pool Pattern（对象池模式）**: `PinnableBlockAllocator` 缓存空闲块复用
- **RAII（Resource Acquisition Is Initialization）**: 析构函数自动释放资源（`DevicePinnedHostAllocator` 和 `PinnableBlockAllocator`）
- **Factory Pattern（工厂模式）**: 通过构造函数参数（Device）配置不同分配行为

### 关键算法
- **大小类选择算法**:
  ```cpp
  for (auto &cls : size_classes_) {
      if (size <= cls.block_size) {  // 第一个满足的大小类
          // 使用此大小类分配
      }
  }
  ```
  时间复杂度: O(1)，最多 11 次比较

- **内存对齐算法**:
  ```cpp
  inline size_t align_up(size_t size, size_t alignment) {
      return (size + alignment - 1) / alignment * alignment;
  }
  ```
  时间复杂度: O(1)

- **大块查找算法**:
  ```cpp
  auto it = std::find_if(large_blocks_.begin(), large_blocks_.end(),
                         [size](const std::shared_ptr<Block> &b) {
                             return b->size >= size && !b->in_use;
                         });
  ```
  时间复杂度: O(n)，n 为大块数量（通常较小）

- **垃圾回收算法**:
  ```cpp
  while (gc_queue_.empty() == false) {
      std::byte *p = gc_queue_.front();
      INFINICORE_CHECK_ERROR(infinirtFreeHost(p));
      gc_queue_.pop();
  }
  ```
  时间复杂度: O(m)，m 为队列长度

### 适用场景
- **HostAllocator**: 临时 CPU 缓冲区、小规模内存分配、不需要跨设备传输的场景
- **DevicePinnedHostAllocator**: 频繁 CPU-GPU 数据传输、DMA 优化、跨设备安全释放
- **StreamOrderedAllocator**: CUDA Graph 记录、流水线深度学习推理、异步内存操作
- **PinnableBlockAllocator**: 深度学习训练（张量内存池）、高频率小对象分配、需要内存固定和缓存的场景

### 限制与注意事项
1. **DevicePinnedHostAllocator 非线程安全**: 不能在多线程环境下并发调用
2. **PinnableBlockAllocator 的内存开销**: 缓存池会占用额外 GPU 内存，需定期调用 `trim()` 释放
3. **流同步责任**: `StreamOrderedAllocator` 的调用者必须手动同步 Stream，否则可能访问已释放内存
4. **大小类浪费**: 小于 32KB 的分配会被升级到 32KB，存在内部碎片
5. **固定模式内存泄漏**: `set_pin_mode(true)` 后未调用 `trim()` 会导致所有缓存块无法释放
6. **异常安全**: `PinnableBlockAllocator` 抛出异常后，部分块可能处于不一致状态（已分配但未加入 `all_blocks_`）
