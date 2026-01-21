# `MemoryPool` Core Implementation Documentation

MemoryPool 是 InfiniLM 框架的核心内存管理组件，实现了基于可合并空闲块的内存池分配器，专门为高性能深度学习推理场景优化。它通过 InfiniRT 运行时抽象层提供跨设备（CUDA/CPU/国产芯片）的统一内存管理接口，采用首次适应算法（First-Fit）配合双向空闲块合并策略，有效降低内存碎片并提升分配效率。

## 1. Module Structure

- **`allocator.hpp`**: 定义 `AllocatorBase` 抽象基类和 `MemoryPool` 具体实现类的完整接口，包含 `Block` 内部数据结构和管理成员变量
- **`memory_allocator.cpp`**: 实现 `MemoryPool` 的核心逻辑，包括内存分配、释放、区域扩展和块合并算法

## 2. Core Classes

### `AllocatorBase`
- **Location**: `allocator.hpp:10-15`
- **Primary Function**: 定义内存分配器的抽象接口，为不同分配策略提供统一的多态基类
- **Core Methods**:
  - `virtual void *alloc(size_t size) = 0`: 分配指定字节数的内存，返回指向内存的指针
  - `virtual void release(void *ptr) = 0`: 释放之前分配的内存
  - `virtual ~AllocatorBase() = default`: 虚析构函数确保派生类正确销毁

### `MemoryPool`
- **Location**: `allocator.hpp:17-52`, `memory_allocator.cpp:4-136`
- **Primary Function**: 高性能内存池实现，通过预分配大块内存区域并在内部进行二次管理，避免频繁调用底层运行时分配函数，同时支持内存块自动合并以减少碎片
- **Key Members**:
  - `size_t _alignment`: 内存对齐要求，必须是 2 的幂，默认 256 字节
  - `std::vector<void *> _base_regions`: 存储所有从 InfiniRT 申请的原始内存区域指针，析构时统一释放
  - `std::set<Block> _all_blocks`: 按地址排序的所有内存块集合（已分配和空闲），用于快速查找相邻块进行合并
  - `std::multimap<size_t, std::set<Block>::iterator> _free_blocks`: 从大小到空闲块迭代器的映射，支持 O(log n) 的最佳/首次适应查找
  - `std::unordered_map<void *, std::set<Block>::iterator> _ptr_to_block`: 从用户指针到块迭代器的哈希表，实现 O(1) 的释放查找
- **Internal Data Structure**:
  - `struct Block`: 内存块描述符
    - `void *base`: 所属原始内存区域的基地址
    - `void *ptr`: 当前块的可用地址（已对齐）
    - `size_t size`: 块大小（字节）
    - `bool is_free`: 是否空闲标记
    - `bool operator<(const Block &other) const`: 按指针地址排序，保证在 `_all_blocks` 中的有序性
- **Core Methods**:
  - `MemoryPool(size_t initialSize, size_t alignment)`: 构造函数，验证对齐参数为 2 的幂，可选预分配初始区域
  - `~MemoryPool()`: 析构函数，遍历 `_base_regions` 调用 `infinirtFree` 释放所有底层内存
  - `void *alloc(size_t size)`: 分配接口，算法复杂度 O(log n)
    1. 对齐请求大小到 `_alignment` 边界（向上取整）
    2. 在 `_free_blocks` 中使用 `lower_bound` 查找首个满足大小的空闲块（首次适应策略）
    3. 若未找到，调用 `allocateNewRegion` 申请新区域
    4. 从 `_free_blocks` 和 `_all_blocks` 中移除选中的块
    5. 创建已分配块并插入 `_all_blocks` 和 `_ptr_to_block`
    6. 若剩余空间 ≥ `_alignment`，分割为新的空闲块
    7. 返回对齐后的指针
  - `void release(void *ptr)`: 释放接口，算法复杂度 O(log n)
    1. 在 `_ptr_to_block` 中查找指针对应的块
    2. 从 `_all_blocks` 中移除并标记为空闲
    3. 重新插入到 `_all_blocks` 和 `_free_blocks`
    4. 调用 `tryCoalesce` 尝试与相邻空闲块合并
  - `void *allocateNewRegion(size_t size)`: 私有方法，通过 `RUN_INFINI(infinirtMalloc(&ptr, size))` 申请底层内存，创建初始空闲块并加入管理结构
  - `void tryCoalesce(const Block &block)`: 私有方法，双向合并算法
    1. 在 `_all_blocks` 中定位当前块
    2. 检查物理地址相邻的后继块，若空闲则合并（更新大小、移除后继块）
    3. 检查物理地址相邻的前驱块，若空闲则合并（更新指针、大小和基地址、移除前驱块）
    4. 将合并后的大块重新插入 `_all_blocks` 和 `_free_blocks`
- **Lifecycle**:
  1. 构造时验证对齐参数，可选预分配初始区域
  2. 运行时通过 `alloc`/`release` 动态管理内存
  3. 析构时释放所有底层区域，无内存泄漏

## 3. API Interface

```cpp
// 抽象分配器接口
class AllocatorBase {
    virtual void *alloc(size_t size) = 0;
    // 分配 size 字节的内存，返回对齐指针，失败抛出 std::bad_alloc

    virtual void release(void *ptr) = 0;
    // 释放之前分配的内存，无效指针抛出 std::runtime_error
};

// 具体内存池实现
class MemoryPool : public AllocatorBase {
    static constexpr size_t DEFAULT_ALIGNMENT = 256;

    explicit MemoryPool(size_t initialSize = 0, size_t alignment = DEFAULT_ALIGNMENT);
    // initialSize: 预分配字节数（0 表示延迟分配）
    // alignment: 对齐要求，必须是 2 的幂

    void *alloc(size_t size) override;
    // 首次适应算法 + 自动分割，复杂度 O(log n)

    void release(void *ptr) override;
    // 自动合并相邻空闲块，复杂度 O(log n)

    size_t getAlignment() const;
    // 获取当前对齐设置
};
```

## 4. Usage Example

```cpp
#include "allocator.hpp"

// 创建默认对齐（256字节）的内存池
MemoryPool pool(1024 * 1024);  // 预分配 1MB

// 分配内存
void *ptr1 = pool.alloc(512);      // 分配 512 字节
void *ptr2 = pool.alloc(4096);     // 分配 4KB
void *ptr3 = pool.alloc(8192);     // 分配 8KB

// 使用内存（需转换为实际类型）
float *tensor_data = static_cast<float *>(ptr1);
tensor_data[0] = 1.0f;

// 释放内存（自动合并相邻空闲块）
pool.release(ptr1);
pool.release(ptr2);

// 再次分配会重用释放的内存
void *ptr4 = pool.alloc(2048);

// 析构时自动释放所有底层内存
// 在栈上：自动调用
// 在堆上：delete pool_ptr;
```

## 5. Implementation Details

- **Memory Management**:
  - 采用区域池（Region Pool）模式，通过 `infinirtMalloc` 从设备运行时申请大块原始内存
  - 内部使用二次管理：将区域分割为块，通过红黑树（`std::set`）和多重映射（`std::multimap`）维护块状态
  - 支持任意 2 的幂对齐，默认 256 字节适配 SIMD/SIMT 指令
  - 对齐计算公式：`(size + alignment - 1) & ~(alignment - 1)`（位优化版本）

- **Concurrency**:
  - **非线程安全**：当前实现未使用锁或原子操作，需外部同步（如每线程独立池或互斥锁保护）
  - `_all_blocks` 的迭代器在 `_free_blocks` 和 `_ptr_to_block` 中缓存，需保证生命周期一致性

- **Performance**:
  - **分配复杂度**：O(log n)，其中 n 为空闲块数量（`std::multimap::lower_bound`）
  - **释放复杂度**：O(log n)，哈希查找 + 合并操作
  - **合并策略**：双向合并（前驱 + 后继），消除外部碎片，最大化连续内存可用性
  - **首次适应算法**：偏向使用小空闲块，保留大块用于后续大分配
  - **空间开销**：每个块约 32 字节元数据（3 个指针 + size_t + bool），加上市图结构开销

- **Error Handling**:
  - 无效对齐参数：构造函数抛出 `std::invalid_argument`
  - 分配失败：`alloc` 抛出 `std::bad_alloc`（底层 InfiniRT 失败或无法分配新区域）
  - 无效指针释放：`release` 抛出 `std::runtime_error`
  - 底层 API 错误：`RUN_INFINI` 宏检测到非 `INFINI_STATUS_SUCCESS` 时打印错误信息并 `exit(EXIT_FAILURE)`

- **Dependencies**:
  - **外部依赖**：
    - `infinirt.h`: InfiniRT 运行时抽象层（跨设备内存管理）
    - `infinicore_infer.h`: InfiniCore 推理框架类型定义
  - **标准库**：
    - `<map>`: `std::multimap` 用于大小到空闲块映射
    - `<set>`: `std::set` 用于地址有序的所有块集合
    - `<unordered_map>`: `std::unordered_map` 用于指针到块的快速查找
    - `<vector>`: `std::vector` 用于管理原始区域

- **Design Patterns**:
  - **Strategy Pattern**: `AllocatorBase` 定义抽象策略，`MemoryPool` 为具体策略
  - **RAII (Resource Acquisition Is Initialization)**: 构造函数分配资源，析构函数自动释放
  - **Buddy System Variant**: 简化版的伙伴系统，支持任意大小块合并（非固定 2 的幂分割）
  - **Object Pool Pattern**: 预分配大块区域，减少频繁调用底层分配器

- **Algorithm Details**:
  - **对齐验证**：`alignment & (alignment - 1) == 0`（位运算检查 2 的幂）
  - **首次适应查找**：`_free_blocks.lower_bound(aligned_size)` 找到首个 ≥ 请求大小的块
  - **块合并条件**：物理地址连续（`prev->ptr + prev->size == curr->ptr`）且均为空闲
  - **最小分割阈值**：剩余空间 ≥ `_alignment` 时才创建新空闲块，避免微小碎片
