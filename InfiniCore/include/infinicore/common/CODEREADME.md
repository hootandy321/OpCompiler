# `infinicore::common` 通用工具模块实现文档

本模块提供 InfiniCore 框架的基础通用工具类，包括哈希组合工具和 LRU 缓存实现。这些工具为框架的高性能计算和内存管理提供核心基础设施。

## 1. 模块结构

- **`hash.hpp`**: 提供通用的哈希值组合工具，支持多种数据类型的组合哈希计算，用于生成复合键的哈希值
- **`LRUCache.hpp`**: 实现最近最少使用（Least Recently Used）缓存策略，支持自定义容量管理和资源清理回调

## 2. 核心组件

### `hash_combine` 函数族
- **位置**: `hash.hpp`
- **主要功能**: 提供类型安全的多参数哈希组合算法，生成统一的 `size_t` 哈希值
- **设计模式**: 函数重载 + 可变参数模板（Variadic Template）

#### 核心算法
```
seed ^= hash(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2)
```

使用 **Boost 哈希组合算法**，其中 `0x9e3779b9` 是黄金比例 φ 的 32 位近似值的无符号整数表示。该算法通过异或、加法、位移操作实现良好的哈希值混合，避免哈希碰撞。

#### 类型特化

1. **算术类型基础模板** (第 10-15 行)
   - 约束: `std::enable_if_t<std::is_arithmetic_v<T>>`
   - 支持所有基础算术类型（int, float, double, size_t 等）
   - 直接使用 `std::hash<T>` 进行哈希计算

2. **Tensor 特化** (第 17-26 行)
   - 哈希组合三个维度：
     - 数据类型 (`dtype`)
     - 形状向量 (`shape`) 的每个维度
     - 步长向量 (`strides`) 的每个步长值
   - 适用于作为缓存键或唯一标识符

3. **std::optional 特化** (第 28-35 行)
   - 先哈希是否有值 (`has_value()`)
   - 如果有值，继续哈希内部值
   - 保证 `optional<T>` 与 `T` 的哈希值不会冲突

4. **std::string 特化** (第 37-40 行)
   - 委托给 `std::hash<std::string>`

5. **const char* 特化** (第 42-45 行)
   - 转换为 `std::string` 后哈希
   - 支持 C 风格字符串

6. **可变参数递归模板** (第 47-57 行)
   - 递归展开参数包
   - 每个参数依次调用 `hash_combine(seed, value)`
   - 基础情况（无参数）为空操作

#### 便捷接口
```cpp
template <typename... Types>
size_t hash_combine(const Types &...values);
```
- 返回组合哈希值，无需预先提供 seed
- 初始 seed 为 0

### `LRUCache<Key, Value>` 类模板
- **位置**: `LRUCache.hpp`
- **主要功能**: 高性能的 LRU 缓存实现，使用哈希表 + 双向链表实现 O(1) 访问和更新
- **数据结构**: `std::unordered_map<Key, ListIt>` + `std::list<KeyValuePair>`

#### 核心成员变量
```cpp
std::list<KeyValuePair> list_;           // front = most recent, back = least recent
std::unordered_map<Key, ListIt> map_;   // O(1) key -> iterator 映射
size_t capacity_;                        // 容量限制
Destructor destructor_;                  // std::function<void(Value&)> 清理回调
```

**设计关键**:
- `list_.front()` 存储最近访问的项
- `list_.back()` 存储最久未访问的项（LRU 淘汰候选）
- `map_` 存储指向 `list_` 节点的迭代器，避免 O(n) 查找
- 使用 `std::list::splice()` 实现节点移动，**不涉及节点拷贝或内存分配**

#### 核心方法

1. **构造函数** (第 19-24 行)
   ```cpp
   explicit LRUCache(size_t capacity = 100, Destructor destructor = nullptr)
   ```
   - 默认容量 100
   - 如果 `capacity == 0`，设置为 `UINT64_MAX`（无界缓存）
   - 接受可选的资源析构回调函数

2. **`put(key, value)`** (第 34-50 行)
   - **时间复杂度**: O(1)
   - **逻辑**:
     - 如果 key 已存在：调用析构器（如果有）、更新值、移动到 front
     - 如果 key 不存在：检查容量、必要时淘汰 LRU 项、插入到 front、更新 map
   - **淘汰策略**: 当 `list_.size() >= capacity_` 时调用 `evictLRU()`

3. **`get(key)`** (第 52-68 行)
   - **重载版本**: const 和 non-const
   - **时间复杂度**: O(1)
   - **non-const 版本**: 调用 `touch()` 将访问项移动到 front（更新访问顺序）
   - **const 版本**: 仅返回值，不修改访问顺序
   - **返回值**: `std::optional<Value>`，未命中时返回 `std::nullopt`

4. **`contains(key)`** (第 30-32 行)
   - O(1) 键存在性检查
   - 不影响访问顺序

5. **`setCapacity(capacity)`** (第 74-79 行)
   - 动态调整容量
   - 如果新容量小于当前大小，循环淘汰直至满足容量约束
   - **线程安全性**: 非线程安全，需要外部同步

6. **`setDestructor(destructor)`** (第 70-72 行)
   - 运行时设置资源清理回调
   - 允许动态修改析构策略

7. **`clear()`** (第 81-89 行)
   - 如果设置析构器，遍历所有项调用 `safeDestruct()`
   - 清空 list 和 map

8. **`getAllItems()`** (第 91-93 行)
   - 返回 list 的常量引用
   - **迭代顺序**: 从最近访问到最久未访问

#### 内部辅助方法

1. **`touch(it)`** (第 99-103 行)
   ```cpp
   void touch(typename std::unordered_map<Key, ListIt>::iterator it) {
       list_.splice(list_.begin(), list_, it->second);
       it->second = list_.begin();
   }
   ```
   - **核心优化**: 使用 `splice()` 将节点移动到 front，**时间复杂度 O(1)**
   - 不涉及元素拷贝，仅调整指针
   - 更新 map 中的迭代器

2. **`safeDestruct(value)`** (第 105-117 行)
   - try-catch 包装析构器调用
   - 捕获 `std::exception`，输出错误信息到 `std::cerr`
   - **容错设计**: 析构失败不会中断缓存操作

3. **`evictLRU()`** (第 119-126 行)
   - 移除 `list_.back()`（最久未使用项）
   - 调用 `safeDestruct()` 清理资源
   - 从 map 中删除 key
   - 从 list 中移除节点

4. **`cleanup()`** (第 128-130 行)
   - 析构函数调用
   - 委托给 `clear()`

#### 析构行为
- 析构时自动调用 `cleanup()` 清空所有缓存项
- 如果设置析构器，会逐个调用清理回调
- **异常安全**: 析构器抛出异常不会导致析构失败

## 3. API 接口

### hash.hpp
```cpp
// 基础类型哈希组合（修改 seed 引用）
template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, void>
hash_combine(size_t &seed, const T &value);

// Tensor 特化
void hash_combine(size_t &seed, Tensor tensor);

// optional 特化
template <typename T>
void hash_combine(size_t &seed, const std::optional<T> &opt);

// 字符串特化
void hash_combine(size_t &seed, const std::string &str);
void hash_combine(size_t &seed, const char *str);

// 可变参数版本（递归展开）
template <typename First, typename... Rest>
void hash_combine(size_t &seed, const First &first, const Rest &...rest);

// 便捷接口：直接返回哈希值
template <typename... Types>
size_t hash_combine(const Types &...values);
```

### LRUCache.hpp
```cpp
template <typename Key, typename Value>
class LRUCache {
public:
    using Destructor = std::function<void(Value &)>;

    // 构造与析构
    explicit LRUCache(size_t capacity = 100, Destructor destructor = nullptr);
    ~LRUCache();

    // 核心操作
    void put(const Key &key, const Value &value);
    std::optional<Value> get(const Key &key);
    bool contains(const Key &key) const;

    // 配置管理
    void setDestructor(Destructor destructor);
    void setCapacity(size_t capacity);
    void clear();

    // 调试与监控
    const std::list<KeyValuePair> &getAllItems() const;
};
```

## 4. 使用示例

### hash.hpp 使用示例
```cpp
#include "infinicore/common/hash.hpp"
#include <string>
#include <optional>

using namespace infinicore;

// 示例 1: 基础类型组合哈希
void basic_hash_combine() {
    size_t seed = 0;
    hash_combine(seed, 42);           // 整数
    hash_combine(seed, 3.14f);        // 浮点数
    hash_combine(seed, "hello");      // C 字符串
    std::cout << "Combined hash: " << seed << std::endl;
}

// 示例 2: 使用便捷接口
void convenient_hash() {
    size_t h = hash_combine(100, 200, 300);
    // 等价于:
    // size_t seed = 0;
    // hash_combine(seed, 100, 200, 300);
}

// 示例 3: Tensor 哈希（用于缓存键）
size_t compute_tensor_key(Tensor tensor) {
    return hash_combine(
        tensor->dtype(),
        tensor->shape()[0],
        tensor->shape()[1]
    );
}

// 示例 4: 复合键哈希
struct CacheKey {
    int op_type;
    std::string device;
    std::optional<size_t> workspace_size;

    size_t hash() const {
        return hash_combine(op_type, device, workspace_size);
    }
};
```

### LRUCache.hpp 使用示例
```cpp
#include "infinicore/common/LRUCache.hpp"
#include <iostream>
#include <string>

using namespace infinicore::common;

// 示例 1: 基础 LRU 缓存
void basic_lru_cache() {
    LRUCache<std::string, int> cache(3);  // 容量为 3

    cache.put("a", 1);
    cache.put("b", 2);
    cache.put("c", 3);

    auto val = cache.get("a");  // 返回 std::optional<int>{1}
    std::cout << "Value of 'a': " << val.value() << std::endl;

    cache.put("d", 4);  // 淘汰 "b"（最久未使用）

    if (!cache.contains("b")) {
        std::cout << "'b' was evicted" << std::endl;
    }
}

// 示例 2: 带资源管理的缓存（模拟 GPU 内存）
struct GPUBuffer {
    void *ptr;
    size_t size;

    ~GPUBuffer() {
        if (ptr) {
            // cudaFree(ptr);  // 释放 GPU 内存
        }
    }
};

void gpu_buffer_cache() {
    // 设置析构器：在淘汰时释放 GPU 内存
    LRUCache<std::string, GPUBuffer> gpu_cache(
        10,  // 容量 10
        [](GPUBuffer &buf) {
            std::cout << "Freeing GPU buffer: " << buf.size << " bytes" << std::endl;
            // cudaFree(buf.ptr);
            buf.ptr = nullptr;
        }
    );

    gpu_cache.put("tensor1", GPUBuffer{/* allocated ptr */, 1024});
    gpu_cache.put("tensor2", GPUBuffer{/* allocated ptr */, 2048});

    // 当缓存满时或手动 clear() 时，会自动调用析构器
}

// 示例 3: 动态调整容量
void dynamic_capacity() {
    LRUCache<int, std::string> cache(100);

    // 运行时调整容量（会触发淘汰）
    cache.setCapacity(50);  // 如果当前有 80 项，会淘汰 30 项
}

// 示例 4: 无界缓存
void unbounded_cache() {
    LRUCache<int, double> cache(0);  // capacity=0 表示无界
    // cache.capacity_ 被设置为 UINT64_MAX
    // 不会自动淘汰，需手动管理
}
```

### 综合示例：Tensor 运算结果缓存
```cpp
#include "infinicore/common/hash.hpp"
#include "infinicore/common/LRUCache.hpp"
#include "infinicore/tensor.hpp"
#include <vector>

using namespace infinicore;
using namespace infinicore::common;

class TensorOpCache {
public:
    using Key = size_t;  // 哈希值作为键

    TensorOpCache(size_t capacity = 100)
        : cache_(capacity,
                 [](Tensor &t) {
                     std::cout << "Evicting tensor from cache" << std::endl;
                     // Tensor 的智能指针会自动管理引用计数
                 }) {}

    // 生成缓存键：操作类型 + 输入张量的形状和类型
    Key compute_key(std::string_view op_name,
                    const std::vector<Tensor> &inputs) {
        size_t seed = 0;
        hash_combine(seed, op_name);
        for (const auto &input : inputs) {
            hash_combine(seed, input);  // 使用 Tensor 特化
        }
        return seed;
    }

    // 缓存查找
    std::optional<Tensor> lookup(std::string_view op_name,
                                 const std::vector<Tensor> &inputs) {
        Key key = compute_key(op_name, inputs);
        return cache_.get(key);
    }

    // 缓存插入
    void insert(std::string_view op_name,
                const std::vector<Tensor> &inputs,
                Tensor result) {
        Key key = compute_key(op_name, inputs);
        cache_.put(key, result);
    }

private:
    LRUCache<Key, Tensor> cache_;
};

// 使用示例
void tensor_cache_demo() {
    TensorOpCache cache(50);

    // 假设有两个输入张量
    Tensor input1 = /* ... */;
    Tensor input2 = /* ... */;

    // 尝试从缓存获取
    auto cached = cache.lookup("matmul", {input1, input2});

    if (cached) {
        std::cout << "Cache hit!" << std::endl;
        // 使用 *cached
    } else {
        std::cout << "Cache miss, computing..." << std::endl;
        Tensor result = /* 执行 matmul */;
        cache.insert("matmul", {input1, input2}, result);
    }
}
```

## 5. 实现细节

### hash.hpp

- **哈希算法**: Boost 哈希组合（基于黄金比例常数）
  - 常数 `0x9e3779b9`: 2^32 × φ ≈ 2654435769
  - 位操作组合：`seed << 6`（乘 64）和 `seed >> 2`（除 4）
  - 异或操作确保信息混合均匀

- **类型安全**: 使用 SFINAE (`std::enable_if_t`) 约束算术类型特化
- **零成本抽象**: 所有函数内联，编译器优化后无额外开销
- **模板递归展开**: 可变参数版本在编译期展开为顺序调用
- **Tensor 特化策略**: 仅哈希元数据（dtype, shape, strides），不哈希实际数据
  - 理由：数据可能很大，且相同形状的张量应被视为相同键

### LRUCache.hpp

- **数据结构设计**: 哈希表 + 双向链表的经典 LRU 实现
  - `std::unordered_map`: O(1) 查找
  - `std::list`: O(1) 插入、删除、移动（使用 splice）
  - **空间复杂度**: O(n)，每个元素在 map 和 list 中各存储一次

- **关键优化**:
  - `std::list::splice()`: 在容器内移动节点，**不涉及内存分配或元素拷贝**
  - map 存储 list 迭代器，避免线性查找
  - const 和 non-const get 分离：const 版本不修改访问顺序

- **异常安全**:
  - 析构器调用使用 try-catch 包装
  - 析构失败输出错误但不中断程序
  - **RAII**: 析构函数自动清理所有资源

- **容量管理**:
  - 0 容量映射为 `UINT64_MAX`（无界）
  - `setCapacity()` 会触发循环淘汰直至满足约束
  - 容量检查在 `put()` 时执行（非严格，允许短暂超出）

- **性能保证**:
  - `put()`: O(1) 平均
  - `get()`: O(1) 平均
  - `contains()`: O(1) 平均
  - `evictLRU()`: O(1)
  - `clear()`: O(n)

- **设计权衡**:
  - **非线程安全**: 多线程访问需要外部加锁（考虑使用 `std::shared_mutex`）
  - **引用语义**: Value 使用引用传递，避免大对象拷贝
  - **迭代器失效**: `list_` 是 public 成员，外部修改可能导致 map 迭代器失效（设计为 protected）

- **内存管理**:
  - 使用 `std::list` 管理节点生命周期
  - 析构器模式允许外部资源（GPU 内存、文件句柄）的定制清理
  - `std::function` 析构器有轻微性能开销，可考虑模板化优化

- **使用场景**:
  - 计算结果缓存（张量运算、图编译）
  - GPU 内存池管理
  - 操作符融合决策缓存
  - JIT 编译缓存
