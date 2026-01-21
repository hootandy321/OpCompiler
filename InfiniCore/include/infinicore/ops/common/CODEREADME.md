# InfiniCore Ops Common 基础设施实现文档

本模块提供 InfiniCore 算子系统的核心基础设施，包括跨设备算子分发调度和多设备算子缓存管理。该模块是所有算子实现的基础设施层，负责设备类型路由和性能优化缓存。

## 1. 模块结构

- **`op.hpp`**: 算子基类的公共头文件入口（仅包含依赖声明）
- **`dispatcher.hpp`**: 算子分发器，实现基于设备类型的函数指针路由表
- **`cache.hpp`**: 算子缓存管理器，实现跨设备多实例的 LRU 缓存容器

## 2. 核心类

### `OpDispatcher<Fn>`
- **位置**: `dispatcher.hpp`
- **主要功能**: 提供零开销抽象的设备类型到函数指针的映射路由表，用于根据当前设备类型分发到对应的算子实现
- **关键成员**:
  - `table_`: `std::array<Fn, static_cast<size_t>(Device::Type::COUNT>` - 编期期确定大小的函数指针数组，每个设备类型对应一个槽位
- **核心方法**:
  - `registerDevice(Device::Type device_type, Fn fn, bool override_existing = true)`: 注册单个设备类型的实现函数。如果槽位已存在且 `override_existing=false`，则保留原有函数
  - `registerDevice(std::initializer_list<Device::Type> device_types, Fn fn, bool override_existing = true)`: 批量注册多个设备类型共享同一实现（适用于 CPU/HYGON 等同构后端）
  - `registerAll(Fn fn, bool override_existing = true)`: 为所有设备类型注册统一的回退实现
  - `lookup(Device::Type device_type) const`: O(1) 时间复杂度查询设备类型对应的函数指针
- **生命周期**: 值类型，通常作为算子类的静态成员变量存在，程序启动时初始化

### `OpCache<Key, Value>`
- **位置**: `cache.hpp`
- **主要功能**: 管理多设备多卡环境下的算子缓存实例，每个设备类型和设备索引组合维护独立的 LRU 缓存，用于缓存编译后的 kernel、算法计划或其他昂贵的算子资源
- **关键成员**:
  - `caches_`: `std::array<CacheVector, static_cast<size_t>(Device::Type::COUNT)>` - 二维数组结构，第一维是设备类型，第二维是该类型下的设备索引列表
  - `capacity_`: `size_t` - 单个缓存实例的容量上限
  - `destructor_`: `Destructor` (别名 `std::function<void(Value&)>`) - 缓存值销毁时的回调函数，用于释放 GPU 资源等
- **核心方法**:
  - `getCache(Device::Type device_type, size_t device_index)`: 获取指定设备的缓存实例引用。如果设备索引超出当前向量范围，自动扩容并初始化新的 LRU 缓存。每次访问都会更新该缓存的 destructor
  - `getCache(Device device)`: 便捷重载，从 Device 对象解包类型和索引
  - `setCapacity(size_t capacity)`: 动态调整所有缓存实例的容量。如果新容量小于当前大小，LRU 策略会自动驱逐多余条目
  - `clear()`: 清空所有缓存。关键实现细节：在清理每个设备的缓存前，会切换到该设备的上下文（`context::setDevice`），确保 GPU 资源释放时设备上下文正确。清理完成后恢复到当前设备。这避免了跨设备访问资源时的上下文错误
- **生命周期**: 由上层算子类持有所有权，析构时自动清理所有设备的缓存资源

## 3. API 接口

```cpp
// OpDispatcher 典型使用模式
namespace infinicore::op::common {

template <typename Fn>
class OpDispatcher {
public:
    // 注册单个设备的实现
    void registerDevice(Device::Type device_type, Fn fn, bool override_existing = true);

    // 批量注册多个设备共享实现
    void registerDevice(std::initializer_list<Device::Type> device_types, Fn fn, bool override_existing = true);

    // 为所有设备注册统一实现
    void registerAll(Fn fn, bool override_existing = true);

    // 查询设备对应的函数指针
    Fn lookup(Device::Type device_type) const;

private:
    std::array<Fn, static_cast<size_t>(Device::Type::COUNT)> table_;
};

} // namespace infinicore::op::common

// OpCache 典型使用模式
namespace infinicore::op::common {

template <typename Key, typename Value>
class OpCache {
public:
    // 构造函数，指定容量和值析构回调
    explicit OpCache(size_t capacity = 100, Destructor destructor = nullptr);

    // 获取指定设备的缓存实例（自动扩容）
    BaseCache &getCache(Device::Type device_type, size_t device_index);
    BaseCache &getCache(Device device);

    // 动态调整所有缓存容量
    void setCapacity(size_t capacity);

    // 清空所有设备缓存（带设备上下文切换）
    void clear();

private:
    size_t capacity_;
    Destructor destructor_;
    std::array<CacheVector, static_cast<size_t>(Device::Type::COUNT)> caches_;
};

} // namespace infinicore::op::common
```

## 4. 使用示例

```cpp
// 示例：实现一个跨设备的 MatMul 算子，使用 dispatcher 和 cache
#include "infinicore/ops/common/dispatcher.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/context/context.hpp"

namespace infinicore::op {

class MatMul {
public:
    using KernelKey = std::tuple<size_t, size_t, size_t>; // (M, N, K)
    using CompiledKernel = void*; // 假设是编译后的 kernel 句柄

    // 1. 定义设备特定实现
    static void nvidiaImpl(void *output, const void *a, const void *b, size_t m, size_t n, size_t k) {
        // CUDA 实现
        context::setDevice(Device::Type::NVIDIA, 0);
        // ... 调用 cuBLAS 或自定义 kernel
    }

    static void cpuImpl(void *output, const void *a, const void *b, size_t m, size_t n, size_t k) {
        // CPU fallback 实现
        // ... 朴素矩阵乘法或调用 BLAS
    }

    // 2. 初始化 dispatcher（静态初始化）
    MatMul() {
        dispatcher_.registerDevice({Device::Type::NVIDIA, Device::Type::KUNLUN}, nvidiaImpl);
        dispatcher_.registerDevice({Device::Type::CPU, Device::Type::HYGON}, cpuImpl);
        // 其他设备使用 cpuImpl 作为回退
        dispatcher_.registerAll(cpuImpl, false); // override_existing=false 保护已注册的实现

        // 3. 初始化 cache，使用自定义析构函数释放 GPU 资源
        cache_ = OpCache<KernelKey, CompiledKernel>(
            1000, // 每个设备缓存 1000 个 kernel
            [](CompiledKernel &kernel) {
                if (kernel) {
                    // 释放 GPU kernel 资源（如 cuModuleUnload）
                }
            }
        );
    }

    // 4. 算子主入口
    void compute(Tensor output, Tensor a, Tensor b) {
        Device device = context::getDevice();

        // 4.1 从 dispatcher 获取设备特定实现
        auto device_fn = dispatcher_.lookup(device.getType());

        // 4.2 构造缓存键（基于张量形状）
        KernelKey key = {a.shape()[0], b.shape()[1], a.shape()[1]};

        // 4.3 查找或编译 kernel
        auto &cache = cache_.getCache(device);
        auto compiled_kernel = cache.get(key);

        if (!compiled_kernel.has_value()) {
            // Cache miss: 编译 kernel
            compiled_kernel = compileKernel(key);
            cache.put(key, *compiled_kernel);
        }

        // 4.4 执行设备特定实现
        device_fn(output.data(), a.data(), b.data(),
                  std::get<0>(key), std::get<1>(key), std::get<2>(key));
    }

private:
    OpDispatcher<void(*)(void*, const void*, const void*, size_t, size_t, size_t)> dispatcher_;
    OpCache<KernelKey, CompiledKernel> cache_;

    std::optional<CompiledKernel> compileKernel(const KernelKey &key) {
        // 根据 (M, N, K) 编译最优 kernel
        // 返回编译后的句柄
        return nullptr;
    }
};

} // namespace infinicore::op
```

## 5. 实现细节

### 内存管理
- **OpCache 多层缓存架构**: 使用 `std::array<std::vector<LRUCache>>` 二维结构，外层维度是设备类型（编译期固定），内层是设备索引（运行期动态扩容）。这种设计支持多卡环境（如 8 张 NVIDIA GPU），每张卡独立缓存
- **LRU 底层实现**: 依赖 `infinicore::common::LRUCache`，使用 `std::list + std::unordered_map` 实现 O(1) 的 put/get 操作。列表头部是最近使用项，尾部是驱逐候选
- **资源清理策略**: `OpCache::clear()` 在清理每个设备的缓存前，必须调用 `context::setDevice()` 切换到该设备的上下文。这是因为 GPU 资源（如 CUDA kernel、texture object）的释放必须在创建它们的设备上下文中进行。清理完成后恢复原设备上下文，避免污染调用者的设备状态

### 并发
- **无锁设计**: `OpDispatcher` 是只读路由表，初始化完成后不再修改，天然线程安全
- **OpCache 线程安全**: 底层 `LRUCache` 本身不提供线程安全保证。如果多线程并发访问同一设备的缓存，上层调用者必须外部加锁（建议使用 `std::shared_mutex` 实现读写锁，因为缓存读多写少）
- **设备上下文切换**: `context::setDevice()` 调用通常不是线程安全的（底层 CUDA/MUSA 等运行时 API 的限制）。`OpCache::clear()` 假设在单线程环境或外部已同步的情况下调用

### 性能
- **零开销路由**: `OpDispatcher::lookup()` 是简单的数组索引访问，编译后等同于 `table_[device_type]`，无分支预测失败，完全可内联
- **缓存局部性**: `OpDispatcher` 的函数指针数组连续存储，访问时 L1 缓存命中率高
- **自动扩容开销**: `OpCache::getCache()` 在首次访问某个设备索引时会触发 `vector::resize()`，涉及 LRU 缓存对象的构造。建议在初始化阶段预热所有目标设备的缓存（如遍历 `context::getDeviceCount(type)`）
- **驱逐策略**: LRU 缓存驱逐时调用用户定义的 destructor，可能涉及 GPU kernel 释放或文件 I/O，这是非廉价操作。应合理设置 `capacity` 平衡内存占用和编译开销

### 错误处理
- **函数指针空指针**: `OpDispatcher::lookup()` 使用 `std::array::at()`，如果设备类型索引越界会抛出 `std::out_of_range`。这通常表示设备类型枚举与底层运行时不匹配，是编程错误
- **LRU destructor 异常**: `LRUCache` 在调用 destructor 时捕获 `std::exception`，打印错误信息到 `stderr`，但不会向上传播。这保证单个资源释放失败不会阻止整个缓存的清理
- **设备上下文切换失败**: `context::setDevice()` 可能失败（如设备不存在），`OpCache::clear()` 没有显式检查返回值。假设所有设备在调用 `clear()` 前都已正确初始化

### 依赖
- **外部模块依赖**:
  - `infinicore::Device`: 设备类型和索引表示（`device.hpp`）
  - `infinicore::context`: 设备上下文管理（`context::setDevice`, `context::getDevice`）
  - `infinicore::common::LRUCache`: LRU 缓存实现
- **第三方库依赖**:
  - 无直接依赖，仅使用标准库（`<array>`, `<functional>`, `<memory>`, `<vector>`）

### 设计模式
- **Strategy 模式**: `OpDispatcher` 是策略模式的实现，设备类型是上下文，函数指针是策略算法
- **Facade 模式**: `OpCache` 对复杂的二维缓存结构提供简化的访问接口，隐藏设备类型和索引的细节
- **RAII**: `OpCache` 的析构函数自动调用 `clear()`，确保缓存资源在对象生命周期结束时正确释放
- **Template Method**: `LRUCache` 的 destructor 是模板方法钩子，允许用户自定义资源清理逻辑
