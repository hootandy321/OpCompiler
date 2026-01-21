# `PagedCaching` 分页缓存操作核心实现文档

本模块实现了 PagedAttention 机制中的分页缓存操作，用于高效管理 Transformer 模型的 KV Cache。通过分页机制，支持动态内存分配和批处理推理，显著提升显存利用率。

## 1. 模块结构

- **`paged_caching.cc`**: 操作接口层，实现 `PagedCaching` 类的公共 API 和分发器管理
- **`paged_caching_infiniop.cc`**: InfiniOP 后端实现，包含描述符缓存机制和具体的 kernel 调度逻辑
- **`include/infinicore/ops/paged_caching.hpp`**: 公共接口定义，声明 `PagedCaching` 类和便捷函数

## 2. 核心类

### `PagedCaching`
- **位置**: `include/infinicore/ops/paged_caching.hpp`, `paged_caching.cc`
- **主要功能**: 提供 PagedCaching 操作的静态接口，管理设备特定实现的分发器
- **关键成员**:
  - `schema`: 类型别名，定义为函数指针 `void (*)(Tensor, Tensor, Tensor, Tensor, Tensor)`，表示操作签名
- **核心方法**:
  - `execute(Tensor k_cache, Tensor v_cache, Tensor k, Tensor v, Tensor slot_mapping)`:
    - 执行分页缓存操作的主入口
    - 验证所有输入张量位于同一设备（使用 `INFINICORE_ASSERT_TENSORS_SAME_DEVICE` 宏）
    - 设置当前设备上下文（`context::setDevice`）
    - 通过分发器根据设备类型查找并调用对应的实现函数
    - 时间复杂度: O(1) 的分发器查找 + O(n) 的实际 kernel 执行
  - `dispatcher()`:
    - 返回静态 `OpDispatcher<schema>` 单例引用
    - 分发器使用函数指针签名 `schema` 来注册和查找设备特定实现
    - 延迟初始化模式：首次调用时构造静态实例
- **生命周期**: 静态类，无需实例化。分发器在首次调用 `dispatcher()` 时初始化，进程生命周期内持久存在

### `OpCache` (模板类)
- **位置**: `include/infinicore/ops/common/cache.hpp`（依赖）
- **主要功能**: 为每个设备提供 LRU 缓存，存储和管理 InfiniOP 描述符对象
- **关键成员**:
  - `capacity_`: 缓存容量，默认 100 个条目
  - `destructor_`: 描述符销毁函数，用于清理资源（`infiniopDestroyPagedCachingDescriptor`）
  - `caches_`: 二维数组结构，第一维按 `Device::Type` 索引（CPU, CUDA 等），第二维按设备索引索引
- **核心方法**:
  - `getCache(Device device)`: 获取指定设备的 LRU 缓存实例，自动扩展缓存向量以容纳新设备
  - `getCache(Device::Type, size_t device_index)`: 按设备类型和索引获取缓存
  - `clear()`: 清空所有设备的缓存，在清除前切换到目标设备上下文以正确释放 GPU 资源
  - `setCapacity(size_t)`: 动态调整所有设备缓存的容量
- **设计模式**:
  - **线程本地存储**: `thread_local` 确保每个线程拥有独立的缓存实例
  - **RAII 资源管理**: 析构函数自动调用 `clear()` 释放所有描述符
  - **设备上下文切换**: 在清理跨设备资源时，自动切换设备上下文确保正确释放

## 3. API 接口

```cpp
// 主操作接口
class PagedCaching {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, Tensor);

    // 执行分页缓存操作
    // k_cache: [num_blocks, block_size, num_heads, head_dim] Key 缓存张量
    // v_cache: [num_blocks, block_size, num_heads, head_dim] Value 缓存张量
    // k: [batch_size, seq_len, num_heads, head_dim] 输入 Key 张量
    // v: [batch_size, seq_len, num_heads, head_dim] 输入 Value 张量
    // slot_mapping: [batch_size, seq_len] 槽位映射张量，指定每个 token 写入缓存的位置
    static void execute(Tensor k_cache, Tensor v_cache, Tensor k, Tensor v, Tensor slot_mapping);

    // 获取设备特定实现的分发器
    static common::OpDispatcher<schema> &dispatcher();
};

// 便捷函数包装器
void paged_caching_(Tensor k_cache, Tensor v_cache, Tensor k, Tensor v, Tensor slot_mapping);
```

## 4. 使用示例

```cpp
#include "infinicore/ops/paged_caching.hpp"

using namespace infinicore;

// 场景: 在批处理推理中更新 KV Cache
// 假设我们有一个批处理请求，每个请求有不同的序列长度

// 初始化张量
Tensor k_cache;   // [1024, 16, 32, 128] - 1024 个块，每块 16 个槽位，32 个注意力头，128 维
Tensor v_cache;   // [1024, 16, 32, 128] - 与 k_cache 结构相同
Tensor k;         // [4, 128, 32, 128] - 批大小 4，序列长度 128
Tensor v;         // [4, 128, 32, 128] - 与 k 结构相同
Tensor slot_mapping; // [4, 128] - 每个元素是 [0, 1024*16) 范围内的整数

// 执行分页缓存操作
// 这将根据 slot_mapping 将 k 和 v 的内容写入对应的缓存槽位
op::PagedCaching::execute(k_cache, v_cache, k, v, slot_mapping);

// 或者使用便捷函数
op::paged_caching_(k_cache, v_cache, k, v, slot_mapping);

// 内部执行流程:
// 1. 验证所有张量位于同一设备（如 CUDA:0）
// 2. 根据张量形状和数据类型生成哈希键
// 3. 查找或创建 InfiniOP 描述符（缓存命中可避免重复创建）
// 4. 分配工作空间内存（如需要）
// 5. 调用底层 kernel 执行实际的张量拷贝和缓存更新
```

## 5. 实现细节

### 内存管理
- **描述符缓存策略**: 使用 `OpCache<size_t, infiniopPagedCachingDescriptor_t>` 实现 LRU 缓存
  - 缓存键：基于所有输入张量的数据类型、形状和步长的组合哈希（使用 `hash_combine` 函数）
  - 缓存值：InfiniOP 描述符指针，封装了张量布局信息和 kernel 参数
  - 容量限制：每个设备 100 个描述符，超出时自动淘汰最久未使用的条目
  - 线程本地：每个线程维护独立缓存，避免多线程竞争

- **工作空间内存管理**:
  - 每次操作调用 `infiniopGetPagedCachingWorkspaceSize` 动态查询所需工作空间大小
  - 使用 `context::allocateMemory` 分配设备内存（可能是 pinned memory 或 GPU 内存）
  - 工作空间在操作完成后自动释放（通过 `shared_ptr` 的 RAII 机制）

### 并发性
- **线程安全**:
  - `thread_local` 缓存确保每个线程有独立的描述符缓存，无需加锁
  - 设备上下文切换（`context::setDevice`）在多线程环境下需要外部同步
  - 实际的 kernel 执行（`infiniopPagedCaching`）可能是异步的，依赖 CUDA stream 的同步机制

- **资源清理策略**:
  - 在 `OpCache::clear()` 中，遍历所有设备的缓存时进行设备上下文切换
  - 使用当前设备的保存和恢复机制，避免影响调用者的设备上下文
  - 每个描述符在淘汰或缓存清空时调用 `infiniopDestroyPagedCachingDescriptor` 释放内部资源

### 性能优化
- **描述符复用**: 通过哈希缓存避免重复创建昂贵的 InfiniOP 描述符
  - 哈希函数：Boost 风格的 `hash_combine` 算法，使用黄金比例常数 `0x9e3779b9` 减少哈希冲突
  - 缓存命中时跳过 `infiniopCreatePagedCachingDescriptor` 调用，节省 CPU 开销

- **设备特定调度**:
  - 使用 `OpDispatcher` 实现运行时多态，根据设备类型（CPU, CUDA, ROCm 等）查找对应的实现函数
  - 分发器在模块加载时通过静态初始化注册所有可用后端（`registerAll(&calculate, false)`）
  - 查找复杂度：O(1) 数组索引（假设设备类型枚举作为索引）

- **零拷贝优化**: 操作直接在设备内存上进行，避免主机和设备之间的数据传输

### 错误处理
- **错误传播**:
  - 使用 `INFINICORE_CHECK_ERROR` 宏包装所有 InfiniOP API 调用
  - 错误时抛出 `std::runtime_error` 异常，包含失败的 API 调用名称和错误描述
  - 断言宏 `INFINICORE_ASSERT_TENSORS_SAME_DEVICE` 在张量设备不匹配时抛出详细错误信息

- **资源泄漏防护**:
  - 描述符使用 RAII 包装，确保异常安全
  - 工作空间内存使用 `shared_ptr` 管理，异常时自动释放

### 依赖关系
- **外部依赖**:
  - `infiniop.h`: InfiniOP 后端库，提供 `infiniopCreatePagedCachingDescriptor`, `infiniopPagedCaching` 等函数
  - `spdlog`: 日志库，用于调试和错误追踪

- **内部依赖**:
  - `infinicore/context/context.hpp`: 设备上下文管理（`getDevice`, `setDevice`, `getInfiniopHandle`, `getStream`）
  - `infinicore/ops/common/cache.hpp`: 操作描述符缓存实现
  - `infinicore/common/hash.hpp`: 张量哈希组合函数
  - `infinicore/ops/common/dispatcher.hpp`: 设备特定实现分发器（推测存在）

### 设计模式
- **策略模式 (Strategy Pattern)**:
  - `OpDispatcher` 根据设备类型动态选择不同的算法实现（CPU vs GPU）
  - 每个设备后端是独立的策略，实现相同的函数签名

- **单例模式 (Singleton Pattern)**:
  - `PagedCaching::dispatcher()` 返回静态单例，确保全局唯一的分发器实例

- **工厂模式 (Factory Pattern)**:
  - `OpCache::getCache` 根据设备类型和索引创建或返回对应的 LRU 缓存实例

- **RAII (Resource Acquisition Is Initialization)**:
  - `OpCache` 的析构函数自动清理所有缓存的描述符
  - 工作空间内存使用智能指针管理

### 算法细节
- **哈希算法**: 使用递归的 `hash_combine` 函数生成复合哈希值
  - 基础算法：`seed ^= hash(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2)`
  - 对于张量：依次哈希数据类型、每个维度大小、每个步长值
  - 最终生成的哈希值作为 LRU 缓存的键，唯一标识一组张量配置

- **PagedCaching 语义**:
  - 根据 `slot_mapping` 张量的指示，将输入的 `k` 和 `v` 张量分散写入到 `k_cache` 和 `v_cache` 的对应槽位
  - 支持不连续写入（即一个序列的 token 可以写入到缓存的不同物理块）
  - 典型应用场景：Transformer 模型的自回归推理，动态分配和释放 KV Cache
