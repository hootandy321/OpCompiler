# `PagedAttentionPrefill` 操作核心实现文档

该模块实现了基于 PagedAttention 机制的 Prefill 阶段注意力计算，用于高效处理大语言模型推理中的 KV-Cache 管理。通过支持分块键值缓存和动态块表映射，实现了对变长序列和批处理请求的高效内存管理。

## 1. 模块结构

- **`paged_attention_prefill.cc`**: 前端接口层，提供公共 API 和设备分发逻辑
- **`paged_attention_prefill_infiniop.cc`**: 后端实现层，基于 InfiniOP 库的具体计算实现，包含 descriptor 缓存机制

## 2. 核心类与组件

### `PagedAttentionPrefill`
- **位置**: `include/infinicore/ops/paged_attention_prefill.hpp`, `paged_attention_prefill.cc`
- **主要功能**: PagedAttention Prefill 操作的门面类，负责设备类型分发和执行调度
- **核心类型定义**:
  ```cpp
  using schema = void (*)(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, std::optional<Tensor>, float);
  ```
  该 schema 定义了操作签名的函数指针类型，包含 8 个输入参数和 1 个浮点缩放因子

- **核心方法**:
  - `execute(out, q, k_cache, v_cache, block_tables, kv_lens, cum_seqlens_q, alibi_slopes, scale)`: 执行注意力计算主入口，先进行设备一致性校验，设置当前设备，然后通过分发器查找对应设备的实现函数并执行
  - `dispatcher()`: 返回静态 `OpDispatcher` 单例，采用 Meyer's Singleton 模式，确保全局唯一的分发器实例

- **生命周期**: 静态分发器在首次调用 `dispatcher()` 时初始化，生命周期与程序相同

### `OpDispatcher<schema>`
- **位置**: `include/infinicore/ops/common/dispatcher.hpp`
- **主要功能**: 设备类型到实现函数的分发表，支持按设备类型注册和查找计算内核
- **数据结构**: `std::array<Fn, static_cast<size_t>(Device::Type::COUNT)> table_` - 固定大小数组，索引为设备类型枚举值
- **核心方法**:
  - `registerDevice(device_type, fn, override_existing)`: 注册单个设备的实现函数
  - `registerAll(fn, override_existing)`: 为所有设备类型注册同一实现函数
  - `lookup(device_type)`: O(1) 时间复杂度查表获取对应设备的函数指针

### `OpCache<Key, Value>`
- **位置**: `include/infinicore/ops/common/cache.hpp`
- **主要功能**: 基于设备分层的 LRU 缓存，存储 InfiniOP descriptor 对象以避免重复创建开销
- **数据结构**:
  - `std::array<CacheVector, static_cast<size_t>(Device::Type::COUNT)> caches_`: 二维数组，第一维为设备类型，第二维为同类型设备的多个实例
  - 每个缓存项为 `LRUCache<Key, Value>` 实例，容量默认 100

- **核心方法**:
  - `getCache(device_type, device_index)`: 获取指定设备的缓存实例，若不存在则自动扩容并创建
  - `clear()`: 遍历所有设备的缓存，在切换到对应设备上下文后清空，确保资源正确释放

- **生命周期**: 全局 thread_local 静态变量，每个线程拥有独立缓存实例，线程退出时自动析构

### `paged_attention_prefill_impl::infiniop::caches`
- **位置**: `paged_attention_prefill_infiniop.cc`
- **主要功能**: thread_local 全局缓存实例，存储 `infiniopPagedAttentionPrefillDescriptor_t` 对象
- **Key 类型**: `size_t` - 通过 hash_combine 函数计算所有输入参数的哈希值
- **Value 类型**: `infiniopPagedAttentionPrefillDescriptor_t` - InfiniOP descriptor 指针
- **析构函数**: 自定义 lambda 函数，调用 `infiniopDestroyPagedAttentionPrefillDescriptor` 释放 descriptor 资源
- **容量**: 100 个 descriptor 条目

## 3. API 接口

```cpp
// 输出张量分配并执行的便捷函数
Tensor paged_attention_prefill(
    Tensor q,                          // Query 张量，打包格式 (packed_qbs x head_dim)
    Tensor k_cache,                    // 物理 Key 缓存，分页格式
    Tensor v_cache,                    // 物理 Value 缓存，分页格式
    Tensor block_tables,               // 逻辑块到物理块的映射表 (num_requests x max_blocks_per_req)
    Tensor total_kv_lens,              // 每个请求的完整 KV 序列长度
    Tensor cum_seqlens_q,              // Query 序列长度的前缀和（用于变长批处理）
    std::optional<Tensor> alibi_slopes, // ALiBi 偏置斜率（可选参数）
    float scale                        // 缩放因子，通常为 1/sqrt(head_size)
);
// 返回: 输出张量，形状与 q 相同

// 预分配输出张量的就地执行版本
void paged_attention_prefill_(
    Tensor out,                        // 预分配的输出张量
    Tensor q,
    Tensor k_cache,
    Tensor v_cache,
    Tensor block_tables,
    Tensor total_kv_lens,
    Tensor cum_seqlens_q,
    std::optional<Tensor> alibi_slopes,
    float scale
);
// 无返回值，结果写入 out

// 静态执行方法（内部使用）
static void PagedAttentionPrefill::execute(
    Tensor out, Tensor q, Tensor k_cache, Tensor v_cache,
    Tensor block_tables, Tensor kv_lens, Tensor cum_seqlens_q,
    std::optional<Tensor> alibi_slopes, float scale
);
```

## 4. 使用示例

```cpp
// 示例: 执行 PagedAttention Prefill 计算
#include "infinicore/ops/paged_attention_prefill.hpp"

using namespace infinicore;

// 假设已有以下张量数据
Tensor q = Tensor::empty({total_qbs, head_dim}, DataType::FLOAT32, Device::cuda(0));
Tensor k_cache = Tensor::empty({num_physical_blocks, block_size, num_heads, head_per_head}, DataType::FLOAT32, Device::cuda(0));
Tensor v_cache = Tensor::empty({num_physical_blocks, block_size, num_heads, head_per_head}, DataType::FLOAT32, Device::cuda(0));
Tensor block_tables = Tensor::empty({num_requests, max_blocks_per_req}, DataType::INT32, Device::cuda(0));
Tensor kv_lens = Tensor::empty({num_requests}, DataType::INT32, Device::cuda(0));
Tensor cum_seqlens_q = Tensor::empty({batch_size + 1}, DataType::INT32, Device::cuda(0));

// ALiBi 偏置（可选）
std::optional<Tensor> alibi_slopes = std::nullopt;
// 如果使用 ALiBi: alibi_slopes = Tensor::empty({num_heads}, DataType::FLOAT32, Device::cuda(0));

// 缩放因子: 1/sqrt(head_dim)
float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

// 方式 1: 自动分配输出张量
Tensor out = op::paged_attention_prefill(q, k_cache, v_cache, block_tables, kv_lens, cum_seqlens_q, alibi_slopes, scale);
// out 现在包含计算结果

// 方式 2: 预分配输出张量（更高效）
Tensor out = Tensor::empty(q->shape(), q->dtype(), q->device());
op::paged_attention_prefill_(out, q, k_cache, v_cache, block_tables, kv_lens, cum_seqlens_q, alibi_slopes, scale);
// 结果已写入 out
```

## 5. 实现细节

### 内存管理
- **Descriptor 缓存策略**: 使用 thread_local LRU 缓存存储 InfiniOP descriptor，键为所有输入张量的形状、数据类型和缩放因子的哈希值，避免重复创建 descriptor 的开销
- **Workspace 动态分配**: 每次执行时调用 `infiniopGetPagedAttentionPrefillWorkspaceSize` 查询所需工作空间大小，通过 `context::allocateMemory` 临时分配，执行完毕后自动释放
- **设备分层缓存**: `OpCache` 为每种设备类型和每个设备实例维护独立的 LRU 缓存，支持多 GPU 并行场景

### 并发控制
- **Thread-Local 缓存**: `caches` 变量声明为 thread_local，每个线程拥有独立的缓存实例，避免多线程竞争
- **设备上下文切换**: `OpCache::clear()` 方法在遍历所有设备缓存时，会先切换到目标设备上下文再释放资源，确保 CUDA/HIP 等后端 API 在正确设备上执行

### 性能优化
- **哈希快速查找**: 利用 `hash_combine` 函数基于所有输入参数的元数据（形状、步长、数据类型）计算哈希键，实现 O(1) 时间复杂度的 descriptor 查找
- **设备类型分发**: `OpDispatcher` 使用固定大小数组存储函数指针，查找时间为 O(1)，优于 std::map 的 O(log n)
- **延迟初始化**: Meyer's Singleton 模式确保分发器在首次使用时才初始化，减少启动开销

### 错误处理
- **设备一致性检查**: `INFINICORE_ASSERT_TENSORS_SAME_DEVICE` 宏在执行前验证所有输入张量位于同一设备，若不一致抛出 std::runtime_error 并包含详细的设备信息
- **InfiniOP 错误传播**: `INFINICORE_CHECK_ERROR` 宏包装所有 InfiniOP API 调用，检查返回值并在失败时通过 `infini_status_string` 获取错误描述，抛出异常
- **异常安全**: Descriptor 的析构通过 RAII 机制确保异常发生时资源仍能正确释放

### 设计模式
- **门面模式 (Facade)**: `PagedAttentionPrefill` 类提供简洁的公共接口，隐藏底层设备分发和 InfiniOP 实现细节
- **策略模式 (Strategy)**: `OpDispatcher` 将不同设备的实现封装为可插拔的策略函数，支持运行时动态注册
- **单例模式 (Singleton)**: 分发器实例采用 Meyer's Singleton，线程安全且延迟初始化
- **RAII (资源获取即初始化)**: `OpCache` 的析构函数自动清理所有缓存 descriptor，无需手动管理生命周期

### 依赖关系
- **InfiniOP 库**: 核心计算后端，提供 `infiniopCreatePagedAttentionPrefillDescriptor`, `infiniopGetPagedAttentionPrefillWorkspaceSize`, `infiniopPagedAttentionPrefill` 等 API
- **InfiniRT 运行时**: 提供设备管理 (`infinirtStream_t`) 和内存分配接口
- **LRUCache 基础容器**: `infinicore/common/LRUCache.hpp` 提供 LRU 淘汰算法实现
- **Hash 工具**: `infinicore/common/hash.hpp` 提供类型安全的哈希组合函数，支持可变参数模板

### 算法复杂度
- **Descriptor 查找**: O(1) - 基于 hash 键的缓存查找
- **设备分发**: O(1) - 数组索引访问
- **哈希计算**: O(n) - n 为输入张量的维度数量，通常为常数（< 10）
- **空间复杂度**: 每个线程最多缓存 100 个 descriptor 对象，每个 descriptor 包含张量形状、设备句柄等元数据
