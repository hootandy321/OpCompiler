# `PagedAttention` 分页注意力算子核心实现文档

本模块实现了 PagedAttention 机制,这是 vLLM 等大模型推理框架中的核心技术。通过分块管理的 KV cache,实现高效的显存管理和可变长度序列的注意力计算,显著提升大模型推理的吞吐量和显存利用率。

## 1. 模块结构

- **`paged_attention.cc`**: PagedAttention 的对外接口层,提供用户友好的 API 和设备分发逻辑
- **`paged_attention_infiniop.cc`**: 基于 InfiniOp 后端的具体实现,包含算子缓存和内核执行逻辑

## 2. 核心类与数据结构

### `PagedAttention`
- **位置**: `paged_attention.cc`, `include/infinicore/ops/paged_attention.hpp`
- **主要职责**: 分页注意力算子的统一入口,负责设备类型分发和执行调度
- **核心成员**:
  - `dispatcher()`: 静态方法,返回全局唯一的 `OpDispatcher` 单例,用于管理不同设备的实现函数
- **核心方法**:
  - `execute(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, Tensor kv_lens, std::optional<Tensor> alibi_slopes, float scale)`: 执行分页注意力计算的主入口,验证设备一致性并根据设备类型分发到具体实现
  - `paged_attention(...)`: 函数式 API,自动分配输出张量并执行计算
  - `paged_attention_(...)`: 就地执行版本,使用用户提供的输出张量
- **生命周期**: 采用静态分发器模式,`dispatcher()` 在首次调用时初始化并全局复用

### `paged_attention_impl::infiniop` 命名空间实现
- **位置**: `paged_attention_infiniop.cc`
- **主要职责**: 封装 InfiniOp 后端的 PagedAttention 实现细节
- **核心组件**:
  - `thread_local OpCache<size_t, infiniopPagedAttentionDescriptor_t> caches`: 线程局部存储的算子描述符缓存,容量为 100,使用 LRU 淘汰策略
    - **缓存键**: 基于 `hash_combine` 生成的 `size_t` 哈希值,包含所有输入张量的形状、数据类型和标度参数
    - **缓存值**: `infiniopPagedAttentionDescriptor_t` 算子描述符指针
    - **析构函数**: 调用 `infiniopDestroyPagedAttentionDescriptor` 销毁描述符
  - `calculate(...)`: 具体实现函数,执行完整的 PagedAttention 计算流程
- **初始化**: 使用函数静态变量 `registered` 在模块加载时自动注册到 `PagedAttention::dispatcher()`,注册所有设备类型

## 3. API 接口

```cpp
// 核心执行接口 - 需要预分配输出张量
void PagedAttention::execute(
    Tensor out,                          // 输出张量 [num_tokens, num_heads, head_size]
    Tensor q,                            // 查询张量 [num_tokens, num_heads, head_size]
    Tensor k_cache,                      // 分页键缓存 [num_blocks, block_size, num_kv_heads, head_size]
    Tensor v_cache,                      // 分页值缓存 [num_blocks, block_size, num_kv_heads, head_size]
    Tensor block_tables,                 // 块表 [num_sequences, max_num_blocks_per_seq]
    Tensor kv_lens,                      // KV 长度 [num_sequences]
    std::optional<Tensor> alibi_slopes,  // 可选的 ALiBI 偏置斜率 [num_heads]
    float scale                          // 注意力缩放因子,通常为 1/sqrt(head_size)
);

// 函数式 API - 自动分配输出张量
Tensor paged_attention(
    Tensor q, Tensor k_cache, Tensor v_cache,
    Tensor block_tables, Tensor kv_lens,
    std::optional<Tensor> alibi_slopes, float scale
);

// 就地执行版本 - 用户管理输出张量内存
void paged_attention_(
    Tensor out, Tensor q, Tensor k_cache, Tensor v_cache,
    Tensor block_tables, Tensor kv_lens,
    std::optional<Tensor> alibi_slopes, float scale
);
```

## 4. 使用示例

```cpp
#include "infinicore/ops/paged_attention.hpp"

using namespace infinicore;

// 示例: 执行 PagedAttention 计算
void run_paged_attention() {
    // 1. 准备输入张量
    auto q = Tensor::empty({128, 32, 128}, DataType::FP16, Device::cuda(0));
    auto k_cache = Tensor::empty({1000, 16, 4, 128}, DataType::FP16, Device::cuda(0));
    auto v_cache = Tensor::empty({1000, 16, 4, 128}, DataType::FP16, Device::cuda(0));
    auto block_tables = Tensor::empty({10, 100}, DataType::INT32, Device::cuda(0));
    auto kv_lens = Tensor::empty({10}, DataType::INT32, Device::cuda(0));

    // 填充张量数据...
    // q->write(...);
    // k_cache->write(...);
    // ...

    // 2. 可选: 准备 ALiBI 偏置
    std::optional<Tensor> alibi_slopes = std::nullopt;
    // auto alibi_slopes = Tensor::empty({32}, DataType::FP32, Device::cuda(0));

    // 3. 设置缩放因子 (通常为 1/sqrt(head_size))
    float scale = 1.0f / std::sqrt(128.0f);

    // 4. 执行计算 (方式1: 自动分配输出)
    auto out = paged_attention(q, k_cache, v_cache, block_tables, kv_lens, alibi_slopes, scale);

    // 4.1 或执行计算 (方式2: 预分配输出)
    // auto out = Tensor::empty(q->shape(), q->dtype(), q->device());
    // paged_attention_(out, q, k_cache, v_cache, block_tables, kv_lens, alibi_slopes, scale);

    // 5. 使用输出结果
    // out->read(...);
}
```

## 5. 实现细节

### 5.1 算子分发机制 (OpDispatcher Pattern)
- **数据结构**: `std::array<Fn, static_cast<size_t>(Device::Type::COUNT)> table_` - 设备类型到函数指针的查找表
- **注册策略**:
  - `registerDevice(Device::Type, Fn, override)`: 注册单个设备的实现,默认覆盖已存在实现
  - `registerAll(Fn, override)`: 为所有设备类型注册同一实现函数(适用于通用后端如 InfiniOp)
- **查询复杂度**: O(1) 数组索引查找,无分支预测开销
- **线程安全**: 只读访问分发器表,写操作仅在初始化阶段

### 5.2 算子描述符缓存 (Descriptor Caching)
- **缓存策略**:
  - **LRU淘汰**: 容量固定为 100,超出时移除最久未使用的描述符
  - **线程隔离**: `thread_local` 确保每个线程独立缓存,避免锁竞争
  - **设备隔离**: `OpCache` 内部为每个 `(device_type, device_index)` 维护独立缓存实例
- **哈希键生成**:
  - 使用 `hash_combine` 递归组合所有参数的哈希值
  - 张量哈希包含: 数据类型、所有维度、所有步长
  - 可选参数哈希包含 `has_value()` 标志和值本身
  - 哈希算法: Boost 风格的 `seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2)`
- **生命周期管理**:
  - 缓存项析构时调用自定义 destructor 调用 `infiniopDestroyPagedAttentionDescriptor`
  - `OpCache::~OpCache()` 遍历所有设备并在对应设备上下文中清理缓存,避免跨设备内存访问错误

### 5.3 内存管理与工作空间分配
- **工作空间查询**: 每次执行前调用 `infiniopGetPagedAttentionWorkspaceSize` 获取所需工作空间大小
- **动态分配**: 使用 `context::allocateMemory(size)` 临时分配工作空间内存
- **生命周期**: 工作空间在函数返回前自动释放 (通过 `shared_ptr` 自定义删除器)
- **设备上下文**: 分配前确保当前设备设置为输出张量所在设备 (`context::setDevice(out->device())`)

### 5.4 错误处理与断言机制
- **设备一致性检查**: `INFINICORE_ASSERT_TENSORS_SAME_DEVICE` 宏验证所有输入张量在同一设备,抛出包含详细设备信息的异常
- **InfiniOp 错误处理**: `INFINICHECK_ERROR` 宏包装所有 InfiniOp API 调用,失败时将错误码转换为可读字符串并抛出异常
- **日志记录**: 使用 spdlog 库记录 API 进入/退出 (DEBUG 级别) 和错误信息

### 5.5 PagedAttention 算法特性
- **分页 KV Cache**:
  - K/V 缓存按固定大小的块 (block, 通常为 16) 组织
  - `block_tables` 映射每个序列的物理块索引,支持非连续存储
  - `kv_lens` 记录每个序列的实际有效长度,支持可变长度序列
- **内存效率**:
  - 避免传统连续 KV cache 的内存碎片问题
  - 支持动态分配和释放块,适应批处理中不同序列的长度差异
  - 显存利用率可达 95% 以上 (相比连续缓存的 60-80%)
- **计算流程**:
  1. 根据 `q` 的 token 数量和位置,从 `block_tables` 查找对应的 KV 块
  2. 从 `k_cache` 和 `v_cache` 中加载相关键值对
  3. 应用 `scale` 缩放因子计算注意力权重
  4. 可选应用 ALiBI 位置编码偏置 (`alibi_slopes`)
  5. 加权求和计算输出并写入 `out`

### 5.6 性能优化技术
- **内核复用**: 描述符缓存避免重复创建内核 (创建成本高昂)
- **零拷贝**: 输入张量直接传递给 InfiniOp,无中间缓冲
- **批量执行**: InfiniOp 后端内部优化批量序列的并行计算
- **设备本地执行**: 所有计算在 GPU 设备上完成,最小化 CPU-GPU 数据传输

### 5.7 多后端支持架构
- **后端抽象**: `OpDispatcher` 机制允许为不同设备类型注册不同实现
- **InfiniOp 后端**: 通过 `registerAll(&calculate, false)` 注册为通用后端,支持所有 InfiniOp 兼容设备
- **扩展性**: 可添加 CUDA、Ascend、CANN 等专用后端,通过 `registerDevice(Device::Type::CUDA, cuda_impl)` 注册
- **向后兼容**: `override_existing = false` 参数允许优先级控制,避免覆盖已注册的高性能实现

### 5.8 依赖关系
- **必需依赖**:
  - `infinicore/tensor.hpp`: 张量抽象层
  - `infinicore/context/context.hpp`: 设备管理和内存分配
  - `infinicore/ops/common/op.hpp`: 算子基类和分发器
  - `infinicore/common/hash.hpp`: 哈希组合函数
  - `infinicore/ops/common/cache.hpp`: 算子描述符缓存
  - `<infiniop.h>`: InfiniOp 外部库接口
- **编译依赖**:
  - C++17 标准库 (std::optional, std::shared_ptr)
  - spdlog 日志库
  - InfiniOp 动态链接库

### 5.9 设计模式总结
- **策略模式 (Strategy Pattern)**: `OpDispatcher` 根据设备类型选择不同实现策略
- **单例模式 (Singleton Pattern)**: 全局唯一的 `dispatcher()` 实例
- **缓存模式 (Cache Pattern)**: `OpCache` 封装 LRU 缓存逻辑,提升性能
- **RAII 惯用语**: 工作空间内存通过 `shared_ptr` 自动管理,描述符通过自定义 destructor 安全销毁
- **线程局部存储 (Thread-Local Storage)**: `thread_local` 缓存避免多线程竞争
