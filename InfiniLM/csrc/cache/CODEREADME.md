# `InfiniLM Cache` Core Implementation Documentation

本模块实现了 Transformer 模型推理和训练中的 Key-Value (KV) 缓存机制,提供静态和分页两种内存管理策略,支持分布式张量并行训练,优化大语言模型的推理性能和显存利用率。

## 1. Module Structure

- **`base_cache.hpp`**: 定义缓存抽象基类 `Cache` 和配置基类 `CacheConfig`,提供多态接口
- **`cache.hpp`**: 模块统一头文件,包含所有缓存实现的导出
- **`kv_cache.hpp`**: 实现静态缓存 `StaticKVCache` 和分页缓存 `PagedKVCache` 及其配置类
- **`kv_cache.cpp`**: 包含静态缓存的完整实现和分页缓存的框架代码

## 2. Core Classes

### `Cache`
- **Location**: `base_cache.hpp`
- **Primary Function**: 抽象基类,定义所有缓存实现的公共接口
- **Key Members**:
  - 无成员变量,仅作为接口定义
- **Core Methods**:
  - `Cache() = default`: 默认构造函数
  - `virtual ~Cache()`: 虚析构函数,支持多态删除
- **Lifecycle**: 接口基类,无独立生命周期,由派生类实现具体功能

### `CacheConfig`
- **Location**: `base_cache.hpp`
- **Primary Function**: 抽象配置基类,定义配置对象的克隆接口
- **Key Members**:
  - 无成员变量
- **Core Methods**:
  - `virtual std::unique_ptr<CacheConfig> unique_copy() const = 0`: 纯虚函数,要求派生类实现深拷贝
- **Lifecycle**: 接口基类,支持配置对象的类型安全复制

### `StaticKVCacheConfig`
- **Location**: `kv_cache.hpp`, `kv_cache.cpp`
- **Primary Function**: 静态 KV 缓存配置类,管理批大小和缓存长度约束
- **Key Members**:
  - `max_batch_size_`: `infinicore::Size` - 最大批次大小
  - `max_cache_len_`: `infinicore::Size` - 最大缓存序列长度
- **Core Methods**:
  - `StaticKVCacheConfig(infinicore::Size _max_batch_size, infinicore::Size _max_cache_len)`: 构造函数,默认值为 batch=1, cache_len=numeric_limits<Size>::max()
  - `unique_copy() const`: 返回 `std::make_unique<StaticKVCacheConfig>(*this)`,实现深拷贝
  - `max_batch_size() const`: 返回最大批次大小
  - `max_cache_len() const`: 返回最大缓存长度
- **Lifecycle**: 值对象,通过 unique_copy 进行复制

### `StaticKVCache`
- **Location**: `kv_cache.hpp`, `kv_cache.cpp`
- **Primary Function**: 静态 KV 缓存实现,为每个批次预分配固定大小的连续显存空间,适合推理场景
- **Key Members**:
  - `k_dim_`, `v_dim_`: `infinicore::Size` - Key 和 Value 的特征维度
  - `num_rank_k_heads_`, `num_rank_v_heads_`: `infinicore::Size` - 张量并行切分后的注意力头数量 (原头数 / tp_size)
  - `rank_batch_size_`: `infinicore::Size` - 当前 rank 的批次大小
  - `cache_len_`: `infinicore::Size` - 缓存序列长度 (取配置值和 max_positional_embedding 的较小值)
  - `rank_num_layers_`: `infinicore::Size` - 当前 rank 负责的层数
  - `dtype_`: `infinicore::DataType` - 缓存数据类型
  - `k_caches_`: `infinicore::Tensor` - 形状为 `[num_layers, max_batch, num_rank_k_heads, max_cache_len, k_dim]` 的 Key 缓存张量
  - `v_caches_`: `infinicore::Tensor` - 形状为 `[num_layers, max_batch, num_rank_v_heads, max_cache_len, v_dim]` 的 Value 缓存张量
- **Core Methods**:
  - `StaticKVCache(...)`: 构造函数,根据模型配置和分布式信息分配张量并行切分后的缓存空间
    - 计算 `num_rank_k_heads_ = num_k_heads / rank_info.tp_size`
    - 计算 `num_rank_v_heads_ = num_v_heads / rank_info.tp_size`
    - 确定缓存长度: 若配置为 max 或 0,则使用 `max_positional_embedding`
    - 使用 `Tensor::empty()` 在指定设备上分配连续显存
  - `update(size_t layer_idx, const Tensor &k, const Tensor &v, const Tensor &cache_lengths)`: 更新指定层的 KV 缓存
    - **输入**:
      - `k`: `[batch, num_rank_k_heads, seq_len, k_dim]` - 新的 Key 张量
      - `v`: `[batch, num_rank_v_heads, seq_len, v_dim]` - 新的 Value 张量
      - `cache_lengths`: 包含当前缓存位置标量的 CPU 张量
    - **算法**:
      1. 从 `cache_lengths` 提取 `cache_pos` (当前写入位置)
      2. 通过 `narrow()` 提取目标层的缓存张量: `k_cache_layer`, `v_cache_layer`
      3. 再次 `narrow()` 定位到 `cache_pos` 位置的更新窗口: `k_cache_update`, `v_cache_update`
      4. 调用 `copy_from()` 将新 KV 数据拷贝到缓存窗口
      5. 返回完整缓存视图 `narrow({2, 0, result_len})` (从 0 到当前位置+新增长度)
    - **输出**: `(full_k, full_v)` - 形状为 `[batch, num_rank_heads, cache_pos + seq_len, dim]` 的完整历史 KV
    - **复杂度**: O(batch_size * seq_len * heads * dim) 内存拷贝
    - **断言检查**:
      - `layer_idx < rank_num_layers_`
      - `result_len <= cache_len_` (防止越界)
      - `batch_size == rank_batch_size_`
- **Memory Management**:
  - 使用 `Tensor::empty()` 预分配固定大小的连续显存
  - 张量形状遵循 5D 布局: `[layers, batch, heads, seq_len, dim]`
  - 显存占用计算: `num_layers * batch_size * (num_heads * cache_len * (k_dim + v_dim) * dtype_size)`
- **Thread Safety**: 无锁,外部调用者需保证线程安全
- **Design Pattern**: RAII (构造时分配,析构时自动释放)

### `PagedKVCacheConfig`
- **Location**: `kv_cache.hpp`, `kv_cache.cpp`
- **Primary Function**: 分页 KV 缓存配置,基于显存预算和块大小管理动态内存分配
- **Key Members**:
  - `max_kv_memory_bytes_`: `size_t` - KV 缓存最大显存预算 (字节)
  - `block_size_`: `size_t` - 每个块的 token 数量,默认 16
- **Core Methods**:
  - `PagedKVCacheConfig(size_t max_kv_memory_bytes, size_t block_size = 16)`: 构造函数
  - `unique_copy() const`: 返回 `std::make_unique<PagedKVCacheConfig>(*this)`
  - `max_kv_memory_bytes() const`: 返回显存预算
  - `block_size() const`: 返回块大小
- **Lifecycle**: 值对象,通过 unique_copy 进行复制

### `PagedKVCache`
- **Location**: `kv_cache.hpp`, `kv_cache.cpp`
- **Primary Function**: 分页 KV 缓存实现,基于块 (block) 的动态显存管理,支持高并发推理场景的灵活显存分配
- **Key Members**:
  - `k_dim_`, `v_dim_`: `infinicore::Size` - KV 特征维度
  - `num_rank_k_heads_`, `num_rank_v_heads_`: `infinicore::Size` - 张量并行切分后的头数
  - `rank_num_layers_`: `infinicore::Size` - 当前 rank 的层数
  - `dtype_`: `infinicore::DataType` - 数据类型
  - `block_size_`: `infinicore::Size` - 每块 token 数量
  - `num_blocks_per_layer_`: `infinicore::Size` - 每层的块数量 (根据显存预算计算)
  - `k_caches_`: `infinicore::Tensor` - 形状为 `[num_layers, num_blocks, num_rank_k_heads, block_size, k_dim]` 的分页 Key 缓存
  - `v_caches_`: `infinicore::Tensor` - 形状为 `[num_layers, num_blocks, num_rank_v_heads, block_size, v_dim]` 的分页 Value 缓存
- **Core Methods**:
  - `PagedKVCache(...)`: 构造函数,根据显存预算动态计算块数量并分配分页缓存
    - **块数量计算公式**:
      ```
      num_blocks_per_layer = max_kv_memory_bytes
                            / (k_dim * num_rank_k_heads + v_dim * num_rank_v_heads)
                            / block_size
                            / dtype_size
      ```
    - 若计算结果为 0,抛出 `std::runtime_error("Not enough memory for KV cache")`
    - 使用 `Tensor::empty()` 分配 5D 分页张量
  - `update(size_t layer_idx, const Tensor &k, const Tensor &v, const Tensor &slot_mapping)`: 更新分页缓存 (框架实现,未完成)
    - **输入**:
      - `k`: `[num_rank_k_heads, seq_len, k_dim]` - 新 Key 张量
      - `v`: `[num_rank_v_heads, seq_len, v_dim]` - 新 Value 张量
      - `slot_mapping`: `[seq_len]` - 每个 token 对应的物理槽位映射
    - **当前实现**: 仅返回当前层的缓存张量视图,包含 `/// @todo: implement paged cache update here` 注释
    - **预期行为** (未实现):
      1. 根据 `slot_mapping` 将分散的 KV tokens 写入对应的物理块槽位
      2. 支持非连续写入,实现 PagedAttention 机制
    - **输出**: `(k_cache_layer, v_cache_layer)` - 当前层的完整分页缓存视图
- **Memory Management**:
  - 预分配块池 (block pool),支持动态分配和释放
  - 显存占用: `num_layers * num_blocks * block_size * num_heads * (k_dim + v_dim) * dtype_size`
  - 相比静态缓存,通过 slot mapping 支持显存的碎片化和复用
- **Performance**:
  - 优势: 支持动态批处理,提高显存利用率 (vLLM PagedAttention 策略)
  - 劣势: 需要额外的 slot mapping 管理,更新路径较慢
- **Implementation Status**: 框架代码,核心逻辑待实现 (todo 注释标记)

## 3. API Interface

```cpp
namespace infinilm::cache {

// ============ Cache Abstraction ============
class Cache {
public:
    Cache() = default;
    virtual ~Cache() {}
};

class CacheConfig {
public:
    CacheConfig() = default;
    virtual ~CacheConfig() {}
    virtual std::unique_ptr<CacheConfig> unique_copy() const = 0;
};

// ============ Static KV Cache API ============
class StaticKVCacheConfig final : public CacheConfig {
public:
    // 构造静态缓存配置
    StaticKVCacheConfig(
        infinicore::Size max_batch_size = 1,
        infinicore::Size max_cache_len = std::numeric_limits<infinicore::Size>::max());

    infinicore::Size max_batch_size() const;  // 获取最大批次
    infinicore::Size max_cache_len() const;    // 获取最大缓存长度
};

class StaticKVCache final : public Cache {
public:
    // 构造静态 KV 缓存
    StaticKVCache(
        infinicore::Size k_dim,                // Key 特征维度
        infinicore::Size v_dim,                // Value 特征维度
        infinicore::Size num_k_heads,          // Key 注意力头数
        infinicore::Size num_v_heads,          // Value 注意力头数
        infinicore::Size num_layers,           // Transformer 层数
        infinicore::Size max_positional_embedding,  // 最大位置编码长度
        infinicore::DataType dtype,            // 数据类型 (fp16/bf16/fp32)
        const StaticKVCacheConfig &config,     // 缓存配置
        const engine::distributed::RankInfo &rank_info);  // 分布式信息

    // 更新 KV 缓存并返回完整历史
    // 输入: k [batch, num_rank_k_heads, seq_len, k_dim]
    //       v [batch, num_rank_v_heads, seq_len, v_dim]
    //       cache_lengths [1] (标量,当前缓存位置)
    // 输出: (full_k, full_v)
    //       full_k: [batch, num_rank_k_heads, cache_pos + seq_len, k_dim]
    //       full_v: [batch, num_rank_v_heads, cache_pos + seq_len, v_dim]
    std::tuple<infinicore::Tensor, infinicore::Tensor>
    update(size_t layer_idx,
           const infinicore::Tensor &k,
           const infinicore::Tensor &v,
           const infinicore::Tensor &cache_lengths);
};

// ============ Paged KV Cache API ============
class PagedKVCacheConfig final : public CacheConfig {
public:
    // 构造分页缓存配置
    PagedKVCacheConfig(
        size_t max_kv_memory_bytes,  // 显存预算 (字节)
        size_t block_size = 16);     // 每块 token 数量

    size_t max_kv_memory_bytes() const;  // 获取显存预算
    size_t block_size() const;           // 获取块大小
};

class PagedKVCache final : public Cache {
public:
    // 构造分页 KV 缓存
    PagedKVCache(
        infinicore::Size k_dim,
        infinicore::Size v_dim,
        infinicore::Size num_k_heads,
        infinicore::Size num_v_heads,
        infinicore::Size num_layers,
        infinicore::DataType dtype,
        const PagedKVCacheConfig &config,
        const engine::distributed::RankInfo &rank_info);

    // 更新分页 KV 缓存 (框架实现,未完成)
    // 输入: k [num_rank_k_heads, seq_len, k_dim]
    //       v [num_rank_v_heads, seq_len, v_dim]
    //       slot_mapping [seq_len] (每个 token 的物理槽位)
    // 输出: (full_k, full_v)
    //       full_k: [num_blocks, num_rank_k_heads, block_size, k_dim]
    //       full_v: [num_blocks, num_rank_v_heads, block_size, v_dim]
    std::tuple<infinicore::Tensor, infinicore::Tensor>
    update(size_t layer_idx,
           const infinicore::Tensor &k,
           const infinicore::Tensor &v,
           const infinicore::Tensor &slot_mapping);
};

} // namespace infinilm::cache
```

## 4. Usage Example

```cpp
#include "infinilm/cache/cache.hpp"
#include "infinicore/context/context.hpp"

using namespace infinilm::cache;
using namespace infinicore;

// ============================
// 示例 1: StaticKVCache 用于单请求推理
// ============================
void static_cache_example() {
    // 模型配置 (GPT-2 124M)
    Size k_dim = 64;
    Size v_dim = 64;
    Size num_k_heads = 12;
    Size num_v_heads = 12;
    Size num_layers = 12;
    Size max_pos_emb = 1024;
    DataType dtype = DataType::fp16;

    // 缓存配置: 单批次,最大缓存 1024 tokens
    StaticKVCacheConfig config(1, 1024);

    // 分布式信息 (单卡)
    engine::distributed::RankInfo rank_info;
    rank_info.device = Device::cuda(0);
    rank_info.tp_size = 1;

    // 创建静态 KV 缓存
    StaticKVCache kv_cache(
        k_dim, v_dim, num_k_heads, num_v_heads,
        num_layers, max_pos_emb, dtype, config, rank_info
    );

    // 模拟推理过程: 预填充阶段 (prompt_len=128)
    Size prompt_len = 128;
    auto k_prompt = Tensor::rand({1, 12, prompt_len, k_dim}, dtype, Device::cuda(0));
    auto v_prompt = Tensor::rand({1, 12, prompt_len, v_dim}, dtype, Device::cuda(0));
    auto cache_lengths = Tensor::from_scalar<int64_t>(0, Device::cuda(0));  // 初始位置 0

    // 更新缓存 (第一次)
    auto [k_full, v_full] = kv_cache.update(0, k_prompt, v_prompt, cache_lengths);
    // k_full 形状: [1, 12, 128, 64]

    // 更新 cache_lengths 标量
    cache_lengths->copy_from(Tensor::from_scalar<int64_t>(128, Device::cuda(0)));

    // 自回归生成阶段 (逐 token 生成)
    for (size_t step = 0; step < 10; ++step) {
        auto k_new = Tensor::rand({1, 12, 1, k_dim}, dtype, Device::cuda(0));
        auto v_new = Tensor::rand({1, 12, 1, v_dim}, dtype, Device::cuda(0));

        // 更新缓存 (追加到位置 128, 129, ...)
        auto [k_cached, v_cached] = kv_cache.update(0, k_new, v_new, cache_lengths);
        // k_cached 形状: [1, 12, 129 + step, 64] (完整历史)

        // 更新 cache_lengths
        cache_lengths->copy_from(Tensor::from_scalar<int64_t>(129 + step, Device::cuda(0)));

        // 使用 k_cached, v_cached 进行注意力计算
        // ...
    }
}

// ============================
// 示例 2: StaticKVCache 用于张量并行推理
// ============================
void static_cache_tp_example() {
    // 模型配置 (Llama-2 7B)
    Size k_dim = 128;
    Size v_dim = 128;
    Size num_k_heads = 32;
    Size num_v_heads = 32;
    Size num_layers = 32;
    Size max_pos_emb = 2048;
    DataType dtype = DataType::bf16;

    // 4 卡张量并行
    int tp_size = 4;
    Size local_batch_size = 8;  // 每卡 8 个请求

    StaticKVCacheConfig config(local_batch_size, 2048);

    for (int rank = 0; rank < tp_size; ++rank) {
        engine::distributed::RankInfo rank_info;
        rank_info.device = Device::cuda(rank);
        rank_info.tp_size = tp_size;

        // 每个 rank 创建切分后的缓存
        StaticKVCache kv_cache(
            k_dim, v_dim, num_k_heads, num_v_heads,
            num_layers, max_pos_emb, dtype, config, rank_info
        );
        // 内部计算:
        // num_rank_k_heads_ = 32 / 4 = 8
        // num_rank_v_heads_ = 32 / 4 = 8
        // k_caches_ 形状: [32, 8, 8, 2048, 128]

        // 模拟输入 (每个 rank 处理 8 个头)
        auto k_input = Tensor::rand({8, 8, 16, k_dim}, dtype, Device::cuda(rank));
        auto v_input = Tensor::rand({8, 8, 16, v_dim}, dtype, Device::cuda(rank));
        auto cache_lengths = Tensor::from_scalar<int64_t>(0, Device::cuda(rank));

        auto [k_out, v_out] = kv_cache.update(0, k_input, v_input, cache_lengths);
        // k_out 形状: [8, 8, 16, 128]
    }
}

// ============================
// 示例 3: PagedKVCache 用于动态批处理推理 (框架代码)
// ============================
void paged_cache_example() {
    // 模型配置
    Size k_dim = 128;
    Size v_dim = 128;
    Size num_k_heads = 32;
    Size num_v_heads = 32;
    Size num_layers = 32;
    DataType dtype = DataType::bf16;

    // 显存预算: 10GB KV 缓存,块大小 16
    size_t max_kv_memory_bytes = 10ULL * 1024 * 1024 * 1024;
    PagedKVCacheConfig config(max_kv_memory_bytes, 16);

    engine::distributed::RankInfo rank_info;
    rank_info.device = Device::cuda(0);
    rank_info.tp_size = 1;

    // 创建分页缓存
    PagedKVCache paged_cache(
        k_dim, v_dim, num_k_heads, num_v_heads,
        num_layers, dtype, config, rank_info
    );
    // 内部计算块数量 (假设 bf16, dsize=2):
    // num_blocks_per_layer = 10GB / (128*32 + 128*32) / 16 / 2
    //                      = 10737418240 / 8192 / 16 / 2 ≈ 40960 块

    // 模拟 slot mapping (非连续写入)
    // 例如: 请求 A 的 token 写入块 5 的槽位 0, 请求 B 的 token 写入块 12 的槽位 3
    auto k_input = Tensor::rand({32, 5, k_dim}, dtype, Device::cuda(0));  // 5 个新 tokens
    auto v_input = Tensor::rand({32, 5, v_dim}, dtype, Device::cuda(0));

    // slot_mapping: [0, 1, 2, 3, 4] -> 物理槽位 (block_id * block_size + offset)
    auto slot_mapping = Tensor::from_vector<int64_t>({5*16+0, 5*16+1, 12*16+3, 12*16+4, 7*16+15},
                                                     Device::cuda(0));

    // 更新分页缓存 (当前为框架实现,未完成)
    auto [k_paged, v_paged] = paged_cache.update(0, k_input, v_input, slot_mapping);
    // k_paged 形状: [40960, 32, 16, 128] (完整块池视图)

    // 后续需要实现: 根据 slot_mapping 散射写入 (scatter write)
    // 使用类似 vLLM 的 PagedAttention 机制
}
```

## 5. Implementation Details

**Memory Management (内存管理策略)**:
- **StaticKVCache**:
  - 预分配策略: 使用 `Tensor::empty()` 在构造时一次性分配连续显存
  - 5D 张量布局: `[layers, batch, heads, seq_len, dim]`, 便于 narrow 操作
  - 显存计算: `layers * batch * heads * cache_len * (k_dim + v_dim) * dtype_size`
  - 适用场景: 单请求或固定批次推理,显存占用可预测
  - 优势: 简单高效,零碎片化
  - 劣势: 显存利用率低 (短序列浪费长序列预留空间)

- **PagedKVCache**:
  - 块池策略: 根据 `max_kv_memory_bytes` 动态计算块数量,预分配块池
  - 5D 分页布局: `[layers, blocks, heads, block_size, dim]`
  - 块数量计算公式: `num_blocks = kv_memory / (k_dim * k_heads + v_dim * v_heads) / block_size / dtype_size`
  - 适用场景: 动态批处理,可变长度序列,高并发推理
  - 优势: 显存利用率高,支持非连续写入和块级复用 (vLLM 风格)
  - 劣势: 需要额外 slot mapping 管理开销

**Distributed Computing (分布式计算)**:
- **张量并行 (Tensor Parallelism, TP)**:
  - `num_rank_k_heads_ = num_k_heads / tp_size`: 注意力头沿 TP 维度切分
  - `num_rank_v_heads_ = num_v_heads / tp_size`: K 和 V 头数均分到各 rank
  - 每个 rank 仅存储和更新本地头对应的 KV 缓存
  - `k_caches_` 和 `v_caches_` 的头维度使用 `num_rank_k_heads_` (而非全局头数)
  - 输入张量形状: `[batch, num_rank_heads, seq_len, dim]` (已切分)
- **数据并行**:
  - 每个 rank 处理独立批次 (`rank_batch_size_`)
  - 缓存张量的 batch 维度切分: `[num_layers, rank_batch_size, num_rank_heads, ...]`
- **设备管理**:
  - 通过 `rank_info.device` 指定分配设备 (CUDA/ROCm/CPU)
  - 缓存张量直接分配在目标设备,避免 CPU-GPU 拷贝

**Concurrency (并发与线程安全)**:
- **无锁设计**: 静态缓存的 `update()` 方法无内部锁,依赖外部同步
- **假设**: 单线程写入 (推理循环) 或外层加锁保护
- **张量操作**: `narrow()` 和 `copy_from()` 假设线程安全 (InfiniCore 内部实现)
- **分页缓存**: 需要实现原子块分配器 (未实现),参考 vLLM 的 `BlockAllocator`

**Performance Optimization (性能优化)**:
- **StaticKVCache**:
  - `narrow()` 返回视图 (零拷贝),仅 `copy_from()` 执行实际内存拷贝
  - 连续内存布局优化缓存局部性,提高内存带宽利用率
  - 更新路径: O(seq_len * heads * dim) 线性拷贝,无额外开销
- **PagedKVCache** (设计目标):
  - 块粒度内存管理,支持细粒度显存复用
  - 预期配合 PagedAttention 内核,减少注意力计算量
  - slot mapping 支持非连续写入,避免显存碎片整理

**Error Handling (错误处理)**:
- **断言检查**:
  - `ASSERT(layer_idx < rank_num_layers_)`: 防止层索引越界
  - `ASSERT(result_len <= cache_len_)`: 防止缓存溢出 (静态)
  - `ASSERT_EQ(batch_size, rank_batch_size_)`: 批次大小一致性校验
- **运行时异常**:
  - `std::runtime_error("Not enough memory for KV cache")`: 分页缓存显存不足时抛出
- **未定义行为**:
  - 若 `cache_lengths` 标量更新不同步,会导致数据覆盖或越界 (外部责任)

**Dependencies (依赖关系)**:
- **InfiniCore**:
  - `infinicore/tensor.hpp`: 张量抽象 (`Tensor::empty`, `narrow`, `copy_from`, `to`)
  - `infinicore/device.hpp`: 设备枚举 (`Device::cuda`, `Device::cpu`)
  - `infinicore/context/context.hpp`: 上下文管理
  - `infinicore::dsize()`: 数据类型字节大小查询
- **Distributed Engine**:
  - `engine/distributed/distributed.hpp`: `RankInfo` 结构体 (tp_size, device)
- **Utils**:
  - `../utils.hpp`: `ASSERT` 和 `ASSERT_EQ` 宏定义
- **spdlog**:
  - 用于日志记录 (头文件引入但未在实现中使用)

**Design Patterns (设计模式)**:
- **策略模式 (Strategy Pattern)**:
  - `Cache` 抽象基类定义接口,`StaticKVCache` 和 `PagedKVCache` 实现不同缓存策略
  - 运行时可选择缓存实现 (静态 vs 分页)
- **工厂模式 (Factory Pattern)** (隐式):
  - `CacheConfig::unique_copy()` 支持配置对象的多态克隆
  - 便于配置管理器和缓存实例的解耦
- **RAII (Resource Acquisition Is Initialization)**:
  - 构造函数分配显存,析构函数自动释放 (`Tensor` 的析构函数管理)
  - 无需手动 free,避免资源泄漏
- **Template Method (模板方法)** (未完全实现):
  - `CacheConfig` 定义 `unique_copy()` 接口,派生类实现具体克隆逻辑

**Incomplete Implementation (未完成部分)**:
- **PagedKVCache::update()**:
  - 当前仅返回缓存张量视图,未实现 slot mapping 的散射写入逻辑
  - TODO 注释: `/// @todo: implement paged cache update here`
  - 需要实现的核心逻辑:
    1. 解析 `slot_mapping` 提取 `(block_id, offset)` 对
    2. 对每个 token,计算目标物理位置并执行散射写入
    3. 处理跨块写入的边界情况
    4. 支持原子块分配和回收 (需要额外的块管理器)
