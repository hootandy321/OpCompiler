# Cache Manager Core Implementation Documentation

本模块实现了 InfiniLM 框架中的两级缓存管理系统：(1) KV Cache 管理器，用于多设备多层的 Key-Value 缓存分配与生命周期管理；(2) Operator Descriptor Cache 管理器，基于 LRU 策略缓存 infiniop 算子描述符以减少重复初始化开销。该模块是 LLM 推理系统中内存管理与性能优化的核心组件。

## 1. Module Structure

- **`kvcache.cpp`**: KV Cache 的创建、复制与销毁实现，支持多设备并行分配和跨设备内存复制
- **`opcache_manager.hpp`**: 基于泛型 LRU 算法的算子描述符缓存管理器，支持十种 infiniop 算子类型的自动缓存与驱逐

## 2. Core Classes

### `KVCache`
- **Location**: `/home/qy/src/Infini/InfiniLM/src/cache.hpp`
- **Primary Function**: 存储多层 Transformer 的 Key 和 Value 缓存张量，采用双层嵌套向量结构组织多设备多层的缓存布局
- **Key Members**:
  - `k`: `std::vector<std::vector<std::shared_ptr<Tensor>>>` - 外层向量按设备索引，内层向量按层索引存储 Key 张量
  - `v`: `std::vector<std::vector<std::shared_ptr<Tensor>>>` - 对应的 Value 张量存储结构
- **Data Layout**:
  - 第一维：`cache->k.size()` = 设备数量 (ndev)
  - 第二维：`cache->k[0].size()` = Transformer 层数 (nlayers)
  - 每个 Tensor 形状：`[max_len, nkvh/ndev, dk或dv]`

### `LRUDescriptorCache<DescriptorType>`
- **Location**: `opcache_manager.hpp`
- **Primary Function**: 泛型 LRU 缓存实现，用于管理任意类型的 infiniop 算子描述符，提供自动驱逐和资源清理机制
- **Key Members**:
  - `cache`: `std::unordered_map<size_t, CacheNode *>` - 哈希表提供 O(1) 查找
  - `head`, `tail`: `CacheNode *` - 双向链表哨兵节点，维护访问顺序（head=最近使用，tail=最久未用）
  - `capacity`: `const size_t` - 缓存容量上限
  - `size`: `size_t` - 当前缓存项数量
  - `destroyer`: `std::unique_ptr<IDescriptorDestroyer>` - 多态资源销毁器接口
- **Core Methods**:
  - `get(size_t key, DescriptorType &out_desc)`: 查找缓存命中时移动节点到链表头部并返回 true，O(1) 复杂度
  - `put(size_t key, const DescriptorType &descriptor)`: 插入新描述符或更新已存在 key，容量超限时自动驱逐尾部节点，O(1) 复杂度
  - `removeNode(CacheNode *node)`: 从双向链表和哈希表中移除节点，调用 destroyer 释放描述符资源
  - `moveToTop(CacheNode *node)`: 将已存在节点移动到链表头部（标记为最近使用）
  - `addToTop(CacheNode *node)`: 新节点插入头部，并在插入后检查是否触发驱逐
- **Algorithm**: 标准 LRU (Least Recently Used) 策略，使用哈希表 + 双向链表实现，查找和更新均为 O(1)

### `CacheManager`
- **Location**: `opcache_manager.hpp`
- **Primary Function**: 统一管理十种 infiniop 算子的描述符缓存，为每种算子类型提供类型安全的 get/put 接口
- **Key Members**:
  - `Add_cache`: `LRUDescriptorCache<infiniopAddDescriptor_t>` - 加法算子描述符缓存
  - `RMSNorm_cache`: `LRUDescriptorCache<infiniopRMSNormDescriptor_t>` - RMS 归一化算子缓存
  - `Gemm_cache`: `LRUDescriptorCache<infiniopGemmDescriptor_t>` - 矩阵乘法算子缓存
  - `RoPE_cache`: `LRUDescriptorCache<infiniopRoPEDescriptor_t>` - 旋转位置编码算子缓存
  - `Rearrange_cache`: `LRUDescriptorCache<infiniopRearrangeDescriptor_t>` - 张量重排算子缓存
  - `CausalSoftmax_cache`: `LRUDescriptorCache<infiniopCausalSoftmaxDescriptor_t>` - 因果 softmax 算子缓存
  - `Topkrouter_cache`: `LRUDescriptorCache<infiniopTopkrouterDescriptor_t>` - Top-K 路由算子缓存
  - `SwiGLU_cache`: `LRUDescriptorCache<infiniopSwiGLUDescriptor_t>` - SwiGLU 激活函数算子缓存
  - `RandomSample_cache`: `LRUDescriptorCache<infiniopRandomSampleDescriptor_t>` - 随机采样算子缓存
  - `DequantizeAWQ_cache`: `LRUDescriptorCache<infiniopDequantizeAWQDescriptor_t>` - AWQ 量化算子缓存
- **Core Methods**:
  - `CacheManager(size_t capacity)`: 构造函数，为所有算子缓存初始化相同容量并绑定对应销毁函数
  - `createDescriptorKey<Tensors...>(Tensors... tensors)`: 静态模板方法，基于多个 Tensor 的 seed() 值通过 FNV-1a 哈希组合生成描述符唯一键
  - `get[OpType]Descriptor(size_t key, [OpType]Descriptor_t &desc)`: 宏生成的类型安全获取接口
  - `put[OpType]Descriptor(size_t key, const [OpType]Descriptor_t &desc)`: 宏生成的类型安全插入接口

### `IDescriptorDestroyer` / `DescriptorDestroyer<DescriptorType>`
- **Location**: `opcache_manager.hpp`
- **Primary Function**: 类型擦除的资源销毁接口，通过多态支持不同算子描述符类型的统一管理
- **Key Members**:
  - `destroyFunc`: `DestroyFunc` (函数指针类型) - 指向特定的 infiniop 算子销毁函数（如 `infiniopDestroyAddDescriptor`）
- **Core Methods**:
  - `destroy(void *descriptor)`: 类型擦除的销毁接口，内部将 void* 转换为正确类型后调用底层销毁函数

## 3. API Interface

```cpp
// KV Cache 管理 API
struct KVCache *createKVCache(
    size_t nlayers,        // Transformer 层数
    size_t max_len,        // 最大序列长度
    size_t nkvh_,          // KV head 总数
    size_t dk,             // Key 维度
    size_t dv,             // Value 维度
    infiniDtype_t dtype,   // 数据类型
    infiniDevice_t device, // 设备类型（CUDA/MUSA/XPU等）
    int *dev_ids,          // 设备 ID 数组
    size_t ndev            // 设备数量
);
// 创建多层多设备 KV Cache，每个设备的 KV head 数量为 nkvh_/ndev

struct KVCache *duplicateKVCache(const KVCache *kv_cache, size_t seq_len);
// 深拷贝现有 KV Cache（仅拷贝前 seq_len 个 token），用于批处理或beam search
// 复杂度：O(ndev * nlayers)，使用 D2D 内存拷贝

void dropKVCache(KVCache *kv_cache);
// 释放所有设备的所有层的 KV Cache 张量，先 reset shared_ptr 再 delete 结构体

// Cache Manager API（通过 DECLARE_OP_CACHE 宏自动生成）
bool getAddDescriptor(size_t key, infiniopAddDescriptor_t &desc);
void putAddDescriptor(size_t key, const infiniopAddDescriptor_t &desc);
// 类似接口还支持：RMSNorm, Gemm, RoPE, Rearrange, CausalSoftmax, Topkrouter, SwiGLU, RandomSample, DequantizeAWQ

size_t CacheManager::createDescriptorKey(Tensors... tensors);
// 基于多个 Tensor 的元数据（形状、步长、数据类型）生成唯一哈希键
```

## 4. Usage Example

```cpp
// 示例 1: 创建和使用 KV Cache
// 场景：32 层 LLaMA-3-70B 模型，8 卡 NVIDIA GPU 分布式推理

size_t nlayers = 32;
size_t max_len = 8192;
size_t nkvh = 8;          // 8 个 KV heads
size_t dk = 128;          // Key 维度
size_t dv = 128;          // Value 维度
infiniDtype_t dtype = INFINI_DTYPE_F16;
int dev_ids[8] = {0, 1, 2, 3, 4, 5, 6, 7};

// 创建 KV Cache（每个设备分配 1 个 KV head）
KVCache *cache = createKVCache(nlayers, max_len, nkvh, dk, dv,
                               dtype, INFINI_DEVICE_CUDA, dev_ids, 8);

// cache->k[idev][layer] 形状为 [8192, 1, 128]
// cache->v[idev][layer] 形状为 [8192, 1, 128]

// 复制前 2048 个 token 的缓存用于新的推理分支
KVCache *branch_cache = duplicateKVCache(cache, 2048);

// 使用完毕后释放
dropKVCache(cache);
dropKVCache(branch_cache);


// 示例 2: 使用 CacheManager 缓存算子描述符
// 场景：RMSNorm 算子描述符缓存以避免重复初始化

CacheManager cache_mgr(100); // 每种算子最多缓存 100 个描述符

// 准备输入张量
auto input = Tensor::buffer(INFINI_DTYPE_F16, {1, 128, 4096});
auto weight = Tensor::buffer(INFINI_DTYPE_F32, {4096});
auto output = Tensor::buffer(INFINI_DTYPE_F16, {1, 128, 4096});

// 生成唯一缓存键（基于张量形状和类型）
size_t key = CacheManager::createDescriptorKey(input, weight, output);

// 尝试从缓存获取
infiniopRMSNormDescriptor_t rms_desc;
if (!cache_mgr.getRMSNormDescriptor(key, rms_desc)) {
    // 缓存未命中，创建新描述符
    RUN_INFINI(infiniopCreateRMSNormDescriptor(
        &rms_desc,
        input->desc(),
        weight->desc(),
        output->desc(),
        1e-5  // epsilon
    ));

    // 放入缓存
    cache_mgr.putRMSNormDescriptor(key, rms_desc);
}

// 使用 rms_desc 进行计算
// RUN_INFINI(infiniopRMSNorm(..., rms_desc, ...));


// 示例 3: CacheManager 在模型推理中的典型使用模式
// 假设我们有一个 Transformer 层需要执行多个算子

class TransformerLayer {
    CacheManager *cache_mgr;

    void forward(Tensor *input, Tensor *output) {
        // 1. RMSNorm 1
        size_t rms_key = CacheManager::createDescriptorKey(input, rms_weight, hidden);
        infiniopRMSNormDescriptor_t rms_desc;
        if (!cache_mgr->getRMSNormDescriptor(rms_key, rms_desc)) {
            infiniopCreateRMSNormDescriptor(&rms_desc, ...);
            cache_mgr->putRMSNormDescriptor(rms_key, rms_desc);
        }
        infiniopRMSNorm(handle, rms_desc, ...);

        // 2. QKV Projection (GEMM)
        size_t gemm_key = CacheManager::createDescriptorKey(hidden, qkv_weight);
        infiniopGemmDescriptor_t gemm_desc;
        if (!cache_mgr->getGemmDescriptor(gemm_key, gemm_desc)) {
            infiniopCreateGemmDescriptor(&gemm_desc, ...);
            cache_mgr->putGemmDescriptor(gemm_key, gemm_desc);
        }
        infiniopGemm(handle, gemm_desc, ...);

        // 3. RoPE
        size_t rope_key = CacheManager::createDescriptorKey(q, k, cos, sin);
        infiniopRoPEDescriptor_t rope_desc;
        if (!cache_mgr->getRoPEDescriptor(rope_key, rope_desc)) {
            infiniopCreateRoPEDescriptor(&rope_desc, ...);
            cache_mgr->putRoPEDescriptor(rope_key, rope_desc);
        }
        infiniopRoPE(handle, rope_desc, ...);

        // ... 后续算子类似
    }
};
```

## 5. Implementation Details

### 内存管理策略
- **KV Cache 分配**: 使用 `Tensor::buffer()` 在每个设备上分配连续内存块，形状为 `[max_len, nkvh/ndev, dk或dv]`，采用 row-major 布局
- **张量生命周期**: KVCache 使用 `std::shared_ptr<Tensor>` 管理 Tensor 对象，支持多引用共享，dropKVCache 通过 reset() 递减引用计数自动释放底层内存
- **跨设备复制**: duplicateKVCache 使用 `infinirtMemcpy` 的 `INFINIRT_MEMCPY_D2D` 模式实现设备间直接内存拷贝（无需经过主机），拷贝字节数通过 `seq_len * nkvh/ndev * dk(dv) * dsize(dtype)` 精确计算

### LRU 缓存实现细节
- **双向链表结构**: 使用哨兵节点（head 和 tail）简化边界处理，head->next 指向最近使用节点，tail->prev 指向最久未用节点
- **驱逐策略**: 当 size >= capacity 时，在 addToTop 中自动调用 removeNode(tail->prev) 驱逐最久未用节点
- **哈希冲突处理**: 使用 `std::unordered_map` 的链地址法处理哈希冲突，value 指向 CacheNode
- **更新语义**: put 操作遇到已存在 key 时，先调用 destroyer 销毁旧描述符，再更新为新的描述符并移到链表头部

### 并发安全性
- **当前实现**: **非线程安全**，LRUDescriptorCache 未使用任何锁机制
- **使用限制**: CacheManager 应该单线程使用，或由上层调用者保证同步
- **潜在优化**: 如需多线程访问，需在 get/put 操作外加锁，或使用读写锁提高并发读性能

### 类型安全与代码生成
- **宏驱动设计**: `DECLARE_OP_CACHE` 宏为每种算子类型生成类型安全的 get/put 方法，避免手动编写重复代码
- **多态销毁器**: 通过 `IDescriptorDestroyer` 接口和 `DescriptorDestroyer<DescriptorType>` 模板实现类型擦除，LRUDescriptorCache 无需知道具体描述符类型即可调用正确的销毁函数
- **模板方法**: `createDescriptorKey` 使用折叠表达式（fold expression）`(..., (tensors ? hash_combine(seed, tensors->seed()) : (void)0))` 递归组合多个 Tensor 的哈希值

### 性能特征
- **时间复杂度**:
  - `createKVCache`: O(ndev * nlayers)，每个设备每层调用两次 Tensor::buffer（分配 O(1)）
  - `duplicateKVCache`: O(ndev * nlayers)，每个设备每层执行两次 D2D 内存拷贝
  - `dropKVCache`: O(ndev * nlayers)，每个设备每层调用两次 shared_ptr::reset（递减引用计数 O(1)）
  - `LRUDescriptorCache::get`: O(1) 平均（哈希查找 + 链表移动）
  - `LRUDescriptorCache::put`: O(1) 平均（哈希查找/插入 + 链表插入/移动 + 驱逐）
- **空间复杂度**:
  - KV Cache: O(ndev * nlayers * max_len * nkvh/ndev * (dk + dv) * dsize(dtype))
  - CacheManager: O(10 * capacity * sizeof(CacheNode))，每个 CacheNode 约 48 字节（key + desc指针 + 两个指针 + 哈希表开销）

### 依赖关系
- **外部依赖**:
  - `infinicore_infer.h`: 提供 infiniDtype_t, infiniDevice_t, RUN_INFINI 宏, infiniop 描述符类型及销毁函数
  - `tensor.hpp`: Tensor, TensorDesc 类及 dsize() 函数
  - `utils.hpp`: hash_combine 函数（用于 createDescriptorKey）
  - InfiniRT: `infinirtSetDevice`, `infinirtMemcpy`, `INFINIRT_MEMCPY_D2D`
- **内部模块交互**:
  - KVCache 被 Transformer 模型推理引擎使用，存储自注意力计算的中间结果
  - CacheManager 被 InfiniLM 的算子调度器使用，缓存频繁使用的算子描述符（如不同形状的 Matmul, RMSNorm）

### 设计模式
- **RAII (Resource Acquisition Is Initialization)**: LRUDescriptorCache 析构函数自动清理所有缓存的描述符，防止资源泄漏
- **策略模式 (Strategy Pattern)**: IDescriptorDestroyer 接口封装不同的资源销毁策略（不同算子有不同销毁函数）
- **模板方法模式 (Template Method)**: createDescriptorKey 定义哈希组合算法骨架，具体由 Tensor::seed() 提供原子哈希值
- **工厂模式 (Factory) 变体**: Tensor::buffer() 作为工厂方法创建 Tensor 对象，KVCache 构造函数批量调用此工厂

### 关键算法实现
- **LRU 驱逐算法**:
  1. 新节点插入时调用 addToTop(node)
  2. addToTop 先将 node 插入 head 之后
  3. 检查 ++size > capacity，若成立则调用 removeNode(tail->prev)
  4. removeNode 从双向链表摘除节点，调用 destroyer->destroy() 释放资源，从哈希表删除，delete 节点
- **哈希键生成**:
  - Tensor::seed() 计算张量元数据（dtype, shape, strides）的哈希值
  - hash_combine 使用 FNV-1a 或类似算法混合多个 seed 值：`seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2)`
  - 最终生成 64 位 size_t 哈希键，冲突概率极低

### 错误处理
- **RUN_INFINI 宏**: 封装 infiniop/infinirt 函数调用，检查返回值并在失败时抛出异常或终止程序（具体行为取决于 utils.hpp 中宏定义）
- **内存分配失败**: Tensor::buffer() 内部调用 Storage::create 或 createFromPool，失败时可能抛出 std::bad_alloc
- **越界访问**: duplicateKVCache 中的 seq_len 参数不应超过 max_len，但代码未显式检查，需调用者保证

### 已知限制与潜在改进
- **duplicateKVCache 循环错误**: 第 49 行存在重复的 `for (unsigned int layer = 0; layer < nlayers; layer++)` 嵌套，导致内层循环重复执行 nlayers 次（应为 1 次），这是明显的 bug
- **无显式缓存统计**: CacheManager 未提供命中率、驱逐次数等性能指标接口，难以监控缓存效果
- **固定容量**: 所有算子共享相同容量，无法根据实际使用频率调整（如 Gemm 缓存可能需要更大容量）
- **手动 Tensor 销毁**: dropKVCache 需手动遍历并 reset，未使用 RAII 自动管理（如改为 unique_ptr 或析构函数自动清理）
