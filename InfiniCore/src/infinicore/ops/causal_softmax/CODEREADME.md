# CausalSoftmax 算子核心实现文档

本模块实现了因果注意力机制中的 softmax 操作，广泛应用于 Transformer 架构的自注意力层。该算子结合了 causal masking（因果掩码）和 softmax 归一化，确保模型在预测时只能访问历史信息。

## 1. 模块结构

- **`causal_softmax.cc`**: 算子前端接口层，提供设备无关的 API 和分发逻辑
- **`causal_softmax_infiniop.cc`**: InfiniOp 后端实现，通过缓存机制优化描述符创建和内核调度

## 2. 核心类与组件

### `CausalSoftmax`
- **位置**: `causal_softmax.cc`
- **主要功能**: 设备无关的算子接口，负责执行设备类型分发和错误处理
- **核心成员**:
  - `dispatcher()`: 静态方法，返回 `OpDispatcher<schema>` 单例，用于管理不同设备类型的实现函数
- **核心方法**:
  - `execute(Tensor output, Tensor input)`: 执行因果 softmax 操作的主入口
    - 验证输入输出张量在同一设备上（`INFINICORE_ASSERT_TENSORS_SAME_DEVICE`）
    - 设置当前设备上下文（`infinicore::context::setDevice`）
    - 根据设备类型查找相应的实现函数
    - 如果未找到实现，抛出 `std::runtime_error` 异常
  - `dispatcher()`: 使用函数静态局部变量实现 Meyers Singleton 模式，确保线程安全的延迟初始化
- **生命周期**: 单例模式，`dispatcher` 实例在首次调用时构造，程序结束时销毁

### `OpDispatcher<schema>`
- **位置**: `/include/infinicore/ops/common/dispatcher.hpp`
- **主要功能**: 设备类型到实现函数的映射表，支持运行时多态分发
- **核心成员**:
  - `table_`: `std::array<Fn, static_cast<size_t>(Device::Type::COUNT>`，存储每个设备类型的函数指针
- **核心方法**:
  - `registerDevice(Device::Type, Fn, bool)`: 注册单个设备类型的实现函数
  - `registerAll(Fn, bool)`: 批量注册所有设备类型（CPU, NVIDIA, CAMBRICON, ASCEND, METAX, MOORE, ILUVATAR, KUNLUN, HYGON, QY）
  - `lookup(Device::Type)`: O(1) 时间复杂度查找设备对应的实现函数
- **设计模式**: Strategy Pattern（策略模式），将算法实现封装为可互换的策略

### `OpCache<Key, Value>`
- **位置**: `/include/infinicore/ops/common/cache.hpp`
- **主要功能**: 多设备 LRU 缓存，为每个设备和设备索引维护独立的缓存实例
- **核心成员**:
  - `caches_`: `std::array<CacheVector, static_cast<size_t>(Device::Type::COUNT)>`，二维数组结构，第一维是设备类型，第二维是设备索引
  - `capacity_`: 缓存容量，默认 100
  - `destructor_`: 值的析构函数，用于资源清理
- **核心方法**:
  - `getCache(Device)`: 获取特定设备的缓存实例，自动扩容 `CacheVector` 以适应设备索引
  - `setCapacity(size_t)`: 动态调整所有缓存的容量，自动驱逐超出容量的条目
  - `clear()`: 清理所有缓存，正确处理设备切换（先切换到目标设备再清理，避免跨设备操作）
- **内存管理**: 使用 `shared_ptr` 管理资源，析构时自动清理所有设备的缓存

### `LRUCache<Key, Value>`
- **位置**: `/include/infinicore/common/LRUCache.hpp`
- **主要功能**: 泛型 LRU（Least Recently Used）缓存实现
- **核心成员**:
  - `list_`: `std::list<KeyValuePair>`，双向链表，front = 最近使用，back = 最久未用
  - `map_`: `std::unordered_map<Key, ListIt>`，哈希表，O(1) 查找
  - `capacity_`: 缓存容量，0 表示无界
  - `destructor_`: `std::function<void(Value &)>`，自定义析构逻辑
- **核心方法**:
  - `put(Key, Value)`: 插入或更新键值对
    - 键已存在：调用析构函数销毁旧值，更新值，移动到 front
    - 键不存在：容量满时驱逐 back 条目，插入到 front
  - `get(Key)`: 查找键，命中则移动到 front 并返回值，未命中返回 `std::nullopt`
  - `touch(typename unordered_map::iterator)`: 内部方法，将条目移动到链表头部，O(1) 时间（`splice` 操作）
  - `evictLRU()`: 驱逐最久未使用的条目（链表 back），调用析构函数
- **算法复杂度**:
  - 插入: O(1) 平均（哈希表插入 + 链表头部插入）
  - 查询: O(1) 平均（哈希表查找）
  - 驱逐: O(1)（链表尾部删除）
- **设计模式**: 使用 `std::list` + `std::unordered_map` 组合实现经典 LRU 缓存

### `causal_softmax_impl::infiniop::calculate`
- **位置**: `causal_softmax_infiniop.cc`
- **主要功能**: InfiniOp 后端的因果 softmax 实现，通过描述符缓存优化性能
- **核心局部变量**:
  - `caches`: `thread_local OpCache<size_t, infiniopCausalSoftmaxDescriptor_t>`，线程本地缓存，容量 100
    - 析构函数：调用 `infiniopDestroyCausalSoftmaxDescriptor` 销毁描述符
- **核心流程**:
  1. 计算缓存键：`hash_combine(output, input)`，融合输出和输入张量的 dtype、shape、strides
  2. 获取当前设备对应的缓存实例
  3. 缓存查找：`cache.get(seed)`，返回 `std::optional<infiniopCausalSoftmaxDescriptor_t>`
  4. 缓存未命中：
     - 调用 `infiniopCreateCausalSoftmaxDescriptor` 创建描述符
     - 将描述符放入缓存：`cache.put(seed, desc)`
  5. 查询工作空间大小：`infiniopGetCausalSoftmaxWorkspaceSize`
  6. 分配工作空间内存：`context::allocateMemory(workspace_size)`
  7. 执行内核：`infiniopCausalSoftmax(desc, workspace, workspace_size, output->data(), input->data(), stream)`
- **性能优化**:
  - 描述符缓存：避免重复创建昂贵的 InfiniOp 描述符对象
  - 线程本地存储：避免多线程竞争，提高并发性能
  - LRU 驱逐策略：自动淘汰最久未使用的描述符，控制内存占用

### `hash_combine`
- **位置**: `/include/infinicore/common/hash.hpp`
- **主要功能**: 泛型哈希组合函数，用于生成缓存键
- **核心算法**:
  ```cpp
  seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  ```
  - 使用 Boost 哈希组合算法的变体
  - `0x9e3779b9`: 黄金比例分数的 32 位表示，用于减少哈希冲突
  - `(seed << 6) + (seed >> 2)`: 位混淆，增强雪崩效应
- **特化版本**:
  - `Tensor`: 哈希 dtype、所有 shape 维度、所有 stride 值
  - `std::optional<T>`: 哈希 `has_value()` 和值（如果存在）
  - `std::string` / `const char*`: 委托给 `std::hash<std::string>`
- **变参模板**: 支持任意数量的参数，递归组合所有参数的哈希值
- **复杂度**: O(n)，n 为所有参数中元素的个数（Tensor 的 shape + stride 维度之和）

## 3. API 接口

```cpp
namespace infinicore::op {

// 主要 API：创建输出张量并执行因果 softmax
Tensor causal_softmax(Tensor input);
// 参数：
//   input - 输入张量，形状通常为 [batch_size, seq_len, seq_len] 或 [batch_size, num_heads, seq_len, seq_len]
// 返回：
//   新分配的输出张量，与输入形状和 dtype 相同
// 注意：
//   该函数会分配新内存，如果需要就地操作，使用 causal_softmax_

// 原地 API：预分配输出张量
void causal_softmax_(Tensor output, Tensor input);
// 参数：
//   output - 预分配的输出张量，必须与输入形状和 dtype 相同
//   input - 输入张量
// 行为：
//   直接调用 CausalSoftmax::execute，内部会验证张量在同一设备上

// 类接口：底层执行入口
class CausalSoftmax {
public:
    using schema = void (*)(Tensor, Tensor);  // 函数签名类型别名

    // 执行因果 softmax 操作
    static void execute(Tensor output, Tensor input);
    // 抛出：
    //   std::runtime_error - 如果输入输出张量不在同一设备，或当前设备类型无实现

    // 获取设备分发器单例
    static common::OpDispatcher<schema> &dispatcher();
    // 返回：
    //   全局唯一的分发器实例引用，用于注册和查询设备实现
};

} // namespace infinicore::op
```

## 4. 使用示例

```cpp
#include "infinicore/ops/causal_softmax.hpp"
#include "infinicore/tensor.hpp"

using namespace infinicore;
using namespace infinicore::op;

// 示例 1：基础用法 - 自动分配输出张量
void example1() {
    // 创建输入张量 [batch_size=2, seq_len=128, seq_len=128]
    Shape shape = {2, 128, 128};
    Device device(Device::Type::NVIDIA, 0);  // 使用 NVIDIA GPU 0
    DataType dtype = DataType::Float32;

    Tensor input = Tensor::zeros(shape, dtype, device);
    // ... 填充输入数据（注意力分数）

    // 执行因果 softmax，自动分配输出张量
    Tensor output = causal_softmax(input);
    // output 形状: [2, 128, 128]，与 input 相同

    // 注意：output 张量中，对于位置 i，只包含位置 0 到 i 的 softmax 归一化结果
    // 位置 j > i 的值被掩码（通常设置为 -inf，softmax 后为 0）
}

// 示例 2：高级用法 - 预分配输出张量
void example2() {
    Shape shape = {4, 8, 256, 256};  // [batch=4, heads=8, seq_len=256, seq_len=256]
    Device device(Device::Type::NVIDIA, 0);
    DataType dtype = DataType::Float16;  // 使用半精度浮点

    Tensor input = Tensor::empty(shape, dtype, device);
    Tensor output = Tensor::empty(shape, dtype, device);

    // ... 填充 input 数据

    // 原地执行，避免额外内存分配
    causal_softmax_(output, input);

    // 输出张量可直接用于后续计算
}

// 示例 3：多设备并行 - 在不同 GPU 上执行
void example3() {
    Shape shape = {1, 512, 512};

    // 在 GPU 0 上执行
    Tensor input0 = Tensor::zeros(shape, DataType::Float32, Device(Device::Type::NVIDIA, 0));
    Tensor output0 = causal_softmax(input0);

    // 在 GPU 1 上执行
    Tensor input1 = Tensor::zeros(shape, DataType::Float32, Device(Device::Type::NVIDIA, 1));
    Tensor output1 = causal_softmax(input1);

    // 两个 GPU 上的操作独立并行执行，无需同步
}

// 示例 4：错误处理 - 捕获异常
void example4() {
    Tensor input = Tensor::zeros({2, 128, 128}, DataType::Float32, Device::cpu());
    Tensor output = Tensor::empty({2, 128, 128}, DataType::Float32, Device(Device::Type::NVIDIA, 0));

    try {
        // 错误：输入和输出不在同一设备
        causal_softmax_(output, input);
    } catch (const std::runtime_error &e) {
        std::cerr << "错误: " << e.what() << std::endl;
        // 输出: "Tensor devices mismatch CPU vs NVIDIA from causal_softmax_ at ..."
    }
}

// 示例 5：集成到 Transformer 自注意力计算
void example5() {
    // 假设 Q, K, V 是查询、键、值张量
    Tensor Q, K, V;  // [batch, heads, seq_len, head_dim]

    // 计算 Q @ K^T，得到注意力分数 [batch, heads, seq_len, seq_len]
    Tensor scores = matmul(Q, transpose(K, {-2, -1}));

    // 缩放（常见于 Transformer）
    scores = scores * (1.0f / std::sqrt(Q.size(-1)));

    // 应用因果 softmax，确保只能看到历史信息
    Tensor attention_weights = causal_softmax(scores);

    // 加权求和得到输出
    Tensor output = matmul(attention_weights, V);  // [batch, heads, seq_len, head_dim]
}
```

## 5. 实现细节

### 内存管理
- **描述符生命周期**: 使用 `OpCache` 管理，缓存容量 100，LRU 驱逐策略自动释放最久未使用的描述符
- **工作空间分配**: 每次执行时动态分配，通过 `context::allocateMemory` 获取 `shared_ptr<Memory>`，函数返回后自动释放
- **线程安全**: `caches` 变量使用 `thread_local` 存储，每个线程拥有独立缓存，无需加锁
- **跨设备资源清理**: `OpCache::clear()` 在清理前会切换到目标设备上下文，确保在正确的设备上释放资源

### 并发控制
- **线程本地缓存**: `thread_local OpCache` 避免多线程竞争，每个线程维护独立的描述符缓存
- **设备上下文隔离**: `OpCache` 为每个 `(Device::Type, device_index)` 对维护独立的缓存实例，防止跨设备访问冲突
- **无锁设计**: LRU 缓存的 `get/put` 操作无原子操作开销，适用于单线程场景（thread_local）

### 性能优化
- **描述符缓存**: InfiniOp 描述符创建开销大（包含内核编译、参数验证），缓存后避免重复创建
- **哈希键计算**: `hash_combine` 使用位混淆算法，快速生成唯一缓存键
- **LRU 驱逐策略**: 自动淘汰最久未使用的描述符，保持高命中率
- **工作空间复用**: 每次执行动态分配，避免预先分配大块内存（工作空间大小因配置而异）

### 错误处理
- **设备不匹配**: `INFINICORE_ASSERT_TENSORS_SAME_DEVICE` 宏检查输入输出张量设备一致性，抛出带详细信息的 `std::runtime_error`
- **未实现设备**: `OpDispatcher::lookup` 返回 nullptr 时，抛出异常并报告设备类型
- **InfiniOp 错误**: `INFINICORE_CHECK_ERROR` 宏检查所有 InfiniOp API 调用，失败时转换错误码为可读字符串
- **缓存析构异常**: `LRUCache::safeDestruct` 捕获析构函数异常，输出到 `std::cerr`，防止程序崩溃

### 依赖关系
- **外部依赖**: `infiniop.h`（InfiniOp 算子库）、`infinirt.h`（InfiniRT 运行时）
- **内部依赖**:
  - `infinicore/context`: 设备管理、内存分配、流管理
  - `infinicore/tensor`: 张量抽象、数据访问
  - `infinicore/device`: 设备类型枚举和设备表示
  - `infinicore/common/hash`: 哈希组合函数
  - `infinicore/ops/common`: 分发器、缓存基础设施

### 设计模式
- **Singleton Pattern**（单例模式）: `CausalSoftmax::dispatcher()` 使用函数静态局部变量实现 Meyers Singleton，确保全局唯一分发器
- **Strategy Pattern**（策略模式）: `OpDispatcher` 将不同设备的实现封装为可互换的策略，运行时选择
- **Template Method Pattern**（模板方法模式）: `causal_softmax` 定义算法骨架（分配输出），`causal_softmax_` 定义具体步骤（执行操作）
- **RAII Pattern**: `LRUCache` 析构时自动清理所有缓存条目，`OpCache` 析构时清理所有设备的缓存
- **Proxy Pattern**（代理模式）: `OpCache` 代理多个 `LRUCache` 实例，提供统一的设备索引接口

### 扩展性
- **新设备支持**: 实现 `calculate` 函数的新版本，调用 `CausalSoftmax::dispatcher().registerDevice` 注册
- **自定义后端**: 继承或替换 `causal_softmax_impl::infiniop::calculate`，提供不同的内核实现
- **缓存策略调整**: 修改 `caches` 的容量参数或替换 `OpCache` 为其他缓存策略（如 LFU、ARC）
- **性能监控**: 在 `calculate` 函数中添加计时逻辑，记录缓存命中率和执行时间

### 线程安全性
- **OpDispatcher**: `table_` 在注册阶段初始化，后续只读，线程安全
- **OpCache::getCache**: 扩容 `CacheVector` 时非线程安全，应在单线程环境或外部加锁
- **LRUCache**: 非 `const` 的 `get/put` 操作修改链表和哈希表，非线程安全（但 thread_local 隔离了线程）
- **InfiniOp 描述符**: 假设底层实现是线程安全的（不同流并发执行不同算子）

### 已知限制
- **缓存容量**: 固定为 100，对于大规模动态形状可能频繁驱逐
- **跨设备张量**: 不支持输入和输出在不同设备上的操作（会抛出异常）
- **同步执行**: `execute` 函数是同步的，异步执行需要手动管理流
- **错误恢复**: InfiniOp 错误直接抛出异常，不支持重试或降级策略

### 适用场景
- **Transformer 自注意力**: GPT、BERT、LLaMA 等架构的解码器层
- **因果语言模型**: 确保预测当前位置时只能看到历史上下文
- **流式推理**: 增量生成时，每次新的 token 都需要因果掩码
- **多 GPU 训练**: 数据并行或张量并行时，每个 GPU 独立执行因果 softmax
