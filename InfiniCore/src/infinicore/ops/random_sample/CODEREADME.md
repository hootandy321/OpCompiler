# Random Sample 算子核心实现文档

该模块实现了基于 logits 的随机采样操作，支持 top-p (nucleus) sampling、top-k sampling 和 temperature scaling，常用于大语言模型的 token 生成阶段。

## 1. 模块结构

- **`random_sample.cc`**: RandomSample 类的核心实现，提供设备无关的调度接口和公开 API
- **`random_sample_infiniop.cc`**: 基于 InfiniOp 后端的实现，包含描述符缓存和 kernel 调度逻辑
- **`include/infinicore/ops/random_sample.hpp`**: 公开 API 接口定义

## 2. 核心类

### `RandomSample`
- **位置**: `include/infinicore/ops/random_sample.hpp`, `random_sample.cc`
- **主要功能**: 提供设备无关的随机采样操作接口，通过 OpDispatcher 实现多设备后端分发
- **类型别名**:
  - `schema`: `void (*)(Tensor, Tensor, float, float, int, float)` - 操作签名类型，定义了函数指针类型
- **核心方法**:
  - `execute(Tensor indices, Tensor logits, float random_val, float topp, int topk, float temperature)`:
    - 执行随机采样操作
    - 验证 indices 和 logits 张量在同一设备上（`INFINICORE_ASSERT_TENSORS_SAME_DEVICE`）
    - 设置当前设备为 logits 所在设备
    - 通过 dispatcher 查找并调用对应设备类型的实现函数
    - 时间复杂度: O(1) 分发开销，实际复杂度取决于后端实现
  - `dispatcher()`: 返回静态 OpDispatcher 实例，采用 Meyer's Singleton 模式
- **生命周期**: 静态类，无需实例化，所有方法均为静态

### `OpDispatcher<schema>`
- **位置**: `include/infinicore/ops/common/dispatcher.hpp`
- **主要功能**: 设备类型到函数指针的查找表，支持运行时多设备分发
- **核心成员**:
  - `std::array<Fn, static_cast<size_t>(Device::Type::COUNT)> table_`: 函数指针数组，按 Device::Type 索引
- **核心方法**:
  - `registerDevice(Device::Type device_type, Fn fn, bool override_existing = true)`: 注册特定设备类型的实现，支持覆盖已存在的注册
  - `registerAll(Fn fn, bool override_existing = true)`: 批量注册所有设备类型，遍历 Device::Type::COUNT
  - `lookup(Device::Type device_type)`: 查找并返回对应设备类型的函数指针，使用 `std::array::at()` 进行边界检查
- **设计模式**: Strategy Pattern + Template Method Pattern

### `OpCache<size_t, infiniopRandomSampleDescriptor_t>`
- **位置**: `include/infinicore/ops/common/cache.hpp`, `random_sample_infiniop.cc`
- **主要功能**: 基于 LRU 策略的 InfiniOp 描述符缓存，按设备类型和设备索引隔离
- **核心成员**:
  - `std::array<CacheVector, static_cast<size_t>(Device::Type::COUNT)> caches_`: 二维数组，第一维是设备类型，第二维是设备索引
  - `size_t capacity_`: 每个 LRU 缓存的容量上限
  - `Destructor destructor_`: 描述符析构函数，用于清理缓存中的 InfiniOp 描述符
- **核心方法**:
  - `getCache(Device device)`: 获取特定设备的 LRU 缓存实例，自动扩容设备索引维度
  - `clear()`: 清空所有设备的缓存，正确处理多 GPU 环境下的设备切换
- **缓存策略**: LRU (Least Recently Used)，容量为 100 个描述符

## 3. API 接口

```cpp
// Out-of-place API: 自动创建输出张量
Tensor random_sample(
    Tensor logits,      // 输入: logits 张量，通常形状为 [vocab_size]
    float random_val,   // 随机种子值，用于采样
    float topp,         // top-p (nucleus) 采样阈值，范围 [0, 1]
    int topk,           // top-k 采样保留的最高概率 token 数量
    float temperature   // 温度参数，控制分布平滑度
);
// 返回: I32 类型的标量张量，包含采样的 token 索引

// In-place API: 用户预分配输出张量
void random_sample_(
    Tensor indices,     // 输出: 预分配的标量张量，I32 类型
    Tensor logits,      // 输入: logits 张量
    float random_val,   // 随机种子值
    float topp,         // top-p 阈值
    int topk,           // top-k 数量
    float temperature   // 温度参数
);

// 底层执行接口
class RandomSample {
    using schema = void (*)(Tensor, Tensor, float, float, int, float);
    static void execute(
        Tensor indices, Tensor logits,
        float random_val, float topp, int topk, float temperature
    );
    static common::OpDispatcher<schema> &dispatcher();
};
```

## 4. 使用示例

```cpp
#include "infinicore/ops/random_sample.hpp"
#include "infinicore/tensor.hpp"

using namespace infinicore;

// 示例 1: Out-of-place 采样
void example_out_of_place() {
    // 假设 logits 是模型输出的预测分布，形状为 [vocab_size]
    Tensor logits = Tensor::empty({32000}, DataType::F32, Device::cuda(0));

    // 填充 logits 数据（通常是模型输出的 log probabilities）

    // 使用 top-p=0.9, top_k=50, temperature=1.0 进行采样
    Tensor sampled_token = op::random_sample(
        logits,
        0.5f,        // random_val (随机值)
        0.9f,        // topp - 保留累计概率 90% 的 tokens
        50,          // topk - 只从概率最高的 50 个 tokens 中采样
        1.0f         // temperature - 不改变原始分布
    );

    // sampled_token 是标量张量，包含采样的 token id
    int token_id = sampled_token->data<int>()[0];
}

// 示例 2: In-place 采样（减少内存分配）
void example_in_place() {
    Tensor logits = Tensor::empty({32000}, DataType::F32, Device::cuda(0));
    Tensor indices = Tensor::empty({}, DataType::I32, Device::cuda(0));

    // greedy decoding: top_k=1, temperature=0
    op::random_sample_(indices, logits, 0.5f, 1.0f, 1, 0.0f);

    int token_id = indices->data<int>()[0];
}

// 示例 3: 温度采样控制多样性
void example_temperature_sampling() {
    Tensor logits = Tensor::empty({32000}, DataType::F32, Device::cuda(0));

    // 低温度 -> 更确定的输出（锐化分布）
    Tensor token_deterministic = op::random_sample(
        logits, 0.5f, 0.95f, 50, 0.7f
    );

    // 高温度 -> 更随机、更多样的输出（平滑分布）
    Tensor token_creative = op::random_sample(
        logits, 0.5f, 0.95f, 50, 1.5f
    );
}
```

## 5. 实现细节

### 内存管理
- **描述符缓存**: 使用 `OpCache` 按设备隔离缓存 `infiniopRandomSampleDescriptor_t`，避免重复创建 InfiniOp 描述符
- **工作空间管理**: 每次调用时通过 `context::allocateMemory()` 动态分配 workspace，由 InfiniOp 后端管理大小
- **内存生命周期**: 描述符由 Lambda 析构函数通过 `infiniopDestroyRandomSampleDescriptor` 释放

### 并发性
- **线程局部存储**: `thread_local OpCache` 确保每个线程拥有独立的缓存实例，避免多线程竞争
- **设备上下文隔离**: `OpCache::getCache(Device)` 按 `(device_type, device_index)` 二维隔离，支持多 GPU 并发
- **流安全**: 所有操作在 `context::getStream()` 返回的 CUDA stream 上执行，支持同一设备上的多流并发

### 性能优化
- **缓存策略**: LRU 缓存容量为 100，减少 `infiniopCreateRandomSampleDescriptor` 调用开销
- **哈希计算**: 使用 `hash_combine` 对 indices 和 logits 张量的 dtype、shape、strides 计算缓存键，避免为不同形状的张量复用描述符
- **零拷贝**: In-place API (`random_sample_`) 允许用户预分配输出张量，减少内存分配开销
- **设备切换优化**: `RandomSample::execute` 仅在必要时设置设备，通过 `INFINICORE_ASSERT_TENSORS_SAME_DEVICE` 验证避免不必要的切换

### 错误处理
- **设备一致性检查**: `INFINICORE_ASSERT_TENSORS_SAME_DEVICE` 确保输入输出张量在同一设备，否则抛出 `std::runtime_error`
- **InfiniOp 错误传播**: `INFINICORE_CHECK_ERROR` 宏检查所有 InfiniOp API 调用，失败时转换为 `std::runtime_error` 并包含错误描述
- **边界检查**: `OpDispatcher::lookup` 使用 `std::array::at()` 进行索引越界检查

### 依赖关系
- **外部依赖**: InfiniOp 库 (`infiniop.h`)，提供底层随机采样 kernel 实现
- **内部模块**:
  - `infinicore::context`: 设备管理、内存分配、stream 获取
  - `infinicore::Tensor`: 张量抽象，提供 data()、desc()、device()、dtype()、shape()、strides() 接口
  - `infinicore::common::LRUCache`: 缓存基类实现
  - `infinicore::common::hash_combine`: 哈希组合函数，用于生成缓存键

### 设计模式
- **Singleton Pattern**: `RandomSample::dispatcher()` 使用 Meyer's Singleton 确保全局唯一分发器
- **Strategy Pattern**: `OpDispatcher` 运行时选择不同设备类型的实现策略
- **Template Method Pattern**: `RandomSample::execute` 定义算法骨架，设备特定实现由 dispatcher 分发
- **Object Pool Pattern**: `OpCache` 缓存昂贵的 InfiniOp 描述符对象，避免重复创建销毁
- **RAII**: `OpCache` 的析构函数自动清理所有缓存的描述符

### 算法细节
- **Top-p (Nucleus) Sampling**: 累积概率达到阈值 topp 的最小 token 集，从该集中采样
- **Top-k Sampling**: 仅从概率最高的 k 个 tokens 中采样
- **Temperature Scaling**: 对 logits 除以 temperature，temperature < 1 锐化分布，> 1 平滑分布
- **工作空间需求**: 由 `infiniopGetRandomSampleWorkspaceSize` 查询，取决于输入张量大小和采样策略

### 多设备支持
- **设备类型**: 支持 CUDA、CPU、ROCm 等所有 `Device::Type` 枚举值
- **多 GPU**: 通过 `Device::getIndex()` 区分同一设备类型的不同设备实例，缓存独立隔离
- **注册机制**: 使用静态初始化器 (`static bool registered = []() {...}()`) 在程序启动时自动注册 InfiniOp 后端到所有设备类型

### 关键常量
- **缓存容量**: 100 个描述符（硬编码在 `thread_local OpCache` 构造函数中）
- **哈希组合算法**: Boost-style hash combine，使用魔数 `0x9e3779b9`（黄金比例分数的 32 位表示）
