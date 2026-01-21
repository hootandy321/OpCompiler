# Attention Operations Core Implementation Documentation

该模块实现了高效的注意力机制操作，支持多硬件后端和KV缓存优化，专门为Transformer模型推理和训练设计。通过InfiniOP后端库提供底层优化实现，并采用LRU缓存机制减少操作描述符创建开销。

## 1. Module Structure

- **`attention.cc`**: 注意力操作的统一接口层，定义了`Attention`类、调度器以及公共API函数，负责设备一致性检查和设备调度
- **`attention_infiniop.cc`**: InfiniOP后端实现，包含线程局部LRU缓存、操作描述符管理、工作空间分配和实际计算执行

## 2. Core Classes

### `Attention`
- **Location**: `include/infinicore/ops/attention.hpp`, `attention.cc`
- **Primary Function**: 提供静态调度接口，使用OpDispatcher模式实现多硬件后端的自动路由和调用
- **Key Members**:
  - `dispatcher()`: 静态方法，返回OpDispatcher单例，管理设备类型到实现函数的映射表
- **Core Methods**:
  - `execute(Tensor out, Tensor q, Tensor k, Tensor v, Tensor k_cache, Tensor v_cache, size_t pos)`: 主执行入口，验证所有输入张量位于同一设备，设置目标设备，通过调度器查找并调用对应后端实现，时间复杂度O(1)
  - `dispatcher()`: 返回`OpDispatcher<schema>`引用，使用函数局部静态初始化（Magic Static）保证线程安全的延迟初始化
- **Lifecycle**: 采用静态单例模式，调度器在首次调用时构造，程序生命周期内持久存在

## 3. API Interface

```cpp
namespace infinicore::op {

// 类型定义：注意力操作函数指针签名
using schema = void (*)(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, size_t);

// 执行注意力计算（输出张量已预分配）
void Attention::execute(
    Tensor out,           // 输出张量 [seq_len, n_q_head, head_dim]
    Tensor q,             // Query张量 [n_q_head, seq_len, head_dim]
    Tensor k,             // Key张量 [n_kv_head, seq_len, head_dim]
    Tensor v,             // Value张量 [n_kv_head, seq_len, head_dim]
    Tensor k_cache,       // Key缓存张量（可选）
    Tensor v_cache,       // Value缓存张量（可选）
    size_t pos            // 当前token位置索引
);

// 自动分配输出张量并执行注意力计算
Tensor attention(
    Tensor q, Tensor k, Tensor v,
    Tensor k_cache, Tensor v_cache,
    size_t pos
);
// 返回新建的输出张量，形状为 {seq_len, n_q_head, head_dim}

// In-place版本，输出张量由调用者预先分配
void attention_(
    Tensor out, Tensor q, Tensor k, Tensor v,
    Tensor k_cache, Tensor v_cache,
    size_t pos
);

} // namespace infinicore::op
```

## 4. Usage Example

```cpp
// 示例：在Transformer推理中使用注意力模块
#include "infinicore/ops/attention.hpp"

using namespace infinicore;

// 1. 初始化上下文和设备
context::initialize(); // 初始化InfiniOP运行时
Device device(Device::Type::NVIDIA, 0); // 使用第一个NVIDIA GPU
context::setDevice(device);

// 2. 准备输入张量（假设batch_size=1, seq_len=128, n_head=32, head_dim=128）
Shape q_shape = {32, 128, 128};  // [n_q_head, seq_len, head_dim]
Shape kv_shape = {4, 128, 128};  // [n_kv_head, seq_len, head_dim] (GQA)
Tensor q = Tensor::empty(q_shape, DType::Float16, device);
Tensor k = Tensor::empty(kv_shape, DType::Float16, device);
Tensor v = Tensor::empty(kv_shape, DType::Float16, device);

// 填充输入数据（实际应用中从模型前一层获取）
// ... 填充 q, k, v 数据 ...

// 3. 可选：准备KV缓存（用于自回归生成）
Shape cache_shape = {4, 4096, 128}; // [n_kv_head, max_cache_len, head_dim]
Tensor k_cache = Tensor::empty(cache_shape, DType::Float16, device);
Tensor v_cache = Tensor::empty(cache_shape, DType::Float16, device);

size_t current_pos = 0; // 当前解码位置

// 4. 方法A：使用自动内存分配的便捷API
Tensor output = op::attention(q, k, v, k_cache, v_cache, current_pos);
// 输出形状: [128, 32, 128] 即 [seq_len, n_q_head, head_dim]

// 5. 方法B：使用预分配输出张量的高性能API
Shape out_shape = {128, 32, 128};
Tensor output_preallocated = Tensor::empty(out_shape, DType::Float16, device);
op::attention_(output_preallocated, q, k, v, k_cache, v_cache, current_pos);

// 6. 在生成循环中逐步更新缓存
for (size_t step = 0; step < 100; ++step) {
    // ... 计算新的 q, k, v (单个token) ...

    // 执行带缓存的注意力计算
    op::attention_(output_preallocated, q, k, v, k_cache, v_cache, current_pos);

    current_pos++; // 更新位置索引
}

// 7. 清理（RAII自动管理，显式调用可选）
context::finalize();
```

## 5. Implementation Details

### 多硬件后端调度机制 (Multi-Backend Dispatching)
- **设计模式**: Strategy Pattern + Template Method
- **实现方式**: 使用`OpDispatcher<schema>`类维护函数指针表，索引为`Device::Type`枚举值
- **支持设备**: CPU, NVIDIA GPU, CAMBRICON, ASCEND, METAX, MOORE, ILUVATAR, KUNLUN, HYGON, QY（共10种）
- **注册时机**: 编译期通过静态初始化（`static bool registered`）自动注册InfiniOP实现到所有设备类型
- **查找复杂度**: O(1) 直接数组索引访问

### 线程局部LRU缓存 (Thread-Local LRU Caching)
- **数据结构**: `OpCache<size_t, infiniopAttentionDescriptor_t>`，基于LRU算法的哈希表+双向链表实现
- **缓存键生成**: 使用`hash_combine()`函数组合所有输入张量的数据类型、形状、步长以及位置索引`pos`，采用Boost风格的哈希组合算法（`seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2)`）
- **容量配置**: 每个线程每个设备默认容量100，可动态调整
- **生命周期**: 线程局部存储（thread_local），线程退出时自动销毁，支持多线程并发执行无锁竞争
- **淘汰策略**: LRU算法，当缓存满时移除最久未使用的描述符，驱逐时调用`infiniopDestroyAttentionDescriptor()`释放底层资源

### 操作描述符管理 (Descriptor Management)
- **创建流程**: 缓存未命中时调用`infiniopCreateAttentionDescriptor()`，传入所有张量的描述符和位置参数
- **描述符内容**: 封装张量布局信息、数据类型、注意力算法配置（如FlashAttention、Memory-Efficient Attention等）
- **复用策略**: 相同张量形状和数据类型的连续调用复用同一描述符，避免重复初始化开销
- **销毁机制**: 使用自定义析构器（lambda函数）在缓存驱逐或清理时调用`infiniopDestroyAttentionDescriptor()`，保证资源不泄漏

### 工作空间动态分配 (Dynamic Workspace Allocation)
- **分配时机**: 每次执行前调用`infiniopGetAttentionWorkspaceSize()`查询所需工作空间大小
- **内存来源**: 通过`context::allocateMemory(size_t)`从设备内存池分配，该函数根据设备类型调用CUDA、hip等对应API
- **生命周期**: 临时使用，函数返回后`std::shared_ptr<Memory>`自动释放（引用计数归零）
- **大小变化**: 工作空间大小随序列长度、头维度、缓存长度等参数动态变化

### 设备一致性验证 (Device Consistency Checking)
- **验证时机**: 执行前通过`INFINICORE_ASSERT_TENSORS_SAME_DEVICE`宏检查
- **检查范围**: 所有输入张量（out, q, k, v, k_cache, v_cache）必须位于同一设备
- **失败处理**: 抛出`std::runtime_error`异常，包含详细设备信息、函数名、源文件位置和行号
- **性能开销**: O(n)遍历检查，n为参数数量（固定6个），开销可忽略

### 错误处理与日志 (Error Handling & Logging)
- **错误检查**: 使用`INFINICORE_CHECK_ERROR`宏包装所有InfiniOP API调用
- **错误传播**: 检测到错误时调用`infini_status_string()`获取错误描述，构造包含API调用名、错误码和错误消息的异常
- **日志级别**: 使用spdlog库，支持环境变量`INFINICORE_LOG_LEVEL`动态配置（默认info级别）
- **调试输出**: DEBUG级别记录每个API调用的入口和退出位置（文件名:行号）

### 流式执行支持 (Stream Execution)
- **异步执行**: 通过`context::getStream()`获取当前设备的CUDA Stream/HIP Queue等，实现计算与主机异步并发
- **同步点**: InfiniOP内部维护流依赖关系，保证同一流内操作顺序执行
- **多流支持**: 不同线程可使用不同流并发执行，通过设备上下文隔离保证线程安全

### KV缓存优化 (KV-Cache Optimization)
- **缓存结构**: 独立的k_cache和v_cache张量，形状为`[n_kv_head, max_cache_len, head_dim]`
- **增量更新**: `pos`参数指定当前token在缓存序列中的位置，支持逐步填充而无需重新计算历史token
- **内存布局**: 使用连续内存存储，缓存友好，支持高效的增量注意力计算（如FlashAttention-2的KV-cache pass）
- **多查询注意力**: 支持GQA（Grouped Query Attention），允许`n_q_head`和`n_kv_head`不同（如32个query head对应4个KV head）

### 性能优化策略 (Performance Optimization)
- **零拷贝**: 所有操作直接在设备内存上进行，避免主机-设备间数据传输
- **内核融合**: InfiniOP后端可能融合softmax、masked fill、矩阵乘算等子操作，减少内核启动开销
- **内存访问模式**: 利用张量步长信息支持非连续布局（如transpose后的视图），自适应最优访问模式
- **缓存命中率**: 典型推理场景（固定形状和dtype）下缓存命中率接近100%，描述符创建开销降至O(1)

### 依赖关系分析 (Dependencies)
- **外部依赖**: InfiniOP库（`libinfiniop.so`），提供底层硬件加速内核
- **内部依赖**:
  - `infinicore/context`: 设备管理、内存分配、流管理
  - `infinicore/tensor`: 张量数据抽象、形状、步长、设备信息
  - `infinicore/common/op.hpp`: 操作基类和类型定义
  - `infinicore/common/LRUCache.hpp`: LRU缓存数据结构
  - `infinicore/common/hash.hpp`: 张量哈希组合函数
  - `spdlog`: 结构化日志记录
- **编译依赖**: C++17标准（支持std::optional, inline variables）

### RAII资源管理 (RAII Resource Management)
- **张量内存**: 使用`std::shared_ptr`管理，离开作用域自动释放
- **缓存描述符**: OpCache析构时遍历所有缓存项调用自定义析构器
- **设备上下文**: 使用RAII包装器（context::DeviceGuard）保证设备切换后恢复
- **异常安全**: 所有资源使用智能指针或RAII包装，异常发生时正确清理

### 编译期多态 (Compile-Time Polymorphism)
- **模板特化**: `OpDispatcher<schema>`根据函数签名类型生成专用调度器
- **静态注册**: 使用函数级静态变量和lambda表达式实现"构造时注册"，无需手动调用注册代码
- **类型安全**: 编译期检查函数签名匹配，避免运行时类型错误
