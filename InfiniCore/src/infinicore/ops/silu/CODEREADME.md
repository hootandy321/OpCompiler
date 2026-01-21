# SiLU (Swish) 激活函数核心实现文档

本模块实现了 SiLU (Sigmoid Linear Unit，也称为 Swish) 激活函数，该函数在现代深度学习中广泛使用，特别是在 Transformer 架构（如 LLaMA、GPT 等大语言模型）中作为门控机制的激活函数。SiLU 的数学定义为：`SiLU(x) = x * sigmoid(x)`，其中 `sigmoid(x) = 1 / (1 + e^(-x))`。

## 1. 模块结构

- **`silu.cc`**: SiLU 操作的核心实现层，包含设备无关的执行逻辑、函数调度器和公共 API
- **`silu_infiniop.cc`**: 基于 InfiniOP 后端的具体实现，提供高效的设备特定计算实现，包含 descriptor 缓存机制
- **`include/infinicore/ops/silu.hpp`**: 公共 API 接口定义

## 2. 核心类

### `Silu`
- **位置**: `include/infinicore/ops/silu.hpp`, `silu.cc`
- **主要功能**: SiLU 操作的调度器和执行入口，采用静态多态模式实现设备无关的操作接口
- **关键成员**:
  - 无数据成员，采用纯静态设计模式
- **核心方法**:
  - `execute(Tensor output, Tensor input)`: 执行 SiLU 操作的入口函数。首先验证输入输出张量在同一设备上，然后根据设备类型查找并调用对应的实现函数。如果找不到对应设备的实现，抛出 `std::runtime_error` 异常。时间复杂度 O(1) 用于调度，实际计算复杂度取决于后端实现。
  - `dispatcher() -> OpDispatcher<schema>&`: 返回静态本地调度器实例（单例模式），用于管理不同设备类型的实现函数。调度器采用函数指针映射表实现 O(1) 查找。
- **生命周期**: 使用静态局部变量（Meyers Singleton）确保调度器在首次调用时初始化，程序结束时自动销毁。

## 3. API 接口

```cpp
// 函数式 API：分配输出张量并执行 SiLU 操作
Tensor silu(Tensor input);
// 参数: input - 输入张量
// 返回: 包含 SiLU(x) 结果的新张量，形状和数据类型与输入相同

// 原地操作 API：在预先分配的输出张量中执行 SiLU
void silu_(Tensor output, Tensor input);
// 参数: output - 输出张量（必须预先分配），input - 输入张量
// 作用: 计算 SiLU(input) 并将结果写入 output

// 类静态方法：直接执行接口
void Silu::execute(Tensor output, Tensor input);
// 参数: output - 输出张量，input - 输入张量
// 作用: 通过调度器查找设备特定实现并执行计算
```

## 4. 使用示例

```cpp
#include "infinicore/ops/silu.hpp"
#include "infinicore/tensor.hpp"

using namespace infinicore;

// 示例 1: 基本使用 - 自动分配输出张量
Tensor input = Tensor::empty({1024, 512}, DataType::FLOAT32, Device::cuda(0));
// ... 填充输入数据 ...

Tensor output = op::silu(input);  // output = input * sigmoid(input)

// 示例 2: 原地操作 - 预先分配输出（适合内存复用场景）
Tensor input = Tensor::empty({256, 256}, DataType::FLOAT32, Device::cuda(0));
Tensor output = Tensor::empty(input->shape(), input->dtype(), input->device());
op::silu_(output, input);  // 结果写入 output

// 示例 3: 使用 Silu 类直接执行
Tensor input = Tensor::load("activation.bin");
Tensor output = Tensor::empty(input->shape(), input->dtype(), input->device());
op::Silu::execute(output, input);  // 与 silu_() 等价
```

## 5. 实现细节

### **多态调度机制 (Polymorphic Dispatcher Pattern)**
- **设计模式**: 策略模式 (Strategy Pattern) + 单例模式 (Singleton Pattern)
- **实现方式**: `OpDispatcher<void(*)(Tensor, Tensor>` 使用函数指针映射表，键为 `Device::Type` 枚举
- **注册机制**: 使用静态初始化器（Static Initializer）在程序启动时自动注册实现
  ```cpp
  // silu_infiniop.cc:45-48
  static bool registered = []() {
      Silu::dispatcher().registerAll(&calculate, false);
      return true;
  }();
  ```
  这里使用 lambda 表达式作为静态变量初始化器，利用 C++ 的静态变量初始化保证在 `main()` 执行前完成注册。`registerAll(&calculate, false)` 将 `calculate` 函数注册到所有设备类型，`false` 参数表示不强制覆盖已有实现。

### **Descriptor 缓存机制 (Descriptor Cache)**
- **数据结构**: `thread_local OpCache<size_t, infiniopSiluDescriptor_t>` 每线程 LRU 缓存
- **缓存容量**: 100 个 descriptor 条目
- **缓存键生成**: 使用 `hash_combine(output, input)` 生成唯一哈希值，综合考虑：
  - 张量数据类型 (`DataType`)
  - 张量形状 (`Shape`: 所有维度)
  - 张量步长 (`Strides`: 所有维度的步长)
- **哈希算法**: Boost 风格的 `hash_combine`，使用黄金比例常数 `0x9e3779b9` 减少哈希冲突
- **生命周期管理**: 自定义析构函数调用 `infiniopDestroySiluDescriptor()` 释放 descriptor 资源

### **内存管理 (Memory Management)**
- **Workspace 分配**: 每次调用时通过 `infiniopGetSiluWorkspaceSize()` 查询所需临时存储空间，使用 `context::allocateMemory()` 分配设备内存
- **RAII 封装**: workspace 使用 `std::shared_ptr<Memory>` 管理，确保异常安全和自动释放
- **设备上下文**: 执行前通过 `context::setDevice()` 设置 CUDA/设备上下文，确保在正确设备上操作

### **并发控制 (Concurrency Control)**
- **线程局部存储**: `thread_local` 确保每个线程拥有独立的 descriptor 缓存，避免多线程竞争
- **线程安全性**: OpCache 内部使用 LRU 缓存（基于 `std::unordered_map` 和 `std::list`），查找和插入操作需要外部同步（但本实现中每个线程独立缓存，天然线程安全）

### **性能优化 (Performance Optimization)**
- **零拷贝语义**: 输入输出张量由调用者管理，实现内部不执行不必要的数据复制
- **Descriptor 复用**: 通过 LRU 缓存避免重复创建昂贵的 InfiniOP descriptor（descriptor 创建通常涉及 JIT 编译或内核查询）
- **设备感知**: 调度器支持多种设备类型（CUDA、CPU、Kunlun、MetaX、Ascend 等），自动选择最优后端实现

### **错误处理 (Error Handling)**
- **设备不匹配**: `INFINICORE_ASSERT_TENSORS_SAME_DEVICE` 宏确保输入输出在同一设备，否则抛出详细异常信息（包含设备类型、函数名、文件名、行号）
- **未实现设备**: 如果调度器找不到对应设备实现，抛出 `std::runtime_error("No Silu implementation found for device type: ...")`
- **InfiniOP 错误**: `INFINICORE_CHECK_ERROR` 宏检查所有 InfiniOP API 返回值，失败时转换为包含错误描述的 `std::runtime_error`

### **依赖关系 (Dependencies)**
- **外部依赖**:
  - `infiniop.h`: InfiniOP 后端库，提供底层算子实现（`infiniopCreateSiluDescriptor`, `infiniopSilu`, `infiniopGetSiluWorkspaceSize`, `infiniopDestroySiluDescriptor`）
  - `spdlog`: 结构化日志库，用于调试日志（`SPDLOG_DEBUG` 宏）
- **内部依赖**:
  - `infinicore/ops/common/op.hpp`: 操作基类和调度器基础设施
  - `infinicore/ops/common/cache.hpp`: 设备感知的 LRU 缓存实现
  - `infinicore/common/hash.hpp`: 张量哈希组合工具
  - `infinicore/context/context.hpp`: 设备上下文管理（`getDevice()`, `setDevice()`, `getInfiniopHandle()`, `getStream()`）
  - `infinicore/tensor.hpp`: 张量抽象（`Tensor` 智能指针类型）

### **算法复杂度 (Algorithm Complexity)**
- **调度开销**: O(1) 哈希表查找
- **缓存命中**: O(1) descriptor 复用
- **缓存未命中**: O(k) descriptor 创建，其中 k 为后端初始化开销（通常涉及内核编译或参数查询，较昂贵）
- **计算复杂度**: O(n)，其中 n 为输入张量元素数量（SiLU 操作对每个元素独立计算）

### **设计模式总结 (Design Patterns Summary)**
- **Strategy Pattern**: 通过 `OpDispatcher` 实现设备特定的算法策略
- **Singleton Pattern**: 静态本地 `dispatcher_` 实例确保全局唯一调度器
- **Factory Pattern**: `Tensor::empty()` 作为张量工厂方法
- **RAII**: `std::shared_ptr<Memory>` 自动管理 workspace 生命周期
- **Template Method**: `execute()` 定义算法骨架，具体实现由调度器查找的函数指针完成
- **Cache Aside Pattern**: descriptor 缓存采用"先查缓存，未命中则创建并写入"模式
