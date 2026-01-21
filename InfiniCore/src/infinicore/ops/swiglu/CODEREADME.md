# SwiGLU 算子核心实现文档

SwiGLU (Swish-Gated Linear Unit) 是一种现代神经网络激活函数，广泛应用于大型语言模型（如 LLaMA、GLM 等）。该模块实现了 SwiGLU 算子的跨设备统一接口，支持多种硬件后端（CUDA、CPU、Kunlun、Ascend 等）的自动派发和 InfiniOP 后端的高效执行。

## 1. 模块结构

- **`swiglu.cc`**: 实现 SwiGLU 算子的公共接口、设备派发逻辑和高阶 API（分配输出张量的 `swiglu` 和原地操作的 `swiglu_`）
- **`swiglu_infiniop.cc`**: 基于 InfiniOP 后端的具体实现，包含算子描述符的 LRU 缓存机制和线程安全的内存管理

## 2. 核心类

### `SwiGLU`
- **位置**: `swiglu.cc`, `include/infinicore/ops/swiglu.hpp`
- **主要功能**: 提供设备无关的 SwiGLU 算子执行接口，通过静态派发器实现多硬件后端支持
- **关键成员**:
  - `schema`: 函数签名类型别名 `void (*)(Tensor, Tensor, Tensor)`，表示接受三个张量参数（输出 c、输入 a、输入 b）的函数指针类型
- **核心方法**:
  - `dispatcher()`: 返回静态单例 `OpDispatcher<schema>` 实例，使用 Meyer's Singleton 模式确保线程安全的延迟初始化
  - `execute(Tensor c, Tensor a, Tensor b)`: 执行 SwiGLU 计算，执行流程：
    1. 断言三个张量必须位于同一设备（使用 `INFINICORE_ASSERT_TENSORS_SAME_DEVICE` 宏）
    2. 根据输出张量 `c` 的设备类型，从派发器查找对应的实现函数
    3. 若查找失败，抛出 `std::runtime_error` 异常（包含设备类型信息）
    4. 调用查找到的函数指针执行计算
- **生命周期**: 单例模式，`dispatcher()` 函数内的 `static` 变量在首次调用时构造，程序结束时析构

### `swiglu_impl::infiniop::calculate`
- **位置**: `swiglu_infiniop.cc`
- **主要功能**: InfiniOP 后端的具体实现函数，负责算子描述符的创建/缓存和内核调度
- **核心逻辑**:
  1. **缓存键生成**: 使用 `hash_combine(c, b, a)` 生成基于输入输出张量形状、数据类型和步长的哈希值作为缓存键
  2. **描述符查找**: 从线程本地缓存 `caches` 中查找是否已存在匹配的 `infiniopSwiGLUDescriptor_t`
  3. **描述符创建** (缓存未命中时):
     - 调用 `infiniopCreateSwiGLUDescriptor` 创建算子描述符
     - 将新描述符存入 LRU 缓存
  4. **工作空间查询**: 调用 `infiniopGetSwiGLUWorkspaceSize` 获取所需临时内存大小
  5. **内存分配**: 通过 `context::allocateMemory` 分配工作空间内存
  6. **内核执行**: 调用 `infiniopSwiGLU` 在当前 CUDA/stream 上执行计算

### `OpCache<size_t, infiniopSwiGLUDescriptor_t>`
- **位置**: `swiglu_infiniop.cc` (线程局部静态变量)
- **主要功能**: 管理每个设备上的算子描述符缓存，使用 LRU 策略自动清理过期描述符
- **关键参数**:
  - 容量: 100 个描述符（每设备）
  - 析构函数: 自定义 lambda，调用 `infiniopDestroySwiGLUDescriptor` 释放描述符资源
- **线程安全性**: 使用 `thread_local` 声明，每个线程拥有独立缓存实例，避免多线程竞争

## 3. API 接口

```cpp
// 高阶 API：自动分配输出张量
Tensor swiglu(Tensor a, Tensor b);
// 参数:
//   a - 第一个输入张量（门控分支）
//   b - 第二个输入张量（线性分支）
// 返回: 新分配的输出张量，形状和数据类型与 a 相同

// 原地操作 API：用户预分配输出张量
void swiglu_(Tensor c, Tensor a, Tensor b);
// 参数:
//   c - 输出张量（必须预分配，形状与 a、b 相同）
//   a - 第一个输入张量
//   b - 第二个输入张量

// 底层执行 API：直接调用派发器
class SwiGLU {
    static void execute(Tensor c, Tensor a, Tensor b);
    static common::OpDispatcher<schema> &dispatcher();
};
```

## 4. 使用示例

```cpp
#include "infinicore/ops/swiglu.hpp"
#include "infinicore/tensor.hpp"

using namespace infinicore;

// 场景 1: 使用高阶 API（自动分配输出）
void example_high_level() {
    // 假设 a 和 b 是已存在的输入张量
    Tensor a = /* ... */;
    Tensor b = /* ... */;

    // 自动分配输出张量并执行计算
    Tensor c = op::swiglu(a, b);

    // c 现在包含 SwiGLU(a, b) 的结果
}

// 场景 2: 使用原地操作 API（预分配输出）
void example_in_place() {
    Tensor a = /* ... */;
    Tensor b = /* ... */;

    // 预分配输出张量
    Shape shape = a->shape();
    Tensor c = Tensor::empty(shape, a->dtype(), a->device());

    // 原地执行计算
    op::swiglu_(c, a, b);
}

// 场景 3: 多设备支持（自动派发）
void example_multi_device() {
    Tensor a_cpu = Tensor::empty({1024}, DataType::FLOAT32, Device::cpu());
    Tensor b_cpu = Tensor::empty({1024}, DataType::FLOAT32, Device::cpu());

    Tensor a_cuda = Tensor::empty({1024}, DataType::FLOAT32, Device::cuda(0));
    Tensor b_cuda = Tensor::empty({1024}, DataType::FLOAT32, Device::cuda(0));

    // 同一个 API 自动适配不同设备
    Tensor c_cpu = op::swiglu(a_cpu, b_cpu);  // 使用 CPU 后端
    Tensor c_cuda = op::swiglu(a_cuda, b_cuda); // 使用 CUDA 后端
}
```

## 5. 实现细节

### 内存管理
- **输出张量分配**: 高阶 API `swiglu` 通过 `Tensor::empty` 在目标设备上分配与输入张量形状相同的连续内存
- **工作空间管理**: InfiniOP 实现通过 `infiniopGetSwiGLUWorkspaceSize` 动态查询内核所需临时内存大小，使用 `context::allocateMemory` 分配，并在函数返回后自动释放（通过 `shared_ptr` 管理生命周期）
- **描述符缓存**: 使用线程局部 LRU 缓存（容量 100）存储已创建的算子描述符，避免重复创建开销。缓存键基于张量的形状、数据类型和步长的组合哈希值

### 并发与线程安全
- **派发器**: `OpDispatcher` 内部使用 `std::array` 存储函数指针表，多线程同时调用 `lookup` 是安全的（只读操作），`registerDevice` 需要在初始化阶段完成（通常在静态初始化时通过 lambda 注册）
- **缓存隔离**: 使用 `thread_local` 关键字声明缓存，每个线程拥有独立的 LRU 缓存实例，完全避免多线程竞争和锁开销
- **设备上下文**: 执行前通过 `context::setDevice(c->device())` 确保在正确的设备上下文中执行，支持多设备环境

### 性能优化
- **描述符缓存**: LRU 缓存避免重复调用 `infiniopCreateSwiGLUDescriptor`（该操作通常涉及内核编译和参数校验，开销较大）
- **哈希键计算**: `hash_combine` 使用 Boost 风格的哈希组合算法（魔法常数 `0x9e3779b9`，黄金比例分数的整数表示），快速生成张量配置的唯一指纹
- **零拷贝语义**: 原地操作 API `swiglu_` 允许用户复用输出张量内存，减少分配/释放开销
- **延迟初始化**: 派发器使用 Meyer's Singleton 模式，仅在首次调用时构造，避免启动时开销

### 错误处理
- **设备不匹配**: `INFINICORE_ASSERT_TENSORS_SAME_DEVICE` 宏在运行时检查所有输入/输出张量是否位于同一设备，失败时抛出包含详细设备信息和调用栈的异常
- **后端未注册**: 若当前设备类型未注册实现函数，`execute` 抛出 `std::runtime_error`，错误消息包含设备类型的数值
- **InfiniOP 错误**: `INFINICORE_CHECK_ERROR` 宏封装所有 InfiniOP API 调用，失败时通过 `infini_status_string` 获取人类可读的错误描述并抛出异常

### 依赖关系
- **内部依赖**:
  - `infinicore/tensor.hpp`: 张量抽象，提供形状、数据类型、设备和内存访问接口
  - `infinicore/device.hpp`: 设备类型枚举和设备标识
  - `infinicore/context/context.hpp`: 设备上下文管理、内存分配和流管理
  - `infinicore/common/hash.hpp`: 张量哈希组合函数，用于缓存键生成
  - `infinicore/ops/common/op.hpp`: 算子基类和派发器接口
  - `infinicore/ops/common/dispatcher.hpp`: 设备类型到函数指针的映射表
  - `infinicore/ops/common/cache.hpp`: 多设备 LRU 缓存封装
  - `infinicore/common/LRUCache.hpp`: LRU 缓存实现，支持自定义析构函数
- **外部依赖**:
  - `<infiniop.h>`: InfiniOP 后端库，提供算子描述符创建、工作空间查询和内核执行接口
  - `spdlog`: 结构化日志库（通过 `utils.hpp` 引入）

### 设计模式
- **Strategy Pattern (策略模式)**: 通过 `OpDispatcher` 实现运行时设备后端选择，客户端代码无需感知具体实现
- **Template Method Pattern (模板方法模式)**: `execute` 定义算法骨架（验证 -> 派发 -> 执行），具体实现由各后端函数指针提供
- **Singleton Pattern (单例模式)**: 派发器使用 Meyer's Singleton 确保全局唯一实例
- **Flyweight Pattern (享元模式)**: 算子描述符缓存共享相同配置的描述符对象，减少重复创建开销
- **RAII (Resource Acquisition Is Initialization)**: 缓存析构函数自动调用 `infiniopDestroySwiGLUDescriptor` 释放资源，工作空间内存通过 `shared_ptr` 自动管理

### 注册机制
InfiniOP 实现通过静态初始化期间的 lambda 表达式自动注册到派发器：
```cpp
static bool registered = []() {
    SwiGLU::dispatcher().registerAll(&calculate, false);
    return true;
}();
```
- `registerAll`: 将 `calculate` 函数注册到所有设备类型（CPU、CUDA、Kunlun 等）
- `false` 参数: 不覆盖已存在的实现（若某设备已有其他实现，优先使用该实现）
- `static bool`: 变量在程序启动时初始化，确保在主函数执行前完成注册
