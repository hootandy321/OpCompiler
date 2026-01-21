# `Mul` (乘法运算) 操作实现模块

本模块实现了 InfiniCore 框架中的张量乘法操作（element-wise multiplication）。它采用分层架构设计，将操作接口定义、调度分发、具体实现解耦，支持多种设备后端（CUDA、CPU、ROCm 等）的统一调用。模块利用 InfiniOP 库进行底层计算，并通过 descriptor 缓存机制和线程局部存储优化性能。

## 1. 模块结构

- **`mul.cc`**: Mul 操作的核心接口层，提供 `Mul` 类、`mul()` 和 `mul_()` 函数的实现，负责设备检查、调度器调用和高层流程控制
- **`mul_infiniop.cc`**: Mul 操作的 InfiniOP 后端实现，负责 descriptor 创建/缓存、workspace 内存分配、底层 API 调用，并自动注册到调度器
- **`mul.hpp`** (在 `include/infinicore/ops/`): 公共接口定义，声明 `Mul` 类及其 schema 类型、`mul()`/`mul_()` 函数

## 2. 核心类

### `Mul`
- **位置**: `include/infinicore/ops/mul.hpp` 和 `mul.cc`
- **主要功能**: 提供乘法操作的静态执行接口，管理设备类型分发器（dispatcher），实现统一的跨设备调用入口
- **关键成员**:
  - `schema`: 类型别名，定义为 `void (*)(Tensor, Tensor, Tensor)`，即接收三个张量（输出 c，输入 a, b）的函数指针类型
  - `dispatcher()`: 静态方法，返回 `OpDispatcher<schema>` 的单例引用，采用 Meyer's Singleton 模式，通过 `static` 局部变量保证线程安全的延迟初始化
- **核心方法**:
  - `execute(Tensor c, Tensor a, Tensor b)`: 执行乘法操作的主入口
    - **前置检查**: 使用 `INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b)` 宏验证三个张量位于同一设备，否则抛出包含详细设备信息的异常
    - **设备切换**: 调用 `infinicore::context::setDevice(c->device())` 确保当前 CUDA/设备上下文与输出张量一致
    - **分发执行**: 从 dispatcher 查找并调用对应设备类型的实现函数 `dispatcher().lookup(c->device().getType())(c, a, b)`，时间复杂度 O(1)
  - `dispatcher()`: 获取分发器实例，内部使用函数指针数组按 `Device::Type` 索引存储各设备实现
- **生命周期**: 单例模式，静态分发器在首次调用 `dispatcher()` 时构造，程序结束时自动销毁；`Mul` 类本身无实例化成员函数，所有方法均为静态

### `OpDispatcher<schema>`
- **位置**: `include/infinicore/ops/common/dispatcher.hpp`
- **主要功能**: 设备类型到函数指针的映射表，提供注册、查找功能，实现运行时多态
- **关键成员**:
  - `table_`: `std::array<Fn, static_cast<size_t>(Device::Type::COUNT)>`，固定大小的数组，索引为设备类型枚举值，存储对应设备的实现函数指针
- **核心方法**:
  - `registerDevice(Device::Type, Fn, bool)`: 注册单个设备类型的实现，`override_existing` 参数控制是否覆盖已有注册
  - `registerAll(Fn, bool)`: 遍历所有设备类型（CUDA、CPU、ROCm、Metax 等），将同一函数指针注册到所有条目，用于通用后端实现
  - `lookup(Device::Type)`: O(1) 时间复杂度查找并返回对应设备的函数指针，使用 `std::array::at()` 进行边界检查
- **设计模式**: Strategy 模式 + Registry 模式，通过函数指针表实现不同设备算法的运行时选择

### `OpCache<Key, Value>`
- **位置**: `include/infinicore/ops/common/cache.hpp`
- **主要功能**: 线程局部的 LRU 缓存管理器，为每个设备类型和设备索引维护独立的缓存实例，用于缓存 InfiniOP descriptor
- **关键成员**:
  - `caches_`: `std::array<CacheVector, static_cast<size_t>(Device::Type::COUNT)>`，二维结构，第一维是设备类型，第二维是该类型下的设备索引列表
  - `capacity_`: 每个 LRU 缓存的容量限制
  - `destructor_`: 值类型的析构函数，用于清理缓存的 descriptor 资源
- **核心方法**:
  - `getCache(Device)`: 获取指定设备的缓存实例，如果设备索引超出当前 vector 大小则自动扩容并初始化新缓存
  - `setCapacity(size_t)`: 动态调整所有缓存的容量
  - `clear()`: 清理所有缓存，切换到每个设备所在上下文后调用 `LRUCache::clear()` 释放资源，防止设备上下文错误
- **设计模式**: Cache Manager 模式，自动处理多设备环境下的资源隔离和生命周期

## 3. API 接口

```cpp
// 高层 API：自动分配输出张量并执行乘法
Tensor mul(Tensor a, Tensor b);
// 功能：创建新张量 c，形状和数据类型与 a 相同，设备与 a 相同，然后执行 c = a * b
// 参数：a, b 为输入张量（必须同设备、同形状、兼容数据类型）
// 返回：新分配的结果张量 c
// 异常：设备不匹配时抛出 std::runtime_error

// 就地 API：使用预分配张量执行乘法
void mul_(Tensor c, Tensor a, Tensor b);
// 功能：执行 c = a * b，c 必须预先分配
// 参数：c 为输出张量（必须与 a, b 同设备、同形状）
// 异常：设备不匹配时抛出 std::runtime_error

// 底层执行接口
Mul::execute(Tensor c, Tensor a, Tensor b);
// 功能：分发器调用入口，查找当前设备类型对应的实现函数并执行
// 前置条件：c, a, b 必须位于同一设备
// 副作用：切换当前设备上下文到 c 所在设备

// InfiniOP 后端实现函数
namespace infinicore::op::mul_impl::infiniop {
    void calculate(Tensor c, Tensor a, Tensor b);
    // 功能：通过 InfiniOP API 执行乘法，包含 descriptor 缓存、workspace 分配
    // 缓存策略：基于张量 dtype + shape + strides 的组合哈希作为 key
    // 线程安全：使用 thread_local 缓存，每线程独立
}
```

## 4. 使用示例

```cpp
#include "infinicore/ops/mul.hpp"
using namespace infinicore;

// 示例 1: 基本乘法操作（自动分配输出）
void example_basic_mul() {
    // 假设已有两个 CUDA 张量
    Tensor a = Tensor::randn({128, 256}, DataType::FLOAT32, Device::cuda(0));
    Tensor b = Tensor::randn({128, 256}, DataType::FLOAT32, Device::cuda(0));

    // 调用 mul()，自动创建并返回结果张量
    Tensor c = mul(a, b);

    // c 现在包含 a * b 的 element-wise 乘积
    // c 的形状、数据类型、设备与 a 相同
}

// 示例 2: 就地乘法（预分配输出）
void example_inplace_mul() {
    Tensor a = Tensor::randn({64, 64}, DataType::FLOAT32, Device::cuda(0));
    Tensor b = Tensor::randn({64, 64}, DataType::FLOAT32, Device::cuda(0));

    // 预分配输出张量（可能来自内存池或复用缓冲区）
    Tensor c = Tensor::empty(a->shape(), a->dtype(), a->device());

    // 执行就地乘法，结果写入 c
    mul_(c, a, b);
}

// 示例 3: 跨设备乘法（错误示例）
void example_device_mismatch() {
    Tensor a = Tensor::randn({10}, DataType::FLOAT32, Device::cuda(0));
    Tensor b = Tensor::randn({10}, DataType::FLOAT32, Device::cpu());  // 不同设备

    try {
        Tensor c = mul(a, b);  // 抛出异常！
    } catch (const std::runtime_error& e) {
        // 异常信息： "Tensor devices mismatch cuda:0 vs cpu from mul at mul.cc:19."
    }
}

// 示例 4: 多设备并行乘法
void example_multi_device() {
    // 在两个不同 GPU 上创建张量
    Tensor a_gpu0 = Tensor::randn({32, 32}, DataType::FLOAT32, Device::cuda(0));
    Tensor b_gpu0 = Tensor::randn({32, 32}, DataType::FLOAT32, Device::cuda(0));

    Tensor a_gpu1 = Tensor::randn({32, 32}, DataType::FLOAT32, Device::cuda(1));
    Tensor b_gpu1 = Tensor::randn({32, 32}, DataType::FLOAT32, Device::cuda(1));

    // 并行执行（需要在不同的线程或流中）
    Tensor c_gpu0 = mul(a_gpu0, b_gpu0);  // 在 GPU 0 上执行
    Tensor c_gpu1 = mul(a_gpu1, b_gpu1);  // 在 GPU 1 上执行

    // OpCache 为每个设备维护独立的 descriptor 缓存
}
```

## 5. 实现细节

### 内存管理
- **Descriptor 缓存**: 使用 `OpCache<size_t, infiniopMulDescriptor_t>` 缓存 InfiniOP descriptor，capacity 为 100，采用 LRU 淘汰策略
  - **缓存键生成**: 通过 `hash_combine(c, b, a)` 组合三个张量的 dtype、shape、strides 生成 64 位哈希，确保相同形状/布局的张量复用 descriptor
  - **析构策略**: 提供 lambda 析构器调用 `infiniopDestroyMulDescriptor()`，在缓存淘汰或程序结束时自动释放资源
- **Workspace 动态分配**: 每次调用 `infiniopGetMulWorkspaceSize()` 查询所需 workspace 大小，通过 `context::allocateMemory()` 分配，传递指针给底层 kernel，函数返回后自动释放（使用 `shared_ptr` 管理）
- **线程局部存储**: `thread_local OpCache` 保证每个线程拥有独立的 descriptor 缓存，避免多线程竞争和锁开销，适合深度学习训练中常见的数据并行模式

### 并发与线程安全
- **Dispatcher 只读访问**: `OpDispatcher::lookup()` 为 const 方法，仅读取 `std::array`，天然线程安全；注册阶段通常在程序初始化单线程完成
- **线程局部缓存**: 每线程独立的 OpCache 实例，无需互斥锁，但可能在不同线程中重复创建相同 descriptor（权衡：避免锁开销 vs 缓存命中率）
- **设备上下文切换**: `Mul::execute()` 在每次调用前执行 `context::setDevice(c->device())`，确保 CUDA 上下文正确；在多设备多线程环境中，调用者需确保线程与设备绑定或正确管理上下文
- **InfiniOP 流管理**: `infiniopMul()` 接收 `context::getStream()`，支持异步执行和 CUDA 流并发

### 性能优化
- **Descriptor 复用**: 避免重复调用 `infiniopCreateMulDescriptor()`，该操作通常涉及 CUDA kernel 编译或参数推导，开销较大（可能数毫秒），缓存后降至微秒级
- **哈希函数**: 使用 Boost 风格的 `hash_combine` 算法（`seed ^= hash(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2)`），黄金比例常数 0x9e3779b9 保证雪崩效应，减少哈希冲突
- **零拷贝设计**: 传递张量的原始数据指针（`c->data()`, `a->data()`, `b->data()`）给 InfiniOP，无额外内存拷贝
- **编译期多态**: 通过函数指针表而非虚函数实现分发，避免虚表查找开销，内联优化可能性更高

### 错误处理
- **运行时断言**: `INFINICORE_ASSERT_TENSORS_SAME_DEVICE` 宏在执行前验证设备一致性，失败时抛出 `std::runtime_error`，包含详细的设备类型、函数名、文件行号信息
- **InfiniOP 错误传播**: `INFINICORE_CHECK_ERROR` 宏包装所有 InfiniOP API 调用，检查 `infiniStatus_t` 返回码，失败时转换为描述性字符串并抛出异常
- **异常安全**: 使用 RAII 管理资源（`shared_ptr<Memory>` 管理 workspace），即使抛出异常也能正确释放；descriptor 缓存使用析构器确保清理
- **调试日志**: 使用 spdlog 记录 InfiniOP API 的入口/出口（通过 `SPDLOG_DEBUG` 宏），可通过环境变量 `INFINICORE_LOG_LEVEL=debug` 启用

### 依赖关系
- **外部依赖**: `infiniop.h`（InfiniOP C API），提供底层算子实现和 descriptor 接口
- **模块间依赖**:
  - `infinicore/tensor.hpp`: Tensor 类型定义、shape/strides/dtype/data 访问接口
  - `infinicore/device.hpp`: Device 类，封装设备类型和索引
  - `infinicore/context/context.hpp`: 上下文管理，提供 `setDevice()`, `getDevice()`, `getInfiniopHandle()`, `getStream()`, `allocateMemory()`
  - `infinicore/common/LRUCache.hpp`: LRU 缓存模板，`OpCache` 的底层存储
  - `infinicore/common/hash.hpp`: 哈希组合函数，用于缓存键生成
  - `infinicore/ops/common/op.hpp`, `dispatcher.hpp`, `cache.hpp`: 操作基类和基础设施

### 设计模式
- **Strategy 模式**: `OpDispatcher` 将不同设备的实现算法封装为统一的函数接口，运行时选择策略
- **Registry 模式**: 使用静态初始化（`static bool registered = []() {...}()`）在程序启动时自动注册 InfiniOP 实现到 dispatcher，实现零配置插件化
- **Singleton 模式**: `Mul::dispatcher()` 使用 Meyer's Singleton，保证全局唯一分发器实例
- **RAII 模式**: `OpCache` 析构函数调用 `clear()`，workspace 使用 `shared_ptr` 管理内存生命周期
- **Template Method 模式**: `mul()`（模板方法）定义高层流程（分配 -> 执行），`mul_()`（原语方法）执行实际计算
- **Cache-Aside 模式**: descriptor 缓存采用旁路缓存策略，先查缓存，未命中则创建并写入
