# Rearrange 操作核心实现文档

本模块实现了 InfiniCore 框架中的张量重排（Rearrange）操作，支持跨多种硬件后端（NVIDIA、Cambricon、Ascend、Kunlun 等）的张量数据布局转换。模块采用分层架构设计，通过操作分发器（OpDispatcher）实现硬件无关的统一接口，并利用 LRU 缓存机制优化描述符创建开销。

## 1. 模块结构

- **`rearrange.cc`**: Rearrange 操作的前端实现，提供统一的用户 API 和设备类型分发逻辑
- **`rearrange_infiniop.cc`**: 基于 InfiniOP 库的后端实现，集成描述符缓存和内核调度

## 2. 核心类

### `Rearrange`
- **位置**: `rearrange.cc` / `include/infinicore/ops/rearrange.hpp`
- **主要功能**: 提供设备无关的张量重排操作静态接口，通过操作分发器将调用路由到特定硬件后端
- **核心成员**:
  - `schema`: 函数类型别名 `void (*)(Tensor, Tensor)`，定义后端实现函数签名
  - `dispatcher_`: 静态局部变量，类型为 `OpDispatcher<schema>`，采用单例模式管理设备类型到实现函数的映射表
- **核心方法**:
  - `execute(Tensor y, Tensor x)`: 执行重排操作的主入口
    - 参数验证：确保输入/输出张量位于同一物理设备（使用 `INFINICORE_ASSERT_TENSORS_SAME_DEVICE` 宏）
    - 设备切换：调用 `context::setDevice(y->device())` 设置当前 CUDA/设备上下文
    - 路由分发：通过 `dispatcher().lookup(y->device().getType())` 获取对应硬件后端的函数指针并执行
  - `dispatcher()`: 静态方法，返回单例 `OpDispatcher` 实例的引用
    - 使用 Meyer's Singleton 模式（函数静态局部变量）确保线程安全的延迟初始化
    - `OpDispatcher` 内部维护大小为 `Device::Type::COUNT` 的 `std::array<Fn>` 函数指针表
- **生命周期**: 采用静态单例模式，程序启动时首次调用 `dispatcher()` 初始化，进程结束时自动销毁

### `OpDispatcher<schema>`
- **位置**: `include/infinicore/ops/common/dispatcher.hpp`
- **主要功能**: 类型安全的设备分发器，将设备枚举类型映射到具体的实现函数
- **核心成员**:
  - `table_`: `std::array<Fn, static_cast<size_t>(Device::Type::COUNT)`，固定大小的函数指针数组，支持 CPU、NVIDIA、CAMBRICON、ASCEND、METAX、MOORE、ILUVATAR、KUNLUN、HYGON、QY 等 10 种设备类型
- **核心方法**:
  - `registerDevice(Device::Type, Fn, bool)`: 注册单个设备类型的实现函数，`override_existing` 参数控制是否覆盖已注册函数（默认允许覆盖）
  - `registerDevice(std::initializer_list<Device::Type>, Fn, bool)`: 批量注册多个设备类型
  - `registerAll(Fn, bool)`: 一次性注册所有设备类型到同一实现函数，用于通用后端实现
  - `lookup(Device::Type)`: O(1) 时间复杂度查表返回对应设备类型的函数指针
- **设计模式**: 策略模式（Strategy Pattern） + 表驱动法（Table-Driven Dispatch）

### `OpCache<Key, Value>`
- **位置**: `include/infinicore/ops/common/cache.hpp`
- **主要功能**: 基于设备隔离的 LRU 缓存容器，为每个物理设备维护独立的缓存实例
- **核心成员**:
  - `caches_`: `std::array<CacheVector, static_cast<size_t>(Device::Type::COUNT)>`，二维数组结构，第一维按设备类型索引，第二维按设备索引（支持多 GPU 场景）
  - `capacity_`: 默认容量 100，控制每个 LRU 缓存的最大条目数
  - `destructor_`: `Destructor` 函数对象类型，用于缓存驱逐时释放资源（如销毁 `infiniopRearrangeDescriptor_t`）
- **核心方法**:
  - `getCache(Device)`: 获取指定设备的缓存引用
    - 动态扩容：如果 `device_index` 超出当前 `CacheVector` 大小，自动 `resize` 并用 `BaseCache(capacity_, destructor_)` 填充新增槽位
    - 析构器更新：每次调用时更新缓存实例的析构器（支持运行时修改）
  - `clear()`: 清理所有设备缓存
    - 设备上下文切换：遍历所有设备类型和设备索引，在清理前切换到目标设备上下文（`context::setDevice(target_device)`），确保正确释放设备资源（如 CUDA 内存/描述符）
    - 恢复原设备：清理完成后恢复到调用前的当前设备上下文
- **线程安全**: 非线程安全，外部需通过 `thread_local` 保证线程隔离
- **内存管理**: RAII 机制，析构时自动调用 `clear()` 释放所有设备资源

## 3. API 接口

```cpp
// 创建并返回重排后的新张量（分配新内存）
Tensor rearrange(Tensor x);
// 功能：创建形状与 x 相同的空张量 y，执行 y ← rearrange(x)，返回 y
// 等价于：Tensor y = Tensor::empty(x->shape(), x->dtype(), x->device());
//        rearrange_(y, x);
//        return y;

// 原地重排操作（用户预分配输出张量）
void rearrange_(Tensor y, Tensor x);
// 功能：执行重排操作 y ← rearrange(x)
// 前置条件：y 必须已预分配，shape 与 x 匹配
// 调用路径：Rearrange::execute(y, x) → dispatcher().lookup(device)(y, x) → calculate(y, x)

// Rearrange 类静态方法
class Rearrange {
    using schema = void (*)(Tensor, Tensor);  // 后端函数签名类型别名

    static void execute(Tensor y, Tensor x);  // 主执行入口
    static OpDispatcher<schema>& dispatcher(); // 获取分发器单例
};
```

## 4. 使用示例

```cpp
#include "infinicore/ops/rearrange.hpp"
#include "infinicore/tensor.hpp"

using namespace infinicore;

// 示例 1：创建新张量接收重排结果
void example_rearrange_new_tensor() {
    // 输入张量：形状 [256, 512], 数据类型 float32, 位于 NVIDIA GPU 0
    Tensor x = Tensor::ones({256, 512}, DataType::Float32, Device(Device::NVIDIA, 0));

    // 重排操作（内部分配新张量）
    Tensor y = rearrange(x);

    // y 现在包含重排后的数据，位于同一 GPU 设备
}

// 示例 2：预分配输出张量的原地操作
void example_rearrange_in_place() {
    Tensor x = Tensor::randn({128, 256}, DataType::Float32, Device(Device::NVIDIA, 0));

    // 用户预分配输出张量
    Tensor y = Tensor::empty({128, 256}, DataType::Float32, Device(Device::NVIDIA, 0));

    // 原地重排
    rearrange_(y, x);

    // y 现在包含重排结果
}

// 示例 3：多 GPU 场景
void example_multi_gpu_rearrange() {
    // 在 GPU 0 上创建输入张量
    Tensor x_gpu0 = Tensor::ones({64, 64}, DataType::Float32, Device(Device::NVIDIA, 0));

    // 在 GPU 1 上创建输出张量
    Tensor y_gpu1 = Tensor::empty({64, 64}, DataType::Float32, Device(Device::NVIDIA, 1));

    // 这将抛出异常（设备不匹配）
    try {
        rearrange_(y_gpu1, x_gpu0);
    } catch (const std::runtime_error& e) {
        // 错误信息：Tensor devices mismatch NVIDIA:0 vs NVIDIA:1
    }

    // 正确方式：确保输入输出在同一设备
    Tensor y_gpu0 = Tensor::empty({64, 64}, DataType::Float32, Device(Device::NVIDIA, 0));
    rearrange_(y_gpu0, x_gpu0);  // 成功
}
```

## 5. 实现细节

### 内存管理
- **张量分配**: 使用 `Tensor::empty(shape, dtype, device)` 在指定设备上分配显存（GPU）/内存（CPU）
- **描述符缓存**: 使用 `OpCache` 缓存 `infiniopRearrangeDescriptor_t` 对象，避免重复调用 `infiniopCreateRearrangeDescriptor` 的开销
- **缓存容量**: 默认 100 条目，驱逐策略为 LRU（Least Recently Used），驱逐时调用自定义析构器销毁描述符（`infiniopDestroyRearrangeDescriptor`）
- **资源清理**: `OpCache` 析构时遍历所有设备上下文，确保正确释放 CUDA 资源（避免设备上下文不匹配导致的释放失败）

### 并发性
- **线程局部缓存**: 使用 `thread_local common::OpCache<size_t, infiniopRearrangeDescriptor_t> caches` 声明，每个线程拥有独立缓存实例，避免多线程竞争锁开销
- **线程安全**: `OpDispatcher` 的注册操作通常在程序启动阶段单线程完成（通过静态初始化的 `registered` 变量），后续查表操作为无状态读取，天然线程安全
- **设备上下文隔离**: 执行前调用 `context::setDevice(y->device())` 确保当前线程绑定到正确的 CUDA 设备上下文（多 GPU 环境关键）

### 性能优化
- **零拷贝语义**: 重排操作直接在目标张量内存上写入，用户可预分配输出缓冲区避免临时分配
- **描述符复用**: 基于 Tensor 属性（dtype、shape、strides）生成 64 位哈希键（`hash_combine` 函数），相同属性张量共享同一描述符
  - 哈希算法：Boost 风格的 `hash_combine`，混合常量 `0x9e3779b9`（黄金比例分数的 32 位近似），位混淆操作 `(seed << 6) + (seed >> 2)` 增强雪崩效应
- **O(1) 分发**: `OpDispatcher::lookup` 通过数组索引实现常数时间查表，比虚函数表或 `if-else` 链更高效
- **延迟初始化**: Meyer's Singleton 模式确保首次使用时才构建分发器和缓存表

### 错误处理
- **设备不匹配断言**: `INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x)` 宏在运行时检查输入/输出张量设备一致性，失败时抛出 `std::runtime_error` 并携带详细位置信息（文件名、行号、函数名）
- **InfiniOP 错误传播**: `INFINICORE_CHECK_ERROR` 宏包装所有 `infiniop*` API 调用，将 `infiniStatus_t` 错误码转换为 C++ 异常，附带错误描述字符串
- **异常安全**: RAII 机制确保异常发生时已分配的资源（描述符、缓存）能正确释放

### 依赖关系
- **外部依赖**:
  - `<infiniop.h>`: 底层张量运算库，提供 `infiniopCreateRearrangeDescriptor`, `infiniopRearrange`, `infiniopDestroyRearrangeDescriptor` 等函数
  - `spdlog`: C++ 日志库，用于调试日志（宏展开后仅在 Debug 模式启用）
- **内部依赖**:
  - `infinicore/tensor.hpp`: Tensor 类定义，提供 `shape()`, `dtype()`, `strides()`, `data()`, `device()`, `desc()` 等访问器
  - `infinicore/context/context.hpp`: 设备上下文管理，提供 `getDevice()`, `setDevice()`, `getInfiniopHandle()`, `getStream()` 等函数
  - `infinicore/ops/common/op.hpp`: 操作基类头文件包含
  - `infinicore/common/hash.hpp`: `hash_combine` 模板函数实现
  - `infinicore/common/LRUCache.hpp`: LRU 缓存容器实现（推测基于 `std::unordered_map` + 双向链表）

### 设计模式
- **单例模式（Singleton）**: `Rearrange::dispatcher()` 使用 Meyer's Singleton 确保全局唯一的分发器实例
- **策略模式（Strategy）**: `OpDispatcher` 将设备类型映射到不同的算法策略（CPU、CUDA、Ascend 等后端实现）
- **模板方法模式（Template Method）**: `rearrange()` 定义算法骨架（分配张量 → 调用 `rearrange_` → 返回结果），具体步骤由子类/后端实现
- **RAII（Resource Acquisition Is Initialization）**: `OpCache` 析构函数自动清理所有设备资源，无需手动调用释放函数
- **注册模式（Registry）**: 静态初始化的 `registered` 变量通过 lambda 表达式在程序启动时自动将 `calculate` 函数注册到所有设备类型的分发器中（利用 C++ 静态局部变量初始化时机保证）

### 后端实现细节（rearrange_infiniop.cc）
- **命名空间**: `infinicore::op::rearrange_impl::infiniop`，嵌套命名空间清晰标识实现路径
- **计算函数签名**: `void calculate(Tensor y, Tensor x)`，与 `schema` 类型完全匹配
- **缓存键生成**: `size_t seed = hash_combine(y, x)` 综合两个张量的 dtype、shape、strides 生成哈希值
- **缓存查找**: `auto desc_opt = cache.get(seed)` 返回 `std::optional<infiniopRearrangeDescriptor_t>`
- **描述符创建路径**:
  1. 缓存未命中：调用 `infiniopCreateRearrangeDescriptor(context::getInfiniopHandle(device), &desc, y->desc(), x->desc())`，传入 InfiniOP 库句柄和输入/输出张量描述符
  2. 插入缓存：`cache.put(seed, desc)`
  3. 缓存命中：直接使用缓存描述符
- **内核启动**: `infiniopRearrange(desc, y->data(), x->data(), context::getStream())` 异步执行重排内核，传入目标设备指针和 CUDA 流
- **自动注册**: `static bool registered = []() { Rearrange::dispatcher().registerAll(&calculate, false); return true; }();` 使用函数静态变量和立即求值 lambda，在程序启动时自动注册后端实现到所有设备类型（`registerAll` 参数 `false` 表示不覆盖已有实现）

### 多硬件后端支持
- **当前实现**: `rearrange_infiniop.cc` 通过 `registerAll(&calculate, false)` 将同一实现注册到所有设备类型（CPU、NVIDIA、Ascend、Kunlun 等）
- **假设前提**: InfiniOP 库内部已根据 `infiniopHandle` 的设备类型选择对应的硬件实现，或提供 CPU fallback
- **扩展性**: 如需为特定硬件提供优化实现，可创建 `rearrange_cuda.cc`, `rearrange_ascend.cc` 等文件，定义专用 `calculate` 函数并通过 `registerDevice(Device::Type::NVIDIA, &calculate_cuda)` 注册，覆盖 InfiniOP 通用实现
