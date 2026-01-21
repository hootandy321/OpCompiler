# `AddRMSNorm` 操作核心实现文档

该模块实现了融合的 Add 和 RMS 归一化操作，这是 Transformer 架构（特别是 LLaMA、GPT 等模型）中的核心计算原语。该融合操作将元素级加法与 RMS 归一化合并为单个 kernel，显著减少了内存访问开销并提升了计算效率。

## 1. 模块结构

- **`add_rms_norm.cc`**: 操作的公共接口层，提供类型安全的 API 和设备调度逻辑
- **`add_rms_norm_infiniop.cc`**: 基于 InfiniOp 后端的具体实现，包含描述符缓存机制和 kernel 调度

## 2. 核心类与组件

### `AddRMSNorm`
- **位置**: `add_rms_norm.cc` 和 `include/infinicore/ops/add_rms_norm.hpp`
- **主要功能**: 提供静态调度器模式的多设备操作分发器，根据输入张量的设备类型动态路由到对应的实现
- **核心成员**:
  - `schema`: 函数指针类型定义，签名为 `void (*)(Tensor, Tensor, Tensor, Tensor, Tensor, float)`，表示 5 个张量参数和 1 个 epsilon 参数的函数
  - `dispatcher()`: 静态方法，返回 `OpDispatcher<schema>` 的单例引用，使用 Meyers Singleton 实现线程安全懒加载
- **核心方法**:
  - `execute(Tensor y, Tensor residual_out, Tensor a, Tensor b, Tensor weight, float epsilon)`:
    - 执行融合操作的入口点
    - **前置断言**: 通过 `INFINICORE_ASSERT_TENSORS_SAME_DEVICE` 宏确保所有输入张量位于同一设备
    - **设备设置**: 调用 `infinicore::context::setDevice(y->device())` 设置当前 CUDA/设备上下文
    - **动态分发**: 根据 `y->device().getType()` 查找调度表并调用对应设备的实现函数
  - **生命周期**: 单例模式，静态 `dispatcher_` 变量在首次调用 `dispatcher()` 时初始化，程序结束时自动销毁

### `OpDispatcher<schema>`
- **位置**: `include/infinicore/ops/common/dispatcher.hpp`
- **主要功能**: 类型擦除的操作分发表，将设备类型枚举映射到具体的函数指针
- **数据结构**: `std::array<Fn, static_cast<size_t>(Device::Type::COUNT)>`，固定大小的函数指针数组，编译期确定大小
- **核心方法**:
  - `registerDevice(Device::Type, Fn, bool)`: 注册单个设备类型的实现，支持覆盖控制
  - `registerAll(Fn, bool)`: 批量注册所有设备类型，使用 `Device::Type::COUNT` 枚举遍历
  - `lookup(Device::Type)`: O(1) 时间复杂度的数组索引访问，返回对应设备的函数指针
- **设计模式**: Strategy Pattern，运行时根据设备类型选择算法实现

### `OpCache<Key, Value>`
- **位置**: `include/infinicore/ops/common/cache.hpp` 和 `add_rms_norm_infiniop.cc`
- **主要功能**: 多设备感知的 LRU 缓存，存储 InfiniOp 描述符以避免重复创建开销
- **核心成员**:
  - `caches_`: 二维数组结构，第一维为设备类型（CUDA、CPU 等），第二维为设备索引（多 GPU 场景）
  - `capacity_`: 缓存容量上限，默认 100 个条目
  - `destructor_`: 自定义析构函数，用于正确释放 InfiniOp 描述符（调用 `infiniopDestroyAddRMSNormDescriptor`）
- **核心方法**:
  - `getCache(Device)`: 返回对应设备的 `LRUCache` 引用，自动扩容以支持任意设备索引
  - `clear()`: 清理所有缓存，在清理前切换到对应设备上下文以正确释放 GPU 资源
- **线程本地存储**: 使用 `thread_local` 关键字，每个线程拥有独立的缓存实例，避免多线程竞争

### `calculate` 函数（InfiniOp 实现）
- **位置**: `add_rms_norm_infiniop.cc` 命名空间 `infinicore::op::add_rms_norm_impl::infiniop`
- **主要功能**: 封装 InfiniOp API 调用的具体实现，包含描述符创建、缓存查询、工作空间分配和 kernel 启动
- **核心算法流程**:
  1. **哈希计算**: 调用 `hash_combine(y, residual_out, a, b, weight, epsilon)` 生成 64 位种子值
     - 哈希策略：对每个张量的 dtype、shape 的每个维度、strides 的每个值进行递归哈希组合
     - 哈希算法：Boost 风格的 `hash_combine`，使用黄金比例常数 `0x9e3779b9` 和位移混合
  2. **缓存查询**: 在 `thread_local OpCache` 中查找是否存在已创建的描述符
  3. **描述符创建（缓存未命中时）**:
     - 调用 `infiniopCreateAddRMSNormDescriptor` 创建 InfiniOp 描述符
     - 传入参数：InfiniOp 句柄、输出张量 y、输入张量 a/b/weight、epsilon 值、残差输出张量
     - 错误检查：使用 `INFINICORE_CHECK_ERROR` 宏包装，失败时抛出包含错误码的 `std::runtime_error`
  4. **工作空间分配**: 调用 `infiniopGetAddRMSNormWorkspaceSize` 获取所需工作空间大小，通过 `context::allocateMemory` 分配 GPU 内存
  5. **Kernel 启动**: 调用 `infiniopAddRMSNorm` 执行计算，传入描述符、工作空间指针、所有张量的数据指针和 CUDA 流

## 3. API 接口

```cpp
// 高级 API：自动分配输出张量并返回结果对
std::pair<Tensor, Tensor> add_rms_norm(
    Tensor a,              // 第一个输入张量（通常是 hidden states）
    Tensor b,              // 第二个输入张量（通常是残差连接）
    Tensor weight,         // 归一化权重（通常是 gamma 缩放参数）
    float epsilon = 1e-5f  // 数值稳定项，防止除零
);
// 返回值：pair.first 是 RMS 归一化结果，pair.second 是加法结果（用于后续残差连接）

// 低级 API：预分配输出张量的就地操作版本
void add_rms_norm_(
    Tensor y,              // 预分配的输出张量（RMSNorm 结果）
    Tensor residual_out,   // 预分配的残差输出张量（Add 结果）
    Tensor a,              // 第一个输入张量
    Tensor b,              // 第二个输入张量
    Tensor weight,         // 归一化权重
    float epsilon = 1e-5f  // 数值稳定项
);
// 语义：执行 y, residual_out = AddRMSNorm(a, b, weight, epsilon)

// 内部调度接口（通常不直接调用）
void AddRMSNorm::execute(
    Tensor y, Tensor residual_out, Tensor a,
    Tensor b, Tensor weight, float epsilon
);
```

## 4. 使用示例

```cpp
#include "infinicore/ops/add_rms_norm.hpp"
using namespace infinicore::op;

// 场景：在 Transformer 层中使用融合的 Add + RMSNorm 操作
void transformer_layer(Tensor hidden_states, Tensor residual, Tensor rms_weight) {
    // 方式 1：使用高级 API（推荐用于大多数场景）
    auto [normalized, new_residual] = add_rms_norm(
        hidden_states,  // 输入：当前层的 hidden states
        residual,       // 输入：前一层的残差
        rms_weight,     // RMSNorm 的可学习参数
        1e-5f          // epsilon，使用默认值
    );

    // normalized 可用于后续的注意力或 FFN 计算
    // new_residual 保存了 a + b 的结果，可用于下一层的残差连接

    // 方式 2：使用低级 API（用于内存敏感场景或需要预分配的情况）
    Tensor y = Tensor::empty(hidden_states->shape(), hidden_states->dtype(), hidden_states->device());
    Tensor residual_out = Tensor::empty(hidden_states->shape(), hidden_states->dtype(), hidden_states->device());

    add_rms_norm_(y, residual_out, hidden_states, residual, rms_weight, 1e-5f);

    // 后续使用 y 和 residual_out...
}
```

## 5. 实现细节

### 内存管理
- **工作空间分配**: 使用 InfiniOp 的动态工作空间机制，通过 `infiniopGetAddRMSNormWorkspaceSize` 查询需求并分配临时 GPU 内存
- **张量内存**: 输出张量通过 `Tensor::empty` 显式分配，使用 RAII 管理生命周期
- **缓存资源**: 描述符缓存的析构函数正确调用 `infiniopDestroyAddRMSNormDescriptor` 释放 InfiniOp 内部资源

### 并发与线程安全
- **调度器**: 使用 Meyers Singleton（C++11 保证线程安全）实现的静态局部变量
- **缓存**: `thread_local` 存储期限，每个工作线程维护独立的缓存实例，消除锁竞争
- **设备上下文**: 执行前通过 `context::setDevice` 切换 CUDA 设备，确保多 GPU 环境下正确性

### 性能优化
- **操作融合**: 将 Add 和 RMSNorm 合并为单个 kernel，减少全局内存访问次数（从 3 次降至 2 次：读写输入、读写输出）
- **描述符缓存**: 基于 LRU 策略缓存 InfiniOp 描述符，避免重复创建的 CPU 开销和 GPU kernel 启动成本
- **哈希优化**: 使用编译期优化的 `hash_combine` 模板，递归组合张量元数据（dtype、shape、stride）生成缓存键
- **零拷贝**: 输出张量通过指针直接传递给底层 kernel，无额外内存拷贝

### 错误处理
- **设备一致性检查**: `INFINICORE_ASSERT_TENSORS_SAME_DEVICE` 宏在运行时验证所有输入张量位于同一物理设备
- **InfiniOp 错误传播**: `INFINICORE_CHECK_ERROR` 宏包装所有 InfiniOp API 调用，失败时抛出包含错误码和描述的异常
- **异常安全**: 使用 RAII 管理资源，异常发生时自动释放已分配的 GPU 内存和描述符

### 依赖关系
- **外部依赖**: `infiniop.h`（InfiniOp C API），提供平台无关的算子描述符接口
- **内部依赖**:
  - `infinicore/context/context.hpp`: 设备管理、内存分配、CUDA 流获取
  - `infinicore/tensor.hpp`: 张量抽象，封装数据指针、形状、数据类型和设备信息
  - `infinicore/common/hash.hpp`: 哈希组合工具，用于生成缓存键
  - `infinicore/ops/common/cache.hpp`: 多设备感知的 LRU 缓存
  - `infinicore/ops/common/dispatcher.hpp`: 设备类型到函数指针的分发表

### 设计模式
- **Singleton Pattern**: `AddRMSNorm::dispatcher()` 使用 Meyers Singleton
- **Strategy Pattern**: `OpDispatcher` 在运行时选择设备特定的实现
- **RAII**: `OpCache` 的析构函数自动清理所有缓存的描述符
- **Template Method**: `execute` 定义算法骨架，具体实现由注册的函数指针提供
- **LRU Cache**: 描述符缓存使用最近最少使用淘汰策略
