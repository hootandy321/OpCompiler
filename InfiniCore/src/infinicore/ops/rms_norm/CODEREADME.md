# RMSNorm 操作核心实现文档

RMSNorm (Root Mean Square Normalization) 操作实现模块，提供高效的张量归一化功能。该模块通过 InfiniOP 后端库实现，支持多设备类型，具备智能描述符缓存机制以优化性能。

## 1. 模块结构

- **`rms_norm.hpp`**: RMSNorm 操作的公共接口定义，声明了 `RMSNorm` 类以及便利函数 `rms_norm` 和 `rms_norm_`
- **`rms_norm.cc`**: RMSNorm 操作的核心调度逻辑，实现设备分发和 API 入口点
- **`rms_norm_infiniop.cc`**: 基于 InfiniOP 库的具体实现，包含描述符缓存和计算执行逻辑

## 2. 核心类

### `RMSNorm`
- **位置**: `rms_norm.hpp`, `rms_norm.cc`
- **主要功能**: 提供静态接口执行 RMSNorm 操作，根据设备类型动态分发到对应的后端实现
- **关键成员**:
  - `schema`: 函数签名类型定义 `void (*)(Tensor, Tensor, Tensor, float)`
  - `dispatcher()`: 静态方法，返回 `OpDispatcher<schema>` 单例引用，用于设备类型到实现函数的映射
- **核心方法**:
  - `execute(Tensor y, Tensor x, Tensor weight, float epsilon)`: 执行 RMSNorm 操作的主入口
    - 验证输入张量 `y`, `x`, `weight` 必须位于同一设备
    - 根据输出张量 `y` 的设备类型查找对应的实现函数
    - 调用查找到的函数执行实际计算
    - 时间复杂度: O(1) 分发开销，实际计算复杂度取决于后端实现
  - `dispatcher()`: 返回单例 `OpDispatcher<schema>` 实例
    - 使用 Meyer's Singleton 模式（函数内静态变量）
    - 线程安全的 C++11 保证
    - 维护设备类型到函数指针的查找表
- **生命周期**:
  - 采用静态单例模式，`dispatcher()` 在首次调用时构造
  - 程序生命周期内持久存在
  - 无需显式销毁，由运行时自动清理

## 3. API 接口

```cpp
namespace infinicore::op {

// RMSNorm 操作执行
void RMSNorm::execute(Tensor y, Tensor x, Tensor weight, float epsilon = 1e-5f);
// 执行 RMSNorm: y = rms_norm(x, weight, epsilon)
// 参数:
//   y - 输出张量，与 x 同形状同设备
//   x - 输入张量
//   weight - 归一化权重参数
//   epsilon - 数值稳定性的小常数，防止除零

// 便利函数：分配输出并执行
Tensor rms_norm(Tensor x, Tensor weight, float epsilon = 1e-5f);
// 创建并返回与 x 同形状的输出张量 y，执行 rms_norm_(y, x, weight, epsilon)

// 原地执行版本
void rms_norm_(Tensor y, Tensor x, Tensor weight, float epsilon = 1e-5f);
// 等价于 RMSNorm::execute(y, x, weight, epsilon)

} // namespace infinicore::op
```

## 4. 使用示例

```cpp
#include "infinicore/ops/rms_norm.hpp"

using namespace infinicore;
using namespace infinicore::op;

// 示例：在 CUDA 设备上执行 RMSNorm
void example_rmsnorm() {
    // 假设已有输入张量和权重
    Tensor x = Tensor::random({32, 128, 768}, DataType::FLOAT32, Device::cuda(0));
    Tensor weight = Tensor::random({768}, DataType::FLOAT32, Device::cuda(0));
    float epsilon = 1e-5f;

    // 方式1：使用便利函数（自动分配输出）
    Tensor y = rms_norm(x, weight, epsilon);
    // y 现在包含 RMSNorm 结果

    // 方式2：使用原地执行版本
    Tensor y_preallocated = Tensor::empty(x->shape(), x->dtype(), x->device());
    rms_norm_(y_preallocated, x, weight, epsilon);
    // y_preallocated 现在包含结果

    // 方式3：直接调用类接口
    Tensor y_another = Tensor::empty(x->shape(), x->dtype(), x->device());
    RMSNorm::execute(y_another, x, weight, epsilon);
}
```

## 5. 实现细节

### 5.1 设备分发机制 (Device Dispatch)

**模式**: 策略模式 (Strategy Pattern) + 函数指针查找表

- **`OpDispatcher<schema>`**: 泛型分发器，维护设备类型到函数指针的映射
  - 使用 `std::array<Fn, (size_t)Device::Type::COUNT>` 作为查找表
  - 提供 `registerDevice()`, `registerAll()`, `lookup()` 方法
  - O(1) 时间复杂度的设备类型查找
- **注册机制**: 使用静态初始化器在程序启动时自动注册实现
  ```cpp
  static bool registered = []() {
      RMSNorm::dispatcher().registerAll(&calculate, false);
      return true;
  }();
  ```
- **线程安全**: `dispatcher()` 使用 Meyer's Singleton，C++11 保证线程安全初始化

### 5.2 描述符缓存 (Descriptor Cache)

**模式**: LRU 缓存 + 线程局部存储

- **缓存结构**:
  ```cpp
  thread_local common::OpCache<size_t, infiniopRMSNormDescriptor_t> caches(
      100, // 容量：100个描述符
      [](infiniopRMSNormDescriptor_t &desc) {
          if (desc != nullptr) {
              INFINICORE_CHECK_ERROR(infiniopDestroyRMSNormDescriptor(desc));
              desc = nullptr;
          }
      });
  ```

- **哈希键生成**: 使用 `hash_combine()` 组合张量属性和参数
  - 输入张量 `x`: 数据类型、形状、步长
  - 权重张量 `weight`: 数据类型、形状、步长
  - 输出张量 `y`: 数据类型、形状、步长
  - 超参数 `epsilon`: 浮点数值
  - 哈希算法: Boost 风格的 `hash_combine`，使用黄金比例常数 `0x9e3779b9`

- **缓存策略**:
  - LRU (Least Recently Used) 淘汰策略
  - 容量限制：100 个描述符 per 设备
  - 线程局部存储：每个线程拥有独立缓存，避免锁竞争
  - 自定义析构器：缓存条目被淘汰时自动调用 `infiniopDestroyRMSNormDescriptor()`

- **多设备支持**: `OpCache` 维护二维数组结构
  ```cpp
  std::array<std::vector<LRUCache<size_t, Descriptor>>,
             static_cast<size_t>(Device::Type::COUNT)> caches_;
  ```
  - 第一维：设备类型 (CUDA, CPU, Kunlun 等)
  - 第二维：同类型设备的索引（支持多 GPU 场景）

### 5.3 内存管理

- **工作空间 (Workspace)**:
  - 通过 `infiniopGetRMSNormWorkspaceSize()` 查询所需大小
  - 使用 `context::allocateMemory()` 分配设备内存
  - 每次调用重新分配，未缓存（假设 InfiniOP 内部有优化）

- **内存生命周期**:
  - Workspace 内存：在 `calculate()` 函数栈上使用 `shared_ptr` 管理
  - 函数返回后自动释放
  - 异常安全：利用 RAII 保证异常时正确释放

### 5.4 错误处理

- **宏机制**:
  - `INFINICORE_CHECK_ERROR(call)`: 检查 InfiniOP API 调用返回值
    - 失败时抛出 `std::runtime_error`
    - 包含错误码和人类可读的错误信息
    - 使用 `infini_status_string()` 转换错误码

  - `INFINICORE_ASSERT_TENSORS_SAME_DEVICE(FIRST, ...)`: 验证张量设备一致性
    - 在编译期展开为循环检查
    - 生成详细错误信息（设备类型、函数名、文件位置、行号）

- **异常传播**: 所有异常向调用者传播，不在内部捕获

### 5.5 并发模型

- **线程局部缓存**: `thread_local` 关键字确保
  - 每个线程拥有独立的 `caches` 实例
  - 无需加锁即可并发访问
  - 适用于多线程推理场景

- **CUDA 流 (Stream)**:
  - 使用 `context::getStream()` 获取当前流
  - 允许异步执行和流并行
  - 多操作可在同一流中流水线执行

- **设备上下文切换**:
  - 调用前: `infinicore::context::setDevice(y->device())`
  - 确保在正确的设备上下文中执行

### 5.6 性能优化

1. **描述符缓存**: 避免重复调用 `infiniopCreateRMSNormDescriptor()`
   - 该函数可能涉及内核编译、启发式优化等昂贵操作
   - 对于相同的张量形状/类型组合，复用描述符

2. **哈希计算**: `hash_combine()` 使用位移和异或，快速生成唯一键
   - 算法复杂度: O(n) where n 是张量维度数
   - 通常 n ≤ 4，开销可忽略

3. **零拷贝分发**:
   - `OpDispatcher::lookup()` 返回函数指针
   - 无虚函数开销
   - 无间接调用开销（现代 CPU 的分支预测可优化）

### 5.7 依赖关系

**外部依赖**:
- `<infiniop.h>`: InfiniOP 后端库，提供实际的 RMSNorm 实现
  - `infiniopCreateRMSNormDescriptor()`: 创建操作描述符
  - `infiniopGetRMSNormWorkspaceSize()`: 查询工作空间大小
  - `infiniopRMSNorm()`: 执行归一化计算
  - `infiniopDestroyRMSNormDescriptor()`: 销毁描述符

**内部依赖**:
- `infinicore/context/context.hpp`: 设备和流管理
- `infinicore/ops/common/op.hpp`: 操作基类定义
- `infinicore/ops/common/dispatcher.hpp`: 设备分发器
- `infinicore/ops/common/cache.hpp`: 操作缓存包装器
- `infinicore/common/hash.hpp`: 哈希组合工具
- `infinicore/common/LRUCache.hpp`: LRU 缓存实现

### 5.8 设计模式总结

1. **Singleton 模式**: `RMSNorm::dispatcher()` 单例分发器
2. **Strategy 模式**: 设备类型到实现函数的映射
3. **Template Method 模式**: `execute()` 定义算法骨架，具体实现由策略提供
4. **RAII 模式**: `shared_ptr<Memory>` 管理工作空间内存
5. **LRU Cache 模式**: 描述符缓存优化重复操作
6. **Thread-Local Storage 模式**: 线程局部缓存避免锁竞争

### 5.9 数学定义

RMSNorm 实现以下数学操作：

```
给定输入张量 x ∈ ℝ^(d1×d2×...×dn)
权重参数 weight ∈ ℝ^D (通常 D 为最后一个维度)

对于每个样本 i (沿最后维度前所有维度):

1. 计算均方根:
   rms_i = sqrt( (1/D) * Σ(j=1 to D) x[i,j]^2 + epsilon )

2. 归一化并缩放:
   y[i,j] = (x[i,j] / rms_i) * weight[j]

其中 epsilon 是数值稳定性常数 (默认 1e-5)
```

该归一化技术在 Transformer 模型（如 LLaMA, GPT-NeoX）中广泛应用，相比 LayerNorm 更简化且效果相当。
