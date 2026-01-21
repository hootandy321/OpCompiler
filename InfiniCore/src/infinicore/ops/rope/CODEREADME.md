# RoPE (Rotary Position Embedding) 操作实现核心文档

本模块实现了 InfiniCore 框架中的 RoPE（旋转位置编码）操作，这是一种在 Transformer 模型中广泛使用的位置编码技术，用于为自注意力机制注入位置信息。该模块提供了基于 InfiniOP 后端的高性能实现，支持 GPT-J 和 GPT-NeoX 两种算法变体。

## 1. 模块结构

- **`rope.cc`**: RoPE 操作的前端实现，包含 OpDispatcher 调度逻辑和公共 API
- **`rope_infiniop.cc`**: 基于 InfiniOP 后端的具体实现，包含描述符缓存和内核调用逻辑

## 2. 核心类与组件

### `infinicore::op::RoPE`
- **位置**: `rope.cc`, `include/infinicore/ops/rope.hpp`
- **主要功能**: 提供 RoPE 操作的静态接口，通过 OpDispatcher 实现跨设备分发
- **核心成员**:
  - `dispatcher()`: 静态方法，返回 `OpDispatcher<schema>` 单例，用于设备类型到实现函数的映射
- **核心方法**:
  - `execute(Tensor x_out, const Tensor &x, const Tensor &pos, const Tensor &sin_table, const Tensor &cos_table, Algo algo)`: 执行 RoPE 操作的主入口
    - **算法流程**:
      1. 使用 `INFINICORE_ASSERT_TENSORS_SAME_DEVICE` 验证所有输入张量在同一设备
      2. 设置当前设备上下文：`context::setDevice(x_out->device())`
      3. 从调度器查询对应设备类型的实现函数
      4. 如果未找到实现，抛出 `std::runtime_error` 异常
      5. 调用底层实现函数完成计算
    - **时间复杂度**: O(1) 用于查询和分发，实际复杂度取决于底层实现
  - `dispatcher()`: 返回静态 `OpDispatcher` 实例（使用函数局部静态初始化模式）
- **生命周期**:
  - **单例模式**: `dispatcher()` 方法使用 `static` 局部变量，保证全局唯一实例
  - **初始化**: 首次调用 `dispatcher()` 时构造 `OpDispatcher<schema>` 实例
  - **注册时机**: 在 `rope_infiniop.cc` 中通过静态变量 `registered` 的初始化完成注册

### `infinicore::op::rope_impl::infiniop::calculate`
- **位置**: `rope_infiniop.cc`
- **主要功能**: 基于 InfiniOP 库的 RoPE 操作具体实现
- **核心逻辑**:
  1. **算法转换**: 将 `infinicore::nn::RoPE::Algo` 枚举转换为 `infiniopRoPEAlgo_t` 类型
     - `GPT_J` → `INFINIOP_ROPE_ALGO_GPT_J`
     - `GPT_NEOX` → `INFINIOP_ROPE_ALGO_GPT_NEOX`
  2. **缓存键生成**: 使用 `hash_combine` 生成描述符缓存键
     ```cpp
     size_t key = hash_combine(x_out, x, pos, sin_cache, cos_cache);
     hash_combine(key, std::hash<int>()(static_cast<int>(infiniop_algo)));
     ```
  3. **描述符缓存查询**:
     - 从 thread_local `OpCache` 获取当前设备的缓存实例
     - 尝试从缓存获取已创建的 `infiniopRoPEDescriptor_t`
  4. **描述符创建（缓存未命中时）**:
     - 调用 `infiniopCreateRoPEDescriptor` 创建描述符
     - 参数包括：输出张量、输入张量、位置张量、sin/cos 缓存张量的描述符，以及算法类型
     - 将新创建的描述符存入缓存
  5. **工作空间分配**:
     - 调用 `infiniopGetRoPEWorkspaceSize` 获取所需工作空间大小
     - 使用 `context::allocateMemory` 分配设备内存
  6. **内核执行**:
     - 调用 `infiniopRoPE` 执行实际的 RoPE 计算
     - 传入工作空间指针和大小、输入/输出数据指针、以及 CUDA/stream 上下文

### `thread_local OpCache<size_t, infiniopRoPEDescriptor_t> caches`
- **位置**: `rope_infiniop.cc`
- **主要功能**: 跨线程隔离的 InfiniOP 描述符缓存
- **类型定义**: `OpCache<size_t, infiniopRoPEDescriptor_t>`
- **容量**: 100 个描述符条目
- **析构函数**: 使用 lambda 表达式自定义资源释放逻辑
  ```cpp
  [](infiniopRoPEDescriptor_t &desc) {
      if (desc != nullptr) {
          INFINICORE_CHECK_ERROR(infiniopDestroyRoPEDescriptor(desc));
          desc = nullptr;
      }
  }
  ```
- **线程安全性**: 使用 `thread_local` 关键字，每个线程拥有独立的缓存实例，避免多线程竞争

## 3. API 接口

### 公共 API

```cpp
// 高级 API：自动分配输出张量
Tensor rope(const Tensor &x, const Tensor &pos, const Tensor &sin_table, const Tensor &cos_table, infinicore::nn::RoPE::Algo algo);
/*
 功能：对输入张量 x 应用 RoPE 变换，自动创建输出张量
 参数：
   - x: 输入张量，形状为 (..., head_dim)，其中 head_dim 必须是偶数
   - pos: 位置 ID 张量，形状通常为 [seq_len] 或 [batch, seq_len]
   - sin_table: 预计算的 sin 值查找表
   - cos_table: 预计算的 cos 值查找表
   - algo: RoPE 算法类型（GPT_J 或 GPT_NEOX）
 返回：包含旋转后结果的新张量
*/

// 低级 API：预分配输出张量
void rope_(Tensor x_out, const Tensor &x, const Tensor &pos, const Tensor &sin_table, const Tensor &cos_table, infinicore::nn::RoPE::Algo algo);
/*
 功能：对输入张量 x 应用 RoPE 变换，写入预分配的输出张量
 参数：
   - x_out: 预分配的输出张量
   - x: 输入张量
   - pos: 位置 ID 张量
   - sin_table: sin 查找表
   - cos_table: cos 查找表
   - algo: RoPE 算法类型
 返回：无（结果写入 x_out）
*/

// 内部调度接口
void RoPE::execute(Tensor x_out, const Tensor &x, const Tensor &pos, const Tensor &sin_table, const Tensor &cos_table, infinicore::nn::RoPE::Algo algo);
/*
 功能：RoPE 操作的内部调度入口，根据设备类型分发到具体实现
 实现：
   1. 验证所有张量在同一设备
   2. 设置设备上下文
   3. 查询并调用设备特定的实现函数
*/
```

### 依赖的枚举类型（infinicore::nn::RoPE::Algo）

```cpp
enum class Algo {
    GPT_J = 0,    // GPT-J 风格：交替使用偶数和奇数维度
    GPT_NEOX = 1, // GPT-NeoX 风格：前半维度用于 sin，后半维度用于 cos
};
```

## 4. 使用示例

### 示例 1：基本用法（使用高级 API）

```cpp
#include "infinicore/ops/rope.hpp"
#include "infinicore/nn/rope.hpp"

using namespace infinicore;

// 1. 创建 RoPE 模块（自动预计算 sin/cos 表）
size_t head_dim = 128;
size_t max_seq_len = 2048;
auto rope_module = std::make_shared<nn::RoPE>(
    head_dim,
    max_seq_len,
    10000.0,           // theta
    nn::RoPE::Algo::GPT_J,
    DataType::F32,
    Device::cuda(0)
);

// 2. 准备输入张量
// 形状: [batch=2, num_heads=8, seq_len=512, head_dim=128]
Shape shape = {2, 8, 512, 128};
Tensor x = Tensor::randn(shape, DataType::F32, Device::cuda(0));

// 3. 准备位置 ID
// 形状: [seq_len=512]
Tensor pos = Tensor::arange(0, 512, DataType::I32, Device::cuda(0));

// 4. 使用 RoPE 模块的前向方法（推荐方式）
Tensor x_rotated = rope_module->forward(x, pos);

// 或者直接调用操作级 API（需要手动提供 sin/cos 表）
Tensor sin_table = rope_module->sin_cache();
Tensor cos_table = rope_module->cos_cache();
Tensor x_rotated2 = op::rope(x, pos, sin_table, cos_table, nn::RoPE::Algo::GPT_J);
```

### 示例 2：原地计算（避免额外内存分配）

```cpp
#include "infinicore/ops/rope.hpp"

using namespace infinicore;

// 预分配输出张量
Shape shape = {2, 8, 512, 128};
Tensor x = Tensor::randn(shape, DataType::F32, Device::cuda(0));
Tensor x_out = Tensor::empty(shape, DataType::F32, Device::cuda(0));

Tensor pos = Tensor::arange(0, 512, DataType::I32, Device::cuda(0));
Tensor sin_table = ...; // 从 RoPE 模块获取或手动创建
Tensor cos_table = ...;

// 原地计算版本（写入预分配的张量）
op::rope_(x_out, x, pos, sin_table, cos_table, nn::RoPE::Algo::GPT_J);

// x_out 现在包含旋转后的结果
```

### 示例 3：使用 GPT-NeoX 算法

```cpp
#include "infinicore/ops/rope.hpp"

using namespace infinicore;

// 创建使用 GPT-NeoX 算法的 RoPE 模块
auto rope_module = std::make_shared<nn::RoPE>(
    128,                     // head_dim
    2048,                    // max_seq_len
    10000.0,                 // theta
    nn::RoPE::Algo::GPT_NEOX, // 使用 GPT-NeoX 算法
    DataType::F32,
    Device::cuda(0)
);

Tensor x = Tensor::randn({2, 8, 512, 128}, DataType::F32, Device::cuda(0));
Tensor pos = Tensor::arange(0, 512, DataType::I32, Device::cuda(0));

// GPT-NeoX 算法会应用不同的旋转模式
Tensor x_rotated = rope_module->forward(x, pos);
```

### 示例 4：在注意力机制中使用

```cpp
#include "infinicore/ops/rope.hpp"
#include "infinicore/nn/rope.hpp"

using namespace infinicore;

// 典型的 Transformer 注意力机制中的 RoPE 使用场景
void attention_with_rope(const Tensor& query, const Tensor& key, const Tensor& pos) {
    auto rope = std::make_shared<nn::RoPE>(
        query->shape().back(),  // head_dim
        2048,                    // max_seq_len
        nn::RoPE::Algo::GPT_J,
        DataType::F32,
        Device::cuda(0)
    );

    // 分别对 query 和 key 应用 RoPE
    Tensor q_rotated = rope->forward(query, pos);
    Tensor k_rotated = rope->forward(key, pos);

    // 继续进行注意力计算...
    // Tensor scores = matmul(q_rotated, transpose(k_rotated));
}
```

## 5. 实现细节

### 内存管理

- **描述符缓存策略**:
  - 使用 `OpCache<size_t, infiniopRoPEDescriptor_t>` 缓存 InfiniOP 描述符
  - 基于 LRU（Least Recently Used）淘汰策略，容量为 100 个条目
  - **缓存键生成**: 使用 `hash_combine` 组合所有输入张量的描述符信息和算法类型
    - 包含张量的数据类型、形状、步长
    - 包含算法类型的哈希值
  - **线程局部存储**: 使用 `thread_local` 关键字，避免多线程环境下的锁竞争

- **工作空间管理**:
  - 每次调用时动态查询所需工作空间大小：`infiniopGetRoPEWorkspaceSize`
  - 使用 `context::allocateMemory` 分配设备内存
  - 工作空间在函数返回后自动释放（通过 `shared_ptr` 管理生命周期）

- **张量数据管理**:
  - 输出张量可以由用户预分配（`rope_`）或自动创建（`rope`）
  - InfiniOP 内核负责从输入张量读取并写入输出张量，内部处理数据复制

### 并发性

- **线程安全**:
  - **OpDispatcher**: 使用静态局部变量，C++11 保证线程安全的初始化
  - **OpCache**: 使用 `thread_local` 关键字，每个线程拥有独立的缓存实例
  - **设备上下文**: 通过 `context::setDevice` 和 `context::getDevice` 管理每线程的设备状态
  - **无显式锁**: 设计避免了多线程竞争，无需使用互斥锁

- **CUDA 流支持**:
  - 使用 `context::getStream()` 获取当前 CUDA 流
  - 允许同一设备上的多个操作在不同流中并发执行

### 性能优化

- **描述符缓存**:
  - 避免重复创建昂贵的 InfiniOP 描述符
  - 对于相同的输入形状和数据类型，描述符可复用
  - LRU 缓存限制内存使用，防止缓存过度增长

- **算法类型**:
  - 支持 GPT-J 和 GPT-NeoX 两种变体
  - 算法选择影响计算模式而非性能（复杂度相同）
  - GPT-J: 交替偶数/奇数维度的旋转
  - GPT-NeoX: 前半维度用 sin，后半维度用 cos

- **设备无关分发**:
  - 使用 OpDispatcher 实现零成本的设备类型分发
  - 编译期确定的设备类型可以内联优化
  - 运行期查询使用数组索引，复杂度 O(1)

### 错误处理

- **异常机制**:
  - 使用 `std::runtime_error` 抛出异常
  - `INFINICORE_CHECK_ERROR` 宏封装 InfiniOP 错误码检查
  - `INFINICORE_ASSERT_TENSORS_SAME_DEVICE` 宏验证张量设备一致性

- **错误消息**:
  - 包含失败的操作名称
  - 包含 InfiniOP 错误码的字符串描述
  - 设备不匹配错误包含详细的张量设备信息

- **断言**:
  - 所有张量必须在同一设备上
  - 设备类型必须有注册的实现
  - 算法类型必须是支持的值

### 依赖关系

- **内部依赖**:
  - `infinicore/context/context.hpp`: 设备上下文管理（`setDevice`, `getDevice`, `getStream`, `allocateMemory`）
  - `infinicore/ops/common/op.hpp`: 操作基础设施
  - `infinicore/ops/common/cache.hpp`: 描述符缓存
  - `infinicore/ops/common/dispatcher.hpp`: 设备类型分发
  - `infinicore/common/hash.hpp`: 缓存键生成

- **外部依赖**:
  - `infiniop.h`: InfiniOP 库接口
    - `infiniopCreateRoPEDescriptor`: 创建 RoPE 操作描述符
    - `infiniopGetRoPEWorkspaceSize`: 查询工作空间大小
    - `infiniopRoPE`: 执行 RoPE 计算
    - `infiniopDestroyRoPEDescriptor`: 销毁描述符
  - `spdlog`: 日志记录（通过 `utils.hpp`）

- **设计模式**:
  - **Strategy Pattern**: OpDispatcher 根据 Device::Type 选择不同的实现策略
  - **Singleton Pattern**: dispatcher() 使用静态局部变量保证全局唯一
  - **Template Method**: RoPE::execute 定义算法骨架，具体实现由注册函数完成
  - **Cache-Aside Pattern**: 描述符缓存采用 Cache-Aside 模式（先查缓存，未命中则创建并缓存）

### 算法说明

**RoPE（旋转位置编码）数学原理**:

RoPE 通过旋转矩阵将位置信息注入到查询和键向量中。对于位置 `m` 和维度 `d`（必须为偶数），计算方式如下：

```
给定位置 m 和输入向量 x ∈ R^d：
1. 将 x 分为两部分：x₁ = x[0:d/2], x₂ = x[d/2:d]
2. 对于每个维度 i ∈ [0, d/2)：
   θᵢ = 10000^(-2i/d)
3. 应用旋转：
   x'₁[i] = x₁[i] * cos(m*θᵢ) - x₂[i] * sin(m*θᵢ)
   x'₂[i] = x₁[i] * sin(m*θᵢ) + x₂[i] * cos(m*θᵢ)
```

**GPT-J vs GPT-NeoX 差异**:

- **GPT-J**: 偶数维度和奇数维度交替应用旋转
  - 索引模式：[0,1], [2,3], [4,5], ..., [d-2,d-1]
  - 每对 (2i, 2i+1) 应用 2D 旋转

- **GPT-NeoX**: 前半维度和后半维度分组应用旋转
  - 索引模式：前 d/2 个维度与后 d/2 个维度配对
  - 每对 (i, i+d/2) 应用 2D 旋转

两种方法数学上等价，但索引排列不同，影响与模型权重预训练时的对齐。

### 注册机制

```cpp
// rope_infiniop.cc 末尾的静态初始化
static bool registered = []() {
    RoPE::dispatcher().registerAll(&calculate, false);
    return true;
}();
```

- **注册时机**: 程序启动时，在进入 main 函数之前的动态初始化阶段
- **注册范围**: 使用 `registerAll` 为所有设备类型（CUDA、CPU、ROCm 等）注册 `calculate` 函数
- **覆盖策略**: `override_existing = false`，不会覆盖已注册的实现（如其他后端的实现）
- **函数指针**: `calculate` 是 `rope_impl::infiniop` 命名空间中的函数，签名与 `RoPE::schema` 完全匹配
