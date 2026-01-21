# GEMM 算子核心实现文档

GEMM (General Matrix Multiply) 算子是深度学习框架中最核心的计算原语之一，实现了高性能的通用矩阵乘法操作 C = alpha * A * B + beta * C。该模块通过 InfiniOP 后端库提供跨硬件平台的统一接口，支持 CPU、NVIDIA GPU、华为昇腾、寒武纪、沐拓等多达 10 种硬件加速器。

## 1. 模块结构

- **`gemm.cc`**: GEMM 算子的公共前端实现，提供用户友好的高层 API（`gemm` 和 `gemm_`），负责输出张量的内存分配和参数验证。
- **`gemm_infiniop.cc`**: GEMM 算子的 InfiniOP 后端实现，通过 Plan-Run 模式与底层 InfiniOP 库交互，实现了描述符缓存、工作空间管理和异步执行。
- **`infinicore/ops/gemm.hpp`**: 算子接口声明，定义了 `Gemm` 类和便捷函数签名。

## 2. 核心类

### `Gemm`
- **Location**: `infinicore/ops/gemm.hpp`, `gemm.cc`
- **Primary Function**: 继承自 `graph::GraphOperator`，是 GEMM 操作的核心执行单元。它通过设备分发器 (Device Dispatcher) 将计算任务路由到特定硬件后端实现（如 InfiniOP、cuBLAS、cuDNN 等）。
- **Key Members**:
  - `planned_meta_`: `void*` 指针，存储后端特定的计划元数据（如 InfiniOP 描述符、工作空间张量等）
  - `runner_`: `run_schema` 函数指针，指向实际执行的函数（如 `infiniop::run`）
  - `deleter_`: `cleanup_schema` 函数指针，指向清理函数（如 `infiniop::cleanup`）
- **Core Methods**:
  - `Gemm(Tensor c, Tensor a, Tensor b, float alpha, float beta)`: 构造函数，执行设备分发逻辑，通过 `INFINICORE_GRAPH_OP_DISPATCH` 宏选择当前设备类型的 plan/run/cleanup 函数，并调用 plan 函数生成 `planned_meta_`
  - `execute(Tensor c, Tensor a, Tensor b, float alpha, float beta)`: 静态执行方法，根据上下文状态决定是记录到计算图还是立即运行。通过 `INFINICORE_GRAPH_OP_RECORD_OR_RUN` 宏实现：如果处于图录制模式（`context::isGraphRecording()`），则将算子添加到图中；否则立即调用 `op->run()`
- **Lifecycle**: 采用计划-执行-清理三阶段生命周期模式。构造时调用 plan 函数生成元数据，执行时调用 run 函数，析构时由 `GraphOperator` 基类调用 deleter 清理资源

### `PlannedMeta` (InfiniOP 后端)
- **Location**: `gemm_infiniop.cc`
- **Primary Function**: 封装 InfiniOP 后端执行所需的全部元数据，包括描述符、工作空间张量、输入输出张量的 Graph 封装和标量参数
- **Key Members**:
  - `descriptor`: `std::shared_ptr<Descriptor>`，智能指针管理的 InfiniOP 描述符，通过 RAII 机制自动调用 `infiniopDestroyGemmDescriptor` 释放资源
  - `workspace`: `graph::GraphTensor`，GEMM 计算所需的临时工作空间，大小由 `infiniopGetGemmWorkspaceSize` 动态查询
  - `c, a, b`: `graph::GraphTensor`，输出和输入张量的 Graph 封装，持有张量描述符和内存指针
  - `alpha, beta`: `float`，GEMM 公式的标量系数：C = alpha * A * B + beta * C
- **Lifecycle**: 在 plan 阶段通过 `new PlannedMeta` 动态分配，在 cleanup 阶段通过 `delete` 释放，生命周期由 `Gemm` 算子的 `planned_meta_` 指针管理

### `Descriptor` (InfiniOP 后端)
- **Location**: `gemm_infiniop.cc`，通过 `INFINIOP_CACHABLE_DESCRIPTOR` 宏定义
- **Primary Function**: 封装原生 InfiniOP 描述符 (`infiniopGemmDescriptor_t`)，提供 RAII 风格的资源管理
- **Key Members**:
  - `desc`: `infiniopGemmDescriptor_t`，指向 InfiniOP 库内部的算子描述符，包含张量布局、数据类型、硬件优化信息
- **Core Methods**:
  - 析构函数自动调用 `infiniopDestroyGemmDescriptor(desc)` 释放底层资源
- **Lifecycle**: 存储在 thread-local 的 LRU 缓存中（容量 100），按设备类型和索引隔离，线程安全。当缓存满时自动驱逐最久未使用的条目

## 3. API 接口

```cpp
// 高层便捷 API：自动分配输出张量 C
// 参数：
//   a: 输入矩阵 A，形状为 [M, K] 或 [Batch, M, K]
//   b: 输入矩阵 B，形状为 [K, N] 或 [Batch, K, N]
//   alpha: A*B 的缩放系数，默认 1.0f
//   beta: C 的缩放系数，默认 0.0f（表示不累加到 C）
// 返回：新分配的输出张量 C，形状为 [M, N] 或 [Batch, M, N]
Tensor gemm(Tensor a, Tensor b, float alpha = 1.0f, float beta = 0.0f);

// 底层就地 API：用户预先分配输出张量 C
// 参数：
//   c: 预分配的输出张量，形状必须为 [M, N] 或 [Batch, M, N]
//   a, b: 输入矩阵
//   alpha, beta: GEMM 标量系数
void gemm_(Tensor c, Tensor a, Tensor b, float alpha = 1.0f, float beta = 0.0f);

// Gemm 类的构造函数（通常不直接调用，通过 execute 间接使用）
Gemm::Gemm(Tensor c, Tensor a, Tensor b, float alpha, float beta);
// 触发设备分发，调用 plan 函数生成 PlannedMeta，建立 run/cleanup 函数指针

// Gemm 类的执行方法
void Gemm::execute(Tensor c, Tensor a, Tensor b, float alpha, float beta);
// 根据图录制状态决定：记录到图 或 立即执行
```

**InfiniOP 后端内部 API**（由 `INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE` 宏注册）：

```cpp
// Plan 阶段：生成执行计划元数据
void* plan(Tensor c, Tensor a, Tensor b, float alpha, float beta);
// 1. 基于 c, a, b 的数据类型、形状、步长计算哈希键
// 2. 从 thread-local LRU 缓存查找或创建 Descriptor（调用 infiniopCreateGemmDescriptor）
// 3. 查询工作空间大小并分配 U8 类型张量
// 4. 构建 PlannedMeta 结构体并返回指针

// Run 阶段：执行 GEMM 计算
void run(void* planned_meta);
// 1. 将 planned_meta 转换为 PlannedMeta* 指针
// 2. 调用 infiniopGemm(desc, workspace, workspace_size, c, a, b, alpha, beta, stream)
// 3. 计算在当前设备流上异步执行

// Cleanup 阶段：释放计划元数据
void cleanup(void** planned_meta_ptr);
// 1. 解引用双重指针获取 PlannedMeta*
// 2. delete 释放内存（触发 PlannedMeta 成员的智能指针自动析构）
// 3. 将指针置空
```

## 4. 使用示例

```cpp
#include "infinicore/ops/gemm.hpp"
using namespace infinicore;
using namespace infinicore::op;

// 示例 1：基本矩阵乘法 C = A * B
// A: [512, 1024], B: [1024, 768] -> C: [512, 768]
Tensor a = Tensor::zeros({512, 1024}, DataType::F32, Device(Device::Type::NVIDIA));
Tensor b = Tensor::zeros({1024, 768}, DataType::F32, Device(Device::Type::NVIDIA));
Tensor c = gemm(a, b);  // alpha=1.0, beta=0.0（默认值）

// 示例 2：带系数的 GEMM C = 2.0 * A * B + 3.0 * C
Tensor c_preallocated = Tensor::empty({512, 768}, DataType::F32, Device(Device::Type::NVIDIA));
// c_preallocated 已有初始值
gemm_(c_preallocated, a, b, 2.0f, 3.0f);  // 就地更新 c_preallocated

// 示例 3：批量 GEMM（Batched GEMM）
// A: [32, 64, 128], B: [32, 128, 96] -> C: [32, 64, 96]
Tensor batch_a = Tensor::zeros({32, 64, 128}, DataType::F32, Device(Device::Type::NVIDIA));
Tensor batch_b = Tensor::zeros({32, 128, 96}, DataType::F32, Device(Device::Type::NVIDIA));
Tensor batch_c = gemm(batch_a, batch_b);

// 示例 4：跨硬件设备（自动分发）
Tensor cpu_a = Tensor::zeros({128, 256}, DataType::F32, Device(Device::Type::CPU));
Tensor cpu_b = Tensor::zeros({256, 192}, DataType::F32, Device(Device::Type::CPU));
Tensor cpu_c = gemm(cpu_a, cpu_b);  // 自动调用 CPU 后端（如 oneDNN）

Tensor ascend_a = Tensor::zeros({128, 256}, DataType::F32, Device(Device::Type::ASCEND));
Tensor ascend_b = Tensor::zeros({256, 192}, DataType::F32, Device(Device::Type::ASCEND));
Tensor ascend_c = gemm(ascend_a, ascend_b);  // 自动调用昇腾后端（HCCL）
```

## 5. 实现细节

### 宏系统驱动的架构分层

该模块深度依赖 C++ 宏元编程实现设备无关的算子注册和分发：

1. **`INFINICORE_GRAPH_OP_CLASS(Gemm, Tensor, Tensor, Tensor, float, float)`**
   - 展开 `Gemm` 类定义，继承 `graph::GraphOperator`
   - 定义三个静态分发器：`plan_dispatcher()`, `run_dispatcher()`, `cleanup_dispatcher()`
   - 每个 `Device::Type` 对应一个函数指针槽位（共 10 种设备类型）
   - 声明构造函数 `Gemm(Tensor c, Tensor a, Tensor b, float alpha, float beta)` 和静态方法 `execute`

2. **`INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Gemm)`**
   - 实例化三个 `OpDispatcher` 单例，使用 Meyer's Singleton 模式（函数静态变量）
   - `OpDispatcher<plan_schema>`: 存储 `void*(*)(Tensor, Tensor, Tensor, float, float)` 类型的函数指针数组
   - `OpDispatcher<run_schema>`: 存储 `void(*)(void*)` 类型的函数指针数组
   - `OpDispatcher<cleanup_schema>`: 存储 `void(*)(void**)` 类型的函数指针数组

3. **`INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Gemm, &plan, &run, &cleanup)`**
   - 利用 C++ 静态初始化在程序启动时自动注册
   - Lambda 表达式 `[]() { return true; }()` 在加载时执行一次
   - 将 `infiniop::plan`, `infiniop::run`, `infiniop::cleanup` 注册到所有设备类型的分发器
   - `override_existing = false` 确保优先使用特定硬件优化实现（如 cuBLAS）

### Plan-Run-Cleanup 三阶段执行模型

借鉴 cuDNN/cuBLAS 的设计理念，将算子执行分解为三个阶段：

**Plan 阶段**（构造时执行一次）：
- 调用 `infiniop::plan()` 生成 `PlannedMeta` 结构体
- **描述符缓存**：通过 `hash_combine(c, a, b)` 计算张量特征哈希（dtype + shape + strides），在 thread-local LRU 缓存中查找或创建 `Descriptor`，避免重复调用 `infiniopCreateGemmDescriptor`（该函数通常耗时 1-10ms）
- **工作空间分配**：查询 `infiniopGetGemmWorkspaceSize`，分配 `Tensor::empty({workspace_size}, DataType::U8, device)`
- **构建元数据**：创建 `PlannedMeta` 并封装所有计算所需参数

**Run 阶段**（每次调用 execute）：
- 直接调用 `infiniopGemm(desc, workspace, workspace_size, c_data, a_data, b_data, alpha, beta, stream)`
- 在 GPU 上异步执行，不阻塞主机线程
- 支持多流并发：不同算子使用不同 `stream` 并行执行

**Cleanup 阶段**（Gemm 析构时）：
- `delete planned_meta` 触发智能指针自动析构
- `~Descriptor()` 调用 `infiniopDestroyGemmDescriptor` 释放 InfiniOP 内部资源（如 PTX 模块、cuBLAS 句柄）
- 工作空间张量的引用计数归零，自动释放显存

### Thread-Local 设备隔离缓存

通过 `INFINIOP_CACHABLE_DESCRIPTOR` 宏实现的高性能缓存机制：

```cpp
thread_local common::OpCache<size_t, std::shared_ptr<Descriptor>> caches(
    100,  // LRU 容量
    [](std::shared_ptr<Descriptor> &desc) { desc = nullptr; }  // 清理回调
);
```

- **Thread-Local 存储隔离**：每个线程拥有独立的缓存实例，避免多线程竞争锁开销
- **设备类型隔离**：`OpCache` 内部使用 `std::array<CacheVector, Device::Type::COUNT>` 结构，为每种硬件（CPU、NVIDIA、ASCEND 等）维护独立的缓存向量
- **设备索引隔离**：每个设备类型下有多个设备实例（如 8 张 GPU 卡），通过 `CacheVector` 为每个 `device_index` 维护独立的 LRU 缓存
- **LRU 驱逐策略**：使用 `infinicore::common::LRUCache` 实现，容量满时自动驱逐最久未使用的描述符
- **RAII 清理**：`OpCache` 析构时遍历所有设备的缓存，正确切换设备上下文后清理资源

### 张量形状推导算法

`gemm()` 函数实现了自动输出形状推导：

```cpp
Shape shape = a->shape();      // 复制 A 的形状
Size size = a->ndim();         // 获取维度数
shape[size - 1] = b->size(size - 1);  // 将 A 的最后一维替换为 B 的最后一维
```

**支持的场景**：
- **二维 GEMM**: A[M, K] * B[K, N] -> C[M, N]（常规矩阵乘法）
- **批量三维 GEMM**: A[Batch, M, K] * B[Batch, K, N] -> C[Batch, M, N]（Batch 必须相等）
- **步长张量**：通过 `a->desc()` 和 `b->desc()` 传递完整的步长信息到 InfiniOP，支持非连续内存布局

**参数验证**：
- `INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b)`：确保三个张量在同一个设备上
- 底层 InfiniOP 会验证形状兼容性（K 维度必须对齐）

### 图录制与即时执行混合模式

通过 `INFINICORE_GRAPH_OP_RECORD_OR_RUN` 宏实现：

```cpp
auto op = std::make_shared<Gemm>(c, a, b, alpha, beta);
if (context::isGraphRecording()) {
    context::addGraphOperator(op);  // 记录到计算图，延迟执行
} else {
    op->run();  // 立即执行，调用 infiniop::run(planned_meta_)
}
```

**应用场景**：
- **训练模式**：关闭图录制，每个算子立即执行，方便调试和动态控制流
- **推理优化**：开启图录制，构建完整计算图后一次性提交，减少内核启动开销

### 错误处理机制

- **`INFINICORE_CHECK_ERROR`**：封装 InfiniOP 返回的 `infiniStatus_t`，失败时抛出 `std::runtime_error` 异常
- **`INFINICORE_ASSERT_TENSORS_SAME_DEVICE`**：设备不匹配时抛出异常并打印详细错误信息
- **空指针保护**：`Descriptor` 析构函数检查 `desc != nullptr` 后才调用 `infiniopDestroyGemmDescriptor`

### 性能优化技术

1. **描述符缓存**：避免重复调用 `infiniopCreateGemmDescriptor`（可节省 90%+ 的初始化时间）
2. **工作空间复用**：`PlannedMeta` 持有工作空间张量，多次调用同一配置的 GEMM 时复用
3. **异步执行**：`infiniopGemm` 在 GPU 流上异步执行，主机线程可继续工作
4. **零拷贝优化**：输入输出张量直接传递设备指针，无需内存拷贝
5. **硬件特定优化**：InfiniOP 后端自动选择最优内核（如 Tensor Core、张量核心指令）
6. **批量处理**：支持 Batched GEMM，一次内核启动处理多个矩阵乘法

### 依赖关系

**外部依赖**：
- **InfiniOP 库** (`libinfiniop.so`)：提供跨硬件的统一算子接口
  - `infiniopCreateGemmDescriptor`: 创建算子描述符
  - `infiniopGetGemmWorkspaceSize`: 查询工作空间大小
  - `infiniopGemm`: 执行 GEMM 计算
  - `infiniopDestroyGemmDescriptor`: 销毁描述符
- **InfiniRT 运行时**：提供设备管理、流管理、内存分配
  - `infinirtStream_t`: 异步执行流
  - `context::getStream()`: 获取当前设备流

**内部依赖**：
- **`tensor.hpp`**: 张量抽象层，提供 `Tensor::empty()`、`shape()`、`data()` 等接口
- **`device.hpp`**: 设备枚举和设备类型定义
- **`graph/graph.hpp`**: `GraphOperator` 基类、`GraphTensor` 包装类、宏定义
- **`context/context.hpp`**: 线程局部存储（当前设备、流、图录制状态）
- **`common/hash.hpp`**: 哈希组合函数，用于缓存键计算
- **`common/cache.hpp`**: LRU 缓存实现，支持设备隔离
- **`common/LRUCache.hpp`**: 底层 LRU 缓存数据结构

### 设计模式

1. **Strategy Pattern（策略模式）**：`OpDispatcher` 允许运行时选择不同的后端实现（InfiniOP、cuBLAS、oneDNN）
2. **RAII（资源获取即初始化）**：`Descriptor` 类管理 InfiniOP 描述符生命周期，`Tensor::empty` 管理设备内存
3. **Meyer's Singleton**：`plan_dispatcher()`, `run_dispatcher()`, `cleanup_dispatcher()` 使用函数静态变量实现延迟初始化
4. **Template Method**：`GraphOperator` 定义执行骨架，子类（如 `Gemm`）实现具体逻辑
5. **Factory（工厂模式）**：`INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE` 在静态初始化时自动注册产品（plan/run/cleanup 函数）
6. **Proxy（代理模式）**：`GraphTensor` 包装 `Tensor`，在图录制模式下提供轻量级代理
7. **Cache-Aside**：描述符缓存采用旁路缓存模式，先查缓存，未命中再创建
8. **Object Pool（对象池）**：工作空间张量可视为对象池，多次执行复用同一内存

### 内存管理

- **设备内存分配**：`Tensor::empty()` 调用 InfiniRT 的 `infinirtMalloc` 分配 GPU 显存或主机内存
- **工作空间管理**：由 `PlannedMeta` 持有 `workspace` 张量，`PlannedMeta` 销毁时自动释放
- **引用计数**：`Tensor` 使用 `std::shared_ptr<TensorImpl>` 管理底层实现，工作空间张量的引用计数归零时自动释放
- **设备切换**：`OpCache::clear()` 在清理多设备缓存时正确切换设备上下文，避免跨设备访问错误

### 并发安全性

- **Thread-Local 缓存**：`thread_local` 关键字确保每个线程独立缓存，无竞争条件
- **只读共享**：`Descriptor` 被 `std::shared_ptr` 管理，多个 `PlannedMeta` 可共享同一个描述符（只读访问）
- **流隔离**：每个 `infiniopGemm` 调用绑定到特定的 `stream`，不同流的计算并行执行
- **设备隔离**：不同设备的缓存和执行上下文完全隔离，支持多设备并行
