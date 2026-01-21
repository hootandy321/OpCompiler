# 目录: infinicore 头文件架构全景

## 1. 子系统职责

`infinicore` 目录是 InfiniCore 框架的公共 API 头文件层，定义了整个框架的对外接口规范。该目录作为框架的"门面"，提供了从底层硬件抽象到高层神经网络构建的完整编程接口体系。它在整个 InfiniCore 架构中承担着承上启下的关键职责：向下对接 `infinirt` 和 `infiniop` 两个底层硬件抽象库，向上为应用层提供统一的设备无关的计算抽象。

该目录的核心设计理念是通过分层模块化和设备分发机制，实现"一次编码，多硬件运行"的目标。所有子模块都遵循统一的设计模式：公共接口定义在头文件中，设备特定实现由后端库提供，通过 `OpDispatcher` 机制在运行时动态分发。这种架构使得 InfiniCore 能够无缝支持 10 种硬件后端（CPU、NVIDIA、CAMBRICON、ASCEND、METAX、MOORE、ILUVATAR、KUNLUN、HYGO、QY），同时保持 API 的简洁性和类型安全。

## 2. 模块导航

* **common**: 通用工具模块，提供框架的基础设施组件
  * *功能*: 实现哈希组合工具和 LRU 缓存数据结构，为框架提供高性能的复合键生成和资源缓存管理能力
  * *职责*: 为上层模块提供零开销的工具类，支持 Tensor 元数据哈希、算子缓存键生成、GPU 内存管理等场景的通用基础设施

* **context**: 运行时上下文管理模块，提供统一的设备、流、内存、事件和计算图管理接口
  * *功能*: 封装 `infinirt` 和 `infiniop` 两个底层库的 C 接口，提供 RAII 风格的 C++ API，管理设备状态、内存分配、数据传输、性能测量和计算图录制
  * *职责*: 作为硬件抽象层的统一入口，为算子执行提供设备上下文管理、异步执行控制、跨设备数据传输和计算图优化基础设施

* **graph**: 计算图执行框架模块，提供声明式算子图构建和批量执行能力
  * *功能*: 实现算子抽象基类、计算图容器和宏驱动的算子注册系统，支持算子录制、图构建和重复执行优化
  * *职责*: 将即时执行的算子调用转换为可优化的计算图表示，通过算子融合、批量执行和 kernel 优化提升性能

* **nn**: 神经网络层模块，提供 PyTorch 风格的高层深度学习组件
  * *功能*: 实现模块基类、参数管理、线性层（含张量并行）、嵌入层、RMS 归一化和旋转位置编码等现代 Transformer 架构的关键组件
  * *职责*: 为大语言模型等深度学习应用提供高层次的构建块，支持层级化参数管理、状态字典序列化和分布式训练的张量并行

* **ops**: 算子库模块，实现框架的核心计算算子集合
  * *功能*: 提供基础数学运算、矩阵乘法、注意力机制（含分页注意力）、激活函数、归一化、位置编码和随机采样等算子，覆盖深度学习推理和训练的核心计算需求
  * *职责*: 为神经网络层和用户代码提供设备无关的计算原语，通过分发机制自动路由到最优的硬件实现

* **ops/common**: 算子基础设施模块，提供算子分发和缓存的底层机制
  * *功能*: 实现 `OpDispatcher` 设备类型路由表和 `OpCache` 多设备 LRU 缓存容器，为所有算子提供统一的注册、查找和性能优化基础设施
  * *职责*: 作为算子系统的底层支撑框架，管理设备特定实现函数的注册和查找，提供跨设备缓存管理以减少编译开销

## 3. 架构逻辑图解

InfiniCore 头文件层采用清晰的分层架构，各模块通过明确的依赖关系和数据流协同工作：

### 自底向上的依赖层次

1. **基础设施层 (common)**: 提供最底层的数据结构工具，为所有上层模块提供哈希计算和缓存管理能力
   - `hash.hpp` 为 `ops/common/cache.hpp` 中的缓存键生成提供哈希组合工具
   - `LRUCache.hpp` 被 `ops/common/cache.hpp` 组合使用，实现算子级的缓存管理

2. **运行时抽象层 (context)**: 作为硬件抽象层的 C++ 封装，管理底层设备状态和资源
   - 为 `ops/common/cache.hpp` 提供设备上下文切换能力（`setDevice/getDevice`）
   - 为 `graph` 模块提供计算图录制状态查询和算子添加接口
   - 为 `ops` 模块中的所有算子提供设备内存分配和数据传输支持
   - 为 `nn` 模块的张量并行层提供通信句柄管理（NCCL）

3. **算子基础设施层 (ops/common)**: 建立在基础设施层之上，为算子系统提供分发和缓存机制
   - `OpDispatcher` 维护设备类型到函数指针的映射表，支持 10 种硬件后端
   - `OpCache` 利用 `common::LRUCache` 和 `context` 的设备管理，实现跨设备的算子结果缓存
   - 该层不直接依赖 `ops` 和 `nn`，而是作为它们的底层支撑

4. **计算图层 (graph)**: 提供算子容器和执行框架，支持声明式图构建
   - 通过 `context` 模块的图录制接口（`isGraphRecording/addGraphOperator`）收集算子
   - 利用 `ops/common` 的分发机制选择算子的硬件特定实现
   - 通过 `GraphOperator` 基类和宏系统（`INFINICORE_GRAPH_OP_CLASS`）实现类型安全的算子注册
   - 提供批量执行能力，减少 kernel 启动开销

5. **算子实现层 (ops)**: 实现具体的计算算子，是框架的计算核心
   - 每个算子（如 `Add/Gemm/Attention`）都使用 `ops/common::OpDispatcher` 注册设备特定实现
   - 通过 `context` 模块获取设备信息、分配内存和执行数据传输
   - 算子函数分为 out-of-place（如 `add`）和 in-place（如 `add_`）两种 API，满足不同场景需求
   - 高级算子（如分页注意力）使用 `common::LRUCache` 缓存编译后的 kernel

6. **神经网络层 (nn)**: 提供高层次深度学习组件，建立在算子层之上
   - 所有神经网络层（`Linear/Embedding/RMSNorm` 等）的 `forward` 方法调用 `ops` 模块的算子实现
   - `Module` 基类使用 `common::hash` 工具生成参数哈希键，用于缓存和序列化
   - 张量并行层（`ColumnParallelLinear/RowParallelLinear`）通过 `context` 获取 NCCL 通信句柄
   - `RoPE` 层使用 `context` 分配设备内存预计算 sin/cos 查找表

### 关键数据流路径

**路径 1: 算子执行流程**
```
用户代码调用 ops::add(a, b)
  ↓
Add::execute() 调用 dispatcher().lookup(device)
  ↓
OpDispatcher 返回当前设备的实现函数指针
  ↓
执行设备特定实现（如 CUDA kernel）
  ↓
如果需要缓存，通过 OpCache 存储结果
```

**路径 2: 计算图录制与执行**
```
context::startGraphRecording()
  ↓
用户调用 INFINICORE_GRAPH_OP_RECORD_OR_RUN(MatMulOp, a, b)
  ↓
context::isGraphRecording() 返回 true，调用 addGraphOperator(op)
  ↓
GraphOperator 构造时通过 OpDispatcher 查找 plan/run/cleanup 函数
  ↓
context::stopGraphRecording() 返回完整 Graph 对象
  ↓
Graph::run() 顺序执行所有算子的 runner 函数
```

**路径 3: 神经网络前向传播**
```
nn::Linear::forward(input)
  ↓
调用 ops::linear(input, weight, bias)
  ↓
linear 算子内部调用 ops::matmul 或 ops::gemm
  ↓
通过 OpDispatcher 路由到设备实现
  ↓
context::allocateMemory 分配输出张量内存
  ↓
返回结果张量
```

**路径 4: 多设备缓存管理**
```
OpCache<Key, Value>::getCache(Device(type=CUDA, index=0))
  ↓
访问 caches_[CUDA][0] 缓存实例
  ↓
如果首次访问，扩容 vector 并创建新 LRU 缓存
  ↓
context::setDevice(CUDA, 0) 切换设备上下文
  ↓
LRUCache::put/get 操作缓存条目
  ↓
驱逐时调用自定义 destructor 释放 GPU 资源
```

### 设计协同机制

1. **设备分发协同**: 所有算子通过 `ops/common::OpDispatcher` 统一管理设备特定实现，确保新硬件后端只需注册函数指针即可接入框架，无需修改上层 API

2. **内存管理协同**: `context` 模块提供的 RAII 风格内存分配与 `Tensor` 类的智能指针结合，自动管理跨设备内存的生命周期，`nn::Parameter` 和 `ops` 算子都依赖此机制避免内存泄漏

3. **计算图优化协同**: `graph` 模块通过 `context` 的录制接口收集算子，算子本身通过 `OpDispatcher` 查找实现，这种解耦设计使得图录制不影响算子的即时执行模式

4. **层级化抽象协同**: `nn` 模块的高层组件（如 `TransformerBlock`）组合多个 `ops` 算子，`ops` 算子组合 `context` 的底层 API，形成清晰的抽象层次，每层职责单一且可复用

5. **性能优化协同**: `ops/common::OpCache` 利用 `common::LRUCache` 和 `context` 的设备管理实现跨设备缓存，为 `ops` 模块的算子提供透明的编译结果缓存，减少重复编译开销

通过这种分层模块化和清晰的依赖关系，`infinicore` 目录形成了一个高内聚、低耦合的公共 API 体系，既保持了接口的简洁性，又实现了多硬件支持和性能优化的灵活性。
