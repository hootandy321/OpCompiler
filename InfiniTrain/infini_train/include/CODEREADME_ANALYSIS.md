# 目录: include 架构全景

## 1. 子系统职责

`infini_train/include` 目录是 InfiniTrain 训练框架的**核心公共接口层**，为整个框架提供统一的数据抽象、设备管理、类型系统和调度机制。该层作为训练框架的"门面"，定义了所有上层应用（如神经网络模块、优化器、数据加载器）依赖的基础类型和接口。

该子系统的核心职责是：
- **定义核心数据抽象**：提供 Tensor（张量）、Device（设备）、DataType（数据类型）等基础类型，建立统一的计算抽象
- **实现自动微分基础设施**：通过 autograd 子系统提供完整的计算图构建和梯度反向传播能力
- **支持神经网络模块**：通过 nn 子系统提供模块化的神经网络层、容器和并行训练策略
- **提供硬件抽象层**：通过 common 子系统屏蔽 CPU/CUDA 等硬件差异，提供统一的计算原语
- **实现混合精度训练**：通过 autocast 机制自动管理数据类型转换，优化训练性能
- **提供性能分析工具**：通过 profiler 支持训练过程的各种性能指标采集

## 2. 模块导航 (Module Navigation)

### 2.1 核心头文件（直接位于 include 目录）

* **`tensor.h`**: 张量核心抽象
    * *功能*: 定义 Tensor 类，封装多维数组的数据存储、设备管理、自动微分和计算操作。提供完整的张量运算接口（算术、比较、归约、变换、矩阵乘法等）和 Eigen 集成
    * *职责*: 作为训练框架的核心数据结构，支持前向计算、反向传播、跨设备迁移和多种数值运算

* **`device.h`**: 设备抽象与管理
    * *功能*: 定义 Device 基类、CpuDevice 和 CudaDevice 派生类，提供设备类型枚举（CPU/CUDA）和设备管理器单例（DeviceManager）。支持设备上下文设置、同步和 CUBLAS 句柄管理
    * *职责*: 统一管理计算设备，提供设备查询、上下文切换和资源分配（如 CUDA stream、CUBLAS handle）

* **`datatype.h`**: 数据类型系统
    * *功能*: 定义 DataType 枚举（支持 UINT8/INT8、FP16/FP32/BF16 等 12 种类型），提供编译期类型映射（TypeMap、DataTypeMap）、类型大小查询（kDataTypeToSize）和类型提升逻辑（WidestType）。支持 CUDA 低精度类型的元编程抽象
    * *职责*: 建立统一的类型系统，支持跨类型安全转换、编译期类型推导和混合精度计算

* **`autocast.h`**: 自动混合精度训练
    * *功能*: 实现 AutocastContext 线程局部上下文和 AutocastGuard RAII 守卫，根据操作类型自动选择合适的计算精度。定义 CastPolicy 枚举（kLowerPrecision、kFP32、kPromoteWidest）和操作到策略的映射表（kOpCastPolicyMap）
    * *职责*: 自动管理前向传播的数据类型转换，在保持数值稳定性的同时提升训练性能

* **`dispatcher.h`**: 算子分发与注册系统
    * *功能*: 提供 Dispatcher 单例和 KernelFunction 函数指针包装器，实现基于 (DeviceType, kernel_name) 的算子注册和分发机制。支持类型感知分发（DispatchFunc）和编译期类型检查
    * *职责*: 管理所有算子内核的注册表，根据设备类型和操作名称动态分发到正确的实现

* **`optimizer.h`**: 优化器抽象接口
    * *功能*: 定义 Optimizer 基类和具体实现（SGD、Adam），提供 ZeroGrad 和 Step 虚函数接口。支持学习率、动量等超参数配置
    * *职责*: 提供参数更新算法，管理梯度清零和参数优化步骤

* **`dataloader.h`**: 数据加载器
    * *功能*: 实现 DataLoader 和 DistributedDataLoader 类，提供迭代器接口（DataLoaderIterator），支持批次划分、分布式采样和多进程数据分片
    * *职责*: 管理训练数据的批次加载和分布式分发，支持单机和多机多卡训练

* **`dataset.h`**: 数据集抽象接口
    * *功能*: 定义 Dataset 抽象基类，提供 operator[] 和 Size 纯虚函数接口
    * *职责*: 定义数据访问的统一接口，支持用户自定义数据集实现

* **`profiler.h`**: 性能分析工具
    * *功能*: 实现 Profiler 单例，提供 StartRecord/EndRecord 记录接口和 Report/PrintRecords 输出功能。支持主机/设备计时、内存使用统计和分组排序（按时间、调用次数等）
    * *职责*: 采集训练过程中的性能数据（kernel 执行时间、设备内存使用），生成性能分析报告

### 2.2 子目录模块

* **`autograd/`**: 自动微分引擎
    * *功能*: 实现动态计算图自动微分系统，提供 Function 基类、梯度模式管理（GradMode/NoGradGuard）、梯度后处理钩子（PostAccumulateGradHook）和各种可微分操作（激活函数、线性层、矩阵乘法、归约操作、分布式通信等）
    * *职责*: 构建前向计算图，自动反向传播梯度，支持分布式训练的梯度同步和多种累积策略

* **`common/`**: 硬件抽象层（HAL）
    * *功能*: 提供跨平台（CPU/CUDA）的类型安全转换、数学运算原语和错误处理机制。包括通用工具宏（common.h）、CPU 后端类型转换（cpu/common_cpu.h）和 CUDA 基础设施（cuda/common_cuda.h、cuda/kernel_helper.cuh）
    * *职责*: 屏蔽底层硬件差异，为上层算子提供高性能计算原语和统一的错误检查机制

* **`nn/`**: 神经网络核心抽象层
    * *功能*: 提供完整的深度学习模型构建、训练和分布式推理能力。包括模块化抽象（Module 基类）、核心算子层（Linear、LayerNorm、Embedding 等）、多维并行支持（DP/TP/PP）和工具函数（functional.h、init.h）
    * *职责*: 支持从简单前馈网络到大规模 3D 并行训练的全场景覆盖，提供 PyTorch 风格的模块接口

## 3. 架构逻辑图解

### 3.1 层次结构

```
include/
├── 基础抽象层: tensor.h, device.h, datatype.h
│   └── 定义 Tensor（数据）、Device（计算设备）、DataType（类型系统）
│
├── 计算图与优化层: autograd/, optimizer.h
│   ├── autograd/: 自动微分引擎（Function、梯度计算、分布式通信）
│   └── optimizer.h: 参数优化算法（SGD、Adam）
│
├── 神经网络层: nn/
│   ├── modules/: 基础层（Linear、LayerNorm、Embedding、容器）
│   ├── parallel/: 并行策略（DP、TP、PP、ProcessGroup）
│   └── 工具: functional.h（函数式 API）、init.h（初始化）
│
├── 硬件抽象层: common/
│   ├── common.h: 平台无关工具宏
│   ├── cpu/: CPU 后端类型转换
│   └── cuda/: CUDA 基础设施、错误处理、数学运算、原子操作
│
├── 系统服务层: dataloader.h, dataset.h, profiler.h
│   ├── dataloader.h/dataset.h: 数据加载与抽象
│   └── profiler.h: 性能分析与计时
│
└── 元编程与分发层: autocast.h, dispatcher.h
    ├── autocast.h: 自动混合精度训练
    └── dispatcher.h: 算子注册与分发
```

### 3.2 数据流与交互关系

#### 训练流程的核心数据流

1. **模型初始化阶段**：
   - 用户通过 `device.h` 的 DeviceManager 获取计算设备（CPU 或 CUDA）
   - 通过 `datatype.h` 定义的数据类型创建参数张量
   - 使用 `nn/init.h` 中的初始化函数初始化模型参数

2. **数据加载阶段**：
   - 用户实现 `dataset.h` 的 Dataset 接口提供数据访问
   - `dataloader.h` 的 DataLoader 将数据分批，DistributedDataLoader 支持多进程分片
   - 数据被加载到 Tensor 中，支持跨设备迁移（Tensor::To）

3. **前向传播阶段**：
   - 输入 Tensor 通过 `nn/modules/` 定义的各层（Linear、LayerNorm 等）
   - 每层的操作调用 `dispatcher.h` 的 Dispatcher 分发到具体算子内核
   - 如果启用 `autocast.h`，操作数会自动转换为合适的精度（FP16/FP32）
   - `autograd/` 的 Function 记录前向操作，构建计算图，保存中间结果到 saved_tensors_

4. **反向传播阶段**：
   - 调用 Tensor::Backward() 触发反向传播
   - `autograd/` 从损失函数开始，按拓扑序遍历计算图
   - 每个 Function::Backward() 计算梯度，梯度沿计算图反向流动
   - 分布式训练时，梯度通过 `autograd/function_hook.h` 的钩子执行 AllReduce 同步
   - 叶节点的梯度累积到 Tensor::grad，由 AccumumulateGrad 管理

5. **参数更新阶段**：
   - `optimizer.h` 的 Optimizer 收集所有参数的梯度
   - 执行 Step() 方法（如 SGD、Adam）更新参数
   - 调用 ZeroGrad() 清空梯度，准备下一轮迭代

#### 类型系统与分发机制

1. **编译期类型推导**：
   - `datatype.h` 的 TypeMap 将 DataType 枚举映射到 C++ 类型（如 kFLOAT32 → float）
   - WidestType 在编译期计算多个类型的提升规则（FP16 + BF16 → float）
   - `dispatcher.h` 的 DispatchFunc 使用 if constexpr 在编译期分发到类型专用内核

2. **运行时分发**：
   - `dispatcher.h` 的 Dispatcher 单例维护 (DeviceType, kernel_name) → KernelFunction 的映射表
   - 算子通过 REGISTER_KERNEL 宏在程序启动时注册到 Dispatcher
   - 运行时根据设备类型和操作名称查找并调用正确的内核函数

3. **混合精度训练**：
   - `autocast.h` 定义操作到转换策略的映射（如 Matmul → kLowerPrecision，Softmax → kFP32）
   - AutocastGuard 在作用域内启用自动转换，tls_autocast_context 保存线程局部状态
   - Dispatcher::Call 在调用内核前调用 AutocastContext::Autocast 转换参数类型

#### 硬件抽象与跨平台支持

1. **设备管理**：
   - `device.h` 的 DeviceManager 单例管理所有可用设备（CPU、多张 GPU）
   - 每个设备封装 CUDA stream、CUBLAS handle 等资源
   - 虚函数接口 SetDevice()、Synchronize() 支持设备上下文切换

2. **后端原语**：
   - `common/cpu/common_cpu.h` 提供 Cast<> 模板，使用完美转发实现零开销类型转换
   - `common/cuda/common_cuda.h` 提供 CUDA 错误检查宏（CUDA_CHECK、CUBLAS_CHECK）和数学运算库
   - `common/cuda/kernel_helper.cuh` 实现类型泛化的数学函数（Sin、Cos、Exp、Sigmoid、Tanh）和向量化原子操作（fastAtomicAdd）

3. **错误处理**：
   - 所有 CUDA API 调用通过 _CHECK 宏包裹，失败时调用 LOG(FATAL) 输出错误位置
   - `common/common.h` 的 LOG_UNSUPPORTED_DTYPE 统一处理不支持的数据类型错误

### 3.3 模块间依赖关系

```
                ┌──────────────────┐
                │   datatype.h     │
                │ (类型系统核心)    │
                └────────┬─────────┘
                         │ 被依赖
        ┌────────────────┼────────────────┐
        │                │                │
   ┌────▼────┐     ┌────▼─────┐    ┌────▼─────┐
   │device.h │     │tensor.h  │    │autocast.h│
   └────┬────┘     └────┬─────┘    └────┬─────┘
        │               │                │
        └───────────────┼────────────────┘
                        │ 被依赖
        ┌───────────────┼────────────────┐
        │               │                │
   ┌────▼─────┐   ┌────▼──────┐   ┌────▼──────┐
   │autograd/ │   │dispatcher │   │  nn/      │
   │optimizer │   │common/    │   │dataloader │
   └──────────┘   └───────────┘   └───────────┘
```

**关键依赖路径**：
- `tensor.h` 依赖 `device.h`（设备管理）、`datatype.h`（类型定义）
- `autograd/` 依赖 `tensor.h`（计算图节点操作 Tensor）、`nn/parallel/`（分布式通信）
- `nn/` 依赖 `tensor.h`（模块参数是 Tensor）、`autograd/`（前向传播构建计算图）、`common/`（算子实现调用后端原语）
- `dispatcher.h` 依赖 `autocast.h`（自动类型转换）、`device.h`（设备分发）、`profiler.h`（性能记录）

### 3.4 设计模式应用

- **抽象工厂模式（Abstract Factory）**：`dispatcher.h` 的 Dispatcher 作为内核工厂，根据 (DeviceType, kernel_name) 创建具体 KernelFunction
- **策略模式（Strategy Pattern）**：`autograd/` 的 Function 定义算法框架，派生类实现具体前向/反向逻辑
- **RAII（资源获取即初始化）**：`autocast.h` 的 AutocastGuard、`autograd/grad_mode.h` 的 NoGradGuard 管理作用域资源
- **单例模式（Singleton Pattern）**：`device.h` 的 DeviceManager、`dispatcher.h` 的 Dispatcher、`profiler.h` 的 Profiler 采用单例
- **观察者模式（Observer Pattern）**：`autograd/function_hook.h` 的 PostAccumulateGradHook 钩子机制，解耦梯度计算与同步
- **模板方法模式（Template Method Pattern）**：`autograd/function.h` 的 Function 定义 Apply 流程，派生类实现 Forward/Backward
- **CRTP（奇异递归模板模式）**：`datatype.h` 的 TypeMap 编译期类型映射，`nn/modules/module.h` 的 CloneableModule
- **迭代器模式（Iterator Pattern）**：`dataloader.h` 的 DataLoaderIterator 提供统一的数据遍历接口

### 3.5 关键技术实现

#### 自动微分的计算图构建
- **动态图**：前向传播时实时构建计算图，每个 Function 节点通过 next_functions_ 指向后继节点
- **梯度累积**：支持多输入节点通过 dependencies_number_/dependencies_reached_ 同步所有输入的梯度到达
- **内存优化**：仅保存反向传播所需中间结果（saved_tensors_），非叶节点梯度在反向后自动释放
- **分布式梯度同步**：通过 PostAccumulateGradHook 钩子在累积梯度后自动执行 AllReduce

#### 混合精度训练
- **操作级策略**：根据操作类型选择转换策略（Matmul 用 FP16，Softmax 用 FP32）
- **线程局部上下文**：tls_autocast_context 保证多线程独立性，支持嵌套作用域
- **自动类型转换**：Dispatcher::Call 在调用内核前转换参数，用户无感知
- **设备相关默认值**：kDeviceDefaultDtype 定义 CPU 用 BF16、GPU 用 FP16

#### 算子分发机制
- **编译期注册**：REGISTER_KERNEL 宏利用静态变量初始化在程序启动时注册内核
- **类型安全分发**：DispatchFunc 使用模板特化和 if constexpr 在编译期检查类型
- **多级分发**：先按设备类型分发，再按数据类型分发，最后按操作名称分发
- **性能记录集成**：PROFILE_MODE 下自动记录 kernel 执行时间和设备内存使用

### 3.6 与外部系统的集成

`include` 目录作为 InfiniTrain 的公共接口层，与以下外部库/系统交互：

- **CUDA 生态**：
  - `device.h` 管理CUDA device、stream、CUBLAS handle
  - `common/cuda/` 依赖 CUDA Runtime、Driver API、CUBLAS、NCCL
  - `datatype.h` 映射 DataType::kFLOAT16 → half，kBFLOAT16 → nv_bfloat16

- **Eigen 库**：
  - `tensor.h` 集成 Eigen::Map，支持张量与 Eigen 矩阵/向量的互操作
  - 利用 Eigen 的高性能线性代数内核（CPU 后端）

- **glog 日志库**：
  - `common/common.h` 的 LOG 宏、`device.h` 的 CHECK 宏
  - 用于错误报告、调试信息和性能分析输出

- **InfiniCore 计算算子库**：
  - `dispatcher.h` 分发的内核函数可能由 InfiniCore 实现
  - 提供底层张量运算（GEMM、卷积、归约等）

### 3.7 使用场景建议

- **CPU 训练**：使用 DeviceManager 获取 CpuDevice，tensor 默认在 CPU 创建，common/cpu/ 提供类型转换
- **GPU 训练**：使用 CudaDevice，dispatcher.h 分发到 CUDA 内核，common/cuda/ 提供数学运算
- **混合精度训练**：启用 AutocastGuard，选择 autocast_dtype 为 FP16/BF16，自动管理类型转换
- **分布式训练**：
  - 数据并行：使用 nn/parallel/distributed_data_parallel.h，梯度通过 autograd/comm.h 同步
  - 张量并行：使用 ColumnParallelLinear/RowParallelLinear，通过 AllGather/AllReduce 协同
  - 流水线并行：使用 nn/parallel/pp/pipeline_parallel.h，通过 ISend/IRecv 传递激活
- **性能分析**：启用 Profiler 采集 kernel 执行时间，Report 输出分析报告

## 4. 架构特色与优势

### 4.1 统一且类型安全的抽象
- Tensor/Device/DataType 提供统一的数据、设备和类型抽象
- 编译期类型映射（TypeMap）和类型推导（WidestType）保证类型安全
- 模板元编程实现零开销抽象，无运行时性能损失

### 4.2 灵活的自动微分系统
- 动态计算图支持控制流和动态模型
- 分布式训练的梯度同步通过钩子机制无缝集成
- 支持高阶导数（create_graph 参数）和梯度累积

### 4.3 完善的并行训练支持
- 通过 nn/parallel/ 提供 DP、TP、PP 三维并行
- ProcessGroup 抽象封装底层通信库（NCCL、MPI）
- 异步通信和计算重叠优化训练效率

### 4.4 PyTorch 兼容性
- API 设计高度模仿 PyTorch（Module、Tensor、optimizer、dataloader）
- 参数命名和状态字典格式兼容，支持模型迁移
- Autocast 机制对应 torch.cuda.amp.autocast

### 4.5 生产级特性
- 多硬件后端支持（CPU、CUDA、Kunlun、Metax、Ascend）
- 完善的错误检查和日志系统（glog 集成）
- 性能分析工具（Profiler）支持瓶颈定位
- 混合精度训练（FP16/BF16）提升性能和减少内存占用

## 总结

`include` 目录作为 InfiniTrain 框架的公共接口层，通过清晰的分层设计和模块化组织，提供了从基础数据抽象到高级分布式训练能力的完整覆盖。核心抽象层（Tensor、Device、DataType）、计算图层（autograd）、神经网络层（nn）、硬件抽象层（common）和系统服务层（dataloader、profiler）各司其职，相互协作，构成了一个类型安全、高性能、易用性强的深度学习训练框架。无论是单机单卡的小规模训练，还是多机多卡的 3D 并行超大规模训练，都能在该子系统中找到合适的构建模块和工具支持。
