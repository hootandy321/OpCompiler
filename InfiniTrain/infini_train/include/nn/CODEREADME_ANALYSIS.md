# 神经网络 (Neural Network) 子系统架构全景

## 1. 子系统职责

`infini_train/include/nn` 目录是 InfiniTrain 框架的**神经网络核心抽象层**，提供完整的深度学习模型构建、训练和分布式推理能力。该子系统实现了类似 PyTorch 的模块化接口设计，支持从简单的前馈网络到大规模分布式 3D 并行训练的全场景覆盖。

该子系统的核心职责是：
- **提供模块化抽象**：通过 `Module` 基类和容器模块，支持灵活的模型组合和嵌套
- **实现核心算子层**：提供线性层、激活函数、归一化层、嵌入层等基础神经网络组件
- **支持多维并行**：集成数据并行（DP）、张量并行（TP）、流水线并行（PP）的 3D 并行训练能力
- **提供工具函数**：通过 `functional.h` 和 `init.h` 提供张量操作和参数初始化工具

## 2. 模块导航 (Module Navigation)

### 2.1 核心模块 (modules/)

* **`module.h`**: 神经网络模块基础抽象
    * *功能*: 定义 `Module` 抽象基类和 `CloneableModule` CRTP 模板基类，实现参数/缓冲区管理、设备迁移、递归应用、状态字典等基础设施
    * *职责*: 提供统一的模块接口，支持所有神经网络层继承和组合

* **`activations.h`**: 激活函数层
    * *功能*: 实现常见的激活函数层，包括 Sigmoid
    * *职责*: 提供非线性变换能力，用于构建深度神经网络

* **`container.h`**: 容器模块
    * *功能*: 提供三种组合模式容器：`Sequential`（顺序执行）、`ModuleList`（有序列表）、`ModuleDict`（字典容器）
    * *职责*: 支持模块的组合和复用，构建复杂网络结构

* **`linear.h`**: 全连接线性层
    * *功能*: 实现线性变换层 `y = xA^T + b`，支持可选偏置项
    * *职责*: 提供基础的矩阵乘法变换，是大多数网络的核心组件

* **`loss.h`**: 损失函数层
    * *功能*: 实现交叉熵损失 `CrossEntropyLoss`，结合 LogSoftmax 和负对数似然
    * *职责*: 提供训练目标的误差度量，用于优化器的目标函数

* **`normalization.h`**: 归一化层
    * *功能*: 实现层归一化 `LayerNorm`，对指定维度应用归一化和仿射变换
    * *职责*: 稳定训练过程，加速收敛，防止梯度消失/爆炸

* **`sparse.h`**: 稀疏层
    * *功能*: 实现词嵌入层 `Embedding`，将离散索引映射到稠密向量
    * *职责*: 处理类别特征和文本 token，提供可学习的嵌入表示

### 2.2 并行训练模块 (parallel/)

* **`global.h`**: 全局并行环境管理
    * *功能*: 提供 `GlobalEnv` 单例和 `Layout` 结构体，管理 DP、TP、PP 三维并行的进程拓扑布局
    * *职责*: 初始化和维护分布式训练的全局元数据，提供 rank 坐标转换

* **`process_group.h`**: 进程组通信抽象
    * *功能*: 定义 `ProcessGroup` 抽象基类，提供集合通信（AllReduce、AllGather、ReduceScatter）和点对点通信（Send、Recv）的异步接口
    * *职责*: 封装底层通信库（NCCL、MPI），为上层并行模块提供统一通信原语

* **`reducer.h`**: 分布式梯度同步
    * *功能*: 实现 `Reducer` 类，管理梯度桶化和异步 AllReduce 同步
    * *职责*: 高效同步梯度，通过桶化和异步通信隐藏通信延迟

* **`data_parallel.h`**: 数据并行模块（简单版本）
    * *功能*: 实现 `DataParallel` 类，将输入切分到多设备，独立计算后聚合输出
    * *职责*: 提供单机多卡的基础数据并行能力

* **`distributed_data_parallel.h`**: 分布式数据并行模块
    * *功能*: 实现 `DistributedDataParallel` 类，封装 `Reducer`，在反向传播时自动同步梯度
    * *职责*: 提供生产级的多机多卡数据并行训练能力

* **`tensor_parallel.h`**: 张量并行模块
    * *功能*: 定义 `ColumnParallelLinear` 和 `RowParallelLinear` 类，实现线性层的按列/按行切分
    * *职责*: 将大型矩阵乘法操作切分到多个设备，支持超大模型训练

* **`pp/`**: 流水线并行子模块（详见下节）

### 2.3 流水线并行子模块 (parallel/pp/)

* **`pipeline_parallel.h`**: 流水线并行主控制器
    * *功能*: 定义 `PipelineParallel` 类，负责模型分割、stage 构建和训练步骤协调
    * *职责*: 管理流水线并行的完整生命周期，协调 micro-batch 的调度

* **`pipeline_schedule.h`**: 流水线调度策略
    * *功能*: 定义 `PipelineSchedule` 抽象基类和 `PipelineParallelScheduler` 生成器，支持 GPipe 和 Interleaved 1F1B 调度
    * *职责*: 实现流水线调度算法，优化计算和通信的重叠

* **`pipeline_stage.h`**: 流水线阶段封装
    * *功能*: 定义 `PipelineStage` 类，封装单个阶段的模型 chunk、优化器和设备上下文
    * *职责*: 提供阶段级别的执行接口和身份查询

* **`send_recv.h`**: 跨设备通信原语
    * *功能*: 提供 `ISend` 和 `IRecv` 异步点对点通信函数
    * *职责*: 实现 stage 间的异步张量传输，支持计算与通信重叠

### 2.4 工具函数模块

* **`functional.h`**: 函数式张量操作接口
    * *功能*: 提供数学运算（三角函数、幂运算、归约）、张量变换（切片、拼接、三角矩阵提取）、激活函数（Sigmoid、Softmax）等函数式 API
    * *职责*: 提供无状态的张量操作，供模块内部或用户直接调用

* **`init.h`**: 参数初始化工具
    * *功能*: 提供 `Normal`、`KaimingUniform`、`Uniform`、`Ones`、`Zeros`、`Arange` 等初始化函数
    * *职责*: 支持多种参数初始化策略，确保训练稳定性

## 3. 架构逻辑图解

### 3.1 层次结构

```
nn/
├── 基础抽象层: modules/module.h
│   └── 定义 Module 基类、参数管理、设备迁移、状态序列化
│
├── 核心组件层: modules/*.h
│   ├── Linear, Embedding: 基础变换层
│   ├── Sigmoid, LayerNorm: 激活和归一化
│   ├── Sequential, ModuleList, ModuleDict: 容器组合
│   └── CrossEntropyLoss: 损失函数
│
├── 并行抽象层: parallel/
│   ├── 全局环境: global.h, rank.h
│   ├── 通信接口: process_group.h, work.h, parallel_functional.h
│   ├── 数据并行: data_parallel.h, distributed_data_parallel.h, reducer.h
│   ├── 张量并行: tensor_parallel.h (ColumnParallelLinear, RowParallelLinear)
│   └── 流水线并行: pp/ (PipelineParallel, PipelineSchedule, PipelineStage)
│
└── 工具函数层: functional.h, init.h
    ├── 数学运算、归约、张量变换
    └── 参数初始化策略
```

### 3.2 数据流与交互关系

#### 单机单卡训练流程

1. **模型构建阶段**：
   - 用户通过 `Linear`、`LayerNorm`、`Sigmoid` 等基础层构建网络
   - 使用 `Sequential` 或 `ModuleList` 容器组合多个层
   - 调用 `init.h` 中的初始化函数初始化参数

2. **前向传播阶段**：
   - 输入张量通过 `Module::Forward()` 依次经过各层
   - 每层可能调用 `functional.h` 中的函数（如 `Sigmoid`、`Softmax`）进行计算
   - 容器模块（如 `Sequential`）自动串联子模块的前向传播

3. **损失计算阶段**：
   - 最后一层的输出传入 `CrossEntropyLoss`
   - Loss 层计算误差标量，准备反向传播

4. **反向传播与参数更新**：
   - 自动微分系统（`autograd/` 子系统）计算梯度
   - 优化器更新模块的参数（通过 `Module::Parameters()` 收集）

#### 3D 并行训练流程（DP + TP + PP）

假设用户配置 DP=2, TP=4, PP=3，总进程数 24：

1. **初始化阶段**：
   - 调用 `GlobalEnv::Init()` 设置并行拓扑
   - `Layout` 计算每个 rank 的 (dp, tp, pp) 坐标
   - 初始化 `ProcessGroup` 建立通信通道

2. **模型构建阶段**：
   - 使用 `ColumnParallelLinear` 和 `RowParallelLinear` 替换 `Linear`，实现 TP 切分
   - 用 `PipelineParallel` 包装完整模型，自动分割层到不同 PP stage
   - 用 `DistributedDataParallel` 包装（隐式或显式），实现 DP 梯度同步

3. **训练执行阶段**（以一个 global batch 为例）：
   - **PP 维度**：`PipelineParallel` 将 batch 分为多个 micro-batch，按 GPipe 或 1F1B 调度执行
   - **Stage 间通信**：通过 `ISend/IRecv` 异步传递激活和梯度
   - **TP 维度**：每个 stage 内的 `ColumnParallelLinear` 执行 `AllGather`，`RowParallelLinear` 执行 `AllReduce`
   - **DP 维度**：反向传播完成后，`DistributedDataParallel` 触发 `Reducer` 执行桶化 AllReduce

4. **通信优化**：
   - **计算与通信重叠**：异步通信允许在等待数据时执行其他计算
   - **梯度桶化**：`Reducer` 将多个小梯度打包，减少通信次数
   - **Micro-batch 流水线**：PP 提高设备利用率，减少空闲时间

### 3.3 模块间依赖关系

```
                    ┌─────────────────────┐
                    │   Module (module.h) │
                    └──────────┬──────────┘
                               │ 继承
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
    ┌────▼────┐         ┌─────▼──────┐       ┌─────▼──────┐
    │ Linear  │         │ LayerNorm  │       │ Embedding  │
    │ Sigmoid │         │  ...       │       │  ...       │
    └────┬────┘         └─────┬──────┘       └─────┬──────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │ 组合
                    ┌──────────▼──────────┐
                    │  Sequential         │
                    │  ModuleList         │
                    └──────────┬──────────┘
                               │ 包装
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
    ┌────▼────────────┐ ┌─────▼──────────┐  ┌──────▼──────────┐
    │ DataParallel    │ │ TensorParallel │  │ PipelineParallel│
    │ DDP (Reducer)   │ │ (Col/RowLinear)│  │ (PP Scheduler) │
    └─────────────────┘ └────────────────┘  └─────────────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │ 依赖
                    ┌──────────▼──────────┐
                    │ ProcessGroup        │
                    │ GlobalEnv           │
                    │ functional.h        │
                    │ init.h              │
                    └─────────────────────┘
```

### 3.4 设计模式应用

- **抽象工厂模式（Abstract Factory）**：`Module` 定义产品接口，具体层（Linear、Sigmoid）作为具体产品
- **组合模式（Composite Pattern）**：`Module` 同时作为叶子节点和组合节点，支持任意深度嵌套
- **装饰器模式（Decorator Pattern）**：`DataParallel`、`DistributedDataParallel`、`PipelineParallel` 包装原始模块，添加并行能力
- **策略模式（Strategy Pattern）**：`PipelineSchedule` 定义调度策略接口，`GPipe` 和 `1F1B` 作为具体策略
- **模板方法模式（Template Method Pattern）**：`Module` 定义 `Forward()` 模板，子类实现具体逻辑
- **外观模式（Facade Pattern）**：`PipelineParallel`、`DistributedDataParallel` 作为高层入口，隐藏复杂并行逻辑
- **单例模式（Singleton Pattern）**：`GlobalEnv` 管理全局并行环境
- **CRTP（奇异递归模板模式）**：`CloneableModule<T>` 提供自动克隆实现

### 3.5 关键技术实现

#### 参数管理机制
- **命名空间访问**：参数存储在 `std::unordered_map<std::string, std::shared_ptr<Tensor>>` 中，支持字符串键访问
- **递归收集**：`Parameters()` 和 `Buffers()` 递归遍历模块树，扁平化所有参数/缓冲区
- **状态序列化**：`StateDict()` 返回点分隔的层级命名（如 "encoder.layer.0.weight"），兼容 PyTorch 格式
- **参数绑定**：通过 `shared_ptr` 支持跨模块共享参数（如权重绑定）

#### 设备与数据类型抽象
- **设备抽象**：使用 `const Device*` 指针表示计算设备（CUDA/CPU/Kunlun/Metax/Ascend）
- **递归迁移**：`To(device)` 和 `To(dtype)` 递归迁移模块及其所有子模块、参数、缓冲区
- **混合设备训练**：支持模型在 GPU、数据在 CPU 的场景

#### 分布式训练支持
- **数据并行**：每个设备复制完整模型，处理不同数据分片，同步梯度
- **张量并行**：切分模型参数到多个设备，通过 AllGather/AllReduce 协同计算
- **流水线并行**：分割模型层到不同阶段，通过 micro-batch 流水线提高吞吐
- **3D 并行组合**：DP×TP×PP 的混合并行，支持超大规模模型训练（如 GPT-3 级别）

#### 通信优化策略
- **异步通信**：所有通信操作返回 `Work` 句柄，支持计算与通信重叠
- **梯度桶化**：`Reducer` 将多个小梯度打包到一个 bucket，减少通信次数
- **点对点通信**：PP 的 `ISend/IRecv` 采用异步非阻塞语义，隐藏通信延迟
- **分桶调度**：DDP 在反向传播完成后触发异步 AllReduce，不阻塞训练流程

### 3.6 与上层系统集成

`nn` 子系统是 InfiniTrain 的核心模块，与以下系统协同工作：

- **`autograd/` 子系统**：提供自动微分能力，所有 `Module::Forward()` 的操作记录到计算图
- **`optimizer/` 子系统**：通过 `Module::Parameters()` 获取参数，执行梯度下降更新
- **`device/` 子系统**：设备抽象，用于选择通信 backend（NCCL for GPU、MPI for CPU）
- **`tensor/` 子系统**：张量抽象，支持跨设备传输和自动微分
- **`datatype.h`**：数据类型定义（float32、float16、bfloat16），支持混合精度训练

### 3.7 使用场景建议

- **小型模型单机训练**：使用 `modules/` 中的基础层 + `Sequential` 容器
- **中等规模单机多卡**：使用 `DataParallel` 或 `DistributedDataParallel`（仅 DP）
- **超大模型单机**：使用 `TensorParallel`（仅 TP）切分层权重
- **超大模型多机**：使用 `PipelineParallel`（仅 PP）分割层到不同设备
- **超大规模模型**：组合 DP+TP+PP（3D Parallel），如 Megatron-LM 风格的训练

## 4. 架构特色与优势

### 4.1 高度模块化
- 所有组件继承自统一 `Module` 接口，支持自由组合和嵌套
- 容器模块提供灵活的组合模式（顺序、列表、字典）
- 函数式 API 和模块化 API 并存，满足不同使用习惯

### 4.2 完整的并行支持
- 提供数据并行、张量并行、流水线并行三种维度的并行策略
- 支持任意组合的 3D 并行，覆盖从小规模到超大规模的训练场景
- 通过异步通信和计算重叠优化训练效率

### 4.3 PyTorch 兼容性
- API 设计高度模仿 PyTorch，降低用户迁移成本
- 参数命名和状态字典格式兼容，支持跨框架模型加载
- 提供类似的模块接口和容器类型

### 4.4 扩展性与灵活性
- CRTP 模式避免虚函数开销，提供编译期多态
- 插件式的并行策略，可轻松添加新的调度算法或通信后端
- 支持自定义层和损失函数，只需继承 `Module` 基类

### 4.5 生产级特性
- 分布式训练的容错和恢复（通过状态字典保存检查点）
- 梯度累积和混合精度训练支持
- 多硬件后端支持（CUDA、CPU、Kunlun、Metax、Ascend 等）

## 总结

`nn` 子系统通过清晰的分层架构和模块化设计，提供了从基础神经网络层到大规模分布式训练的完整能力。核心抽象层（`Module`）、组件层（`modules/`）、并行层（`parallel/`）和工具层（`functional.h`、`init.h`）各司其职，相互协作，构成了一个灵活、高效、易用的深度学习训练框架。无论是简单的 MLP 还是复杂的 3D 并行 Transformer，都能在该子系统中找到合适的构建模块。
