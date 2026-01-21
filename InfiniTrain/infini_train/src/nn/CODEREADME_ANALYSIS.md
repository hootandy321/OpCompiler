# 📂 目录: nn 架构全景

## 1. 子系统职责

`nn` 目录是 InfiniTrain 框架的神经网络层核心实现，提供模块化的神经网络构建基础设施。该目录在整体架构中承担以下关键职责：

1. **模块化神经网络组件**：通过 `modules` 子系统提供类似 PyTorch 的 Module 基类和具体层实现（Linear、LayerNorm、Embedding 等），支持前向传播、参数管理和设备迁移。

2. **分布式训练并行策略**：通过 `parallel` 子系统实现数据并行、张量并行和流水线并行三种核心并行策略，基于 NCCL 实现高效的 GPU 间通信和梯度同步。

3. **模型组合与容器化**：提供 Sequential、ModuleList、ModuleDict 等容器模块，支持灵活的模型拓扑构建。

4. **自动微分集成**：所有神经网络模块通过委托给 `autograd` 子系统实现可微分的前向传播，无缝集成反向传播和梯度计算。

该目录在 InfiniTrain 框架中处于核心位置，连接下层的基础设施（autograd、tensor、device）和上层的模型应用（transformer、llm 实现），是构建大规模深度学习模型的基石。

## 2. 模块导航

* **📂 modules**:
  * *功能*: 实现神经网络的核心层组件和模块化基础设施，提供 Module 基类、常用层（Linear、LayerNorm、Embedding、Sigmoid、CrossEntropyLoss）以及容器模块（Sequential、ModuleList、ModuleDict）。
  * *职责*: 提供可组合的神经网络构建块，支持参数管理、设备迁移、状态字典维护和数据并行复制，所有模块通过 autograd 实现自动微分。

* **📂 parallel**:
  * *功能*: 实现分布式训练的并行策略，包括数据并行（DataParallel、DistributedDataParallel）、张量并行（ColumnParallelLinear、RowParallelLinear、VocabParallelEmbedding、VocabParallelCrossEntropy）和流水线并行（PipelineParallel 及其调度器和通信原语）。
  * *职责*: 提供高效的 GPU 间通信（基于 NCCL）、梯度同步（Reducer 和梯度桶）、张量分片和流水线调度，支持 DP/TP/PP 混合并行和 3D 并行拓扑布局。

* **📂 parallel/pp**:
  * *功能*: 流水线并行（Pipeline Parallel）的核心实现，包含 PipelineStage（流水线阶段封装）、PipelineSchedule（GPipe 和 1F1B 调度器）和 PipelineParallel（顶层流水线管理器）。
  * *职责*: 实现大规模深度学习模型的流水线切分和调度，通过将模型按层分片到多个 GPU 并在微批次级别交错执行前向和反向传播，实现内存优化和计算并行化。

## 3. 架构逻辑图解

### 3.1 模块层次结构

`nn` 目录采用清晰的分层架构，从底层的通信基础设施到顶层的并行训练接口：

```
nn/
├── modules/           # 基础神经网络层（用户 API 层）
│   ├── Module         # 所有模块的抽象基类
│   ├── Linear, LayerNorm, Embedding  # 具体层实现
│   └── Sequential, ModuleList, ModuleDict  # 容器模块
│
└── parallel/          # 分布式并行基础设施（系统层）
    ├── 进程组管理     # ProcessGroup, ProcessGroupFactory
    ├── 通信原语       # AllReduce, AllGather, ReduceScatter, Send/Recv
    ├── 梯度同步       # Reducer, DistributedDataParallel
    ├── 张量并行       # ColumnParallelLinear, RowParallelLinear
    └── 流水线并行     # PipelineParallel, PipelineSchedule, PipelineStage
```

### 3.2 数据流与交互

#### 3.2.1 前向传播流程

用户通过 `modules` 子系统构建模型，前向传播时的数据流如下：

1. **模型构建阶段**：
   - 用户使用 `Sequential` 容器组合多个 `Module`（如 Linear、LayerNorm）
   - 每个模块在构造函数中初始化参数（`parameters_`）和缓冲区（`buffers_`）
   - 模块通过 `modules_` 映射表管理子模块，形成树形结构

2. **前向传播阶段**：
   - 调用 `model->Forward(input_tensors)` 触发前向传播
   - Sequential 容器迭代执行子模块：`output = module->Forward(input)`
   - 具体层（如 Linear）委托给 `autograd::Linear->Apply(input_tensors)` 实现可微分计算
   - 前向传播过程中自动构建计算图（autograd Function 链）

3. **并行执行阶段**（如果使用并行策略）：
   - **张量并行**：`ColumnParallelLinear` 首先复制输入到各 TP rank（如果需要），执行本地矩阵乘法，然后 AllGather 输出（如果启用）；`RowParallelLinear` 首先 Scatter 输入，执行本地矩阵乘法，然后 AllReduce 输出。
   - **流水线并行**：`PipelineParallel::TrainStep` 将 input 切分为微批次，`PipelineSchedule` 按照 GPipe/1F1B 调度表执行每个 chunk 的前向传播，通过 ISend/IRecv 在相邻 stage 间传递激活值。
   - **数据并行**：`DistributedDataParallel` 调用底层模块的 Forward，同时注册梯度钩子（`PostAccumulateGradHook`），准备反向传播时的梯度同步。

#### 3.2.2 反向传播流程

反向传播通过 autograd 子系统自动触发，并行策略的梯度同步流程如下：

1. **梯度计算阶段**：
   - 调用 `loss->Backward()` 触发反向传播
   - Autograd 系统按照计算图的逆序执行每个 Function 的 Backward 方法

2. **梯度同步阶段**（并行策略介入）：
   - **张量并行**：
     - `CopyToTPRegion::Backward` 对梯度执行 AllReduce（与 Forward 的复制操作对应）
     - `GatherFromTPRegion::Backward` 对梯度执行 Split（与 Forward 的 AllGather 对应）
     - `ScatterToTPRegion::Backward` 对梯度执行 AllGather（与 Forward 的 Split 对应）
     - `ReduceFromTPRegion::Backward` 复制梯度（与 Forward 的 AllReduce 对应）
   - **流水线并行**：
     - `IRecv::Backward` 发送梯度到上一 stage（与 Forward 的接收对应）
     - `ISend::Backward` 从下一 stage 接收梯度（与 Forward 的发送对应）
   - **数据并行**：
     - 每个参数的 `PostAccumulateGradHook` 触发 `Reducer::MarkVariableReadyDense`
     - 梯度被拷贝到对应的梯度桶（`bucket`）或直接作为桶的视图（如果启用 `gradient_as_bucket_view`）
     - 当桶的所有参数梯度就绪时，触发 `Reducer::MarkBucketReady`
     - 按顺序启动桶的 AllReduce（避免乱序完成），等待所有桶完成后调用 `Reducer::FinalizeBackward` 将梯度写回参数

#### 3.2.3 设备与进程组管理

1. **设备管理**：
   - `Module::To(device)` 递归将模块的所有参数和缓冲区迁移到目标设备
   - `Module::To(dtype)` 转换所有参数和缓冲区的数据类型
   - 每个模块维护 `device_` 指针，默认使用 `DeviceManager::Instance()->GetDefaultDevice()`

2. **进程组管理**：
   - `ProcessGroupFactory::Instance()` 单例工厂管理所有进程组（DP、TP、PP）
   - 每个进程组维护 NCCL 通信器（`ncclComm_t`）和独立的通信流（`cudaStream_t`）
   - 通过 `GlobalEnv` 管理 3D 并行拓扑（Layout），将全局 rank 映射到 (dp_rank, tp_rank, pp_rank) 三元组
   - `Rank` 类封装进程级和线程级的 rank 信息，计算全局 rank = `process_rank_ * thread_size_ + thread_rank_`

3. **通信流隔离**：
   - 所有 NCCL 操作在独立的通信流上执行
   - 通过 `cudaEventRecord(compute_stream) -> cudaStreamWaitEvent(comm_stream) -> NCCL op -> cudaEventRecord(comm_stream) -> cudaStreamWaitEvent(compute_stream)` 实现流同步
   - 避免通信阻塞计算，提高吞吐量

### 3.3 并行策略协同

`parallel` 子系统支持 DP/TP/PP 混合并行，三种策略的协同工作流程：

1. **拓扑初始化**：
   - 用户调用 `GlobalEnv::Init(nthread_per_process, tp_size, sequence_parallel, pp_size, vpp_size)` 初始化全局并行环境
   - 自动计算 `data_parallel_size = world_size / (tp_size * pp_size)`
   - 创建 3D 拓扑布局（Layout），支持 `RankOf(dp, tp, pp)` 和 `CoordOf(rank, dp, tp, pp)` 坐标转换

2. **模型构建**：
   - 使用张量并行层（ColumnParallelLinear、RowParallelLinear、VocabParallelEmbedding）替换标准层
   - 将模型按层切分为多个 chunk，每个 chunk 对应一个流水线阶段
   - 使用 PipelineParallel 包装模型，指定 num_stages（PP 大小）和 chunks_per_stage（VPP 大小）

3. **训练执行**：
   - 每次前向传播：
     - TP：在 ColumnParallelLinear 和 RowParallelLinear 中自动执行 AllGather/AllReduce/Scatter/Gather
     - PP：在 PipelineParallel 中调度微批次，通过 ISend/IRecv 在相邻 stage 间传递激活值
     - DP：在 DistributedDataParallel 中通过 Reducer 同步梯度
   - 三种并行策略相互独立，通过不同的进程组（DP、TP、PP）隔离通信

### 3.4 内存优化策略

1. **梯度桶（Bucket）**：
   - Reducer 将参数梯度按大小分组到桶中，减少 AllReduce 启动次数
   - 支持动态桶重建：第一次迭代后根据实际梯度就绪顺序重新分配桶
   - 梯度视图优化：当 `gradient_as_bucket_view=true` 时，参数的 `grad` 直接指向桶内的视图，避免拷贝

2. **微批次切分**：
   - PipelineParallel 将 input 切分为 `num_micro_batches`，减少峰值内存
   - 梯度累积：`loss = loss / n` 在末尾 chunk 反向时平均梯度

3. **Activation 缓存**：
   - PipelineParallel 使用 `activations[vpp_size][n]` 二维数组缓存中间结果供反向传播使用
   - 支持虚拟流水线（Virtual Pipeline Parallelism），每个 stage 包含多个 chunk，进一步优化内存占用

### 3.5 关键设计模式

1. **Composite Pattern**：Module 作为树形结构的基类，支持任意深度的模块嵌套（如 Sequential 中包含 Sequential）

2. **Template Method Pattern**：`Forward()` 定义算法骨架，具体计算委托给 `autograd::` 子系统

3. **CRTP（Curiously Recurring Template Pattern）**：CloneableModule 使用静态多态实现类型安全的复制，避免虚函数开销

4. **Strategy Pattern**：PipelineSchedule 支持多种调度策略（GPipe、1F1B），可扩展新算法

5. **Observer Pattern**：`autograd::PostAccumulateGradHook` 在参数梯度就绪时触发 Reducer 的同步逻辑

6. **Singleton Pattern**：ProcessGroupFactory 和 GlobalEnv 使用 Meyer's Singleton 管理全局状态

7. **Adapter Pattern**：nn::modules 类将 autograd 函数包装为 Module 接口，解耦自动微分与模型构建

### 3.6 与其他子系统的依赖关系

1. **依赖 autograd 子系统**：
   - 所有具体层（Linear、Sigmoid、LayerNorm、Embedding、CrossEntropyLoss）通过 `autograd::` 命名空间下的对应函数类实现前向传播和自动微分
   - 并行通信原语（ISend、IRecv）继承 `autograd::Function`，实现自定义前向/反向逻辑

2. **依赖 tensor 子系统**：
   - Module 的参数和缓冲区是 `std::shared_ptr<Tensor>`
   - 依赖 `Tensor::To(device)`, `Tensor::To(dtype)`, `Tensor::RequiresGrad()` 等方法

3. **依赖 device 子系统**：
   - 使用 `DeviceManager::Instance()->GetDevice("cuda:0")` 获取 CUDA 设备
   - ProcessGroup 封装 NCCL 通信器和 CUDA 流

4. **依赖 nn/functional 子系统**：
   - 使用 `Concat`、`Split`、`MaskedFill` 等函数式操作
   - 用于张量分片和组合（如 VocabParallelCrossEntropy 的 mask 填充）

### 3.7 扩展性与限制

**扩展性**：
- 新增神经网络层：继承 Module 基类，实现 Forward 方法，委托给 autograd 函数
- 新增并行策略：扩展 ProcessGroup 支持新的通信后端（如 MPI、Gloo）
- 新增流水线调度算法：在 PipelineParallelScheduler 中添加新的生成函数

**已知限制**：
- NCCL 依赖：仅支持 CUDA 设备，不支持 CPU
- 单节点限制：当前实现假设所有设备在同一节点（`cudaDeviceCanAccessPeer`）
- ModuleDict Forward：未实现基于字典键的路由逻辑
- 流水线通信：IRecv 在 Forward 前需要预先知道接收张量的形状
- 设备索引未使用：ReplicateForDataParallel(int device_idx) 的 device_idx 参数当前未实际使用

## 4. 总结

`nn` 目录是 InfiniTrain 框架的神经网络层核心实现，通过 `modules` 和 `parallel` 两个子系统协同工作，提供从基础神经网络组件到大规模分布式训练并行策略的完整解决方案。`modules` 子系统提供模块化的神经网络构建块，所有模块通过 autograd 实现自动微分；`parallel` 子系统基于 NCCL 实现数据并行、张量并行和流水线并行，支持 DP/TP/PP 混合并行和 3D 并行拓扑布局。两个子系统通过清晰的接口（Module 基类、ProcessGroup 通信原语）解耦，各自独立可测试，共同支撑起 InfiniTrain 框架的分布式训练能力。
