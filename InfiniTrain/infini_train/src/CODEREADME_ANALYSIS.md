# 架构全景: src 目录

## 1. 子系统职责

`src` 目录是 InfiniTrain 训练框架的核心实现层，承载着深度学习训练的完整技术栈。该目录在整体架构中处于承上启下的关键位置：向上为应用层（如 Transformer、LLM 实现）提供模块化的神经网络组件和分布式训练接口，向下依赖基础设施层（tensor、device、dispatcher）实现具体的计算和通信逻辑。

该目录承担以下核心职责：
1. **自动微分引擎**：通过 `autograd` 子系统实现完整的函数式自动微分系统，支持动态计算图构建、梯度累积和分布式训练场景
2. **高性能计算内核**：通过 `kernels` 子系统提供 CPU/CUDA 双后端的深度学习算子库，实现前向传播和反向传播的高性能计算
3. **神经网络组件**：通过 `nn/modules` 子系统提供类似 PyTorch 的模块化神经网络层（Linear、LayerNorm、Embedding 等）和容器（Sequential、ModuleList、ModuleDict）
4. **分布式并行训练**：通过 `nn/parallel` 子系统实现数据并行（DP）、张量并行（TP）、流水线并行（PP）三种核心并行策略，支持 DP/TP/PP 混合并行的 3D 并行拓扑

## 2. 模块导航

### 2.1 autograd - 自动微分引擎

- **功能**: 实现完整的自动微分系统，支持动态计算图构建、梯度累积和分布式训练场景。核心采用函数式自动微分模式，通过 `Function` 基类统一前向传播和反向传播接口，利用 `Dispatcher` 机制实现硬件后端解耦，并提供了 50+ 种可微分算子的实现。
- **职责**: 提供计算图构建机制（Function::Apply）、梯度累积策略（AccumulateGrad）、反向传播执行（BackwardPartial）、梯度模式控制（GradMode）和分布式钩子（AllReducePostAccumulateHook），确保所有算子通过 Dispatcher 自动路由到正确的硬件后端（CPU/CUDA/KUNLUN/METAX 等）。

**核心组件**:
- **Function 基类**: 计算图节点的抽象基类，管理前向传播时的图构建和反向传播时的梯度流传播，维护 saved_tensors、next_functions、grad_outputs 等核心状态
- **AccumulateGrad**: 叶节点张量的梯度累加器，连接计算图和优化器，支持多次反向传播累积梯度到同一张量，提供分布式优化（ConsumeGradOverwriteFlag）和钩子机制（post_accumulate_grad_hook）
- **50+ 算子实现**: 包括线性代数（Linear、Matmul、Outer）、归一化（LayerNorm、Softmax）、损失函数（CrossEntropy）、激活函数（Sigmoid）、通信原语（Scatter、Gather、Broadcast、ReduceAddCoalesced）、逐元素运算（30+ 种算术、三角、比较、逻辑运算）、归约操作（Mean、Sum、Max、Min）、张量操作（Split、IndexGather、Slice、Stack、Concat、Transpose、Tril、Triu、Mask、RepeatInterleave）、稀疏操作（Embedding）

### 2.2 kernels - 计算内核库

- **功能**: 实现所有深度学习训练所需的前向传播和反向传播算子，作为训练框架的基础设施层，向上层提供统一的张量计算接口，同时针对不同硬件平台（CPU/CUDA）提供优化的实现策略。
- **职责**: 提供完整的深度学习算子库（覆盖线性代数、归一化、激活函数、张量操作等核心计算）、硬件加速（通过 CPU 的 Eigen/OpenMP 和 CUDA 的 cuBLAS/CUB 实现跨平台性能优化）、自动微分（每个算子同时实现前向和反向传播，支持梯度反向传播链式法则）、优化器集成（内置梯度累积和 Adam 优化器更新逻辑）、分布式支持（CUDA 后端提供通信原语，支持多 GPU 并行训练）。

#### CPU 后端 (kernels/cpu/)

- **功能**: CPU 算子库实现，提供纯 C++ 编写的深度学习计算内核。
- **职责**: 为无 GPU 环境或 CPU 推理场景提供高效的算子实现，使用 Eigen 库加速矩阵运算（LinearForward、OuterForward），利用 OpenMP 并行化梯度更新（AdamAccumulateGrad），基于 memcpy 的高效内存拷贝（concat、split、stack），Softmax 和 CrossEntropy 使用最大值减法避免指数溢出，LayerNorm 使用 epsilon 保护数值稳定性。

**关键算子分类**:
1. 梯度优化: accumulate_grad.cc（梯度累积、Adam 更新）
2. 线性代数: linear.cc（矩阵乘法、线性变换）、outer.cc（外积）
3. 归一化: layernorm.cc（3D 张量层归一化）、cross_entropy.cc（交叉熵损失）
4. 激活函数: softmax.cc、sigmoid.cc
5. 逐元素运算: elementwise.cc（一元/二元操作、广播机制）
6. 张量操作: concat.cc、split.cc、stack.cc、slice.cc、gather.cc
7. 归约操作: reduction.cc（mean、sum、max、min）
8. 索引与变换: embedding.cc、transform.cc、cast.cc

#### CUDA 后端 (kernels/cuda/)

- **功能**: GPU 算子库实现，提供 CUDA 核函数实现的深度学习计算内核。
- **职责**: 为 NVIDIA GPU 提供高性能并行计算实现，是训练性能的关键优化层。采用 256 线程/块的网格配置，大规模张量使用 grid-stride loop，集成 cuBLAS（矩阵乘法）和 CUB（并行归约）性能优化库，使用 cudaMallocAsync/cudaFreeAsync 实现流有序内存分配，EmbeddingBackward 使用 atomicAdd 处理重复 token 梯度累积，Elementwise BinaryBackward 对 BF16/half 使用 fastAtomicAdd。

**关键算子分类**:
1. 梯度优化: accumulate_grad.cu（梯度累积、Adam 更新，使用 FMA 指令）
2. 线性代数: linear.cu（批量矩阵乘法 cuBLAS GEMM）、outer.cu
3. 归一化: layernorm.cu（CUB BlockReduce 并行归约）、cross_entropy.cu
4. 激活函数: softmax.cu（2D 网格并行化，每个线程块计算一个 softmax）
5. 逐元素运算: elementwise.cu（支持广播，低精度类型使用三阶段策略优化）
6. 张量操作: concat.cu（二分查找定位源张量）、split.cu、stack.cu、slice.cu
7. 归约操作: reduction.cu（通用归约框架，支持 sum/mean/max/min）
8. 索引与变换: embedding.cu、gather.cu、transform.cu（转置、三角掩码、掩码、重复插值）
9. 通信原语: comm.cu（broadcast、scatter、gather、reduce_add_coalesced）
10. 分布式专用: vocab_parallel_cross_entropy.cu（词汇表并行交叉熵）

### 2.3 nn/modules - 神经网络模块

- **功能**: 实现神经网络的核心层组件和模块化基础设施，提供 Module 基类、常用层（Linear、LayerNorm、Embedding、Sigmoid、CrossEntropyLoss）以及容器模块（Sequential、ModuleList、ModuleDict）。
- **职责**: 提供可组合的神经网络构建块，支持参数管理（Parameters 方法递归收集本模块及所有子模块的参数，使用 unordered_set 防止重复收集）、设备迁移（To 方法递归将模块的所有参数和缓冲区迁移到目标设备）、状态字典维护（StateDict 构建完整的状态字典，包含所有参数和 buffers，跳过名称以 "__pp" 开头的子模块用于 Pipeline Parallel）和数据并行复制（ReplicateForDataParallel），所有模块通过 autograd 实现自动微分。

**核心组件**:
- **Module 基类**: 所有神经网络模块的抽象基类，提供参数管理、设备管理、模块组合的核心基础设施，支持递归收集参数和 buffers、状态字典保存/加载、设备迁移、树形模块遍历（Apply 方法深度优先遍历所有子模块并应用函数）
- **CloneableModule<Derived>**: CRTP (Curiously Recurring Template Pattern) 模板基类，为派生类提供类型安全的复制功能，避免虚函数开销
- **具体层实现**:
  - **Linear**: 全连接层，实现仿射变换 `y = xA^T + b`，支持可选偏置项，权重使用 Kaiming Uniform (He 初始化) 适用于 ReLU 激活函数
  - **LayerNorm**: 层归一化，实现 `y = (x - mean) / sqrt(var + eps) * gamma + beta`，在特征维度上归一化，weight 初始化为 1（保持原始方差），bias 初始化为 0（保持原始均值）
  - **Embedding**: 稀疏嵌入查找表，用于词嵌入和类别特征，实现离散索引到连续向量的查找表
  - **Sigmoid**: 激活函数层，实现 `sigmoid(x) = 1 / (1 + e^{-x})`
  - **CrossEntropyLoss**: 交叉熵损失函数，实现 `H(p, q) = -sum(p_i * log(q_i))`，用于分类任务
- **容器模块**:
  - **Sequential**: 顺序容器，按顺序执行子模块，前一个模块的输出是后一个模块的输入
  - **ModuleList**: 可迭代模块列表，支持索引访问，但不在 Forward 中自动执行
  - **ModuleDict**: 字典容器，通过字符串键访问子模块

### 2.4 nn/parallel - 分布式并行训练

- **功能**: 实现分布式训练的并行策略，包括数据并行（DataParallel、DistributedDataParallel）、张量并行（ColumnParallelLinear、RowParallelLinear、VocabParallelEmbedding、VocabParallelCrossEntropy）和流水线并行（PipelineParallel 及其调度器和通信原语）。
- **职责**: 提供高效的 GPU 间通信（基于 NCCL）、梯度同步（Reducer 和梯度桶）、张量分片和流水线调度，支持 DP/TP/PP 混合并行和 3D 并行拓扑布局。

**核心组件**:
- **进程组管理**: ProcessGroup、ProcessGroupFactory 封装 NCCL 通信器（ncclComm_t）和独立的通信流（cudaStream_t），提供 AllReduce、AllGather、ReduceScatter、Send、Recv、Broadcast、Scatter、Gather、ReduceAddCoalesced 等通信原语，通过 GlobalEnv 管理 3D 并行拓扑（Layout），将全局 rank 映射到 (dp_rank, tp_rank, pp_rank) 三元组
- **数据并行**:
  - **DataParallel**: 单机数据并行，通过模块复制和多线程实现，调用 function::Scatter 将输入散射到各设备，调用 function::Replicate 复制模块到各设备，在多线程中并行执行 Forward，调用 function::Gather 将输出收集到 output_device
  - **DistributedDataParallel**: 分布式数据并行包装器，整合 Reducer 实现梯度同步，支持梯度桶（bucket）管理和动态重建策略，提供梯度视图优化（gradient_as_bucket_view）减少拷贝开销
- **张量并行**:
  - **ColumnParallelLinear**: 列并行线性层，权重按列分割（每个 rank 拥有 `out_features / tp_size` 列），输入需要复制（Copy）或已经在各 rank 上分片（input_is_parallel），输出可选 Gather（gather_output 参数），支持 Sequence Parallel 和 skip_bias_add（fused kernel 优化）
  - **RowParallelLinear**: 行并行线性层，权重按行分割（每个 rank 拥有 `in_features / tp_size` 行），输入需要 Scatter 到各 rank（除非 input_is_parallel=true），输出需要 AllReduce（reduce_output 参数）或 ReduceScatter（sequence_parallel=true）
  - **VocabParallelEmbedding**: 词汇表并行的嵌入层，按词表分割嵌入矩阵，前向传播通过 mask + 偏移 + 本地查找 + AllReduce 实现
  - **VocabParallelCrossEntropy**: 并行的交叉熵损失，支持 label smoothing，通过 Mask 填充 + 全局最大值计算 + 数值稳定性处理 + softmax 计算 + 预测 logit 提取 + 损失计算实现
- **流水线并行**: 见 nn/parallel/pp 子目录

### 2.5 nn/parallel/pp - 流水线并行实现

- **功能**: 流水线并行（Pipeline Parallel）的核心实现，包含 PipelineStage（流水线阶段封装）、PipelineSchedule（GPipe 和 1F1B 调度器）和 PipelineParallel（顶层流水线管理器）。
- **职责**: 实现大规模深度学习模型的流水线切分和调度，通过将模型按层分片到多个 GPU 并在微批次级别交错执行前向和反向传播，实现内存优化和计算并行化。

**核心组件**:
- **PipelineParallel**: 流水线并行的顶层管理类，负责模块切分、Stage 构建、调度器初始化，将完整模型按 `kPPChunkNamePrefix + chunk_id` 切分 chunks，为首尾 stage 添加 `kPPFirstStageName` / `kPPLastStageName` 模块
- **PipelineStage**: 单个流水线阶段的执行引擎，封装模型 chunks、设备、优化器、邻居 rank 信息，维护 stage_index（当前 stage 的 rank）、prev_rank/next_rank（相邻 stage 的 rank）、chunks（当前 stage 持有的模型 chunk 列表，Sequential 容器）、recv_shape（接收张量的形状信息）
- **PipelineSchedule**: 执行流水线调度，按照生成的 Task 序列协调前向/反向传播和跨阶段通信，管理 activations 缓存（vpp_size x n 矩阵），调用 ReceiveFromPrev（通过 IRecv 从上一 stage 接收张量）和 SendToNext（通过 ISend 向下一 stage 发送张量）
- **PipelineParallelScheduler**: 静态工具类，生成不同调度策略的任务序列：
  - **GPipe Schedule**: 阶段为 Warmup（仅前向） -> Steady（前向+后向） -> Cooldown（仅后向），总步数为 `2 * (n + num_stages * vpp_size - 1)`，特点：简单易实现，但内存占用高
  - **1F1B Schedule** (Interleaved 1F1B): 阶段为 Warmup（仅前向） -> Steady（每步前向+后向） -> Cooldown（仅后向），总步数为 `2 * (num_stages * vpp_size - 1) + n`，特点：内存效率更高，适合大模型训练
- **ISend / IRecv**: 基于 Autograd 的异步通信原语，前向发送/接收张量，反向自动交换梯度，继承 autograd::Function，自动微分集成（通信操作纳入计算图，反向时自动交换梯度），异步执行（Send/Recv 的 blocking=false 参数实现非阻塞通信）

**调度执行细节**:
- 将 input/target 切分为 microbatches，调用 optimizer->ZeroGrad/Step
- 按照 schedule 表执行每个 Task，管理 activations 缓存
- 遍历 schedule，仅执行 task.stage_id == stage_idx 的任务
- Forward 时：如果是首 chunk 则使用 microbatch input，否则从上一 stage 接收
- Backward 时：如果是末尾 chunk 则计算 loss 并反向传播，否则使用 dummy gradient

## 3. 架构逻辑图解

### 3.1 训练数据流全景图

```
训练输入数据 (Input/Target)
     ↓
[nn/modules: 模型构建]
     ├─ Sequential 容器组合多个 Module
     ├─ Linear, LayerNorm, Embedding, Sigmoid 等具体层
     └─ 所有层通过 autograd::Function 实现可微分前向传播
     ↓
[autograd: 计算图构建]
     ├─ Function::Apply 执行前向传播
     ├─ 如果 GradMode::IsEnabled()，为输出张量设置 grad_fn
     ├─ 构建计算图节点链（next_functions_）
     └─ SetupContext 保存反向传播所需张量（saved_tensors_）
     ↓
[kernels: 硬件加速计算]
     ├─ Dispatcher 根据 DeviceType 自动路由到对应后端
     ├─ CPU 后端: Eigen/OpenMP 优化，串行/并行混合
     └─ CUDA 后端: cuBLAS/CUB 加速，256 线程/块并行
     ↓
损失标量 (Loss Scalar)
     ↓
[nn/parallel: 分布式训练介入点]
     ├─ 数据并行 (DP):
     │    ├─ DistributedDataParallel 包装模型
     │    ├─ Reducer 管理梯度桶和 AllReduce 同步
     │    └─ 所有 DP rank 独立训练，梯度同步
     ├─ 张量并行 (TP):
     │    ├─ ColumnParallelLinear: AllGather 输入 → Forward → ReduceScatter 输出
     │    ├─ RowParallelLinear: AllGather 输入 → Forward → AllReduce 输出
     │    └─ VocabParallelEmbedding: 本地查找 + AllReduce
     └─ 流水线并行 (PP):
          ├─ PipelineParallel 将模型切分到多个 GPU
          ├─ PipelineSchedule 按照 GPipe/1F1B 调度微批次
          └─ ISend/IRecv 在相邻 stage 间传递激活值
     ↓
[autograd: 反向传播触发]
     ├─ loss->Backward() 触发反向传播
     ├─ 按照计算图的逆序执行每个 Function 的 Backward 方法
     └─ Function::BackwardPartial 累积多输出路径的梯度
     ↓
梯度张量 (Gradients)
     ↓
[梯度同步阶段 (并行策略介入)]
     ├─ 数据并行: Reducer->FinalizeBackward 等待所有桶 AllReduce 完成
     ├─ 张量并行:
     │    ├─ CopyToTPRegion::Backward: AllReduce 梯度
     │    ├─ GatherFromTPRegion::Backward: Split 梯度
     │    └─ ScatterToTPRegion::Backward: AllGather 梯度
     └─ 流水线并行:
          ├─ IRecv::Backward: 发送梯度到上一 stage
          └─ ISend::Backward: 从下一 stage 接收梯度
     ↓
[autograd: 梯度累积]
     ├─ AccumulateGrad::Backward 累积梯度到叶节点张量
     ├─ 支持多次反向传播累积梯度（适用于小批量训练）
     ├─ 调用 post_accumulate_grad_hook（如 AllReducePostAccumulateHook）
     └─ 调用 ResetAccumulator 清空累加器状态
     ↓
[优化器更新]
     ├─ kernels::AdamAccumulateGrad 更新参数
     ├─ m = beta1 * m + (1 - beta1) * grad
     ├─ v = beta2 * v + (1 - beta2) * grad^2
     ├─ m_hat = m / (1 - beta1^t)
     ├─ v_hat = v / (1 - beta2^t)
     └─ param -= lr * m_hat / (sqrt(v_hat) + eps)
     ↓
参数更新 (Parameters Updated)
```

### 3.2 模块层次结构与依赖关系

```
src/
├── autograd/              # 自动微分引擎（核心基础层）
│   ├── function.cc       # Function 基类，计算图节点抽象
│   ├── accumulate.cc     # 梯度累加器，连接计算图和优化器
│   ├── grad_mode.cc      # 全局梯度模式控制（thread_local）
│   ├── function_hook.cc  # 梯度后处理钩子（AllReduce 同步）
│   └── 50+ 算子实现      # 所有算子通过 Dispatcher 路由到硬件后端
│
├── kernels/               # 计算内核库（硬件加速层）
│   ├── cpu/              # CPU 后端
│   │   ├── accumulate_grad.cc  # 梯度累积、Adam 更新（OpenMP 并行）
│   │   ├── linear.cc            # Eigen 矩阵乘法
│   │   ├── elementwise.cc       # 逐元素运算（广播机制）
│   │   ├── layernorm.cc         # 3D 张量层归一化
│   │   ├── softmax.cc           # 数值稳定版本
│   │   ├── concat.cc            # memcpy 批量传输
│   │   └── ...
│   └── cuda/             # CUDA 后端
│       ├── accumulate_grad.cu  # FMA 指令优化
│       ├── linear.cu            # cuBLAS GEMM
│       ├── elementwise.cu       # 三阶段策略优化 BF16/half
│       ├── layernorm.cu         # CUB BlockReduce 并行归约
│       ├── softmax.cu           # 2D 网格并行化
│       ├── concat.cu            # 二分查找定位
│       ├── vocab_parallel_cross_entropy.cu  # 分布式专用
│       └── ...
│
└── nn/                    # 神经网络组件（应用层）
    ├── modules/          # 基础神经网络层（用户 API 层）
    │   ├── Module         # 所有模块的抽象基类
    │   ├── Linear, LayerNorm, Embedding  # 具体层实现
    │   ├── Sequential, ModuleList, ModuleDict  # 容器模块
    │   └── 所有层委托给 autograd:: 函数实现前向传播
    │
    └── parallel/         # 分布式并行基础设施（系统层）
        ├── 进程组管理     # ProcessGroup, ProcessGroupFactory
        ├── 通信原语       # AllReduce, AllGather, ReduceScatter, Send/Recv
        ├── 梯度同步       # Reducer, DistributedDataParallel
        ├── 张量并行       # ColumnParallelLinear, RowParallelLinear
        └── 流水线并行     # PipelineParallel, PipelineSchedule, PipelineStage
```

**依赖关系链**:
1. **nn/modules** 依赖 **autograd**: 所有具体层（Linear、Sigmoid、LayerNorm、Embedding、CrossEntropyLoss）通过 `autograd::` 命名空间下的对应函数类实现前向传播和自动微分
2. **autograd** 依赖 **kernels**: 所有算子通过 Dispatcher 调用 kernels 层的硬件后端实现（Forward/Backward kernel）
3. **kernels** 依赖 **基础设施**: Tensor 类（张量数据结构）、Device 类（设备抽象）、Dispatcher（类型分发和算子注册）
4. **nn/parallel** 依赖 **nn/modules** 和 **autograd**: 并行层（ColumnParallelLinear、RowParallelLinear）继承自 nn::modules::Linear，通信原语（ISend、IRecv）继承自 autograd::Function

### 3.3 并行策略协同工作流

```
[全局环境初始化]
     ↓
GlobalEnv::Init(nthread_per_process, tp_size, sequence_parallel, pp_size, vpp_size)
     ↓
计算 data_parallel_size = world_size / (tp_size * pp_size)
     ↓
创建 3D 拓扑布局（Layout），支持 RankOf(dp, tp, pp) 和 CoordOf(rank, dp, tp, pp) 坐标转换
     ↓
[模型构建]
     ↓
使用张量并行层（ColumnParallelLinear、RowParallelLinear、VocabParallelEmbedding）替换标准层
     ↓
将模型按层切分为多个 chunk，每个 chunk 对应一个流水线阶段
     ↓
使用 PipelineParallel 包装模型，指定 num_stages（PP 大小）和 chunks_per_stage（VPP 大小）
     ↓
使用 DistributedDataParallel 包装模型（如果启用 DP）
     ↓
[训练执行]
     ↓
每次前向传播:
     ├─ TP: 在 ColumnParallelLinear 和 RowParallelLinear 中自动执行 AllGather/AllReduce/Scatter/Gather
     ├─ PP: 在 PipelineParallel 中调度微批次，通过 ISend/IRecv 在相邻 stage 间传递激活值
     └─ DP: 在 DistributedDataParallel 中通过 Reducer 同步梯度
     ↓
三种并行策略相互独立，通过不同的进程组（DP、TP、PP）隔离通信
```

### 3.4 数值精度管理策略

**类型转换策略**:
- CPU 后端: 主要支持 float32，cast 算子提供类型转换
- CUDA 后端: 支持 float32 和 bfloat16，低精度算子使用类型提升（WidestType_t）
- 计算-存储分离: BF16 存储权重，FP32 进行累积（cuBLAS compute precision CUDA_R_32F）

**梯度精度**:
- CPU: 统一使用 float32 梯度
- CUDA: BF16/half 反向传播使用三阶段策略（无广播/直方图/块归约），确保梯度精度

**数值稳定性保证**:
- Softmax/CrossEntropy: 使用最大值减法避免 exp 溢出
- LayerNorm: 使用 epsilon 保护（`rsqrt(var + eps)`）
- Adam: 使用偏差修正（`m_hat = m / (1 - beta1^t)`）

### 3.5 内存优化策略

1. **梯度桶（Bucket）**: Reducer 将参数梯度按大小分组到桶中，减少 AllReduce 启动次数，支持动态桶重建（第一次迭代后根据实际梯度就绪顺序重新分配桶），梯度视图优化（当 `gradient_as_bucket_view=true` 时，参数的 `grad` 直接指向桶内的视图，避免拷贝）
2. **微批次切分**: PipelineParallel 将 input 切分为 `num_micro_batches`，减少峰值内存，梯度累积（`loss = loss / n` 在末尾 chunk 反向时平均梯度）
3. **Activation 缓存**: PipelineParallel 使用 `activations[vpp_size][n]` 二维数组缓存中间结果供反向传播使用，支持虚拟流水线（Virtual Pipeline Parallelism），每个 stage 包含多个 chunk，进一步优化内存占用
4. **内存复用**: 前向传播缓存中间结果（mean、rstd、softmax 输出）供反向使用，反向传播完成后立即清理 `saved_tensors_` 和 `grad_outputs_`，节省内存
5. **稀疏梯度**: EmbeddingBackward 使用原子操作仅更新访问过的 token 梯度

### 3.6 关键设计模式

1. **Composite Pattern**: Module 作为树形结构的基类，支持任意深度的模块嵌套（如 Sequential 中包含 Sequential）
2. **Template Method Pattern**: `Forward()` 定义算法骨架，具体计算委托给 `autograd::` 子系统
3. **CRTP（Curiously Recurring Template Pattern）**: CloneableModule 使用静态多态实现类型安全的复制，避免虚函数开销
4. **Strategy Pattern**: PipelineSchedule 支持多种调度策略（GPipe、1F1B），可扩展新算法
5. **Observer Pattern**: `autograd::PostAccumulateGradHook` 在参数梯度就绪时触发 Reducer 的同步逻辑
6. **Singleton Pattern**: ProcessGroupFactory 和 GlobalEnv 使用 Meyer's Singleton 管理全局状态
7. **Adapter Pattern**: nn::modules 类将 autograd 函数包装为 Module 接口，解耦自动微分与模型构建
8. **RAII**: `NoGradGuard` 使用构造/析构函数自动管理梯度模式，`WorkNccl` 析构时自动销毁 CUDA 事件

## 4. 扩展性与限制

### 4.1 扩展性

- **新增神经网络层**: 继承 Module 基类，实现 Forward 方法，委托给 autograd 函数
- **新增并行策略**: 扩展 ProcessGroup 支持新的通信后端（如 MPI、Gloo）
- **新增流水线调度算法**: 在 PipelineParallelScheduler 中添加新的生成函数
- **添加新算子**: 继承 `Function`，实现 `Forward`, `SetupContext`, `Backward` 三个虚函数
- **添加新硬件**: 在 Dispatcher 中注册新的 kernel 实现，无需修改 `autograd` 层代码
- **自定义梯度**: 重写算子的 `Backward` 方法实现特定梯度计算逻辑
- **钩子扩展**: 实现自定义 `FunctionHook` 子类，支持梯度后处理（如量化、裁剪）

### 4.2 已知限制

- **NCCL 依赖**: 仅支持 CUDA 设备，不支持 CPU
- **单节点限制**: 当前实现假设所有设备在同一节点（`cudaDeviceCanAccessPeer`）
- **CPU 后端限制**: LayerNorm 仅支持 3D 张量 [bs, seq_len, embed_dim]，广播仅支持单向（低维 → 高维），Matmul 使用朴素实现，未优化
- **ModuleDict Forward**: 未实现基于字典键的路由逻辑
- **流水线通信**: IRecv 在 Forward 前需要预先知道接收张量的形状
- **设备索引未使用**: ReplicateForDataParallel(int device_idx) 的 device_idx 参数当前未实际使用
- **ISend/IRecv 数据类型限制**: 当前硬编码为 kFLOAT32（需要改进以支持半精度通信）

## 5. 总结

`src` 目录是 InfiniTrain 训练框架的核心实现层，通过 `autograd`、`kernels`、`nn/modules` 和 `nn/parallel` 四个子系统协同工作，提供从自动微分引擎到分布式并行训练的完整解决方案。`autograd` 子系统实现函数式自动微分，提供计算图构建和梯度反向传播能力；`kernels` 子系统通过 CPU/CUDA 双后端提供高性能计算内核；`nn/modules` 子系统提供模块化的神经网络构建块；`nn/parallel` 子系统基于 NCCL 实现数据并行、张量并行和流水线并行，支持 DP/TP/PP 混合并行和 3D 并行拓扑布局。四个子系统通过清晰的接口（Function 基类、Module 基类、ProcessGroup 通信原语、Dispatcher 算子调度）解耦，各自独立可测试，共同支撑起 InfiniTrain 框架的大规模分布式训练能力。
