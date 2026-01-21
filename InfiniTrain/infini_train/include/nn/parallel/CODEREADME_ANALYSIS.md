# 并行训练 (Parallel Training) 模块架构全景

## 1. 子系统职责

`infini_train/include/nn/parallel` 目录是 InfiniTrain 分布式训练框架的**核心并行子系统**，负责实现大规模深度学习模型训练的多种并行策略。该子系统提供了完整的三维并行（3D Parallel）支持，包括数据并行（Data Parallel）、张量并行（Tensor Parallel）和流水线并行（Pipeline Parallel），以及底层的进程组通信和全局并行布局管理。

该子系统的核心职责是：
- **抽象并行策略**：将不同的并行方式封装为独立的模块，用户可根据需求选择或组合使用
- **提供通信原语**：通过 ProcessGroup 和 Work 类提供集合通信（AllReduce、AllGather、ReduceScatter）和点对点通信（ISend/IRecv）的统一接口
- **管理并行拓扑**：通过 GlobalEnv 和 Layout 类管理多维并行的进程拓扑布局，支持 DP-TP-PP 的混合并行
- **支持梯度同步**：通过 DistributedDataParallel 和 Reducer 实现高效的分布式梯度同步，支持桶化（bucket）和异步通信优化

## 2. 模块导航 (Module Navigation)

### 2.1 头文件模块（parallel 目录根）

* **`global.h`**: 全局并行环境与拓扑布局管理
    * *功能*: 提供全局单例 `GlobalEnv` 管理分布式训练的并行配置（包括 DP、TP、PP 维度），定义 `Layout` 结构体描述多维并行拓扑的进程映射关系（rank 到 (dp, tp, pp) 坐标的双向转换）
    * *职责*: 初始化和维护并行训练的全局元数据，提供进程组查询和 rank 坐标转换

* **`process_group.h`**: 进程组通信抽象接口
    * *功能*: 定义 `ProcessGroup` 抽象基类，提供集合通信（AllReduce、AllGather、ReduceScatter、Broadcast）和点对点通信（Send、Recv）的异步接口，返回 `Work` 对象用于同步等待
    * *职责*: 封装底层通信库（NCCL、MPI 等）的差异，为上层并行模块提供统一的通信原语

* **`reducer.h`**: 分布式梯度同步与桶化优化
    * *功能*: 实现 `Reducer` 类，管理分布式训练中的梯度同步。支持梯度桶化（Gradient Bucketing）以减少通信次数，提供异步 AllReduce 和梯度累积优化，与 `DistributedDataParallel` 配合使用
    * *职责*: 高效地同步梯度，通过桶化和异步通信隐藏通信延迟

* **`data_parallel.h`**: 数据并行模块（简单版本）
    * *功能*: 实现 `DataParallel` 类，将输入张量按指定维度切分并分发到多个设备，每个设备独立计算前向传播，最后聚合输出。支持多设备单机场景
    * *职责*: 提供基础的数据并行能力（通常用于单机多卡，分布式场景推荐使用 `DistributedDataParallel`）

* **`distributed_data_parallel.h`**: 分布式数据并行模块
    * *功能*: 实现 `DistributedDataParallel` 类，封装 `Reducer`，在反向传播时自动同步梯度，支持多机多卡的大规模数据并行训练
    * *职责*: 提供生产级的分布式数据并行训练能力，自动处理梯度同步和参数更新

* **`tensor_parallel.h`**: 张量并行模块
    * *功能*: 定义 `ColumnParallelLinear` 和 `RowParallelLinear` 类，实现线性层的按列/按行切分，支持 Sequence Parallel（序列并行）优化。通过 `thread_local int tp_rank` 标识当前线程的 TP rank
    * *职责*: 将大型矩阵乘法操作切分到多个设备，实现模型级别的并行（如 Transformer 的 MLP 和 Attention 层）

* **`parallel_functional.h`**: 并行函数式接口
    * *功能*: 提供并行操作的函数式 API（如 AllReduce、AllGather 等），作为 `ProcessGroup` 的补充或便捷调用接口
    * *职责*: 提供更低层次的并行原语，供高级模块或用户自定义操作使用

* **`rank.h`**: Rank 标识与查询工具
    * *功能*: 定义 rank 相关的辅助函数或常量，可能用于查询当前进程在各个并行维度（DP、TP、PP）的 rank
    * *职责*: 提供并行身份标识的便捷查询

* **`reduce_op_type.h`**: 归约操作类型定义
    * *功能*: 定义归约操作的枚举类型（如 SUM、AVG、MAX、MIN 等），用于 AllReduce 和 ReduceScatter 操作的归约语义
    * *职责*: 标准化归约操作的类型定义

* **`utils.h`**: 并行工具函数
    * *功能*: 提供并行训练相关的通用工具函数（如设备管理、张量分片辅助、通信协调等）
    * *职责*: 收集并行模块共用的辅助功能

* **`work.h`**: 异步通信工作句柄
    * *功能*: 定义 `Work` 类，表示异步通信操作的状态，提供 `Wait()`、`IsCompleted()` 等接口用于同步等待
    * *职责*: 封装异步通信操作的生命周期，支持通信与计算的重叠

### 2.2 子目录模块

* **`pp/`**: 流水线并行（Pipeline Parallel）子模块
    * *功能*: 实现 GPipe 和 Interleaved 1F1B 两种流水线调度策略，将模型层分割到多个阶段（stage），每个阶段在不同设备上执行，通过 micro-batch 的流水线执行实现计算和通信的重叠
    * *职责*:
        - **`pipeline_parallel.h`**: 主入口类 `PipelineParallel`，负责模型分割、stage 构建和训练步骤协调
        - **`pipeline_schedule.h`**: 调度策略抽象类 `PipelineSchedule` 和调度生成器 `PipelineParallelScheduler`，提供 GPipe 和 1F1B 调度算法
        - **`pipeline_stage.h`**: 单个流水线阶段 `PipelineStage`，封装该阶段的模型 chunk、通信接口和优化器
        - **`send_recv.h`**: 跨设备通信原语 `ISend` 和 `IRecv`，用于 stage 间的异步张量传输

## 3. 架构逻辑图解

### 3.1 层次结构

```
parallel/
├── 全局层: global.h, rank.h
│   └── 管理并行拓扑布局、进程 rank 映射、环境初始化
│
├── 通信层: process_group.h, work.h, parallel_functional.h, reduce_op_type.h, utils.h
│   └── 提供统一的集合通信和点对点通信接口，封装底层通信库（NCCL、MPI）
│
├── 策略层: data_parallel.h, distributed_data_parallel.h, tensor_parallel.h, reducer.h
│   ├── DataParallel: 单机多卡数据并行
│   ├── DistributedDataParallel: 分布式数据并行（基于 Reducer）
│   ├── TensorParallel (ColumnParallelLinear, RowParallelLinear): 模型张量切分并行
│   └── Reducer: 梯度桶化和同步优化
│
└── 流水线层: pp/
    ├── PipelineParallel: 流水线并行主控制器
    ├── PipelineSchedule: 调度策略抽象（GPipe、1F1B）
    ├── PipelineStage: 单个流水线阶段封装
    └── ISend/IRecv: stage 间异步通信原语
```

### 3.2 数据流与交互关系

#### 初始化阶段
1. **用户调用 `GlobalEnv::Init()`**，设置 DP、TP、PP 并行度和进程拓扑
2. **`GlobalEnv` 根据 `Layout`** 计算 rank 到 (dp, tp, pp) 坐标的映射关系
3. **初始化 `ProcessGroup`** 实例（如 `ProcessGroupNCCL`），建立设备间的通信通道

#### 训练执行阶段（以 3D 并行为例）

假设用户同时启用 DP=2, TP=4, PP=3，总进程数为 2×4×3=24：

1. **流水线并行（PP）维度**：
   - `PipelineParallel` 将模型层分为 3 个 stage（每个 stage 负责连续的若干层）
   - 每个 PP stage（0, 1, 2）在不同的设备组上执行，通过 `ISend/IRecv` 传递激活和梯度
   - `PipelineSchedule` 按 GPipe 或 1F1B 调度 micro-batch 的前向/反向传播

2. **张量并行（TP）维度**：
   - 在每个 PP stage 内部，如果包含线性层（如 MLP、Attention），使用 `ColumnParallelLinear` 和 `RowParallelLinear` 切分权重矩阵
   - TP=4 表示每个矩阵被切分为 4 份，每个设备持有 1/4 的参数
   - 前向传播时，`ColumnParallelLinear` 先执行 AllGather（如果 input_is_parallel=false），再计算本地矩阵乘；`RowParallelLinear` 计算后执行 AllReduce 聚合输出

3. **数据并行（DP）维度**：
   - 在 (TP, PP) 的每个唯一组合上，DP=2 表示有 2 份模型副本处理不同的数据样本
   - `DistributedDataParallel` 在反向传播后自动触发 `Reducer` 同步梯度
   - `Reducer` 将梯度组织到桶（bucket）中，每个 bucket 执行一次 AllReduce，同步所有 DP rank 的梯度

#### 通信协调

- **跨维度隔离**：DP、TP、PP 三个维度的通信通过不同的 `ProcessGroup` 隔离，避免通信冲突
- **通信与计算重叠**：
  - PP 维度：`ISend/IRecv` 异步传输 stage 间的激活，允许计算与通信重叠
  - TP 维度：`AllGather` 和 `AllReduce` 尽可能采用异步模式，允许后续计算先执行
  - DP 维度：`Reducer` 在反向传播完成后异步执行 AllReduce，不阻塞训练流程

#### 依赖关系

- **`global.h` 是所有并行模块的基础**，提供进程拓扑查询
- **`process_group.h` 是通信的基础设施**，被 `reducer.h`、`tensor_parallel.h`、`pp/send_recv.h` 依赖
- **`reducer.h` 是 `distributed_data_parallel.h` 的核心组件**，提供梯度同步逻辑
- **`pp/` 子模块相对独立**，通过 `send_recv.h` 的通信原语与其他模块解耦
- **`work.h` 提供异步操作的同步机制**，被所有需要等待通信完成的模块使用

### 3.3 设计模式应用

- **策略模式（Strategy Pattern）**：`PipelineSchedule` 定义调度策略接口，`GPipeSchedule` 和 `Interleaved1F1BSchedule` 实现具体算法
- **外观模式（Facade Pattern）**：`PipelineParallel`、`DistributedDataParallel` 作为高层入口，隐藏复杂的并行逻辑
- **单例模式（Singleton Pattern）**：`GlobalEnv` 采用单例模式管理全局并行环境
- **工厂模式（Factory Pattern）**：`PipelineParallelScheduler` 的静态工厂方法生成不同策略的调度任务
- **模板方法模式（Template Method Pattern）**：`PipelineSchedule::Step()` 定义算法骨架，子类实现 `StepMicroBatches()` 具体步骤

### 3.4 内存与性能优化

- **梯度桶化（Gradient Bucketing）**：`Reducer` 将多个小张量的梯度打包到一个 bucket，减少通信次数
- **异步通信**：所有通信操作均返回 `Work` 句柄，支持计算与通信的重叠
- **Micro-batch 流水线**：PP 模块通过将 global batch 分为多个 micro-batch，提高设备利用率
- **虚拟流水线并行（VPP）**：Interleaved 1F1B 调度通过增加每个 stage 的 chunk 数量，减少内存占用（代价是吞吐略降）

### 3.5 与上层系统的集成

`parallel` 子系统是 `InfiniTrain` 框架的核心组件，与以下模块协同工作：
- **`nn/modules/module.h`**：所有并行模块（`DataParallel`、`DistributedDataParallel`、`PipelineParallel`、`ColumnParallelLinear` 等）都继承自 `Module` 基类
- **`autograd/function.h`**：张量并行操作的自动微分函数，确保反向传播正确同步梯度
- **`optimizer.h`**：每个并行 stage 维护独立的优化器实例，更新本地持有的参数
- **`device.h`**：设备抽象，用于选择通信 backend（NCCL for GPU、MPI for CPU）

### 3.6 使用场景建议

- **单机多卡小模型**：使用 `DataParallel` 或 `DistributedDataParallel`（仅 DP）
- **超大模型单机**：使用 `TensorParallel`（仅 TP）切分层权重
- **超大模型多机**：使用 `PipelineParallel`（仅 PP）分割层到不同设备
- **超大模型超大规模**：组合 DP+TP+PP（3D Parallel），如 Megatron-LM 风格的训练

---

## 总结

`parallel` 子系统通过清晰的层次分离和模块化设计，提供了灵活且高效的分布式训练能力。从底层的通信原语到高层的并行策略，每个模块职责明确，支持独立使用或组合使用，覆盖了从小规模单机训练到大规模 3D 并行训练的全场景需求。
