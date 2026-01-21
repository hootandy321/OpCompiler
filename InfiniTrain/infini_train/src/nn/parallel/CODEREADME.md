# Parallel 模块核心实现文档

本模块实现了 InfiniTrain 框架的分布式训练并行策略，包括数据并行（Data Parallel）、张量并行（Tensor Parallel）、流水线并行（Pipeline Parallel）以及混合并行策略。模块基于 NCCL 进行 GPU 间通信，提供完整的梯度同步、张量分片和流水线调度功能。

## 1. 模块结构

- **`work.cc`**: 异步通信工作对象（WorkNccl）的实现，封装 NCCL 操作的事件同步机制
- **`utils.cc`**: 并行工具函数，提供获取各并行组名称和 rank 列表的接口
- **`tensor_parallel.cc`**: 张量并行核心实现，包含 ColumnParallelLinear、RowParallelLinear、VocabParallelEmbedding、VocabParallelCrossEntropy 等模块
- **`reducer.cc`**: 梯度Reducer，实现梯度桶（bucket）管理和动态重建策略，支持 AllReduce 梯度同步
- **`process_group.cc`**: 进程组（ProcessGroup）实现，封装 NCCL 通信原语（AllReduce、AllGather、ReduceScatter、Send、Recv、Broadcast、Scatter、Gather）
- **`rank.cc`**: Rank 管理，实现进程级和线程级的 rank 编号与转换
- **`parallel_functional.cc`**: 并行功能函数，提供 Scatter、Gather、Broadcast、Replicate 等张量操作
- **`global.cc`**: 全局并行环境配置，管理 DP/TP/PP 混合并行的拓扑布局（Layout）
- **`distributed_data_parallel.cc`**: 分布式数据并行（DDP）模块，整合 Reducer 实现梯度同步
- **`data_parallel.cc`**: 单机数据并行（DataParallel），通过模块复制和多线程实现
- **`pp/send_recv.cc`**: 流水线并行的通信原语，实现 ISend/IRecv 的 autograd 函数
- **`pp/pipeline_stage.cc`**: 流水线阶段（PipelineStage）封装，管理单个阶段的模型分片和优化器
- **`pp/pipeline_schedule.cc`**: 流水线调度器，实现 GPipe 和 1F1B 两种调度策略
- **`pp/pipeline_parallel.cc`**: 流水线并行（PipelineParallel）模块，构建和管理整个流水线模型

## 2. 核心类

### 2.1 WorkNccl
- **位置**: `work.cc`
- **主要功能**: 封装 NCCL 异步通信操作，提供同步和异步等待接口
- **关键成员**:
  - `ready_event_`: CUDA 事件，标记输入数据就绪
  - `done_event_`: CUDA 事件，标记通信完成
  - `completed_`: `std::atomic<bool>`，操作完成标志
  - `success_`: `std::atomic<bool>`，操作成功标志
  - `exception_`: `std::exception_ptr`，保存异常信息
- **核心方法**:
  - `WaitBlocking(timeout)`: 阻塞等待，支持超时机制。使用 `cudaEventSynchronize` 等待 done_event，并通过 `ncclCommGetAsyncError` 检查 NCCL 异步错误
  - `WaitNonBlocking()`: 非阻塞等待，使用 `cudaStreamWaitEvent` 将通信事件插入计算流
  - `IsCompleted()`: 查询操作状态，使用 `cudaEventQuery` 轮询完成状态
  - `CheckNcclStatus()`: 检查 NCCL 通信器的异步错误状态
- **生命周期**: 由 ProcessGroupNCCL 创建，在通信操作完成后由调用方管理生命周期

### 2.2 ProcessGroup / ProcessGroupNCCL
- **位置**: `process_group.cc`
- **主要功能**: 封装进程组通信语义，管理 NCCL 通信器和 CUDA 流
- **关键成员**:
  - `world_size_`: 进程组大小
  - `name_`: 进程组名称（如 "DP0", "TP1", "PP2"）
  - `comms_`: `std::vector<ncclComm_t>`，每个设备的 NCCL 通信器
  - `devices_`: `std::vector<const Device*>`，进程组内的设备列表
  - `device_comm_map_`: `std::unordered_map<const Device*, ncclComm_t>`，设备到通信器的映射
  - `comm_streams_`: `std::vector<cudaStream_t>`，通信流（独立于计算流）
  - `global_group_rank_map_`: `std::unordered_map<int, int>`，全局 rank 到组内 rank 的映射
- **核心方法**:
  - `AllReduce(tensor, reduce_op, async_op)`: 执行 AllReduce 操作，使用 `ncclAllReduce`，通过事件同步计算流和通信流
  - `AllGather(output, input, async_op)`: 执行 AllGather 操作，沿维度 0 收集张量
  - `ReduceScatter(output, input, reduce_op, async_op)`: 执行 ReduceScatter 操作，先规约后分散
  - `Send(tensors, dest_rank, async_op)`: 点对点发送，支持多张量连续发送
  - `Recv(tensors, src_rank, async_op)`: 点对点接收
  - `BroadCast(input_tensors)`: 广播操作，将根设备的数据广播到组内所有设备
  - `Scatter(tensor, devices, dim)`: 散射操作，将张量分片发送到不同设备
  - `Gather(tensors, destination, dim)`: 收集操作，从多设备收集张量到目标设备
  - `ReduceAddCoalesced(grads, destination)`: 合并规约，对多个梯度张量执行 ReduceAdd
- **初始化流程**:
  1. 单进程模式：调用 `ncclCommInitAll` 初始化所有通信器
  2. 多进程模式：
     - 主进程生成 `ncclUniqueId` 并写入文件
     - 其他进程读取 `ncclUniqueId`
     - 使用 `ncclCommInitRank` 初始化每个设备的通信器
  3. 创建高优先级的通信流（`cudaStreamCreateWithPriority`）
- **通信流隔离**: 所有通信操作在独立的通信流上执行，通过 `cudaEventRecord` + `cudaStreamWaitEvent` 与计算流同步

### 2.3 ProcessGroupFactory
- **位置**: `process_group.cc`
- **主要功能**: 单例工厂，管理所有进程组的创建和获取
- **关键成员**:
  - `name_to_group_`: `std::unordered_map<std::string, std::unique_ptr<ProcessGroup>>`，进程组名称到进程组的映射
  - `mutex_`: `std::mutex`，保护进程组注册表
- **核心方法**:
  - `Instance()`: 获取单例实例，使用 Meyer's Singleton 实现
  - `GetOrCreate(name, device_indices)`: 根据设备索引列表创建或获取进程组
  - `Get(name)`: 根据名称获取已注册的进程组
  - `GetDefaultProcessGroup()`: 获取默认进程组（包含所有设备）

### 2.4 Reducer
- **位置**: `reducer.cc`
- **主要功能**: 管理梯度桶（bucket）和 AllReduce 操作，支持动态桶重建和梯度视图优化
- **关键成员**:
  - `params_`: `std::vector<std::shared_ptr<Tensor>>`，需要同步的参数列表
  - `buckets_`: `std::vector<Bucket>`，梯度桶列表
  - `locators_`: `std::vector<BucketLocator>`，参数到桶位置的映射（桶索引 + 桶内索引）
  - `opts_`: `ReducerOptions`，配置选项（桶大小、是否启用梯度视图等）
  - `next_bucket_`: `size_t`，下一个待处理的桶索引
  - `grad_ready_order_indices_`: `std::vector<size_t>`，梯度就绪顺序记录
  - `comm_hook_`: `std::shared_ptr<autograd::PostAccumulateGradHook>`，自定义通信钩子
- **核心方法**:
  - `BuildBuckets(bucket_indices)`: 根据参数索引构建梯度桶，每个桶包含：
    - `variables`: 桶内的参数张量
    - `contents`: 扁平化的梯度存储（1D tensor）
    - `offsets`: 每个参数在 contents 中的偏移量（元素个数）
    - `lengths`: 每个参数的元素个数
    - `bucket_views_in/out`: 梯度视图（避免拷贝）
  - `ComputeBucketAssignmentBySize(...)`: 静态方法，根据参数大小和容量限制分配桶
    - 支持多级容量限制（首桶、普通桶）
    - 按设备和数据类型分组
    - 默认按反向顺序分配（接近梯度就绪顺序）
  - `RebuildBuckets()`: 根据实际梯度就绪顺序动态重建桶，第一次迭代后触发
  - `MarkVariableReadyDense(variable_index)`: 标记参数梯度就绪
    - 将参数梯度拷贝到桶中（如果未启用梯度视图）
    - 递减桶的 pending 计数
    - 当 pending == 0 时，调用 `MarkBucketReady` 启动 AllReduce
  - `MarkBucketReady(bucket_index)`: 按顺序启动桶的 AllReduce
    - 仅当 `bucket_index == next_bucket_` 时启动
    - 连续启动所有就绪的桶
    - 所有桶完成后触发 `FinalizeBackward`
  - `FinalizeBucketDense(bucket_index)`: 对单个桶执行 AllReduce（kAvg 操作）
  - `FinalizeBackward()`: 等待所有 AllReduce 完成，将梯度写回参数
    - 使用 `WaitNonBlocking` 避免主机同步
  - `PrepareForBackward()`: 每次前向传播后准备反向传播
    - 重置 `next_bucket_` 和桶的 pending 计数
    - 如果启用 `gradient_as_bucket_view`，将参数的梯度直接设置为桶的视图
- **同步机制**:
  - 使用 `std::mutex` 保护桶状态
  - 使用 `std::atomic<size_t>` 记录已完成的桶数量
  - 桶的 AllReduce 按顺序启动，避免乱序完成导致的梯度覆盖问题
- **梯度视图优化**: 当 `gradient_as_bucket_view=true` 时，参数的梯度直接指向桶内的内存视图，减少拷贝开销

### 2.5 ColumnParallelLinear / RowParallelLinear
- **位置**: `tensor_parallel.cc`
- **主要功能**: 实现张量并行的线性层，按列或行分割权重矩阵
- **ColumnParallelLinear 特性**:
  - 权重按列分割：每个 rank 拥有 `out_features / tp_size` 列
  - 输入需要复制（Copy）或已经在各 rank 上分片（input_is_parallel）
  - 输出可选 Gather（gather_output 参数）
  - 支持 Sequence Parallel（sequence_parallel 参数）
  - 支持 skip_bias_add（fused kernel 优化）
- **RowParallelLinear 特性**:
  - 权重按行分割：每个 rank 拥有 `in_features / tp_size` 行
  - 输入需要 Scatter 到各 rank（除非 input_is_parallel=true）
  - 输出需要 AllReduce（reduce_output 参数）或 ReduceScatter（sequence_parallel=true）
- **Autograd 集成**:
  - `CopyToTPRegion`: 前向复制输入，反向 AllReduce 梯度
  - `GatherFromTPRegion`: 前向 AllGather 输出，反向 Split 梯度
  - `ScatterToTPRegion`: 前向 Split 输入，反向 AllGather 梯度
  - `ReduceFromTPRegion`: 前向 AllReduce 输出，反向复制梯度

### 2.6 VocabParallelEmbedding
- **位置**: `tensor_parallel.cc`
- **主要功能**: 实现词汇表并行的嵌入层，按词表分割嵌入矩阵
- **关键成员**:
  - `vocab_size_global_`: 全局词汇表大小
  - `vocab_size_per_partition_`: 每个 rank 的词汇表大小
  - `vocab_start_index_` / `vocab_end_index_`: 当前 rank 负责的词汇表范围
- **核心算法**:
  - 前向传播：
    1. 创建 mask：`tokens < vocab_start_index` OR `tokens >= vocab_end_index`
    2. 将 tokens 偏移到本地范围：`tokens - vocab_start_index`，超出范围的 mask 为 0
    3. 执行本地 Embedding 查找
    4. 将 mask 位置的嵌入置零
    5. 执行 AllReduce（或 ReduceScatter，如果启用 sequence_parallel）
  - 反向传播：自动推导，AllReduce 的梯度传播到本地嵌入矩阵

### 2.7 VocabParallelCrossEntropy / VocabParallelCrossEntropyLoss
- **位置**: `tensor_parallel.cc`
- **主要功能**: 实现并行的交叉熵损失，支持 label smoothing
- **核心算法**（前向）：
  1. **Mask 填充**: 将 logits 中 padding 部分填充为 `-inf`
  2. **计算全局最大值**: 本地 Max + AllReduce(Max)
  3. **数值稳定性处理**: `logits - global_max`
  4. **计算 softmax**: 本地 `exp(shifted) / sum_exp`，其中 sum_exp 通过 AllReduce 聚合
  5. **计算预测 logit**: 根据 target mask 提取本地 logit，通过 AllReduce 聚合
  6. **计算损失**: `log(sum_exp) - predicted_logit`
  7. **Label Smoothing**（可选）:
     - 计算有效 token 的平均 logp: `(sum_i_in_valid shifted_i) / vocab_size_original_ - log_sum_exp`
     - 混合: `loss = loss * (1 - smoothing) - mean_logp * smoothing`
- **优化**:
  - 显式转换 logits 为 FP32（与 Megatron-LM 一致）
  - 使用 AllReduce 和 AllGather 通信原语
  - 支持词汇表 padding（vocab_size_original_ <= vocab_size_global_）

### 2.8 GlobalEnv / Layout
- **位置**: `global.cc`
- **主要功能**: 管理全局并行环境配置，实现 DP/TP/PP 混合并行的拓扑布局
- **关键成员**:
  - `nnodes_`: 节点数量
  - `nproc_per_node_`: 每节点进程数
  - `nthread_per_process_`: 每进程线程数
  - `world_size_`: 总线程数（全局 rank 数量）
  - `tensor_parallel_size_`: TP 大小
  - `pipeline_parallel_size_`: PP 大小
  - `data_parallel_size_`: DP 大小（自动计算：world_size / TP / PP）
  - `layout_`: `Layout` 对象，描述 3D 并行拓扑
- **Layout 结构**:
  - `sizes[AXIS_COUNT]`: 各轴大小（DP, TP, PP）
  - `order[AXIS_COUNT]`: 轴顺序（默认 PP -> TP -> DP）
  - `strides[AXIS_COUNT]`: 各轴步长（用于坐标转换）
- **核心方法**:
  - `InitStrides()`: 根据轴顺序计算步长，实现多维数组布局
  - `RankOf(dp, tp, pp)`: 将 3D 坐标转换为全局 rank
  - `CoordOf(rank, &dp, &tp, &pp)`: 将全局 rank 转换为 3D 坐标
  - `GroupId(target, dp, tp, pp)`: 计算给定坐标在目标轴（DP/TP/PP）的进程组 ID
  - `GroupRanks(target, fixed_dp, fixed_tp, fixed_pp)`: 返回目标轴进程组内的所有 rank
- **初始化**: 从环境变量读取配置（NNODES、NPROC_PER_NODE、PROC_WORLD_SIZE、GLOBAL_PROC_RANK、LOCAL_PROC_RANK）

### 2.9 Rank
- **位置**: `rank.cc`
- **主要功能**: 封装进程级和线程级的 rank 信息
- **关键成员**:
  - `process_rank_`: 进程 rank（全局）
  - `thread_rank_`: 线程 rank（进程内）
  - `process_size_`: 总进程数
  - `thread_size_`: 每进程线程数
- **核心方法**:
  - `GlobalRank()`: 计算全局 rank = `process_rank_ * thread_size_ + thread_rank_`
  - `IsParallel()`: 判断是否启用并行（`process_size_ * thread_size_ > 1`）
  - `IsMainRank()`: 判断是否为主 rank（`GlobalRank() == 0`）

### 2.10 DistributedDataParallel
- **位置**: `distributed_data_parallel.cc`
- **主要功能**: 分布式数据并行包装器，自动同步梯度
- **关键成员**:
  - `modules_`: `std::unordered_map<std::string, std::shared_ptr<Module>>`，包装的模块
  - `reducer_`: `std::shared_ptr<Reducer>`，梯度同步器（可选）
- **初始化流程**:
  1. 检查所有参数和缓冲区在同一设备上
  2. 如果 `gradient_bucketing_enabled=false`，为每个参数注册 `AllReducePostAccumulateHook`
  3. 如果 `gradient_bucketing_enabled=true`：
     - 调用 `ComputeBucketAssignmentBySize` 分配桶
     - 创建 `Reducer` 实例
     - 调用 `reducer_->AttachHooksToParameters()` 注册钩子
- **Forward**:
  - 调用底层模块的 Forward
  - 调用 `reducer_->PrepareForBackward()` 准备反向传播

### 2.11 DataParallel
- **位置**: `data_parallel.cc`
- **主要功能**: 单机数据并行，通过多线程并行执行前向传播
- **关键成员**:
  - `dim_`: Scatter/Gather 的维度（默认为 0）
  - `devices_`: 所有可用设备列表
  - `output_device_`: 输出设备（默认为 devices_[0]）
  - `src_device_`: 源设备（模型所在设备）
- **核心流程**:
  1. 检查所有参数在 `src_device_` 上
  2. 调用 `function::Scatter` 将输入散射到各设备
  3. 如果仅单设备，直接执行 Forward
  4. 否则：
     - 调用 `function::Replicate` 复制模块到各设备
     - 调用 `ParallelApply` 在多线程中并行执行 Forward
  5. 调用 `function::Gather` 将输出收集到 `output_device_`
- **ParallelApply 实现**:
  - 预分配结果数组（避免锁）
  - 每个 worker 线程执行：`device->SetDevice(); module->Forward(inputs);`
  - 主线程等待所有 worker 完成

### 2.12 PipelineStage
- **位置**: `pp/pipeline_stage.cc`
- **主要功能**: 封装单个流水线阶段的模型分片
- **关键成员**:
  - `stage_index_`: 阶段索引（即 pp_rank）
  - `num_stages_`: 总阶段数
  - `prev_rank_` / `next_rank_`: 前驱/后继阶段 rank
  - `recv_shape_`: 接收张量的形状
  - `optimizer_`: 优化器（每个阶段独立）
  - `chunks_`: `std::vector<std::shared_ptr<Module>>`，虚拟流水线分片（Virtual Pipeline Parallelism）
- **核心方法**:
  - `ForwardOneChunk(inputs, local_chunk_idx)`: 执行单个 chunk 的前向传播
  - `IsFirstStage()` / `IsLastStage()`: 判断是否为首/尾阶段

### 2.13 PipelineParallelScheduler / PipelineSchedule
- **位置**: `pp/pipeline_schedule.cc`
- **主要功能**: 生成和执行流水线调度策略
- **调度类型**:
  1. **GPipe Schedule**:
     - 阶段：Warmup（仅前向） -> Steady（前向+后向） -> Cooldown（仅后向）
     - 总步数：`2 * (n + num_stages * vpp_size - 1)`
     - 特点：简单易实现，但内存占用高
  2. **1F1B Schedule**（Interleaved 1F1B）:
     - 阶段：Warmup（仅前向） -> Steady（每步前向+后向） -> Cooldown（仅后向）
     - 总步数：`2 * (num_stages * vpp_size - 1) + n`
     - 特点：内存效率更高，适合大模型训练
- **核心方法**:
  - `GenerateGPipeSchedule(n, num_stages, vpp_size)`: 生成 GPipe 调度表
  - `GenerateInterleaved1F1BSchedule(n, num_stages, vpp_size)`: 生成 1F1B 调度表
  - `StepMicroBatches(...)`: 执行调度表
    - 遍历调度表的每个任务
    - 如果任务属于当前阶段：
      - 前向：接收输入（如果不是首 chunk） -> 执行前向 -> 发送输出（如果不是尾 chunk）
      - 后向：
        - 如果是尾 chunk：计算损失 -> 反向传播
        - 否则：使用 dummy gradient 反向传播
  - `ReceiveFromPrev(peer_rank)`: 调用 `IRecv` 接收张量
  - `SendToNext(tensors, peer_rank)`: 调用 `ISend` 发送张量
- **Task 结构**:
  - `step`: 任务步骤
  - `microbatch_id`: 微批次 ID
  - `global_chunk_id`: 全局 chunk ID（0 到 num_stages * vpp_size - 1）
  - `local_chunk_idx`: 本地 chunk ID（global_chunk_id / num_stages）
  - `stage_id`: 负责执行该任务的阶段 ID
  - `is_forward`: 是否为前向任务
  - `is_first_chunk` / `is_last_chunk`: 是否为首/尾 chunk

### 2.14 ISend / IRecv
- **位置**: `pp/send_recv.cc`
- **主要功能**: 流水线并行的点对点通信 autograd 函数
- **ISend 实现**:
  - Forward: 调用 `ProcessGroup::Send` 发送张量，返回输入张量（不变）
  - Backward: 调用 `ProcessGroup::Recv` 接收梯度张量
- **IRecv 实现**:
  - Forward: 调用 `ProcessGroup::Recv` 接收张量
  - SetupContext: 记录当前设备
  - Backward: 调用 `ProcessGroup::Send` 发送梯度张量
- **数据流**: 前向时从 stage i 发送到 stage i+1，反向时从 stage i+1 发送梯度到 stage i

### 2.15 PipelineParallel
- **位置**: `pp/pipeline_parallel.cc`
- **主要功能**: 流水线并行包装器，构建和执行完整的流水线模型
- **关键成员**:
  - `num_stages_`: 阶段数（PP 大小）
  - `rank_`: 当前阶段 rank
  - `pipeline_stage_`: `std::shared_ptr<PipelineStage>`，流水线阶段对象
  - `schedule_`: `std::shared_ptr<PipelineSchedule>`，调度器
- **核心方法**:
  - `BuildPipelineStage(...)`: 构建 PipelineStage，包含多个 chunk
  - `SetupSchedule(num_micro_batches)`: 初始化调度器
  - `GetStageInfo(total_layers, pp_size, rank, chunks_per_stage)`: 静态方法，计算每个阶段负责的层范围
    - 支持均匀分配和余数分配
    - 返回 `StageInfo` 结构（is_first_stage, is_last_stage, layer_ranges_per_chunk）
  - `TrainStep(input, target, loss_fn, dtype)`: 执行一次训练步骤
    - 如果是首阶段，将 input 分割为微批次
    - 如果是尾阶段，将 target 分割为微批次
    - 调用 `optimizer->ZeroGrad()`
    - 调用 `schedule_->Step(...)` 执行流水线调度
    - 调用 `optimizer->Step()`
- **模型构建**:
  - 每个阶段包含多个 chunk（Virtual Pipeline Parallelism）
  - 每个 chunk 是一个 Sequential 模块，包含：
    - 首阶段：`kPPFirstStageName` + `kPPChunkNamePrefix + chunk_id`
    - 中间阶段：`kPPChunkNamePrefix + chunk_id`
    - 尾阶段：`kPPChunkNamePrefix + chunk_id` + `kPPLastStageName`

## 3. API 接口

### 3.1 进程组管理

```cpp
// 获取进程组工厂单例
ProcessGroupFactory* ProcessGroupFactory::Instance();

// 创建或获取进程组（按设备数量）
const ProcessGroup* ProcessGroupFactory::GetOrCreate(
    const std::string& name,
    int comm_size
);

// 创建或获取进程组（按设备索引列表）
const ProcessGroup* ProcessGroupFactory::GetOrCreate(
    const std::string& name,
    const std::vector<int>& device_indices
);

// 获取已注册的进程组
const ProcessGroup* ProcessGroupFactory::Get(const std::string& name) const;
```

### 3.2 通信原语

```cpp
// AllReduce: 在进程组内对张量执行规约并广播结果
std::shared_ptr<Work> ProcessGroup::AllReduce(
    const std::shared_ptr<Tensor>& tensor,
    ReduceOpType reduce_op,  // kSum, kProd, kMax, kAvg
    bool async_op
) const;

// AllGather: 收集所有进程的张量并拼接
std::shared_ptr<Work> ProcessGroup::AllGather(
    const std::shared_ptr<Tensor>& output,
    const std::shared_ptr<Tensor>& input,
    bool async_op
) const;

// ReduceScatter: 先规约后分散
std::shared_ptr<Work> ProcessGroup::ReduceScatter(
    const std::shared_ptr<Tensor>& output,
    const std::shared_ptr<Tensor>& input,
    ReduceOpType reduce_op,
    bool async_op
) const;

// Send: 点对点发送
std::shared_ptr<Work> ProcessGroup::Send(
    std::vector<std::shared_ptr<Tensor>> tensors,
    int dest_rank,
    bool async_op
) const;

// Recv: 点对点接收
std::shared_ptr<Work> ProcessGroup::Recv(
    std::vector<std::shared_ptr<Tensor>> tensors,
    int src_rank,
    bool async_op
) const;
```

### 3.3 并行工具函数

```cpp
// 获取数据并行进程组名称
std::string GetDataParallelProcessGroupName(int global_rank);

// 获取张量并行进程组名称
std::string GetTensorParallelProcessGroupName(int global_rank);

// 获取流水线并行进程组名称
std::string GetPipelineParallelProcessGroupName(int global_rank);

// 获取数据并行组内的所有 rank
std::vector<int> GetDataParallelGroupRanks(int global_rank);

// 获取张量并行组内的所有 rank
std::vector<int> GetTensorParallelGroupRanks(int global_rank);

// 获取流水线并行组内的所有 rank
std::vector<int> GetPipelineParallelGroupRanks(int global_rank);
```

### 3.4 张量并行模块

```cpp
// 列并行线性层
ColumnParallelLinear::ColumnParallelLinear(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool gather_output,           // 是否在输出时 AllGather
    bool input_is_parallel,        // 输入是否已分片
    bool skip_bias_add,           // 是否跳过 bias 加法（fused kernel）
    bool sequence_parallel         // 是否启用 Sequence Parallel
);

// 行并行线性层
RowParallelLinear::RowParallelLinear(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool reduce_output,            // 是否在输出时 AllReduce
    bool input_is_parallel,        // 输入是否已分片
    bool skip_bias_add,
    bool sequence_parallel
);

// 词汇表并行嵌入层
VocabParallelEmbedding::VocabParallelEmbedding(
    int64_t num_embeddings,
    int64_t embedding_dim,
    bool reduce_scatter_embeddings  // 是否使用 ReduceScatter 替代 AllReduce
);

// 词汇表并行交叉熵损失
VocabParallelCrossEntropyLoss::VocabParallelCrossEntropyLoss(
    int64_t vocab_size_original,   // 原始词汇表大小（未 padding）
    float label_smoothing = 0.0f   // Label smoothing 系数
);
```

### 3.5 数据并行模块

```cpp
// 分布式数据并行（DDP）
DistributedDataParallel::DistributedDataParallel(
    std::shared_ptr<nn::Module> module,
    int device_id,
    const ReducerOptions& opts  // 梯度桶配置
);

// ReducerOptions 配置
struct ReducerOptions {
    size_t first_bucket_cap_mb = 25;      // 首桶容量（MB）
    size_t normal_bucket_cap_mb = 25;     // 普通桶容量（MB）
    bool gradient_bucketing_enabled = true; // 是否启用梯度桶
    bool gradient_as_bucket_view = false;   // 是否使用梯度视图优化
};

// 单机数据并行
DataParallel::DataParallel(
    const std::shared_ptr<Module>& module,
    int dim = 0  // Scatter/Gather 维度
);
```

### 3.6 流水线并行模块

```cpp
// 流水线并行
PipelineParallel::PipelineParallel(
    const std::shared_ptr<Module> module,    // 包含所有 chunk 的模块
    int num_stages,                          // PP 大小
    int num_micro_batches,                   // 微批次数量
    const std::vector<std::vector<int64_t>>& recv_shape,  // 接收张量形状
    int pp_rank,                             // 当前阶段 rank
    const std::shared_ptr<Optimizer>& optimizer,
    int device_id,
    int chunk_size                           // VPP 大小
);

// 获取阶段负责的层范围
StageInfo PipelineParallel::GetStageInfo(
    int total_layers,
    int pp_size,
    int rank,
    int chunks_per_stage
);

// 执行训练步骤
float PipelineParallel::TrainStep(
    const std::vector<std::shared_ptr<Tensor>>& input,
    const std::vector<std::shared_ptr<Tensor>>& target,
    const std::shared_ptr<Module>& loss_fn,
    DataType dtype
);

// 流水线通信函数
std::vector<std::shared_ptr<Tensor>> ISend(
    const std::vector<std::shared_ptr<Tensor>>& input_tensors,
    const Device* target_device,
    int cur_rank,
    int peer_rank,
    const std::vector<std::vector<int64_t>>& shape
);

std::vector<std::shared_ptr<Tensor>> IRecv(
    const std::vector<std::shared_ptr<Tensor>>& outputs,
    const Device* src_device,
    int cur_rank,
    int peer_rank
);
```

### 3.7 全局配置

```cpp
// 初始化全局并行环境
void GlobalEnv::Init(
    int nthread_per_process,           // 每进程线程数
    int tensor_parallel_size,          // TP 大小
    bool sequence_parallel_enabled,     // 是否启用 Sequence Parallel
    int pipeline_parallel_size,         // PP 大小
    int virtual_pipeline_parallel_size  // VPP 大小
);

// 获取全局配置单例
GlobalEnv& GlobalEnv::Instance();

// 查询配置
int GlobalEnv::world_size() const;
int GlobalEnv::tensor_parallel_size() const;
int GlobalEnv::data_parallel_size() const;
int GlobalEnv::pipeline_parallel_size() const;
Layout GlobalEnv::layout() const;
```

## 4. 使用示例

### 4.1 初始化并行环境

```cpp
#include "infini_train/include/nn/parallel/global.h"

using namespace infini_train::nn::parallel::global;

// 初始化：8 卡，TP=2, PP=4, 每进程 4 线程
GlobalEnv::Instance().Init(
    4,    // nthread_per_process
    2,    // tensor_parallel_size
    true, // sequence_parallel_enabled
    4,    // pipeline_parallel_size
    2     // virtual_pipeline_parallel_size
);

// 自动计算：DP = 8 / (2 * 4) = 1
// 打印进程组概览
LOG(INFO) << ProcessGroupOverview(GlobalEnv::Instance().layout(), true);
```

### 4.2 使用张量并行构建 Transformer 层

```cpp
#include "infini_train/include/nn/parallel/tensor_parallel.h"

using namespace infini_train::nn::parallel;

// 创建 ColumnParallelLinear（QKV 投影）
auto qkv_layer = std::make_shared<ColumnParallelLinear>(
    hidden_size,      // in_features
    3 * hidden_size,  // out_features
    true,             // bias
    true,             // gather_output
    false,            // input_is_parallel
    false,            // skip_bias_add
    false             // sequence_parallel
);

// 创建 RowParallelLinear（输出投影）
auto output_layer = std::make_shared<RowParallelLinear>(
    hidden_size,      // in_features
    vocab_size,       // out_features
    false,            // bias
    true,             // reduce_output
    true,             // input_is_parallel
    false,            // skip_bias_add
    true              // sequence_parallel
);

// 创建 VocabParallelEmbedding
auto embedding = std::make_shared<VocabParallelEmbedding>(
    vocab_size,
    hidden_size,
    true  // reduce_scatter_embeddings
);

// 创建 VocabParallelCrossEntropyLoss
auto loss_fn = std::make_shared<VocabParallelCrossEntropyLoss>(
    vocab_size,    // vocab_size_original（可能被 padding）
    0.1f          // label_smoothing
);
```

### 4.3 使用分布式数据并行

```cpp
#include "infini_train/include/nn/parallel/distributed_data_parallel.h"

using namespace infini_train::nn::parallel;

// 配置 Reducer
ReducerOptions opts;
opts.first_bucket_cap_mb = 25;
opts.normal_bucket_cap_mb = 25;
opts.gradient_bucketing_enabled = true;
opts.gradient_as_bucket_view = true;  // 减少拷贝

// 包装模型
auto model = BuildTransformerModel();
auto ddp_model = std::make_shared<DistributedDataParallel>(
    model,
    device_id,
    opts
);

// 训练循环
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    auto batch = data_loader.NextBatch();
    auto output = ddp_model->Forward(batch);
    auto loss = loss_fn->Forward({output, target})[0];
    loss->Backward();
    optimizer->Step();
}
```

### 4.4 使用流水线并行

```cpp
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"

using namespace infini_train::nn::parallel;

// 构建包含所有 chunk 的模块
auto module = std::make_shared<Module>();
// ... 添加 submodules：kPPFirstStageName, kPPChunkNamePrefix + "0", ...
// ... kPPChunkNamePrefix + "1", kPPLastStageName

// 获取当前阶段负责的层范围
auto stage_info = PipelineParallel::GetStageInfo(
    total_layers,
    pp_size,        // 4
    pp_rank,        // 0, 1, 2, or 3
    chunks_per_stage // 2
);

// 创建流水线并行模型
auto pp_model = std::make_shared<PipelineParallel>(
    module,
    4,                              // num_stages
    8,                              // num_micro_batches
    {{batch_size, seq_len, hidden}}, // recv_shape
    pp_rank,
    optimizer,
    device_id,
    2                               // chunk_size
);

// 训练循环
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    auto batch = data_loader.NextBatch();
    float loss = pp_model->TrainStep(
        {batch.input},
        {batch.target},
        loss_fn,
        DataType::kFLOAT16
    );
    LOG(INFO) << "Epoch " << epoch << ", Loss: " << loss;
}
```

### 4.5 混合并行（DP + TP + PP）

```cpp
// 假设总卡数 = 32，配置：DP=2, TP=4, PP=4
// 每个 DP 组内：TP=4, PP=4（共 16 卡）
// 共有 2 个 DP 组

// 初始化全局环境
GlobalEnv::Instance().Init(
    16,   // nthread_per_process（1 个进程，16 个线程）
    4,    // tensor_parallel_size
    true, // sequence_parallel_enabled
    4,    // pipeline_parallel_size
    2     // virtual_pipeline_parallel_size
);

// 每个 DP rank 独立训练
auto dp_rank = global::GetGroupId(global::DP, global::GetGlobalRank());

// 创建数据并行模型（内含 TP + PP）
auto model = BuildTPPPModel();  // 包含 ColumnParallelLinear, RowParallelLinear, PipelineParallel

auto ddp_model = std::make_shared<DistributedDataParallel>(
    model,
    device_id,
    reducer_opts
);

// 每次前向传播：
// 1. TP：在 ColumnParallelLinear 和 RowParallelLinear 中自动 AllGather/AllReduce
// 2. PP：在 PipelineParallel 中调度微批次
// 3. DP：在 DDP 中通过 Reducer 同步梯度
auto output = ddp_model->Forward(input);
```

## 5. 实现细节

### 5.1 内存管理

- **梯度桶（Bucket）**:
  - 每个桶维护一个扁平化的 `contents` tensor（1D）
  - 参数梯度通过 `bucket_views_in/out` 映射到 `contents` 的子视图
  - 支持动态重建：第一次迭代后根据实际梯度就绪顺序重新分配桶
  - 避免碎片化：按设备和数据类型分组，确保同一桶内的张量在同一设备和同一类型

- **梯度视图优化**:
  - 当 `gradient_as_bucket_view=true` 时，参数的 `grad` 直接指向桶内的视图
  - 避免梯度从参数到桶的拷贝（`CopyGradToBucket`）
  - 需要在下次梯度累积时覆盖写入（通过 `MarkGradOverwriteOnNextAccum` 标记）

### 5.2 并发控制

- **Reducer 同步**:
  - 使用 `std::mutex` 保护桶状态（`buckets_`, `next_bucket_`, `locators_`）
  - 使用 `std::atomic<size_t>` 记录已完成的桶数量（`buckets_finished_`）
  - 桶的 AllReduce 按顺序启动（`next_bucket_` 递增），避免乱序完成导致的梯度覆盖

- **ProcessGroup 线程安全**:
  - 每个设备有独立的 NCCL 通信器和 CUDA 流
  - 使用 `cudaEvent` 同步计算流和通信流
  - 多进程初始化：通过文件共享 `ncclUniqueId`，使用轮询等待文件创建

### 5.3 性能优化

- **通信流隔离**:
  - 所有 NCCL 操作在独立的通信流上执行
  - 通过 `cudaEventRecord(compute_stream) -> cudaStreamWaitEvent(comm_stream) -> NCCL op -> cudaEventRecord(comm_stream) -> cudaStreamWaitEvent(compute_stream)` 实现流同步
  - 避免通信阻塞计算，提高吞吐量

- **梯度桶优化**:
  - 首桶容量限制：`first_bucket_cap_mb`（默认 25MB）
  - 普通桶容量限制：`normal_bucket_cap_mb`（默认 25MB）
  - 多级容量策略：迭代 `bucket_size_limits` 直到最后一个容量
  - 按反向顺序分配：接近梯度实际就绪顺序，减少等待时间

- **动态桶重建**:
  - 第一次迭代记录梯度就绪顺序（`grad_ready_order_indices_`）
  - 第二次迭代前根据实际顺序重建桶（`RebuildBuckets`）
  - 减少 AllReduce 启动延迟

- **Sequence Parallel**:
  - 在 TP 基础上进一步沿序列维度分片
  - `ColumnParallelLinear`: AllGather 输入 -> Forward -> ReduceScatter 输出
  - `RowParallelLinear`: AllGather 输入 -> Forward -> AllReduce 输出
  - 减少激活值内存占用，支持更长序列

### 5.4 错误处理

- **NCCL 异步错误检测**:
  - 使用 `ncclCommGetAsyncError` 检查通信器的异步错误状态
  - 在 `WorkNccl::CheckNcclStatus()` 中定期检查
  - 错误信息通过 `std::exception_ptr` 保存，避免在异步上下文中抛出异常

- **超时机制**:
  - `WorkNccl::WaitBlocking` 支持超时参数
  - 超时后轮询 `cudaEventQuery`，避免忙等待
  - 每次轮询后 sleep 50 微秒（`std::this_thread::sleep_for(std::chrono::microseconds(50))`）

### 5.5 数值稳定性

- **VocabParallelCrossEntropy**:
  - 显式转换为 FP32（`input_tensors[0]->To(DataType::kFLOAT32)`）
  - 全局 Max 减法：`logits - global_max` 避免溢出
  - Log-Sum-Exp 技巧：`log(sum_exp) - predicted_logit`

- **AllReduce 平均**:
  - Reducer 使用 `ReduceOpType::kAvg` 而非 `kSum`
  - NCCL 的 `ncclAvg` 自动除以 world_size，避免手动除法

### 5.6 依赖关系

- **外部依赖**:
  - NCCL（NVIDIA Collective Communications Library）：GPU 间通信
  - CUDA Runtime API：事件和流管理
  - glog：日志记录

- **内部模块依赖**:
  - `autograd/Function`: 自动微分框架
  - `autograd/FunctionHook`: 梯度钩子（`PostAccumulateGradHook`）
  - `nn/modules/Module`: 神经网络模块基类
  - `nn/functional`: 神经网络函数（`Concat`, `Split`, `MaskedFill` 等）
  - `device`: 设备管理（`CudaDevice`, `DeviceManager`）
  - `tensor`: 张量抽象
  - `datatype`: 数据类型定义
  - `dispatcher`: 算子分发器（用于 VocabParallelCrossEntropyBackward）

### 5.7 设计模式

- **单例模式**:
  - `ProcessGroupFactory::Instance()`: Meyer's Singleton（线程安全，C++11 保证）
  - `GlobalEnv::Instance()`: Meyer's Singleton
  - `DeviceManager::Instance()`: 依赖的单例

- **工厂模式**:
  - `ProcessGroupFactory`: 根据配置创建 `ProcessGroupNCCL`
  - 支持按设备数量或设备索引列表创建

- **策略模式**:
  - `PipelineSchedule`: 支持多种调度策略（GPipe, 1F1B）
  - `Reducer`: 支持自定义通信钩子（`comm_hook_`）

- **观察者模式**:
  - `autograd::PostAccumulateGradHook`: 参数梯度就绪时触发 Reducer 的 `MarkVariableReadyDense`
  - 支持多个钩子注册（链式调用）

- **模板方法模式**:
  - `PipelineSchedule::StepMicroBatches`: 定义调度执行流程，具体通信由 `ReceiveFromPrev` 和 `SendToNext` 实现

- **RAII（资源获取即初始化）**:
  - `WorkNccl`: 析构时自动销毁 CUDA 事件（`cudaEventDestroy`）
  - `ProcessGroupNCCL`: 析构时自动销毁 NCCL 通信器和 CUDA 流

## 6. 关键算法复杂度

- **AllReduce**: O(log P) 通信轮次，P = 进程组大小
- **AllGather**: O(log P) 通信轮次
- **ReduceScatter**: O(log P) 通信轮次
- **Send/Recv**: O(1) 通信轮次（点对点）
- **Bucket Assignment**: O(N log N)，N = 参数数量（排序）
- **GPipe 调度**: O(n * vpp * num_stages)，n = 微批次数量
- **1F1B 调度**: O(n * vpp * num_stages)

## 7. 限制与注意事项

- **NCCL 依赖**: 仅支持 CUDA 设备，不支持 CPU
- **单节点限制**: 当前实现假设所有设备在同一节点（`cudaDeviceCanAccessPeer`）
- **流水线通信**: `IRecv` 在 Forward 前分配接收张量，需要预先知道形状
- **梯度视图**: 启用 `gradient_as_bucket_view` 时，用户不应手动修改参数梯度
- **VocabParallelCrossEntropy**: 要求词汇表大小被 TP 大小整除（或 padding）
- **混合精度**: VocabParallelCrossEntropy 显式使用 FP32，与 Megatron-LM 对齐
