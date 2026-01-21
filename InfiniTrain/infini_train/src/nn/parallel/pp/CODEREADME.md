# Pipeline Parallel (PP) 核心实现文档

Pipeline Parallel (PP) 模块实现大规模深度学习模型的流水线并行训练，支持 GPipe 和 1F1B (Interleaved) 两种调度策略，通过将模型切分到多个 GPU 设备实现内存优化和计算并行化。

## 1. 模块结构

- **`pipeline_parallel.cc/h`**: 流水线并行主入口，负责构建 PipelineStage 和 PipelineSchedule，提供模型层切分逻辑
- **`pipeline_schedule.cc/h`**: 核心调度器实现，生成 GPipe/1F1B 调度表，执行前向/反向传播的时序控制
- **`pipeline_stage.cc/h`**: 流水线阶段抽象，封装单个 Stage 的前向计算、设备管理、邻居通信逻辑
- **`send_recv.cc/h`**: 异步通信原语，基于 Autograd 实现的 ISend/IRecv 操作，支持自动微分梯度传播

## 2. 核心类

### `PipelineParallel`
- **Location**: `pipeline_parallel.cc/h`
- **Primary Function**: 流水线并行的顶层管理类，负责模块切分、Stage 构建、调度器初始化
- **Key Members**:
  - `num_stages_`: 流水线总阶段数（等于 GPU 数量）
  - `rank_`: 当前进程在流水线中的 rank [0, num_stages-1]
  - `pipeline_stage_`: 当前 Stage 的执行引擎
  - `schedule_`: 调度器实例（GPipe 或 1F1B）
- **Core Methods**:
  - `BuildPipelineStage()`: 构建 PipelineStage 实例，将 chunks 移动到目标设备
  - `SetupSchedule()`: 创建 PipelineSchedule，传入 num_micro_batches 参数
  - `TrainStep()`: 单步训练入口，处理 input/target 分片，调用 scheduler->Step()
  - `GetStageInfo()`: 静态方法，计算每个 stage 的层分配范围（处理均匀分配和余数分配）
- **Lifecycle**:
  1. 构造时接收完整模型，按 `kPPChunkNamePrefix + chunk_id` 切分 chunks
  2. 为首尾 stage 添加 `kPPFirstStageName` / `kPPLastStageName` 模块
  3. 调用 BuildPipelineStage() 和 SetupSchedule() 完成初始化

### `PipelineSchedule`
- **Location**: `pipeline_schedule.cc/h`
- **Primary Function**: 执行流水线调度，按照生成的 Task 序列协调前向/反向传播和跨阶段通信
- **Key Members**:
  - `stage_`: 持有的 PipelineStage 实例
  - `num_micro_batches_`: 微批次数量（每个 microbatch 的 batch size = total_batch / num_micro_batches）
- **Core Methods**:
  - `Step()`: 主训练循环，将 input/target 切分为 microbatches，调用 optimizer->ZeroGrad/Step
  - `StepMicroBatches()`: 按照 schedule 表执行每个 Task，管理 activations 缓存（vpp_size x n 矩阵）
  - `ReceiveFromPrev()`: 从上一 stage 接收张量（通过 IRecv），创建空张量并启动异步接收
  - `SendToNext()`: 向下一 stage 发送张量（通过 ISend），触发异步发送
- **Scheduling Algorithm**:
  - 调用 `PipelineParallelScheduler::GenerateGPipeSchedule(n, num_stages, vpp_size)` 生成任务表
  - Task 包含：step, microbatch_id, global_chunk_id, local_chunk_idx, is_forward, stage_id
  - 遍历 schedule，仅执行 `task.stage_id == stage_idx` 的任务
  - Forward 时：如果是首 chunk 则使用 microbatch input，否则从上一 stage 接收
  - Backward 时：如果是末尾 chunk 则计算 loss 并反向传播，否则使用 dummy gradient
- **Activation Management**: 使用 `activations[vpp_size][n]` 二维数组缓存中间结果，供反向传播使用

### `PipelineParallelScheduler`
- **Location**: `pipeline_schedule.cc/h`
- **Primary Function**: 静态工具类，生成不同调度策略的任务序列
- **Core Methods**:
  - `GenerateGPipeSchedule(n, num_stages, vpp_size)`: 生成 GPipe 调度表
    - 总步数：2 * (n + num_stages * vpp_size - 1)
    - 前向阶段：按 step-mb = global_chunk_id 公式生成任务
    - 反向阶段：按 global_chunk_id = (total_steps-1-step) - mb 公式生成任务
    - 按 step 和 local_chunk_idx 排序保证执行顺序
  - `GenerateInterleaved1F1BSchedule(n, num_stages, vpp_size)`: 生成 1F1B 调度表
    - Warmup 阶段：仅前向，步数 = total_global_chunks - 1
    - Steady 阶段：每个 step 同时执行 1 个前向和 1 个反向，步数 = n
    - Cool-down 阶段：仅反向，清理剩余梯度
  - `CreateTask()`: 工厂方法，构造 Task 对象并计算 derived fields

### `PipelineStage`
- **Location**: `pipeline_stage.cc/h`
- **Primary Function**: 单个流水线阶段的执行引擎，封装模型 chunks、设备、优化器、邻居 rank 信息
- **Key Members**:
  - `stage_index_`: 当前 stage 的 rank [0, num_stages-1]
  - `num_stages_`: 总 stage 数量
  - `prev_rank_` / `next_rank_`: 相邻 stage 的 rank（边界为 -1）
  - `chunks_`: 当前 stage 持有的模型 chunk 列表（Sequential 容器）
  - `device_`: CUDA 设备指针（通过 DeviceManager::GetAllAvailableDevices().at(device_id) 获取）
  - `recv_shape_`: 接收张量的形状信息（用于预分配内存）
  - `optimizer_`: 优化器实例
- **Core Methods**:
  - `ForwardOneChunk(inputs, local_chunk_idx)`: 执行指定 chunk 的前向传播
    - 边界检查：local_chunk_idx 必须在 [0, chunks_.size()) 范围内
    - 调用 chunks_[local_chunk_idx]->Forward(inputs)
  - `IsFirstStage()` / `IsLastStage()`: 边界判断（stage_index == 0 或 num_stages-1）
  - `stage_index()`, `prev_rank()`, `next_rank()`, `num_stages()`: 访问器方法
  - `device()`, `recv_shape()`, `optimizer()`, `chunks()`, `mutable_chunks()`: 数据访问接口
- **Lifecycle**: 构造时移动 chunks 所有权，通过 DeviceManager 获取设备指针

### `ISend` / `IRecv` (Functions)
- **Location**: `send_recv.cc/h`
- **Primary Function**: 基于 Autograd 的异步通信原语，前向发送/接收张量，反向自动交换梯度
- **ISend 实现**:
  - 继承 `autograd::Function`，类型为 "ISendFunction"
  - **Forward**: 调用 `pp_group->Send(input_tensors, peer_rank_, false)` 发送到对端
    - 从 `ProcessGroupFactory` 获取 PP 通信组（按 global rank 命名）
    - 返回输入张量（形成计算图边）
  - **Backward**: 接收来自对端的梯度张量
    - 根据 `shapes_` 预分配 float32 张量
    - 调用 `pp_group->Recv(recv_tensors, peer_rank_, false)`
    - 返回接收到的梯度（反向传播给上游）
- **IRecv 实现**:
  - 继承 `autograd::Function`，类型为 "IRecvFunction"
  - **Forward**: 接收张量并返回（预分配的空张量传入）
    - 调用 `pp_group->Recv(recv_tensors, peer_rank_, false)`
    - SetupContext 记录 `cur_device_` 用于反向通信
  - **Backward**: 发送梯度到对端
    - 调用 `pp_group->Send(grad_outputs, peer_rank_, false)`
    - 返回 grad_outputs（保持梯度形状）
- **关键特性**:
  - 自动微分集成：通信操作纳入计算图，反向时自动交换梯度
  - 异步执行：Send/Recv 的 `blocking=false` 参数实现非阻塞通信
  - 数据类型限制：当前硬编码为 kFLOAT32（FIXME 注释指出需要改进）

## 3. API 接口

```cpp
// 顶层流水线并行构造器
PipelineParallel::PipelineParallel(
    const std::shared_ptr<nn::Module> module,      // 完整模型（内部包含切分好的 chunks）
    int num_stages,                                 // 总 stage 数量（= GPU 数量）
    int num_micro_batches,                          // 微批次数量
    const std::vector<std::vector<int64_t>> &recv_shape,  // 跨阶段张量形状
    int rank,                                       // 当前 stage 的 rank
    const std::shared_ptr<Optimizer> &optimizer,    // 优化器
    int device_id,                                  // CUDA 设备 ID
    int vpp                                         // 每个 stage 的 chunk 数量（virtual pipeline size）
);

// 单步训练入口
float PipelineParallel::TrainStep(
    const std::vector<std::shared_ptr<Tensor>> &input,      // 输入张量（仅在 rank 0 使用）
    const std::vector<std::shared_ptr<Tensor>> &target,     // 目标张量（仅在最后一 stage 使用）
    const std::shared_ptr<nn::Module> &loss_fn,             // 损失函数模块
    DataType dtype                                          // 计算精度（如 fp16/bf16）
);

// 静态方法：计算层分配范围
StageInfo PipelineParallel::GetStageInfo(
    int total_layers,        // 模型总层数
    int pp_size,             // 流水线总 stage 数
    int pp_rank,             // 当前 stage rank
    int chunks_per_stage     // 每个 stage 的 chunk 数量（默认 1）
);

// 异步通信原语（纳入 Autograd）
std::vector<std::shared_ptr<Tensor>> ISend(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors,  // 要发送的张量
    const Device *target_device,                                  // 目标设备
    int cur_rank,                                                 // 当前 rank
    int peer_rank,                                                // 对端 rank
    const std::vector<std::vector<int64_t>> &shape               // 反向时接收梯度的形状
);

std::vector<std::shared_ptr<Tensor>> IRecv(
    const std::vector<std::shared_ptr<Tensor>> &outputs,  // 预分配的接收缓冲区
    const Device *src_device,                              // 源设备
    int cur_rank,                                          // 当前 rank
    int peer_rank                                          // 对端 rank
);

// 调度器生成接口
std::vector<PipelineParallelScheduler::Task>
PipelineParallelScheduler::GenerateGPipeSchedule(
    int n,              // 微批次数量
    int num_stages,     // 总 stage 数
    int vpp_size        // 每个 stage 的 chunk 数量（虚拟流水线深度）
);
```

## 4. 使用示例

```cpp
// 假设场景：72 层 Transformer 模型，4 卡流水线并行，每 stage 2 个 chunk
int total_layers = 72;
int num_stages = 4;         // pp_size
int chunks_per_stage = 2;   // vpp (virtual pipeline parallel size)
int num_micro_batches = 8;

// 1. 计算每个 stage 的层分配范围
StageInfo stage_info = PipelineParallel::GetStageInfo(
    total_layers, num_stages, my_rank, chunks_per_stage
);
// stage_info.layer_ranges_per_chunk 包含：
//   rank 0: [(0, 9), (36, 45)]     // chunk 0, 2 的层范围
//   rank 1: [(9, 18), (45, 54)]    // chunk 1, 3 的层范围
//   rank 2: [(18, 27), (54, 63)]
//   rank 3: [(27, 36), (63, 72)]

// 2. 构建流水线并行模型
std::vector<std::vector<int64_t>> recv_shape = {{batch_size, hidden_dim}};
auto pp_model = std::make_shared<PipelineParallel>(
    module,                    // 包含 chunks 的模型
    num_stages,                // 4 个 stage
    num_micro_batches,         // 8 个 microbatch
    recv_shape,                // 跨 stage 张量形状
    my_rank,                   // 当前进程 rank [0-3]
    optimizer,                 // AdamW 优化器
    device_id,                 // CUDA:0/1/2/3
    chunks_per_stage           // vpp = 2
);

// 3. 训练循环
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (auto &batch : dataloader) {
        float loss = pp_model->TrainStep(
            {batch.input},      // 输入张量
            {batch.target},     // 目标张量
            loss_fn,            // CrossEntropyLoss
            DataType::kFLOAT16  // 混合精度训练
        );

        if (my_rank == 0) {
            std::cout << "Loss: " << loss << std::endl;
        }
    }
}

// 调度执行细节（GPipe 示例）：
// Step 0: rank 0 执行 mb0 的 chunk0 前向
// Step 1: rank 0 执行 mb1 的 chunk0 前向，rank 1 执行 mb0 的 chunk1 前向
// Step 2: rank 0 执行 mb2 的 chunk0 前向，rank 1 执行 mb1 的 chunk1 前向，rank 2 执行 mb0 的 chunk2 前向
// ...
// Step 7: 所有 rank 并行执行不同 mb 的前向（流水线充满）
// Step 8: rank 0 开始 mb0 的反向，同时 rank 3 执行 mb7 的前向
// ...
```

## 5. 实现细节

- **模型切分策略**:
  - 均匀分配：`layers_per_chunk = total_layers / (pp_size * chunks_per_stage)`
  - 余数处理：前 `remainder % total_global_chunks` 个 chunk 各多分配 1 层
  - 边界 chunk：首 stage 添加 `kPPFirstStageName`（如 embedding），尾 stage 添加 `kPPLastStageName`（如 head）
  - 切分命名约定：`kPPChunkNamePrefix + chunk_id`（如 "chunk_0", "chunk_1"）

- **调度算法 (GPipe)**:
  - 时间复杂度：O((n + num_stages * vpp_size) * n) 生成任务表
  - 空间复杂度：O(n * vpp_size) 缓存 activations
  - 流水线填充阶段：前 n 个 step 逐步激活所有 stage
  - 稳定阶段：所有 stage 同时执行不同 microbatch 的不同 chunk
  - 排水阶段：最后 n 个 step 逐步完成剩余前向和反向
  - 总步数公式：`2 * (n + num_stages * vpp_size - 1)`（前向 + 反向）

- **通信机制**:
  - 使用 ProcessGroup 实现点对点通信（Send/Recv 原语）
  - ProcessGroup 命名：`GetPipelineParallelProcessGroupName(global_rank)` 支持多组流水线
  - 异步执行：`blocking=false` 允许计算与通信重叠
  - Autograd 集成：ISend/IRecv 纳入计算图，反向时自动交换梯度（无需手动同步）
  - 内存管理：ReceiveFromPrev 预分配 float32 张量作为接收缓冲区

- **内存优化**:
  - 微批次切分：`input->Split(batch_size / num_micro_batches)` 减少峰值内存
  - 梯度累积：`loss = loss / n` 在末尾 chunk 反向时平均梯度
  - Dummy Gradient：非末尾 chunk 的反向使用形状匹配的零张量触发 Autograd
  - Activation 缓存：`activations[vpp_size][n]` 按需释放（反向完成后可回收）

- **混合精度支持**:
  - AutocastGuard：前向传播时自动转换 dtype（如 fp16 计算(fp32 权重)
  - 损失缩放：`loss / n` 缩放避免梯度溢出
  - 硬编码限制：ISend/IRecv 当前强制使用 kFLOAT32（需要改进以支持半精度通信）

- **线程安全**:
  - Thread-local storage：`thread_local int pp_rank` 支持多线程环境（每个线程独立 rank）
  - 无锁设计：ProcessGroup 通信由底层库（NCCL）保证线程安全

- **错误处理**:
  - 边界检查：ForwardOneChunk 检查 local_chunk_idx 范围，越界时 LOG(FATAL) 终止
  - 设备检查：DeviceManager::GetAllAvailableDevices().at(device_id) 越界抛出异常
  - 空指针检查：IRecv::SetupContext 使用 CHECK_NOTNULL(src_device_)

- **依赖项**:
  - glog：日志记录（INFO/FATAL 级别）
  - ProcessGroup：通信组管理（依赖 NCCL 后端）
  - Autograd：自动微分框架（Function 基类）
  - DeviceManager：CUDA 设备管理
  - Sequential：模块容器（串联 chunk_parts）

- **设计模式**:
  - Strategy Pattern：PipelineParallelScheduler 可扩展多种调度算法（GPipe/1F1B）
  - Builder Pattern：PipelineParallel 构造函数分步构建 Stage 和 Schedule
  - Template Method：PipelineSchedule::Step() 定义训练骨架，StepMicroBatches() 交由子类实现
  - Autograd Function：ISend/IRecv 继承 autograd::Function，实现自定义前向/反向逻辑
