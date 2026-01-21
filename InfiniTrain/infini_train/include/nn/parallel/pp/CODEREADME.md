# Pipeline Parallel (PP) 模块核心实现文档

本模块实现了流水线并行（Pipeline Parallelism）训练框架，支持 GPipe 和 Interleaved 1F1B（一前向一反向）两种流水线调度策略，用于在大规模分布式训练中跨多个设备分割模型层，实现高效的内存利用和计算重叠。

## 1. 模块结构

- **`pipeline_parallel.h`**: 定义流水线并行的主入口类 `PipelineParallel`，负责模块分割、stage 构建和训练步骤协调
- **`pipeline_schedule.h`**: 定义流水线调度策略抽象类 `PipelineSchedule` 和调度生成器 `PipelineParallelScheduler`，支持 GPipe 和 Interleaved 1F1B 调度算法
- **`pipeline_stage.h`**: 定义单个流水线阶段 `PipelineStage`，封装该阶段负责的模型层 chunk、通信接口和优化器
- **`send_recv.h`**: 提供跨设备通信原语 `ISend` 和 `IRecv`，用于流水线阶段间的异步张量传输

## 2. 核心类

### `PipelineParallel`
- **位置**: `pipeline_parallel.h`
- **主要功能**: 流水线并行训练的主控制器，接收完整模型，将其按层分割到多个 pipeline stage，协调 micro-batch 的前向/反向传播
- **关键成员**:
  - `num_stages_`: 流水线总阶段数（即 pipeline parallel size）
  - `rank_`: 当前进程的 pipeline rank（0 到 num_stages-1）
  - `schedule_`: 流水线调度器实例（GPipe 或 1F1B），负责执行 micro-batch 调度
  - `pipeline_stage_`: 当前 pipeline 阶段实例，包含分配到本阶段的模型 chunk
  - `pp_rank`: thread_local 变量，存储当前线程的 pipeline rank，用于在并行上下文中标识当前阶段
- **核心方法**:
  - `PipelineParallel(module, num_stages, num_micro_batches, recv_shape, rank, optimizer, device_id, vpp)`: 构造函数。将完整 `module` 按 `num_stages` 分割，每个 stage 分配若干 chunk（chunk 数量由 `vpp` 参数控制，即 virtual pipeline parallel size）
  - `TrainStep(input, target, loss_fn, dtype)`: 执行一次完整的训练步骤。内部调用 `PipelineSchedule::Step()`，按调度策略执行所有 micro-batch 的前向/反向传播，并返回平均 loss
  - `GetStageInfo(total_layers, pp_size, pp_rank, chunks_per_stage)`: 静态工具函数，计算给定 rank 的 stage 负责的层索引范围。返回 `StageInfo` 结构体，包含 `is_first_stage`、`is_last_stage` 标志和 `layer_ranges_per_chunk`（每个 chunk 的 [起始层, 结束层) 半开区间）
  - `BuildPipelineStage(optimizer, recv_shape, device_id, chunks)`: 私有方法。创建 `PipelineStage` 实例，将分割好的模型 chunk 移入 stage
  - `SetupSchedule(num_micro_batches)`: 私有方法。根据配置创建 `PipelineSchedule` 子类实例（GPipe 或 1F1B）
  - `mutable_chunks()`: 返回 stage 中 chunk 的可变指针，用于调试或动态修改模型
- **生命周期**: 由用户显式构造，训练过程中多次调用 `TrainStep()`，析构时自动释放 schedule 和 stage 资源

### `PipelineSchedule`
- **位置**: `pipeline_schedule.h`
- **主要功能**: 流水线调度的抽象基类，定义了 micro-batch 执行流程的通用接口和通信原语
- **关键成员**:
  - `stage_`: 持有的 `PipelineStage` 实例，封装当前阶段的模型和优化器
  - `num_micro_batches_`: 每个 global batch 的 micro-batch 数量
- **核心方法**:
  - `Step(input, target, loss_fn, dtype)`: 公共入口，接收单个 global batch 的 input/target，内部将其分割为 `num_micro_batches_` 个 micro-batch，调用虚函数 `StepMicroBatches()` 执行调度，返回平均 loss
  - `StepMicroBatches(arg_mbs, target_mbs, loss_fn, dtype)`: 纯虚函数，由子类实现具体的调度逻辑（如 GPipe 或 1F1B）。接收 micro-batch 列表，按时间步编排 forward/backward，处理 stage 间的依赖和数据传递
  - `ReceiveFromPrev(peer_rank)`: 从前一阶段接收张量。如果当前是第一 stage，返回空 vector；否则调用 `IRecv()` 异步接收前向激活或反向梯度
  - `SendToNext(tensors, peer_rank)`: 向下一阶段发送张量。如果当前是最后 stage，返回空 vector；否则调用 `ISend()` 异步发送前向激活或反向梯度
- **生命周期**: 由 `PipelineParallel::SetupSchedule()` 创建，每次 `TrainStep` 调用其 `Step()` 方法

### `PipelineParallelScheduler`
- **位置**: `pipeline_schedule.h`
- **主要功能**: 流水线调度任务生成器的静态工具类，提供 GPipe 和 Interleaved 1F1B 两种经典调度算法
- **关键成员**:
  - `Task` 结构体：描述单个调度任务
    - `step`: 任务在调度序列中的时间步索引
    - `microbatch_id`: 该任务处理的 micro-batch ID
    - `global_chunk_id`: 使用的 chunk 全局 ID（用于 Interleaved 调度）
    - `local_chunk_idx`: 该 chunk 在当前 stage 中的本地索引
    - `is_forward`: 是否为前向传播（false 表示反向传播）
    - `stage_id`: 任务所属的 stage ID
    - `is_first_chunk`: 是否是当前 stage 的第一个 chunk（用于 1F1B 调度的特殊处理）
    - `is_last_chunk`: 是否是当前 stage 的最后一个 chunk
- **核心方法**:
  - `CreateTask(step, mb, global_chunk, num_stages, total_chunks, is_forward)`: 静态工厂方法，构造单个 Task 对象，自动计算 `local_chunk_idx`、`is_first_chunk`、`is_last_chunk` 等派生字段
  - `GenerateGPipeSchedule(n, num_stages, vpp_size)`: 生成 GPipe（非交错）流水线调度。参数 `n` 是 micro-batch 数量。GPipe 的特点是：每个 stage 只负责一个 chunk，所有 micro-batches 按"先全部 forward，再全部 backward"的顺序执行，warmup 阶段需要 `num_stages-1` 个 micro-batch 填充流水线，steady 阶段每个时间步一个 forward 和一个 backward 并行，cooldown 阶段排空流水线。时间复杂度：O(n + num_stages) 个时间步
  - `GenerateInterleaved1F1BSchedule(n, num_stages, vpp_size)`: 生成 Interleaved 1F1B（一前向一反向）调度。参数 `vpp_size`（virtual pipeline parallel size）表示每个 stage 分配的 chunk 数量。1F1B 的特点是：每个 stage 交错执行多个 chunk 的 forward 和 backward，在 warmup 阶段每个 chunk 先 forward 完成并暂存激活，进入 steady 阶段后每个时间步执行一个新 chunk 的 forward 和一个已完成 forward 的 chunk 的 backward，显著减少内存占用（因为不需要暂存所有 micro-batch 的激活）。相比 GPipe，1F1B 的吞吐量略低但内存效率更高
- **生命周期**: 纯静态类，无实例化

### `PipelineStage`
- **位置**: `pipeline_stage.h`
- **主要功能**: 封装单个流水线阶段的模型 chunk、优化器和设备上下文，提供前向传播执行接口和 stage 身份查询
- **关键成员**:
  - `stage_index_`: 当前 stage 在流水线中的索引（0 到 num_stages-1）
  - `num_stages_`: 流水线总阶段数
  - `prev_rank_`: 前一阶段的进程 rank（-1 表示当前是第一 stage）
  - `next_rank_`: 下一阶段的进程 rank（-1 表示当前是最后 stage）
  - `device_`: 当前 stage 绑定的计算设备（如 GPU）
  - `chunks_`: 分配到本 stage 的模型 chunk 列表（每个 chunk 是一个 `Module` 实例，包含连续的若干层）
  - `optimizer_`: 优化器实例，用于更新本 stage 包含的参数
  - `recv_shape_`: 接收张量的形状信息，用于 `IRecv` 预分配内存
- **核心方法**:
  - `PipelineStage(stage_index, num_stages, recv_shape, optimizer, device_id, chunks)`: 构造函数。初始化 stage 元数据，计算 `prev_rank_` 和 `next_rank_`（基于 `stage_index_` 和 `num_stages_`），移动接收的 chunk 列表
  - `ForwardOneChunk(inputs, local_chunk_idx)`: 对指定的 chunk 执行前向传播。`inputs` 是前一 stage 传来的激活（或第一 stage 的原始输入），内部调用 `chunks_[local_chunk_idx]->Forward()`，返回该 chunk 的输出激活
  - `IsFirstStage()`: 判断当前是否为流水线的第一 stage（不需要接收前一 stage 的数据）
  - `IsLastStage()`: 判断当前是否为流水线的最后 stage（不需要发送给下一 stage）
  - `stage_index()`, `prev_rank()`, `next_rank()`, `num_stages()`: 访问器方法，返回 stage 元数据
  - `device()`, `recv_shape()`, `optimizer()`, `chunks()`, `mutable_chunks()`: 访问器方法，返回内部组件指针或引用
- **生命周期**: 由 `PipelineParallel::BuildPipelineStage()` 创建，与 `PipelineParallel` 同生命周期

### `ISend` / `IRecv` 函数
- **位置**: `send_recv.h`
- **主要功能**: 提供跨设备的异步点对点通信原语，用于流水线 stage 间的激活和梯度传输
- **核心方法**:
  - `ISend(input_tensors, target_device, cur_rank, peer_rank, shape)`: 异步发送一批张量到目标设备的对等进程。`target_device` 指明目标设备类型（用于确定通信 backend，如 NCCL、MPI），`cur_rank` 和 `peer_rank` 标识当前和目标进程的 rank，`shape` 提供张量形状信息。返回接收端的张量占位符（用于异步等待完成）。非阻塞操作，发送操作在后台进行
  - `IRecv(outputs, src_device, cur_rank, peer_rank)`: 异步从源设备的对等进程接收一批张量。`outputs` 是预分配的输出张量容器（需要提前知道形状），`src_device` 指明源设备类型，`cur_rank` 和 `peer_rank` 标识当前和源进程的 rank。返回接收到的张量列表。非阻塞操作，接收操作在后台进行
- **通信语义**: 采用异步非阻塞模式，允许计算与通信重叠。`PipelineSchedule` 在发送/接收后可立即执行其他任务，无需等待完成，从而隐藏通信延迟

## 3. API 接口

```cpp
// 主流水线并行训练接口
class PipelineParallel : public Module {
    // 构造流水线并行实例
    // module: 完整模型（将被分割到多个 stage）
    // num_stages: 流水线阶段数（等于 pipeline parallel group size）
    // num_micro_batches: 每个 global batch 的 micro-batch 数量
    // recv_shape: 接收张量的形状信息（每个 stage 一组形状）
    // rank: 当前进程在 pipeline group 中的 rank
    // optimizer: 优化器实例（每个 stage 独立持有）
    // device_id: 绑定的计算设备 ID
    // vpp: virtual pipeline parallel size（每个 stage 的 chunk 数量，用于 Interleaved 调度）
    PipelineParallel(const std::shared_ptr<nn::Module> module,
                     int num_stages,
                     int num_micro_batches,
                     const std::vector<std::vector<int64_t>> &recv_shape,
                     int rank,
                     const std::shared_ptr<Optimizer> &optimizer,
                     int device_id,
                     int vpp);

    // 执行一次训练步骤
    // input: 输入张量列表（仅在第一 stage 使用）
    // target: 目标张量列表（仅在最后 stage 使用）
    // loss_fn: 损失函数模块
    // dtype: 计算数据类型（如 float16、bfloat16）
    // 返回: 平均 loss 值
    float TrainStep(const std::vector<std::shared_ptr<Tensor>> &input,
                    const std::vector<std::shared_ptr<Tensor>> &target,
                    const std::shared_ptr<nn::Module> &loss_fn,
                    DataType dtype);

    // 静态方法：计算指定 stage 负责的层范围
    // total_layers: 模型总层数
    // pp_size: 流水线并行度（stage 数量）
    // pp_rank: 当前 stage 的 rank
    // chunks_per_stage: 每个 stage 的 chunk 数量（默认 1，即非 Interleaved）
    // 返回: StageInfo 结构体，包含层索引范围和 stage 身份标志
    static StageInfo GetStageInfo(int total_layers,
                                  int pp_size,
                                  int pp_rank,
                                  int chunks_per_stage = 1);
};

// 流水线调度器接口（子类需实现）
class PipelineSchedule {
    // 执行 micro-batch 调度（由子类实现具体策略）
    // arg_mbs: 所有 micro-batch 的输入张量列表
    // target_mbs: 所有 micro-batch 的目标张量列表
    // loss_fn: 损失函数模块
    // dtype: 计算数据类型
    // 返回: 平均 loss 值
    virtual float StepMicroBatches(const std::vector<std::shared_ptr<Tensor>> &arg_mbs,
                                   const std::vector<std::shared_ptr<Tensor>> &target_mbs,
                                   const std::shared_ptr<nn::Module> &loss_fn,
                                   DataType dtype);

    // 从前一 stage 接收张量
    // peer_rank: 前一 stage 的进程 rank
    // 返回: 接收到的张量列表（第一 stage 返回空列表）
    std::vector<std::shared_ptr<Tensor>> ReceiveFromPrev(int peer_rank);

    // 向下一 stage 发送张量
    // tensors: 待发送的张量列表
    // peer_rank: 下一 stage 的进程 rank
    // 返回: 发送操作的句柄张量列表（最后 stage 返回空列表）
    std::vector<std::shared_ptr<Tensor>> SendToNext(const std::vector<std::shared_ptr<Tensor>> &tensors,
                                                    int peer_rank);
};

// 流水线 stage 执行接口
class PipelineStage {
    // 构造 stage 实例
    // stage_index: 当前 stage 的索引（0-based）
    // num_stages: 流水线总 stage 数量
    // recv_shape: 接收张量的形状信息
    // optimizer: 优化器实例
    // device_id: 绑定的设备 ID
    // chunks: 分配到本 stage 的模型 chunk 列表（右值引用，将被移动）
    PipelineStage(int stage_index,
                  int num_stages,
                  const std::vector<std::vector<int64_t>> &recv_shape,
                  std::shared_ptr<Optimizer> optimizer,
                  int device_id,
                  std::vector<std::shared_ptr<Module>> &&chunks);

    // 对指定 chunk 执行前向传播
    // inputs: 输入张量列表（第一 stage 为原始输入，其他 stage 为前一 stage 的输出）
    // local_chunk_idx: 要执行的 chunk 在本 stage 中的索引
    // 返回: 该 chunk 的输出激活
    std::vector<std::shared_ptr<Tensor>> ForwardOneChunk(
        const std::vector<std::shared_ptr<Tensor>> &inputs,
        int local_chunk_idx = 0);
};

// 通信原语接口
namespace infini_train::nn::parallel {
    // 异步发送张量到对等进程
    // input_tensors: 待发送的张量列表
    // target_device: 目标设备类型（用于选择通信 backend）
    // cur_rank: 当前进程的 rank
    // peer_rank: 目标进程的 rank
    // shape: 张量形状信息（用于预分配接收端内存）
    std::vector<std::shared_ptr<Tensor>> ISend(
        const std::vector<std::shared_ptr<Tensor>> &input_tensors,
        const Device *target_device,
        int cur_rank,
        int peer_rank,
        const std::vector<std::vector<int64_t>> &shape);

    // 异步从对等进程接收张量
    // outputs: 预分配的输出张量容器（接收完成后将填充数据）
    // src_device: 源设备类型（用于选择通信 backend）
    // cur_rank: 当前进程的 rank
    // peer_rank: 源进程的 rank
    std::vector<std::shared_ptr<Tensor>> IRecv(
        const std::vector<std::shared_ptr<Tensor>> &outputs,
        const Device *src_device,
        int cur_rank,
        int peer_rank);
}
```

## 4. 使用示例

```cpp
// 假设我们有一个 24 层的 Transformer 模型，想在 4 个 GPU 上进行流水线并行训练
// 每个 GPU 负责 6 层，使用 8 个 micro-batches，Interleaved 1F1B 调度，每个 stage 分 2 个 chunk

#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"

using namespace infini_train;
using namespace infini_train::nn::parallel;

// 1. 定义完整的模型（24 层 Transformer）
auto model = std::make_shared<Transformer>(
    vocab_size = 50000,
    hidden_dim = 1024,
    num_layers = 24,        // 总共 24 层
    num_heads = 16
);

// 2. 定义优化器
auto optimizer = std::make_shared<AdamOptimizer>(
    model->parameters(),
    lr = 1e-3
);

// 3. 配置流水线并行参数
int num_stages = 4;           // 流水线分为 4 个 stage
int num_micro_batches = 8;    // 每个 global batch 分为 8 个 micro-batch
int vpp = 2;                  // 每个 stage 分 2 个 chunk（Interleaved 模式）
int rank = GetMyPipelineRank(); // 获取当前进程在 pipeline group 中的 rank（0-3）
int device_id = rank;         // 每个 stage 绑定一个 GPU

// 接收张量的形状信息（示例：假设 stage 间传递 [batch_size, seq_len, hidden_dim] 的激活）
std::vector<std::vector<int64_t>> recv_shape = {{8, 512, 1024}};  // [micro_batch_size, seq_len, hidden_dim]

// 4. 创建 PipelineParallel 实例（内部会自动将模型的 24 层分割到 4 个 stage，每个 stage 2 个 chunk）
// 分割结果：stage 0 负责 chunk 0-1（层 0-5），stage 1 负责 chunk 2-3（层 6-11），
//          stage 2 负责 chunk 4-5（层 12-17），stage 3 负责 chunk 6-7（层 18-23）
auto pp_model = std::make_shared<PipelineParallel>(
    model,               // 完整模型
    num_stages,          // 4 个 stage
    num_micro_batches,   // 8 个 micro-batch
    recv_shape,          // 接收形状
    rank,                // 当前 stage 的 rank
    optimizer,           // 优化器
    device_id,           // GPU 设备 ID
    vpp                  // 每个 stage 2 个 chunk（启用 Interleaved 调度）
);

// 5. 定义损失函数
auto loss_fn = std::make_shared<CrossEntropyLoss>();

// 6. 训练循环
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (auto& batch : dataloader) {
        // batch 包含 inputs 和 targets（仅在第一 stage 和最后 stage 使用）
        auto inputs = batch.inputs;
        auto targets = batch.targets;

        // 执行一次训练步骤
        // 内部流程：
        // - 将 inputs 分割为 8 个 micro-batch（每个 micro-batch 的 batch_size = global_batch_size / 8）
        // - 按照 Interleaved 1F1B 调度策略执行：
        //   * Warmup 阶段：stage 0 执行 chunk 0 的 forward（所有 micro-batch），
        //                 stage 1 执行 chunk 2 的 forward，...
        //   * Steady 阶段：每个时间步执行一个新 chunk 的 forward 和一个已完成 forward 的 chunk 的 backward
        //   * Cooldown 阶段：执行剩余的 backward
        // - Stage 间通过 ISend/IRecv 异步传递激活和梯度
        // - 每个 stage 的 optimizer 更新本 stage 包含的参数
        float avg_loss = pp_model->TrainStep(inputs, targets, loss_fn, DataType::kFloat16);

        if (rank == 0) {  // 仅在第一 stage 打印日志
            std::cout << "Epoch " << epoch << ", Loss: " << avg_loss << std::endl;
        }
    }
}

// 关键点：
// 1. 只有 stage 0（rank == 0）需要提供原始 inputs，其他 stage 的 inputs 参数会被忽略
// 2. 只有 stage 3（rank == num_stages-1）需要提供 targets，其他 stage 的 targets 参数会被忽略
// 3. 每个 stage 只更新分配到本 stage 的 chunk 的参数（通过本地的 optimizer 实例）
// 4. Interleaved 调度通过交错执行多个 chunk 的 forward/backward，减少内存占用（相比 GPipe 不需要暂存所有 micro-batch 的激活）
// 5. 通信与计算重叠：ISend/IRecv 是异步的，stage 在等待数据时可执行其他 chunk 的计算
```

## 5. 实现细节

### 模型分割策略
- **均匀分割**: `GetStageInfo()` 将模型的 `total_layers` 层均匀分配到 `pp_size` 个 stage。例如 24 层模型分为 4 个 stage，每个 stage 负责连续的 6 层
- **Interleaved 分割**: 当 `chunks_per_stage > 1` 时，每个 stage 的层进一步分为多个 chunk。例如 24 层模型分为 4 个 stage，每个 stage 2 个 chunk，则总共有 8 个 chunk（chunk 0-7），chunk 0 包含层 0-2，chunk 1 包含层 3-5，...，chunk 7 包含层 21-23。注意这里的分割是"轮转式"的：stage 0 分到 chunk 0 和 chunk 1（层 0-5），stage 1 分到 chunk 2 和 chunk 3（层 6-11），依此类推。这种分割方式允许 Interleaved 1F1B 调度交错执行不同 chunk 的 forward 和 backward

### GPipe 调度算法
- **Warmup 阶段**: 需要花费 `num_stages-1` 个时间步填充流水线。例如 4 个 stage 时，时间步 0: stage 0 执行 mb0 的 forward；时间步 1: stage 0 执行 mb1 的 forward，stage 1 执行 mb0 的 forward；时间步 2: stage 0 执行 mb2 的 forward，stage 1 执行 mb1 的 forward，stage 2 执行 mb0 的 forward
- **Steady 阶段**: 从时间步 `num_stages-1` 开始，每个时间步所有 stage 都在工作：同时执行一个 micro-batch 的 forward 和另一个 micro-batch 的 backward。例如时间步 3: stage 0 执行 mb3 的 forward，stage 1 执行 mb2 的 forward，stage 2 执行 mb1 的 forward，stage 3 执行 mb0 的 forward；时间步 4: stage 0 执行 mb0 的 backward，stage 1 执行 mb3 的 forward，stage 2 执行 mb2 的 forward，stage 3 执行 mb1 的 forward
- **Cooldown 阶段**: 最后的 `num_stages-1` 个时间步排空流水线，只执行 backward
- **总时间步**: `num_micro_batches + 2 * (num_stages - 1)`。例如 8 个 micro-batch，4 个 stage，总时间步 = 8 + 2*3 = 14
- **内存开销**: 需要暂存所有 micro-batch 的中间激活（用于 backward），因此内存占用与 `num_micro_batches * num_stages` 成正比

### Interleaved 1F1B 调度算法
- **核心思想**: 每个 stage 交错执行多个 chunk 的 forward 和 backward，在 warmup 阶段尽快让所有 chunk 完成 forward 并进入 steady 阶段，此时每个时间步执行一个新 chunk 的 forward 和一个已完成 forward 的 chunk 的 backward
- **Warmup 阶段**: 类似 GPipe，但每个 stage 同时启动多个 chunk 的 forward。例如 4 个 stage，每个 stage 2 个 chunk（总共 8 个 chunk），warmup 阶段持续 `num_stages + vpp - 2` 个时间步
- **Steady 阶段**: 每个时间步，stage 执行一个新 chunk 的 forward 和一个已完成 forward 的 chunk 的 backward。这样每个 micro-batch 只需暂存其所在 chunk 的激活，而非所有 micro-batch 的激活
- **内存优势**: 内存占用约为 `num_micro_batches * (num_layers / num_chunks)`，相比 GPipe 的 `num_micro_batches * num_layers` 显著降低（降低倍数为 `vpp`）
- **吞吐损失**: 由于 chunk 间的依赖关系，1F1B 的 steady 阶段吞吐量略低于 GPipe（大约降低 5-10%），但内存效率提升使得可以训练更大的模型或使用更大的 batch size

### 通信机制
- **异步非阻塞**: `ISend` 和 `IRecv` 采用非阻塞语义，立即返回控制权，允许后续计算与通信重叠
- **通信 backend**: 根据 `Device` 类型选择通信实现。对于 GPU 设备，通常使用 NCCL（NVIDIA Collective Communications Library）；对于 CPU 设备，可能使用 MPI 或 gRPC
- **点对点通信**: 流水线 stage 间采用一对一的点对点通信模式，每个 stage 只与前一和后一 stage 通信
- **张量切分**: 每个 micro-batch 的张量可能进一步切分（例如按 sequence length 切分），以减少单次通信的消息大小
- **通信与计算重叠**: 在 1F1B 调度中，stage 在等待接收前一 stage 的激活时，可以执行另一个 chunk 的 backward，从而隐藏通信延迟

### 线程局部存储
- **`thread_local int pp_rank`**: 存储当前线程的 pipeline rank，用于在多线程环境下区分不同 pipeline stage。某些实现可能为每个 stage 分配独立的线程池，每个线程通过 `pp_rank` 识别当前 stage 身份
- **线程安全性**: 当前设计假设每个 stage 在独立线程或进程上执行，因此不需要额外的锁保护。如果多个线程共享同一个 stage，则需要额外的同步机制

### 内存管理
- **Chunk 持有**: 每个 `PipelineStage` 通过 `std::vector<std::shared_ptr<Module>> chunks_` 持有分配到本 stage 的模型 chunk，使用共享指针管理生命周期
- **激活内存**: 前向传播的中间激活存储在每个 chunk 的内部缓存中（由 `Module` 基类管理），backward 时复用这些缓存计算梯度
- **接收缓冲区**: `IRecv` 需要预分配接收缓冲区，形状由 `recv_shape_` 参数提供。缓冲区可能在 stage 初始化时一次性分配，后续 micro-batch 复用
- **优化器状态**: 每个 stage 持有独立的 `Optimizer` 实例，存储本 stage 参数的梯度、动量等状态，内存占用与参数量成正比

### 错误处理
- **Rank 检查**: 构造 `PipelineStage` 时会根据 `stage_index_` 计算 `prev_rank_` 和 `next_rank_`，第一 stage 的 `prev_rank_` 为 -1，最后 stage 的 `next_rank_` 为 -1，用于边界条件处理
- **形状验证**: 用户提供的 `recv_shape` 必须与实际接收到的张量形状匹配，否则 `IRecv` 会失败（底层通信库会检测到大小不匹配）
- **通信超时**: 如果某 stage 崩溃或卡死，其他 stage 在 `IRecv` 时会阻塞或超时（取决于通信 backend 的配置）

### 依赖关系
- **外部依赖**:
  - `Module` 基类（`infini_train/include/nn/modules/module.h`）：所有模型 chunk 都继承自 `Module`，提供 `Forward()` 接口
  - `Tensor` 类（`infini_train/include/tensor.h`）：张量抽象，支持跨设备传输
  - `Device` 类（`infini_train/include/device.h`）：设备抽象，标识计算设备类型（CPU/GPU/NPU）
  - `Optimizer` 类（`infini_train/include/optimizer.h`）：优化器接口，用于更新参数
  - `DataType` 枚举（`infini_train/include/datatype.h`）：数据类型定义（float32、float16、bfloat16 等）
- **内部模块依赖**: 本模块是 `InfiniTrain` 框架的并行训练子模块，与数据并行（`dp`）、张量并行（`tp`）模块协同工作，构成完整的 3D 并行训练体系

### 设计模式
- **Strategy Pattern（策略模式）**: `PipelineSchedule` 作为调度策略的抽象基类，`GPipeSchedule` 和 `Interleaved1F1BSchedule`（未在本头文件中展示，应为子类实现）作为具体策略，用户可通过构造函数参数选择调度算法
- **Facade Pattern（外观模式）**: `PipelineParallel` 作为流水线并行的统一入口，隐藏了内部 stage 初始化、调度器创建、micro-batch 执行等复杂逻辑
- **Builder Pattern（构建器模式）**: `PipelineParallelScheduler` 的静态工厂方法（`GenerateGPipeSchedule`、`GenerateInterleaved1F1BSchedule`）封装了复杂的调度算法实现，用户只需调用静态方法即可获得任务列表
- **Composite Pattern（组合模式）**: 每个 `PipelineStage` 持有多个 `Module` chunk，chunk 本身是一个完整的模型（包含多个层），形成层级结构
- **Iterator Pattern（迭代器模式）**: 调度算法生成的 `Task` 列表可视为时间步的迭代器，`StepMicroBatches()` 按顺序执行每个任务
