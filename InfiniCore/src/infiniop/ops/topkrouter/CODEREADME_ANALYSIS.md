# TopKRouter 算子架构全景

## 1. 子系统职责

TopKRouter 是混合专家模型（MoE，Mixture of Experts）中的核心路由算子，负责从 256 个专家中动态选择 Top-K 个最优专家，并计算归一化的专家权重。该算子在 InfiniOp 框架中实现了跨多种硬件后端的高性能专家路由，支持 NVIDIA CUDA、华为昇腾（MetaX、 kunlun）和通用 CPU 平台。TopKRouter 通过层级化的 Warp 级并行排序策略，在 GPU 上实现了 O(256 log 256) 时间复杂度的专家选择，并支持 FP32、FP16 和 BF16 三种数据类型，为大模型 MoE 推理提供了高效的路由决策能力。

## 2. 模块导航

* **cpu**: 文档缺失，CPU 后端实现细节待补充
* **cuda**: 文档缺失，CUDA 通用实现核心文档待补充
* **kunlun**: 文档缺失，昆仑芯片后端实现文档待补充
* **metax**: MetaX 后端实现了基于华为昇腾设备的 TopK 专家路由算子，采用层级化 Warp 级排序策略，通过 CUB 原语在 CUDA Block 内进行高效的 Top-K 选择，支持 Sigmoid 激活、偏置校正和 Softmax 归一化，固定配置为 width=256 和 BLOCK_SIZE=256
* **nvidia**: NVIDIA CUDA 后端实现了高性能专家路由算子，通过层级化并行排序策略（Warp 级预选 → Block 级精选 → 归一化）完成 Top-K 专家选择，使用 CUB 库的 WarpMergeSort 和 BlockRadixSort 优化排序性能，每个 token 独立分配一个 Block，实现 token 间完全并行

## 3. 架构逻辑图解

TopKRouter 算子的数据流和执行流程如下：

**输入阶段**：算子接收输入张量 `x [N, 256]`（N 个 token 对应 256 个专家的 logits）和校正偏置 `correction_bias [256]`。各后端首先验证输入张量的元数据（数据类型、形状、步长），要求输入为 2D 浮点张量（F32/F16/BF16），且第二维必须连续存储（stride[1] == 1）。MetaX 和 NVIDIA 后端通过 `TopkrouterInfo` 类统一管理张量元信息，并创建包含设备句柄的 Descriptor 对象。

**计算阶段**：采用分层并行策略在 GPU 上执行 Top-K 路由。每个 token 分配一个独立的 CUDA Block（256 个线程，对应 256 个专家）。在 Block 内部，首先对每个专家的 logit 执行 Sigmoid 激活并加上校正偏置，然后将 256 个线程分为 8 个 Warp（每组 32 线程）。每个 Warp 内部使用 `cub::WarpMergeSort` 对其 32 个专家权重降序排序，保留前 2 个最大值，共产生 16 个候选专家（8 Warp × 2）。接着，每个 Warp 的前 2 个值求和得到 8 个组分数，Warp 0 对这 8 个组分数排序并标记前 4 个组为有效，淘汰后 4 个组的所有专家。最后，全 Block 使用 `cub::BlockRadixSort` 对所有 256 个专家权重降序排序，选取前 K 个专家。

**输出阶段**：对选出的 Top-K 专家进行 Softmax 归一化。Warp 0 对 Top-K 专家对应的原始 sigmoid 值求和，然后将每个值除以该和并乘以路由缩放因子 `routed_scaling_factor`，最终输出归一化的专家权重 `values [N, topk]` 和对应的专家索引 `indices [N, topk]`。

**并发与优化**：各后端采用 Block 级并行（每 token 一个 Block）、Warp 级协作（CUB 原语）和共享内存优化（`share_data[256]` 存储中间结果）来最大化 GPU 利用率。MetaX 和 NVIDIA 后端共享相同的 CUDA kernel 实现（定义在 `cuda/kernel.cuh`），通过模板特化支持多种数据类型，实现了零拷贝工作空间和类型安全的设备函数（`exp_func`、`sigmoid_func`）。
