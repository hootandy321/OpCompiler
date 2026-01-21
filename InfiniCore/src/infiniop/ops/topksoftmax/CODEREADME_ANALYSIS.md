# TopKSoftmax 算子架构全景

## 1. 子系统职责

TopKSoftmax 算子是 InfiniOP 中的关键融合操作，专门为 Mixture of Experts (MoE) 模型设计。该子系统实现了 **Softmax 归一化 + TopK 选择 + 可选二次归一化** 的三阶段融合操作，用于专家路由选择机制。系统针对四种不同的硬件平台（CPU、NVIDIA GPU、MetaX GPU、通用 CUDA）提供了高度优化的后端实现，通过共享的 CUDA Kernel 核心代码和设备特定的调度逻辑，实现了跨平台的高效专家选择计算。

## 2. 模块导航

* **cpu**: CPU 后端实现，提供基于串行算法的 TopKSoftmax 操作。使用 std::sort 进行排序，时间复杂度 O(N * width * log(width))，适用于小规模推理或无 GPU 环境。支持 F32/F16/BF16 数据类型，通过模板特化实现类型安全的计算。
* **cuda**: CUDA Kernel 核心实现，定义了 `softmax_topk_row_kernel` 融合 kernel。使用 CUB 库的高性能原语（BlockReduce、BlockRadixSort、WarpReduce）实现 Softmax 归一化、块内排序和 TopK 提取的并行计算。这是所有 CUDA 兼容后端（nvidia、metax）共享的核心 kernel 代码。
* **metax**: 沐曦 MetaX GPU 设备后端实现。封装 MetaX 设备句柄（`device::metax::Handle::Internal`），支持根据输入宽度自适应选择 Block Size（128/256/512）。复用 cuda 目录下的 kernel 实现，通过 HCUDA（沐曦的 CUDA 兼容层）API 进行 kernel 调度和内存管理。
* **nvidia**: NVIDIA CUDA 原生后端实现。封装 NVIDIA CUDA 设备句柄，提供与 MetaX 类似的自适应 Block Size 选择策略。直接调用 CUDA Runtime API 和共享的 kernel 实现，是标准的 NVIDIA GPU 后端参考实现。

## 3. 架构逻辑图解

### 数据流与模块交互

```
用户 API 调用
    |
    v
设备后端选择 (cpu / nvidia / metax)
    |           |           |
    |           |           +---> MetaX 实现路径
    |           |                  |
    |           |                  +-- 1. Descriptor::create() 验证张量元数据
    |           |                  +-- 2. calculate() 根据 width 选择 BLOCK_SIZE
    |           |                  +-- 3. launch_topksoftmax<BLOCK_SIZE>() 调度
    |           |                  +-- 4. 调用共享的 softmax_topk_row_kernel
    |           |
    |           +---> NVIDIA 实现路径
    |                  |
    |                  +-- 1. Descriptor::create() 验证张量元数据
    |                  +-- 2. calculate() 根据 width 选择 BLOCK_SIZE
    |                  +-- 3. launch_topksoftmax<BLOCK_SIZE>() 调度
    |                  +-- 4. 调用共享的 softmax_topk_row_kernel
    |
    +---> CPU 实现路径
           |
           +-- 1. Descriptor::create() 验证张量元数据和内存连续性
           +-- 2. calculate() 分发到 topksoftmax_cpu_func<T> 模板函数
           +-- 3. 对每个 token 调用 topksoftmax_cpu_one_token()
           +-- 4. 串行执行 max 查找、exp 计算、softmax、排序
```

### CUDA Kernel 并行执行流程（nvidia 和 metax 共享）

```
CUDA Grid 启动: N 个 Block（每行一个 Block）
    |
    v
每个 CUDA Block 处理一行（一个 token 的所有专家分数）
    |
    +-- 阶段 1: 计算最大值（数值稳定性）
    |      使用 cub::BlockReduce 规约求 max
    |
    +-- 阶段 2: 计算指数和（Softmax 分母）
    |      每线程计算 exp(x - max)，BlockReduce 规约求和
    |
    +-- 阶段 3: Softmax 归一化
    |      每元素除以指数和
    |
    +-- 阶段 4: TopK 排序
    |      使用 cub::BlockRadixSort 对 <value, index> 对降序排序
    |
    +-- 阶段 5: TopK 求和（可选归一化准备）
    |      Warp 0 使用 cub::WarpReduce 对前 topk 个值求和
    |
    +-- 阶段 6: 二次归一化（norm=true 时）
    |      将 TopK 值除以它们的和，确保概率和为 1
    |
    +-- 阶段 7: 写入全局内存
           前 topk 个线程写入 values 和 indices 输出数组
```

### 模块依赖关系

**共享组件**:
- `info.h`: `TopksoftmaxInfo` 元数据结构（所有后端共享）
- `topksoftmax.h`: Descriptor 宏定义（生成设备特定描述符类）
- `cuda/kernel.cuh`: CUDA Kernel 实现（nvidia 和 metax 共享）

**后端特定依赖**:
- `cpu`: 依赖标准库（`<algorithm>`, `<vector>`），无外部 GPU 库依赖
- `cuda`: 依赖 CUB 库（`cub/block/block_reduce.cuh`, `cub/block/block_radix_sort.cuh`）
- `nvidia`: 依赖 CUDA Runtime、NVIDIA 设备句柄（`device::nvidia::Handle`）
- `metax`: 依赖 MetaX 驱动（hcblas/hcdnn）、MetaX 设备句柄

### 关键设计差异

| 维度 | CPU | NVIDIA/MetaX (CUDA) |
|------|-----|---------------------|
| **并行模型** | 串行执行，O(N*width*log(width)) | 数据并行，O(N*width/log(BLOCK_SIZE)) |
| **排序算法** | std::sort（内省排序） | cub::BlockRadixSort（基数排序） |
| **归约策略** | 串行累加 | cub::BlockReduce/WarpReduce（并行规约） |
| **内存需求** | 无额外工作空间 | 依赖 CUB 共享内存（自动管理） |
| **数值稳定性** | max 减法技巧 | max 减法技巧 + CUDA 内置函数 |
| **Block Size** | 不适用 | 自适应选择 128/256/512 |

### 性能优化策略对比

**CPU 实现优化**:
- 内存布局约束：强制最后一维连续（步长为 1）提升缓存命中率
- 编译期类型特化：`if constexpr` 避免运行时类型判断
- 数值稳定性：使用 `exp(x - max)` 防止溢出

**CUDA 实现优化**（nvidia + metax 共享）:
- 融合 Kernel：将 Softmax、排序、TopK 提取融合为单 kernel，减少全局内存访问
- 自适应并行度：根据输入宽度动态选择 Block Size（128/256/512），最大化 GPU 利用率
- CUB 原语集成：使用高度优化的 BlockReduce、BlockRadixSort、WarpReduce
- 最小化全局内存写入：仅前 topk 个线程写入输出结果
- Warp 级优化：TopK 提取仅使用第一个 warp，减少线程分歧

### 典型应用场景

该子系统主要用于 **Mixture of Experts (MoE)** 模型的专家路由：

1. **输入**: 门控网络对每个 token 输出的专家分数 logits `[N, num_experts]`
2. **Softmax**: 将分数转换为概率分布（所有专家概率和为 1）
3. **TopK**: 选择概率最高的 K 个专家（例如 top-8）
4. **Norm（可选）**: 对 TopK 概率重新归一化，确保这 K 个专家权重和为 1
5. **输出**: TopK 专家的索引和归一化权重，用于后续的专家激活和加权聚合

**输出用途**:
- `indices`: 确定激活哪些专家
- `values`: 专家输出的加权系数

### 约束与限制

**所有后端通用约束**:
- 输入必须是 2D 张量 `[N, width]`
- 最后一维必须内存连续（`strides[1] == 1`）
- 支持的数据类型：F32、F16、BF16

**后端特定约束**:
- **CPU**: 无额外约束，但性能随 width 增长而下降
- **CUDA (nvidia/metax)**: width 必须 <= 512（Block Size 上限）
- **CUDA**: topk 必须 <= 32（单个 warp 处理能力限制）
