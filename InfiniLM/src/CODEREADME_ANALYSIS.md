# 目录: InfiniLM/src 架构全景

## 1. 子系统职责

`InfiniLM/src` 是 InfiniLM 项目的核心源代码目录，负责实现大语言模型（LLM）的高性能推理引擎。该目录采用分层模块化架构，从底层的张量计算和内存管理，到中层的缓存管理和权重加载，再到上层的多模型实现（Jiuge、DeepSeek V3、JiugeAWQ），形成了完整的推理技术栈。该目录的设计目标是支持多设备并行推理、灵活的量化策略和高效的内存管理，为生产环境的大语言模型部署提供基础设施。

## 2. 模块导航

* **allocator**:
    * *功能*: 实现高性能内存池（MemoryPool），支持快速分配、自动合并和对齐管理
    * *职责*: 为推理过程中的临时缓冲区提供零拷贝内存管理，避免频繁的设备内存分配/释放开销，采用最佳适应（best-fit）算法和自动合并策略优化内存利用率

* **cache_manager**:
    * *功能*: 提供 KV Cache（Key-Value Cache）的创建、复制和销毁接口
    * *职责*: 管理多层多设备的 KV 缓存，支持张量并行下的 KV 分片（每设备 nkvh/ndev 个头），为增量推理提供缓存复用机制，减少重复计算

* **dataloader**:
    * *功能*: 实现权重加载器（WeightLoader），支持 FULL/ROW/COLUMN 三种分发策略
    * *职责*: 从主机内存加载模型权重到设备内存，支持多设备并行加载和张量分片（COLUMN 模式下按列重排），通过异步流提升加载效率，提供 C 兼容接口 `loadModelWeight()`

* **models/jiuge**:
    * *功能*: 实现九格（Jiuge）大语言模型的完整推理引擎，支持多设备张量并行和 GQA（分组查询注意力）
    * *职责*: 提供 32 层 Transformer 的前向传播、RoPE 位置编码、批量推理和随机采样（temperature/top-k/top-p），每设备一个推理线程，通过条件变量实现主线程与工作线程的同步，支持 W8A8 量化和混合精度计算（FP16/BF16/FP32）

* **models/deepseek_v3**:
    * *功能*: 实现 DeepSeek V3 混合专家模型（MoE），结合密集层和稀疏层，支持 MLA（多头潜在注意力）和 Top-K 路由
    * *职责*: 提供 W8A8 量化的 MoE 推理，通过 KV 压缩（r_kv=512）大幅减少缓存占用，共享专家 + Top-8 路由专家的加权输出，多设备 AllReduce 同步，支持 61 层（24 密集 + 37 稀疏）的流水线执行

* **models/jiuge_awq**:
    * *功能*: 实现九格模型的 AWQ（Activation-aware Weight Quantization）INT4 量化推理
    * *职责*: 支持分组量化（group_size=128），每 4 个 INT4 权重打包存储，反量化与矩阵乘融合为单次算子调用，与 Jiuge 模型共享相同的架构设计（GQA、RoPE、多线程调度），在保持精度的同时将权重显存减少 4 倍

* **tensor**:
    * *功能*: 实现张量抽象层（Tensor），提供设备无关的数据容器和描述符
    * *职责*: 封装存储（Storage）、描述符（TensorDesc）和视图操作，支持连续/非连续内存布局、形状变换（view/view_as）、数据加载（H2D/D2H）和张量复制，为上层算子提供统一的接口

## 3. 架构逻辑图解

InfiniLM/src 的架构设计遵循"分层抽象、垂直集成"的原则，从底层硬件抽象到上层模型实现形成清晰的数据流和依赖关系：

### 3.1 数据流与模块交互

```
权重加载阶段：
主机权重文件 → dataloader (WeightLoader)
              ↓ [COLUMN 分发策略]
              多设备权重张量 (各设备独立流)

推理初始化阶段：
allocator (MemoryPool) → 预分配 128MB 显存池
cache_manager (createKVCache) → 分配多层 KV Cache
              ↓
models/* (createModel) → 加载权重、启动推理线程、同步初始化

批量推理阶段：
用户请求 → models/* (inferBatch/forwardBatch)
         ↓ [主线程设置 InferRequest，唤醒工作线程]
         [设备线程并行执行]
         ↓
         1. tensor (Tensor::buffer) → 分配中间缓冲区
         2. 词嵌入查找 (infiniop)
         3. 逐层计算：
            - RMSNorm (infiniop)
            - QKV 投影 (dequant_linear for AWQ/V3)
            - RoPE 旋转 (infiniop)
            - cache_manager (KV Cache 更新)
            - 注意力计算 (GQA/MQA)
            - FFN (SwiGLU/MoE)
            - AllReduce (多设备)
         4. 输出采样或 logits 返回
```

### 3.2 核心依赖关系

**纵向依赖（从底层到上层）**：
- `tensor` → 被所有模块依赖，提供数据容器和算子接口
- `allocator` → 被 `tensor::buffer()` 和模型推理（临时缓冲区）依赖
- `cache_manager` → 被 `models/*` 依赖，管理推理过程中的 KV 缓存
- `dataloader` → 被 `models/*` 的权重加载阶段依赖
- `models/*` → 顶层模块，聚合所有底层组件实现完整推理

**横向协作（同级模块）**：
- `models/jiuge`、`models/deepseek_v3`、`models/jiuge_awq` → 共享相同的基础设施（tensor、allocator、cache_manager），实现不同的模型架构和量化策略
- `dataloader` 与 `cache_manager` → 分别负责静态权重加载和动态缓存管理，共同支撑推理的内存需求

### 3.3 设计模式与架构决策

**内存管理策略**：
- **预分配池**：`allocator` 通过 MemoryPool 预分配大块显存，推理过程中从池中分配，避免系统调用开销
- **KV Cache 复用**：`cache_manager` 的 KV Cache 跨请求共享，仅更新新增 token 的 K/V
- **权重共享**：所有权重使用 `std::shared_ptr` 管理，多设备间共享只读权重

**并行计算策略**：
- **张量模型并行**：`models/*` 将注意力头和 FFN 中间层按设备切分，每层后 AllReduce 聚合
- **线程级并行**：每设备一个专用推理线程（`launchDevice`），主线程通过条件变量调度
- **流水线并行**：`dataloader` 使用异步流加载权重，隐藏 PCIe 传输延迟

**量化优化策略**：
- **Jiuge**：支持标准精度（FP16/BF16/FP32）
- **JiugeAWQ**：INT4 分组量化，权重显存减少 4 倍
- **DeepSeek V3**：W8A8 量化 + MLA KV 压缩，综合显存优化

### 3.4 关键技术实现

**GQA（分组查询注意力）**：
- `models/jiuge` 和 `models/jiuge_awq` 实现了 nh 个查询头共享 nkvh 个 KV 头（nkvh << nh）
- 每 `ngroup = nh/nkvh` 个查询头共享一组 KV，减少 KV Cache 内存占用
- 计算时将 Q 重排为 `[nkvh, ngroup, seq_len, dh]`，K/V 为 `[nkvh, total_len, dh]`

**MoE（混合专家）**：
- `models/deepseek_v3` 实现共享专家 + Top-8 路由专家的加权输出
- 门控网络计算每个 token 对所有专家的亲和度，选择 Top-8 并归一化
- 每个专家独立计算 FFN，最终输出 = 共享专家输出 + Σ(路由权重 × 专家输出)

**RoPE（旋转位置编码）**：
- 所有模型实现 RoPE v2，预计算 sin/cos 表 `[dctx, dh/2]`
- 支持任意长度位置索引，频率公式 `θ_i = θ^(-2i/dh)`

**线程同步机制**：
- 主线程（生产者）准备 `InferRequest`，通过 `cv_start` 通知所有设备
- 设备线程（消费者）等待 `proceed` 标志，执行推理后通过 `cv_done` 通知主线程
- 初始化时等待 `cv_load` 确保所有设备就绪，销毁时设置 `exit_flag` 优雅退出

### 3.5 扩展性设计

**新模型接入**：继承 `tensor`、`allocator`、`cache_manager` 基础设施，实现新的推理内核
**新量化格式**：扩展 `QuantLinearWeight` 结构，更新 `dequant_linear` 算子
**新硬件后端**：通过 `infiniDevice_t` 抽象，支持 CUDA、ROCm、Kunlun 等设备
**新分发策略**：`dataloader` 支持自定义 `DistributionType`（FULL/ROW/COLUMN）

---

**文档生成说明**：本文档基于子目录的 CODEREADME.md 和源代码分析生成，覆盖所有 7 个子目录的架构信息。对于无文档的模块（allocator、cache_manager、dataloader、tensor），通过源代码分析提取了核心功能和职责。建议为这些模块补充独立的 CODEREADME.md 以提供更详细的实现细节。
