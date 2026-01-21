# InfiniTrain 项目架构全景

## 1. 项目职责

**InfiniTrain** 是一个从零开始构建的大规模模型 C++ 训练框架，专注于支持**多维分布式并行训练**。该项目在 Infini 生态系统中承担着深度学习模型训练的核心职责，向下依赖 InfiniCore 提供的张量运算基础设施，向上为大规模语言模型（如 GPT-2、LLaMA 3）和深度学习模型提供完整的训练能力。

InfiniTrain 的核心定位包括：

1. **全功能训练框架**：提供从数据加载、模型定义、前向传播、自动微分、反向传播到参数优化的完整训练流程
2. **多维并行支持**：支持数据并行（DP）、分布式数据并行（DDP）、张量并行（TP）、序列并行（SP）和流水线并行（PP），并可任意组合实现 3D/4D 混合并行
3. **生产级性能**：通过计算-通信重叠、梯度桶优化、自动混合精度、高性能计算内核等技术实现高效的大规模训练
4. **多硬件后端**：支持 CPU、CUDA（含 NCCL）、Kunlun、Metax 等多种硬件加速器
5. **PyTorch 兼容**：API 设计高度模仿 PyTorch，便于模型迁移和开发者上手

## 2. 模块导航 (Module Navigation)

### 2.1 核心代码库 (infini_train/)

* **📂 `infini_train/`**：训练框架核心实现
  * *功能*：提供完整的深度学习训练基础设施，包括统一数据抽象层（Tensor、Device、DataType）、自动微分引擎（autograd）、高性能计算内核库（kernels）、神经网络模块系统（nn/modules）和分布式并行训练基础设施（nn/parallel）
  * *职责*：作为整个训练框架的技术基石，从基础数据结构到大规模分布式并行训练的全栈实现
  * *核心子系统*：
    - **include/**：公共接口层，定义 tensor.h、device.h、datatype.h、autocast.h、dispatcher.h、optimizer.h、dataloader.h、dataset.h、profiler.h 等核心头文件
    - **src/autograd/**：自动微分引擎实现，提供 Function 基类、梯度模式管理、梯度后处理钩子和 50+ 种可微分算子
    - **src/kernels/**：计算内核库，提供 CPU/CUDA 双后端的深度学习算子实现（线性代数、归一化、损失函数、激活函数、张量操作、分布式通信原语）
    - **src/nn/modules/**：神经网络模块系统，提供 Module 基类和具体层实现（Linear、LayerNorm、Embedding、Sigmoid、CrossEntropyLoss）及容器（Sequential、ModuleList、ModuleDict）
    - **src/nn/parallel/**：分布式并行训练实现，支持数据并行（DP）、张量并行（TP）、流水线并行（PP）和混合并行策略

### 2.2 模型示例 (example/)

* **📂 `example/`**：训练示例与模型实现
  * *功能*：提供各类模型的训练示例，包括 GPT-2、LLaMA 3、MNIST 等模型的完整训练代码
  * *职责*：作为框架使用的参考实现，展示如何使用 InfiniTrain 构建和训练不同类型的深度学习模型
  * *子目录*：
    - **common/**：公共工具代码和数据加载器
    - **gpt2/**：GPT-2 模型训练示例（Decoder-only Transformer 语言模型）
    - **llama3/**：LLaMA 3 模型训练示例（现代 LLaMA-family Transformer 架构）
    - **mnist/**：MNIST 手写数字识别示例（入门级图像分类任务）
  * *文档状态*：无独立文档，示例代码即为使用说明

### 2.3 构建系统 (cmake/)

* **📂 `cmake/`**：CMake 构建配置
  * *功能*：提供 CMake 构建脚本和模块查找配置
  * *职责*：管理项目的编译配置、依赖项查找（如 FindNCCL.cmake）和构建选项
  * *核心组件*：
    - **FindNCCL.cmake**：NCCL（NVIDIA Collective Communications Library）查找脚本，用于分布式 GPU 训练
  * *文档状态*：无独立文档

### 2.4 工具脚本 (scripts/)

* **📂 `scripts/`**：辅助工具与自动化脚本
  * *功能*：提供代码格式化、性能测试、结果分析等辅助工具
  * *职责*：提升开发效率和训练流程自动化程度
  * *核心组件*：
    - **format.py**：代码格式化工具
    - **run_models_and_profile.bash**：自动化模型运行和性能分析脚本
    - **test_config.json**：测试配置文件
    - **write_to_feishu_sheet.py**：飞书表格导出工具（自动化性能报告生成）
  * *文档状态*：无独立文档

### 2.5 运行工具 (tools/)

* **📂 `tools/`**：训练启动工具
  * *功能*：提供分布式训练的启动和管理工具
  * *职责*：简化多机多卡训练的启动流程
  * *核心组件*：
    - **infini_run/**：分布式训练启动器，支持多节点、多进程的并行训练启动
  * *文档状态*：无独立文档

### 2.6 第三方依赖 (third_party/)

* **📂 `third_party/`**：第三方依赖库
  * *功能*：管理项目所需的外部依赖项
  * *职责*：隔离第三方库，避免与系统库冲突
  * *文档状态*：无独立文档（通常包含第三方库的源码或预编译二进制）

## 3. 架构逻辑图解

### 3.1 项目整体架构

```
InfiniTrain/
├── infin_train/              # 核心训练框架实现
│   ├── include/              # 公共接口层（声明）
│   │   ├── 基础抽象层:
│   │   │   ├── tensor.h      # Tensor 核心数据结构
│   │   │   ├── device.h      # 设备管理（CPU/CUDA）
│   │   │   └── datatype.h    # 类型系统（FP16/FP32/BF16）
│   │   │
│   │   ├── 计算图与优化层:
│   │   │   ├── autograd/     # 自动微分引擎接口
│   │   │   └── optimizer.h   # 优化器抽象（SGD、Adam）
│   │   │
│   │   ├── 神经网络层:
│   │   │   ├── nn/modules/   # 基础层接口（Linear、LayerNorm 等）
│   │   │   └── nn/parallel/  # 并行策略接口（DP、TP、PP）
│   │   │
│   │   ├── 硬件抽象层:
│   │   │   └── common/       # 跨平台工具（CPU/CUDA）
│   │   │
│   │   └── 系统服务层:
│   │       ├── dataloader.h  # 数据加载器
│   │       ├── dataset.h     # 数据集抽象
│   │       ├── profiler.h    # 性能分析工具
│   │       ├── autocast.h    # 自动混合精度
│   │       └── dispatcher.h  # 算子分发系统
│   │
│   └── src/                  # 核心实现层（实现）
│       ├── autograd/         # 自动微分引擎实现（50+ 算子）
│       ├── kernels/          # 计算内核库（CPU/CUDA 双后端）
│       └── nn/               # 神经网络组件实现
│           ├── modules/      # 神经网络模块
│           └── parallel/     # 分布式并行实现
│
├── example/                  # 模型训练示例
│   ├── common/               # 公共工具代码
│   ├── gpt2/                 # GPT-2 训练示例
│   ├── llama3/               # LLaMA 3 训练示例
│   └── mnist/                # MNIST 训练示例
│
├── cmake/                    # 构建系统
│   └── FindNCCL.cmake        # NCCL 查找脚本
│
├── scripts/                  # 辅助工具
│   ├── format.py             # 代码格式化
│   ├── run_models_and_profile.bash  # 性能测试脚本
│   └── write_to_feishu_sheet.py     # 飞书报表导出
│
├── tools/                    # 运行工具
│   └── infini_run/           # 分布式训练启动器
│
└── third_party/              # 第三方依赖
```

### 3.2 模块间依赖关系

```
┌─────────────────────────────────────────────────────┐
│                  用户应用层                          │
│  (example/gpt2, example/llama3, example/mnist)     │
└────────────────────┬────────────────────────────────┘
                     │ 使用
                     ↓
┌─────────────────────────────────────────────────────┐
│           infin_train 核心框架层                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  nn/modules/                                │   │
│  │  (Linear, LayerNorm, Embedding, ...)        │   │
│  └─────────────────────────────────────────────┘   │
│                     │ 依赖                          │
│  ┌────────────────▼─────────────────────────────┐  │
│  │  autograd/                                   │  │
│  │  (自动微分引擎，50+ 可微分算子)              │  │
│  └─────────────────────────────────────────────┘  │
│                     │ 调用                          │
│  ┌────────────────▼─────────────────────────────┐  │
│  │  kernels/                                    │  │
│  │  (CPU/CUDA 双后端计算内核)                   │  │
│  └─────────────────────────────────────────────┘  │
│                     │ 依赖                          │
│  ┌────────────────▼─────────────────────────────┐  │
│  │  基础抽象层                                   │  │
│  │  (tensor.h, device.h, datatype.h, ...)       │  │
│  └─────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                     │ 依赖
                     ↓
┌─────────────────────────────────────────────────────┐
│           InfiniCore 基础设施层                      │
│     (张量运算、设备管理、基础算子库)                 │
└─────────────────────────────────────────────────────┘
```

### 3.3 分布式训练数据流

```
[训练启动阶段]
     ↓
tools/infini_run/infini_run 启动多进程
     ↓
设置环境变量（WORLD_SIZE、RANK、MASTER_ADDR等）
     ↓
example/llama3/llama3 可执行文件启动
     ↓
     ↓
[环境初始化阶段]
     ↓
GlobalEnv::Init(nthread_per_process, tp_size, sequence_parallel, pp_size, vpp_size)
     ↓
计算 data_parallel_size = world_size / (tp_size * pp_size)
     ↓
创建 3D 拓扑布局（Layout），支持 RankOf(dp, tp, pp) 坐标转换
     ↓
创建进程组（ProcessGroup）：DP、TP、PP 独立通信域
     ↓
     ↓
[模型构建阶段]
     ↓
用户使用 nn::modules::Module 构建神经网络
     ↓
┌─────────────────────────────────────────┐
│ 根据并行策略替换层                       │
│ - TP: ColumnParallelLinear、            │
│        RowParallelLinear、              │
│        VocabParallelEmbedding           │
│ - PP: 将模型按层切分为多个 chunk         │
│ - DP: 使用 DistributedDataParallel 包装 │
└─────────────────────────────────────────┘
     ↓
     ↓
[训练迭代阶段]
     ↓
┌──────────────────────────────────────────┐
│ 前向传播                                  │
│ ├─ 数据加载器读取批次                     │
│ ├─ 按微批次切分（PP）                     │
│ ├─ 各层前向计算                           │
│ ├─ 构建计算图（autograd）                 │
│ ├─ TP: AllGather/AllReduce 通信          │
│ └─ PP: ISend/IRecv 传递激活值            │
└──────────────────────────────────────────┘
     ↓
┌──────────────────────────────────────────┐
│ 反向传播                                  │
│ ├─ 计算损失函数                           │
│ ├─ 按计算图逆序反向传播                   │
│ ├─ TP: AllReduce/Split 同步梯度           │
│ ├─ PP: IRecv/ISend 传递梯度               │
│ └─ DP: Reducer 收集梯度到桶               │
└──────────────────────────────────────────┘
     ↓
┌──────────────────────────────────────────┐
│ 梯度同步与参数更新                        │
│ ├─ DP: Reducer 触发梯度桶 AllReduce       │
│ ├─ 优化器更新参数（SGD/Adam）             │
│ └─ 清空梯度，准备下一轮迭代               │
└──────────────────────────────────────────┘
```

### 3.4 并行策略协同工作流

```
[全局环境初始化]
     ↓
根据命令行参数计算并行度：
- total_batch_size = 10240
- batch_size = 40
- tensor_parallel = 2 (TP)
- pipeline_parallel = 2 (PP)
- nthread_per_process = 8
     ↓
推导隐含并行度：
- data_parallel_size = 8 / (2 * 2) = 2 (DP)
- num_micro_batches = total_batch_size / batch_size = 256
     ↓
创建 3D 拓扑布局：
Layout(DP=2, TP=2, PP=2) → 8 个进程
     ↓
创建独立进程组：
- dp_group: 2 个进程（跨节点）
- tp_group: 2 个进程（节点内）
- pp_group: 2 个进程（流水线阶段）
     ↓
[模型构建与包装]
     ↓
原始模型（如 LLaMA 3）
     ↓
┌─────────────────────────────────────────┐
│ 应用张量并行（TP）                        │
│ - Linear → ColumnParallelLinear/         │
│            RowParallelLinear             │
│ - Embedding → VocabParallelEmbedding     │
└─────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────┐
│ 应用流水线并行（PP）                      │
│ - 将模型 layers 分为 2 个 chunks         │
│ - PipelineParallel 包装，指定 chunks     │
└─────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────┐
│ 应用数据并行（DP）                        │
│ - DistributedDataParallel 包装           │
│ - 初始化 Reducer 和梯度桶                │
└─────────────────────────────────────────┘
     ↓
[训练执行]
     ↓
每次迭代处理 256 个微批次：
     ↓
┌──────────────────────────────────────────┐
│ 微批次 k 的前向传播                       │
│ ├─ PP: PipelineSchedule 调度             │
│ │   - GPipe: 简单的阶段流水线            │
│ │   - 1F1B: 内存优化的调度               │
│ ├─ TP: 在 ColumnParallelLinear 中        │
│ │   - AllGather 输入（如果需要）         │
│ │   - 本地 GEMM 计算                     │
│ │   - 在 RowParallelLinear 中            │
│ │   - 本地 GEMM 计算                     │
│ │   - AllReduce 输出（如果需要）          │
│ └─ PP: ISend 发送激活值到下一 stage      │
└──────────────────────────────────────────┘
     ↓
┌──────────────────────────────────────────┐
│ 微批次 k 的反向传播                       │
│ ├─ PP: IRecv 从下一 stage 接收梯度        │
│ ├─ TP: 在 ColumnParallelLinear 中        │
│ │   - AllReduce 梯度                     │
│ │   - 在 RowParallelLinear 中            │
│ │   - Split 梯度                         │
│ └─ DP: Reducer 收集梯度到桶              │
└──────────────────────────────────────────┘
     ↓
[所有微批次完成后]
     ↓
┌──────────────────────────────────────────┐
│ 梯度同步（DP）                            │
│ - Reducer->FinalizeBackward()            │
│ - 等待所有梯度桶 AllReduce 完成           │
│ - 所有 DP 进程获得平均梯度                │
└──────────────────────────────────────────┘
     ↓
┌──────────────────────────────────────────┐
│ 参数更新                                  │
│ - Optimizer::Step() 更新参数              │
│ - Optimizer::ZeroGrad() 清空梯度          │
└──────────────────────────────────────────┘
```

### 3.5 核心技术特性

**1. 自动微分系统**：
- 动态计算图构建：前向传播时自动构建计算图
- 自动反向传播：loss->Backward() 触发梯度计算
- 50+ 可微分算子：覆盖线性代数、归一化、损失函数、激活函数等
- 分布式梯度同步：通过 PostAccumulateGradHook 钩子集成 AllReduce

**2. 多硬件后端支持**：
- CPU 后端：基于 Eigen 库优化
- CUDA 后端：集成 cuBLAS（矩阵乘法）、CUB（并行归约）
- 其他后端：Kunlun、Metax、Ascend 等国产芯片
- Dispatcher 机制：根据设备类型自动分发到对应后端

**3. 高性能优化**：
- 计算-通信重叠：异步通信与计算并行执行
- 梯度桶优化：减少 AllReduce 启动次数
- 自动混合精度（Autocast）：BF16 计算、FP32 累积
- 流水线调度：GPipe 和 1F1B 调度算法
- 微批次切分：降低峰值内存占用

**4. PyTorch 兼容性**：
- Module、Tensor、Optimizer、DataLoader API 设计
- 参数命名和状态字典格式兼容
- 支持 no_grad 推理模式
- Autocast 机制对应 torch.cuda.amp.autocast

## 4. 架构特色与优势

### 4.1 全栈自研
- 从零构建的训练框架，完全控制底层实现
- 不依赖 PyTorch 等第三方训练框架，仅依赖 InfiniCore 和标准库
- 便于针对特定硬件和应用场景进行深度优化

### 4.2 多维并行支持
- 支持 DP、DDP、TP、SP、PP 五种并行策略
- 可任意组合实现 3D/4D 混合并行（如 DDP + TP + SP + PP）
- 独立进程组隔离不同并行维度的通信，避免通信冲突

### 4.3 生产级性能
- 计算内核高度优化：CPU 使用 Eigen，CUDA 使用 cuBLAS/CUB
- 通信优化：NCCL 高速通信、梯度桶减少延迟、异步流水线
- 内存优化：微批次切分、梯度视图、activation 缓存复用
- 混合精度训练：BF16 存储、FP32 累积，兼顾性能和精度

### 4.4 易用性与可扩展性
- PyTorch 风格 API，降低学习成本
- 模块化设计，易于添加新的神经网络层
- Dispatcher 机制，易于添加新硬件后端支持
- 完善的错误检查和性能分析工具（Profiler）

### 4.5 大规模训练能力
- 支持单机多卡和多机多卡训练
- 通过 3D 并行可扩展到数百张 GPU
- 支持超大规模模型训练（如 DeepSeek-V3 MoE 模型）

## 5. 应用场景

### 5.1 已支持模型
- **GPT-2**：Decoder-only Transformer 语言模型
- **LLaMA 3**：现代 LLaMA-family Transformer 架构
- **MNIST**：手写数字识别（入门示例）

### 5.2 计划支持模型
- **DeepSeek-V3**：大规模 MoE（Mixture of Experts）语言模型

### 5.3 适用场景
- 大语言模型预训练和微调
- 多模态模型训练
- 超大规模神经网络训练
- 需要自定义并行策略的研究项目

## 6. 性能指标

根据项目 README 提供的信息，InfiniTrain 的性能特性包括：

- **精度支持**：FP32、BF16 及混合精度训练
- **并行策略**：DP、DDP、TP、SP、PP 及混合并行
- **优化技术**：计算-通信重叠、DDP 梯度桶优化、ZeRO-DP（开发中）
- **工具链**：内置性能分析器、自动化性能基准测试

## 7. 开发与构建

### 7.1 系统要求
- **硬件**：推荐 NVIDIA Ampere-class GPU (A100/A800) 或更新
- **软件**：
  - CUDA / NCCL：最新稳定版
  - gcc / g++：13+
  - CMake：3.13+

### 7.2 构建流程
```bash
mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DUSE_NCCL=ON
make -j
```

### 7.3 运行训练
- **单节点**：直接运行 `./llama3 --device cuda ...`
- **多节点**：使用 `./infini_run --nnodes=2 ... ./llama3 ...`

## 8. 项目演进路线

根据项目 README 提供的路线图：

- **v0.1.0** (2025/03/10)：初始框架原型，MNIST CPU 训练
- **v0.3.0** (2025/04/30)：添加 Autograd 支持，GPT-2 CPU/CUDA 训练
- **v0.4.0** (2025/07/09)：内核注册机制、LLaMA CPU/CUDA 训练、BF16 精度、数据并行
- **v0.5.0** (2025/12/31)：Autocast、多维分布式并行（DDP/TP/SP/PP）、多节点训练、no_grad 模式、计算-通信重叠

## 9. 总结

InfiniTrain 是一个**全栈自研的大规模模型 C++ 训练框架**，通过清晰的分层架构和模块化设计，提供从基础数据抽象到多维分布式并行训练的完整解决方案。

**核心优势**：
- **完整性**：涵盖训练流程的所有环节（数据、模型、训练、优化、部署）
- **高性能**：多硬件后端、计算-通信重叠、混合精度等优化技术
- **可扩展**：支持 DP、TP、PP 混合并行，可扩展到数百张 GPU
- **易用性**：PyTorch 风格 API，完善的工具链和示例代码

**项目定位**：
在 Infini 生态系统中，InfiniTrain 承担着**深度学习模型训练**的核心职责，是连接底层基础设施（InfiniCore）和上层应用（大语言模型、多模态模型）的桥梁，为大规模 AI 模型训练提供高效、灵活、可扩展的解决方案。
