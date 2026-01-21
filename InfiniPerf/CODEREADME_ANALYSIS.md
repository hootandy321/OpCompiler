# InfiniPerf 架构全景

## 1. 子系统职责

InfiniPerf 是 Infini 生态系统中的**全谱系性能基准测试套件**，负责评估加速器卡的计算性能、硬件特性和框架兼容性。作为性能评估的核心子系统，它通过多层次、多维度的测试方案，为硬件开发者、框架优化者和终端用户提供客观、可复现的性能指标。

该子系统采用分层架构设计，从底层硬件特性（访存带宽、缓存性能、通信延迟）到中间层算子性能（矩阵乘、不同精度计算），再到上层框架兼容性（Megatron-LM、PaddlePaddle、InfiniTensor）和实际大模型训练推理性能，形成完整的性能评估体系。

InfiniPerf 在整体架构中的定位：
- **上游依赖**：依赖 InfiniCore 提供的算子测试框架（InfiniCore/test/infiniop-test）
- **横向协作**：与 InfiniLM（大模型训练）和 InfiniTrain（训练优化）形成性能验证闭环
- **输出价值**：为硬件选型、框架优化和系统调优提供数据支撑

## 2. 模块导航

### 一级子目录概览

- **benchmarks**: 核心基准测试套件，包含所有性能测试模块
  - 功能：组织和管理四类测试（兼容性、计算、硬件、大模型），提供统一的测试入口
  - 职责：性能测试框架的主容器，协调不同维度的评估任务

- **docs**: 项目文档目录（文档缺失，仅有空目录结构）
  - 功能：未知（未提供文档）
  - 职责：未记录

- **scripts**: 项目级脚本工具
  - 功能：提供项目级别的辅助脚本（当前文档仅包含标题，功能未详细说明）
  - 职责：测试流程的自动化和工具支持

### 二级子目录详解

#### **benchmarks/compatibility** - 兼容性测评套件
- **功能**：验证 CUDA 兼容性和深度学习框架在目标硬件上的正确运行能力
- **测试覆盖**：
  1. **CUDA 兼容性**：通过 cuda-samples 仓库测试 CUDA SDK 示例程序的编译和运行
  2. **Megatron-LM 框架测试**：执行八卡 Llama2-7B 预训练任务，验证大模型框架的分布式训练能力
  3. **NCCL 通信测试**：通过 nccl-tests 验证单机多卡通信性能和正确性
  4. **PaddlePaddle 框架测试**：覆盖多个领域模型（GAN、LSTM、ResNet18、Transformer、VGG11、YOLOv3）
  5. **InfiniTensor 框架测试**：使用 16 个模型覆盖图像分类、NLP、对抗生成、超分辨率四大领域
- **职责**：确保软件栈（CUDA、框架、应用）在目标硬件上的功能正确性，发现兼容性问题

#### **benchmarks/computation** - 计算性能测评
- **功能**：评估加速器的峰值算力和算子性能
- **测试内容**：
  1. **峰值算力测试**：采用 8192×8192 矩阵乘，测试 FP32、FP16 两种精度的理论峰值
  2. **矩阵乘算子性能**：测试非方阵场景 [512, 5120] × [5120, 5120] 和 [512, 5120] × [5120, 13824]
  3. **算子测试框架**：
     - **scripts/**：使用 InfiniCore 测试框架进行算子测试
     - **archive/scripts/**：使用旧版 Infiniop-test 框架（保留用于兼容性）
- **职责**：量化硬件计算能力和算子库性能，为优化提供基准数据

#### **benchmarks/hardware** - 硬件性能测评
- **功能**：测试加速器的底层硬件特性（访存、缓存、通信）
- **硬件平台支持**：
  1. **CUDA 平台**（cuda/）：NVIDIA GPU 硬件特性测试
     - 访存性能：cuda-memcpy（Host↔Device）、cuda-stream（Copy/Scale/Add/Triad 模式）
     - 缓存性能：gpu-cache（L1 Cache）、gpu-l2-cache（L2 Cache，含 SYCL 实现）
     - 通信性能：bandwidthTest（PCIe/NVLink 带宽）
     - 辅助工具：CUDA Helper Library（NVIDIA 官方辅助工具，提供设备初始化、错误检查、计时、图像处理、命令行解析）
     - 性能指标：cuda_metrics（基于 NVPW/CUPTI 的 GPU 硬件计数器采集，支持 DRAM 带宽、L2 命中率、Tensor Core 利用率等指标）
  2. **寒武纪平台**（bang/）：MLU 硬件特性测试
     - 访存性能：cnrt-memcpy（CNRT 库接口测试）
     - CNVS 工具：测试 PCIe 通信带宽、MLULink 通信带宽、矩阵乘性能
- **职责**：揭示硬件微观性能特征，定位性能瓶颈（访存受限、计算受限、通信受限）

#### **benchmarks/llm** - 大模型能力测评
- **功能**：评估大模型训练与推理性能（文档仅包含标题，详细功能未说明）
- **职责**：端到端的大模型工作负载性能评估，验证实际场景下的性能表现

### 三级及更深层级的关键组件

#### **benchmarks/computation/archive/scripts/testcases** - GEMM 测试用例生成器
- **功能**：为 Infiniop 算子库生成 GEMM（通用矩阵乘法）测试用例
- **核心组件**：
  - `gemm.py`：测试用例生成器，包含参考算法、测试数据生成和 GGUF 格式序列化
  - 随机张量生成：小值范围 [-5e-4, 5e-4] 用于数值稳定性测试
  - 数据类型支持：FP16、FP32、FP64、BF16、多种整数类型
  - 步长支持：测试非连续内存布局（转置、广播、子矩阵）
- **测试场景覆盖**：
  - 大规模方阵（8192×8192）：FP32 和 FP16 版本
  - 高方阵（512×5120 × 5120×5120）：模拟 Transformer 注意力矩阵
  - 长矩形矩阵（512×5120 × 5120×13824）：测试非对称维度
- **职责**：提供高精度、可复现的算子测试数据，验证算子库正确性和性能

#### **benchmarks/hardware/cuda/bandwidthTest/include** - CUDA 辅助工具库
- **功能**：NVIDIA 官方提供的 CUDA SDK 辅助工具集合，简化应用程序开发
- **核心模块**：
  - `exception.h`：带位置信息的异常处理模板
  - `helper_cuda.h`：CUDA 设备管理、错误检查、架构查询（支持 Kepler 到 Blackwell 全架构）
  - `helper_functions.h`：通用工具聚合头文件
  - `helper_image.h`：PGM/PPM 图像加载、保存、比较
  - `helper_string.h`：命令行参数解析、文件路径查找
  - `helper_timer.h`：跨平台高精度计时器（Windows QueryPerformanceCounter/Linux gettimeofday）
- **关键能力**：
  - 自动设备选择（`gpuGetMaxGflopsDeviceId`）
  - 架构查询（SM 版本转核心数、架构名称）
  - 图像处理（支持灰度和彩色图像的读写与比较）
  - 文件路径搜索（约 60 个预定义路径的自动查找）
- **职责**：为 CUDA 测试程序提供标准化的基础设施，减少重复代码

#### **benchmarks/hardware/cuda/gpu-l2-cache/sycl** - SYCL GPU L2 缓存测试
- **功能**：基于 SYCL（兼容 NVIDIA CUDA）的 L2 缓存性能探测工具
- **测试原理**：
  - 通过指数增长的数据集大小，测试不同内存跨度下的访问带宽
  - 观察带宽突变点，识别 L2 缓存容量
  - 使用精心设计的索引计算控制内存访问跨度
- **实现特点**：
  - 编译目标：NVPTX64-NVIDIA-CUDA（sm_80 架构，Ampere 及更新）
  - 内存访问模式：防止编译器优化的永假分支、浮点常量、无用赋值
  - 统计采样：11 次重复测试，取最小值排除系统抖动
- **职责**：科学测量 GPU L2 缓存大小和带宽特性，为内核优化提供数据

#### **benchmarks/hardware/cuda/include/cuda_metrics** - CUDA 性能指标测量系统
- **功能**：通过 NVIDIA PerfWorks (NVPW) 和 CUPTI API 实现 GPU 硬件计数器的自动化采集
- **核心组件**：
  - `Eval.hpp`：指标评估引擎，从计数器数据提取语义化指标值
  - `Metric.hpp`：指标配置生成器，创建 Profiler 配置图像
  - `measureMetricPW.hpp`：测量会话管理器，提供 Start/Stop 接口
  - `Parser.h`：指标名称解析器（支持孤立模式、实例保留等修饰符）
  - `pythonInterface.cpp`：Python 绑定层
- **预配置测量接口**：
  - `measureDRAMBytesStart/Stop`：DRAM 读写带宽
  - `measureL2BytesStart/Stop`：L2 缓存读写带宽（32-byte sector 单位）
  - `measureMetricsStart/Stop`：自定义指标列表测量
- **测量流程**：
  1. 初始化 CUPTI Profiler 和 NVPW Host
  2. 查询芯片名称和计数器可用性
  3. 生成配置图像和计数器数据前缀
  4. 启动 KernelReplay 模式的 Profiler 会话
  5. 执行 GPU 操作（自动多次重放采集所有计数器）
  6. 结束会话并评估指标值
- **支持的指标示例**：
  - `dram__bytes_read.sum`：DRAM 读取字节数
  - `lts__t_sectors_srcunit_tex_op_read.sum`：L2 缓存读取 sector 数
  - `sm__pipe_tensor_cycles_active.sum`：Tensor Core 激活周期
  - `dram__throughput.avg.pct_of_peak_dram`：DRAM 带宽利用率
- **职责**：提供细粒度的 GPU 硬件性能计数器采集能力，支持深度性能剖析

## 3. 架构逻辑图解

### 数据流与依赖关系

```
┌─────────────────────────────────────────────────────────────────┐
│                         InfiniPerf 总控层                        │
│              (scripts/ 项目级自动化与流程编排)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  兼容性测评   │    │  计算性能    │    │  硬件性能    │
│compatibility/│    │computation/  │    │ hardware/    │
└──────────────┘    └──────────────┘    └──────────────┘
         │                    │                    │
         ├─ CUDA Samples      ├─ 峰值算力测试      ├─ CUDA 平台
         ├─ Megatron-LM       │   (8192×8192)     │  ├─ 访存性能
         ├─ NCCL Tests        ├─ 算子性能测试      │  │  (memcpy, stream)
         ├─ PaddlePaddle      │   (GEMM)          │  ├─ 缓存性能
         └─ InfiniTensor      │                   │  │  (L1, L2)
                             └─ InfiniCore 框架   │  ├─ 通信带宽
                                 依赖             │  │  (PCIe, NVLink)
                                                  │  └─ 硬件计数器
                                                  │    (NVPW/CUPTI)
                                                  └─ 寒武纪平台
                                                     ├─ CNRT memcpy
                                                     └─ CNVS 工具

┌──────────────┐
│  大模型测评   │
│     llm/     │
└──────────────┘
         │
         ├─ InfiniLM
         └─ InfiniLM-Rust
```

### 测试执行流程（以硬件性能测试为例）

1. **测试准备阶段**：
   - 调用 CUDA Helper Library 的设备管理功能自动选择最高性能 GPU
   - 使用 Helper Timer 创建高精度计时器
   - 调用 cuda_metrics 的 `measureMetricsStart` 初始化 Profiler 会话

2. **测试执行阶段**：
   - 访存测试：cuda-memcpy 执行 Host↔Device 数据传输，记录带宽
   - 缓存测试：gpu-l2-cache/sycl 渐进式增加数据集，观察带宽突变
   - 通信测试：bandwidthTest 测试设备间传输带宽
   - 计数器采集：CUPTI Profiler 在 KernelReplay 模式下自动采集硬件计数器

3. **结果分析阶段**：
   - 调用 `measureMetricsStop` 结束 Profiler 会话
   - 使用 Eval.hpp 的 `GetMetricGpuValue` 评估语义化指标
   - 使用 Helper Image Library 对比参考图像（如适用）
   - 输出带宽、延迟、利用率等性能指标

### 兼容性测试流程

```
┌─────────────────┐
│ 框架选择        │
├─────────────────┤
│• Megatron-LM    │ → 分布式训练测试（8卡 Llama2-7B）
│• PaddlePaddle   │ → 多领域模型验证（GAN/CV/NLP）
│• InfiniTensor   │ → 16模型四领域测试
│• CUDA Samples   │ → SDK 示例编译运行
│• NCCL Tests     │ → 通信正确性验证
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ 环境准备        │
│• build.sh       │ → 编译框架/模型
│• 数据准备       │ → process_data.sh
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ 测试执行        │
│• 预训练/推理    │ → 记录 log 和 checkpoint
│• 缓存管理       │ → remove_cache_data.sh
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ 结果验证        │
│• 检查输出正确性 │
│• 性能指标收集   │
└─────────────────┘
```

### 计算性能测试流程（使用 InfiniCore 框架）

```
┌─────────────────────────────────────────────┐
│   InfiniCore 算子测试框架                   │
│   (benchmarks/computation/scripts/op/)     │
└─────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│ 峰值算力测试    │    │ 算子性能测试    │
│ 8192×8192 GEMM  │    │ 多种形状精度    │
│ FP32/FP16       │    │ FP32/FP16       │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │ 结果汇总与报告        │
         │• 峰值算力 (TFLOPS)    │
         │• 算子性能 (μs)        │
         │• 带宽利用率 (%)       │
         └───────────────────────┘
```

### 关键技术路径

1. **性能计数器采集路径**（cuda_metrics）：
   ```
   用户调用 measureMetricsStart()
     → 查询 GPU 芯片名称（cuptiDeviceGetChipName）
     → 获取计数器可用性（cuptiProfilerGetCounterAvailability）
     → 生成配置图像（GetConfigImage → NVPW_CUDA_RawMetricsConfig）
     → 生成计数器数据前缀（GetCounterDataPrefixImage → NVPW_CounterDataBuilder）
     → 开始 Profiler 会话（cuptiProfilerBeginSession + CUPTI_KernelReplay）
   [用户执行 GPU 操作]
   用户调用 measureMetricsStop()
     → 结束会话（runTestEnd）
     → 评估指标值（GetMetricGpuValue → NVPW_MetricsEvaluator）
     → 返回语义化指标（如 "DRAM 带宽：600 GB/s"）
   ```

2. **测试用例生成路径**（GEMM testcases）：
   ```
   运行 gemm.py
     → 定义 6 个测试用例（不同形状、精度、参数）
     → 对每个用例调用 GemmTestCase.write_test()
       → 计算参考答案（NumPy float64 精度）
       → 序列化到 GGUF 文件（张量数据 + 元数据）
     → 生成 gemm.gguf 文件
   [Infiniop 测试框架读取]
     → 反序列化 GGUF 文件
     → 调用 Infiniop 算子实现
     → 对比结果与参考答案（容差检查）
   ```

3. **L2 缓存探测路径**（gpu-l2-cache/sycl）：
   ```
   主循环控制 blockRun（内存跨度）从 3 指数增长
     → 分配设备缓冲区（malloc_device，大小随 blockRun 增长）
     → 并行执行测试 kernel（200,000 work-groups × 1,024 work-items）
       → 精心设计的索引访问模式（控制内存跨度）
       → 编译器对抗技术（防止优化）
     → 记录执行时间（11 次采样，取最小值）
     → 计算有效带宽（数据量 / 时间）
   观察带宽突变点
     → 识别 L2 缓存容量（如 40 MB 处带宽从 70 GB/s 降至 30 GB/s）
   ```

### 多硬件平台支持策略

InfiniPerf 采用平台子目录隔离的策略支持不同硬件后端：

- **CUDA 平台**（benchmarks/hardware/cuda/）：完整的硬件特性测试套件
- **寒武纪平台**（benchmarks/hardware/bang/）：MLU 特定的测试工具（CNRT、CNVS）
- **未来扩展**：可添加其他平台子目录（如 ascend/、rocm/、kunlun/）

每个平台子目录包含：
- 平台特定的访存测试（调用平台原生的内存拷贝 API）
- 平台特定的缓存测试（利用架构特性）
- 平台特定的通信测试（使用平台专用互联技术）
- 平台特定的性能计数器接口（调用平台的性能分析工具）

### 测试结果流向

```
硬件性能测试
   ├─ 访存带宽（GB/s）
   ├─ 缓存大小与命中率
   └─ 通信带宽
         ↓
计算性能测试
   ├─ 峰值算力（TFLOPS）
   └─ 算子性能（μs，GFLOPS）
         ↓
兼容性测试
   ├─ 框架运行正确性
   └─ 分布式训练稳定性
         ↓
大模型测试
   ├─ 训练吞吐量（tokens/s）
   └─ 推理延迟（ms）
         ↓
性能分析报告
   ├─ 瓶颈识别（计算/访存/通信）
   ├─ 优化建议（内核调优、数据布局）
   └─ 横向对比（不同硬件/框架）
```

### 与 Infini 生态系统的集成

- **依赖 InfiniCore**：computation/scripts/ 使用 InfiniCore 测试框架，复用算子测试基础设施
- **验证 InfiniLM**：llm/ 测试套件验证大模型库的实际性能
- **反馈到 InfiniTrain**：性能瓶颈识别为训练优化提供目标
- **独立于具体实现**：硬件测试与框架解耦，可评估任何基于 CUDA/兼容硬件的实现

### 设计优势

1. **分层清晰**：从硬件到应用的完整测试栈，每层独立可测
2. **平台可扩展**：子目录隔离支持新增硬件平台，不影响现有测试
3. **自动化程度高**：脚本驱动的测试流程，支持 CI/CD 集成
4. **深度性能剖析**：硬件计数器级别的细粒度分析
5. **标准化输出**：统一的性能指标格式，便于对比和分析
