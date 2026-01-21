# 目录: dequantize_awq 架构全景

## 1. 子系统职责

本模块实现了 AWQ (Activation-aware Weight Quantization) 算法的权重解量化算子，支持多种硬件后端。AWQ 是一种高效的 4-bit 权重量化方案，专门为大型语言模型（LLM）推理设计。该算子的核心功能是将压缩存储的 4-bit 量化权重（packed 格式）解量化为 FP16 半精度浮点格式，以便在 GPU 上执行高性能矩阵运算。

在 InfiniOp 算子库中，dequantize_awq 属于量化相关的预处理算子，位于 `infiniop/ops/` 子系统下，是 LLM 推理 pipeline 中关键的权重复原步骤。该模块采用硬件后端隔离的架构设计，每个硬件厂商（NVIDIA、Moore Threads、Iluvatar）都有独立的实现目录。

## 2. 模块导航

* **iluvatar**:
  * *功能*: *文档缺失* - 该目录包含天数智芯（Iluvatar）GPU 的实现代码（.cu/.cuh 文件），但尚未生成 CODEREADME.md 文档。
  * *职责*: 提供天数智芯 GPU（如 Enflame 系列加速卡）的 AWQ 权重量化解量化内核实现，使用天数智芯的 CUDA 兼容编程模型。

* **moore**:
  * *功能*: *文档缺失* - 该目录包含摩尔线程（Moore Threads）GPU 的实现代码（.mu 内核文件），但尚未生成 CODEREADME.md 文档。
  * *职责*: 提供摩尔线程国产 GPU（如 S3000、S4000 系列加速卡）的 AWQ 权重量化解量化内核实现，使用摩尔线程的 MUSA 编程模型。

* **nvidia**:
  * *功能*: NVIDIA GPU 后端的 AWQ 权重量化解量化实现，提供高度优化的 CUDA 内核。该实现包含设备端解量化函数、PTX 内联汇编优化以及针对不同 GPU 架构（Volta/Turing/Ampere/Hopper）的双路优化代码。
  * *职责*: 实现将 int4 packed 格式的量化权重解量化为 FP16 格式的 CUDA 内核，核心功能包括：4-bit 到 FP16x2 的高效转换、零点减法、缩放因子乘法，支持分组量化（group_size 可配置，通常为 128）。该实现利用 PTX 汇编指令（lop3.b32、fma.rn.f16x2、sub.f16x2）实现接近理论峰值的内存带宽性能。

## 3. 架构逻辑图解

本模块采用典型的**多后端策略模式**架构，通过目录隔离实现不同硬件厂商的算子实现。该设计使得上层 InfiniOp 框架可以透明地调用不同硬件后端的解量化算子，而无需关心底层实现细节。

**模块间数据流与依赖关系**：

1. **统一接口层**（父目录）：`dequantize_awq.h` 定义了所有后端共享的算子接口宏 `DESCRIPTOR(NAMESPACE)`，该宏为每个命名空间生成一致的 Descriptor 类接口，确保不同后端对外提供的 API 完全一致（`create()` 和 `calculate()` 方法）。

2. **元数据验证层**（父目录）：`info.h` 定义了 `DequantizeAWQInfo` 类，负责在算子创建时验证张量形状兼容性（维度数、数据类型、分组大小匹配），并提取元数据供后端实现使用。所有后端的 `Descriptor::create()` 方法都会调用此类进行形状验证。

3. **硬件后端实现层**（子目录）：
   - **nvidia** 目录：包含完整的 CUDA 实现代码，采用头文件分离设计（.cuh 声明、.cu 实现）。设备端内核函数 `dequantize_s4_to_fp16x2()` 负责核心的 int4 到 FP16 转换，主机端 `Descriptor` 类管理内核启动和 CUDA 资源。该实现针对不同 GPU 架构提供双路优化（计算能力 < 7.5 使用标准 CUDA API，>= 7.5 使用 PTX 内联汇编），最大化性能。
   - **iluvatar** 目录：包含天数智芯 GPU 的 .cu/.cuh 实现文件，应与 NVIDIA 实现采用相似的接口设计（通过 `DESCRIPTOR(iluvatar)` 宏生成），但使用天数智芯的设备驱动和编译工具链。
   - **moore** 目录：包含摩尔线程 GPU 的 .mu（MUSA 汇编/源码）实现文件，采用摩尔线程的专有编程模型，但对外接口保持一致。

4. **算子注册与分发**（父目录 `operator.cc`）：上层框架通过编译时条件宏（`ENABLE_NVIDIA_API`、`ENABLE_ILUVATAR_API`、`ENABLE_MOORE_API`）选择编译哪些后端实现。在运行时，InfiniOp Handle 根据设备类型动态分发到对应的后端实现。

**执行流程**：
当用户调用 AWQ 解量化算子时，数据流如下：用户代码 → `infiniopCreateTensorDescriptor()` 创建张量描述符 → `op::dequantize_awq::<backend>::Descriptor::create()` 验证形状并创建算子描述符 → `Descriptor::calculate()` 启动硬件内核 → GPU 执行解量化（读取 packed int4 权重、缩放因子、zero point，执行 `int4 - zero * scale = FP16`，输出 FP16 权重矩阵）→ 返回给用户用于后续矩阵乘法。

**架构优势**：
- **硬件隔离**：各厂商实现互不干扰，可以独立优化和维护
- **接口统一**：所有后端对外 API 一致，上层代码无需修改即可支持新硬件
- **性能优化**：每个后端可以利用硬件专有特性（如 NVIDIA 的 PTX 汇编）实现极致性能
- **扩展性强**：新增硬件后端只需添加新目录并实现相同接口，无需修改现有代码
