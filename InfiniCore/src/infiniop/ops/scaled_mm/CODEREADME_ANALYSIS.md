# scaled_mm 模块架构全景

## 1. 子系统职责

`scaled_mm` 模块实现了 INT8 量化矩阵乘法操作符（scaled matrix multiplication），这是深度学习推理中的核心计算内核，用于加速 Transformer 模型（如 GPT、BERT）中的量化矩阵运算。该模块支持逐行（per-token）和逐列（per-channel）的动态缩放量化，将 INT8 输入矩阵乘法结果转换为 FP16 或 BF16 输出，并支持可选的偏置加法。

模块采用高度工程化的 CUTLASS 库实现，针对 NVIDIA GPU 的不同架构（Turing、Ampere、Ada Lovelace、Hopper）进行了深度优化，利用 Tensor Core 硬件加速，在保持数值精度的同时最大化计算吞吐量。

## 2. 模块导航

### 硬件后端实现

* **nvidia (NVIDIA GPU 后端)**
    * **功能**: 基于 CUTLASS 库的完整 INT8 量化 GEMM 实现，支持 SM75/80/89 架构的 CUTLASS 2.x 内核和 SM90 架构的 CUTLASS 3.x 内核
    * **职责**: 提供跨多代 NVIDIA GPU 的高性能 INT8 矩阵乘法，通过架构自适应调度策略选择最优的 tile 形状、流水线深度和指令配置，实现逐行逐列缩放和偏置融合的后处理逻辑

### 头文件定义

* **info.h**
    * **功能**: 定义矩阵布局验证工具函数 `I8GemmInfo`
    * **职责**: 在算子创建阶段验证输入矩阵（A、B、缩放因子、偏置）的数据类型、布局和维度是否满足 INT8 GEMM 的要求，提取关键的形状信息（M、N、K）供后续内核调度使用

* **int8_gemm.h**
    * **功能**: 定义 INT8 GEMM 算子的公共 C API 接口和数据结构
    * **职责**: 声明跨硬件后端的统一接口函数（如 `infiniopCreateInt8GemmDescriptor`、`infiniopInt8Gemm`），定义算子描述符的抽象基类或接口规范，为上层应用提供标准化的调用方式

* **operator.cc**
    * **功能**: 算子的 C++ 实现层，连接上层 C API 和底层硬件后端
    * **职责**: 实现 C API 函数的具体逻辑，根据设备类型（通过 `infiniopHandle_t` 识别）分发到相应的硬件后端实现（如 nvidia::Descriptor），处理参数验证、错误码转换和资源管理

## 3. 架构逻辑图解

### 数据流与组件交互

scaled_mm 模块的架构呈现清晰的分层设计，从上层应用 API 到底层硬件内核的完整数据流如下：

```
上层应用
    |
    | 1. 创建张量描述符 (out_desc, a_desc, b_desc, a_scale_desc, b_scale_desc, bias_desc)
    | 2. 调用 infinioopCreateInt8GemmDescriptor
    v
operator.cc (C++ 接口层)
    |
    | 3. 验证输出数据类型 (必须是 F16 或 BF16)
    | 4. 调用 I8GemmInfo::create() 验证矩阵布局
    | 5. 根据设备类型分发到硬件后端
    v
nvidia::Descriptor (NVIDIA GPU 算子描述符)
    |
    | 6. 初始化 GPU 句柄和资源
    | 7. 等待计算调用
    v
nvidia::Descriptor::calculate() (计算入口)
    |
    | 8. 运行时架构检测 (getSMVersion)
    | 9. 根据架构选择最优内核配置
    |    - SM75 (Turing): sm75_dispatch_shape
    |    - SM80 (Ampere): sm80_dispatch_shape
    |    - SM86/89 (Ada): sm89_dispatch_shape
    |    - SM90 (Hopper): sm90_dispatch_shape
    v
cutlass_int8_scaled_mm / cutlass_int8_scaled_mm_sm90 (内核实例化)
    |
    | 10. 构造 CUTLASS GEMM 内核
    |     - CUTLASS 2.x: DefaultGemmConfiguration + EpilogueVisitorPerRowPerCol
    |     - CUTLASS 3.x: CollectiveBuilder + Epilogue Fusion EVT
    | 11. 准备内核参数 (数据指针、跨步、缩放因子、偏置)
    v
CUTLASS GEMM 内核执行 (设备端)
    |
    | 12. CUDA Grid 启动
    |     - Threadblock 并行: M-N 平面 2D 划分
    |     - Warp 并行: Threadblock 内子块处理
    |     - Tensor Core: 16x8x32 或 8x8x16 微内核计算
    | 13. MMA 主循环: INT8 矩阵乘法累加 (INT32 accumulator)
    v
Epilogue 阶段 (后处理)
    |
    | 14. EpilogueVisitorPerRowPerCol 或 EVT 执行
    |     - 加载逐行缩放因子 (per-token scale from a_scale)
    |     - 加载逐列缩放因子 (per-channel scale from b_scale)
    |     - 加载偏置向量 (bias, 可选)
    |     - 对每个累加器元素应用: result = float(accum) * scale_row * scale_col + bias
    |     - 转换为 FP16/BF16 并写入输出矩阵
    v
输出结果 (GPU 全局内存)
```

### 关键设计决策

1. **分层架构设计**
   - **接口抽象层** (`int8_gemm.h`, `operator.cc`): 提供跨硬件后端的统一 C API，隔离不同硬件实现细节
   - **硬件后端层** (`nvidia/`): 针对特定 GPU 平台的优化实现，利用硬件特性最大化性能
   - **内核抽象层** (CUTLASS): 通过 CUTLASS 库的模板元编程，自动生成针对不同架构的最优内核

2. **架构自适应调度策略**
   - 模块不依赖单一配置，而是根据运行时检测到的 GPU 计算能力（SM 版本）动态选择最优内核实现
   - 每代架构的调度函数（`sm75_dispatch_shape`, `sm80_dispatch_shape`, `sm89_dispatch_shape`, `sm90_dispatch_shape`）内置了针对该架构硬件特性的预优化配置表
   - 配置表根据问题规模（M、N、K）选择最优的 ThreadblockShape、WarpShape、InstructionShape 和流水线阶段数

3. **量化感知的矩阵乘法**
   - 标准矩阵乘法: `D = A × B` (FP32/FP16 计算)
   - 量化矩阵乘法: `D = scale(A) × scale(B)`，其中 scale 是逐元素或逐行/列的动态缩放因子
   - 该模块实现的关键创新在于 Epilogue 阶段的融合后处理：在 Tensor Core 完成 INT8×INT8→INT32 的累加后，立即应用缩放因子和偏置，避免额外的内核启动和内存读写

4. **CUTLASS 2.x 与 3.x 的双路径支持**
   - **CUTLASS 2.x 路径** (SM75/80/89): 使用传统的 `DefaultGemmConfiguration` + 自定义 `EpilogueVisitor`，通过访问者模式注入逐行逐列缩放逻辑
   - **CUTLASS 3.x 路径** (SM90): 使用新的 `CollectiveBuilder` 和 Epilogue Visitor Tree (EVT)，通过表达式树融合实现更高效的后处理，支持 TMA (Tensor Memory Accelerator) 硬件加速
   - 两条路径在功能上完全等价，但在 SM90 架构上 CUTLASS 3.x 能获得更高的性能和更低的延迟

5. **内存对齐与布局优化**
   - 强制要求 A 矩阵的 K 维度和 B 矩阵的 K 维度 128-bit 对齐（INT8 为 16 元素），以确保 Tensor Core 加载指令的高效执行
   - B 矩阵采用 Column Major 存储（逻辑上相当于转置的 Row Major），优化内存访问模式和缓存利用率
   - 缩放因子作为 1D 向量存储（per-token: M 元素，per-channel: N 元素），在 Epilogue 阶段通过迭代器按需加载，减少全局内存带宽压力

### 性能优化要点

- **Tensor Core 利用率**: 通过精心选择 InstructionShape (如 SM80 的 16×8×32)，最大化每个 Tensor Core 指令的 INT8 计算吞吐量（理论上是 FP32 的 4 倍）
- **流水线深度优化**: 根据架构的 shared memory 容量（Turing 限制、Ampere 160KB、Ada 100KB）选择最优的流水线阶段数，隐藏全局内存延迟
- **Tile 形状自适应**: 小矩阵（M≤32）使用小 tile 减少线程空闲，大矩阵（M>128）使用大 tile 提高占用率
- **后处理融合**: 将缩放、偏置、类型转换全部融合在 Epilogue 阶段，避免额外的内核启动和中间结果存储
- **TMA 加速** (SM90): 利用 Hopper 架构的 Tensor Memory Accelerator 硬件单元，自动管理数据传输，进一步降低延迟

### 与 InfiniOP 生态的集成

scaled_mm 模块作为 InfiniOP 算子库的核心组件，遵循统一的设计规范：
- 通过 `infiniopHandle_t` 访问 GPU 设备资源和内存池
- 通过 `infiniopTensorDescriptor_t` 描述张量的数据类型、形状和布局
- 通过标准的 InfiniStatus 错误码与上层通信
- 支持批处理（batched GEMM）和流式并发执行（通过 CUDA stream）

该模块的高性能实现为上层框架（如 InfiniLM 的推理引擎）提供了关键的算力支撑，使得大语言模型可以在保持精度的同时通过 INT8 量化显著降低显存占用和推理延迟。
