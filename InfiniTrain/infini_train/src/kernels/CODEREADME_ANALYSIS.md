# 架构全景: Kernels 目录

## 1. 子系统职责

该目录是 InfiniTrain 框架的核心计算引擎，负责实现所有深度学习训练所需的前向传播和反向传播算子。作为训练框架的基础设施层，它向上层提供统一的张量计算接口，同时针对不同硬件平台（CPU/CUDA）提供优化的实现策略。

该模块承担以下核心职责：
- **算子实现**：提供完整的深度学习算子库，覆盖线性代数、归一化、激活函数、张量操作等核心计算
- **硬件加速**：通过 CPU（Eigen/OpenMP）和 CUDA（cuBLAS/CUB）两个后端实现跨平台性能优化
- **自动微分**：每个算子同时实现前向和反向传播，支持梯度反向传播链式法则
- **优化器集成**：内置梯度累积和 Adam 优化器更新逻辑
- **分布式支持**：CUDA 后端提供通信原语，支持多 GPU 并行训练

## 2. 模块导航

### 2.1 CPU 后端 (cpu/)

- **功能**: CPU 算子库实现，提供纯 C++ 编写的深度学习计算内核
- **职责**: 为无 GPU 环境或 CPU 推理场景提供高效的算子实现

**核心特性**：
- **数据类型支持**: 主要支持 float32，部分算子利用模板机制支持多种类型
- **并行化策略**: 使用 OpenMP 并行化梯度更新（AdamAccumulateGrad），其余算子主要为串行实现
- **数学库依赖**: 使用 Eigen 库加速矩阵运算（LinearForward、OuterForward）
- **内存优化**: 基于 memcpy 的高效内存拷贝（concat、split、stack），避免逐元素复制
- **数值稳定性**: Softmax 和 CrossEntropy 使用最大值减法避免指数溢出，LayerNorm 使用 epsilon 保护

**关键算子分类**：
1. **梯度优化**: `accumulate_grad.cc`（梯度累积、Adam 更新）
2. **线性代数**: `linear.cc`（矩阵乘法、线性变换）、`outer.cc`（外积）
3. **归一化**: `layernorm.cc`（3D 张量层归一化）、`cross_entropy.cc`（交叉熵损失）
4. **激活函数**: `softmax.cc`、`sigmoid.cc`
5. **逐元素运算**: `elementwise.cc`（一元/二元操作、广播机制）
6. **张量操作**: `concat.cc`、`split.cc`、`stack.cc`、`slice.cc`、`gather.cc`
7. **归约操作**: `reduction.cc`（mean、sum、max、min）
8. **索引与变换**: `embedding.cc`、`transform.cc`、`cast.cc`

**性能特征**：
- 矩阵乘法采用朴素三重循环 O(bs * m * n * k)，Linear 使用 Eigen 优化
- 广播机制支持从低维到高维的单向广播，通过 stride 计算实现索引映射
- LayerNorm 对每个 [bs, seq_len, embed_dim] 位置计算均值和方差，复杂度 O(bs * seq_len * embed_dim)

### 2.2 CUDA 后端 (cuda/)

- **功能**: GPU 算子库实现，提供 CUDA 核函数实现的深度学习计算内核
- **职责**: 为 NVIDIA GPU 提供高性能并行计算实现，是训练性能的关键优化层

**核心特性**：
- **数据类型支持**: 支持 float32 和 bfloat16，使用类型提升策略确保计算精度
- **并行架构**: 采用 256 线程/块的网格配置，大规模张量使用 grid-stride loop
- **性能优化库**: 集成 cuBLAS（矩阵乘法）和 CUB（并行归约）
- **内存管理**: 使用 cudaMallocAsync/cudaFreeAsync 实现流有序内存分配，无需显式同步
- **原子操作**: EmbeddingBackward 使用 atomicAdd 处理重复 token 梯度累积，Elementwise BinaryBackward 对 BF16/half 使用 fastAtomicAdd

**关键算子分类**：
1. **梯度优化**: `accumulate_grad.cu`（梯度累积、Adam 更新，使用 FMA 指令）
2. **线性代数**: `linear.cu`（批量矩阵乘法 cuBLAS GEMM）、`outer.cu`
3. **归一化**: `layernorm.cu`（CUB BlockReduce 并行归约）、`cross_entropy.cu`
4. **激活函数**: `softmax.cu`（2D 网格并行化，每个线程块计算一个 softmax）
5. **逐元素运算**: `elementwise.cu`（支持广播，低精度类型使用三阶段策略优化）
6. **张量操作**: `concat.cu`（二分查找定位源张量）、`split.cu`、`stack.cu`、`slice.cu`
7. **归约操作**: `reduction.cu`（通用归约框架，支持 sum/mean/max/min）
8. **索引与变换**: `embedding.cu`、`gather.cu`、`transform.cu`（转置、三角掩码、掩码、重复插值）
9. **通信原语**: `comm.cu`（broadcast、scatter、gather、reduce_add_coalesced）
10. **分布式专用**: `vocab_parallel_cross_entropy.cu`（词汇表并行交叉熵）

**性能特征**：
- 矩阵乘法使用 cuBLAS StridedBatchedEx API，支持 FP32 和 BF16 计算，FP32 累积
- LayerNorm 和 Softmax 使用 CUB BlockReduce 实现 O(log block_size) 并行归约
- 低精度反向传播（BF16/half）采用自适应策略：
  - 无广播：直接写入（无原子操作）
  - 小广播维度（K <= 4096）：共享内存直方图累积
  - 大广播维度：填充共享内存（SoA 布局）+ 快速原子操作
- CrossEntropy 每个样本分配一个线程块，三次并行归约计算最大值、指数和和损失

**数值稳定性**：
- Softmax/CrossEntropy 使用最大值减法：`exp(x - max) / sum(exp(x - max))`
- LayerNorm 使用 `rsqrt(var + eps)` 提高精度
- Adam 使用偏差修正：`m_hat = m / (1 - beta1^t)`

## 3. 架构逻辑图解

### 3.1 数据流与依赖关系

```
训练输入数据流
     ↓
[前向传播算子链]
     ├─ Linear/Matmul → 线性变换（cuBLAS/Eigen）
     ├─ LayerNorm → 归一化（CUB BlockReduce/串行遍历）
     ├─ Softmax/Sigmoid → 激活函数（数值稳定版本）
     ├─ Embedding → 查表操作（直接内存拷贝/原子操作）
     └─ CrossEntropy → 损失计算（并行归约）
     ↓
损失标量
     ↓
[反向传播算子链]
     ├─ CrossEntropyBackward → Softmax - OneHot
     ├─ LayerNormBackward → 链式法则梯度计算
     ├─ LinearBackward → 两次 GEMM 计算梯度
     └─ 其他算子反向 → 对应梯度计算
     ↓
梯度张量
     ↓
[梯度优化器]
     ├─ AccumulateGrad → 简单梯度累积
     └─ AdamAccumulateGrad → Adam 优化器更新（OpenMP/并行）
     ↓
参数更新
```

### 3.2 硬件后端差异

**CPU 后端实现策略**：
- **矩阵运算**: 优先使用 Linear（Eigen 库调用 BLAS），Matmul 采用朴素实现
- **并行粒度**: Adam 优化器使用 OpenMP 元素级并行，其余算子串行
- **内存访问**: 行优先存储，手动计算 strides 实现索引映射
- **广播机制**: 单向广播（低维 → 高维），通过维度填充和步长计算实现
- **优化方向**: 减少内存拷贝（memcpy 批量传输）、利用 Eigen 表达式模板

**CUDA 后端实现策略**：
- **矩阵运算**: 全面使用 cuBLAS SGEMM/GEMMEx，支持批量矩阵乘法
- **并行粒度**: 线程级并行，256 线程/块，大规模张量使用 grid-stride loop
- **内存访问**: 流有序内存分配，指针数组间接寻址（concat）
- **广播机制**: 运行时 stride 计算，支持动态广播模式
- **优化方向**: CUB 并行归约、warp 级原语、共享内存优化、原子操作融合

### 3.3 算子分类与交互模式

**1. 基础运算层**（最底层，无依赖）
- 逐元素运算（elementwise）、类型转换（cast）、填充（fill）、无操作（no_op）

**2. 张量操作层**（依赖基础运算）
- 形状变换：reshape（no_op）、transpose、slice、split、stack、concat
- 索引操作：gather、embedding
- 掩码操作：mask、tril、triu

**3. 归约与归一化层**（依赖张量操作）
- 归约：sum、mean、max、min
- 归一化：layernorm、softmax
- 损失函数：cross_entropy

**4. 线性代数层**（依赖归约层）
- 矩阵乘法：matmul、linear、outer

**5. 优化与通信层**（最上层，依赖所有层）
- 梯度优化：accumulate_grad、adam
- 分布式通信：broadcast、scatter、gather、reduce_add_coalesced

### 3.4 计算图执行模式

**前向传播执行序列**：
1. **数据准备**: Embedding 查表 → 形状变换（transpose/slice）
2. **特征提取**: Linear 线性变换 → LayerNorm 归一化 → 激活函数（softmax/sigmoid）
3. **损失计算**: CrossEntropy/其他损失函数 → 输出标量损失

**反向传播执行序列**：
1. **损失梯度**: 损失函数反向 → 输入梯度
2. **链式求导**: 激活函数反向 → LayerNorm 反向 → Linear 反向
3. **参数更新**: 梯度累积 → Adam 优化器更新 → 写回参数

**关键优化点**：
- **内存复用**: 前向传播缓存中间结果（mean、rstd、softmax 输出）供反向使用
- **梯度稀疏性**: EmbeddingBackward 使用原子操作仅更新访问过的 token 梯度
- **批量融合**: CUDA 后端使用 cuBLAS StridedBatchedEx 融合批量矩阵乘法
- **通信计算重叠**: comm.cu 提供的通信原语支持流水线并行

### 3.5 数值精度管理

**类型转换策略**：
- CPU 后端: 主要支持 float32，cast 算子提供类型转换
- CUDA 后端: 支持 float32 和 bfloat16，低精度算子使用类型提升（WidestType_t）
- 计算-存储分离: BF16 存储权重，FP32 进行累积（cuBLAS compute precision CUDA_R_32F）

**梯度精度**：
- CPU: 统一使用 float32 梯度
- CUDA: BF16/half 反向传播使用三阶段策略（无广播/直方图/块归约），确保梯度精度

## 4. 技术对比与设计考量

### 4.1 性能权衡

| 维度 | CPU 后端 | CUDA 后端 |
|------|---------|-----------|
| 矩阵乘法 | Eigen BLAS（优化） | cuBLAS（高度优化） |
| 并行归约 | 串行遍历 | CUB BlockReduce（O(log n)） |
| 逐元素运算 | 串行（除 Adam） | 256 线程/块并行 |
| 内存拷贝 | memcpy 批量传输 | 流有序异步拷贝 |
| 低精度支持 | 有限 | 完善（BF16/half 原子操作） |
| 适用场景 | 小规模训练/CPU 推理 | 大规模训练/GPU 加速 |

### 4.2 功能差异

**CUDA 后端独有功能**：
- 通信原语（comm.cu）：支持分布式训练
- 词汇表并行交叉熵（vocab_parallel_cross_entropy.cu）
- 低精度类型优化（BF16/half 原子操作）
- Warp 级原语优化

**CPU 后端限制**：
- LayerNorm 仅支持 3D 张量 [bs, seq_len, embed_dim]
- 广播仅支持单向（低维 → 高维）
- Matmul 使用朴素实现，未优化

### 4.3 设计一致性

**API 接口统一**：
- 两后端使用相同的函数签名（Forward/Backward 模式）
- 通过 Dispatcher 根据设备类型自动路由到对应实现
- 注册宏 REGISTER_KERNEL 统一管理算子表

**数值稳定性保证**：
- Softmax/CrossEntropy 使用 max-subtraction
- LayerNorm 使用 epsilon 保护
- Adam 使用偏差修正

## 5. 扩展性与维护性

**模块化设计**：
- 每个算子独立文件（.cc/.cu），便于单独测试和优化
- 通用模板框架（elementwise、reduction）减少代码重复
- 策略模式（GenericReduceKernel）支持多种归约操作

**类型安全**：
- DispatchFunc 编译时类型特化，避免运行时类型错误
- CHECK 宏进行形状和数据类型验证
- 模板分发机制（DataTypeList）确保类型覆盖

**未来扩展方向**：
- CPU 后端：增加 OpenMP 并行化、支持更多数据类型、优化 Matmul
- CUDA 后端：支持 Tensor Core、Flash Attention、算子融合
- 分布式：扩展通信原语、支持模型并行和流水线并行
