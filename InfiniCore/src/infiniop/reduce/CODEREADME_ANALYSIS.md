# reduce 目录架构全景

## 1. 子系统职责

`reduce` 目录是 InfiniOp 运算库中的 **归约操作 (Reduction Operations) 核心实现层**。该子系统负责为深度学习计算中的归约类操作提供高性能、跨硬件平台的底层算子实现，包括求和 (sum)、求最大值 (max) 和平方和 (sumSquared) 等基础归约运算。

该目录位于 `infiniop` 基础操作层之下，作为原子算子模块被上层的高维张量操作（如矩阵乘法、卷积、注意力机制等）调用。其设计核心在于 **硬件异构抽象统一**：通过命名空间隔离不同硬件平台的实现细节，对外暴露一致的函数接口，使上层算法无需关心底层硬件差异。

**关键设计特征**：
- **跨平台支持**：覆盖 CPU、CUDA、昆仑 (Kunlun)、寒武纪 (Bang) 四类主流硬件后端
- **模板化泛型**：利用 C++ 模板实现数据类型泛型，支持 float、double、int 系列及 fp16/bf16 半精度浮点
- **性能优化分级**：CPU 后端支持 OpenMP 并行，加速器后端利用块级归约 (block reduction) 和专用指令集
- **内存布局灵活性**：支持 stride 参数，可处理连续或非连续内存布局的数据

## 2. 模块导航 (Module Navigation)

### **cpu (CPU 通用实现)**
- **功能**：基于标量循环和 OpenMP 并行化的 CPU 归约算子实现
- **职责**：提供跨 x86/ARM 等 CPU 架构的通用归约操作基线实现，支持全精度浮点和整数类型的原生归约运算，以及 fp16/bf16 半精度类型转换为 float32 后的归约计算

**核心实现**：
- `reduce.h`：模板函数定义，利用 `std::disjunction` 进行类型萃取，为 float/double/int 系列类型提供原生的 `sum()`、`max()`、`sumSquared()` 函数；半精度类型通过函数声明转至实现文件
- `reduce.cc`：半精度类型特化实现，通过 `sum_half_impl()`、`max_half_impl()`、`sumSquared_half_impl()` 模板函数统一处理 fp16/bf16，先调用 `utils::cast<float>` 转换为 float32 再执行归约

**性能特征**：
- 支持 OpenMP 并行（通过 `#ifdef ENABLE_OMP` 条件编译）
- 使用 stride 参数支持非连续内存布局
- 半精度计算通过类型转换保证精度

### **cuda (CUDA GPU 实现)**
- **功能**：基于 NVIDIA CUDA 和 CUB 库的 GPU 并行归约算子
- **职责**：利用 CUDA 线程束并行性和 CUB 块级归约原语，在 GPU 上实现高性能归约计算，确保仅 thread 0 的结果正确，手动广播至其他线程

**核心实现**：
- `reduce.cuh`：设备端函数定义，利用 `cub::BlockReduce` 实现线程块内归约
  - `sum()`：线程并行累加后通过 `BlockReduce::Sum()` 聚合
  - `sumSquared()`：先计算平方再累加，使用相同的块级归约策略
  - `max()`：采用条件编译适配 CUDA 版本差异（>= 12.9 使用 `::cuda::maximum()`，旧版本使用 `cub::Max()` 或自定义 lambda），通过 `BlockReduce::Reduce()` 聚合

**性能特征**：
- 模板参数 `BLOCK_SIZE` 允许调用者指定线程块大小
- 分离数据类型 `Tdata` 和计算类型 `Tcompute`，支持计算精度提升（如 fp16 数据用 float32 计算）
- 使用 `__shared__` 内存存储临时归约存储，避免全局内存访问
- 兼容 HYGON API（通过 `#ifdef ENABLE_HYGON_API` 条件编译）

### **kunlun (昆仑 AI 加速器实现)**
- **功能**：适配昆仑 AI 芯片的归约算子实现
- **职责**：利用昆仑硬件的原生集群同步和原子操作原语，实现分布式计算单元间的归约聚合

**核心实现**：
- `reduce_kunlun.h`：设备端函数，使用昆仑特定的内存标记和同步原语
  - `sum()` / `sumSquared()`：通过 `core_id()` 获取计算单元 ID 进行循环分片，使用 `__shared__` 内存作为共享累加器，通过 `atomicAdd()` 原子累加，`sync_cluster()` 保证集群同步
  - `max()`：采用 `atomicMax()` 原子操作求最大值，`fmax()` 函数进行浮点比较

**性能特征**：
- 使用 `__shared_ptr__` 标记共享内存指针
- 通过 `core_id()` 替代 CUDA 的 `threadIdx.x` 实现计算单元索引
- 原子操作 + 集群同步确保数据一致性
- 初始化逻辑（`if (core_id() == 0) temp_storage = ...`）避免脏数据

### **bang (寒武纪 MLU 实现)**
- **功能**：适配寒武纪 MLU (Machine Learning Unit) 的归约算子实现
- **职责**：利用寒武纪专用向量指令（`__bang_*` 系列函数）和 NRAM（Near RAM）片上内存，实现高吞吐归约计算

**核心实现**：
- `reduce_bang.h`：MLU 函数端实现，包含显式的 NRAM 管理和数据搬运逻辑
  - `sum()` / `sumBatched()`：分批从 GDRAM 搬运数据至 NRAM，对半精度类型先转换为 float32，使用 `__bang_sumpool()` 和 `__bang_reduce_sum()` 向量化归约
  - `sumSquared()` / `sumSquaredBatched()`：先通过 `__bang_mul()` 计算平方，再调用 `sumInternal()` 归约，或逐元素计算累加
  - `max()` / `maxBatched()`：使用 `__bang_maxpool()` 进行最大池化归约，`__bang_argmax()` 获取最大值索引

**性能特征**：
- 显式内存管理：使用 `__memcpy(GDRAM2NRAM)` 在全局内存和片上内存间搬运数据
- 批处理策略：通过 `max_batch` 参数控制单批处理元素数，`batch_size = 128 / sizeof(float)` 确保内存对齐
- 向量化优化：宽度 >= 4 时使用 `__bang_sumpool()` / `__bang_maxpool()` 向量化指令，小批量回退到标量循环
- 对齐处理：`aligned_batch` 处理对齐部分，`remainder` 逐元素处理尾部未对齐数据
- 半精度优化：`__bang_half2float()` / `__bang_bfloat162float()` 硬件加速转换

## 3. 架构逻辑图解

### 数据流向与调用链

```
上层张量操作 (MatMul/Conv/Attention)
         |
         v
infiniop 高维操作层
         |
         +-- 根据硬件后端分发 ----------------+
         |                                  |
         v                                  v
[CPU 路径]                        [加速器路径]
   |                                  |
   v                                  v
cpu::reduce_op              cuda::reduce_op  kunlun::reduce_op  bang::reduce_op
(OpenMP + 标量循环)          (CUB 块归约)      (原子操作 + 集群同步)  (向量指令 + NRAM)
```

### 硬件后端选择逻辑

系统在编译时或运行时根据以下规则选择后端实现：

1. **CPU 后端**：作为通用回退方案，适用于所有支持 C++ 的平台，无硬件依赖
2. **CUDA 后端**：检测到 NVIDIA GPU 且 CUDA 版本匹配时启用，依赖 `cub` 库
3. **昆仑后端**：检测到昆仑硬件且启用 `-DENABLE_KUNLUN` 时使用
4. **寒武纪后端**：检测到寒武纪 MLU 且启用 `-DENABLE_BANG` 时使用

### 归约计算模式对比

| 后端    | 并行模型      | 同步机制       | 内存层级         | 半精度处理          | 适用场景               |
|---------|-------------|--------------|----------------|-------------------|---------------------|
| CPU     | OpenMP 线程  | 隐式同步       | DRAM           | 转换为 float32      | 小规模数据、CPU 推理    |
| CUDA    | 线程块级      | __syncthreads | Shared + Global | 支持 fp16/bf16 计算 | 大规模并行、GPU 训练    |
| Kunlun  | 计算集群      | sync_cluster  | Shared Memory  | 转换为 float32      | 昆仑云端推理          |
| Bang    | 标量并行      | 隐式同步       | NRAM + GDRAM   | 硬件加速转换         | 寒武纪边缘设备        |

### 关键性能优化策略

1. **CUDA 平台**：
   - 块级归约利用 Shared Memory 避免全局内存竞争
   - 使用 CUB 库的最优归约算法（Warp Shuffle + 共享内存）
   - 分离数据类型和计算类型，在 fp16 数据上使用 float32 累加避免溢出

2. **Kunlun 平台**：
   - 原子操作替代显式锁，减少同步开销
   - 集群同步 (`sync_cluster`) 确保所有计算单元在原子操作前完成写入
   - 初始化共享内存避免未定义行为

3. **Bang 平台**：
   - 数据预取至 NRAM 片上内存，减少 GDRAM 访问延迟
   - 向量化指令（`__bang_sumpool` / `__bang_maxpool`）利用 SIMD 并行
   - 批处理 + 内存对齐最大化内存带宽利用率
   - 对齐数据和尾部未对齐数据分别处理，平衡性能与通用性

4. **CPU 平台**：
   - OpenMP 并行循环利用多核 CPU
   - 编译器模板特化生成类型优化代码
   - Stride 支持处理 C-order / Fortran-order 张量布局

### 类型转换与数值精度

- **半精度浮点**：所有后端均将 fp16/bf16 转换为 float32 后计算，确保累加精度
- **整数类型**：CPU 后端直接计算（原生支持），加速器后端通过模板参数控制计算类型
- **溢出保护**：CUDA 分离 `Tdata` 和 `Tcompute` 允许在 fp16 数据上使用 float32 累加

### 扩展性设计

该模块采用 **命名空间隔离 + 模板泛型** 的设计模式，新增硬件后端仅需：

1. 创建新目录（如 `ascend` / `rocm`）
2. 实现同名命名空间（如 `op::common_ascend::reduce_op`）
3. 提供相同签名的模板函数（`sum` / `max` / `sumSquared`）
4. 适配该硬件的并行原语和内存层级

上层代码通过条件编译或工厂模式自动选择对应后端，无需修改调用逻辑。
