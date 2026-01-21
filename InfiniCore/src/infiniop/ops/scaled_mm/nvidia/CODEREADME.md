# `scaled_mm (nvidia)` INT8 量化矩阵乘法核心实现文档

本模块实现了基于 NVIDIA GPU 的 INT8 量化矩阵乘法操作符，支持逐行和逐列缩放（per-token 和 per-channel quantization），输出为 FP16 或 BF16 格式。该实现基于 CUTLASS 库，针对不同 GPU 架构（Turing、Ampere、Hopper）进行了深度优化。

## 1. 模块结构

- **`epilogue_per_row_per_col_scale.h`**: 自定义 Epilogue Visitor，实现逐行逐列缩放的后处理逻辑
- **`gemm_universal_base_compat.h`**: CUTLASS 2.x 兼容层设备端 GEMM 基类，提供网格形状计算、工作空间管理和内核启动功能
- **`gemm_with_epilogue_visitor.h`**: 带 Epilogue Visitor 的 GEMM 内核实现，支持自定义后处理操作
- **`int8_gemm_kernel.cuh`**: INT8 量化 GEMM 的核心内核实现，包含不同架构的调度策略和内核实例化
- **`int8_gemm_nvidia.cu`**: NVIDIA GPU 后端的算子描述符实现，负责架构检测和内核分发
- **`int8_gemm_nvidia.cuh`**: NVIDIA 后端公共接口定义

## 2. 核心类

### `EpilogueVisitorPerRowPerCol`
- **位置**: `epilogue_per_row_per_col_scale.h`
- **主要功能**: 实现支持逐行和逐列缩放的 Epilogue Visitor，在 GEMM 累加后对结果应用缩放因子和偏置

#### 关键成员变量
- `params_`: Epilogue 参数引用，包含逐元素函数参数和批跨步
- `elementwise_`: 逐元素仿函数，用于激活或其他后处理操作
- `with_bias_`: 布尔标志，指示是否添加偏置
- `per_token_quant_`: 布尔标志，启用逐 token（行）量化
- `per_channel_quant_`: 布尔标志，启用逐 channel（列）量化
- `ptr_alpha_row_`: 行缩放因子指针（per-token scale）
- `ptr_alpha_col_`: 列缩放因子指针（per-channel scale）
- `iterator_alpha_col_`: 列缩放因子的 Tile 迭代器
- `iterator_C_`: 偏置矩阵的 Tile 迭代器
- `iterator_D_`: 输出矩阵的 Tile 迭代器
- `element_alpha_row_`: 当前行的缩放标量（非逐行量化时为常数）
- `element_alpha_col_`: 当前列的缩放标量（非逐列量化时为常数）

#### 核心方法
- `EpilogueVisitorPerRowPerCol(...)`: 构造函数，初始化所有迭代器和标志位。如果是全局缩放而非逐元素缩放，会在构造时直接加载缩放标量

- `set_batch_index(int batch_idx)`: 设置批处理索引，更新迭代器的指针偏移量（按批跨步）

- `begin_epilogue()`: Epilogue 开始时调用，加载列缩放因子（如果启用逐列量化）和偏置数据

- `begin_row(int row_idx)`: 处理新行时调用，如果是逐行量化模式，使用全局内存加载当前行的缩放因子 `alpha_row`

- `visit(int iter_idx, int row_idx, int column_idx, int frag_idx, AccumulatorFragment const &accum)`:
  - **功能**: 处理每个累加器片段，应用缩放和偏置
  - **算法**:
    1. 将累加器类型从 `ElementAccumulator` (int32_t) 转换为 `ElementCompute` (float)
    2. 如果启用逐列量化：`result = accum * (scale_col[i] * scale_row)`
    3. 否则：`result = accum * (scale_col * scale_row)`
    4. 如果有偏置：`result = result + bias`
    5. 转换为输出类型 (FP16/BF16)
  - **复杂度**: O(kElementsPerAccess)，其中 kElementsPerAccess 通常为 1-8 个元素

- `end_step(int step_idx)`: 将处理后的片段写入输出矩阵，并递增迭代器

#### 生命周期
作为 `EpilogueWithVisitor` 的一部分，在每个 threadblock 的 epilogue 阶段被实例化和使用。由 `GemmWithEpilogueVisitor` 内核的 epilogue 阶段调用。

### `GemmUniversalBaseCompat<GemmKernel_>`
- **位置**: `gemm_universal_base_compat.h`
- **主要功能**: CUTLASS 2.x 风格的设备端 GEMM 封装类，提供参数初始化、工作空间计算、内核启动等功能

#### 关键成员变量
- `params_`: 内核参数对象，包含所有 GEMM 操作的数据指针和布局信息

#### 核心方法
- `can_implement(Arguments const &args)`: 静态方法，检查给定问题是否可以执行。验证网格维度是否超过 uint16_t 限制

- `get_workspace_size(Arguments const &args)`: 静态方法，计算所需工作空间大小
  - Split-K 并行模式：需要临时存储多个 partial results，空间为 `sizeof(ElementC) * batch_stride_D * grid_tiled_shape.k()`
  - Serial Split-K：如果 K 维度被分割，需要同步空间，大小为 `sizeof(int) * grid_tiled_shape.m() * grid_tiled_shape.n()`
  - **复杂度**: O(1)

- `get_grid_shape(Arguments const &args)`: 静态方法，计算 CUDA 网格维度
  - 使用 `ThreadblockSwizzle` 将问题尺寸映射到 3D 网格
  - 返回 `dim3` 结构 (grid.x, grid.y, grid.z)
  - **复杂度**: O(1)

- `maximum_active_blocks(int smem_capacity)`: 静态方法，查询每个 SM 最大可激活的 block 数量
  - 如果 shared memory <= 48KB：使用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 直接查询
  - 否则：先查询零 shared memory 时的占用率，再根据 SM 容量计算受限值：`min(max_active_blocks, smem_capacity / smem_size)`
  - **复杂度**: O(1) CUDA API 调用

- `initialize(Arguments const &args, void *workspace, cudaStream_t stream)`: 初始化 GEMM 状态
  - 计算并分配工作空间（如果需要）
  - 对于 serial split-K 模式，使用 `cudaMemsetAsync` 清零工作空间
  - 构造 `Params` 结构
  - 如果 shared memory >= 48KB，使用 `cudaFuncSetAttribute` 设置动态共享内存上限
  - **复杂度**: O(workspace_size) 用于清零操作

- `run(cudaStream_t stream)`: 启动内核执行
  - 使用 `ThreadblockSwizzle` 计算网格形状
  - 启动 CUTLASS GEMM 内核：`cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params_)`
  - 检查 CUDA 错误并返回状态
  - **复杂度**: O(1) 启动开销，实际执行时间取决于问题规模

#### 生命周期
作为算子描述符的成员，在算子创建时构造，每次调用 `calculate()` 时重新初始化参数并运行。

### `GemmWithEpilogueVisitor<Mma_, Epilogue_, ThreadblockSwizzle_>`
- **位置**: `gemm_with_epilogue_visitor.h`
- **主要功能**: GEMM 内核实现，集成主循环 MMA 计算和自定义 Epilogue Visitor

#### 关键类型定义
- `Mma`: Threadblock 级别的矩阵乘累加单元
- `Epilogue`: Epilogue 模板，包含 `Visitor` 类型
- `EpilogueVisitor`: 实际的后处理 visitor（如 `EpilogueVisitorPerRowPerCol`）
- `ElementA`, `ElementB`: 输入矩阵元素类型（int8_t）
- `ElementC`: 输出/偏置矩阵元素类型（float16/bfloat16）
- `ThreadblockShape`: Threadblock 级别的 tile 形状（如 128x128x64）
- `WarpShape`: Warp 级别的 tile 形状（如 64x64x64）
- `InstructionShape`: Tensor Core 指令形状（如 16x8x32）

#### 核心方法
- `can_implement(...)`: 静态方法，验证矩阵对齐是否符合要求
  - 检查 A、B、C 矩阵的维度是否满足对齐要求（kAlignmentA, kAlignmentB, kAlignmentC）
  - 支持多种布局：RowMajor, ColumnMajor, Interleaved
  - **复杂度**: O(1)

- `run_kernel_(Params const &params, SharedStorage &shared_storage)`: 设备端内核主函数
  - **步骤 1 - Threadblock 定位**：
    - 使用 `ThreadblockSwizzle` 计算当前 threadblock 在全局网格中的偏移
    - 提前退出如果超出有效范围
  - **步骤 2 - Split-K 处理**：
    - 根据模式（Gemm, Batched, Array）调整 K 维度偏移和指针
    - 计算 `offset_k` 和实际 `problem_size_k`
  - **步骤 3 - 迭代器构造**：
    - 创建 A 矩阵迭代器：`IteratorA(params_A, ptr_A, {m, problem_size_k}, thread_idx, tb_offset_A)`
    - 创建 B 矩阵迭代器：`IteratorB(params_B, ptr_B, {problem_size_k, n}, thread_idx, tb_offset_B)`
  - **步骤 4 - 主循环 MMA**：
    - 计算迭代次数：`gemm_k_iterations = (problem_size_k - offset_k + Shape::kK - 1) / Shape::kK`
    - 执行累加：`mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators)`
    - **复杂度**: O(gemm_k_iterations) 次 Tensor Core 操作
  - **步骤 5 - Epilogue Visitor 执行**：
    - 检查是否有偏置：`with_bias = (params.ptr_C != nullptr)`
    - 构造 visitor：`EpilogueVisitor(..., with_bias, true/*per_token*/, true/*per_channel*/, ...)`
    - 设置 K 分割或批索引：`set_k_partition()` 或 `set_batch_index()`
    - 执行 epilogue：`epilogue(epilogue_visitor, accumulators)`
  - **总复杂度**: O(M * N * K / num_threads) 实际计算复杂度

- `operator()(Params const &params, SharedStorage &shared_storage)`: 设备端入口函数，根据架构标签分发到 `run_kernel_`

#### 生命周期
作为设备端内核函数，由 CUDA grid 中的每个 threadblock 实例执行一次。函数栈生命周期仅限于内核执行期间。

### `cutlass_int8_scaled_mm<ElementOutput, ArchTag, ThreadblockShape, WarpShape, InstructionShape, NumStages>`
- **位置**: `int8_gemm_kernel.cuh`
- **主要功能**: INT8 量化 GEMM 的模板函数实例化，使用 CUTLASS 2.x 的 DefaultGemm 配置

#### 关键模板参数
- `ElementOutput`: 输出类型（cutlass::half_t 或 cutlass::bfloat16_t）
- `ArchTag`: 架构标签（cutlass::arch::Sm75, Sm80, Sm90）
- `ThreadblockShape`: Threadblock tile 形状（如 GemmShape<128, 128, 64>）
- `WarpShape`: Warp tile 形状（如 GemmShape<64, 64, 64>）
- `InstructionShape`: Tensor Core 指令形状（如 GemmShape<16, 8, 32> 或 <8, 8, 16>）
- `NumStages`: 流水线阶段数（2-6，取决于架构和 shared memory 容量）

#### 核心实现
```cpp
template <typename ElementOutput, typename ArchTag, typename ThreadblockShape, ...>
void cutlass_int8_scaled_mm(...) {
    // 类型定义
    using ElementAccumulator = int32_t;
    using ElementCompute = float;
    using ElementInputA = int8_t;
    using ElementInputB = int8_t;
    using OperatorClass = cutlass::arch::OpClassTensorOp;

    // 1. 构造默认 GEMM 配置
    using DefaultGemmConf = cutlass::gemm::device::DefaultGemmConfiguration<...>;

    // 2. 生成基础 GEMM 内核
    using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemm<...>::GemmKernel;

    // 3. 构造列缩放因子迭代器
    using AlphaColTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<...>;

    // 4. 构造自定义 Epilogue Visitor
    using EpilogueVisitor = cutlass::epilogue::threadblock::EpilogueVisitorPerRowPerCol<
        ThreadblockShape, GemmKernel_::kThreadCount, AlphaColTileIterator,
        typename GemmKernel_::Epilogue::OutputTileIterator,
        ElementAccumulator, ElementCompute, EpilogueOutputOp>;

    // 5. 从现有 epilogue 构造带 visitor 的 epilogue
    using Epilogue = typename cutlass::epilogue::threadblock::
        EpilogueWithVisitorFromExistingEpilogue<EpilogueVisitor, typename GemmKernel_::Epilogue>::Epilogue;

    // 6. 构造最终 GEMM 内核
    using GemmKernel = cutlass::gemm::kernel::GemmWithEpilogueVisitor<
        typename GemmKernel_::Mma, Epilogue, ThreadblockSwizzle>;

    // 7. 设备端 GEMM 包装
    using Gemm = cutlass::gemm::device::GemmUniversalBaseCompat<GemmKernel>;

    // 8. 执行 GEMM
    Gemm gemm_op;
    typename Gemm::Arguments args{...};
    gemm_op(args, nullptr, stream);
}
```

#### 算法复杂度
- 时间复杂度: O(M * N * K / (tensor_core_throughput))，具体取决于 ThreadblockShape、WarpShape 和 InstructionShape 的配置
- 空间复杂度: O(1) 额外空间（除了输入输出），加上 shared memory 用于流水线（NumStages 个 tile）

### `cutlass_int8_scaled_mm_sm90<ElementOutput, TileShape, ClusterShape, MainloopScheduleType, WithBias>`
- **位置**: `int8_gemm_kernel.cuh`
- **主要功能**: CUTLASS 3.x 风格的 Hopper (SM90) INT8 GEMM 实现，使用 Collective Builder 和 Epilogue Fusion

#### 关键特性
- 使用 CUTLASS 3.x 的新架构，基于 `cute` 库的张量抽象
- 支持 TMA (Tensor Memory Accelerator) 用于高效数据传输
- 使用 Epilogue Visitor Tree (EVT) 实现融合后处理操作

#### 核心实现步骤
```cpp
template <typename ElementOutput, typename TileShape, typename ClusterShape, ...>
void cutlass_int8_scaled_mm_sm90(...) {
    using ArchTag = cutlass::arch::Sm90;
    using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedulerType = cutlass::gemm::PersistentScheduler;

    // 1. 定义 Epilogue Fusion 操作
    // 列广播（逐 token 缩放）
    using XScale = cutlass::epilogue::fusion::
        Sm90ColBroadcast<0, TileShape, ElementCompute, ElementCompute, ...>;

    // 行广播（逐 channel 缩放）
    using WScale = cutlass::epilogue::fusion::
        Sm90RowBroadcast<0, TileShape, ElementCompute, ElementCompute, ...>;

    // 行广播（偏置）
    using Bias = cutlass::epilogue::fusion::
        Sm90RowBroadcast<0, TileShape, ElementOutput, ElementOutput, ...>;

    // 获取累加器
    using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

    // 构建计算树：先乘 WScale，再乘 XScale，最后加 Bias（如果有）
    using Compute0 = cutlass::epilogue::fusion::Sm90Compute<multiplies, ...>;
    using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<Compute0, WScale, Accum>;

    using Compute1 = cutlass::epilogue::fusion::Sm90Compute<multiplies, ...>;
    using EVTCompute1 = cutlass::epilogue::fusion::Sm90EVT<Compute1, XScale, EVTCompute0>;

    using ComputeWithBias = cutlass::epilogue::fusion::Sm90Compute<multiply_add, ...>;
    using EVTComputeWithBias = cutlass::epilogue::fusion::Sm90EVT<ComputeWithBias, XScale, EVTCompute0, Bias>;

    using EpilogueEVT = conditional<WithBias, EVTComputeWithBias, EVTCompute1>;

    // 2. 构造 Collective Epilogue
    using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute, ElementOutput,
        cutlass::layout::RowMajor, AlignmentC,
        ElementOutput, cutlass::layout::RowMajor, AlignmentOutput,
        EpilogueScheduleType, EpilogueEVT>::CollectiveOp;

    // 3. 自动计算 Stage 数量（减去 epilogue 占用的 shared memory）
    using Stages = cutlass::gemm::collective::StageCountAutoCarveout<
        sizeof(typename CollectiveEpilogue::SharedStorage)>;

    // 4. 构造 Collective Mainloop
    using CollectiveMainloop = cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementInputA, cutlass::layout::RowMajor, AlignmentA,
        ElementInputB, cutlass::layout::ColumnMajor, AlignmentB,
        ElementAccumulator, TileShape, ClusterShape, Stages, MainloopScheduleType>::CollectiveOp;

    // 5. 构造完整内核
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, TileSchedulerType>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // 6. 准备参数
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, 1));

    // 7. 构造 epilogue 参数（包含缩放因子和偏置）
    typename Gemm::Arguments args = {
        cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k, 1},
        {a_ptr, stride_a, b_ptr, stride_b},
        {{}, // epilogue.thread
         nullptr, stride_c, o_ptr, stride_d}
    };

    if constexpr (WithBias) {
        args.epilogue.thread = {
            {a_s_ptr},           // per-token scale
            {{b_s_ptr}, {}, {}}, // per-channel scale
            {bias_ptr},          // bias
            {}
        };
    } else {
        args.epilogue.thread = {
            {a_s_ptr},           // per-token scale
            {{b_s_ptr}, {}, {}}, // per-channel scale
            {},                  // no bias
        };
    }

    // 8. 执行
    gemm_op(args, nullptr, stream);
}
```

#### 算法复杂度
- 时间复杂度: O(M * N * K / (hopper_tensor_core_throughput))，在 Hopper 架构上通过 TMA 和 Cluster 优化获得更高吞吐量
- 空间复杂度: O(1) 额外空间，使用更高效的 shared memory 管理

### `Descriptor` (op::i8gemm::nvidia::Descriptor)
- **位置**: `int8_gemm_nvidia.cu`
- **主要功能**: INT8 GEMM 算子的描述符类，提供算子创建和计算接口

#### 核心方法
- `create(...)`: 静态方法，创建算子描述符
  - 验证输出数据类型为 F16 或 BF16
  - 调用 `I8GemmInfo::create()` 验证矩阵布局并提取形状信息
  - 构造 `Descriptor` 对象并初始化 GPU 句柄
  - **复杂度**: O(1)

- `calculate(...)`: 执行 INT8 量化 GEMM 计算
  - **步骤 1 - 架构检测**：
    - 调用 `getSMVersion()` 获取计算能力版本
    - 返回格式：`sm_major * 10 + sm_minor`（例如 80, 86, 89, 90）
  - **步骤 2 - 内核分发**：
    - **Turing (SM75)**: 仅支持 FP16 输出，使用 `sm75_dispatch_shape`
      - 根据 M 维度选择 tile 形状（<=32, <=64, <=256, >256）
      - InstructionShape: `GemmShape<8, 8, 16>`
      - 流水线阶段: 2
    - **Ampere (SM80)**: 支持 FP16 和 BF16，使用 `sm80_dispatch_shape`
      - 根据 M 和 N 维度选择最优 tile
      - InstructionShape: `GemmShape<16, 8, 32>`
      - 流水线阶段: 5-6（取决于 N 维度）
    - **Ada Lovelace (SM86/89)**: 优化版本，使用 `sm89_dispatch_shape`
      - 针对 100KB shared memory 限制调整 tile 形状和阶段数
      - InstructionShape: `GemmShape<16, 8, 32>`
      - 流水线阶段: 3-5（更保守以适应较小 shared memory）
    - **Hopper (SM90)**: 使用 CUTLASS 3.x，调用 `sm90_dispatch_shape`
      - 基于 cute 库的新架构，使用 `cutlass_int8_scaled_mm_sm90`
      - 支持 TMA 和 Cluster 优化
      - 如果 CUDA >= 12.0，使用 SM90 原生内核，否则回退到 SM80 兼容模式
  - **步骤 3 - 调度策略**：
    - 所有 dispatch 函数根据 M、N 维度查找预定义的最优 tile 配置表
    - **决策树**：
      ```
      if (m <= threshold1):
          if (n <= threshold2): use config_A
          else: use config_B
      else if (m <= threshold3):
          ...
      else:
          use default_config
      ```
  - **复杂度**:
    - 架构检测: O(1) CUDA API 调用
    - 调度: O(1) 条件判断
    - 内核执行: O(M * N * K / throughput)

#### 生命周期
- 在用户调用 `infiniopCreateInt8GemmDescriptor` 时创建
- 在调用 `infiniopInt8Gemm` 时重复使用（每次调用可能重新初始化参数）
- 在用户销毁描述符时析构

## 3. API 接口

### 公共 C API
```cpp
// 创建算子描述符
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,                // InfiniOP 句柄
    Descriptor **desc_ptr,                  // 输出：描述符指针
    infiniopTensorDescriptor_t out_desc,    // 输出张量描述符 (M x N, F16/BF16)
    infiniopTensorDescriptor_t bias_desc,   // 偏置张量描述符 (N,) 或 nullptr
    infiniopTensorDescriptor_t a_desc,      // 矩阵 A 描述符 (M x K, INT8)
    infiniopTensorDescriptor_t a_scale_desc, // A 的缩放因子 (M,) 或 (1,)
    infiniopTensorDescriptor_t b_desc,      // 矩阵 B 描述符 (K x N, INT8)
    infiniopTensorDescriptor_t b_scale_desc  // B 的缩放因子 (N,) 或 (1,)
);
// 返回: INFINI_STATUS_SUCCESS 或错误码

// 执行计算
infiniStatus_t Descriptor::calculate(
    void *workspace,          // 工作空间指针（可为 nullptr）
    size_t workspace_size,    // 工作空间大小
    void *out,                // 输出矩阵 (M x N)
    const void *bias,         // 偏置向量 (N,) 或 nullptr
    const void *a,            // 矩阵 A (M x K, INT8)
    const void *a_scale,      // A 的缩放因子
    const void *b,            // 矩阵 B (K x N, INT8)
    const void *b_scale,      // B 的缩放因子
    void *stream              // CUDA 流
) const;
// 返回: INFINI_STATUS_SUCCESS 或错误码
```

### 内核模板接口
```cpp
// CUTLASS 2.x 内核 (SM75/80/89)
template <
    typename ElementOutput,      // cutlass::half_t 或 cutlass::bfloat16_t
    typename ArchTag,            // cutlass::arch::Sm75/Sm80
    typename ThreadblockShape,   // 如 cutlass::gemm::GemmShape<128, 128, 64>
    typename WarpShape,          // 如 cutlass::gemm::GemmShape<64, 64, 64>
    typename InstructionShape,   // 如 cutlass::gemm::GemmShape<16, 8, 32>
    int NumStages>               // 2-6 流水线阶段数
void cutlass_int8_scaled_mm(
    void *out, const void *a, const void *b,
    const void *a_scale, const void *b_scale, const void *bias,
    int m, int n, int k,
    int lda, int ldb, int ldd, void *stream
);

// CUTLASS 3.x 内核 (SM90)
template <
    typename ElementOutput,      // cutlass::half_t 或 cutlass::bfloat16_t
    typename TileShape,          // 如 Shape<_128, _128, _128>
    typename ClusterShape,       // 如 Shape<_2, _1, _1>
    typename MainloopScheduleType,// 如 KernelTmaWarpSpecialized
    bool WithBias>               // true/false
void cutlass_int8_scaled_mm_sm90(
    void *out, const void *a, const void *b,
    const void *a_scale, const void *b_scale, const void *bias,
    int m, int n, int k,
    int lda, int ldb, int ldd, void *stream
);
```

### Epilogue Visitor 接口
```cpp
// 访问者回调接口（由 GEMM 内核调用）
CUTLASS_DEVICE void visit(
    int iter_idx,           // 迭代索引
    int row_idx,            // 行索引
    int column_idx,         // 列索引
    int frag_idx,           // 片段索引
    AccumulatorFragment const &accum  // 累加器片段 [int32_t]
);
// 作用：对每个累加器元素应用缩放和偏置：
//   1. result = float(accum)
//   2. result *= (scale_col * scale_row)
//   3. if (bias) result += bias
//   4. output = ElementType(result)
```

## 4. 使用示例

### 基本用法
```cpp
#include "infiniop.h"
#include "ops/scaled_mm/int8_gemm.h"

// 1. 创建张量描述符
infiniopTensorDescriptor_t out_desc, a_desc, b_desc, a_scale_desc, b_scale_desc, bias_desc;

// 输出: M x N, FP16
infiniopCreateTensorDescriptor(handle, &out_desc,
    INFINI_DTYPE_F16, 2, {M, N}, {N, 1});

// 矩阵 A: M x K, INT8, Row Major
infiniopCreateTensorDescriptor(handle, &a_desc,
    INFINI_DTYPE_I8, 2, {M, K}, {K, 1});

// 矩阵 B: K x N, INT8, Column Major (转置存储)
infiniopCreateTensorDescriptor(handle, &b_desc,
    INFINI_DTYPE_I8, 2, {K, N}, {N, 1});

// Per-token 缩放: M 个元素 (每行一个 scale)
infiniopCreateTensorDescriptor(handle, &a_scale_desc,
    INFINI_DTYPE_F32, 1, {M}, {1});

// Per-channel 缩放: N 个元素 (每列一个 scale)
infiniopCreateTensorDescriptor(handle, &b_scale_desc,
    INFINI_DTYPE_F32, 1, {N}, {1});

// 偏置: N 个元素
infiniopCreateTensorDescriptor(handle, &bias_desc,
    INFINI_DTYPE_F16, 1, {N}, {1});

// 2. 创建算子描述符
op::i8gemm::nvidia::Descriptor *gemm_desc;
auto status = op::i8gemm::nvidia::Descriptor::create(
    handle, &gemm_desc, out_desc, bias_desc,
    a_desc, a_scale_desc, b_desc, b_scale_desc
);

// 3. 分配 GPU 内存
int8_t *d_A, *d_B;
float *d_a_scale, *d_b_scale;
half *d_out, *d_bias;

cudaMalloc(&d_A, M * K * sizeof(int8_t));
cudaMalloc(&d_B, K * N * sizeof(int8_t));
cudaMalloc(&d_a_scale, M * sizeof(float));
cudaMalloc(&d_b_scale, N * sizeof(float));
cudaMalloc(&d_bias, N * sizeof(half));
cudaMalloc(&d_out, M * N * sizeof(half));

// 拷贝数据到 GPU
cudaMemcpy(d_A, h_A, M * K * sizeof(int8_t), cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, K * N * sizeof(int8_t), cudaMemcpyHostToDevice);
cudaMemcpy(d_a_scale, h_a_scale, M * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_b_scale, h_b_scale, N * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_bias, h_bias, N * sizeof(half), cudaMemcpyHostToDevice);

// 4. 执行计算
cudaStream_t stream;
cudaStreamCreate(&stream);

status = gemm_desc->calculate(
    nullptr, 0,              // 无需工作空间
    d_out,                   // 输出
    d_bias,                  // 偏置
    d_A, d_a_scale,          // 矩阵 A 和其缩放因子
    d_B, d_b_scale,          // 矩阵 B 和其缩放因子
    stream                   // CUDA 流
);

// 5. 获取结果
cudaMemcpy(h_out, d_out, M * N * sizeof(half), cudaMemcpyDeviceToHost);

// 6. 清理
cudaFree(d_A); cudaFree(d_B);
cudaFree(d_a_scale); cudaFree(d_b_scale);
cudaFree(d_bias); cudaFree(d_out);
cudaStreamDestroy(stream);
delete gemm_desc;
```

### 高级用法：动态批处理
```cpp
// 3D 张量支持 (Batch x M x K)
infiniopTensorDescriptor_t batch_a_desc;
infiniopCreateTensorDescriptor(handle, &batch_a_desc,
    INFINI_DTYPE_I8, 3,
    {Batch, M, K},           // 形状
    {M * K, K, 1}            // 跨步（紧凑布局）
);

// Batch x K x N
infiniopTensorDescriptor_t batch_b_desc;
infiniopCreateTensorDescriptor(handle, &batch_b_desc,
    INFINI_DTYPE_I8, 3,
    {Batch, K, N},
    {K * N, N, 1}
);

// Batch x M x N
infiniopTensorDescriptor_t batch_out_desc;
infiniopCreateTensorDescriptor(handle, &batch_out_desc,
    INFINI_DTYPE_F16, 3,
    {Batch, M, N},
    {M * N, N, 1}
);

// 创建算子（自动检测批处理模式）
op::i8gemm::nvidia::Descriptor *batch_gemm_desc;
status = op::i8gemm::nvidia::Descriptor::create(
    handle, &batch_gemm_desc,
    batch_out_desc, bias_desc,
    batch_a_desc, a_scale_desc,
    batch_b_desc, b_scale_desc
);

// 执行批量 GEMM（每个 batch 独立计算）
status = batch_gemm_desc->calculate(
    nullptr, 0, d_batch_out, d_bias,
    d_batch_A, d_a_scale,
    d_batch_B, d_b_scale,
    stream
);
```

### 架构特定优化示例
```cpp
// 获取 GPU 计算能力
int getSMVersion() {
    int device, sm_major, sm_minor;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device);
    return sm_major * 10 + sm_minor;  // 例如 80, 86, 89, 90
}

// 根据架构手动选择最优配置
int sm_version = getSMVersion();

if (sm_version == 86 || sm_version == 89) {
    // Ada Lovelace: 使用保守的 shared memory 配置
    // 自动使用 sm89_dispatch_shape
    printf("Using Ada Lovelace optimized kernels (100KB SMEM)\n");
} else if (sm_version == 80) {
    // Ampere: 使用大 shared memory 配置
    // 自动使用 sm80_dispatch_shape
    printf("Using Ampere optimized kernels (160KB SMEM)\n");
} else if (sm_version == 90) {
    // Hopper: 使用 CUTLASS 3.x 和 TMA
    // 自动使用 sm90_dispatch_shape
    printf("Using Hopper optimized kernels (TMA + Cluster)\n");
}

// 所有优化都自动应用，无需手动配置
```

## 5. 实现细节

### 内存管理
- **输入布局**:
  - 矩阵 A: Row Major (M x K)，leading dimension 为 K
  - 矩阵 B: Column Major (K x N)，逻辑上相当于转置的 Row Major，leading dimension 为 N
  - 输出 D: Row Major (M x N)，leading dimension 为 N
  - 缩放因子: `a_scale` 为 1D 向量 (M,)，`b_scale` 为 1D 向量 (N,)
  - 偏置: 1D 向量 (N,)，行广播
- **对齐要求**:
  - A 矩阵: K 维度需 128-bit 对齐（int8 为 16 元素）
  - B 矩阵: K 维度需 128-bit 对齐（int8 为 16 元素）
  - 输出: N 维度需 128-bit 对齐（FP16/BF16 为 8 元素）
- **Shared Memory 策略**:
  - SM75/80: 使用软件管理的双缓冲流水线（2-6 阶段）
  - SM86/89: 限制为 100KB shared memory，减少流水线阶段（3-5）
  - SM90: 使用硬件 TMA 自动管理数据传输，无需显式双缓冲

### 并发控制
- **Threadblock 并行**:
  - 使用 `GemmIdentityThreadblockSwizzle` 将 M-N 平面划分为 2D grid
  - 每个 threadblock 处理一个 tile (如 128x128)
  - Grid 形状计算: `grid_m = ceil(M / tile_m)`, `grid_n = ceil(N / tile_n)`
- **Warp 并行**:
  - 每个 threadblock 包含多个 warp（通常 2-4 个）
  - 每个 warp 处理 tile 的子块（如 64x64）
  - Warp 内使用 Tensor Core 并行计算 16x8x32 或 8x8x16 的微内核
- **线程同步**:
  - 主循环使用 pipeline barrier 同步 shared memory 加载
  - Epilogue 使用 `__syncthreads()` 确保所有累加器完成后再写入结果
- **Split-K 支持**:
  - Serial Split-K: 多个 threadblock 顺序处理 K 维度的不同切片，需要原子操作归约
  - Parallel Split-K: 每个 K 切片独立计算部分结果，最后在 workspace 中累加

### 性能优化
- **算法选择**:
  - 使用 Tensor Core 硬件加速 INT8 矩阵乘法
  - SM75: INT8 Tensor Core吞吐量为 FP32 的 4 倍
  - SM80+: 支持 INT8 MMA 指令，更高效的寄存器压力管理
  - SM90: 使用 Hopper Tensor Core，支持 TMA 和 Cluster 优化
- **Tile 形状优化**:
  - 小 M (<=32): 使用小 tile 减少空闲线程 (16x64x128)
  - 中 M (64-128): 使用平衡形状 (64x128x128)
  - 大 M (>128): 使用大 tile 提高占用率 (128x128x64 或 256x128x64)
  - 根据 N 维度调整 N-tile 大小以减少全局内存访问
- **流水线深度**:
  - SM75: 2 阶段（Turing shared memory 较小）
  - SM80: 5-6 阶段（Ampere 大 shared memory）
  - SM86/89: 3-5 阶段（Ada 100KB 限制）
  - SM90: 自动阶段数（基于 shared memory 使用情况动态计算）
- **指令级优化**:
  - 使用 `CUTLASS_PRAGMA_UNROLL` 展开内层循环
  - Epilogue 中的逐元素操作完全向量化（处理 kElementsPerAccess 个元素）
  - 使用 `arch::global_load` 进行缓存控制的全局内存加载

### 错误处理
- **对齐检查**:
  - `can_implement()` 检查 A、B、C 矩阵的维度是否满足对齐要求
  - 不满足时返回 `Status::kErrorMisalignedOperand`
- **网格限制检查**:
  - 验证 grid.y 和 grid.z 不超过 uint16_t 最大值 (65535)
  - 超过时返回 `Status::kErrorInvalidProblem`
- **工作空间验证**:
  - 如果需要工作空间但未提供，返回 `Status::kErrorWorkspaceNull`
  - Split-K 模式失败时返回错误状态
- **CUDA 错误传播**:
  - 使用 `check_cutlass_status()` 包装 CUTLASS 状态码
  - 转换为 InfiniOP 的 `INFINI_STATUS_INTERNAL_ERROR`
  - 内核启动失败时通过 `cudaGetLastError()` 捕获

### 依赖关系
- **外部依赖**:
  - CUTLASS 2.x: 用于 SM75/80/89 实现
    - `cutlass/gemm/device/gemm.h`
    - `cutlass/gemm/kernel/default_gemm.h`
    - `cutlass/epilogue/threadblock/epilogue_with_visitor.h`
  - CUTLASS 3.x: 用于 SM90 实现（如果 CUDA >= 12.0）
    - `cutlass/gemm/collective/collective_builder.hpp`
    - `cutlass/epilogue/collective/collective_builder.hpp`
    - `cute/tensor.hpp`
  - CUDA Toolkit: 需要兼容的 CUDA 驱动和运行时
- **内部依赖**:
  - `../../../devices/nvidia/nvidia_handle.cuh`: GPU 句柄管理
  - `../../../devices/nvidia/nvidia_kernel_common.cuh`: 通用 CUDA 内核工具
  - `../int8_gemm.h`: 算子接口定义
  - `../info.h`: 矩阵布局验证工具
- **架构兼容性**:
  - SM75 (Turing): 支持 FP16 输出，INT8 Tensor Core
  - SM80 (Ampere): 支持 FP16/BF16 输出，增强 INT8 Tensor Core
  - SM86/89 (Ada Lovelace): 优化 shared memory 使用
  - SM90 (Hopper): CUTLASS 3.x 完整支持，TMA 加速

### 设计模式
- **模板方法模式**:
  - `GemmUniversalBaseCompat` 定义 GEMM 执行骨架（initialize -> run）
  - 具体内核类型（`GemmKernel`）实现细节
- **策略模式**:
  - `EpilogueVisitor` 抽象接口，`EpilogueVisitorPerRowPerCol` 具体实现
  - 不同架构使用不同的调度策略（`sm75_dispatch_shape`, `sm80_dispatch_shape`, 等）
- **访问者模式**:
  - `EpilogueVisitor` 访问累加器矩阵的每个元素
  - `visit()` 方法定义对每个元素的后处理逻辑
- **适配器模式**:
  - `GemmUniversalBaseCompat` 适配 CUTLASS 2.x 内核到统一接口
  - `GemmUniversalAdapter` (SM90) 适配 CUTLASS 3.x 内核
- **工厂模式**:
  - `Descriptor::create()` 根据架构选择最优内核实现
  - `DefaultGemmConfiguration` 自动生成最优 GEMM 配置
- **组合模式**:
  - CUTLASS 3.x Epilogue Fusion 使用表达式树组合多个操作（Sm90Compute, Sm90Broadcast, 等）
- **Builder 模式**:
  - `CollectiveBuilder` (SM90) 构造复杂的主循环和 epilogue 配置
  - `cutlass::make_cute_packed_stride()` 构造张量跨步

### 关键算法
1. **INT8 量化矩阵乘法**:
   ```
   // 输入: A[MxK] (int8), B[KxN] (int8)
   // 输出: D[MxN] (fp16/bf16)
   // 算法:
   for i in 0..M:
       for j in 0..N:
           acc = 0
           for k in 0..K:
               acc += A[i,k] * B[k,j]  // INT8 乘法，INT32 累加
           // 后处理
           result = float(acc)
           result *= a_scale[i] * b_scale[j]  // 逐行逐列缩放
           if bias:
               result += bias[j]
           D[i,j] = fp16(result)
   ```

2. **Threadblock Tiling**:
   ```
   // 将 MxN 输出划分为 tile_m x tile_n 的网格
   for tb_m in 0..ceil(M/tile_m):
       for tb_n in 0..ceil(N/tile_n):
           // 加载当前 tile 的 A 和 B 子块到 shared memory
           load_A_to_smem(A[tb_m*tile_m : (tb_m+1)*tile_m, 0:K])
           load_B_to_smem(B[0:K, tb_n*tile_n : (tb_n+1)*tile_n])
           // 使用 Tensor Core 计算
           C_tile = tensor_core_multiply(smem_A, smem_B)
           // 应用 epilogue
           D_tile = epilogue(C_tile, scales, bias)
           store_D_to_gmem(D_tile, D[tb_m*tile_m:, tb_n*tile_n:])
   ```

3. **Warp-level Tiling**:
   ```
   // 每个 threadblock 内的 warp 并行
   // Threadblock tile: 128x128
   // Warp tile: 64x64
   // 每个 warp 处理 threadblock tile 的 1/4
   warp_id = threadIdx.x / 32
   warp_m = warp_id % 2  // 0 or 1
   warp_n = warp_id / 2  // 0 or 1
   // 每个 warp 计算 64x64 子块
   subtile_C = tensor_core_multiply(
       A[warp_m*64 : (warp_m+1)*64, :],
       B[:, warp_n*64 : (warp_n+1)*64]
   )
   ```

4. **Tensor Core 微内核** (SM80 InstructionShape 16x8x32):
   ```
   // 每个 Tensor Core 操作计算 16x8 输出，使用 32 个累加维度
   // 输入: 16x32 和 32x8 的 INT8 矩阵片段
   // 输出: 16x8 的 INT32 累加器矩阵
   // 指令: mma.sync.aligned.m16n8k32.row.col.f32.i8.i8
   fragment_C = mma.sync(fragment_A, fragment_B, fragment_C)
   // 执行: C = A * B + C (INT8 x INT8 -> INT32)
   ```

5. **Epilogue Visitor Tree** (SM90):
   ```
   // 构建融合表达式树
   Accum = load_accumulator()  // 从 MMA 获取累加器
   WScale = broadcast_row(b_scale)  // 广播逐 channel 缩放
   XScale = broadcast_col(a_scale)  // 广播逐 token 缩放
   Bias = broadcast_row(bias)       // 广播偏置

   // 计算树
   temp0 = multiply(Accum, WScale)   // accum * b_scale[j]
   temp1 = multiply(temp0, XScale)   // (accum * b_scale[j]) * a_scale[i]
   result = multiply_add(temp1, Bias, 1.0)  // result + bias[j]

   // 优化: 编译期融合为单个内核操作，无中间存储
   ```

6. **Split-K 归约** (Serial Split-K):
   ```
   // 当 K 维度很大时，将其分为多个切片
   K_slices = ceil(K / gemm_k_size)
   for slice in 0..K_slices:
       k_start = slice * gemm_k_size
       k_end = min((slice+1) * gemm_k_size, K)
       // 计算部分结果
       partial_D = gemm(A[:, k_start:k_end], B[k_start:k_end, :])
       // 归约到最终结果
       if slice == 0:
           D = partial_D
       else:
           D += partial_D  // 原子加或串行归约
   ```
