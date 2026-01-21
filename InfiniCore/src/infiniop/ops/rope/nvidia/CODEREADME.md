# RoPE (Rotary Position Embedding) NVIDIA CUDA 实现文档

本模块实现了 Rotary Position Embedding (RoPE) 旋转位置编码的 NVIDIA CUDA 后端，支持 GPT-J 和 GPT-NeoX 两种主流位置编码算法，用于 Transformer 模型中注入相对位置信息。

## 1. 模块结构

- **`rope_nvidia.cuh`**: NVIDIA 后端描述符定义头文件，通过宏 `DESCRIPTOR(nvidia)` 展开生成完整的 `op::rope::nvidia::Descriptor` 类声明
- **`rope_nvidia.cu`**: 核心实现文件，包含描述符创建、销毁以及 RoPE 计算的主调度逻辑

## 2. 核心类与数据结构

### `RoPEInfo`
- **位置**: `../rope.h`
- **功能**: 封装 RoPE 操作的所有张量形状、步长和数据类型信息，通过工厂方法 `createRoPEInfo()` 在运行时验证输入张量的合法性
- **核心成员**:
  - `data_type`: 输入/输出张量的数据类型 (支持 F16/BF16/F32/F64)
  - `pos_type`: 位置 ID 的数据类型 (支持所有有符号/无符号整数类型)
  - `batch, seqlen, nhead, dhead`: 张量的四个维度（批量大小、序列长度、注意力头数、每个头的维度）
  - `table_len, table_dim`: sin/cos 查找表的长度和维度（`dhead = table_dim * 2`）
  - `y_stride_batch, y_stride_seqlen, y_stride_nhead`: 输出张量在三个维度上的步长
  - `x_stride_batch, x_stride_seqlen, x_stride_nhead`: 输入张量在三个维度上的步长
  - `has_batch_dim`: 是否包含批量维度（支持 3D 和 4D 张量）
  - `pos_has_batch_dim`: 位置 ID 张量是否包含批量维度（支持 1D 共享或 2D 每 batch 独立）
  - `algo`: RoPE 算法类型（GPT-J 或 GPT-NeoX）

### `op::rope::nvidia::Descriptor`
- **位置**: `rope_nvidia.cu` (通过 `rope_nvidia.cuh` 中的宏展开生成)
- **功能**: RoPE 操作的 NVIDIA 设备描述符，管理 CUDA 设备句柄和计算配置
- **核心成员**:
  - `_opaque`: 不透明指针，持有 `device::nvidia::Handle::Internal` 的共享所有权
  - `_info`: `RoPEInfo` 实例，存储所有张量元信息
  - `_workspace_size`: 工作空间大小（当前实现为 0）
- **核心方法**:
  - `create(handle, desc_ptr, y_desc, x_desc, pos_desc, sin_desc, cos_desc, algo)`: 静态工厂方法，验证输入张量并创建描述符实例
    - 调用 `RoPEInfo::createRoPEInfo()` 进行形状验证
    - 检查 `y_desc`, `x_desc`, `pos_desc`, `sin_desc`, `cos_desc` 非空
    - 验证所有数据张量类型一致且为浮点类型
    - 验证位置 ID 类型为整数类型
    - 支持 3D 张量 `[seqlen, nhead, dhead]` 和 4D 张量 `[batch, seqlen, nhead, dhead]`
    - 验证 `dhead = table_dim * 2`（RoPE 要求头部维度是查找表维度的两倍）
    - 复杂度: O(1)
  - `calculate(workspace, workspace_size, y, x, pos_ids, sin_table, cos_table, stream)`: 执行 RoPE 计算
    - 根据 `data_type` 和 `pos_type` 进行双重分发，生成 32 种特化实现
    - 为每个 `(seq_idx, head_idx, batch_idx)` 元素启动一个 CUDA block
    - 每个块内的线程并行处理 `table_dim` 个旋转角度
    - 复杂度: O(batch × seqlen × nhead × table_dim / blockDim)
  - `~Descriptor()`: 析构函数，释放 `_opaque` 指针
- **生命周期**: 由用户通过 `infiniopCreateRoPEDescriptor()` 创建，通过 `infiniopDestroyRoPEDescriptor()` 销毁

### `op::rope::nvidia::Descriptor::Opaque`
- **位置**: `rope_nvidia.cu`
- **功能**: 封装 NVIDIA 设备相关的内部状态
- **核心成员**:
  - `internal`: `std::shared_ptr<device::nvidia::Handle::Internal>`，持有 CUDA 设备上下文和流管理器

## 3. CUDA 核函数接口

### `ropeThreadPerItemKernel<Tdata, Tindex, Tangle>`
- **位置**: `rope_nvidia.cu:9-34`
- **功能**: RoPE 计算的 CUDA 核函数入口，每个线程块处理一个 `(seqlen, nhead, batch)` 位置的元素
- **模板参数**:
  - `IsGPTJ`: 布尔编译期常量，`true` 表示 GPT-J 算法，`false` 表示 GPT-NeoX 算法
  - `Tdata`: 数据类型 (half, cuda_bfloat16, float, double)
  - `Tindex`: 位置 ID 类型 (uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t)
  - `Tangle`: 角度查找表类型 (通常与 Tdata 相同)
- **参数**:
  - `y_`: 输出张量指针
  - `x_`: 输入张量指针
  - `pos_ids`: 位置 ID 张量指针（1D `[seqlen]` 或 2D `[batch, seqlen]`）
  - `sin_table, cos_table`: 预计算的 sin/cos 查找表，形状 `[table_len, table_dim]`
  - `table_dim`: 查找表的维度（`dhead / 2`）
  - `pos_stride_batch`: 位置 ID 在批量维度上的步长（0 表示 1D 张量）
  - `pos_has_batch_dim`: 位置 ID 是否有批量维度
  - `has_batch_dim`: 数据张量是否有批量维度
  - `y_stride_batch, y_stride_seqlen, y_stride_nhead`: 输出张量步长
  - `x_stride_batch, x_stride_seqlen, x_stride_nhead`: 输入张量步长
- **执行配置**:
  - Grid Dim: `(seqlen, nhead, batch)` 对于 4D 张量，`(seqlen, nhead)` 对于 3D 张量
  - Block Dim: `max(table_dim, maxThreadsPerBlock)`，通常为 1024 或设备最大线程数
- **实现逻辑**: 直接委托给 `ropeThreadPerItemBlock<IsGPTJ>()` 函数执行核心计算

### `ropeThreadPerItemBlock<IsGPTJ, Tdata, Tindex, Tangle>`
- **位置**: `../cuda/kernel.cuh:4-104`
- **功能**: RoPE 核心计算逻辑，每个线程处理一个旋转角度对
- **线程索引映射**:
  - `blockIdx.x`: 序列索引 `seq_idx` ∈ `[0, seqlen)`
  - `blockIdx.y`: 注意力头索引 `head_idx` ∈ `[0, nhead)`
  - `blockIdx.z`: 批量索引 `batch_idx` ∈ `[0, batch)` (仅 4D 张量)
  - `threadIdx.x`: 旋转角度索引 `i` ∈ `[0, table_dim)`，每个线程处理一个角度，跨步循环覆盖所有维度
- **关键算法**:
  ```cpp
  // 计算当前元素的内存偏移
  size_t batch_idx = has_batch_dim ? blockIdx.z : 0;
  size_t seq_idx = blockIdx.x;
  size_t head_idx = blockIdx.y;

  // 计算位置 ID
  size_t pos_offset = pos_has_batch_dim ? (batch_idx * pos_stride_batch + seq_idx) : seq_idx;
  size_t pos_id = pos_ids[pos_offset];

  // 查找 sin/cos 值
  size_t table_offset = pos_id * table_dim;
  Tangle sin__ = sin_table[table_offset + i];
  Tangle cos__ = cos_table[table_offset + i];
  ```
- **GPT-J 算法** (`IsGPTJ == true`):
  - 旋转逻辑: 对相邻的 `(x[2i], x[2i+1])` 对应用 2D 旋转矩阵
  - 公式:
    ```
    y[2i]   =  cos(θ) * x[2i]   - sin(θ) * x[2i+1]
    y[2i+1] =  sin(θ) * x[2i]   + cos(θ) * x[2i+1]
    ```
  - 特化优化:
    - `half` 类型: 使用 `half2` 向量指令，一次处理两个 FP16 值
    - `cuda_bfloat16` 类型: 使用 `cuda_bfloat162` 向量指令，通过 `__low2bfloat16()`/`__high2bfloat16()` 提取分量
    - `float/double` 类型: 标量计算
- **GPT-NeoX 算法** (`IsGPTJ == false`):
  - 旋转逻辑: 对 `(x[i], x[i+table_dim])` 跨越半个维度的元素对应用旋转
  - 公式:
    ```
    y[i]             =  cos(θ) * x[i]             - sin(θ) * x[i+table_dim]
    y[i+table_dim]   =  sin(θ) * x[i]             + cos(θ) * x[i+table_dim]
    ```
  - 特化优化:
    - `half` 类型: 通过 `__half2float()`/`__float2half()` 进行 FP16↔FP32 转换后计算
    - `cuda_bfloat16` 类型: 通过 `__bfloat162float()`/`__float2bfloat16()` 进行 BF16↔FP32 转换
    - `float/double` 类型: 标量计算
- **性能优化**:
  - 使用 `__restrict__` 指针限定符，提示编译器消除内存别名
  - 使用 `const` 限定符启用编译器优化
  - 循环跨步: `for (size_t i = threadIdx.x; i < table_dim; i += blockDim.x)` 允许任意 blockDim 覆盖所有 table_dim
  - 编译期 `if constexpr` 消除分支，为每种数据类型生成最优代码

## 4. RoPE 算法说明

### 数学原理
RoPE (Rotary Position Embedding) 通过旋转矩阵将位置信息注入到注意力计算的 Query 和 Key 向量中。对于位置 `m` 和 `n`，旋转后的向量满足:
```
⟨RoPE(x, m), RoPE(x, n)⟩ = ⟨x, x⟩ × f(m - n)
```
即点积仅依赖于相对位置差 `m - n`，而非绝对位置。

### 两种算法变体

#### GPT-J 风格 (`INFINIOP_ROPE_ALGO_GPT_J`)
- **排列方式**: 交错排列 (interleaved)
- **旋转对**: `(x[0], x[1]), (x[2], x[3]), ..., (x[dhead-2], x[dhead-1])`
- **适用模型**: GPT-J, LLaMA 系列
- **优点**: 内存访问更连续，缓存友好

#### GPT-NeoX 风格 (`INFINIOP_ROPE_ALGO_GPT_NEOX`)
- **排列方式**: 分半排列 (halved)
- **旋转对**: `(x[0], x[table_dim]), (x[1], x[table_dim+1]), ..., (x[table_dim-1], x[dhead-1])`
- **适用模型**: GPT-NeoX, BLOOM, ChatGLM
- **优点**: 与某些注意力实现（如分组查询注意力）的对齐方式更自然

## 5. API 使用示例

```cpp
// 假设我们有以下张量:
// x: [batch, seqlen, nhead, dhead] - 输入 Query 或 Key 向量
// pos_ids: [batch, seqlen] - 每个位置的 ID（通常为 0, 1, 2, ..., seqlen-1）
// sin_table, cos_table: [max_seq_len, table_dim] - 预计算的旋转角度表
// y: [batch, seqlen, nhead, dhead] - 输出 RoPE 编码后的向量

// 1. 创建描述符
infiniopRoPEDescriptor_t rope_desc;
infiniStatus_t status = infiniopCreateRoPEDescriptor(
    handle,                          // InfiniHandle（已初始化的 CUDA 上下文）
    &rope_desc,                      // 输出描述符指针
    y_desc,                          // 输出张量描述符
    x_desc,                          // 输入张量描述符
    pos_desc,                        // 位置 ID 张量描述符
    sin_desc,                        // sin 查找表描述符
    cos_desc,                        // cos 查找表描述符
    INFINIOP_ROPE_ALGO_GPT_J         // 算法选择（GPT-J 或 GPT-NeoX）
);

if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误（形状不匹配、数据类型不支持等）
}

// 2. 查询工作空间大小
size_t workspace_size;
status = infiniopGetRoPEWorkspaceSize(rope_desc, &workspace_size);
// 当前实现返回 0，无需额外工作空间

void *workspace = nullptr;  // 如果需要，分配 workspace_size 字节的 GPU 内存

// 3. 执行 RoPE 计算
status = infiniopRoPE(
    rope_desc,      // 描述符
    workspace,      // 工作空间（可为 nullptr）
    workspace_size, // 工作空间大小
    d_y,            // 输出张量的设备指针
    d_x,            // 输入张量的设备指针
    d_pos_ids,      // 位置 ID 的设备指针
    d_sin_table,    // sin 表的设备指针
    d_cos_table,    // cos 表的设备指针
    stream          // CUDA 流
);

// 4. 同步并使用结果
cudaStreamSynchronize(stream);
// y 现在包含 RoPE 编码后的向量，可直接用于注意力计算

// 5. 清理
infiniopDestroyRoPEDescriptor(rope_desc);
```

### 支持的张量形状

#### 4D 张量（带批量维度）
```cpp
// 形状: [batch, seqlen, nhead, dhead]
// 要求: dhead = table_dim * 2
// 示例: [4, 128, 32, 128] 其中 table_dim = 64
```

#### 3D 张量（无批量维度）
```cpp
// 形状: [seqlen, nhead, dhead]
// 要求: dhead = table_dim * 2
// 示例: [128, 32, 128] 其中 table_dim = 64
```

#### 位置 ID 张量
```cpp
// 1D 共享位置: [seqlen] - 所有 batch 使用相同的位置编码
// 2D 独立位置: [batch, seqlen] - 每个 batch 有独立的位置 ID
```

## 6. 实现细节

### 类型分发机制
代码使用宏实现双重类型分发，支持 4 × 8 = 32 种特化组合：
- **数据类型** (`Tdata`): `half`, `cuda_bfloat16`, `float`, `double`
- **位置 ID 类型** (`Tindex`): `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`, `int8_t`, `int16_t`, `int32_t`, `int64_t`

```cpp
// 外层宏: 根据数据类型分发
#define ROPE_TYPE(TDATA)                        \
    switch (_info.pos_type) {                   \
    case INFINI_DTYPE_U8:                       \
        return CALCULATE_ROPE(TDATA, uint8_t);  \
    // ... 其他整数类型
    }

// 内层宏: 根据位置 ID 类型分发
#define CALCULATE_ROPE(TDATA, TINDEX)           \
    calculateRoPE(_info,                        \
                  _opaque->internal->maxThreadsPerBlock(), \
                  (TDATA *)y,                   \
                  (const TDATA *)x,             \
                  (const TINDEX *)pos_ids,      \
                  (const TDATA *)sin_table,     \
                  (const TDATA *)cos_table,     \
                  (cudaStream_t)stream)
```

### CUDA 内核启动配置
- **Grid 维度**:
  - 4D 张量: `dim3(seqlen, nhead, batch)` - 总共 `seqlen × nhead × batch` 个块
  - 3D 张量: `dim3(seqlen, nhead)` - 总共 `seqlen × nhead` 个块
- **Block 维度**: `max(table_dim, maxThreadsPerBlock)` - 通常为 1024，确保能覆盖所有 table_dim
- **共享内存**: 使用 0 字节共享内存
- **流**: 支持异步执行，通过 `stream` 参数指定

### 内存访问模式
- **输入/输出张量**: 使用步长 (stride) 进行间接索引，支持非连续内存布局
  - 偏移计算: `batch_idx * stride_batch + seq_idx * stride_seqlen + head_idx * stride_nhead + elem_idx`
  - 要求: 最后一维（维度 `dhead`）必须连续（stride 为 1）
- **sin/cos 查找表**: 要求完全连续 (contiguous)，直接线性索引 `pos_id * table_dim + i`
- **位置 ID 张量**: 根据是否有 batch 维度计算偏移
  - 1D: `seq_idx`
  - 2D: `batch_idx * seqlen + seq_idx`

### 数据类型特化优化

#### FP16 (half) - GPT-J
```cpp
// 使用 half2 向量化，一次处理两个 FP16 值
auto &y = reinterpret_cast<half2&>(y_[y_offset + 2 * i]);
auto &x = reinterpret_cast<const half2&>(x_[x_offset + 2 * i]);
Tangle y0 = x.x * cos__ - x.y * sin__;
Tangle y1 = x.x * sin__ + x.y * cos__;
y = half2(y0, y1);
```

#### BF16 (cuda_bfloat16) - GPT-J
```cpp
// 使用 bfloat162 向量指令，需要手动提取高低位
auto &y = reinterpret_cast<cuda_bfloat162&>(y_[y_offset + 2 * i]);
auto &x = reinterpret_cast<const cuda_bfloat162&>(x_[x_offset + 2 * i]);
Tangle x0 = __low2bfloat16(x);  // 提取低位元素
Tangle x1 = __high2bfloat16(x); // 提取高位元素
Tangle y0 = x0 * cos__ - x1 * sin__;
Tangle y1 = x0 * sin__ + x1 * cos__;
y = __floats2bfloat162_rn(y0, y1);  // 舍入模式为 RN (Round to Nearest)
```

#### BF16 - GPT-NeoX
```cpp
// 非相邻元素访问，需要先转换为 FP32 计算
Tangle x0 = __bfloat162float(x_[x_offset + pos0]);
Tangle x1 = __bfloat162float(x_[x_offset + pos1]);
Tangle y0 = x0 * cos__ - x1 * sin__;
Tangle y1 = x0 * sin__ + x1 * cos__;
y_[y_offset + pos0] = __float2bfloat16(y0);
y_[y_offset + pos1] = __float2bfloat16(y1);
```

### 编译期优化
- **模板特化**: 通过 `template <bool IsGPTJ, typename Tdata, typename Tindex, typename Tangle>` 在编译期生成 2 × 4 × 8 × 1 = 64 个核函数变体（算法 × 数据 × 索引 × 角度）
- **if constexpr**: 编译期条件分支，完全消除运行时判断
  ```cpp
  if constexpr (IsGPTJ) {
      // GPT-J 分支代码
  } else {
      // GPT-NeoX 分支代码
  }
  // 编译后只保留一个分支，零运行时开销
  ```
- **强制内联**: `__forceinline__` 提示编译器内联小函数（如 `indexToOffset`）

### 错误处理
- **形状验证**: 在 `RoPEInfo::createRoPEInfo()` 中进行严格的形状检查
  - 维度数量必须是 3 或 4
  - 数据张量类型必须一致且为浮点类型
  - 位置 ID 必须为整数类型
  - `dhead = table_dim * 2` 硬约束
  - 最后一维必须连续
  - sin/cos 表必须完全连续
- **运行时错误**: 返回 `infiniStatus_t` 错误码
  - `INFINI_STATUS_NULL_POINTER`: 输入描述符为空
  - `INFINI_STATUS_BAD_TENSOR_DTYPE`: 数据类型不匹配或不支持
  - `INFINI_STATUS_BAD_TENSOR_SHAPE`: 张量形状不合法
  - `INFINI_STATUS_BAD_TENSOR_STRIDES`: 步长不满足要求

### 设备兼容性
- **CUDA 架构**: 支持所有现代 CUDA 架构（通过 `maxThreadsPerBlock()` 自适应）
- **块大小限制**:
  - 定义了 `CUDA_BLOCK_SIZE_4096`, `CUDA_BLOCK_SIZE_1024`, `CUDA_BLOCK_SIZE_512` 宏
  - 实际使用 `internal->maxThreadsPerBlock()` 动态获取设备限制
- **混合精度**: 支持 FP16/BF16 低精度计算，加速推理并减少显存占用
- **跨平台**: 通过条件编译支持 Hygon DCU、Iluvatar、QY 等兼容 CUDA 的设备

### 性能特性
- **并行度**: `O(seqlen × nhead × batch)` 个线程块完全并行执行
- **内存带宽**: 每个线程读取 2 个输入值 + 2 个查找表值，写入 2 个输出值
- **计算强度**: 每个元素执行 4 次浮点乘法和 2 次浮点加法（8 FLOPs）
- **延迟**: 主导因素是全局内存访问延迟，而非计算延迟
- **吞吐量优化**:
  - 使用 `half2`/`cuda_bfloat162` 向量指令提升吞吐量 2x
  - 跨步循环减少 warp 内分歧（所有线程执行相同指令流）
  - `__restrict__` 指针帮助编译器优化缓存预取

### 依赖关系
- **内部依赖**:
  - `../rope.h`: RoPEInfo 类定义和 DESCRIPTOR 宏
  - `../cuda/kernel.cuh`: `ropeThreadPerItemBlock` 核心计算函数
  - `../../../devices/nvidia/nvidia_common.cuh`: NVIDIA 设备类型定义
  - `../../../devices/nvidia/nvidia_kernel_common.cuh`: CUDA 内核公共宏和类型定义
- **外部依赖**:
  - CUDA Toolkit: `cuda_bf16.h`, `cuda_fp16.h` (半精度和 bfloat16 支持)
  - InfiniCore 基础库: `infinicore.h`, `operator.h`, `tensor.h`, `utils.h`

### 设计模式
- **工厂模式**: `Descriptor::create()` 静态工厂方法封装复杂的对象构建逻辑
- **策略模式**: 通过模板参数 `IsGPTJ` 在编译期选择算法实现，零运行时开销
- **不透明指针模式**: `Opaque` 结构体隐藏 NVIDIA 设备相关的实现细节
- **RAII**: `std::shared_ptr` 管理 `device::nvidia::Handle::Internal` 生命周期
- **特化模板**: 使用模板特化为不同数据类型生成最优代码
