# CPU Convolution Operator Core Implementation Documentation

该模块实现了基于 CPU 的高性能多维卷积运算，支持 1D、2D 和 3D 卷积，提供 F32、F16 和 BF16 三种数据类型的完整支持，包括自动填充（padding）、步长（stride）和膨胀（dilation）等高级特性。

## 1. 模块结构

- **`conv_cpu.h`**: CPU 卷积算子声明文件，通过宏定义 `DESCRIPTOR(cpu)` 生成 Descriptor 类
- **`conv_cpu.cc`**: CPU 卷积算子核心实现，包含所有卷积算法、填充逻辑和并行计算策略

## 2. 核心类

### `Descriptor`
- **位置**: 通过 `DESCRIPTOR(cpu)` 宏在 `conv_cpu.h` 中生成
- **主要功能**: CPU 卷积算子的操作描述符，负责资源计算、参数验证和算子调度
- **关键成员**:
  - `_dtype`: 数据类型 (F16/F32/BF16)
  - `_info`: `ConvInfo` 对象，封装所有卷积参数和维度信息
  - `_workspace_size`: 工作空间大小（字节），用于存储填充输入和 FP32 累加器
  - `_opaque`: 不透明指针，保留给未来扩展
- **核心方法**:
  - `create(handle, desc_ptr, y_desc, x_desc, w_desc, b_desc, pads, strides, dilations, n)`: 静态工厂方法，验证输入参数并创建描述符。计算工作空间需求（填充输入大小 + FP16/BF16 的 float 累加器空间）
  - `calculate(workspace, workspace_size, y, x, w, bias, stream)`: 执行卷积运算的主入口，根据 `_dtype` 分发到对应的模板特化实现
- **生命周期**: 由用户通过 `create()` 静态方法创建，使用完成后由用户负责释放

### `ConvInfo` (来自 `info.h`)
- **位置**: `../info.h`
- **主要功能**: 封装卷积运算的所有元数据，提供紧凑的内存布局
- **关键成员**:
  - `_meta`: `std::vector<size_t>`，紧凑存储所有维度和参数数据
  - `_ndim`: 空间维度数（1D/2D/3D）
  - `_batch`: 批次大小
  - `_in_channels`: 输入通道数
  - `_out_channels`: 输出通道数
  - `_spatial_sizes`: 输出张量的空间维度乘积
  - `_bias_dims_size`: 偏置张量的维度数
  - `_padded_shape_size`: 填充后张量的维度数（0 表示无需填充）
- **数据布局**: `_meta` 数组按顺序存储：
  1. `input_dims[ndim]`: 输入空间维度
  2. `kernel_dims[ndim]`: 卷积核空间维度
  3. `output_dims[ndim]`: 输出空间维度
  4. `bias_dims[bias_dims_size]`: 偏置张量形状
  5. `pads_info[ndim]`: 每个维度的填充大小
  6. `strides_info[ndim]`: 每个维度的步长（ptrdiff_t 类型）
  7. `dilations_info[ndim]`: 每个维度的膨胀系数
  8. `padded_shape[padded_shape_size]`: 填充后的张量形状
- **核心方法**:
  - `create()`: 静态工厂方法，验证张量形状兼容性，计算输出维度，填充所有元数据
  - `input_dim(i)`, `kernel_dim(i)`, `output_dim(i)`: 获取第 i 个空间维度的大小
  - `pad_info(i)`, `stride_info(i)`, `dilation_info(i)`: 获取第 i 个维度的卷积参数
  - `getPaddedShape()`: 返回填充后的张量形状数组
  - `getPadsInfo()`, `getStridesInfo()`, `getDilationsInfo()`: 返回参数数组的指针

## 3. API 接口

```cpp
// 核心计算接口（通过 Descriptor 调用）
infiniStatus_t Descriptor::calculate(
    void *workspace,              // 工作空间指针，至少 workspaceSize() 字节
    size_t workspace_size,        // 工作空间大小
    void *y,                      // 输出张量 [batch, out_channels, spatial_dims...]
    const void *x,                // 输入张量 [batch, in_channels, spatial_dims...]
    const void *w,                // 卷积核 [out_channels, in_channels, kernel_dims...]
    const void *bias,             // 偏置向量 [out_channels] 或 nullptr
    void *stream) const;          // 流指针（CPU 实现忽略）

// 描述符创建接口
static infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,      // 设备句柄
    Descriptor **desc_ptr,        // 输出：描述符指针
    infiniopTensorDescriptor_t y, // 输出张量描述符
    infiniopTensorDescriptor_t x, // 输入张量描述符
    infiniopTensorDescriptor_t w, // 卷积核描述符
    infiniopTensorDescriptor_t b, // 偏置描述符（可为 nullptr）
    const void *pads,             // 填充数组 [ndim] 或 nullptr
    const void *strides,          // 步长数组 [ndim] 或 nullptr
    const void *dilations,        // 膨胀数组 [ndim] 或 nullptr
    size_t n);                    // 空间维度数（1=1D, 2=2D, 3=3D）

// 工作空间查询
size_t Descriptor::workspaceSize() const;
```

## 4. 使用示例

```cpp
// 示例：执行 2D 卷积（batch=4, in_ch=3, out_ch=64, H=224, W=224）
// 卷积核：3x3, stride=1, padding=1

// 1. 准备张量描述符
size_t x_shape[] = {4, 3, 224, 224};
size_t w_shape[] = {64, 3, 3, 3};
size_t y_shape[] = {4, 64, 224, 224};
size_t b_shape[] = {64};

infiniopTensorDescriptor_t x_desc, w_desc, y_desc, b_desc;
infiniopCreateTensorDesc(&x_desc, INFINI_DTYPE_F32, 4, x_shape);
infiniopCreateTensorDesc(&w_desc, INFINI_DTYPE_F32, 4, w_shape);
infiniopCreateTensorDesc(&y_desc, INFINI_DTYPE_F32, 4, y_shape);
infiniopCreateTensorDesc(&b_desc, INFINI_DTYPE_F32, 1, b_shape);

// 2. 设置卷积参数
size_t pads[] = {1, 1};           // 2D padding
ptrdiff_t strides[] = {1, 1};     // stride=1
size_t dilations[] = {1, 1};      // dilation=1

// 3. 创建描述符
op::conv::cpu::Descriptor *conv_desc = nullptr;
infiniStatus_t status = op::conv::cpu::Descriptor::create(
    handle, &conv_desc, y_desc, x_desc, w_desc, b_desc,
    pads, strides, dilations, 2  // 2D 卷积
);

// 4. 分配工作空间
size_t workspace_size = conv_desc->workspaceSize();
void *workspace = malloc(workspace_size);

// 5. 准备数据（假设已填充 x_data, w_data, b_data, y_data）
float *x_data = ...;  // [4*3*224*224]
float *w_data = ...;  // [64*3*3*3]
float *b_data = ...;  // [64]
float *y_data = ...;  // [4*64*224*224]

// 6. 执行卷积
status = conv_desc->calculate(workspace, workspace_size,
                              y_data, x_data, w_data, b_data, nullptr);

// 7. 清理
free(workspace);
delete conv_desc;
```

## 5. 实现细节

### 内存管理策略
- **工作空间复用**:
  - 对于需要填充的情况：前段存储填充后的输入张量
  - 对于 FP16/BF16 类型：后段存储 FP32 累加器（避免精度损失）
  - 填充输入大小通过 `calculatePaddedInputSize()` 计算：`batch * in_channels * ∏(input_dim[i] + 2*pad[i])`
  - FP32 累加器大小通过 `calculateOutputSize()` 计算：`batch * out_channels * ∏output_dim[i]`

- **数据类型特化处理**:
  - **FP32**: 直接计算，无需累加器，输出初始化为 0.0f
  - **FP16/BF16**: 使用 FP32 累加器避免精度损失
    - 输入 FP16/BF16 → 转换为 FP32 → 卷积计算 → 转换回 FP16/BF16
    - 这种策略遵循标准深度学习框架的做法（如 cuDNN）

### 并发策略
- **OpenMP 并行化**:
  - 外层循环使用 `#pragma omp parallel for schedule(dynamic)` 并行化批次和输出通道
  - 循环迭代数：`batch_size * out_channels`，使用 `ptrdiff_t` 索引确保线程安全
  - 动态调度（`schedule(dynamic)`）适用于不规则的负载分布
  - 内层循环（输入通道）串行执行，避免数据竞争
  - 偏置加法也使用 OpenMP 并行化（独立元素操作）

### 卷积算法
- **递归直接卷积** (`_applyConv`):
  - 时间复杂度：O(batch × out_ch × in_ch × ∏(output_dim × kernel_dim))
  - 空间复杂度：O(1) 额外空间（除了工作空间）
  - 算法思路：
    1. 从最内层空间维度开始，递归向外层展开
    2. 每个维度计算步数：`steps = (dim_size - dilation*(kernel_size-1) - 1) / stride + 1`
    3. 对每个输出位置，遍历所有卷积核元素，累加 `x[curr_x_index] * w[curr_w_index]`
    4. 膨胀卷积实现：输入索引偏移 `i * stride + k * dilation`
  - 优点：代码简洁，支持任意维度（1D/2D/3D），参数灵活
  - 缺点：未使用 im2col 或 Winograd 等优化算法，性能较低（参考实现）

- **填充处理** (`fillPaddedInput`):
  - 递归算法，将原始输入拷贝到填充后的缓冲区中心位置
  - 填充区域初始化为 0（使用类型安全的零值）
  - 支持任意维度的填充，填充偏移量通过 `pad_offset = info.pad_info(ndim - 2)` 计算

### 性能优化技术
- **循环展开**: 手动实现的递归展开，避免硬编码维度
- **内存连续性**: 输出张量按行优先顺序填充，利用 CPU 缓存
- **类型特化**: 编译期通过 `if constexpr` 消除分支，生成三个独立的代码版本
- **零初始化优化**: 使用 `std::fill` 批量清零，避免逐元素赋值
- **编译期类型选择**:
  ```cpp
  if constexpr (std::is_same<Xdata, fp16_t>::value || std::is_same<Xdata, bf16_t>::value) {
      y[y_index] += utils::cast<float>(x[curr_x_index]) * utils::cast<float>(w[curr_w_index]);
  } else {
      y[y_index] += x[curr_x_index] * w[curr_w_index];
  }
  ```

### 错误处理
- **参数验证**:
  - `ConvInfo::create()` 验证张量形状兼容性（batch、channels 必须匹配）
  - 输出维度通过公式验证：`output = (input + 2*pad - dilation*(kernel-1) - 1) / stride + 1`
  - 检查 stride != 0、dilation != 0、kernel_size != 0
  - 检查填充后输入足够大：`padded_input >= effective_kernel_size`

- **运行时错误检查**:
  - `calculate()` 检查工作空间是否足够（`workspace_size < _workspace_size` 返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`）
  - 不支持的数据类型返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`

### 设计模式
- **策略模式 (Strategy Pattern)**:
  - 通过模板特化实现不同数据类型的计算策略（`conv_cpu<T>`）
  - FP32: 直接计算
  - FP16/BF16: FP32 累加器 + 类型转换

- **工厂模式 (Factory Pattern)**:
  - `Descriptor::create()` 作为静态工厂方法，封装复杂的对象构建逻辑

- **模板方法模式 (Template Method Pattern)**:
  - `_conv_cpu()` 定义算法骨架（填充判断 → 卷积计算）
  - `applyConv()` 定义并行化策略
  - `_applyConv()` 定义核心卷积递归逻辑

- **CRTP (Curiously Recurring Template Pattern)** 的变体:
  - `DESCRIPTOR(NAMESPACE)` 宏在指定命名空间生成 Descriptor 类
  - 不同后端（cpu, cuda 等）通过相同宏生成不同实现

### 依赖关系
- **外部依赖**:
  - `../conv.h`: 提供 `DESCRIPTOR` 宏定义和 `ConvInfo`
  - `../../../devices/cpu/common_cpu.h`: 提供 `getPaddedSize()` 计算填充后张量大小
  - `../../../utils.h`: 提供 `fp16_t`, `bf16_t` 类型定义和 `utils::cast<>` 转换函数
  - `<algorithm>`: 提供 `std::fill`

- **内部模块依赖**:
  - 依赖 `ConvInfo` 的元数据布局（`getPaddedShape()`, `getPadsInfo()` 等）
  - 依赖 `device::cpu::Handle` 类型（虽然未直接使用其方法）

### 辅助函数

#### `calculatePaddedInputSize(const ConvInfo &info)`
计算填充后输入张量的元素总数。
```cpp
shape[0] = info.batch()
shape[1] = info.in_channels()
shape[2..ndim+1] = info.input_dim(i)
return common_cpu::getPaddedSize(ndim+2, shape.data(), info.getPadsInfo())
```

#### `calculateOutputSize(const ConvInfo &info)`
计算输出张量的元素总数。
```cpp
size = info.batch() * info.out_channels()
for (i in 0..ndim-1):
    size *= info.output_dim(i)
return size
```

#### `needsPadding(const ConvInfo &info)`
检查是否需要填充（任意维度的 pad > 0）。

#### `fillPaddedInput<Tdata>(info, x, padded_x_shape, padded_x, x_index, padded_x_index, ndim)`
递归填充函数，将原始输入 `x` 拷贝到 `padded_x` 的正确位置，处理任意维度。
- **参数**:
  - `ndim`: 当前递归维度（0=batch, 1=in_channels, 2+=空间维度）
  - `pad_offset`: 空间维度的填充偏移量（仅当 `ndim >= 2` 且 `x_shape != padded_x_shape` 时非零）

#### `_applyConv<Xdata, Ydata>(info, y, x, w, x_shape, x_index, w_index, y_index, ndim)`
核心卷积递归函数，从最内层空间维度开始逐层向外计算。
- **终止条件**: `ndim < 2`（到达 batch 或 channels 维度）
- **递归逻辑**:
  1. 计算当前维度的输出步数：`steps = (dim_size - dilation*(kernel_size-1) - 1) / stride + 1`
  2. 对每个输出位置 `i`（0 到 steps-1）：
     - 对每个卷积核元素 `k`（0 到 kernel_size-1）：
       - 计算输入索引：`curr_x_index = x_index + i*stride + k*dilation`
       - 计算卷积核索引：`curr_w_index = w_index + k`
       - 如果是最内层维度（`ndim == info.ndim() + 1`）：
         - 累加乘积到输出：`y[y_index] += x[curr_x_index] * w[curr_w_index]`
       - 否则递归到下一维度

#### `applyConv<Xdata, Ydata>(info, y, x, w, x_shape)`
并行化的卷积外层函数。
- **并行范围**: `[0, batch_size * out_channels)`，使用动态调度
- **循环体**:
  1. 解析批次索引 `i = iter / out_channels`
  2. 解析输出通道索引 `j = iter % out_channels`
  3. 遍历所有输入通道 `k`（0 到 in_channels-1）：
     - 计算 `x_index = i * in_channels + k`
     - 计算 `w_index = j * in_channels + k`
     - 调用 `_applyConv(..., ndim=2)` 从第一个空间维度开始递归

#### `_conv_cpu<Xdata, Ydata>(info, workspace, workspace_size, y, x, w)`
卷积执行的主函数，处理填充逻辑。
- **逻辑**:
  1. 如果 `needsPadding(info)` 为真：
     - 使用 `workspace` 作为填充缓冲区
     - 用类型安全的零值填充整个缓冲区
     - 调用 `fillPaddedInput()` 拷贝数据
     - 调用 `applyConv(y, padded_x, w, getPaddedShape())`
  2. 否则：
     - 构造原始输入形状数组 `shape = [batch, in_channels, input_dims...]`
     - 调用 `applyConv(y, x, w, shape)`

#### `conv_cpu<Tdata>(info, workspace, workspace_size, y, x, w, bias)`
通用模板函数，支持 F32/F16/BF16。
- **流程**:
  1. 输出张量 `y` 初始化为 0
  2. 调用 `_conv_cpu<Tdata, Tdata>()` 执行卷积
  3. 如果 `bias != nullptr`：
     - 并行化遍历所有输出元素
     - 计算通道索引：`channel_idx = (i / spatial_sizes) % out_channels`
     - 累加偏置：`y[i] += bias[channel_idx]`

#### `conv_cpu<fp16_t>` 和 `conv_cpu<bf16_t>` 特化
针对半精度浮点的特殊处理，避免精度损失。
- **工作空间布局**:
  ```
  +-----------------------+-----------------------+
  | FP32 累加器 (output)  | 填充输入 (optional)   |
  +-----------------------+-----------------------+
  ```
- **流程**:
  1. `y_float = reinterpret_cast<float*>(workspace)`：工作空间前段作为 FP32 累加器
  2. `std::fill(y_float, y_float + output_size, 0.0f)`：初始化为 0
  3. `conv_workspace = y_float + output_size`：工作空间后段用于填充输入
  4. `_conv_cpu<fp16_t/bf16_t, float>(...)`：半精度输入，FP32 计算
  5. 如果 `bias != nullptr`：
     - 并行化遍历所有输出元素
     - 转换偏置：`bias_value = utils::cast<float>(bias_half[channel_idx])`
     - 累加偏置：`y_float[i] += bias_value`
     - 转换输出：`y_half[i] = utils::cast<fp16_t/bf16_t>(y_float[i])`
  6. 否则仅转换输出（不加偏置）

### 性能特征
- **计算复杂度**:
  - 2D 卷积：O(N × O × I × H × W × kH × kW)，其中 N=batch, O=out_ch, I=in_ch
  - 3D 卷积：O(N × O × I × D × H × W × kD × kH × kW)

- **内存访问模式**:
  - 输入：随机访问（受 dilation 和 stride 影响）
  - 卷积核：顺序访问（缓存友好）
  - 输出：顺序写入（缓存友好）

- **并行扩展性**:
  - 理想情况：线性扩展到 `batch_size * out_channels` 个核心
  - 实际限制：内存带宽瓶颈，每个线程的计算量相对较小

### 已知限制
1. **算法性能**: 使用直接卷积算法，未针对大规模卷积进行优化（如 im2col + GEMM 或 Winograd）
2. **缓存效率**: 递归实现可能导致缓存局部性不佳
3. **无数据打包**: 未使用数据打包（如 vNNI、AVX-512）或 SIMD 指令
4. **无 tiling**: 未实现分块优化，无法充分利用 L1/L2 缓存
5. **仅支持 OMP**: 仅支持 OpenMP 并行化，无向量化优化

### 适用场景
- **推荐用途**:
  - 小规模卷积（kernel ≤ 3×3，channels ≤ 64）
  - 验证和调试（实现简单，易于理解）
  - 不具备专用硬件加速器的环境

- **不推荐用途**:
  - 大规模深度学习推理/训练（性能远低于 cuDNN、MKL-DNN 等）
  - 实时应用（延迟较高）

### 未来优化方向
1. 实现 im2col + GEMM 算法（利用高性能矩阵乘法）
2. 添加 SIMD 向量化（AVX2/AVX-512）
3. 实现分块（tiling）优化
4. 支持 Winograd 算法（针对 3×3 卷积）
5. 添加多线程调度优化（如工作窃取）
