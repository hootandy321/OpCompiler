# NVIDIA Layer Norm 算子核心实现文档

本文档详细描述了 Infini 框架中针对 NVIDIA GPU 设备的 Layer Normalization（层归一化）算子实现。该实现提供了高效的 CUDA kernel，支持多种数据类型（FP16、FP32、BF16），并根据特征维度大小自动选择最优的并行策略。

## 1. 模块结构

- **`layer_norm_nvidia.cuh`**: NVIDIA 后端算子描述符声明，定义了 `Descriptor` 类的接口
- **`layer_norm_nvidia.cu`**: NVIDIA CUDA 实现主文件，包含算子描述符实现、内核调度逻辑和 workspace 管理

## 2. 核心类

### `Descriptor`
- **位置**: `layer_norm_nvidia.cuh` (声明), `layer_norm_nvidia.cu` (实现)
- **主要功能**: Layer Norm 算子的 NVIDIA 设备描述符，负责算子初始化、workspace 计算和内核调度
- **关键成员**:
  - `_opaque`: 指向 `Opaque` 结构的指针，封装 NVIDIA 设备句柄的内部状态
  - `_info`: `LayerNormInfo` 对象，存储算子的元数据（形状、步长、归一化维度等）
  - `_workspace_size`: 执行计算所需的工作空间大小（字节）

#### `Opaque` 内部结构
```cpp
struct Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};
```
封装 NVIDIA 设备句柄的内部实现，提供设备能力查询（如最大线程块大小）。

- **核心方法**:

  **`create(handle, desc_ptr, output_desc, input_standardization_desc, input_std_deviation_desc, input_desc, weight_desc, bias_desc, eps)`**
  - 静态工厂方法，创建并初始化 Layer Norm 描述符
  - 参数验证：检查输出、输入、标准化的形状一致性；验证 weight 和 bias 的维度
  - 数据类型支持：FP16、FP32、BF16
  - Workspace 计算：`ndim * (4 * sizeof(ptrdiff_t) + sizeof(size_t))`
  - 创建 `LayerNormInfo` 对象并封装到描述符中
  - 返回：`INFINI_STATUS_SUCCESS` 或错误码

  **`calculate(workspace, workspace_size, output, input_standardization, input_std_deviation, input, weight, bias, stream)`**
  - 执行 Layer Norm 计算
  - Workspace 验证：检查提供的 workspace 大小是否足够
  - 根据设备架构（`maxThreadsPerBlock`）选择线程块大小（1024/512/4096）
  - 根据数据类型分派到相应的模板实例化（`half`/`float`/`__nv_bfloat16`）
  - 调用 `calculate_layer_norm` 函数执行内核启动
  - 返回：`INFINI_STATUS_SUCCESS` 或错误码

- **生命周期**:
  1. 用户调用 `Descriptor::create()` 创建描述符
  2. 描述符持有 `LayerNormInfo` 和 `Opaque` 内部状态
  3. 用户调用 `Descriptor::calculate()` 执行计算
  4. 析构函数释放 `Opaque` 资源

## 3. API 接口

### 算子创建接口

```cpp
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,                          // Infini 运行时句柄
    Descriptor **desc_ptr,                            // 输出：创建的描述符指针
    infiniopTensorDescriptor_t output_desc,           // 输出张量描述符 [batch, ..., normalized_dim]
    infiniopTensorDescriptor_t input_standardization_desc,  // 标准化结果张量 [batch, ..., normalized_dim]
    infiniopTensorDescriptor_t input_std_deviation_desc,    // 标准差张量 [batch, ..., normalized_dim-1]
    infiniopTensorDescriptor_t input_desc,            // 输入张量 [batch, ..., normalized_dim]
    infiniopTensorDescriptor_t weight_desc,           // 可学习权重 γ [normalized_dim]
    infiniopTensorDescriptor_t bias_desc,             // 可学习偏置 β [normalized_dim] (可选，nullptr 表示不存在)
    float eps                                         // 数值稳定性的小常数
);
```

### 计算接口

```cpp
infiniStatus_t Descriptor::calculate(
    void *workspace,                                  // 设备上的工作空间指针
    size_t workspace_size,                            // 工作空间大小（字节）
    void *output,                                     // 输出张量设备指针
    void *input_standardization,                      // 标准化结果设备指针
    void *input_std_deviation,                        // 标准差设备指针
    const void *input,                                // 输入张量设备指针
    const void *weight,                               // 权重设备指针
    const void *bias,                                 // 偏置设备指针（可选）
    void *stream                                      // CUDA 流
) const;
```

## 4. 使用示例

```cpp
// 1. 准备张量描述符
// 输入: [batch_size, seq_len, hidden_dim]
infiniopTensorDescriptor_t input_desc, output_desc;
infiniopTensorDescriptor_t weight_desc, bias_desc;
infiniopTensorDescriptor_t input_standardization_desc, input_std_deviation_desc;

// 2. 创建算子描述符
op::layer_norm::nvidia::Descriptor *layer_norm_desc;
infiniStatus_t status = op::layer_norm::nvidia::Descriptor::create(
    handle, &layer_norm_desc,
    output_desc,
    input_standardization_desc,
    input_std_deviation_desc,
    input_desc,
    weight_desc,
    bias_desc,
    1e-5f  // eps
);

// 3. 分配 workspace
size_t workspace_size = layer_norm_desc->workspaceSize();
void *workspace;
cudaMalloc(&workspace, workspace_size);

// 4. 执行计算
cudaStream_t stream;
cudaStreamCreate(&stream);

status = layer_norm_desc->calculate(
    workspace, workspace_size,
    d_output, d_input_standardization, d_input_std_deviation,
    d_input, d_weight, d_bias,
    stream
);

// 5. 清理
cudaFree(workspace);
delete layer_norm_desc;
cudaStreamDestroy(stream);
```

## 5. 实现细节

### 5.1 内存管理

**Workspace 组织**:
工作空间用于在内核执行期间存储设备端元数据（步长和形状），避免在内核中重复计算地址。

布局（从基址开始）：
- `input_strides_cuda`: `ndim * sizeof(ptrdiff_t)` 字节
- `output_strides_cuda`: `ndim * sizeof(ptrdiff_t)` 字节
- `input_standardization_strides_cuda`: `(ndim - 1) * sizeof(ptrdiff_t)` 字节
- `input_std_deviation_strides_cuda`: `(ndim - 1) * sizeof(ptrdiff_t)` 字节
- `shape_cuda`: `ndim * sizeof(size_t)` 字节

总大小：`ndim * (4 * sizeof(ptrdiff_t) + sizeof(size_t))` 字节

**数据传输**:
使用 `cudaMemcpyAsync` 异步传输步长和形状数据到设备，以最小化主机-设备同步开销。

### 5.2 并行策略

实现根据归一化维度（`normalized_size`）大小自动选择最优的 kernel 实现：

**策略 1: Warp-Level 并行** (`dimsize <= 1024`)
- **Kernel**: `warpLayernorm<T, BLOCK_SIZE_x, BLOCK_SIZE_y>`
- **线程块配置**: `dim3(32, 32, 1)` = 1024 线程/块
- **并行模式**:
  - Y 维度（`threadIdx.y`）：处理不同的归一化组（batch 中不同的样本）
  - X 维度（`threadIdx.x`）：在同一组内并行处理特征元素
- **归约策略**: 使用 `WarpAllReduce` 通过 `__shfl_xor_sync` 在 warp 内进行归约
- **Shared Memory**: 存储中间结果（均值 `mu` 和标准差倒数 `sigma2`），每个 Y 线程一个值
- **优势**: 当归一化维度较小时，多个样本可以并行处理，提高设备利用率

**策略 2: Block-Level 并行** (`dimsize > 1024`)
- **Kernel**: `blockLayernorm<T, BLOCK_SIZE>`
- **线程块配置**: `BLOCK_SIZE` 线程/块（1024/512/4096，取决于设备架构）
- **并行模式**: 每个线程块处理一个归一化组
- **归约策略**: 使用 CUB 的 `BlockReduce` 在线程块内进行归约
- **Shared Memory**: 存储 CUB 的临时存储和全局均值/标准差
- **优势**: 当归一化维度较大时，充分利用线程块内的并行性

### 5.3 性能优化

**数值稳定性**:
- 方差计算采用两遍算法：先计算平方和，再计算 `var = sum_squared / n - mean^2`
- 标准差计算添加 epsilon: `std_dev = sqrt(var + eps)`
- 在 kernel 中使用 `__fdividef(x, y)` 内置函数进行快速除法

**内存访问优化**:
- 所有张量访问通过步长计算，支持非连续张量
- 输入/输出指针在 kernel 开始时计算一次，避免重复计算
- 权重和偏置的访问通过 `weight_stride` 和 `bias_stride` 参数化

**归约优化**:
- 使用 CUB 库的高度优化块级归约原语
- Warp-level 归约使用 `__shfl_xor_sync` 避免共享内存访问
- 两阶段归约：首先每个线程计算部分和，然后通过归约操作合并

**设备适配**:
- 根据设备能力（`maxThreadsPerBlock`）选择线程块大小
- 支持 CUDA_BLOCK_SIZE_1024、CUDA_BLOCK_SIZE_512、CUDA_BLOCK_SIZE_4096
- 自动处理不同架构的线程数限制

### 5.4 错误处理

**数据类型检查**:
```cpp
CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
```
只支持 FP16、FP32、BF16 三种数据类型。

**张量形状验证**（在 `LayerNormInfo::createLayerNormInfo` 中）:
- 输出、输入、标准化的形状必须完全一致
- Weight 和 bias 必须是 1D 张量，长度等于归一化维度
- 标准差张量的维度必须是 `ndim - 1`（最后一个维度被归约）

**Workspace 验证**:
在 `calculate()` 中检查提供的 workspace 大小是否足够，不足时返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`。

**设备架构支持**:
如果不支持的设备架构（无法匹配 1024/512/4096 线程块配置），返回 `INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED`。

### 5.5 依赖关系

**内部依赖**:
- `../layer_norm.h`: Layer Norm 算子的通用定义和宏
- `../info.h`: `LayerNormInfo` 类定义
- `../cuda/kernel.cuh`: CUDA kernel 实现模板
- `../../../reduce/cuda/reduce.cuh`: 归约操作 CUDA 实现（`sum`、`sumSquared`）
- `../../../devices/nvidia/nvidia_common.cuh`: NVIDIA 通用 CUDA 定义
- `../../../devices/nvidia/nvidia_handle.cuh`: NVIDIA 设备句柄实现
- `../../../devices/nvidia/nvidia_kernel_common.cuh`: NVIDIA kernel 通用宏和工具

**外部依赖**:
- CUB 库（CUDA 原语）：`<cub/block/block_reduce.cuh>` 用于高效的块级归约
- CUDA Runtime: `cudaMemcpyAsync`、`cudaStream_t` 等

### 5.6 设计模式

**策略模式 (Strategy Pattern)**:
根据归一化维度大小（`dimsize`）动态选择 kernel 实现（warp-level vs block-level），优化不同场景下的性能。

**模板方法模式 (Template Method Pattern)**:
使用 C++ 模板实现类型通用和数据类型多态，避免为每种数据类型重复编写相同逻辑。

**Pimpl 惯用法 (Pointer to Implementation)**:
`Descriptor` 类通过 `_opaque` 指针封装 NVIDIA 特定的实现细节，隐藏设备句柄的内部状态。

**RAII (Resource Acquisition Is Initialization)**:
`Opaque` 成员使用 `std::shared_ptr` 管理设备句柄的内部状态，确保资源正确释放。

## 6. 算法细节

### Layer Norm 数学公式

对于输入张量 `x`，在最后一个维度上进行归一化：

```
μ = (1/n) * Σ(x[i])                    # 均值
σ² = (1/n) * Σ((x[i] - μ)²)            # 方差
x̂[i] = (x[i] - μ) / √(σ² + eps)        # 标准化
y[i] = γ[i] * x̂[i] + β[i]              # 仿射变换
```

其中：
- `n`: 归一化维度大小（`normalized_size`）
- `γ`: 可学习的权重参数（weight）
- `β`: 可学习的偏置参数（bias，可选）
- `eps`: 数值稳定性常数（通常 1e-5）

### Kernel 计算流程

**Warp-Level Kernel** (`warpLayernormKernel`):
1. 每个线程块处理多行数据（`BLOCK_SIZE_y = 32` 行）
2. 对每一行：
   a. 线程协作计算输入的和（使用 warp shuffle 归约）
   b. 计算均值 `mu = sum / dimsize`
   c. 线程协作计算方差（使用 warp shuffle 归约）
   d. 计算标准差倒数 `sigma2 = 1 / sqrt(var + eps)`
   e. 并行应用仿射变换到每个元素

**Block-Level Kernel** (`blockLayernormKernel`):
1. 每个线程块处理一行数据
2. 对该行：
   a. 使用 CUB `BlockReduce` 计算输入的和
   b. 线程 0 计算均值并存储到共享内存
   c. 使用 CUB `BlockReduce` 计算方差
   d. 线程 0 计算标准差倒数并存储到共享内存
   e. 所有线程并行应用仿射变换到分配的元素

### 复杂度分析

- **时间复杂度**:
  - 均值计算: O(normalized_size)（每个元素访问一次）
  - 方差计算: O(normalized_size)（每个元素访问一次）
  - 仿射变换: O(normalized_size)（每个元素访问一次）
  - 总计: O(normalized_size) per 归一化组

- **空间复杂度**:
  - 额外共享内存: O(BLOCK_SIZE_y) 用于存储中间结果（warp kernel）
  - Workspace: O(ndim) 用于存储元数据

- **并行度**:
  - Warp kernel: min(othersize, 32) 组同时处理（每个线程块处理 32 行）
  - Block kernel: othersize 组同时处理（每个线程块处理 1 行）

## 7. 支持的数据类型

| 数据类型 | C++ 类型 | 说明 |
|---------|---------|------|
| FP16 | `half` | 半精度浮点数，适合推理和训练加速 |
| FP32 | `float` | 单精度浮点数，标准精度 |
| BF16 | `__nv_bfloat16` | Brain 浮点数，范围与 FP32 相同但精度较低，适合混合精度训练 |

所有计算在内部使用 `float` 进行累加以确保数值精度，最终结果转换回目标数据类型。
