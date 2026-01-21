# TopkSoftmax NVIDIA CUDA Core Implementation Documentation

该模块实现了基于 NVIDIA CUDA 的 TopkSoftmax 算子，用于在 MoE (Mixture of Experts) 模型中计算 softmax 概率分布并提取 top-k 专家。该实现针对 CUDA 架构进行了深度优化，使用 CUB 库实现高效的并行规约和排序操作。

## 1. Module Structure

- **`topksoftmax_nvidia.cuh`**: NVIDIA 实现的头文件声明，通过宏定义生成 `Descriptor` 类
- **`topksoftmax_nvidia.cu`**: NVIDIA CUDA 实现的核心源文件，包含算子创建、核函数调度和计算执行逻辑

## 2. Core Classes

### `Descriptor::Opaque`
- **Location**: `topksoftmax_nvidia.cu`
- **Primary Function**: 封装 NVIDIA 设备句柄的内部状态，持有 CUDA 设备句柄的共享指针
- **Key Members**:
  - `internal`: `std::shared_ptr<device::nvidia::Handle::Internal>` - NVIDIA CUDA 设备句柄的内部实现指针
- **Lifecycle**: 由 `Descriptor::create` 构造，在 `Descriptor` 析构时释放

### `Descriptor`
- **Location**: `topksoftmax_nvidia.cuh` (通过 `DESCRIPTOR(nvidia)` 宏生成), `topksoftmax_nvidia.cu`
- **Primary Function**: TopkSoftmax 算子的 NVIDIA CUDA 后端描述符，管理算子生命周期并提供计算接口
- **Key Members**:
  - `_opaque`: `Opaque *` - NVIDIA 设备句柄的内部状态指针
  - `_info`: `TopksoftmaxInfo` - 输入张量的元数据信息（数据类型、形状、步长等）
  - `_workspace_size`: `size_t` - 工作区大小（当前为 0）
- **Core Methods**:
  - `~Descriptor()`: 析构函数，释放 `_opaque` 指针
  - `create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t x_desc)`: 静态工厂方法，创建描述符实例。验证输入张量的步长（要求 `x_strides[1] == 1`，即最后一维连续），初始化设备句柄和元数据，返回 `INFINI_STATUS_SUCCESS` 或错误码
  - `calculate(void *workspace, size_t workspace_size, float *values, int *indices, const void *x, const size_t topk, const bool norm, void *stream) const`: 执行 TopkSoftmax 计算。根据输入宽度选择最优的 CUDA Block Size（128/256/512），调度对应的核函数
- **Lifecycle**: 单例模式，通过 `create` 工厂方法构造，用户负责销毁

## 3. API Interface

```cpp
// 创建 TopkSoftmax 描述符（NVIDIA CUDA 后端）
// 参数:
//   handle - Infini 运行时句柄
//   desc_ptr - 输出参数，指向新创建的描述符指针
//   x_desc - 输入张量描述符 [N, width]，必须是连续内存（最后一维步长为 1）
// 返回: infiniStatus_t 状态码（SUCCESS / BAD_TENSOR_STRIDES）
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc);

// 执行 TopkSoftmax 计算
// 参数:
//   workspace - 工作区指针（当前未使用，传 nullptr 即可）
//   workspace_size - 工作区大小（必须 >= _workspace_size）
//   values - 输出 top-k 值 [N, topk]，float 类型
//   indices - 输出 top-k 索引 [N, topk]，int 类型
//   x - 输入数据 [N, width]，支持 F32/F16/BF16
//   topk - 提取的前 k 个元素数量
//   norm - 是否对 top-k 值进行归一化（使其和为 1）
//   stream - CUDA 流
// 返回: infiniStatus_t 状态码
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    float *values,
    int *indices,
    const void *x,
    const size_t topk,
    const bool norm,
    void *stream) const;
```

## 4. Usage Example

```cpp
// 示例：在 MoE 模型中使用 TopkSoftmax 选择 top-k 专家

// 1. 准备输入数据
const size_t num_tokens = 1024;    // N: token 数量
const size_t num_experts = 256;    // width: 专家数量
const size_t top_k = 8;            // 选择前 8 个专家

// 输入 logits: [num_tokens, num_experts]
float* d_logits;  // CUDA 设备内存

// 输出缓冲区
float* d_topk_values;   // [num_tokens, top_k]
int* d_topk_indices;    // [num_tokens, top_k]

// 2. 创建张量描述符
std::vector<size_t> shape = {num_tokens, num_experts};
std::vector<ptrdiff_t> strides = {num_experts, 1};  // 最后一维必须连续
infiniopTensorDescriptor_t x_desc = new TensorDescriptor(
    INFINI_DTYPE_F32, shape, strides, 0, 0);

// 3. 创建 TopkSoftmax 描述符
op::topksoftmax::nvidia::Descriptor* topk_desc = nullptr;
infiniStatus_t status = op::topksoftmax::nvidia::Descriptor::create(
    handle, &topk_desc, x_desc);

if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 4. 获取 CUDA 流
cudaStream_t stream;
cudaStreamCreate(&stream);

// 5. 执行 TopkSoftmax 计算
// 对每个 token 的专家 logits 做 softmax，然后选择 top-8 专家
// norm=true 表示对 top-8 个专家的 softmax 值进行归一化（使其和为 1）
status = topk_desc->calculate(
    nullptr,           // workspace (当前未使用)
    0,                 // workspace_size
    d_topk_values,     // 输出: top-k softmax 值
    d_topk_indices,    // 输出: top-k 专家索引
    d_logits,          // 输入: 原始 logits
    top_k,             // k=8
    true,              // 归一化 top-k 值
    stream             // CUDA 流
);

// 6. 使用结果进行专家路由
// d_topk_indices[i] 存储第 i 个 token 的 top-8 专家索引
// d_topk_values[i] 存储第 i 个 token 的 top-8 专家权重（已归一化）

// 7. 清理资源
delete topk_desc;
cudaStreamDestroy(stream);
```

## 5. Implementation Details

### 算法流程（7 阶段 CUDA Kernel）

**核心 Kernel 函数**: `softmax_topk_row_kernel<T, BLOCK_SIZE>`

每个 CUDA Block 处理输入张量的一行（一个 token 的所有专家 logits）：

1. **计算最大值**（数值稳定性）:
   - 使用 `cub::BlockReduce` 在 Block 内规约求最大值
   - 防止 `exp` 溢出：`max = max(input)`

2. **计算指数和**（Softmax 分母）:
   - 每个线程计算 `exp(input[i] - max)`
   - 使用 `cub::BlockReduce` 规约求和得到 `sum_exp`

3. **计算 Softmax**:
   - 每个元素归一化：`softmax[i] = exp(input[i] - max) / sum_exp`

4. **块内排序**（Top-K 选择）:
   - 使用 `cub::BlockRadixSort` 对整个 Block 的 softmax 值进行降序排序
   - 同时排序对应的索引

5. **Top-K 求和**（可选归一化）:
   - 仅第一个 Warp（warp_id == 0）参与
   - 使用 `cub::WarpReduce` 对前 k 个值求和

6. **归一化**（可选）:
   - 如果 `norm == true`，将 top-k 值除以它们的和：`value /= (sum_topk + 1e-9f)`
   - 确保 top-k 权重和为 1

7. **写入结果**:
   - 仅第一个 Warp 的前 k 个线程将值和索引写入输出

### 内存管理

- **工作区**: 当前未使用，`_workspace_size = 0`。所有计算都在核函数内完成，无需额外临时内存
- **共享内存**: 依赖 CUB 库自动管理，用于 BlockReduce、BlockRadixSort、WarpReduce 的临时存储
- **全局内存**: 输入/输出数据必须位于 CUDA 设备内存

### 并发策略

- **CUDA Kernel 启动配置**:
  - Grid: `(N)` - 每行一个 Block
  - Block: `(BLOCK_SIZE)` - 根据 width 自适应选择（128/256/512）
- **Block Size 选择策略**:
  - `width <= 128`: BLOCK_SIZE = 128
  - `width <= 256`: BLOCK_SIZE = 256
  - `width <= 512`: BLOCK_SIZE = 512
  - `width > 512`: 返回错误（当前不支持）
- **线程协作**:
  - 所有线程参与 Block 级规约和排序
  - 仅第一个 Warp（32 个线程）参与结果收集和输出
- **同步点**:
  - `__syncthreads()`: 确保 Block 内所有线程完成规约后再使用共享变量
  - `__syncwarp()`: 确保 Warp 内线程完成求和

### 性能优化

- **数值稳定性**: 使用最大值归一化技巧避免 `exp` 溢出
- **自适应 Block Size**: 根据输入宽度选择最优 Block Size，最大化寄存器和共享内存利用率
- **CUB 原语**: 使用 NVIDIA CUB 库的高性能原语（BlockReduce、BlockRadixSort、WarpReduce），这些库针对 CUDA 架构进行了极致优化
- **最小化全局内存访问**: 每行数据仅读取一次，所有计算在寄存器/共享内存中完成
- **Warp 优化**: 最后阶段仅使用第一个 Warp，减少空闲线程

### 数据类型支持

- **输入**: `float` (F32), `half` (F16), `__nv_bfloat16` (BF16)
- **输出**:
  - `values`: 始终为 `float`（Softmax 结果）
  - `indices`: 始终为 `int`
- **类型转换**:
  - F16/BF16 在 `exp_func` 中转换为 float 进行计算
  - 使用 CUDA 内置函数：`__half2float`, `__bfloat162float`

### 错误处理

- **Stride 验证**: 要求 `x_strides[1] == 1`（最后一维连续），否则返回 `INFINI_STATUS_BAD_TENSOR_STRIDES`
- **数据类型验证**: 仅支持 F32/F16/BF16，否则返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
- **工作区检查**: 要求 `workspace_size >= _workspace_size`，否则返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
- **宽度限制**: `width` 必须 <= 512，否则返回 `INFINI_STATUS_INTERNAL_ERROR`

### 依赖关系

- **外部依赖**:
  - `../../../devices/nvidia/nvidia_common.cuh`: NVIDIA 设备通用定义
  - `../../../devices/nvidia/nvidia_kernel_common.cuh`: CUDA Kernel 通用工具
  - `../../../reduce/cuda/reduce.cuh`: 规约操作（当前未直接使用）
- **上层依赖**:
  - `../topksoftmax.h`: 算子接口定义（宏生成 Descriptor 类）
  - `../cuda/kernel.cuh`: CUDA Kernel 实现（`softmax_topk_row_kernel`）
  - `../info.h`: `TopksoftmaxInfo` 元数据类
- **CUDA 库**:
  - `cub/block/block_reduce.cuh`: Block 级规约
  - `cub/block/block_radix_sort.cuh`: Block 级基数排序
  - `cub/block/block_load.cuh`, `cub/block/block_store.cuh`: 块加载/存储（隐式使用）

### 设计模式

- **工厂模式**: `Descriptor::create` 静态工厂方法负责对象构造
- **宏元编程**: 使用 `DESCRIPTOR(nvidia)` 宏在编译期生成 `Descriptor` 类定义
- **策略模式**: 根据 `width` 动态选择 Block Size（128/256/512）
- **模板特化**: Kernel 函数使用模板参数支持多种数据类型（float/half/__nv_bfloat16）
- **Pimpl 惯用法**: `Descriptor::Opaque` 封装 NVIDIA 设备句柄的内部实现细节
