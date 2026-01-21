# Add RMS Norm NVIDIA 算子核心实现文档

本模块实现了 Add RMS Norm（残差连接 + RMS 归一化）融合算子在 NVIDIA GPU 上的高性能 CUDA 实现。该算子将两个输入张量相加后进行 RMS 归一化操作，是 Transformer 模型（特别是 GPT-2、BLOOM 等架构）中的核心计算单元。

## 1. 模块结构

- **`add_rms_norm_nvidia.cuh`**: NVIDIA 后端的描述符声明文件，定义 `op::add_rms_norm::nvidia::Descriptor` 类
- **`add_rms_norm_nvidia.cu`**: NVIDIA CUDA 实现主文件，包含核函数启动逻辑、类型分派和描述符方法实现

**依赖文件**:
- **`../add_rms_norm.h`**: 算子接口定义，使用宏 `DESCRIPTOR(NAMESPACE)` 生成各后端描述符类
- **`../cuda/kernel.cuh`**: CUDA 核心计算逻辑 `add_rmsnormBlock`，实现单 block 内的融合计算
- **`../info.h`**: `AddRMSNormInfo` 类，定义算子的元数据（shape、stride、数据类型、epsilon 等）
- **`../../devices/nvidia/nvidia_kernel_common.cuh`**: CUDA 常量（`CUDA_BLOCK_SIZE_*`）和工具函数
- **`../../../reduce/cuda/reduce.cuh`**: CUDA reduce 操作的通用实现
- **`../../../devices/nvidia/nvidia_common.cuh`**: NVIDIA 设备通用定义

## 2. 核心类与数据结构

### `op::add_rms_norm::nvidia::Descriptor`

**位置**: `add_rms_norm_nvidia.cu` (通过宏定义在 `add_rms_norm.h` 中生成)

**主要功能**: 封装 Add RMS Norm 算子在 NVIDIA GPU 上的执行配置和状态

**关键成员**:
- `Opaque *_opaque`: 不透明指针，指向 NVIDIA 设备相关的内部状态
  - `std::shared_ptr<device::nvidia::Handle::Internal> internal`: NVIDIA 设备句柄的内部实现，包含设备能力信息（如 maxThreadsPerBlock）
- `AddRMSNormInfo _info`: 算子元数据
  - `infiniDtype_t atype`: 激活值数据类型（a, b, y, residual_out 的类型）
  - `infiniDtype_t wtype`: 权重数据类型
  - `float epsilon`: RMS 计算的数值稳定性常数
  - `std::vector<size_t> shape`: 输出张量的形状（支持 2D 或 3D）
  - `std::vector<ptrdiff_t> y_strides, a_strides, b_strides, residual_out_strides`: 各张量的步长信息
- `size_t _workspace_size`: 所需工作空间大小（当前为 0）

**核心方法**:

#### `Descriptor::create()`
```cpp
static infiniStatus_t create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t weight_desc,
    float epsilon,
    infiniopTensorDescriptor_t residual_out_desc);
```
**功能**: 创建描述符实例，验证输入张量参数
- **验证逻辑**:
  - 检查 a, b, y 的数据类型一致
  - 支持的类型组合:
    - FP16 激活值 + FP16/BF16/FP32 权重
    - BF16 激活值 + BF16/FP16/FP32 权重
    - FP32 激活值 + FP32 权重
  - 形状支持: `(batch, dim)` 或 `(batch, nhead, dim)`
  - 要求最后一维连续（stride 为 1）
  - 强制要求 `residual_out_desc` 非空
- **返回**: `INFINI_STATUS_SUCCESS` 或相应的错误码

#### `Descriptor::calculate()`
```cpp
infiniStatus_t calculate(
    void *workspace, size_t workspace_size,
    void *y, const void *a, const void *b, const void *weight,
    void *residual_out, void *stream) const;
```
**功能**: 执行 Add RMS Norm 计算
- **工作流**:
  1. 检查 workspace_size >= _workspace_size
  2. 提取张量的 stride 信息
  3. 根据 GPU 能力选择 block size (512/1024/4096)
  4. 调用 `launchKernel<BLOCK_SIZE>()` 启动 CUDA 核函数
- **Kernel 配置**:
  - Grid size: `batch_size * nhead`（每个 head 启动一个 block）
  - Block size: 根据 GPU 架构选择（512/1024/4096）

### `Descriptor::Opaque`
**位置**: `add_rms_norm_nvidia.cu:40-42`
```cpp
struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};
```
**功能**: 封装 NVIDIA 设备相关的内部状态，提供设备能力查询（如 `maxThreadsPerBlock()`）

## 3. CUDA 核心实现

### `add_rmsnormKernel` (全局核函数)
**位置**: `add_rms_norm_nvidia.cu:11-36`
```cpp
template <unsigned int BLOCK_SIZE, typename Tcompute, typename Tdata, typename Tweight>
INFINIOP_CUDA_KERNEL add_rmsnormKernel(
    Tdata *__restrict__ y,
    Tdata *__restrict__ residual_out,
    ptrdiff_t stride_y_batch, ptrdiff_t stride_y_nhead,
    ptrdiff_t stride_residual_out_batch, ptrdiff_t stride_residual_out_nhead,
    const Tdata *__restrict__ a,
    ptrdiff_t stride_a_batch, ptrdiff_t stride_a_nhead,
    const Tdata *__restrict__ b,
    ptrdiff_t stride_b_batch, ptrdiff_t stride_b_nhead,
    const Tweight *__restrict__ w,
    size_t nhead, size_t dim, float epsilon);
```
**功能**: CUDA 全局核函数入口，直接调用 `add_rmsnormBlock` 执行计算
- **参数**: 所有的输入/输出张量指针和 stride
- **类型参数**:
  - `BLOCK_SIZE`: 线程块大小（编译时常量）
  - `Tcompute`: 计算类型（通常为 float，确保精度）
  - `Tdata`: 激活值类型（half/__nv_bfloat16/float）
  - `Tweight`: 权重类型（half/__nv_bfloat16/float）

### `add_rmsnormBlock` (设备函数)
**位置**: `../cuda/kernel.cuh:6-61`
```cpp
template <unsigned int BLOCK_SIZE, typename Tcompute, typename Tdata, typename Tweight>
__device__ void add_rmsnormBlock(
    Tdata *__restrict__ y,
    Tdata *__restrict__ residual_out,
    ptrdiff_t stride_y_batch, ptrdiff_t stride_y_nhead,
    ptrdiff_t stride_residual_out_batch, ptrdiff_t stride_residual_out_nhead,
    const Tdata *__restrict__ a,
    ptrdiff_t stride_a_batch, ptrdiff_t stride_a_nhead,
    const Tdata *__restrict__ b,
    ptrdiff_t stride_b_batch, ptrdiff_t stride_b_nhead,
    const Tweight *__restrict__ w,
    size_t nhead, size_t dim, float epsilon);
```
**功能**: 在单个 CUDA block 内执行完整的 Add RMS Norm 计算
- **计算流程**:
  1. **索引计算** (第 26-33 行):
     - 从 `blockIdx.x` 解析 `batch_idx` 和 `head_idx`
     - 计算各张量的起始指针

  2. **第一阶段：Add + 平方和计算** (第 36-41 行):
     ```cpp
     Tcompute sum_squared = 0;
     for (size_t i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
         Tcompute sum_val = Tcompute(a_ptr[i]) + Tcompute(b_ptr[i]);
         residual_out_ptr[i] = Tdata(sum_val); // 存储 a+b 结果
         sum_squared += sum_val * sum_val;      // 累加平方和
     }
     ```
     - 每个线程处理 `BLOCK_SIZE` 步长的元素
     - 同时计算 `a + b` 和 `(a + b)^2` 的累加和
     - 将 `a + b` 结果写入 `residual_out`

  3. **Block 级 Reduce** (第 43-46 行):
     ```cpp
     using BlockReduce = cub::BlockReduce<Tcompute, BLOCK_SIZE>;
     __shared__ typename BlockReduce::TempStorage temp_storage;
     sum_squared = BlockReduce(temp_storage).Sum(sum_squared);
     ```
     - 使用 CUB 库的 `BlockReduce` 进行 warp 级和 block 级归约
     - 时间复杂度: O(log BLOCK_SIZE)

  4. **RMS 计算** (第 48-53 行):
     ```cpp
     __shared__ Tcompute rms;
     if (threadIdx.x == 0) {
         rms = Tcompute(rsqrtf(sum_squared / Tcompute(dim) + epsilon));
     }
     __syncthreads();
     ```
     - 线程 0 计算 RMS = `1 / sqrt(mean(square_sum) + epsilon)`
     - 存储到 shared memory 供所有线程访问

  5. **第二阶段：归一化** (第 55-60 行):
     ```cpp
     for (size_t i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
         Tcompute sum_val = Tcompute(residual_out_ptr[i]); // 重用存储的 a+b
         y_ptr[i] = Tdata(sum_val * Tcompute(w_ptr[i]) * rms);
     }
     ```
     - 重用 `residual_out` 中的 `a + b` 结果
     - 计算 `y = (a + b) * w * rms`

**优化技术**:
- **融合计算**: Add + RMS Norm 单次 kernel 启动完成
- **数据重用**: `residual_out` 作为中间结果存储，避免重复计算 `a + b`
- **Block 并行**: 每个 head 独立处理，无跨 block 同步
- **类型提升**: 使用 `Tcompute=float` 确保数值稳定性

## 4. 类型分派与 Kernel 启动

### `launchKernel()` 模板函数
**位置**: `add_rms_norm_nvidia.cu:70-121`
```cpp
template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(
    uint32_t batch_size, size_t nhead, size_t dim,
    void *y, infiniDtype_t atype, ptrdiff_t stride_y_batch, ptrdiff_t stride_y_nhead,
    void *residual_out, ptrdiff_t stride_residual_out_batch, ptrdiff_t stride_residual_out_nhead,
    const void *a, ptrdiff_t stride_a_batch, ptrdiff_t stride_a_nhead,
    const void *b, ptrdiff_t stride_b_batch, ptrdiff_t stride_b_nhead,
    const void *w, infiniDtype_t wtype,
    float epsilon, cudaStream_t cuda_stream);
```
**功能**: 根据数据类型实例化并启动对应的 CUDA 核函数
- **支持的类型组合**:
  | 激活值类型 (atype) | 权重类型 (wtype) | 计算类型 (Tcompute) |
  |-------------------|------------------|---------------------|
  | FP16              | FP16             | float               |
  | FP16              | BF16             | float               |
  | FP16              | FP32             | float               |
  | BF16              | BF16             | float               |
  | BF16              | FP16             | float               |
  | BF16              | FP32             | float               |
  | FP32              | FP32             | float               |

- **宏展开**: `LAUNCH_KERNEL(Tdata, Tweight, Tcompute)` 将类型参数展开为完整的 kernel 启动语句
- **Kernel 配置**:
  ```cpp
  add_rmsnormKernel<BLOCK_SIZE, Tcompute, Tdata, Tweight>
      <<<batch_size * nhead, BLOCK_SIZE, 0, cuda_stream>>>(/* args */);
  ```
  - Grid: `batch_size * nhead` 个 block
  - Block: `BLOCK_SIZE` 个线程

### `Descriptor::calculate()` 中的 Block Size 选择
**位置**: `add_rms_norm_nvidia.cu:146-172`
```cpp
if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
    CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_1024>(...));
} else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
    CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_512>(...));
} else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
    CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_4096>(...));
} else {
    return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
}
```
**功能**: 运行时根据 GPU 计算能力选择最优 block size
- **支持的 Block Size**:
  - 512: 适用于较老的 GPU（如 Kepler、Maxwell）
  - 1024: 适用于现代 GPU（如 Pascal、Volta、Turing、Ampere）
  - 4096: 适用于高吞吐 GPU（如 Hopper）

## 5. API 使用示例

```cpp
#include "infinicore.h"
#include "infiniop/operator.h"

// 1. 创建句柄
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_NVIDIA, 0);

// 2. 准备张量描述符（假设 batch=2, nhead=4, dim=128）
std::vector<size_t> shape = {2, 4, 128};
std::vector<ptrdiff_t> strides = {4 * 128, 128, 1}; // 连续内存

infiniopTensorDescriptor_t y_desc, a_desc, b_desc, w_desc, residual_out_desc;
infiniopCreateTensorDescriptor(&y_desc, INFINI_DTYPE_F16, shape, strides);
infiniopCreateTensorDescriptor(&a_desc, INFINI_DTYPE_F16, shape, strides);
infiniopCreateTensorDescriptor(&b_desc, INFINI_DTYPE_F16, shape, strides);
infiniopCreateTensorDescriptor(&w_desc, INFINI_DTYPE_F16, {128}, {1});
infiniopCreateTensorDescriptor(&residual_out_desc, INFINI_DTYPE_F16, shape, strides);

// 3. 创建算子描述符
op::add_rms_norm::nvidia::Descriptor *add_rms_norm_desc;
float epsilon = 1e-5f;
auto status = op::add_rms_norm::nvidia::Descriptor::create(
    handle, &add_rms_norm_desc,
    y_desc, a_desc, b_desc, w_desc, epsilon, residual_out_desc);

// 4. 分配内存并初始化数据
size_t elem_count = 2 * 4 * 128;
half *a, *b, *w, *y, *residual_out;
cudaMalloc(&a, elem_count * sizeof(half));
cudaMalloc(&b, elem_count * sizeof(half));
cudaMalloc(&w, 128 * sizeof(half));
cudaMalloc(&y, elem_count * sizeof(half));
cudaMalloc(&residual_out, elem_count * sizeof(half));
// ... 初始化输入数据 ...

// 5. 创建 CUDA stream
cudaStream_t stream;
cudaStreamCreate(&stream);

// 6. 执行计算
size_t workspace_size = add_rms_norm_desc->workspaceSize();
void *workspace = nullptr; // 本算不需要 workspace
status = add_rms_norm_desc->calculate(
    workspace, workspace_size,
    y, a, b, w, residual_out, stream);

// 7. 同步并清理
cudaStreamSynchronize(stream);
// ... 使用结果 ...
delete add_rms_norm_desc;
cudaFree(a); cudaFree(b); cudaFree(w); cudaFree(y); cudaFree(residual_out);
```

## 6. 实现细节

### 内存管理
- **工作空间**: 本算子不需要额外工作空间（`_workspace_size = 0`）
- **Shared Memory**: 每个 block 使用 `sizeof(typename BlockReduce::TempStorage)` 字节的 shared memory（通常为 512-1024 字节）
- **Global Memory**: 直接读写输入/输出张量，无中间缓冲

### 并发策略
- **Block 级并行**: 每个 `(batch, head)` 对独立分配一个 CUDA block
- **Thread 级并行**: Block 内线程使用 stride 循环（`i += BLOCK_SIZE`）协同处理 dim 维度
- **Warp 级优化**: CUB 的 `BlockReduce` 自动使用 warp shuffle 指令进行高效归约
- **流处理**: 支持异步执行，通过 `cudaStream_t` 参数支持多 stream 并发

### 性能优化技术
1. **融合 Kernel**:
   - Add、平方和计算、归一化融合为单次 kernel 启动
   - 减少全局内存访问（`a + b` 结果只写一次到 `residual_out`）

2. **类型策略**:
   - 计算类型固定为 `float`，确保数值稳定性
   - 支持混合精度（如 FP16 激活 + FP32 权重）
   - 避免半精度累加导致的精度损失

3. **数据重用**:
   - `residual_out` 同时作为中间结果存储和最终输出
   - 第二阶段直接从 `residual_out` 读取 `a + b`，避免重复计算

4. **Shared Memory 优化**:
   - 仅存储 RMS 标量（4 字节）
   - CUB 的 `TempStorage` 自动优化 shared memory 使用

5. **Block Size 自适应**:
   - 根据 GPU 能力选择最优 block size
   - 兼顾并行度和资源利用率

### 数值稳定性
- **Epsilon 参数**: `epsilon` 防止除零错误（默认 `1e-5`）
- **精度提升**:
  - 使用 `rsqrtf()` 计算 `1 / sqrt(x)`
  - 平方和累加使用 `float` 而非半精度
  - 避免下溢/上溢问题

### 错误处理
- **输入验证**:
  - 检查数据类型兼容性（`INFINI_STATUS_BAD_TENSOR_DTYPE`）
  - 验证形状匹配（`INFINI_STATUS_BAD_TENSOR_SHAPE`）
  - 检查 stride 合法性（`INFINI_STATUS_BAD_TENSOR_STRIDES`）
  - 强制 `residual_out` 非空（`INFINI_STATUS_BAD_PARAM`）

- **运行时检查**:
  - Workspace 大小验证（`INFINI_STATUS_INSUFFICIENT_WORKSPACE`）
  - GPU 架构不支持时返回 `INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED`

- **宏辅助**: `CHECK_STATUS()` 和 `CHECK_RESULT()` 用于错误传播

### 依赖关系
- **外部依赖**:
  - **CUDA Toolkit**: 核心并行计算框架
  - **CUB 库**: 提供 `BlockReduce` 原语（CUDA 包含）

- **内部依赖**:
  - `InfiniopDescriptor`: 基类接口
  - `device::nvidia::Handle`: NVIDIA 设备管理
  - `AddRMSNormInfo`: 算子元数据验证和存储
  - `infinicore.h`: 核心类型定义和常量

### 设计模式
- **策略模式**: 根据 GPU 能力选择 block size
- **模板元编程**: 编译期生成所有类型组合的核函数
- **RAII**: 描述符使用析构函数释放 `_opaque`
- **Pimpl 惯例**: `Opaque` 结构体隐藏 NVIDIA 实现细节

### 限制与约束
1. **维度约束**: 仅支持 2D `(batch, dim)` 或 3D `(batch, nhead, dim)` 张量
2. **连续性要求**: 最后一维必须连续（stride = 1）
3. **residual_out 强制**: 必须提供 `residual_out` 参数（不同于可选的早期版本）
4. **GPU 架构**: 仅支持 NVIDIA GPU，不支持 CPU/AMD/Intel 等其他设备
5. **Block Size 限制**: dim 必须小于等于最大 block size（否则部分线程会空转）

### 性能特征
- **时间复杂度**: O(batch * nhead * dim)
- **空间复杂度**: O(1) 额外空间（除输入输出外）
- **内存访问**:
  - 每个元素读取 2 次（a, b 各一次）+ 写入 1 次（residual_out）+ 读取 1 次（residual_out 重用）+ 写入 1 次（y）
  - 权重读取 1 次（w）
- **并行效率**:
  - Block 间无同步，完全并行
  - Block 内使用 CUB 的高效归约算法
  - 适合大 batch 和多 head 场景

### 与其他算子的关系
- **输入依赖**: 需要 `a` 和 `b` 两个激活值张量
- **输出**: `y`（归一化结果）和 `residual_out`（中间加法结果）
- **权重**: 使用可学习的 `w` 张量进行仿射变换
- **上层算子**: 常用于 Transformer 的 Pre-Norm 或 Post-Norm 层
- **组合使用**: 通常与 MatMul、Attention、FFN 等算子串联使用

## 7. 总结

本模块实现了高效的 Add RMS Norm 算子，通过以下关键技术实现高性能：

1. **CUDA 并行**: 充分利用 GPU 的 massively parallel 架构
2. **融合计算**: Add + RMS Norm 单次 kernel 完成，减少内存访问
3. **类型系统**: 灵活支持 FP16/BF16/FP32 的多种组合
4. **自适应优化**: 根据 GPU 能力选择最优 block size
5. **数据重用**: `residual_out` 作为中间结果存储，避免重复计算

该实现是 Infini 框架中神经网络计算的关键组件，特别适用于大语言模型（LLM）的训练和推理场景。
