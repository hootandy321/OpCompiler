# Causal Softmax NVIDIA 实现核心文档

本模块实现了基于 NVIDIA GPU 的因果掩码 softmax 操作（Causal Softmax），这是 Transformer 模型中自注意力机制的核心计算组件。该实现针对 CUDA 架构进行了深度优化，支持多种数据类型和 GPU 架构配置。

## 1. 模块结构

- **`causal_softmax_nvidia.cuh`**: NVIDIA 实现的头文件声明，通过宏定义生成 Descriptor 类
- **`causal_softmax_nvidia.cu`**: NVIDIA GPU 的完整实现，包含 kernel 调度逻辑、描述符管理和类型特化

## 2. 核心类与组件

### `op::causal_softmax::nvidia::Descriptor`
- **位置**: `causal_softmax_nvidia.cuh`（通过宏定义生成）、`causal_softmax_nvidia.cu`
- **主要功能**: NVIDIA GPU 后端的因果 softmax 操作描述符，继承自 `InfiniopDescriptor`，负责管理计算资源和执行调度
- **关键成员**:
  - `_opaque`: `Opaque *` 类型，封装设备内部状态（`device::nvidia::Handle::Internal` 共享指针）
  - `_info`: `CausalSoftmaxInfo` 类型，存储张量形状、步长、数据类型等元信息
  - `_workspace_size`: `size_t` 类型，工作空间大小（当前实现未使用）
- **核心方法**:
  - `create(handle, desc_ptr, y_desc, x_desc)`: 静态工厂方法，验证张量描述符兼容性并创建描述符实例。调用 `CausalSoftmaxInfo::create` 获取元信息，初始化 NVIDIA 设备句柄的内部状态引用
  - `calculate(workspace, workspace_size, y, x, stream_)`: 执行因果 softmax 计算的主入口。根据设备 maxThreadsPerBlock 能力（1024/512/4096）选择合适的 kernel 模板实例，配置 CUDA grid 维度为 `(seq_len, batch_size, 1)`
  - `~Descriptor()`: 析构函数，释放 `_opaque` 内部状态
- **生命周期**: 由用户调用 `create` 静态方法创建，计算完成后由调用者负责销毁

### `Descriptor::Opaque`
- **位置**: `causal_softmax_nvidia.cu` 内部结构体
- **主要功能**: 封装 NVIDIA GPU 设备相关的不透明状态
- **关键成员**:
  - `internal`: `std::shared_ptr<device::nvidia::Handle::Internal>`，共享的 CUDA 设备句柄内部状态（包含设备属性、流管理等）
- **生命周期**: 由 `Descriptor::create` 动态分配，由 `~Descriptor` 释放

### `causalSoftmax<BLOCK_SIZE, Tdata, Tcompute>` (CUDA Kernel 函数)
- **位置**: `causal_softmax_nvidia.cu`
- **主要功能**: CUDA kernel 启动器，包装下层 `causalSoftmaxKernel` 设备函数
- **模板参数**:
  - `BLOCK_SIZE`: `unsigned int`，线程块大小（支持 512、1024、4096）
  - `Tdata`: 数据类型（`half`、`__nv_bfloat16`、`float`）
  - `Tcompute`: 计算类型（统一使用 `float` 保证数值稳定性）
- **核心逻辑**: 直接调用 `causalSoftmaxKernel`，传递批次、高度、宽度及步长参数
- **并行策略**: Grid 维度为 `(seq_len, batch_size, 1)`，即每个序列位置对应一个 block

### `causalSoftmaxKernel<BLOCK_SIZE, Tdata, Tcompute>` (设备函数)
- **位置**: `../cuda/kernel.cuh`
- **主要功能**: 实现因果掩码 softmax 的核心计算逻辑
- **算法流程**（四阶段标准 softmax）:
  1. **最大值归约**: 沿行方向（宽度维度）对当前行及之前位置的元素（`width - height + 1 + blockIdx.x` 个元素）执行 `reduce_op::max`，结果存储于共享内存 `max_`
  2. **指数运算与因果掩码**: 对有效区域（满足 `width + blockIdx.x >= col + height` 条件）计算 `exp(x[col] - max_)`，否则填充 0。使用 `hexp` 函数处理半精度浮点
  3. **求和归约**: 对更新后的行执行 `reduce_op::sum`，结果存储于共享内存 `sum_`
  4. **归一化**: 将每个元素除以总和 `sum_`，得到 softmax 概率分布
- **内存访问模式**:
  - 使用步长（stride）索引：`y + blockIdx.y * y_stride_b + blockIdx.x * y_stride_h + threadIdx.x`
  - 支持非连续张量布局（如 NHWC vs NCHW）
- **同步机制**: 在每个归约和元素级操作后使用 `__syncthreads()` 保证共享内存一致性
- **数值稳定性**: 通过减去最大值避免指数上溢，使用 `float` 作为中间计算类型

## 3. API 接口

```cpp
// 创建描述符
static infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,              // Infini 运行时句柄
    Descriptor **desc_ptr,                 // [输出] 描述符指针
    infiniopTensorDescriptor_t y_desc,    // 输出张量描述符
    infiniopTensorDescriptor_t x_desc     // 输入张量描述符
);
// 返回: 成功返回 INFINI_STATUS_SUCCESS，失败返回对应错误码（如类型不兼容）

// 执行计算
infiniStatus_t Descriptor::calculate(
    void *workspace,                       // 工作空间（当前未使用，传 nullptr）
    size_t workspace_size,                 // 工作空间大小（传 0）
    void *y,                               // [输出] GPU 端输出张量指针
    const void *x,                         // [输入] GPU 端输入张量指针
    void *stream_                          // CUDA 流（cudaStream_t）
) const;
// 返回: 成功返回 INFINI_STATUS_SUCCESS，失败返回架构不支持或类型错误
```

## 4. 使用示例

```cpp
// 初始化 CUDA 环境
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_NVIDIA, 0);

// 准备张量描述符（形状: [batch_size, seq_len, total_seq_len]）
infiniopTensorDescriptor_t x_desc, y_desc;
// ... 假设已创建描述符，数据类型为 INFINI_DTYPE_F16

// 创建因果 softmax 描述符
op::causal_softmax::nvidia::Descriptor *softmax_desc;
auto status = op::causal_softmax::nvidia::Descriptor::create(
    handle, &softmax_desc, y_desc, x_desc);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 获取并分配 GPU 内存
half *d_x, *d_y;
cudaMalloc(&d_x, batch_size * seq_len * total_seq_len * sizeof(half));
cudaMalloc(&d_y, batch_size * seq_len * total_seq_len * sizeof(half));

// 拷贝输入数据到 GPU
cudaMemcpyAsync(d_x, h_x, size, cudaMemcpyHostToDevice, stream);

// 执行因果 softmax 计算
cudaStream_t stream;
cudaStreamCreate(&stream);
status = softmax_desc->calculate(nullptr, 0, d_y, d_x, stream);

// 同步并取回结果
cudaStreamSynchronize(stream);
cudaMemcpyAsync(h_y, d_y, size, cudaMemcpyDeviceToHost, stream);

// 清理资源
delete softmax_desc;
cudaFree(d_x);
cudaFree(d_y);
infiniopDestroyHandle(handle);
```

## 5. 实现细节

### 内存管理
- **张量布局**: 通过步长（stride）抽象支持任意内存布局（如连续、交错、分块）
- **工作空间**: 当前实现无需额外工作空间（workspace_size 固定为 0）
- **设备状态管理**: 使用 `std::shared_ptr` 共享 CUDA 设备句柄的内部状态，避免重复初始化

### 并发与并行
- **线程块配置**: 根据设备能力动态选择 block size（512/1024/4096），通过 `_opaque->internal->maxThreadsPerBlock()` 查询
- **Grid 策略**: 二维 grid `(seq_len, batch_size)`，每个序列位置对应一个独立的 block 处理一行
- **线程协作**: 使用 CUB 库的 `BlockReduce` 原语进行高效的块内归约（在 `reduce_op::max/sum` 中实现）
- **流式执行**: 支持异步 CUDA 流，允许多个计算流重叠执行

### 性能优化
- **类型特化**: 支持半精度（`half`）、bfloat16（`__nv_bfloat16`）和单精度（`float`）浮点，在存储受限场景使用半精度减少显存占用
- **计算类型分离**: 数据类型 `Tdata` 和计算类型 `Tcompute` 分离，统一使用 `float` 作为中间计算类型避免精度损失
- **归约优化**: 利用 CUB 库的高效归约算法（基于 warp shuffle 指令），复杂度 O(log n)
- **因果掩码融合**: 在 softmax 计算过程中应用因果掩码，避免单独的掩码 kernel 调用
- **内存合并访问**: 线程按连续顺序访问全局内存，最大化合并传输效率

### 错误处理
- **错误码传播**: 使用 `CHECK_RESULT` 和 `CHECK_STATUS` 宏统一处理错误码，失败时立即返回
- **类型验证**: `launchKernel` 中检查数据类型是否为 `F16`/`BF16`/`F32`，否则返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
- **架构兼容性**: 检测设备 block size 能力，不支持时返回 `INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED`

### 依赖项
- **CUDA Runtime**: 核心并行计算平台
- **CUB 库**: `cub::BlockReduce` 用于块内归约操作
- **Infini 内部模块**:
  - `devices/nvidia/nvidia_common.cuh`: CUDA 常量定义（如 `CUDA_BLOCK_SIZE_1024`）
  - `devices/nvidia/nvidia_kernel_common.cuh`: kernel 通用工具
  - `reduce/cuda/reduce.cuh`: 归约操作实现（`reduce_op::max/sum`）
  - `causal_softmax/info.h`: 张量元信息验证

### 设计模式
- **策略模式**: 通过模板参数 `BLOCK_SIZE`、`Tdata`、`Tcompute` 实现编译期策略选择，避免运行时分支
- **工厂模式**: `create` 静态方法作为工厂，封装复杂的初始化逻辑
- **适配器模式**: `Descriptor` 适配统一的 `InfiniopDescriptor` 接口，隐藏 NVIDIA 特定实现
- **RAII**: 使用 `std::shared_ptr` 自动管理设备句柄生命周期

### 算法复杂度
- **时间复杂度**: O(batch_size × seq_len × total_seq_len × log(total_seq_len))，归约操作主导
- **空间复杂度**: O(1) 额外空间（仅使用共享内存，大小为 2 × BLOCK_SIZE × sizeof(float/半精度)）
