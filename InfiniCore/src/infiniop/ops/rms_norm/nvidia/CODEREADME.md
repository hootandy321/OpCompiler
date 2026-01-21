# RMSNorm NVIDIA GPU 算子核心实现文档

本模块实现了 RMS (Root Mean Square) 归一化算子在 NVIDIA GPU 上的高性能 CUDA kernel，支持 FP16、BF16 和 FP32 数据类型，并采用 CUB 库的块级归约优化技术实现高效的并行计算。该算子广泛应用于 Transformer 模型（如 GPT、BERT、LLaMA）的层归一化层。

## 1. 模块结构

- **`rms_norm_nvidia.cuh`**: RMSNorm NVIDIA 算子的头文件，通过宏 `DESCRIPTOR(nvidia)` 展开生成完整的 Descriptor 类声明
- **`rms_norm_nvidia.cu`**: RMSNorm NVIDIA 算子的核心实现，包含算子描述符的创建、kernel 启动逻辑和多数据类型特化

## 2. 核心类

### `Descriptor`
- **位置**: `rms_norm_nvidia.cuh` (通过宏展开), `rms_norm_nvidia.cu`
- **主要功能**: RMSNorm 算子的描述符类，继承自 `InfiniopDescriptor`，负责管理算子的元数据、工作空间大小、设备句柄，并提供算子创建和计算接口
- **关键成员**:
  - `_opaque`: `Opaque*` 类型，指向 NVIDIA 设备特定的内部状态（包含 CUDA 设备句柄、设备属性等）
  - `_info`: `RMSNormInfo` 类型，存储算子的张量描述信息（形状、步长、数据类型、epsilon 参数）
  - `_workspace_size`: `size_t` 类型，算子执行所需的工作空间大小（本实现中固定为 0）
- **核心方法**:
  - `create(infiniopHandle_t, Descriptor**, y_desc, x_desc, w_desc, epsilon)`: 静态工厂方法，验证输入/输出/权重张量的数据类型和形状兼容性，创建并初始化描述符对象。支持的数据类型组合包括：FP16/FP16, FP16/BF16, FP16/FP32, BF16/BF16, BF16/FP16, BF16/FP32, FP32/FP32。复杂度为 O(1)。
  - `calculate(workspace, workspace_size, y, x, w, stream)`: 执行 RMSNorm 计算。根据 GPU 设备的最大线程块大小（512/1024/4096）选择对应的 kernel 配置，提取张量步长和形状信息，启动 CUDA kernel。工作空间大小必须满足 `_workspace_size` 要求。
  - `workspaceSize() const`: 返回算子所需的工作空间大小（本实现为 0）
- **生命周期**: 由 `create` 静态方法构造，析构函数释放 `_opaque` 指向的内存

### `Descriptor::Opaque`
- **位置**: `rms_norm_nvidia.cu`
- **主要功能**: 封装 NVIDIA 设备特定的内部状态
- **关键成员**:
  - `internal`: `std::shared_ptr<device::nvidia::Handle::Internal>` 类型，共享指针管理 NVIDIA 设备句柄的内部实现，包含 CUDA 设备属性、流管理等资源

## 3. API 接口

```cpp
// 创建 RMSNorm 描述符
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,             // [in] InfiniOP 句柄，包含设备和上下文信息
    Descriptor **desc_ptr,               // [out] 输出创建的描述符指针
    infiniopTensorDescriptor_t y_desc,   // [in] 输出张量描述符 (形状: [batch] 或 [batch, nhead] 或 [batch, nhead, dim])
    infiniopTensorDescriptor_t x_desc,   // [in] 输入张量描述符 (形状与 y_desc 相同)
    infiniopTensorDescriptor_t w_desc,   // [in] 权重张量描述符 (形状: [dim])
    float epsilon                        // [in] 防止除零的小常数 (如 1e-5)
);
// 返回: 成功返回 INFINI_STATUS_SUCCESS，失败返回对应的错误码（类型不匹配、形状不兼容、步长非法等）

// 执行 RMSNorm 计算
infiniStatus_t Descriptor::calculate(
    void *workspace,          // [in] 工作空间指针（本实现中未使用，可传 nullptr）
    size_t workspace_size,    // [in] 工作空间大小（必须 >= workspaceSize()）
    void *y,                  // [out] 输出张量数据指针
    const void *x,            // [in] 输入张量数据指针
    const void *w,            // [in] 权重张量数据指针
    void *stream              // [in] CUDA 流指针（cudaStream_t）
) const;
// 返回: 成功返回 INFINI_STATUS_SUCCESS，失败返回 INFINI_STATUS_INSUFFICIENT_WORKSPACE 或 INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED

// 获取工作空间大小
size_t Descriptor::workspaceSize() const;
// 返回: 算子所需的工作空间大小（本实现固定返回 0）
```

## 4. 使用示例

```cpp
// 示例：在 NVIDIA GPU 上使用 RMSNorm 算子对 Transformer 隐藏状态进行归一化

// 1. 准备张量描述符
// 假设输入形状为 [batch_size, num_heads, head_dim]
infiniopTensorDescriptor_t x_desc, y_desc, w_desc;
infiniopCreateTensorDescriptor(handle, &x_desc);
infiniopSetTensorDescriptor(x_desc, INFINI_DTYPE_FP16, 3,
                            {batch_size, num_heads, head_dim},
                            {num_heads * head_dim, head_dim, 1});

infiniopCreateTensorDescriptor(handle, &y_desc);
infiniopSetTensorDescriptor(y_desc, INFINI_DTYPE_FP16, 3,
                            {batch_size, num_heads, head_dim},
                            {num_heads * head_dim, head_dim, 1});

infiniopCreateTensorDescriptor(handle, &w_desc);
infiniopSetTensorDescriptor(w_desc, INFINI_DTYPE_FP32, 1,
                            {head_dim}, {1});

// 2. 创建 RMSNorm 描述符
op::rms_norm::nvidia::Descriptor *rmsnorm_desc;
infiniStatus_t status = op::rms_norm::nvidia::Descriptor::create(
    handle, &rmsnorm_desc, y_desc, x_desc, w_desc, 1e-5f);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 3. 分配 GPU 内存
half *d_x, *d_y;
float *d_w;
cudaMalloc(&d_x, batch_size * num_heads * head_dim * sizeof(half));
cudaMalloc(&d_y, batch_size * num_heads * head_dim * sizeof(half));
cudaMalloc(&d_w, head_dim * sizeof(float));

// 4. 初始化权重（通常从 CPU 拷贝）
float *h_w = new float[head_dim];
for (size_t i = 0; i < head_dim; ++i) h_w[i] = 1.0f; // 全1权重
cudaMemcpy(d_w, h_w, head_dim * sizeof(float), cudaMemcpyHostToDevice);

// 5. 创建 CUDA 流
cudaStream_t stream;
cudaStreamCreate(&stream);

// 6. 执行 RMSNorm 计算
status = rmsnorm_desc->calculate(
    nullptr, 0,           // 无需工作空间
    d_y, d_x, d_w,        // 输入、输出、权重
    stream                // CUDA 流
);

// 7. 同步并检查错误
cudaStreamSynchronize(stream);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理计算错误
}

// 8. 清理资源
delete rmsnorm_desc;
cudaFree(d_x);
cudaFree(d_y);
cudaFree(d_w);
cudaStreamDestroy(stream);
```

## 5. 实现细节

### RMSNorm 算法原理
RMS (Root Mean Square) 归一化是一种无需中心化（减去均值）的归一化方法，计算公式为：

```
y = x * w / sqrt(mean(x^2) + epsilon)
```

其中：
- `x` 为输入张量，形状为 `[batch, nhead, dim]` 或 `[batch, dim]`
- `w` 为可学习的缩放权重，形状为 `[dim]`
- `mean(x^2)` = `sum(x^2) / dim`，为输入元素平方的均值
- `epsilon` 为防止除零的小常数（通常取 1e-5）
- `y` 为归一化后的输出张量

与 LayerNorm 不同，RMSNorm 省略了减去均值的步骤，计算更简单且效果相当，在 LLaMA 等大语言模型中被广泛采用。

### Kernel 并行策略
- **Block-Head 映射**: 每个 CUDA Block 处理一个 Batch 中的一个 Head 的所有维度元素。Grid 大小为 `batch_size * nhead`，Block 大小为 512/1024/4096（根据 GPU 架构自动选择）。
- **Strided 访问**: 线程以步长 `BLOCK_SIZE` 遍历数据，即线程 `i` 处理索引 `i, i+BLOCK_SIZE, i+2*BLOCK_SIZE, ...` 的元素，保证良好的内存合并访问。

### 计算流程 (Kernel 层面)
1. **块级归约求平方和**: 调用 `op::common_cuda::reduce_op::sumSquared<BLOCK_SIZE, Tdata, Tcompute>`，使用 CUB 的 `BlockReduce` 原语在 Block 内归约所有线程计算的局部平方和，得到完整的 `sum(x^2)`。时间复杂度 O(dim / BLOCK_SIZE)。
2. **线程 0 计算 RMS**: 只有 `threadIdx.x == 0` 的线程执行 `rms = rsqrtf(ss / dim + epsilon)`，将结果写入共享内存变量 `__shared__ Tcompute rms`。
3. **同步广播**: 通过 `__syncthreads()` 确保 RMS 值对 Block 内所有线程可见。
4. **并行应用归一化**: 所有线程并行遍历其负责的元素，计算 `y[i] = x[i] * w[i] * rms`，每个元素的计算独立无依赖，可充分并行。时间复杂度 O(dim / BLOCK_SIZE)。

### 多数据类型支持
- **计算类型提升**: 所有半精度类型（FP16/BF16）的计算均在 `float` 类型中进行（`Tcompute = float`），避免精度损失和溢出。
- **类型组合支持**: 通过宏 `LAUNCH_KERNEL` 实现编译期类型分发，支持以下组合：
  - FP16 输入 + FP16 权重
  - FP16 输入 + BF16 权重
  - FP16 输入 + FP32 权重
  - BF16 输入 + BF16 权重
  - BF16 输入 + FP16 权重
  - BF16 输入 + FP32 权重
  - FP32 输入 + FP32 权重

### 内存管理
- **零工作空间设计**: 算子不需要额外的临时缓冲区，所有中间结果（平方和、RMS 值）均通过共享内存和寄存器完成。
- **共享内存使用**: 每个 Block 使用 `sizeof(Tcompute)` 字节的共享内存存储 RMS 值，以及 CUB `BlockReduce` 所需的临时存储（约 `BLOCK_SIZE * sizeof(Tcompute)` 字节）。

### 性能优化
- **CUB 库加速**: 使用 NVIDIA CUB 库的 `BlockReduce` 原语实现 Warp 原语优化的块级归约，充分利用 CUDA Warp Shuffle 指令，避免手动实现低效的共享内存归约。
- **自适应 Block 大小**: 根据 GPU 设备的 `maxThreadsPerBlock` 属性（512、1024 或 4096）选择最优的 Block 大小，最大化并行度和占用率。
- **步长支持**: 支持非连续张量（通过 `stride_y_batch`, `stride_y_nhead`, `stride_x_batch`, `stride_x_nhead` 参数），适用于转置、视图等复杂内存布局场景。

### 错误处理
- **类型验证**: `RMSNormInfo::create` 验证输入/输出/权重的数据类型兼容性，拒绝不支持的组合（如 FP64 输入、半精度输入 + FP64 权重）。
- **形状验证**: 检查张量维度（仅支持 2D 或 3D）、形状一致性（batch、nhead、dim 必须匹配）和最后一维连续性（stride 必须为 1）。
- **设备架构检查**: `calculate` 方法检查 GPU 是否支持 512/1024/4096 线程的 Block，不支持的架构返回 `INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED`。
- **工作空间检查**: 执行前验证 `workspace_size >= _workspace_size`，否则返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`。

### 依赖关系
- **外部依赖**:
  - CUDA Runtime API (`cudaStream_t`, `__restrict__`, `__device__`, `__shared__`, `__syncthreads()`)
  - NVIDIA CUB 库 (`cub::BlockReduce`)
- **内部依赖**:
  - `../rms_norm.h`: 算子描述符的宏定义声明
  - `../rms_norm/info.h`: `RMSNormInfo` 类，包含张量形状、步长、数据类型等元信息和验证逻辑
  - `../../devices/nvidia/nvidia_common.cuh`: NVIDIA 设备公共头文件（设备属性常量）
  - `../../devices/nvidia/nvidia_kernel_common.cuh`: NVIDIA kernel 公共宏和工具
  - `../../reduce/cuda/reduce.cuh`: 通用的 CUDA 归约 kernel 函数（`sumSquared`, `sum`, `max`）
  - `../cuda/kernel.cuh`: RMSNorm CUDA kernel 的设备函数实现（`rmsnormBlock`）

### 设计模式
- **宏代码生成**: 通过 `DESCRIPTOR(NAMESPACE)` 宏在不同命名空间中展开相同的类结构，避免代码重复。
- **策略模式**: `launchKernel` 函数模板根据数据类型组合编译期生成不同的 kernel 实例，避免运行时分支开销。
- **RAII 资源管理**: `Opaque` 结构体使用 `std::shared_ptr` 管理 CUDA 设备句柄，确保资源安全释放。
