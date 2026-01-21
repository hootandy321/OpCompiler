# NVIDIA CUDA 卷积操作核心实现文档

本模块实现了基于 NVIDIA GPU 的卷积操作（Convolution），通过 cuDNN 库提供高性能的前向传播卷积计算，支持 1D/2D/3D 卷积、偏置项融合、多种数据类型（FP16/FP32/BF16）以及灵活的卷积参数配置。

## 1. 模块结构

- **`conv_nvidia.cuh`**: NVIDIA 卷积操作头文件，定义了 Descriptor 类并使用宏机制生成标准接口
- **`conv_nvidia.cu`**: NVIDIA 卷积操作核心实现，封装 cuDNN API 并管理卷积描述符生命周期

## 2. 核心类

### `Descriptor::Opaque`
- **位置**: `conv_nvidia.cu`
- **主要功能**: 封装 cuDNN 卷积操作的所有底层状态，包括张量描述符、卷积描述符、算法选择和工作空间管理
- **关键成员**:
  - `internal`: `std::shared_ptr<device::nvidia::Handle::Internal>` - NVIDIA 设备句柄的内部表示，用于访问 cuDNN 上下文
  - `workspace_size`: `size_t` - cuDNN 卷积算法所需工作空间大小（字节）
  - `x_desc`: `cudnnTensorDescriptor_t` - 输入张量的 cuDNN 描述符
  - `y_desc`: `cudnnTensorDescriptor_t` - 输出张量的 cuDNN 描述符
  - `w_desc`: `cudnnFilterDescriptor_t` - 卷积核（滤波器）的 cuDNN 描述符
  - `b_desc`: `cudnnTensorDescriptor_t` - 偏置项张量的 cuDNN 描述符（可选）
  - `act_desc`: `cudnnActivationDescriptor_t` - 激活函数描述符（固定为 IDENTITY）
  - `conv_desc`: `cudnnConvolutionDescriptor_t` - 卷积操作描述符（填充、步长、扩张等）
  - `algo`: `cudnnConvolutionFwdAlgo_t` - 选定的前向卷积算法（默认 IMPLICIT_GEMM）

#### 核心方法

- **`initializeDimensionArrays()`**: 初始化输入/输出/滤波器的维度数组
  - 处理 1D 卷积特殊场景（转换为 4D 张量，第二维设为 1）
  - 生成 NCHW 格式的维度数组（Batch, Channels, Spatial Dims...）
  - 计算输入和输出张量的步长（stride）

- **`initializeConvolutionParams()`**: 初始化卷积参数数组
  - 提取填充（padding）、步长（stride）、扩张（dilation）信息
  - 1D 卷积转换为 2D 格式（第一维填充/步长/扩张设为 1）

- **`calculateStrides()`**: 计算张量步长
  - 采用 NCHW 内存布局，从最后一维向前计算步长
  - 保证 `strides[i] = strides[i+1] * dims[i+1]`

- **`getCudnnDataType()`**: 数据类型映射
  - 将 Infini 的 `infiniDtype_t`（F16/F32/BF16）转换为 cuDNN 的 `cudnnDataType_t`
  - 拒绝不支持的数据类型，返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`

- **`createBasicDescriptors()`**: 创建基础 cuDNN 描述符
  - 创建输入、输出、滤波器、卷积描述符
  - 设置张量格式为 `CUDNN_TENSOR_NCHW`
  - 时间复杂度：O(1)，仅涉及描述符创建

- **`createBiasDescriptors()`**: 创建偏置项和激活描述符
  - 如果没有偏置项（`bias_dims_size() == 0`），跳过创建
  - 偏置张量形状为 `[1, out_channels, 1, 1, ...]`
  - 激活函数固定为 `CUDNN_ACTIVATION_IDENTITY`（不应用激活）

- **`setupConvolutionDescriptor()`**: 配置卷积描述符
  - 设置填充、步长、扩张参数
  - 卷积模式为 `CUDNN_CROSS_CORRELATION`（互相关，标准深度学习卷积）
  - 计算数据类型固定为 `CUDNN_DATA_FLOAT`（即使输入是 FP16/BF16）

- **`setupAlgorithmWithoutBias()`**: 无偏置卷积的算法选择
  - 固定使用 `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM` 算法
  - 调用 `cudnnGetConvolutionForwardWorkspaceSize()` 查询所需工作空间大小

- **`setupAlgorithmWithBias()`**: 带偏置卷积的算法选择
  - 调用 `cudnnGetConvolutionForwardAlgorithmMaxCount()` 获取最大算法数量
  - 使用 `cudnnFindConvolutionForwardAlgorithm()` 自动搜索最优算法
  - 遍历性能结果，选择第一个可用算法并查询其工作空间需求
  - 启发式策略：优先选择 cuDNN 评估的最佳算法

- **`initializeCudnnContext()`**: 完整初始化 cuDNN 上下文
  - 协调整个初始化流程：维度数组 → 卷积参数 → 基础描述符 → 偏置描述符 → 卷积描述符 → 算法选择
  - 根据是否有偏置项调用不同的算法设置函数
  - 返回 `INFINI_STATUS_SUCCESS` 或错误码

- **`create()`**: 静态工厂方法，创建 Opaque 对象
  - 构造 Opaque 实例并调用 `initializeCudnnContext()`
  - 使用移动语义返回 `utils::Result<Opaque>`
  - 如果未启用 cuDNN（`ENABLE_CUDNN_API` 未定义），返回 `INFINI_STATUS_NOT_IMPLEMENTED`

- **生命周期**: 采用 RAII 模式
  - 构造时初始化 cuDNN 描述符
  - 移动构造转移所有权，将原对象指针置空
  - 析构时通过 `CLEANUP_CUDNN_DESCRIPTORS()` 宏释放所有 cuDNN 描述符

### `Descriptor`
- **位置**: `conv_nvidia.cu`（通过 `conv.h` 中的 `DESCRIPTOR(nvidia)` 宏生成）
- **主要功能**: 对外暴露的卷积操作描述符接口，继承自 `InfiniopDescriptor`
- **关键成员**:
  - `_opaque`: `Opaque *` - 指向内部实现对象的指针
  - `_dtype`: `infiniDtype_t` - 卷积操作的数据类型
  - `_info`: `ConvInfo` - 卷积操作的元数据（维度、参数等）
  - `_workspace_size`: `size_t` - cuDNN 工作空间大小

#### 核心方法

- **`create()`**: 创建卷积描述符
  - 参数验证：检查数据类型是否为 F16/F32/BF16
  - 调用 `ConvInfo::create()` 生成卷积元数据
  - 调用 `Opaque::create()` 初始化 cuDNN 上下文
  - 构造 Descriptor 对象并返回
  - 时间复杂度：O(n)，n 为卷积维度数

- **`calculate()`**: 执行卷积计算
  - **无偏置模式**：调用 `cudnnConvolutionForward()`
    - alpha = 1.0, beta = 0.0（直接覆盖输出）
  - **有偏置模式**：调用 `cudnnConvolutionBiasActivationForward()`
    - 融合卷积、偏置加法、激活函数（IDENTITY）于单次 kernel 调用
    - 减少内存访问和中间结果存储
  - 使用 CUDA Stream 异步执行
  - 时间复杂度：取决于 cuDNN 算法，通常 O(N × C × K × S)，其中 N 为输出元素数，C 为输入通道数，K 为卷积核大小，S 为输出空间尺寸

## 3. API 接口

```cpp
// 创建卷积描述符
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,              // [in] 设备句柄
    Descriptor **desc_ptr,                 // [out] 输出的描述符指针
    infiniopTensorDescriptor_t y_desc,     // [in] 输出张量描述符
    infiniopTensorDescriptor_t x_desc,     // [in] 输入张量描述符
    infiniopTensorDescriptor_t w_desc,     // [in] 卷积核张量描述符
    infiniopTensorDescriptor_t b_desc,     // [in] 偏置张量描述符（可为 nullptr）
    const void *pads,                      // [in] 填充数组（size_t[]）
    const void *strides,                   // [in] 步长数组（ptrdiff_t[]）
    const void *dilations,                 // [in] 扩张数组（size_t[]）
    size_t n                               // [in] 卷积维度数（1/2/3）
);

// 执行卷积计算
infiniStatus_t Descriptor::calculate(
    void *workspace,                       // [in] cuDNN 工作空间指针
    size_t workspace_size,                 // [in] 工作空间大小（字节）
    void *y,                               // [out] 输出张量数据
    const void *x,                         // [in] 输入张量数据
    const void *w,                         // [in] 卷积核数据
    const void *bias,                      // [in] 偏置数据（可为 nullptr）
    void *stream                           // [in] CUDA 流
) const;
```

## 4. 使用示例

```cpp
// 1. 定义卷积参数
constexpr size_t ndim = 2;  // 2D 卷积
size_t pads[] = {1, 1};     // 上下左右各填充 1
ptrdiff_t strides[] = {1, 1};  // 步长为 1
size_t dilations[] = {1, 1};   // 扩张率为 1

// 2. 创建卷积描述符
op::conv::nvidia::Descriptor *conv_desc = nullptr;
auto status = op::conv::nvidia::Descriptor::create(
    handle,
    &conv_desc,
    y_desc,  // 输出: [batch, out_channels, out_h, out_w]
    x_desc,  // 输入: [batch, in_channels, in_h, in_w]
    w_desc,  // 卷积核: [out_channels, in_channels, kernel_h, kernel_w]
    b_desc,  // 偏置: [out_channels] 或 nullptr
    pads,
    strides,
    dilations,
    ndim
);

// 3. 分配工作空间
size_t workspace_size = conv_desc->workspaceSize();
void *workspace = nullptr;
if (workspace_size > 0) {
    cudaMalloc(&workspace, workspace_size);
}

// 4. 执行卷积计算
status = conv_desc->calculate(
    workspace,
    workspace_size,
    y_data,    // 输出数据
    x_data,    // 输入数据
    w_data,    // 卷积核数据
    b_data,    // 偏置数据（或 nullptr）
    cuda_stream
);

// 5. 清理资源
cudaFree(workspace);
delete conv_desc;
```

## 5. 实现细节

### 内存管理
- **工作空间分配**: cuDNN 要求的工作空间由调用方在 GPU 内存中分配，通过 `workspaceSize()` 查询大小
- **描述符生命周期**: cuDNN 描述符在 Opaque 对象析构时自动释放，使用 `DESTROY_CUDNN_DESCRIPTOR` 宏确保异常安全
- **引用计数**: `device::nvidia::Handle::Internal` 使用 `std::shared_ptr` 管理，避免悬空指针

### 并发性
- **CUDA Stream**: 卷积计算在指定的 CUDA Stream 上异步执行，支持与其他 CUDA 操作并发
- **线程安全**: cuDNN Handle 通过 `device::nvidia::Handle::Internal` 管理，每个线程应使用独立的 Handle
- **工作空间复用**: 工作空间可在多次卷积调用间复用，减少内存分配开销

### 性能优化
- **算法自动选择**: 带偏置卷积使用 `cudnnFindConvolutionForwardAlgorithm()` 启发式搜索最优算法，考虑 GPU 架构、张量形状、工作空间限制
- **算子融合**: 偏置加法和激活函数通过 `cudnnConvolutionBiasActivationForward()` 融合到单次 kernel 启动，减少全局内存访问
- **数据类型**: 计算精度固定为 FP32，即使输入是 FP16/BF16，避免精度损失
- **1D 卷积优化**: 1D 卷积转换为 2D 张量处理（第二维为 1），复用 cuDNN 的高性能 2D 卷积实现

### 错误处理
- **错误传播**: 所有 cuDNN API 调用通过 `CHECK_CUDNN` 宏包装，错误码转换为 `infiniStatus_t`
- **参数验证**: 创建描述符时验证数据类型、张量形状一致性、输出尺寸计算正确性
- **Result 类型**: 使用 `utils::Result<T>` 封装可能失败的操作，避免异常开销

### 依赖关系
- **cuDNN**: 强依赖，通过 `ENABLE_CUDNN_API` 宏控制编译。未启用时返回 `INFINI_STATUS_NOT_IMPLEMENTED`
- **CUDA Runtime**: 用于内存分配和 Stream 管理
- **设备抽象层**: 依赖 `device::nvidia::Handle` 访问 cuDNN 上下文
- **元数据处理**: 使用 `ConvInfo` 类统一管理卷积参数和形状信息

### 设计模式
- **Pimpl 模式**: `Descriptor` 通过 `Opaque` 指针隐藏 cuDNN 实现细节，减少头文件依赖
- **工厂模式**: `create()` 静态方法封装对象创建逻辑，返回 Result 类型
- **RAII**: 资源获取即初始化，析构函数自动释放 cuDNN 描述符
- **宏生成接口**: `DESCRIPTOR(nvidia)` 宏为不同后端生成统一的 Descriptor 类接口
