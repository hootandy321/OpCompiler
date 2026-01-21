# GELU NVIDIA GPU 算子核心实现文档

本模块实现了 GELU（Gaussian Error Linear Unit）激活函数的 NVIDIA GPU CUDA 后端，支持 FP16、BF16、FP32 和 FP64 四种浮点数据类型，基于 InfiniOp 的逐元素操作（elementwise）框架构建。

## 1. 模块结构

- **`gelu_nvidia.cuh`**: GELU NVIDIA 描述符的 API 声明头文件，通过宏定义生成完整的描述符类
- **`gelu_nvidia.cu`**: GELU NVIDIA GPU 实现主文件，包含描述符的创建、计算和析构逻辑

## 2. 核心类与组件

### `Descriptor` 类
- **位置**: `gelu_nvidia.cuh`（通过 `ELEMENTWISE_DESCRIPTOR` 宏生成）
- **命名空间**: `op::gelu::nvidia`
- **继承**: 继承自 `InfiniopDescriptor`
- **主要功能**: 封装 GELU 操作的 CUDA 描述符，管理设备信息、工作空间大小和数据类型

#### 核心成员变量
```cpp
infiniDtype_t _dtype;                                         // 输出/输入张量的数据类型
op::elementwise::ElementwiseInfo _info;                       // 张量形状、步幅等元数据
std::unique_ptr<op::elementwise::nvidia::DeviceImpl> _device_info; // CUDA 设备实现
size_t _workspace_size;                                       // 设备端工作空间大小（字节）
```

#### 核心方法

##### `create()`
```cpp
static infiniStatus_t create(
    infiniopHandle_t handle_,              // InfiniOp 设备句柄
    Descriptor **desc_ptr,                 // [输出] 创建的描述符指针
    infiniopTensorDescriptor_t out_desc,   // 输出张量描述符
    std::vector<infiniopTensorDescriptor_t> input_desc_vec); // 输入张量描述符向量
```
- **功能**: 创建并初始化 GELU 操作描述符
- **执行流程**:
  1. 将句柄转换为 NVIDIA 设备句柄
  2. 验证输出数据类型（支持 BF16、F16、F32、F64）
  3. 验证输入和输出张量形状一致性（`CHECK_SAME_SHAPE`）
  4. 通过 `CREATE_ELEMENTWISE_CUDA_DESCRIPTOR` 宏创建逐元素操作描述符：
     - 调用 `ElementwiseInfo::create()` 生成元数据
     - 计算工作空间大小 = 元数据大小 + 输入指针数组大小
     - 创建 `DeviceImpl` 实例
  5. 构造并返回 `Descriptor` 对象
- **时间复杂度**: O(ndim)，其中 ndim 是张量维度数
- **错误处理**: 返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`、`INFINI_STATUS_BAD_TENSOR_SHAPE` 等错误码

##### `calculate()`
```cpp
infiniStatus_t calculate(
    void *workspace,                      // 设备端工作空间指针
    size_t workspace_size,                // 工作空间大小（字节）
    void *output,                         // 输出张量设备指针
    std::vector<const void *> inputs,     // 输入张量设备指针向量
    void *stream) const;                  // CUDA 流指针
```
- **功能**: 在 GPU 上执行 GELU 激活计算
- **执行流程**:
  1. 验证工作空间大小是否足够
  2. 根据 `_dtype` 分发到对应的模板特化：
     - `INFINI_DTYPE_BF16`: 调用 `DeviceImpl::calculate<256, cuda::GeluOp, cuda_bfloat16>`
     - `INFINI_DTYPE_F16`: 调用 `DeviceImpl::calculate<256, cuda::GeluOp, half>`
     - `INFINI_DTYPE_F32`: 调用 `DeviceImpl::calculate<256, cuda::GeluOp, float>`
     - `INFINI_DTYPE_F64`: 调用 `DeviceImpl::calculate<256, cuda::GeluOp, double>`
  3. 设备端执行逐元素 GELU 计算
- **线程配置**: 固定使用 256 线程/块（BLOCK_SIZE = 256）
- **空间复杂度**: O(1)，额外空间仅用于元数据存储

##### `~Descriptor()`
```cpp
~Descriptor() = default;
```
- **功能**: 析构函数，使用默认实现（智能指针自动管理资源）

### `GeluOp` 函数对象
- **位置**: `../cuda/kernel.cuh`
- **命名空间**: `op::gelu::cuda`
- **功能**: 定义 GELU 的 CUDA 设备端计算逻辑

#### 核心方法
```cpp
template <typename T>
__device__ __forceinline__ T operator()(const T &x) const;
```
- **输入**: 单个标量值 `x`
- **输出**: 应用 GELU 激活后的值
- **数学公式**: `GELU(x) = 0.5 * x * (1 + erf(x / √2))`

#### 数据类型特化实现
```cpp
// BF16: 先转换为 float 计算，再转回 BF16
if constexpr (std::is_same_v<T, cuda_bfloat16>) {
    float x_f = __bfloat162float(x);
    float result = 0.5 * x_f * (1 + erf(x_f / sqrt(2.0f)));
    return __float2bfloat16(result);
}

// FP16: 先转换为 float 计算，再转回 half
else if constexpr (std::is_same_v<T, half>) {
    float x_f = __half2float(x);
    float result = 0.5 * x_f * (1 + erf(x_f / sqrt(2.0f)));
    return __float2half(result);
}

// FP32: 直接计算
else if constexpr (std::is_same_v<T, float>) {
    return 0.5 * x * (1 + erf(x / sqrt(2.0f)));
}

// FP64: 直接计算（双精度）
else {
    return 0.5 * x * (1 + erf(x / sqrt(2.0)));
}
```
- **性能优化**:
  - `__device__ __forceinline__`: 强制内联以减少函数调用开销
  - 低精度类型（BF16/FP16）在 float 精度下计算以保证数值稳定性
  - 使用 CUDA 内置函数 `erf()` 计算误差函数
- **数值稳定性**: 避免在低精度下直接计算累积误差

### `DeviceImpl` 类（继承自 elementwise 框架）
- **位置**: `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/nvidia/elementwise_nvidia.cuh`
- **命名空间**: `op::elementwise::nvidia`
- **设计模式**: Pimpl（Pointer to Implementation）模式，通过 `Opaque` 结构体隐藏实现

#### 核心方法
```cpp
template <unsigned int BLOCK_SIZE, typename Op, typename Tdata, typename... Args>
infiniStatus_t calculate(
    const op::elementwise::ElementwiseInfo &info,
    void *workspace,
    void *output,
    const std::vector<const void *> &inputs,
    void *stream,
    Args &&...args);
```
- **功能**: 启动逐元素操作的 CUDA 核函数
- **参数说明**:
  - `BLOCK_SIZE`: 编译时常量，指定每个线程块的线程数（GELU 使用 256）
  - `Op`: 操作类型（GELU 使用 `cuda::GeluOp`）
  - `Tdata`: 输入和输出的数据类型
- **执行流程**:
  1. 将元数据和输入指针数组异步拷贝到设备（`cudaMemcpyAsync`）
  2. 计算网格维度：`gridDims = min(CEIL_DIV(output_size, BLOCK_SIZE), gridSizeX)`
  3. 分多次启动核函数（支持大型张量的分段执行）
  4. 每次传递线性偏移量 `offset` 以支持分段计算

### `ElementwiseInfo` 结构体
- **位置**: `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/elementwise.h`
- **功能**: 封装逐元素操作的元数据，包括形状、步幅、连续性标志等

#### 内存布局
元数据在设备端的线性内存布局（以 size_t 对齐）：
```
┌─────────────────────┐
│ output_shape        │ [ndim * size_t]
├─────────────────────┤
│ output_strides      │ [ndim * ptrdiff_t]
├─────────────────────┤
│ input_shapes        │ [input_size * ndim * size_t]
├─────────────────────┤
│ input_strides       │ [input_size * ndim * ptrdiff_t]
├─────────────────────┤
│ input_contiguous    │ [input_size * bool]
├─────────────────────┤
│ input_broadcasted   │ [input_size * bool]
└─────────────────────┘
```

#### 关键方法
```cpp
static ResultType create(infiniopTensorDescriptor_t output_desc,
                         std::vector<infiniopTensorDescriptor_t> input_descs);
```
- **功能**: 从张量描述符构建元数据结构
- **验证**:
  - 输出张量不能有广播维度
  - 自动检测输入张量的连续性和广播标志

## 3. API 接口

### 公共 API（通过描述符暴露）

```cpp
// 创建 GELU 描述符
infiniStatus_t infinio pCreateGeluDescriptor(
    infiniopHandle_t handle,
    infinio pDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc);

// 执行 GELU 计算
infiniStatus_t infinio pGelu(
    infinio pDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream);

// 销毁描述符
void infinio pDestroyGeluDescriptor(infinio pDescriptor_t desc);
```

### 内部 CUDA 核函数签名

```cpp
template <size_t N, typename Op, typename Tdata, typename... Args>
__global__ void elementwiseKernel(
    size_t output_size,                  // 输出元素总数
    size_t ndim,                         // 张量维度数
    bool output_contiguous,              // 输出是否连续
    const bool *__restrict__ input_contiguous,    // 各输入是否连续
    const bool *__restrict__ input_broadcasted,   // 各输入是否广播
    const size_t *__restrict__ output_shape,      // 输出形状
    const size_t *__restrict__ input_shapes,      // 输入形状
    const ptrdiff_t *__restrict__ output_strides, // 输出步幅
    const ptrdiff_t *__restrict__ input_strides,  // 输入步幅
    Tdata *output,                       // 输出缓冲区
    const void *const *inputs,           // 输入指针数组
    size_t offset,                       // 线性偏移量（用于分段执行）
    Args... args);                       // 额外参数（GELU 不使用）
```

## 4. 使用示例

```cpp
#include "gelu_nvidia.cuh"
#include "../../utils.h"

// 1. 创建 InfiniOp 句柄
infiniopHandle_t handle;
infiniopCreateHandle(&handle, 0); // 设备 ID = 0

// 2. 准备张量描述符（假设输入为 [1024, 1024] FP32 张量）
int64_t shape[] = {1024, 1024};
int64_t strides[] = {1024, 1};

infiniopTensorDescriptor_t input_desc, output_desc;
infiniopCreateTensorDescriptor(&input_desc, INFINI_DTYPE_F32, 2, shape, strides);
infiniopCreateTensorDescriptor(&output_desc, INFINI_DTYPE_F32, 2, shape, strides);

// 3. 创建 GELU 描述符
op::gelu::nvidia::Descriptor *gelu_desc;
auto status = op::gelu::nvidia::Descriptor::create(
    handle,
    &gelu_desc,
    output_desc,
    {input_desc});

if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
    return;
}

// 4. 分配设备内存
size_t numel = 1024 * 1024;
float *d_input, *d_output;
cudaMalloc(&d_input, numel * sizeof(float));
cudaMalloc(&d_output, numel * sizeof(float));

// 5. 准备工作空间
size_t workspace_size = gelu_desc->workspaceSize();
void *d_workspace;
cudaMalloc(&d_workspace, workspace_size);

// 6. 拷贝输入数据到设备
cudaMemcpy(d_input, h_input, numel * sizeof(float), cudaMemcpyHostToDevice);

// 7. 创建 CUDA 流并执行计算
cudaStream_t stream;
cudaStreamCreate(&stream);

status = gelu_desc->calculate(
    d_workspace,
    workspace_size,
    d_output,
    {d_input},
    stream);

if (status != INFINI_STATUS_SUCCESS) {
    // 处理计算错误
}

// 8. 同步并拷贝结果回主机
cudaStreamSynchronize(stream);
cudaMemcpy(h_output, d_output, numel * sizeof(float), cudaMemcpyDeviceToHost);

// 9. 清理资源
delete gelu_desc;
infiniopDestroyTensorDescriptor(input_desc);
infiniopDestroyTensorDescriptor(output_desc);
cudaFree(d_input);
cudaFree(d_output);
cudaFree(d_workspace);
cudaStreamDestroy(stream);
infiniopDestroyHandle(handle);
```

## 5. 实现细节

### 内存管理策略
- **元数据存储**: 使用统一的 `ElementwiseInfo` 结构体存储所有张量的形状、步幅和标志位
- **工作空间计算**:
  - `workspace_size = info.getMetaMemSize() + info.getInputSize() * sizeof(void*)`
  - 元数据部分对齐到 size_t 边界
  - 输入指针数组存储在工作空间起始位置
- **设备端内存传输**:
  - 使用 `cudaMemcpyAsync` 异步拷贝元数据和输入指针数组
  - 指针偏移计算：`d_meta_start = workspace + input_arr_size`

### 并发执行模型
- **CUDA 线程配置**:
  - 每个线程块固定 256 线程（编译时常量）
  - 网格维度动态计算：`gridDims = min(CEIL_DIV(output_size, 256), device_grid_size_x)`
  - 支持的最大网格大小由设备属性决定（`internal->gridSizeX()`）
- **分段执行**:
  - 对于大型张量（output_size > gridDims * blockDims），循环多次启动核函数
  - 每次传递不同的 `offset` 值以处理不同的数据段
  - 步长：`step = gridDims.x * blockDims.x`
- **线程安全**:
  - 每个线程处理一个输出元素（`idx = blockIdx.x * blockDim.x + threadIdx.x + offset`）
  - 无竞争条件，无需原子操作

### 性能优化技术
1. **编译时模板特化**:
   - 为每种数据类型生成专用核函数，避免运行时分支
   - 使用 `if constexpr` 在编译期消除死代码
2. **强制内联**:
   - `__device__ __forceinline__` 确保 `GeluOp::operator()` 被完全内联
3. **类型转换优化**:
   - BF16/FP16 在 float 精度下计算（精度与速度的平衡）
   - 使用 CUDA 内置转换函数（`__bfloat162float`, `__half2float`）
4. **内存访问优化**:
   - 连续张量使用线性索引（`is_contiguous ? idx : indexToOffset(...)`）
   - 广播张量通过 `InputIndexer` 计算偏移
5. **数学函数优化**:
   - 使用 CUDA 标准库的 `erf()` 函数（设备端优化实现）
   - 预计算常量 `sqrt(2.0f)` 在编译期优化

### 错误处理机制
- **类型验证**:
  - `CHECK_DTYPE` 宏确保数据类型在支持列表中
  - 不支持的类型返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
- **形状验证**:
  - `CHECK_SAME_SHAPE` 确保输入和输出形状完全一致
  - 广播通过 `input_broadcasted` 标志在运行时处理
- **工作空间验证**:
  - 运行时检查 `workspace_size >= _workspace_size`
  - 不足时返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
- **CUDA 错误传播**:
  - `CHECK_CUDA` 宏将 `cudaError_t` 转换为 `infiniStatus_t`
  - 内核启动失败会传播到上层调用者

### 设计模式应用
1. **策略模式（Strategy Pattern）**:
   - `DeviceImpl` 封装不同的计算策略（相同输入类型 vs 混合输入类型）
2. **模板方法模式（Template Method Pattern）**:
   - `calculate()` 定义算法骨架，`calculateImpl()` 提供具体实现
3. **工厂模式（Factory Pattern）**:
   - `Descriptor::create()` 作为静态工厂方法构造描述符
4. **Pimpl 模式（Pointer to Implementation）**:
   - `DeviceImpl::Opaque` 隐藏实现细节，减少编译依赖

### 依赖关系
- **内部依赖**:
  - `../../../elementwise/nvidia/elementwise_nvidia.cuh`: 逐元素操作 CUDA 框架
  - `../cuda/kernel.cuh`: GELU 核函数实现
  - `../../devices/nvidia/nvidia_common.cuh`: NVIDIA 设备通用定义
  - `../../utils.h`: 工具类和错误处理宏
- **外部依赖**:
  - CUDA Runtime API（`cudaMemcpyAsync`, `cudaMemcpy`）
  - CUDA 标准库（`erf`, `sqrt`）
  - C++ 标准库（`vector`, `memory`, `cmath`）

### 数值特性
- **GELU 公式**:
  - 精确实现：`GELU(x) = 0.5 * x * (1 + erf(x / √2))`
  - 近似实现：未使用（可选项为 `x * sigmoid(1.702 * x)`）
- **精度保证**:
  - BF16/FP16: 在 float 精度下计算，避免累积误差
  - FP32/FP64: 直接在对应精度下计算
- **边界情况**:
  - 输入为 0：输出为 0（`GELU(0) = 0`）
  - 大正数：趋近于 `x`（`GELU(+∞) ≈ x`）
  - 大负数：趋近于 0（`GELU(-∞) ≈ 0`）

### 限制与约束
1. **形状要求**: 输入和输出张量形状必须完全相同（不支持广播）
2. **数据类型**: 仅支持浮点类型（BF16、FP16、FP32、FP64）
3. **设备限制**:
   - 最大线程块大小：`maxThreadsPerBlock`（通常为 1024）
   - 最大网格 X 维度：`gridSizeX`（通常为 2^31 - 1）
4. **工作空间**: 必须在设备端预先分配足够的连续内存

### 可扩展性
- **添加新数据类型**:
  1. 在 `calculate()` 中添加新的 case 分支
  2. 在 `GeluOp::operator()` 中添加对应的 `if constexpr` 分支
- **支持批量操作**:
  - 当前实现已通过 `InputIndexer` 支持多输入（虽然 GELU 只用 1 个）
  - 可扩展为逐元素的多元操作
- **自定义激活函数**:
  - 复制 `kernel.cuh` 并实现新的 `Op` 函数对象
  - 使用相同的 `ELEMENTWISE_DESCRIPTOR` 宏生成描述符

---

**关键文件路径**:
- `/home/qy/src/Infini/InfiniCore/src/infiniop/ops/gelu/nvidia/gelu_nvidia.cuh`
- `/home/qy/src/Infini/InfiniCore/src/infiniop/ops/gelu/nvidia/gelu_nvidia.cu`
- `/home/qy/src/Infini/InfiniCore/src/infiniop/ops/gelu/cuda/kernel.cuh`
- `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/nvidia/elementwise_nvidia.cuh`
- `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/nvidia/elementwise_nvidia_api.cuh`
- `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/elementwise.h`
