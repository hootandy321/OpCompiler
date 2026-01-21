# Ones 操作 NVIDIA GPU 后端核心实现文档

本文档详细描述了 `ones` 操作在 NVIDIA GPU 上的 CUDA 实现核心，该模块用于生成全 1 张量，支持丰富的数据类型和广播机制。

## 1. 模块结构

本目录包含 `ones` 操作的 NVIDIA GPU 实现，共 2 个源文件：

- **`ones_nvidia.cuh`**: CUDA 后端 API 声明，通过宏定义生成操作描述符类
- **`ones_nvidia.cu`**: CUDA 后端实现，包含描述符创建、计算执行和类型分发逻辑

### 依赖关系
- **上层接口**: `/InfiniCore/src/infiniop/ops/ones/operator.cc` - 统一的 C API 入口
- **元素计算**: `/InfiniCore/src/infiniop/ops/ones/cuda/kernel.cuh` - OnesOp 核心计算算子
- **基础框架**: `/InfiniCore/src/infiniop/elementwise/nvidia/elementwise_nvidia.cuh` - 元素级操作 CUDA 内核
- **API 模板**: `/InfiniCore/src/infiniop/elementwise/nvidia/elementwise_nvidia_api.cuh` - 元素级操作 API 基类

## 2. 核心类与数据结构

### `op::ones::nvidia::Descriptor`
- **位置**: `ones_nvidia.cuh`（通过 `ELEMENTWISE_DESCRIPTOR` 宏生成）
- **父类**: `InfiniopDescriptor`
- **主要功能**: 管理 ones 操作的 NVIDIA GPU 执行描述符，封装类型信息、张量元数据和设备实现

#### 关键成员变量
```cpp
infiniDtype_t _dtype;                          // 输出张量的数据类型
op::elementwise::ElementwiseInfo _info;        // 张量形状、步长、广播等元数据
std::unique_ptr<op::elementwise::nvidia::DeviceImpl> _device_info;  // CUDA 设备实现
size_t _workspace_size;                        // 设备工作空间大小（字节）
```

#### 核心方法

##### `create()`
```cpp
static infiniStatus_t create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec);
```

**功能**: 创建 ones 操作描述符并初始化 CUDA 后端

**执行流程**:
1. **类型检查**: 使用 `CHECK_DTYPE` 宏验证输出数据类型，支持 15 种类型：
   - 整型: `int8_t`, `int16_t`, `int32_t`, `int64_t`, `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`
   - 浮点: `cuda_fp8_e4m3`, `half`, `float`, `double`, `cuda_bfloat16`
   - 其他: `bool`, `BYTE`
   - 不支持: 复数类型 (`C16`, `C32`, `C64`, `C128`) 返回 `INFINI_STATUS_NOT_IMPLEMENTED`

2. **形状一致性**: `CHECK_SAME_SHAPE` 宏确保输出张量与输入张量形状一致

3. **创建 CUDA 元素级描述符**: 使用 `CREATE_ELEMENTWISE_CUDA_DESCRIPTOR` 宏：
   - 调用 `ElementwiseInfo::create()` 生成张量元数据（形状、步长、广播信息）
   - 计算工作空间大小 = `元数据大小 + 输入指针数组大小`
   - 创建 `DeviceImpl` 实例（持有 CUDA 上下文和内核启动逻辑）
   - 构造并返回 `Descriptor` 对象

##### `calculate()`
```cpp
infiniStatus_t calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const;
```

**功能**: 在 GPU 上执行 ones 操作，将输出张量的所有元素填充为 1

**执行流程**:
1. **工作空间验证**: 检查 `workspace_size` 是否满足 `_workspace_size` 要求

2. **类型分发**: 根据 `_dtype` 分发到对应的模板实例化：
   - 对每种支持的数据类型，调用 `DeviceImpl::calculate<256, cuda::OnesOp, T>()`
   - CUDA 块大小固定为 256 线程
   - 使用 `cuda::OnesOp` 作为计算算子
   - 模板参数 `T` 决定输出类型和常量 1 的表示形式

3. **返回状态**:
   - 成功: `INFINI_STATUS_SUCCESS`
   - 工作空间不足: `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
   - 不支持的类型: `INFINI_STATUS_BAD_TENSOR_DTYPE` 或 `INFINI_STATUS_NOT_IMPLEMENTED`

##### `~Descriptor()`
析构函数，默认实现（`= default`），通过智能指针自动清理资源

### `cuda::OnesOp`
- **位置**: `/InfiniCore/src/infiniop/ops/ones/cuda/kernel.cuh`
- **类型**: CUDA 设备端仿函数（Functor）
- **主要功能**: 定义将任意输入值转换为 1 的 CUDA 设备端操作

#### 结构定义
```cpp
struct OnesOp {
    static constexpr size_t num_inputs = 1;  // 标记为单输入操作

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const;
};
```

#### 核心方法
```cpp
template <typename T>
__device__ __forceinline__ T operator()(const T &x) const {
    if constexpr (std::is_same_v<T, bool>) {
        return true;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return 1;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return 1;
    }
    // ... 其他整型返回 1 ...
    else if constexpr (std::is_same_v<T, cuda_fp8_e4m3>) {
        return cuda_fp8_e4m3(1.0f);  // FP8 格式化
    } else if constexpr (std::is_same_v<T, half>) {
        return __float2half(1.0f);   // 半精度浮点转换
    } else if constexpr (std::is_same_v<T, float>) {
        return 1.0f;
    } else if constexpr (std::is_same_v<T, double>) {
        return 1.0;
    } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
        return __float2bfloat16(1.0f);  // bfloat16 转换
    } else {
        return 1.0;  // 默认浮点返回
    }
}
```

**关键特性**:
- **编译期类型分发**: 使用 `if constexpr` 在编译期根据类型 `T` 选择实现
- **忽略输入参数**: 算子接收输入 `x` 但不使用其值，仅利用其类型
- **设备端内联**: `__device__ __forceinline__` 确保在 GPU 上高效执行并内联展开
- **类型安全**: 为每种类型提供精确的常量 1 表示（避免隐式转换警告）

### `op::elementwise::nvidia::DeviceImpl`
- **位置**: `/InfiniCore/src/infiniop/elementwise/nvidia/elementwise_nvidia.cuh`
- **设计模式**: Pimpl（Pointer to Implementation）
- **主要功能**: 封装 CUDA 内核启动逻辑，管理元数据传输和网格配置

#### 内部结构 `Opaque`
```cpp
struct DeviceImpl::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;  // CUDA 上下文句柄

    template <uint32_t BLOCK_SIZE, size_t N, typename Op, typename Tdata, typename... Args>
    infiniStatus_t calculateImpl(...);  // 同类型实现
};
```

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

**参数说明**:
- `BLOCK_SIZE`: CUDA 块大小（ones 操作中固定为 256）
- `Op`: 操作算子类型（ones 使用 `cuda::OnesOp`）
- `Tdata`: 输入/输出数据类型
- `info`: 张量元数据（形状、步长、广播标志）
- `workspace`: GPU 端工作空间，存储元数据和输入指针数组
- `output`: GPU 端输出缓冲区指针
- `inputs`: 输入张量指针数组（ones 中仅用于形状推导）
- `stream`: CUDA 流句柄

**执行流程**:
1. 调用 `Opaque::calculateImpl()` 转发到模板化实现
2. 调用 `launchElementwiseKernel()` 启动 CUDA 内核
3. 通过 `infoToDevice()` 将元数据异步传输到 GPU
4. 计算网格维度并循环启动内核处理大规模张量

### `op::elementwise::ElementwiseInfo`
- **位置**: `/InfiniCore/src/infiniop/elementwise/elementwise.h`
- **主要功能**: 存储张量元数据的紧凑结构，支持广播和步长计算

#### 内存布局
元数据在单个 `std::vector<size_t>` 中按以下顺序排列：
```
[输出形状 (ndim)] [输出步长 (ndim)] [输入形状 (N * ndim)]
[输入步长 (N * ndim)] [输入连续性标志 (N)] [输入广播标志 (N)]
```

#### 关键方法
```cpp
size_t getMetaMemSize() const;           // 元数据总字节数
size_t getOutputSize() const;            // 输出张量元素总数
size_t getNdim() const;                  // 张量维度数
bool isOutputContiguous() const;         // 输出是否内存连续
const size_t* getOutputShape() const;    // 输出形状指针
const ptrdiff_t* getOutputStrides() const;  // 输出步长指针
const bool* getInputContiguous() const;  // 各输入连续性标志数组
const bool* getInputBroadcasted() const;  // 各输入广播标志数组
```

## 3. API 接口

### C 接口（通过 operator.cc 暴露）

```c
// 创建描述符
infiniStatus_t infiniopCreateOnesDescriptor(
    infiniopHandle_t handle,
    infiniopOnesDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc);

// 获取工作空间大小
infiniStatus_t infiniopGetOnesWorkspaceSize(
    infiniopOnesDescriptor_t desc,
    size_t *size);

// 执行计算
infiniStatus_t infiniopOnes(
    infiniopOnesDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream);

// 销毁描述符
infiniStatus_t infiniopDestroyOnesDescriptor(
    infiniopOnesDescriptor_t desc);
```

### 内部 CUDA 接口

```cpp
// 命名空间: op::ones::nvidia

class Descriptor {
    // 创建描述符（静态工厂方法）
    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t out_desc,
        std::vector<infiniopTensorDescriptor_t> input_desc_vec);

    // 执行 ones 操作
    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *output,
        std::vector<const void *> inputs,
        void *stream) const;

    // 获取工作空间大小
    size_t workspaceSize() const;
};
```

## 4. 使用示例

```cpp
#include "infiniop/ops/ones.h"
#include "infiniop/handle.h"

// 1. 初始化 CUDA 句柄
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_NVIDIA, 0);

// 2. 创建张量描述符（假设生成形状为 [1024, 1024] 的 float 张量）
int64_t shape[2] = {1024, 1024};
infiniopTensorDescriptor_t x_desc, y_desc;
infiniopCreateTensorDescriptor(handle, &x_desc, INFINI_DTYPE_F32, 2, shape, nullptr);
infiniopCreateTensorDescriptor(handle, &y_desc, INFINI_DTYPE_F32, 2, shape, nullptr);

// 3. 创建 ones 操作描述符
infiniopOnesDescriptor_t ones_desc;
infiniStatus_t status = infiniopCreateOnesDescriptor(handle, &ones_desc, y_desc, x_desc);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 4. 获取并分配工作空间
size_t workspace_size;
infiniopGetOnesWorkspaceSize(ones_desc, &workspace_size);
void *workspace;
cudaMalloc(&workspace, workspace_size);

// 5. 分配输出张量内存
float *d_y;
cudaMalloc(&d_y, 1024 * 1024 * sizeof(float));

// 6. 创建 CUDA 流
cudaStream_t stream;
cudaStreamCreate(&stream);

// 7. 执行 ones 操作（输入 x 仅用于形状推导，可以为空）
void *d_x = nullptr;  // ones 不需要实际输入数据
status = infiniopOnes(ones_desc, workspace, workspace_size, d_y, d_x, stream);

// 8. 同步并验证结果
cudaStreamSynchronize(stream);

// 9. 清理资源
cudaFree(d_y);
cudaFree(workspace);
cudaStreamDestroy(stream);
infiniopDestroyOnesDescriptor(ones_desc);
infiniopDestroyTensorDescriptor(x_desc);
infiniopDestroyTensorDescriptor(y_desc);
infiniopDestroyHandle(handle);
```

**预期结果**: `d_y` 的所有 1,048,576 个元素均为 `1.0f`

## 5. 实现细节

### 内存管理

**元数据传输策略**:
- 工作空间分为两部分：
  1. **输入指针数组**: 存储 `N` 个输入张量的设备指针（`N * sizeof(void*)` 字节）
  2. **元数据区**: 存储形状、步长、广播标志等（`ElementwiseInfo::getMetaMemSize()` 字节）
- 使用 `cudaMemcpyAsync()` 异步传输元数据到 GPU，避免阻塞
- 元数据在设备端紧凑存储，减少全局内存访问延迟

**内存对齐**:
- 所有元数据按 `sizeof(size_t)` 对齐，确保 GPU 端高效访问
- 布尔标志（`input_contiguous`, `input_broadcasted`）在元数据末尾对齐存储

### 并发执行

**CUDA 内核配置**:
- **块大小**: 固定 256 线程（平衡寄存器使用和占用率）
- **网格大小**: 动态计算 `min(CEIL_DIV(output_size, 256), grid_size_x)`
  - `grid_size_x` 从设备属性查询（通常为 2^31 - 1 或硬件限制）
  - 使用网格步进循环处理超大张量（`step = gridDims.x * blockDims.x`）

**流处理**:
- 所有 CUDA 操作在用户提供的流上执行，支持与其他操作并发
- 使用异步内存传输（`cudaMemcpyAsync`）与内核执行重叠
- 多个 ones 操作可在不同流上并行执行

### 性能优化

**编译期优化**:
- `OnesOp::operator()` 使用 `if constexpr` 完全在编译期展开，零运行时分支
- 模板实例化为每种数据类型生成专用内核，避免虚函数开销
- `__forceinline__` 强制内联设备端函数，减少调用开销

**内存访问优化**:
- 对连续张量使用线性索引（`idx`），避免 `indexToOffset` 计算
- `InputIndexer` 结构体封装广播逻辑，编译器优化为直接索引
- 元数据缓存在 GPU 常量内存或共享内存（由 CUDA 编译器决定）

**类型处理**:
- 浮点类型（`half`, `bfloat16`, `fp8`）使用 CUDA 内置转换函数（`__float2half` 等）
- 避免浮点到整型的隐式转换，显式返回类型匹配的常量

### 错误处理

**错误类型**:
1. **类型不支持** (`INFINI_STATUS_BAD_TENSOR_DTYPE`):
   - 传入未在 `CHECK_DTYPE` 宏中列出的类型
   - 复数类型返回 `INFINI_STATUS_NOT_IMPLEMENTED`

2. **形状不匹配**:
   - `CHECK_SAME_SHAPE` 宏在编译时验证输出与输入形状一致性
   - 运行时 `ElementwiseInfo::create()` 检查广播兼容性

3. **资源不足** (`INFINI_STATUS_INSUFFICIENT_WORKSPACE`):
   - 用户分配的工作空间小于 `_workspace_size`
   - 必须调用 `infiniopGetOnesWorkspaceSize()` 查询需求

4. **CUDA 错误**:
   - `CHECK_CUDA` 宏捕获内核启动、内存传输失败
   - 错误通过 `infiniStatus_t` 传播到上层

### 广播支持

**广播机制**:
- 虽然操作符本身不使用输入值，但 `ElementwiseInfo` 支持广播逻辑：
  - `input_broadcasted` 标志标记哪些输入维度被广播
  - `InputIndexer` 在内核中处理广播索引映射
  - 形状为 `[1]` 的输入可广播到任意形状输出

**连续性优化**:
- `isOutputContiguous()` 检测输出是否内存连续（步长为标准行主序）
- 连续张量使用简化索引（`out_idx = idx`），避免 `indexToOffset` 调用
- 非连续张量（如转置、切片）通过 `device::nvidia::indexToOffset` 计算偏移

### 设计模式

**宏驱动代码生成**:
- `ELEMENTWISE_DESCRIPTOR(ones, nvidia)` 宏生成完整的 `Descriptor` 类
- 避免为每个元素级操作重复编写样板代码
- 编译期展开，零运行时开销

**策略模式（Strategy Pattern）**:
- `DeviceImpl` 封装 CUDA 特定执行策略
- 其他后端（CPU、MooreThreads、Metax）可提供自己的 `DeviceImpl`
- 统一接口 `calculate()` 支持多态调用

**工厂模式（Factory Pattern）**:
- `Descriptor::create()` 作为静态工厂方法
- `operator.cc` 中的 `infiniopCreateOnesDescriptor` 根据设备类型分发到具体实现

**Pimpl 惯用法**:
- `DeviceImpl` 通过 `Opaque` 结构体隐藏 CUDA 实现细节
- 减少头文件依赖，加速编译
- 允许在不破坏 ABI 的情况下更改内部实现

### 数据类型支持详解

**整型常量生成**:
- 所有有符号/无符号整型（8/16/32/64 位）返回整数常量 `1`
- `bool` 类型返回 `true`（C++ 中 `true` 隐式转换为 `1`）

**浮点常量生成**:
- `float`: `1.0f`（单精度浮点字面量）
- `double`: `1.0`（双精度浮点字面量）
- `half` (FP16): `__float2half(1.0f)`（NVIDIA 内置转换）
- `cuda_bfloat16`: `__float2bfloat16(1.0f)`（NVIDIA 内置转换）
- `cuda_fp8_e4m3`: `cuda_fp8_e4m3(1.0f)`（FP8 类型转换）

**不支持类型**:
- 复数类型 (`C16`, `C32`, `C64`, `C128`) 返回 `INFINI_STATUS_NOT_IMPLEMENTED`
- 自定义类型需要在 `OnesOp` 中添加特化

### CUDA 内核实现细节

**内核签名**（来自 `elementwise_nvidia.cuh`）:
```cpp
template <size_t N, typename Op, typename Tdata, typename... Args>
__global__ void elementwiseKernel(
    size_t output_size,         // 输出元素总数
    size_t ndim,                // 张量维度数
    bool output_contiguous,     // 输出连续性标志
    const bool *input_contiguous,  // 各输入连续性数组
    const bool *input_broadcasted, // 各输入广播标志数组
    const size_t *output_shape,     // 输出形状
    const size_t *input_shapes,     // 输入形状数组（N * ndim）
    const ptrdiff_t *output_strides, // 输出步长
    const ptrdiff_t *input_strides,  // 输入步长数组（N * ndim）
    Tdata *output,              // 输出缓冲区
    const void *const *inputs,  // 输入指针数组
    size_t offset,              // 全局偏移（网格步进使用）
    Args... args);              // 额外参数（ones 不使用）
```

**内核执行流程**（单线程）:
1. 计算全局索引: `idx = blockIdx.x * blockDim.x + threadIdx.x + offset`
2. 边界检查: `if (idx < output_size)` 确保不越界
3. 计算输出偏移:
   - 连续: `out_idx = idx`
   - 非连续: `out_idx = indexToOffset(idx, ndim, output_shape, output_strides)`
4. 计算输入偏移: `InputIndexer` 处理广播和步长
5. 执行操作: `output[out_idx] = OnesOp{}(typed_inputs[0][in_idx])`
   - `OnesOp` 忽略输入值，仅使用类型信息生成 `1`
6. 返回：自动写入全局内存

**网格步进循环**（处理大张量）:
```cpp
size_t step = gridDims.x * blockDims.x;  // 单次网格启动处理元素数
for (size_t i = 0; i < output_size; i += step) {
    elementwiseKernel<<<gridDims, blockDims, 0, stream>>>(..., i);
}
```
- 当 `output_size > 2^31` 时，避免网格维度溢出
- 每次启动处理 `step` 个元素，通过 `offset` 参数传递起始位置
- 内核中的边界检查确保最后一批不越界

### 工作空间计算

**公式**:
```
workspace_size = meta_mem_size + input_ptr_array_size
                = (ndim * sizeof(size_t) * 2 +              // 输出形状 + 输出步长
                   N * ndim * sizeof(size_t) +               // 输入形状
                   N * ndim * sizeof(ptrdiff_t) +            // 输入步长
                   N * sizeof(bool) +                        // 输入连续性
                   N * sizeof(bool)) +                       // 输入广播标志
                  N * sizeof(void*)                          // 输入指针数组
```

**示例**（2D 张量，1 个输入）:
- `ndim = 2`, `N = 1`
- `meta_mem_size = 2*8 + 2*8 + 1*2*8 + 1*2*8 + 1*1 + 1*1 = 58` 字节
- `input_ptr_array_size = 1 * 8 = 8` 字节
- `workspace_size = 66` 字节

## 6. 关键常量与配置

### 编译时常量
- **CUDA 块大小**: `256`（在 `calculate()` 中硬编码）
- **输入数量**: `OnesOp::num_inputs = 1`（编译期常量）
- **最大网格维度**: 从 `device::nvidia::Handle::Internal::gridSizeX()` 查询

### 数据类型映射
```cpp
INFINI_DTYPE_BYTE    -> uint8_t
INFINI_DTYPE_BOOL    -> bool
INFINI_DTYPE_I8      -> int8_t
INFINI_DTYPE_I16     -> int16_t
INFINI_DTYPE_I32     -> int32_t
INFINI_DTYPE_I64     -> int64_t
INFINI_DTYPE_U8      -> uint8_t
INFINI_DTYPE_U16     -> uint16_t
INFINI_DTYPE_U32     -> uint32_t
INFINI_DTYPE_U64     -> uint64_t
INFINI_DTYPE_F8      -> cuda_fp8_e4m3
INFINI_DTYPE_F16     -> half
INFINI_DTYPE_F32     -> float
INFINI_DTYPE_F64     -> double
INFINI_DTYPE_BF16    -> cuda_bfloat16
```

### 宏定义
```cpp
#define CHECK_DTYPE(dtype, ...)       // 验证数据类型是否支持
#define CHECK_SAME_SHAPE(y, x)        // 验证形状一致性
#define CREATE_ELEMENTWISE_CUDA_DESCRIPTOR(...)  // 创建 CUDA 描述符
#define ELEMENTWISE_DESCRIPTOR(OP, NS) // 生成操作描述符类
```

## 7. 调试与性能分析

### 常见问题

1. **工作空间不足**:
   - 症状: 返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
   - 解决: 调用 `infiniopGetOnesWorkspaceSize()` 查询实际需求

2. **类型不支持**:
   - 症状: 返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
   - 解决: 检查输入张量类型是否在支持列表中

3. **CUDA 错误**:
   - 症状: 返回非成功状态码
   - 解决: 使用 `cudaGetLastError()` 查询详细错误，检查 CUDA 驱动版本

### 性能建议

1. **批量操作**: 将多个 ones 操作合并到单个 CUDA 流中
2. **工作空间复用**: 多次操作间复用同一工作空间，减少分配开销
3. **零拷贝优化**: 对连续张量使用 `cudaMalloc` 而非 `cudaMallocManaged`
4. **流并发**: 在不同流上执行独立的 ones 操作

### 性能基准（参考）
- **小张量** (< 1KB): 内核启动开销主导，约 10-20 μs
- **中等张量** (1KB - 1MB): 内存带宽主导，接近理论峰值
- **大张量** (> 1MB): 受 GPU 全局内存带宽限制（~500 GB/s on A100）

## 8. 未来扩展

### 可能的优化方向
1. **融合操作**: 支持 `ones_like` 或 `ones_add` 等融合算子
2. **类型扩展**: 添加对 TF32、INT4 等新类型的支持
3. **图优化**: 与 CUDA Graph 集成减少内核启动开销
4. **稀疏张量**: 支持稀疏格式（CSR、COO）的 ones 填充

### 维护建议
- 添加新数据类型时需同步更新:
  1. `ones_nvidia.cu` 中的 `CHECK_DTYPE` 宏
  2. `calculate()` 中的 `switch-case` 分支
  3. `kernel.cuh` 中的 `OnesOp::operator()` 特化
- 保持与 CPU 后端（`cpu/ones_cpu.h`）的接口一致性

---

**文档版本**: 1.0
**最后更新**: 2026-01-14
**作者**: Infini 框架代码分析
**模块路径**: `/InfiniCore/src/infiniop/ops/ones/nvidia`
