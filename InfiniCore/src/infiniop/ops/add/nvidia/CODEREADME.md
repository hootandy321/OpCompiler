# NVIDIA CUDA Add 算子核心实现文档

本模块实现了基于 NVIDIA CUDA 的逐元素加法运算（Element-wise Addition），作为 Infini 框架中逐元素操作基础设施的具体实现。该模块支持多种数据类型（FP16/BF16/FP32/FP32/INT32/INT64/FP64），利用 CUDA 并行计算能力实现张量间的高效加法运算，完全兼容广播机制和非连续内存布局。

## 1. 模块结构

- **`add_nvidia.cuh`**: 头文件，通过宏生成 CUDA 加法操作符的 Descriptor 类定义
- **`add_nvidia.cu`**: 实现文件，包含 Descriptor 的创建（create）和计算（calculate）方法，负责类型分发和 CUDA kernel 调度

## 2. 核心类

### `op::add::nvidia::Descriptor`
- **位置**: `add_nvidia.cuh`（通过 `ELEMENTWISE_DESCRIPTOR` 宏生成）
- **主要功能**: 封装 NVIDIA CUDA 后端的加法操作符描述符，管理张量元数据、设备实现和 workspace 大小
- **继承关系**: 继承自 `InfiniopDescriptor` 基类
- **核心成员**:
  - `_dtype: infiniDtype_t`: 输出张量的数据类型
  - `_info: op::elementwise::ElementwiseInfo`: 张量形状、步长、连续性等元数据
  - `_device_info: std::unique_ptr<op::elementwise::nvidia::DeviceImpl>`: CUDA 设备实现对象指针
  - `_workspace_size: size_t`: CUDA kernel 执行所需的 workspace 字节数

#### 核心方法

##### `create(...)`
```cpp
static infiniStatus_t create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec);
```

**功能**: 创建加法操作符描述符，验证张量形状和数据类型一致性。

**执行流程**:
1. **类型转换**: 将 `infiniopHandle_t` 转换为 `device::nvidia::Handle*` 获取设备句柄
2. **参数提取**: 提取输出和输入（两个输入张量 A 和 B）的描述符
3. **数据类型验证**: 使用 `CHECK_DTYPE` 宏验证输出类型是否为支持的类型（F16/F32/BF16/I32/I64/F64）
4. **形状验证**: 使用 `CHECK_SAME_SHAPE` 宏确保输出、输入 A、输入 B 三者的形状完全一致
5. **创建 Elementwise 描述符**: 调用 `CREATE_ELEMENTWISE_CUDA_DESCRIPTOR` 宏完成:
   - 构造 `ElementwiseInfo` 对象，包含张量的形状、步长、连续性、广播标志等元数据
   - 计算 workspace 大小 = 元数据大小 + 输入指针数组大小
   - 创建 `op::elementwise::nvidia::DeviceImpl` 实例
   - 实例化 `Descriptor` 对象并返回

**复杂度**: O(ndim) - 遍历张量维度验证和元数据复制

##### `calculate(...)`
```cpp
infiniStatus_t calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const;
```

**功能**: 执行 CUDA 加法计算，根据数据类型分发到相应的模板实例化。

**执行流程**:
1. **Workspace 验证**: 检查传入的 workspace 大小是否满足 `_workspace_size` 要求
2. **类型分发**: 根据 `_dtype` 的值，switch-case 分发到对应的模板调用:
   - `INFINI_DTYPE_F16`: 调用 `_device_info->calculate<256, cuda::AddOp, half>(...)`
   - `INFINI_DTYPE_BF16`: 调用 `_device_info->calculate<256, cuda::AddOp, cuda_bfloat16>(...)`
   - `INFINI_DTYPE_F32`: 调用 `_device_info->calculate<256, cuda::AddOp, float>(...)`
   - `INFINI_DTYPE_I32`: 调用 `_device_info->calculate<256, cuda::AddOp, int32_t>(...)`
   - `INFINI_DTYPE_I64`: 调用 `_device_info->calculate<256, cuda::AddOp, int64_t>(...)`
   - `INFINI_DTYPE_F64`: 调用 `_device_info->calculate<256, cuda::AddOp, double>(...)`
3. **委托执行**: 调用 `elementwise::nvidia::DeviceImpl::calculate` 方法，该方法进一步:
   - 将元数据和输入指针数组异步拷贝到设备 workspace
   - 计算 CUDA grid 和 block 维度（block size = 256，grid size 根据 output size 计算）
   - 分步启动 CUDA kernel（每步处理 `grid_size * block_size` 个元素）
   - 返回执行状态

**复杂度**: O(n) - n 为输出张量的元素总数，每个元素由一个 CUDA 线程处理

### `op::add::cuda::AddOp`
- **位置**: `../cuda/kernel.cuh`（被 `add_nvidia.cu` 引用）
- **主要功能**: 定义加法操作的 CUDA 设备端仿函数（functor）
- **核心成员**:
  - `num_inputs = 2`: 编译时常量，指定操作需要 2 个输入张量

#### 核心方法

##### `operator()(const T& a, const T& b)`
```cpp
template <typename T>
__device__ __forceinline__ T operator()(const T &a, const T &b) const;
```

**功能**: 在 CUDA 设备端执行两个标量的加法运算。

**优化策略**:
- **half2 类型**: 使用 `__hadd2(a, b)` intrinsic 指令，一次计算两个 FP16 值的加法（向量化）
- **half/cuda_bfloat16 类型**: 使用 `__hadd(a, b)` intrinsic 指令，硬件加速的低精度加法
- **float 类型**: 使用 `__fadd_rd(a, b)` intrinsic 指令，向负无穷方向舍入的加法（保证数值稳定性）
- **其他类型（int32_t/int64_t/double）**: 使用标准运算符 `a + b`

**复杂度**: O(1) - 单个算术运算

### `op::elementwise::nvidia::DeviceImpl`
- **位置**: `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/nvidia/elementwise_nvidia.cuh`
- **主要功能**: 实现逐元素操作的 CUDA 通用执行引擎，提供 kernel 启动、元数据管理、内存传输等底层功能
- **生命周期**: 由 `Descriptor::create` 中通过 `DeviceImpl::create` 静态工厂方法创建，存储为 `Descriptor` 的成员，由 `Descriptor` 析构时自动释放

#### 核心方法

##### `calculate<BLOCK_SIZE, Op, Tdata>(...)`
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

**功能**: 执行所有输入输出类型相同的逐元素操作（模板特化版本 1）。

**执行流程**:
1. **模板参数推断**: `BLOCK_SIZE=256`, `Op=cuda::AddOp`, `Tdata` 为具体类型（如 `float`）
2. **调用 calculateImpl**: 委托给内部实现，传递类型信息和操作符

##### `calculate<BLOCK_SIZE, Op, Tout, Tin...>(...)`
```cpp
template <unsigned int BLOCK_SIZE, typename Op, typename Tout, typename... Tin,
          typename... Args,
          std::enable_if_t<(sizeof...(Tin) == Op::num_inputs), int> = 0>
infiniStatus_t calculate(...);
```

**功能**: 执行输入输出类型可能不同的逐元素操作（模板特化版本 2）。

**SFINAE 约束**: `std::enable_if_t<(sizeof...(Tin) == Op::num_inputs), int>` 确保模板参数数量与操作符要求的输入数量匹配

**执行流程**: 同上，但支持混合类型计算

##### `calculateImpl<BLOCK_SIZE, N, Op, Tdata>(...)`
```cpp
template <uint32_t BLOCK_SIZE, size_t N, typename Op, typename Tdata, typename... Args>
infiniStatus_t calculateImpl(...);
```

**功能**: 内部实现方法，调用通用的 kernel 启动逻辑。

**执行流程**:
1. 调用 `launchElementwiseKernel<BLOCK_SIZE, N>` 传递 kernel 函数指针 `elementwiseKernel<N, Op, Tdata, Args...>`

##### `launchElementwiseKernel<BLOCK_SIZE, N, KernelFunc, Tout>(...)`
```cpp
template <uint32_t BLOCK_SIZE, size_t N, typename KernelFunc, typename Tout, typename... Args>
infiniStatus_t launchElementwiseKernel(...);
```

**功能**: 启动 CUDA kernel 的核心方法。

**详细流程**:
1. **空张量检查**: 如果 `output_size == 0`，直接返回成功
2. **元数据传输**: 调用 `infoToDevice<N>` 将主机端元数据异步拷贝到设备 workspace
3. **计算执行维度**:
   - `blockDims.x = min(256, maxThreadsPerBlock)`: 每个 block 的线程数
   - `gridDims.x = min(ceil(output_size / blockDims.x), gridSizeX)`: grid 中的 block 数
   - `step = gridDims.x * blockDims.x`: 每次 kernel 启动处理的元素数
4. **分步启动 kernel**: 使用 for 循环处理大规模张量（超过 grid 容量）:
   ```cpp
   for (size_t i = 0; i < output_size; i += step) {
       kernel_func<<<gridDims, blockDims, 0, stream>>>(
           output_size, ndim, output_contiguous,
           d_input_contiguous, d_input_broadcasted,
           d_output_shape, d_input_shapes,
           d_output_strides, d_input_strides,
           output, d_inputs_arr, i, args...);
   }
   ```
   每次调用处理一个 step，通过 `offset=i` 参数实现连续处理

##### `infoToDevice<N>(...)`
```cpp
template <size_t N>
infiniStatus_t infoToDevice(
    const op::elementwise::ElementwiseInfo &info,
    void *workspace,
    const void *const *h_inputs_arr,
    const void **&d_inputs_arr,
    const bool *&d_input_contiguous,
    const bool *&d_input_broadcasted,
    const size_t *&d_output_shape,
    const ptrdiff_t *&d_output_strides,
    const size_t *&d_input_shapes,
    const ptrdiff_t *&d_input_strides,
    cudaStream_t stream) const;
```

**功能**: 将主机端的元数据和输入指针数组异步拷贝到设备 workspace，并设置设备端指针偏移。

**内存布局**:
```
workspace layout:
+------------------+
| d_inputs_arr     |  <- N * sizeof(void*) 字节，存储输入设备指针数组
+------------------+
| d_output_shape   |  <- ndim * sizeof(size_t) 字节
+------------------+
| d_output_strides |  <- ndim * sizeof(ptrdiff_t) 字节
+------------------+
| d_input_shapes   |  <- N * ndim * sizeof(size_t) 字节
+------------------+
| d_input_strides  |  <- N * ndim * sizeof(ptrdiff_t) 字节
+------------------+
| d_input_contiguous |  <- N * sizeof(bool) 字节
+------------------+
| d_input_broadcasted |  <- N * sizeof(bool) 字节
+------------------+
```

**执行流程**:
1. 计算 workspace 中元数据的起始偏移: `d_meta_start = workspace + N * sizeof(void*)`
2. 使用 `cudaMemcpyAsync` 异步拷贝输入指针数组和元数据到设备
3. 设置各指针的设备端地址（通过指针算术）

##### `elementwiseKernel<N, Op, Tdata>(...)`
```cpp
template <size_t N, typename Op, typename Tdata, typename... Args>
INFINIOP_CUDA_KERNEL elementwiseKernel(
    size_t output_size, size_t ndim, bool output_contiguous,
    const bool *__restrict__ input_contiguous,
    const bool *__restrict__ input_broadcasted,
    const size_t *__restrict__ output_shape,
    const size_t *__restrict__ input_shapes,
    const ptrdiff_t *__restrict__ output_strides,
    const ptrdiff_t *__restrict__ input_strides,
    Tdata *output,
    const void *const *inputs,
    size_t offset,
    Args... args);
```

**功能**: CUDA kernel 函数，执行逐元素加法计算。

**线程映射**: 每个线程处理一个输出元素，全局线程 ID = `blockIdx.x * blockDim.x + threadIdx.x + offset`

**执行流程**:
1. **边界检查**: `if (idx < output_size)` 确保不越界
2. **类型转换**: 将 `inputs` 从 `const void**` 转换为 `const Tdata*const*`
3. **计算输出索引**:
   - 如果输出连续: `out_idx = idx`
   - 如果输出非连续: `out_idx = indexToOffset(idx, ndim, output_shape, output_strides)`
4. **创建索引器**: `InputIndexer` 结构体封装输入索引计算逻辑
5. **执行操作**: 使用 `unpackInputsAndApply` 和 C++17 折叠表达式展开参数包:
   ```cpp
   unpackInputsAndApply(
       [&](auto... Is) {
           output[out_idx] = Op{}(typed_inputs[Is.value][indexer(Is.value)]..., args...);
       },
       std::make_index_sequence<N>{});
   ```
   - `std::make_index_sequence<N>{}` 生成编译时索引序列 `0, 1, ..., N-1`
   - lambda 表达式接收索引常量 `Is...`，访问对应的输入张量
   - `Op{}(inputs[0][idx0], inputs[1][idx1], ...)` 调用 `cuda::AddOp::operator()`

**复杂度**: O(1) - 每个线程执行常数时间操作

##### `elementwiseKernel<Op, Tout, Tin...>(...)`
```cpp
template <typename Op, typename Tout, typename... Tin>
INFINIOP_CUDA_KERNEL elementwiseKernel(...);
```

**功能**: 支持混合输入输出类型的 CUDA kernel 重载版本。

**区别**:
- 不要求所有输入类型相同，每个输入有独立的类型模板参数 `Tin...`
- 使用 `typedInputPtr<Tin>(inputs[Is.value])` 分别转换每个输入指针
- 调用 `Op{}.template operator()<Tout, Tin...>(...)` 进行类型转换

### `op::elementwise::ElementwiseInfo`
- **位置**: `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/elementwise.h`
- **主要功能**: 封装逐元素操作所需的所有元数据，使用扁平化内存布局存储所有输入输出张量的形状、步长、连续性、广播标志
- **内存管理**: 内部使用 `std::vector<size_t>` 存储，通过指针算术访问不同区域

#### 核心方法

##### `create(output_desc, input_descs)`
```cpp
static ResultType create(
    infiniopTensorDescriptor_t output_desc,
    std::vector<infiniopTensorDescriptor_t> input_descs);
```

**功能**: 从张量描述符构造 `ElementwiseInfo` 对象。

**执行流程**:
1. **参数验证**: 检查输出描述符非空，输入描述符数组非空
2. **广播约束**: 输出张量不能有广播维度（`output_desc->hasBroadcastDim() == false`）
3. **提取基础信息**: ndim, output_size, input_size, output_contiguous
4. **计算元数据大小**:
   ```cpp
   size_t meta_mem_size =
       ndim * (sizeof(size_t) + sizeof(ptrdiff_t)) +  // output shape + strides
       input_size * ndim * sizeof(size_t) +            // input shapes
       input_size * ndim * sizeof(ptrdiff_t) +         // input strides
       2 * input_size * sizeof(bool);                  // contiguous + broadcasted flags
   ```
5. **分配内存**: `std::vector<size_t> meta(CEIL_DIV(meta_mem_size, sizeof(size_t)))`
6. **设置内部指针**: 通过指针算术为各区域设置起始地址
7. **填充数据**:
   - 拷贝输出形状和步长
   - 遍历每个输入，拷贝形状、步长，并计算:
     - `input_contiguous[i] = desc->isContiguous()`
     - `input_broadcasted[i] = !input_contiguous[i] && (desc->ndim() != ndim || desc->hasBroadcastDim())`
8. **构造对象**: 返回 `ResultType(std::move(info))`

**复杂度**: O(input_size * ndim) - 遍历所有输入的维度

## 3. API 接口

```cpp
// 创建加法操作符描述符
infiniStatus_t infiniopCreateAddDescriptor(
    infiniopHandle_t handle,
    infiniopDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_a_desc,
    infiniopTensorDescriptor_t input_b_desc);
// 功能: 分配并初始化一个加法操作符的描述符，验证形状和类型一致性
// 返回: 成功返回 INFINI_STATUS_SUCCESS，失败返回对应错误码

// 查询所需 workspace 大小
size_t infiniopGetAddWorkspaceSize(infiniopDescriptor_t desc);
// 功能: 返回执行该操作所需的设备 workspace 字节数
// 返回: workspace 大小（字节）

// 执行加法计算
infiniStatus_t infiniopAdd(
    infiniopDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input_a,
    const void *input_b,
    void *stream);
// 功能: 在 CUDA 设备上执行加法计算，output = input_a + input_b
// 参数:
//   - workspace: 设备端 workspace 指针，用于存储元数据
//   - workspace_size: workspace 大小，必须 >= descriptor 返回的大小
//   - output: 输出张量的设备指针
//   - input_a, input_b: 输入张量的设备指针
//   - stream: CUDA stream，用于异步执行
// 返回: 成功返回 INFINI_STATUS_SUCCESS，失败返回对应错误码
```

## 4. 使用示例

```cpp
#include "infiniop.h"
#include "infiniop_descriptor.h"

// 初始化 CUDA 环境
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_NVIDIA, 0);

// 定义张量形状和类型（示例：两个 [1024, 1024] 的 FP32 张量）
int64_t shape[] = {1024, 1024};
int64_t strides[] = {1024, 1};  // 行主序（C 风格）
infiniopTensorDescriptor_t input_a_desc, input_b_desc, output_desc;
infiniopCreateTensorDescriptor(&input_a_desc, INFINI_DTYPE_F32, 2, shape, strides);
infiniopCreateTensorDescriptor(&input_b_desc, INFINI_DTYPE_F32, 2, shape, strides);
infiniopCreateTensorDescriptor(&output_desc, INFINI_DTYPE_F32, 2, shape, strides);

// 创建加法操作符描述符
infiniDescriptor_t add_desc;
infiniopCreateAddDescriptor(handle, &add_desc, output_desc, input_a_desc, input_b_desc);

// 查询 workspace 大小并分配
size_t workspace_size = infiniopGetAddWorkspaceSize(add_desc);
void *d_workspace;
cudaMalloc(&d_workspace, workspace_size);

// 分配输入输出张量的设备内存
void *d_input_a, *d_input_b, *d_output;
size_t tensor_size = 1024 * 1024 * sizeof(float);
cudaMalloc(&d_input_a, tensor_size);
cudaMalloc(&d_input_b, tensor_size);
cudaMalloc(&d_output, tensor_size);

// 填充输入数据（示例）
float h_input_a[1024 * 1024], h_input_b[1024 * 1024];
// ... 初始化 h_input_a 和 h_input_b ...
cudaMemcpy(d_input_a, h_input_a, tensor_size, cudaMemcpyHostToDevice);
cudaMemcpy(d_input_b, h_input_b, tensor_size, cudaMemcpyHostToDevice);

// 获取 CUDA stream
cudaStream_t stream;
cudaStreamCreate(&stream);

// 执行加法计算
infiniopAdd(add_desc, d_workspace, workspace_size, d_output, d_input_a, d_input_b, stream);

// 等待完成并取回结果
cudaStreamSynchronize(stream);
cudaMemcpy(h_input_a, d_output, tensor_size, cudaMemcpyDeviceToHost);

// 清理资源
cudaFree(d_workspace);
cudaFree(d_input_a);
cudaFree(d_input_b);
cudaFree(d_output);
cudaStreamDestroy(stream);
infiniopDestroyDescriptor(add_desc);
infiniopDestroyTensorDescriptor(input_a_desc);
infiniopDestroyTensorDescriptor(input_b_desc);
infiniopDestroyTensorDescriptor(output_desc);
infiniopDestroyHandle(handle);
```

## 5. 实现细节

### 内存管理
- **Workspace 布局**: 采用扁平化内存布局，所有元数据连续存储在设备端，减少 kernel 启动时的参数传递开销
- **内存所有权**: `ElementwiseInfo` 使用 `std::vector<size_t>` 自动管理元数据生命周期，通过移动语义避免深拷贝
- **异步传输**: 使用 `cudaMemcpyAsync` 将元数据传输到设备，利用 CUDA stream 实现计算与传输重叠

### 并发性
- **CUDA Kernel 配置**: Block size 固定为 256，Grid size 根据 `output_size` 动态计算，受限于硬件的 `maxThreadsPerBlock` 和 `gridSizeX` 约束
- **分步执行**: 对于超大规模张量（元素数 > grid_size * block_size），使用循环多次启动 kernel，每次处理一个 step，确保所有元素都被处理
- **线程安全**: 描述符对象在创建后为只读，多线程可安全并发调用 `calculate` 方法（需各自提供独立的 workspace 和 stream）

### 性能优化
- **向量化指令**: 针对 `half2` 类型使用 `__hadd2` intrinsic，一次计算两个 FP16 值，吞吐量翻倍
- **硬件加速**: 对 `half`/`cuda_bfloat16` 使用 `__hadd`，对 `float` 使用 `__fadd_rd`，充分利用 Tensor Core 和 FP 单元
- **连续路径优化**: 对于连续张量，kernel 使用线性索引直接访问内存，避免 `indexToOffset` 的维度遍历开销
- **广播支持**: 通过 `InputIndexer` 结构体统一处理广播和非连续访问，保持代码简洁性
- **编译期展开**: 使用 C++17 折叠表达式和 `std::make_index_sequence` 在编译期展开输入参数包，避免运行时分支

### 错误处理
- **参数验证**:
  - `CHECK_DTYPE` 宏验证数据类型支持
  - `CHECK_SAME_SHAPE` 宏验证输出和输入形状一致性
  - `CHECK_CUDA` 宏检查 CUDA API 调用状态
  - `CHECK_RESULT` 宏检查 `utils::Result` 类型
- **错误传播**: 所有错误通过 `infiniStatus_t` 枚举返回，包括:
  - `INFINI_STATUS_SUCCESS`: 成功
  - `INFINI_STATUS_BAD_PARAM`: 参数错误（空指针等）
  - `INFINI_STATUS_BAD_TENSOR_DTYPE`: 不支持的数据类型
  - `INFINI_STATUS_BAD_TENSOR_STRIDES`: 张量步长无效（如输出有广播维度）
  - `INFINI_STATUS_INSUFFICIENT_WORKSPACE`: workspace 大小不足
- **异常安全**: 使用 RAII 和智能指针（`std::unique_ptr`, `std::shared_ptr`）管理资源，异常时自动释放

### 依赖关系
- **上游依赖**:
  - `infiniop/utils.h`: 通用工具宏（`CHECK_DTYPE`, `CHECK_SAME_SHAPE`, `CEIL_DIV` 等）和类型定义
  - `infiniop/operator.h`: `InfiniopDescriptor` 基类和算子接口
  - `infiniop/tensor.h`: 张量描述符相关类型和方法
  - `infiniop/devices/nvidia/nvidia_common.cuh`: NVIDIA 设备通用定义（`device::nvidia::Handle`）
  - `infiniop/devices/nvidia/nvidia_kernel_common.cuh`: CUDA kernel 通用工具（`indexToOffset`, `INFINIOP_CUDA_KERNEL`）
- **下游依赖**: 无（该模块为叶子节点，不依赖其他算子）
- **外部依赖**: CUDA Toolkit（提供 `__hadd2`, `__hadd`, `__fadd_rd` 等 intrinsic）

### 设计模式
- **宏生成模式**: `ELEMENTWISE_DESCRIPTOR` 宏为每个逐元素操作生成相同的 `Descriptor` 类结构，避免代码重复
- **策略模式**: `DeviceImpl` 封装 CUDA 执行策略，`Descriptor` 通过组合使用该策略
- **模板方法模式**: `calculate` 方法定义算法骨架，`calculateImpl` 和 `launchElementwiseKernel` 实现具体步骤
- **工厂模式**: `Descriptor::create` 和 `DeviceImpl::create` 使用静态工厂方法封装对象创建逻辑
- **RAII**: `Descriptor` 析构时自动释放 `DeviceImpl`，`ElementwiseInfo` 使用 `std::vector` 自动管理内存
- **类型擦除**: 使用 `void*` 和 `void**` 传递无类型指针，在 kernel 内部通过模板参数恢复类型信息
- **CRTP（奇异递归模板模式）**: `ELEMENTWISE_DESCRIPTOR` 宏生成类时，类名和命名空间作为宏参数，实现代码复用
