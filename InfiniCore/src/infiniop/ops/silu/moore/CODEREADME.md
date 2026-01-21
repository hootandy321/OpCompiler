# Moore 平台 SiLU (Swish) 激活函数算子实现文档

## 模块概述

本模块实现了 Moore 平台（摩尔线程 GPU）上的 SiLU (Swish) 激活函数算子。SiLU 是一种平滑的非单调激活函数，定义为 `SiLU(x) = x * sigmoid(x) = x / (1 + e^(-x))`。该实现基于 InfiniOp 的 elementwise 基础设施，支持 FP16、BF16、FP32 和 FP64 四种数据类型，并针对 Moore 平台的 MUSA 架构进行了优化。

## 1. 模块结构

- **`silu_moore.h`**: 算子描述符的公共 API 定义，通过宏 `ELEMENTWISE_DESCRIPTOR` 生成完整的 Descriptor 类接口
- **`silu_moore.mu`**: 算子的核心实现，包括描述符创建和计算调度逻辑
- **`silu_moore_kernel.h`**: SiLU 操作的 CUDA/MUSA 设备端内核实现，包含多类型特化的 functor

## 2. 核心类与组件

### `op::silu::moore::Descriptor`
- **位置**: `silu_moore.h` (通过宏生成), `silu_moore.mu` (实现)
- **主要功能**: SiLU 算子的描述符类，继承自 `InfiniopDescriptor`，负责管理算子的元数据、设备实现和执行参数
- **关键成员**:
  - `_dtype`: `infiniDtype_t` - 算子支持的数据类型 (BF16/F16/F32/F64)
  - `_info`: `op::elementwise::ElementwiseInfo` - 张量的形状、步长、连续性等元数据
  - `_device_info`: `std::unique_ptr<op::elementwise::moore::DeviceImpl>` - Moore 设备端实现封装
  - `_workspace_size`: `size_t` - 设备端所需工作空间大小
- **核心方法**:
  - `create(handle, desc_ptr, out_desc, input_desc_vec)`: 静态工厂方法，创建算子描述符
    - 验证数据类型支持 (BF16/F16/F32/F64)
    - 验证输入输出张量形状一致性
    - 计算 elementwise 元信息和工作空间大小
    - 创建 Moore 设备实现对象
    - 返回 `INFINI_STATUS_SUCCESS` 或错误码
  - `calculate(workspace, workspace_size, output, inputs, stream)`: 执行 SiLU 计算
    - 检查工作空间是否充足
    - 根据 `_dtype` 分发到对应的类型特化模板
    - 调用 `DeviceImpl::calculate<256, moore::SiluOp, T>` 执行内核
    - 支持的数据类型映射: BF16→cuda_bfloat16, F16→half, F32→float, F64→double
- **生命周期**: 通过 `create` 静态方法构造，析构函数默认实现，由调用方管理生命周期

### `op::silu::moore::SiluOp`
- **位置**: `silu_moore_kernel.h`
- **主要功能**: 设备端 functor，实现 SiLU 的数学计算逻辑，使用模板特化支持不同数据类型的优化路径
- **关键成员**:
  - `num_inputs`: `static constexpr size_t = 1` - 单输入操作符标记
- **核心方法**:
  - `operator()(const T &x) const`: SiLU 函数实现
    - **half2 特化**: 使用向量化指令优化
      ```cpp
      return __hmul2(x, __h2div(__float2half2_rn(1.0f),
                                __hadd2(__float2half2_rn(1.0f), h2exp(__hneg2(x)))));
      ```
      利用 half2 SIMD 指令并行处理两个 FP16 值
    - **half 特化**: 为保证 MUSA 平台兼容性，先转换为 FP32 计算，再转回 FP16
      ```cpp
      float x_f = __half2float(x);
      float sigmoid_f = 1.0f / (1.0f + __expf(-x_f));
      return __float2half(x_f * sigmoid_f);
      ```
    - **cuda_bfloat16 特化**: 同样通过 FP32 中转计算
    - **float 特化**: 使用 Moore 平台内置指令优化
      ```cpp
      return __fmul_rn(x, __frcp_rn(__fadd_rn(1.0f, __expf(-x))));
      ```
      使用 `__frcp_rn` (快速倒数) 和 `__fmul_rn` (舍入乘法) 提升性能
    - **double 特化**: 直接使用标准数学函数 `exp()`
- **设计模式**: Functor 模式，支持 `__device__ __forceinline__` 内联调用

### `op::elementwise::moore::DeviceImpl`
- **位置**: `elementwise_moore.h` (依赖的基础设施)
- **主要功能**: Moore 平台的 elementwise 操作通用执行引擎，管理内核启动、内存复制和网格配置
- **关键成员**:
  - `_opaque`: `std::shared_ptr<Opaque>` - Pimpl 模式的实现封装
- **核心方法**:
  - `calculate<BLOCK_SIZE, Op, Tdata>(...)`: 单类型统一输入的计算入口
    - BLOCK_SIZE 固定为 256
    - Op 为 `moore::SiluOp`
    - Tdata 为具体数据类型 (half2/half/cuda_bfloat16/float/double)
  - `calculate<BLOCK_SIZE, Op, Tout, Tin...>(...)`: 支持不同输入输出类型的重载版本
- **Opaque 内部实现**:
  - `calculateImpl(...)`: 调用 `launchElementwiseKernel` 启动内核
  - `infoToDevice(...)`: 将元数据从主机复制到设备
    - 复制输入指针数组
    - 复制形状、步长、连续性标志等元数据
    - 计算设备端内存布局偏移量
  - `launchElementwiseKernel(...)`: 配置并执行 CUDA/MUSA 内核
    - 动态计算 block 和 grid 维度 (block.x ≤ maxThreadsPerBlock, grid.x ≤ gridSizeX)
    - 支持大张量的分步循环处理 (step = grid.x * block.x)
    - 内核签名为 `elementwiseKernel<N, Op, Tdata, Args...>`

## 3. API 接口

```cpp
namespace op::silu::moore {

// 算子描述符类 (由 ELEMENTWISE_DESCRIPTOR 宏生成)
class Descriptor final : public InfiniopDescriptor {
public:
    ~Descriptor();

    // 获取所需工作空间大小
    size_t workspaceSize() const;

    // 创建算子描述符
    // @param handle: InfiniOp 句柄，包含设备和上下文信息
    // @param desc_ptr: 输出参数，用于返回创建的描述符指针
    // @param out_desc: 输出张量描述符
    // @param input_desc_vec: 输入张量描述符向量（SiLU 只需1个输入）
    // @return: INFINI_STATUS_SUCCESS 或错误码 (BAD_TENSOR_DTYPE, BAD_TENSOR_SHAPE 等)
    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t out_desc,
        std::vector<infiniopTensorDescriptor_t> input_desc_vec);

    // 执行 SiLU 计算
    // @param workspace: 设备端工作空间指针
    // @param workspace_size: 工作空间大小，必须 ≥ workspaceSize()
    // @param output: 输出张量设备指针
    // @param inputs: 输入张量设备指针向量
    // @param stream: MUSA 流
    // @return: INFINI_STATUS_SUCCESS 或 INSUFFICIENT_WORKSPACE/BAD_TENSOR_DTYPE
    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *output,
        std::vector<const void *> inputs,
        void *stream) const;
};

// 设备端 SiLU 操作 functor
struct SiluOp {
    static constexpr size_t num_inputs = 1;

    // 对标量执行 SiLU 操作
    // @param x: 输入值 (支持 half2/half/cuda_bfloat16/float/double)
    // @return: SiLU(x) = x * sigmoid(x)
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const;
};

} // namespace op::silu::moore
```

## 4. 使用示例

```cpp
#include "silu_moore.h"
#include <vector>

// 假设已有环境和句柄
infiniopHandle_t handle;          // InfiniOp 句柄
musaStream_t stream;               // MUSA 流

// 定义张量形状和类型
std::vector<size_t> shape = {1024, 1024};
auto dtype = INFINI_DTYPE_F16;

// 创建输入输出张量描述符
infiniopTensorDescriptor_t input_desc, output_desc;
infiniCreateTensorDescriptor(handle, &input_desc, dtype, shape.size(), shape.data(), nullptr);
infiniCreateTensorDescriptor(handle, &output_desc, dtype, shape.size(), shape.data(), nullptr);

// 分配设备内存
half *d_input, *d_output;
musaMalloc(&d_input, 1024 * 1024 * sizeof(half));
musaMalloc(&d_output, 1024 * 1024 * sizeof(half));

// 创建 SiLU 算子描述符
op::silu::moore::Descriptor *silu_desc = nullptr;
auto status = op::silu::moore::Descriptor::create(
    handle,
    &silu_desc,
    output_desc,
    {input_desc}
);

if (status != INFINI_STATUS_SUCCESS) {
    // 错误处理
}

// 获取工作空间大小
size_t workspace_size = silu_desc->workspaceSize();
void *d_workspace = nullptr;
if (workspace_size > 0) {
    musaMalloc(&d_workspace, workspace_size);
}

// 执行 SiLU 计算
std::vector<const void *> inputs = {d_input};
status = silu_desc->calculate(d_workspace, workspace_size, d_output, inputs, stream);

// 同步流
musaStreamSynchronize(stream);

// 清理资源
delete silu_desc;
musaFree(d_input);
musaFree(d_output);
if (d_workspace) musaFree(d_workspace);
```

## 5. 实现细节

### 数学定义与算法

**SiLU (Swish) 函数**:
```
SiLU(x) = x * sigmoid(x) = x / (1 + e^(-x))
```

**计算流程**:
1. 计算 `-x` (取负)
2. 计算 `exp(-x)` (指数函数)
3. 计算 `1 + exp(-x)` (加法)
4. 计算倒数 `1 / (1 + exp(-x))` (sigmoid)
5. 计算 `x * sigmoid(x)` (乘法)

**时间复杂度**: O(n)，其中 n 为张量元素数量
**空间复杂度**: O(1) 额外空间 (除了输入输出)

### 内存管理

**工作空间组成**:
- **输入指针数组**: `sizeof(void*) * num_inputs` 字节，用于存储输入张量的设备指针
- **元数据区**: 由 `ElementwiseInfo::getMetaMemSize()` 计算，包含:
  - `output_shape`: `ndim * sizeof(size_t)` - 输出张量形状
  - `output_strides`: `ndim * sizeof(ptrdiff_t)` - 输出张量步长
  - `input_shapes`: `num_inputs * ndim * sizeof(size_t)` - 所有输入张量形状
  - `input_strides`: `num_inputs * ndim * sizeof(ptrdiff_t)` - 所有输入张量步长
  - `input_contiguous`: `num_inputs * sizeof(bool)` - 输入连续性标志
  - `input_broadcasted`: `num_inputs * sizeof(bool)` - 输入广播标志

**内存传输策略**:
- 使用 `musaMemcpyAsync` 异步复制元数据到设备
- 元数据和指针数组打包在连续的 workspace 中，减少传输次数

### 并发执行

**内核配置**:
- **BLOCK_SIZE**: 固定为 256 线程/块
- **Grid 大小**: 动态计算
  - `blockDims.x = min(256, maxThreadsPerBlock)`
  - `gridDims.x = min(ceil_div(output_size, blockDims.x), gridSizeX)`
- **步进循环**: 对于大张量 (output_size > grid.x * block.x)，使用循环多次启动内核处理
  ```cpp
  for (size_t i = 0; i < output_size; i += step) {
      kernel<<<grid, block, 0, stream>>>(..., i);
  }
  ```

**线程安全性**:
- 算子描述符创建阶段: 非线程安全，需调用方同步
- 计算阶段: 线程安全，不同流可并发执行不同算子实例
- MUSA 流隔离: 使用不同流保证并发执行的安全性

### 性能优化技术

1. **向量化优化 (half2)**:
   - 对于 FP16 类型，使用 `half2` 向量指令同时处理两个元素
   - 相关指令: `__hmul2`, `__h2div`, `__hadd2`, `h2exp`, `__hneg2`
   - 理论吞吐量提升 2倍

2. **内置函数使用**:
   - FP32 使用 `__frcp_rn` (快速倒数近似) 替代除法
   - FP32 使用 `__fmul_rn` (舍入乘法) 保证精度
   - 指数计算使用 `__expf` 而非标准 `exp`

3. **条件编译优化**:
   - `if constexpr` 在编译期选择类型特化，零运行时开销
   - 每个 `operator()` 实例都是独立编译的模板特化

4. **连续内存路径**:
   - `ElementwiseInfo` 检测张量连续性 (`isOutputContiguous`)
   - 连续张量使用线性索引 `idx`，避免 `indexToOffset` 转换开销
   - 非连续张量通过 `device::moore::indexToOffset` 计算实际偏移

5. **广播支持**:
   - `InputIndexer` 封装输入索引计算逻辑
   - 自动处理形状广播和步长转换
   - 允许输入张量形状与输出不完全一致

### 错误处理

**错误码**:
- `INFINI_STATUS_SUCCESS`: 操作成功
- `INFINI_STATUS_BAD_TENSOR_DTYPE`: 不支持的数据类型 (仅支持 BF16/F16/F32/F64)
- `INFINI_STATUS_BAD_TENSOR_SHAPE`: 输入输出形状不匹配
- `INFINI_STATUS_INSUFFICIENT_WORKSPACE`: 提供的 workspace 小于所需大小

**验证逻辑**:
- `create` 阶段使用 `CHECK_DTYPE` 宏验证数据类型
- `create` 阶段使用 `CHECK_SAME_SHAPE` 宏验证形状一致性
- `calculate` 阶段检查 `workspace_size < _workspace_size`

**宏定义**:
- `CHECK_RESULT`: 检查 `Result<T>` 类型并提取值或返回错误
- `CHECK_MOORE`: 检查 MUSA API 调用状态

### 平台兼容性

**Moore 平台特性**:
- 使用 MUSA (Moore Unified Stream Architecture) 编程模型
- API 兼容 CUDA (musaMemcpyAsync, musaStream_t 等)
- 内核使用 `__device__`, `__forceinline__` CUDA 扩展
- 内核启动使用 `<<<grid, block, 0, stream>>>` 语法

**数据类型映射**:
- `INFINI_DTYPE_F16` → `half` / `half2` (向量化)
- `INFINI_DTYPE_BF16` → `cuda_bfloat16` (兼容 CUDA 类型定义)
- `INFINI_DTYPE_F32` → `float`
- `INFINI_DTYPE_F64` → `double`

**依赖项**:
- `device::moore::Handle`: Moore 设备句柄
- `device::moore::indexToOffset`: 设备端索引计算函数
- `op::elementwise::ElementwiseInfo`: Elementwise 操作元数据管理
- `INFINIOP_MOORE_KERNEL`: MUSA 内核声明宏

### 设计模式

1. **Pimpl 模式 (Pointer to Implementation)**:
   - `DeviceImpl` 通过 `std::shared_ptr<Opaque>` 隐藏实现细节
   - 减少 `elementwise_moore_api.h` 的编译依赖

2. **CRTP (Curiously Recurring Template Pattern)**:
   - `ELEMENTWISE_DESCRIPTOR` 宏生成派生自 `InfiniopDescriptor` 的类
   - 将公共逻辑抽象到宏定义中

3. **Functor 模式**:
   - `SiluOp` 重载 `operator()` 实现可调用对象
   - 便于作为模板参数传递给通用内核

4. **类型擦除**:
   - `Descriptor` 统一接口，内部存储 `infiniDtype_t`
   - `calculate` 中通过 switch 分发到类型特化实现

5. **策略模式**:
   - `elementwiseKernel` 是通用的 elementwise 执行策略
   - `SiluOp` 是具体的计算策略实现
