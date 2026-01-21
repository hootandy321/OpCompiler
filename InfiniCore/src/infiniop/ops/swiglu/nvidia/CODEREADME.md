# SwiGLU NVIDIA CUDA 算子核心实现文档

本模块实现了 SwiGLU (Swish-Gated Linear Unit) 激活函数的 NVIDIA CUDA 后端，这是现代大语言模型（如 LLaMA、PaLM）中常用的关键激活函数。该实现基于通用的逐元素操作框架，支持多种浮点数据类型，并针对 CUDA 架构进行了深度优化。

## 1. 模块结构

- **`swiglu_nvidia.cuh`**: API 声明头文件，通过宏定义复用逐元素操作的描述符结构
- **`swiglu_nvidia.cu`**: 核心实现文件，包含描述符创建、计算调度和数据类型分发逻辑
- **依赖项**:
  - `../../../elementwise/nvidia/elementwise_nvidia_api.cuh`: 逐元素操作 CUDA API 框架
  - `../cuda/kernel.cuh`: SwiGLU CUDA 核函数实现（sigmoid 计算和向量化操作）

## 2. 核心类与结构

### `op::swiglu::nvidia::Descriptor`
- **位置**: `swiglu_nvidia.cuh`（通过 `ELEMENTWISE_DESCRIPTOR` 宏生成）
- **基类**: `InfiniopDescriptor`
- **主要功能**: 封装 SwiGLU 算子的元数据、设备实现和工作空间需求

#### 关键成员变量
- **`_dtype` (`infiniDtype_t`)**: 输出张量的数据类型（支持 F16/BF16/F32/F64）
- **`_info` (`op::elementwise::ElementwiseInfo`)**: 存储张量形状、步长、广播信息的元数据结构
- **`_device_info` (`std::unique_ptr<op::elementwise::nvidia::DeviceImpl>`)**: CUDA 设备实现的智能指针
- **`_workspace_size` (`size_t`)**: 设备端所需工作空间大小（元数据 + 输入指针数组）

#### 核心方法

##### `create()`
```cpp
static infiniStatus_t create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec);
```
- **功能**: 创建并初始化 SwiGLU 描述符
- **算法流程**:
  1. 将通用句柄转换为 NVIDIA 设备句柄
  2. 验证数据类型（F16/BF16/F32/F64）
  3. 检查输入（up、gate）与输出形状一致性
  4. 通过 `CREATE_ELEMENTWISE_CUDA_DESCRIPTOR` 宏构造逐元素操作元数据
  5. 创建 CUDA 设备实现并计算工作空间大小
- **复杂度**: O(ndim)，其中 ndim 为张量维度数
- **返回值**: `INFINI_STATUS_SUCCESS` 或相应错误码

##### `calculate()`
```cpp
infiniStatus_t calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const;
```
- **功能**: 在 CUDA 流上执行 SwiGLU 计算
- **算法流程**:
  1. 验证工作空间大小是否满足需求
  2. 根据 `_dtype` 分发到对应的模板实例化
  3. 调用 `DeviceImpl::calculate<256, cuda::SwiGLUOp, T>()` 启动核函数
- **关键参数**:
  - `BLOCK_SIZE = 256`: CUDA 线程块大小
  - `cuda::SwiGLUOp`: 操作符函数对象（定义在 `../cuda/kernel.cuh`）
- **返回值**: `INFINI_STATUS_SUCCESS`、`INFINI_STATUS_INSUFFICIENT_WORKSPACE` 或 `INFINI_STATUS_BAD_TENSOR_DTYPE`

### `op::swiglu::cuda::SwiGLUOp`
- **位置**: `../cuda/kernel.cuh`
- **类型**: 函数对象（Functor）
- **模板参数**: `T`（数据类型）
- **静态常量**: `num_inputs = 2`

#### 核心操作符重载
```cpp
template <typename T>
__device__ __forceinline__ T operator()(const T &up, const T &gate) const;
```
- **数学定义**: `output = up × sigmoid(gate) × gate`
- **实现优化**:
  - **half2**: 使用 `__hmul2`、`h2rcp`、`h2exp`、`__hneg2` 等 intrinsic 指令实现向量化计算
  - **cuda_bfloat162**: 转换为 float 计算，再转换回 bfloat16（避免精度损失）
  - **float**: 使用 `__fmul_rn`、`__frcp_rn`、`__fadd_rn` 等 IEEE-754 舍入模式指令
  - **double**: 标准库实现（`std::exp`、乘法）

#### 私有辅助方法
```cpp
template <typename T>
__device__ __forceinline__ T sigmoid(const T &x) const;
```
- **数学定义**: `sigmoid(x) = 1 / (1 + e^(-x))`
- **实现优化**:
  - **half2**: `h2rcp(1 + h2exp(-x))`（向量化倒数+指数）
  - **cuda_bfloat162**: 分别计算两个元素的 `1.0f / (1.0f + expf(-x))`
  - **float**: `__frcp_rn(1 + expf(-x))`
- **数值稳定性**: 使用倒数指令而非除法，提高性能并保持精度

## 3. API 接口

### 公共 API（C 兼容接口）
通过 `ELEMENTWISE_DESCRIPTOR` 宏自动生成以下标准接口：

```cpp
// 创建算子描述符
infiniStatus_t infiniswigluNvidiaDescriptorCreate(
    infiniopHandle_t handle,
    infiniswigluNvidiaDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t up_desc,
    infiniopTensorDescriptor_t gate_desc);

// 销毁算子描述符
infiniStatus_t infiniswigluNvidiaDescriptorDestroy(
    infiniswigluNvidiaDescriptor_t desc);

// 获取工作空间大小
size_t infiniswigluNvidiaGetWorkspaceSize(
    const infiniswigluNvidiaDescriptor_t desc);

// 执行计算
infiniStatus_t infiniswigluNvidiaCalculate(
    const infiniswigluNvidiaDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *up,
    const void *gate,
    void *stream);
```

### 内部实现接口

```cpp
namespace op::swiglu::nvidia {
    // 描述符创建（实现）
    infiniStatus_t Descriptor::create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t out_desc,
        std::vector<infiniopTensorDescriptor_t> input_desc_vec);

    // 计算执行（实现）
    infiniStatus_t Descriptor::calculate(
        void *workspace,
        size_t workspace_size,
        void *output,
        std::vector<const void *> inputs,
        void *stream) const;
}
```

## 4. 使用示例

```cpp
#include "swiglu_nvidia.cuh"
#include "../../devices/nvidia/nvidia_handle.h"

// 示例：在 CUDA 设备上执行 SwiGLU 激活
void exampleSwiGLU() {
    // 1. 初始化 CUDA 句柄
    infiniopHandle_t handle;
    infiniopNvidiaHandleCreate(&handle, 0); // 设备 ID = 0

    // 2. 创建张量描述符（假设形状为 [1024, 512]）
    int64_t shape[] = {1024, 512};
    int64_t strides[] = {512, 1}; // 连续内存布局

    infiniopTensorDescriptor_t up_desc, gate_desc, out_desc;
    infiniopCreateTensorDescriptor(&up_desc, INFINI_DTYPE_F16, 2, shape, strides);
    infiniopCreateTensorDescriptor(&gate_desc, INFINI_DTYPE_F16, 2, shape, strides);
    infiniopCreateTensorDescriptor(&out_desc, INFINI_DTYPE_F16, 2, shape, strides);

    // 3. 创建 SwiGLU 描述符
    op::swiglu::nvidia::Descriptor *swiglu_desc;
    auto status = op::swiglu::nvidia::Descriptor::create(
        handle,
        &swiglu_desc,
        out_desc,
        {up_desc, gate_desc});
    if (status != INFINI_STATUS_SUCCESS) {
        // 错误处理
        return;
    }

    // 4. 分配设备内存和工作空间
    half *d_up, *d_gate, *d_out;
    size_t nelem = 1024 * 512;
    cudaMalloc(&d_up, nelem * sizeof(half));
    cudaMalloc(&d_gate, nelem * sizeof(half));
    cudaMalloc(&d_out, nelem * sizeof(half));

    size_t workspace_size = swiglu_desc->workspaceSize();
    void *d_workspace;
    cudaMalloc(&d_workspace, workspace_size);

    // 5. 创建 CUDA 流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 6. 执行计算
    status = swiglu_desc->calculate(
        d_workspace,
        workspace_size,
        d_out,
        {d_up, d_gate},
        stream);

    // 7. 同步并检查错误
    cudaStreamSynchronize(stream);
    if (status != INFINI_STATUS_SUCCESS) {
        // 错误处理
    }

    // 8. 清理资源
    cudaFree(d_up);
    cudaFree(d_gate);
    cudaFree(d_out);
    cudaFree(d_workspace);
    cudaStreamDestroy(stream);
    delete swiglu_desc;
    infiniopDestroyTensorDescriptor(up_desc);
    infiniopDestroyTensorDescriptor(gate_desc);
    infiniopDestroyTensorDescriptor(out_desc);
    infiniopNvidiaHandleDestroy(handle);
}
```

## 5. 实现细节

### 内存管理
- **元数据打包**: `ElementwiseInfo` 将形状、步长、广播标志打包到连续的 `std::vector<size_t>` 中，通过指针偏移访问不同区域
- **工作空间布局**:
  ```
  [输入指针数组 (N * sizeof(void*))]
  [输出形状 (ndim * sizeof(size_t))]
  [输出步长 (ndim * sizeof(ptrdiff_t))]
  [输入形状 (N * ndim * sizeof(size_t))]
  [输入步长 (N * ndim * sizeof(ptrdiff_t))]
  [连续标志 (N * sizeof(bool))]
  [广播标志 (N * sizeof(bool))]
  ```
- **设备传输**: 使用 `cudaMemcpyAsync` 异步传输元数据到设备，与计算重叠

### 并发策略
- **CUDA 流**: 所有操作在用户提供的 CUDA 流上异步执行
- **网格配置**:
  - 块大小: 固定 256 线程（`BLOCK_SIZE`）
  - 网格大小: `min(ceil(output_size / 256), max_grid_size_x)`
  - 步进循环: 处理超过最大网格尺寸的大张量（`for (i = 0; i < output_size; i += step)`）
- **线程安全**: 每个线程处理独立的输出元素，无需同步原语

### 性能优化
- **向量化指令**:
  - **half2**: 单指令处理两个 FP16 值，吞吐量翻倍
  - **cuda_bfloat162**: 虽需转换，但利用 CUDA Core 的流水线
- **融合操作**: `gate * sigmoid(gate) * up` 在单个核函数中完成，减少全局内存访问
- **快速数学函数**:
  - `__hmul2`, `__fmul_rn`: 融合乘加指令
  - `h2rcp`, `__frcp_rn`: 硬件倒数（近似但更快）
  - `h2exp`, `__expf`: 快速指数函数
- **广播优化**:
  - 连续张量直接使用线性索引（`idx`）
  - 非连续张量通过 `indexToOffset` 计算物理偏移
  - 广播维度自动识别，避免越界访问

### 错误处理
- **类型检查**: `CHECK_DTYPE` 宏验证数据类型是否在支持列表中
- **形状验证**: `CHECK_SAME_SHAPE` 宏确保 up、gate、输出形状一致
- **工作空间验证**: 运行时检查 `workspace_size < _workspace_size`，返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
- **空张量处理**: `output_size == 0` 时直接返回成功，避免无效核函数启动
- **CUDA 错误传播**: `CHECK_CUDA` 宏检查所有 CUDA API 调用状态

### 设计模式
- **CRTP (奇异递归模板模式)**: `ELEMENTWISE_DESCRIPTOR` 宏为每个操作生成独立描述符类
- **策略模式**: 数据类型分发（switch-case）选择不同的模板实例化
- **工厂方法**: `Descriptor::create` 作为构造函数的封装
- **RAII**: `std::unique_ptr` 管理设备实现对象生命周期
- **类型擦除**: `void *` 输入指针在核函数内部转换为具体类型

### 依赖关系
- **上游依赖**:
  - `op::elementwise::ElementwiseInfo`: 元数据管理
  - `op::elementwise::nvidia::DeviceImpl`: CUDA 逐元素操作基础设施
  - `device::nvidia::Handle`: NVIDIA 设备句柄
  - `infiniopTensorDescriptor`: 张量描述符接口
- **下游使用**: 大语言模型推理框架中的前馈网络层（FFN）
- **外部依赖**: CUDA Toolkit（需支持 half2、bfloat16 intrinsic）

### 算法复杂度
- **时间复杂度**: O(n)，其中 n 为输出张量元素数量（每个元素执行固定计算）
- **空间复杂度**:
  - 额外工作空间: O(ndim * input_size)（元数据存储）
  - 设备端内存: O(output_size)（输出张量）
- **吞吐量**: 理论上受限于 CUDA Core 的浮点运算能力（FP16: ~2×TFLOPS, BF16: ~2×TFLOPS, FP32: ~1×TFLOPS）

### 数值特性
- **半精度 (FP16)**:
  - 动态范围: [6.1e-5, 6.5e4]
  - sigmoid 中间值可能溢出/下溢，但实现中已通过 intrinsic 指令优化
- **脑浮点 (BF16)**:
  - 指数位与 FP32 相同，尾数位截断至 7 位
  - 避免了 FP16 的溢出问题，适合大数值梯度
- **精度损失**:
  - `gate * sigmoid(gate) * up` 包含两次乘法，可能累积舍入误差
  - FP16/BF16 在极端情况下可能损失 1-2 位有效数字

### 扩展性
- **新数据类型支持**: 在 `calculate()` 的 switch-case 中添加分支
- **自定义操作**: 继承 `SwiGLUOp` 模式，实现新的函数对象
- **多 GPU**: 通过不同 `device_id` 创建独立句柄，实现数据并行
- **混合精度**: 扩展 `operator()` 模板支持不同输入/输出类型组合
