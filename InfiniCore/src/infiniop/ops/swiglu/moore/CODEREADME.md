# SwiGLU Moore 后端实现文档

本模块实现了 SwiGLU (Swish-Gated Linear Unit) 激活函数在 Moore (MUSA) 硬件平台上的后端支持。SwiGLU 是现代 Transformer 架构（如 LLaMA、GLM 等大语言模型）中广泛使用的前馈网络激活函数，通过门控机制结合 Swish 激活函数，相比传统的 ReLU/GLU 具有更好的性能表现。

## 1. 模块结构

- **`swiglu_moore.h`**: Moore 后端 API 描述符定义，继承自 elementwise 通用框架
- **`swiglu_moore.mu`**: Moore 后端实现主文件，包含描述符创建与计算调度逻辑
- **`siwglu_moore_kernel.h`**: 设备端核函数实现，包含 SwiGLU 操作的 MUSA 内核代码（注：文件名拼写错误，应为 `swiglu_moore_kernel.h`）

## 2. 核心类与组件

### `op::swiglu::moore::Descriptor`
- **位置**: 通过 `ELEMENTWISE_DESCRIPTOR` 宏在 `swiglu_moore.h` 中自动生成
- **主要功能**: SwiGLU 操作的 Moore 设备描述符，继承自 `InfiniopDescriptor` 基类
- **核心成员变量**:
  - `_dtype`: `infiniDtype_t` - 数据类型（支持 F16/BF16/F32/F64）
  - `_info`: `op::elementwise::ElementwiseInfo` - 张量形状、步长、布局等元数据
  - `_device_info`: `std::unique_ptr<op::elementwise::moore::DeviceImpl>` - Moore 设备实现指针
  - `_workspace_size`: `size_t` - 所需工作空间大小
- **核心方法**:
  - `create(handle, desc_ptr, out_desc, input_desc_vec)`: 创建描述符并验证张量形状一致性
    - 验证输入输出张量形状完全相同（通过 `CHECK_SAME_SHAPE` 宏）
    - 验证数据类型支持（F16/BF16/F32/F32/F64）
    - 调用 `CREATE_ELEMENTWISE_MOORE_DESCRIPTOR` 宏初始化底层实现
  - `calculate(workspace, workspace_size, output, inputs, stream)`: 执行 SwiGLU 计算
    - 检查工作空间大小是否充足
    - 根据数据类型分发到对应的模板特化（F16/BF16/F32/F64）
    - 调用 `DeviceImpl::calculate<256, cuda::SwiGLUOp, T>` 启动内核（块大小固定为 256）
- **生命周期**: 由 `operator.cc` 中的工厂函数 `infiniopCreateSwiGLUDescriptor` 创建，由 `infiniopDestroySwiGLUDescriptor` 销毁

### `op::swiglu::cuda::SwiGLUOp`
- **位置**: `siwglu_moore_kernel.h` (namespace 仍使用 `cuda` 以保持代码兼容性)
- **主要功能**: 设备端仿函数（Functor），定义 SwiGLU 操作的数学运算
- **核心成员**:
  - `num_inputs`: `static constexpr size_t = 2` - 输入数量（up 分支和 gate 分支）
- **核心方法**:
  - `sigmoid(x)`: 私有方法，计算 Sigmoid 激活函数
    - **half2 向量化**: 使用 `h2rcp(__hadd2(make_half2(1, 1), h2exp(__hneg2(x))))` 实现高效计算
    - **half 标量**: 由于 MUSA 平台不支持 `hrcp` 内置函数，采用提升到 float 精度的实现方式：
      ```cpp
      float xf = __half2float(x);
      float sigf = 1.0f / (1.0f + std::exp(-xf));
      return __float2half(sigf);
      ```
    - **cuda_bfloat162 向量化**: 分别提取低高位，使用 `__frcp_rn(__fadd_rn(1.0f, __expf(-x)))` 计算
    - **float/double 标量**: 使用标准浮点运算 `1 / (1 + std::exp(-x))`
  - `operator()(up, gate)`: 公有方法，执行 SwiGLU 计算 `output = SiLU(gate) * up = gate * sigmoid(gate) * up`
    - **half2**: 使用 `__hmul2(__hmul2(gate, sigmoid(gate)), up)` 进行向量化乘法
    - **cuda_bfloat162**:
      - MUSA 平台使用 `__low2float()` / `__high2float()` 直接提取并转换（替代 CUDA 的两步操作）
      - 分别计算两个元素的乘积：`res = gate * sigmoid(gate) * up`
      - 使用 `__floats2bfloat162_rn(res0, res1)` 组合结果
    - **其他类型**: 标量乘法 `gate * sigmoid(gate) * up`
- **关键实现细节**:
  - 所有计算均采用向量化指令（half2/bfloat162）以提高吞吐量
  - 使用 `__forceinline__` 强制内联以减少函数调用开销
  - 使用 `if constexpr` 实现编译期类型分发，避免运行时分支

### `op::elementwise::moore::DeviceImpl`
- **位置**: `elementwise_moore.h` (通过 `swiglu_moore.mu` 间接依赖)
- **主要功能**: Moore 设备端的通用逐元素操作执行引擎
- **核心成员**:
  - `_opaque`: `std::shared_ptr<Opaque>` - Pimpl 模式隐藏实现细节
- **核心方法**:
  - `calculate<BLOCK_SIZE, Op, Tdata>(info, workspace, output, inputs, stream)`: 统一计算入口
    - BLOCK_SIZE 固定为 256（由 swiglu_moore.mu 指定）
    - Op 为 `cuda::SwiGLUOp` 操作符
    - Tdata 为数据类型（half/cuda_bfloat16/float/double）
  - `Opaque::calculateImpl()`: 实际内核启动逻辑
    - 调用 `launchElementwiseKernel()` 准备内核参数
    - 使用 MUSA 异步内存复制将元数据传输到设备
    - 计算网格和块维度（考虑设备 `maxThreadsPerBlock` 和 `gridSizeX` 限制）
    - 对于大张量，使用步进式多内核启动策略（`for (size_t i = 0; i < output_size; i += step)`）
- **关键实现细节**:
  - **工作空间布局**:
    ```
    [输入指针数组 (N * sizeof(void*))] [元数据 (shape, strides, contiguous flags)]
    ```
  - **广播支持**: 通过 `InputIndexer` 和 `device::moore::indexToOffset()` 实现自动广播
  - **内存对齐**: 元数据使用 `std::vector<size_t>` 确保对齐到 size_t 边界

## 3. API 接口

```cpp
// 公共 C API (定义在 operator.cc，通过宏调度到 moore 后端)

__C infiniStatus_t infiniopCreateSwiGLUDescriptor(
    infiniopHandle_t handle,                        // Moore 设备句柄
    infiniopSwiGLUDescriptor_t *desc_ptr,          // 输出：描述符指针
    infiniopTensorDescriptor_t c_desc,             // 输出张量描述符
    infiniopTensorDescriptor_t a_desc,             // up 分支输入张量描述符
    infiniopTensorDescriptor_t b_desc);            // gate 分支输入张量描述符
// 功能：创建 SwiGLU 描述符，验证输入输出形状必须完全相同
// 返回：INFINI_STATUS_SUCCESS / INFINI_STATUS_BAD_TENSOR_DTYPE / INFINI_STATUS_BAD_TENSOR_STRIDES

__C infiniStatus_t infiniopGetSwiGLUWorkspaceSize(
    infiniopSwiGLUDescriptor_t desc,               // SwiGLU 描述符
    size_t *size);                                 // 输出：所需工作空间大小（字节）
// 功能：查询内核执行所需的工作空间大小（用于存储元数据和输入指针数组）
// 返回：INFINI_STATUS_SUCCESS / INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED

__C infiniStatus_t infiniopSwiGLU(
    infiniopSwiGLUDescriptor_t desc,               // SwiGLU 描述符
    void *workspace,                               // 工作空间指针（设备内存）
    size_t workspace_size,                         // 工作空间大小
    void *c,                                       // 输出张量 (output = SiLU(gate) * up)
    const void *a,                                 // up 分支输入张量
    const void *b,                                 // gate 分支输入张量
    void *stream);                                 // MUSA 流
// 功能：执行 SwiGLU 计算
// 返回：INFINI_STATUS_SUCCESS / INFINI_STATUS_INSUFFICIENT_WORKSPACE / 错误码

__C infiniStatus_t infiniopDestroySwiGLUDescriptor(
    infiniopSwiGLUDescriptor_t desc);              // SwiGLU 描述符
// 功能：销毁描述符并释放内存
// 返回：INFINI_STATUS_SUCCESS / INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED
```

## 4. 使用示例

```cpp
// 示例：在 Moore 设备上使用 SwiGLU 操作

#include "infiniop/ops/swiglu.h"
#include "infiniop/handle.h"

// 1. 创建 Moore 设备句柄
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_MOORE, 0);

// 2. 定义张量形状 (假设形状为 [batch_size, seq_len, hidden_dim])
std::vector<int64_t> shape = {32, 2048, 4096};
std::vector<int64_t> strides = {2048 * 4096, 4096, 1};  // 连续内存布局

// 3. 创建张量描述符
infiniopTensorDescriptor_t up_desc, gate_desc, out_desc;
infiniopCreateTensorDescriptor(&up_desc, INFINI_DTYPE_F16, shape.size(), shape.data(), strides.data());
infiniopCreateTensorDescriptor(&gate_desc, INFINI_DTYPE_F16, shape.size(), shape.data(), strides.data());
infiniopCreateTensorDescriptor(&out_desc, INFINI_DTYPE_F16, shape.size(), shape.data(), strides.data());

// 4. 创建 SwiGLU 描述符
infiniopSwiGLUDescriptor_t swiglu_desc;
infiniStatus_t status = infiniopCreateSwiGLUDescriptor(handle, &swiglu_desc, out_desc, up_desc, gate_desc);

// 5. 查询并分配工作空间
size_t workspace_size;
infiniopGetSwiGLUWorkspaceSize(swiglu_desc, &workspace_size);
void *d_workspace;
musaMalloc(&d_workspace, workspace_size);

// 6. 分配并初始化张量内存
void *d_up, *d_gate, *d_out;
size_t tensor_size = 32 * 2048 * 4096 * sizeof(half);  // batch * seq * hidden * sizeof(F16)
musaMalloc(&d_up, tensor_size);
musaMalloc(&d_gate, tensor_size);
musaMalloc(&d_out, tensor_size);
// ... 从主机复制数据到设备 ...

// 7. 获取或创建 MUSA 流
musaStream_t stream;
musaStreamCreate(&stream, 0);

// 8. 执行 SwiGLU 计算
status = infiniopSwiGLU(swiglu_desc, d_workspace, workspace_size, d_out, d_up, d_gate, stream);

// 9. 同步流以等待计算完成
musaStreamSynchronize(stream);

// 10. 清理资源
infiniopDestroySwiGLUDescriptor(swiglu_desc);
infiniopDestroyTensorDescriptor(up_desc);
infiniopDestroyTensorDescriptor(gate_desc);
infiniopDestroyTensorDescriptor(out_desc);
musaFree(d_workspace);
musaFree(d_up);
musaFree(d_gate);
musaFree(d_out);
musaStreamDestroy(stream);
infiniopDestroyHandle(handle);
```

## 5. 实现细节

### 核心算法与数学公式
SwiGLU 激活函数定义为：
```
output = SiLU(gate) × up
       = gate × σ(gate) × up
       = gate × (1 / (1 + e^(-gate))) × up
```
其中：
- `up` 和 `gate` 是两个线性变换的输出（形状必须相同）
- `σ(x)` 是 Sigmoid 函数
- SiLU (Swish) 函数 = `x × σ(x)`

### 内存管理策略
- **工作空间分配**:
  - 大小 = `info.getMetaMemSize() + info.getInputSize() * sizeof(void*)`
  - 元数据包含：输出形状、步长、所有输入的形状、步长、连续性标志、广播标志
  - 使用 MUSA 异步内存复制（`musaMemcpyAsync`）传输到设备
- **张量内存**: 由调用者在设备端预分配，内核直接读写
- **对齐要求**: 元数据使用 `std::vector<size_t>` 存储并按 `sizeof(size_t)` 对齐，通过 `CEIL_DIV(meta_mem_size, sizeof(size_t))` 计算所需 size_t 单元数

### 并发与执行模型
- **并行策略**:
  - 使用一维网格和块配置（`blockIdx.x * blockDim.x + threadIdx.x`）
  - 块大小固定为 256 线程（在 `swiglu_moore.mu` 的 `calculate` 方法中硬编码）
  - 网格大小动态计算：`min(CEIL_DIV(output_size, blockDims.x), internal->gridSizeX())`
- **流并发**: 支持在任意 MUSA 流上异步执行，使用 `<<<grid, block, 0, stream>>>` 启动内核
- **大张量处理**: 对于超过单个网格容量的张量，采用步进式多内核启动（`for` 循环 + `offset` 参数）确保所有元素被处理
- **原子操作**: 不需要（逐元素操作，各元素独立计算）

### 性能优化技术
- **向量化计算**:
  - 对 half 类型使用 `half2` 向量指令（2 个 FP16 元素打包处理）
  - 对 bfloat16 类型使用 `cuda_bfloat162` 向量指令
  - 向量化版本吞吐量约为标量版本的 2 倍
- **内置函数优化**:
  - Sigmoid 函数使用 `h2rcp` / `__frcp_rn` 快速倒数指令
  - 指数运算使用 `h2exp` / `__expf` 硬件加速指令
  - 乘法使用 `__hmul2` / `__fmul_rn` 融合乘加指令
- **编译期优化**:
  - 使用 `if constexpr` 实现模板特化，避免运行时类型判断
  - 使用 `__forceinline__` 强制内联小型函数（sigmoid、operator()）
  - 使用 `__restrict__` 指针限定符帮助编译器优化
- **内存访问优化**:
  - 连续内存路径：直接使用线性索引 `idx`，避免地址计算
  - 非连续/广播内存：调用 `device::moore::indexToOffset()` 计算物理偏移
  - 元数据只读存储在设备全局内存，通过指针间接访问

### 错误处理与验证
- **形状验证**:
  - 输出张量不能有广播维度（`output_desc->hasBroadcastDim()` 返回 false）
  - 三个张量形状必须完全相同（`CHECK_SAME_SHAPE` 宏验证）
- **数据类型支持**:
  - 支持的精度：FP16、BF16、FP32、FP64
  - 不支持的类型返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
- **工作空间检查**:
  - 运行时验证 `workspace_size >= _workspace_size`
  - 不满足时返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
- **错误传播**:
  - 使用 `CHECK_RESULT` / `CHECK_MOORE` / `CHECK_STATUS` 宏检查中间操作
  - 失败时立即返回错误码，不继续执行

### 依赖关系
- **内部依赖**:
  - `infiniop/elementwise/moore/elementwise_moore_api.h`: 通用逐元素操作 API 宏（`ELEMENTWISE_DESCRIPTOR`、`CREATE_ELEMENTWISE_MOORE_DESCRIPTOR`）
  - `infiniop/elementwise/moore/elementwise_moore.h`: Moore 设备端内核执行引擎
  - `infiniop/devices/moore/moore_common.h`: Moore 设备通用定义（`INFINIOP_MOORE_KERNEL` 宏、`CHECK_MOORE` 宏）
  - `infiniop/devices/moore/moore_kernel_common.h`: Moore 内核工具函数（`indexToOffset` 等）
- **外部依赖**:
  - MUSA 运行时 API（`musaStream_t`、`musaMemcpyAsync`、`musaMalloc` 等）
  - MUDA 内置函数（`half2`、`__hmul2`、`__frcp_rn` 等）
  - C++ 标准库（`std::vector`、`std::shared_ptr`、`std::enable_if_t` 等）
- **硬件要求**:
  - Moore 系列 GPU（如 MTT S80 等）
  - 支持 FP16/BF16 硬件加速（推荐使用 Tensor Core 单元）

### 设计模式
- **Pimpl 模式 (Pointer to Implementation)**:
  - `DeviceImpl` 使用 `Opaque` 内部结构隐藏实现细节
  - 通过 `std::shared_ptr<Opaque>` 管理生命周期
- **策略模式 (Strategy Pattern)**:
  - `SwiGLUOp` 作为可插拔的策略对象，支持不同的逐元素操作
  - 通过模板参数传入 `DeviceImpl::calculate<Op, Tdata>()`
- **工厂模式 (Factory Pattern)**:
  - `infiniopCreateSwiGLUDescriptor` 作为工厂函数，根据设备类型创建相应后端的描述符
  - `operator.cc` 中的宏 `CREATE(CASE, NAMESPACE)` 实现多后端分发
- **CRTP (Curiously Recurring Template Pattern)** 变体:
  - `ELEMENTWISE_DESCRIPTOR` 宏使用递归继承模式生成子类
  - 通过 `op::swiglu::moore::Descriptor` 命名空间约定避免名称冲突

### MUSA 平台特殊适配
本实现针对 MUSA 平台（沐璨 Moore GPU 的 CUDA 兼容平台）进行了特殊优化，与标准 CUDA 实现的主要区别：

1. **Half 精度 Sigmoid 实现**:
   - CUDA: 使用 `hrcp` 内置函数直接计算 half 精度倒数
   - MUSA: 由于缺少 `hrcp` 支持，提升到 float 精度计算（`__half2float` -> `1.0f / (1.0f + std::exp(-xf))` -> `__float2half`）

2. **BFloat16 向量化指令**:
   - CUDA: 使用 `__low2bfloat16` + `__bfloat162float` 两步提取和转换
   - MUSA: 使用 `__low2float()` / `__high2float()` 一步完成提取和转换（更高效）

3. **命名空间兼容性**:
   - 代码仍使用 `op::swiglu::cuda` 命名空间，以保持与 CUDA 后端的代码对齐
   - 通过 `moore` 特定的实现文件和宏编译隔离

4. **API 命名**:
   - 使用 MUSA 特定的类型名称（`musaStream_t` 替代 `cudaStream_t`）
   - 使用 `CHECK_MOORE` 宏替代 `CHECK_CUDA` 宏
