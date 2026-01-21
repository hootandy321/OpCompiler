# NVIDIA CUDA 减法运算算子核心实现文档

本模块实现了基于 NVIDIA CUDA 的张量逐元素减法运算（Element-wise Subtraction），作为 Infini 框架算子体系的一部分，提供高性能的 GPU 加速减法计算能力。该实现支持多种浮点数据类型（FP16/BF16/FP32/FP64），并具备广播（Broadcasting）、步长张量（Strided Tensor）等高级特性。

## 1. 模块结构

- **`sub_nvidia.cuh`**：定义减法算子的描述符接口，通过宏 `ELEMENTWISE_DESCRIPTOR` 生成统一的 API 接口类
- **`sub_nvidia.cu`**：实现减法算子的核心逻辑，包括描述符创建（`create`）和计算执行（`calculate`）
- **`kernel.cuh`**（位于 `../cuda/`）：定义 CUDA 设备端减法运算函数对象（`SubOp`），针对不同数据类型使用优化的硬件指令

## 2. 核心类

### `op::sub::nvidia::Descriptor`
- **位置**：通过 `ELEMENTWISE_DESCRIPTOR` 宏定义于 `sub_nvidia.cuh`
- **主要功能**：封装 CUDA 减法算子的所有元数据和执行逻辑，管理张量描述信息、设备实现对象和工作空间大小
- **核心成员**：
  - `infiniDtype_t _dtype`：输出张量的数据类型（FP16/F32/F64/BF16）
  - `op::elementwise::ElementwiseInfo _info`：存储张量形状、步长、广播等元数据的结构体
  - `std::unique_ptr<op::elementwise::nvidia::DeviceImpl> _device_info`：CUDA 设备端实现对象，负责内核启动
  - `size_t _workspace_size`：执行计算所需的 GPU 工作空间大小（存储元数据和输入指针数组）
- **核心方法**：
  - `create(handle_, desc_ptr, out_desc, input_desc_vec)`：静态工厂方法，构造减法描述符对象。执行类型检查（仅支持浮点类型）、形状一致性验证（要求所有输入输出形状完全相同或可广播），并初始化底层 Elementwise 元数据和 CUDA 设备实现。
  - `calculate(workspace, workspace_size, output, inputs, stream)`：执行减法运算的异步 CUDA 内核调用。根据 `_dtype` 分发到对应的数据类型特化版本，调用 `DeviceImpl::calculate` 启动 GPU 计算。工作空间必须满足最小大小要求，否则返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE` 错误。
- **生命周期**：由 `create` 静态方法构造，通过 `std::unique_ptr` 管理 `DeviceImpl` 资源，析构时自动释放

### `op::sub::cuda::SubOp`
- **位置**：`../cuda/kernel.cuh`
- **主要功能**：CUDA 设备端函数对象（Functor），定义减法运算的语义
- **静态常量**：
  - `num_inputs = 2`：固定两个输入张量
- **核心方法**：
  - `operator()(const T& a, const T& b)`：对两个标量执行 `a - b`。使用 `if constexpr` 编译期分支针对不同数据类型优化：
    - `half2`/`cuda_bfloat162`（向量类型）：调用 `__hsub2` 硬件指令（FP16 向量化减法）
    - `half`/`cuda_bfloat16`：调用 `__hsub` 硬件指令（FP16 标量减法）
    - `float`：调用 `__fsub_rd` 硬件指令（向负无穷舍入的浮点减法）
    - 其他类型（如 `double`）：直接使用 `-` 运算符

### `op::elementwise::nvidia::DeviceImpl`
- **位置**：`../../elementwise/nvidia/elementwise_nvidia.cuh`
- **主要功能**：封装 CUDA 内核启动逻辑，管理 GPU 内存传输和内核执行参数
- **实现模式**：Pimpl（Pointer to Implementation）模式，通过 `std::shared_ptr<Opaque>` 隐藏实现细节
- **核心方法**：
  - `calculate<BLOCK_SIZE, Op, Tdata>(...)`：当所有输入输出类型相同时，启动统一类型的 CUDA 内核。模板参数 `BLOCK_SIZE=256` 控制每个线程块的线程数，`Op=SubOp` 指定运算逻辑，`Tdata` 为数据类型。
  - `calculate<BLOCK_SIZE, Op, Tout, Tin...>(...)`：当输入输出类型不同时（如类型转换），启动混合类型的 CUDA 内核。编译期检查 `sizeof...(Tin) == Op::num_inputs` 确保输入数量匹配。
- **内核启动细节**：
  - 将主机端元数据（形状、步长、广播标志）异步拷贝到 GPU 工作空间
  - 计算网格维度：`gridDims.x = min(CEIL_DIV(output_size, 256), device_grid_size_limit)`
  - 大张量自动分块执行（step-based partitioning），每个网格处理 `gridDims.x * blockDims.x` 个元素
  - 支持非连续张量：通过 `indexToOffset` 将线性索引映射到实际内存偏移

## 3. API 接口

```cpp
namespace op::sub::nvidia {

// 创建减法算子描述符
static infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,                // [入] Infini 设备句柄（包含 CUDA 上下文）
    Descriptor **desc_ptr,                   // [出] 输出的描述符指针
    infiniopTensorDescriptor_t out_desc,     // [入] 输出张量描述符
    std::vector<infiniopTensorDescriptor_t> input_desc_vec  // [入] 输入张量描述符向量（必须恰好2个）
);
// 返回：INFINI_STATUS_SUCCESS / INFINI_STATUS_BAD_TENSOR_DTYPE / INFINI_STATUS_BAD_SHAPE

// 执行减法计算
infiniStatus_t Descriptor::calculate(
    void *workspace,                         // [入] GPU 工作空间指针
    size_t workspace_size,                   // [入] 工作空间大小（字节）
    void *output,                            // [出] 输出张量 GPU 指针
    std::vector<const void *> inputs,        // [入] 输入张量 GPU 指针向量（必须恰好2个）
    void *stream                             // [入] CUDA 流（cudaStream_t）
) const;
// 返回：INFINI_STATUS_SUCCESS / INFINI_STATUS_INSUFFICIENT_WORKSPACE / INFINI_STATUS_BAD_TENSOR_DTYPE

}
```

## 4. 使用示例

```cpp
// 初始化 Infini 句柄
infiniopHandle_t handle;
infiniopCreateHandle(reinterpret_cast<infiniHandle_t>(&cuda_context),
                     INFINI_DEVICE_NVIDIA, 0, &handle);

// 准备张量描述符（假设形状为 [1024, 1024]）
int64_t shape[] = {1024, 1024};
int64_t strides[] = {1024, 1};  // 行主序连续张量
infiniopTensorDescriptor_t a_desc, b_desc, c_desc;
infiniopCreateTensorDescriptor(&a_desc, INFINI_DTYPE_F32, 2, shape, strides);
infiniopCreateTensorDescriptor(&b_desc, INFINI_DTYPE_F32, 2, shape, strides);
infiniopCreateTensorDescriptor(&c_desc, INFINI_DTYPE_F32, 2, shape, strides);

// 创建减法算子描述符
op::sub::nvidia::Descriptor *sub_desc;
auto status = op::sub::nvidia::Descriptor::create(
    handle, &sub_desc, c_desc, {a_desc, b_desc});
if (status != INFINI_STATUS_SUCCESS) {
    // 错误处理：检查 dtype 是否为浮点类型，形状是否兼容
}

// 分配 GPU 内存并初始化输入数据
float *d_a, *d_b, *d_c;
cudaMalloc(&d_a, 1024 * 1024 * sizeof(float));
cudaMalloc(&d_b, 1024 * 1024 * sizeof(float));
cudaMalloc(&d_c, 1024 * 1024 * sizeof(float));
// ... 填充 d_a, d_b 数据 ...

// 分配工作空间
size_t workspace_size = sub_desc->workspaceSize();
void *d_workspace;
cudaMalloc(&d_workspace, workspace_size);

// 创建 CUDA 流并执行计算
cudaStream_t stream;
cudaStreamCreate(&stream);
status = sub_desc->calculate(d_workspace, workspace_size, d_c, {d_a, d_b}, stream);

// 同步等待计算完成
cudaStreamSynchronize(stream);

// 清理资源
cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_workspace);
cudaStreamDestroy(stream);
delete sub_desc;
infiniopDestroyHandle(handle);
```

## 5. 实现细节

### 内存管理
- **工作空间布局**：工作空间存储两部分数据：
  1. 输入指针数组（`N * sizeof(void*)`，N 为输入数量）
  2. 元数据区域（通过 `ElementwiseInfo::getMetaMemSize()` 计算），包含输出形状、步长、所有输入的形状/步长/连续性/广播标志
- **内存传输**：使用 `cudaMemcpyAsync` 异步拷贝元数据到 GPU，由 `infoToDevice` 方法执行
- **张量访问**：内核通过 `InputIndexer` 函数对象计算每个输入的元素偏移，自动处理广播和步长

### 并发与性能
- **线程配置**：固定使用 `BLOCK_SIZE = 256` 线程/块，平衡寄存器使用和 occupancy
- **网格限制**：网格 X 维度受限于 `internal->gridSizeX()`（通常为 2^31 - 1 或设备特定限制）
- **内核分区**：当 `output_size > gridDims.x * blockDims.x` 时，通过循环多次启动内核，每次偏移 `i += step`
- **硬件指令优化**：
  - FP16：使用 `__hsub` / `__hsub2` 指令（Tensor Core 加速）
  - FP32：使用 `__fsub_rd` 舍入指令（可配置舍入模式）
  - FP64：直接使用 `-` 运算符（依赖 PTX 指令）
- **向量化**：通过 `half2` 和 `cuda_bfloat162` 类型实现 2 元素向量 SIMD 指令（需编译器自动向量化或手动扩展）

### 错误处理
- **类型检查**：`CHECK_DTYPE` 宏验证输出 dtype 是否在 `INFINI_DTYPE_F16/F32/F64/BF16` 中
- **形状验证**：`CHECK_SAME_SHAPE` 宏确保输出和所有输入形状完全匹配（不支持广播减法，要求严格形状对齐）
- **工作空间验证**：`calculate` 方法检查 `workspace_size < _workspace_size`，返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
- **CUDA 错误传播**：使用 `CHECK_CUDA` 宏将 `cudaError_t` 转换为 `infiniStatus_t`
- **Result 类型**：`DeviceImpl::create` 和 `ElementwiseInfo::create` 返回 `utils::Result<T>`，通过 `CHECK_RESULT` 宏解包错误

### 依赖关系
- **Elementwise 框架**：复用 `op::elementwise::nvidia::DeviceImpl` 和 `ElementwiseInfo`，实现通用的逐元素算子基础设施
- **CUDA 基础库**：依赖 `device::nvidia::Handle`（设备上下文）、`device::nvidia::indexToOffset`（索引计算）、`INFINIOP_CUDA_KERNEL`（内核启动宏）
- **张量抽象**：使用 `infiniopTensorDescriptor_t` 查询 `dtype()`、`shape()`、`ndim()` 等元信息
- **宏定义**：
  - `ELEMENTWISE_DESCRIPTOR(sub, nvidia)`：在 `sub_nvidia.cuh` 中生成 `Descriptor` 类定义
  - `CREATE_ELEMENTWISE_CUDA_DESCRIPTOR`：在 `sub_nvidia.cu` 的 `create` 方法中构造 `ElementwiseInfo` 和 `DeviceImpl`

### 设计模式
- **策略模式（Strategy）**：`SubOp` 作为可插拔的运算策略，`ElementwiseInfo` 作为张量元数据策略，`DeviceImpl` 作为设备执行策略
- **模板方法模式（Template Method）**：`calculate` 方法定义算法骨架，`calculateImpl` 模板方法实现类型特化
- **工厂模式（Factory）**：`create` 静态方法作为工厂，封装对象构造逻辑
- **Pimpl 惯用法**：`DeviceImpl` 通过 `Opaque` 结构体隐藏 CUDA 实现细节，减少头文件依赖
- **CRTP（Curiously Recurring Template Pattern）**：`ELEMENTWISE_DESCRIPTOR` 宏生成继承自 `InfiniopDescriptor` 的类，通过模板参数定制算子名称和命名空间

### 算法复杂度
- **时间复杂度**：O(n)，其中 n 为输出张量元素数量。每个元素恰好执行一次减法运算，无额外循环或依赖
- **空间复杂度**：O(1) 额外空间（工作空间大小与 n 无关，仅与元数据大小成正比）
- **并行度**：理论并行度为 n（每个元素独立计算），实际受限于 GPU 核心数量和内存带宽
