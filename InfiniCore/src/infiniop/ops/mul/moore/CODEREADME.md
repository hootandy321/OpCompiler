# Moore Backend 乘法运算实现文档

本文档详细描述了 Moore 硬件后端上的逐元素乘法（Element-wise Multiplication）运算实现。该模块基于通用的逐元素运算框架，针对 Moore 平台（MUSA 架构）进行了优化实现。

## 1. 模块结构

- **`mul_moore.h`**: API 接口定义层，通过宏展开生成 Descriptor 类声明
- **`mul_moore_kernel.h`**: 核心计算内核定义，包含 MulOp 函数对象及各种数据类型的乘法实现
- **`mul_moore.mu`**: Descriptor 类实现，包含算子创建和计算调度逻辑

## 2. 核心类与数据结构

### `op::mul::moore::Descriptor` 类
- **位置**: 通过 `ELEMENTWISE_DESCRIPTOR` 宏在 `mul_moore.h` 中自动生成，实现在 `mul_moore.mu`
- **主要功能**: 封装 Moore 平台上乘法运算的完整执行上下文，继承自 `InfiniopDescriptor` 基类
- **关键成员变量**:
  - `_dtype`: `infiniDtype_t` 类型，存储输出张量的数据类型（F16/BF16/F32/F64）
  - `_info`: `op::elementwise::ElementwiseInfo` 类型，存储输入/输出张量的形状、步幅、连续性等元数据
  - `_device_info`: `std::unique_ptr<op::elementwise::moore::DeviceImpl>` 类型，持有设备端实现细节（Opaque 模式）
  - `_workspace_size`: `size_t` 类型，设备端工作空间大小（用于存放元数据和输入指针数组）

- **核心方法**:
  - **`create(handle_, desc_ptr, out_desc, input_desc_vec)`**: 静态工厂方法，负责：
    1. 类型检查：验证输出 dtype 为 F16/F32/F64/BF16 之一
    2. 形状验证：确保所有输入和输出张量形状完全一致（`CHECK_SAME_SHAPE`）
    3. 元数据构建：调用 `ElementwiseInfo::create()` 生成广播和步幅信息
    4. 设备实现创建：通过 `CREATE_ELEMENTWISE_MOORE_DESCRIPTOR` 宏初始化 DeviceImpl
    5. 内存计算：工作空间 = `info.getMetaMemSize() + info.getInputSize() * sizeof(void*)`

  - **`calculate(workspace, workspace_size, output, inputs, stream)`**: 执行乘法运算：
    1. 工作空间大小检查：不足则返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
    2. 类型分发：根据 `_dtype` 调用模板化的 DeviceImpl::calculate 方法：
       - `INFINI_DTYPE_F16` → `calculate<256, moore::MulOp, half>`
       - `INFINI_DTYPE_BF16` → `calculate<256, moore::MulOp, cuda_bfloat16>`
       - `INFINI_DTYPE_F32` → `calculate<256, moore::MulOp, float>`
       - `INFINI_DTYPE_F64` → `calculate<256, moore::MulOp, double>`
    3. 线程配置：固定使用 BLOCK_SIZE=256 的块大小

- **生命周期**: 由 `create()` 静态方法构造，析构函数默认实现（`= default`），通过智能指针管理 DeviceImpl 生命周期

### `op::mul::moore::MulOp` 结构体
- **位置**: `mul_moore_kernel.h`
- **主要功能**: 设备端函数对象（Functor），定义单个元素的乘法操作
- **关键成员**:
  - `num_inputs`: `static constexpr size_t` 值为 2，声明该操作需要两个输入

- **核心方法**:
  - **`operator()(const T& a, const T& b) const`**: 模板化函数调用运算符：
    - **`half2` 类型**: 调用 `__hmul2(a, b)` 执行 SIMD 向量乘法（每条指令处理两个 FP16 值）
    - **`half` 类型**: 调用 `__hmul(a, b)` 执行标量 FP16 乘法
    - **`cuda_bfloat16` 类型**: Moore 平台特殊处理路径
      - 通过 `__bfloat162float(a)` 将 BF16 转换为 FP32
      - 执行 FP32 乘法 `a_f * b_f`
      - 通过 `__float2bfloat16_rn()` 结果舍入并转换回 BF16（RN = Round to Nearest）
    - **`float` 类型**: 调用 `__fmul_rn(a, b)` 使用 Moore 平台特定的乘法指令（带舍入控制）
    - **其他类型**: 使用原生 C++ `*` 运算符（如 double 类型）

### `op::elementwise::moore::DeviceImpl` 类（依赖模块）
- **位置**: `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/moore/elementwise_moore.h`
- **主要功能**: 通用的逐元素运算设备端执行引擎，采用 Pimpl（Pointer to Implementation）模式
- **内部结构**:
  - **`Opaque` 内部类**: 持有 `std::shared_ptr<device::moore::Handle::Internal>` 设备句柄
  - **`calculateImpl` 模板方法**: 执行内核启动逻辑
  - **`infoToDevice` 私有方法**: 将主机端元数据异步复制到设备端
  - **`launchElementwiseKernel` 私有方法**: 计算网格/块维度并启动 CUDA 内核

## 3. API 接口

```cpp
// 创建乘法算子描述符
namespace op::mul::moore {
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,                      // Moore 设备句柄
    Descriptor **desc_ptr,                         // 输出：创建的描述符指针
    infiniopTensorDescriptor_t out_desc,           // 输出张量描述符
    std::vector<infiniopTensorDescriptor_t> input_desc_vec  // 输入张量描述符向量 [A, B]
);
// 返回值：INFINI_STATUS_SUCCESS / INFINI_STATUS_BAD_TENSOR_DTYPE / 错误码

// 执行乘法计算
infiniStatus_t Descriptor::calculate(
    void *workspace,                               // 设备端工作空间指针（预分配）
    size_t workspace_size,                         // 工作空间大小（字节）
    void *output,                                  // 输出张量设备指针
    std::vector<const void *> inputs,              // 输入张量设备指针向量 [A_ptr, B_ptr]
    void *stream                                   // MUSA 流指针
) const;
// 返回值：INFINI_STATUS_SUCCESS / INFINI_STATUS_INSUFFICIENT_WORKSPACE / INFINI_STATUS_BAD_TENSOR_DTYPE
}
```

## 4. 使用示例

```cpp
#include "mul_moore.h"

// 1. 初始化 Moore 设备句柄（假设已存在）
infiniopHandle_t handle;
// ... handle 初始化逻辑 ...

// 2. 创建张量描述符（假设输入和输出形状相同）
int64_t shape[] = {1024, 1024};
int64_t strides[] = {1024, 1};

infiniopTensorDescriptor_t A_desc, B_desc, C_desc;
infiniopCreateTensorDescriptor(&A_desc, INFINI_DTYPE_F16, 2, shape, strides);
infiniopCreateTensorDescriptor(&B_desc, INFINI_DTYPE_F16, 2, shape, strides);
infiniopCreateTensorDescriptor(&C_desc, INFINI_DTYPE_F16, 2, shape, strides);

// 3. 创建乘法算子描述符
op::mul::moore::Descriptor *mul_desc = nullptr;
std::vector<infiniopTensorDescriptor_t> inputs_desc = {A_desc, B_desc};
auto status = op::mul::moore::Descriptor::create(handle, &mul_desc, C_desc, inputs_desc);
if (status != INFINI_STATUS_SUCCESS) {
    // 错误处理
}

// 4. 分配设备内存和工作空间
half *d_A, *d_B, *d_C;
size_t tensor_size = 1024 * 1024 * sizeof(half);
musaMalloc(&d_A, tensor_size);
musaMalloc(&d_B, tensor_size);
musaMalloc(&d_C, tensor_size);

size_t workspace_size = mul_desc->workspaceSize();
void *d_workspace;
musaMalloc(&d_workspace, workspace_size);

// 5. 准备输入数据（主机到设备拷贝）
// musaMemcpyAsync(d_A, h_A, tensor_size, musaMemcpyHostToDevice, stream);
// musaMemcpyAsync(d_B, h_B, tensor_size, musaMemcpyHostToDevice, stream);

// 6. 执行乘法运算
musaStream_t stream;
// ... 获取或创建流 ...

std::vector<const void *> inputs = {d_A, d_B};
status = mul_desc->calculate(d_workspace, workspace_size, d_C, inputs, stream);
if (status != INFINI_STATUS_SUCCESS) {
    // 错误处理
}

// 7. 同步并读取结果
// musaStreamSynchronize(stream);
// musaMemcpyAsync(h_C, d_C, tensor_size, musaMemcpyDeviceToHost, stream);

// 8. 清理资源
musaFree(d_A);
musaFree(d_B);
musaFree(d_C);
musaFree(d_workspace);
delete mul_desc;
infiniopDestroyTensorDescriptor(A_desc);
infiniopDestroyTensorDescriptor(B_desc);
infiniopDestroyTensorDescriptor(C_desc);
```

## 5. 实现细节

### 5.1 内存管理策略
- **工作空间布局**: 线性布局分为两部分：
  1. 输入指针数组（`input_size * sizeof(void*)` 字节）
  2. 元数据区域（`info.getMetaMemSize()` 字节），包含：
     - 输出形状（ndim 个 size_t）
     - 输出步幅（ndim 个 ptrdiff_t）
     - 所有输入的形状（input_size × ndim 个 size_t）
     - 所有输入的步幅（input_size × ndim 个 ptrdiff_t）
     - 输入连续性标志（input_size 个 bool）
     - 输入广播标志（input_size 个 bool）
- **异步拷贝**: 通过 `musaMemcpyAsync(..., musaMemcpyHostToDevice, stream)` 将主机端元数据传输到设备端，避免阻塞
- **设备端指针偏移**: 使用指针算术定位元数据各部分，避免二次分配

### 5.2 并发与线程调度
- **线程块配置**: 固定使用 `BLOCK_SIZE=256`，实际块维度为 `min(256, maxThreadsPerBlock)`
- **网格配置**: 网格 X 维度为 `min(ceil(output_size / blockDims.x), gridSizeX)`
- **大张量分块处理**: 当 `output_size > gridDims.x * blockDims.x` 时，通过循环多次启动内核处理不同片段：
  ```cpp
  size_t step = gridDims.x * blockDims.x;
  for (size_t i = 0; i < output_size; i += step) {
      kernel_func<<<gridDims, blockDims, 0, stream>>>(..., offset=i, ...);
  }
  ```
- **无竞争写入**: 每个线程处理唯一的输出元素，通过 `idx = blockIdx.x * blockDim.x + threadIdx.x + offset` 计算全局索引

### 5.3 性能优化技术
- **向量化指令**:
  - `half2` 类型使用 `__hmul2` 实现 SIMD 并行（每指令 2 个 FP16 乘法）
  - BF16 通过先转 FP32 再乘法，利用 Moore 平台的 FP32 单元吞吐量
- **类型特化优化**:
  - FP32 使用 `__fmul_rn` 硬件指令（带 IEEE 754 舍入模式）
  - 避免通用乘法的类型转换开销
- **快速路径优化**: 对连续内存布局（`is_contiguous == true`）的输出，直接使用线性索引 `idx`，跳过昂贵的 `indexToOffset` 坐标转换
- **编译期优化**:
  - `constexpr size_t num_inputs = 2` 允许编译器展开输入索引循环
  - `if constexpr` 进行编译期类型分支，零运行时开销

### 5.4 错误处理机制
- **类型验证**: `CHECK_DTYPE` 宏确保支持的数据类型（F16/F32/F64/BF16），否则返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
- **形状一致性**: `CHECK_SAME_SHAPE` 验证所有输入和输出张量的形状完全匹配
- **工作空间检查**: `calculate()` 方法验证 `workspace_size >= _workspace_size`，不足时返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
- **错误传播**: 使用 `CHECK_RESULT` 和 `CHECK_MOORE` 宏封装底层 API 调用，统一错误码转换

### 5.5 平台适配特性
- **命名空间策略**: 使用 `op::mul::moore` 命名空间而非 `op::mul::musa`，保持代码架构的硬件中立性（Moore 为 MUSA 的硬件代号）
- **BF16 兼容性**: Moore 平台可能缺乏原生 BF16 乘法指令，采用 FP32 转换路径（牺牲部分性能换取正确性）
- **MUSA 生态集成**:
  - 流类型使用 `musaStream_t`（而非 CUDA 的 `cudaStream_t`）
  - 内核标记使用 `INFINIOP_MOORE_KERNEL` 宏
  - 设备句柄转换：`reinterpret_cast<device::moore::Handle *>(handle_)`

### 5.6 设计模式
- **Pimpl 模式**: `DeviceImpl` 通过 `Opaque` 内部类隐藏实现细节，减少头文件依赖
- **策略模式**: `MulOp` 函数对象作为可插拔的计算策略，与通用的逐元素框架解耦
- **CRTP（奇异递归模板模式）**: `ELEMENTWISE_DESCRIPTOR` 宏通过宏展开自动生成完整的 Descriptor 类定义，避免代码重复
- **工厂模式**: `create()` 静态方法封装复杂的对象构造逻辑
- **模板特化**: 通过 `std::enable_if_t` 和 `if constexpr` 实现编译期多态

### 5.7 依赖关系
- **上游依赖**:
  - `op::elementwise::ElementwiseInfo`: 提供张量元数据管理
  - `op::elementwise::moore::DeviceImpl`: 提供内核启动和内存管理基础设施
  - `device::moore::Handle`: MUSA 设备抽象层
- **外部依赖**:
  - MUSA Runtime API (`musaMemcpyAsync`, `musaStream_t`)
  - Moore 内建函数 (`__hmul2`, `__fmul_rn`, `__bfloat162float`)
  - CUDA 类型定义（`half`, `half2`, `cuda_bfloat16`）

### 5.8 算法复杂度
- **时间复杂度**: O(N)，其中 N 为输出张量的元素数量（线性扫描每个元素一次）
- **空间复杂度**: O(N) 输出内存 + O(M) 工作空间（M = 元数据大小，与张量维度和输入数量成正比）
- **并行度**: 理论上可并发处理 N 个元素（受限于 GPU 设备的并行计算单元数量）

---

## 补充说明

**Moore 平台背景**: Moore 是 Moore Threads（摩尔线程）公司开发的 GPU 硬件架构，运行 MUSA（类似于 CUDA）软件栈。该实现针对 Moore 平台的指令集特性进行了专门优化，同时保持与 CUDA 实现的代码结构一致性。
