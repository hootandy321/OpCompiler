# SiLU METAX 算子核心实现文档

本模块实现了 SiLU (Sigmoid Linear Unit) 激活函数在 Metax 硬件加速器上的后端实现。SiLU 函数定义为 `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`，广泛应用于现代深度学习模型（如 LLaMA、PaLM 等）作为平滑的激活函数。本实现通过复用 Elementwise 通用框架和 CUDA 核函数，在 Metax 设备上提供高性能的计算支持。

## 1. 模块结构

- **`silu_metax.h`**: SiLU 算子的 MetAX 后端声明文件，通过宏展开生成 Descriptor 类接口
- **`silu_metax.maca`**: SiLU 算子的 MetAX 后端实现文件，包含 Descriptor 的创建与计算逻辑

## 2. 核心类

### `op::silu::metax::Descriptor`
- **位置**: `silu_metax.h` (宏展开), `silu_metax.maca`
- **主要功能**: SiLU 激活函数的 MetAX 设备描述符，负责管理算子的元数据、工作空间大小以及设备实现
- **核心成员**:
  - `_dtype`: `infiniDtype_t` - 输入输出张量的数据类型（BF16/F16/F32/F64）
  - `_info`: `op::elementwise::ElementwiseInfo` - 封装输入/输出张量的形状、步长、连续性等元数据
  - `_device_info`: `std::unique_ptr<op::elementwise::metax::DeviceImpl>` - MetAX 设备实现的具体操作对象
  - `_workspace_size`: `size_t` - 设备执行所需的工作空间大小（字节）
- **核心方法**:
  - `create(handle, desc_ptr, out_desc, input_desc_vec)`: 静态工厂方法，构造 Descriptor 实例
    - 验证数据类型支持（BF16, F16, F32, F64）
    - 验证输入输出张量形状一致性
    - 调用 `CREATE_ELEMENTWISE_METAX_DESCRIPTOR` 宏初始化 Elementwise 通用框架
  - `calculate(workspace, workspace_size, output, inputs, stream)`: 执行 SiLU 计算
    - 检查工作空间是否足够
    - 根据 `_dtype` 分发到对应的模板特化调用 `_device_info->calculate<256, cuda::SiluOp, T>()`
    - 块大小固定为 256 线程，使用 CUDA 定义的 `SiluOp` 算子
- **生命周期**:
  - 通过 `create()` 静态方法构造，分配内存并初始化所有成员
  - 析构函数默认实现，由智能指针自动管理资源
  - 所有权的唯一性由 `std::unique_ptr` 保证

## 3. API 接口

```cpp
// 创建 SiLU MetAX 描述符
static infiniStatus_t create(
    infiniopHandle_t handle,                    // MetAX 设备句柄
    Descriptor **desc_ptr,                      // [输出] 指向新创建的描述符指针
    infiniopTensorDescriptor_t out_desc,        // 输出张量描述符
    std::vector<infiniopTensorDescriptor_t> input_desc_vec  // 输入张量描述符向量（仅1个）
);
// 返回: INFINI_STATUS_SUCCESS / 错误码（类型不支持、形状不匹配等）

// 执行 SiLU 计算
infiniStatus_t calculate(
    void *workspace,                            // 设备工作空间指针
    size_t workspace_size,                      // 工作空间大小（字节）
    void *output,                               // 输出张量的设备指针
    std::vector<const void *> inputs,           // 输入张量的设备指针向量（inputs[0]为x）
    void *stream                                // MetAX 计算流（hcStream_t）
) const;
// 返回: INFINI_STATUS_SUCCESS / INFINI_STATUS_INSUFFICIENT_WORKSPACE / INFINI_STATUS_BAD_TENSOR_DTYPE
```

## 4. 使用示例

```cpp
// 示例：在 MetAX 设备上执行 SiLU 激活函数
#include "silu_metax.h"

// 假设已初始化 MetAX 句柄和流
infiniopHandle_t handle;           // MetAX 设备句柄
hcStream_t stream;                 // MetAX 计算流

// 定义张量形状 [batch_size, seq_len, hidden_dim]
std::vector<int64_t> shape = {32, 1024, 4096};
std::vector<int64_t> strides = {1024 * 4096, 4096, 1};  // 连续内存布局
infiniDtype_t dtype = INFINI_DTYPE_F16;  // 使用半精度浮点

// 创建输入输出张量描述符
infiniopTensorDescriptor_t input_desc, output_desc;
infiniopCreateTensorDescriptor(&input_desc, dtype, shape.size(),
                               shape.data(), strides.data());
infiniopCreateTensorDescriptor(&output_desc, dtype, shape.size(),
                               shape.data(), strides.data());

// 分配设备内存
void *d_input, *d_output;
hcMalloc(&d_input, 32 * 1024 * 4096 * sizeof(half));
hcMalloc(&d_output, 32 * 1024 * 4096 * sizeof(half));

// 创建 SiLU 描述符
op::silu::metax::Descriptor *silu_desc = nullptr;
infiniStatus_t status = op::silu::metax::Descriptor::create(
    handle, &silu_desc, output_desc, {input_desc}
);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误（类型不支持或形状不匹配）
}

// 查询并分配工作空间
size_t workspace_size = silu_desc->workspaceSize();
void *d_workspace;
hcMalloc(&d_workspace, workspace_size);

// 上传输入数据到设备（假设 h_input 为主机数据）
hcMemcpyAsync(d_input, h_input, 32 * 1024 * 4096 * sizeof(half),
              hcMemcpyHostToDevice, stream);

// 执行 SiLU 计算: output = silu(input)
status = silu_desc->calculate(d_workspace, workspace_size, d_output,
                              {d_input}, stream);

// 下载结果到主机（可选）
hcMemcpyAsync(h_output, d_output, 32 * 1024 * 4096 * sizeof(half),
              hcMemcpyDeviceToHost, stream);
hcStreamSynchronize(stream);

// 清理资源
hcFree(d_input);
hcFree(d_output);
hcFree(d_workspace);
delete silu_desc;
```

## 5. 实现细节

### 算法实现
- **数学公式**: `silu(x) = x / (1 + exp(-x))`，等价于 `x * sigmoid(x)`
- **CUDA 核函数复用**: 直接使用 `cuda::SiluOp` 函数对象（定义在 `../cuda/kernel.cuh`），该函数针对不同数据类型优化：
  - **half2 (FP16向量化)**: 使用 `__hmul2`, `__hadd2`, `h2exp`, `__hneg2` 等 CUDA intrinsic 实现向量化计算，吞吐量翻倍
  - **cuda_bfloat16 (BF16)**: 转换为 FP32 计算再转回 BF16，避免精度损失
  - **half (FP16)**: 同 BF16，转 FP32 计算再转回
  - **float (FP32)**: 直接使用 `__expf` 快速指数函数
  - **double (FP64)**: 使用标准 `exp` 函数保证双精度精度
- **模板特化策略**: `calculate()` 方法通过 `switch(_dtype)` 分发到 4 种模板特化，每种类型调用 `_device_info->calculate<256, cuda::SiluOp, T>()`

### 内存管理
- **工作空间布局**:
  - 输入指针数组：`sizeof(void*) * num_inputs` 字节
  - Elementwise 元数据（从 `ElementwiseInfo` 复制）：
    - 输出形状：`ndim * sizeof(size_t)`
    - 输出步长：`ndim * sizeof(ptrdiff_t)`
    - 所有输入形状：`num_inputs * ndim * sizeof(size_t)`
    - 所有输入步长：`num_inputs * ndim * sizeof(ptrdiff_t)`
    - 输入连续性标志：`num_inputs * sizeof(bool)`
    - 输入广播标志：`num_inputs * sizeof(bool)`
  - 总大小：`info.getMetaMemSize() + info.getInputSize() * sizeof(void*)`
- **设备端内存传递**: 使用 `hcMemcpyAsync` 将主机端元数据和工作空间指针异步传输到设备

### 并发执行
- **核函数启动参数**:
  - **块大小 (BLOCK_SIZE)**: 固定为 256 线程，与 MetAX 设备的 warp/wavefront 大小对齐
  - **网格大小**: `min(ceil_div(output_size, 256), max_grid_size)`，确保不超过设备限制
  - **大张量分块**: 对于超过 `grid_size * block_size` 的输出，通过循环分步启动核函数，每步处理 `step = gridDims.x * blockDims.x` 个元素
- **流式执行**: 计算在传入的 `hcStream_t` 流上异步执行，支持与其他算子重叠
- **线程安全**: 每个 Descriptor 实例的 `calculate()` 方法是线程安全的（只读成员），但多个实例不应共享同一流的不同工作空间

### 性能优化
- **连续内存快速路径**: 当张量连续时 (`is_contiguous == true`)，直接使用线性索引 `idx`，避免昂贵的 `indexToOffset` 计算
- **向量化计算**: FP16 类型优先使用 `half2` 向量指令，一次处理两个元素，理论吞吐量提升 2x
- **广播优化**: 通过 `InputIndexer` 结构体统一处理广播和步长，避免分支判断
- **元数据预计算**: `ElementwiseInfo` 在创建时预计算所有形状、步长、连续性标志，运行时直接复制到设备，避免重复计算
- **时间复杂度**: O(n)，其中 n 为输出张量元素数量，每个元素独立计算

### 错误处理
- **数据类型验证**: 仅支持 BF16, F16, F32, F64，其他类型返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
- **形状一致性检查**: 使用 `CHECK_SAME_SHAPE` 宏验证输入输出形状完全匹配，否则返回错误
- **工作空间验证**: 如果 `workspace_size < _workspace_size`，返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
- **元数据构造错误**: `ElementwiseInfo::create()` 失败时，通过 `CHECK_RESULT` 宏传播错误码
- **设备内存复制失败**: 使用 `CHECK_METAX` 宏捕获 `hcMemcpyAsync` 错误并返回对应状态

### 依赖关系
- **Elementwise 框架**:
  - `op::elementwise::ElementwiseInfo`: 封装张量元数据的结构体
  - `op::elementwise::metax::DeviceImpl`: MetAX 设备的通用逐元素计算实现
  - `CREATE_ELEMENTWISE_METAX_DESCRIPTOR` 宏：自动创建 Descriptor 的辅助宏
- **CUDA 核函数复用**:
  - `cuda::SiluOp`: 定义在 `../cuda/kernel.cuh` 的函数对象，提供 `operator()(T x)` 方法
  - 该设计允许 MetAX 后端直接复用 NVIDIA 平台验证过的数学计算逻辑
- **MetAX 设备接口**:
  - `device::metax::Handle`: MetAX 设备句柄，封装硬件连接
  - `hcStream_t`, `hcMemcpyAsync`, `hcMalloc`: MetAX 的 HIP/CUDA 兼容 API
  - `INFINIOP_METAX_KERNEL` 宏：标记设备核函数

### 设计模式
- **宏生成模式 (Macro-based Code Generation)**:
  - `ELEMENTWISE_DESCRIPTOR(silu, metax)` 宏自动生成完整的 Descriptor 类定义
  - 避免手动编写重复的样板代码，确保所有逐元素算子接口一致性
- **模板方法模式 (Template Method)**:
  - `Descriptor::calculate()` 定义算法骨架（类型检查、工作空间验证），具体计算委托给 `_device_info->calculate<>()`
  - 子类（如不同硬件后端）可复用相同的接口，只需替换 `DeviceImpl` 实现
- **策略模式 (Strategy)**:
  - `cuda::SiluOp` 作为策略对象，通过模板参数传入通用核函数框架
  - 支持运行时选择不同的数据类型策略（BF16/F16/F32/F64）
- **RAII 资源管理**:
  - `Descriptor` 使用 `std::unique_ptr` 管理 `DeviceImpl`，自动释放设备资源
  - `ElementwiseInfo` 使用移动语义避免不必要的内存拷贝

### 硬件后端特性
- **MetAX 架构适配**:
  - MetAX 是沐曦（Moore Threads）的国产 GPU 架构，兼容 HIP/CUDA 编程模型
  - 本实现通过复用 CUDA 核函数和 Elementwise 框架，实现代码跨平台复用
  - 核函数使用 `INFINIOP_METAX_KERNEL` 宏标记，确保在 MetAX 工具链中正确编译
- **多硬件后端统一抽象**:
  - 目录结构 `infiniop/ops/silu/{cuda, metax, cpu, kunlun, bang}` 体现相同算子在不同硬件的实现
  - MetAX 后端与 CUDA 后端共享 `kernel.cuh` 中的数学逻辑，仅设备接口层不同
- **性能可移植性**:
  - 块大小 (256) 适配 MetAX 的 SIMT 宽度，保证高占用率
  - 核函数网格启动逻辑考虑 MetAX 设备的 `maxThreadsPerBlock` 和 `gridSizeX` 限制
