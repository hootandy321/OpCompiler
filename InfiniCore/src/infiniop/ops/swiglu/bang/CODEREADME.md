# SwiGLU BANG 算子核心实现文档

SwiGLU (Swish-Gated Linear Unit) 是一种高效的激活函数，广泛应用于现代大语言模型（如 LLaMA、PaLM 等）。本模块实现了 SwiGLU 算子在寒武纪 BANG 硬件加速平台上的高性能版本，针对 MLU 设备进行了深度优化，支持 FP16、BF16 和 FP32 三种数据类型。

## 1. 模块结构

- **`swiglu_bang.h`**: 公共 API 声明，通过 ELEMENTWISE_DESCRIPTOR 宏定义算子描述符接口
- **`swiglu_bang.mlu`**: 算子描述符实现，包含 Descriptor 类的 create/calculate 方法以及 SwiGLUOp 主机端启动器
- **`swiglu_bang_internal.mlu`**: 设备端核心实现，包含 SwiGLUOp 设备函数和内核启动逻辑

## 2. 核心类

### `op::swiglu::bang::Descriptor`
- **位置**: `swiglu_bang.mlu`
- **主要功能**: SwiGLU 算子的顶层描述符，继承自通用的 Elementwise BANG 描述符基类，管理算子生命周期和执行调度
- **关键成员** (继承自基类):
  - `_workspace_size`: size_t - 设备端工作空间大小（存储元数据）
  - `_dtype`: infiniopDtype_t - 输出/输入张量的数据类型
  - `_device_info`: std::unique_ptr<DeviceImpl> - 设备特定实现的智能指针
  - `_info`: ElementwiseInfo - 张量形状、步长、广播等元数据
- **核心方法**:
  - `create(infiniopHandle_t, Descriptor**, infiniopTensorDescriptor_t, std::vector<infiniopTensorDescriptor_t>)`: 静态工厂方法，验证张量形状一致性（输出、up、gate 三者形状必须相同），检查数据类型（仅支持 F16/F32/BF16），创建 BANG Elementwise 描述符实例
  - `calculate(void *workspace, size_t workspace_size, void *output, std::vector<const void*> inputs, void *queue) const`: 主计算入口，根据数据类型分发到模板实例化（half/bfloat16_t/float），调用 DeviceImpl::calculate 启动内核执行，返回 INFINI_STATUS_SUCCESS 或错误码（如 INFINI_STATUS_INSUFFICIENT_WORKSPACE）
  - `~Descriptor()`: 默认析构函数，由编译器自动生成
- **生命周期**: 用户通过 create() 创建实例（构造时分配设备元数据），通过 calculate() 执行计算，析构时自动释放 DeviceImpl 资源

### `SwiGLUOp` (设备端算子)
- **位置**: `swiglu_bang_internal.mlu`
- **主要功能**: 实现 SwiGLU 数学运算 `output = up * sigmoid(gate)` 的设备端核心逻辑，针对不同数据类型使用不同的 BANG 指令优化路径
- **关键成员**:
  - `num_inputs`: static constexpr size_t = 2 - 输入张量数量（up 和 gate）
- **核心方法**:
  - `operator()(T *out, const T *up, const T *gate, size_t num_elements) const`: __mlu_device__ 修饰的设备端函数，执行逐元素 SwiGLU 计算：
    - **half/bfloat16_t 优化路径**: 使用 BANG 内置指令 `__bang_active_sigmoid` 计算 sigmoid，然后通过两次 `__bang_mul` 实现乘法（先 sigmoid*gate，再 *up），利用硬件加速单元
    - **float 优化路径**: 通过 `__bang_neg` + `__bang_active_exphp` + `__bang_add_scalar` + `__bang_div` 组合实现 sigmoid（基于数值稳定的公式 sigmoid(x) = 1 / (1 + e^(-x))），最后 `__bang_mul(up, result)` 完成 gate*up
    - **通用回退路径**: for 循环逐元素计算 `out[i] = up[i] * gate[i] / (1.0 + exp(-gate[i]))`，确保其他类型可编译但性能较低
- **执行模型**: 作为 Elementwise 内核的算子模板参数，每个 MLU 核心处理一组元素，无跨步长协作

### `SwiGLUOp` (主机端启动器)
- **位置**: `swiglu_bang.mlu`
- **主要功能**: 主机端静态包装器，提供类型擦除的启动接口，连接上层 Descriptor 和下层设备内核
- **关键成员**:
  - `num_inputs`: static constexpr size_t = 2 - 指明内核需要 2 个输入张量
- **核心方法**:
  - `launch<Tdata>(Args... args)`: 静态模板方法，转发参数到 `launchSwiGLUKernel<Tdata>(args...)`，该函数在 `swiglu_bang_internal.mlu` 中通过 LAUNCH_ELEMENTWISE_KERNEL_IMPL 宏生成，返回 INFINI_STATUS_SUCCESS

## 3. API 接口

```cpp
// 创建 SwiGLU 算子描述符
infiniStatus_t op::swiglu::bang::Descriptor::create(
    infiniopHandle_t handle_,                   // [输入] BANG 设备句柄
    Descriptor **desc_ptr,                      // [输出] 返回的描述符指针
    infiniopTensorDescriptor_t out_desc,        // [输入] 输出张量描述符
    std::vector<infiniopTensorDescriptor_t> input_desc_vec  // [输入] {up_desc, gate_desc}
);
// 返回 INFINI_STATUS_SUCCESS 或错误码（如类型不匹配、形状不一致）

// 执行 SwiGLU 计算
infiniStatus_t op::swiglu::bang::Descriptor::calculate(
    void *workspace,                            // [输入] 设备端工作空间指针（元数据传输用）
    size_t workspace_size,                      // [输入] 工作空间大小（字节）
    void *output,                               // [输入/输出] 输出张量的设备指针
    std::vector<const void *> inputs,           // [输入] {up_device_ptr, gate_device_ptr}
    void *queue                                 // [输入] cnrtQueue_t 队列，用于异步执行
) const;
// 返回 INFINI_STATUS_SUCCESS, INFINI_STATUS_INSUFFICIENT_WORKSPACE, 或 INFINI_STATUS_BAD_TENSOR_DTYPE
```

## 4. 使用示例

```cpp
// 示例：在 MLU 设备上执行 SwiGLU 激活函数
#include "infiniop/ops/swiglu/bang/swiglu_bang.h"

// 1. 初始化 BANG 设备句柄
infiniopHandle_t handle;
cnrtQueue_t queue;
// ... (假设已创建 handle 和 queue)

// 2. 准备张量描述符（假设形状为 {batch_size, seq_len, hidden_dim}）
int64_t shape[] = {32, 1024, 4096};
infinopTensorDescriptor_t up_desc, gate_desc, out_desc;
infiniopCreateTensorDescriptor(&up_desc, INFINI_DTYPE_F16, 3, shape);
infiniopCreateTensorDescriptor(&gate_desc, INFINI_DTYPE_F16, 3, shape);
infiniopCreateTensorDescriptor(&out_desc, INFINI_DTYPE_F16, 3, shape);

// 3. 创建 SwiGLU 算子描述符
op::swiglu::bang::Descriptor *swiglu_desc;
std::vector<infiniopTensorDescriptor_t> inputs = {up_desc, gate_desc};
auto status = op::swiglu::bang::Descriptor::create(handle, &swiglu_desc, out_desc, inputs);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理创建失败（形状不匹配或类型不支持）
}

// 4. 分配设备内存并获取工作空间大小
half *d_up, *d_gate, *d_out;
size_t tensor_size = 32 * 1024 * 4096 * sizeof(half);
cnrtMalloc(&d_up, tensor_size);
cnrtMalloc(&d_gate, tensor_size);
cnrtMalloc(&d_out, tensor_size);

size_t workspace_size = swiglu_desc->getWorkspaceSize();
void *d_workspace;
cnrtMalloc(&d_workspace, workspace_size);

// 5. 填充输入数据（从主机到设备）
cnrtMemcpy(d_up, h_up_data, tensor_size, CNRT_MEM_TRANS_DIR_HOST2DEV);
cnrtMemcpy(d_gate, h_gate_data, tensor_size, CNRT_MEM_TRANS_DIR_HOST2DEV);

// 6. 执行计算
std::vector<const void *> input_ptrs = {d_up, d_gate};
status = swiglu_desc->calculate(d_workspace, workspace_size, d_out, input_ptrs, queue);

// 7. 同步队列并取回结果
cnrtQueueSync(queue);
cnrtMemcpy(h_out_data, d_out, tensor_size, CNRT_MEM_TRANS_DIR_DEV2HOST);

// 8. 清理资源
delete swiglu_desc;
cnrtFree(d_up); cnrtFree(d_gate); cnrtFree(d_out); cnrtFree(d_workspace);
```

## 5. 实现细节

### 数学公式与优化策略
- **公式**: SwiGLU = up * sigmoid(gate)，其中 sigmoid(x) = 1 / (1 + e^(-x))
- **FP16/BF16 路径**: 直接调用 `__bang_active_sigmoid` 利用 MLU 硬件加速单元计算 sigmoid，通过两次乘法完成，避免中间溢出（先 gate*sigmoid 结果，再乘以 up）
- **FP32 路径**: 使用数值稳定实现 `1 / (1 + exp(-x))`，通过 `__bang_neg` + `__bang_active_exphp` (指数函数) + `__bang_add_scalar` + `__bang_div` 组合，避免大负数时 exp 溢出
- **回退路径**: 标准 C++ 实现，保证代码可编译性但无硬件加速

### 内存管理与数据流
- **工作空间布局**: 工作空间前部存储输入指针数组（N 个指针），后部存储元数据（通过 ElementwiseInfo::getMetaStart() 获取），包括输出形状、步长、输入形状、输入步长、连续性标志、广播标志
- **数据传输方向**: 主机到设备（H2D）通过 cnrtMemcpy 将输入指针和元数据复制到工作空间，内核执行完成后调用 cnrtQueueSync 同步，确保计算完成
- **内存分配策略**: 用户负责分配输入/输出张量的设备内存和工作空间，算子不进行任何内存分配，仅使用用户提供的工作空间进行元数据传输

### 并发执行与同步
- **任务级并行**: 每个 MLU 核心独立处理一组元素，通过 Elementwise 框架的内核启动逻辑分配任务，核心间无通信
- **同步机制**: calculate() 方法在内核启动后立即调用 cnrtQueueSync(queue) 阻塞等待完成，确保执行完成后返回，实现同步 API 语义
- **队列管理**: 用户提供的 cnrtQueue_t 队列用于内核提交，支持将多个算子提交到同一队列实现流水线执行

### 性能优化技术
- **硬件加速指令**: 针对 half/bfloat16_t 使用 `__bang_active_sigmoid` 直接调用硬件 sigmoid 单元，避免逐元素计算；float 类型使用 `__bang_active_exphp` 硬件指数函数
- **内存访问模式**: 连续张量访问模式，MLU 的 DMA 引擎可高效预取，非连续张量通过步长数组支持泛型广播
- **模板实例化**: 仅实例化 half/bfloat16_t/float 三种常用类型，避免代码膨胀，通过 LAUNCH_ELEMENTWISE_KERNEL_INSTANTIATE 宏生成
- **编译期分支**: 使用 if constexpr 在编译期选择数据类型路径，避免运行时分支开销

### 错误处理与验证
- **类型检查**: create() 方法通过 CHECK_DTYPE 宏验证数据类型为 F16/F32/BF16，否则返回错误
- **形状验证**: CHECK_SAME_SHAPE 宏确保输出、up、gate 三者形状完全一致，SwiGLU 不支持广播
- **工作空间验证**: calculate() 方法检查 workspace_size >= _workspace_size，不足时返回 INFINI_STATUS_INSUFFICIENT_WORKSPACE
- **错误传播**: 所有 CNRT 调用（cnrtMemcpy, cnrtQueueSync）通过 CNRT_CHECK 宏包装，失败时返回对应错误码

### 设计模式
- **策略模式 (Strategy Pattern)**: 通过模板参数 Op 抽象算子逻辑，SwiGLUOp 作为具体策略实现，Elementwise 框架在编译期静态分发
- **工厂模式 (Factory Pattern)**: Descriptor::create() 静态工厂方法封装对象创建逻辑，验证参数并构造实例
- **模板方法模式 (Template Method Pattern)**: DeviceImpl::calculateImpl 定义 Elementwise 算法的骨架（元数据传输 → 内核启动 → 同步），SwiGLUOp::launch 提供可变部分的实现
- **RAII (Resource Acquisition Is Initialization)**: DeviceImpl 使用 std::shared_ptr<Opaque> 管理设备资源，析构时自动释放

### 依赖关系
- **外部依赖**: 寒武纪 BANG SDK（cnrt.h, bang.h, bang_device_functions.h），提供设备运行时 API 和硬件加速函数
- **内部依赖**: elementwise/bang/elementwise_bang.h（通用 Elementwise BANG 框架），elementwise/bang/elementwise_bang_kernel.h（内核启动宏和基础设施），device/bang/common_bang.h（BANG 设备通用定义），utils.h（工具类和错误码定义）
- **宏系统**: 依赖 LAUNCH_ELEMENTWISE_KERNEL_IMPL（生成内核包装函数）、LAUNCH_ELEMENTWISE_KERNEL_INSTANTIATE（生成类型实例化）、ELEMENTWISE_DESCRIPTOR（生成 Descriptor 类声明）等宏简化重复代码
