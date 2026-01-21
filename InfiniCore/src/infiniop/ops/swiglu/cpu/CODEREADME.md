# SwiGLU CPU 算子核心实现文档

SwiGLU (Swish-Gated Linear Unit) 激活函数的 CPU 后端实现。该模块基于 InfiniOP 的 elementwise 通用框架，通过模板化算子和 OpenMP 并行化，实现了高效的 CPU 端 SwiGLU 激活计算，支持 FP16、BF16、FP32 和 FP64 四种精度。

## 1. 模块结构

- **`swiglu_cpu.h`**: 定义 SwiGLU 核心算子类和接口宏展开，包含 sigmoid 激活函数和 SwiGLU 前向计算逻辑
- **`swiglu_cpu.cc`**: 实现 Descriptor 的创建和计算调度，处理 dtype 分发和参数校验

## 2. 核心类

### `SwiGLUOp`
- **位置**: `swiglu_cpu.h`
- **主要功能**: 实现 SwiGLU 激活函数的逐元素计算算子
- **关键成员**:
  - `num_inputs`: 静态常量，固定为 2，表示需要两个输入张量（up 和 gate）
- **核心方法**:
  - `sigmoid<T>(const T &x)`: 私有辅助方法，计算 sigmoid 激活函数 σ(x) = 1 / (1 + e^(-x))，使用标准库 `std::exp` 实现指数运算
  - `operator()<T>(const T &up, const T &gate)`: 公共仿函数接口，实现 SwiGLU 前向计算：`output = gate × σ(gate) × up`。该函数会先对 gate 应用 sigmoid，再与 up 相乘，最终时间复杂度 O(1)
- **生命周期**: 无状态仿函数（Stateless Functor），支持栈上构造和值传递，无需动态内存分配

### `op::swiglu::cpu::Descriptor`
- **位置**: 通过 `ELEMENTWISE_DESCRIPTOR(swiglu, cpu)` 宏展开生成在 `swiglu_cpu.h`
- **主要功能**: SwiGLU CPU 算子的描述符类，继承自 `InfiniopDescriptor`，管理算子元数据和执行资源
- **关键成员**:
  - `_dtype`: `infiniDtype_t` 类型，存储输出张量的数据类型（FP16/BF16/F32/F64）
  - `_info`: `op::elementwise::ElementwiseInfo` 类型，封装输入/输出张量的形状、步幅、连续性等元数据
  - `_device_info`: `std::unique_ptr<op::elementwise::cpu::DeviceImpl>` 类型，CPU 设备特定实现（当前为空实现）
  - `_workspace_size`: `size_t` 类型，工作空间大小（当前固定为 0）
- **核心方法**:
  - `create(...)`: 静态工厂方法，执行以下步骤：
    1. 类型转换：将 `infiniopHandle_t` 转换为 `device::cpu::Handle*`
    2. 参数校验：检查输出 dtype 必须为 FP16/BF16/F32/F64 之一，使用 `CHECK_DTYPE` 宏；验证输入和输出形状完全一致，使用 `CHECK_SAME_SHAPE` 宏
    3. 元数据提取：从 `input_desc_vec.at(0)` 获取 up 张量描述符，从 `input_desc_vec.at(1)` 获取 gate 张量描述符
    4. 描述符构建：调用 `CREATE_ELEMENTWISE_CPU_DESCRIPTOR` 宏创建 `ElementwiseInfo` 并实例化 Descriptor 对象
    5. 返回 `INFINI_STATUS_SUCCESS` 或错误码
  - `calculate(...)`: 常量方法，执行 SwiGLU 计算：
    1. Dtype 分发：根据 `_dtype` 成员变量，在四种支持的浮点类型中 switch
    2. 模板实例化：调用 `_device_info->calculate<SwiGLUOp, T>(_info, output, inputs, stream)`，其中 T 为对应的 C++ 类型（`fp16_t`, `bf16_t`, `float`, `double`）
    3. 下层调度：`DeviceImpl::calculate` 使用 OpenMP 并行遍历输出张量，每个元素调用 `SwiGLUOp::operator()`
    4. 特殊处理：对于 FP16/BF16，先转换为 float 计算，再转回原类型（通过 `utils::cast`）
    5. 返回状态码或 `INFINI_STATUS_BAD_TENSOR_DTYPE`（不支持的 dtype）
- **生命周期**: 由 `create` 方法在堆上分配（`new Descriptor`），用户负责调用析构函数释放（析构函数默认实现）

## 3. API 接口

```cpp
// 创建 SwiGLU CPU 算子描述符
infiniStatus_t op::swiglu::cpu::Descriptor::create(
    infiniopHandle_t handle,                    // CPU 设备句柄
    Descriptor **desc_ptr,                      // 输出：构造的描述符指针
    infiniopTensorDescriptor_t out_desc,        // 输出张量描述符
    std::vector<infiniopTensorDescriptor_t> input_desc_vec  // 输入张量描述符向量
);
// 返回 INFINI_STATUS_SUCCESS 或错误码（如 dtype 不支持、形状不匹配）

// 执行 SwiGLU 计算
infiniStatus_t Descriptor::calculate(
    void *workspace,            // 工作空间指针（当前未使用，传 nullptr）
    size_t workspace_size,      // 工作空间大小（当前未使用，传 0）
    void *output,               // 输出张量数据指针
    std::vector<const void *> inputs,  // 输入数据指针向量 [0]=up, [1]=gate
    void *stream                // 执行流（CPU 后端未使用，传 nullptr）
) const;
// 返回 INFINI_STATUS_SUCCESS 或 INFINI_STATUS_BAD_TENSOR_DTYPE
```

## 4. 使用示例

```cpp
// 示例：使用 SwiGLU CPU 算子进行前向计算
#include "swiglu_cpu.h"

// 1. 准备张量描述符和数据
const std::vector<size_t> shape = {1024, 512};
auto up_desc = createTensorDescriptor(shape, INFINI_DTYPE_F32);      // up 张量
auto gate_desc = createTensorDescriptor(shape, INFINI_DTYPE_F32);    // gate 张量
auto out_desc = createTensorDescriptor(shape, INFINI_DTYPE_F32);     // 输出张量

// 2. 分配内存
float *up_data = new float[1024 * 512];
float *gate_data = new float[1024 * 512];
float *out_data = new float[1024 * 512];

// 填充输入数据（示例）
for (size_t i = 0; i < 1024 * 512; ++i) {
    up_data[i] = /* up 值 */;
    gate_data[i] = /* gate 值 */;
}

// 3. 创建算子描述符
op::swiglu::cpu::Descriptor *swiglu_desc = nullptr;
infiniopHandle_t cpu_handle = /* 获取 CPU 句柄 */;
std::vector<infiniopTensorDescriptor_t> inputs = {up_desc, gate_desc};
auto status = op::swiglu::cpu::Descriptor::create(
    cpu_handle, &swiglu_desc, out_desc, inputs);

if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 4. 执行计算
std::vector<const void *> input_ptrs = {up_data, gate_data};
status = swiglu_desc->calculate(
    nullptr,      // workspace（未使用）
    0,            // workspace_size
    out_data,     // 输出
    input_ptrs,   // 输入
    nullptr       // stream（CPU 后端未使用）
);

// 5. 清理资源
delete swiglu_desc;
delete[] up_data;
delete[] gate_data;
delete[] out_data;
```

## 5. 实现细节

### 内存管理
- **零拷贝策略**: 所有计算直接在用户提供的输入/输出缓冲区上进行，不分配中间内存
- **栈上算子**: `SwiGLUOp` 仿函数无状态，编译器优化后完全内联，无运行时开销
- **元数据封装**: `ElementwiseInfo` 使用单个 `std::vector<size_t>` 紧凑存储所有张量元数据（形状、步幅、连续性标志），内存布局为连续数组，减少缓存未命中

### 并发性
- **OpenMP 并行**: `DeviceImpl::calculate` 使用 `#pragma omp parallel for` 并行化元素循环（当输出元素数 > 1024 时启用，避免小张量的线程创建开销）
- **无共享状态**: 每个线程独立计算不重叠的输出元素，无需锁或同步原语
- **线程安全**: 算子函数 `operator()` 是纯函数（无副作用），多线程并发调用安全
- **静态调度**: OpenMP 默认使用 static 调度，工作负载均匀分布

### 性能
- **算法复杂度**: O(N)，其中 N 为输出张量元素总数，每个元素执行常数次运算（1 次指数、3 次乘法、1 次加法）
- **向量化潜力**: 循环体为简单浮点运算，编译器可自动生成 SIMD 指令（AVX2/AVX-512）
- **低精度加速**: FP16/BF16 通过先转为 float 再计算（精度提升），避免直接半精度运算的数值不稳定
- **连续内存优化**: 对连续张量使用平坦索引 `i`，对非连续张量调用 `indexToOffset` 计算偏移量（时间复杂度 O(ndim)，但 ndim 通常较小）

### 错误处理
- **Dtype 校验**: `CHECK_DTYPE` 宏在运行时验证输出 dtype，不支持的类型返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
- **形状校验**: `CHECK_SAME_SHAPE` 宏确保 up、gate、output 三个张量形状完全一致，否则返回错误码
- **错误传播**: `ElementwiseInfo::create` 和 `Descriptor::create` 使用 `Result<T>` 模式封装返回值，支持链式错误检查

### 依赖
- **内部依赖**:
  - `op::elementwise::ElementwiseInfo`: 元数据管理（定义在 `elementwise/elementwise.h`）
  - `op::elementwise::cpu::DeviceImpl`: CPU 设备执行器（定义在 `elementwise/cpu/elementwise_cpu.h`）
  - `op::common_cpu::indexToOffset`: 张量索引计算工具（定义在 `devices/cpu/common_cpu.h`）
  - `utils::cast`: 类型转换工具（FP16/BF16 ↔ float）
- **外部依赖**:
  - OpenMP（可选，`#ifdef ENABLE_OMP`）
  - C++ 标准库：`std::exp`（数学函数）、`std::vector`（容器）

### 设计模式
- **CRTP 模式**: `ELEMENTWISE_DESCRIPTOR` 宏使用命名空间注入生成派生类，避免代码重复
- **策略模式**: `DeviceImpl` 封装设备特定行为，支持未来扩展（如 GPU 后端）
- **工厂模式**: `Descriptor::create` 静态方法作为构造器，集中参数校验和对象构建
- **仿函数模式**: `SwiGLUOp` 重载 `operator()`，实现类似函数对象的接口，支持模板泛型
- **RAII**: `ElementwiseInfo` 使用 `std::vector` 自动管理内存，析构时释放
