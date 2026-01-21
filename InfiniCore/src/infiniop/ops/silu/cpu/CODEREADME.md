# SiLU CPU 实现核心文档

本模块实现 Swish 激活函数（又称 SiLU：Sigmoid Linear Unit）的 CPU 后端，基于 InfiniOP 的逐元素运算框架。该模块通过模板元编程和 OpenMP 并行化，为 BF16、FP16、FP32 和 FP64 数据类型提供高性能的逐元素激活函数计算。

## 1. 模块结构

- **`silu_cpu.h`**: 定义 SiLU 运算的核心算子类和描述符宏，实现 SiLU 数学公式 `x / (1 + e^(-x))`
- **`silu_cpu.cc`**: 实现 SiLU 描述符的创建和计算调度逻辑，处理多数据类型分发

## 2. 核心类

### `op::silu::cpu::SiluOp`
- **位置**: `silu_cpu.h`
- **主要功能**: 实现 SiLU 激活函数的逐元素变换算子
- **关键成员**:
  - `num_inputs`: 编译时常量，固定为 1，表示单输入运算
- **核心方法**:
  - `operator()(const T &x) const`: SiLU 数学实现，计算公式 `x / (1 + exp(-x))`
    - **算法**: 直接计算 sigmoid 函数 `σ(x) = 1 / (1 + e^(-x))`，然后返回 `x * σ(x)`
    - **数值稳定性**: 使用 `std::exp(-x)` 计算，对负值输入数值稳定
    - **复杂度**: O(1) 每元素
    - **模板参数**: `T` 支持浮点类型（float, double, bf16_t, fp16_t）
- **生命周期**: 无状态仿函数（Stateless Functor），可被值传递和复制

### `op::silu::cpu::Descriptor`
- **位置**: 通过 `ELEMENTWISE_DESCRIPTOR` 宏在编译时生成
- **主要功能**: SiLU 操作的 CPU 描述符，管理运算元数据和设备信息
- **关键成员**:
  - `_dtype`: `infiniDtype_t`，存储输出张量的数据类型
  - `_info`: `op::elementwise::ElementwiseInfo`，封装张量的形状、步长、连续性等元数据
  - `_device_info`: `std::unique_ptr<op::elementwise::cpu::DeviceImpl>`，CPU 设备特定实现
  - `_workspace_size`: `size_t`，工作空间大小（当前实现为 0）
- **核心方法**:
  - `create(...)`: 静态工厂方法，构造并验证 SiLU 描述符
    - **参数验证**:
      - 检查数据类型是否为 BF16/F16/F32/F64（通过 `CHECK_DTYPE` 宏）
      - 验证输入和输出形状完全一致（通过 `CHECK_SAME_SHAPE` 宏）
    - **元数据创建**: 调用 `ElementwiseInfo::create()` 提取张量布局信息
    - **描述符实例化**: 使用 `CREATE_ELEMENTWISE_CPU_DESCRIPTOR` 宏分配内存并构造对象
    - **复杂度**: O(n) 其中 n 为张量维度数
  - `calculate(...)`: 执行 SiLU 计算
    - **类型分发**: 根据 `_dtype` 分发到对应的模板实例化（BF16/F16/F32/F64）
    - **计算委托**: 调用 `_device_info->calculate<SiluOp, T>()` 执行实际计算
    - **并行策略**: 底层使用 OpenMP `#pragma omp parallel for` 并行化（当元素数 > 1024 时启用）
    - **复杂度**: O(N) 其中 N 为输出张量元素总数
- **继承层次**: 继承自 `InfiniopDescriptor`，遵循 InfiniOP 描述符接口规范

## 3. API 接口

```cpp
// SiLU 描述符创建 API
namespace op::silu::cpu {
    infiniStatus_t Descriptor::create(
        infiniopHandle_t handle_,              // [in] CPU 设备句柄
        Descriptor **desc_ptr,                 // [out] 输出描述符指针
        infiniopTensorDescriptor_t out_desc,   // [in] 输出张量描述符
        std::vector<infiniopTensorDescriptor_t> input_desc_vec  // [in] 输入张量描述符向量（单元素）
    );
    // 返回 INFINI_STATUS_SUCCESS 成功，或错误码：
    // - INFINI_STATUS_BAD_TENSOR_DTYPE: 数据类型不支持
    // - INFINI_STATUS_BAD_TENSOR_STRIDES: 形状不匹配
}

// SiLU 计算执行 API
infiniStatus_t Descriptor::calculate(
    void *workspace,                           // [in] 工作空间指针（当前未使用，可传 nullptr）
    size_t workspace_size,                     // [in] 工作空间大小（当前为 0）
    void *output,                              // [out] 输出张量数据指针
    std::vector<const void *> inputs,          // [in] 输入张量数据指针向量（单元素）
    void *stream                               // [in] 执行流（CPU 实现中未使用，可传 nullptr）
) const;
// 返回 INFINI_STATUS_SUCCESS 成功，或 INFINI_STATUS_BAD_TENSOR_DTYPE 类型错误
```

### 核心 SiluOp 算子接口

```cpp
template <typename T>
struct SiluOp {
    static constexpr size_t num_inputs = 1;  // 编译期输入数量

    // SiLU 激活函数：f(x) = x * sigmoid(x) = x / (1 + e^(-x))
    template <typename T>
    T operator()(const T &x) const {
        return x / (static_cast<T>(1) + std::exp(-x));
    }
};
```

## 4. 使用示例

```cpp
// 示例：在 CPU 上执行 SiLU 激活函数计算

#include "silu_cpu.h"
#include "../devices/cpu/cpu_handle.h"

using namespace op::silu::cpu;

// 1. 准备输入输出张量描述符
constexpr size_t ndim = 2;
size_t shape[2] = {1024, 1024};           // 1M 元素的张量
ptrdiff_t strides[2] = {1024, 1};         // 行主序布局

infiniopTensorDescriptor_t input_desc;
infiniopTensorDescriptor_t output_desc;

// 创建描述符（假设已有 CPU handle）
infiniopCreateTensorDescriptor(handle, &input_desc,
    INFINI_DTYPE_F32, ndim, shape, strides);
infiniopCreateTensorDescriptor(handle, &output_desc,
    INFINI_DTYPE_F32, ndim, shape, strides);

// 2. 创建 SiLU 描述符
Descriptor* silu_desc = nullptr;
std::vector<infiniopTensorDescriptor_t> inputs = {input_desc};
auto status = Descriptor::create(handle, &silu_desc, output_desc, inputs);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误：类型不支持或形状不匹配
}

// 3. 分配内存并初始化输入数据
float* d_input = new float[1024 * 1024];
float* d_output = new float[1024 * 1024];
// ... 填充 d_input 数据 ...

// 4. 执行 SiLU 计算
status = silu_desc->calculate(
    nullptr,              // 无需工作空间
    0,                    // 工作空间大小为 0
    d_output,             // 输出缓冲区
    {d_input},            // 输入数据指针数组
    nullptr               // CPU 实现无需流对象
);

// 5. 清理资源
delete silu_desc;
delete[] d_input;
delete[] d_output;
infiniopDestroyTensorDescriptor(input_desc);
infiniopDestroyTensorDescriptor(output_desc);

// 结果：d_output[i] = d_input[i] / (1 + exp(-d_input[i]))
```

## 5. 实现细节

### 内存管理
- **描述符分配**: 使用 `new Descriptor(...)` 在堆上分配，由调用者负责 `delete`
- **元数据存储**: `ElementwiseInfo` 使用单个 `std::vector<size_t>` 紧凑存储所有张量的形状、步长、连续性和广播标志
  - 元数据大小计算公式：`ndim * (sizeof(size_t) + sizeof(ptrdiff_t)) + input_size * ndim * (sizeof(size_t) + sizeof(ptrdiff_t)) + 2 * input_size * sizeof(bool)`
  - 对于单输入 SiLU，input_size = 1
- **张量数据**: 调用者负责分配和释放输入/输出缓冲区，描述符仅持有指针不管理生命周期

### 并发控制
- **并行策略**: 使用 OpenMP 并行 for 循环（`#pragma omp parallel for`）
  - 自动线程数调度（由 OMP_NUM_THREADS 或硬件并发度控制）
  - 条件并行：仅在输出元素数 > 1024 时启用（避免小数组并行开销）
- **同步机制**: 无显式锁，依赖 OpenMP 的隐式屏障同步
- **线程安全**: `calculate()` 方法是 const 的，只读访问描述符状态，多线程并发安全
- **数据竞争保护**: 每个线程处理独立的输出元素索引范围，无共享写操作

### 性能优化
- **算法选择**: 直接逐元素计算，O(N) 复杂度（N 为元素总数）
- **缓存友好**: 连续内存访问模式（行主序遍历）
  - 当张量连续时，使用线性索引 `i`
  - 非连续张量通过 `indexToOffset()` 计算物理偏移（可能降低性能）
- **向量化**: 编译器可自动对 FP32/FP64 进行 SIMD 优化（需 `-O3 -mavx2` 等）
- **半精度处理**: FP16/BF16 先提升至 FP32 计算，再转回原类型（`utils::cast<float>`）
  - 避免精度损失和下溢问题
- **分支消除**: 使用编译期 `if constexpr` 在半精度和全精度路径间编译时分派
- **内存开销**: 零运行时工作空间（workspace_size = 0），所有状态在栈上

### 错误处理
- **错误传播**: 使用 `infiniStatus_t` 枚举返回状态码
  - `INFINI_STATUS_SUCCESS`: 操作成功
  - `INFINI_STATUS_BAD_TENSOR_DTYPE`: 不支持的数据类型（非浮点类型）
  - `INFINI_STATUS_BAD_TENSOR_STRIDES`: 输入输出形状不匹配或广播设置错误
- **参数验证**:
  - `CHECK_DTYPE` 宏：编译期生成类型检查代码（仅允许 BF16/F16/F32/F64）
  - `CHECK_SAME_SHAPE` 宏：运行时验证输入输出张量的形状完全一致
  - `ElementwiseInfo::create()` 检查输出张量不能有广播维度
- **异常安全**: 不使用 C++ 异常，错误通过返回码传播
- **资源清理**: 描述符析构函数默认生成（`= default`），自动释放成员资源

### 依赖关系
- **外部依赖**:
  - `std::exp`: C++ 标准库数学函数，计算指数
  - OpenMP (可选): `#pragma omp parallel for` 需要 `-fopenmp` 编译选项
- **内部模块依赖**:
  - `op::elementwise::cpu::DeviceImpl`: 提供通用逐元素计算基础设施
  - `op::elementwise::ElementwiseInfo`: 张量布局元数据管理
  - `op::common_cpu::indexToOffset()`: 扁平索引到多维偏移的转换
  - `utils::cast<T>()`: 类型安全转换工具（处理半精度提升）
- **继承层次**:
  - `Descriptor` → `InfiniopDescriptor` (基类)
  - `DeviceImpl` (组合，非继承)

### 设计模式
- **模板方法模式**: `Descriptor::calculate()` 定义类型分发框架，`SiluOp::operator()` 实现具体算法
- **策略模式**: 数据类型作为编译期策略，生成不同的特化代码
- **仿函数模式**: `SiluOp` 实现 `operator()`，可被高阶算法调用
- **宏生成模式**: `ELEMENTWISE_DESCRIPTOR` 宏减少重复代码，自动生成描述符类
- **RAII (Resource Acquisition Is Initialization)**: 描述符成员使用智能指针和值类型管理生命周期
- **编译期多态**: 通过模板特化实现零开销抽象，避免虚函数调用

### 关键特性
- **数学精确性**: SiLU 公式 `f(x) = x / (1 + e^(-x))` 与 `x * sigmoid(x)` 等价，直接实现避免中间乘法
- **数值范围**: 对所有浮点数值稳定（包括大正数和大负数）
  - 当 `x >> 0`，`e^(-x) → 0`，结果趋向 `x`
  - 当 `x << 0`，`e^(-x) → ∞`，结果趋向 `0`
- **自动微分友好**: 逐元素变换，链式法则简单，易于集成至反向传播
- **零拷贝**: 不修改输入张量，仅写入输出缓冲区
- **广播限制**: 当前实现要求输入输出形状完全相同（不支持自动广播）
