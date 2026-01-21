# Elementwise CPU Backend Core Implementation Documentation

这是一个为逐元素运算（elementwise operations）提供 CPU 后端实现的模板化计算引擎，专门设计用于高效处理张量的逐元素操作，支持广播、非连续内存布局和 OpenMP 并行化。

## 1. 模块结构

- **`elementwise_cpu.h`**: 定义 CPU 专用实现类 `DeviceImpl` 和核心计算模板函数，提供统一类型与异构类型两种计算模式

## 2. 核心类

### `DeviceImpl`
- **位置**: `elementwise_cpu.h`
- **主要功能**: 封装 CPU 设备特定的逐元素运算执行逻辑，作为具体操作（如 Add, Mul, Relu 等）的后端计算引擎
- **关键成员**:
  - `struct Opaque`: 前向声明的不透明类型，用于 Pimpl 模式（指针实现），隐藏实现细节
  - `std::shared_ptr<Opaque> _opaque`: 指向不透明实现对象的共享指针，采用共享所有权语义
- **核心方法**:
  - `create(Args &&...args)`: 静态工厂方法，当前实现返回 `INFINI_STATUS_NOT_IMPLEMENTED`，保留用于未来扩展
  - `calculate<Op, Tdata, Args...>(info, output, inputs, stream, args...)`: **同构类型计算入口**，当所有输入和输出类型相同时调用，时间复杂度 O(n)，n 为输出张量元素总数
  - `calculate<Op, Tout, Tin..., Args...>(info, output, inputs, stream, args...)`: **异构类型计算入口**，支持每个输入有不同的数据类型，类型数量必须在编译期与 `Op::num_inputs` 匹配，使用 SFINAE 进行编译期类型检查
- **生命周期**: 采用 Pimpl 惯用模式，使用 `std::shared_ptr` 管理 `Opaque` 对象，支持拷贝和移动语义（默认析构函数）

### `ElementwiseInfo` (依赖)
- **位置**: `../elementwise.h`
- **主要功能**: 存储逐元素操作的元数据，包括形状、步幅、连续性标志和广播信息
- **关键数据结构**:
  - `std::vector<size_t> _meta`: 单一连续向量存储所有元数据，内部通过指针偏移访问不同区域（输出形状、输出步幅、输入形状数组、输入步幅数组、输入连续性标志数组、输入广播标志数组）
  - `size_t _output_size`: 输出张量的元素总数
  - `size_t _input_size`: 输入张量的数量
  - `size_t _ndim`: 张量的维度数
  - `bool _output_contiguous`: 输出张量是否内存连续
- **内存布局**: 使用扁平化 `std::vector<size_t>` 存储，通过 `reinterpret_cast` 指针运算访问不同类型的子区域，减少内存分配次数并提高缓存局部性

### `Op` 操作概念 (模板参数)
- **约束**: 必须定义静态 constexpr 成员 `num_inputs` 表示输入数量
- **接口**: 必须实现可调用对象（functor）或 `operator()` 模板方法，接受输入值并返回计算结果
- **示例**: `AddOp`, `MulOp`, `ReluOp` 等具体操作通过函数对象实现

## 3. API 接口

```cpp
namespace op::elementwise::cpu {

// 同构类型计算：所有输入和输出类型相同
template <typename Op, typename Tdata, typename... Args>
infiniStatus_t calculate(
    const op::elementwise::ElementwiseInfo &info,
    void *output,
    const std::vector<const void *> &inputs,
    void *stream,
    Args &&...args);

// 异构类型计算：每个输入可有不同类型
template <typename Op, typename Tout, typename... Tin, typename... Args,
          std::enable_if_t<(sizeof...(Tin) == Op::num_inputs), int> = 0>
infiniStatus_t calculate(
    const op::elementwise::ElementwiseInfo &info,
    void *output,
    const std::vector<const void *> &inputs,
    void *stream,
    Args &&...args);

// 宏：用于快速创建 CPU 元素描述符
#define CREATE_ELEMENTWISE_CPU_DESCRIPTOR(HANDLE, DTYPE, OUT_DESC, INPUT_DESC_VEC)
}
```

## 4. 使用示例

```cpp
// 示例：在 Add 操作的 CPU 实现中使用 elementwise_cpu.h

#include "elementwise/cpu/elementwise_cpu.h"

namespace op::add::cpu {

// 定义操作符：两个浮点数相加
struct AddOp {
    static constexpr size_t num_inputs = 2;

    template <typename T, typename U1, typename U2>
    T operator()(U1 a, U2 b) const {
        return static_cast<T>(a + b);
    }
};

// 描述符创建
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    // 使用宏创建 CPU 描述符，自动初始化 ElementwiseInfo
    CREATE_ELEMENTWISE_CPU_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec);

    return INFINI_STATUS_SUCCESS;
}

// 计算执行
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_F32:
        // 同构类型：所有输入和输出都是 float
        return _device_info->calculate<AddOp, float>(_info, output, inputs, stream);
    case INFINI_DTYPE_F16:
        // fp16_t 会在内部提升到 float 计算，然后再转回 fp16_t
        return _device_info->calculate<AddOp, fp16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_BF16:
        // bf16_t 会在内部提升到 float 计算，然后再转回 bf16_t
        return _device_info->calculate<AddOp, bf16_t>(_info, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}
}
```

**复杂操作示例（异构类型）**:
```cpp
// 示例：混合精度操作，输入为 float16 和 float32，输出为 float32
struct MixedPrecisionOp {
    static constexpr size_t num_inputs = 2;

    template <typename Tout, typename Tin1, typename Tin2>
    Tout operator()(Tin1 a, Tin2 b) const {
        // 自动类型提升和转换
        return static_cast<Tout>(static_cast<float>(a) + static_cast<float>(b));
    }
};

// 调用：输入1是 fp16_t，输入2是 float，输出是 float
infiniStatus_t status = _device_info->calculate<MixedPrecisionOp, float, fp16_t, float>(
    _info, output, inputs, stream);
```

## 5. 实现细节

### 内存管理
- **元数据存储**: 使用单一 `std::vector<size_t>` 扁平化存储所有元数据，通过 `reinterpret_cast` 指针算术访问不同类型的子区域，减少多次内存分配开销
- **共享所有权**: `DeviceImpl` 使用 `std::shared_ptr<Opaque>` 管理内部状态，支持拷贝语义和多处共享描述符
- **空实现优化**: `DeviceImpl::Opaque` 定义为空结构体，CPU 实现无需维护设备特定状态（相比 CUDA 需要管理 kernels 和 streams）

### 并发策略
- **OpenMP 并行化**: 使用 `#pragma omp parallel for` 对元素循环进行并行化
- **条件并行**: 同构类型版本在 `output_size > 1024` 时才启用 OpenMP，避免小张量的并行开销（`#pragma omp parallel for if (output_size > 1024)`）
- **异构类型版本**: 始终使用 OpenMP 并行（`#pragma omp parallel for`），无阈值检查
- **线程安全**: 每个迭代处理独立的输出元素，无数据竞争，只读输入和独立输出写入

### 性能优化
- **连续内存快速路径**: 检查 `info.isOutputContiguous()` 和 `info.getInputContiguous()[input_id]`，连续张量使用线性索引 `i`，避免 `indexToOffset` 计算
- **非连续索引计算**: 使用 `op::common_cpu::indexToOffset` 将扁平索引转换为多维张量的实际内存偏移，支持任意步幅布局
- **半精度浮点提升**: 对于 `fp16_t` 和 `bf16_t`，在计算前提升到 `float` 类型进行运算（`utils::cast<float>()`），计算后再转回原类型，避免精度损失和硬件支持问题
- **编译期优化**: 使用 `constexpr` 和模板元编程，编译期展开类型分发和循环，避免运行期分支开销
- **内联优化**: `ElementwiseInfo` 的所有访问方法都标记为 `inline`，消除函数调用开销

### 错误处理
- **Result 类型**: `ElementwiseInfo::create()` 返回 `utils::Result<ElementwiseInfo>`，使用值语义封装错误状态，避免异常开销
- **状态码**: 使用 `infiniStatus_t` 枚举报告错误（`INFINI_STATUS_SUCCESS`, `INFINI_STATUS_BAD_PARAM`, `INFINI_STATUS_NOT_IMPLEMENTED` 等）
- **编译期检查**: 使用 `std::enable_if_t` 和 `static_assert` 在编译期验证类型参数正确性
- **空指针保护**: `DeviceImpl::create()` 当前返回 `INFINI_STATUS_NOT_IMPLEMENTED`，保留接口用于未来扩展

### 依赖关系
- **外部依赖**:
  - `common_cpu.h`: 提供 `indexToOffset` 函数用于索引映射
  - `elementwise.h`: 定义 `ElementwiseInfo` 元数据结构和描述符宏
  - `utils.h`: 提供 `Result<T>` 类型、`CHECK_RESULT` 宏和 `utils::cast` 类型转换函数
  - `<utility>`: 提供 `std::forward` 完美转发
- **编译器要求**: 支持 OpenMP（可选，通过 `ENABLE_OMP` 宏控制）、C++14 或更高版本（`std::enable_if_t`）

### 设计模式
- **Pimpl 惯用模式**: `DeviceImpl` 通过 `Opaque` 前向声明隐藏实现细节，减少编译依赖和 ABI 稳定性
- **策略模式**: `DeviceImpl` 作为计算策略的接口，通过模板参数 `Op` 注入具体算法
- **模板方法模式**: `calculate` 定义计算骨架，`calculate_impl` 实现具体步骤，通过索引序列展开参数包
- **类型擦除**: 使用 `void*` 和 `std::vector<const void*>` 在运行期处理不同类型的张量数据，结合模板在编译期恢复类型信息
- **CRTP (奇异递归模板模式) 准备**: `Op` 模板参数可以是继承自基类的具体操作，支持静态多态
- **宏生成代码模式**: `CREATE_ELEMENTWISE_CPU_DESCRIPTOR` 宏自动化描述符创建流程，减少重复代码

### 算法复杂度
- **时间复杂度**: O(n)，其中 n 是输出张量的元素总数，每个元素独立计算一次
- **空间复杂度**: O(1) 额外空间（不包括输入输出），仅使用少量局部变量和索引计算
- **并行度**: O(n) 理论并行度，OpenMP 线程池自动调度，实际性能受 CPU 核心数和内存带宽限制

### 广播语义
- **输入广播**: 支持 NumPy 风格的广播规则，通过 `info.getInputBroadcasted()` 标记哪些输入被广播
- **形状检查**: 在 `ElementwiseInfo::create()` 中验证输出张量不能有广播维度（`hasBroadcastDim()`）
- **索引适配**: `get_input_idx` lambda 根据输入是否为广播张量，选择使用线性索引或计算多维偏移
