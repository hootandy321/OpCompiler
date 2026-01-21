# Ones Operation CPU Backend Core Implementation Documentation

本模块实现了 Infini 框架中 `ones` 操作的 CPU 后端，这是一个基于逐元素（elementwise）操作框架的特殊算子，用于生成全 1 张量。该模块采用了策略模式和模板元编程技术，支持多种数据类型的并行计算。

## 1. Module Structure

- **`ones_cpu.h`**: 头文件，定义 `OnesOp` 操作结构体并通过 `ELEMENTWISE_DESCRIPTOR` 宏声明 Descriptor 类接口
- **`ones_cpu.cc`**: 实现文件，包含 Descriptor 的构造函数、工厂方法 `create()` 和核心计算方法 `calculate()` 的具体实现

## 2. Core Classes

### `OnesOp` 结构体
- **Location**: `ones_cpu.h:9-16`
- **Primary Function**: 定义 ones 操作的核心语义，是一个仿函数（functor），用于将输入值转换为 1.0
- **Key Members**:
  - `num_inputs` (static constexpr size_t): 声明该操作需要 1 个输入张量（尽管实际不使用输入值）
- **Core Methods**:
  - `operator()(const T &x) const`: 仿函数调用运算符，接收任意类型的输入参数 `x`（未使用），返回类型 `T` 的常量 1.0。使用 `static_cast<T>(1.0)` 确保类型安全的转换
- **Lifecycle**: 编译期常量结构体，无运行时构造/析构开销

### `Descriptor` 类（由宏生成）
- **Location**: 通过 `ELEMENTWISE_DESCRIPTOR(ones, cpu)` 宏在 `elementwise.h` 中生成，实现在 `ones_cpu.cc`
- **Primary Function**: 管理 ones 操作的执行描述符，封装数据类型、张量元数据和设备实现
- **Key Members**:
  - `_dtype` (infiniDtype_t): 输出张量的数据类型
  - `_info` (op::elementwise::ElementwiseInfo): 存储输入/输出张量的形状、步幅、连续性等元数据
  - `_device_info` (std::unique_ptr<op::elementwise::cpu::DeviceImpl>): CPU 设备特定的实现对象（此处为空实现，因为 CPU 后端的 DeviceImpl::Opaque 为空结构体）
  - `_workspace_size` (size_t): 工作空间大小（此处为 0）
- **Core Methods**:
  - `create(handle_, desc_ptr, out_desc, input_desc_vec)`: 静态工厂方法，验证输入输出张量的数据类型兼容性和形状一致性，创建 `ElementwiseInfo` 元数据，构造 Descriptor 实例。支持 14 种数据类型（BYTE, BOOL, I8, I16, I32, I64, U8, U16, U32, U64, F16, F32, F64, BF16），使用 `CHECK_DTYPE` 宏进行编译期类型检查，使用 `CHECK_SAME_SHAPE` 验证输入输出形状匹配
  - `calculate(workspace, workspace_size, output, inputs, stream)`: 核心计算方法，根据 `_dtype` 分派到相应的模板特化，调用 `DeviceImpl::calculate<OnesOp, Tdata>()` 执行并行填充。对于 FP8、复数类型（C16, C32, C64, C128）返回 `INFINI_STATUS_NOT_IMPLEMENTED`，对其他类型使用 OpenMP 并行循环逐元素写入 1.0
- **Lifecycle**: 由 `create()` 工厂方法动态分配（`new Descriptor`），由智能指针管理生命周期，析构函数默认实现

## 3. API Interface

```cpp
namespace op::ones::cpu {

// Descriptor 类（由 ELEMENTWISE_DESCRIPTOR 宏生成）
class Descriptor final : public InfiniopDescriptor {
public:
    // 虚析构函数
    ~Descriptor();

    // 获取所需工作空间大小（返回 0，因为 ones 操作无需额外缓冲区）
    size_t workspaceSize() const;

    // 工厂方法：创建 Ones 操作描述符
    // @param handle_ 设备句柄（CPU）
    // @param desc_ptr 输出参数，用于返回新创建的 Descriptor 指针
    // @param out_desc 输出张量描述符
    // @param input_desc_vec 输入张量描述符向量（必须包含 1 个元素，尽管实际不使用其数据）
    // @return infiniStatus_t 状态码（SUCCESS/BAD_PARAM/BAD_TENSOR_DTYPE 等）
    static infiniStatus_t create(
        infiniopHandle_t handle_,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t out_desc,
        std::vector<infiniopTensorDescriptor_t> input_desc_vec);

    // 执行 ones 计算
    // @param workspace 工作空间缓冲区（此处未使用）
    // @param workspace_size 工作空间大小（必须为 0）
    // @param output 输出张量数据指针
    // @param inputs 输入张量数据指针向量（包含 1 个元素，实际不读取）
    // @param stream 执行流（CPU 后端未使用，保留参数以兼容 CUDA/GPU 接口）
    // @return infiniStatus_t 状态码
    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *output,
        std::vector<const void *> inputs,
        void *stream) const;
};

// Ones 操作的核心仿函数
struct OnesOp {
    static constexpr size_t num_inputs = 1;  // 声明需要 1 个输入张量（形式上）

    // 仿函数调用运算符
    // @tparam T 目标数据类型
    // @param x 输入值（未使用，仅为了符合 elementwise 框架接口）
    // @return T 类型转换后的 1.0
    template <typename T>
    T operator()(const T &x) const {
        return static_cast<T>(1.0);
    }
};

} // namespace op::ones::cpu
```

## 4. Usage Example

```cpp
#include "infiniop/ops/ones/cpu/ones_cpu.h"

using namespace op::ones::cpu;

// 示例：在 CPU 上创建一个形状为 {2, 3} 的 float 类型全 1 张量

void example_ones_cpu() {
    // 1. 准备设备句柄（假设已初始化）
    infiniopHandle_t handle = /* 获取 CPU 设备句柄 */;

    // 2. 创建输入/输出张量描述符
    // 注意：虽然 ones 操作不读取输入，但框架要求提供一个形状匹配的输入描述符
    std::vector<int64_t> shape = {2, 3};
    std::vector<int64_t> strides = {3, 1};  // 行主序（row-major）

    infiniopTensorDescriptor_t input_desc;
    infiniopCreateTensorDescriptor(&input_desc,
                                   INFINI_DEVICE_CPU, 0,
                                   INFINI_DTYPE_F32,
                                   2, shape.data(), strides.data());

    infiniopTensorDescriptor_t output_desc;
    infiniopCreateTensorDescriptor(&output_desc,
                                   INFINI_DEVICE_CPU, 0,
                                   INFINI_DTYPE_F32,
                                   2, shape.data(), strides.data());

    // 3. 创建 Ones 操作描述符
    Descriptor* ones_desc = nullptr;
    infiniStatus_t status = Descriptor::create(
        handle,
        &ones_desc,
        output_desc,
        {input_desc}  // 输入描述符向量
    );

    if (status != INFINI_STATUS_SUCCESS) {
        // 错误处理：数据类型不支持或形状不匹配
        return;
    }

    // 4. 分配输出内存
    float* output_data = new float[6];  // 2 * 3 = 6 个元素

    // 5. 执行计算（stream 传 nullptr，CPU 后端不使用）
    float dummy_input = 0.0f;  // 虚拟输入，实际不会被读取
    status = ones_desc->calculate(
        nullptr,              // workspace = nullptr
        0,                    // workspace_size = 0
        output_data,          // 输出缓冲区
        {&dummy_input},       // 输入数据指针向量
        nullptr               // stream = nullptr
    );

    if (status == INFINI_STATUS_SUCCESS) {
        // output_data 现在包含 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
        // 打印结果
        for (int i = 0; i < 6; ++i) {
            printf("%f ", output_data[i]);  // 输出: 1.0 1.0 1.0 1.0 1.0 1.0
        }
        printf("\n");
    }

    // 6. 清理资源
    delete ones_desc;
    delete[] output_data;
    infiniopDestroyTensorDescriptor(input_desc);
    infiniopDestroyTensorDescriptor(output_desc);
}

// 高级示例：使用不同的数据类型
void example_ones_with_bfloat16() {
    infiniopHandle_t handle = /* CPU handle */;
    std::vector<int64_t> shape = {4};

    infiniopTensorDescriptor_t input_desc, output_desc;
    // ... 创建描述符（使用 INFINI_DTYPE_BF16）...

    Descriptor* ones_desc = nullptr;
    Descriptor::create(handle, &ones_desc, output_desc, {input_desc});

    bf16_t* output = new bf16_t[4];
    bf16_t dummy_input;

    ones_desc->calculate(nullptr, 0, output, {&dummy_input}, nullptr);

    // output 现在包含 4 个 bfloat16 类型的 1.0

    delete ones_desc;
    delete[] output;
}
```

## 5. Implementation Details

### **设计模式与架构**
- **策略模式（Strategy Pattern）**: `OnesOp` 作为一个策略对象，通过仿函数接口定义操作语义，允许 elementwise 框架以统一方式处理不同操作
- **工厂模式（Factory Pattern）**: `Descriptor::create()` 作为工厂方法，封装对象创建逻辑和验证流程
- **CRTP（奇异递归模板模式）基础**: `ELEMENTWISE_DESCRIPTOR` 宏使用命名空间注入（`op::OP::NAMESPACE`）生成特定操作的 Descriptor 类，避免代码重复
- **模板元编程**: `DeviceImpl::calculate<Op, Tdata>()` 使用模板特化在编译期生成针对不同数据类型的高效代码

### **内存管理**
- **所有权语义**: `Descriptor::create()` 返回裸指针（`*desc_ptr`），调用者负责管理生命周期（通过 `delete` 释放）
- **零工作空间设计**: `_workspace_size` 硬编码为 0，因为 ones 操作无需临时缓冲区，这是与需要中间结果的 elementwise 操作（如某些超越函数）的关键区别
- **元数据封装**: `ElementwiseInfo` 使用单一 `std::vector<size_t>` 存储所有形状、步幅、连续性和广播标志，通过指针偏移访问不同区域，减少内存分配开销

### **并发与性能优化**
- **OpenMP 并行化**: `calculate_impl()` 使用 `#pragma omp parallel for` 指令，启用多线程并行填充。对于大型张量（`output_size > 1024`），自动并行化；小型张量串行执行以避免线程创建开销
- **连续性优化**:
  - 通过 `info.isOutputContiguous()` 和 `info.getInputContiguous()` 检测张量内存布局
  - 连续张量直接使用线性索引 `i`，非连续张量调用 `indexToOffset()` 计算物理偏移
  - 对 FP16 和 BF16 类型，先提升到 float 计算后再转换回原类型，利用现代 CPU 的 SIMD 指令（如 AVX-512 的 VFP16DP 指令）
- **分支预测友好**: `calculate()` 方法使用大型 switch 语句（而非 if-else 链），编译器可优化为跳转表（jump table），降低分支预测失败率

### **类型系统与错误处理**
- **编译期类型检查**: `CHECK_DTYPE` 宏展开为静态断言，在编译期验证数据类型合法性，防止无效类型实例化
- **运行时类型分派**: `calculate()` 方法使用 switch-on-dtype 模式，避免虚函数调用开销（静态绑定）
- **错误码传播**: 所有错误通过 `infiniStatus_t` 返回值传播，不使用 C++ 异常（符合 C 风格 API 设计）
- **未实现类型处理**: FP8（F8）和复数类型（C16, C32, C64, C128）返回 `INFINI_STATUS_NOT_IMPLEMENTED`，而非静默失败

### **数据类型支持**
模块支持 14 种标量数据类型，分为四类：
1. **整型**: BYTE (uint8_t), BOOL, I8, I16, I32, I64, U8, U16, U32, U64
2. **浮点型**: F16 (fp16_t), F32 (float), F64 (double)
3. **Brain 浮点**: BF16 (bf16_t)
4. **不支持**: F8（FP8），C16/C32/C64/C128（复数类型）

类型转换使用 `static_cast<T>(1.0)` 确保从浮点字面量到目标类型的精确转换，避免截断未定义行为。

### **依赖关系**
- **直接依赖**: `elementwise_cpu.h`（提供 `DeviceImpl` 和 `CREATE_ELEMENTWISE_CPU_DESCRIPTOR` 宏）
- **间接依赖**:
  - `common_cpu.h`（提供 `indexToOffset()` 索引计算工具）
  - `operator.h`, `tensor.h`（定义张量描述符和设备句柄类型）
  - `utils.h`（提供 `CHECK_RESULT`, `CHECK_DTYPE`, `CHECK_SAME_SHAPE` 宏）
- **外部依赖**: OpenMP（`#pragma omp parallel for`），标准 C++ 库（`<vector>`, `<cstdint>`）

### **关键算法复杂度**
- **时间复杂度**: O(n)，其中 n 为输出张量的元素数量（`output_size`）。每个元素写入常数时间操作（写入常量 1.0）
- **空间复杂度**: O(1) 额外空间（除输入输出缓冲区外）。元数据存储大小为 O(ndim * (1 + num_inputs))，与张量大小无关
- **并行扩展性**: 理想情况下为 O(n/p)，其中 p 为 OpenMP 线程数。实际性能受内存带宽限制，属于内存带宽密集型（memory-bound）操作

### **与 Elementwise 框架的集成**
本模块巧妙地复用了 elementwise 框架的基础设施：
- `OnesOp::num_inputs = 1`: 声明为单输入操作，符合 `elementwise_cpu.h` 中的模板参数 `sizeof...(Tin) == Op::num_inputs` 约束
- `operator()` 忽略输入参数: 虽然 `calculate_impl` 会读取输入数据（`ins[Is][get_input_idx(Is)]`），但 `OnesOp` 不使用该值，仅返回常量 1.0
- 广播和形状检查: 通过 `CHECK_SAME_SHAPE` 和 `ElementwiseInfo::create()` 自动处理输入输出的形状匹配和广播逻辑

这种设计使得 ones 操作无需重写并行循环、索引计算和内存布局处理代码，实现了高度代码复用。
