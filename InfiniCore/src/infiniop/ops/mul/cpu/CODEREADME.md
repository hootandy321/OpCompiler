# CPU 逐元素乘法运算核心实现文档

本模块实现了 InfiniOp 框架中 CPU 后端的逐元素乘法（Element-wise Multiplication）操作。该模块基于通用的逐元素运算框架，通过 OpenMP 并行化技术实现高性能的张量乘法运算，支持多种浮点数据类型（FP16、FP32、FP64、BF16）。

## 1. 模块结构

- **`mul_cpu.h`**: 定义乘法操作的函数符（Functor）和描述符类接口，通过宏展开生成完整的 Descriptor 类定义
- **`mul_cpu.cc`**: 实现 Descriptor 的构造方法（`create`）和计算调度方法（`calculate`），包含数据类型验证和计算核函数的分发逻辑

## 2. 核心类

### `op::mul::cpu::MulOp`

- **位置**: `mul_cpu.h:9-16`
- **主要功能**: 定义逐元素乘法的运算语义，作为可调用对象传递给通用的逐元素计算框架
- **核心成员**:
  - `num_inputs` (静态常量): 值为 2，标识该操作需要两个输入张量
- **核心方法**:
  - `operator()(const T& a, const T& b) const`: 对两个标量值执行乘法运算
    - **模板参数**: `T` - 数据类型（float、double、fp16_t、bf16_t 等）
    - **算法**: 直接返回 `a * b`，利用 C++ 原生乘法运算符
    - **时间复杂度**: O(1) 标量运算
- **生命周期**: 无状态设计，作为纯函数符使用，无需初始化或析构

### `op::mul::cpu::Descriptor`

- **位置**: 通过 `ELEMENTWISE_DESCRIPTOR(mul, cpu)` 宏在 `mul_cpu.h:6` 自动生成
- **继承体系**: 继承自 `InfiniopDescriptor`（定义于 `operator.h`）
- **主要功能**: 封装 CPU 后端乘法操作的元数据、计算资源和执行接口
- **核心成员**:
  - `_dtype`: `infiniDtype_t` 类型，存储输出张量的数据类型（F16/F32/F64/BF16）
  - `_info`: `op::elementwise::ElementwiseInfo` 类型，存储输入/输出张量的形状、步长、连续性等元数据
  - `_device_info`: `std::unique_ptr<op::elementwise::cpu::DeviceImpl>` 类型，管理 CPU 设备特定的实现对象（实际为空实现）
  - `_workspace_size`: `size_t` 类型，工作空间大小（当前实现固定为 0）
- **核心方法**:

#### `create(...)`
- **位置**: `mul_cpu.cc:7-30`
- **功能**: 构造并初始化乘法操作描述符
- **参数**:
  - `handle_`: 设备句柄（转换为 `device::cpu::Handle*`）
  - `desc_ptr`: 输出参数，用于返回构造的 Descriptor 指针
  - `out_desc`: 输出张量描述符
  - `input_desc_vec`: 输入张量描述符向量（包含两个输入张量）
- **算法流程**:
  1. **数据类型验证**（第 22 行）: 使用 `CHECK_DTYPE` 宏验证输出数据类型是否为支持的浮点类型（F16/F32/F64/BF16）
  2. **形状一致性检查**（第 24 行）: 使用 `CHECK_SAME_SHAPE` 宏验证三个张量（输出、输入 A、输入 B）的形状完全匹配
  3. **元数据提取**（第 18-20 行）: 从张量描述符中提取形状信息
  4. **逐元素信息构造**（第 26-27 行）: 调用 `CREATE_ELEMENTWISE_CPU_DESCRIPTOR` 宏，该宏内部调用 `ElementwiseInfo::create()` 生成包含形状、步长、连续性标志的元数据结构
  5. **对象构造**（第 22-28 行）: 使用 `new` 运算符分配 Descriptor 对象，传入 dtype、ElementwiseInfo、设备信息等
- **返回值**: `infiniStatus_t`，成功时返回 `INFINI_STATUS_SUCCESS`
- **错误处理**:
  - 不支持的数据类型返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
  - 形状不匹配返回 `INFINI_STATUS_BAD_TENSOR_SHAPE`
  - 元数据构造失败返回相应的错误码

#### `calculate(...)`
- **位置**: `mul_cpu.cc:32-52`
- **功能**: 执行逐元素乘法计算的调度入口
- **参数**:
  - `workspace`: 工作空间指针（未使用）
  - `workspace_size`: 工作空间大小（未使用）
  - `output`: 输出张量数据指针
  - `inputs`: 输入张量数据指针向量（inputs[0] 为张量 A，inputs[1] 为张量 B）
  - `stream`: 执行流指针（CPU 后端未使用）
- **算法流程**:
  1. **类型分发**（第 39-50 行）: 根据 `_dtype` 成员变量分发到不同的模板特化：
     - `INFINI_DTYPE_F16`: 调用 `_device_info->calculate<MulOp, fp16_t>(...)`
     - `INFINI_DTYPE_F32`: 调用 `_device_info->calculate<MulOp, float>(...)`
     - `INFINI_DTYPE_F64`: 调用 `_device_info->calculate<MulOp, double>(...)`
     - `INFINI_DTYPE_BF16`: 调用 `_device_info->calculate<MulOp, bf16_t>(...)`
  2. **模板实例化**: 每个分支调用 `DeviceImpl::calculate<MulOp, Tdata>()` 模板方法
  3. **并行计算执行**: 底层调用 `calculate_impl<Op, Tdata>`（定义于 `elementwise_cpu.h:152-181`）
- **时间复杂度**: O(n)，其中 n 为输出张量的元素数量
- **空间复杂度**: O(1) 额外空间（原地修改或仅使用输出缓冲区）

## 3. API 接口

```cpp
// 创建乘法操作描述符
infiniStatus_t op::mul::cpu::Descriptor::create(
    infiniopHandle_t handle,                  // [输入] 设备句柄
    Descriptor **desc_ptr,                    // [输出] 返回的描述符指针
    infiniopTensorDescriptor_t output_desc,   // [输入] 输出张量描述符
    std::vector<infiniopTensorDescriptor_t> input_descs  // [输入] 输入张量描述符向量
);
// 功能：初始化乘法操作，验证张量形状和数据类型，构造元数据
// 返回：成功返回 INFINI_STATUS_SUCCESS，失败返回相应错误码

// 执行乘法计算
infiniStatus_t op::mul::cpu::Descriptor::calculate(
    void *workspace,                          // [输入] 工作空间（未使用）
    size_t workspace_size,                    // [输入] 工作空间大小（未使用）
    void *output,                             // [输出] 输出数据缓冲区
    std::vector<const void *> inputs,         // [输入] 输入数据缓冲区向量
    void *stream                              // [输入] 执行流（CPU 未使用）
) const;
// 功能：执行逐元素乘法运算 output = inputs[0] * inputs[1]
// 返回：成功返回 INFINI_STATUS_SUCCESS

// 乘法运算符
template <typename T>
T op::mul::cpu::MulOp::operator()(const T &a, const T &b) const;
// 功能：计算两个标量的乘积
// 返回：a * b 的结果
```

## 4. 使用示例

```cpp
// 示例：在 CPU 上执行逐元素乘法运算 C = A * B

#include "mul_cpu.h"
#include "infiniop/operator_descriptor.h"

using namespace op::mul::cpu;

// 1. 准备张量描述符和数据
constexpr size_t num_elements = 1024;
constexpr size_t ndim = 2;
size_t shape[2] = {32, 32};
size_t strides[2] = {32, 1};

// 假设已初始化的数据指针
float *A = new float[num_elements];  // 输入张量 A
float *B = new float[num_elements];  // 输入张量 B
float *C = new float[num_elements];  // 输出张量 C

// 填充输入数据
for (size_t i = 0; i < num_elements; ++i) {
    A[i] = 2.0f;
    B[i] = 3.0f;
}

// 2. 创建张量描述符
infiniopTensorDescriptor_t desc_A, desc_B, desc_C;
infiniopCreateTensorDescriptor(&desc_A, INFINI_DTYPE_F32, ndim, shape, strides);
infiniopCreateTensorDescriptor(&desc_B, INFINI_DTYPE_F32, ndim, shape, strides);
infiniopCreateTensorDescriptor(&desc_C, INFINI_DTYPE_F32, ndim, shape, strides);

// 3. 创建乘法操作描述符
infiniopHandle_t handle;  // 假设已初始化的 CPU 设备句柄
Descriptor *mul_desc = nullptr;
std::vector<infiniopTensorDescriptor_t> inputs = {desc_A, desc_B};
auto status = Descriptor::create(handle, &mul_desc, desc_C, inputs);

if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
    return;
}

// 4. 执行乘法计算
std::vector<const void *> input_data = {A, B};
status = mul_desc->calculate(
    nullptr,  // 无需工作空间
    0,        // 工作空间大小为 0
    C,        // 输出缓冲区
    input_data,  // 输入缓冲区向量
    nullptr   // CPU 不需要 stream
);

if (status != INFINI_STATUS_SUCCESS) {
    // 处理计算错误
    return;
}

// 5. 验证结果
// C 中应包含 A * B 的结果，即全为 6.0
bool correct = true;
for (size_t i = 0; i < num_elements; ++i) {
    if (std::abs(C[i] - 6.0f) > 1e-6) {
        correct = false;
        break;
    }
}

// 6. 清理资源
delete mul_desc;
infiniopDestroyTensorDescriptor(desc_A);
infiniopDestroyTensorDescriptor(desc_B);
infiniopDestroyTensorDescriptor(desc_C);
delete[] A;
delete[] B;
delete[] C;
```

## 5. 实现细节

### 5.1 基于宏的代码生成策略

- **设计模式**: 使用 `ELEMENTWISE_DESCRIPTOR(mul, cpu)` 宏（定义于 `elementwise.h:15-54`）自动生成完整的 Descriptor 类
- **宏展开内容**:
  - 继承自 `InfiniopDescriptor` 基类
  - 声明 `_dtype`、`_info`、`_device_info`、`_workspace_size` 四个私有成员变量
  - 声明构造函数（初始化所有成员）
  - 声明 `workspaceSize()` 访问器
  - 声明静态 `create()` 方法和实例 `calculate()` 方法
- **优势**: 避免为每个逐元素操作重复编写相同的样板代码，确保所有操作的接口一致性

### 5.2 逐元素计算框架集成

- **继承层次**: 乘法操作复用通用的逐元素运算基础设施（`op::elementwise::cpu` 命名空间）
- **关键组件**:
  - **`ElementwiseInfo`**（定义于 `elementwise.h:69-203`）: 封装张量元数据的结构体
    - 使用紧凑的内存布局（`std::vector<size_t> _meta`）存储所有张量的形状、步长、连续性标志
    - 提供 `getOutputShape()`、`getInputStrides(index)`、`isOutputContiguous()` 等访问方法
    - 支持广播维度检测（`getInputBroadcasted()`）
  - **`DeviceImpl::calculate<Op, Tdata>()`**（定义于 `elementwise_cpu.h:184-193`）: 同类型输入的计算分发模板
    - 接收操作符类型 `Op`（此处为 `MulOp`）和数据类型 `Tdata`
    - 调用 `calculate_impl<Op, Tdata>` 执行实际计算
  - **`calculate_impl<Op, Tdata>()`**（定义于 `elementwise_cpu.h:152-181`）: 核心并行循环实现
    - 使用 `#pragma omp parallel for if (output_size > 1024)` 自动并行化（第 163 行）
    - 仅在元素数量大于 1024 时启用 OpenMP，避免小张量的并行开销
    - 对每个输出元素 `i`：
      1. 计算输出的线性偏移量 `out_idx`（连续张量直接使用 `i`，否则调用 `indexToOffset`）
      2. 对每个输入计算对应的线性偏移量 `get_input_idx(input_id)`
      3. 调用 `Op{}(ins[0][idx0], ins[1][idx1])` 执行乘法
      4. 将结果写入 `out[out_idx]`

### 5.3 内存布局与索引计算

- **连续内存优化**: 当张量连续时（`isOutputContiguous()` 或 `getInputContiguous()[id]` 为 true），直接使用扁平索引 `i`，避免昂贵的 `indexToOffset` 调用
- **非连续内存处理**: 调用 `op::common_cpu::indexToOffset()`（定义于 `common_cpu.h:19`）将线性索引映射到实际内存偏移量
  - 原理：通过形状和步长信息计算多维索引，再转换为线性偏移
  - 复杂度：O(ndim)，其中 ndim 为张量维度数
- **半精度浮点特殊处理**（第 175-179 行）:
  - 对 `fp16_t` 和 `bf16_t` 类型，先转换为 `float` 执行计算，再转换回原类型
  - 使用 `utils::cast<float>()` 和 `utils::cast<Tdata>()` 进行类型转换
  - 原因：避免半精度运算的精度损失和溢出风险

### 5.4 并行化策略

- **并行框架**: OpenMP（需要编译时定义 `ENABLE_OMP` 宏）
- **调度策略**:
  - 条件并行：仅在 `output_size > 1024` 时启用（`elementwise_cpu.h:163`）
  - 自动调度：OpenMP runtime 自动选择循环调度方式（static、dynamic、guided）
  - 默认线程数：使用 `omp_get_num_threads()` 返回的值（通常为 CPU 核心数）
- **线程安全性**: 每个线程处理独立的输出元素，无数据竞争，无需同步原语
- **性能特征**:
  - 小张量（≤1024 元素）：串行执行，避免线程创建开销
  - 大张量（>1024 元素）：并行执行，理论上可获得接近线性的加速比（受内存带宽限制）

### 5.5 数据类型支持

- **支持的 dtype**:
  - `INFINI_DTYPE_F16`: 16 位半精度浮点（IEEE 754 binary16）
  - `INFINI_DTYPE_F32`: 32 位单精度浮点（IEEE 754 binary32）
  - `INFINI_DTYPE_F64`: 64 位双精度浮点（IEEE 754 binary64）
  - `INFINI_DTYPE_BF16`: 16 位脑浮点（Brain Floating Point，8 位指数 + 7 位尾数）
- **类型验证**: 在 `create()` 方法中使用 `CHECK_DTYPE` 宏（定义于 `/home/qy/src/Infini/InfiniCore/src/utils/check.h`）进行编译期和运行时检查
- **类型转换**:
  - 同类型操作：所有输入和输出使用相同的 `Tdata` 类型
  - 混合类型：框架支持异构输入类型（通过 `calculate<Op, Tout, Tin...>` 模板），但当前乘法实现未使用此特性

### 5.6 错误处理机制

- **错误传播**: 使用 `infiniStatus_t` 枚举类型表示错误状态
  - `INFINI_STATUS_SUCCESS`: 操作成功
  - `INFINI_STATUS_BAD_TENSOR_DTYPE`: 不支持的数据类型
  - `INFINI_STATUS_BAD_TENSOR_SHAPE`: 张量形状不匹配
  - `INFINI_STATUS_BAD_PARAM`: 参数为空或无效
- **验证层次**:
  1. **参数验证**（`ElementwiseInfo::create` 第 150-157 行）: 检查描述符指针非空，输出不能有广播维度
  2. **类型验证**（第 22 行）: 使用 `CHECK_DTYPE` 宏验证 dtype 合法性
  3. **形状验证**（第 24 行）: 使用 `CHECK_SAME_SHAPE` 宏确保三个张量形状完全一致
  4. **元数据构造验证**（第 19 行）: `CHECK_RESULT` 宏检查 `ElementwiseInfo::create()` 的返回状态
- **错误恢复**: 所有错误检查失败时立即返回错误码，不执行部分计算，避免未定义行为

### 5.7 依赖关系

- **外部依赖**:
  - `elementwise_cpu.h`: 提供通用逐元素计算基础设施和 `DeviceImpl` 类
  - `elementwise.h`: 定义 `ElementwiseInfo` 元数据结构和 `ELEMENTWISE_DESCRIPTOR` 宏
  - `common_cpu.h`: 提供 `indexToOffset` 索引计算函数
  - `operator.h`: 定义 `InfiniopDescriptor` 基类
  - `tensor.h`: 定义张量描述符接口（`infiniopTensorDescriptor_t`）
- **编译依赖**:
  - OpenMP（可选，通过 `ENABLE_OMP` 宏控制）
  - C++14 或更高标准（支持 `std::enable_if_t`、变量模板等特性）
- **运行时依赖**:
  - 无外部库依赖，纯 CPU 实现
  - 不需要 BLAS、DNN 等加速库

### 5.8 设计模式

- **策略模式（Strategy Pattern）**:
  - `MulOp` 函数符封装乘法算法，作为策略对象传递给通用的 `calculate_impl` 框架
  - 其他逐元素操作（Add、Sub、Div 等）可复用相同框架，仅替换策略对象
- **模板方法模式（Template Method Pattern）**:
  - `Descriptor::create` 定义操作符创建的骨架（验证、元数据构造、对象分配）
  - 具体操作通过宏参数定制类型和命名空间
- **工厂模式（Factory Pattern）**:
  - 静态 `create()` 方法作为工厂函数，封装对象构造逻辑
  - 用户无需直接访问构造函数，通过工厂接口创建对象
- **RAII（Resource Acquisition Is Initialization）**:
  - `Descriptor` 使用智能指针 `std::unique_ptr` 管理 `DeviceImpl` 生命周期
  - 析构函数自动释放 `DeviceImpl` 资源（第 5 行定义）

### 5.9 性能优化技巧

1. **分支消除**: 通过模板特化避免运行时类型判断，编译器为每个 dtype 生成专用代码
2. **循环展开**: OpenMP 和编译器自动进行循环展开，减少循环控制开销
3. **缓存友好**: 连续内存访问模式充分利用 CPU 缓存行（通常为 64 字节）
4. **SIMD 潜力**: 简单的乘法运算易于编译器自动向量化（SSE/AVX 指令）
5. **惰性并行化**: 仅在数据量大时启用并行，避免小任务的调度开销
6. **内存预取**: 连续访问模式允许硬件预取器工作，减少缓存未命中

### 5.10 限制与约束

1. **不支持广播**: 当前实现要求所有输入和输出形状完全一致（`CHECK_SAME_SHAPE`），不支持 NumPy 风格的广播机制
2. **无原地操作**: 输出缓冲区必须与输入缓冲区不同，不支持 `A *= B` 式的原地更新
3. **固定工作空间**: `_workspace_size` 固定为 0，不支持需要额外临时内存的优化算法
4. **同步执行**: CPU 实现为同步调用，不支持异步执行流（`stream` 参数未使用）
5. **无设备管理**: CPU 实现没有真正的设备状态（`DeviceImpl::Opaque` 为空结构体）
