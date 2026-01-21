# Zeros CPU 操作核心实现文档

该模块实现了 Infini 框架中 Zeros 操作的 CPU 后端，用于生成所有元素为零的张量。这是一个逐元素（elementwise）操作的特殊实现，完全基于 Infini 的逐元素操作框架构建。

## 1. 模块结构

- **`zeros_cpu.h`**: 定义 Zeros 操作的 CPU 描述符和操作符结构体，通过宏继承通用逐元素操作框架
- **`zeros_cpu.cc`**: 实现 Zeros 描述符的创建、验证和计算逻辑，支持 15 种数据类型

## 2. 核心类

### `op::zeros::cpu::ZerosOp`
- **位置**: `zeros_cpu.h:9-16`
- **主要功能**: 定义 Zeros 操作的核心语义，将任意输入转换为零值
- **关键成员**:
  - `num_inputs`: 静态常量，值为 1，表示该操作接受 1 个输入张量（尽管实际不使用输入）
- **核心方法**:
  - `template <typename T> T operator()(const T &x) const`: 返回类型 T 的零值（`static_cast<T>(0.0)`），忽略输入参数 x
- **生命周期**: 无状态结构体，编译期常量，可作为模板参数传递

### `op::zeros::cpu::Descriptor`
- **位置**: 通过 `ELEMENTWISE_DESCRIPTOR` 宏在 `zeros_cpu.h:6` 展开定义
- **主要功能**: Zeros 操作的 CPU 描述符，管理操作的元数据、设备信息和执行逻辑
- **关键成员**:
  - `_dtype`: `infiniDtype_t`，输出张量的数据类型
  - `_info`: `op::elementwise::ElementwiseInfo`，张量形状、步长等元数据
  - `_device_info`: `std::unique_ptr<op::elementwise::cpu::DeviceImpl>`，CPU 设备特定实现
  - `_workspace_size`: `size_t`，工作空间大小（对于 Zeros 操作为 0）
- **核心方法**:
  - `static infiniStatus_t create(...)`: 创建描述符实例，验证数据类型和形状一致性
  - `infiniStatus_t calculate(...)`: 执行 Zeros 操作，根据数据类型分发到相应的模板实例
  - `~Descriptor()`: 析构函数（默认实现）
  - `size_t workspaceSize() const`: 返回所需工作空间大小（始终为 0）
- **生命周期**:
  - 通过静态 `create()` 方法构造，执行运行时验证
  - 使用 RAII 管理资源，`_device_info` 通过 `unique_ptr` 自动释放
  - 析构时自动清理 `ElementwiseInfo` 和 `DeviceImpl` 资源

## 3. API 接口

```cpp
// 创建 Zeros 操作描述符
static infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,                    // CPU 设备句柄
    Descriptor **desc_ptr,                      // 输出：描述符指针
    infiniopTensorDescriptor_t out_desc,        // 输出张量描述符
    std::vector<infiniopTensorDescriptor_t> input_desc_vec  // 输入张量描述符向量（需包含 1 个输入）
);
// 返回：INFINI_STATUS_SUCCESS 或错误码（INFINI_STATUS_BAD_TENSOR_DTYPE、INFINI_STATUS_BAD_TENSOR_SHAPE）

// 执行 Zeros 计算
infiniStatus_t Descriptor::calculate(
    void *workspace,                            // 工作空间指针（可为 nullptr）
    size_t workspace_size,                      // 工作空间大小
    void *output,                               // 输出张量数据指针
    std::vector<const void *> inputs,           // 输入张量数据指针向量（内容被忽略）
    void *stream                                // CUDA 流（CPU 后端忽略此参数）
) const;
// 返回：INFINI_STATUS_SUCCESS 或 INFINI_STATUS_NOT_IMPLEMENTED（对于 F8 和复数类型）

// 查询工作空间大小
size_t workspaceSize() const;
// 返回：0（Zeros 操作不需要额外工作空间）
```

## 4. 使用示例

```cpp
// 示例：创建形状为 {3, 4} 的零张量（float32 类型）
#include "zeros_cpu.h"
#include "cpu_handle.h"

// 1. 准备张量描述符
std::vector<size_t> shape = {3, 4};
std::vector<ptrdiff_t> strides = {4, 1};  // 行主序
auto out_desc = new TensorDescriptor(INFINI_DTYPE_F32, shape, strides);
auto input_desc = new TensorDescriptor(INFINI_DTYPE_F32, shape, strides);  // 形状必须一致

// 2. 创建 CPU 句柄
infiniopHandle_t handle = new device::cpu::Handle();

// 3. 创建 Zeros 描述符
op::zeros::cpu::Descriptor* zeros_desc = nullptr;
std::vector<infiniopTensorDescriptor_t> inputs = {input_desc};
auto status = op::zeros::cpu::Descriptor::create(handle, &zeros_desc, out_desc, inputs);
if (status != INFINI_STATUS_SUCCESS) {
    // 错误处理
}

// 4. 分配输出内存
float* output_data = new float[out_desc->numel()];

// 5. 执行计算（输入数据可为任意值）
float* dummy_input = new float[out_desc->numel()];
status = zeros_desc->calculate(nullptr, 0, output_data, {dummy_input}, nullptr);

// 6. 清理资源
delete zeros_desc;
delete[] output_data;
delete[] dummy_input;
delete out_desc;
delete input_desc;
```

## 5. 实现细节

### 5.1 数据类型支持
支持 15 种标量数据类型，通过 `calculate()` 方法中的 switch-case 语句分发：

**整数类型（9 种）**:
- `INFINI_DTYPE_BYTE` (uint8_t)
- `INFINI_DTYPE_BOOL` (bool)
- `INFINI_DTYPE_I8`, `I16`, `I32`, `I64`
- `INFINI_DTYPE_U8`, `U16`, `U32`, `U64`

**浮点类型（6 种）**:
- `INFINI_DTYPE_F16` (fp16_t) - 通过 fp32 中间计算
- `INFINI_DTYPE_F32` (float)
- `INFINI_DTYPE_F64` (double)
- `INFINI_DTYPE_BF16` (bf16_t) - 通过 fp32 中间计算

**不支持**（返回 `INFINI_STATUS_NOT_IMPLEMENTED`）:
- `INFINI_DTYPE_F8`
- 所有复数类型（`C16`, `C32`, `C64`, `C128`）

### 5.2 元数据管理
`ElementwiseInfo` 结构体存储所有张量的元数据，内存布局采用单一连续缓冲区：

**内存布局**（从低地址到高地址）:
1. 输出形状 (`size_t * ndim`)
2. 输出步长 (`ptrdiff_t * ndim`)
3. 所有输入形状 (`size_t * input_size * ndim`)
4. 所有输入步长 (`ptrdiff_t * input_size * ndim`)
5. 输入连续性标志 (`bool * input_size`)
6. 输入广播标志 (`bool * input_size`)

对于 Zeros 操作，`input_size` 固定为 1，尽管输入数据不被使用。

### 5.3 计算内核
计算逻辑继承自 `elementwise::cpu::DeviceImpl::calculate()` 模板方法：

**执行流程**:
1. 将输出指针转换为类型 `Tdata*`
2. 将输入指针数组转换为 `const Tdata*` 数组（虽然内容被忽略）
3. 使用 OpenMP 并行化遍历所有输出元素（当元素数 > 1024 时）
4. 对于每个元素：
   - 如果输出连续：直接使用线性索引 `i`
   - 否则：调用 `indexToOffset()` 根据形状和步长计算偏移量
   - 调用 `ZerosOp::operator()` 生成零值
   - 对于 fp16/bf16 类型：通过 `utils::cast` 从 float 转换
5. 将零值写入输出张量

**索引计算**（非连续张量）:
```cpp
size_t out_idx = info.isOutputContiguous()
                   ? i
                   : op::common_cpu::indexToOffset(i, info.getNdim(),
                                                    info.getOutputShape(),
                                                    info.getOutputStrides());
```

### 5.4 并行化策略
- 使用 OpenMP `#pragma omp parallel for` 指令
- 自动并行化条件：`output_size > 1024`
- 每个线程独立处理不同的输出元素，无共享状态，无需同步
- 线程安全：`ZerosOp` 是无状态的，`operator()` 为纯函数

### 5.5 类型转换机制
通过 `utils::cast` 模板函数处理特殊类型：

**fp16/bf16 转换路径**:
```cpp
// 元素级计算（elementwise_cpu.h:175-176）
out[out_idx] = utils::cast<Tdata>(
    Op{}(utils::cast<float>(ins[Is][get_input_idx(Is)])...)
);
```

对于 Zeros 操作：
1. `ZerosOp::operator()` 返回 `0.0` (double)
2. `utils::cast<float>(0.0)` → `0.0f`
3. `utils::cast<Tdata>(0.0f)` → fp16/bf16 零值（通过 `_f32_to_f16` 或 `_f32_to_bf16`）

### 5.6 验证机制
在 `create()` 方法中执行严格验证：

**数据类型检查**（`CHECK_DTYPE` 宏）:
```cpp
CHECK_DTYPE(dtype,
    INFINI_DTYPE_BYTE, INFINI_DTYPE_BOOL, INFINI_DTYPE_I8,
    INFINI_DTYPE_I16, INFINI_DTYPE_I32, INFINI_DTYPE_I64,
    INFINI_DTYPE_U8, INFINI_DTYPE_U16, INFINI_DTYPE_U32, INFINI_DTYPE_U64,
    INFINI_DTYPE_F8, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64,
    INFINI_DTYPE_BF16
);
```

**形状一致性检查**（`CHECK_SAME_SHAPE` 宏）:
```cpp
CHECK_SAME_SHAPE(y_shape, x_shape);
```
确保输出张量和输入张量的形状完全一致，即使输入数据不被使用。

**Result 类型检查**:
```cpp
auto info_result = op::elementwise::ElementwiseInfo::create(out_desc, input_desc_vec);
CHECK_RESULT(info_result);  // 如果失败，返回错误码
```

### 5.7 内存管理
- **描述符内存**: 通过 `new Descriptor(...)` 动态分配，由用户负责 `delete`
- **元数据内存**: `ElementwiseInfo` 使用 `std::vector<size_t>` 存储，RAII 自动管理
- **设备实现**: `DeviceImpl` 通过 `std::unique_ptr` 管理，析构时自动释放
- **工作空间**: 不需要额外工作空间（`_workspace_size = 0`）

### 5.8 设计模式
- **策略模式（Strategy Pattern）**: `ZerosOp` 作为可调用策略，通过模板参数注入 `DeviceImpl::calculate()`
- **工厂模式（Factory Pattern）**: 静态 `create()` 方法作为工厂，封装验证和构造逻辑
- **模板方法模式（Template Method Pattern）**: `ELEMENTWISE_DESCRIPTOR` 宏定义描述符骨架，子类实现 `create()` 和 `calculate()`
- **RAII（Resource Acquisition Is Initialization）**: 使用智能指针和 STL 容器自动管理资源

### 5.9 错误处理
错误传播采用 `infiniStatus_t` 枚举：

- `INFINI_STATUS_SUCCESS`: 操作成功
- `INFINI_STATUS_BAD_PARAM`: 参数为空（输入描述符向量为空）
- `INFINI_STATUS_BAD_TENSOR_DTYPE`: 不支持的数据类型
- `INFINI_STATUS_BAD_TENSOR_STRIDES`: 输出张量有广播维度（不允许）
- `INFINI_STATUS_BAD_TENSOR_SHAPE`: 输入/输出形状不匹配
- `INFINI_STATUS_NOT_IMPLEMENTED`: F8 或复数类型

错误信息通过 `std::cerr` 输出到标准错误流，包含失败条件、函数名、文件名和行号。

### 5.10 依赖关系
**外部依赖**:
- OpenMP（可选，通过 `ENABLE_OMP` 宏控制）
- C++ STL（`<vector>`, `<variant>`, `<tuple>`, `<cstring>`, `<type_traits>`）

**模块间依赖**:
- `infiniop/elementwise/cpu/elementwise_cpu.h`: 逐元素操作 CPU 框架
- `infiniop/elementwise/elementwise.h`: `ElementwiseInfo` 和 `ELEMENTWISE_DESCRIPTOR` 宏
- `infiniop/devices/cpu/common_cpu.h`: CPU 通用工具（如 `indexToOffset`）
- `utils/check.h`: 验证宏（`CHECK_DTYPE`, `CHECK_SAME_SHAPE`, `CHECK_RESULT`）
- `utils/result.hpp`: `Result<T>` 类型用于错误处理
- `utils/custom_types.h`: fp16/bf16 类型定义和转换函数
- `infinicore.h`: 核心类型定义（`infiniStatus_t`, `infiniDtype_t`）

### 5.11 性能特性
- **时间复杂度**: O(n)，其中 n 为输出张量的元素数量
- **空间复杂度**: O(1) 额外空间（不使用工作空间）
- **并行性**: OpenMP 并行循环，线程数由运行时决定（通常为 CPU 核心数）
- **缓存友好**: 线性遍历输出张量，对于连续张量实现顺序内存访问
- **分支预测**: `isOutputContiguous()` 检查在循环外，分支预测器优化良好

### 5.12 特殊优化
尽管 Zeros 操作不读取输入数据，框架仍然：
1. 验证输入张量的形状一致性（确保 API 一致性）
2. 调用 `ZerosOp::operator()` 时传递输入值（但被忽略）
3. 对于 fp16/bf16，通过 float 中间类型确保精度（即使零值不需要此转换）

这种设计保持与其他逐元素操作的代码一致性，但略显冗余。未来的优化可能引入专门的"无输入"操作类型。
