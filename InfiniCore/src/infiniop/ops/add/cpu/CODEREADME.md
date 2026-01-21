# CPU Backend 加法操作实现文档

## 概述

本模块实现了 InfiniOp 框架中加法操作 (Add) 的 CPU 后端。作为逐元素 (element-wise) 操作的一种，加法操作对两个输入张量执行逐元素加法运算，并将结果写入输出张量。该实现基于通用的逐元素操作 CPU 框架，支持多种数据类型，并利用 OpenMP 实现多线程并行计算以提升性能。

## 1. 模块结构

- **`add_cpu.h`**: 定义加法操作的描述符类和操作算子结构体
- **`add_cpu.cc`**: 实现描述符的创建和计算方法

## 2. 核心类与结构

### 2.1 `AddOp` 结构体

- **位置**: `add_cpu.h:9-16`
- **主要功能**: 定义加法操作的语义，作为可调用对象传递给底层计算框架
- **关键成员**:
  - `num_inputs` (static constexpr size_t): 值为 2，指定操作需要两个输入
- **核心方法**:
  - `template <typename T> T operator()(const T &a, const T &b) const`: 执行加法运算 `return a + b`，支持任意可加类型
- **设计模式**: 函数对象 (Functor) 模式，使操作可作为模板参数传递

### 2.2 `Descriptor` 类

- **位置**: 通过 `ELEMENTWISE_DESCRIPTOR(add, cpu)` 宏在 `add_cpu.h:6` 自动生成
- **继承**: 继承自 `InfiniopDescriptor`
- **主要功能**: 管理加法操作的元数据、设备信息和计算逻辑
- **关键成员**:
  - `_dtype` (infiniDtype_t): 存储操作的数据类型
  - `_info` (op::elementwise::ElementwiseInfo): 存储张量形状、步长等元数据
  - `_device_info` (std::unique_ptr<op::elementwise::cpu::DeviceImpl>): CPU 设备实现指针
  - `_workspace_size` (size_t): 工作空间大小（本操作为 0）
- **生命周期**:
  - **构造**: 通过静态工厂方法 `create()` 创建
  - **析构**: `~Descriptor()` 在 `add_cpu.cc:5` 中默认实现
  - **所有权**: 调用者负责管理 Descriptor 实例的生命周期

## 3. API 接口

### 3.1 描述符创建接口

```cpp
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,              // [in] 设备句柄，指向 device::cpu::Handle
    Descriptor **desc_ptr,                 // [out] 输出参数，返回创建的描述符指针
    infiniopTensorDescriptor_t out_desc,   // [in] 输出张量描述符
    std::vector<infiniopTensorDescriptor_t> input_desc_vec  // [in] 输入张量描述符向量
);
// 返回: INFINI_STATUS_SUCCESS 成功，错误码失败
// 功能: 创建加法操作描述符，验证数据类型和形状一致性
```

**实现细节** (`add_cpu.cc:7-30`):
1. 类型转换：将 `handle_` 转换为 `device::cpu::Handle*`
2. 数据类型验证：使用 `CHECK_DTYPE` 宏验证支持的数据类型
   - 支持类型：F16, F32, F64, BF16, I32, I64
3. 形状一致性检查：使用 `CHECK_SAME_SHAPE` 宏确保三个张量形状完全相同
4. 元数据创建：调用 `CREATE_ELEMENTWISE_CPU_DESCRIPTOR` 宏创建 `ElementwiseInfo`

### 3.2 计算接口

```cpp
infiniStatus_t Descriptor::calculate(
    void *workspace,                       // [in] 工作空间指针（未使用）
    size_t workspace_size,                 // [in] 工作空间大小（未使用）
    void *output,                          // [out] 输出张量数据指针
    std::vector<const void *> inputs,      // [in] 输入张量数据指针向量，inputs[0]=a, inputs[1]=b
    void *stream                           // [in] 执行流（CPU 后端未使用）
) const;
// 返回: INFINI_STATUS_SUCCESS 成功，错误码失败
// 功能: 执行逐元素加法计算
```

**实现细节** (`add_cpu.cc:32-57`):
1. 数据类型分发：根据 `_dtype` switch 到对应的模板特化
2. 类型映射：
   - `INFINI_DTYPE_F16` → `fp16_t`
   - `INFINI_DTYPE_F32` → `float`
   - `INFINI_DTYPE_F64` → `double`
   - `INFINI_DTYPE_BF16` → `bf16_t`
   - `INFINI_DTYPE_I32` → `int32_t`
   - `INFINI_DTYPE_I64` → `int64_t`
3. 调用底层计算：`_device_info->calculate<AddOp, Ttype>(_info, output, inputs, stream)`

## 4. 使用示例

```cpp
#include "add_cpu.h"
#include <vector>

using namespace op::add::cpu;

// 1. 准备张量描述符（假设已创建）
infiniopTensorDescriptor_t a_desc, b_desc, c_desc;
// ... 初始化描述符，确保形状相同，数据类型为支持的类型 ...

// 2. 创建设备句柄（假设已初始化）
infiniopHandle_t handle;
// ... 初始化 CPU 设备句柄 ...

// 3. 创建操作描述符
Descriptor* add_desc = nullptr;
infiniStatus_t status = Descriptor::create(
    handle,
    &add_desc,
    c_desc,      // 输出描述符
    {a_desc, b_desc}  // 输入描述符向量
);

if (status != INFINI_STATUS_SUCCESS) {
    // 错误处理
}

// 4. 准备数据指针
void* a_ptr;  // 输入 A 的数据指针
void* b_ptr;  // 输入 B 的数据指针
void* c_ptr;  // 输出 C 的数据指针
// ... 分配并初始化内存 ...

// 5. 执行计算
status = add_desc->calculate(
    nullptr,           // 无需工作空间
    0,                 // 工作空间大小为 0
    c_ptr,             // 输出指针
    {a_ptr, b_ptr},    // 输入指针向量
    nullptr            // CPU 后端不使用流
);

// 6. 清理资源
delete add_desc;
```

## 5. 实现细节

### 5.1 宏驱动的架构设计

**`ELEMENTWISE_DESCRIPTOR` 宏** (`elementwise.h:15-54`):
- 展开为 `op::add::cpu::Descriptor` 类的完整定义
- 自动生成成员变量、构造函数和公共接口声明
- 实现了代码复用，所有逐元素操作共享相同的描述符结构

**关键设计**:
```cpp
ELEMENTWISE_DESCRIPTOR(add, cpu)
// 展开为:
namespace op::add::cpu {
    class Descriptor final : public InfiniopDescriptor {
        infiniDtype_t _dtype;
        op::elementwise::ElementwiseInfo _info;
        std::unique_ptr<op::elementwise::cpu::DeviceImpl> _device_info;
        size_t _workspace_size;
        // ... 构造函数、析构函数、create、calculate 方法声明 ...
    };
}
```

### 5.2 元数据管理：ElementwiseInfo 结构

**位置**: `elementwise.h:69-203`

**内存布局** (使用单一 `std::vector<size_t>` 存储所有元数据):
```
_meta[0 ... ndim-1]           : 输出形状 (size_t[])
_meta[ndim ... 2*ndim-1]      : 输出步长 (ptrdiff_t[])
_meta[2*ndim ... (2+n)*ndim-1]: 所有输入形状 (size_t[])
_meta[(2+n)*ndim ... (2+2n)*ndim-1]: 所有输入步长 (ptrdiff_t[])
_meta[(2+2n)*ndim ... (2+2n)*ndim+n-1]: 输入连续性标志 (bool[])
_meta[(2+2n)*ndim+n ... (2+2n)*ndim+2n-1]: 输入广播标志 (bool[])
```

**访问方法**:
- `getOutputSize()`: 返回输出张量的元素总数
- `getNdim()`: 返回张量维度数
- `isOutputContiguous()`: 输出是否连续存储
- `getOutputShape()/getOutputStrides()`: 输出形状和步长
- `getInputShape(index)/getInputStrides(index)`: 指定输入的形状和步长
- `getInputContiguous()[index]`: 指定输入是否连续
- `getInputBroadcasted()[index]`: 指定输入是否需要广播

**创建逻辑** (`ElementwiseInfo::create`, `elementwise.h:146-202`):
1. 验证输出不能有广播维度
2. 计算元数据所需总内存大小
3. 分配对齐的 `std::vector<size_t>` 缓冲区
4. 使用 `std::memcpy` 复制形状和步长数据
5. 设置连续性和广播标志

### 5.3 CPU 计算核心：DeviceImpl::calculate

**位置**: `elementwise_cpu.h:152-193`

**算法流程** (`calculate_impl` 函数):
```cpp
template <typename Op, typename Tdata, size_t... Is, typename... Args>
void calculate_impl(
    const ElementwiseInfo &info,
    void *output,
    const std::vector<const void *> &inputs,
    std::index_sequence<Is...>,
    Args &&...args
) {
    Tdata *out = reinterpret_cast<Tdata *>(output);
    std::array<const Tdata *, sizeof...(Is)> ins = {
        reinterpret_cast<const Tdata *>(inputs[Is])...
    };
    const ptrdiff_t output_size = info.getOutputSize();

    #pragma omp parallel for if (output_size > 1024)
    for (ptrdiff_t i = 0; i < output_size; ++i) {
        // 计算输出索引（处理非连续情况）
        size_t out_idx = info.isOutputContiguous()
            ? i
            : op::common_cpu::indexToOffset(i, info.getNdim(),
                                            info.getOutputShape(),
                                            info.getOutputStrides());

        // 获取输入索引的 lambda
        auto get_input_idx = [&](size_t input_id) {
            return info.getInputContiguous()[input_id]
                ? i
                : op::common_cpu::indexToOffset(i, info.getNdim(),
                                                info.getInputShape(input_id),
                                                info.getInputStrides(input_id));
        };

        // 执行加法操作（针对 fp16/bf16 特殊处理）
        if constexpr (std::is_same_v<Tdata, fp16_t> ||
                      std::is_same_v<Tdata, bf16_t>) {
            out[out_idx] = utils::cast<Tdata>(
                Op{}(utils::cast<float>(ins[Is][get_input_idx(Is)])...,
                     std::forward<Args>(args)...)
            );
        } else {
            out[out_idx] = Op{}(ins[Is][get_input_idx(Is)]...,
                                std::forward<Args>(args)...);
        }
    }
}
```

**关键特性**:
1. **并行化**: 使用 OpenMP `#pragma omp parallel for if (output_size > 1024)`
   - 小数据量（≤1024 元素）串行执行，避免线程开销
   - 大数据量自动并行化，利用多核 CPU

2. **索引优化**:
   - 连续张量：直接使用线性索引 `i`
   - 非连续张量：调用 `indexToOffset` 计算物理偏移量

3. **半精度浮点特殊处理**:
   - `fp16_t` 和 `bf16_t` 先提升为 `float` 运算
   - 运算结果再转换回半精度
   - 避免精度损失和溢出问题

4. **零拷贝设计**:
   - 直接操作原始数据指针
   - 无中间缓冲区分配

### 5.4 类型转换与精度处理

**utils::cast 模板函数** (`custom_types.h:23-48`):
```cpp
template <typename TypeTo, typename TypeFrom>
TypeTo cast(TypeFrom val) {
    if constexpr (std::is_same<TypeTo, TypeFrom>::value) {
        return val;  // 同类型，直接返回
    } else if constexpr (std::is_same<TypeTo, fp16_t>::value &&
                        std::is_same<TypeFrom, float>::value) {
        return _f32_to_f16(val);  // float → fp16
    } else if constexpr (std::is_same<TypeFrom, fp16_t>::value &&
                        std::is_same<TypeTo, float>::value) {
        return _f16_to_f32(val);  // fp16 → float
    }
    // ... 其他类型转换特化 ...
    else {
        return static_cast<TypeTo>(val);  // 标准转换
    }
}
```

**设计原理**:
- 编译期 `if constexpr` 消除死代码，零运行时开销
- 使用 `std::is_same_v` 编译期类型检查
- 半精度类型通过专门的转换函数维护数值精度

### 5.5 错误处理机制

**CHECK 宏系列** (`check.h`):

1. **CHECK_DTYPE** (`check.h:47-60`):
```cpp
#define CHECK_DTYPE(DT, ...)               \
    do {                                   \
        auto dtype_is_supported = false;   \
        for (auto dt : {__VA_ARGS__}) {    \
            if (dt == DT) {                \
                dtype_is_supported = true; \
                break;                     \
            }                              \
        }                                  \
        CHECK_OR_DO(dtype_is_supported,    \
            { std::cerr << "Unsupported dtype: " << \
                infiniDtypeToString(DT) << ". "; \
                return INFINI_STATUS_BAD_TENSOR_DTYPE; }); \
    } while (0)
```
- 使用初始化列表 `{__VA_ARGS__}` 构造支持类型数组
- 迭代检查是否匹配，失败时输出错误信息并返回错误码

2. **CHECK_SAME_SHAPE** (`check.h:76`):
```cpp
#define CHECK_SAME_SHAPE(FIRST, ...) \
    CHECK_SAME_VEC(INFINI_STATUS_BAD_TENSOR_SHAPE, FIRST, __VA_ARGS__)

#define CHECK_SAME_VEC(ERR, FIRST, ...)              \
    do {                                             \
        for (const auto &shape___ : {__VA_ARGS__}) { \
            if (FIRST != shape___) {                 \
                return ERR;                          \
            }                                        \
        }                                            \
    } while (0)
```
- 使用 `std::vector` 的 `operator==` 比较形状
- 所有输入必须与输出形状完全相同

3. **CHECK_RESULT** (`result.hpp:8-11`):
```cpp
#define CHECK_RESULT(RESULT)    \
    if (!RESULT) {              \
        return RESULT.status(); \
    }
```
- 利用 `Result<T>` 的 `operator bool()` 检查状态
- 失败时提取并返回错误码

### 5.6 内存管理策略

**智能指针模式**:
- `_device_info`: 使用 `std::unique_ptr` 管理设备实现
- 自动析构，防止内存泄漏
- 不可复制，仅可移动（通过 unique_ptr 语义）

**元数据存储**:
- `ElementwiseInfo` 使用单一 `std::vector<size_t>` 存储所有元数据
- 内存对齐：`CEIL_DIV(meta_mem_size, sizeof(size_t))` 确保按 size_t 对齐
- 零散数据紧凑排列，提升缓存命中率

**工作空间**:
- 加法操作无需额外工作空间，`_workspace_size = 0`
- `calculate()` 方法的 workspace 参数保留以兼容接口

### 5.7 性能优化技术

1. **分支预测友好**:
   - 连续性检查移到循环外
   - 使用 `if constexpr` 编译期消除分支

2. **缓存局部性**:
   - 元数据紧凑存储，减少缓存 miss
   - 线性扫描数据模式

3. **SIMD 就绪设计**:
   - 简单的逐元素操作易于向量化
   - 编译器可自动生成 SIMD 指令

4. **并行粒度控制**:
   - `if (output_size > 1024)` 条件并行
   - 避免小数据量的并行开销

### 5.8 设计模式应用

1. **策略模式 (Strategy Pattern)**:
   - `AddOp` 作为可替换的操作策略
   - 相同的计算框架支持不同逐元素操作（加、减、乘等）

2. **工厂模式 (Factory Pattern)**:
   - `Descriptor::create()` 静态工厂方法
   - 封装复杂的对象创建逻辑

3. **模板方法模式 (Template Method Pattern)**:
   - `ELEMENTWISE_DESCRIPTOR` 宏定义类模板
   - 具体操作（add、sub）填充特定逻辑

4. **RAII (Resource Acquisition Is Initialization)**:
   - 智能指针自动管理资源生命周期
   - 异常安全保证

### 5.9 依赖关系

**外部依赖**:
- `infinicore.h`: 核心类型定义（`infiniDtype_t`, `infiniStatus_t` 等）
- OpenMP: 多线程并行（可选，通过 `ENABLE_OMP` 宏控制）

**内部依赖**:
- `elementwise_cpu.h`: CPU 逐元素操作框架
- `elementwise.h`: 通用逐元素操作元数据
- `common_cpu.h`: CPU 通用工具（`indexToOffset` 等）
- `cpu_handle.h`: CPU 设备句柄定义
- `operator.h`: 基础操作描述符
- `tensor.h`: 张量描述符接口
- `utils/check.h`: 验证宏
- `utils/result.hpp`: 错误处理类型
- `utils/custom_types.h`: 自定义类型（`fp16_t`, `bf16_t`）

**编译条件**:
- 需要 C++14 或更高标准（`std::enable_if_t`, `if constexpr`）
- 支持的编译器：GCC 5+, Clang 3.8+, MSVC 2015+

## 6. 数据类型支持

| 枚举值 | 类型 | 字节数 | 说明 |
|--------|------|--------|------|
| `INFINI_DTYPE_F16` | `fp16_t` | 2 | 半精度浮点（IEEE 754） |
| `INFINI_DTYPE_F32` | `float` | 4 | 单精度浮点 |
| `INFINI_DTYPE_F64` | `double` | 8 | 双精度浮点 |
| `INFINI_DTYPE_BF16` | `bf16_t` | 2 | 脑浮点 16（Brain Float） |
| `INFINI_DTYPE_I32` | `int32_t` | 4 | 32 位整数 |
| `INFINI_DTYPE_I64` | `int64_t` | 8 | 64 位整数 |

## 7. 限制与约束

1. **形状要求**: 三个张量（两个输入、一个输出）必须具有完全相同的形状
2. **广播**: 本实现不支持广播，所有张量维度必须严格匹配
3. **连续性**: 输出张量不能有广播维度（`hasBroadcastDim()` 必须为 false）
4. **数据类型**: 输入和输出必须使用相同的数据类型
5. **线程安全**: `calculate()` 方法是线程安全的（只读操作），但同一个 `Descriptor` 实例不能被多线程同时调用

## 8. 扩展性

本实现展示了如何通过以下方式扩展到其他逐元素操作：

1. **定义新操作算子**:
```cpp
namespace op::mul::cpu {
typedef struct MulOp {
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        return a * b;
    }
} MulOp;
}
```

2. **使用宏生成描述符**:
```cpp
ELEMENTWISE_DESCRIPTOR(mul, cpu)
```

3. **实现 create 和 calculate**:
```cpp
infiniStatus_t Descriptor::create(...) {
    // 与 add_cpu.cc:7-30 相同逻辑
}

infiniStatus_t Descriptor::calculate(...) {
    switch (_dtype) {
    case INFINI_DTYPE_F32:
        return _device_info->calculate<MulOp, float>(...);
    // ... 其他类型
    }
}
```

这种设计使得添加新操作只需修改算子定义，无需重写框架代码。

---

**文档版本**: 1.0
**最后更新**: 2026-01-14
**分析深度**: 全量代码级分析
