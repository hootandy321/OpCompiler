# CPU Reduce Operations Core Implementation Documentation

该模块实现了CPU上的归约操作(reduction operations)，为张量计算提供基础的sum、max、sumSquared等数学运算，支持多种数据类型包括标准整数/浮点类型和半精度浮点类型(fp16/bf16)。该模块是Infini框架中高性能张量运算的基础组件。

## 1. Module Structure

- **`reduce.h`**: 头文件，定义了归约操作的模板函数接口，包含类型萃取器(type traits)和函数声明，使用SFINAE技术实现类型安全的模板重载
- **`reduce.cc`**: 实现文件，提供了半精度浮点类型(fp16_t和bf16_t)的具体实现，通过模板函数避免代码重复

## 2. Core Classes

### `ReduceToSame<T>` Type Trait
- **Location**: `reduce.h` (lines 16-27)
- **Primary Function**: 编译期类型检查，判断类型T是否可以直接进行归约操作而不需要类型转换
- **Type Definition**: `std::disjunction` 组合多个 `std::is_same` 特征
- **Supported Types**:
  - 浮点类型: `float`, `double`
  - 有符号整数: `int8_t`, `int16_t`, `int32_t`, `int64_t`
  - 无符号整数: `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`
- **Design Pattern**: 使用C++17的`std::disjunction`实现逻辑OR的类型特征组合，在编译期提供类型约束

### `sum()` Function Family
- **Location**: `reduce.h` (lines 29-40) and `reduce.cc` (lines 5-36)
- **Primary Function**: 对输入数据数组进行求和归约，支持步长(stride)访问
- **Template Version** (reduce.h, lines 30-37):
  - **Template Parameters**: `typename T` with constraint `ReduceToSame<T>::value`
  - **Function Signature**: `T sum(const T *data, size_t len, ptrdiff_t stride = 1)`
  - **Algorithm**: 简单线性累加，时间复杂度 O(n)，空间复杂度 O(1)
  - **Implementation**: 初始化result为0，遍历数组以stride为步长累加元素

- **Half-Precision Versions** (reduce.h lines 39-40, reduce.cc lines 34-36, 47-49):
  - **Function Signature**: `float sum(const fp16_t *data, size_t len, ptrdiff_t stride = 1)`
  - **Function Signature**: `float sum(const bf16_t *data, size_t len, ptrdiff_t stride = 1)`
  - **Implementation**: 通过`sum_half_impl<T>()`模板函数统一实现，先将半精度数据转换为float后累加
  - **Return Type**: `float` (防止fp16/bf16溢出，保持精度)
  - **Type Conversion**: 使用`utils::cast<float>()`进行类型转换

### `max()` Function Family
- **Location**: `reduce.h` (lines 42-54) and `reduce.cc` (lines 14-21, 38-40, 51-53)
- **Primary Function**: 对输入数据数组求最大值，支持步长访问
- **Template Version** (reduce.h, lines 43-50):
  - **Function Signature**: `T max(const T *data, size_t len, ptrdiff_t stride = 1)`
  - **Algorithm**: 初始化result为首元素，遍历比较更新最大值，时间复杂度 O(n)
  - **Implementation**: 使用`std::max()`进行逐元素比较

- **Half-Precision Versions** (reduce.cc):
  - **Function Signature**: `float max(const fp16_t *data, size_t len, ptrdiff_t stride = 1)`
  - **Function Signature**: `float max(const bf16_t *data, size_t len, ptrdiff_t stride = 1)`
  - **Implementation**: 通过`max_half_impl<T>()`模板函数实现，先转换为float再比较
  - **Return Type**: `float` (保持与sum函数一致的返回类型策略)

### `sumSquared()` Function Family
- **Location**: `reduce.h` (lines 55-68) and `reduce.cc` (lines 23-31, 42-44, 55-57)
- **Primary Function**: 对输入数据数组的每个元素平方后求和，常用于方差、L2范数计算
- **Template Version** (reduce.h, lines 56-64):
  - **Function Signature**: `T sumSquared(const T *data, size_t len, ptrdiff_t stride = 1)`
  - **Algorithm**: 遍历数组，对每个元素平方后累加，时间复杂度 O(n)
  - **Implementation**: `val * val` 计算平方，然后累加到result

- **Half-Precision Versions** (reduce.cc):
  - **Function Signature**: `float sumSquared(const fp16_t *data, size_t len, ptrdiff_t stride = 1)`
  - **Function Signature**: `float sumSquared(const bf16_t *data, size_t len, ptrdiff_t stride = 1)`
  - **Implementation**: 通过`sumSquared_half_impl<T>()`模板函数实现
  - **Precision Strategy**: 转换为float后再平方和累加，避免fp16/bf16精度损失

### `sum_half_impl<HalfType>()` Template
- **Location**: `reduce.cc` (lines 5-12)
- **Primary Function**: 半精度浮点类型的sum实现模板，避免fp16和bf16的代码重复
- **Template Parameters**: `typename HalfType` (可以是fp16_t或bf16_t)
- **Algorithm**: 先将每个元素转换为float，然后累加
- **Type Conversion**: 使用`utils::cast<float>(data[i * stride])`进行类型转换

### `max_half_impl<HalfType>()` Template
- **Location**: `reduce.cc` (lines 14-21)
- **Primary Function**: 半精度浮点类型的max实现模板
- **Algorithm**: 初始化为第一个元素的float转换值，然后遍历比较
- **Implementation**: `std::max(result, utils::cast<float>(data[i * stride]))`

### `sumSquared_half_impl<HalfType>()` Template
- **Location**: `reduce.cc` (lines 23-31)
- **Primary Function**: 半精度浮点类型的sumSquared实现模板
- **Algorithm**: 将元素转换为float，平方后累加
- **Implementation**: `float val = utils::cast<float>(data[i * stride]); result += val * val;`

## 3. API Interface

```cpp
namespace op::common_cpu::reduce_op {

// 类型萃取器：判断类型T是否可直接归约
template <typename T>
using ReduceToSame = std::disjunction<...>;

// 求和操作 - 标准类型版本
template <typename T, typename = std::enable_if_t<ReduceToSame<T>::value>>
T sum(const T *data, size_t len, ptrdiff_t stride = 1);
// 返回数组data的累加和，len为元素个数，stride为访问步长
// 时间复杂度: O(n), 空间复杂度: O(1)

// 求和操作 - 半精度浮点版本
float sum(const fp16_t *data, size_t len, ptrdiff_t stride = 1);
float sum(const bf16_t *data, size_t len, ptrdiff_t stride = 1);
// 返回float类型以避免溢出和保持精度

// 求最大值 - 标准类型版本
template <typename T, typename = std::enable_if_t<ReduceToSame<T>::value>>
T max(const T *data, size_t len, ptrdiff_t stride = 1);
// 返回数组中的最大值
// 前置条件: len > 0

// 求最大值 - 半精度浮点版本
float max(const fp16_t *data, size_t len, ptrdiff_t stride = 1);
float max(const bf16_t *data, size_t len, ptrdiff_t stride = 1);
// 返回float类型

// 平方和 - 标准类型版本
template <typename T, typename = std::enable_if_t<ReduceToSame<T>::value>>
T sumSquared(const T *data, size_t len, ptrdiff_t stride = 1);
// 返回 Σ(data[i]²)，用于方差和L2范数计算

// 平方和 - 半精度浮点版本
float sumSquared(const fp16_t *data, size_t len, ptrdiff_t stride = 1);
float sumSquared(const bf16_t *data, size_t len, ptrdiff_t stride = 1);
// 返回float类型，在float精度下计算避免精度损失

} // namespace op::common_cpu::reduce_op
```

## 4. Usage Example

```cpp
#include "infiniop/reduce/cpu/reduce.h"
#include <vector>
#include <iostream>

using namespace op::common_cpu::reduce_op;

int main() {
    // 示例1: 标准浮点数求和
    std::vector<float> float_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float sum_result = sum(float_data.data(), float_data.size());
    std::cout << "Sum: " << sum_result << std::endl;  // 输出: Sum: 15

    // 示例2: 整数求最大值
    std::vector<int32_t> int_data = {10, 5, 20, 15, 8};
    int32_t max_result = max(int_data.data(), int_data.size());
    std::cout << "Max: " << max_result << std::endl;  // 输出: Max: 20

    // 示例3: 带步长的平方和计算
    std::vector<double> double_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double sum_sq = sumSquared(double_data.data(), 3, 2);  // 计算索引0,2,4的平方和
    std::cout << "SumSquared with stride: " << sum_sq << std::endl;  // 输出: 35 (1+9+25)

    // 示例4: FP16半精度浮点求和
    std::vector<fp16_t> fp16_data = {
        {0x3c00},  // 1.0
        {0x4000},  // 2.0
        {0x4200}   // 3.0
    };
    float fp16_sum = sum(fp16_data.data(), fp16_data.size());
    std::cout << "FP16 Sum: " << fp16_sum << std::endl;  // 输出: FP16 Sum: 6

    // 示例5: BF16半精度浮点求最大值
    std::vector<bf16_t> bf16_data = {
        {0x3f80},  // 1.0
        {0x4000},  // 2.0
        {0x4080}   // 3.0
    };
    float bf16_max = max(bf16_data.data(), bf16_data.size());
    std::cout << "BF16 Max: " << bf16_max << std::endl;  // 输出: BF16 Max: 3

    // 示例6: 计算向量的L2范数 (平方根的平方和)
    std::vector<float> vector = {3.0f, 4.0f};
    float squared_norm = sumSquared(vector.data(), vector.size());
    float l2_norm = std::sqrt(squared_norm);
    std::cout << "L2 Norm: " << l2_norm << std::endl;  // 输出: L2 Norm: 5

    return 0;
}
```

## 5. Implementation Details

### Memory Management
- **Zero Allocation**: 所有函数均为纯计算，不进行任何动态内存分配
- **Pointer Based**: 使用原始指针`const T*`访问输入数据，避免智能指针开销
- **Stride Access**: 支持非连续内存访问模式，stride参数以`ptrdiff_t`类型传递，支持负步长

### Concurrency
- **OpenMP Support**: 头文件包含`#ifdef ENABLE_OMP`和`#include <omp.h>`预处理指令，表明该模块支持OpenMP并行化
- **Current Implementation**: 当前实现为串行版本，未显式使用OpenMP指令
- **Parallelization Potential**: 可以通过添加`#pragma omp parallel for reduction(+:result)`实现并行化

### Performance
- **Algorithm Complexity**: 所有操作均为O(n)线性复杂度，n为数组长度
- **Loop Unrolling**: 编译器可通过优化自动进行循环展开
- **Cache Friendliness**: 顺序访问模式对CPU缓存友好，stride=1时性能最优
- **Type Conversion Cost**: 半精度浮点版本每次迭代都需要类型转换(fp16->float)，存在性能开销
- **Branch Prediction**: 简单的线性循环，分支预测友好

### Error Handling
- **No Exception**: 所有函数均不抛出异常，遵循C++标准库的数值函数设计
- **Undefined Behavior**:
  - `max()`函数在len=0时访问data[0]导致未定义行为
  - 传入nullptr指针会导致段错误
- **Type Safety**: 使用SFINAE和`std::enable_if`在编译期确保类型安全，不支持类型会产生编译错误

### Dependencies
- **Internal Dependencies**:
  - `../../../utils.h`: 提供通用的工具函数和类型定义
  - `utils/custom_types.h`: 定义fp16_t和bf16_t类型及类型转换函数
  - `utils::cast<T>()`: 编译期类型转换模板函数
- **External Dependencies**:
  - `<cstddef>`: 提供size_t和ptrdiff_t类型
  - `<type_traits>`: 提供std::disjunction, std::is_same, std::enable_if等类型特征
  - `<algorithm>` (隐式): 提供std::max函数
- **Conditional Compilation**: ENABLE_OMP宏控制是否包含OpenMP头文件

### Design Patterns

1. **Template Specialization with SFINAE**:
   - 使用`std::enable_if_t<ReduceToSame<T>::value>`约束模板函数
   - 为标准类型提供通用模板实现
   - 为fp16_t和bf16_t提供显式特化版本

2. **Type Traits**:
   - `ReduceToSame<T>`类型萃取器封装了"是否支持直接归约"的类型判断逻辑
   - 使用`std::disjunction`组合多个类型特征，提高代码可维护性

3. **Code Reuse via Template Helper Functions**:
   - `sum_half_impl<T>()`, `max_half_impl<T>()`, `sumSquared_half_impl<T>()`避免fp16和bf16的代码重复
   - 编译器会为每种半精度类型生成独立的实例化代码

4. **Namespace Encapsulation**:
   - `op::common_cpu::reduce_op`命名空间隔离归约操作
   - 避免与其他模块的符号冲突

5. **Const Correctness**:
   - 所有输入参数均为`const T*`，确保不修改输入数据
   - 函数副作用(side effect)最小化

### Numeric Precision Considerations

- **Floating Point Accumulation**: 使用简单的顺序累加，可能产生舍入误差
  - 对于大数组，误差累积可能显著
  - 未使用Kahan求和算法或其他高精度累加方法

- **Half-Precision Strategy**:
  - fp16/bf16的归约结果转换为float返回
  - 防止fp16(5位指数，10位尾数)和bf16(8位指数，7位尾数)的范围限制
  - 在float精度下进行计算保持中间结果的准确性

- **Integer Overflow**:
  - 模板版本对整数类型使用T类型存储结果
  - 大数组求和可能导致整数溢出，未提供溢出保护

### Compiler Optimization Opportunities

- **Inline Expansion**: 所有函数定义为模板或在头文件中，有利于内联展开
- **SIMD Vectorization**: 编译器可自动向量化，特别是stride=1的情况
- **Loop Unrolling**: 编译器可根据-O2/-O3优化级别自动展开循环
- **Constant Propagation**: stride参数若为编译期常量，可进一步优化

### Extension Points

- **Additional Reduction Operations**: 可扩展min、mean、prod、argmax等操作
- **Parallel Implementation**: 使用OpenMP或std::thread并行化
- **GPU Backend**: 可实现CUDA/HIP版本的归约kernel
- **Higher Precision Accumulation**: 为整数类型提供double precision累加选项
