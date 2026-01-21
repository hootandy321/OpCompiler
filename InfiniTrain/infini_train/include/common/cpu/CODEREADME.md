# `CPU Common Utilities` Core Implementation Documentation

这是一个轻量级的CPU通用工具模块，提供类型安全的高性能类型转换功能。该模块采用C++模板元编程技术，通过完美转发（perfect forwarding）机制实现类型转换，同时保留值的类别（左值/右值）和CV限定符（const/volatile），主要用于深度学习训练框架中CPU后端的数据类型转换场景。

## 1. Module Structure

- **`common_cpu.h`**: 提供CPU通用的类型安全转换模板函数，实现跨类型的值转换并保留值类别特性

## 2. Core Classes

### `Cast` Template Function
- **Location**: `common_cpu.h`
- **Primary Function**: 提供编译期类型安全检查的通用类型转换，支持任意类型间的值转换，同时通过完美转发保留输入值的值类别（lvalue/rvalue）和CV限定符
- **Template Parameters**:
  - `DST`: 目标类型（Destination Type），编译期自动推导
  - `SRC`: 源类型（Source Type），编译期自动推导，作为转发引用
- **Key Implementation Details**:
  - 使用 `static_assert` 在编译期禁止返回引用类型，确保返回值语义
  - 采用 `std::forward<SRC>(x)` 实现完美转发，保留原始值的值类别
  - 通过C风格的强制转换 `(DST)` 执行实际类型转换
- **Core Methods**:
  - `DST Cast(SRC&& x)`: 执行类型转换，时间复杂度 O(1)，编译期内联展开
    - **参数**: `x` - 转发引用，接受左值或右值输入，保留const/volatile限定符
    - **返回**: 转换后的 `DST` 类型值（非引用）
    - **编译期检查**: 如果 `DST` 是引用类型，触发静态断言失败
- **Lifecycle**: 无状态模板函数，编译期实例化，无运行时构造/析构开销
- **Known Limitations**:
  - 当前未实现CPU版本的 fp16 (半精度浮点) 和 bf16 (脑浮点) 类型支持（标记为TODO）

## 3. API Interface

```cpp
namespace infini_train::common::cpu {

// 通用类型转换函数
template <typename DST, typename SRC>
DST Cast(SRC &&x);
// 执行类型安全转换，保留值类别
// @tparam DST 目标类型（不可为引用类型）
// @tparam SRC 源类型（推导为转发引用）
// @param x 输入值（可以是左值或右值）
// @return 转换为DST类型的值
// @throw 编译期static_assert错误如果DST是引用类型

} // namespace infini_train::common::cpu
```

## 4. Usage Example

```cpp
#include "common/cpu/common_cpu.h"
using namespace infini_train::common::cpu;

// 基本类型转换
int int_val = 42;
float float_val = Cast<float>(int_val);  // int -> float
double double_val = Cast<double>(100);    // rvalue int -> double

// 保留值类别的转换
std::string str = "hello";
size_t len = Cast<size_t>(str.length());  // 从函数返回的临时值转换

// 错误示例：尝试返回引用类型（编译期失败）
// int& ref = Cast<int&>(int_val);  // static_assert失败！

// 常量性保留
const int cint = 10;
int copy = Cast<int>(cint);  // 正确：const int -> int

// 未来支持（当前TODO）
// fp16_t half = Cast<fp16_t>(float_val);  // 待实现
// bf16_t bfloat = Cast<bf16_t>(double_val);  // 待实现
```

## 5. Implementation Details

**核心设计决策**:

- **类型安全策略**: 使用 `static_assert(!std::is_reference_v<DST>)` 在编译期捕获引用类型返回的错误，防止悬空引用和未定义行为
- **完美转发机制**: 采用 `SRC&&` 转发引用配合 `std::forward`，精确保留调用者的值类别（lvalue/rvalue）和CV限定符，避免不必要的拷贝
- **转换语义**: 选择C风格强制转换 `(DST)` 而非 `static_cast/DST>`，因为该函数定位为"任意类型间的通用转换"，需要兼容内置类型和用户自定义类型的转换场景
- **零运行时开销**: 模板函数在编译期完全内联，无额外的函数调用开销，编译器可进行充分的优化
- **命名空间隔离**: 使用三层命名空间 `infini_train::common::cpu`，明确标识该模块属于训练框架的通用CPU工具库

**已知的扩展点**:
- TODO注释表明需要添加CPU端的 fp16 (float16) 和 bf16 (bfloat16) 类型支持，这些是深度学习中常用的低精度浮点格式
- 当前实现依赖C风格转换，未来可针对特定类型对提供特化版本（如 `Cast<__fp16, float>`）

**性能特性**:
- 时间复杂度: O(1)，单次类型转换操作
- 空间复杂度: 无额外内存分配
- 编译期优化: 完全内联，编译器可进行常量折叠和死代码消除

**使用场景**:
该模块主要服务于 InfiniTrain 框架中CPU后端的数据预处理、类型转换、张量操作等需要跨类型转换的场景，特别是在混合精度训练中需要在 float32、float16、bfloat16、int32 等类型间转换时提供统一的接口。
