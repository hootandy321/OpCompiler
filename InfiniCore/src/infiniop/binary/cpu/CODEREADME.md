# Binary CPU Operations Core Implementation Documentation

该模块实现了 CPU 后端的二元运算（Binary Operations）核心计算逻辑，支持张量广播、不同数据类型混合运算以及 OpenMP 并行加速。

## 1. Module Structure

- **`binary_cpu.h`**: CPU 后端二元运算的通用模板计算核心，提供高性能并行计算接口

## 2. Core Components

### `binary_op::calculate()` - 多类型版本
- **Location**: `binary_cpu.h:14-28`
- **Primary Function**: 当输入张量（a, b）和输出张量（c）具有不同数据类型时，执行通用的二元运算计算
- **Template Parameters**:
  - `Tc`: 输出张量的数据类型
  - `Ta`: 第一个输入张量的数据类型
  - `Tb`: 第二个输入张量的数据类型
  - `BinaryOp`: 二元运算符类型（仿函数/Functor）
  - `Args...`: 可变参数包，传递给 BinaryOp 的额外参数
- **Key Members**:
  - `info`: `op::binary::BinaryInfo` 结构体，包含张量形状、步长、连续性等元数据
  - `c, a, b`: 输出和两个输入张量的原始指针（void* 类型）
- **Core Algorithm**:
  1. 将 void* 指针转换为具体的类型指针 `Tc*`, `Ta*`, `Tb*`
  2. 使用 `#pragma omp parallel for` 启动 OpenMP 并行循环
  3. 对每个输出元素索引 `i`：
     - 根据 `info.contiguous` 标志决定使用线性索引还是计算偏移量
     - 如果非连续，调用 `op::common_cpu::indexToOffset()` 计算每个张量的实际内存偏移
     - 调用 `BinaryOp{}(a_[a_index], b_[b_index], args...)` 执行计算
     - 将结果写入 `c_[c_index]`
- **Performance**: OpenMP 并行化，时间复杂度 O(n)，n 为输出张量元素总数

### `binary_op::calculate()` - 单类型版本
- **Location**: `binary_cpu.h:32-52`
- **Primary Function**: 当所有张量（a, b, c）共享相同数据类型时执行优化的二元运算
- **Template Parameters**:
  - `Tdata`: 统一的数据类型（a, b, c 均使用此类型）
  - `BinaryOp`: 二元运算符类型
  - `Args...`: 传递给 BinaryOp 的额外参数
- **Key Optimization - fp16_t Specialization**:
  - 使用 `if constexpr (std::is_same_v<Tdata, fp16_t>)` 编译期类型判断
  - 对于 `fp16_t` 类型：
    - 将输入转换为 `float` 进行高精度计算
    - 调用 `BinaryOp` 在 float 域执行运算
    - 将结果转回 `fp16_t`
  - 对于其他类型：直接在原类型上执行运算，避免类型转换开销
- **Index Calculation Logic**:
  ```cpp
  size_t a_index = info.contiguous ?
      i :  // 连续内存：直接使用线性索引
      op::common_cpu::indexToOffset(i, info.ndim,
                                    info.a_shape.data(),
                                    info.a_strides.data());  // 非连续：计算多维偏移
  ```
- **Thread Safety**: OpenMP 并行循环，每个迭代独立无数据竞争

## 3. API Interface

```cpp
namespace op::common_cpu::binary_op {

// 多类型混合计算接口
template <typename Tc, typename Ta, typename Tb, typename BinaryOp, typename... Args>
void calculate(
    op::binary::BinaryInfo info,    // 张量元数据（形状、步长、连续性等）
    void *c,                         // 输出张量指针（类型 Tc*）
    const void *a,                   // 第一个输入张量指针（类型 Ta*）
    const void *b,                   // 第二个输入张量指针（类型 Tb*）
    Args &&...args                   // 转发给 BinaryOp 的额外参数
);

// 单类型优化计算接口
template <typename Tdata, typename BinaryOp, typename... Args>
void calculate(
    op::binary::BinaryInfo info,    // 张量元数据
    void *c,                         // 输出张量指针（类型 Tdata*）
    const void *a,                   // 第一个输入张量指针（类型 Tdata*）
    const void *b,                   // 第二个输入张量指针（类型 Tdata*）
    Args &&...args                   // 转发给 BinaryOp 的额外参数
);

} // namespace op::common_cpu::binary_op
```

### 辅助结构体：`op::binary::BinaryInfo`
```cpp
namespace op::binary {

struct BinaryInfo {
    size_t c_data_size;                  // 输出张量元素总数
    size_t ndim;                         // 张量维度数
    bool contiguous;                     // 所有张量是否内存连续
    bool broadcasted;                    // 是否涉及广播操作
    std::vector<size_t> c_shape;         // 输出张量形状
    std::vector<size_t> a_shape;         // 输入张量 A 形状
    std::vector<size_t> b_shape;         // 输入张量 B 形状
    std::vector<ptrdiff_t> c_strides;    // 输出张量步长（元素单位）
    std::vector<ptrdiff_t> a_strides;    // 输入张量 A 步长
    std::vector<ptrdiff_t> b_strides;    // 输入张量 B 步长
};

} // namespace op::binary
```

## 4. Usage Example

```cpp
#include "infiniop/binary/cpu/binary_cpu.h"
#include <algorithm>

// 示例1：相同类型的加法（float32）
struct AddOp {
    float operator()(float a, float b) const {
        return a + b;
    }
};

void float_addition_example() {
    // 准备输入输出数据
    constexpr size_t size = 1000;
    float a[size], b[size], c[size];

    // 填充测试数据
    for (size_t i = 0; i < size; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }

    // 准备元数据（假设连续内存）
    op::binary::BinaryInfo info;
    info.c_data_size = size;
    info.ndim = 1;
    info.contiguous = true;
    info.broadcasted = false;
    info.c_shape = {size};
    info.a_shape = {size};
    info.b_shape = {size};
    info.c_strides = {1};
    info.a_strides = {1};
    info.b_strides = {1};

    // 执行加法（OpenMP 并行）
    op::common_cpu::binary_op::calculate<float, AddOp>(
        info, c, a, b
    );

    // c 现在包含 a + b 的结果
}

// 示例2：混合类型计算（float32 + int32 -> float32）
struct MixedAddOp {
    float operator()(float a, int b, float scale) const {
        return a + b * scale;  // 带额外参数的运算
    }
};

void mixed_type_example() {
    float a[100];
    int b[100];
    float c[100];
    float scale = 0.5f;

    op::binary::BinaryInfo info;
    info.c_data_size = 100;
    info.ndim = 1;
    info.contiguous = true;
    // ... 其他元数据初始化 ...

    // 类型自动转换：int -> float
    op::common_cpu::binary_op::calculate<float, float, int, MixedAddOp>(
        info, c, a, b, scale  // scale 作为额外参数转发给 MixedAddOp
    );
}

// 示例3：广播计算（2D + 1D -> 2D）
void broadcast_example() {
    // 张量 A: shape [3, 4], 张量 B: shape [4] -> 广播到 [3, 4]
    float a[12];  // 3x4
    float b[4];   // 4 (广播到每一行)
    float c[12];  // 3x4

    op::binary::BinaryInfo info;
    info.c_data_size = 12;
    info.ndim = 2;
    info.contiguous = true;
    info.broadcasted = true;
    info.c_shape = {3, 4};
    info.a_shape = {3, 4};
    info.b_shape = {1, 4};  // 广播维度
    info.c_strides = {4, 1};
    info.a_strides = {4, 1};
    info.b_strides = {0, 1};  // stride=0 表示广播

    op::common_cpu::binary_op::calculate<float, AddOp>(
        info, c, a, b
    );
}
```

## 5. Implementation Details

### 内存管理 (Memory Management)
- **零拷贝设计**: 函数直接操作传入的原始指针（void*），不分配额外内存
- **类型安全**: 使用模板在编译期确保类型转换的正确性
- **指针转换**: 通过 `reinterpret_cast` 将 void* 转换为具体类型指针，无运行时开销

### 并发控制 (Concurrency)
- **并行策略**: 使用 OpenMP `#pragma omp parallel for` 实现数据并行
- **线程安全**: 每个循环迭代处理独立的输出元素，无共享可变状态
- **负载均衡**: OpenMP 运行时自动调度迭代到可用线程（通常使用 dynamic 或 static 调度）
- **无锁设计**: 不需要任何互斥锁或原子操作，完全无竞争

### 性能优化 (Performance)
1. **分支优化**:
   - 使用 `info.contiguous` 布尔标志在循环外判断，避免每次迭代的条件分支
   - 连续内存路径：直接使用线性索引 `i`
   - 非连续路径：调用 `indexToOffset()` 计算多维索引

2. **编译期优化**:
   - `if constexpr` 确保 fp16 特殊化代码在编译其他类型时完全消除
   - 模板内联：BinaryOp 调用会被编译器内联，消除函数调用开销
   - `std::forward<Args>(args)...` 完美转发，避免不必要的拷贝

3. **缓存友好性**:
   - 连续内存访问模式最大化空间局部性
   - 每个线程处理连续的索引范围（OpenMP 默认调度），提高缓存命中率

4. **浮点精度处理**:
   - fp16_t 自动提升为 float 进行计算，避免累积误差
   - 使用 `utils::cast` 进行类型转换（可能包含舍入策略）

### 广播机制 (Broadcasting)
- **检测**: `info.broadcasted` 标志由 `createBinaryInfo()` 在运行时计算
- **索引映射**: 通过 `indexToOffset()` 函数处理：
  - stride = 0 的维度表示广播（该维度索引映射到偏移量 0）
  - 例如：shape [1, 4], strides [0, 1] 表示沿第 0 维广播
- **限制**: 输出张量不能有广播维度（`!c_desc->hasBroadcastDim()`）

### 错误处理 (Error Handling)
- **无异常**: 模板函数不抛出异常，符合 C++ 性能关键代码的最佳实践
- **前置条件**: 调用者需确保：
  - 指针 `a, b, c` 有效且对齐
  - `info` 结构体正确填充（通过 `createBinaryInfo()` 生成）
  - 内存区域不重叠（或允许重叠时的正确处理）

### 设计模式 (Design Patterns)
- **策略模式 (Strategy Pattern)**: `BinaryOp` 模板参数允许用户自定义运算逻辑
- **模板方法模式 (Template Method)**: `calculate()` 定义算法骨架，BinaryOp 定义具体步骤
- **类型推导 (Type Erasure)**: 使用 void* 接口结合模板实现泛型，同时保持 C 兼容性
- **编译期多态 (Compile-time Polymorphism)**: 模板实例化生成高度优化的特化代码

### 依赖关系 (Dependencies)
- **必需依赖**:
  - `../../devices/cpu/common_cpu.h`: 提供 `indexToOffset()` 索引计算函数
  - `../binary.h`: 提供 `BinaryInfo` 结构体定义
  - `<utility>`: 提供 `std::forward` 用于完美转发
- **条件依赖**:
  - OpenMP (`<omp.h>`): 通过 `#ifdef ENABLE_OMP` 条件编译，未启用时退化到串行
- **类型依赖**:
  - `fp16_t`: 半精度浮点类型定义（可能在项目中自定义或使用第三方库）
  - `utils::cast<T>()`: 类型转换工具函数

### 算法复杂度 (Complexity)
- **时间复杂度**: O(n)，其中 n = `info.c_data_size`（输出元素数量）
- **空间复杂度**: O(1) 额外空间（不分配临时数组）
- **并行加速**: 理想情况下达到 O(n/p)，p 为 CPU 核心数（实际受内存带宽限制）

### 使用场景 (Use Cases)
- **逐元素运算**: Add, Sub, Mul, Div, Max, Min, Pow 等基础数学运算
- **逻辑运算**: And, Or, Xor, Equal, Greater, Less 等比较运算
- **混合精度计算**: 低精度输入 + 高精度累加（如 int8 -> float32）
- **深度学习层**: 实现二元运算层的 CPU 后端（如 Add 层、Mul 层）
- **广播机制**: 实现类似 NumPy 的广播语义
