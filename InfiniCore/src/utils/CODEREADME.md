# InfiniCore Utils 模块核心实现文档

InfiniCore Utils 模块是整个框架的基础设施层，提供类型安全的错误处理机制、自定义浮点格式转换（FP16/BF16）、高效的多维张量重排算法，以及编译时断言宏系统。该模块专注于底层工具函数，为上层的张量计算和算子实现提供支撑。

## 1. 模块结构

- **`check.h`**: 编译时宏断言系统，提供条件检查、API调用验证、数据类型验证和形状一致性检查
- **`custom_types.h`** / **`custom_types.cc`**: 自定义浮点类型定义与转换实现，支持 FP16 和 BF16 与 Float32 的双向转换
- **`infini_status_string.h`**: 状态码到可读字符串的映射表，用于错误信息输出
- **`rearrange.h`** / **`rearrange.cc`**: 多维张量重排算法实现，支持跨步张量的高效内存布局转换
- **`result.hpp`**: 类型安全的 Result<T> 模板类，封装返回值或错误状态

## 2. 核心类与组件

### `Result<T>`
- **Location**: `result.hpp`
- **Primary Function**: 提供类型安全的错误处理机制，封装操作成功返回的值或失败时的错误状态码
- **Key Members**:
  - `std::variant<infiniStatus_t, T> _result`: 使用 std::variant 存储错误状态或成功值
- **Core Methods**:
  - `explicit Result(T value)`: 构造成功结果，移动语义传入值
  - `Result(infiniStatus_t status)`: 构造失败结果，状态码不能为 SUCCESS
  - `infiniStatus_t status() const`: 返回操作状态，成功返回 INFINI_STATUS_SUCCESS
  - `T take()`: 提取并移动内部值，需确保 status() 为 SUCCESS
  - `operator bool() const`: 布尔转换，成功返回 true
  - `T* operator->()` / `T& operator*()`: 指针/解引用操作符，像智能指针一样访问值
- **Lifecycle**: 值语义，支持移动语义。构造时根据参数类型决定持有错误码或值
- **Design Pattern**: Sum Type（类似 Rust 的 Result<T, E>），使用 std::variant 实现类型安全的 Either 模式

### `RearrangeMeta`
- **Location**: `rearrange.h` / `rearrange.cc`
- **Primary Function**: 编译期优化多维张量重排的元数据，生成高效的索引计算和内存复制方案
- **Key Members**:
  - `std::vector<ptrdiff_t> _meta`: 紧凑的元数据数组，格式为 [unit, idx_stride_0..n, dst_stride_0..n, src_stride_0..n]
    - `unit`: 连续内存块大小（字节数）
    - `idx_stride`: 索引计算的累积步长
    - `dst_stride` / `src_stride`: 目标和源内存的字节步长
- **Core Methods**:
  - `static Result<RearrangeMeta> create(...)`: 根据张量形状和步长生成优化的重排元数据
    - **算法流程**：
      1. 过滤长度为 1 的维度
      2. 按目标步长绝对值降序排序（优化缓存局部性）
      3. 合并连续维度（减少循环嵌套）
      4. 递归计算索引步长（用于多维索引转换）
    - **时间复杂度**: O(ndim * log(ndim))，主要来自排序
  - `void launch(void *dst, const void *src) const`: 执行重排操作
    - **算法**：OpenMP 并行化的多维索引计算，每个线程处理一部分元素
    - **索引计算**：根据 idx_strides 将线性索引分解为多维坐标，再计算 dst/src 的字节偏移
    - **内存复制**：对于连续的 unit 使用 memcpy，否则逐元素复制
  - `Result<RearrangeMeta> distributeUnit(const std::vector<size_t>&) const`: 拆分 unit 以增加并行度
    - **用途**：当 unit 过大导致并行度不足时，将其拆分为更小的连续块
    - **算法**：选择能整除当前 unit 的最大候选值，插入新的维度
- **Accessors**:
  - `size_t ndim() const`: 返回有效维度数
  - `size_t unit() const`: 返回连续内存块大小
  - `size_t count() const`: 返回总块数（元素数 / unit）
  - `const ptrdiff_t* idx_strides()`: 索引步长数组
  - `const ptrdiff_t* dst_strides()`: 目标步长数组
  - `const ptrdiff_t* src_strides()`: 源步长数组
- **Implementation Details**:
  - 使用 `do { ... } while(0)` 宏包装确保单语句语义
  - OpenMP 并行仅在 count > 1 时启用，避免小数据开销
  - 维度合并条件：`b.dst * len == f.dst && b.src * len == f.src`（连续且比例一致）

### `fp16_t` / `bf16_t`
- **Location**: `custom_types.h`
- **Primary Function**: 定义半精度浮点和 Brain 浮点的存储类型，提供类型安全的转换接口
- **Key Members**:
  - `uint16_t _v`: 底层 16 位整数存储（避免未定义的类型双关）
- **Conversion Functions** (位于 `custom_types.cc`):
  - `float _f16_to_f32(fp16_t)`: FP16 → FP32 转换
    - **算法**：位级解析 IEEE 754 格式（1 符号位 + 5 指数位 + 10 尾数位）
    - 特殊值处理：无穷大（指数 31，尾数 0）、NaN（指数 31，尾数非 0）、非规格化数（指数 0，尾数非 0）
    - 指数偏置调整：FP16 偏置 15，FP32 偏置 127，需加 112
  - `fp16_t _f32_to_f16(float)`: FP32 → FP16 转换
    - **算法**：提取 FP32 的符号、指数、尾数，根据范围决定处理策略
    - 溢出处理：指数 >= 16 返回无穷大，指数 < -24 返回有符号零
    - 非规格化处理：指数在 [-24, -14) 时，添加隐含的前导 1 并右移尾数
  - `float _bf16_to_f32(bf16_t)`: BF16 → FP32 转换
    - **算法**：直接左移 16 位（BF16 与 FP32 共享指数偏置和格式，仅尾数截断）
  - `bf16_t _f32_to_bf16(float)`: FP32 → BF16 转换
    - **算法**：Round-to-nearest-even 舍入策略
    - **舍入偏置**：`0x7FFF + ((bits32 >> 16) & 1)`（根据最低有效位决定向上/向下舍入）
- **Design Pattern**: 类型安全的包装器，使用 memcpy 避免严格别名规则违规

### `utils::cast<TypeTo, TypeFrom>`
- **Location**: `custom_types.h`
- **Primary Function**: 编译期类型转换模板，支持自定义浮点类型的自动双向转换
- **Core Logic**:
  - 使用 `if constexpr` 在编译期选择转换路径
  - 转换规则：
    1. 同类型直接返回
    2. FP16 ↔ Float32: 调用专用转换函数
    3. FP16 ↔ 其他类型: 先转 Float32，再转目标类型（通过 float 中间表示）
    4. BF16 ↔ Float32: 调用专用转换函数
    5. BF16 ↔ 其他类型: 先转 Float32，再转目标类型
    6. 其他类型: 使用 static_cast
- **Complexity**: O(1) 编译期计算，零运行时开销

## 3. API 接口

### 错误检查宏 (check.h)

```cpp
// 通用条件检查，失败时执行自定义动作
CHECK_OR_DO(CONDITION, ACTION)
// 示例：CHECK_OR_DO(ptr != nullptr, { return ERROR_NULL_POINTER; });

// 条件检查，失败时返回错误码
CHECK_OR_RETURN(CONDITION, ERROR)
// 示例：CHECK_OR_RETURN(size > 0, INFINI_STATUS_BAD_PARAM);

// API 调用检查，验证返回值是否符合预期
CHECK_API_OR(API, EXPECT, ACTION)
// 示例：CHECK_API_OR(cudaMalloc(&ptr, size), cudaSuccess, { return ERROR_OOM; });

// 内部 API 检查，失败返回内部错误
CHECK_INTERNAL(API, EXPECT)
// 示例：CHECK_INTERNAL(op_status, SUCCESS);

// 状态码检查，失败时打印错误信息并返回
CHECK_STATUS(API)
// 示例：CHECK_STATUS(infiniOpExecute(op));

// 数据类型检查，验证是否在支持列表中
CHECK_DTYPE(DT, ...)
// 示例：CHECK_DTYPE(dtype, INFINI_DTYPE_F32, INFINI_DTYPE_F16);

// 整数类型检查（快捷方式）
CHECK_DTYPE_ANY_INT(DT)
// 示例：CHECK_DTYPE_ANY_INT(tensor->dtype());

// 形状一致性检查
CHECK_SAME_SHAPE(FIRST, ...)
// 示例：CHECK_SAME_SHAPE(input->shape(), output->shape(), weight->shape());

// 步长一致性检查
CHECK_SAME_STRIDES(FIRST, ...)
// 示例：CHECK_SAME_STRIDES(src_strides, dst_strides);
```

### 张量重排接口 (rearrange.h)

```cpp
namespace utils {

// 创建重排元数据（推荐使用，性能最优）
Result<RearrangeMeta> RearrangeMeta::create(
    const size_t *shape,        // 张量形状数组 [dim0, dim1, ..., dimN]
    const ptrdiff_t *dst_strides, // 目标步长（字节单位）
    const ptrdiff_t *src_strides, // 源步长（字节单位）
    size_t ndim,                // 维度数量
    size_t element_size         // 单元素大小（字节）
);
// 返回：成功时包含 RearrangeMeta 对象，失败时包含错误状态码

// 一站式重排函数（内部调用 create + launch）
void rearrange(
    void *dst,                  // 目标内存指针
    const void *src,            // 源内存指针
    const size_t *shape,        // 张量形状
    const ptrdiff_t *dst_strides, // 目标步长
    const ptrdiff_t *src_strides, // 源步长
    size_t ndim,                // 维度数量
    size_t element_size         // 元素大小
);

} // namespace utils
```

### 类型转换接口 (custom_types.h)

```cpp
// FP16/BF16 转换函数（C 风格）
float _f16_to_f32(fp16_t val);   // FP16 → Float32
fp16_t _f32_to_f16(float val);   // Float32 → FP16
float _bf16_to_f32(bf16_t val);  // BF16 → Float32
bf16_t _f32_to_bf16(float val);  // Float32 → BF16

// 通用类型转换模板（C++ 风格）
namespace utils {
template <typename TypeTo, typename TypeFrom>
TypeTo cast(TypeFrom val);  // 自动选择最优转换路径
}

// 使用示例：
float f32_value = 3.14f;
fp16_t f16_value = utils::cast<fp16_t>(f32_value);  // Float32 → FP16
int int_value = utils::cast<int>(f16_value);        // FP16 → Float32 → int
```

### 结果类型接口 (result.hpp)

```cpp
namespace utils {

template <typename T>
class Result {
public:
    // 构造成功结果
    explicit Result(T value);

    // 构造失败结果（status 不能是 SUCCESS）
    Result(infiniStatus_t status);

    // 获取状态码
    infiniStatus_t status() const;

    // 提取值（移动语义）
    T take();

    // 布尔转换（成功返回 true）
    explicit operator bool() const;

    // 智能指针风格访问
    T* operator->();
    const T* operator->() const;
    T& operator*();
    const T& operator&() const;
};

// 辅助宏：检查 Result 并在失败时提前返回
#define CHECK_RESULT(RESULT) \
    if (!RESULT) { \
        return RESULT.status(); \
    }

} // namespace utils
```

## 4. 使用示例

### 示例 1：张量转置（2D 矩阵）

```cpp
#include "rearrange.h"
#include "check.h"

// 将 3x4 矩阵转置为 4x3（行优先存储）
void transpose_matrix(float *dst, const float *src, size_t rows, size_t cols) {
    size_t shape[] = {rows, cols};
    ptrdiff_t src_strides[] = {cols * sizeof(float), sizeof(float)};  // [row_stride, col_stride]
    ptrdiff_t dst_strides[] = {rows * sizeof(float), sizeof(float)};  // 交换了行列

    auto scheme = utils::RearrangeMeta::create(
        shape, dst_strides, src_strides, 2, sizeof(float)
   );
    CHECK_RESULT(scheme);

    scheme->launch(dst, src);
}

// 调用示例：
float matrix[3][4] = {{1,2,3,4}, {5,6,7,8}, {9,10,11,12}};
float transposed[4][3];
transpose_matrix(&transposed[0][0], &matrix[0][0], 3, 4);
// 结果：transposed = {{1,5,9}, {2,6,10}, {3,7,11}, {4,8,12}}
```

### 示例 2：FP16 量化与计算

```cpp
#include "custom_types.h"
#include "check.h"

infiniStatus_t process_fp16_tensor(const float *input, fp16_t *output, size_t size) {
    // 检查输入指针
    CHECK_OR_RETURN(input != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(output != nullptr, INFINI_STATUS_NULL_POINTER);

    // 转换并限制范围
    for (size_t i = 0; i < size; ++i) {
        float clamped = std::max(-65504.0f, std::min(65504.0f, input[i]));
        output[i] = _f32_to_f16(clamped);
    }

    return INFINI_STATUS_SUCCESS;
}

// 调用示例：
float float_data[] = {1.0f, -2.5f, 3.14f};
fp16_t fp16_data[3];
auto status = process_fp16_tensor(float_data, fp16_data, 3);
CHECK_STATUS(status);
```

### 示例 3：使用 Result<T> 的错误处理

```cpp
#include "result.hpp"
#include "rearrange.h"

utils::Result<RearrangeMeta> create_rearrange_scheme(
    const std::vector<size_t> &shape,
    const std::vector<ptrdiff_t> &dst_strides,
    const std::vector<ptrdiff_t> &src_strides,
    size_t element_size) {

    // 参数验证
    if (shape.size() != dst_strides.size() || shape.size() != src_strides.size()) {
        return INFINI_STATUS_BAD_PARAM;
    }

    // 创建元数据
    auto scheme = utils::RearrangeMeta::create(
        shape.data(), dst_strides.data(), src_strides.data(),
        shape.size(), element_size
    );

    // 直接返回 Result（可能成功或失败）
    return scheme;
}

// 调用示例：
auto result = create_rearrange_scheme({2, 3}, {12, 4}, {4, 1}, sizeof(float));
if (!result) {
    std::cerr << "Error: " << infini_status_string(result.status()) << std::endl;
    return;
}
// 使用 scheme
result->launch(dst_ptr, src_ptr);
```

### 示例 4：增加重排并行度

```cpp
#include "rearrange.h"

void parallel_rearrange(float *dst, const float *src, size_t n) {
    size_t shape[] = {n};
    ptrdiff_t strides[] = {sizeof(float)};

    auto scheme = utils::RearrangeMeta::create(shape, strides, strides, 1, sizeof(float));
    CHECK_RESULT(scheme);

    // 如果 unit 太大（导致并行度不足），拆分为 64 字节的块
    if (scheme->unit() > 64) {
        auto distributed = scheme->distributeUnit({64, 32, 16, 8, 4, 2, 1});
        CHECK_RESULT(distributed);
        distributed->launch(dst, src);
    } else {
        scheme->launch(dst, src);
    }
}
```

## 5. 实现细节

### 内存管理
- **零拷贝设计**：所有接口使用原始指针，调用者负责内存分配和释放
- **Result<T> 移动语义**：`take()` 方法移动内部值，避免不必要的拷贝
- **RearrangeMeta 紧凑存储**：单个 vector 存储所有元数据，减少内存分配次数

### 并发与并行
- **OpenMP 并行化**：`RearrangeMeta::launch()` 使用 `#pragma omp parallel for` 并行化元素级循环
- **无锁设计**：所有函数都是线程安全的（只读操作或独立内存区域）
- **并行度优化**：`distributeUnit()` 可将大 unit 拆分为小块，增加线程利用率

### 性能优化
- **编译期计算**：`utils::cast` 使用 `if constexpr` 完全展开为直接指令，无运行时分支
- **维度合并**：`RearrangeMeta::create()` 自动合并连续维度，减少循环嵌套层级
- **缓存友好**：按步长绝对值降序排序维度，优先处理大跨度（外层循环）
- **批量复制**：连续内存块使用 `memcpy`（比逐元素复制快 10-100 倍）
- **位操作转换**：FP16/BF16 转换使用位操作而非浮点运算，避免精度损失和性能开销

### 错误处理
- **宏驱动断言**：所有检查宏在失败时打印文件名、行号、函数名和条件表达式
- **状态码字符串化**：`infini_status_string()` 将错误码转换为可读消息
- **Result<T> 类型安全**：强制检查返回值，编译期防止忽略错误
- **自定义动作支持**：`CHECK_OR_DO` 允许失败时执行任意清理代码

### 依赖关系
- **外部依赖**：
  - `<infinicore.h>`: 提供 `infiniStatus_t` 枚举和数据类型常量（INFINI_DTYPE_*）
  - OpenMP (可选): 定义 `ENABLE_OMP` 时启用并行化
- **内部依赖**：
  - `check.h` 被 `result.hpp` 和 `rearrange.cc` 依赖
  - `result.hpp` 被 `rearrange.h` 依赖
  - `infini_status_string.h` 被 `check.h` 依赖

### 设计模式
- **RAII（Resource Acquisition Is Initialization）**: Result<T> 管理值或错误的生命周期
- **Sum Type**: Result<T> 使用 std::variant 实现 Either 模式（值或错误，二选一）
- **Strategy Pattern**: `utils::cast` 根据类型组合选择不同的转换策略
- **Builder Pattern**: `RearrangeMeta::create()` 从参数构建优化的执行方案
- **Template Method**: `CHECK_*` 宏系列提供统一的检查框架，自定义失败处理
