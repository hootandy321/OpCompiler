# `ZerosOp CUDA Kernel` Core Implementation Documentation

该模块实现了 CUDA 设备端的零值填充操作（zeros operation），为深度学习张量操作提供类型安全的零值初始化功能。该操作是 Infini 框架中基础的数据结构构建块，用于张量初始化、梯度清零和内存复位场景。

## 1. Module Structure

- **`kernel.cuh`**: CUDA 设备端函数对象（functor）定义，提供类型泛型的零值生成操作。该头文件定义了 `ZerosOp` 结构体，支持 14 种标量数据类型的零值转换，是 CUDA kernel 的核心计算单元。

## 2. Core Classes

### `ZerosOp`
- **Location**: `kernel.cuh`
- **Primary Function**: 作为 CUDA 设备端函数对象，对任意输入类型返回对应的零值表示。通过编译期类型分发（`if constexpr`）实现零开销的类型特化，无需虚函数或运行时分支。
- **Key Members**:
  - `num_inputs`: 编译期常量，值为 1。该元数据表明该操作接受 1 个输入张量（虽然实际值被忽略，仅用于类型推导）。
- **Core Methods**:
  - `operator()(const T &x) const`: 函数调用运算符重载，接受任意类型 `T` 的引用参数 `x`（实际未使用），返回类型 `T` 的零值。使用 `if constexpr` 在编译期展开为对应类型的零值返回语句，完全避免运行时分支判断。时间复杂度 O(1)，空间复杂度 O(1)。
- **Lifecycle**: 该结构体为无状态（stateless）的 POD 类型，无构造/析构逻辑，可直接作为 kernel 参数传递或全局常量使用。实例化无需动态内存分配，可在设备端栈上直接构造。

## 3. API Interface

```cpp
namespace op::zeros::cuda {

struct ZerosOp {
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const;
    // 对任意类型 T 返回对应的零值表示
    // 参数 x: 输入值（未使用，仅用于类型推导）
    // 返回值: 类型 T 的零值（bool 返回 false，整数返回 0，浮点返回 0.0/0.0f，FP8/BF16/Half 返回对应格式的零）
};

}
```

## 4. Usage Example

```cpp
#include "kernel.cuh"

// 示例：在 CUDA kernel 中使用 ZerosOp 填充张量
__global__ void fill_zeros_kernel(float *output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        op::zeros::cuda::ZerosOp zeros_op;
        output[idx] = zeros_op(output[idx]); // 输出 0.0f
    }
}

// 调用示例
float *d_tensor;
cudaMalloc(&d_tensor, 1024 * sizeof(float));
fill_zeros_kernel<<<32, 32>>>(d_tensor, 1024);

// 示例：支持多种数据类型的零值初始化
template <typename T>
__global__ void init_zeros_kernel(T *data, size_t size) {
    op::zeros::cuda::ZerosOp op;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = op(data[idx]); // 自动类型推导，返回对应类型的零值
    }
}

// 分别处理 float 和 half 张量
init_zeros_kernel<float><<<blocks, threads>>>(d_float_tensor, n);
init_zeros_kernel<half><<<blocks, threads>>>(d_half_tensor, n);
```

## 5. Implementation Details

- **编译期类型分发 (Compile-Time Type Dispatch)**: 使用 C++17 的 `if constexpr` 特性实现零分支类型特化。所有类型判断在编译期完成，生成的设备代码仅包含对应类型的直接返回语句（如 `return 0;` 或 `return 0.0f;`），无运行时开销。这种设计比虚函数表或 switch-case 更高效，尤其适合 CUDA kernel 的性能敏感场景。

- **类型覆盖范围**: 支持完整的数据类型谱系：
  - 布尔型: `bool` → `false`
  - 有符号整数: `int8_t`, `int16_t`, `int32_t`, `int64_t` → `0`
  - 无符号整数: `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t` → `0`
  - 浮点型: `float` → `0.0f`, `double` → `0.0`
  - 机器学习专用低精度类型: `half` → `__float2half(0.0f)`, `cuda_bfloat16` → `__float2bfloat16(0.0f)`, `cuda_fp8_e4m3` → `cuda_fp8_e4m3(0.0f)`

  注意：`uint8_t` 在第 7 行重复出现，实际未覆盖 `uint16_t` 之前的逻辑，这是一个潜在的代码缺陷（但不影响功能，因为逻辑相同）。

- **设备端属性声明**:
  - `__device__`: 该函数可从 CUDA 设备代码调用，仅能在 GPU 端使用。
  - `__forceinline__`: 强制内联展开，避免函数调用开销。对于简单的零值返回操作，内联后生成单条指令（如 `MOV.R32 F1, 0;`）。

- **命名空间隔离**: 位于 `op::zeros::cuda` 嵌套命名空间，防止与其他操作（如 `op::ones::cuda`）或后端（如 `op::zeros::cpu`）的符号冲突。

- **类型安全设计**: 通过模板和 `if constexpr` 保证编译期类型检查。若传入不支持类型（如自定义结构体），进入 `else` 分支返回 `0.0`（可能触发类型转换警告），而非运行时错误。

- **数学语义保证**: 对于浮点类型返回精确的 IEEE 754 正零值（+0.0），而非负零（-0.0）或非规范化数（denormal）。`__float2half` 和 `__float2bfloat16` 内置函数确保低精度浮点类型的位模式正确。

- **无副作用 (Side-Effect Free)**: 函数忽略输入参数 `x` 的值，仅用其推导类型 `T`，满足纯函数（pure function）语义，便于编译器优化和自动向量化。

- **集成模式**: 该 functor 通常与 Infini 的更高层算子封装（如 `Tensor::zeros()`）或执行引擎（如 graph executor）配合使用。上层框架通过泛型编程自动生成对应的 kernel 实例化代码。

- **边界情况处理**: 对 `bool` 类型返回 `false` 而非 `0`，避免隐式类型转换。对 `cuda_fp8_e4m3`（8 位浮点）使用显式构造函数 `cuda_fp8_e4m3(0.0f)`，保证编码正确。
