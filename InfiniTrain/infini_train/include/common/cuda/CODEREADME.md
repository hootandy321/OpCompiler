# CUDA Common Utilities Core Implementation Documentation

本模块提供 InfiniTrain 框架的 CUDA 基础设施层，实现了统一的错误处理宏、类型转换系统和数学运算库，支持 fp16、bf16 等低精度计算类型，为上层训练内核提供高性能的 CUDA 原语。

## 1. Module Structure

- **`common_cuda.h`**: CUDA API 错误检查宏定义，封装 CUDA Runtime、CUDA Driver、CUBLAS 和 NCCL 的错误处理逻辑
- **`kernel_helper.cuh`**: CUDA 内核辅助函数库，提供类型安全的模板化数学运算、类型转换和原子操作优化

## 2. Core Classes

### Error Handling Macros (无状态宏定义)
- **Location**: `common_cuda.h`
- **Primary Function**: 提供 CUDA 相关 API 的统一错误检测和日志输出机制，在发生错误时通过 glog 记录详细的位置信息并终止程序
- **Key Members**:
  - `CUDA_CHECK`: 检查 `cudaError_t` 类型的 CUDA Runtime API 调用
  - `CUBLAS_CHECK`: 检查 `cublasStatus_t` 类型的 CUBLAS 库调用
  - `CUDA_DRIVER_CHECK`: 检查 `CUresult` 类型的 CUDA Driver API 调用
  - `NCCL_CHECK`: 条件编译宏，仅在 `USE_NCCL` 定义时检查 `ncclResult_t` 类型的 NCCL 调用
- **Core Methods**:
  - 所有宏均使用 `do { ... } while(0)` 惯用法确保作为单条语句使用时的安全性
  - 错误发生时调用 `LOG(FATAL)` 输出错误字符串、文件名和行号
- **Lifecycle**: 编译期宏展开，无运行时生命周期

### Type Cast Function
- **Location**: `kernel_helper.cuh`
- **Primary Function**: 提供类型安全、值类别保留的类型转换模板，专门优化 CUDA 低精度类型（half/bf16）与标准浮点类型的转换
- **Key Members**:
  - `template<DST, SRC>`: 源类型和目标类型的双参数模板推导
  - 使用 `std::remove_cv_t` 和 `std::remove_reference_t` 剥离 const/volatile 和引用修饰符
- **Core Methods**:
  - `Cast<SRC>(x)`: 接受转发引用参数，通过 `if constexpr` 在编译期选择最优转换路径
    - `nv_bfloat16 -> float`: 调用 `__bfloat162float` CUDA 内在函数
    - `half -> float`: 调用 `__half2float` CUDA 内在函数
    - `float -> half/bf16`: 调用 `__float2half`/`__float2bfloat16` 硬件加速转换
    - 其他类型: 使用 `static_cast` 进行标准 C++ 转换
  - 编译期断言禁止返回引用类型，确保返回值语义
- **Lifecycle**: 纯函数模板，无状态

### Math Operation Templates
- **Location**: `kernel_helper.cuh`
- **Primary Function**: 实现类型泛化的数学运算库，通过 `if constexpr` 为不同精度类型选择最优硬件指令
- **Core Methods**:
  - `Neg<T>(x)`: 取反运算，half/bf16 使用 `__hneg` 内在函数
  - `Reciprocal<T>(x)`: 计算倒数，half/bf16 使用 `__hdiv(1.0, x)` 避免类型转换
  - `Sin<T>`, `Cos<T>`: 三角函数，half/bf16 先转换到 float 计算，再转回原类型
  - `Tanh<T>(x)`: 双曲正切，half/bf16 使用 `htanh` 内在函数，float 使用 `tanhf`
  - `Pow<T>(x, exponent)`: 幂运算，包含 NaN 检查逻辑，当 `__powf` 返回 NaN 时回退到 `std::pow`
  - `Rsqrt<T>(x)`: 平方根倒数，float/half/bf16 使用 `rsqrtf` 硬件加速指令
  - `Exp<T>`, `Log<T>`: 指数和对数，float 使用 `__expf`/`__logf`，half/bf16 使用 `hexp` 或先转换后计算
  - `Add<T>`, `Sub<T>`, `Mul<T>`, `Div<T>`: 四则运算，half/bf16 使用 `__hadd`/`__hsub`/`__hmul`/`__hdiv`
  - `Sigmoid<T>(x)`: Sigmoid 激活函数，实现为 `1 / (1 + exp(-x))`
  - `Max<T>`, `Min<T>`: 极值运算，half/bf16 使用 `__hle` 比较指令选择分支
  - `Fma<T>(x, y, z)`: 融合乘加运算，half 使用 `__hfma`，bf16 先转换到 float 后使用 `__fmaf_rn`，float 使用 `__fmaf_rn`，double 使用 `std::fma`

### Fast Atomic Operations
- **Location**: `kernel_helper.cuh`
- **Primary Function**: 针对 half 和 bfloat16 类型的原子加法优化，利用向量化内存访问提升性能
- **Key Methods**:
  - `fastSpecializedAtomicAdd<__half>(tensor, index, num_elements, value)`:
    - 检查目标地址是否 2 字节对齐（`sizeof(__half2)` 对齐）
    - 如果对齐且不是最后一个元素：构造 `__half2` 向量 `(value, 0)` 并执行 `atomicAdd` 到 `__half2*`
    - 如果不对齐且不是第一个元素：构造 `__half2` 向量 `(0, value)` 并执行 `atomicAdd` 到前一个 `__half2*`
    - 边界情况回退到标量 `atomicAdd`
  - `fastSpecializedAtomicAdd<__nv_bfloat16>(tensor, index, num_elements, value)`:
    - 实现逻辑与 half 版本相同，但使用 `__nv_bfloat162` 向量化类型
  - `fastSpecializedAtomicAdd<其他类型>(tensor, index, num_elements, value)`:
    - 直接调用标准 `atomicAdd(tensor + index, value)`
  - `fastAtomicAdd(tensor, index, num_elements, value, fast_atomics)`:
    - 根据 `fast_atomics` 布尔标志选择优化路径或标准路径

## 3. API Interface

```cpp
// 错误检查宏 (必须作为独立语句使用)
CUDA_CHECK(cudaMalloc(&ptr, size));
CUBLAS_CHECK(cublasCreate(&handle));
CUDA_DRIVER_CHECK(cuCtxGetCurrent(&ctx));
NCCL_CHECK(ncclAllReduce(...));  // 仅在 USE_NCCL 定义时可用

// 类型转换函数
template <typename DST, typename SRC>
__host__ __device__ DST Cast(SRC&& x);
// 返回转换后的值，保留 const/volatile 修饰符，但不返回引用

// 数学运算函数 (仅设备代码)
template <typename T>
__device__ T Neg(const T& x);        // 取反
__device__ T Reciprocal(const T& x); // 倒数
__device__ T Sin(const T& x);        // 正弦
__device__ T Cos(const T& x);        // 余弦
__device__ T Tanh(const T& x);       // 双曲正切
__device__ T Pow(const T& x, const T& exponent);  // 幂运算
__device__ T Rsqrt(const T& x);      // 平方根倒数
__device__ T Exp(const T& x);        // 指数
__device__ T Log(const T& x);        // 自然对数
__device__ T Add(const T& a, const T& b);   // 加法
__device__ T Sub(const T& a, const T& b);   // 减法
__device__ T Mul(const T& a, const T& b);   // 乘法
__device__ T Div(const T& a, const T& b);   // 除法
__device__ T Sigmoid(const T& x);    // Sigmoid 激活
__device__ T Max(const T& a, const T& b);   // 最大值
__device__ T Min(const T& a, const T& b);   // 最小值
__device__ T Fma(const T& x, const T& y, const T& z);  // 融合乘加

// 原子加法优化 (仅设备代码)
template <typename scalar_t, typename index_t>
__device__ void fastAtomicAdd(scalar_t* tensor, index_t index,
                              const index_t num_elements,
                              scalar_t value, bool fast_atomics);
// 参数:
//   tensor - 目标张量指针
//   index - 写入索引
//   num_elements - 张量总元素数 (用于边界检查)
//   value - 要加的值
//   fast_atomics - 是否启用向量化优化
```

## 4. Usage Example

```cpp
#include "common/cuda/common_cuda.h"
#include "common/cuda/kernel_helper.cuh"

__global__ void sigmoid_kernel(__half* output, const __half* input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __half x = input[idx];
        // 使用类型安全的数学运算
        __half result = Sigmoid<__half>(x);
        output[idx] = result;
    }
}

__global__ void gradient_accumulation_kernel(__half* gradients,
                                             const __half* local_grad,
                                             int num_elements,
                                             bool fast_atomics) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        __half grad = local_grad[idx];
        // 使用优化的原子加法
        fastAtomicAdd(gradients, idx, num_elements, grad, fast_atomics);
    }
}

void launch_operations() {
    // 检查 CUDA 错误
    __half *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));

    // 启动内核
    sigmoid_kernel<<<blocks, threads>>>(d_output, d_input, n);
    CUDA_CHECK(cudaGetLastError());  // 检查内核启动错误

    // 梯度累加
    gradient_accumulation_kernel<<<blocks, threads>>>(gradients, local_grad, n, true);

    // 清理
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

// 主机端类型转换示例
void host_conversion_example() {
    float f = 3.14f;
    __half h = Cast<__half>(f);         // float -> half
    nv_bfloat16 bf16 = Cast<nv_bfloat16>(f);  // float -> bfloat16
    float f2 = Cast<float>(h);          // half -> float
}
```

## 5. Implementation Details

- **类型推导策略**: 使用 `std::remove_cv_t<std::remove_reference_t<SRC>>` 剥离类型修饰符，确保 `if constexpr` 分支匹配到基础类型，同时保留 `std::forward<SRC>(x)` 的值类别（左值/右值）转发特性

- **编译期分支选择**: 所有数学运算和类型转换均使用 `if constexpr` 在编译期生成针对特定类型的优化代码，避免运行时分支开销，支持 half/bf16/float/double 的任意组合

- **低精度类型优化**: 针对 `__half` 和 `__nv_bfloat16` 类型，优先使用 CUDA 内在函数（如 `__hadd`, `__hmul`, `__hfma`）直接操作硬件指令，避免转换到 float 再计算的开销

- **向量化原子操作**: 利用 `__half2` 和 `__nv_bfloat162` 类型（2 个 16 位浮点数的打包），通过地址对齐检查执行 32 位原子操作，相比 16 位标量原子操作可获得接近 2 倍性能提升，但需处理边界不对齐情况

- **NaN 处理**: `Pow` 函数在 `__powf` 返回 NaN 时回退到 `std::pow`，增强数值稳定性，适用于某些 `__powf` 硬件实现不覆盖的边界情况

- **融合乘加**: `Fma` 函数针对不同类型选择最优指令路径，half 使用 `__hfma` 单指令，bf16 先转换到 float 再使用 `__fmaf_rn`（round-to-nearest），float/double 使用标准 FMA 指令，确保精度和性能的最优平衡

- **错误处理机制**: 所有检查宏使用 `do { ... } while(0)` 包裹，确保在 `if-else` 语句中使用时的语法正确性，错误时通过 `LOG(FATAL)` 自动终止程序并记录完整的调用栈信息（文件名、行号、错误字符串）

- **依赖关系**: 依赖 CUDA Toolkit（`cuda_runtime.h`, `cuda_bf16.h`, `cuda_fp16.h`）、CUBLAS 库（`cublas_v2.h`）、可选的 NCCL 库（条件编译）、glog 日志库（`glog/logging.h`）

- **设计模式**: 纯函数式模板设计，无状态、无副作用，使用编译期多态（`if constexpr` + 模板特化）实现类型分发策略，所有函数标记为 `__device__` 或 `__host__ __device__` 以支持设备端和主机端调用
