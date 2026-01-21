# NVIDIA/CUDA 设备后端核心实现文档

本模块为 InfiniOp 提供 NVIDIA GPU 及其兼容硬件（Iluvatar、QY、Hygon）的统一 CUDA 设备后端实现。模块封装了 CUBLAS、CUDNN 库的句柄管理、设备属性查询、类型映射以及 CUDA 内核开发的通用基础设施。

## 1. 模块结构

- **`nvidia_handle.h`**: 定义 Handle 类层次结构，为 NVIDIA、Iluvatar、QY、Hygon 四种设备提供统一接口
- **`nvidia_handle.cuh`**: Handle::Internal 实现，管理 CUBLAS/CUDNN 句柄池和设备属性
- **`nvidia_common.cu`**: Handle 类构造函数、句柄管理方法实现、类型转换函数
- **`nvidia_common.cuh`**: 导出类型转换接口（getCudnnDtype）
- **`nvidia_kernel_common.cuh`**: CUDA 内核开发工具宏、类型定义、数学函数、索引计算

## 2. 核心类

### `device::nvidia::Handle`
- **位置**: `nvidia_handle.h`, `nvidia_handle.cuh`, `nvidia_common.cu`
- **主要功能**: NVIDIA GPU 设备句柄，管理设备资源和库句柄池
- **继承关系**: 继承自 `InfiniopHandle`
- **关键成员**:
  - `_internal: std::shared_ptr<Internal>`: Pimpl 模式隐藏实现细节，管理内部状态
- **核心方法**:
  - `create(InfiniopHandle **handle_ptr, int device_id)`: 工厂方法，创建 Handle 实例并初始化为 INFINI_DEVICE_NVIDIA 类型
  - `internal() const`: 返回内部实现的共享指针，用于访问设备属性和句柄池
- **生命周期**: 通过 `create()` 工厂方法动态分配，使用裸指针管理（与 C 接口兼容）

### `device::nvidia::Handle::Internal`
- **位置**: `nvidia_handle.cuh`, `nvidia_common.cu`
- **主要功能**: 管理设备属性查询和库句柄对象池
- **关键成员**:
  - `blas_handles: Pool<cublasHandle_t>`: CUBLAS 句柄对象池，避免频繁创建/销毁开销
  - `dnn_handles: Pool<cudnnHandle_t>`: CUDNN 句柄对象池（条件编译 ENABLE_CUDNN_API）
  - `_warp_size: int`: 设备 warp 大小（通常为 32）
  - `_max_threads_per_block: int`: 单个线程块最大线程数
  - `_block_size[3]: int`: 线程块各维度最大尺寸 [x, y, z]
  - `_grid_size[3]: int`: 网格各维度最大尺寸 [x, y, z]
- **核心方法**:
  - `Internal(int device_id)`: 构造函数，调用 `cudaGetDeviceProperties` 查询设备能力并缓存属性
  - `useCublas(cudaStream_t stream, const Fn<cublasHandle_t> &f) const`: 从池中获取或创建 CUBLAS 句柄，设置流，执行用户回调，归还句柄到池
  - `useCudnn(cudaStream_t stream, const Fn<cudnnHandle_t> &f) const`: 类似 useCublas，管理 CUDNN 句柄
  - `warpSize()`, `maxThreadsPerBlock()`, `blockSizeX/Y/Z()`, `gridSizeX/Y/Z()`: 设备属性访问器
- **生命周期**: 由 Handle 通过 shared_ptr 管理，句柄池在析构时自动释放资源

### `device::iluvatar::Handle`
- **位置**: `nvidia_handle.h`, `nvidia_common.cu`
- **主要功能**: Iluvatar GPU（天数智芯）设备句柄，兼容 CUDA 生态
- **继承关系**: 继承自 `device::nvidia::Handle`
- **核心方法**:
  - `create(InfiniopHandle **handle_ptr, int device_id)`: 创建 Handle 实例并初始化为 INFINI_DEVICE_ILUVATAR 类型
- **设计模式**: 继承复用，通过父类 nvidia::Handle 实现所有功能

### `device::qy::Handle`
- **位置**: `nvidia_handle.h`, `nvidia_common.cu`
- **主要功能**: QY GPU 设备句柄
- **继承关系**: 继承自 `device::nvidia::Handle`
- **核心方法**:
  - `create(InfiniopHandle **handle_ptr, int device_id)`: 创建 Handle 实例并初始化为 INFINI_DEVICE_QY 类型
- **特殊处理**: 部分代码路径禁用 F64 和 I64 支持（见 nvidia_common.cu:68, #ifndef ENABLE_QY_API）

### `device::hygon::Handle`
- **位置**: `nvidia_handle.h`, `nvidia_common.cu`
- **主要功能**: Hygon DCU（海光）设备句柄，兼容 CUDA 生态
- **继承关系**: 继承自 `device::nvidia::Handle`
- **核心方法**:
  - `create(InfiniopHandle **handle_ptr, int device_id)`: 创建 Handle 实例并初始化为 INFINI_DEVICE_HYGON 类型
- **特殊处理**: 使用不同的 bfloat16 类型定义和内核启动配置（见 nvidia_kernel_common.cuh:5, 24-25）

## 3. API 接口

```cpp
// 设备句柄创建工厂方法
namespace device::nvidia {
    infiniStatus_t Handle::create(InfiniopHandle **handle_ptr, int device_id);
    // 创建 NVIDIA GPU 设备句柄，成功返回 INFINI_STATUS_SUCCESS
    // device_id: CUDA 设备编号（0, 1, 2, ...）
    // handle_ptr: 输出参数，返回新分配的 Handle 指针
}

namespace device::iluvatar {
    infiniStatus_t Handle::create(InfiniopHandle **handle_ptr, int device_id);
    // 创建 Iluvatar GPU 设备句柄
}

namespace device::qy {
    infiniStatus_t Handle::create(InfiniopHandle **handle_ptr, int device_id);
    // 创建 QY GPU 设备句柄
}

namespace device::hygon {
    infiniStatus_t Handle::create(InfiniopHandle **handle_ptr, int device_id);
    // 创建 Hygon DCU 设备句柄
}

// CUBLAS 句柄访问接口（通过 Handle::Internal）
infiniStatus_t Handle::Internal::useCublas(
    cudaStream_t stream,
    const Fn<cublasHandle_t> &f
) const;
// 从句柄池获取或创建 CUBLAS 句柄，绑定到指定 CUDA 流，
// 执行用户函数 f，自动归还句柄到池
// 流式操作安全，支持并发调用

// CUDNN 句柄访问接口（条件编译 ENABLE_CUDNN_API）
infiniStatus_t Handle::Internal::useCudnn(
    cudaStream_t stream,
    const Fn<cudnnHandle_t> &f
) const;
// 类似 useCublas，管理 CUDNN 句柄

// 类型转换接口
cudnnDataType_t getCudnnDtype(infiniDtype_t dt);
// 将 Infini 统一数据类型映射到 CUDNN 数据类型
// 支持类型：F16, F32, F64(部分设备), BF16, I8, I32, I64(部分设备), U8
// 默认返回 CUDNN_DATA_FLOAT
```

## 4. 使用示例

```cpp
// 示例：创建 NVIDIA GPU 设备句柄并执行 CUBLAS 矩阵乘法
#include "infiniop/devices/nvidia/nvidia_handle.h"

using namespace device::nvidia;

// 1. 创建设备句柄
InfiniopHandle* handle_raw;
infiniStatus_t status = Handle::create(&handle_raw, 0); // 使用 GPU 0
Handle* handle = static_cast<Handle*>(handle_raw);

// 2. 访问设备属性
int warp_size = handle->internal()->warpSize(); // 32
int max_threads = handle->internal()->maxThreadsPerBlock(); // 1024 或更多

// 3. 使用 CUBLAS 执行矩阵乘法
cudaStream_t stream;
cudaStreamCreate(&stream);

status = handle->internal()->useCublas(stream, [](cublasHandle_t blas_handle) {
    int m = 1024, n = 1024, k = 1024;
    const float alpha = 1.0f, beta = 0.0f;
    const float* A = /* 矩阵 A */;
    const float* B = /* 矩阵 B */;
    float* C = /* 矩阵 C */;

    return cublasSgemm(
        blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k, &alpha, B, n, A, k, &beta, C, n
    );
});

// 4. 清理资源
cudaStreamDestroy(stream);
delete handle;

// 示例：获取 Infini 张量对应的 CUDNN 数据类型
#ifdef ENABLE_CUDNN_API
infiniDtype_t dtype = INFINI_DTYPE_F16;
cudnnDataType_t cudnn_dtype = device::nvidia::getCudnnDtype(dtype); // CUDNN_DATA_HALF
#endif

// 示例：在 CUDA 内核中使用索引计算函数
#include "infiniop/devices/nvidia/nvidia_kernel_common.cuh"

__global__ void custom_kernel(
    float* output, const float* input,
    size_t ndim, const size_t* shape, const ptrdiff_t* strides
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t offset = device::nvidia::indexToOffset(idx, ndim, shape, strides);
    output[offset] = input[offset] * 2.0f;
}
```

## 5. 实现细节

### 内存管理
- **句柄池模式**: 使用 `Pool<T>` 模板类（定义于 `../pool.h`）管理 CUBLAS 和 CUDNN 句柄，避免频繁创建/销毁的系统调用开销
- **对象池策略**: 池初始为空，首次请求时创建新句柄，使用后归还池中供后续复用，Handle 析构时自动释放所有池中句柄
- **Pimpl 惯用法**: Handle 类通过 shared_ptr<Internal> 隐藏实现细节，降低编译依赖，便于后续扩展

### 并发与线程安全
- **流式并发**: useCublas 和 useCudnn 方法接受 cudaStream_t 参数，支持多个 CUDA 流并发执行不同操作
- **句柄池线程安全性**: Pool<T> 的实现需保证线程安全（通过 mutex 或 lock-free 机制），本模块依赖外部实现
- **常量成员方法**: useCublas/useCudnn 标记为 const，但通过 mutable Pool<> 实现逻辑上的常量性

### 性能优化
- **设备属性缓存**: 构造时一次性查询所有设备属性（warpSize、maxThreadsPerBlock 等），避免重复调用 cudaGetDeviceProperties
- **句柄复用**: 对象池显著减少库句柄创建开销，CUBLAS/CUDNN 句柄创建通常涉及数百微秒的初始化
- **零拷贝语义**: 句柄池使用移动语义（push/pop 接口），避免不必要的句柄拷贝

### 错误处理
- **宏封装错误检查**: CHECK_CUBLAS 和 CHECK_CUDNN 宏将库返回状态码转换为统一 infiniStatus_t
- **错误传播**: useCublas/useCudnn 内部任何错误都会立即中止并返回错误码，不会将损坏的句柄归还池中
- **类型映射降级**: getCudnnDtype 对不支持的类型返回 CUDNN_DATA_FLOAT 作为安全降级

### 多硬件兼容性
- **条件编译策略**: 通过 ENABLE_CUDNN_API、ENABLE_ILUVATAR_API、ENABLE_QY_API、ENABLE_HYGON_API 宏适配不同硬件
- **类型别名统一**: nvidia_kernel_common.cuh 定义 cuda_bfloat16、cuda_bfloat162、cuda_fp8_e4m3 别名，统一不同厂商的类型名称
- **内核启动配置**: Hygon DCU 使用 __launch_bounds__(1024) 强制内核最大线程数限制（NVIDIA GPU 不需要）
- **功能裁剪**: QY GPU 禁用 F64/I64 支持（可能硬件不支持），Iluvatar GPU 禁用 I64 支持

### 数据类型映射
| Infini 类型      | CUDNN 类型           | 说明                     |
|------------------|----------------------|--------------------------|
| INFINI_DTYPE_F16 | CUDNN_DATA_HALF      | 16 位浮点                |
| INFINI_DTYPE_F32 | CUDNN_DATA_FLOAT     | 32 位浮点（默认降级）     |
| INFINI_DTYPE_F64 | CUDNN_DATA_DOUBLE    | 64 位浮点（QY 不支持）   |
| INFINI_DTYPE_BF16| CUDNN_DATA_BFLOAT16  | 16 位脑浮点              |
| INFINI_DTYPE_I8  | CUDNN_DATA_INT8      | 8 位整数                 |
| INFINI_DTYPE_I32 | CUDNN_DATA_INT32     | 32 位整数                |
| INFINI_DTYPE_I64 | CUDNN_DATA_INT64     | 64 位整数（部分不支持）  |
| INFINI_DTYPE_U8  | CUDNN_DATA_UINT8     | 8 位无符号整数           |

### 数学函数支持
nvidia_kernel_common.cuh 提供跨平台的 exp_ 模板函数：
- `float`: expf（单精度快速指数）
- `double`: exp（双精度指数）
- `long double`: expl（扩展精度，仅 NVIDIA/Iluvatar/QY）
- `__half`: hexp（半精度 FP16）
- `__nv_bfloat16`: hexp（脑浮点 BF16）

### 索引计算算法
`indexToOffset` 函数实现**扁平索引到内存偏移的映射**，支持任意步长张量：
- **输入**: 扁平索引（线程索引）、维度数、形状数组、步长数组
- **算法**: 从最低维到最高维迭代，对每维度执行模除运算提取坐标，乘以步长并累加
- **复杂度**: O(ndim)，ndim 为张量维度数（通常 ≤ 8）
- **使用场景**: CUDA 内核中根据线程索引计算访问位置，支持非连续张量布局
- **属性**: __forceinline__ + __device__ + __host__，确保设备端高性能内联，同时支持主机端测试

### 设计模式
- **工厂方法**: Handle::create 静态方法封装对象创建逻辑
- **Pimpl（指针实现）**: Handle 类通过 shared_ptr<Internal> 隐藏实现细节
- **对象池**: Pool<cublasHandle_t> 和 Pool<cudnnHandle_t> 复用昂贵资源
- **模板方法模式**: useCublas/useCudnn 定义句柄获取-使用-归还的算法骨架，用户通过回调函数自定义操作
- **继承层次**: Iluvatar/QY/Hygong 继承自 nvidia::Handle，复用所有功能
- **RAII（外部）**: Pool<T> 的 push/pop 操作隐含资源所有权管理

### 依赖关系
- **外部库**: CUDA Runtime API、CUBLAS、CUDNN（可选）
- **内部模块**:
  - `../../handle.h`: 基类 InfiniopHandle 定义
  - `../pool.h`: Pool<T> 对象池模板实现
  - `../../../utils.h`: CHECK_INTERNAL 宏定义（统一错误检查）
  - `infinicore.h`: 核心数据类型和设备类型枚举（infiniDtype_t, infiniDevice_t）
- **硬件相关**: cuda_bf16.h, cuda_fp16.h, cuda_fp8.h（CUDA 头文件）

### 代码组织
- **头文件分离**: .h 文件声明公共接口，.cuh 文件声明 CUDA 相关实现（.cu 文件包含）
- **命名空间**: 所有代码位于 device::nvidia 及其子命名空间，避免符号冲突
- **编译单元**: nvidia_common.cu 是唯一的 .cu 实现文件，简化 CUDA 编译流程
