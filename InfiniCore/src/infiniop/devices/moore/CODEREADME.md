# Moore GPU 设备适配层核心实现文档

Moore 设备适配层为 InfiniOp 框架提供了对 Moore 系列 GPU（Moore Threads 架构）的完整支持，包括设备句柄管理、MUBLAS 和 MUDNN 库的封装、以及 MUSA 内核开发的基础设施。

## 1. 模块结构

- **`moore_handle.h`**: Moore 设备句柄的公共接口定义，继承自 `InfiniopHandle` 基类
- **`moore_handle.cc`**: 设备句柄实现，包括设备属性查询和 MUBLAS/MUDNN 句柄池管理
- **`moore_common.h`**: 内部实现类定义，包含句柄池和设备属性缓存
- **`moore_kernel_common.h`**: MUSA 内核开发基础设施，包括类型映射、数学函数包装和内存索引计算

## 2. 核心类

### `device::moore::Handle`
- **位置**: `moore_handle.h`, `moore_handle.cc`
- **主要功能**: Moore 设备的顶层句柄，管理设备生命周期和资源，封装底层 MUBLAS 和 MUDNN 库的访问
- **关键成员**:
  - `_internal`: `std::shared_ptr<Internal>` - 内部实现的共享指针，采用 Pimpl 模式隐藏实现细节
- **核心方法**:
  - `Handle(int device_id)`: 构造函数，初始化 Moore 设备句柄，设备类型固定为 `INFINI_DEVICE_MOORE`
  - `create(InfiniopHandle **handle_ptr, int device_id)`: 静态工厂方法，创建设备句柄实例并返回状态码
  - `internal() const`: 访问内部实现的常量引用，用于访问底层功能
- **生命周期**: 工厂模式创建，通过 `shared_ptr` 管理内部实现，支持多线程共享访问

### `device::moore::Handle::Internal`
- **位置**: `moore_common.h` (声明), `moore_handle.cc` (实现)
- **主要功能**: 设备资源的实际管理者，维护 MUBLAS/MUDNN 句柄池和设备硬件属性缓存
- **关键成员**:
  - `mublas_handles`: `Pool<std::unique_ptr<mublasHandle_t>>` - MUBLAS 库句柄的对象池，实现句柄复用
  - `mudnn_handles`: `Pool<std::unique_ptr<::musa::dnn::Handle>>` - MUDNN 库句柄的对象池
  - `_warp_size`: `int` - 设备的 warp 大小（通常为 32）
  - `_max_threads_per_block`: `int` - 每个 thread block 的最大线程数
  - `_block_size[3]`: `int[3]` - thread block 在 X/Y/Z 维度的最大尺寸
  - `_grid_size[3]`: `int[3]` - grid 在 X/Y/Z 维度的最大尺寸
  - `_device_id`: `int` - 设备 ID，用于 MUDNN 句柄初始化
- **核心方法**:
  - `Internal(int device_id)`: 构造函数，调用 `musaGetDeviceProperties` 查询硬件属性并缓存
  - `useMublas(musaStream_t stream, const Fn<mublasHandle_t> &f)`: 从池中获取或创建 MUBLAS 句柄，设置流，执行回调函数，归还句柄到池。使用 `CHECK_MUBLAS` 宏验证 API 调用
  - `useMudnn(musaStream_t stream, const Fn<::musa::dnn::Handle &> &f)`: 从池中获取或创建 MUDNN 句柄，设置流，执行回调，归还句柄。使用 `CHECK_MUDNN` 宏验证状态
  - `warpSize()`, `maxThreadsPerBlock()`, `blockSizeX/Y/Z()`, `gridSizeX/Y/Z()`: 访问缓存的硬件属性，O(1) 时间复杂度
- **生命周期**: 由 `Handle` 通过 `shared_ptr` 管理，句柄池采用对象池模式避免频繁创建/销毁开销

## 3. API 接口

```cpp
// 设备句柄创建 API
infiniStatus_t Handle::create(InfiniopHandle **handle_ptr, int device_id);
// 在指定 Moore GPU 上创建设备句柄，返回 INFINI_STATUS_SUCCESS 成功
// handle_ptr: 输出参数，指向新创建的句柄指针
// device_id: Moore GPU 设备 ID（0-based）

// MUBLAS 库操作执行器
infiniStatus_t Handle::Internal::useMublas(
    musaStream_t stream,
    const Fn<mublasHandle_t> &f
) const;
// 从句柄池获取 MUBLAS 句柄，绑定到指定流，执行用户回调函数 f
// stream: MUSA 异步流
// f: 接收 mublasHandle_t 的函数对象，执行具体的 BLAS 操作

// MUDNN 库操作执行器
infiniStatus_t Handle::Internal::useMudnn(
    musaStream_t stream,
    const Fn<::musa::dnn::Handle &> &f
) const;
// 从句柄池获取 MUDNN 句柄，绑定到指定流，执行用户回调函数 f
// stream: MUSA 异步流
// f: 接收 musa::dnn::Handle 引用的函数对象，执行具体的 DNN 操作

// 设备属性查询 API
int Handle::Internal::warpSize() const;           // 返回 warp 大小（通常 32）
int Handle::Internal::maxThreadsPerBlock() const; // 返回每块最大线程数（如 1024）
int Handle::Internal::blockSizeX/Y/Z() const;     // 返回 block 在各维度的最大尺寸
int Handle::Internal::gridSizeX/Y/Z() const;      // 返回 grid 在各维度的最大尺寸
```

## 4. 使用示例

```cpp
// 示例：在 Moore GPU 上创建句柄并执行矩阵乘法
#include "infiniop/devices/moore/moore_handle.h"

// 1. 创建 Moore 设备句柄
InfiniopHandle *raw_handle;
infiniStatus_t status = device::moore::Handle::create(&raw_handle, 0);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}
auto *moore_handle = static_cast<device::moore::Handle *>(raw_handle);

// 2. 查询设备硬件属性
int warp_size = moore_handle->internal()->warpSize();
int max_threads = moore_handle->internal()->maxThreadsPerBlock();
printf("Moore GPU: warp_size=%d, max_threads=%d\n", warp_size, max_threads);

// 3. 使用 MUBLAS 执行矩阵乘法
musaStream_t stream;
musaStreamCreate(&stream);
status = moore_handle->internal()->useMublas(stream, [](mublasHandle_t blas_handle) {
    const float alpha = 1.0f, beta = 0.0f;
    return mublasSgemm(blas_handle,
                       MUBLAS_OP_N, MUBLAS_OP_N,
                       m, n, k,
                       &alpha,
                       d_A, lda,
                       d_B, ldb,
                       &beta,
                       d_C, ldc);
});

// 4. 使用 MUDNN 执行卷积操作
status = moore_handle->internal()->useMudnn(stream, [](::musa::dnn::Handle &dnn_handle) {
    ::musa::dnn::ConvolutionDescriptor conv_desc;
    // 配置卷积参数...
    return conv_desc.SetAlgorithmMode(::musa::dnn::ConvolutionAlgorithmMode::FWD_DEFAULT);
});

// 5. 清理资源（句柄由框架管理，只需销毁流）
musaStreamDestroy(stream);
```

## 5. 实现细节

- **内存管理**:
  - 句柄池采用 `std::unique_ptr` 管理原生 MUBLAS/MUDNN 句柄，确保自动销毁
  - 内部实现使用 `std::shared_ptr` 支持多个外部句柄共享同一资源
  - 对象池模式避免频繁调用 `mublasCreate`/`mublasDestroy` 和 `mudnn::Handle` 构造/析构的开销

- **并发控制**:
  - MUBLAS/MUDNN 句柄池使用 `Pool<T>` 容器（假设为线程安全队列），支持多线程并发获取/归还句柄
  - 句柄池在每次使用时动态绑定到指定 MUSA 流，确保流间隔离
  - 设备属性在构造时一次性查询并缓存为只读，后续访问无锁

- **性能优化**:
  - 设备属性查询（`musaGetDeviceProperties`）仅在初始化时执行一次，O(1) 后续访问
  - 句柄池复用策略减少库句柄创建开销（MUBLAS/MUDNN 句柄初始化成本较高）
  - 使用 `__forceinline__` 和 `__device__` 标记内核函数，确保 MUSA 编译器内联优化

- **错误处理**:
  - 使用 `CHECK_MUBLAS(API)` 宏验证 MUBLAS API 调用，失败时返回 `MUBLAS_STATUS_SUCCESS` 以外的状态
  - 使用 `CHECK_MUDNN(API)` 宏验证 MUDNN API 调用，失败时返回非 `::musa::dnn::Status::SUCCESS` 状态
  - 使用 `CHECK_STATUS` 宏传播用户回调函数的 `infiniStatus_t` 错误码
  - 所有公开 API 返回 `infiniStatus_t` 枚举类型，统一错误处理机制

- **依赖项**:
  - **外部库**: MUSA Runtime (`musa_runtime_api.h`), MUBLAS (`mublas.h`), MUDNN (`mudnn.h`), MUSA 数学库 (`musa_fp16_mtgpu.h`)
  - **内部依赖**: `../../handle.h` (设备句柄基类), `../pool.h` (句柄池容器), `../../../utils.h` (宏定义和工具函数)
  - **硬件要求**: Moore Threads GPU（Moore 架构），支持 `musaGetDeviceProperties` 查询的设备

- **设计模式**:
  - **Pimpl (Pointer to Implementation)**: `Handle` 通过 `shared_ptr<Internal>` 隐藏实现细节，减少编译依赖
  - **对象池模式 (Object Pool)**: MUBLAS/MUDNN 句柄池管理昂贵的库句柄资源
  - **工厂模式 (Factory)**: `Handle::create` 静态方法封装对象创建逻辑
  - **RAII (Resource Acquisition Is Initialization)**: `unique_ptr` 自动管理原生句柄生命周期
  - **模板方法模式 (Template Method)**: `useMublas`/`useMudnn` 定义句柄获取、使用、归还的骨架，用户通过回调函数 `f` 注入具体操作

- **类型映射与兼容层** (`moore_kernel_common.h`):
  - 定义 `cuda_bfloat16` → `mt_bfloat16`, `cuda_bfloat162` → `mt_bfloat162`, `cuda_fp8_e4m3` → `__mt_fp8_e4m3` 实现与 CUDA 类型的二进制兼容
  - 提供 `exp_(T)` 函数模板重载，支持 `float`/`double`/`long double`/`__half`/`__mt_bfloat16` 的指数运算，解决 MUSA 数学库中 `exp` 函数的歧义调用问题（`long double` 与 `double` 冲突）
  - 实现 `indexToOffset` 函数：将扁平化索引转换为多维张量的内存偏移量，采用行优先（C 风格）布局，算法复杂度 O(ndim)，适用于设备端和主机端代码

- **内核开发基础设施** (`moore_kernel_common.h`):
  - 定义 `INFINIOP_MOORE_KERNEL` 宏为 `__global__ void`，用于声明 MUDA 内核函数
  - 提供 `MOORE_BLOCK_SIZE_2048/1024/512` 常量，用于根据硬件能力选择合适的 block 大小
  - 定义 `CHECK_MOORE(API)` 宏验证 MUSA Runtime API 调用，失败时返回非 `musaSuccess` 状态
  - 包含 MUSA 低精度数据类型头文件 (`musa_bf16.h`, `musa_fp16.h`, `musa_fp8.h`)，支持 BF16/FP16/FP8 计算
