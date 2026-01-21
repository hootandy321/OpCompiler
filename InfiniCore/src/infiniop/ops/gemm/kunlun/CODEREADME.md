# 昆仑 (Kunlun) GEMM 算子核心实现文档

本模块实现了昆仑 XPU 加速卡上的通用矩阵乘法 (GEMM) 算子，通过 cuBLAS 兼容层提供高性能的批量矩阵乘法运算。该实现支持 FP16、BF16 和 FP32 三种数据类型，利用昆仑硬件的 Tensor Core 加速单元实现高效计算。

## 1. 模块结构

- **`gemm_kunlun.h`**: 算子描述符声明，使用宏 `DESCRIPTOR(kunlun)` 展开生成完整的 `Descriptor` 类定义，隐藏硬件相关的内部实现细节
- **`gemm_kunlun.cc`**: 算子核心实现，包含描述符创建、矩阵元数据解析、类型映射转换以及 cuBLAS 批量 GEMM 调度逻辑

## 2. 核心类

### `Descriptor::Opaque`
- **位置**: `gemm_kunlun.cc:9-11`
- **主要功能**: 封装昆仑硬件相关的句柄内部状态，采用 PImpl (Pointer to Implementation) 模式隐藏硬件细节
- **关键成员**:
  - `std::shared_ptr<HandleInternal> internal`: 共享指针管理昆仑 BLAS 句柄的内部实现，生命周期由引用计数自动管理
- **生命周期**: 由 `Descriptor` 构造函数分配，在 `Descriptor` 析构函数中显式释放 (第 13-15 行)

### `op::gemm::kunlun::Descriptor`
- **位置**: 通过 `DESCRIPTOR(kunlun)` 宏展开定义在 `gemm_kunlun.h`
- **主要功能**: 昆仑 GEMM 算子的完整描述符，继承自 `InfiniopDescriptor`，负责算子创建、工作空间管理和计算执行
- **关键成员** (继承自宏定义):
  - `Opaque *_opaque`: 指向硬件相关内部状态的指针
  - `infiniDtype_t _dtype`: 计算数据类型 (FP16/BF16/F32)
  - `MatmulInfo _info`: 矩阵乘法元数据，包含 M/N/K 维度、batch 大小、布局信息和 stride 信息
  - `size_t _workspace_size`: 工作空间大小 (当前实现为 0)
- **核心方法**:
  - `~Descriptor()`: 析构函数，释放 `_opaque` 指针，避免内存泄漏
  - `static create(...)`: 工厂方法，解析张量描述符，验证数据类型，创建 `MatmulInfo` 元数据，构造并返回算子描述符实例
  - `calculate(...)`: 执行批量 GEMM 计算，映射 InfiniOP 类型到 cuBLAS 类型，处理矩阵转置逻辑，通过句柄池调度 cuBLAS 内核
- **生命周期**:
  - **创建**: 通过静态工厂方法 `create()` 构造，执行类型检查和元数据验证
  - **使用**: 调用 `calculate()` 执行实际计算，可多次调用复用同一描述符
  - **销毁**: 显式 `delete` 或通过 RAII 管理，析构时释放 Opaque 内部状态

### `device::kunlun::blas::Handle::Internal`
- **位置**: 引用自 `kunlun_xblas.h:27-34`
- **主要功能**: 管理 cuBLAS 句柄池的生命周期，提供线程安全的句柄获取和执行接口
- **关键成员**:
  - `Pool<cublasHandle_t> blas_handles`: 无锁并发句柄池，使用 lock-free 的 `compare_exchange_weak` 原子操作实现高效的 push/pop
- **核心方法**:
  - `useCublas(cudaStream_t stream, Fn<cublasHandle_t> f)`: 从句柄池获取或创建 cuBLAS 句柄，在指定 CUDA 流上执行用户传入的 lambda 函数，执行完毕后归还句柄到池中
- **设计模式**:
  - **对象池模式**: 重用昂贵的 cuBLAS 句柄创建开销，避免重复初始化
  - **RAII**: 句柄生命周期由 `std::shared_ptr` 自动管理，确保资源正确释放

## 3. API 接口

```cpp
// 算子描述符创建接口
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,                    // [入参]昆仑设备句柄，包含 device_id 和内部状态
    Descriptor **desc_ptr,                        // [出参]返回创建的描述符指针
    infiniopTensorDescriptor_t c_desc,            // [入参]输出张量 C 的描述符 (形状: [batch, M, N] 或 [M, N])
    infiniopTensorDescriptor_t a_desc,            // [入参]输入张量 A 的描述符 (形状: [batch, M, K] 或 [M, K])
    infiniopTensorDescriptor_t b_desc             // [入参]输入张量 B 的描述符 (形状: [batch, K, N] 或 [K, N])
);
// 返回值: INFINI_STATUS_SUCCESS 表示成功，错误码表示失败类型 (如不支持的 dtype、形状不匹配等)

// 矩阵乘法计算接口
infiniStatus_t Descriptor::calculate(
    void *workspace,                              // [入参]工作空间指针 (当前未使用，传 nullptr)
    size_t workspace_size,                        // [入参]工作空间大小 (当前为 0)
    void *c,                                      // [出参]输出矩阵 C 的设备指针
    float beta,                                   // [入参]C 的缩放因子 (C = alpha * op(A) * op(B) + beta * C)
    const void *a,                                // [入参]输入矩阵 A 的设备指针
    const void *b,                                // [入参]输入矩阵 B 的设备指针
    float alpha,                                  // [入参]A*B 的缩放因子
    void *stream                                  // [入参]昆仑计算流 (XPUStream_t)
) const;
// 返回值: INFINI_STATUS_SUCCESS 表示成功，错误码表示 cuBLAS 调用失败
```

## 4. 使用示例

```cpp
// 示例: 在昆仑设备上执行批量 FP16 矩阵乘法 C = alpha * A * B + beta * C

#include "infiniop/ops/gemm/kunlun/gemm_kunlun.h"

// 1. 创建昆仑设备句柄
int device_id = 0;
device::kunlun::blas::Handle *kunlun_handle;
auto status = device::kunlun::blas::Handle::create(
    reinterpret_cast<InfiniopHandle **>(&kunlun_handle),
    device_id
);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理句柄创建失败
}

// 2. 准备张量描述符 (假设形状: batch=2, M=128, N=256, K=512)
// A: [2, 128, 512], B: [2, 512, 256], C: [2, 128, 256]
infiniopTensorDescriptor_t a_desc, b_desc, c_desc;
// ... 初始化张量描述符 (略，包含维度、步长、数据类型等)

// 3. 创建 GEMM 算子描述符
op::gemm::kunlun::Descriptor *gemm_desc;
status = op::gemm::kunlun::Descriptor::create(
    kunlun_handle,
    &gemm_desc,
    c_desc,  // 输出张量
    a_desc,  // 左矩阵
    b_desc   // 右矩阵
);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理描述符创建失败 (如类型不支持、形状不匹配)
}

// 4. 分配设备内存并初始化数据
void *d_a, *d_b, *d_c;
size_t size_a = 2 * 128 * 512 * sizeof(float16_t);
size_t size_b = 2 * 512 * 256 * sizeof(float16_t);
size_t size_c = 2 * 128 * 256 * sizeof(float16_t);
xpu_malloc(&d_a, size_a);
xpu_malloc(&d_b, size_b);
xpu_malloc(&d_c, size_c);
// ... 从主机拷贝数据到设备 (略)

// 5. 创建计算流
XPUStream stream;
xpu_stream_create(&stream);

// 6. 执行矩阵乘法: C = 1.0 * A * B + 0.0 * C
float alpha = 1.0f, beta = 0.0f;
status = gemm_desc->calculate(
    nullptr,     // 无工作空间
    0,           // 工作空间大小为 0
    d_c,         // 输出矩阵 C
    beta,        // beta = 0.0 表示不累加
    d_a,         // 输入矩阵 A
    d_b,         // 输入矩阵 B
    alpha,       // alpha = 1.0 表示标准矩阵乘法
    stream       // 昆仑计算流
);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理计算失败
}

// 7. 同步流并拷贝结果回主机
xpu_wait(stream);  // 等待计算完成
// ... 从 d_c 拷贝数据回主机 (略)

// 8. 清理资源
delete gemm_desc;      // 释放算子描述符
xpu_stream_destroy(stream);
xpu_free(d_a);
xpu_free(d_b);
xpu_free(d_c);
delete kunlun_handle;  // 释放设备句柄
```

## 5. 实现细节

### 数据类型映射与精度策略
- **FP16 (半精度浮点)**:
  - cuBLAS 类型: `CUDA_R_16F`
  - 计算类型: `CUBLAS_COMPUTE_32F` (累加使用 FP32 保证精度)
- **BF16 (脑浮点)**:
  - cuBLAS 类型: `CUDA_R_16BF`
  - 计算类型: `CUBLAS_COMPUTE_32F` (累加使用 FP32)
- **FP32 (单精度浮点)**:
  - cuBLAS 类型: `CUDA_R_32F`
  - 计算类型: `CUBLAS_COMPUTE_32F_FAST_TF32` (使用 TF32 加速，在 A100/Tensor Core 上性能更高)

### 矩阵布局自动转置逻辑
本实现采用**列优先 (COL_MAJOR)** 存储约定，并在 `calculate()` 方法中实现智能布局适配:

1. **输入验证** (第 28 行): 调用 `MatmulInfo::create()` 验证张量形状一致性 (M, N, K 维度匹配、batch 一致性)
2. **自动转置检测** (第 68-70 行):
   - 如果 `_info.is_transed` 为 true，交换矩阵 A 和 B 的指针
   - 这解决了行优先和列优先布局的兼容性问题
3. **操作标志推导** (第 72-73 行):
   - `row_stride == 1` 表示行优先 → 使用 `CUBLAS_OP_N` (不转置)
   - `col_stride == 1` 表示列优先 → 使用 `CUBLAS_OP_T` (转置)
   - 映射到 BLAS 的经典约定: 始终按列优先解释矩阵

### cuBLAS 批量 GEMM 调度
- **内核选择**: `cublasGemmStridedBatchedEx` (第 79-102 行)
  - 支持批量矩阵乘法 (batch 维度并行)
  - 使用 Tensor Core 加速 (`CUBLAS_GEMM_DEFAULT_TENSOR_OP`)
  - 显式指定数据类型和计算类型，提供最佳性能
- **参数映射**:
  - `op_a`, `op_b`: 根据 stride 自动推导是否转置
  - `m`, `n`, `k`: 矩阵乘法维度 (静态转换为 int)
  - `lda`, `ldb`, `ldc`: 主维度 (leading dimension)，通过 `BlasMatrix::ld()` 计算
  - `stride_a`, `stride_b`, `stride_c`: batch 维度步长
  - `batch`: 批量大小 (支持 batch=1 的退化情况)

### 内存管理策略
- **句柄池**: `Pool<cublasHandle_t>` 使用无锁栈结构 (基于 `std::atomic` 的 `compare_exchange_weak`):
  - **push**: 使用原子操作将句柄放回池顶 (O(1) 均摊时间)
  - **pop**: 使用 CAS 循环从池顶弹出句柄 (无锁并发，无线程阻塞)
  - **优势**: 避免每次调用都创建/销毁 cuBLAS 句柄的开销 (句柄初始化涉及设备侧资源分配)
- **智能指针**: `std::shared_ptr<HandleInternal>` 管理内部状态生命周期
  - 支持多描述符共享同一句柄池
  - 自动引用计数管理，无需手动释放

### 并发与同步
- **流并发**: 每次计算调用接收 `stream` 参数，支持同一设备上的多个流并行执行
- **句柄池并发**: 无锁栈支持多线程同时获取/归还句柄，无互斥锁竞争
- **同步点**: `xpu_wait(stream)` (第 106 行) 在计算完成后插入同步点，确保内核执行完毕再返回

### 错误处理机制
- **类型检查** (第 26 行): 宏 `CHECK_DTYPE` 验证 dtype 必须为 FP16/F32/BF16，否则返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
- **形状验证** (第 28-29 行): `MatmulInfo::create()` 检查矩阵维度一致性、batch 匹配性、stride 合法性
- **cuBLAS 错误传播** (第 78 行): `CHECK_CUBLAS` 宏将 cuBLAS 错误码转换为 InfiniOP 状态码
- **工作空间验证**: 当前不使用工作空间，但接口保留扩展性

### 性能优化技术
1. **Tensor Core 利用**: 使用 `CUBLAS_GEMM_DEFAULT_TENSOR_OP` 启用 Tensor Core 加速 (昆仑硬件兼容)
2. **FP32 累加**: FP16/BF16 输入时使用 FP32 累加 (`CUBLAS_COMPUTE_32F`)，避免精度损失
3. **句柄复用**: 对象池避免重复初始化 cuBLAS 上下文
4. **批量计算**: 单次 kernel 启动完成 batch 个矩阵乘法，降低 kernel 启动开销
5. **TF32 加速**: FP32 计算时使用 `CUBLAS_COMPUTE_32F_FAST_TF32` 在支持 TF32 的硬件上提升吞吐量

### 依赖关系
- **硬件层依赖**:
  - `kunlun_common.h`: 昆仑 XPU 运行时 (`xpu/runtime.h`, `xpu/xdnn.h`)
  - `kunlun_xblas.h`: cuBLAS 兼容层和句柄池实现
- - **算子框架依赖**:
  - `gemm.h`: 算子描述符宏定义和 PImpl 模式
  - `info.h`: `MatmulInfo` 矩阵元数据解析和布局转换
  - `operator.h`: 基类 `InfiniopDescriptor` 和 `InfiniopHandle`
- **外部依赖**:
  - cuBLAS (`cublas_v2.h`): NVIDIA CUDA BLAS 库 (通过昆仑兼容层调用)
  - XPU 运行时 (`xpu/runtime.h`): 昆仑设备管理、流和内存操作

### 设计模式应用
1. **PImpl (Pointer to Implementation)**: 通过 `Opaque` 结构体隐藏硬件相关类型，避免头文件暴露 cuBLAS 句柄等实现细节
2. **工厂方法模式**: `create()` 静态方法封装复杂的对象构建逻辑，提供清晰的错误处理
3. **对象池模式**: `Pool<cublasHandle_t>` 重用昂贵资源，降低初始化开销
4. **策略模式**: 通过 `compute_type` 参数选择不同的计算精度策略 (FP32/TF32)
5. **RAII**: 智能指针和析构函数确保资源自动释放，防止内存泄漏
