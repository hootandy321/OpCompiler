# Moore mublas GEMM 算子实现文档

## 概述

本模块实现了基于 Moore 架构（摩尔线程 GPU）的通用矩阵乘法（GEMM）算子，通过调用 muBLAS 库的 `mublasGemmStridedBatchedEx` API 实现高性能的批量矩阵乘法运算。该模块支持 FP16、BF16 和 FP32 三种数据类型，并针对列主序（Column-Major）布局进行了优化。

## 1. 模块结构

- **`gemm_mublas.h`**: 算子描述符的公共头文件，通过 `DESCRIPTOR(mublas)` 宏定义 `op::gemm::mublas::Descriptor` 类接口
- **`gemm_mublas.mu`**: 算子的具体实现文件，包含描述符的创建、析构和计算逻辑

## 2. 核心类

### `Descriptor` (通过 DESCRIPTOR 宏定义)

- **命名空间**: `op::gemm::mublas`
- **继承关系**: `public InfiniopDescriptor`
- **主要功能**: 封装 muBLAS GEMM 算子的描述符，管理算子元数据和硬件特定资源

#### 关键成员变量

- `Opaque *_opaque`: 指向硬件特定封装结构的指针（PImpl 模式），用于隐藏 Moore 设备相关实现
- `infiniDtype_t _dtype`: 矩阵元素的数据类型（F16/BF16/F32）
- `MatmulInfo _info`: 矩阵乘法的维度信息（M, N, K, batch）和矩阵布局信息
- `size_t _workspace_size`: 所需工作空间大小（当前实现为 0）

#### 核心方法

**`~Descriptor()`**
- **功能**: 析构函数，释放 `_opaque` 指针指向的资源
- **实现**: 直接 `delete _opaque`

**`static infiniStatus_t create(infiniopHandle_t handle_, Descriptor **desc_ptr, infiniopTensorDescriptor_t c_desc, infiniopTensorDescriptor_t a_desc, infiniopTensorDescriptor_t b_desc)`**
- **功能**: 工厂方法，创建并初始化 GEMM 描述符
- **参数验证**:
  - 检查数据类型是否为 `INFINI_DTYPE_F16`、`INFINI_DTYPE_F32` 或 `INFINI_DTYPE_BF16`
  - 通过 `MatmulInfo::create()` 验证张量形状和步长是否兼容
- **初始化流程**:
  1. 将 `infiniopHandle_t` 转换为 `device::moore::Handle*`
  2. 提取输出矩阵的数据类型
  3. 调用 `MatmulInfo::create()` 生成矩阵布局信息，强制使用 `MatrixLayout::COL_MAJOR`
  4. 构造 `Descriptor` 对象，保存 Moore 设备句柄的内部引用
- **返回值**: 成功返回 `INFINI_STATUS_SUCCESS`，失败返回相应的错误码

**`infiniStatus_t calculate(void *workspace, size_t workspace_size, void *c, float beta, const void *a, const void *b, float alpha, void *stream) const`**
- **功能**: 执行批量矩阵乘法运算，计算 `C = alpha * op(A) * op(B) + beta * C`
- **核心算法**: 调用 muBLAS 库的 `mublasGemmStridedBatchedEx()` 函数
- **数据类型映射**:
  - **F16**: `MUSA_R_16F`, `MUBLAS_COMPUTE_16F`, 需将 alpha/beta 转换为 `half` 类型
  - **BF16**: `MUSA_R_16BF`, `MUBLAS_COMPUTE_32F`（计算使用 FP32）
  - **F32**: `MUSA_R_32F`, `MUBLAS_COMPUTE_32F_FAST_TF32`（启用 Tensor Core 加速）
- **矩阵操作标志推导**:
  - `op_a`: 若 `row_stride == 1` 则为 `MUBLAS_OP_N`（行主序，不转置），否则为 `MUBLAS_OP_T`（列主序，转置）
  - `op_b`: 同 `op_a` 的推导逻辑
- **转置优化**: 若 `_info.is_transed` 为真，交换 A 和 B 矩阵的指针以适配布局要求
- **执行流程**:
  1. 根据数据类型设置 muBLAS 数据类型和计算类型
  2. 对于 F16 类型，使用 `__float2half()` 将 alpha/beta 转换为半精度
  3. 获取 Moore 设备句柄的 muBLAS handle（通过 `_opaque->internal->useMublas()`）
  4. 调用 `mublasGemmStridedBatchedEx()` 执行批量 GEMM
  5. 检查 muBLAS 返回状态并转换为 InfiniOP 状态码
- **性能特性**: 使用 `MUBLAS_GEMM_DEFAULT` 算法，由 muBLAS 库自动选择最优 kernel

### `Descriptor::Opaque` (PImpl 实现)

- **定义位置**: `gemm_mublas.mu`
- **访问级别**: `private`（对外部完全隐藏）
- **核心成员**:
  - `std::shared_ptr<device::moore::Handle::Internal> internal`: Moore 设备句柄的内部实现引用
- **生命周期**: 由 `Descriptor` 构造时分配，析构时释放

## 3. API 接口

```cpp
namespace op::gemm::mublas {

class Descriptor final : public InfiniopDescriptor {
public:
    // 析构函数
    ~Descriptor();

    // 获取工作空间大小（当前实现返回 0）
    size_t workspaceSize() const;

    // 创建描述符的工厂方法
    static infiniStatus_t create(
        infiniopHandle_t handle,              // [输入] Moore 设备句柄
        Descriptor **desc_ptr,                // [输出] 描述符指针的指针
        infiniopTensorDescriptor_t c_desc,    // [输入] 输出矩阵 C 的张量描述符
        infiniopTensorDescriptor_t a_desc,    // [输入] 输入矩阵 A 的张量描述符
        infiniopTensorDescriptor_t b_desc);   // [输入] 输入矩阵 B 的张量描述符

    // 执行矩阵乘法计算
    infiniStatus_t calculate(
        void *workspace,       // [输入] 工作空间指针（当前未使用）
        size_t workspace_size, // [输入] 工作空间大小（当前未使用）
        void *c,               // [输入/输出] 输出矩阵 C，公式 C = alpha*A*B + beta*C
        float beta,            // [输入] 标量 beta，用于 C 的缩放
        const void *a,         // [输入] 输入矩阵 A
        const void *b,         // [输入] 输入矩阵 B
        float alpha,           // [输入] 标量 alpha，用于 A*B 的缩放
        void *stream) const;   // [输入] MUSA 流指针
};

}
```

## 4. 使用示例

```cpp
#include "infiniop/ops/gemm/moore/mublas/gemm_mublas.h"
#include "infiniop/devices/moore/moore_handle.h"

// 1. 创建 Moore 设备句柄
device::moore::Handle *moore_handle;
auto status = device::moore::Handle::create(
    reinterpret_cast<InfiniopHandle **>(&moore_handle),
    device_id  // 设备 ID
);

// 2. 准备张量描述符（假设已创建）
// 形状要求：C = [batch, M, N], A = [batch, M, K], B = [batch, K, N]
// 步长要求：至少有一个维度是连续的（row_stride 或 col_stride 为 1）
infiniopTensorDescriptor_t c_desc, a_desc, b_desc;
// ... (填充张量描述符)

// 3. 创建 GEMM 描述符
op::gemm::mublas::Descriptor *gemm_desc;
status = op::gemm::mublas::Descriptor::create(
    moore_handle,
    &gemm_desc,
    c_desc,  // 输出：[batch, M, N]
    a_desc,  // 输入 A：[batch, M, K]
    b_desc   // 输入 B：[batch, K, N]
);

if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 4. 分配工作空间（当前 muBLAS 实现不需要额外工作空间）
size_t workspace_size = gemm_desc->workspaceSize();  // 返回 0
void *workspace = nullptr;  // 或分配 workspace_size 字节

// 5. 分配矩阵内存（假设已在 MUSA 设备上分配）
void *d_A, *d_B, *d_C;
musaMalloc(&d_A, a_size * dtype_size);
musaMalloc(&d_B, b_size * dtype_size);
musaMalloc(&d_C, c_size * dtype_size);

// 6. 创建 MUSA 流
musaStream_t stream;
musaStreamCreate(&stream);

// 7. 执行矩阵乘法：C = 2.0f * A * B + 1.0f * C
status = gemm_desc->calculate(
    workspace,      // 工作空间（可为 nullptr）
    workspace_size, // 工作空间大小
    d_C,            // 输出矩阵
    1.0f,           // beta
    d_A,            // 输入矩阵 A
    d_B,            // 输入矩阵 B
    2.0f,           // alpha
    stream          // MUSA 流
);

// 8. 同步流并等待计算完成
musaStreamSynchronize(stream);

// 9. 清理资源
delete gemm_desc;
musaFree(d_A);
musaFree(d_B);
musaFree(d_C);
musaStreamDestroy(stream);
// ... (清理张量描述符和设备句柄)
```

## 5. 实现细节

### 内存管理
- **工作空间策略**: 当前实现不分配额外工作空间，`workspaceSize()` 返回 0。muBLAS 库内部管理所需的临时存储
- **句柄复用**: 通过 `device::moore::Handle::Internal` 的 `Pool<std::unique_ptr<mublasHandle_t>>` 实现线程安全的 muBLAS 句柄池
- **智能指针**: 使用 `std::shared_ptr` 管理 `device::moore::Handle::Internal` 的生命周期，确保资源正确释放

### 并发控制
- **句柄池机制**: `useMublas()` 方法内部从线程池获取或创建 muBLAS 句柄，避免多线程竞争
- **流隔离**: 每次调用 `calculate()` 传入独立的 MUSA 流，支持不同算子实例的并行执行
- **原子操作**: muBLAS 库内部使用锁机制保护句柄的初始化和销毁

### 性能优化
- **批量计算**: 使用 `mublasGemmStridedBatchedEx()` 一次性处理多个矩阵乘法，减少 kernel 启动开销
- **Tensor Core 加速**: 对于 F32 类型，使用 `MUBLAS_COMPUTE_32F_FAST_TF32` 启用 TF32 模式，在 Ampere 及更新架构上显著加速
- **自动调优**: `MUBLAS_GEMM_DEFAULT` 允许 muBLAS 库根据矩阵形状和数据类型自动选择最优算法
- **零拷贝转置**: 通过 `_info.is_transed` 标志和指针交换实现逻辑转置，无需实际数据移动

### 错误处理
- **类型检查**: 构造时验证数据类型，仅支持 F16/BF16/F32
- **形状验证**: `MatmulInfo::create()` 检查矩阵维度兼容性（M×K × K×N = M×N）
- **步长验证**: 要求至少一个维度是连续的（`row_stride == 1 || col_stride == 1`），否则返回 `INFINI_STATUS_BAD_TENSOR_STRIDES`
- **批处理一致性**: 检查 A、B、C 矩阵的 batch 维度是否匹配（允许广播 batch=1）
- **错误传播**: 使用 `CHECK_MUBLAS()` 宏将 muBLAS 错误码转换为 InfiniOP 状态码

### 依赖关系
- **外部库**:
  - `mublas.h`: Moore 线程的 BLAS 库，提供 GPU 加速的矩阵运算
  - `musa.h`: Moore 线程的 CUDA 兼容运行时 API
  - `musa_fp16_mtgpu.h`: 半精度浮点数支持（`__float2half()` 函数）
- **内部模块**:
  - `device::moore::Handle`: Moore 设备句柄，封装 muBLAS 和 muDNN 资源
  - `op::gemm::MatmulInfo`: 矩阵布局信息和形状验证逻辑
  - `op::gemm::BlasMatrix`: BLAS 矩阵元数据（维度、步长、批处理）

### 设计模式
- **PImpl (Pointer to Implementation)**: 通过 `Opaque` 结构体隐藏 Moore 设备相关的实现细节，保持头文件的平台无关性
- **工厂模式**: `create()` 静态方法封装复杂的构造逻辑和验证流程
- **策略模式**: 通过 `MatrixLayout::COL_MAJOR` 参数指定矩阵布局策略
- **RAII (Resource Acquisition Is Initialization)**: 使用析构函数自动释放 `_opaque` 资源

### 特殊处理
- **标量类型适配**: MUSA 的 GEMM 要求 alpha/beta 类型与矩阵数据类型一致。对于 F16 矩阵，使用 `__float2half()` 将 float 参数转换为 half 类型
- **布局转置优化**: 若 C 矩阵为行主序（`col_stride == 1`），则在 `MatmulInfo::create()` 中将所有矩阵转置，并交换 A 和 B 的角色，使计算适配列主序优化的 muBLAS 实现
- **指针交换技巧**: 在 `calculate()` 中通过 `std::swap(a, b)` 交换输入矩阵指针，避免实际数据转置
