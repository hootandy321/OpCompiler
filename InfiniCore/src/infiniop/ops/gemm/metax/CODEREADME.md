# Metax GEMM 算子核心实现文档

本文档详细描述了 Moore Threads Metax GPU 后端的通用矩阵乘法（GEMM）算子实现，该模块基于 HCBLAS 库提供高性能的批量矩阵乘法运算，支持 FP16、BF16 和 FP32 三种精度。

## 1. 模块结构

- **`gemm_metax.h`**: 定义 Metax GEMM 描述符类，使用宏声明实现与硬件无关的公共接口
- **`gemm_metax.cc`**: 实现 Metax 设备上的 GEMM 算子核心逻辑，包括描述符创建、数据类型映射和 HCBLAS 调用

## 2. 核心类

### `Descriptor`
- **位置**: `gemm_metax.h`, `gemm_metax.cc`
- **主要功能**: Metax GPU 设备上的 GEMM 算子描述符，封装矩阵乘法运算所需的所有元数据和硬件相关状态
- **关键成员**:
  - `_opaque`: 指向 `Opaque` 结构体的指针，使用 PImpl 模式隐藏硬件相关实现细节（`std::shared_ptr<device::metax::Handle::Internal>`）
  - `_dtype`: 输出张量的数据类型（`infiniDtype_t`），支持 FP16/BF16/F32
  - `_info`: 矩阵乘法运算信息（`MatmulInfo`），包含 M/N/K 维度、batch 数量、矩阵布局和 leading dimensions
  - `_workspace_size`: 工作空间大小（当前实现为 0，HCBLAS 内部管理）
- **核心方法**:
  - `create(handle_, desc_ptr, c_desc, a_desc, b_desc)`: 静态工厂方法，验证输入张量并构造描述符实例。执行数据类型检查（仅支持 FP16/F32/BF16），创建 `MatmulInfo`（强制列主序布局 `MatrixLayout::COL_MAJOR`），初始化 Opaque 结构并存储 Metax 设备句柄的 `internal` 指针。时间复杂度 O(1)。
  - `calculate(workspace, workspace_size, c, beta, a, b, alpha, stream)`: 执行批量 GEMM 计算。将 Infini 数据类型映射到 HCBLAS 数据类型（`HPCC_R_16F`/`HPCC_R_16BF`/`HPCC_R_32F`），确定计算精度（`HCBLAS_COMPUTE_32F` 或 `HCBLAS_COMPUTE_32F_FAST_TF32`），根据 `row_stride` 自动推导矩阵转置标志（`HCBLAS_OP_N` 或 `HCBLAS_OP_T`），处理 `is_transed` 情况下的 A/B 矩阵交换，通过 `useMcblas` 获取 HCBLAS 句柄并调用 `hcblasGemmStridedBatchedEx` 执行批量张量核 GEMM。时间复杂度 O(batch × m × n × k)，空间复杂度 O(1)。
- **生命周期**: 由 `create` 工厂方法动态分配，析构函数释放 Opaque 指针（不销毁 `shared_ptr` 指向的 `Handle::Internal`，由外部管理）

### `Descriptor::Opaque`
- **位置**: `gemm_metax.cc:7-9`
- **主要功能**: 封装 Metax 硬件相关状态，使用 PImpl（Pointer to Implementation）模式隔离硬件特定类型
- **关键成员**:
  - `internal`: `std::shared_ptr<device::metax::Handle::Internal>`，共享的 Metax 设备内部句柄，用于访问 HCBLAS/HCDNN 句柄池和设备属性
- **设计模式**: PImpl 模式（不透明指针），隐藏硬件相关实现，确保头文件对硬件无关部分可见

## 3. API 接口

```cpp
namespace op::gemm::metax {

class Descriptor final : public InfiniopDescriptor {
public:
    // 析构函数：释放 Opaque 资源
    ~Descriptor();

    // 获取所需工作空间大小（当前实现固定返回 0）
    size_t workspaceSize() const;

    // 创建 GEMM 描述符实例
    // 参数:
    //   handle_: Infini 运行时句柄（实际类型为 device::metax::Handle*）
    //   desc_ptr: 输出参数，返回新创建的描述符指针
    //   c_desc, a_desc, b_desc: 输出/输入张量描述符
    // 返回: INFINI_STATUS_SUCCESS 或错误码
    static infiniStatus_t create(
        infiniopHandle_t handle_,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t c_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc);

    // 执行批量 GEMM 计算：C = alpha * op(A) * op(B) + beta * C
    // 参数:
    //   workspace, workspace_size: 工作空间缓冲区及大小（当前未使用）
    //   c: 输出张量指针
    //   beta: C 张量的缩放系数
    //   a, b: 输入矩阵指针
    //   alpha: A*B 的缩放系数
    //   stream: Metax 计算流（hcStream_t）
    // 返回: INFINI_STATUS_SUCCESS 或错误码
    infiniStatus_t calculate(
        void *workspace, size_t workspace_size,
        void *c,
        float beta,
        const void *a,
        const void *b,
        float alpha,
        void *stream) const;
};

}
```

## 4. 使用示例

```cpp
// 示例：在 Metax GPU 上执行批量 FP16 GEMM
// 假设输入为形状 [batch, m, k] 和 [batch, k, n] 的张量

#include "infiniop/ops/gemm/metax/gemm_metax.h"

// 1. 创建 Metax 设备句柄（假设已初始化）
infiniopHandle_t handle;
// ... handle 初始化代码 ...

// 2. 准备张量描述符（列主序布局）
int64_t c_dims[3] = {batch, m, n};
int64_t c_strides[3] = {m * n, n, 1};  // 列主序
int64_t a_dims[3] = {batch, m, k};
int64_t a_strides[3] = {m * k, k, 1};
int64_t b_dims[3] = {batch, k, n};
int64_t b_strides[3] = {k * n, n, 1};

infiniopTensorDescriptor_t c_desc, a_desc, b_desc;
infiniopCreateTensorDescriptor(handle, INFINI_DTYPE_F16, 3, c_dims, c_strides, &c_desc);
infiniopCreateTensorDescriptor(handle, INFINI_DTYPE_F16, 3, a_dims, a_strides, &a_desc);
infiniopCreateTensorDescriptor(handle, INFINI_DTYPE_F16, 3, b_dims, b_strides, &b_desc);

// 3. 创建 GEMM 描述符
op::gemm::metax::Descriptor *gemm_desc;
auto status = op::gemm::metax::Descriptor::create(handle, &gemm_desc, c_desc, a_desc, b_desc);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 4. 分配设备内存并填充数据
half *d_a, *d_b, *d_c;
size_t a_bytes = batch * m * k * sizeof(half);
size_t b_bytes = batch * k * n * sizeof(half);
size_t c_bytes = batch * m * n * sizeof(half);
// ... 使用 Metax runtime 分配内存并复制数据 ...

// 5. 获取计算流
hcStream_t stream;
// ... 获取或创建 Metax 流 ...

// 6. 执行 GEMM 计算
float alpha = 1.0f, beta = 0.0f;
status = gemm_desc->calculate(nullptr, 0, d_c, beta, d_a, d_b, alpha, stream);

// 7. 同步并读取结果
hcStreamSynchronize(stream);
// ... 将结果从设备复制回主机 ...

// 8. 清理资源
delete gemm_desc;
infiniopDestroyTensorDescriptor(c_desc);
infiniopDestroyTensorDescriptor(a_desc);
infiniopDestroyTensorDescriptor(b_desc);
```

## 5. 实现细节

- **内存管理**: 使用 PImpl 模式通过 `std::shared_ptr<device::metax::Handle::Internal>` 管理设备句柄，确保句柄生命周期正确延长。工作空间由 HCBLAS 库内部管理，用户层 `workspace_size` 固定为 0。张量内存由外部分配和释放，描述符仅持有引用。

- **并发控制**: 依赖 HCBLAS 库内部的线程安全机制。通过 `Handle::Internal::useMcblas` 方法从句柄池中获取 HCBLAS 句柄，该方法可能使用互斥锁保护句柄池访问。计算在用户提供的 `hcStream_t` 流上异步执行，通过流同步保证操作顺序。

- **性能优化**: 使用 `hcblasGemmStridedBatchedEx` API 执行批量 GEMM，支持张量核（Tensor Core）加速（`HCBLAS_GEMM_DEFAULT_TENSOR_OP` 标志）。数据类型到计算类型的映射策略：FP16/BF16 使用 32 位浮点累加（`HCBLAS_COMPUTE_32F`），FP32 使用 TF32 快速模式（`HCBLAS_COMPUTE_32F_FAST_TF32`）。矩阵布局优化：根据 `row_stride` 自动检测并设置转置标志（`HCBLAS_OP_N`/`HCBLAS_OP_T`），避免不必要的转置操作。算法复杂度为 O(batch × m × n × k)，理论峰值性能受限于 Metax GPU 的张量核计算能力和内存带宽。

- **错误处理**: 使用 `CHECK_DTYPE` 宏验证数据类型，不支持时返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`。`MatmulInfo::create` 验证张量形状匹配性（维度、步幅），失败时返回 `INFINI_STATUS_BAD_TENSOR_SHAPE` 或 `INFINI_STATUS_BAD_TENSOR_STRIDES`。HCBLAS 调用失败时通过 `CHECK_MCBLAS` 宏捕获并传播错误码。所有错误路径均保证资源正确释放（RAII 语义）。

- **依赖项**:
  - **外部库**: HCBLAS（Moore Threads BLAS 库），提供 `hcblasGemmStridedBatchedEx` 函数；HCDNN（Moore Threads DNN 库，间接依赖）。根据编译选项可选择 MCBLAS/MCDNN API（`ENABLE_METAX_MC_API` 宏）。
  - **内部模块**: `infiniop/ops/gemm/gemm.h`（DESCRIPTOR 宏定义和 `MatmulInfo` 类），`infiniop/devices/metax/metax_common.h`（设备句柄和类型映射），`infiniop/devices/metax/metax_handle.h`（Metax 设备句柄类），`infiniop/utils.h`（错误处理宏和 Result 类型）。

- **设计模式**:
  - **PImpl（Pointer to Implementation）**: 通过 `Opaque` 结构体隐藏硬件特定类型，使头文件对硬件无关代码可见
  - **工厂模式**: 静态 `create` 方法封装对象构造逻辑和参数验证
  - **策略模式**: `MatmulInfo` 类根据 `MatrixLayout` 参数自动适配矩阵布局（列主序/行主序）
  - **RAII（Resource Acquisition Is Initialization）**: 析构函数自动释放 Opaque 资源，shared_ptr 管理 Handle::Internal 生命周期
