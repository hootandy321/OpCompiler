# NVIDIA GEMM 算子核心实现文档

本模块实现了在 NVIDIA GPU 上的通用矩阵乘法（GEMM）算子，基于 cuBLAS 库提供高性能的矩阵乘法计算能力。该实现支持 FP16、BF16 和 FP32 三种数据类型，支持批量矩阵乘法，并能自动处理矩阵转置和内存布局优化。

## 1. 模块结构

- **`gemm_nvidia.cuh`**: 头文件，通过 `DESCRIPTOR(nvidia)` 宏声明 NVIDIA 硬件的 GEMM 描述符类，定义了公共接口
- **`gemm_nvidia.cu`**: 实现文件，包含描述符的创建、销毁和计算逻辑的完整实现

## 2. 核心类

### `Descriptor`
- **位置**: `gemm_nvidia.cuh`, `gemm_nvidia.cu`
- **主要功能**: NVIDIA GPU 上的 GEMM 算子描述符，封装了矩阵乘法的所有必要信息和硬件特定资源
- **继承关系**: 继承自 `InfiniopDescriptor` 基类
- **生命周期**:
  - 通过静态工厂方法 `create()` 创建
  - 构造时初始化数据类型、矩阵信息、工作空间大小和硬件特定的 Opaque 对象
  - 析构时释放 Opaque 对象中持有的资源

#### 关键成员

**私有成员**:
- `Opaque *_opaque`: 硬件特定的不透明指针，使用 PImpl 模式隐藏 NVIDIA 特定实现
- `infiniDtype_t _dtype`: 矩阵元素的数据类型（FP16/BF16/F32）
- `MatmulInfo _info`: 矩阵乘法的详细信息（M/N/K 维度、批次、步长等）
- `size_t _workspace_size`: 所需工作空间大小（当前实现为 0）

**PImpl 结构体**:
```cpp
struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
    // 持有 CUDA 设备句柄的内部实现，用于管理 cuBLAS 句柄池
};
```

#### 核心方法

**`Descriptor::create()`**
```cpp
static infiniStatus_t create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc);
```
- **功能**: 工厂方法，创建并初始化 GEMM 描述符
- **算法流程**:
  1. 类型转换：将通用句柄转换为 NVIDIA 设备句柄
  2. 数据类型验证：检查输出张量类型是否为 FP16、BF16 或 F32
  3. 矩阵信息创建：调用 `MatmulInfo::create()` 解析张量描述符，提取 M/N/K 维度、步长、批次等信息
  4. 描述符构造：创建新的 Descriptor 对象，初始化所有成员
- **复杂度**: O(1)，仅做元数据处理
- **错误处理**: 使用 `CHECK_DTYPE` 和 `CHECK_RESULT` 宏进行参数验证

**`Descriptor::~Descriptor()`**
```cpp
~Descriptor();
```
- **功能**: 析构函数，释放 Opaque 对象持有的资源
- **实现**: 简单的 `delete _opaque`
- **资源管理**: Opaque 中持有 `shared_ptr`，自动管理 cuBLAS 句柄池的生命周期

**`Descriptor::calculate()`**
```cpp
infiniStatus_t calculate(
    void *workspace, size_t workspace_size,
    void *c, float beta,
    const void *a, const void *b, float alpha,
    void *stream) const;
```
- **功能**: 执行批量矩阵乘法计算，实现 `C = alpha * A * B + beta * C`
- **核心算法**: 基于 cuBLAS 的 `cublasGemmStridedBatchedEx()` API
- **参数映射**:
  - `a, b, c`: 输入和输出矩阵的设备指针
  - `alpha, beta`: 标量系数（float 类型）
  - `stream`: CUDA 流指针，用于异步执行
  - `workspace`: 预留接口，当前未使用

**实现细节**:
1. **数据类型映射** (第 45-80 行):
   - 根据 `_dtype` 确定 CUDA 数据类型和计算类型
   - FP16/BF16: 使用 `CUDA_R_16F`/`CUDA_R_16BF`，计算类型为 `CUBLAS_COMPUTE_32F`
   - F32: 使用 `CUDA_R_32F`，计算类型为 `CUBLAS_COMPUTE_32F_FAST_TF32`（启用 Tensor Core 加速）

2. **矩阵转置处理** (第 82-87 行):
   - 如果 `_info.is_transed` 为真，交换 `a` 和 `b` 指针
   - 根据行主序/列主序布局确定 cuBLAS 操作类型：
     - `row_stride == 1`: 列主序，使用 `CUBLAS_OP_N`
     - `col_stride == 1`: 行主序，使用 `CUBLAS_OP_T`（转置）

3. **cuBLAS 调用** (第 89-118 行):
   - 使用 `_opaque->internal->useCublas()` 从句柄池获取 cuBLAS 句柄
   - 调用 `cublasGemmStridedBatchedEx()` 执行批量 GEMM：
     - `op_a, op_b`: 矩阵 A 和 B 的操作类型（是否转置）
     - `m, n, k`: 矩阵维度
     - `alpha, beta`: 标量系数
     - `a, b, c`: 数据指针
     - `lda, ldb, ldc`: 主导维度（leading dimension）
     - `stride_a, stride_b, stride_c`: 批次间的步长
     - `batch`: 批次大小
     - `compute_type`: 计算精度
     - `CUBLAS_GEMM_DEFAULT_TENSOR_OP`: 启用 Tensor Core 加速

4. **错误处理**: 使用 `CHECK_CUBLAS` 宏检查 cuBLAS API 返回状态

### `MatmulInfo`
- **位置**: `../info.h`
- **主要功能**: 封装矩阵乘法的几何信息和内存布局，独立于具体硬件实现
- **数据成员**:
  - `BlasMatrix a_matrix, b_matrix, c_matrix`: 三个矩阵的详细描述（维度、步长、批次）
  - `size_t m, n, k`: GEMM 的三个维度
  - `size_t batch`: 批次大小
  - `bool is_transed`: 标记是否进行了转置优化

**核心方法 `create()`**:
- 验证张量形状兼容性（2D 或 3D 张量）
- 检查矩阵维度匹配（C 的行数 = A 的行数，C 的列数 = B 的列数，A 的列数 = B 的行数）
- 验证批次一致性（所有矩阵的批次必须相同或为 1）
- 自动转置优化：如果 C 的内存布局与期望不符，转置所有矩阵并交换 A 和 B

### `BlasMatrix`
- **位置**: `../info.h`
- **主要功能**: 描述单个矩阵的内存布局和维度信息
- **数据成员**:
  - `size_t ndim`: 张量维度（2 或 3）
  - `size_t batch`: 批次大小
  - `ptrdiff_t stride`: 批次间的步长
  - `size_t rows, cols`: 矩阵的行数和列数
  - `ptrdiff_t row_stride, col_stride`: 行和列的步长

**关键方法**:
- `create()`: 从张量描述符创建 BlasMatrix，验证内存布局合法性
- `transpose()`: 交换行/列维度和步长
- `ld()`: 返回主导维度（leading dimension），用于 BLAS API

## 3. API 接口

```cpp
// 创建 GEMM 描述符
infiniStatus_t op::gemm::nvidia::Descriptor::create(
    infiniopHandle_t handle,              // [in]  InfiniOp 句柄
    Descriptor **desc_ptr,                // [out] 输出的描述符指针
    infiniopTensorDescriptor_t c_desc,    // [in]  输出张量 C 的描述符
    infiniopTensorDescriptor_t a_desc,    // [in]  输入张量 A 的描述符
    infiniopTensorDescriptor_t b_desc);   // [in]  输入张量 B 的描述符
// 返回: INFINI_STATUS_SUCCESS 或错误码

// 获取所需工作空间大小
size_t Descriptor::workspaceSize() const;
// 返回: 当前固定为 0

// 执行矩阵乘法计算
infiniStatus_t Descriptor::calculate(
    void *workspace,           // [in]  工作空间指针（当前未使用）
    size_t workspace_size,     // [in]  工作空间大小
    void *c,                   // [out] 输出矩阵 C 的设备指针
    float beta,                // [in]  C 的缩放系数
    const void *a,             // [in]  输入矩阵 A 的设备指针
    const void *b,             // [in]  输入矩阵 B 的设备指针
    float alpha,               // [in]  A*B 的缩放系数
    void *stream) const;       // [in]  CUDA 流指针
// 返回: INFINI_STATUS_SUCCESS 或错误码
// 计算: C = alpha * (A @ B) + beta * C
```

## 4. 使用示例

```cpp
// 示例：使用 NVIDIA GEMM 算子执行批量矩阵乘法
// 场景：计算 (batch, M, K) @ (batch, K, N) -> (batch, M, N)

#include "infiniop/ops/gemm/nvidia/gemm_nvidia.cuh"

// 1. 创建张量描述符（假设使用 FP16）
size_t batch = 32, M = 512, K = 768, N = 1024;
size_t shape_a[3] = {batch, M, K};
size_t shape_b[3] = {batch, K, N};
size_t shape_c[3] = {batch, M, N};

ptrdiff_t strides_a[3] = {M * K, K, 1};  // 列主序布局
ptrdiff_t strides_b[3] = {K * N, N, 1};
ptrdiff_t strides_c[3] = {M * N, N, 1};

infiniopTensorDescriptor_t a_desc, b_desc, c_desc;
infiniopCreateTensorDescriptor(&a_desc, INFINI_DTYPE_F16, 3, shape_a, strides_a);
infiniopCreateTensorDescriptor(&b_desc, INFINI_DTYPE_F16, 3, shape_b, strides_b);
infiniopCreateTensorDescriptor(&c_desc, INFINI_DTYPE_F16, 3, shape_c, strides_c);

// 2. 创建 GEMM 描述符
op::gemm::nvidia::Descriptor *gemm_desc;
auto status = op::gemm::nvidia::Descriptor::create(
    handle, &gemm_desc, c_desc, a_desc, b_desc);

// 3. 分配 GPU 内存
void *d_a, *d_b, *d_c;
size_t size_a = batch * M * K * sizeof(fp16_t);
size_t size_b = batch * K * N * sizeof(fp16_t);
size_t size_c = batch * M * N * sizeof(fp16_t);
cudaMalloc(&d_a, size_a);
cudaMalloc(&d_b, size_b);
cudaMalloc(&d_c, size_c);

// 4. 拷贝数据到 GPU（假设 h_a, h_b 是主机端数据）
cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

// 5. 创建 CUDA 流
cudaStream_t stream;
cudaStreamCreate(&stream);

// 6. 执行矩阵乘法：C = 1.0 * A * B + 0.0 * C
float alpha = 1.0f, beta = 0.0f;
status = gemm_desc->calculate(
    nullptr, 0,      // 无需工作空间
    d_c, beta,        // 输出和 beta
    d_a, d_b, alpha,  // 输入和 alpha
    stream);          // CUDA 流

// 7. 同步并取回结果
cudaStreamSynchronize(stream);
cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);

// 8. 清理资源
delete gemm_desc;
cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
cudaStreamDestroy(stream);
infiniopDestroyTensorDescriptor(a_desc);
infiniopDestroyTensorDescriptor(b_desc);
infiniopDestroyTensorDescriptor(c_desc);
```

## 5. 实现细节

### 内存管理
- **零拷贝设计**: 当前实现不需要额外工作空间，`workspace_size` 固定为 0
- **共享指针**: Opaque 结构体使用 `std::shared_ptr` 管理 NVIDIA 句柄的内部实现，确保多个算子可以共享同一个句柄池
- **PImpl 模式**: 通过 `Opaque` 前向声明和指针隐藏 NVIDIA 特定类型，避免在头文件中暴露 CUDA/cuBLAS 类型

### 并发控制
- **句柄池机制**: 通过 `device::nvidia::Handle::Internal` 的 `Pool<cublasHandle_t>` 管理多个 cuBLAS 句柄
- **流式执行**: 所有计算在用户提供的 CUDA 流上异步执行，支持并发内核和流水线重叠
- **线程安全**: cuBLAS 句柄从池中获取和归还，支持多线程并发调用不同流上的 GEMM

### 性能优化
- **Tensor Core 加速**:
  - FP16: 使用 `CUBLAS_COMPUTE_32F` 计算类型
  - BF16: 使用 `CUBLAS_COMPUTE_32F` 计算类型
  - F32: 使用 `CUBLAS_COMPUTE_32F_FAST_TF32`，启用 TF32 模式加速矩阵乘法
- **批量处理**: 使用 `cublasGemmStridedBatchedEx()` 一次性计算多个矩阵乘法，减少内核启动开销
- **自动布局优化**:
  - `MatmulInfo::create()` 自动检测并转置矩阵以匹配 cuBLAS 的列主序预期
  - 根据 `row_stride` 和 `col_stride` 自动判断是否需要转置操作
- **零拷贝转置**: 通过交换指针和调整 `op_a/op_b` 参数实现逻辑转置，无需实际内存拷贝

### 错误处理
- **类型验证**: `CHECK_DTYPE` 宏限制只支持 FP16、BF16 和 F32 三种类型
- **形状验证**: `MatmulInfo::create()` 检查维度兼容性、批次一致性、内存布局合法性
- **API 错误检查**:
  - `CHECK_RESULT`: 检查 `utils::Result<T>` 类型返回值
  - `CHECK_CUBLAS`: 检查 cuBLAS API 调用状态，返回 `INFINI_STATUS_BAD_TENSOR_DTYPE` 等错误码

### 跨平台兼容性
- **硬件后端抽象**: 通过预编译宏支持不同的 GPU 厂商 API：
  - `ENABLE_ILUVATAR_API`: 熙泰（Iluvatar）GPU
  - `ENABLE_HYGON_API`: 海光（Hygon）GPU
  - 标准 NVIDIA: 使用原生 `cublasComputeType_t`
- **条件编译**: 在第 46-50 行和 54-76 行根据编译选项选择不同的数据类型枚举

### 设计模式
- **PImpl (Pointer to Implementation)**: Opaque 结构体隐藏硬件特定实现，头文件只暴露公共接口
- **工厂模式**: `create()` 静态方法封装复杂的对象构造逻辑
- **RAII**: 析构函数自动管理资源释放，使用智能指针管理句柄生命周期
- **策略模式**: 通过 `DESCRIPTOR(NAMESPACE)` 宏为不同硬件生成独立的描述符类，共享接口但实现独立
