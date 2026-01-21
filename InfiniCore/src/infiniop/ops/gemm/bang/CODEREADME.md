# GEMM BANG 后端实现文档

本模块实现了通用矩阵乘法（GEMM）算子在寒武纪（Cambricon）MLU 硬件平台上的 BANG 后端支持。该实现通过 CNNL（Cambricon CNN Library）提供的高性能矩阵乘法接口，支持 2D 和 3D 张量的批量矩阵乘运算，并采用启发式算法选择最优计算内核。

## 1. 模块结构

- **`gemm_bang.h`**: 头文件，通过 `DESCRIPTOR(bang)` 宏声明 BANG 后端的 GEMM 描述符类，定义公共接口
- **`gemm_bang.cc`**: 实现文件，包含完整的 GEMM 算子在 BANG 硬件上的具体实现逻辑

## 2. 核心类

### `Descriptor::Opaque`
- **位置**: `gemm_bang.cc` (第 7-22 行)
- **主要功能**: PImpl 模式的内部实现类，封装所有 CNNL 相关的硬件特定类型和描述符，对外隐藏实现细节
- **关键成员**:
  - `cnnlMatMulDescriptor_t op`: CNNL 矩阵乘法操作描述符，定义矩阵乘计算的配置
  - `cnnlMatMulAlgo_t algo`: CNNL 矩阵乘算法描述符，通过启发式搜索选择的最优算法
  - `cnnlMatMulHeuristicResult_t algoResult`: 启发式算法搜索结果，包含算法选择和所需工作空间大小
  - `cnnlTensorDescriptor_t a, b, c`: 分别为矩阵 A、B、C 的张量描述符，描述形状、布局、数据类型
  - `std::shared_ptr<device::bang::Handle::Internal> internal`: BANG 设备句柄的内部实现，管理 CNNL 句柄池
- **生命周期**:
  - 在 `Descriptor::create()` 中构造并初始化所有 CNNL 资源
  - 析构函数自动释放所有 CNNL 描述符，遵循 RAII 原则

### `Descriptor`
- **位置**: `gemm_bang.h` (通过 `DESCRIPTOR(bang)` 宏展开)
- **主要功能**: BANG 后端的 GEMM 算子描述符，继承自 `InfiniopDescriptor`，封装算子元数据和执行接口
- **关键成员**:
  - `Opaque *_opaque`: PImpl 指针，指向硬件特定的实现细节
  - `infiniDtype_t _dtype`: 输出数据类型（支持 F16、BF16、F32）
  - `MatmulInfo _info`: 矩阵乘运算的元数据（BMNK 维度、批处理大小、布局信息）
  - `size_t _workspace_size`: CNNL 算法所需的工作空间大小（字节）
- **核心方法**:
  - `~Descriptor()`: 析构函数，释放 `_opaque` 指针
  - `size_t workspaceSize() const`: 返回所需工作空间大小
  - `static infiniStatus_t create(...)`: 静态工厂方法，根据张量描述符创建并初始化算子描述符
  - `infiniStatus_t calculate(...)`: 执行矩阵乘计算，支持 alpha 和 beta 缩放因子
- **生命周期**: 通过静态工厂方法创建，用户负责调用析构销毁

## 3. API 接口

```cpp
// 工厂方法：创建 BANG 后端的 GEMM 描述符
static infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,                    // BANG 设备句柄
    Descriptor **desc_ptr,                      // [输出] 创建的描述符指针
    infiniopTensorDescriptor_t c_desc,          // 输出张量 C 的描述符
    infiniopTensorDescriptor_t a_desc,          // 输入张量 A 的描述符
    infiniopTensorDescriptor_t b_desc);         // 输入张量 B 的描述符
// 返回 INFINI_STATUS_SUCCESS 成功，否则返回错误码

// 执行矩阵乘计算：C = alpha * (A @ B) + beta * C
infiniStatus_t Descriptor::calculate(
    void *workspace,            // CNNL 工作空间缓冲区
    size_t workspace_size,      // 工作空间大小（字节）
    void *c,                    // 输出张量 C 的设备指针
    float beta,                 // C 的缩放因子
    const void *a,              // 输入张量 A 的设备指针
    const void *b,              // 输入张量 B 的设备指针
    float alpha,                // A @ B 的缩放因子
    void *stream) const;        // CNRT 计算流
// 返回 INFINI_STATUS_SUCCESS 成功，否则返回错误码

// 查询所需工作空间大小
size_t workspaceSize() const;
// 返回 CNNL 算法所需的工作空间字节数
```

## 4. 使用示例

```cpp
// 示例：在 BANG 硬件上执行批量矩阵乘 C = A @ B
// 假设张量形状: A[batch, M, K], B[batch, K, N], C[batch, M, N]

// 1. 准备张量描述符（行主序布局）
std::vector<size_t> a_shape = {batch, M, K};
std::vector<size_t> b_shape = {batch, K, N};
std::vector<size_t> c_shape = {batch, M, N};
std::vector<ptrdiff_t> a_strides = {M * K, K, 1};
std::vector<ptrdiff_t> b_strides = {K * N, N, 1};
std::vector<ptrdiff_t> c_strides = {M * N, N, 1};

auto a_desc = new InfiniopTensorDescriptor(INFINI_DTYPE_F16, 3, a_shape.data(), a_strides.data());
auto b_desc = new InfiniopTensorDescriptor(INFINI_DTYPE_F16, 3, b_shape.data(), b_strides.data());
auto c_desc = new InfiniopTensorDescriptor(INFINI_DTYPE_F16, 3, c_shape.data(), c_strides.data());

// 2. 创建 GEMM 算子描述符（自动选择最优 CNNL 算法）
op::gemm::bang::Descriptor *gemm_desc = nullptr;
auto status = op::gemm::bang::Descriptor::create(handle, &gemm_desc, c_desc, a_desc, b_desc);
if (status != INFINI_STATUS_SUCCESS) {
    // 错误处理
}

// 3. 分配 CNNL 工作空间
size_t workspace_size = gemm_desc->workspaceSize();
void *workspace = nullptr;
cnrtMalloc(&workspace, workspace_size);

// 4. 分配设备内存并拷贝数据（假设已有数据）
void *d_a, *d_b, *d_c;
cnrtMalloc(&d_a, a_desc->numel() * sizeof(half));
cnrtMalloc(&d_b, b_desc->numel() * sizeof(half));
cnrtMalloc(&d_c, c_desc->numel() * sizeof(half));
cnrtMemcpy(d_a, h_a, ..., CNRT_MEM_HOST2DEV);
cnrtMemcpy(d_b, h_b, ..., CNRT_MEM_HOST2DEV);

// 5. 在 CNRT 流上执行矩阵乘
cnrtQueue_t stream;
cnrtQueueCreate(&stream);
status = gemm_desc->calculate(workspace, workspace_size, d_c, 0.0f, d_a, d_b, 1.0f, stream);
if (status != INFINI_STATUS_SUCCESS) {
    // 错误处理
}

// 6. 同步并取回结果
cnrtQueueSync(stream);
cnrtMemcpy(h_c, d_c, ..., CNRT_MEM_DEV2HOST);

// 7. 清理资源
cnrtFree(d_a);
cnrtFree(d_b);
cnrtFree(d_c);
cnrtFree(workspace);
cnrtQueueDestroy(stream);
delete gemm_desc;
delete a_desc;
delete b_desc;
delete c_desc;
```

## 5. 实现细节

### 内存管理
- **RAII 资源管理**: `Opaque` 结构体的析构函数自动释放所有 CNNL 描述符（张量、矩阵乘、算法、启发式结果），避免资源泄漏
- **句柄池共享**: 通过 `std::shared_ptr<device::bang::Handle::Internal>` 共享设备句柄的 CNNL 句柄池，避免重复创建
- **工作空间分配**: 用户负责根据 `workspaceSize()` 分配足够大的设备内存，CNNL 在计算时用作临时存储

### 并发与同步
- **流式执行**: `calculate()` 方法接受 CNRT 流参数，支持与其它操作在同一个流中排队执行，保证顺序性
- **队列同步**: 调用 `cnrtQueueSync()` 同步等待计算完成，确保结果可访问后才返回
- **句柄池并发安全**: `device::bang::Handle::Internal` 内部使用 `Pool<cnnlHandle_t>` 管理线程安全的 CNNL 句柄池

### 性能优化
- **启发式算法选择**: 使用 `cnnlGetBatchMatMulAlgoHeuristic()` 根据张量形状、数据类型、硬件特性自动选择最优矩阵乘算法，避免用户手动调优
- **批量计算**: 通过 `cnnlBatchMatMulBCast_v2` 支持批量矩阵乘，充分利用 MLU 的并行计算能力
- **stride 支持**: 通过 `CNNL_MATMUL_USE_STRIDE` 属性启用 stride 模式，支持非连续张量，避免不必要的内存拷贝
- **布局转换**: 在 `MatmulInfo::create()` 中自动检测张量布局（行主序/列主序），必要时进行转置优化（`is_transed` 标志）

### 错误处理
- **错误传播**: 所有 CNNL API 调用通过 `CHECK_BANG` 宏检查返回值，失败时返回 `infiniStatus_t` 错误码
- **数据类型验证**: `create()` 方法验证输出数据类型，仅支持 F16、BF16、F32，否则返回 `INFINI_STATUS_BAD_DTYPE`
- **形状验证**: `MatmulInfo::create()` 验证矩阵维度兼容性（M、N、K 对齐、批处理大小匹配），失败时返回 `INFINI_STATUS_BAD_TENSOR_SHAPE`
- **stride 约束**: `BlasMatrix::create()` 要求至少一个维度的 stride 为 1（行或列连续），否则返回 `INFINI_STATUS_BAD_TENSOR_STRIDES`

### 依赖关系
- **CNNL (Cambricon CNN Library)**: 核心依赖，提供高性能矩阵乘法 API
  - `cnnlCreateTensorDescriptor` / `cnnlSetTensorDescriptorEx`: 创建和配置张量描述符
  - `cnnlMatMulDescCreate` / `cnnlSetMatMulDescAttr`: 创建矩阵乘操作描述符
  - `cnnlMatMulAlgoCreate` / `cnnlGetBatchMatMulAlgoHeuristic`: 算法选择
  - `cnnlBatchMatMulBCast_v2`: 执行批量矩阵乘
- **CNRT (Cambricon Runtime)**: 设备管理和流 API
  - `cnrtQueue_t`: 计算流，用于异步执行
  - `cnrtQueueSync`: 流同步
  - `cnrtMalloc` / `cnrtFree`: 设备内存管理
- **设备抽象层**: `device::bang::Handle` 和 `device::bang::Handle::Internal`，封装 CNNL 句柄管理和设备信息
- **通用算子框架**: `op::gemm::MatmulInfo` 和 `op::gemm::BlasMatrix`，提供硬件无关的矩阵乘元数据计算和布局处理

### 设计模式
- **PImpl (Pointer to Implementation)**: `Descriptor` 类通过 `Opaque` 前向声明隐藏 CNNL 特定类型，实现接口与实现分离，头文件不暴露硬件细节
- **工厂模式**: `Descriptor::create()` 静态工厂方法封装复杂的创建逻辑，包括验证、资源分配、算法选择
- **策略模式**: 通过 `cnnlMatMulAlgo_t` 支持多种矩阵乘算法，运行时通过启发式选择最优策略
- **RAII (Resource Acquisition Is Initialization)**: 资源（描述符、句柄）在对象构造时获取，析构时自动释放

### 算法复杂度
- **矩阵乘法**: 标准 O(M × N × K) 浮点运算，CNNL 根据硬件优化选择不同的分块和并行化策略
- **启发式搜索**: `cnnlGetBatchMatMulAlgoHeuristic()` 时间复杂度取决于内部候选算法数量，通常为 O(1) 或 O(log n)
- **空间复杂度**: 工作空间大小由启发式结果决定，通常为 O(1) 或 O(min(M, N, K))，具体取决于算法
