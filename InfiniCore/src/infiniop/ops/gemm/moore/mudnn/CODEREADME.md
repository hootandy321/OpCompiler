# Moore muDNN GEMM 算子实现文档

本模块实现了基于 Moore 架构（使用 MUSA/muDNN）的通用矩阵乘法（GEMM）算子，支持半精度浮点（FP16）、单精度浮点（FP32）和 BFloat16（BF16）三种数据类型。该实现封装了 Moore 硬件平台的 muDNN BatchMatMul 算子，提供统一的矩阵乘法接口。

## 1. 模块结构

- **`gemm_mudnn.h`**: 头文件，通过 `DESCRIPTOR(mudnn)` 宏声明 `Descriptor` 类，定义公开接口
- **`gemm_mudnn.mu`**: 实现文件，包含 muDNN 特定的 GEMM 算子实现逻辑，负责与 muDNN API 交互

## 2. 核心类与数据结构

### `Descriptor` 类（通过宏定义生成）
- **命名空间**: `op::gemm::mudnn`
- **继承关系**: 继承自 `InfiniopDescriptor`
- **主要功能**: 封装 muDNN GEMM 算子的描述符，管理算子生命周期和执行参数

#### 关键成员变量
- **`Opaque *_opaque`**: 不透明指针，指向 muDNN 特定的内部状态（PImpl 模式）
- **`infiniDtype_t _dtype`**: 数据类型（FP16/FP32/BF16）
- **`MatmulInfo _info`**: 矩阵乘法运算信息（BMNK 维度、步长、布局等）
- **`size_t _workspace_size`**: 工作空间大小（当前实现中固定为 0）

### `Descriptor::Opaque` 结构体
```cpp
struct Descriptor::Opaque {
    std::shared_ptr<device::moore::Handle::Internal> internal;
};
```
- **作用**: 保存 Moore 设备句柄的内部状态，用于管理 muDNN 句柄的生命周期
- **生命周期**: 由 `Descriptor` 构造时分配，析构时释放

### `MatmulInfo` 类（定义在 `info.h`）
- **位置**: `/home/qy/src/Infini/InfiniCore/src/infiniop/ops/gemm/info.h`
- **主要功能**: 封装矩阵乘法的形状和布局信息

#### 关键成员
- **`BlasMatrix a_matrix, b_matrix, c_matrix`**: 三个矩阵的元数据（维度、步长、布局）
- **`size_t m, n, k, batch`**: 矩阵乘法的维度信息
  - `m`: 矩阵 A 的行数 / 矩阵 C 的行数
  - `n`: 矩阵 B 的列数 / 矩阵 C 的列数
  - `k`: 矩阵 A 的列数 / 矩阵 B 的行数
  - `batch`: 批处理大小
- **`bool is_transed`**: 是否进行了转置适配标志

### `BlasMatrix` 结构体（定义在 `info.h`）
```cpp
struct BlasMatrix {
    size_t ndim;           // 张量维度数（2 或 3）
    size_t batch;          // 批次大小
    ptrdiff_t stride;      // 批次步长
    size_t rows, cols;     // 矩阵行数和列数
    ptrdiff_t row_stride;  // 行步长（连续存储时为 1）
    ptrdiff_t col_stride;  // 列步长（连续存储时为 1）

    ptrdiff_t ld() const;  // 返回主维度（leading dimension）
};
```

## 3. API 接口

### `Descriptor::create()` - 算子描述符创建
```cpp
static infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,           // [输入] Moore 设备句柄
    Descriptor **desc_ptr,              // [输出] 创建的描述符指针
    infiniopTensorDescriptor_t c_desc,  // [输入] 输出矩阵 C 的张量描述符
    infiniopTensorDescriptor_t a_desc,  // [输入] 输入矩阵 A 的张量描述符
    infiniopTensorDescriptor_t b_desc   // [输入] 输入矩阵 B 的张量描述符
);
```
**返回值**: `INFINI_STATUS_SUCCESS` 或错误码
**功能**:
1. 验证数据类型（支持 FP16/FP32/BF16）
2. 调用 `MatmulInfo::create()` 生成矩阵元数据（强制行主序布局）
3. 构造 `Descriptor` 对象，初始化 `_opaque` 成员（持有 Moore 设备内部句柄）

### `Descriptor::calculate()` - 矩阵乘法计算
```cpp
infiniStatus_t Descriptor::calculate(
    void *workspace,        // [输入] 工作空间指针（未使用）
    size_t workspace_size,  // [输入] 工作空间大小（未使用）
    void *c,                // [输出] 输出矩阵 C 的设备指针
    float beta,             // [输入] 缩放系数 beta（C = alpha * op(A) @ op(B) + beta * C）
    const void *a,          // [输入] 输入矩阵 A 的设备指针
    const void *b,          // [输入] 输入矩阵 B 的设备指针
    float alpha,            // [输入] 缩放系数 alpha
    void *stream            // [输入] MUSA 流指针
) const;
```
**返回值**: `INFINI_STATUS_SUCCESS` 或错误码
**功能**:
1. 根据 `_dtype` 分派到对应的模板函数实例化
2. 调用 `calculate<Tdata>()` 执行实际的 muDNN 矩阵乘法

### `calculate<Tdata>()` - 模板计算函数（内部）
```cpp
template <typename Tdata>
infiniStatus_t calculate(
    const MatmulInfo &info,                              // [输入] 矩阵元数据
    std::shared_ptr<device::moore::Handle::Internal> &_internal, // [输入] Moore 设备内部句柄
    void *c, float beta, const void *a, const void *b, float alpha, void *stream
);
```
**支持的模板参数**: `half` (FP16), `float` (FP32), `__mt_bfloat16` (BF16)

**执行流程**:
1. **创建 BatchMatMul 算子**: `std::make_unique<::musa::dnn::BatchMatMul>()`
2. **设置计算模式**: `SetComputeMode(::musa::dnn::BatchMatMul::ComputeMode::TENSOR)`
3. **获取 muDNN 句柄**: 通过 `_internal->useMudnn()` 在 RAII 作用域内获取 muDNN 句柄
4. **配置张量类型**: 根据模板参数设置三个张量的数据类型（HALF/FLOAT/BFLOAT16）
5. **绑定张量地址**: `SetAddr()` 设置设备内存指针
6. **配置张量形状**:
   - 根据 `col_stride` 判断矩阵布局（行主序或列主序）
   - 调用 `SetNdInfo()` 设置维度数组和步长数组
   - 左矩阵形状: `[batch, k, m]` (列主序) 或 `[batch, m, k]` (行主序)
   - 右矩阵形状: `[batch, n, k]` (列主序) 或 `[batch, k, n]` (行主序)
   - 输出矩阵形状: 固定为 `[batch, m, n]` (行主序)
7. **配置转置标志**:
   - `SetTranspose(transA, transB)` 根据 `col_stride` 判断是否需要转置
   - 行主序（`col_stride == 1`）: `trans = false`
   - 列主序（`col_stride != 1`）: `trans = true`
8. **分配工作空间**: 使用 lambda 函数创建 `MemoryMaintainer`，通过 `musaMalloc/musaFree` 管理临时内存
9. **查询工作空间大小**: `GetWorkspaceSize()` 获取所需字节数
10. **设置缩放系数**: `SetAlpha(alpha)`, `SetBeta(beta)`, `SetGamma(0.0)`
11. **执行计算**: `Run()` 启动 BatchMatMul 核函数
    - 传入维度参数: `batch, m, n, k`
    - 传入主维度: `ld(A), ld(B), ld(C)`
    - 传入批次步长: `stride(A), stride(B), stride(C)`
    - 传入内存维护器: `maintainer`

### `Descriptor::~Descriptor()` - 析构函数
```cpp
Descriptor::~Descriptor();
```
**功能**: 释放 `_opaque` 指向的内存

## 4. 使用示例

```cpp
// 1. 创建 Moore 设备句柄
InfiniopHandle *handle;
device::moore::Handle::create(&handle, device_id);

// 2. 创建张量描述符（假设 shape 为 [batch, m, k], [batch, k, n], [batch, m, n]）
infiniopTensorDescriptor_t a_desc, b_desc, c_desc;
// ... 初始化张量描述符（省略）

// 3. 创建 GEMM 描述符
op::gemm::mudnn::Descriptor *gemm_desc;
auto status = op::gemm::mudnn::Descriptor::create(
    handle, &gemm_desc, c_desc, a_desc, b_desc
);
if (status != INFINI_STATUS_SUCCESS) {
    // 错误处理
}

// 4. 分配设备内存
void *d_a, *d_b, *d_c;
size_t size_a = batch * m * k * sizeof(half);
size_t size_b = batch * k * n * sizeof(half);
size_t size_c = batch * m * n * sizeof(half);
musaMalloc(&d_a, size_a);
musaMalloc(&d_b, size_b);
musaMalloc(&d_c, size_c);

// 5. 拷贝数据到设备（省略）
// cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);

// 6. 获取或创建 MUSA 流
musaStream_t stream;
musaStreamCreate(&stream);

// 7. 执行矩阵乘法 C = 1.0 * A @ B + 0.0 * C
status = gemm_desc->calculate(
    nullptr, 0,      // 工作空间（未使用）
    d_c, 0.0f,       // 输出矩阵，beta = 0.0
    d_a, d_b, 1.0f,  // 输入矩阵，alpha = 1.0
    stream           // MUSA 流
);

// 8. 同步并拷贝结果回主机（省略）
// musaStreamSynchronize(stream);
// cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);

// 9. 清理资源
musaFree(d_a);
musaFree(d_b);
musaFree(d_c);
musaStreamDestroy(stream);
delete gemm_desc;
delete handle;
```

## 5. 实现细节

### 内存管理策略
- **设备内存分配**: 使用 muDNN 的 `MemoryMaintainer` 机制，通过 lambda 函数封装 `musaMalloc/musaFree`
- **RAII 模式**: `MemoryHandler` 使用自定义删除器，确保异常安全
- **工作空间**: 动态分配，算子内部通过 `GetWorkspaceSize()` 查询需求

### 并发与同步
- **流式执行**: 支持传入 `musaStream_t` 流指针，实现异步执行
- **muDNN 句柄管理**: 通过 `_internal->useMudnn(stream, callback)` 在回调作用域内获取 muDNN 句柄，保证句柄生命周期安全
- **线程安全**: 每个描述符独立持有内部状态，无共享全局状态

### 性能优化
- **批量处理**: 使用 `BatchMatMul` 算子，充分利用张量核心（Tensor Core）加速能力
- **计算模式**: 设置为 `TENSOR` 模式（相对于 `SM` 模式），针对张量形状优化
- **布局自适应**: 根据 `col_stride` 自动检测矩阵布局（行主序/列主序），避免显式转置
- **零拷贝转置**: 通过 `SetTranspose()` 配置逻辑转置，无需物理移动数据

### 错误处理
- **数据类型验证**: 仅支持 FP16/FP32/BF16，其他类型返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
- **形状验证**: `MatmulInfo::create()` 检查矩阵维度兼容性
  - C 的行数必须等于 A 的行数
  - C 的列数必须等于 B 的列数
  - A 的列数必须等于 B 的行数
  - 批次大小必须匹配（允许广播 1）
- **步长验证**: 要求至少一个维度是连续的（`row_stride == 1 || col_stride == 1`）
- **错误传播**: 使用 `CHECK_DTYPE` 和 `CHECK_RESULT` 宏进行早期返回

### 依赖关系
- **外部库**:
  - `musa_bfloat16.h`: Moore BFloat16 数据类型定义
  - `mudnn_base.h`, `mudnn_math.h`, `mudnn.h`: muDNN 核心 API
- **内部模块**:
  - `../../gemm.h`: DESCRIPTOR 宏定义和基础接口
  - `../../../../devices/moore/moore_handle.h`: Moore 设备句柄
  - `../../../../devices/moore/moore_common.h`: Moore 通用定义（如 `CHECK_DTYPE` 宏）
  - `info.h`: 矩阵元数据结构

### 设计模式
- **PImpl (Pointer to Implementation)**: 通过 `Opaque` 结构体隐藏硬件相关类型
  - 头文件仅声明 `struct Opaque;`，不暴露实现细节
  - 实现文件中定义 `Opaque`，持有 `device::moore::Handle::Internal`
  - 保证头文件可被硬件无关代码（operator.cc）包含
- **策略模式**: 通过模板函数 `calculate<Tdata>` 支持多种数据类型，避免虚函数开销
- **RAII (Resource Acquisition Is Initialization)**: `MemoryMaintainer` lambda 确保动态分配的内存在作用域结束时释放

### 矩阵布局处理逻辑
本实现强制输出矩阵为行主序（`MatrixLayout::ROW_MAJOR`），并通过以下逻辑适配输入矩阵：

1. **布局检测**: 通过 `col_stride == 1` 判断是否为行主序
2. **转置配置**: 根据 `col_stride` 设置 `SetTranspose(transA, transB)`
   - `col_stride == 1` (行主序): `trans = false`
   - `col_stride != 1` (列主序): `trans = true`
3. **维度调整**: 根据 `col_stride` 调整传入 muDNN 的维度数组
   - 列主序左矩阵: `[batch, k, m]` (存储维度为 k×m)
   - 行主序左矩阵: `[batch, m, k]` (存储维度为 m×k)

这种设计允许算子无缝处理行主序和列主序的输入矩阵，无需显式转置操作。
