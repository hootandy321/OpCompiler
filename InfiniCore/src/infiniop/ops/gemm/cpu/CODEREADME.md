# CPU GEMM 算子核心实现文档

本文档详细描述了 Infini 框架中 CPU 设备上的通用矩阵乘法（GEMM: General Matrix Multiply）算子实现。该模块提供了在 CPU 上执行批量矩阵乘法运算的核心功能，支持多种浮点数据类型（F16、BF16、F32），并采用 OpenMP 并行化以充分利用多核处理器性能。

## 1. 模块结构

- **`gemm_cpu.h`**: CPU GEMM 算子的声明文件，通过宏展开定义 `Descriptor` 类结构
- **`gemm_cpu.cc`**: CPU GEMM 算子的核心实现，包含描述符创建和矩阵乘法计算逻辑

## 2. 核心类与数据结构

### 2.1 `Descriptor` 类

**位置**: `gemm_cpu.h` (通过 `DESCRIPTOR(cpu)` 宏定义生成)

**主要功能**: 封装 CPU 设备上 GEMM 运算的所有必要信息，提供算子创建和计算的公共接口。

**关键成员**:
- `struct Opaque *_opaque`: PImpl 模式的不透明指针，用于隐藏硬件相关的私有实现细节（CPU 版本中未使用，设为 `nullptr`）
- `infiniDtype_t _dtype`: 矩阵元素的数据类型（F16/BF16/F32）
- `MatmulInfo _info`: 矩阵乘法的几何信息（BMNK 维度、步长、布局等）
- `size_t _workspace_size`: 工作空间大小（CPU 实现中为 0，无需额外缓冲区）

**继承关系**:
```
InfiniopDescriptor (基类)
    ↓
op::gemm::cpu::Descriptor
```

**核心方法**:

- **`~Descriptor()`**: 析构函数（默认实现，无特殊清理逻辑）

- **`static infiniStatus_t create(...)`**:
  - **功能**: 工厂方法，根据输入张量描述符创建 GEMM 算子描述符
  - **参数**:
    - `infiniopHandle_t handle_`: CPU 设备句柄（转换为 `device::cpu::Handle*`）
    - `Descriptor **desc_ptr`: 输出参数，返回创建的描述符指针
    - `infiniopTensorDescriptor_t c_desc, a_desc, b_desc`: 输出矩阵 C 和输入矩阵 A、B 的张量描述符
  - **执行流程**:
    1. 将句柄转换为 CPU 设备句柄
    2. 验证数据类型必须是 F16、BF16 或 F32（`CHECK_DTYPE` 宏）
    3. 调用 `MatmulInfo::create()` 创建矩阵乘法信息，强制使用列主序（`MatrixLayout::COL_MAJOR`）
    4. 验证矩阵形状兼容性（`CHECK_RESULT` 宏）
    5. 创建 `Descriptor` 实例并返回
  - **时间复杂度**: O(1)
  - **返回值**: 成功返回 `INFINI_STATUS_SUCCESS`，失败返回相应错误码

- **`infiniStatus_t calculate(...) const`**:
  - **功能**: 执行矩阵乘法计算 C = alpha * A @ B + beta * C
  - **参数**:
    - `void *workspace, size_t workspace_size`: 工作空间缓冲区及其大小（CPU 实现中未使用）
    - `void *c`: 输出矩阵 C 的数据指针
    - `float beta`: 标量 beta，用于累加到原 C 值
    - `const void *a, const void *b`: 输入矩阵 A、B 的数据指针
    - `float alpha`: 标量 alpha，用于缩放乘积结果
    - `void *stream`: 执行流（CPU 实现中未使用）
  - **执行流程**:
    1. 根据 `_dtype` 分发到对应的模板特化版本：
       - `INFINI_DTYPE_F16` → 调用 `cpu::calculate<fp16_t>()`
       - `INFINI_DTYPE_BF16` → 调用 `cpu::calculate<bf16_t>()`
       - `INFINI_DTYPE_F32` → 调用 `cpu::calculate<float>()`
    2. 其他数据类型返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
  - **返回值**: 成功返回 `INFINI_STATUS_SUCCESS`

**生命周期**:
- 由 `Descriptor::create()` 工厂方法动态分配（`new Descriptor`）
- 所有权转移给调用者，调用者负责释放（`delete`）
- 析构函数为默认实现，C++ RAII 机制自动管理成员资源

### 2.2 `MatmulInfo` 结构

**位置**: `../info.h`

**功能**: 封装矩阵乘法的几何布局和形状信息，提供矩阵形状验证和转置优化逻辑。

**关键成员**:
- `BlasMatrix a_matrix, b_matrix, c_matrix`: 三个矩阵的布局信息（行/列步长、维度）
- `size_t m, n, k`: 矩阵乘法的核心维度（A: m×k, B: k×n, C: m×n）
- `size_t batch`: 批量维度大小（支持批量矩阵乘法）
- `bool is_transed`: 标记是否进行了转置优化（交换 A/B）

**关键方法**:
- **`static utils::Result<MatmulInfo> create(...)`**:
  - **功能**: 从张量描述符创建并验证矩阵乘法信息
  - **转置优化**: 当检测到 C 矩阵不符合目标布局时，自动转置所有矩阵并交换 A/B，将运算转换为符合硬件习惯的形式
  - **返回值**: 成功返回 `MatmulInfo` 实例，失败返回错误码

### 2.3 `BlasMatrix` 结构

**位置**: `../info.h`

**功能**: 描述 BLAS 级别的矩阵布局，支持 2D 和批量 3D 张量。

**关键成员**:
- `size_t ndim`: 张量维度数（2 或 3）
- `size_t batch`: 批量大小（3D 张量有效）
- `ptrdiff_t stride`: 批量维度的步长
- `size_t rows, cols`: 矩阵的行数和列数
- `ptrdiff_t row_stride, col_stride`: 行和列方向的步长（用于支持非连续内存）

**关键方法**:
- **`static utils::Result<BlasMatrix> create(...)`**: 从张量描述符提取矩阵信息，验证步长合法性（行/列步长必须有一个为 1）
- **`void transpose()`**: 交换行/列维度和步长
- **`ptrdiff_t ld() const`**: 返回主维度步长（leading dimension）

## 3. 核心算法实现

### 3.1 模板函数 `calculate<Tdata>`

**位置**: `gemm_cpu.cc:29-70`

**功能**: 通用的 CPU 矩阵乘法实现，支持多种数据类型，使用 OpenMP 并行化。

**算法复杂度**: O(batch × m × n × k)

**核心逻辑**:

```cpp
template <typename Tdata>
void calculate(
    const MatmulInfo &info,
    void *c, float beta,
    const void *a, const void *b,
    float alpha)
```

**实现细节**:

1. **转置处理**（第 37-39 行）:
   - 如果 `info.is_transed` 为真，交换 A 和 B 的指针
   - 这是因为在创建 `MatmulInfo` 时可能为了优化内存布局而交换了矩阵顺序

2. **OpenMP 并行化**（第 41 行）:
   - 使用 `#pragma omp parallel for` 将最外层循环（C 矩阵的每个元素）并行化
   - 线程数由 OpenMP 运行时自动管理（通常等于 CPU 核心数）
   - `schedule(static)` 默认调度策略，均匀分配迭代次数

3. **多维索引计算**（第 42-48 行）:
   - 将扁平化索引 `index`（范围：0 到 batch×m×n-1）分解为三维索引：
     - `i`: 批量索引（0 到 batch-1）
     - `m_`: C 矩阵的行索引（0 到 m-1）
     - `n_`: C 矩阵的列索引（0 到 n-1）
   - 使用整数除法和模运算逆向计算索引：
     ```cpp
     size_t n_ = ind % info.n;        // 列索引
     ind /= info.n;
     size_t m_ = ind % info.m;        // 行索引
     ind /= info.m;
     size_t i = ind;                  // 批量索引
     ```

4. **指针算术计算**（第 49 行）:
   - 根据 `info.c_matrix` 的步长信息，计算当前 C 元素的内存地址：
     ```cpp
     auto c_ = c_base + i * stride + m_ * row_stride + n_ * col_stride
     ```
   - 支持非连续内存布局（如转置矩阵、切片等）

5. **内积计算**（第 50-59 行）:
   - 对每个 k 维度（0 到 k-1）执行点积：
     - 计算 A 矩阵元素地址：`a_base + i*stride + m_*row_stride + k_*col_stride`
     - 计算 B 矩阵元素地址：`b_base + i*stride + n_*col_stride + k_*row_stride`
   - **类型特化处理**：
     - 如果是 `fp16_t` 或 `bf16_t`（半精度浮点），先转换为 `float` 再相乘，避免精度损失：
       ```cpp
       sum += utils::cast<float>(*a_) * utils::cast<float>(*b_);
       ```
     - 如果是 `float`，直接相乘：`sum += *a_ * (*b_)`

6. **结果写入与累加**（第 60-68 行）:
   - **低精度类型**（F16/BF16）：
     - 如果 `beta == 0`：直接写入 `alpha * sum` 的转换结果
     - 如果 `beta != 0`：将原 C 值转换为 float，累加后转回原类型：
       ```cpp
       *c_ = cast<Tdata>(beta * cast<float>(*c_) + alpha * sum);
       ```
   - **float 类型**：直接进行浮点运算：
     ```cpp
     *c_ = beta * (*c_) + alpha * sum;
     ```

**内存访问模式**:
- A 矩阵：按行遍历（内循环 k_ 连续访问，利用缓存局部性）
- B 矩阵：按列遍历（可能造成缓存未命中，但对小规模矩阵可接受）
- C 矩阵：写入时按行优先（与 BLAS 列主序约定一致）

**并行效率**:
- 每个线程独立处理 C 矩阵的不同元素，无数据竞争
- 负载均衡：每个迭代的工作量完全相同（k 次乘加）
- 扩展性：线程数可线性扩展至 CPU 核心数

## 4. 公共 API 接口

```cpp
// 创建 CPU GEMM 算子描述符
infiniStatus_t op::gemm::cpu::Descriptor::create(
    infiniopHandle_t handle,              // [输入] CPU 设备句柄
    Descriptor **desc_ptr,                // [输出] 返回的描述符指针
    infiniopTensorDescriptor_t c_desc,    // [输入] 输出矩阵 C 的张量描述符
    infiniopTensorDescriptor_t a_desc,    // [输入] 左矩阵 A 的张量描述符
    infiniopTensorDescriptor_t b_desc     // [输入] 右矩阵 B 的张量描述符
);
// 返回: 成功 INFINI_STATUS_SUCCESS，失败返回错误码（如类型不支持、形状不匹配等）

// 执行矩阵乘法计算 C = alpha * A @ B + beta * C
infiniStatus_t op::gemm::cpu::Descriptor::calculate(
    void *workspace,       // [输入] 工作空间（CPU 实现中为 nullptr）
    size_t workspace_size, // [输入] 工作空间大小（CPU 实现中为 0）
    void *c,               // [输入/输出] 输出矩阵 C 的数据指针
    float beta,            // [输入] C 的缩放系数
    const void *a,         // [输入] 左矩阵 A 的数据指针
    const void *b,         // [输入] 右矩阵 B 的数据指针
    float alpha,           // [输入] 乘积的缩放系数
    void *stream           // [输入] 执行流（CPU 实现中为 nullptr）
) const;
// 返回: 成功 INFINI_STATUS_SUCCESS，失败返回错误码

// 获取所需工作空间大小
size_t Descriptor::workspaceSize() const;
// 返回: 始终返回 0（CPU 实现无需额外工作空间）
```

## 5. 使用示例

```cpp
#include "infiniop/ops/gemm/cpu/gemm_cpu.h"

// 1. 创建 CPU 设备句柄
infiniopHandle_t handle;
infiniStatus_t status = device::cpu::Handle::create(&handle, 0);
assert(status == INFINI_STATUS_SUCCESS);

// 2. 准备张量描述符（假设形状：A=[2,3], B=[3,4], C=[2,4]）
infiniopTensorDescriptor_t a_desc, b_desc, c_desc;
// ... 创建张量描述符的代码（略） ...

// 3. 创建 GEMM 算子描述符
op::gemm::cpu::Descriptor *gemm_desc;
status = op::gemm::cpu::Descriptor::create(handle, &gemm_desc, c_desc, a_desc, b_desc);
assert(status == INFINI_STATUS_SUCCESS);

// 4. 分配并初始化数据
float *a = new float[2 * 3]; // 矩阵 A
float *b = new float[3 * 4]; // 矩阵 B
float *c = new float[2 * 4]; // 矩阵 C
// ... 填充数据 ...

// 5. 执行矩阵乘法 C = 1.0 * A @ B + 0.0 * C
status = gemm_desc->calculate(
    nullptr, 0,  // 无工作空间
    c, 0.0f,     // 输出矩阵 C，beta = 0（不累加原值）
    a, b, 1.0f,  // 输入矩阵 A、B，alpha = 1.0
    nullptr      // 无执行流
);
assert(status == INFINI_STATUS_SUCCESS);

// 6. 清理资源
delete gemm_desc;
delete[] a;
delete[] b;
delete[] c;
// ... 清理句柄和描述符 ...
```

## 6. 实现细节

### 6.1 内存管理

**策略**: 纯 CPU 内存操作，无需特殊分配器
- 所有数据指针由调用者管理（通常在 CPU 主存上）
- 算子内部不分配任何动态内存（workspace_size = 0）
- 使用 RAII 机制：`Descriptor` 由 `new` 创建，由用户负责 `delete`

**内存访问**:
- 支持非连续内存布局（通过步长信息）
- 允许多张量共享底层内存（视图机制）
- 无内存对齐要求（现代 CPU 自动处理未对齐访问）

### 6.2 并发控制

**OpenMP 并行化**:
- 使用 OpenMP 的 `parallel for` 指令实现数据并行
- **同步原语**: 无显式锁，依赖 OpenMP 运行时的隐式同步
- **数据竞争避免**: 每个线程写入 C 的不同元素，无共享写操作
- **线程安全**: 完全线程安全，多个 GEMM 实例可在不同线程并发执行

**死锁预防**: 不适用（无锁机制）

### 6.3 性能优化

**算法选择**:
- **朴素三重循环**: O(m×n×k) 的标准矩阵乘法
- 未使用高级优化技术（如分块、向量化、Strassen 算法）
- **适用场景**: 中小规模矩阵（m, n, k < 512）或原型验证

**优化技术**:
1. **OpenMP 并行**: 利用多核 CPU，理论加速比接近核心数
2. **类型特化**: 编译期为不同数据类型生成特化版本，避免运行时分支
3. **if constexpr**: 半精度类型在编译期展开为 float 运算，零运行时开销
4. **缓存局部性**: A 矩阵按行访问（连续内存），充分利用 L1/L2 缓存

**性能限制**:
- 未使用 SIMD 指令（如 AVX-512、ARM NEON）
- 未实现缓存分块（cache blocking），大矩阵可能因缓存抖动而降速
- 未使用线程亲和性绑定，可能导致线程迁移

**复杂度保证**:
- 时间复杂度: Θ(batch × m × n × k)
- 空间复杂度: Θ(1) 额外空间（除了输入输出）

### 6.4 错误处理

**错误传播机制**:
- 使用 `infiniStatus_t` 枚举返回错误码
- 宏辅助：`CHECK_DTYPE`、`CHECK_RESULT` 在失败时提前返回
- **错误码**:
  - `INFINI_STATUS_SUCCESS`: 成功
  - `INFINI_STATUS_BAD_TENSOR_DTYPE`: 不支持的数据类型（非 F16/BF16/F32）
  - `INFINI_STATUS_BAD_TENSOR_SHAPE`: 矩阵形状不匹配（如 A 的列数 ≠ B 的行数）
  - `INFINI_STATUS_BAD_TENSOR_STRIDES`: 步长不合法（行和列步长不能都不为 1）

**恢复机制**: 无自动恢复，错误时立即返回，调用者负责清理资源

### 6.5 依赖关系

**外部依赖**:
- **OpenMP**: 并行化框架（需编译器支持，如 `-fopenmp`）
- **C++ 标准库**: `<algorithm>`（std::swap）

**内部依赖**:
- `../gemm.h`: 提供描述符宏定义
- `../info.h`: 提供 `MatmulInfo` 和 `BlasMatrix` 结构
- `../../../devices/cpu/common_cpu.h`: CPU 设备通用定义
- `../../../utils.h`: 工具函数和类型定义

**硬件依赖**: 无特定硬件要求，兼容所有 x86_64/ARM64 CPU

### 6.6 设计模式

**1. PImpl (Pointer to Implementation) 模式**:
- **目的**: 隐藏硬件相关的私有实现细节
- **实现**: `struct Opaque;` 前向声明 + 私有指针 `_opaque`
- **优势**: 头文件不暴露硬件类型，保持 ABI 兼容性

**2. 工厂模式 (Factory Pattern)**:
- **实现**: `Descriptor::create()` 静态工厂方法
- **优势**: 封装创建逻辑，集中处理错误，避免构造函数抛异常

**3. 策略模式 (Strategy Pattern)**:
- **实现**: 不同硬件后端（CPU/CUDA/Ascend）继承同一基类 `InfiniopDescriptor`
- **优势**: 运行时多态，统一接口，易于扩展新硬件

**4. 模板方法模式 (Template Method)**:
- **实现**: `calculate<Tdata>()` 模板函数，根据数据类型特化
- **优势**: 零开销抽象，编译期类型检查

## 7. 与其他模块的关系

**上游依赖**:
- `device::cpu::Handle`: 提供设备信息（device_type, device_id）
- `infiniopTensorDescriptor_t`: 提供张量形状和布局信息
- `utils::Result<T>`: 用于错误传播和结果封装

**下游调用**:
- 无直接下游模块（叶子节点算子）

**同级模块**:
- `../cuda/`: CUDA 设备的 GEMM 实现（使用 cuBLAS）
- `../kunlun/`: 昆仑芯片的 GEMM 实现
- `../ascend/`: 华为昇腾的 GEMM 实现（使用 ACL）

**接口兼容性**: 所有后端实现相同的 `Descriptor` 接口，保证算子级别的硬件无关性
