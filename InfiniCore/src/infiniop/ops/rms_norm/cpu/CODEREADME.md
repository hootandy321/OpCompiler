# RMSNorm CPU 后端核心实现文档

RMSNorm（Root Mean Square Normalization）是深度学习中常用的归一化层，该模块实现了 RMSNorm 操作的 CPU 后端，支持多种数据类型（FP32、FP64、FP16、BF16）和多维度张量（2D 和 3D），采用 OpenMP 并行化技术提升计算性能。

## 1. 模块结构

- **`rms_norm_cpu.h`**: CPU 后端Descriptor类声明，通过宏定义实现与父类接口的绑定
- **`rms_norm_cpu.cc`**: CPU 后端核心实现，包含算子创建、计算核函数以及针对不同精度的特化版本

## 2. 核心类

### `Descriptor`
- **位置**: `rms_norm_cpu.h` (宏定义), `rms_norm_cpu.cc` (实现)
- **主要功能**: RMSNorm CPU 后端算子描述符，负责算子初始化、输入验证和计算调度
- **核心成员**:
  - `_opaque`: 不透明指针（当前未使用，保留用于设备特定资源）
  - `_info`: `RMSNormInfo` 结构体，存储张量形状、步长、数据类型等元信息
  - `_workspace_size`: 工作空间大小（当前为 0）
- **核心方法**:
  - `create(handle, desc_ptr, y_desc, x_desc, w_desc, epsilon)`: 静态工厂方法，验证输入张量类型和形状兼容性，创建 Descriptor 实例，返回 `INFINI_STATUS_SUCCESS` 或错误码
  - `calculate(workspace, workspace_size, y, x, w, stream)`: 执行 RMSNorm 计算，根据输入数据类型分派到不同的模板特化实现，支持异步流接口（当前 CPU 实现中 stream 参数未使用）
  - `~Descriptor()`: 析构函数（空实现，无资源释放）
- **生命周期**: 由 `create` 方法动态分配，调用方负责内存释放，遵循 RAII 原则

### `RMSNormInfo` (父类定义)
- **位置**: `../info.h`
- **主要功能**: 存储和验证 RMSNorm 操作的元数据，执行类型和形状兼容性检查
- **核心成员**:
  - `wtype`: 权重张量数据类型（`infiniDtype_t`）
  - `atype`: 激活张量数据类型（`infiniDtype_t`）
  - `epsilon`: 数值稳定项（float），防止除零错误
  - `shape`: 输出张量形状（`std::vector<size_t>`），支持 2D `[batch, dim]` 或 3D `[batch, nhead, dim]`
  - `y_strides`, `x_strides`: 输出和输入张量的步长（`std::vector<ptrdiff_t>`）
- **核心方法**:
  - `create(y_desc, x_desc, w_desc, epsilon)`: 静态工厂方法，验证数据类型组合（FP16/BF16 支持混合精度，FP32/FP64 要求类型一致），检查张量维度和最后一维连续性，返回 `Result<RMSNormInfo>` 或错误码
  - `ndim()`: 返回张量维度数（2 或 3）
  - `dim()`: 返回归一化维度大小（最后一个维度）

## 3. API 接口

```cpp
// 算子创建 API
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,                // 全局句柄，包含设备和设备ID信息
    Descriptor **desc_ptr,                  // [输出] 创建的描述符指针
    infiniopTensorDescriptor_t y_desc,      // 输出张量描述符
    infiniopTensorDescriptor_t x_desc,      // 输入张量描述符
    infiniopTensorDescriptor_t w_desc,      // 权重张量描述符（1D，大小为 dim）
    float epsilon                           // 数值稳定项
);
// 返回: INFINI_STATUS_SUCCESS | INFINI_STATUS_BAD_TENSOR_DTYPE | INFINI_STATUS_BAD_TENSOR_SHAPE | INFINI_STATUS_BAD_TENSOR_STRIDES

// 计算执行 API
infiniStatus_t Descriptor::calculate(
    void *workspace,         // 工作空间（当前未使用）
    size_t workspace_size,   // 工作空间大小（当前为 0）
    void *y,                 // [输出] 输出数据指针
    const void *x,           // [输入] 输入数据指针
    const void *w,           // [输入] 权重数据指针
    void *stream             // 流句柄（CPU 实现中未使用）
) const;
// 返回: INFINI_STATUS_SUCCESS | INFINI_STATUS_BAD_TENSOR_DTYPE
```

## 4. 使用示例

```cpp
// 示例: 在 FP32 数据上执行 RMSNorm
#include "rms_norm_cpu.h"
#include "../../tensor.h"

// 1. 创建张量描述符
const size_t batch = 128;
const size_t nhead = 8;
const size_t dim = 512;

std::vector<size_t> shape = {batch, nhead, dim};
std::vector<size_t> strides = {nhead * dim, dim, 1};

// 创建 3D 张量 [batch, nhead, dim]
auto x_desc = infiniopCreateTensorDescriptor(INFINI_DTYPE_F32, 3, shape.data(), strides.data());
auto y_desc = infiniopCreateTensorDescriptor(INFINI_DTYPE_F32, 3, shape.data(), strides.data());

// 创建 1D 权重张量 [dim]
std::vector<size_t> w_shape = {dim};
std::vector<size_t> w_strides = {1};
auto w_desc = infiniopCreateTensorDescriptor(INFINI_DTYPE_F32, 1, w_shape.data(), w_strides.data());

// 2. 创建 RMSNorm 算子
float epsilon = 1e-5f;
op::rms_norm::cpu::Descriptor *rms_norm_desc = nullptr;
auto status = op::rms_norm::cpu::Descriptor::create(
    handle, &rms_norm_desc, y_desc, x_desc, w_desc, epsilon);

if (status != INFINI_STATUS_SUCCESS) {
    // 错误处理
}

// 3. 分配内存并执行计算
float *x = new float[batch * nhead * dim];
float *w = new float[dim];
float *y = new float[batch * nhead * dim];

// 填充输入数据...
for (size_t i = 0; i < batch * nhead * dim; ++i) {
    x[i] = /* 输入数据 */;
}
for (size_t i = 0; i < dim; ++i) {
    w[i] = /* 权重数据 */;
}

// 执行 RMSNorm 计算
status = rms_norm_desc->calculate(nullptr, 0, y, x, w, nullptr);

// 4. 清理资源
delete rms_norm_desc;
delete[] x;
delete[] w;
delete[] y;
```

## 5. 实现细节

### 核心算法
RMSNorm 公式：`y = x * w / sqrt(mean(x^2) + epsilon)`

计算步骤：
1. **平方和归约**：对最后一个维度计算 `sum(x^2)`，使用 `op::common_cpu::reduce_op::sumSquared` 函数
2. **RMS 计算**：`rms = 1 / sqrt(sum / dim + epsilon)`
3. **逐元素应用**：`y[i] = x[i] * w[i] * rms`

### 并行化策略
- **OpenMP 并行**：使用 `#pragma omp parallel for` 对 batch 和 head 维度并行化
- **线程粒度**：每个线程处理一个 `(batch, head)` 对，即 `total_blocks = batch * nhead` 个独立任务
- **负载均衡**：OpenMP runtime 负责任务调度，采用 static 或 dynamic 调度（取决于编译器配置）
- **无共享状态**：每个线程操作独立的内存区域，无需锁机制

### 内存访问模式
- **输入指针计算**：`x_ptr = x + i * x_strides[0] + j * x_strides[1]`，支持非连续张量布局
- **连续性假设**：最后一维必须连续（stride = 1），确保向量化和高效缓存利用
- **步长支持**：通过 strides 数组支持广播和不规则内存布局

### 数据类型支持
- **FP32/FP64**：使用 `rmsnorm<T>` 模板，所有计算在原精度执行
- **FP16/BF16**：使用 `rmsnormHalfPrecision<T, Tw>` 模板，关键特性：
  - **中间精度提升**：归约计算在 FP32 执行，避免溢出和精度损失
  - **混合权重类型**：支持权重为 FP16、BF16 或 FP32，通过 `if constexpr` 编译时分支优化
  - **类型转换**：使用 `utils::cast<T>()` 进行安全类型转换

### 性能优化技术
1. **向量化友好**：最后一维连续存储，编译器可自动生成 SIMD 指令（AVX2/AVX-512）
2. **缓存优化**：每个线程处理的数据块大小为 `dim`，适配 L1/L2 缓存
3. **归约优化**：`sumSquared` 函数可能使用级联归约或 Kahan 求和算法（需查看 reduce 模块实现）
4. **编译时常量**：`dim` 和 `epsilon` 在循环内不变，编译器可优化除法为乘法

### 错误处理
- **类型验证**：`RMSNormInfo::create` 检查数据类型组合合法性
- **形状验证**：检查张量维度（2D/3D）和最后一维大小匹配
- **连续性检查**：确保最后一维 stride 为 1
- **错误传播**：使用 `CHECK_RESULT` 和 `CHECK_STATUS` 宏传播错误码

### 依赖关系
- **父模块**：`../rms_norm.h`（Descriptor 基类定义）、`../info.h`（元信息类）
- **设备通用模块**：`../../../devices/cpu/common_cpu.h`（CPU 通用工具）
- **归约算子**：`../../../reduce/cpu/reduce.h`（提供 `sumSquared` 函数）
- **工具库**：`../../../utils.h`（类型转换、Result 错误处理）
- **外部依赖**：OpenMP 并行运行时，标准库 `<cmath>`（数学函数）

### 设计模式
- **策略模式**：通过数据类型分派到不同的模板实例化（FP32/FP64 vs FP16/BF16）
- **工厂模式**：`create` 静态方法封装对象创建和验证逻辑
- **CRTP (Curiously Recurring Template Pattern)**：`DESCRIPTOR(cpu)` 宏在 `op::rms_norm::cpu` 命名空间中生成 `Descriptor` 类
- **零成本抽象**：模板和 `if constexpr` 确保无运行时类型检查开销
