# `AddRMSNorm CPU Backend` 核心实现文档

本模块实现了 AddRMSNorm（加法 + RMS 归一化）融合算子的 CPU 后端，支持 float16、bfloat16、float32 和 float64 数据类型。该算子在 Transformer 模型中广泛应用于残差连接后的归一化操作，将元素加法与 RMS 归一化融合为单一算子以提升性能。

## 1. 模块结构

- **`add_rms_norm_cpu.h`**: CPU 后端描述符的头文件定义，通过宏 `DESCRIPTOR(cpu)` 展开生成 `Descriptor` 类
- **`add_rms_norm_cpu.cc`**: CPU 后端实现文件，包含算子创建、类型分发和核心计算内核

## 2. 核心类

### `op::add_rms_norm::cpu::Descriptor`
- **位置**: `add_rms_norm_cpu.h` (通过宏展开), `add_rms_norm_cpu.cc`
- **主要功能**: CPU 后端的 AddRMSNorm 算子描述符，负责算子实例的创建、配置和执行调度
- **关键成员**:
  - `_opaque`: 不透明句柄（当前未使用，保留用于未来扩展）
  - `_info`: `AddRMSNormInfo` 结构体，存储张量形状、步长、数据类型和 epsilon 参数
  - `_workspace_size`: 工作空间大小（当前为 0，无需额外工作空间）
  - 继承自 `InfiniopDescriptor`，包含 `device_type` 和 `device_id`
- **核心方法**:
  - `~Descriptor()`: 析构函数（空实现）
  - `create(handle, desc_ptr, y_desc, a_desc, b_desc, weight_desc, epsilon, residual_out_desc)`: 静态工厂方法，验证张量描述符并创建算子实例。调用 `AddRMSNormInfo::create()` 进行参数验证，然后分配 `Descriptor` 对象。时间复杂度 O(1)。
  - `calculate(workspace, workspace_size, y, a, b, weight, residual_out, stream)`: 执行算子计算的核心接口，根据输入数据类型（`atype`）和权重类型（`wtype`）分发到对应的模板实例化函数
- **生命周期**: 动态分配（`new Descriptor`），由调用者负责释放

### `op::add_rms_norm::AddRMSNormInfo`
- **位置**: `../info.h` (父目录，被当前模块引用)
- **主要功能**: 算子元数据容器，封装张量形状、步长、数据类型验证逻辑
- **关键成员**:
  - `wtype`, `atype`: 权重和激活值的数据类型
  - `epsilon`: RMS 归一化的数值稳定项（默认约 1e-6）
  - `shape`: 输出张量的形状向量 `[batch, nhead, dim]` 或 `[batch, dim]`
  - `y_strides`, `a_strides`, `b_strides`, `residual_out_strides`: 各张量的步长向量
  - `has_residual_out`: 布尔标志（当前恒为 `true`）
- **核心方法**:
  - `create(...)`: 静态工厂方法，执行严格的前置条件验证：
    - 检查数据类型兼容性（FP16/BF16 支持混合精度权重，FP32/FP64 要求类型一致）
    - 验证形状兼容性（支持 2D `[batch, dim]` 和 3D `[batch, nhead, dim]` 输入）
    - 确保最内维连续（步长为 1）
    - 验证 `residual_out` 张量的形状和类型与输入一致
  - `ndim()`: 返回张量维度数（2 或 3）
  - `dim()`: 返回最内维大小（归一化的特征维度）
- **生命周期**: 值对象，通过 `utils::Result<AddRMSNormInfo>` 返回

## 3. API 接口

```cpp
// 算子创建接口
infiniStatus_t op::add_rms_norm::cpu::Descriptor::create(
    infiniopHandle_t handle,                          // InfiniOp 上下文句柄
    Descriptor **desc_ptr,                            // 输出：算子描述符指针
    infiniopTensorDescriptor_t y_desc,                // 输出张量描述符
    infiniopTensorDescriptor_t a_desc,                // 输入张量 A（如残差）
    infiniopTensorDescriptor_t b_desc,                // 输入张量 B（如隐藏状态）
    infiniopTensorDescriptor_t weight_desc,           // 归一化权重向量
    float epsilon,                                    // RMS 归一化的稳定项
    infiniopTensorDescriptor_t residual_out_desc      // 残差加法输出
);
// 返回 INFINI_STATUS_SUCCESS 或错误码

// 算子计算接口
infiniStatus_t Descriptor::calculate(
    void *workspace,           // 工作空间（当前未使用）
    size_t workspace_size,     // 工作空间大小（当前为 0）
    void *y,                   // 输出缓冲区：归一化后的结果
    const void *a,             // 输入缓冲区 A
    const void *b,             // 输入缓冲区 B
    const void *weight,        // 权重向量
    void *residual_out,        // 残差加法输出：a + b
    void *stream               // CUDA 流（CPU 后端忽略）
) const;
```

## 4. 使用示例

```cpp
// 示例：在 CPU 上执行 AddRMSNorm（融合残差加法 + RMS 归一化）
#include "infiniop/ops/add_rms_norm/cpu/add_rms_norm_cpu.h"

// 1. 定义张量形状 (batch=2, nhead=4, dim=128)
const size_t batch = 2, nhead = 4, dim = 128;
std::vector<size_t> shape = {batch, nhead, dim};

// 2. 创建张量描述符
auto y_desc = TensorDescriptor::create(INFINI_DTYPE_F16, shape);
auto a_desc = TensorDescriptor::create(INFINI_DTYPE_F16, shape);
auto b_desc = TensorDescriptor::create(INFINI_DTYPE_F16, shape);
auto weight_desc = TensorDescriptor::create(INFINI_DTYPE_F32, {dim});  // FP32 权重
auto residual_out_desc = TensorDescriptor::create(INFINI_DTYPE_F16, shape);

// 3. 创建算子描述符
op::add_rms_norm::cpu::Descriptor *add_rms_norm_desc = nullptr;
float epsilon = 1e-6f;
auto status = op::add_rms_norm::cpu::Descriptor::create(
    handle, &add_rms_norm_desc, y_desc.get(), a_desc.get(),
    b_desc.get(), weight_desc.get(), epsilon, residual_out_desc.get()
);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误（如类型不兼容、形状不匹配）
}

// 4. 准备数据缓冲区（假设已分配）
fp16_t *y = new fp16_t[batch * nhead * dim];
fp16_t *a = new fp16_t[batch * nhead * dim];  // 输入 A（如残差连接）
fp16_t *b = new fp16_t[batch * nhead * dim];  // 输入 B（如层输出）
float *weight = new float[dim];                // 归一化权重
fp16_t *residual_out = new fp16_t[batch * nhead * dim];

// 5. 执行算子
status = add_rms_norm_desc->calculate(
    nullptr, 0,          // 无需工作空间
    y, a, b, weight,     // 数据指针
    residual_out,        // 残差加法输出
    nullptr              // CPU 后端忽略流参数
);

// 6. 清理
delete add_rms_norm_desc;
delete[] y, a, b, weight, residual_out;

// 执行后：
// - residual_out 包含 a + b 的逐元素加法结果
// - y 包含归一化结果：rms((a + b) * weight)，其中 rms(x) = x / sqrt(mean(x²) + ε)
```

## 5. 实现细节

### 算法实现

**AddRMSNorm 融合算子**（公式）:
```
residual_out = a + b
rms = 1 / sqrt(mean(residual_out²) + ε)
y = residual_out ⊙ weight ⊙ rms
```
其中 `⊙` 表示逐元素乘法，`mean` 在特征维度（最内维）上计算。

**FP32/FP64 全精度内核** (`add_rmsnorm<T>`):
- **第一阶段（循环融合）**: 遍历每个样本的最内维，一次遍历完成加法和平方和累积
  ```cpp
  for (size_t k = 0; k < dim; k++) {
      T sum_val = a_ptr[k] + b_ptr[k];
      residual_out_ptr[k] = sum_val;      // 保存加法结果
      sum_squared += sum_val * sum_val;   // 累积平方和
  }
  ```
- **第二阶段（计算 RMS）**: 单次除法 + 平方根
  ```cpp
  T rms = (T)1 / std::sqrt(sum_squared / (T)(dim) + (T)(epsilon));
  ```
- **第三阶段（应用归一化）**: 重新遍历，逐元素应用权重和 RMS
  ```cpp
  for (size_t k = 0; k < dim; k++) {
      y_ptr[k] = residual_out_ptr[k] * w[k] * rms;
  }
  ```

**FP16/BF16 半精度内核** (`add_rmsnormHalfPrecision<T, Tw>`):
- **精度提升策略**: 所有中间计算在 `float` 中进行以避免溢出和精度损失
  ```cpp
  float sum_val = utils::cast<float>(a_ptr[k]) + utils::cast<float>(b_ptr[k]);
  residual_out_ptr[k] = utils::cast<T>(sum_val);  // 存回半精度
  sum_squared += sum_val * sum_val;
  ```
- **权重类型处理**: 使用 `if constexpr` 编译时分支支持混合精度权重
  - `Tw = float`: 直接使用 `w[k]`（无需转换）
  - `Tw = fp16_t/bf16_t`: 先转换为 `float` 再计算
  ```cpp
  if constexpr (std::is_same<Tw, float>::value) {
      val = sum_val * w[k] * rms;
  } else {
      val = sum_val * utils::cast<float>(w[k]) * rms;
  }
  ```

### 并发策略

- **OpenMP 并行化**: 使用 `#pragma omp parallel for` 在最外层（`batch * nhead`）并行
  - 每个线程独立处理一个样本（batch index + head index 组合）
  - 无数据竞争：各样本完全独立，无跨样本依赖
  - 负载均衡：静态调度，假设各样本计算量相等
- **线程安全性**: 无共享可变状态，仅读写独立内存区域

### 内存管理

- **工作空间**: 无需额外工作空间（`workspace_size = 0`），所有计算原地完成
- **内存布局**: 要求最内维连续（步长为 1），外层维度支持非连续步长
- **内存效率**: 两遍算法但第二遍复用 `residual_out` 缓冲区，避免重复加法

### 性能特征

- **时间复杂度**: O(batch × nhead × dim) - 每个元素执行常数次算术运算
- **空间复杂度**: O(1) 额外空间（除输入输出外）
- **融合优势**: 相比分离的 "Add + RMSNorm"，减少一次内存读写和 kernel 启动开销
- **缓存友好**: 两遍算法在半精度内核中通过复用 `residual_out` 缓存加法结果

### 错误处理

- **类型验证**: `AddRMSNormInfo::create()` 拒绝不支持的类型组合
  - FP16/BF16: 权重可以是同类型或 FP32
  - FP32/FP64: 权重必须同类型
- **形状验证**: 仅支持 2D 或 3D 张量，且维度必须匹配
- **步长验证**: 强制最内维连续（`stride[-1] == 1`）
- **错误传播**: 使用 `CHECK_RESULT` 和 `CHECK_STATUS` 宏统一处理错误码
- **Fallback**: 不支持的类型返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`

### 类型分发机制

**`calculate()` 方法的类型分发树**:
```
if (atype == FP16) {
    if (wtype == FP16)      → add_rmsnormHalfPrecision<fp16_t, fp16_t>
    else if (wtype == F32)  → add_rmsnormHalfPrecision<fp16_t, float>
    else if (wtype == BF16) → add_rmsnormHalfPrecision<fp16_t, bf16_t>
}
else if (atype == BF16) {
    if (wtype == BF16)      → add_rmsnormHalfPrecision<bf16_t, bf16_t>
    else if (wtype == F32)  → add_rmsnormHalfPrecision<bf16_t, float>
    else if (wtype == F16)  → add_rmsnormHalfPrecision<bf16_t, fp16_t>
}
else if (atype == FP32)     → add_rmsnorm<float>
else if (atype == FP64)     → add_rmsnorm<double>
```
通过模板实例化避免运行时分支，提升性能。

### 依赖关系

- **外部依赖**:
  - OpenMP (`#pragma omp parallel for`)：多线程并行
  - C++ 标准库：`std::sqrt`, `std::abort`
  - `utils::cast<T>()`：安全类型转换工具（可能处理饱和/舍入）
  - `../add_rms_norm.h`：父目录的算子接口定义（宏 `DESCRIPTOR`）
  - `../../reduce/cpu/reduce.h`：未在当前代码中使用（可能用于其他实现）
  - `../../../devices/cpu/common_cpu.h`：CPU 通用工具（可能包含宏定义）
- **内部依赖**: `AddRMSNormInfo`（位于父目录 `info.h`）

### 设计模式

- **策略模式**: 通过 `atype` 和 `wtype` 选择不同的模板实例化策略
- **模板方法模式**: `calculate()` 定义类型分发框架，具体算法由模板函数实现
- **RAII**: 虽未显式使用智能指针，但描述符通过析构函数管理资源
- **CRTP（奇异递归模板模式）**: `DESCRIPTOR(cpu)` 宏通过命名空间注入生成类定义

### 数值稳定性

- **epsilon 作用**: 防止除零和数值下溢，通常设置为 1e-6
- **半精度提升**: FP16/BF16 计算在 float 中进行，避免平方和溢出
- **求和顺序**: 从左到右累积（Kahan 求和可能更精确，但当前实现未使用）

### 优化技巧

- **循环融合**: 第一阶段同时计算加法和平方和，减少一遍遍历
- **结果复用**: 第二阶段归一化时重用 `residual_out` 中的加法结果
- **编译时分支**: `if constexpr` 在半精度内核中消除权重类型的运行时开销
- **步长缓存**: 在 `AddRMSNormInfo` 中预存步长，避免重复计算指针偏移
