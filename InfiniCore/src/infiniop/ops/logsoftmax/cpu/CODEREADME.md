# LogSoftmax CPU 后端实现文档

本模块实现了 LogSoftmax 操作的 CPU 后端，支持 2D 和 3D 张量，提供多种浮点数据类型（FP16、BF16、F32）的组合计算。采用 OpenMP 并行化策略和数值稳定的 log-softmax 算法，确保在保持精度的同时实现高性能计算。

## 1. 模块结构

- **`logsoftmax_cpu.h`**: CPU 后端描述符声明文件，通过宏 `DESCRIPTOR(cpu)` 展开生成完整的 Descriptor 类定义
- **`logsoftmax_cpu.cc`**: CPU 后端核心实现，包含描述符创建、log-softmax 计算内核以及类型分发逻辑

## 2. 核心类

### `op::logsoftmax::cpu::Descriptor`
- **位置**: `logsoftmax_cpu.h`（宏展开）, `logsoftmax_cpu.cc`
- **主要功能**: CPU 设备的 LogSoftmax 操作描述符，管理计算所需元数据和执行调度
- **继承关系**: 继承自 `InfiniopDescriptor` 基类
- **关键成员**:
  - `Opaque *_opaque`: 不透明指针，保留用于未来扩展（当前初始化为 nullptr）
  - `LogSoftmaxInfo _info`: 存储张量形状、步长、数据类型等元数据
  - `size_t _workspace_size`: 工作空间大小（当前实现为 0，无需额外内存）
- **核心方法**:
  - `create(handle, desc_ptr, y_desc, x_desc)`: 静态工厂方法，验证张量描述符一致性并创建 Descriptor 实例。调用 `LogSoftmaxInfo::create()` 进行元数据初始化，返回标准状态码
  - `calculate(workspace, workspace_size, y, x, stream)`: 执行 log-softmax 计算的主入口。根据输入输出数据类型分发到对应的模板特化实现
  - `~Descriptor()`: 析构函数，当前为空实现（无动态资源释放）
- **生命周期**: 由用户通过 `create()` 静态方法创建，计算完成后由调用者负责销毁。遵循 RAII 语义，但当前实现无状态清理需求

### `op::logsoftmax::LogSoftmaxInfo`
- **位置**: `../info.h`（父目录共享定义）
- **主要功能**: 编译期张量元数据容器，在 `create()` 阶段完成所有形状和步长计算
- **关键成员**:
  - `infiniDtype_t x_dtype, y_dtype`: 输入输出张量的数据类型（F16/BF16/F32）
  - `size_t batch_size`: 批次维度大小（2D 为 shape[0]，3D 为 shape[0]×shape[1]）
  - `size_t probs_size`: 概率维度大小（2D 为 shape[1]，3D 为 shape[2]）
  - `size_t ndim`: 张量维度数（2 或 3）
  - `size_t seq_len`: 序列长度（仅 3D 张量使用，等于 shape[1]）
  - `ptrdiff_t y_stride_b, x_stride_b`: 展平后批次间步长
  - `ptrdiff_t y_stride_p, x_stride_p`: 概率维度内步长
  - `ptrdiff_t y_stride_0/1/2, x_stride_0/1/2`: 原始 3D 张量的各维度步长
- **核心方法**:
  - `create(y_desc, x_desc)`: 静态工厂方法，执行数据类型校验、形状一致性检查、步长计算。对 2D 张量直接使用原始步长，对 3D 张量展平前两维度并重新计算批次步长。返回 `utils::Result<LogSoftmaxInfo>` 类型

## 3. API 接口

```cpp
// 创建 CPU LogSoftmax 描述符
infiniStatus_t op::logsoftmax::cpu::Descriptor::create(
    infiniopHandle_t handle,              // InfiniOP 全局句柄
    Descriptor **desc_ptr,                // [输出] 创建的描述符指针
    infiniopTensorDescriptor_t y_desc,    // 输出张量描述符
    infiniopTensorDescriptor_t x_desc     // 输入张量描述符
);
// 返回: INFINI_STATUS_SUCCESS 成功; INFINI_STATUS_BAD_TENSOR_DTYPE 类型不支持; INFINI_STATUS_BAD_TENSOR_SHAPE 形状不合规

// 执行 LogSoftmax 计算
infiniStatus_t op::logsoftmax::cpu::Descriptor::calculate(
    void *workspace,          // 工作空间指针（当前未使用）
    size_t workspace_size,    // 工作空间大小（当前未使用）
    void *y,                  // [输出] 输出张量数据指针
    const void *x,            // [输入] 输入张量数据指针
    void *stream              // 流指针（CPU 后端未使用）
) const;
// 返回: INFINI_STATUS_SUCCESS 成功; INFINI_STATUS_BAD_TENSOR_DTYPE 类型组合不支持

// 查询工作空间需求
size_t op::logsoftmax::cpu::Descriptor::workspaceSize() const;
// 返回: 当前固定返回 0（无需额外工作空间）
```

## 4. 使用示例

```cpp
// 示例：在 CPU 上执行 3D 张量的 LogSoftmax（形状 [2, 5, 1024]）
#include "logsoftmax/cpu/logsoftmax_cpu.h"

// 1. 创建输入输出张量描述符
std::vector<size_t> x_shape = {2, 5, 1024};
std::vector<ptrdiff_t> x_strides = {5120, 1024, 1};  // 连续内存
auto x_desc = new TensorDescriptor(INFINI_DTYPE_F16, x_shape, x_strides);

auto y_desc = new TensorDescriptor(INFINI_DTYPE_F32, x_shape, x_strides);

// 2. 创建 LogSoftmax 描述符
op::logsoftmax::cpu::Descriptor *logsoftmax_desc = nullptr;
auto status = op::logsoftmax::cpu::Descriptor::create(
    handle, &logsoftmax_desc, y_desc, x_desc);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误（类型不匹配或形状非法）
}

// 3. 分配内存并执行计算
fp16_t *x_data = new fp16_t[2 * 5 * 1024];
float *y_data = new float[2 * 5 * 1024];
// ... 填充输入数据 ...

status = logsoftmax_desc->calculate(
    nullptr, 0,              // 无需工作空间
    y_data,                  // 输出
    x_data,                  // 输入
    nullptr                  // CPU 后端忽略 stream
);

// 4. 清理资源
delete logsoftmax_desc;
delete x_desc;
delete y_desc;
delete[] x_data;
delete[] y_data;
```

## 5. 实现细节

### 算法实现
- **数值稳定性策略**: 采用经典的 log-softmax 稳定化算法 `log_softmax(x_i) = x_i - max(x) - log(Σ exp(x_j - max(x)))`，通过减去最大值避免 `exp()` 上溢出
- **计算流程**（每个 batch 独立执行）:
  1. 使用 `op::common_cpu::reduce_op::max()` 找到当前 batch 的最大值 `max_val`
  2. 遍历计算 `sum = Σ exp(x_i - max_val)`
  3. 计算 `log_sum = log(sum)`
  4. 对每个元素输出 `y_i = x_i - max_val - log_sum`

### 并行化策略
- **OpenMP 并行**: 使用 `#pragma omp parallel for` 对 batch 维度并行化，每个线程处理一个独立的 batch
- **无竞争设计**: 不同 batch 读写完全独立，无需同步操作
- **负载均衡**: 静态调度策略适用于各 batch 计算量均匀的场景（probs_size 相同）

### 类型系统
- **模板特化**: `logsoftmax<Tx, Ty>()` 模板函数支持 9 种类型组合（FP16/BF16/F32 × FP16/BF16/F32）
- **半精度处理**: 对 `fp16_t` 和 `bf16_t` 类型在计算时转换为 `float` 确保精度，仅最后写入时转换回原类型
- **类型分发**: `calculate()` 方法通过三层 if-else 嵌套根据 `_info.x_dtype` 和 `_info.y_dtype` 分发到正确的模板实例化

### 内存访问
- **步长支持**: 通过 `x_stride_p` 和 `y_stride_p` 支持非连续内存布局（如转置张量）
- **2D 张量寻址**: `offset = batch * stride_b + i * stride_p`
- **3D 张量寻址**: 先将线性 batch 索引分解为 `(batch_idx, seq_idx)`，然后计算 `offset = batch_idx * stride_0 + seq_idx * stride_1 + i * stride_2`
- **展平优化**: 对 3D 张量在元数据层将前两维度展平（`batch_size = shape[0] × shape[1]`），简化并行循环逻辑

### 错误处理
- **编译期检查**: 使用 `CHECK_DTYPE` 宏在 `LogSoftmaxInfo::create()` 阶段拒绝非法数据类型
- **形状验证**: 仅支持 2D 或 3D 张量，其他维度返回 `INFINI_STATUS_BAD_TENSOR_SHAPE`
- **运行时分发**: `calculate()` 方法遇到不支持的数据类型组合返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
- **结果类型封装**: `LogSoftmaxInfo::create()` 返回 `utils::Result<T>` 类型，通过 `CHECK_RESULT()` 宏进行统一错误处理

### 性能特性
- **时间复杂度**: O(batch_size × probs_size)，每个元素需三次遍历（找最大值、求和、计算结果）
- **空间复杂度**: O(1) 额外空间（仅在栈上分配 `max_val`, `sum`, `log_sum` 等标量）
- **缓存友好**: 对连续内存布局（stride_p = 1）实现顺序访问，充分利用 CPU 缓存行
- **SIMD 潜力**: 当前实现为标量循环，未来可通过向量化指令（AVX-512/NEON）加速 `exp` 和 `log` 计算

### 依赖关系
- **外部依赖**:
  - `op::common_cpu::reduce_op::max()`: 从 `reduce/cpu/reduce.h` 导入的最大值归约操作
  - `utils::cast<T>()`: 类型转换工具函数（半精度↔单精度）
  - `CHECK_RESULT`, `CHECK_DTYPE`, `CHECK_SAME_SHAPE`: 宏定义工具，用于错误检查
- **内部依赖**:
  - `../logsoftmax.h`: 定义 `DESCRIPTOR` 宏和基类 `InfiniopDescriptor`
  - `../info.h`: 定义 `LogSoftmaxInfo` 元数据结构
  - `../../operator.h`: 定义算子基类和通用类型

### 设计模式
- **CRTP (奇异递归模板模式)**: 通过 `DESCRIPTOR(NAMESPACE)` 宏生成命名空间特定的 `Descriptor` 类，避免代码重复
- **工厂方法模式**: `create()` 静态方法封装对象创建和验证逻辑
- **策略模式**: 数据类型组合通过模板特化实现不同的计算策略
- **零开销抽象**: 编译期内联所有模板调用，无虚函数开销
