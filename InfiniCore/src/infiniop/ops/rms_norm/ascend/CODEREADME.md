# RMS Norm Ascend ACLNN 实现文档

本模块实现了 RMS (Root Mean Square) Layer Normalization 操作的华为昇腾 (Ascend) AI 处理器后端，基于华为 CANN (Compute Architecture for Neural Networks) 框架的 ACLNN (Ascend Compute Library Neural Network) API。该模块为 InfiniCore 框架提供了在昇腾 NPU 上高效执行 RMS 归一化的能力。

## 1. 模块结构

- **`rms_norm_aclnn.h`**: 声明文件，通过 `DESCRIPTOR(ascend)` 宏展开生成完整的 `Descriptor` 类定义，继承自父目录的通用 RMS Norm 接口模板
- **`rms_norm_aclnn.cc`**: 实现文件，包含昇腾 NPU 特定的 RMS Norm 算子实现，封装了 ACLNN API 调用、张量描述符管理和执行器生命周期管理

## 2. 核心类

### `Descriptor`
- **位置**: `rms_norm_aclnn.h` (通过宏展开定义), `rms_norm_aclnn.cc` (实现)
- **主要功能**: RMS Norm 操作的昇腾后端描述符，负责算子初始化、工作空间计算和内核执行调度
- **继承关系**: 继承自 `InfiniopDescriptor` 基类，通过 `DESCRIPTOR(ascend)` 宏在命名空间 `op::rms_norm::ascend` 中实例化
- **关键成员**:
  - `Opaque *_opaque`: 指向不透明数据结构的指针，封装 ACLNN 特定的内部实现细节
  - `RMSNormInfo _info`: 存储张量形状、数据类型和步长信息的验证后元数据
  - `size_t _workspace_size`: 计算所需的显存工作空间总大小（包括 ACLNN 内部工作空间 + rstd 输出缓冲区）

### `Descriptor::Opaque`
- **位置**: `rms_norm_aclnn.cc` (第 7-23 行)
- **主要功能**: 封装 ACLNN API 特定的资源句柄，实现 RAII (Resource Acquisition Is Initialization) 模式
- **关键成员**:
  - `aclnnTensorDescriptor_t y`: 输出张量的 ACLNN 描述符（单行切片视图）
  - `aclnnTensorDescriptor_t x`: 输入张量的 ACLNN 描述符（单行切片视图）
  - `aclnnTensorDescriptor_t w`: 权重（gamma）张量的 ACLNN 描述符（完整参数向量）
  - `aclnnTensorDescriptor_t rstd`: 均方根倒数张量的 ACLNN 描述符（1x1 标量张量，ACLNN API 强制要求非空）
  - `size_t workspaceSize`: ACLNN 算子内部所需工作空间大小（通过 `aclnnRmsNormGetWorkspaceSize` 查询获得）
  - `aclOpExecutor *executor`: ACLNN 算子执行器句柄，编译后的算子可执行对象
- **析构函数**:
  - 释放所有 `aclnnTensorDescriptor_t` 对象（调用 `delete` 触发 `aclnnTensorDescriptor` 析构函数）
  - 调用 `aclDestroyAclOpExecutor(executor)` 销毁执行器，释放 NPU 资源

## 3. API 接口

### `Descriptor::create()`
```cpp
static infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,                    // [in] 昇腾设备句柄 (device::ascend::Handle*)
    Descriptor **desc_ptr,                      // [out] 输出创建的描述符指针
    infiniopTensorDescriptor_t y_desc,          // [in] 输出张量描述符（形状: [batch, dim] 或 [batch, nhead, dim]）
    infiniopTensorDescriptor_t x_desc,          // [in] 输入张量描述符（形状同 y_desc）
    infiniopTensorDescriptor_t w_desc,          // [in] 权重张量描述符（形状: [dim]，1D 向量）
    float epsilon);                             // [in] 数值稳定项，防止除以零（典型值: 1e-6）
```

**功能**: 创建并初始化 RMS Norm 操作的昇腾后端描述符，验证输入参数并预分配 NPU 资源

**执行流程**:
1. 调用 `RMSNormInfo::create()` 验证张量形状、数据类型和步长的合法性
2. 构造单行切片张量描述符（`slice_shape = {dim}`），因为 ACLNN 的 RMS Norm 算子每次处理一个向量
3. 创建 `rstd` 标量张量描述符（形状 `[1]`，数据类型 `ACL_FLOAT`），以满足 ACLNN API 的非空要求
4. 调用 `aclnnRmsNormGetWorkspaceSize()` 查询算子所需工作空间大小并编译算子执行器
5. 调用 `aclSetAclOpExecutorRepeatable(executor)` 标记执行器可重用（优化批量处理性能）
6. 计算总工作空间大小：`workspace_size + rstd->numel() * aclDataTypeSize(rstd->dataType)`
7. 构造并返回 `Descriptor` 对象

**返回值**:
- `INFINI_STATUS_SUCCESS`: 成功创建
- `INFINI_STATUS_BAD_TENSOR_DTYPE`: 数据类型不兼容（如 x 和 y 类型不匹配）
- `INFINI_STATUS_BAD_TENSOR_SHAPE`: 张量形状不合法（非 2D/3D 或维度不匹配）
- `INFINI_STATUS_BAD_TENSOR_STRIDES`: 最后一维不连续（步长不为 1）

---

### `Descriptor::calculate()`
```cpp
infiniStatus_t Descriptor::calculate(
    void *workspace,             // [in] 预分配的工作空间缓冲区（设备内存）
    size_t workspace_size,       // [in] 工作空间大小（必须 >= workspaceSize()）
    void *y,                     // [out] 输出张量设备指针
    const void *x,               // [in] 输入张量设备指针
    const void *w,               // [in] 权重张量设备指针
    void *stream) const;         // [in] Ascend Stream 句柄（aclrtStream）
```

**功能**: 在 NPU 上异步执行 RMS Norm 计算，对输入张量的每一行（或最后一维）进行归一化

**执行流程**:
1. 验证工作空间大小是否满足要求（`workspace_size < workspaceSize()` 时返回错误）
2. 从工作空间末尾分配 `rstd` 输出缓冲区：`rstdPtr = workspace + _opaque->workspaceSize`
3. 固定不变参数绑定：
   - 索引 1 (权重 `w`): 通过 `AclSetTensorAddr()` 绑定到执行器
   - 索引 3 (rstd 输出): 绑定到工作空间末尾的 `rstdPtr`
4. 批量处理循环（处理 batch 维度）：
   - 计算当前行的数据指针偏移：`x + i * x_strides[0] * unit`, `y + i * y_strides[0] * unit`
   - 动态绑定输入/输出地址（索引 0: `x`, 索引 2: `y`）
   - 调用 `aclnnRmsNorm(workspace, _opaque->workspaceSize, _opaque->executor, stream)` 提交 NPU 内核
5. 所有操作在给定 stream 上异步执行，需通过同步原语等待完成

**数学公式**:
```
对于输入张量 x 的每一行（或最后一个维度）：
rstd = 1 / sqrt(mean(x²) + epsilon)
y = x * rstd * w
```

**返回值**:
- `INFINI_STATUS_SUCCESS`: 成功提交 NPU 内核
- `INFINI_STATUS_INSUFFICIENT_WORKSPACE`: 工作空间不足

---

### `Descriptor::~Descriptor()`
```cpp
Descriptor::~Descriptor();
```
**功能**: 析构函数，自动释放 `_opaque` 指向的 Opaque 对象及其管理的所有 ACLNN 资源

### `Descriptor::workspaceSize()`
```cpp
size_t workspaceSize() const;
```
**功能**: 返回所需工作空间的字节数（包括 ACLNN 内部缓冲区 + rstd 输出缓冲区）

## 4. 使用示例

```cpp
#include "infiniop/ops/rms_norm/rms_norm.h"
#include "infiniop/devices/ascend/ascend_handle.h"

using namespace op::rms_norm::ascend;

// 1. 初始化昇腾设备和句柄
device::ascend::Handle *handle;
device::ascend::Handle::create(&handle, 0);  // device_id = 0

// 2. 创建张量描述符
// 输入/输出张量形状: [batch_size=128, hidden_dim=768]
int64_t shape[] = {128, 768};
int64_t strides[] = {768, 1};  // 行主序（C 风格）
infiniopTensorDescriptor_t x_desc = new TensorDescriptor(
    INFINI_DTYPE_F16, 2, shape, strides, INFINI_DEVICE_ASCEND, 0);
infiniopTensorDescriptor_t y_desc = new TensorDescriptor(
    INFINI_DTYPE_F16, 2, shape, strides, INFINI_DEVICE_ASCEND, 0);

// 权重张量形状: [hidden_dim=768]
int64_t w_shape[] = {768};
int64_t w_strides[] = {1};
infiniopTensorDescriptor_t w_desc = new TensorDescriptor(
    INFINI_DTYPE_F16, 1, w_shape, w_strides, INFINI_DEVICE_ASCEND, 0);

// 3. 创建 RMS Norm 描述符并查询工作空间大小
Descriptor *rms_norm_desc;
infiniStatus_t status = Descriptor::create(
    handle, &rms_norm_desc, y_desc, x_desc, w_desc, 1e-6f);

if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误（形状不匹配、数据类型不支持等）
}

size_t workspace_size = rms_norm_desc->workspaceSize();

// 4. 分配设备内存和工作空间
void *x_d, *y_d, *w_d, *workspace;
aclrtMalloc(&x_d, x_desc->nbytes() * sizeof(half), ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(&y_d, y_desc->nbytes() * sizeof(half), ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(&w_d, w_desc->nbytes() * sizeof(half), ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);

// 从主机复制数据到设备（假设 x_h 和 w_h 是主机端指针）
aclrtMemcpy(x_d, x_desc->nbytes() * sizeof(half), x_h, ..., ACL_MEMCPY_HOST_TO_DEVICE);
aclrtMemcpy(w_d, w_desc->nbytes() * sizeof(half), w_h, ..., ACL_MEMCPY_HOST_TO_DEVICE);

// 5. 在 Stream 上执行 RMS Norm（异步）
aclrtStream stream;
aclrtCreateStream(&stream);

status = rms_norm_desc->calculate(workspace, workspace_size, y_d, x_d, w_d, stream);

if (status != INFINI_STATUS_SUCCESS) {
    // 处理执行错误（工作空间不足等）
}

// 6. 同步 Stream 并复制结果回主机
aclrtSynchronizeStream(stream);
aclrtMemcpy(y_h, y_desc->nbytes() * sizeof(half), y_d, ..., ACL_MEMCPY_DEVICE_TO_HOST);

// 7. 清理资源
aclrtFree(x_d);
aclrtFree(y_d);
aclrtFree(w_d);
aclrtFree(workspace);
aclrtDestroyStream(stream);
delete rms_norm_desc;
delete handle;
```

## 5. 实现细节

### 内存管理策略
- **RAII 资源管理**: `Opaque` 结构体通过析构函数自动释放所有 ACLNN 描述符和执行器，防止资源泄漏
- **工作空间分区**: 工作空间被划分为两个区域（从头到尾）：
  1. `ACLNN 内部工作空间` (大小: `_opaque->workspaceSize`): 由 `aclnnRmsNormGetWorkspaceSize()` 查询获得，用于算子内部临时计算
  2. `rstd 输出缓冲区` (大小: `rstd->numel() * aclDataTypeSize(rstd->dataType)`): 存储均方根倒数结果，虽然用户通常不需要此数据，但 ACLNN API 强制要求提供非空指针
- **设备内存分配**: 调用方负责在调用 `calculate()` 前通过 `aclrtMalloc()` 分配足够大的工作空间缓冲区

### 并发与执行模型
- **异步执行**: `aclnnRmsNorm()` 在给定的 Ascend Stream 上异步执行，立即返回不阻塞 CPU
- **批量处理优化**: 使用 `aclSetAclOpExecutorRepeatable(executor)` 标记执行器可重用，避免在循环中重复编译算子
- **循环提交策略**: 对 batch 维度的每一行单独提交一次内核调用（`for (size_t i = 0; i < shape[0]; ++i)`），每次动态更新输入/输出张量地址，但保持权重和 rstd 地址不变
- **Stream 同步**: 调用方需通过 `aclrtSynchronizeStream()` 或其他同步机制等待内核完成后再访问输出数据

### 性能优化技术
- **执行器重用**: 算子执行器在 `create()` 阶段编译一次，在 `calculate()` 循环中重复使用，减少编译开销
- **内存对齐**: ACLNN API 内部自动处理张量数据的内存对齐要求（通常为 32 字节对齐）
- **标量切片优化**: 构造单行切片张量描述符 (`slice_shape = {dim}`) 而非完整张量，减少 ACLNN API 的内部参数验证开销
- **数据类型支持**:
  - 输入/输出: 支持 FP16, BF16, FP32, FP64（通过 `RMSNormInfo::create()` 验证）
  - 权重: 对于半精度类型（FP16/BF16），允许权重为 FP32 以提高数值稳定性

### 错误处理与边界检查
- **类型兼容性检查**: `RMSNormInfo::create()` 验证输入、输出和权重的数据类型组合合法性（例如：FP16 输入 + FP16/BF16/FP32 权重）
- **形状验证**: 仅支持 2D `[batch, dim]` 或 3D `[batch, nhead, dim]` 张量，拒绝其他维度
- **连续性检查**: 要求最后一维步长为 1（紧凑存储），确保 ACLNN API 能正确访问向量元素
- **工作空间验证**: `calculate()` 在执行前检查 `workspace_size < workspaceSize()` 并返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
- **ACL 错误传播**: 使用 `CHECK_ACL()` 宏包装 ACL API 调用，自动将 `ACL_SUCCESS` 以外的错误码转换为 `infiniStatus_t`

### 依赖项
- **华为 CANN 框架**:
  - `acl/acl.h`: 基础 ACL (Ascend Computing Language) 运行时 API
  - `aclnn/acl_meta.h`: ACLNN 神经网络算子元数据定义
  - `aclnnop/aclnn_rms_norm.h`: RMS Norm 算子的 ACLNN API 声明
- **InfiniCore 内部组件**:
  - `infiniop/operator.h`: 基础算子描述符接口 (`InfiniopDescriptor`)
  - `infiniop/tensor.h`: 张量描述符定义 (`infiniopTensorDescriptor_t`)
  - `infiniop/devices/ascend/common_ascend.h`: 昇腾后端公共工具（`aclnnTensorDescriptor`, `toAclDataType()`, `CHECK_ACL` 宏）
  - `infiniop/ops/rms_norm/info.h`: RMS Norm 元数据验证类 (`RMSNormInfo`)
- **C++ 标准库**: `std::vector<int64_t>` 用于张量形状和步长存储

### 设计模式
- **Pimpl (Pointer to Implementation) 模式**: 使用 `Opaque` 结构体隐藏 ACLNN 特定的实现细节，保持公共头文件简洁
- **工厂方法模式**: `create()` 静态方法作为构造函数的替代，提供错误返回能力（C 风格的 `**desc_ptr` 输出参数）
- **RAII (Resource Acquisition Is Initialization)**: 通过析构函数自动管理 ACLNN 资源生命周期
- **策略模式**: 通过命名空间 `op::rms_norm::ascend` 与其他硬件后端（如 CUDA, CPU）隔离，共享相同的 `Descriptor` 接口定义（通过 `DESCRIPTOR` 宏）

### ACLNN API 约束与变通方案
- **rstd 参数非空要求**: ACLNN 的 `aclnnRmsNorm()` API 要求 `rstdTensor` 参数不能为 `nullptr`（见源代码第 59-60 行注释），即使调用者不需要此输出。实现中通过分配 1x1 标量张量描述符并从工作空间末尾分配缓冲区来满足此要求
- **向量级处理**: ACLNN RMS Norm 算子设计为处理单个向量（1D 张量），因此对多维张量需按 batch 循环调用，每次处理一行（通过动态更新张量地址实现）
- **地址绑定机制**: `AclSetTensorAddr()` 宏（具体实现依赖于 ACLNN 内部 API）用于在执行前动态绑定设备内存地址到算子参数，支持同一执行器处理不同数据缓冲区
