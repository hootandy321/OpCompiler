# Causal Softmax CPU Backend Core Implementation Documentation

该模块实现了因果掩码softmax操作的CPU后端，主要用于Transformer模型的自注意力机制。该模块对输入张量的每一行应用因果掩码（即只保留当前位置及之前的位置），然后执行softmax归一化操作。支持FP16、BF16和FP32三种数据类型，并使用OpenMP实现多线程并行计算。

## 1. Module Structure

- **`causal_softmax_cpu.h`**: CPU后端描述符的宏定义声明，通过宏展开生成完整的Descriptor类
- **`causal_softmax_cpu.cc`**: CPU后端的具体实现，包含描述符创建、核心计算算法和类型分发逻辑

## 2. Core Classes

### `op::causal_softmax::cpu::Descriptor`
- **Location**: `causal_softmax_cpu.h` (macro定义), `causal_softmax_cpu.cc` (实现)
- **Primary Function**: CPU后端的因果softmax操作描述符，继承自`InfiniopDescriptor`基类。负责初始化操作所需的元数据（如步长、形状信息），并执行实际的因果softmax计算
- **Key Members**:
  - `Opaque *_opaque`: 不透明指针，用于存储后端特定的私有数据（当前CPU实现中未使用，设为nullptr）
  - `CausalSoftmaxInfo _info`: 存储张量形状、步长和数据类型的元数据结构体，包含：
    - `dtype`: 数据类型（F16/BF16/F32）
    - `batch_size`: 批次大小（对于2D张量为1）
    - `seq_len`: 当前序列长度（张量的倒数第二维）
    - `total_seq_len`: 总序列长度（张量的最后一维）
    - `x_stride_b/i/j`: 输入张量的batch、seq_len、total_seq_len维度的步长
    - `y_stride_b/i/j`: 输出张量的batch、seq_len、total_seq_len维度的步长
  - `size_t _workspace_size`: 工作空间大小（当前CPU实现为0，不需要额外工作空间）
- **Core Methods**:
  - `~Descriptor()`: 析构函数，当前为空实现
  - `create(infiniopHandle_t, Descriptor**, infiniopTensorDescriptor_t, infiniopTensorDescriptor_t)`: 静态工厂方法，验证输入/输出张量的形状和数据类型一致性，创建`CausalSoftmaxInfo`元数据，并构造Descriptor实例
  - `calculate(void*, size_t, void*, const void*, void*) const`: 执行因果softmax计算的主入口，根据数据类型分发到对应的模板函数
  - `workspaceSize() const`: 返回所需工作空间大小（当前返回0）
- **Lifecycle**: 由`create`静态方法动态分配（使用`new Descriptor`），由调用者负责释放。采用RAII模式，构造函数初始化所有成员变量

### `causal_softmax<T>` (Template Function)
- **Location**: `causal_softmax_cpu.cc` (lines 20-57)
- **Primary Function**: 核心计算模板函数，实现因果掩码softmax的完整算法流程。对于每个批次和序列位置，执行三阶段计算：掩码填充、最大值减法、指数化和归一化
- **Algorithm Details**:
  1. **掩码阶段**: 将当前位置`i`之后的未来位置（从`total_seq_len - seq_len + i + 1`到`total_seq_len - 1`）填充为0
  2. **最大值计算**: 对有效区域（0到`total_seq_len - seq_len + i`）计算最大值，用于数值稳定性
  3. **指数化**: 对有效区域计算`exp(x - max)`，结果存储到输出张量
  4. **归一化**: 计算指数和，并对每个元素除以该和值，得到softmax概率分布
- **Complexity**: O(batch_size × seq_len × total_seq_len)，使用OpenMP并行化外层循环
- **Thread Safety**: 使用`#pragma omp parallel for`实现线程级并行，每个线程处理独立的(batch, seq_len)位置对

## 3. API Interface

```cpp
// 创建因果softmax操作描述符
static infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,                          // [in] InfiniOp操作句柄
    Descriptor **desc_ptr,                            // [out] 输出的描述符指针
    infiniopTensorDescriptor_t y_desc,                // [in] 输出张量描述符
    infiniopTensorDescriptor_t x_desc                 // [in] 输入张量描述符
);
// 返回: 成功返回INFINI_STATUS_SUCCESS，失败返回对应错误码（BAD_TENSOR_DTYPE/BAD_TENSOR_SHAPE）

// 执行因果softmax计算
infiniStatus_t Descriptor::calculate(
    void *workspace,                                  // [in] 工作空间指针（可为nullptr）
    size_t workspace_size,                            // [in] 工作空间大小（字节）
    void *y,                                          // [out] 输出数据指针
    const void *x,                                    // [in] 输入数据指针
    void *stream                                      // [in] 流对象（CPU实现中未使用）
) const;
// 返回: 成功返回INFINI_STATUS_SUCCESS，失败返回BAD_TENSOR_DTYPE

// 获取所需工作空间大小
size_t workspaceSize() const;
// 返回: 0（CPU实现不需要额外工作空间）
```

## 4. Usage Example

```cpp
// 示例：在CPU上执行因果softmax操作
// 假设输入形状为 [batch_size, seq_len, total_seq_len] = [2, 4, 8]

#include "infiniop/ops/causal_softmax/cpu/causal_softmax_cpu.h"

// 1. 创建张量描述符
std::vector<size_t> shape = {2, 4, 8};
infiniopTensorDescriptor_t x_desc, y_desc;
// ... 初始化描述符（省略具体代码）

// 2. 创建因果softmax操作描述符
op::causal_softmax::cpu::Descriptor* causal_softmax_desc;
infiniStatus_t status = op::causal_softmax::cpu::Descriptor::create(
    handle,              // InfiniOp句柄
    &causal_softmax_desc,// 输出的描述符指针
    y_desc,              // 输出张量描述符
    x_desc               // 输入张量描述符
);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 3. 执行计算
float* x_data = /* 输入数据指针 */;
float* y_data = /* 输出数据指针 */;

status = causal_softmax_desc->calculate(
    nullptr,              // 工作空间（CPU实现不需要）
    0,                    // 工作空间大小
    y_data,               // 输出数据
    x_data,               // 输入数据
    nullptr               // 流对象（CPU实现中忽略）
);

// 4. 清理资源
delete causal_softmax_desc;

// 计算结果说明：
// 对于batch=0, seq_len位置i=2（第3个位置）:
// - 有效区域: j ∈ [0, 8-4+2] = [0, 6]，即前7个元素
// - 掩码区域: j ∈ [7, 7]，即最后一个元素被填充为0
// - softmax仅对有效区域进行归一化
```

## 5. Implementation Details

### 算法设计
- **因果掩码策略**: 通过计算有效长度`valid_len = total_seq_len - seq_len + i + 1`实现动态掩码，随着位置`i`增加，有效区域逐渐扩大，确保当前位置只能看到之前的信息
- **数值稳定性**: 使用经典的max-subtraction技巧，在指数化前减去最大值避免溢出，公式为`exp(x_i - max(x)) / Σ exp(x_j - max(x))`
- **三阶段处理**: 将计算分解为掩码→指数化→归一化三个独立循环，虽然增加内存访问次数，但代码清晰且易于维护

### 内存管理
- **零工作空间**: CPU实现完全在原计算，不需要临时缓冲区，减少内存分配开销
- **步长感知**: 通过`info->x_stride_j`和`info->y_stride_j`支持非连续内存布局，可处理转置、切片等张量视图
- **原地计算**: 理论上支持原地操作（输入输出指针相同），但当前实现通过独立指针避免别名问题

### 并发策略
- **OpenMP并行**: 外层循环使用`#pragma omp parallel for`并行化，每个线程独立处理一个(batch, i)组合，避免数据竞争
- **负载均衡**: 采用静态调度策略，适合所有迭代计算量均匀的场景（每个位置处理长度不同，但差异较小）
- **线程安全**: `reduce_op::max`和`reduce_op::sum`为只读操作，无共享状态修改

### 数据类型处理
- **模板特化**: 使用`if constexpr`编译期分支，为FP16/BF16类型添加`utils::cast<T>`类型转换，FP32直接计算
- **精度保持**: FP16/BF16的中间计算转换为FP32进行（reduce和exp操作），最终结果转回原类型，平衡精度和性能
- **类型验证**: 在`create`阶段检查输入输出类型一致性，不支持其他类型（如INT8、FP64）

### 依赖关系
- **上游依赖**:
  - `CausalSoftmaxInfo`: 提供张量元数据验证和提取
  - `op::common_cpu::reduce_op::max/sum`: 提供最大值和求和归约操作
  - `utils::cast<T>`: 提供FP16/BF16与FP32之间的类型转换
- **编译期依赖**:
  - OpenMP (`#ifdef ENABLE_OMP`): 可选依赖，未定义时退化为单线程
  - C++17 `if constexpr`: 编译期条件编译
- **运行时依赖**: 无外部库依赖，纯CPU实现

### 性能特征
- **时间复杂度**: O(B × S × T)，其中B为batch_size，S为seq_len，T为total_seq_len
- **空间复杂度**: O(1)额外空间（除输入输出外）
- **内存带宽**: 每个元素读取1次（输入），写入3次（掩码、指数、归一化），共4次内存访问
- **并行扩展性**: 理论加速比可达O(min(B×S, 线程数))，受内存带宽限制

### 错误处理
- **验证时机**: 在`create`阶段验证张量形状（2D或3D）、维度约束（total_seq_len >= seq_len）和数据类型
- **错误传播**: 使用`CHECK_RESULT`和`CHECK_STATUS`宏，遇到错误立即返回错误码
- **错误类型**: `INFINI_STATUS_BAD_TENSOR_DTYPE`（类型不匹配或不支持）、`INFINI_STATUS_BAD_TENSOR_SHAPE`（形状非法）

### 设计模式
- **策略模式**: 通过`DESCRIPTOR(cpu)`宏定义不同后端的统一接口，CPU、CUDA、BANG等后端各自实现
- **模板方法模式**: `calculate`作为非模板入口，分发到模板函数`causal_softmax<T>`，实现类型多态
- **CRTP (奇异递归模板模式)**: 通过宏定义生成类，避免代码重复，同时保持类型安全
