# TopKSoftmax CPU 实现文档

TopKSoftmax CPU 算子实现模块，提供在 CPU 上执行 Top-K 选择与 Softmax 归一化的组合操作。该模块主要用于 MoE (Mixture of Experts) 模型中的专家路由选择，实现对输入张量每一行执行 Softmax 后选择 Top-K 个最大值及其索引。

## 1. 模块结构

- **`topksoftmax_cpu.h`**: CPU 描述符声明文件，通过宏定义生成 `op::topksoftmax::cpu::Descriptor` 类
- **`topksoftmax_cpu.cc`**: CPU 实现文件，包含 TopKSoftmax 算子的核心算法逻辑和类型特化处理

## 2. 核心类

### `op::topksoftmax::cpu::Descriptor`
- **位置**: `topksoftmax_cpu.h` (通过 `DESCRIPTOR(cpu)` 宏生成), `topksoftmax_cpu.cc`
- **主要功能**: 管理 CPU 设备上的 TopKSoftmax 算子描述符，提供算子创建和计算接口
- **继承关系**: 继承自 `InfiniopDescriptor`
- **关键成员**:
  - `_opaque`: 不透明指针，用于未来扩展
  - `_info`: `TopksoftmaxInfo` 结构体，存储输入张量的元数据（数据类型、形状、步长、N、width）
  - `_workspace_size`: 工作空间大小（CPU 实现为 0）
  - `device_type`: 设备类型（CPU）
  - `device_id`: 设备 ID
- **核心方法**:
  - `~Descriptor()`: 析构函数，空实现
  - `create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t x_desc)`: 静态工厂方法，创建描述符实例并验证输入张量约束（要求第二维步长为 1，即内存连续），复杂度 O(1)
  - `calculate(void *workspace, size_t workspace_size, float *values, int *indices, const void *x, const size_t topk, const bool norm, void *stream) const`: 执行 TopKSoftmax 计算，根据输入数据类型分发到模板特化函数，时间复杂度 O(N * width * log(width))
- **生命周期**: 通过 `create` 工厂方法创建，使用 `new` 分配内存，调用方负责释放

### `TopksoftmaxInfo`
- **位置**: `../info.h`
- **主要功能**: 存储 TopKSoftmax 算子的输入张量元数据信息
- **关键成员**:
  - `xtype`: 输入张量数据类型（F32/F16/BF16）
  - `shape`: 输入张量形状向量 [N, width]
  - `x_strides`: 输入张量步长向量
  - `N`: 批次大小（token 数量）
  - `width`: 特征维度大小（专家数量）
- **核心方法**:
  - `create(infiniopTensorDescriptor_t x_desc)`: 静态工厂方法，从张量描述符创建 `TopksoftmaxInfo`，验证数据类型（仅支持 F32/F16/BF16）和形状（必须是 2D 张量），返回 `Result<TopksoftmaxInfo>`，复杂度 O(1)
  - `ndim()`: 返回张量维度数（始终为 2）
  - `dim()`: 返回最后一维大小（width）

## 3. API 接口

```cpp
// 创建 TopKSoftmax 描述符
static infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,           // InfiniOP 操作句柄
    Descriptor **desc_ptr,             // [输出] 创建的描述符指针
    infiniopTensorDescriptor_t x_desc  // 输入张量描述符 (shape: [N, width])
);
// 返回: 成功返回 INFINI_STATUS_SUCCESS，失败返回对应错误码
// 约束: x_desc 必须是 2D 张量，dtype 为 F32/F16/BF16，第二维步长必须为 1（内存连续）

// 执行 TopKSoftmax 计算
infiniStatus_t Descriptor::calculate(
    void *workspace,        // 工作空间指针（CPU 实现不使用）
    size_t workspace_size,  // 工作空间大小（CPU 实现为 0）
    float *values,          // [输出] Top-K 个最大值，形状 [N, topk]
    int *indices,           // [输出] Top-K 个最大值的索引，形状 [N, topk]
    const void *x,          // [输入] 输入数据，形状 [N, width]
    const size_t topk,      // 选择的 Top-K 个数
    const bool norm,        // 是否对 Top-K 结果进行归一化
    void *stream            // 流指针（CPU 实现不使用）
) const;
// 返回: 成功返回 INFINI_STATUS_SUCCESS，失败返回 INFINI_STATUS_BAD_TENSOR_DTYPE
```

## 4. 使用示例

```cpp
// 示例: 使用 TopKSoftmax CPU 算子进行专家路由选择

#include "infiniop/ops/topksoftmax/cpu/topksoftmax_cpu.h"

// 1. 准备输入数据 (假设 N=2 个 token, width=4 个专家)
float input_data[2][4] = {
    {1.0f, 3.0f, 2.0f, 4.0f},  // token 0 的专家得分
    {0.5f, 2.5f, 1.5f, 3.5f}   // token 1 的专家得分
};

// 2. 创建输入张量描述符
infiniopTensorDescriptor_t x_desc;
size_t shape[2] = {2, 4};
size_t strides[2] = {4, 1};  // 第二维步长必须为 1
infiniopCreateTensorDescriptor(&x_desc, INFINI_DTYPE_F32, 2, shape, strides);

// 3. 创建 TopKSoftmax 描述符
op::topksoftmax::cpu::Descriptor* topk_softmax_desc;
infiniStatus_t status = op::topksoftmax::cpu::Descriptor::create(
    handle, &topk_softmax_desc, x_desc
);

if (status == INFINI_STATUS_SUCCESS) {
    // 4. 准备输出缓冲区
    const size_t topk = 2;  // 选择 Top-2 个专家
    float values[2][2];     // 存储选中的得分
    int indices[2][2];      // 存储选中的专家索引

    // 5. 执行 TopKSoftmax 计算
    status = topk_softmax_desc->calculate(
        nullptr,                    // 无需工作空间
        0,                          // 工作空间大小为 0
        (float*)values,             // 输出: Top-K 得分
        (int*)indices,              // 输出: Top-K 索引
        (const void*)input_data,    // 输入数据
        topk,                       // Top-K 个数
        true,                       // 对结果进行归一化
        nullptr                     // CPU 实现不使用流
    );

    // 预期输出:
    // Token 0: Softmax归一化后选择Top-2
    //   - values[0] = {0.705, 0.259} (近似值，归一化后)
    //   - indices[0] = {3, 1} (专家 3 和 1)
    // Token 1: Softmax归一化后选择Top-2
    //   - values[1] = {0.703, 0.259} (近似值，归一化后)
    //   - indices[1] = {3, 1} (专家 3 和 1)
}

// 6. 清理资源
delete topk_softmax_desc;
infiniopDestroyTensorDescriptor(x_desc);
```

## 5. 实现细节

### 核心算法: `topksoftmax_cpu_one_token`

针对单个 token 的 TopKSoftmax 计算核心函数，执行以下 6 个步骤：

1. **计算最大值**: 遍历查找输入向量中的最大值，用于数值稳定性（防止 exp 溢出），复杂度 O(width)
2. **指数计算**: 对每个元素执行 `exp(x_i - max)`，计算 softmax 的分子部分，同时累加求和，复杂度 O(width)
3. **Softmax 归一化**: 将所有元素除以 exp_sum，得到概率分布，复杂度 O(width)
4. **排序**: 使用 `std::sort` 对 `<value, index>` 对按降序排序，复杂度 O(width * log(width))，使用 lambda 比较器
5. **Top-K 提取**: 提取前 topk 个值及其索引，累加前 topk 个值，复杂度 O(topk)
6. **可选归一化**: 如果 `norm=true`，将 Top-K 值除以其和，使其和为 1，复杂度 O(topk)

总体时间复杂度: **O(width * log(width))**（排序主导）
空间复杂度: **O(width)**（存储 `<value, index>` 对）

### 模板函数: `topksoftmax_cpu_func<T>`

批处理模板函数，处理 N 个 token：

1. **类型转换循环**: 对每个 token，将输入数据从 T 类型 (fp16_t/bf16_t/float) 转换为 float，使用编译期 `if constexpr` 进行类型特化：
   - `fp16_t`: 调用 `_f16_to_f32()` 转换
   - `bf16_t`: 调用 `_bf16_to_f32()` 转换
   - `float`: 直接赋值
   构造 `<value, index>` 对，复杂度 O(N * width)

2. **调用核心函数**: 对每个 token 调用 `topksoftmax_cpu_one_token()`

总体时间复杂度: **O(N * width * log(width))**
空间复杂度: **O(width)**（每个 token 的临时数组，复用）

### 描述符创建: `Descriptor::create()`

验证步骤：
1. 调用 `TopksoftmaxInfo::create()` 验证张量元数据（数据类型、形状）
2. 验证第二维步长 `x_strides[1] == 1`（确保内存连续，优化访问性能）
3. 创建 `Descriptor` 实例，`workspace_size` 设为 0（CPU 实现不需要额外工作空间）

### 计算分发: `Descriptor::calculate()`

根据输入数据类型 `_info.xtype` 分发到对应的模板特化：
- `INFINI_DTYPE_F32`: 调用 `topksoftmax_cpu_func<float>`
- `INFINI_DTYPE_F16`: 调用 `topksoftmax_cpu_func<fp16_t>`
- `INFINI_DTYPE_BF16`: 调用 `topksoftmax_cpu_func<bf16_t>`
- 其他: 返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`

### 内存管理

- **工作空间**: CPU 实现不使用额外工作空间，`workspace_size = 0`
- **临时内存**: 每个 token 使用 `std::vector<std::pair<float, size_t>>` 存储中间结果，大小为 width，在 token 循环内复用
- **类型转换**: 半精度 (FP16/BF16) 在计算前转换为 float (32-bit) 以保证精度

### 并发性

- **线程安全**: 当前实现是单线程的，不同批次需要串行执行或由调用方并行化
- **无锁设计**: 不使用任何锁机制或同步原语
- **流支持**: `stream` 参数未使用，CPU 实现是同步的

### 性能优化

1. **内存布局优化**: 强制要求第二维步长为 1（内存连续），提升缓存命中率
2. **数值稳定性**: 使用 max 减法防止 exp 溢出，避免计算 `exp(大值)` 导致无穷大
3. **类型特化**: 编译期 `if constexpr` 避免运行时类型判断开销
4. **排序算法**: 使用 `std::sort`（通常是内省排序 Introsort，结合快速排序、堆排序和插入排序），平均复杂度 O(n log n)
5. **局部性原理**: 每个 token 独立处理，数据局部性好

### 错误处理

- **类型错误**: 输入数据类型不支持时返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
- **形状错误**: 输入不是 2D 张量时，在 `TopksoftmaxInfo::create()` 中返回 `INFINI_STATUS_BAD_TENSOR_SHAPE`
- **步长错误**: 第二维步长不为 1 时返回 `INFINI_STATUS_BAD_TENSOR_STRIDES`
- **Result 类型**: 使用 `utils::Result<T>` 进行错误传播，支持函数式错误处理

### 依赖项

- **标准库**: `<algorithm>` (std::sort), `<vector>` (动态数组)
- **内部模块**:
  - `../topksoftmax.h`: 父类描述符宏定义
  - `../info.h`: TopksoftmaxInfo 元数据结构
  - `../../../../utils.h`: 通用工具函数（如 Result 类型）
  - `../../../devices/cpu/common_cpu.h`: CPU 通用定义
  - `../../../reduce/cpu/reduce.h`: 归约操作（当前未使用）
- **外部依赖**: 无（纯 CPU 实现，不依赖 CUDA 或其他加速库）

### 设计模式

- **策略模式 (Strategy Pattern)**: 通过模板特化支持不同的输入数据类型（F32/F16/BF16）
- **工厂模式 (Factory Pattern)**: `create()` 静态方法封装对象创建和验证逻辑
- **RAII (Resource Acquisition Is Initialization)**: 使用析构函数管理资源（当前为空实现）
- **宏生成模式**: `DESCRIPTOR(cpu)` 宏生成描述符类，减少重复代码
