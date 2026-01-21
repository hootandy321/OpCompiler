# CPU Softplus 算子核心实现文档

该模块实现了 Softplus 激活函数的 CPU 后端，基于 InfiniOP 的逐元素运算(elementwise)框架，为神经网络前向推理提供数值稳定的 Softplus 计算支持。Softplus 函数定义为 log(1 + exp(x))，是 ReLU 的平滑近似。

## 1. 模块结构

- **`softplus_cpu.h`**: Softplus 算子的 CPU 特化头文件，定义核心计算逻辑 `SoftplusOp` 函数子和类型注册宏
- **`softplus_cpu.cc`**: Softplus 描述符的实现文件，包含算子创建(`create`)和计算调度(`calculate`)方法

## 2. 核心类与数据结构

### `op::softplus::cpu::SoftplusOp`
- **位置**: `softplus_cpu.h:9-20`
- **主要功能**: 实现 Softplus 激活函数的逐元素计算逻辑，采用函数子(functor)设计模式
- **关键成员**:
  - `num_inputs`: 静态常量，值为 1，标识该算子为单输入逐元素操作
- **核心方法**:
  - `operator()(const T &x) const`: Softplus 计算的主函数
    - **算法**: 使用数值稳定策略，当 x > 20 时直接返回 x（因为 log(1+exp(x)) ≈ x），否则计算 log(1 + exp(x))
    - **时间复杂度**: O(1) 每元素
    - **空间复杂度**: O(1)
    - **数值稳定性**: 通过 x > 20 的阈值避免大数指数溢出，对于 x <= 20 的值使用标准公式
- **生命周期**: 无状态(stateless)函数子，编译期可构造

### `op::softplus::cpu::Descriptor`
- **位置**: 通过 `ELEMENTWISE_DESCRIPTOR(softplus, cpu)` 宏展开生成
- **主要功能**: 继承自 `InfiniopDescriptor`，管理 Softplus 算子的元数据、类型信息和计算调度
- **关键成员**:
  - `_dtype`: `infiniDtype_t`，存储输出张量的数据类型(F16/F32/F64/BF16)
  - `_info`: `op::elementwise::ElementwiseInfo`，封装输入/输出张量的形状、步幅、连续性等元数据
  - `_device_info`: `std::unique_ptr<op::elementwise::cpu::DeviceImpl>`，CPU 设备特定实现的智能指针
  - `_workspace_size`: `size_t`，工作空间大小(当前实现为 0，无需额外内存)
- **核心方法**:
  - `~Descriptor()`: 析构函数(默认实现)
  - `create(...)`: 静态工厂方法，构造并验证 Softplus 描述符
    - **参数验证**:
      - 数据类型检查: 仅支持 `INFINI_DTYPE_F16`, `INFINI_DTYPE_F32`, `INFINI_DTYPE_F64`, `INFINI_DTYPE_BF16`
      - 形状一致性: 通过 `CHECK_SAME_SHAPE` 确保输入输出张量形状完全匹配
    - **元数据构建**: 调用 `CREATE_ELEMENTWISE_CPU_DESCRIPTOR` 宏实例化 `ElementwiseInfo`
    - **返回值**: 成功返回 `INFINI_STATUS_SUCCESS`，失败返回对应错误码
  - `calculate(...)`: 执行 Softplus 计算的调度函数
    - **分发逻辑**: 根据 `_dtype` 进行 switch-case 分发，为四种浮点类型分别调用模板化的 `_device_info->calculate<SoftplusOp, T>()`
    - **类型特化**:
      - `INFINI_DTYPE_F16`: 调用 `calculate<SoftplusOp, fp16_t>`
      - `INFINI_DTYPE_F32`: 调用 `calculate<SoftplusOp, float>`
      - `INFINI_DTYPE_F64`: 调用 `calculate<SoftplusOp, double>`
      - `INFINI_DTYPE_BF16`: 调用 `calculate<SoftplusOp, bf16_t>`
    - **底层执行**: 委托给 `op::elementwise::cpu::DeviceImpl::calculate`，后者使用 OpenMP 并行化遍历张量元素
- **生命周期**: 由 `create` 方法构造，用户负责内存管理(当前使用 `new` 分配)

## 3. API 接口

```cpp
// Softplus 算子创建函数
infiniStatus_t op::softplus::cpu::Descriptor::create(
    infiniopHandle_t handle,                // CPU 设备句柄
    Descriptor **desc_ptr,                  // 输出参数：构造的描述符指针
    infiniopTensorDescriptor_t out_desc,    // 输出张量描述符
    std::vector<infiniopTensorDescriptor_t> input_descs  // 输入张量描述符向量
);
// 返回值: INFINI_STATUS_SUCCESS | INFINI_STATUS_BAD_TENSOR_DTYPE | INFINI_STATUS_BAD_TENSOR_STRIDES

// Softplus 计算执行函数
infiniStatus_t op::softplus::cpu::Descriptor::calculate(
    void *workspace,                         // 工作空间指针(当前未使用)
    size_t workspace_size,                   // 工作空间大小
    void *output,                           // 输出张量数据指针
    std::vector<const void *> inputs,       // 输入张量数据指针向量
    void *stream                            // CUDA stream(未使用，CPU 后端)
) const;
// 返回值: INFINI_STATUS_SUCCESS | INFINI_STATUS_BAD_TENSOR_DTYPE
```

## 4. 使用示例

```cpp
// 示例：使用 CPU Softplus 算子对浮点数组进行激活
#include "softplus_cpu.h"

// 1. 准备张量描述符
std::vector<size_t> shape = {1024, 1024};
infiniopTensorDescriptor_t x_desc, y_desc;
infiniopCreateTensorDescriptor(&x_desc, INFINI_DTYPE_F32, shape.data(), shape.size());
infiniopCreateTensorDescriptor(&y_desc, INFINI_DTYPE_F32, shape.data(), shape.size());

// 2. 分配内存
size_t numel = 1024 * 1024;
float *x = new float[numel];  // 输入数据
float *y = new float[numel];  // 输出缓冲
// ... 初始化 x 数组 ...

// 3. 创建算子描述符
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_CPU, 0);

op::softplus::cpu::Descriptor *softplus_desc = nullptr;
auto status = op::softplus::cpu::Descriptor::create(
    handle,
    &softplus_desc,
    y_desc,
    {x_desc}
);

if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 4. 执行 Softplus 计算
status = softplus_desc->calculate(
    nullptr,  // 无需工作空间
    0,        // 工作空间大小为 0
    y,        // 输出
    {x},      // 输入
    nullptr   // CPU 后端不使用 stream
);

// 5. 清理资源
delete softplus_desc;
infiniopDestroyTensorDescriptor(x_desc);
infiniopDestroyTensorDescriptor(y_desc);
infiniopDestroyHandle(handle);
delete[] x;
delete[] y;
```

## 5. 实现细节

### 宏驱动架构
- **`ELEMENTWISE_DESCRIPTOR(softplus, cpu)`**: 在 `softplus_cpu.h:6` 展开，生成 `op::softplus::cpu::Descriptor` 类的完整定义
  - 继承 `InfiniopDescriptor` 基类
  - 自动实现工作空间查询接口 `workspaceSize()`
  - 声明 `create` 和 `calculate` 方法，由 `.cc` 文件提供实现

### 数值稳定性策略
- **大数优化**: 当输入 x > 20 时，直接返回 x 原值
  - 数学依据: lim(x→∞) log(1 + exp(x)) = x
  - 避免大数溢出: exp(20) ≈ 4.85×10⁸，exp(21) 可能导致精度损失
- **标准计算路径**: 对于 x ≤ 20，使用 `std::log(1 + std::exp(x))`
  - 依赖标准库的数值稳定实现
  - 对于 x < 0 的情况，log(1 + exp(x)) 不会产生下溢

### 类型转换与低精度计算
- **FP16/BF16 提升**: 在 `elementwise_cpu.h:175-178` 中，半精度类型(fp16_t/bf16_t)被提升为 float 进行计算
  ```cpp
  if constexpr (std::is_same_v<Tdata, fp16_t> || std::is_same_v<Tdata, bf16_t>) {
      out[out_idx] = utils::cast<Tdata>(Op{}(utils::cast<float>(ins[Is][get_input_idx(Is)])...));
  }
  ```
  - 防止半精度运算的精度损失和溢出风险
  - 计算完成后通过 `utils::cast<Tdata>` 转回原类型

### 并行化执行
- **OpenMP 并行**: 在 `elementwise_cpu.h:163` 使用 `#pragma omp parallel for if (output_size > 1024)`
  - 条件并行: 仅当张量元素数 > 1024 时启用多线程
  - 负载均衡: OpenMP runtime 自动分配循环迭代给线程池
  - 线程安全: 每个线程写入独立的输出元素，无需同步

### 内存布局处理
- **非连续张量支持**: 通过 `ElementwiseInfo` 存储的步幅(stride)信息处理非连续内存布局
  - 输出索引计算: `info.isOutputContiguous() ? i : indexToOffset(i, ...)`
  - 输入索引计算: 依据 `info.getInputContiguous()[input_id]` 选择线性索引或多维偏移量计算
  - 广播支持: `info.getInputBroadcasted()` 标识输入是否需要广播到输出形状

### 错误处理机制
- **类型检查**: 使用 `CHECK_DTYPE` 宏在 `create` 阶段验证数据类型
- **形状验证**: `CHECK_SAME_SHAPE` 宏确保输入输出形状完全匹配
- **返回值传播**: 所有错误通过 `infiniStatus_t` 枚举返回，调用者可据此处理异常情况

### 设计模式
- **Strategy Pattern**: `SoftplusOp` 作为策略对象，可替换为其他激活函数(如 ReLU, Sigmoid)
- **Template Method**: `ELEMENTWISE_DESCRIPTOR` 宏定义算法骨架，子类填充具体计算逻辑
- **RAII(Resource Acquisition Is Initialization)**: `Descriptor` 使用 `std::unique_ptr` 管理 `DeviceImpl` 生命周期

### 性能特性
- **复杂度**: O(n) 时间，其中 n 为张量元素数量；O(1) 额外空间
- **缓存友好**: 顺序访问模式优化 CPU 缓存命中率
- **分支预测**: x > 20 的分支在大多数输入下可预测，现代 CPU 分支预测器开销可忽略
