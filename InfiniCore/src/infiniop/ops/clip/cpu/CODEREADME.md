# CLIP CPU 算子核心实现文档

本模块实现了 Clip（裁剪）算子的 CPU 后端，提供元素级张量裁剪功能，将输入张量的值限制在指定范围内。这是 Infini 框架中基础的一元（实际上是三元，包含输入和上下界）逐元素操作。

## 1. 模块结构

- **`clip_cpu.h`**: Clip 算子的核心操作定义和描述符声明，使用宏生成标准化的 Descriptor 类接口
- **`clip_cpu.cc`**: CPU 后端的算子创建和计算逻辑实现，包含类型分发和执行调度

## 2. 核心类与数据结构

### `op::clip::cpu::ClipOp`
- **位置**: `clip_cpu.h`
- **主要功能**: 定义 Clip 操作的语义，实现元素级的数值裁剪逻辑
- **核心成员**:
  - `num_inputs` (static constexpr size_t): 值为 3，表示需要三个输入张量（数据、最小值、最大值）
- **核心方法**:
  ```cpp
  template <typename T>
  T operator()(const T &x, const T &min_val, const T &max_val) const
  ```
  - **算法**: 使用 `std::min` 和 `std::max` 组合实现裁剪
  - **计算公式**: `return std::max(std::min(x, max_val), min_val);`
  - **语义**: 先将 x 限制在不超过 max_val，再确保不小于 min_val
  - **时间复杂度**: O(1) 每元素
- **生命周期**: 无状态函数对象（Stateless Functor），可安全复制和传递

### `op::clip::cpu::Descriptor`
- **位置**: 通过 `ELEMENTWISE_DESCRIPTOR(clip, cpu)` 宏生成于 `clip_cpu.h`
- **主要功能**: CPU 后端的算子描述符，管理张量元数据和执行上下文
- **核心成员** (继承自宏生成):
  - `_dtype` (infiniDtype_t): 输出张量的数据类型（F16/F32/F64/BF16）
  - `_info` (op::elementwise::ElementwiseInfo): 张量形状、步长、连续性等元数据
  - `_device_info` (std::unique_ptr<op::elementwise::cpu::DeviceImpl>): CPU 设备实现指针
  - `_workspace_size` (size_t): 工作空间大小（本算子为 0）
- **核心方法**:
  - **`create(...)`**: 静态工厂方法，创建并初始化描述符
    - 验证数据类型支持（F16/F32/F64/BF16）
    - 验证输入/输出形状一致性
    - 构建元数据信息（ElementwiseInfo）
    - 返回 INFINI_STATUS_SUCCESS 或错误码

  - **`calculate(...)`**: 执行裁剪计算
    - 根据 `_dtype` 分发到对应的模板实例化
    - 调用底层 `_device_info->calculate<ClipOp, T>()` 执行
    - 支持的数据类型：
      - `INFINI_DTYPE_F16`: 半精度浮点（fp16_t）
      - `INFINI_DTYPE_F32`: 单精度浮点（float）
      - `INFINI_DTYPE_F64`: 双精度浮点（double）
      - `INFINI_DTYPE_BF16`: BFloat16 格式（bf16_t）

## 3. API 接口

```cpp
// 创建 Clip 算子描述符
infiniStatus_t op::clip::cpu::Descriptor::create(
    infiniopHandle_t handle_,                    // CPU 设备句柄
    Descriptor **desc_ptr,                       // [输出] 描述符指针
    infiniopTensorDescriptor_t out_desc,         // 输出张量描述符
    std::vector<infiniopTensorDescriptor_t> input_desc_vec  // 输入张量描述符向量
);
// 返回: 成功返回 INFINI_STATUS_SUCCESS
//       失败返回 BAD_TENSOR_DTYPE / BAD_TENSOR_SHAPE

// 执行 Clip 计算
infiniStatus_t op::clip::cpu::Descriptor::calculate(
    void *workspace,                             // 工作空间指针（可为 nullptr）
    size_t workspace_size,                       // 工作空间大小
    void *output,                                // 输出张量数据指针
    std::vector<const void *> inputs,            // 输入数据指针向量 [data, min, max]
    void *stream                                 // 执行流（CPU 后端忽略）
) const;
// 返回: 成功返回 INFINI_STATUS_SUCCESS
//       失败返回 BAD_TENSOR_DTYPE
```

## 4. 使用示例

```cpp
// 假设有初始化的 CPU 句柄和张量描述符
infiniopHandle_t cpu_handle;
infiniopTensorDescriptor_t input_desc, min_desc, max_desc, output_desc;

// 准备张量数据（例如 float 类型）
float* input_data = new float[100];     // 输入数据
float* min_data = new float[100];       // 最小值边界
float* max_data = new float[100];       // 最大值边界
float* output_data = new float[100];    // 输出缓冲区

// 填充示例数据
for (int i = 0; i < 100; ++i) {
    input_data[i] = (float)(i - 50);    // -50 到 49
    min_data[i] = -10.0f;               // 统一最小值
    max_data[i] = 10.0f;                // 统一最大值
}

// 创建 Clip 描述符
op::clip::cpu::Descriptor* clip_desc = nullptr;
auto status = op::clip::cpu::Descriptor::create(
    cpu_handle,
    &clip_desc,
    output_desc,
    {input_desc, min_desc, max_desc}
);

if (status == INFINI_STATUS_SUCCESS) {
    // 执行裁剪操作
    std::vector<const void*> inputs = {input_data, min_data, max_data};
    status = clip_desc->calculate(
        nullptr,           // 无需工作空间
        0,                 // 工作空间大小为 0
        output_data,
        inputs,
        nullptr            // CPU 后端忽略流参数
    );

    // 结果: output_data[i] 会被裁剪到 [-10, 10] 范围内
    // 例如: -50 -> -10, 49 -> 10, 5 -> 5

    delete clip_desc;
}

// 清理资源
delete[] input_data;
delete[] min_data;
delete[] max_data;
delete[] output_data;
```

## 5. 实现细节

### 5.1 继承体系与代码生成
- **宏驱动设计**: 使用 `ELEMENTWISE_DESCRIPTOR(clip, cpu)` 宏自动生成标准化的 Descriptor 类
  - 避免手动编写重复的样板代码
  - 保证所有逐元素操作的一致性接口
  - 宏展开定义于 `infiniop/elementwise/elementwise.h:15-54`

### 5.2 类型分发策略
- **运行时类型检查**: 在 `create()` 中使用 `CHECK_DTYPE` 宏验证支持的数据类型
- **编译时特化**: `calculate()` 使用 switch 语句分发到四个模板实例化
  - 每个数据类型生成独立的机器码，避免运行时分支开销
  - 半精度类型（fp16_t/bf16_t）在计算前转换为 float 进行中间计算

### 5.3 内存管理
- **零工作空间**: Clip 是纯逐元素操作，不需要额外缓冲区，`_workspace_size = 0`
- **元数据封装**: `ElementwiseInfo` 结构体紧凑存储所有张量的形状、步长和连续性标志
  - 使用连续的 `std::vector<size_t>` 存储，减少内存碎片
  - 内存布局: `[output_shape][output_strides][input_shapes...][input_strides...][contiguous_flags][broadcast_flags]`

### 5.4 并行计算
- **OpenMP 并行化**: 底层 `elementwise_cpu.h` 中的 `calculate_impl` 使用 `#pragma omp parallel for`
  - 自动将元素循环分配到多个 CPU 核心
  - 条件并行: 当 `output_size > 1024` 时才启用并行（避免小数据集的线程创建开销）
  - 默认调度策略: 静态调度，负载均衡

### 5.5 广播与步长支持
- **非连续张量**: 支持任意步长的张量，通过 `indexToOffset` 函数计算线性索引
  - 连续张量优化路径: 直接使用扁平索引 `i`
  - 非连续回退路径: 运行时计算多维到线性的偏移量
- **广播机制**: 元数据中记录每个输入的广播标志，自动处理形状对齐

### 5.6 数据类型转换
- **半精度提升**: fp16_t 和 bf16_t 在运算时转换为 float（通过 `utils::cast<float>`）
  - 避免半精度运算的精度损失
  - 计算结果再转回原类型写入输出
- **通用类型转换**: `utils::cast<To, From>` 模板函数使用 `if constexpr` 编译时分支

### 5.7 错误处理
- **创建时验证**:
  - `CHECK_DTYPE`: 数据类型不在支持列表时返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
  - `CHECK_SAME_SHAPE`: 输入/输出形状不匹配时返回 `INFINI_STATUS_BAD_TENSOR_SHAPE`
- **执行时检查**:
  - 未知数据类型返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
  - 所有错误在 stderr 打印详细诊断信息（文件名、行号、条件）

### 5.8 性能特性
- **时间复杂度**: O(n)，n 为输出张量元素总数
- **空间复杂度**: O(1) 额外空间（原地修改或写入独立输出）
- **缓存友好**:
  - 连续内存访问模式
  - OpenMP 并行时每个线程处理连续的块，减少 false sharing
- **分支预测友好**: 内层循环无分支（使用条件运算符 std::max/min）

### 5.9 设计模式
- **策略模式 (Strategy Pattern)**: ClipOp 作为可插拔的策略对象，传递给通用的 elementwise 计算框架
- **工厂模式 (Factory Pattern)**: `create()` 静态方法封装对象构造逻辑
- **模板方法模式 (Template Method)**: 基类框架定义算法骨架，子类特化具体操作
- **RAII**: 使用 `unique_ptr` 管理 DeviceImpl 生命周期

### 5.10 依赖关系
- **核心依赖**:
  - `elementwise/cpu/elementwise_cpu.h`: 提供 ElementwiseInfo 和 DeviceImpl 框架
  - `devices/cpu/common_cpu.h`: 提供 indexToOffset 索引计算函数
  - `utils/custom_types.h`: 提供 fp16_t/bf16_t 类型定义和转换函数
  - `utils/check.h`: 提供 CHECK_DTYPE/ CHECK_SAME_SHAPE 宏
- **外部依赖**: OpenMP（可选，通过 ENABLE_OMP 编译标志控制）
