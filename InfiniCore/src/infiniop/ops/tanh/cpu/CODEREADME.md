# Tanh CPU 操作核心实现文档

Tanh CPU 操作是 InfiniOp 框架中双曲正切激活函数的 CPU 后端实现，提供对 1D、2D、3D 及更高维张量的逐元素 tanh 计算支持。该模块采用模板元编程和宏驱动的架构设计，通过继承通用元素操作框架实现类型安全和跨类型复用。

## 1. 模块结构

- **`tanh_cpu.h`**: 核心头文件，定义 `TanhOp` 函数子（Functor）和 `Descriptor` 类结构，使用 `ELEMENTWISE_DESCRIPTOR` 宏生成完整的描述符类
- **`tanh_cpu.cc`**: 实现文件，包含描述符的析构函数、创建逻辑（`create`）和计算调度（`calculate`）

## 2. 核心类

### `TanhOp`
- **位置**: `tanh_cpu.h:10-18`
- **主要功能**: 定义双曲正切操作的函数子（Functor），作为元素操作的可调用单元
- **关键成员**:
  - `num_inputs` (静态常量 `size_t = 1`): 指定操作所需输入张量数量，tanh 为单输入操作
- **核心方法**:
  ```cpp
  template <typename T>
  T operator()(const T &input) const
  ```
  - **功能**: 对单个标量输入计算 tanh 值
  - **算法**: 调用标准库 `std::tanh(input)`，使用平台优化的数学库实现
  - **复杂度**: O(1) 每元素
  - **类型支持**: 模板化设计，支持 `float`, `double`, `fp16_t`, `bf16_t` 等浮点类型

### `Descriptor`
- **位置**: 通过 `ELEMENTWISE_DESCRIPTOR(tanh, cpu)` 宏生成于 `tanh_cpu.h:7`
- **主要功能**: 管理张量元数据和计算资源，协调输入验证、内存布局分析和计算内核调度
- **关键成员** (由宏展开生成):
  - `_dtype` (infiniDtype_t): 输出张量的数据类型（F16/F32/F64/BF16）
  - `_info` (ElementwiseInfo): 封装张量形状、步幅、连续性等元数据
  - `_device_info` (unique_ptr<DeviceImpl>): CPU 设备特定实现的管理对象
  - `_workspace_size` (size_t): 临时缓冲区大小（tanh 操作为 0）
- **核心方法**:

  **析构函数**:
  ```cpp
  ~Descriptor()
  ```
  - **实现**: `default` (定义于 `tanh_cpu.cc:5`)
  - **生命周期**: 智能指针自动管理 `_device_info`，无需手动释放资源

  **创建函数** (静态方法):
  ```cpp
  static infiniStatus_t create(
      infiniopHandle_t handle_,               // CPU 设备句柄
      Descriptor **desc_ptr,                  // 输出参数：指向新创建的描述符指针
      infiniopTensorDescriptor_t out_desc,    // 输出张量描述符
      std::vector<infiniopTensorDescriptor_t> input_desc_vec) // 输入张量描述符列表
  ```
  - **功能**: 验证张量兼容性，构建 `ElementwiseInfo` 元数据，实例化描述符对象
  - **验证流程**:
    1. **类型检查** (`CHECK_DTYPE`, 行 20): 验证输出数据类型在支持列表内（F16/F32/F64/BF16）
    2. **形状一致性** (`CHECK_SAME_SHAPE`, 行 22): 确保输入输出张量形状完全匹配（不支持广播）
  - **元数据构建** (行 25):
    ```cpp
    CREATE_ELEMENTWISE_CPU_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec)
    ```
    调用链:
    - `ElementwiseInfo::create()`: 扁平化存储形状/步幅数组，计算连续性标志
    - 分配元数据内存: `ndim * (shape + strides) + input_count * ndim * (shape + strides) + 2 * input_count * bool`
    - 复制张量属性到紧凑 `_meta` 向量
  - **错误处理**: 返回 `INFINI_STATUS_SUCCESS` 或相应错误码（类型不匹配/形状不一致/内存分配失败）

  **计算函数**:
  ```cpp
  infiniStatus_t calculate(
      void *workspace,              // 未使用（tanh 不需要临时缓冲）
      size_t workspace_size,        // 必须为 0
      void *output,                 // 输出张量数据指针
      std::vector<const void *> inputs, // 输入张量数据指针列表
      void *stream) const           // CPU 流上下文（未使用）
  ```
  - **功能**: 根据数据类型调度计算内核
  - **调度策略** (行 37-48): 基于 `_dtype` 的 switch-case 分发
    - `INFINI_DTYPE_F16`: 调用 `_device_info->calculate<TanhOp, fp16_t>(...)`
    - `INFINI_DTYPE_F32`: 调用 `_device_info->calculate<TanhOp, float>(...)`
    - `INFINI_DTYPE_F64`: 调用 `_device_info->calculate<TanhOp, double>(...)`
    - `INFINI_DTYPE_BF16`: 调用 `_device_info->calculate<TanhOp, bf16_t>(...)`
  - **内核委托**: 最终调用 `DeviceImpl::calculate` (定义于 `elementwise_cpu.h:184-193`)
    - **同类型模板实例化**: 所有输入类型相同时优化
    - **并行化**: 使用 OpenMP (`#pragma omp parallel for if (output_size > 1024)`)
    - **索引计算**:
      - 连续张量: 直接线性索引 `i`
      - 非连续张量: `indexToOffset()` 行列式计算物理偏移
    - **类型转换** (行 175-179):
      - FP16/BF16: 先提升至 `float` 调用 `operator()`，再转换回原类型（避免精度损失）
      - F32/F64: 直接计算
  - **复杂度**: O(n)，n 为输出张量元素总数
  - **线程安全**: 只读操作，多线程独立处理不同元素，无数据竞争

## 3. API 接口

### 公共 C 接口 (通过 `Descriptor` 类暴露)
```cpp
// 创建 tanh 操作描述符
infiniStatus_t infiniopCreateTanhCpuDescriptor(
    infiniopHandle_t handle,
    infiniopDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc);

// 执行 tanh 计算
infiniStatus_t infiniopTanhCpu(
    infiniopDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream);
```

### 内部操作符接口
```cpp
namespace op::tanh::cpu {
    struct TanhOp {
        static constexpr size_t num_inputs = 1;

        template <typename T>
        T operator()(const T &input) const {
            return std::tanh(input);
        }
    };
}
```

## 4. 使用示例

```cpp
// 初始化 CPU 设备句柄
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_CPU, 0);

// 创建输入/输出张量描述符（形状: [2, 3], 类型: F32）
int64_t shape[] = {2, 3};
int64_t strides[] = {3, 1}; // 行主序
infiniopTensorDescriptor_t input_desc, output_desc;
infiniopCreateTensorDescriptor(&input_desc, INFINI_DTYPE_F32, 2, shape, strides);
infiniopCreateTensorDescriptor(&output_desc, INFINI_DTYPE_F32, 2, shape, strides);

// 创建 tanh 操作描述符
op::tanh::cpu::Descriptor *tanh_desc;
auto status = op::tanh::cpu::Descriptor::create(
    handle,
    &tanh_desc,
    output_desc,
    {input_desc});

// 准备数据
float input_data[6] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
float output_data[6];

// 执行计算
status = tanh_desc->calculate(
    nullptr,           // workspace (tanh 不需要)
    0,                 // workspace_size
    output_data,       // 输出缓冲区
    {input_data},      // 输入缓冲区列表
    nullptr);          // stream (CPU 后端未使用)

// 结果: output_data ≈ {-0.964, -0.762, 0.0, 0.762, 0.964, 0.995}

// 清理资源
delete tanh_desc;
infiniopDestroyTensorDescriptor(input_desc);
infiniopDestroyTensorDescriptor(output_desc);
infiniopDestroyHandle(handle);
```

### 非连续张量计算示例
```cpp
// 处理转置矩阵（形状: [3, 2], 步幅: [1, 3]）
int64_t transposed_shape[] = {3, 2};
int64_t transposed_strides[] = {1, 3}; // 列主序（转置后）
infiniopTensorDescriptor_t transposed_desc;
infiniopCreateTensorDescriptor(&transposed_desc, INFINI_DTYPE_F32,
                               2, transposed_shape, transposed_strides);

// tanh 自动处理非连续布局
op::tanh::cpu::Descriptor *tanh_desc;
op::tanh::cpu::Descriptor::create(
    handle, &tanh_desc, transposed_desc, {transposed_desc});

// calculate() 内部会使用 indexToOffset() 计算正确索引
```

## 5. 实现细节

### 内存管理
- **元数据扁平化**: `ElementwiseInfo` 使用单一 `std::vector<size_t>` 存储所有张量的形状/步幅，避免嵌套结构开销
  - 布局: `[输出形状(ndim)] [输出步幅(ndim)] [输入0形状(ndim)] [输入0步幅(ndim)] ... [连续性标志(n)] [广播标志(n)]`
  - 内存对齐: 使用 `CEIL_DIV(meta_mem_size, sizeof(size_t))` 确保对齐
- **零拷贝计算**: 直接在用户提供缓冲区上操作，无中间分配
- **智能指针**: `_device_info` 使用 `unique_ptr`，析构时自动释放

### 并发策略
- **OpenMP 并行化**: 使用 `#pragma omp parallel for if (output_size > 1024)` 条件并行
  - 小张量（≤1024 元素）: 串行执行，避免线程调度开销
  - 大张量（>1024 元素）: 多线程并行，默认使用所有可用 CPU 核心
- **无数据竞争**: 每个线程写入不同输出元素，只读共享输入，无需锁
- **调度策略**: OpenMP runtime 自动选择静态调度（通常为均等 chunk 分配）

### 性能优化
- **类型提升策略**: FP16/BF16 计算时临时提升至 `float`，避免半精度浮点的精度损失和性能问题
  - 理由: x86/ARM CPU 对半精度浮点缺乏硬件加速，提升至标准精度更高效
- **连续性优化**: 通过 `isOutputContiguous()` 和 `getInputContiguous()[i]` 快路径
  - 连续张量: 省略 `indexToOffset()` 行列式计算，直接使用线性索引
  - 代价: 每元素减少 O(ndim) 乘加运算
- **缓存友好**: 按最内层维度顺序遍历（行主序），最大化空间局部性
- **编译器优化**: 模板实例化为每种类型生成专用代码，允许内联和循环展开

### 错误处理
- **类型验证** (`CHECK_DTYPE`): 编译期+运行期双重检查
  - 支持类型白名单: `F16`, `F32`, `F64`, `BF16`
  - 不支持整数类型（tanh 为浮点函数）
- **形状验证** (`CHECK_SAME_SHAPE`): 运行期逐维比较
  - 拒绝形状不匹配: 返回 `INFINI_STATUS_BAD_TENSOR_STRIDES`
  - 不支持广播: 要求输入输出形状完全相同
- **Result 类型** (父类 `ElementwiseInfo::create` 返回值):
  - 类 Rust 的错误传播: `utils::Result<ElementwiseInfo>`
  - `CHECK_RESULT` 宏自动解包错误，提前返回
- **异常安全**: 所有路径保证 RAII，无异常抛出

### 设计模式
- **CRTP (奇异递归模板模式)**: `ELEMENTWISE_DESCRIPTOR` 宏将操作名（`tanh`）和命名空间（`cpu`）作为模板参数
- **策略模式**: `TanhOp` 作为可调用策略，通过模板参数注入 `DeviceImpl::calculate`
- **工厂模式**: `Descriptor::create()` 静态方法封装实例化逻辑
- **桥接模式**: `Descriptor` (接口) 与 `DeviceImpl` (实现) 分离，支持多后端扩展
- **模板方法模式**: `calculate_impl` 定义算法骨架，`TanhOp::operator()` 定义可变步骤

### 依赖关系
- **直接依赖**:
  - `elementwise/cpu/elementwise_cpu.h`: 通用元素操作框架
  - `<cmath>`: 标准数学库（`std::tanh`）
  - `<utility>`: `std::move`, `std::forward` (由父类包含)
- **间接依赖**:
  - `elementwise/elementwise.h`: `ElementwiseInfo` 元数据结构
  - `devices/cpu/common_cpu.h`: CPU 通用工具函数（如 `indexToOffset`）
  - `utils/custom_types.h`: `fp16_t`, `bf16_t` 半精度类型定义
  - `utils/result.h`: `Result<T>` 错误处理类型
- **外部依赖**:
  - **OpenMP**: 并行化支持（编译器标志: `-fopenmp`）
  - **C++ 标准库**: 数学函数（`libm`）
  - **硬件优化库**: Intel MKL / AMD libm (可选，通过 `std::tanh` 调用)

### 算法正确性
- **数学定义**: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
  - 数值范围: (-1, 1)
  - 渐近行为: lim_{x→∞} tanh(x) = 1, lim_{x→-∞} tanh(x) = -1
  - 奇函数: tanh(-x) = -tanh(x)
- **边界情况**:
  - x = 0: tanh(0) = 0（精确）
  - x → ∞: tanh(x) → 1（数值精度内）
  - x = ±∞: tanh(±inf) = ±1（IEEE 754 定义）
  - x = NaN: tanh(NaN) = NaN（IEEE 754 传播）
- **类型精度**:
  - F32: ~7 位十进制精度，最大误差 < 1 ULP
  - F64: ~15 位十进制精度，最大误差 < 1 ULP
  - FP16/BF16: 通过 F32 提升计算，误差受半精度存储限制

### 扩展性
- **添加新数据类型**: 修改 `CHECK_DTYPE` 白名单和 `calculate()` switch case
- **支持广播**: 移除 `CHECK_SAME_SHAPE`，使用 `ElementwiseInfo` 的 `getInputBroadcasted()` 标志
- **SIMD 优化**: 在 `calculate_impl` 中使用 SIMD intrinsics (如 AVX-512 `_mm256_tanh_ps`)
- **自定义调度**: 替换 OpenMP 为 TBB / 自定义线程池
