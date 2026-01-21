# Sigmoid CPU Operator Core Implementation Documentation

该模块实现了 Sigmoid 激活函数的 CPU 后端计算，基于 InfiniOp 的元素级操作框架。Sigmoid 函数是神经网络中常用的激活函数，将输入映射到 (0, 1) 区间。

## 1. Module Structure

- **`sigmoid_cpu.h`**: 定义 Sigmoid 操作的核心算子和描述符宏，包含数学公式的 functor 实现
- **`sigmoid_cpu.cc`**: 实现 CPU 描述符的创建、验证和计算分发逻辑

## 2. Core Classes

### `SigmoidOp` (Functor)
- **Location**: `sigmoid_cpu.h:9-16`
- **Primary Function**: 实现 Sigmoid 激活函数的数学计算逻辑
- **Key Members**:
  - `num_inputs` (static constexpr size_t): 固定为 1，表示单输入操作
- **Core Methods**:
  - `operator()(const T &x) const`: 计算 Sigmoid 函数值
    - **算法**: σ(x) = 1 / (1 + e^(-x))
    - **实现细节**: 使用 `std::exp(-x)` 计算指数，通过 `T(1)` 确保类型安全的字面量
    - **时间复杂度**: O(1) per element，主要开销在指数函数计算
    - **数值稳定性**: 标准实现，对于极端负值可能下溢（但不影响本模块范围）
- **Lifecycle**: 无状态 functor，可按值构造和传递

### `Descriptor` (Macro-generated Class)
- **Location**: 通过 `ELEMENTWISE_DESCRIPTOR(sigmoid, cpu)` 宏生成 (定义于 `elementwise.h:15-54`)
- **Primary Function**: 管理 Sigmoid 操作的元数据、设备和执行状态
- **Key Members**:
  - `_dtype` (infiniDtype_t): 输出张量的数据类型（F16/F32/F64/BF16）
  - `_info` (ElementwiseInfo): 预计算的张量形状、步长、广播信息
  - `_device_info` (unique_ptr<DeviceImpl>): CPU 设备特定实现封装
  - `_workspace_size` (size_t): 额外工作空间需求（本操作为 0）
  - 继承自 `InfiniopDescriptor` 包含 `device_type` 和 `device_id`
- **Core Methods**:
  - `create(handle_, desc_ptr, out_desc, input_desc_vec)`:
    - **功能**: 创建并验证描述符实例
    - **验证项**:
      - 数据类型必须是 F16、F32、F64 或 BF16
      - 输入输出形状必须完全一致（无广播语义）
    - **执行流程**:
      1. 类型转换句柄到 `device::cpu::Handle*`
      2. 调用 `ElementwiseInfo::create()` 预计算元数据
      3. 调用 `CREATE_ELEMENTWISE_CPU_DESCRIPTOR` 宏初始化描述符
    - **返回**: `INFINI_STATUS_SUCCESS` 或相应的错误码

  - `calculate(workspace, workspace_size, output, inputs, stream)`:
    - **功能**: 执行 Sigmoid 计算的主入口
    - **分发策略**: 基于 `_dtype` 进行编译期类型分发
      - `INFINI_DTYPE_F16` → 调用 `_device_info->calculate<SigmoidOp, fp16_t>(...)`
      - `INFINI_DTYPE_F32` → 调用 `_device_info->calculate<SigmoidOp, float>(...)`
      - `INFINI_DTYPE_F64` → 调用 `_device_info->calculate<SigmoidOp, double>(...)`
      - `INFINI_DTYPE_BF16` → 调用 `_device_info->calculate<SigmoidOp, bf16_t>(...)`
    - **半精度处理**: FP16/BF16 会先转为 float 计算，再转回原类型（在 `elementwise_cpu.h:175-176`）
    - **返回**: `INFINI_STATUS_SUCCESS` 或 `INFINI_STATUS_BAD_TENSOR_DTYPE`

  - `~Descriptor()`: 默认析构函数 (defined in `sigmoid_cpu.cc:5`)

- **Lifecycle**:
  1. **创建**: 通过静态 `create()` 方法，由 `ElementwiseInfo::create()` 构造元数据
  2. **配置**: 构造时传入 dtype、info、device_info 等
  3. **执行**: 多次调用 `calculate()` 执行计算
  4. **销毁**: 显式调用析构函数释放资源（`_device_info` 自动管理）

## 3. API Interface

```cpp
// 公共 API：创建 Sigmoid 描述符
infiniStatus_t op::sigmoid::cpu::Descriptor::create(
    infiniopHandle_t handle_,              // [in] CPU 设备句柄
    Descriptor **desc_ptr,                 // [out] 输出描述符指针
    infiniopTensorDescriptor_t out_desc,   // [in] 输出张量描述符
    std::vector<infiniopTensorDescriptor_t> input_desc_vec  // [in] 输入张量描述符向量（大小为1）
);
// 返回：INFINI_STATUS_SUCCESS / INFINI_STATUS_BAD_TENSOR_DTYPE / INFINI_STATUS_BAD_TENSOR_SHAPE

// 公共 API：执行 Sigmoid 计算
infiniStatus_t op::sigmoid::cpu::Descriptor::calculate(
    void *workspace,                       // [in] 工作空间缓冲区（本操作不使用）
    size_t workspace_size,                 // [in] 工作空间大小
    void *output,                          // [out] 输出张量数据指针
    std::vector<const void *> inputs,      // [in] 输入张量数据指针向量（大小为1）
    void *stream                           // [in] CPU 流（当前未使用）
) const;
// 返回：INFINI_STATUS_SUCCESS / INFINI_STATUS_BAD_TENSOR_DTYPE
```

## 4. Usage Example

```cpp
#include "infiniop/ops/sigmoid/cpu/sigmoid_cpu.h"
#include "infiniop/handle.h"

// 初始化 CPU 设备句柄
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_CPU, 0);

// 准备张量描述符（假设输入输出都是 float32 类型，形状为 {1024, 1024}）
std::vector<size_t> shape = {1024, 1024};
std::vector<ptrdiff_t> strides = {1024, 1};  // 行主序

infiniopTensorDescriptor_t x_desc, y_desc;
infiniopCreateTensorDescriptor(&x_desc, INFINI_DTYPE_F32, shape.data(), strides.data(), 2);
infiniopCreateTensorDescriptor(&y_desc, INFINI_DTYPE_F32, shape.data(), strides.data(), 2);

// 创建 Sigmoid 操作描述符
op::sigmoid::cpu::Descriptor* sigmoid_desc = nullptr;
auto status = op::sigmoid::cpu::Descriptor::create(
    handle,
    &sigmoid_desc,
    y_desc,
    {x_desc}
);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误（dtype 不支持或形状不匹配）
    return;
}

// 分配并初始化输入数据
float* x_data = new float[1024 * 1024];
float* y_data = new float[1024 * 1024];
// ... 填充 x_data ...

// 执行 Sigmoid 计算
status = sigmoid_desc->calculate(
    nullptr,              // workspace
    0,                    // workspace_size
    y_data,               // output
    {x_data},             // inputs
    nullptr               // stream
);

// 清理资源
delete sigmoid_desc;
delete[] x_data;
delete[] y_data;
infiniopDestroyTensorDescriptor(x_desc);
infiniopDestroyTensorDescriptor(y_desc);
infiniopDestroyHandle(handle);
```

## 5. Implementation Details

### 内存管理
- **所有权模型**: 描述符使用裸指针 new 分配（见 `CREATE_ELEMENTWISE_CPU_DESCRIPTOR` 宏第22行），调用方负责 delete
- **元数据存储**: `ElementwiseInfo` 使用单一 `std::vector<size_t>` 压缩存储所有张量的形状和步长，通过指针偏移访问不同部分
- **零拷贝计算**: `calculate()` 直接操作用户提供的输入输出指针，无内部缓冲区分配
- **半精度转换**: FP16/BF16 在计算时临时转为 float（`utils::cast` 在 `elementwise_cpu.h:176`），避免精度损失

### 并发
- **并行策略**: 使用 OpenMP 并行化（`#pragma omp parallel for`），定义于 `elementwise_cpu.h:121` 和 `163`
- **任务调度**:
  - 小张量（≤1024 元素）：串行执行（避免线程开销）
  - 大张量（>1024 元素）：自动并行，运行时调度
- **数据竞争**: 每个线程处理独立的输出元素，无共享可变状态
- **流支持**: 当前 CPU 实现忽略 stream 参数（未来可能用于队列管理）

### 性能
- **算法复杂度**: O(n) where n 是输出张量的元素总数
- **关键路径**:
  1. `std::exp()` 调用（CPU 密集，约 10-20 CPU 周期）
  2. 除法操作（约 10-40 CPU 周期，取决于架构）
  3. 半精度类型的两次转换（FP16/BF16 ↔ float）
- **优化技术**:
  - **连续内存优化**: 检测 `_output_contiguous` 和 `_input_contiguous`，连续张量使用线性索引（`elementwise_cpu.h:123-125`）
  - **编译期分发**: dtype 通过 switch-case 分发，实现模板特化，消除分支预测失败
  - **向量化潜力**: 循环体独立，编译器可自动向量化 SIMD（需 -O3 -march=native）
- **缓存友好**: 顺序访问模式，预取器友好

### 错误处理
- **错误传播**:
  - `create()`: 返回 `infiniStatus_t` 错误码，不抛异常
  - `calculate()`: 返回错误码，调用方检查返回值
- **验证宏**:
  - `CHECK_DTYPE(dtype, ...)`: 遍历变参列表验证 dtype 合法性（`utils/check.h:47-60`）
  - `CHECK_SAME_SHAPE(y_shape, x_shape)`: 逐元素比较形状向量（`utils/check.h:67-74`）
  - 失败时打印详细错误信息到 stderr（函数名、文件、行号）
- **运行时安全**:
  - dtype 不匹配时返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
  - 形状不一致时返回 `INFINI_STATUS_BAD_TENSOR_SHAPE`

### 依赖关系
- **外部依赖**:
  - `std::exp`: C 标准库指数函数（`<cmath>`）
  - OpenMP（可选，需 `ENABLE_OMP` 编译标志）
- **内部模块依赖**:
  - `op::elementwise::ElementwiseInfo`: 元数据计算和存储（`elementwise.h:69-203`）
  - `op::elementwise::cpu::DeviceImpl`: CPU 特定计算调度（`elementwise_cpu.h:39-98`）
  - `op::common_cpu::indexToOffset`: 扁平化索引到内存偏移的转换（`common_cpu.h:19`）
  - `utils::cast<T>`: 半精度浮点类型转换（`custom_types.h:22-49`）
  - `fp16_t` / `bf16_t`: 半精度浮点类型定义（`custom_types.h:6-14`）

### 设计模式
- **CRTP (奇异递归模板模式)**: `ELEMENTWISE_DESCRIPTOR` 宏通过命名空间约定（`op::OP::NAMESPACE`）生成描述符类
- **Strategy Pattern**: `DeviceImpl` 封装 CPU 特定行为，可替换为 CUDA/ROCm 等其他后端
- **Functor Pattern**: `SigmoidOp` 实现函数对象，支持模板化通用计算
- **Type Erasure**: `Descriptor` 通过 `_dtype` + `calculate()` 的 switch-case 实现运行时多态
- **RAII**: `ElementwiseInfo` 使用移动语义管理内部 vector，`unique_ptr<DeviceImpl>` 自动释放设备资源

### 特殊考虑
- **数值稳定性**: 标准实现 σ(x) = 1/(1+e^(-x))，对于 x < -100 时 e^(-x) 上溢导致 σ(x) ≈ 0（可接受行为）
- **广播语义**: Sigmoid 严格禁止广播（CHECK_SAME_SHAPE），与逐元素操作的通用设计不同
- **线程安全**: `calculate()` 是 const 方法，无状态，多线程可并发调用不同描述符实例
- **ABI 稳定性**: 描述符布局固定，但 `ElementwiseInfo` 内部表示可能变化（通过 create() 工厂隔离）
