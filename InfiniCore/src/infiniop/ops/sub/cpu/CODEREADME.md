# CPU 减法操作核心实现文档

本模块实现了 Infini 框架中 CPU 后端的逐元素张量减法操作（Element-wise Subtraction），支持多种浮点数据类型（F16、F32、F64、BF16），并利用 OpenMP 并行化优化计算性能。

## 1. 模块结构

- **`sub_cpu.h`**: 定义减法操作的描述符类和核心操作算子，通过宏扩展生成完整的 Descriptor 接口
- **`sub_cpu.cc`**: 实现 Descriptor 的创建和计算调度逻辑，处理数据类型分发和设备信息管理

## 2. 核心类

### `op::sub::cpu::SubOp`

- **位置**: `sub_cpu.h:9-16`
- **主要功能**: 定义减法操作的语义，作为可调用对象实现逐元素减法运算
- **关键成员**:
  - `num_inputs`: 静态常量，值为 2，标识减法为二元操作
- **核心方法**:
  - `operator()(const T &a, const T &b) const`: 返回 `a - b`，实现模板化的减法运算，支持任意类型 T
- **生命周期**: 无状态结构（Stateless），仅作为编译期策略对象使用

### `op::sub::cpu::Descriptor`

- **位置**: 通过 `ELEMENTWISE_DESCRIPTOR(sub, cpu)` 宏展开定义在 `sub_cpu.h:6`
- **主要功能**: CPU 减法操作的完整描述符，继承自 `InfiniopDescriptor`，管理操作元数据、设备信息和执行逻辑
- **关键成员**:
  - `_dtype`: `infiniDtype_t`，存储输出张量的数据类型
  - `_info`: `op::elementwise::ElementwiseInfo`，封装张量形状、步长、连续性等元数据
  - `_device_info`: `std::unique_ptr<op::elementwise::cpu::DeviceImpl>`，CPU 设备特定实现（本例中为空指针，因 CPU 实现无需额外资源）
  - `_workspace_size`: `size_t`，工作空间大小（恒为 0）
  - `device_type`: `infiniDevice_t`（继承），设备类型（CPU）
  - `device_id`: `int`（继承），设备 ID
- **核心方法**:
  - `~Descriptor()`: 析构函数，默认实现
  - `workspaceSize() const`: 返回 0，表示无需额外工作空间
  - `static create(...)`: 工厂方法，验证输入输出张量并构造 Descriptor 实例
  - `calculate(...) const`: 执行减法计算，根据 `_dtype` 分发到对应类型的模板特化
- **生命周期**:
  - 通过 `Descriptor::create()` 静态方法构造
  - 使用 RAII 管理资源，`_device_info` 通过 `unique_ptr` 自动释放
  - 析构时自动清理 `_info` 中的元数据内存

## 3. API 接口

```cpp
// 工厂方法：创建减法操作描述符
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,              // CPU 设备句柄
    Descriptor **desc_ptr,                 // [输出] 指向新创建的描述符指针
    infiniopTensorDescriptor_t out_desc,   // 输出张量描述符
    std::vector<infiniopTensorDescriptor_t> input_desc_vec  // 输入张量描述符向量（2个：被减数、减数）
);
// 返回 INFINI_STATUS_SUCCESS 成功，否则返回错误码（类型不支持、形状不匹配等）

// 执行减法计算
infiniStatus_t Descriptor::calculate(
    void *workspace,                       // 工作空间指针（本实现不使用）
    size_t workspace_size,                 // 工作空间大小（必须为 0）
    void *output,                          // [输出] 输出张量数据指针
    std::vector<const void *> inputs,      // 输入张量数据指针向量（2个）
    void *stream                           // 流指针（CPU 实现不使用）
) const;
// 返回 INFINI_STATUS_SUCCESS 成功，否则返回错误码（类型不支持）

// 减法操作算子（内部使用）
struct SubOp {
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        return a - b;
    }
};
```

## 4. 使用示例

```cpp
// 示例：CPU 上的张量减法 C = A - B（形状 [256, 256]，类型 FP32）

#include "sub_cpu.h"

// 1. 准备张量描述符
size_t shape[2] = {256, 256};
ptrdiff_t strides[2] = {256, 1};  // 行主序

auto a_desc = new InfiniopTensorDescriptor(INFINI_DTYPE_F32, 2, shape, strides);
auto b_desc = new InfiniopTensorDescriptor(INFINI_DTYPE_F32, 2, shape, strides);
auto c_desc = new InfiniopTensorDescriptor(INFINI_DTYPE_F32, 2, shape, strides);

// 2. 获取 CPU 设备句柄
InfiniopHandle *handle;
device::cpu::Handle::create(&handle, 0);

// 3. 创建减法操作描述符
op::sub::cpu::Descriptor *sub_desc = nullptr;
std::vector<infiniopTensorDescriptor_t> inputs = {a_desc, b_desc};
auto status = op::sub::cpu::Descriptor::create(handle, &sub_desc, c_desc, inputs);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误（类型不支持或形状不匹配）
}

// 4. 分配并初始化数据
float *a = new float[256 * 256];
float *b = new float[256 * 256];
float *c = new float[256 * 256];
// ... 填充 a 和 b 数据 ...

// 5. 执行减法计算
std::vector<const void *> input_data = {a, b};
status = sub_desc->calculate(nullptr, 0, c, input_data, nullptr);
// 计算：c[i] = a[i] - b[i] for all i

// 6. 清理资源
delete sub_desc;
delete[] a;
delete[] b;
delete[] c;
```

## 5. 实现细节

### 设计模式与架构

- **CRTP（奇异递归模板模式）**: 通过 `ELEMENTWISE_DESCRIPTOR` 宏实现，将通用逐元素操作逻辑与具体算子解耦
- **策略模式**: `SubOp` 作为编译期策略对象，通过模板参数注入到通用计算框架中
- **RAII 资源管理**: `unique_ptr` 管理设备实现，`ElementwiseInfo` 内部使用移动语义避免内存拷贝
- **类型擦除**: 使用 `std::variant` 和模板特化实现运行时类型分发

### 元数据管理（`ElementwiseInfo` 结构）

- **内存布局**: 所有元数据（形状、步长、连续性、广播标记）紧凑存储在单一 `std::vector<size_t>` 中，通过指针偏移访问不同部分
- **布局结构**:
  ```
  _meta[0..ndim-1]:           输出形状（size_t）
  _meta[ndim..2*ndim-1]:      输出步长（ptrdiff_t）
  _meta[2*ndim..(2+n)*ndim-1]: 所有输入形状（size_t）
  _meta[(2+n)*ndim..(2+2n)*ndim-1]: 所有输入步长（ptrdiff_t）
  _meta[(2+2n)*ndim..(2+2n)*ndim+n-1]: 输入连续性标志（bool）
  _meta[(2+2n)*ndim+n..(2+2n)*ndim+2n-1]: 输入广播标志（bool）
  ```
- **内存对齐**: 使用 `CEIL_DIV(meta_mem_size, sizeof(size_t))` 确保 `size_t` 对齐
- **移动语义**: 构造函数通过右值引用接收 `meta` 向量，避免不必要的内存拷贝

### 并行化策略

- **OpenMP 并行**: 使用 `#pragma omp parallel for` 并行化主循环
- **自适应并行阈值**: 仅当 `output_size > 1024` 时启用并行化（`elementwise_cpu.h:163`），避免小张量的并行开销
- **无共享状态**: 每个线程独立计算不同输出元素，无需锁机制

### 索引计算算法

- **平坦索引转偏移量**（`indexToOffset`）:
  - **算法**: 从最低维到最高维迭代，通过模除运算分解多维索引
  - **时间复杂度**: O(ndim)，每个输出元素需遍历所有维度
  - **实现**: `common_cpu.cc:5-16`
    ```cpp
    size_t res = 0;
    for (size_t i = ndim; i-- > 0;) {
        res += (flat_index % shape[i]) * strides[i];
        flat_index /= shape[i];
    }
    return res;
    ```
- **连续性优化**: 如果张量连续（`isContiguous()` 为真），直接使用平坦索引 `i` 作为偏移量，跳过计算

### 数据类型处理

- **支持类型**: F16、F32、F64、BF16（通过 `CHECK_DTYPE` 宏验证）
- **半精度提升**: F16 和 BF16 运算时先转换为 float 类型进行计算，再转换回原类型（`elementwise_cpu.h:175-176`）
  ```cpp
  if constexpr (std::is_same_v<Tdata, fp16_t> || std::is_same_v<Tdata, bf16_t>) {
      out[out_idx] = utils::cast<Tdata>(Op{}(utils::cast<float>(ins[Is][get_input_idx(Is)])...));
  }
  ```
- **类型转换**: 使用 `utils::cast` 模板函数处理类型转换，支持 FP16/F32 互转（`custom_types.h:22-49`）

### 错误处理机制

- **类型检查**: `CHECK_DTYPE` 宏验证数据类型在支持列表中，失败返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
- **形状验证**: `CHECK_SAME_SHAPE` 宏确保所有输入输出张量形状完全一致，失败返回 `INFINI_STATUS_BAD_TENSOR_SHAPE`
- **结果类型**: `ElementwiseInfo::create()` 返回 `Result<ElementwiseInfo>`，封装成功值或错误码
  - 成功时：通过 `take()` 获取值
  - 失败时：通过 `CHECK_RESULT` 宏提前返回错误码
- **广播禁止**: 输出张量不能包含广播维度（`elementwise.h:155-157`），防止语义歧义

### 性能优化技术

- **零拷贝元数据**: `ElementwiseInfo` 复用张量描述符中的形状/步长指针，仅在必要时拷贝
- **分支预测优化**: 连续性检查（`isOutputContiguous()`、`getInputContiguous()`）置于循环外，编译器可优化条件分支
- **缓存友好性**: OpenMP 调度器默认使用静态调度，保证线程访问局部性
- **内联优化**: `indexToOffset` 等小型函数通过 `inline` 声明，减少函数调用开销

### 依赖关系

- **外部依赖**:
  - `elementwise.h`: 提供逐元素操作通用框架（`ELEMENTWISE_DESCRIPTOR` 宏、`ElementwiseInfo` 结构）
  - `elementwise_cpu.h`: CPU 特定计算实现（`DeviceImpl::calculate` 模板方法）
  - `common_cpu.h`: 通用 CPU 工具函数（`indexToOffset`）
  - `utils.h`: 工具宏（`CEIL_DIV`）和类型转换函数
  - `check.h`: 验证宏（`CHECK_DTYPE`、`CHECK_SAME_SHAPE`、`CHECK_RESULT`）
  - `tensor.h`: 张量描述符定义（`InfiniopTensorDescriptor`）
  - `custom_types.h`: 自定义类型（`fp16_t`、`bf16_t`）和转换工具
- **内部依赖**:
  - 无子模块依赖，完全依赖上层框架提供的模板和工具

### 编译期多态

- **宏展开**: `ELEMENTWISE_DESCRIPTOR(sub, cpu)` 在编译期生成完整的 `op::sub::cpu::Descriptor` 类定义
- **模板特化**:
  - 同质类型版本：`template <typename Op, typename Tdata, typename... Args>`（所有输入输出类型相同）
  - 异质类型版本：`template <typename Op, typename Tout, typename... Tin, ...>`（支持不同输入类型）
- **静态断言**: 编译期验证输入类型数量与 `Op::num_inputs` 一致（`elementwise_cpu.h:146`）

### 设备抽象

- **CPU 特化**: `DeviceImpl::Opaque` 为空结构（`elementwise_cpu.h:101`），因 CPU 实现无需设备上下文
- **统一接口**: `DeviceImpl::create()` 返回 `INFINI_STATUS_NOT_IMPLEMENTED`（`elementwise_cpu.h:104-106`），本实现不使用该接口
- **流语义**: `stream` 参数保留但未使用，保持与其他设备后端接口一致性
