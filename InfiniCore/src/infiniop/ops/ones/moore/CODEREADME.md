# Moore Ones 操作核心实现文档

本模块实现了针对 Moore GPU 硬件后端的 Ones（全1张量生成）操作。作为逐元素（elementwise）操作的特例，该模块在 Moore 架构上生成所有元素均为1的张量，支持14种数据类型包括布尔、整数、浮点数以及8位浮点数（FP8）和BFloat16。

## 1. 模块结构

- **`ones_moore.h`**: Moore Ones 操作的 API 声明，通过宏定义定义 Descriptor 类
- **`ones_moore_kernel.h`**: CUDA 设备端内核实现，包含 `OnesOp` 函数对象的类型特化
- **`ones_moore.mu`**: Moore 设备端实现，包含描述符创建和计算调度逻辑

## 2. 核心类

### `op::ones::moore::Descriptor`
- **位置**: `ones_moore.h` (通过 `ELEMENTWISE_DESCRIPTOR` 宏展开)
- **主要功能**: Moore Ones 操作的描述符类，管理操作的生命周期和执行
- **继承关系**: 继承自 `InfiniopDescriptor` 基类
- **关键成员**:
  - `_dtype`: `infiniDtype_t` - 输出张量的数据类型
  - `_info`: `op::elementwise::ElementwiseInfo` - 逐元素操作的元数据（形状、步长、布局信息）
  - `_device_info`: `std::unique_ptr<op::elementwise::moore::DeviceImpl>` - Moore 设备实现对象
  - `_workspace_size`: `size_t` - 所需工作空间大小

- **核心方法**:
  - `create(handle_, desc_ptr, out_desc, input_desc_vec)`: 静态工厂方法，创建 Ones 操作描述符
    - **输入验证**: 检查输出张量的数据类型是否在支持的14种类型范围内
    - **形状一致性**: 验证输入和输出张量形状完全一致
    - **工作空间计算**: `workspace_size = info.getMetaMemSize() + info.getInputSize() * sizeof(void*)`
    - **设备实现创建**: 通过 `CREATE_ELEMENTWISE_MOORE_DESCRIPTOR` 宏初始化设备端实现

  - `calculate(workspace, workspace_size, output, inputs, stream)`: 执行 Ones 计算操作
    - **工作空间检查**: 验证提供的工作空间是否足够
    - **类型分派**: 根据数据类型调用模板化的设备计算函数
    - **线程块配置**: 使用固定的 `BLOCK_SIZE=256` 进行 CUDA 内核启动
    - **返回值**: `infiniStatus_t` 状态码表示操作成功或失败

- **生命周期**:
  - **创建**: 通过 `Descriptor::create()` 静态方法构造
  - **所有权**: 调用者负责管理 Descriptor 指针的生命周期
  - **销毁**: 析构函数默认实现（在 `ones_moore.mu:8` 定义）

### `op::ones::cuda::OnesOp`
- **位置**: `ones_moore_kernel.h`
- **主要功能**: CUDA 设备端函数对象，实现类型化的全1值生成逻辑
- **特性**: C++ 函数对象模式，支持在 CUDA 内核中的编译期类型分派
- **静态成员**:
  - `num_inputs`: `constexpr size_t = 1` - 指定操作输入数量（虽然 Ones 操作实际不使用输入）

- **核心方法**:
  - `operator()(const T& x)`: 模板化函数调用运算符
    - **编译期类型分派**: 使用 `if constexpr` 在编译时为不同类型生成专用代码
    - **支持的类型映射**:
      | 类型索引 | C++ 类型 | 返回值 | 说明 |
      |---------|---------|--------|------|
      | 1 | `bool` | `true` | 布尔真值 |
      | 2 | `uint8_t` | `1` | 无符号8位整数 |
      | 3 | `int8_t` | `1` | 有符号8位整数 |
      | 4 | `int16_t` | `1` | 有符号16位整数 |
      | 5 | `int32_t` | `1` | 有符号32位整数 |
      | 6 | `int64_t` | `1` | 有符号64位整数 |
      | 7 | `uint8_t` | `1` | 无符号8位整数（重复） |
      | 8 | `uint16_t` | `1` | 无符号16位整数 |
      | 9 | `uint32_t` | `1` | 无符号32位整数 |
      | 10 | `uint64_t` | `1` | 无符号64位整数 |
      | 11 | `cuda_fp8_e4m3` | `cuda_fp8_e4m3(1.0f)` | 8位浮点数（E4M3格式） |
      | 12 | `half` | `__float2half(1.0f)` | 16位浮点数 |
      | 13 | `float` | `1.0f` | 32位浮点数 |
      | 14 | `double` | `1.0` | 64位浮点数 |
      | 19 | `cuda_bfloat16` | `__float2bfloat16(1.0f)` | BFloat16格式 |
    - **未支持类型**: 复数类型（C16, C32, C64, C128）返回 `INFINI_STATUS_NOT_IMPLEMENTED`
    - **默认分支**: 对于未知类型返回 `1.0`（兜底逻辑）

## 3. API 接口

```cpp
// 创建 Ones 操作描述符
infiniStatus_t op::ones::moore::Descriptor::create(
    infiniopHandle_t handle_,              // [in] Moore 设备句柄
    Descriptor **desc_ptr,                 // [out] 输出描述符指针
    infiniopTensorDescriptor_t out_desc,   // [in] 输出张量描述符
    std::vector<infiniopTensorDescriptor_t> input_desc_vec  // [in] 输入张量描述符向量（仅用于形状匹配）
);
// 功能：创建 Moore Ones 操作描述符，验证数据类型和形状一致性
// 返回：INFINI_STATUS_SUCCESS 或错误码（类型不支持/形状不匹配/参数无效）

// 执行 Ones 计算
infiniStatus_t op::ones::moore::Descriptor::calculate(
    void *workspace,                       // [in] 工作空间指针
    size_t workspace_size,                 // [in] 工作空间大小（字节）
    void *output,                          // [out] 输出张量数据指针
    std::vector<const void *> inputs,      // [in] 输入张量数据指针向量（未使用）
    void *stream                           // [in] CUDA 流指针
) const;
// 功能：在 Moore GPU 上执行全1张量生成操作
// 返回：INFINI_STATUS_SUCCESS 或 INFINI_STATUS_INSUFFICIENT_WORKSPACE
```

## 4. 使用示例

```cpp
// 示例：在 Moore GPU 上生成形状为 {1024, 1024} 的全1 F32 张量

#include "ones_moore.h"
#include "../../tensor.h"
#include "../../handle.h"

// 1. 初始化 Moore 设备句柄
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_MOORE, 0);

// 2. 创建输出张量描述符（1024x1024 F32 矩阵）
std::vector<size_t> shape = {1024, 1024};
std::vector<ptrdiff_t> strides = {1024, 1};  // 行主序布局
auto out_desc = new TensorDescriptor(INFINI_DTYPE_F32, shape, strides);

// 3. 创建虚拟输入张量描述符（仅用于形状匹配）
auto in_desc = new TensorDescriptor(INFINI_DTYPE_F32, shape, strides);
std::vector<infiniopTensorDescriptor_t> input_descs = {in_desc};

// 4. 创建 Ones 操作描述符
op::ones::moore::Descriptor *ones_desc;
auto status = op::ones::moore::Descriptor::create(
    handle, &ones_desc, out_desc, input_descs);

if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误（如类型不支持或形状不匹配）
    return;
}

// 5. 分配输出张量内存和工作空间
size_t workspace_size = ones_desc->workspaceSize();
void *d_output, *d_workspace;
cudaMalloc(&d_output, 1024 * 1024 * sizeof(float));
cudaMalloc(&d_workspace, workspace_size);

// 6. 准备输入向量（Ones 操作不读取输入，但需要提供占位符）
std::vector<const void *> inputs = {nullptr};  // 未使用

// 7. 创建 CUDA 流
cudaStream_t stream;
cudaStreamCreate(&stream);

// 8. 执行 Ones 操作
status = ones_desc->calculate(d_workspace, workspace_size, d_output, inputs, stream);

// 9. 同步并验证结果
cudaStreamSynchronize(stream);

// 10. 清理资源
delete ones_desc;
delete out_desc;
delete in_desc;
cudaFree(d_output);
cudaFree(d_workspace);
cudaStreamDestroy(stream);
infiniopDestroyHandle(handle);
```

## 5. 实现细节

### 宏生成架构
- **`ELEMENTWISE_DESCRIPTOR(ones, moore)`**: 位于 `ones_moore.h:6`
  - 自动生成 `op::ones::moore::Descriptor` 类定义
  - 继承自 `InfiniopDescriptor` 基类
  - 提供标准的创建和计算接口
  - 管理设备实现对象和元数据

### 设备实现复用
- **逐元素操作框架**: 复用 `elementwise_moore` 的通用基础设施
  - `ElementwiseInfo`: 存储形状、步长、广播、连续性等元数据
  - `DeviceImpl`: Moore 设备端实现模板类，提供 `calculate()` 方法
  - 支持广播和 stride 操作（虽然 Ones 操作输入形状必须一致）

### 内存布局优化
- **元数据打包**: `ElementwiseInfo` 将所有元数据紧凑打包到单一 `std::vector<size_t>` 中
  - 内存布局：`[输出形状][输出步长][所有输入形状][所有输入步长][连续性标志][广播标志]`
  - 减少内核启动时的参数传递开销

- **工作空间计算**:
  ```
  workspace_size = meta_mem_size + input_size * sizeof(void*)
                 = (ndim * 2 * sizeof(size_t) + input_size * ndim * (sizeof(size_t) + sizeof(ptrdiff_t) + 2 * sizeof(bool)) / sizeof(size_t) * sizeof(size_t)
                 + input_size * sizeof(void*)
  ```
  - `meta_mem_size`: 存储 `ElementwiseInfo` 元数据所需的字节大小
  - `input_size * sizeof(void*)`: 存储输入指针数组的额外空间

### 类型分派策略
- **编译期优化**: 使用 `if constexpr` 实现零运行时开销的类型分派
  - 每种数据类型生成独立的内核实例
  - 避免内核内的分支判断
  - 编译器可为每种类型生成最优化的代码

- **运行时分派**: `Descriptor::calculate()` 中的 `switch` 语句
  - 根据 `_dtype` 成员调用正确的模板实例化
  - 支持的14种类型映射到 CUDA 内核的不同模板特化

### Moore 架构特定配置
- **线程块大小**: 固定使用 `BLOCK_SIZE = 256`
  - 平衡寄存器使用和线程调度效率
  - 适用于 Moore 架构的 warp 大小和执行单元配置

- **数据类型支持**:
  - **完全支持**: 布尔、所有整数类型（8/16/32/64位）、FP16/FP32/FP64
  - **特殊硬件类型**: FP8 (E4M3格式)、BFloat16
  - **不支持**: 复数类型（返回 `INFINI_STATUS_NOT_IMPLEMENTED`）

### 错误处理机制
- **类型检查**: `CHECK_DTYPE` 宏（line 24-40）
  - 在描述符创建阶段验证数据类型
  - 不支持的类型立即返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`

- **形状验证**: `CHECK_SAME_SHAPE` 宏（line 42）
  - 确保输入和输出张量形状完全匹配
  - 返回 `INFINI_STATUS_BAD_TENSOR_STRIDES` 表示形状不一致

- **工作空间验证**: 运行时检查（line 57-59）
  - 如果提供的工作空间小于 `_workspace_size`，返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
  - 防止内核执行时的内存越界访问

### 依赖关系
- **上游依赖**:
  - `elementwise_moore_api.h`: 提供设备实现接口和 `CREATE_ELEMENTWISE_MOORE_DESCRIPTOR` 宏
  - `elementwise_moore.h`: 提供 `DeviceImpl` 类和元数据结构
  - `cuda/kernel.cuh`: 提供 CUDA 通用内核模板（Ones 操作不直接使用）
  - `device/moore/handle.h`: Moore 设备句柄管理

- **设计模式**:
  - **策略模式**: `OnesOp` 函数对象封装了"生成全1"的算法策略
  - **工厂模式**: `Descriptor::create()` 静态方法作为对象构造工厂
  - **模板方法模式**: `DeviceImpl::calculate()` 定义算法骨架，`OnesOp` 提供具体实现
  - **RAII**: 使用智能指针 (`std::unique_ptr`) 管理设备实现对象的生命周期

### 性能特征
- **计算复杂度**: O(N) 其中 N 为输出张量元素数量
- **内存访问模式**:
  - 写密集型操作（只写输出，不读输入）
  - 输出张量访问模式由步长决定，支持任意布局
- **并行化策略**:
  - 元素级并行：每个 CUDA 线程处理一个输出元素
  - 使用 256 线程块以充分利用 Moore GPU 的 SIMT 架构
- **零拷贝优化**: 输入数据未被读取，减少内存带宽消耗
