# Tanh 操作 NVIDIA GPU 实现核心文档

本模块实现了 Infini 框架中 Tanh（双曲正切）激活函数的 NVIDIA GPU 后端，通过 CUDA 核函数在 GPU 上执行逐元素的 tanh 运算。该实现利用逐元素操作框架，支持 FP16、FP32、FP64 和 BF16 四种精度，并针对向量化指令（half2, bfloat162）进行了优化。

## 1. 模块结构

- **`tanh_nvidia.cuh`**: 声明头文件，通过 `ELEMENTWISE_DESCRIPTOR` 宏定义 `op::tanh::nvidia::Descriptor` 类的结构和接口
- **`tanh_nvidia.cu`**: 实现文件，包含描述符的创建 (`create`) 和计算 (`calculate`) 方法的具体实现，负责数据类型校验、形状检查和 CUDA 核函数调度

## 2. 核心类

### `Descriptor` (通过宏生成)
- **位置**: `tanh_nvidia.cuh` (第 6 行)
- **主要功能**: 定义 Tanh 操作的 NVIDIA GPU 描述符，继承自 `InfiniopDescriptor`，封装元数据、设备信息和工作空间大小
- **关键成员**:
  - `_dtype`: `infiniDtype_t` - 输出/输入张量的数据类型 (F16/F32/F64/BF16)
  - `_info`: `op::elementwise::ElementwiseInfo` - 封装张量形状、步长、连续性、广播等元数据的结构体
  - `_device_info`: `std::unique_ptr<op::elementwise::nvidia::DeviceImpl>` - CUDA 设备实现，负责核函数启动
  - `_workspace_size`: `size_t` - 所需的 GPU 工作空间大小（元数据 + 输入指针数组）
- **核心方法**:
  - `~Descriptor()`: 析构函数，默认实现
  - `workspaceSize()`: 返回 `_workspace_size`，用户需根据此值分配 GPU 内存
  - `create(handle, desc_ptr, out_desc, input_desc_vec)`: 静态工厂方法，校验参数、构建 `ElementwiseInfo`、创建 `DeviceImpl` 并初始化描述符实例
  - `calculate(workspace, workspace_size, output, inputs, stream)`: 执行 tanh 计算，根据 `_dtype` 调用对应的模板化 `DeviceImpl::calculate`，使用 `BLOCK_SIZE=256` 和 `cuda::TanhOp` 算子
- **生命周期**: 由用户调用 `create` 构造，使用 `delete` 销毁；内部采用 RAII 管理资源

## 3. API 接口

```cpp
// 创建 Tanh 操作描述符
infiniStatus_t op::tanh::nvidia::Descriptor::create(
    infiniopHandle_t handle,              // NVIDIA 设备句柄
    Descriptor **desc_ptr,                // [输出] 指向新创建描述符的指针
    infiniopTensorDescriptor_t out_desc,  // 输出张量描述符
    std::vector<infiniopTensorDescriptor_t> input_desc_vec  // 输入张量描述符向量（仅1个）
);
// 返回值: 成功返回 INFINI_STATUS_SUCCESS，失败返回对应错误码（如类型/形状不匹配）
// 副作用: 成功时 *desc_ptr 指向新分配的 Descriptor 实例，需用户负责释放

// 执行 Tanh 计算
infiniStatus_t Descriptor::calculate(
    void *workspace,                      // GPU 工作空间指针，大小 >= workspaceSize()
    size_t workspace_size,                // 工作空间大小（字节）
    void *output,                         // GPU 输出缓冲区指针
    std::vector<const void *> inputs,     // GPU 输入缓冲区指针向量（仅1个）
    void *stream                          // CUDA 流（void* 类型）
) const;
// 返回值: 成功返回 INFINI_STATUS_SUCCESS，失败返回错误码（如工作空间不足、类型错误）
// 核函数配置: 固定使用 256 线程/块，数据类型由 _dtype 决定
```

## 4. 使用示例

```cpp
// 示例: 在 NVIDIA GPU 上执行 Tanh 激活函数
#include "infiniop/ops/tanh/nvidia/tanh_nvidia.cuh"

// 1. 创建设备句柄（假设已初始化）
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_NVIDIA, 0);

// 2. 准备张量描述符（假设输入/输出形状相同: {batch, hidden}）
int64_t shape[] = {32, 768};
int64_t strides[] = {768, 1};
auto *in_desc = new TensorDescriptor(INFINI_DTYPE_F16, 2, shape, strides);
auto *out_desc = new TensorDescriptor(INFINI_DTYPE_F16, 2, shape, strides);

// 3. 创建 Tanh 描述符
op::tanh::nvidia::Descriptor *tanh_desc = nullptr;
auto status = op::tanh::nvidia::Descriptor::create(
    handle, &tanh_desc, out_desc, {in_desc});

// 4. 分配 GPU 内存和工作空间
half *d_input, *d_output;
cudaMalloc(&d_input, in_desc->numel() * sizeof(half));
cudaMalloc(&d_output, out_desc->numel() * sizeof(half));
void *d_workspace;
cudaMalloc(&d_workspace, tanh_desc->workspaceSize());

// 5. 复制数据到 GPU（省略错误检查）
cudaMemcpy(d_input, h_input, in_desc->numel() * sizeof(half), cudaMemcpyHostToDevice);

// 6. 执行 Tanh 计算
cudaStream_t stream;
cudaStreamCreate(&stream);
status = tanh_desc->calculate(d_workspace, tanh_desc->workspaceSize(),
                              d_output, {d_input}, stream);
cudaStreamSynchronize(stream);

// 7. 获取结果并清理
cudaMemcpy(h_output, d_output, out_desc->numel() * sizeof(half), cudaMemcpyDeviceToHost);
delete tanh_desc;
cudaFree(d_workspace); cudaFree(d_input); cudaFree(d_output);
```

## 5. 实现细节

### 内存管理
- **工作空间布局**: 分配连续 GPU 内存，包含两部分:
  1. `ElementwiseInfo` 元数据（形状、步长、连续/广播标志），通过 `getMetaMemSize()` 计算
  2. 输入张量指针数组（`input_size * sizeof(void*)`），支持多输入逐元素操作
- **元数据对齐**: 使用 `std::vector<size_t>` 存储，自动按 `sizeof(size_t)` 对齐，确保 GPU 访问效率
- **RAII 模式**: `Descriptor` 使用 `std::unique_ptr` 管理 `DeviceImpl`，自动析构释放资源

### 并发与执行
- **CUDA 流**: 支持异步执行，用户传入的 `stream` 参数用于核函数启动，允许操作流水线化
- **线程块配置**: 固定使用 `BLOCK_SIZE=256` 线程/块，这是 NVIDIA GPU 的典型配置，平衡占用率和寄存器压力
- **网格大小**: 由 `DeviceImpl::calculate` 根据 `ElementwiseInfo` 自动计算（通常为 `(total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE`）
- **向量化优化**: 通过 `cuda::TanhOp` 算子支持 `half2` 和 `cuda_bfloat162` 向量类型，每个线程处理 2 个 FP16/BF16 元素，提升吞吐量

### 性能优化
- **模板特化**: `calculate` 方法使用编译期类型分发（switch-case 模板实例化），为 F16/F32/F64/BF16 生成专用核函数，避免运行时分支
- **内联函数**: `cuda::TanhOp::operator()` 使用 `__device__ __forceinline__`，确保编译器完全内联，减少函数调用开销
- **数学库函数**: FP32 使用 `tanhf`，FP64 使用 `std::tanh`，均为 CUDA 标准库提供的优化实现
- **连续性检测**: `ElementwiseInfo` 存储 `isOutputContiguous()` 和 `getInputContiguous()` 标志，核函数可据此优化内存访问模式（如向量化加载）

### 错误处理
- **数据类型校验**: `create` 方法使用 `CHECK_DTYPE` 宏，限制支持 F16/F32/F64/BF16，否则返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
- **形状校验**: 使用 `CHECK_SAME_SHAPE` 宏，确保输入/输出形状完全一致，否则返回 `INFINI_STATUS_BAD_TENSOR_SHAPE`
- **工作空间检查**: `calculate` 开头验证 `workspace_size >= _workspace_size`，不满足则返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
- **错误传播**: 所有子操作（如 `ElementwiseInfo::create`, `DeviceImpl::create`）通过 `CHECK_RESULT` 宏传播错误状态

### 依赖关系
- **上级依赖**:
  - `op::elementwise::nvidia::DeviceImpl`: 提供 CUDA 核函数启动逻辑（位于 `elementwise/nvidia/elementwise_nvidia_api.cuh`）
  - `op::elementwise::ElementwiseInfo`: 元数据容器（位于 `elementwise/elementwise.h`）
  - `cuda::TanhOp`: tanh 算子的 CUDA 设备函数定义（位于 `tanh/cuda/kernel.cuh`）
- **框架集成**: 遵循 InfiniOp 的 `InfiniopDescriptor` 接口，可与设备无关的算子调度层集成
- **编译依赖**: 需要 CUDA 工具包（`__half2float`, `__float22half2_rn` 等内联函数）

### 设计模式
- **工厂模式**: `create` 静态方法负责构造描述符，封装复杂的初始化逻辑和错误处理
- **模板方法模式**: `ELEMENTWISE_DESCRIPTOR` 宏定义描述符骨架，子类（如 `tanh`）只需提供类型和命名空间
- **策略模式**: `DeviceImpl` 封装 CUDA 执行策略，`TanhOp` 算子作为可插拔的计算策略
- **CRTP (奇异递归模板模式)**: `ELEMENTWISE_DESCRIPTOR(OP, NAMESPACE)` 宏在 `op::OP::NAMESPACE` 命名空间生成类，结合命名约定实现代码复用

### 核函数调度细节
- **类型分发**:
  ```cpp
  switch (_dtype) {
    case INFINI_DTYPE_F16:
      return _device_info->calculate<256, cuda::TanhOp, half>(...);  // FP16 标量
    case INFINI_DTYPE_BF16:
      return _device_info->calculate<256, cuda::TanhOp, cuda_bfloat16>(...);  // BF16 标量
    case INFINI_DTYPE_F32:
      return _device_info->calculate<256, cuda::TanhOp, float>(...);  // FP32 标量
    case INFINI_DTYPE_F64:
      return _device_info->calculate<256, cuda::TanhOp, double>(...);  // FP64 标量
  }
  ```
  `_device_info->calculate` 是模板函数，编译器为每个类型实例化，核函数内部调用 `cuda::TanhOp::operator()`，后者使用 `if constexpr` 在编译期选择正确的类型转换路径

- **向量化处理**: 当 `Tdata` 为 `half2` 时，`cuda::TanhOp` 执行:
  1. `__half22float2(input)`: 将 FP16 向量转换为 FP32 向量 (float2)
  2. 分别计算 `tanh_f32_func(vf.x)` 和 `tanh_f32_func(vf.y)`
  3. `__float22half2_rn(vr)`: 将 FP32 向量转换回 FP16 向量（舍入到最近偶数）
  BF16 同理（使用 `__bfloat162float`, `__floats2bfloat162_rn`）

### 广播支持
- 虽然当前 `create` 方法检查 `CHECK_SAME_SHAPE`，但框架底层支持广播:
  - `ElementwiseInfo` 存储每个输入的 `input_broadcasted` 标志（第 197 行）
  - 核函数通过步长信息处理广播维度（如将标量广播到张量）
  - 未来可扩展支持不同形状的输入（如 `tanh(scalar + tensor)`）

### 精度与数值稳定性
- **FP16/BF16 处理**: 先提升到 FP32 计算，再转换回原精度，避免下溢和精度损失
- **舍入模式**: FP16 使用 `__float2half_rn`（IEEE 754 舍入到最近偶数），BF16 使用 `__float2bfloat16_rn`
- **边界情况**: `tanhf` 和 `std::tanh` 在输入 ±∞ 时返回 ±1，输入 NaN 时返回 NaN，符合 IEEE 754 标准

### 实现复杂度
- **时间复杂度**: O(n)，n 为输出张量元素数量，每个元素独立计算
- **空间复杂度**: O(1) 额外空间（仅工作空间用于元数据和指针数组，不随输入规模增长）
- **并行度**: 理论上可并行处理所有 n 个元素，实际受 GPU 流多处理器 (SM) 数量和占用率限制

---

**关键要点**: 本模块通过模板元编程和 CUDA 内联函数优化，实现了高性能的 Tanh 激活函数。其设计充分利用了逐元素操作框架的代码复用能力，同时通过向量化指令和多精度支持，覆盖了深度学习推理和训练的常见场景。错误检查严格，接口清晰，易于集成到更大型的计算图中。
