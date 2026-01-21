# NVIDIA CUDA Clip 算子核心实现文档

本模块实现了 Clip（裁剪）算子在 NVIDIA GPU 上的 CUDA 后端，通过元素级操作将输入张量的值限制在指定范围内。该实现基于通用的逐元素计算框架，支持多种浮点数据类型和广播机制。

## 1. 模块结构

- **`clip_nvidia.cuh`**: NVIDIA 后端的 Clip 算子描述符接口定义，通过宏复用通用元素级操作框架
- **`clip_nvidia.cu`**: Clip 算子 NVIDIA CUDA 实现的主文件，包含描述符创建和计算核心逻辑

## 2. 核心类与组件

### `op::clip::nvidia::Descriptor`
- **位置**: `clip_nvidia.cuh` (宏定义), `clip_nvidia.cu` (实现)
- **主要功能**: Clip 算子的 NVIDIA 设备描述符，继承自 `InfiniopDescriptor`，管理算子元数据、设备信息和工作空间需求
- **关键成员**:
  - `_dtype`: `infiniDtype_t` - 输出张量的数据类型 (F16/F32/F64/BF16)
  - `_info`: `op::elementwise::ElementwiseInfo` - 包含所有输入输出张量的形状、步幅、连续性等元数据
  - `_device_info`: `std::unique_ptr<op::elementwise::nvidia::DeviceImpl>` - CUDA 设备实现对象，负责内核启动
  - `_workspace_size`: `size_t` - 设备端工作空间大小（用于存储元数据和输入指针数组）
- **核心方法**:
  - `create(infiniopHandle_t, Descriptor**, infiniopTensorDescriptor_t, std::vector<infiniopTensorDescriptor_t>)`:
    - **功能**: 静态工厂方法，构造并验证 Clip 描述符
    - **参数**: 设备句柄、输出描述符指针、输入描述符向量（3个：输入、最小值、最大值）
    - **验证逻辑**:
      - 数据类型检查：必须是 F16/F32/F64/BF16 之一
      - 形状一致性：输出、输入、min、max 四个张量形状必须完全匹配（不支持不同形状间的广播）
    - **初始化**: 通过 `CREATE_ELEMENTWISE_CUDA_DESCRIPTOR` 宏创建 `ElementwiseInfo` 和 `DeviceImpl`，计算工作空间大小
  - `calculate(void*, size_t, void*, std::vector<const void*>, void*) const`:
    - **功能**: 执行 Clip 计算内核
    - **参数**: 工作空间指针及大小、输出指针、输入指针向量、CUDA 流
    - **实现**: 根据 `_dtype` 分发到对应的模板特化调用，统一使用 256 线程/块的配置
    - **数据类型分发**:
      - `INFINI_DTYPE_F16`: 调用 `_device_info->calculate<256, cuda::ClipOp, half>`
      - `INFINI_DTYPE_F32`: 调用 `_device_info->calculate<256, cuda::ClipOp, float>`
      - `INFINI_DTYPE_F64`: 调用 `_device_info->calculate<256, cuda::ClipOp, double>`
      - `INFINI_DTYPE_BF16`: 调用 `_device_info->calculate<256, cuda::ClipOp, cuda_bfloat16>`
  - **生命周期**: 由 `create` 方法构造，析构函数默认实现，使用智能指针管理设备实现对象

### `op::clip::cuda::ClipOp`
- **位置**: `/home/qy/src/Infini/InfiniCore/src/infiniop/ops/clip/cuda/kernel.cuh`
- **主要功能**: Clip 操作的 CUDA 设备函数对象，在 GPU 内核中执行逐元素裁剪
- **关键成员**:
  - `num_inputs`: `static constexpr size_t = 3` - 输入数量固定为 3（x, min, max）
- **核心方法**:
  - `operator()(const T&, const T&, const T&) const`:
    - **功能**: 实现 clamp 操作，将输入 x 限制在 [min_val, max_val] 范围内
    - **模板特化**:
      - 标量类型 (`half`, `float`, `double`, `cuda_bfloat16`): 使用 `std::clamp(x, min_val, max_val)`
      - 向量化类型 (`half2`, `cuda_bfloat162`): 使用 `__hmax2(__hmin2(x, max_val), min_val)` 进行 SIMD 操作（天析 API 使用标准库回退）
    - **复杂度**: O(1) 常数时间，无分支（使用硬件内在函数）

### `op::elementwise::nvidia::DeviceImpl`
- **位置**: `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/nvidia/elementwise_nvidia.cuh`
- **主要功能**: 元素级操作的 CUDA 设备执行引擎，处理元数据传输和内核启动
- **关键成员**:
  - `_opaque`: `std::shared_ptr<Opaque>` - Pimpl 模式的实现对象，隐藏 CUDA 特定细节
- **核心方法**:
  - `calculate<BLOCK_SIZE, Op, Tdata>(...)`:
    - **功能**: 同类型输入的元素级计算入口
    - **模板参数**:
      - `BLOCK_SIZE`: 块大小（Clip 固定为 256）
      - `Op`: 操作类型（`cuda::ClipOp`）
      - `Tdata`: 数据类型（`half`/`float`/`double`/`cuda_bfloat16`）
    - **实现**: 委托给内部 `_opaque->calculateImpl<256, 3, Op, Tdata>`
  - `calculate<BLOCK_SIZE, Op, Tout, Tin...>(...)`:
    - **功能**: 支持混合输入类型的重载版本（Clip 不使用此版本）
    - **约束**: `sizeof...(Tin) == Op::num_inputs` 的 SFINAE 约束
  - **内核实现细节** (`Opaque::calculateImpl` 和 `launchElementwiseKernel`):
    1. **元数据传输**: 通过 `infoToDevice<3>` 将主机端元数据复制到设备工作空间
       - 传输内容：输入指针数组、形状、步幅、连续性标志、广播标志
       - 异步复制：使用 `cudaMemcpyAsync` 在指定流上执行
    2. **网格配置**:
       - 块大小：`min(256, maxThreadsPerBlock)`
       - 网格大小：`min(CEIL_DIV(output_size, block_size), gridSizeX)`
       - 分步执行：对于大型张量，使用 `step = grid_size * block_size` 分多次启动内核
    3. **内核调用**: 启动 `elementwiseKernel<3, cuda::ClipOp, Tdata>` 内核

### `op::elementwise::ElementwiseInfo`
- **位置**: `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/elementwise.h`
- **主要功能**: 封装元素级操作所需的张量元数据（形状、步幅、布局）
- **数据布局**: 单一 `std::vector<size_t>` 扁平化存储所有元数据
  - 输出形状 (`ndim` 个 `size_t`)
  - 输出步幅 (`ndim` 个 `ptrdiff_t`)
  - 所有输入形状 (`3 * ndim` 个 `size_t`)
  - 所有输入步幅 (`3 * ndim` 个 `ptrdiff_t`)
  - 输入连续性标志 (3 个 `bool`)
  - 输入广播标志 (3 个 `bool`)
- **关键方法**:
  - `create(...)`: 从张量描述符构造元数据，验证形状兼容性
  - `getMetaMemSize()`: 返回元数据总字节数
  - `getOutputSize()`: 返回输出元素总数
  - `getNdim()`: 返回张量维度数
  - `isOutputContiguous()`: 输出是否内存连续
  - 各种访问器：获取形状、步幅、标志的指针

## 3. API 接口

```cpp
// Clip 算子描述符创建（C API 外部调用）
infiniStatus_t infiniopCreateClipDescriptor(
    infiniopHandle_t handle,              // [in] CUDA 设备句柄
    infiniopDescriptor_t *desc_ptr,       // [out] 输出描述符指针
    infiniopTensorDescriptor_t input_desc,  // [in] 输入张量描述符
    infiniopTensorDescriptor_t min_desc,    // [in] 最小值张量描述符
    infiniopTensorDescriptor_t max_desc,    // [in] 最大值张量描述符
    infiniopTensorDescriptor_t output_desc  // [in] 输出张量描述符
);
// 返回: INFINI_STATUS_SUCCESS / INFINI_STATUS_BAD_TENSOR_DTYPE / 其他错误码

// Clip 算子计算执行
infiniStatus_t infiniopClip(
    infiniopDescriptor_t desc,        // [in] Clip 描述符
    void *workspace,                  // [in] 设备工作空间指针
    size_t workspace_size,            // [in] 工作空间大小
    void *output,                     // [out] 输出张量设备指针
    const void *input,                // [in] 输入张量设备指针
    const void *min_val,              // [in] 最小值张量设备指针
    const void *max_val,              // [in] 最大值张量设备指针
    void *stream                      // [in] CUDA 流
);
// 返回: INFINI_STATUS_SUCCESS / INFINI_STATUS_INSUFFICIENT_WORKSPACE / 其他错误码
```

## 4. 使用示例

```cpp
// 示例：在 NVIDIA GPU 上执行 Clip 操作
#include "clip_nvidia.cuh"
#include "../../infiniop.h"

// 1. 创建 CUDA 设备句柄
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_NVIDIA, 0);

// 2. 定义张量描述符（假设形状为 {64, 64}，数据类型为 FP32）
std::vector<int64_t> shape = {64, 64};
std::vector<int64_t> strides = {64, 1};  // 行主序

infiniopTensorDescriptor_t input_desc, min_desc, max_desc, output_desc;
infiniopCreateTensorDescriptor(&input_desc, INFINI_DTYPE_F32, shape.size(), shape.data(), strides.data());
infiniopCreateTensorDescriptor(&min_desc, INFINI_DTYPE_F32, shape.size(), shape.data(), strides.data());
infiniopCreateTensorDescriptor(&max_desc, INFINI_DTYPE_F32, shape.size(), shape.data(), strides.data());
infiniopCreateTensorDescriptor(&output_desc, INFINI_DTYPE_F32, shape.size(), shape.data(), strides.data());

// 3. 创建 Clip 算子描述符
infiniopDescriptor_t clip_desc;
auto status = op::clip::nvidia::Descriptor::create(
    handle,
    reinterpret_cast<op::clip::nvidia::Descriptor**>(&clip_desc),
    output_desc,
    {input_desc, min_desc, max_desc}
);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 4. 分配设备内存
size_t num_elements = 64 * 64;
size_t tensor_bytes = num_elements * sizeof(float);
float *d_input, *d_min, *d_max, *d_output;
cudaMalloc(&d_input, tensor_bytes);
cudaMalloc(&d_min, tensor_bytes);
cudaMalloc(&d_max, tensor_bytes);
cudaMalloc(&d_output, tensor_bytes);

// 初始化 min 和 max（例如：限制在 [-1.0, 1.0]）
float h_min_value = -1.0f, h_max_value = 1.0f;
std::vector<float> h_min(num_elements, h_min_value);
std::vector<float> h_max(num_elements, h_max_value);
cudaMemcpy(d_min, h_min.data(), tensor_bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_max, h_max.data(), tensor_bytes, cudaMemcpyHostToDevice);

// 5. 分配工作空间
size_t workspace_size = reinterpret_cast<op::clip::nvidia::Descriptor*>(clip_desc)->workspaceSize();
void *d_workspace;
cudaMalloc(&d_workspace, workspace_size);

// 6. 创建 CUDA 流
cudaStream_t stream;
cudaStreamCreate(&stream);

// 7. 执行 Clip 计算
std::vector<const void*> inputs = {d_input, d_min, d_max};
status = reinterpret_cast<op::clip::nvidia::Descriptor*>(clip_desc)->calculate(
    d_workspace, workspace_size,
    d_output, inputs, stream
);

// 8. 同步并清理
cudaStreamSynchronize(stream);

// 使用结果...
// cudaMemcpy(h_output, d_output, tensor_bytes, cudaMemcpyDeviceToHost);

// 9. 释放资源
cudaFree(d_input);
cudaFree(d_min);
cudaFree(d_max);
cudaFree(d_output);
cudaFree(d_workspace);
cudaStreamDestroy(stream);
infiniopDestroyDescriptor(clip_desc);
infiniopDestroyTensorDescriptor(input_desc);
infiniopDestroyTensorDescriptor(min_desc);
infiniopDestroyTensorDescriptor(max_desc);
infiniopDestroyTensorDescriptor(output_desc);
infiniopDestroyHandle(handle);
```

## 5. 实现细节

### 5.1 内存管理策略
- **工作空间分配**: 使用单一连续设备内存块存储：
  - 输入指针数组（3 个 `const void*`，24 字节）
  - 元数据（形状、步幅、标志），总计 `meta_size = (8 * ndim + 8 * 3 * ndim + 6) * sizeof(size_t)` 字节
  - 总工作空间大小：`input_size * sizeof(void*) + meta_size`
- **Pimpl 模式**: `DeviceImpl` 使用 `Opaque` 指针隐藏 CUDA 特定实现，避免在头文件中暴露 CUDA API

### 5.2 并发与线程安全
- **CUDA 流支持**: 所有内核启动和内存传输都在用户提供的 CUDA 流上执行，支持同一流内的顺序性和跨流并发
- **异步执行**: 使用 `cudaMemcpyAsync` 进行异步主机到设备内存传输
- **无主机端同步**: 计算函数立即返回，用户需手动同步流以检查完成

### 5.3 性能优化
- **向量化指令**: 对 `half2` 和 `cuda_bfloat162` 类型使用 SIMD 内在函数 `__hmax2`/`__hmin2`，每个指令处理 2 个元素
- **块大小选择**: 固定使用 256 线程/块，平衡寄存器使用和占用率
- **网格分区**: 对大型张量（`output_size > grid_size * block_size`），分多次启动内核，每次处理 `step` 个元素
- **连续内存优化**: 检测张量连续性，连续张量使用线性索引，非连续张量调用 `indexToOffset` 计算偏移
- **广播支持**: 通过 `InputIndexer` 结构处理输入张量的广播，自动将输出索引映射到正确的输入索引

### 5.4 错误处理
- **描述符创建阶段**:
  - `INFINI_STATUS_BAD_TENSOR_DTYPE`: 不支持的数据类型
  - `INFINI_STATUS_INVALID_ARGUMENT`: 形状不匹配或空描述符
- **计算阶段**:
  - `INFINI_STATUS_INSUFFICIENT_WORKSPACE`: 工作空间大小不足
  - `INFINI_STATUS_BAD_TENSOR_DTYPE`: 运行时数据类型分发失败（不应该发生）
- **CUDA 错误传播**: 使用 `CHECK_CUDA` 宏检查 CUDA API 调用，错误状态向上传播

### 5.5 依赖关系
- **内部依赖**:
  - `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/nvidia/elementwise_nvidia.cuh`: 通用元素级操作 CUDA 框架
  - `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/elementwise.h`: 元数据结构 `ElementwiseInfo`
  - `/home/qy/src/Infini/InfiniCore/src/infiniop/ops/clip/cuda/kernel.cuh`: CUDA 设备函数 `ClipOp`
  - `/home/qy/src/Infini/InfiniCore/src/infiniop/devices/nvidia/nvidia_common.cuh`: 通用 CUDA 工具（如 `indexToOffset`）
- **外部依赖**: CUDA Toolkit, C++ 标准库 (`std::clamp`, `std::vector`, `std::shared_ptr`)

### 5.6 设计模式
- **宏代码生成**: `ELEMENTWISE_DESCRIPTOR(clip, nvidia)` 和 `CREATE_ELEMENTWISE_CUDA_DESCRIPTOR` 宏避免重复代码
- **模板元编程**: 使用编译期索引序列 `std::index_sequence` 展开变参模板，生成无循环的内联代码
- **策略模式**: `ClipOp` 作为策略对象，可在运行时替换为其他逐元素操作
- **RAII**: 使用 `std::unique_ptr` 和 `std::shared_ptr` 自动管理资源生命周期
- **类型擦除**: `void*` 类型的输入指针和流，在内部转换为强类型

### 5.7 广播语义
- **严格形状匹配**: Clip 操作要求所有 4 个张量（输入、min、max、输出）形状完全相同，不支持不同形状间的广播
- **连续性优化**: 即使形状相同，如果步幅不同（如转置张量），也会通过 `InputIndexer` 正确处理

### 5.8 数据类型支持
- **半精度浮点**: `half` (FP16) 和 `cuda_bfloat16` (BF16)
- **单精度浮点**: `float` (FP32)
- **双精度浮点**: `double` (FP64)
- **向量化**: 自动使用 `half2` 和 `cuda_bfloat162` 进行 SIMD 优化（取决于 CUDA 内核配置）

### 5.9 设备兼容性
- **NVIDIA GPU**: 所有支持 CUDA 的 GPU
- **天析 GPU**: 通过 `ENABLE_ILUVATAR_API` 宏启用标准库回退实现（不使用 `__hmax2`/`__hmin2`）
- **计算能力**: 依赖 CUDA 内在函数，需要足够新的架构（SM >= 5.0 以支持 half/half2）
