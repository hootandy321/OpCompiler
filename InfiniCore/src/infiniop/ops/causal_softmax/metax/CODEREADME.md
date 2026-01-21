# Causal Softmax MetAX 后端核心实现文档

本模块实现了 Moore Threads MooreThreads AIGPU (MetAX) 硬件平台的因果注意力掩码 Softmax 操作。这是 Transformer 模型中自注意力机制的核心计算组件，专门针对 MetAX GPU 架构进行了优化，支持 FP16、BF16 和 FP32 三种数据类型。

## 1. 模块结构

- **`causal_softmax_metax.h`**: MetAX 后端的描述符声明文件，通过宏定义生成 `op::causal_softmax::metax::Descriptor` 类
- **`causal_softmax_metax.maca`**: MetAX 后端的核心实现文件，包含 GPU 内核封装、描述符实现、内核启动逻辑

## 2. 核心类

### `op::causal_softmax::metax::Descriptor`
- **位置**: `causal_softmax_metax.h` (宏定义) + `causal_softmax_metax.maca` (实现)
- **主要功能**: MetAX 平台的因果 Softmax 操作符描述符，负责内核验证、工作空间管理、GPU 内核启动
- **继承关系**: 继承自 `InfiniopDescriptor`，实现跨设备统一接口
- **关键成员**:
  - `_opaque`: 指向 `Opaque` 结构体的指针，封装 MetAX 设备内部状态
  - `_info`: `CausalSoftmaxInfo` 对象，存储张量形状、步长、数据类型等元数据
  - `_workspace_size`: 工作空间大小（本实现为 0，无需额外工作空间）

#### `Descriptor::Opaque`
- **位置**: `causal_softmax_metax.maca:26-28`
- **主要功能**: 封装 MetAX 设备句柄的内部状态，提供设备架构查询能力
- **关键成员**:
  - `internal`: `std::shared_ptr<device::metax::Handle::Internal>`，MetAX 设备内部句柄，用于查询设备参数（如最大线程块大小）

### `causalSoftmax<Tdata, Tcompute>` (GPU 内核封装)
- **位置**: `causal_softmax_metax.maca:15-22`
- **主要功能**: MetAX GPU 内核的薄封装层，将统一的内核签名转发到 CUDA 友好的通用内核实现
- **模板参数**:
  - `BLOCK_SIZE`: 线程块大小（512 或 1024），根据设备架构动态选择
  - `Tdata`: 数据类型（`half`、`__hpcc_bfloat16`、`float`）
  - `Tcompute`: 计算类型（通常为 `float`，用于提高数值精度）
- **核心方法**:
  - 直接调用 `causalSoftmaxKernel<BLOCK_SIZE, Tdata, Tcompute>()`，该内核在 `../cuda/kernel.cuh` 中实现
  - 支持批处理（`batch`）、高度（`height`，即序列长度）、宽度（`width`，即总序列长度）三维参数
  - 接收输入输出张量的步长参数，支持非连续内存布局

### `launchKernel<BLOCK_SIZE>` (内核启动函数)
- **位置**: `causal_softmax_metax.maca:47-76`
- **主要功能**: 根据数据类型分发模板特化，配置并启动 GPU 内核
- **模板参数**:
  - `BLOCK_SIZE`: 从描述符的 `_opaque->internal->maxThreadsPerBlock()` 查询，决定使用 512 或 1024 线程块
- **内核配置**:
  - Grid 维度: `dim3(seq_len, batch_size, 1)`，即 X 维度为序列长度，Y 维度为批大小
  - Block 维度: `BLOCK_SIZE`（512 或 1024），一维线程块
- **数据类型支持**:
  - `INFINI_DTYPE_F16`: 使用 `half` 类型，调用 `hexp()` 进行指数运算
  - `INFINI_DTYPE_BF16`: 使用 `__hpcc_bfloat16` 类型（MetAX 扩展类型），调用 `hexp()`
  - `INFINI_DTYPE_F32`: 使用 `float` 类型，调用标准 `exp()`
- **返回值**: `infiniStatus_t`，成功返回 `INFINI_STATUS_SUCCESS`，不支持的数据类型返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`

## 3. API 接口

```cpp
// 描述符创建接口
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,                  // MetAX 设备句柄
    Descriptor **desc_ptr,                    // 输出：创建的描述符指针
    infiniopTensorDescriptor_t y_desc,        // 输出张量描述符
    infiniopTensorDescriptor_t x_desc         // 输入张量描述符
);
// 验证张量形状和数据类型，提取元信息，初始化 Opaque 结构

// 计算接口
infiniStatus_t Descriptor::calculate(
    void *workspace,          // 工作空间指针（本实现未使用，传 nullptr）
    size_t workspace_size,    // 工作空间大小（必须为 0）
    void *y,                  // 输出张量指针
    const void *x,            // 输入张量指针
    void *stream_             // MetAX 流（hcStream_t 类型）
) const;
// 启动 GPU 内核，执行因果 Softmax 计算

// 工作空间查询
size_t Descriptor::workspaceSize() const;
// 返回 0，本实现无需额外工作空间
```

## 4. 使用示例

```cpp
// 示例：在 MetAX GPU 上执行因果 Softmax（用于 Transformer 自注意力）

// 1. 准备张量描述符（假设 batch_size=8, seq_len=128, total_seq_len=256）
std::vector<int64_t> shape = {8, 128, 256};
std::vector<int64_t> stride = {128 * 256, 256, 1};
infiniopTensorDescriptor_t x_desc, y_desc;
infiniopCreateTensorDescriptor(&x_desc, INFINI_DTYPE_F16, shape.data(), stride.data(), 3);
infiniopCreateTensorDescriptor(&y_desc, INFINI_DTYPE_F16, shape.data(), stride.data(), 3);

// 2. 创建 MetAX 设备句柄
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_METAX, device_id);

// 3. 创建操作描述符
op::causal_softmax::metax::Descriptor *softmax_desc;
infiniStatus_t status = op::causal_softmax::metax::Descriptor::create(
    handle, &softmax_desc, y_desc, x_desc);

// 4. 分配 GPU 内存（假设数据已在 GPU 上）
half *d_x, *d_y;
hcMalloc((void **)&d_x, 8 * 128 * 256 * sizeof(half));
hcMalloc((void **)&d_y, 8 * 128 * 256 * sizeof(half));

// 5. 创建 MetAX 流
hcStream_t stream;
hcStreamCreate(&stream);

// 6. 执行计算（无需工作空间）
status = softmax_desc->calculate(nullptr, 0, d_y, d_x, stream);
// 内部自动选择 BLOCK_SIZE（512 或 1024），根据设备架构启动对应内核

// 7. 同步并清理
hcStreamSynchronize(stream);
hcStreamDestroy(stream);
hcFree(d_x);
hcFree(d_y);
delete softmax_desc;
infiniopDestroyHandle(handle);
```

## 5. 实现细节

### 内存管理
- **零拷贝设计**: 输入输出张量直接在 GPU 全局内存中操作，无需主机端缓冲
- **工作空间策略**: 本实现无需额外工作空间（`workspace_size = 0`），所有临时数据使用 GPU 共享内存（`__shared__`）
- **共享内存使用**: 每个线程块使用共享内存存储行最大值（`max_`）和行和（`sum_`），大小为 `BLOCK_SIZE * sizeof(Tdata)`

### 并发与线程配置
- **线程块大小策略**: 根据 MetAX 设备架构动态选择
  - `_opaque->internal->maxThreadsPerBlock() == METAX_BLOCK_SIZE_1024`: 使用 1024 线程块
  - `_opaque->internal->maxThreadsPerBlock() == METAX_BLOCK_SIZE_512`: 使用 512 线程块
  - 其他情况返回 `INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED`
- **Grid-Block 映射**:
  - Grid X 维度: 对应序列中的每个位置（`seq_len`）
  - Grid Y 维度: 对应批处理中的每个样本（`batch_size`）
  - 每个 Block 处理一行（一个序列位置的整个宽度）
- **线程级并行**: 每个线程通过步长（`BLOCK_SIZE`）处理行中的多个元素，实现循环展开

### 性能优化
- **减少操作优化**: 使用 CUB 库的 `BlockReduce` 原语进行高效的块内归约（求最大值、求和），复杂度 O(log BLOCK_SIZE)
- **类型提升策略**: `Tcompute` 模板参数允许使用 `float` 作为计算类型，即使 `Tdata` 为 `half`/`bfloat16`，避免 FP16/BF16 累积误差
- **因果掩码融合**: 掩码操作直接融合在 Softmax 计算中，避免单独的掩码内核
  - 掩码条件: `width + blockIdx.x >= col + height`（即下三角区域）
  - 掩码外元素直接置零，无需额外内存访问
- **编译器特性**: 使用 `__forceinline__` 强制内联减少函数调用开销

### 错误处理
- **数据类型验证**: 仅支持 `INFINI_DTYPE_F16`、`INFINI_DTYPE_BF16`、`INFINI_DTYPE_F32`，其他类型返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
- **设备架构检查**: 不支持的 MetAX 架构（最大线程数非 512/1024）返回 `INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED`
- **宏检查**: 使用 `CHECK_RESULT`、`CHECK_STATUS` 宏进行错误传播（定义在 `../../../utils.h`）

### 依赖项
- **核心依赖**:
  - `../causal_softmax.h`: 父操作符定义，通过 `DESCRIPTOR(metax)` 宏生成描述符类
  - `../cuda/kernel.cuh`: 通用 CUDA 内核实现（跨平台共享），包含 `causalSoftmaxKernel` 模板函数
  - `../../../devices/metax/metax_common.h`: MetAX 设备公共头文件，定义设备句柄、编译标志
  - `../../../devices/metax/metax_kernel_common.h`: MetAX 内核常量（`METAX_BLOCK_SIZE_1024`、`METAX_BLOCK_SIZE_512`）
  - `../../../reduce/cuda/reduce.cuh`: 设备无关的归约原语（`sum`、`max`）
- **第三方库**:
  - CUB 库: `cub/block/block_reduce.cuh`，用于块级归约
  - 编译时根据 `ENABLE_METAX_MC_API` 宏选择:
    - 启用: 使用 `<cub/block/block_reduce.cuh>`（MooreThreads MC API）
    - 禁用: 使用 `<hccub/block/block_reduce.cuh>`（Hygon C CUDA 兼容层）

### 设计模式
- **策略模式 (Strategy Pattern)**: `launchKernel` 根据数据类型选择不同的模板特化，实现类型分发逻辑
- **模板方法模式 (Template Method Pattern)**: `causalSoftmax` 内核封装调用通用 `causalSoftmaxKernel`，实现平台无关与平台相关的分离
- **桥接模式 (Bridge Pattern)**: `Descriptor` 通过 `Opaque` 结构体桥接到 MetAX 设备内部状态，隐藏实现细节
- **工厂模式 (Factory Pattern)**: `Descriptor::create` 作为工厂方法，统一创建描述符实例

### 数值稳定性
- **最大值归一化**: 内核先计算行最大值（`max_`），然后用 `exp(x - max_)` 避免指数溢出
- **因果掩码语义**: 掩码区域（上三角）在指数化前置零，确保 Softmax 概率仅来自有效位置
- **类型分离**: `Tdata`（存储类型）与 `Tcompute`（计算类型）分离，FP16/BF16 输入使用 FP32 累加，保证精度
