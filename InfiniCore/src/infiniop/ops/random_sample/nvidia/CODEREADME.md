# Random Sample NVIDIA CUDA 实现文档

本模块实现了基于 NVIDIA GPU 的随机采样算子，支持 Top-K 和 Top-P（nucleus）采样策略，广泛应用于大语言模型（LLM）的文本生成场景。该实现利用 CUB 库的高性能原语进行归约、排序和前缀和操作，实现了完整的 softmax 归一化和基于概率分布的采样过程。

## 1. 模块结构

- **`random_sample_nvidia.cuh`**: 头文件，包含 NVIDIA 后端的描述符声明，通过 DESCRIPTOR 宏注册算子实现
- **`random_sample_nvidia.cu`**: 主实现文件，包含描述符类的创建、workspace 计算和计算调度逻辑
- **`random_sample_kernel.cuh`**: CUDA kernel 实现，包含所有 GPU 核函数、CUB 库封装以及采样算法的具体实现

## 2. 核心类与结构

### `Descriptor::Opaque`
- **位置**: `random_sample_nvidia.cu`
- **主要功能**: 封装 NVIDIA 设备句柄的内部状态，持有 `device::nvidia::Handle::Internal` 的共享指针
- **关键成员**:
  - `internal`: `std::shared_ptr<device::nvidia::Handle::Internal>` - 持有 NVIDIA 设备句柄的内部实现，用于获取 CUDA 流配置和设备参数

### `Descriptor`
- **位置**: `random_sample_nvidia.cu`
- **主要功能**: 随机采样算子的 NVIDIA 后端描述符，管理算子的生命周期、workspace 分发和计算执行
- **关键成员**:
  - `_opaque`: `Opaque*` - 持有设备句柄内部状态的指针
  - `_info`: `RandomSampleInfo` - 从父类继承的张量描述信息（数据类型、维度大小等）
  - `_min_workspace_size`: `size_t` - 计算所需的最小 workspace 大小
- **核心方法**:
  - `create(handle_, desc_ptr, result_desc, probs_desc)`: 静态工厂方法，根据输入/输出张量的数据类型（整数类型和浮点类型组合）实例化描述符。通过双重 switch-case 宏展开，支持 8 种整数类型（int8_t/int16_t/int32_t/int64_t/uint8_t/uint16_t/uint32_t/uint64_t）与 4 种浮点类型（half/__nv_bfloat16/float/double）的所有组合。调用 `calculateWorkspace<Tidx, Tval>(info.n)` 计算 workspace 大小，返回 `INFINI_STATUS_SUCCESS` 或相应错误码
  - `minWorkspaceSize()`: 返回 `_min_workspace_size`，表示执行计算所需的最小临时存储空间
  - `calculate(workspace, workspace_size, result, probs, random_val, topp, topk, temperature, stream)`: 执行随机采样计算。首先验证 workspace 大小是否充足，然后从设备句柄获取 CUDA block 配置（`blockSizeX()`），最后调用 `Calculate::calculate<Algo>()` 触发 GPU kernel 执行。返回 `INFINI_STATUS_SUCCESS` 或 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
- **生命周期**: 由 `create()` 静态方法构造，析构函数释放 `_opaque` 指针。采用 RAII 模式管理资源

### `Algo`
- **位置**: `random_sample_kernel.cuh`
- **主要功能**: 采样算法的策略类，封装 ArgMax（贪心采样）和 Random（随机采样）两种模式的 GPU 计算流程
- **关键成员**:
  - `block_size`: `int` - CUDA kernel 执行的 block 大小，从设备句柄的 `blockSizeX()` 配置获取
- **核心方法**:
  - `argmax(workspace, workspace_size, result, probs, n, stream)`: 模板方法（`<Tidx, Tval_>`），执行贪心采样。使用 CUB 的 `DeviceReduce::ArgMax` 找到概率最大的元素索引。调用 `argMax_<Tval>()` 计算归约操作，并通过 `castIdx<<<1, 1>>>` kernel 将 `cub::KeyValuePair<int, Tval>` 的 key 字段提取到结果。返回 `INFINI_STATUS_SUCCESS`
  - `random(workspace, workspace_size, result, probs, n, random_val, topp, topk, temperature, stream)`: 模板方法（`<Tidx, Tval_>`），执行完整的随机采样流程。通过指针算术将 workspace 划分为三个对齐缓冲区：indices（索引数组）、sorted（排序后的概率值）、indices_out（排序后的索引数组）。执行 5 个阶段的 kernel 调用：
    1. 调用 `fillIndices<<<grid, block>>>` 填充初始索引序列 `[0, 1, 2, ..., n-1]`
    2. 调用 `radixSort()` 使用 CUB 的 `DeviceRadixSort::SortPairsDescending` 按概率值降序排列，同时排列索引
    3. 调用 `partialSoftmaxKernel<<<grid, block>>>` 对排序后的概率执行温度缩放的 softmax 归一化（`exp((x - max) / temperature)`）
    4. 调用 `setSoftmaxMaxKernel<<<1, 1>>>` 将最大值位置设为 1（即 `exp(0)`）
    5. 调用 `inclusiveSum()` 使用 CUB 的 `DeviceScan::InclusiveSum` 计算累积概率分布
    6. 调用 `randomSampleKernel<<<1, 1>>>` 执行线性扫描采样，根据 `random_val`（0~1 之间的均匀分布随机数）在累积分布中找到第一个超过阈值的索引
     返回 `INFINI_STATUS_SUCCESS`

## 3. 工具函数与 Kernel

### CUB 库封装函数
- **`argMax_<T>(kv_pair, logits, n, workspace_ptr, workspace_len, stream)`**: 模板函数，封装 `cub::DeviceReduce::ArgMax`，返回 logits 数组中最大值的键值对（索引和值）
- **`radixSort<Tval, Tidx>(workspace_ptr, workspace_len, key_in, key_out, val_in, val_out, n, stream)`**: 模板函数，封装 `cub::DeviceRadixSort::SortPairsDescending`，对键值对进行降序排序，排序范围覆盖整个 `Tval` 类型的所有比特位（`0` 到 `sizeof(Tval) * 8`）
- **`inclusiveSum<T>(workspace_ptr, workspace_len, data, n, stream)`**: 模板函数，封装 `cub::DeviceScan::InclusiveSum`，对数组进行原地前缀和计算

### 辅助函数
- **`align256(size)`**: `static constexpr` 函数，将大小对齐到 256 字节边界，通过 `(size + 255) & (~255)` 位运算实现，用于满足 CUDA 设备内存的合并访问要求

### Workspace 计算函数
- **`calculateWorkspace<Tidx, Tval>(n_)`**: 模板函数（返回 `utils::Result<size_t>`），计算执行采样所需的最小临时存储空间。首先调用 CUB API（传入 nullptr）查询 `argMax`、`radixSort` 和 `inclusiveSum` 各自需要的 workspace 大小。然后将数组存储空间（`sizeof(Tidx) * n` 和 `sizeof(Tval) * n`）按 256 字节对齐累加。最后通过 `cuda::maximum()` 或 `cub::Max()`（根据 CUDA 版本分支）取各阶段所需 workspace 的最大值，确保可复用同一块内存

### 类型特化
- **`CudaTval<Tval>`**: 模板结构体，将抽象的浮点类型映射到 CUDA 原生类型。特化版本包括：`fp16_t` → `half`，`bf16_t` → `__nv_bfloat16`，其他类型保持不变

### CUDA Kernel 函数
- **`castIdx<Tidx, Tval><<<1, 1>>>(result, kv_pair)`**: 单线程 kernel，从 `cub::KeyValuePair<int, Tval>` 中提取索引字段并写入结果
- **`fillIndices<Tidx><<<grid, block>>>(indices, n)`**: 并行填充 kernel，生成 `[0, 1, 2, ..., n-1]` 序列，每个线程处理一个元素，通过 `i < n` 边界检查
- **`partialSoftmaxKernel<T><<<grid, block>>>(data, n, temperature)`**: 并行 softmax kernel，对已排序数组（`data[0]` 为最大值）执行温度缩放。使用 `__ldg(data)` 通过只读缓存加载最大值以减少延迟。对于 `i > 0` 的元素，计算 `exp((data[i] - max) / temperature)`，索引 0 的元素保留待后续处理
- **`setSoftmaxMaxKernel<T><<<1, 1>>>(data)`**: 单线程 kernel，将 `data[0]` 设为 1（对应 `exp(0)`），完成 softmax 归一化
- **`randomSampleKernel<Tval, Tidx><<<1, 1>>>(result, sorted, indices_out, n, random, topp, topk)`**: 采样 kernel，实现线性扫描采样算法。计算采样阈值 `p = random * min(topp * sorted[n-1], sorted[topk-1])`，其中 `sorted[n-1]` 是累积概率总和，`topp * sorted[n-1]` 限制了 Top-P（nucleus）采样的质量阈值，`sorted[topk-1]` 限制了 Top-K 采样的范围。通过 `for` 循环遍历累积概率数组，找到第一个满足 `sorted[i] >= p` 的索引，通过 `indices_out[i]` 映射回原始位置并写入结果

## 4. API 接口

```cpp
// 算子描述符创建 API
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,            // [输入] Infini 运行时句柄
    Descriptor **desc_ptr,               // [输出] 创建的描述符指针
    infiniopTensorDescriptor_t result_desc,  // [输入] 输出张量描述（整数类型）
    infiniopTensorDescriptor_t probs_desc    // [输入] 输入概率张量描述（浮点类型）
);
// 根据张量数据类型组合实例化描述符并计算 workspace 大小

// 获取 workspace 大小 API
size_t Descriptor::minWorkspaceSize() const;
// 返回执行计算所需的最小临时存储空间字节数

// 执行采样计算 API
infiniStatus_t Descriptor::calculate(
    void *workspace,      // [输入] 临时存储空间
    size_t workspace_size, // [输入] workspace 大小
    void *result,         // [输出] 采样结果（单个整数索引）
    const void *probs,    // [输入] 概率分布数组（未归一化的 logits）
    float random_val,     // [输入] 0~1 之间的均匀分布随机数
    float topp,           // [输入] Top-P 采样阈值（0~1）
    int topk,             // [输入] Top-K 采样范围
    float temperature,    // [输入] 温度参数（控制分布平滑度）
    void *stream          // [输入] CUDA 流
);
// 执行完整的采样流程，返回成功状态或错误码
```

## 5. 使用示例

```cpp
// 示例：在 LLM 推理中使用 Random Sample 算子进行 Top-K/Top-P 采样

// 1. 准备输入张量描述符
infiniopTensorDescriptor_t logits_desc;
infiniopTensorDescriptor_t token_desc;
// 假设 vocab_size = 50000，数据类型为 float（logits）和 int32_t（token）
createTensorDesc(&logits_desc, INFINI_DTYPE_F32, {50000});
createTensorDesc(&token_desc, INFINI_DTYPE_I32, {});

// 2. 创建算子描述符
op::random_sample::nvidia::Descriptor *sample_desc;
infiniStatus_t status = op::random_sample::nvidia::Descriptor::create(
    handle, &sample_desc, token_desc, logits_desc);
if (status != INFINI_STATUS_SUCCESS) {
    // 错误处理
}

// 3. 分配 workspace 和 GPU 内存
size_t workspace_size = sample_desc->minWorkspaceSize();
void *workspace = nullptr;
cudaMalloc(&workspace, workspace_size);

float *d_logits;  // GPU 上的 logits 数组
int32_t *d_token; // GPU 上的采样结果
cudaMalloc(&d_logits, 50000 * sizeof(float));
cudaMalloc(&d_token, sizeof(int32_t));

// 将 logits 从模型输出拷贝到 GPU
cudaMemcpy(d_logits, model_output, 50000 * sizeof(float), cudaMemcpyHostToDevice);

// 4. 生成随机数（在 CPU 或 GPU 上均可）
float random_val = generate_uniform_random();  // 0~1 之间的均匀分布

// 5. 执行采样
cudaStream_t stream;
cudaStreamCreate(&stream);

status = sample_desc->calculate(
    workspace, workspace_size,  // 临时存储
    d_token, d_logits,           // 输出/输入
    random_val,                  // 随机数
    0.9f,                        // Top-P 阈值（保留累积概率 90% 的 token）
    50,                          // Top-K 范围（只从前 50 个概率中采样）
    0.8f,                        // 温度参数（较低使分布更尖锐）
    stream                       // CUDA 流
);

if (status != INFINI_STATUS_SUCCESS) {
    // 错误处理：可能是 workspace 不足或其他 CUDA 错误
}

// 6. 获取采样结果
int32_t sampled_token;
cudaMemcpy(&sampled_token, d_token, sizeof(int32_t), cudaMemcpyDeviceToHost);
printf("Sampled token ID: %d\n", sampled_token);

// 7. 清理资源
cudaFree(workspace);
cudaFree(d_logits);
cudaFree(d_token);
cudaStreamDestroy(stream);
delete sample_desc;
```

## 6. 实现细节

### 内存管理
- **Workspace 划分策略**: 将临时存储划分为三个对齐缓冲区（indices、sorted、indices_out），每个缓冲区按 256 字节对齐以满足 CUDA 合并访问要求。通过指针算术实现零拷贝的内存复用
- **类型映射**: 通过 `CudaTval<Tval>` 特化将抽象浮点类型（`fp16_t`、`bf16_t`）映射到 CUDA 原生类型（`half`、`__nv_bfloat16`），确保与 CUB 库的兼容性
- **内存对齐**: 使用 `align256()` 函数强制所有数组按 256 字节边界对齐，优化 GPU 内存控制器的访问效率

### 并发与性能
- **CUDA 流支持**: 所有 kernel 调用和 CUB 操作都接受 `cudaStream_t` 参数，支持与同一流中的其他操作并发执行，实现计算与数据传输的重叠
- **Kernel 启动配置**: 动态计算 grid 和 block 大小（`block = min(block_size, n)`，`grid = (n + block - 1) / block`），根据数据规模自适应调整并行度
- **CUB 异步执行**: CUB 的 DeviceReduce/DeviceRadixSort/DeviceScan API 均为异步操作，允许 host 端继续执行后续任务

### 算法优化
- **排序后简化 softmax**: 利用降序排序后的特性（最大值位于索引 0），将标准 softmax 的两次遍历（找最大值 + 计算指数和）优化为一次并行 kernel 调用，并通过 `setSoftmaxMaxKernel` 补齐第一个元素
- **Top-K/Top-P 联合裁剪**: 在 `randomSampleKernel` 中，通过 `min(topp * sorted[n-1], sorted[topk-1])` 同时应用 Top-K 和 Top-P 限制，避免遍历整个累积概率数组
- **CUB 库利用**: 充分利用 NVIDIA CUB 库的高性能原语：
  - `DeviceReduce::ArgMax`：O(log n) 并行归约
  - `DeviceRadixSort::SortPairsDescending`：O(n) 基数排序（相比比较排序更快）
  - `DeviceScan::InclusiveSum`：O(n) 前缀和（采用 Blelloch 或 Hillis-Steele 算法）

### 错误处理
- **Workspace 验证**: 在 `calculate()` 方法中检查 `workspace_size < _min_workspace_size`，返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE` 错误码，避免 CUB 操作因内存不足崩溃
- **CUDA 错误传播**: 使用 `CHECK_CUDA` 宏检查所有 CUB API 的返回值（`cudaError_t`），在失败时提前终止并返回错误状态
- **类型安全**: 通过模板特化和编译时 `switch-case` 展开，确保所有数据类型组合在编译期被验证，避免运行时类型错误

### 依赖关系
- **外部依赖**:
  - `cub/device/device_radix_sort.cuh`：CUB 基数排序 API
  - `cub/device/device_reduce.cuh`：CUB 归约操作 API（ArgMax）
  - `cub/device/device_scan.cuh`：CUB 前缀和扫描 API
  - `../../../devices/nvidia/nvidia_handle.cuh`：NVIDIA 设备句柄和流管理
  - `../../../devices/nvidia/nvidia_kernel_common.cuh`：CUDA kernel 通用工具和宏定义
  - `"infinicore.h"`：Infini 核心数据类型定义（`fp16_t`、`bf16_t` 等）
- **内部依赖**:
  - `../random_sample.h`：算子的通用定义和接口声明
  - `../info.h`：`RandomSampleInfo` 结构体，封装张量的数据类型和维度信息

### 设计模式
- **策略模式（Strategy Pattern）**: `Algo` 结构体封装两种采样策略（ArgMax 和 Random），通过模板方法统一接口，支持运行时选择采样算法
- **工厂模式（Factory Pattern）**: `Descriptor::create()` 静态方法作为工厂，根据数据类型组合实例化具体的描述符对象
- **RAII（Resource Acquisition Is Initialization）**: `Descriptor` 的析构函数自动释放 `_opaque` 指针，确保资源生命周期与对象绑定
- **模板特化（Template Specialization）**: 通过 `CudaTval<Tval>` 的部分特化实现类型映射，避免在每个 kernel 中重复编写类型转换代码
