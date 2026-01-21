# Metax Random Sample 操作符核心实现文档

本模块实现了在 Moore Threads Metax GPU 设备上的随机采样操作（random sampling），支持 top-p 和 top-k 采样策略，广泛应用于大语言模型的文本生成场景。该实现利用 CUB 库的高性能原语（排序、归约、扫描）完成 GPU 端的并行采样，避免 CPU-GPU 数据传输开销。

## 1. 模块结构

- **`random_sample_metax.h`**: 前向声明头文件，通过宏 `DESCRIPTOR(metax)` 定义完整的 Descriptor 类结构
- **`random_sample_kernel.h`**: 核心计算内核实现，包含 CUB API 封装、workspace 计算算法、采样 kernels 以及算法策略类 `Algo`
- **`random_sample_metax.maca`**: Metax 后端实现文件，包含 Descriptor 类的方法实现（创建、workspace 计算、采样调度）

## 2. 核心类与结构

### `RandomSampleInfo`
- **位置**: `/home/qy/src/Infini/InfiniCore/src/infiniop/ops/random_sample/info.h`
- **主要功能**: 验证和提取随机采样操作的张量元数据
- **关键成员**:
  - `dt_i`: 输出索引的数据类型（必须为整数类型：int8/16/32/64, uint8/16/32/64）
  - `dt_p`: 输入概率/对数值的数据类型（支持 fp16, bf16, fp32, fp64）
  - `n`: 概率分布的长度（1D 张量的维度）
- **核心方法**:
  - `create(result_desc, probs_desc)`: 静态工厂方法，验证张量形状（输出为标量，输入为 1D 且连续），返回 `Result<RandomSampleInfo>`
- **设计模式**: 值对象模式，不可变数据结构

### `op::random_sample::Descriptor`
- **位置**: 通过 `DESCRIPTOR(metax)` 宏在 `random_sample_metax.h` 中定义
- **主要功能**: Metax 后端的随机采样操作符描述符，管理设备句柄、workspace 大小和类型信息
- **关键成员**:
  - `_opaque`: 指向 `Opaque` 结构的指针（Pimpl 惯用法，封装 Metax 设备句柄）
  - `_info`: `RandomSampleInfo` 实例，存储张量类型信息
  - `_min_workspace_size`: 运行时所需的最小 workspace 字节数
- **核心方法**:
  - `create(handle, desc_ptr, result_desc, probs_desc)`: 静态构造函数，根据张量类型特化计算 workspace 大小，创建描述符实例。内部使用嵌套 switch（外层索引类型，内层概率类型）遍历所有支持的类型组合
  - `minWorkspaceSize()`: 返回预计算的 workspace 大小
  - `calculate(workspace, workspace_size, result, probs, random_val, topp, topk, temperature, stream)`: 执行采样操作，内部调用 `Calculate::calculate<Algo>` 分发到具体算法
- **生命周期**: 由用户通过 `create()` 构造，析构函数释放 `_opaque` 资源

### `op::random_sample::metax::Descriptor::Opaque`
- **位置**: `random_sample_metax.maca` 第 10-12 行
- **主要功能**: 封装 Metax 设备的内部句柄，隐藏实现细节
- **关键成员**:
  - `internal`: `std::shared_ptr<device::metax::Handle::Internal>`，共享指针管理设备句柄生命周期
- **设计模式**: Pimpl (Pointer to Implementation) 惯用法，隔离 Metax 特定类型

### `op::random_sample::metax::Algo`
- **位置**: `random_sample_kernel.h` 第 183-264 行
- **主要功能**: 封装随机采样的算法策略，提供 argmax 和随机采样两种执行路径
- **关键成员**:
  - `block_size`: CUDA/Metax block 大小，从设备句柄查询获得
- **核心方法**:
  - `argmax<Tidx, Tval_>(workspace, workspace_size, result, probs, n, stream)`: 快速路径，当采样参数退化为确定性行为时（如 topk=1 或 temperature=0），直接计算 logits 的 argmax。内部调用 CUB 的 `DeviceReduce::ArgMax` 找到最大值索引，然后通过 `castIdx` kernel 提取索引
  - `random<Tidx, Tval_>(...)`: 完整采样路径，执行排序、softmax、前缀和、采样四个阶段：
    1. **排序**: 使用 `fillIndices` 生成原始索引，然后调用 `radixSort` 按 logits 值降序排序
    2. **Softmax**: 并行计算 `exp((logits[i] - max) / temperature)`，其中 max 为已排序数组的第一个元素（最大值），然后将第一个元素置为 1
    3. **前缀和**: 使用 CUB 的 `DeviceScan::InclusiveSum` 计算累积概率分布
    4. **采样**: 通过 `randomSampleKernel` 在 GPU 上遍历累积分布，找到第一个大于随机数 `p = random * min(topp * CDF[n-1], CDF[topk-1])` 的索引
- **算法复杂度**: 排序阶段 O(n log n)，其他阶段 O(n)

### `op::random_sample::Calculate`
- **位置**: `/home/qy/src/Infini/InfiniCore/src/infiniop/ops/random_sample/random_sample.h`
- **主要功能**: 类型分发器，根据运行时类型信息特化模板函数
- **核心方法**:
  - `calculate<Algo>(...)`: 主入口，使用嵌套 switch 语句分发到正确的 `Tidx` × `Tval` 组合，内部调用 `switch_f` 根据采样参数选择 `argmax` 或 `random` 算法
- **设计模式**: 策略模式 + 模板元编程，编译时类型特化

## 3. 核心内核函数

### `castIdx<Tidx, Tval>`
```cuda
template <class Tidx, class Tval>
static __global__ void castIdx(Tidx *result, const cub::KeyValuePair<int, Tval> *kv_pair);
```
- **功能**: 从 CUB 的 `KeyValuePair` 中提取索引字段，解决 MACA toolkit 11.x 的 CUB 库只返回 `KeyValuePair<int, Tval>` 类型的问题
- **执行配置**: `<<<1, 1, 0, stream>>>`，单线程执行
- **时间复杂度**: O(1)

### `fillIndices<Tidx>`
```cuda
template <class Tidx>
static __global__ void fillIndices(Tidx *indices, int n);
```
- **功能**: 并行生成序列 [0, 1, 2, ..., n-1]，为后续排序提供原始索引
- **实现**: 每个线程处理一个元素 `i = blockIdx.x * blockDim.x + threadIdx.x`，写入 `indices[i] = i`
- **执行配置**: grid = `(n + block - 1) / block`，block = `min(block_size, n)`

### `partialSoftmaxKernel<T>`
```cuda
template <class T>
static __global__ void partialSoftmaxKernel(T *__restrict__ data, int n, float temperature);
```
- **功能**: 计算简化的 softmax（指数归一化），利用已排序数组的特性
- **算法**:
  - 最大值 `data[0]` 通过 `__ldg(data)` 只读加载（避免写后读冲突）
  - 对于 i > 0，计算 `data[i] = exp((data[i] - max) / temperature)`
  - 第一个元素稍后由 `setSoftmaxMaxKernel` 置为 1（即 exp(0)）
- **数值稳定性**: 使用最大值减法避免溢出，适用于 logits 已排序的场景

### `setSoftmaxMaxKernel<T>`
```cuda
template <class T>
static __global__ void setSoftmaxMaxKernel(T *__restrict__ data);
```
- **功能**: 将已排序数组的首元素置为 1，对应 `exp(0) = 1`
- **执行配置**: `<<<1, 1, 0, stream>>>`，单线程执行
- **协调**: 必须在 `partialSoftmaxKernel` 之后调用

### `randomSampleKernel<Tval, Tidx>`
```cuda
template <class Tval, class Tidx>
static __global__ void randomSampleKernel(
    Tidx *__restrict__ result,
    const Tval *__restrict__ sorted,
    const Tidx *__restrict__ indices_out,
    size_t n,
    float random, float topp, size_t topk);
```
- **功能**: 在 GPU 上执行线性搜索采样，避免数据传输到 CPU
- **算法**:
  1. 计算 top-k 和 top-p 的阈值：`threshold = min(topp * sorted[n-1], sorted[topk-1])`
  2. 生成随机采样点：`p = random * threshold`
  3. 线性遍历累积分布 `sorted[i]`，找到第一个满足 `sorted[i] >= p` 的索引
  4. 返回对应的原始索引 `indices_out[i]`
- **执行配置**: `<<<1, 1, 0, stream>>>`，单线程串行搜索（因为 n 通常较小，且为单次采样）
- **时间复杂度**: O(n) 最坏情况

## 4. CUB API 封装函数

### `argMax_<T>`
```cuda
template <class T>
static hcError_t argMax_(
    cub::KeyValuePair<int, T> *kv_pair,
    const T *logits,
    int n,
    void *workspace_ptr,
    size_t &workspace_len,
    hcStream_t stream);
```
- **功能**: 封装 CUB 的 `DeviceReduce::ArgMax`，简化模板参数
- **返回**: 设备上的键值对（索引和最大值）
- **Workspace**: 通过 nullptr 查询所需大小，实际调用时需提供预分配内存

### `radixSort<Tval, Tidx>`
```cuda
template <class Tval, class Tidx>
static hcError_t radixSort(
    void *workspace_ptr, size_t &workspace_len,
    const Tval *key_in, Tval *key_out,
    const Tidx *val_in, Tidx *val_out,
    int n,
    hcStream_t stream);
```
- **功能**: 封装 CUB 的 `DeviceRadixSort::SortPairsDescending`，按键（logits 值）降序排序，同时排列值（索引）
- **排序范围**: 第 0 位到最高位（`sizeof(Tval) * 8`），全比特排序
- **应用场景**: 将 logits 和配套索引按概率降序排列，为 top-p/top-k 采样做准备

### `inclusiveSum<T>`
```cuda
template <class T>
static hcError_t inclusiveSum(
    void *workspace_ptr, size_t &workspace_len,
    T *data, int n,
    hcStream_t stream);
```
- **功能**: 封装 CUB 的 `DeviceScan::InclusiveSum`，计算原地前缀和
- **应用场景**: 将 softmax 归一化后的概率转换为累积分布函数（CDF）

## 5. Workspace 计算算法

### `calculateWorkspace<Tidx, Tval>`
```cuda
template <class Tidx, class Tval>
utils::Result<size_t> calculateWorkspace(size_t n_);
```
- **功能**: 计算采样操作所需的最大 workspace 字节数
- **内存布局**:
  ```
  [argmax workspace (256B aligned)] [indices array (256B aligned)] [sorted logits (256B aligned)]
  [indices_out (256B aligned)] [radix_sort/inclusive_sum workspace (max of both)]
  ```
- **对齐策略**: 所有缓冲区按 256 字节对齐（`align256` 函数），满足 Metax GPU 的内存合并访问要求
- **计算步骤**:
  1. 查询 `argMax` 所需空间并加 256 字节预留
  2. 计算 indices 数组大小（`sizeof(Tidx) * n`，向上对齐到 256）
  3. 计算 sorted 数组大小（`sizeof(Tval) * n`，对齐）
  4. 计算 indices_out 大小（`sizeof(Tidx) * n`，对齐）
  5. 查询 `radixSort` 和 `inclusiveSum` 所需空间，取最大值
  6. 返回 `max(argmax_total, sum(indices + sorted + indices_out + max(sort, sum)))`
- **返回值**: `Result<size_t>`，失败时包含 Metax 错误码

### `CudaTval<Tval>` 类型映射
- **功能**: 将 Infini 的自定义浮点类型（`fp16_t`, `bf16_t`）映射到 CUDA/Metax 原生类型（`half`, `__hpcc_bfloat16`）
- **特化**:
  - `fp16_t` → `half` (16-bit 半精度浮点)
  - `bf16_t` → `__hpcc_bfloat16` (16-bit brain float)
  - 其他类型保持不变
- **应用场景**: 模板函数中统一类型处理

## 6. 类型系统

### 支持的类型组合
该实现通过编译时模板特化和运行时 switch 分发，支持以下类型组合：

**索引类型 (`Tidx`)**:
- 有符号整数: `int8_t`, `int16_t`, `int32_t`, `int64_t`
- 无符号整数: `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`

**概率类型 (`Tval`)**:
- `half` (fp16): 16-bit IEEE 754 半精度
- `__hpcc_bfloat16` (bf16): 16-bit brain float（8-bit 指数，7-bit 尾数）
- `float`: 32-bit 单精度
- `double`: 64-bit 双精度

### 类型分发策略
1. **Descriptor::create**: 使用嵌套 switch（外层索引类型 `CASE_I`，内层概率类型 `CASE_P`）特化 `calculateWorkspace`
2. **Calculate::calculate**: 先按索引类型分发（`CASE` 宏），再按概率类型分发到 `switch_f`
3. **算法执行**: 在 `Algo::argmax` 和 `Algo::random` 中通过 `CudaTval<Tval_>::Type` 转换为 Metax 原生类型

## 7. 使用示例

```cpp
// 示例：在 Metax GPU 上执行 top-p (nucleus) 采样

#include "infinicore.h"

// 1. 创建句柄和流
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_METAX, 0);
hcStream_t stream;
hcStreamCreate(&stream);

// 2. 准备张量描述符
// 概率分布：vocab_size = 32000, fp16 类型
infiniopTensorDescriptor_t probs_desc;
int64_t probs_dim = 32000;
infiniopCreateTensorDescriptor(
    &probs_desc,
    INFINI_DTYPE_F16,
    1, &probs_dim,     // 1D 张量
    nullptr,           // 默认步长（紧凑布局）
    0                  // 无额外标志
);

// 输出索引：单个 int32 值
infiniopTensorDescriptor_t result_desc;
infiniopCreateTensorDescriptor(
    &result_desc,
    INFINI_DTYPE_I32,
    0, nullptr, nullptr, 0  // 标量
);

// 3. 创建操作符描述符
op::random_sample::metax::Descriptor *sample_desc;
auto status = op::random_sample::metax::Descriptor::create(
    handle,
    &sample_desc,
    result_desc,
    probs_desc
);
if (status != INFINI_STATUS_SUCCESS) {
    // 错误处理
}

// 4. 分配 GPU 内存
half *d_probs;
int32_t *d_result;
hcMalloc(&d_probs, sizeof(half) * 32000);
hcMalloc(&d_result, sizeof(int32_t));

// 5. 分配 workspace
size_t workspace_size = sample_desc->minWorkspaceSize();
void *d_workspace;
hcMalloc(&d_workspace, workspace_size);

// 6. 填充输入数据（假设从 CPU 传输）
hcMemcpyH2DAsync(d_probs, h_logits, sizeof(half) * 32000, stream);

// 7. 执行采样：top-p=0.9, top-k=0 (无限制), temperature=1.0
float random_val = 0.345f;  // [0, 1) 随机数（从 CPU 或 GPU 生成）
float topp = 0.9f;
int topk = 0;       // 0 表示不限制 top-k
float temperature = 1.0f;

sample_desc->calculate(
    d_workspace,
    workspace_size,
    d_result,
    d_probs,
    random_val,
    topp,
    topk,
    temperature,
    stream
);

// 8. 同步并读取结果
hcStreamSynchronize(stream);
int32_t sampled_token;
hcMemcpyD2HAsync(&sampled_token, d_result, sizeof(int32_t), stream);
hcStreamSynchronize(stream);

printf("Sampled token ID: %d\n", sampled_token);

// 9. 清理资源
hcFree(d_probs);
hcFree(d_result);
hcFree(d_workspace);
delete sample_desc;
hcStreamDestroy(stream);
infiniopDestroyHandle(handle);
```

### 特殊场景示例

**场景 1：确定性采样（退化为 argmax）**
```cpp
// 当 topk=1 或 temperature=0 时，自动优化为 argmax 路径
sample_desc->calculate(
    d_workspace, workspace_size,
    d_result, d_probs,
    0.0f,    // random_val（忽略）
    0.0f,    // topp（忽略）
    1,       // topk=1，触发 argmax
    0.0f,    // temperature（忽略）
    stream
);
```

**场景 2：Top-k 采样**
```cpp
// 仅从概率最高的 50 个 token 中采样
sample_desc->calculate(
    d_workspace, workspace_size,
    d_result, d_probs,
    random_val,
    1.0f,    // topp=1.0（禁用 top-p）
    50,      // topk=50
    1.0f,    // temperature=1.0
    stream
);
```

**场景 3：温度缩放采样**
```cpp
// 降低温度使分布更集中，提高确定性
sample_desc->calculate(
    d_workspace, workspace_size,
    d_result, d_probs,
    random_val,
    0.9f,    // topp
    0,       // topk（禁用）
    0.7f,    // temperature=0.7（低于 1.0 使分布更尖锐）
    stream
);
```

## 8. 实现细节

### 内存管理策略
- **Workspace 预分配**: 在描述符创建时一次性计算并分配最大所需内存，避免运行时动态分配
- **256 字节对齐**: 所有中间缓冲区按 256 字节对齐，满足 Metax GPU 的缓存行和内存合并访问要求，提升带宽利用率
- **原地操作**: CUB 的前缀和（`InclusiveSum`）原地修改输入数组，节省内存
- **共享指针管理**: `Opaque::internal` 使用 `std::shared_ptr`，支持多个描述符共享同一设备句柄，自动管理生命周期

### 并发与线程安全
- **单操作执行**: 每次采样调用使用单个 CUDA/Metax stream，不支持多 stream 并发执行同一操作符
- **只读输入**: 输入 `probs` 数组只读，不修改原始 logits，允许跨操作共享
- **Kernel 并行性**:
  - `fillIndices`: 并行填充，O(n) 时间，O(n) 线程
  - `partialSoftmaxKernel`: 并行计算，除第一个元素外独立
  - `randomSampleKernel`: 单线程执行（适合 n 较小的场景，避免同步开销）
- **流同步**: 所有 kernels 在同一 stream 上顺序执行，隐式同步保证依赖关系

### 性能优化技术
- **类型特化**: 编译时模板展开，避免运行时分支开销
- **快速路径**: 当采样参数退化时（topk=1 或 temperature=0），自动选择 argmax 路径，跳过排序和 softmax
- **CUB 库利用**: 使用 NVIDIA/ Moore Threads 高性能原语（radix sort、reduce、scan），这些内核经过高度优化，充分利用 GPU 的共享内存和 warp 原语
- **内存合并访问**: 所有内核使用 256 字节对齐的缓冲区，确保全局内存访问合并
- **只读缓存 (`__ldg`)**: `partialSoftmaxKernel` 使用 `__ldg` 内置函数读取最大值，利用 GPU 的只读缓存，减少内存延迟
- **就地计算**: softmax 和前缀和原地修改数组，减少内存拷贝

### 错误处理机制
- **Result 类型**: `calculateWorkspace` 和 `RandomSampleInfo::create` 返回 `utils::Result<T>`，封装错误码或成功值
- **宏辅助**: `CHECK_METAX`、`CHECK_RESULT`、`CHECK_DTYPE` 等宏简化错误检查，失败时提前返回
- **Abort 调用**: 当遇到不可达的 default 分支时，调用 `std::abort()` 终止程序（用于类型分发逻辑）
- **Workspace 验证**: `calculate` 方法检查 `workspace_size < _min_workspace_size`，返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE` 错误

### 设备兼容性
- **Metax 特定**: 使用 Moore Threads 的自定义类型（`hcStream_t`, `hcError_t`, `hcMalloc` 等），通过 `metax_kernel_common.h` 和 `metax_common.h` 提供
- **CUB 版本适配**: 支持 `ENABLE_METAX_MC_API` 宏控制的两个 CUB 版本（MC API 和标准 API）
- **半精度浮点**: 支持 `half` (fp16) 和 `__hpcc_bfloat16` (bf16)，通过 `CudaTval` 特化映射
- **块大小查询**: 从设备内部句柄 (`blockSizeX()`) 获取最优 block 大小，适应不同 Metax GPU 型号

### 算法特性
- **确定性排序**: 基数排序 (`radixSort`) 是稳定排序，保证相同 logits 值时保留原始索引顺序
- **数值稳定性**: Softmax 使用最大值减法 (`exp((x - max) / temp)`) 避免溢出，适用于大负数 logits
- **Top-p/Top-k 联合裁剪**: `randomSampleKernel` 计算阈值 `min(topp * CDF[n-1], CDF[topk-1])`，同时应用两个约束
- **线性搜索采样**: 单个采样场景下，线性搜索累积分布足够高效（n 通常为几千到几万，O(n) 可接受）
- **GPU 端采样**: 完全在 GPU 上执行采样循环，避免 CPU-GPU 同步和传输延迟

### 设计模式应用
- **策略模式**: `Algo` 类封装 argmax 和 random 两种策略，运行时根据参数选择
- **模板方法模式**: `Calculate::calculate` 定义类型分发骨架，`Algo` 定义具体算法步骤
- **Pimpl 惯用法**: `Descriptor::Opaque` 隐藏 Metax 特定实现细节，减少头文件依赖
- **工厂模式**: `Descriptor::create` 作为静态工厂方法，封装复杂的对象构造逻辑
- **RAII**: 使用 `std::shared_ptr` 自动管理设备句柄生命周期
