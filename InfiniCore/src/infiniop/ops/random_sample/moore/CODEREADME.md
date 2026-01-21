# Random Sample Moore 后端实现文档

本模块实现了随机采样（Random Sample）算子的 Moore（摩尔线程 GPU）后端。该算子在 LLM 推理中用于从概率分布中采样下一个 token，支持 Top-K、Top-P（nucleus sampling）和温度采样等策略。

## 1. 模块结构

- **`random_sample_kernel.h`**: 核心实现文件，包含所有 CUDA 内核函数和算法逻辑
- **`random_sample_moore.h`**: Moore 后端描述符的宏定义声明
- **`random_sample_moore.mu`**: 描述符的具体实现，包括工厂方法、工作空间计算和算子调度

## 2. 核心数据结构

### `RandomSampleInfo`
- **位置**: `../info.h`
- **功能**: 封装随机采样算子的张量描述信息
- **成员变量**:
  - `dt_i`: 结果张量的数据类型（整数类型：int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t）
  - `dt_p`: 概率张量的数据类型（浮点类型：fp16_t, bf16_t, float, double）
  - `n`: 概率分布的元素个数（词汇表大小）

### `Descriptor`
- **位置**: `random_sample_moore.mu`, `random_sample.h`
- **功能**: Moore 后端的随机采样算子描述符，继承自 `InfiniopDescriptor`
- **内部结构**:
  - `Opaque`: 封装 `device::moore::Handle::Internal` 智能指针，持有 Moore 设备句柄的内部状态
  - `_info`: `RandomSampleInfo` 实例，存储张量类型和形状信息
  - `_min_workspace_size`: 所需最小工作空间大小
- **核心接口**:
  - `create()`: 工厂方法，根据张量描述符创建描述符实例并计算工作空间
  - `minWorkspaceSize()`: 返回工作空间大小
  - `calculate()`: 执行采样计算

### `Algo`
- **位置**: `random_sample_kernel.h`
- **功能**: 核心算法实现类，封装 Top-K/Top-P 采样的完整流程
- **成员变量**:
  - `block_size`: CUDA kernel 的块大小，从设备句柄动态获取
- **核心方法**:
  - `argmax<Tidx, Tval_>()`: 简化路径，直接返回最大概率的索引（当 random_val=0 或 topp=0 或 topk=1 或 temperature=0 时）
  - `random<Tidx, Tval_>()`: 完整采样路径，执行排序、softmax、累积和、采样

## 3. 核心函数与内核

### 辅助转换内核

#### `bfloat16_to_float_kernel`
```cpp
__global__ void bfloat16_to_float_kernel(const __mt_bfloat16 *in, float *out, int n)
```
- **功能**: 将 BF16 格式的 logits 转换为 float，用于后续 CUB 操作
- **实现**: 每个线程处理一个元素，简单类型转换
- **网格配置**: `grid_size = (n + 255) / 256`, `block_size = 256`

#### `float_to_bfloat16_kernel`
```cpp
__global__ void float_to_bfloat16_kernel(const float *in, __mt_bfloat16 *out, int n)
```
- **功能**: 将 float 格式转换回 BF16
- **实现**: 同样是单线程单元素转换

#### `write_kv_bfloat16_kernel`
```cpp
__global__ void write_kv_bfloat16_kernel(
    cub::KeyValuePair<int, __mt_bfloat16> *dst,
    const cub::KeyValuePair<int, float> *src)
```
- **功能**: 将 float 类型的 KeyValuePair（CUB ArgMax 结果）转换为 BF16 类型
- **配置**: 单线程单块执行 `<<<1, 1>>>`

### CUB 封装函数

#### `argMax_<T>()`
```cpp
template <class T>
static musaError_t argMax_(
    cub::KeyValuePair<int, T> *kv_pair,
    const T *logits,
    int n,
    void *workspace_ptr,
    size_t &workspace_len,
    musaStream_t stream)
```
- **功能**: 在 logits 中查找最大值及其索引
- **BF16 特殊处理**:
  1. 分配临时 float 缓冲区存储转换后的 logits
  2. 分配临时 `KeyValuePair<int, float>` 缓冲区
  3. 执行 `bfloat16_to_float_kernel` 转换
  4. 调用 `cub::DeviceReduce::ArgMax` 在 float 数据上执行 argmax
  5. 调用 `write_kv_bfloat16_kernel` 将结果转回 BF16
- **非 BF16**: 直接调用 `cub::DeviceReduce::ArgMax`
- **内存对齐**: 所有临时缓冲区 256 字节对齐

#### `radixSort<Tval, Tidx>()`
```cpp
template <class Tval, class Tidx>
static musaError_t radixSort(
    void *workspace_ptr, size_t &workspace_len,
    const Tval *key_in, Tval *key_out,
    const Tidx *val_in, Tidx *val_out,
    int n,
    musaStream_t stream)
```
- **功能**: 基于键值对的降序排序（键为概率，值为索引）
- **算法**: 使用 CUB 的 `DeviceRadixSort::SortPairsDescending`
- **BF16 特殊处理**:
  1. 分配两个 float 缓冲区：`temp_key_in` 和 `temp_key_out`
  2. 转换 BF16 logits 到 float
  3. 在 float 数据上执行排序
  4. 将排序后的 float 键转回 BF16
- **时间复杂度**: O(n log n)（基数排序）

#### `inclusiveSum<T>()`
```cpp
template <class T>
static musaError_t inclusiveSum(
    void *workspace_ptr, size_t &workspace_len,
    T *data, int n,
    musaStream_t stream)
```
- **功能**: 计算前缀和（inclusive scan），用于计算累积概率分布
- **算法**: 使用 CUB 的 `DeviceScan::InclusiveSum`
- **BF16 特殊处理**:
  1. 分配临时 float 缓冲区
  2. 转换 BF16 到 float
  3. 在 float 数据上执行 inclusive sum
  4. 将结果转回 BF16
- **时间复杂度**: O(n)

### 采样内核

#### `castIdx<Tidx, Tval>`
```cpp
template <class Tidx, class Tval>
static __global__ void castIdx(Tidx *result, const cub::KeyValuePair<int, Tval> *kv_pair)
```
- **功能**: 从 KeyValuePair 中提取索引（key），用于 argmax 路径
- **配置**: `<<<1, 1>>>` 单线程执行

#### `fillIndices<Tidx>`
```cpp
template <class Tidx>
static __global__ void fillIndices(Tidx *indices, int n)
```
- **功能**: 初始化索引数组为 `[0, 1, 2, ..., n-1]`，用于排序时追踪原始位置
- **实现**: 每个线程写入 `indices[i] = i`
- **配置**: `grid_size = (n + 255) / 256`, `block_size = 256`

#### `partialSoftmaxKernel<T>`
```cpp
template <class T>
static __global__ void partialSoftmaxKernel(
    T *__restrict__ data, int n,
    float temperature)
```
- **功能**: 计算简化的 softmax，仅执行指数部分（exp((x - max) / temperature)）
- **优化**: 由于数据已排序，最大值必定在 `data[0]`，无需调用 `__ldg` 或 reduce
- **实现细节**:
  - 跳过 `data[0]`（稍后单独设置为 1.0）
  - 对每个元素计算 `exp((data[i] - data[0]) / temperature)`
- **配置**: 256 线程/块

#### `setSoftmaxMaxKernel<T>`
```cpp
template <class T>
static __global__ void setSoftmaxMaxKernel(T *__restrict__ data)
```
- **功能**: 将第一个元素（最大概率）设置为 1.0，即 `exp(0)`
- **实现**: 显式转换为 `T` 类型以避免 BF16 赋值的二义性
- **配置**: `<<<1, 1>>>`

#### `randomSampleKernel<Tval, Tidx>`
```cpp
template <class Tval, class Tidx>
static __global__ void randomSampleKernel(
    Tidx *__restrict__ result,
    const Tval *__restrict__ sorted,
    const Tidx *__restrict__ indices_out,
    size_t n,
    float random, float topp, size_t topk)
```
- **功能**: 根据随机数和累积概率分布执行采样
- **算法**:
  1. 计算阈值 `p = random * min(topp * sorted[n-1], sorted[topk-1])`（Top-P 和 Top-K 的交集）
  2. 线性遍历排序后的累积概率数组
  3. 返回第一个满足 `sorted[i] >= p` 的索引
- **配置**: `<<<1, 1>>>` 单线程串行搜索
- **复杂度**: O(n) 最坏情况

### 工作空间计算

#### `calculateWorkspace<Tidx, Tval>()`
```cpp
template <class Tidx, class Tval>
utils::Result<size_t> calculateWorkspace(size_t n_)
```
- **功能**: 预先计算所需工作空间大小
- **组成**:
  - ArgMax 工作区 + 256 字节 kv_pair 对齐
  - 三个主缓冲区：`indices` (Tidx*n), `sorted` (Tval*n), `indices_out` (Tidx*n)
  - CUB RadixSort 工作区
  - CUB InclusiveSum 工作区
  - BF16 额外开销：4 * n * sizeof(float)（ArgMax 1n, InclusiveSum 1n, RadixSort 2n）
- **返回**: `max(argmax_total, radix_sort + inclusive_sum + temp_buffers)`

## 4. 算法流程

### Argmax 路径（简化模式）
**触发条件**: `random_val == 0 || topp == 0 || topk == 1 || temperature == 0`

1. 调用 `argMax_<T>()` 查找最大概率索引
2. 调用 `castIdx` 提取索引并写入结果

### 完整采样路径
**适用场景**: 真随机采样，支持 Top-K 和 Top-P

1. **初始化索引** (`fillIndices`)
   - 生成 `[0, 1, 2, ..., n-1]` 数组

2. **排序** (`radixSort`)
   - 基于 logits 降序排序键值对
   - 排序后：`sorted[0]` 是最大概率，`indices_out[0]` 是其对应索引

3. **Softmax** (`partialSoftmaxKernel` + `setSoftmaxMaxKernel`)
   - 计算 `exp((logits[i] - logits[0]) / temperature)`
   - 将 `sorted[0]` 设为 1.0

4. **累积和** (`inclusiveSum`)
   - 计算 `sorted[i] += sorted[i-1]`
   - 结果：`sorted[i]` 是前 i 个概率的累积和

5. **采样** (`randomSampleKernel`)
   - 计算 `threshold = random * min(topp * sorted[n-1], sorted[topk-1])`
   - 线性搜索第一个 `sorted[i] >= threshold` 的位置
   - 返回 `indices_out[i]`

## 5. BF16 支持策略

由于 Moore 的 CUB 库不直接支持 BF16 类型，代码采用**类型提升策略**：

### 设计原则
- **输入**: BF16 logits (`__mt_bfloat16`)
- **提升**: 在所有 CUB 操作前临时转换为 float
- **回降**: 操作结果立即转回 BF16
- **内存管理**: 使用 256 字节对齐的缓冲区切分，避免数据覆盖

### 具体实现
1. **ArgMax**: BF16 logits → float temp_logits → float kv_pair → BF16 kv_pair
2. **RadixSort**: BF16 keys → float temp_keys → 排序 → float sorted_keys → BF16 sorted_keys
3. **InclusiveSum**: BF16 data → float temp_data → scan → float temp_data → BF16 data

### 内存开销
对于词汇表大小 n，额外需要 `4 * n * sizeof(float)` = 16n 字节临时显存。

## 6. 类型转换特化

### `CudaTval<Tval>` 模板结构
```cpp
template <class Tval>
struct CudaTval {
    using Type = Tval;
};

template <>
struct CudaTval<fp16_t> {
    using Type = half;  // 映射 fp16_t 到 CUDA half
};

template <>
struct CudaTval<bf16_t> {
    using Type = __mt_bfloat16;  // 映射 bf16_t 到 Moore BF16
};
```
- **功能**: 将抽象数据类型（`fp16_t`, `bf16_t`）映射到具体硬件类型（`half`, `__mt_bfloat16`）
- **使用**: 在模板函数中通过 `typename CudaTval<Tval_>::Type` 获取实际类型

## 7. API 接口

### 创建描述符
```cpp
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t result_desc,
    infiniopTensorDescriptor_t probs_desc);
```
- **输入**:
  - `handle`: Moore 设备句柄
  - `result_desc`: 输出标量张量（采样结果索引）
  - `probs_desc`: 输入 1D 张量（概率分布）
- **输出**: 构造完成的描述符实例
- **逻辑**: 验证张量形状 → 类型分发 → 计算工作空间 → 构造描述符

### 获取工作空间大小
```cpp
size_t Descriptor::minWorkspaceSize() const;
```
- **返回**: 执行采样所需的最小 GPU 显存量（字节）

### 执行计算
```cpp
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *result,
    const void *probs,
    float random_val,
    float topp,
    int topk,
    float temperature,
    void *stream) const;
```
- **输入**:
  - `workspace`: GPU 工作空间指针
  - `workspace_size`: 工作空间大小
  - `probs`: GPU 指针，概率分布 logits
  - `random_val`: [0, 1) 随机数，由 CPU 生成
  - `topp`: Top-P 阈值（nucleus sampling）
  - `topk`: Top-K 限制
  - `temperature`: 温度参数，控制分布平滑度
  - `stream`: MUSA 流
- **输出**: `result`（GPU 指针），写入单个索引值

## 8. 使用示例

```cpp
// 1. 创建张量描述符
infiniopTensorDescriptor_t probs_desc, result_desc;
infiniopCreateTensorDescriptor(&probs_desc, INFINI_DTYPE_F16, nullptr, 1, &vocab_size);
infiniopCreateTensorDescriptor(&result_desc, INFINI_DTYPE_I32, nullptr, 0, nullptr);

// 2. 创建算子描述符
op::random_sample::moore::Descriptor *desc;
auto status = op::random_sample::moore::Descriptor::create(
    handle, &desc, result_desc, probs_desc);

// 3. 分配 GPU 内存
size_t workspace_size = desc->minWorkspaceSize();
void *workspace, *d_probs, *d_result;
musaMalloc(&d_probs, vocab_size * sizeof(fp16_t));
musaMalloc(&d_result, sizeof(int32_t));
musaMalloc(&workspace, workspace_size);

// 4. 上传概率分布（假设 h_probs 是 CPU 侧 logits）
musaMemcpyAsync(d_probs, h_probs, vocab_size * sizeof(fp16_t),
                musaMemcpyHostToDevice, stream);

// 5. 生成随机数并调用采样
float random_val = generate_random();  // CPU 侧生成 [0, 1)
status = desc->calculate(
    workspace, workspace_size,
    d_result, d_probs,
    random_val, 0.9f, 50, 1.0f,  // topp=0.9, topk=50, temperature=1.0
    stream);

// 6. 下载结果
int32_t h_result;
musaMemcpyAsync(&h_result, d_result, sizeof(int32_t),
                musaMemcpyDeviceToHost, stream);
musaStreamSynchronize(stream);

// 7. 使用采样结果
printf("Sampled token ID: %d\n", h_result);
```

## 9. 实现细节

### 内存管理
- **工作空间分区**: 使用指针偏移和递减 `workspace_len` 的方式动态切分大块内存
- **256 字节对齐**: 所有临时缓冲区通过 `align256()` 对齐，保证 MUSA Coalesced Access 性能
- **零拷贝优化**: ArgMax 结果直接使用前 256 字节，避免额外分配

### 并发与流
- **异步执行**: 所有内核调用都是异步的，依赖流管理
- **内核同步**: 通过 stream 隐式排序，无需显式同步点
- **单流设计**: 假设所有操作在同一 stream 上，不支持多流并发

### 性能优化
1. **BF16 优化**: 仅在必要时转换，避免冗余类型提升
2. **Softmax 简化**: 利用已排序性质，省略全局 Max Reduce
3. **CUB 复用**: 临时缓冲区在 ArgMax、RadixSort、InclusiveSum 间共享
4. **线性搜索**: 采样 kernel 使用串行搜索，避免复杂的二分查找实现

### 错误处理
- **工作空间不足**: 返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
- **类型不匹配**: 编译期 `abort()`（代码 unreachable）
- **MUSA 错误传播**: 使用 `CHECK_MOORE` 宏封装 CUDA/MUSA 错误检查

### 依赖关系
- **上层依赖**:
  - `infinicore.h`: Infini 核心类型定义
  - `../random_sample.h`: 基类描述符和 `RandomSampleInfo`
  - `../../operator.h`: `InfiniopDescriptor` 基类
- **设备层依赖**:
  - `../../../devices/moore/moore_kernel_common.h`: MUSA 内核通用定义
  - `../../../devices/moore/moore_common.h`: Moore 设备抽象
  - `../../../devices/moore/moore_handle.h`: Moore 设备句柄
- **外部库**:
  - CUB: NVIDIA/Moore 并发原语库（radix sort, reduce, scan）
  - MUSA Runtime: Moore GPU 驱动 API

### 设计模式
- **策略模式**: `Algo` 类封装采样算法，`Calculate::calculate` 通过模板多态调度
- **工厂方法**: `Descriptor::create` 根据 dtype 分支创建特化实例
- **RAII**: `Opaque` 使用 `std::shared_ptr` 管理设备句柄生命周期
- **模板特化**: `CudaTval` 和 `if constexpr` 实现编译期类型分发

## 10. 算法复杂度分析

### 时间复杂度
- **ArgMax 路径**: O(n)（CUB Reduce）
- **完整采样路径**:
  - 填充索引: O(n)
  - RadixSort: O(n log n)
  - Softmax: O(n)
  - InclusiveSum: O(n)
  - 采样搜索: O(n) 最坏情况
  - **总计**: O(n log n)（受排序主导）

### 空间复杂度
- **主缓冲区**: 3n（indices + sorted + indices_out）
- **CUB 工作区**: O(n)（具体取决于 CUB 实现）
- **BF16 临时**: 4n * sizeof(float) = 16n 字节
- **总计**: O(n)

### 实际性能特征
- **小词汇表**（n < 8k）: 内存带宽受限，排序开销相对较小
- **大词汇表**（n > 32k）: 排序成为瓶颈，BF16 转换开销显著
- **Top-K 限制**: 排序后只需前 K 个元素，但当前实现仍全量排序（优化空间）
