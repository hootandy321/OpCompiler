# MetaX TopKSoftmax 算子实现文档

本模块实现了在沐曦 MetaX GPU 设备上的 TopKSoftmax 操作，结合了 TopK 选择和 Softmax 归一化两个功能，主要用于 MoE (Mixture of Experts) 模型中的专家路由选择和门控机制。

## 1. 模块结构

- **`topksoftmax_metax.cuh`**: MetaX 设备实现的头文件声明，通过宏 `DESCRIPTOR(metax)` 定义 Descriptor 类
- **`topksoftmax_metax.maca`**: MetaX 设备实现的核心源文件，包含描述符创建、kernel 启动逻辑和计算调度

## 2. 核心类

### `op::topksoftmax::metax::Descriptor`

- **位置**: 通过 `topksoftmax_metax.cuh` 中的宏定义生成
- **继承**: `InfiniopDescriptor`
- **主要功能**: 管理 MetaX 设备上的 TopKSoftmax 操作描述符，封装设备句柄和操作配置信息

#### 内部结构

**`struct Descriptor::Opaque`**
- **位置**: `topksoftmax_metax.maca:14-16`
- **功能**: 封装 MetaX 设备的内部句柄，使用 Pimpl 模式隐藏实现细节
- **成员**:
  - `std::shared_ptr<device::metax::Handle::Internal> internal`: MetaX 设备句柄的智能指针，管理设备资源生命周期

#### 关键成员变量

- **`Opaque *_opaque`**: 不透明指针，指向设备特定的内部状态
- **`TopksoftmaxInfo _info`**: 张量形状和类型信息
  - `xtype`: 输入数据类型 (F32/F16/BF16)
  - `shape`: 张量形状 [N, width]
  - `x_strides`: 张量步长
  - `N`: 批次大小 (token 数量)
  - `width`: 特征维度 (专家数量)
- **`size_t _workspace_size`**: 所需工作空间大小 (当前实现为 0)

#### 核心方法

**`~Descriptor()`**
- **位置**: `topksoftmax_metax.maca:18-20`
- **功能**: 析构函数，释放 Opaque 内部状态内存

**`static create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t x_desc)`**
- **位置**: `topksoftmax_metax.maca:22-40`
- **功能**: 创建并初始化 MetaX TopKSoftmax 描述符
- **算法流程**:
  1. 调用 `TopksoftmaxInfo::create()` 验证输入张量描述符
  2. 检查步长约束: `x_strides[1] == 1` (要求最后一维连续)
  3. 从 `infiniopHandle_t` 提取 MetaX 设备内部句柄
  4. 构造 Descriptor 实例并返回
- **返回值**:
  - `INFINI_STATUS_SUCCESS`: 创建成功
  - `INFINI_STATUS_BAD_TENSOR_STRIDES`: 步长不满足约束
  - 其他错误码由 `TopksoftmaxInfo::create()` 传播

**`calculate(void *workspace, size_t workspace_size, float *values, int *indices, const void *x, const size_t topk, const bool norm, void *stream_) const`**
- **位置**: `topksoftmax_metax.maca:65-92`
- **功能**: 执行 TopKSoftmax 计算的主入口函数
- **参数**:
  - `workspace`: 工作空间指针 (当前未使用)
  - `workspace_size`: 工作空间大小 (需 >= 0)
  - `values`: 输出 TopK 值数组 `[N, topk]`
  - `indices`: 输出 TopK 索引数组 `[N, topk]`
  - `x`: 输入数据 `[N, width]`
  - `topk`: 选择的 Top-K 个数
  - `norm`: 是否对 TopK 结果进行归一化 (除以 TopK 之和)
  - `stream_`: MetaX 计算流 (`hcStream_t`)
- **算法流程**:
  1. 验证工作空间大小
  2. 根据 `width` 维度选择合适的 Block Size:
     - `width <= 128`: 使用 BLOCK_SIZE=128
     - `width <= 256`: 使用 BLOCK_SIZE=256
     - `width <= 512`: 使用 BLOCK_SIZE=512
     - 其他: 返回错误 (当前不支持 >512 的维度)
  3. 调用 `launch_topksoftmax<BLOCK_SIZE>()` 启动 kernel
- **返回值**:
  - `INFINI_STATUS_SUCCESS`: 计算成功
  - `INFINI_STATUS_INSUFFICIENT_WORKSPACE`: 工作空间不足
  - `INFINI_STATUS_INTERNAL_ERROR`: 不支持的宽度维度
  - `INFINI_STATUS_BAD_TENSOR_DTYPE`: 不支持的数据类型

## 3. 内部辅助函数

### `launch_topksoftmax<float *d_values_out, int *d_indices_out, const void *d_input, const size_t N, const size_t width, const size_t topk, const bool norm, infiniDtype_t xtype, hcStream_t stream>`

- **位置**: `topksoftmax_metax.maca:44-61`
- **模板参数**: `BLOCK_SIZE` (128/256/512)
- **功能**: 根据输入数据类型分发到相应的 CUDA kernel
- **Kernel 启动配置**:
  - Grid: `(N, 1, 1)` - 每一行一个 block
  - Block: `(BLOCK_SIZE, 1, 1)` - 固定线程数
- **支持的数据类型**:
  - `INFINI_DTYPE_F32`: 单精度浮点
  - `INFINI_DTYPE_F16`: 半精度浮点 (half)
  - `INFINI_DTYPE_BF16`: bfloat16 格式
- **调用的 Kernel**: `softmax_topk_row_kernel<T, BLOCK_SIZE>` (定义在 `../cuda/kernel.cuh`)

## 4. Kernel 实现细节 (位于 ../cuda/kernel.cuh)

### `softmax_topk_row_kernel<T, BLOCK_SIZE>`

- **位置**: `topksoftmax/cuda/kernel.cuh:22-144`
- **功能**: 对输入矩阵的每一行执行 Softmax + TopK + 可选归一化
- **算法流程** (7 个步骤):

#### 步骤 1: 计算行最大值 (数值稳定性)
- 使用 CUB `BlockReduce` 进行归约
- 每个线程处理一个元素，找出局部最大值
- 块内归约得到全局最大值 `shared_max`
- **API**: `BlockReduce(temp_storage_max).Reduce(thread_max, cuda::maximum() / cub::Max())`

#### 步骤 2: 计算指数和 (Softmax 分母)
- 每个线程计算: `exp(x[i] - max_x)` (防止溢出)
- 块内归约得到指数和 `shared_sum`
- **API**: `BlockReduce(temp_storage_sum).Sum(exp_val)`
- **辅助函数**: `exp_func<T>()` 处理 F16/BF16 到 F32 的转换

#### 步骤 3: 计算 Softmax 概率
- 归一化: `exp_val /= shared_sum`
- 此时 `exp_val` 为 Softmax 后的概率值

#### 步骤 4: TopK 排序选择
- 初始化: `thread_values = -FLT_MAX`, `thread_indices = -1`
- 每个线程保留自身的 (概率值, 索引) 对
- 使用 CUB `BlockRadixSort` 进行块内降序排序
- **API**: `BlockRadixSort(temp_storage).SortDescending(thread_values, thread_indices)`
- 排序后, Block 内所有元素按概率降序排列

#### 步骤 5: TopK 之和 (可选归一化准备)
- 仅 Warp 0 (前 32 个线程) 参与
- 使用 `WarpReduce` 计算 TopK 个概率之和
- **API**: `WarpReduce(temp_storage).Sum(value)`
- 添加小常数 `1e-9f` 防止除零

#### 步骤 6: Norm 归一化 (可选)
- 条件: `norm == true`
- 对 TopK 的每个概率值除以 TopK 之和
- 结果: TopK 概率的重新归一化分布

#### 步骤 7: 输出结果
- 仅前 `topk` 个线程执行写操作
- 写入 `values_topk_output[tid]` 和 `indices_topk_output[tid]`
- 输出形状: `[N, topk]`

### 并行模式
- **Grid-Stride**: N 个 block 并行处理 N 行
- **Block 内协作**:
  - 所有线程参与最大值和指数和的归约
  - Warp 0 (前 32 线程) 负责 TopK 选择和输出
- **同步点**:
  - `__syncthreads()`: 块内同步 (共享变量读写)
  - `__syncwarp()`: Warp 内同步

## 5. API 接口

```cpp
// 创建描述符
infiniStatus_t op::topksoftmax::metax::Descriptor::create(
    infiniopHandle_t handle,                      // MetaX 设备句柄
    Descriptor **desc_ptr,                        // 输出: 描述符指针
    infiniopTensorDescriptor_t x_desc             // 输入张量描述符 [N, width]
);

// 执行计算
infiniStatus_t op::topksoftmax::metax::Descriptor::calculate(
    void *workspace,                              // 工作空间 (可为 nullptr)
    size_t workspace_size,                        // 工作空间大小 (需 >= 0)
    float *values,                                // 输出: TopK 值 [N, topk]
    int *indices,                                 // 输出: TopK 索引 [N, topk]
    const void *x,                                // 输入数据 [N, width]
    const size_t topk,                            // TopK 个数
    const bool norm,                              // 是否归一化 TopK 结果
    void *stream_                                 // MetaX 计算流
) const;

// 查询工作空间大小
size_t op::topksoftmax::metax::Descriptor::workspaceSize() const;
```

## 6. 使用示例

```cpp
// 初始化 MetaX 设备和句柄
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_METAX, 0);

// 准备输入张量描述符: [batch_size, num_experts]
std::vector<size_t> shape = {128, 256};  // 128 tokens, 256 experts
std::vector<ptrdiff_t> strides = {256, 1};
infiniopTensorDescriptor_t x_desc;
infiniopCreateTensorDescriptor(&x_desc, INFINI_DTYPE_F16, shape, strides);

// 创建 TopKSoftmax 描述符
op::topksoftmax::metax::Descriptor *desc;
auto status = op::topksoftmax::metax::Descriptor::create(handle, &desc, x_desc);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 分配内存
half *d_input;
float *d_values;
int *d_indices;
size_t input_bytes = 128 * 256 * sizeof(half);
size_t output_bytes = 128 * 6 * sizeof(float);  // topk=6
hcMalloc((void**)&d_input, input_bytes);
hcMalloc((void**)&d_values, output_bytes);
hcMalloc((void**)&d_indices, 128 * 6 * sizeof(int));

// 获取或创建计算流
hcStream_t stream;
hcStreamCreate(&stream);

// 执行 TopKSoftmax 计算
size_t topk = 6;  // 选择 Top-6 专家
bool norm = true; // 对 Top-6 概率重新归一化
status = desc->calculate(
    nullptr,           // workspace (当前实现不需要)
    0,                 // workspace_size
    d_values,          // 输出: Top-6 概率值
    d_indices,         // 输出: Top-6 专家索引
    d_input,           // 输入: 门控分数
    topk,
    norm,
    stream             // 计算流
);

// 同步并取回结果
hcStreamSynchronize(stream);
std::vector<float> h_values(128 * 6);
std::vector<int> h_indices(128 * 6);
hcMemcpyDtoH(h_values.data(), d_values, output_bytes);
hcMemcpyDtoH(h_indices.data(), d_indices, 128 * 6 * sizeof(int));

// 清理资源
delete desc;
infiniopDestroyTensorDescriptor(x_desc);
hcFree(d_input);
hcFree(d_values);
hcFree(d_indices);
hcStreamDestroy(stream);
infiniopDestroyHandle(handle);
```

## 7. 实现细节

### 约束条件

1. **输入维度**: 必须是 2D 张量 `[N, width]`
   - `N`: 批次大小 (通常为 token 数量)
   - `width`: 特征维度 (通常为专家数量)

2. **步长约束**: `x_strides[1] == 1` (最后一维必须连续存储)

3. **支持的数据类型**:
   - `INFINI_DTYPE_F32`: 32-bit 浮点
   - `INFINI_DTYPE_F16`: 16-bit 浮点 (half)
   - `INFINI_DTYPE_BF16`: bfloat16

4. **宽度限制**: `width <= 512`
   - 当前实现仅支持 128/256/512 三种 Block Size

### 内存管理

- **零拷贝工作空间**: 当前实现不需要额外工作空间 (`_workspace_size = 0`)
- **设备端计算**: 所有计算在 MetaX GPU 上执行
- **句柄管理**: 使用 `std::shared_ptr` 管理设备内部状态，自动释放资源

### 并发模型

- **Grid-Level 并行**: N 个 Block 并行处理 N 行数据
- **Block-Level 协作**:
  - 所有线程参与归约操作 (最大值、指数和)
  - Warp 0 (前 32 线程) 负责 TopK 选择和输出
- **同步机制**:
  - `__syncthreads()`: 块内屏障同步
  - `__syncwarp()`: Warp 内同步
- **无竞争写入**: 每个 Block 独立写入输出数组的不同行

### 性能优化策略

1. **Block Size 自适应选择**:
   - 根据 `width` 动态选择 128/256/512，最大化并行效率
   - 避免 Block 内线程空闲

2. **数值稳定性**:
   - Softmax 使用 `exp(x - max_x)` 防止溢出
   - TopK 归一化添加 `1e-9f` 防止除零

3. **CUB 原语**:
   - `BlockReduce`: 高效块内归约
   - `BlockRadixSort`: 块内排序 (O(log n) 时间)
   - `WarpReduce`: Warp 级归约 (单指令多数据)

4. **类型转换优化**:
   - `exp_func<T>()` 编译期特化，零运行时开销
   - F16/BF16 统一转为 F32 计算，避免精度损失

5. **内存访问模式**:
   - 合并读取: Block 内线程连续访问输入数组
   - 写入分散: 仅前 32 个线程写入 TopK 结果

### 错误处理

- **编译时类型检查**: 使用 `if constexpr` 确保类型安全
- **运行时验证**:
  - 工作空间大小检查
  - 数据类型合法性检查
  - 张量形状和步长约束检查
- **错误传播**: 通过 `Result<T>` 模式和状态码传播错误

### 设计模式

1. **Pimpl 模式**: `struct Opaque` 隐藏 MetaX 设备特定实现
2. **策略模式**: 根据 `width` 选择不同 Block Size 策略
3. **模板元编程**: `launch_topksoftmax<BLOCK_SIZE>()` 编译期生成多个 kernel 版本
4. **RAII**: `std::shared_ptr` 管理设备句柄生命周期

### 依赖关系

**外部依赖**:
- CUB 库: `cub/block/block_reduce.cuh`, `cub/block/block_radix_sort.cuh`
- MetaX 驱动: `hcblas`, `hcdnn` (或 `mcblas`, `mcdnn`)
- CUDA Runtime: `__expf`, 类型转换函数

**内部依赖**:
- `../topksoftmax.h`: Descriptor 宏定义
- `../info.h`: `TopksoftmaxInfo` 类
- `../../../devices/metax/metax_common.h`: MetaX 设备句柄
- `../../../devices/metax/metax_kernel_common.h`: Kernel 公共定义
- `../cuda/kernel.cuh`: CUDA kernel 实现

### 典型应用场景

本算子主要用于 **Mixture of Experts (MoE)** 模型:
1. **输入**: 门控网络 (Gating Network) 对每个 token 输出的专家分数 `[batch_size, num_experts]`
2. **Softmax**: 将分数转换为概率分布
3. **TopK**: 选择概率最高的 K 个专家
4. **Norm (可选)**: 对 TopK 概率重新归一化，确保和为 1
5. **输出**: TopK 专家的索引和归一化权重，用于后续的专家激活和加权聚合

### 算法复杂度

- **时间复杂度**: O(N * width)
  - 最大值归约: O(width)
  - 指数和归约: O(width)
  - 块内排序: O(log width) (使用基数排序)
- **空间复杂度**: O(N * topk)
  - 仅存储 TopK 结果，不保留完整 Softmax 输出
- **并行度**: O(N * min(BLOCK_SIZE, width))
