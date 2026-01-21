# Causal Softmax Bang 实现核心文档

本模块实现了 Cambricon (寒武纪) MLU 硬件上的因果掩码 softmax 操作，专为 Transformer 模型的自注意力机制优化。该实现通过分块处理、类型转换优化和向量化计算，在有限的片上内存 (NRAM) 约束下实现高性能的因果 softmax 计算。

## 1. 模块结构

- **`causal_softmax_bang.h`**: Bang 后端描述符声明文件，通过 DESCRIPTOR 宏定义 Descriptor 类结构
- **`causal_softmax_bang.mlu`**: 核心 MLU 内核实现，包含 softmax 分步计算函数、全局 kernel 函数和描述符方法实现

## 2. 核心类与结构

### `op::causal_softmax::CausalSoftmaxInfo`
- **位置**: `../info.h` (父目录)
- **主要功能**: 存储因果 softmax 操作的元数据和步长信息
- **关键成员**:
  - `dtype`: 数据类型 (F16/BF16/F32)
  - `batch_size`: 批次大小 (3D tensor) 或 1 (2D tensor)
  - `seq_len`: 查询序列长度 (最后一维的维度数)
  - `total_seq_len`: 键值对总序列长度
  - `y_stride_b/i/j`: 输出张量的批次、序列、位置维度的步长
  - `x_stride_b/i/j`: 输入张量的批次、序列、位置维度的步长
- **静态方法**:
  - `create(y_desc, x_desc)`: 验证张量描述符并构造 CausalSoftmaxInfo 对象，支持 2D/3D 张量，要求 `total_seq_len >= seq_len`

### `op::causal_softmax::bang::Descriptor`
- **位置**: `causal_softmax_bang.mlu`
- **主要功能**: Bang 设备上的因果 softmax 操作符描述符，管理设备句柄和内核启动
- **内部结构**:
  - `struct Opaque`: 封装设备内部句柄 (`std::shared_ptr<device::bang::Handle::Internal>`)
  - `CausalSoftmaxInfo _info`: 操作元数据
  - `size_t _workspace_size`: 工作空间大小 (当前为 0)
- **核心方法**:
  - `create(handle_, desc_ptr, y_desc, x_desc)`: 静态工厂方法，验证张量兼容性，初始化描述符
  - `calculate(workspace, workspace_size, y, x, stream)`: 执行因果 softmax 计算，根据 dtype 分发到对应的模板实例化
  - `~Descriptor()`: 析构函数，释放 Opaque 内部句柄

## 3. API 接口

### MLU 设备函数

```cpp
// 分步处理 softmax 的核心函数
template <typename T>
__mlu_func__ void processSoftmaxStep(
    T *output, const T *input, float scalar,
    int num_elements, int stride, bool is_exp_phase);
```
- **功能**: 执行 softmax 的指数阶段或归一化阶段
- **参数**:
  - `output`: 输出缓冲区指针
  - `input`: 输入缓冲区指针 (指数阶段为 x，归一化阶段为 output)
  - `scalar`: 标量值 (指数阶段为 max_val，归一化阶段为 1.0f/sum_val)
  - `num_elements`: 处理元素数量
  - `stride`: 内存步长 (字节)
  - `is_exp_phase`: true 为指数计算，false 为归一化
- **实现细节**:
  - 使用 NRAM 分块处理 (chunk_size)，支持 F16/BF16/F32
  - 非浮点类型通过 `__bang_half2float` / `__bang_bfloat162float` 转换后计算
  - 2D memcpy 实现分散/gather (GDRAM2NRAM/NRAM2GDRAM)
  - 指数阶段调用 `__bang_sub_scalar` + `__bang_active_exphp`
  - 归一化阶段调用 `__bang_mul_scalar`

```cpp
// 因果 softmax 全局 kernel
template <typename T>
__mlu_global__ void causalSoftmax(
    T *y, const T *x,
    size_t batch_size, size_t seq_len, size_t total_seq_len,
    ptrdiff_t y_stride_b, ptrdiff_t y_stride_i, ptrdiff_t y_stride_j,
    ptrdiff_t x_stride_b, ptrdiff_t x_stride_i, ptrdiff_t x_stride_j);
```
- **功能**: 并行计算因果 softmax，每个核心处理多个 (batch, seq_len) 对
- **负载均衡**:
  - 总任务数: `batch_size * seq_len`
  - 每核心任务数: `(total_tasks + task_num - 1) / task_num`
  - 任务索引: `index = task_id * tasks_per_core + local_index`
- **因果掩码实现**:
  - 计算 `valid_len = total_seq_len - seq_len + i + 1`
  - 将 `[valid_len, total_seq_len)` 位置置零
- **Softmax 算法**:
  1. 调用 `maxBatched` 计算最大值 (数值稳定性)
  2. 调用 `processSoftmaxStep(..., max_val, ..., true)` 计算 exp(x - max)
  3. 调用 `sumBatched` 计算指数和
  4. 调用 `processSoftmaxStep(..., 1.0f/sum_val, ..., false)` 归一化

### 主机端函数

```cpp
template <typename T>
void causalSoftmaxUnion(
    void *workspace, int core_per_cluster, int cluster_count,
    cnrtQueue_t queue, void *y, const void *x,
    const op::causal_softmax::CausalSoftmaxInfo *info);
```
- **功能**: 配置并启动 MLU kernel
- **Kernel 配置**:
  - `kernel_dim.x = core_per_cluster`: 每个集群的核心数
  - `kernel_dim.y = cluster_count`: 集群数量
  - `kernel_type = CNRT_FUNC_TYPE_UNION1`: Union kernel 类型
- **同步**: 调用 `cnrtQueueSync(queue)` 等待完成

```cpp
infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y, const void *x, void *stream) const;
```
- **功能**: 公共 API，根据 `_info.dtype` 分发到 `causalSoftmaxUnion<T>`
- **支持类型**: F16 (half), BF16 (bfloat16_t), F32 (float)
- **错误处理**: 不支持的类型返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`

## 4. 使用示例

```cpp
// 1. 创建句柄和描述符
infiniopHandle_t handle;
infiniopCreateHandle(&handle, device_id);

infiniopTensorDescriptor_t x_desc, y_desc;
// 假设 shape = {batch_size, seq_len, total_seq_len}
infiniopCreateTensorDescriptor(&x_desc, INFINI_DTYPE_F16, ndim, shape, strides);
infiniopCreateTensorDescriptor(&y_desc, INFINI_DTYPE_F16, ndim, shape, strides);

// 2. 创建因果 softmax 描述符
op::causal_softmax::bang::Descriptor *softmax_desc;
auto status = op::causal_softmax::bang::Descriptor::create(
    handle, &softmax_desc, y_desc, x_desc);

// 3. 分配工作空间 (当前为 0)
size_t workspace_size = softmax_desc->workspaceSize();
void *workspace = nullptr;
if (workspace_size > 0) {
    cnrtMalloc(&workspace, workspace_size);
}

// 4. 执行计算
void *d_x, *d_y;
cnrtMalloc(&d_x, tensor_size);
cnrtMalloc(&d_y, tensor_size);
// ... 拷贝输入数据到 d_x ...

cnrtQueue_t queue;
cnrtQueueCreate(&queue);
softmax_desc->calculate(workspace, workspace_size, d_y, d_x, queue);

// 5. 同步和清理
cnrtQueueSync(queue);
// ... 使用 d_y 数据 ...

cnrtFree(d_x);
cnrtFree(d_y);
if (workspace) cnrtFree(workspace);
delete softmax_desc;
```

## 5. 实现细节

### 内存管理
- **NRAM 分配策略**: 全局静态 buffer `__nram__ char nram_buffer[NRAM_MAX_SIZE]` (240KB)
- **分区方案**:
  - `SRC_MAX_SIZE = NRAM_MAX_SIZE / 4` (60KB)
  - `float *float_buffer`: 前半部分用于浮点计算
  - `T *temp_buffer`: 后半部分用于 F16/BF16 临时存储
  - F16/BF16: `chunk_size = SRC_MAX_SIZE / (2 * sizeof(float))` ≈ 7.5K 元素
  - F32: `chunk_size = SRC_MAX_SIZE / sizeof(float)` ≈ 15K 元素
- **分块处理**: 对超过 chunk_size 的输入分批迭代，每批处理 `min(chunk_size, remaining)` 元素

### 并发与并行
- **任务并行**: 2D 网格 (core_per_cluster × cluster_count) 并行处理 (batch, seq_len) 对
- **负载均衡**: 静态分块，每个核心分配连续的任务范围，避免动态调度开销
- **线程局部**: 每个 MLU 核心独立的 NRAM buffer，无需同步
- **无竞争**: 不同核心写入 `y` 的不同行，无数据竞争

### 性能优化
- **类型转换开销**: F16/BF16 仅在 NRAM 中转换为 float 计算，避免 GDRAM 往返
- **向量化指令**:
  - `__bang_active_exphp`: 高性能指数计算
  - `__bang_sumpool` / `__bang_maxpool`: 向量化规约
  - `__bang_half2float` / `__bang_bfloat162float`: 批量类型转换
- **2D Memcpy**: 利用 `__memcpy` 的 stride 参数实现非连续内存的 gather/scatter
- **数值稳定性**: 先减最大值再计算指数，避免 exp 溢出
- **对齐访问**: 128 字节对齐 (`ALIGN_SIZE`)，优化内存带宽

### 错误处理
- **张量验证**: 在 `CausalSoftmaxInfo::create` 中检查维度和数据类型
- **步长支持**: 支持任意步长布局 (非连续张量)
- **类型不匹配**: 输入输出 dtype 不一致返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
- **形状检查**: 拒绝 `total_seq_len < seq_len` 的无效形状

### 依赖关系
- **CNNL 驱动**: 依赖 `cnrt.h` 和 `cnnl.h` 的 MLU 运行时接口
- **设备抽象**: `device::bang::Handle::Internal` 提供 core_per_cluster 和 cluster_count
- **规约原语**: `op::common_bang::reduce_op` 命名空间提供 `maxBatched` 和 `sumBatched` 函数
  - `maxBatched`: 使用 `__bang_maxpool` + `__bang_argmax` 计算最大值
  - `sumBatched`: 使用 `__bang_sumpool` + `__bang_reduce_sum` 计算和
  - 对小向量 (<32 元素) 回退到标量循环，避免向量化开销

### 设计模式
- **模板元编程**: 使用 `if constexpr` 实现编译期类型分发，避免运行时分支
- **CRTP (奇异递归模板模式)**: `DESCRIPTOR(bang)` 宏定义命名空间专用类
- **Pimpl 惯例**: `struct Opaque` 隐藏设备句柄实现细节
- **策略模式**: 通过 `is_exp_phase` 参数统一两个计算阶段的接口
- **RAII**: `Descriptor` 管理句柄生命周期，析构时自动释放资源

### 算法复杂度
- **时间复杂度**: O(batch_size × seq_len × total_seq_len)
- **空间复杂度**: O(NRAM_MAX_SIZE) ≈ 240KB 片上内存，O(1) 额外 GDRAM
- **内存访问**: 每个元素读取 2 次 (x + output)，写入 1 次 (output)
