# Kunlun Random Sample 核心实现文档

该模块实现了基于昆仑（XPU）设备的随机采样算子，支持 Top-K 采样、核采样（Top-p/Nucleus）和温度缩放，用于大语言模型推理中的 token 生成。该实现针对昆仑硬件架构进行了深度优化，采用了多层次并行策略和混合精度计算。

## 1. 模块结构

- **`kernel.h`**: 设备端核心算子实现，包含 Top-K 选择、softmax 归一化、采样算法及相关的 CUDA-like kernel 函数
- **`random_sample_kunlun.h`**: 昆仑后端的算子描述符声明，包含 DESCRIPTOR 宏定义
- **`random_sample_kunlun.xpu`**: 主机端接口实现，包含算子描述符的生命周期管理、workspace 计算和 kernel 启动逻辑

## 2. 核心类与函数

### 设备端核心函数 (kernel.h)

#### `swap<Tval>()`
- **位置**: kernel.h:10-14
- **功能**: 交换两个本地内存中的值
- **实现**: 模板函数，使用临时变量进行原地交换
- **使用场景**: Top-K 排序算法中的值交换

#### `findTopk<Tval, Tidx>()`
- **位置**: kernel.h:17-57
- **功能**: 在全局内存中查找 Top-K 个最大值及其索引
- **算法**: 选择排序变种，时间复杂度 O(topk * size)
- **内存模式**: 使用 GM2LM/LM2GM 宏进行全局内存与本地内存的数据传输
- **支持类型**: float, half (FP16), bfloat16_t (BF16)
- **比较策略**:
  - float: 直接比较
  - half/bfloat16_t: 转换为 float 后比较
- **输入**: values 数组、indices 数组、数组大小、k 值
- **副作用**: 原地修改 values 和 indices 数组，前 k 个位置存储最大值

#### `findTopkLocal<Tval, Tidx>()`
- **位置**: kernel.h:60-87
- **功能**: 在本地内存（__local__）中查找 Top-K
- **算法**: 同样使用选择排序，但操作本地内存
- **优化**: 避免频繁的全局内存访问
- **使用场景**: 当 buf_size > step_easy 时，本地内存可容纳全部数据

#### `findTopOne<Tval, Tidx>()`
- **位置**: kernel.h:90-123
- **功能**: 在全局内存中查找单个最大值（argmax 的核心）
- **算法**: 单次线性扫描，O(size) 时间复杂度
- **初始值**: 使用 -INFINITY 作为初始最大值
- **输出**: 最大值存储在 values[0]，对应索引存储在 indices[0]

#### `findTopOneLocal<Tval, Tidx>()`
- **位置**: kernel.h:126-154
- **功能**: 在本地内存中查找单个最大值
- **使用场景**: TopOneKernel 的本地内存优化版本

#### `TopkKernel<Tval, Tidx>()`
- **位置**: kernel.h:156-250
- **功能**: 多级并行的 Top-K 计算主函数
- **并行策略**:
  - 三层并行：Cluster × Core × 内存分块
  - 负载均衡: 使用"前 remain 个线程多分配 1 个元素"策略
- **参数**:
  - `voc`: 词汇表大小（输入数组长度）
  - `topk`: 需要选择的 Top-K 数量
  - `buf_size`: 本地缓冲区大小（固定 128）
- **执行路径**:
  1. **topk >= step_easy**: 仅 core_id=0 的线程执行全局 findTopk
  2. **buf_size > step_easy**: 本地内存一次性加载全部数据，调用 findTopkLocal
  3. **topk > buf_size**: 全局内存 findTopk，分块传输结果
  4. **buf_size ≥ topk**: 滑动窗口算法，每次处理 buf_size 个元素，保留前 topk
- **输出**: values_global[0:topk] 存储 Top-K 值，indices_global[0:topk] 存储对应索引

#### `softmaxSum<CLUSTER_SIZE, BLOCK_SIZE, Tval, Tcompute>()`
- **位置**: kernel.h:253-326
- **功能**: 计算 softmax 的归一化因子 Σ exp((x - max) / temperature)
- **并行策略**:
  - **Cluster 级**: 每个 cluster 处理一块共享内存（SM_SIZE）大小的数据
  - **Core 级**: cluster 内使用 BLOCK_SIZE 个 core 并行处理
- **内存层次**:
  - 全局内存 → 共享内存（GM2SM）→ 私有寄存器计算
  - 使用 reduce_op::sum 进行 warp 级规约
- **数学稳定性**: 使用 max-value 技术避免 exp 溢出
- **支持精度**: Tval（输入类型）和 Tcompute（计算类型，通常为 float）分离
- **归约步骤**:
  1. 每个 cluster 计算局部和
  2. 存储 cluster 结果到 sum_global
  3. Core 0 读取所有 cluster 结果
  4. 再次归约得到全局和 all_sum

#### `sample<Tval, Tcompute, Tidx>()`
- **位置**: kernel.h:328-386
- **功能**: 基于 Top-p (Nucleus) 采样策略从 Top-K 候选中选择一个 token
- **算法**:
  1. **Top-p 截断**: 从高到低累加概率，直到超过 topp 阈值，确定 end 索引
  2. **随机采样**: 将 random_val 缩放到 [0, cumsum)，再次累加概率直到超过 random_val
- **执行线程**: 仅 thread_id=0 执行（串行逻辑）
- **内存访问**: 分块从 values_global 读取数据（buf_size=128）
- **输出**: result[0] 存储被选中的 token 索引

#### `randomSampleKernel<CLUSTER_SIZE, BLOCK_SIZE, Tval, Tcompute, Tidx>()`
- **位置**: kernel.h:388-431
- **功能**: 随机采样的全局入口函数（__global__ kernel）
- **执行流程**:
  1. 调用 TopkKernel 获取 Top-K 候选
  2. 同步所有 cluster（sync_cluster）
  3. 提取最大值 max_value = values_global[0]
  4. 调用 softmaxSum 计算归一化因子
  5. 调用 sample 执行 Top-p 采样
- **本地内存分配**: `values_local[256]` 和 `indices_local[256]`（2 * buf_size）
- **配置**: CLUSTER_SIZE=8, BLOCK_SIZE=64

#### `TopOneKernel<Tval, Tidx>()`
- **位置**: kernel.h:433-495
- **功能**: Argmax 的核心实现（查找单个最大值）
- **并行策略**: 与 TopkKernel 类似，但每个线程只需找到局部最大值
- **路径选择**:
  - **buf_size > step_easy**: 本地内存加载全部数据，findTopOneLocal
  - **buf_size ≤ step_easy**: 滑动窗口算法，每次保留当前最大值
- **最终归约**: Thread 0 调用 findTopOne 从 nthreads 个结果中选择全局最大值
- **输出**: result[0] = argmax 索引

#### `argmaxKernel<Tval, Tidx>()`
- **位置**: kernel.h:497-514
- **功能**: Greedy decoding（贪婪解码）的全局入口
- **用途**: 当 temperature=0 或不满足采样条件时，直接选择概率最大的 token
- **实现**: 封装 TopOneKernel

### 主机端接口 (random_sample_kunlun.xpu)

#### `launchKernel<Tval, Tidx>()`
- **位置**: random_sample_kunlun.xpu:8-56
- **功能**: 启动设备 kernel 的模板函数
- **参数**:
  - workspace: 预分配的设备内存缓冲区
  - result: 输出 token 索引（设备指针）
  - probs: 输入概率分布（设备指针）
  - random_val, topp, topk, temperature: 采样参数
  - n: 词汇表大小
  - stream: 昆仑流
- **Workspace 布局**:
  ```
  [0]: values[n]                    // 概率值副本
  [n]: values_global[cluster_num * core_num * topk]  // Top-K 值
  [n + cluster_num * core_num * topk]: sum_global[cluster_num]  // softmax 和
  [偏移后]: indices[n]              // 索引数组
  [n]: indices_global[...]          // Top-K 索引
  ```
- **采样条件判断**:
  ```cpp
  dosample = (topk_ > 1) && (temperature != 0.0f) &&
             (topp != 0.0f) && (random_val != 0.0f);
  ```
- **Kernel 选择**:
  - dosample=true: randomSampleKernel
  - dosample=false: argmaxKernel（贪婪解码）

#### `Descriptor::Opaque`
- **位置**: random_sample_kunlun.xpu:63-65
- **功能**: 不透明指针，封装昆仑设备句柄的内部实现
- **成员**: `std::shared_ptr<device::kunlun::Handle::Internal> internal`
- **生命周期**: 由 Descriptor 管理，使用 shared_ptr 共享所有权

#### `Descriptor::~Descriptor()`
- **位置**: random_sample_kunlun.xpu:67-69
- **功能**: 析构函数，释放 Opaque 资源

#### `Descriptor::create()`
- **位置**: random_sample_kunlun.xpu:71-94
- **功能**: 算子描述符工厂函数
- **参数**:
  - handle_: 全局昆仑句柄
  - result_desc: 输出张量描述符（标量，索引类型）
  - probs_desc: 输入张量描述符（1D 张量，概率类型）
- **验证**: 调用 RandomSampleInfo::create 检查张量有效性
- **Workspace 计算**:
  ```cpp
  size_t workspace_size =
      (n + cluster_num * core_num * n) *
      (sizeof(Tval) + sizeof(Tidx)) +
      cluster_num * sizeof(float);
  ```
  - 最坏情况: topk=n（全量排序）
- **配置**: cluster_num=8, core_num=64
- **返回**: 新分配的 Descriptor 指针

#### `Descriptor::minWorkspaceSize()`
- **位置**: random_sample_kunlun.xpu:96-98
- **功能**: 返回所需的最小 workspace 大小

#### `Descriptor::calculate()`
- **位置**: random_sample_kunlun.xpu:100-149
- **功能**: 执行随机采样计算的主入口
- **参数验证**: 检查 workspace_size 是否足够
- **类型分发**:
  - 索引类型: int32_t 或 int64_t
  - 概率类型: FP16, BF16, FP32
- **实现**: 使用 LAUNCH_KERNEL 宏实例化并调用 launchKernel
- **错误处理**: 不支持的 dtype 返回 INFINI_STATUS_BAD_TENSOR_DTYPE

## 3. API 接口

```cpp
// 算子描述符创建
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,              // 昆仑设备句柄
    Descriptor **desc_ptr,                // 输出：描述符指针
    infiniTensorDescriptor_t result_desc, // 输出张量（索引，标量）
    infiniTensorDescriptor_t probs_desc   // 输入张量（概率，1D向量）
);

// 查询 workspace 大小
size_t Descriptor::minWorkspaceSize() const;

// 执行计算
infiniStatus_t Descriptor::calculate(
    void *workspace,          // 设备内存缓冲区
    size_t workspace_size,    // 缓冲区大小
    void *result,             // 输出：选中的 token 索引（设备指针）
    const void *probs,        // 输入：概率分布（设备指针）
    float random_val,         // 随机数 [0, 1)
    float topp,               // Top-p 阈值（核采样）
    int topk,                 // Top-K 数量
    float temperature,        // 温度参数
    void *stream              // 昆仑流
) const;
```

## 4. 使用示例

```cpp
#include "infiniop/ops/random_sample/kunlun/random_sample_kunlun.h"

// 1. 创建算子描述符
infiniopHandle_t handle;
infiniopCreateHandle(&handle, device_id);

infiniTensorDescriptor_t probs_desc, result_desc;
// ... 初始化张量描述符
// probs_desc: dtype=FP16, shape=[vocab_size]
// result_desc: dtype=I32, shape=[1]

op::random_sample::kunlun::Descriptor *sample_desc;
auto status = op::random_sample::kunlun::Descriptor::create(
    handle, &sample_desc, result_desc, probs_desc
);

// 2. 分配 workspace
size_t workspace_size = sample_desc->minWorkspaceSize();
void *workspace;
xpu_malloc(&workspace, workspace_size);

// 3. 准备输入数据（设备端）
void *d_probs;
xpu_malloc(&d_probs, vocab_size * sizeof(half));
// ... 复制概率数据到 d_probs

void *d_result;
xpu_malloc(&d_result, sizeof(int32_t));

// 4. 执行采样
XPUStream stream;
xpu_stream_create(&stream);

float random_val = 0.5f;  // 从随机数生成器获取
float topp = 0.9f;        // Top-p (nucleus) 采样
int topk = 50;            // 从 Top-50 候选中选择
float temperature = 0.8f; // 温度缩放

status = sample_desc->calculate(
    workspace, workspace_size,
    d_result, d_probs,
    random_val, topp, topk, temperature,
    stream
);

// 5. 同步并获取结果
xpu_stream_synchronize(stream);
int32_t token_id;
xpu_memcpy(&token_id, d_result, sizeof(int32_t), XPU_DEVICE_TO_HOST);

// 6. 清理资源
xpu_free(workspace);
xpu_free(d_probs);
xpu_free(d_result);
xpu_stream_destroy(stream);
delete sample_desc;
```

## 5. 实现细节

### 内存层次与优化策略

- **三级存储层次**:
  - **全局内存 (GM)**: 大容量，高延迟，存储输入/输出数据
  - **本地内存 (LM)**: 类似 CUDA 的 local memory，每个 thread 私有，低容量
  - **共享内存 (SM)**: Cluster 内共享，中等容量，用于 softmax 的中间结果
- **数据传输宏**:
  - `GM2LM/GM2SM`: 全局 → 本地/共享
  - `LM2GM/SM2GM`: 本地/共享 → 全局
  - `sizeof(Tval)`: 精确的字节长度传输

### 并行计算模型

- **硬件拓扑**: 8 Clusters × 64 Cores = 512 并行单元
- **负载均衡算法**:
  ```cpp
  int remain = voc % nthreads;
  int step_easy = (voc - remain) / nthreads;
  int step_hard = step_easy + 1;
  int step = (thread_id < remain ? step_hard : step_easy);
  ```
  保证线程间任务分配差距不超过 1 个元素
- **同步原语**:
  - `sync_cluster()`: Cluster 内同步（barrier）
  - `core_id()`, `cluster_id()`: 线程识别

### Top-K 算法优化

- **自适应路径选择**: 根据 topk、buf_size、step 的关系选择最优算法
  1. **topk ≥ step_easy**: 直接全局排序（避免冗余计算）
  2. **buf_size > step_easy**: 本地内存一次性加载（最小化全局访问）
  3. **topk > buf_size**: 全局排序 + 分块传输
  4. **buf_size ≥ topk**: 滑动窗口（流式处理，节省内存）
- **滑动窗口技巧**:
  ```cpp
  // 每次迭代保留前 topk
  values_local[i] = values_local[i - buf_size]; // 滚动窗口
  findTopkLocal(values_local, indices_local, buf_size + topk, topk);
  ```

### Softmax 数值稳定性

- **Max-value 减法**: 避免大指数导致溢出
  ```cpp
  exp((x_sm[index] - float(max_value)) / temperature)
  ```
- **类型提升**: FP16/BF16 输入在计算前转为 float
  ```cpp
  __half2float(x_sm[index])
  __bfloat162float(x_sm[index])
  ```

### Top-p (Nucleus) 采样实现

- **两阶段扫描**:
  1. **第一阶段**: 累加概率直到超过 topp，确定截断点 `end`
  2. **第二阶段**: 从 [0, end) 中根据随机值采样
- **概率归一化**: 除以 `all_sum` 确保概率和为 1
- **边界处理**: 如果 topk=1 或温度=0，退化为 argmax

### 混合精度支持

- **输入类型 (Tval)**: FP16, BF16, FP32
- **计算类型 (Tcompute)**: float（保证精度）
- **索引类型 (Tidx)**: int32_t, int64_t
- **类型分发策略**: 编译时模板实例化 + 运行时 switch-case

### Workspace 内存布局

```
偏移 0:                    values[n]
偏移 n:                    values_global[cluster_num * core_num * topk]
偏移 n + cluster_num * core_num * topk:     sum_global[cluster_num]
偏移 n + cluster_num * core_num * topk + cluster_num * sizeof(float):    indices[n]
偏移 2n + cluster_num * core_num * topk + cluster_num * sizeof(float):   indices_global[cluster_num * core_num * topk]
```

### 错误处理

- **Workspace 不足**: 返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
- **不支持的 dtype**: 返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
- **张量验证**: 委托给 `RandomSampleInfo::create` 进行检查

### 性能特征

- **Top-K 时间复杂度**:
  - 最坏情况: O(topk × voc)（当 topk 接近 voc）
  - 最佳情况: O(buf_size × voc / nthreads)（滑动窗口）
- **Softmax 时间复杂度**: O(voc / (CLUSTER_SIZE × BLOCK_SIZE))
- **采样时间复杂度**: O(topk)（串行，仅在 thread 0 执行）
- **内存访问模式**:
  - Top-K: 多次遍历（写入密集型）
  - Softmax: 单次遍历（读写平衡）
  - 采样: 两次遍历 Top-K 结果（读密集型）

### 设计模式

- **策略模式**: 根据 dosample 标志选择 randomSampleKernel 或 argmaxKernel
- **模板方法模式**: kernel.h 中的模板函数提供通用算法，类型特化处理精度差异
- **Pimpl 惯用法**: Descriptor::Opaque 隐藏设备句柄实现细节
- **RAII**: 使用 std::shared_ptr 管理 Handle::Internal 生命周期
