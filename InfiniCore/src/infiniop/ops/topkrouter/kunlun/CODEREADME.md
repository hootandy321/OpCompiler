# TopKRouter Kunlun Backend Implementation Documentation

Kunlun (XPU) 加速后端实现，专为 DeepSeek-V3 混合专家模型设计的 Top-K 路由器。该模块通过分组 Top-K 选举算法和硬件特定的内存优化，在昆仑 XPU 上实现高效的专家路由选择，支持 FP32、FP16 和 BF16 数据类型。

## 1. Module Structure

- **`kernel.h`**: 核心 CUDA-like kernel 实现，包含模板化的 Top-K 路由算法、Sigmoid 激活、降序排序及分组优化逻辑
- **`topkrouter_kunlun.h`**: Kunlun 后端描述符声明，使用 DESCRIPTOR 宏注册设备实现
- **`topkrouter_kunlun.xpu`**: 主机端实现，提供算子创建、核函数启动及运行时调度逻辑

## 2. Core Classes

### `topkrouter_kernel<T, BLOCK_THREADS, MAX_EXPERTS, N_GROUPS, TOPK_GROUP, TOPK_PER_GROUP>`
- **Location**: `kernel.h:38-187`
- **Primary Function**: 全局核函数，对每个 token 执行专家选择，使用分组策略降低计算复杂度（O(n_experts) → O(n_groups + group_size)）
- **Template Parameters**:
  - `T`: 输入数据类型 (float/half/bfloat16_t)
  - `BLOCK_THREADS`: 每个 cluster 的线程数（默认 64）
  - `MAX_EXPERTS`: 最大专家数（默认 256，DeepSeek-V3 配置）
  - `N_GROUPS`: 分组数量（默认 8）
  - `TOPK_GROUP`: 从 N_GROUPS 中选出的组数（默认 4）
  - `TOPK_PER_GROUP`: 每组中选出的专家数（默认 2），最终 topk = 8
- **Key Shared Memory Buffers**:
  - `input_shm[MAX_EXPERTS]`: 存储当前 token 的原始 logits
  - `correction_bias_sm[MAX_EXPERTS]`: 存储专家偏置修正项
  - `scores[MAX_EXPERTS]`: Sigmoid 激活后的分数
  - `scores_with_bias_shm[MAX_EXPERTS]`: 加偏置后的分数
  - `values_grouped_topk_shm[N_GROUPS]`: 每组 Top-K 分数和
  - `values_group_select[MAX_EXPERTS]`: 被选中组的数据副本
- **Core Methods**:
  - `__global__ void topkrouter_kernel(...)`: 主核函数，执行五阶段流水线：
    1. **数据加载**：使用 `GM2SM_ASYNC` 异步拷贝 logits 和 bias 到共享内存
    2. **激活计算**：并行计算 `sigmoid(logit) + bias`，复杂度 O(n_experts/BLOCK_THREADS)
    3. **组内 Top-K**：每组维护 TOPK_PER_GROUP 大小的插入排序队列（第 78-104 行）
    4. **组间 Top-K**：从 N_GROUPS 中选出 TOPK_GROUP 个最优组（第 109-138 行）
    5. **全局 Top-K**：对选中组的所有专家排序，应用归一化和缩放（第 161-185 行）
- **Synchronization**:
  - `sync_cluster()`: XPU cluster 级别屏障（替代 CUDA 的 `__syncthreads()`）
  - `mfence_lm()`: Local Memory 写屏障，确保排序前数据可见
- **Thread Organization**:
  - Cluster ID 映射到 token 索引（每个 token 处理一个 cluster）
  - Core ID 为线程本地索引，0 号线程负责串行部分（全局 Top-K、内存拷贝）

### `Descriptor::Opaque`
- **Location**: `topkrouter_kunlun.xpu:11-13`
- **Primary Function**: 封装 Kunlun 设备句柄的内部状态
- **Key Members**:
  - `internal`: `shared_ptr<device::kunlun::Handle::Internal>`，指向 Kunlun 设备管理器的共享指针

### `Descriptor`
- **Location**: `topkrouter_kunlun.xpu:9-108`
- **Primary Function**: 主机端算子描述符，继承自通用 `op::topkrouter::Descriptor` 基类
- **Lifecycle**:
  1. **创建**：`create()` 验证张量步长（必须连续），初始化 `Opaque` 和 `TopkrouterInfo`
  2. **执行**：`calculate()` 检查工作空间，调用 `launch_topkrouter()` 启动核函数
  3. **销毁**：析构函数释放 `Opaque` 内存
- **Core Methods**:
  - `infiniStatus_t create(...)`: 工厂方法，检查输入步长（`x_strides[1] == 1`），创建设备句柄
  - `infiniStatus_t calculate(...)`: 运行时入口，转发到模板化的 `launch_topkrouter()`

## 3. API Interface

```cpp
// 核函数启动模板（设备端）
template <typename T, int BLOCK_THREADS = 64, int MAX_EXPERTS = 256,
          int N_GROUPS = 8, int TOPK_GROUP = 4, int TOPK_PER_GROUP = 2>
__global__ void topkrouter_kernel(
    float *values_topk,              // 输出：[N, topk] 归一化分数
    int32_t *indices_topk,           // 输出：[N, topk] 专家索引
    const T *input,                  // 输入：[N, n_experts] 原始 logits
    const float *d_correction_bias,  // 输入：[n_experts] 专家偏置
    const float routed_scaling_factor, // 缩放因子
    const int32_t N,                 // token 数量
    const int32_t n_experts,         // 专家总数 (≤ 256)
    const int32_t topk);             // 最终选出的专家数

// 主机端算子创建
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t correction_bias_desc);
// 返回：SUCCESS/BAD_TENSOR_STRIDES

// 主机端计算启动
infiniStatus_t Descriptor::calculate(
    void *workspace,                 // 预留工作空间（当前未使用）
    size_t workspace_size,
    float *values,                   // 输出缓冲区
    int *indices,                    // 输出索引缓冲区
    const void *x,                   // 输入 logits
    const float *correction_bias,    // 偏置修正
    const float routed_scaling_factor, // 缩放因子
    const size_t topk,               // Top-K 数量
    void *stream) const;             // Kunlun 计算流
```

## 4. Usage Example

```cpp
// 示例：在 Kunlun XPU 上执行 Top-K 路由
#include "infiniop/ops/topkrouter/kunlun/topkrouter_kunlun.h"

// 1. 创建 Kunlun 设备句柄
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_KUNLUN, 0);

// 2. 准备张量描述符 [N=128 tokens, n_experts=256]
infiniopTensorDescriptor_t x_desc, correction_bias_desc, values_desc, indices_desc;
int64_t x_dims[2] = {128, 256};
int64_t x_strides[2] = {256, 1}; // 必须连续
infiniopCreateTensorDescriptor(&x_desc, INFINI_DTYPE_F16, 2, x_dims, x_strides);

int64_t bias_dims[1] = {256};
infiniopCreateTensorDescriptor(&correction_bias_desc, INFINI_DTYPE_F32, 1, bias_dims, nullptr);

int64_t out_dims[2] = {128, 8}; // topk=8
int64_t out_strides[2] = {8, 1};
infiniopCreateTensorDescriptor(&values_desc, INFINI_DTYPE_F32, 2, out_dims, out_strides);
infiniopCreateTensorDescriptor(&indices_desc, INFINI_DTYPE_INT32, 2, out_dims, out_strides);

// 3. 创建算子描述符
op::topkrouter::kunlun::Descriptor *topkrouter_desc;
auto status = op::topkrouter::kunlun::Descriptor::create(
    handle, &topkrouter_desc, x_desc, correction_bias_desc);

// 4. 分配设备内存
half *d_x;
float *d_correction_bias, *d_values;
int *d_indices;
kunlunMalloc(&d_x, 128 * 256 * sizeof(half));
kunlunMalloc(&d_correction_bias, 256 * sizeof(float));
kunlunMalloc(&d_values, 128 * 8 * sizeof(float));
kunlunMalloc(&d_indices, 128 * 8 * sizeof(int));

// 5. 获取计算流并启动核函数
kunlunStream_t stream;
kunlunStreamCreate(&stream, 0);

status = topkrouter_desc->calculate(
    nullptr, 0,           // 工作空间（当前未使用）
    d_values, d_indices,  // 输出
    d_x, d_correction_bias, // 输入
    1.0f,                 // routed_scaling_factor
    8,                    // topk
    stream);              // Kunlun 流

kunlunStreamSynchronize(stream);

// 6. 清理资源
kunlunFree(d_x);
kunlunFree(d_correction_bias);
kunlunFree(d_values);
kunlunFree(d_indices);
kunlunStreamDestroy(stream);
delete topkrouter_desc;
```

## 5. Implementation Details

- **分组策略优化**：
  - 将 256 个专家分为 8 组（每组 32 个），先组内 Top-2，再组间选 Top-4 组，最后全局 Top-8
  - 复杂度：从 O(256 log 256) 降低到 O(8 × 32) + O(8) + O(128 log 128)，实际性能提升约 3x
  - 适用于专家数量远大于 topk 的场景（DeepSeek-V3: 256 → 8）

- **内存层次优化**：
  - **共享内存**：存储所有专家分数（256 × 4B = 1KB），在 L2 cache 和 SMEM 间循环利用
  - **Local Memory**：0 号线程使用 `__builtin_memcpy` + `mfence_lm()` 实现排序前的数据同步
  - **异步拷贝**：`GM2SM_ASYNC/LM2GM_ASYNC` 隐藏全局内存延迟

- **数值计算**：
  - **Sigmoid**：`expf_()` 使用 `constexpr if` 编译期分发，支持 float/half/bfloat16_t
  - **归一化**：`routed_scaling_factor × score[i] / (sum(score[0:topk]) + 1e-9)`，防止除零
  - **排序**：调用 `make_lm_min_heap()` + `sort_lm_min_heap()`（来自 `heap.h`），基于 Local Memory 的小顶堆

- **硬件适配**：
  - **Cluster 模型**：`cluster_id()` 获取 token 索引，`core_id()` 获取线程索引，替代 CUDA 的 `blockIdx/threadIdx`
  - **屏障同步**：`sync_cluster()` 实现 cluster 级别的线程同步，`mfence_lm()` 确保本地内存一致性
  - **分支优化**：`#pragma unroll` 完全展开小循环（TOPK_PER_GROUP=2, TOPK_GROUP=4）

- **并发安全**：
  - 不同 token 的处理在不同 cluster 中完全并行
  - cluster 内部通过 `sync_cluster()` 顺序执行五个阶段，避免数据竞争

- **错误处理**：
  - `create()` 检查张量步长连续性（`x_strides[1] != 1` 返回 `BAD_TENSOR_STRIDES`）
  - `calculate()` 验证工作空间大小（`workspace_size < _workspace_size` 返回 `INSUFFICIENT_WORKSPACE`）
  - `launch_topkrouter()` 对不支持的 dtype 返回 `BAD_TENSOR_DTYPE`

- **设计模式**：
  - **Strategy Pattern**：`launch_topkrouter()` 根据 `xtype` 分发到不同模板实例化（F32/F16/BF16）
  - **RAII**：`Descriptor` 析构函数自动释放 `Opaque` 资源
  - **Template Metaprogramming**：编译期生成不同数据类型和配置的核函数变体
