# TopKRouter MetaX 后端实现文档

TopKRouter MetaX 后端模块实现了基于华为昇腾 MetaX 设备的 Top-K 专家路由算子，用于混合专家模型（MoE）中的动态专家选择。该模块通过层级化排序策略从 256 个专家中选择 Top-K 个专家，并对权重进行归一化处理。

## 1. 模块结构

- **`topkrouter_metax.h`**: MetaX 后端描述符声明文件，定义 DESCRIPTOR 宏以生成 op::topkrouter::metax::Descriptor 类
- **`topkrouter_metax.maca`**: MetaX 后端核心实现，包含描述符创建、CUDA Kernel 启动逻辑及 TopK 计算调度

## 2. 核心类

### `op::topkrouter::metax::Descriptor`
- **位置**: `topkrouter_metax.h` (通过 DESCRIPTOR 宏定义) 和 `topkrouter_metax.maca`
- **主要功能**: 封装 MetaX 设备上的 TopKRouter 算子描述符，管理设备句柄、张量信息和执行配置
- **关键成员**:
  - `Opaque *_opaque`: 内部不透明指针，持有 `std::shared_ptr<device::metax::Handle::Internal>` 设备内部句柄
  - `TopkrouterInfo _info`: 存储输入张量元信息（数据类型、形状、步长）
  - `size_t _workspace_size`: 工作空间大小（当前实现为 0）
- **核心方法**:
  - `create(handle, desc_ptr, x_desc, correction_bias_desc)`: 构建描述符实例，验证输入张量步长是否连续（`x_strides[1] == 1`），初始化设备句柄和元信息，返回 `INFINI_STATUS_SUCCESS` 或错误码
  - `calculate(workspace, workspace_size, values, indices, x, correction_bias, routed_scaling_factor, topk, stream)`: 执行 TopK 路由计算，启动 CUDA kernel，仅支持 `width == 256` 的场景，否则返回 `INFINI_STATUS_BAD_PARAM`
  - `~Descriptor()`: 析构函数，释放 `_opaque` 指针
- **生命周期**: 由用户通过 `create()` 静态方法创建，计算完成后由用户负责销毁（调用 `delete`）

### `Descriptor::Opaque`
- **位置**: `topkrouter_metax.maca` (line 17-19)
- **主要功能**: 封装 MetaX 设备内部句柄，使用共享指针管理生命周期
- **关键成员**:
  - `std::shared_ptr<device::metax::Handle::Internal> internal`: MetaX 设备句柄的内部实现

## 3. API 接口

```cpp
namespace op::topkrouter::metax {

// 创建 TopKRouter MetaX 描述符
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,                          // [输入] MetaX 设备句柄
    Descriptor **desc_ptr,                            // [输出] 描述符指针
    infiniopTensorDescriptor_t x_desc,                // [输入] 输入张量描述符 [N, 256]
    infiniopTensorDescriptor_t correction_bias_desc   // [输入] 校正偏置张量描述符 [256]
);
// 返回: INFINI_STATUS_SUCCESS (成功), INFINI_STATUS_BAD_TENSOR_STRIDES (步长不连续)

// 执行 TopK 路由计算
infiniStatus_t Descriptor::calculate(
    void *workspace,              // [输入] 工作空间指针（当前未使用）
    size_t workspace_size,        // [输入] 工作空间大小
    float *values,                // [输出] Top-K 权重值 [N, topk]
    int *indices,                 // [输出] Top-K 专家索引 [N, topk]
    const void *x,                // [输入] 输入 logits [N, 256]
    const float *correction_bias, // [输入] 校正偏置 [256]
    const float routed_scaling_factor, // [输入] 路由缩放因子
    const size_t topk,            // [输入] 选择的专家数量 K
    void *stream                  // [输入] CUDA 流
) const;
// 返回: INFINI_STATUS_SUCCESS (成功), INFINI_STATUS_INSUFFICIENT_WORKSPACE (工作空间不足),
//       INFINI_STATUS_BAD_PARAM (width != 256)

}
```

## 4. 使用示例

```cpp
// 示例: 使用 TopKRouter MetaX 后端选择 Top-4 专家
#include "infiniop/ops/topkrouter/metax/topkrouter_metax.h"

// 1. 准备输入张量描述符 [batch_size, 256]
std::vector<size_t> x_shape = {128, 256};  // 128 个 token, 256 个专家
std::vector<ptrdiff_t> x_strides = {256, 1}; // 连续内存布局
infiniopTensorDescriptor_t x_desc = createTensorDesc(x_shape, x_strides, INFINI_DTYPE_F16);

// 2. 准备校正偏置描述符 [256]
std::vector<size_t> bias_shape = {256};
infiniopTensorDescriptor_t bias_desc = createTensorDesc(bias_shape, INFINI_DTYPE_F32);

// 3. 创建 TopKRouter 描述符
op::topkrouter::metax::Descriptor* topk_desc;
infiniStatus_t status = op::topkrouter::metax::Descriptor::create(
    metax_handle, &topk_desc, x_desc, bias_desc);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 4. 分配输出缓冲区 [128, 4]
float* d_values = allocateDeviceMemory<float>(128 * 4);
int* d_indices = allocateDeviceMemory<int>(128 * 4);

// 5. 执行 TopK 路由计算
const float kScalingFactor = 1.0f;
const size_t kTopK = 4;
status = topk_desc->calculate(
    nullptr, 0,              // 无需工作空间
    d_values, d_indices,     // 输出缓冲区
    d_input_logits,          // 输入 logits (FP16)
    d_correction_bias,       // 校正偏置
    kScalingFactor,          // 缩放因子
    kTopK,                   // 选择 Top-4
    cuda_stream              // CUDA 流
);

// 6. 清理资源
delete topk_desc;
freeDeviceMemory(d_values);
freeDeviceMemory(d_indices);
```

## 5. 实现细节

### 算法流程
TopKRouter 采用**层级化 Warp 级排序策略**，通过 CUB 原语在 CUDA Block 内进行高效的 Top-K 选择：

1. **Sigmoid 激活与偏置校正** (line 67-72):
   - 对每个专家的 logit 执行 `sigmoid(x) = 1 / (1 + exp(-x))`
   - 加上校正偏置 `value += correction_bias[tid]`

2. **Warp 级排序** (line 77-86):
   - 将 256 个线程分为 8 个 Warp（每组 32 线程）
   - 使用 `cub::WarpMergeSort` 对每个 Warp 内的 32 个值降序排序
   - 每个 Warp 保留最大的两个值

3. **组间 Top-4 选择** (line 90-115):
   - 计算每个 Warp 的最大两个值之和（共 8 个组值）
   - 第 0 个 Warp 使用 `WarpMergeSort` 对这 8 个组值排序
   - 标记 Top-4 组对应的 `share_data_group_mask[indices] = 1.0f`

4. **最终 Top-K 提取** (line 120-128):
   - 使用掩码屏蔽非 Top-4 组的所有值 `value *= mask`
   - 使用 `cub::BlockRadixSort` 对 Block 内 256 个值降序排序
   - 提取前 K 个值

5. **Softmax 归一化** (line 133-154):
   - 对 Top-K 值求和 `sum = Σ(values[i]) + 1e-9f`
   - 归一化并应用缩放因子 `output[i] = routed_scaling_factor * value[i] / sum`

### 内存管理
- **零拷贝设计**: 所有计算在设备端完成，无需主机端内存拷贝
- **共享内存优化**: 使用 `__shared__` 内存存储中间排序结果，减少全局内存访问
  - `share_data[256]`: 存储 Warp 级排序后的所有值
  - `share_data_group[8]`: 存储 8 个 Warp 的组累加和
  - `share_data_group_mask[8]`: 存储 Top-4 组的掩码
  - `share_sum`: 存储 Softmax 归一化分母

### 并发策略
- **Block 并行**: 每个样本 (token) 分配一个 Block，`dim3 blocks(N)`
- **Warp 级协作**: 使用 CUB Warp 级原语（`WarpMergeSort`, `WarpReduce`）实现低延迟排序
- **Block 级同步**: 通过 `__syncthreads()` 和 `__syncwarp()` 保证数据一致性

### 性能特性
- **固定配置约束**: 当前实现仅支持 `width = 256` (固定专家数量) 和 `BLOCK_SIZE = 256` (固定 Block 大小)
- **数据类型支持**: FP32 (float), FP16 (half), BF16 (cuda_bfloat16)
- **时间复杂度**:
  - Warp 级排序: O(32 log 32) per Warp
  - 组间排序: O(8 log 8)
  - Block 级排序: O(256 log 256)
  - 总体: O(N * 256 log 256) where N 为 token 数量

### 错误处理
- **输入验证**:
  - 检查张量步长连续性 `x_strides[1] == 1`，返回 `INFINI_STATUS_BAD_TENSOR_STRIDES`
  - 检查数据类型是否为 FP32/FP16/BF16，返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
  - 检查 width 是否为 256，返回 `INFINI_STATUS_BAD_PARAM`
  - 检查工作空间大小，返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`

### 依赖关系
- **设备层**: `device::metax::Handle` (MetaX 设备句柄)
- **公共算子**: `op::topkrouter::TopkrouterInfo` (TopK 路由元信息)
- **CUDA 库**: CUB (Block/Warp 级排序和归约原语)
- **Kernel 实现**: `topkrouter_kernel<T, BLOCK_THREADS>` (定义在 `../cuda/kernel.cuh`)

### 设计模式
- **工厂模式**: `create()` 静态方法负责对象构造和验证
- **RAII**: 使用 `std::shared_ptr` 管理设备句柄生命周期
- **策略模式**: 通过模板特化支持多种数据类型 (FP32/FP16/BF16)
- **宏驱动生成**: `DESCRIPTOR(metax)` 宏生成完整的 Descriptor 类定义
