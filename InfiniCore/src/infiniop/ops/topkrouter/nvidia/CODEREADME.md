# TopKRouter NVIDIA CUDA 实现核心文档

TopKRouter NVIDIA 模块实现了基于 CUDA 的高性能专家路由（Expert Routing）算子，专门用于混合专家模型（MoE，Mixture of Experts）中的 Top-K 专家选择与路由决策。该模块通过层级化的并行排序策略，在 GPU 上高效地完成专家权重计算、分组筛选、Top-K 选取和归一化流程。

## 1. 模块结构

- **`topkrouter_nvidia.cuh`**: 头文件，定义 NVIDIA 特化版本的 Descriptor 类，通过宏展开声明公共接口
- **`topkrouter_nvidia.cu`**: 核心实现文件，包含 Descriptor 类的方法实现、类型分发逻辑和 CUDA kernel 启动封装

## 2. 核心类与数据结构

### `op::topkrouter::TopkrouterInfo`
- **位置**: `../info.h`（被本模块引用）
- **功能**: 封装 TopKRouter 算子的输入张量元数据和配置信息
- **关键成员**:
  - `xtype`: 输入数据类型（支持 FP32、FP16、BF16）
  - `shape`: 输入张量形状 [N, width]
  - `x_strides`: 输入张量步长，要求第二维连续（stride[1] == 1）
  - `N`: 批次大小（token 数量）
  - `width`: 专家总数（当前实现固定为 256）
- **验证规则**:
  - 输入必须是 2D 张量
  - 仅支持浮点类型（F32/F16/BF16）
  - 第二维必须连续存储

### `op::topkrouter::nvidia::Descriptor`
- **位置**: `topkrouter_nvidia.cu`
- **功能**: TopKRouter 算子的 NVIDIA 设备描述符，管理算子生命周期和执行
- **继承**: 继承自 `InfiniopDescriptor`（基类定义在 `topkrouter.h` 中通过宏展开）
- **关键成员**:
  - `_opaque`: `Opaque*` 类型，封装设备句柄的内部状态（`std::shared_ptr<device::nvidia::Handle::Internal>`）
  - `_info`: `TopkrouterInfo` 实例，存储输入张量的元数据
  - `_workspace_size`: 工作空间大小（当前实现为 0）
- **生命周期**:
  - **创建**: 通过静态工厂方法 `Descriptor::create()` 构造，验证输入张量格式并初始化设备句柄
  - **销毁**: 析构函数释放 `_opaque` 指针（使用 `delete`）
- **核心方法**:
  - **`create(handle, desc_ptr, x_desc, correction_bias_desc)`**:
    - 算法：调用 `TopkrouterInfo::create()` 验证并提取张量元数据，检查步长约束
    - 复杂度：O(1)
    - 错误处理：返回 `INFINI_STATUS_BAD_TENSOR_STRIDES` 若 stride[1] != 1
  - **`calculate(workspace, workspace_size, values, indices, x, correction_bias, routed_scaling_factor, topk, stream)`**:
    - 功能：执行 TopKRouter 计算，启动 CUDA kernel
    - 参数校验：检查工作空间大小（虽然当前实现不需要额外 workspace）
    - Kernel 启动：根据输入类型（F32/F16/BF16）分发到特化模板实例

### `op::topkrouter::nvidia::Descriptor::Opaque`
- **位置**: `topkrouter_nvidia.cu` 中定义
- **功能**: 封装 NVIDIA 设备句柄的内部实现，避免暴露设备层细节
- **成员**:
  - `internal`: `std::shared_ptr<device::nvidia::Handle::Internal>`，共享指针管理 CUDA 上下文资源

## 3. API 接口

### 创建算子描述符
```cpp
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,                          // [输入] InfiniOp 全局句柄
    Descriptor **desc_ptr,                            // [输出] 创建的描述符指针
    infiniopTensorDescriptor_t x_desc,                // [输入] 输入张量描述符 [N, 256]
    infiniopTensorDescriptor_t correction_bias_desc   // [输入] 修正偏置张量 [256]
);
// 返回: INFINI_STATUS_SUCCESS 成功, INFINI_STATUS_BAD_TENSOR_STRIDES 步长不合法
```

### 执行 TopKRouter 计算
```cpp
infiniStatus_t Descriptor::calculate(
    void *workspace,                  // [输入] 工作空间指针（当前未使用）
    size_t workspace_size,            // [输入] 工作空间大小
    float *values,                    // [输出] Top-K 值，形状 [N, topk]
    int *indices,                     // [输出] Top-K 索引，形状 [N, topk]
    const void *x,                    // [输入] 输入 logits，形状 [N, 256]
    const float *correction_bias,     // [输入] 修正偏置，形状 [256]
    const float routed_scaling_factor,// [输入] 路由缩放因子
    const size_t topk,                // [输入] 选取的 Top-K 数量
    void *stream                      // [输入] CUDA 流
) const;
// 返回: INFINI_STATUS_SUCCESS 成功, INFINI_STATUS_INSUFFICIENT_WORKSPACE 工作空间不足,
//       INFINI_STATUS_BAD_PARAM 参数不合法（width 必须为 256）
```

## 4. CUDA Kernel 实现

### `topkrouter_kernel<T, BLOCK_THREADS>`
- **位置**: `../cuda/kernel.cuh`
- **功能**: 对每个 token 执行层级化 Top-K 专家路由算法
- **模板参数**:
  - `T`: 输入数据类型（float/half/__nv_bfloat16）
  - `BLOCK_THREADS`: Block 线程数（固定 256）
- **Kernel 配置**:
  - Grid: `(N, 1, 1)` — 每个 token 一个 Block
  - Block: `(256, 1, 1)` — 每个 Block 256 个线程（对应 256 个专家）

#### 算法流程（单 Block 处理单个 Token）：

**阶段 1：预处理与 Sigmoid**
- 线程 `tid` 读取 `input[tid]`（对应专家 `tid` 的 logit）
- 计算 `sigmoid(input[tid])`，将 logits 转换为概率分布
- 加上修正偏置：`value += correction_bias[tid]`

**阶段 2：Warp 级排序**
- 将 256 个线程分为 8 个 Warp（每个 Warp 32 个线程）
- 每个 Warp 内部使用 `cub::WarpMergeSort` 对其 32 个专家权重降序排序
- 每个 Warp 选取前 2 个最大值（共 8 Warp × 2 = 16 个候选）
- 时间复杂度：O(32 log 32) ≈ O(1)（Warp 内排序）

**阶段 3：Warp 间分组筛选**
- 每个 Warp 的前 2 个值求和，得到 8 个组分数（`share_data_group[warp_id]`）
- Warp 0 使用 `WarpMergeSort` 对这 8 个组分数降序排序
- 标记前 4 个组为有效（`share_data_group_mask[indices] = 1.0f`）
- 淘汰后 4 个组的所有专家

**阶段 4：最终 Top-K 选取**
- 将无效组的专家权重置零（`value *= share_data_group_mask[warp_id]`）
- 全 Block 使用 `cub::BlockRadixSort` 对所有 256 个专家权重降序排序
- 选取前 `topk` 个专家（对应索引 `thread_indices[0..topk-1]`）

**阶段 5：归一化**
- 仅 Warp 0 参与归一化计算
- 对 Top-K 专家对应的原始 sigmoid 值求和：`sum = Σ sigmoid(input[indices[i]])`
- 每个值归一化：`output[i] = routed_scaling_factor * sigmoid_value / sum`

#### 共享内存布局：
```cpp
__shared__ float share_data[256];              // Warp 排序结果
__shared__ float share_data_group[8];          // Warp 组分数
__shared__ float share_data_group_mask[8];     // 有效组掩码（0 或 1）
__shared__ float share_sum;                    // Top-K 权重和（归一化分母）
```

### 辅助设备函数

#### `exp_func<T>(x)`
- **功能**: 类型安全的指数函数，支持 float/half/bfloat16
- **实现**: 使用 CUDA 内建函数 `__expf`，对于半精度类型先转换为 float

#### `sigmoid_func<T>(x)`
- **功能**: 计算 sigmoid 激活函数 `σ(x) = 1 / (1 + e^(-x))`
- **实现**: 组合 `exp_func` 和算术运算

#### `CustomLess`
- **功能**: CUB 排序自定义比较器，实现降序排序（`operator()` 返回 `lhs > rhs`）

## 5. 使用示例

```cpp
// 1. 创建 TopKRouter 描述符
infiniopHandle_t handle; // 假设已初始化
infiniopTensorDescriptor_t x_desc;    // [N, 256] 输入 logits
infiniopTensorDescriptor_t bias_desc; // [256] 修正偏置

op::topkrouter::nvidia::Descriptor* topkrouter_desc;
auto status = op::topkrouter::nvidia::Descriptor::create(
    handle, &topkrouter_desc, x_desc, bias_desc);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 2. 分配输出内存
float* d_values;     // [N, topk] 输出权重
int* d_indices;      // [N, topk] 输出专家索引
cudaMalloc(&d_values, N * topk * sizeof(float));
cudaMalloc(&d_indices, N * topk * sizeof(int));

// 3. 执行 TopKRouter
const void* d_x = ...;       // [N, 256] 输入 logits（GPU）
const float* d_bias = ...;   // [256] 修正偏置（GPU）
const float scaling_factor = 1.0f;
const size_t topk = 4;
cudaStream_t stream = 0;

status = topkrouter_desc->calculate(
    nullptr, 0,              // 无需工作空间
    d_values, d_indices,
    d_x, d_bias, scaling_factor, topk,
    stream
);

// 4. 使用结果
// d_values[i] 和 d_indices[i] 存储第 i 个 token 的 top-4 专家及归一化权重

// 5. 清理
cudaFree(d_values);
cudaFree(d_indices);
delete topkrouter_desc;
```

## 6. 实现细节

### 并行策略
- **Block 分解**: 每个 token 独立分配一个 Block，实现 token 间完全并行
- **Warp 分组**: 256 线程 = 8 Warp × 32 线程，利用 Warp 原语加速排序
- **层级筛选**: Warp 级预选 → Block 级精选，减少全局排序开销

### 内存访问模式
- **合并访问**: 每个线程访问 `input[tid]`，连续线程访问连续内存
- **共享内存**: 256 个线程对应 256 个专家，无 bank conflict（float 对齐）
- **只读偏置**: `correction_bias` 被 256 个线程同时读取，广播机制高效

### 性能优化
- **CUB 原语**: 使用 `WarpMergeSort`（O(32 log 32)）和 `BlockRadixSort`（O(256 log 256)）优化排序
- **提前淘汰**: Warp 分组筛选后淘汰 50% 专家（4/8 组无效），减少最终排序压力
- **类型特化**: 编译期为 F32/F16/BF16 生成特化 kernel，避免运行时分支
- **零拷贝工作空间**: 当前实现不需要额外 workspace，降低内存占用

### 限制与约束
- **固定专家数**: 硬编码 `width == 256`，否则返回 `INFINI_STATUS_BAD_PARAM`
- **连续内存**: 要求 `x_strides[1] == 1`，保证行主序连续存储
- **Block 大小**: 固定 256 线程，不可配置
- **Top-K 上限**: 最大为 8（每个 Warp 保留前 2 个的约束）

### 依赖关系
- **外部依赖**:
  - CUDA Toolkit（CUB 库）
  - `device::nvidia::Handle`（设备管理）
  - `infiniopTensorDescriptor_t`（张量描述符）
- **内部依赖**:
  - `../topkrouter.h`（Descriptor 基类宏定义）
  - `../info.h`（TopkrouterInfo 元数据类）
  - `../cuda/kernel.cuh`（CUDA kernel 实现）
  - `../../devices/nvidia/nvidia_common.cuh`（NVIDIA 通用工具）
  - `../../devices/nvidia/nvidia_kernel_common.cuh`（Kernel 通用工具）

### 错误处理
- **类型错误**: 返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`（非浮点类型）
- **形状错误**: 返回 `INFINI_STATUS_BAD_TENSOR_SHAPE`（非 2D 张量）
- **步长错误**: 返回 `INFINI_STATUS_BAD_TENSOR_STRIDES`（非连续存储）
- **参数错误**: 返回 `INFINI_STATUS_BAD_PARAM`（width != 256）
- **工作空间**: 返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`（当前不应触发）

### 设计模式
- **RAII**: 使用 `std::shared_ptr` 管理设备句柄生命周期
- **工厂方法**: `create()` 静态方法封装对象构造与验证
- **策略模式**: 模板特化实现类型分发（F32/F16/BF16）
- **PImpl 惯用法**: `Opaque` 结构隐藏 NVIDIA 设备层实现细节
