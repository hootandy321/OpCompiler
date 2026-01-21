# RoPE Kunlun 算子核心实现文档

本文档详细描述了 RoPE (Rotary Position Embedding) 算子在昆仑 (Kunlun) XPU 设备上的实现。该实现支持两种 RoPE 模式（GPT-J 和标准模式），并针对昆仑 XPU 的架构特点进行了优化。

## 1. 模块结构

- **`rope_kunlun.h`**: RoPE 算子的昆仑设备后端头文件，定义了 `op::rope::kunlun::Descriptor` 类
- **`rope_kunlun.xpu`**: RoPE 算子的 XPU 内核实现，包含核心计算 kernel 和设备接口

## 2. 核心类与数据结构

### `Descriptor` 类
- **位置**: `rope_kunlun.h` / `rope_kunlun.xpu`
- **主要功能**: RoPE 算子的昆仑设备描述符，继承自 `InfiniopDescriptor`，负责算子实例创建、参数验证和 kernel 调度
- **关键成员**:
  - `_opaque`: `Opaque*` 类型，持有 `std::shared_ptr<device::kunlun::Handle::Internal>`，管理昆仑设备的内部句柄
  - `_info`: `RoPEInfo` 类型，存储张量维度、步长、数据类型等元数据
  - `_workspace_size`: `size_t` 类型，工作空间大小（当前实现为 0）
- **核心方法**:
  - `create(handle, desc_ptr, y_desc, x_desc, pos_desc, sin_desc, cos_desc, algo)`: 静态工厂方法，创建描述符实例。调用 `RoPEInfo::createRoPEInfo` 验证参数并构建元数据，初始化 Opaque 内部句柄
  - `calculate(workspace, workspace_size, y, x, pos_ids, sin_table, cos_table, stream)`: 执行 RoPE 计算。根据数据类型和位置 ID 类型分发到对应的模板特化 kernel，使用 `<<<8, 64, stream>>>` 配置启动 kernel
- **生命周期**: 通过 `new Descriptor(...)` 动态分配，析构函数释放 `_opaque` 指针

### `RoPEInfo` 类
- **位置**: 父级 `rope.h`（共享定义）
- **主要功能**: 封装 RoPE 算子的所有元数据和验证逻辑
- **关键成员**:
  - `data_type`: `infiniDtype_t`，数据类型（F32/F16/BF16）
  - `pos_type`: `infiniDtype_t`，位置 ID 类型（I32/U32）
  - `batch`, `seqlen`, `nhead`, `dhead`: `size_t`，张量维度
  - `table_len`, `table_dim`: `size_t`，sin/cos 表维度（`table_dim = dhead / 2`）
  - `x_stride_*`, `y_stride_*`: `ptrdiff_t`，各维度步长
  - `has_batch_dim`: `bool`，输入张量是否包含 batch 维度（3D 或 4D）
  - `pos_has_batch_dim`: `bool`，位置 ID 是否为 2D 张量
  - `algo`: `infiniopRoPEAlgo_t`，RoPE 模式（GPT-J 或标准模式）
- **核心方法**:
  - `createRoPEInfo(...)`: 静态方法，验证张量形状、数据类型、步长等约束条件，返回 `Result<RoPEInfo>`。验证包括：数据类型一致性、维度匹配（`dhead == table_dim * 2`）、最后维度连续性、sin/cos 表完全连续

## 3. 核心算法

### `RoPEKernel<T, Tindex>` (模板函数)
- **位置**: `rope_kunlun.xpu` (第 10-123 行)
- **功能**: RoPE 计算的设备 kernel，实现旋转位置编码的核心数学运算
- **签名**:
  ```cpp
  __global__ void RoPEKernel(
      T *destination, const T *source,
      const Tindex *pos_ids, const T *sin_table, const T *cos_table,
      uint32_t seqlen, uint32_t nhead, uint32_t dhead,
      int32_t x_stride_seqlen, int32_t x_stride_nhead,
      int32_t y_stride_seqlen, int32_t y_stride_nhead,
      bool IsGPTJ, XPUStream stream)
  ```
- **算法逻辑**:
  1. **线程分配** (第 18-32 行): 使用静态负载均衡策略，将 `seqlen * nhead` 个任务分配给所有 XPU 核心。每个线程处理 `step` 个任务，起始索引为 `ind_start`
  2. **局部内存分配** (第 34-39 行): 每个核心分配 4 个 256 元素的 `__local__` 缓冲区（`x_local`, `y_local`, `sin_local`, `cos_local`）和 1 个位置 ID 缓冲区 `pos_local`
  3. **位置索引计算** (第 49-58 行): 根据 `ind_i` 计算 3D 张量索引（seqlen, nhead, dhead），通过 `GM2LM` 从全局内存加载位置 ID 到局部内存
  4. **模式分支**:
     - **GPT-J 模式** (`IsGPTJ == true`, 第 60-87 行): 顺序处理连续的元素对。每次迭代读取 `buf_size` 个元素（通常 256），对前半部分应用旋转公式：
       ```
       y[2k]   = x[2k] * cos[k] - x[2k+1] * sin[k]
       y[2k+1] = x[2k] * sin[k] + x[2k+1] * cos[k]
       ```
       BF16 类型通过 `__bfloat162float` 转换为 float 计算后再转回
     - **标准模式** (`IsGPTJ == false`, 第 89-121 行): 同时读取前半部分和后半部分到 `x_local[0:buf_table]` 和 `x_local[buf_table:2*buf_table]`，应用旋转：
       ```
       y[k]           = x[k] * cos[k] - x[k+buf_table] * sin[k]
       y[k+buf_table] = x[k] * sin[k] + x[k+buf_table] * cos[k]
       ```
  5. **内存同步与写回** (第 85, 117 行): `mfence()` 确保局部内存写入完成，`LM2GM` 将结果写回全局内存
- **复杂度**: O(seqlen * nhead * dhead)，每个元素执行常数次浮点运算
- **优化技术**:
  - 分块加载（`buf_size = 256`）减少全局内存访问次数
  - 本地内存缓存 sin/cos 表，减少重复加载
  - 静态负载均衡避免核心空闲

### `RoPE<T, Tindex>` (模板函数)
- **位置**: `rope_kunlun.xpu` (第 125-137 行)
- **功能**: 封装 kernel 启动的包装函数
- **签名**:
  ```cpp
  void RoPE(
      void *destination, const void *source,
      const void *pos_ids, const void *sin_table, const void *cos_table,
      uint32_t seqlen, uint32_t nhead, uint32_t dhead,
      int32_t x_stride_seqlen, int32_t x_stride_nhead,
      int32_t y_stride_seqlen, int32_t y_stride_nhead,
      bool IsGPTJ, XPUStream stream)
  ```
- **实现**: 使用 `<<<8, 64, stream>>>` 配置启动 `RoPEKernel`，8 个 cluster，每 cluster 64 个核心

## 4. API 接口

### 公开 API

```cpp
// 创建 RoPE 描述符
infiniStatus_t op::rope::kunlun::Descriptor::create(
    infiniopHandle_t handle,                    // 昆仑设备句柄
    Descriptor **desc_ptr,                      // 输出：描述符指针
    infiniopTensorDescriptor_t y_desc,          // 输出张量描述符 [batch?, seqlen, nhead, dhead]
    infiniopTensorDescriptor_t x_desc,          // 输入张量描述符 [batch?, seqlen, nhead, dhead]
    infiniopTensorDescriptor_t pos_desc,        // 位置 ID [seqlen] 或 [batch, seqlen]
    infiniopTensorDescriptor_t sin_desc,        // Sin 表 [table_len, table_dim]
    infiniopTensorDescriptor_t cos_desc,        // Cos 表 [table_len, table_dim]
    infiniopRoPEAlgo_t algo);                   // RoPE 模式：INFINIOP_ROPE_ALGO_GPT_J 或其他

// 执行 RoPE 计算
infiniStatus_t Descriptor::calculate(
    void *workspace,              // 工作空间（未使用，传 nullptr）
    size_t workspace_size,        // 工作空间大小（传 0）
    void *y,                      // 输出数据指针
    const void *x,                // 输入数据指针
    const void *pos_ids,          // 位置 ID 数据指针
    const void *sin_table,        // Sin 表数据指针
    const void *cos_table,        // Cos 表数据指针
    void *stream) const;          // 昆仑流指针
```

### 支持的数据类型组合
- **数据类型**: `INFINI_DTYPE_F32`, `INFINI_DTYPE_F16`, `INFINI_DTYPE_BF16`
- **位置 ID 类型**: `INFINI_DTYPE_I32`, `INFINI_DTYPE_U32`
- **约束**: 所有张量（x, y, sin, cos）的数据类型必须一致

## 5. 使用示例

```cpp
// 示例：在昆仑 XPU 上使用 RoPE 算子（标准模式，3D 张量）

#include "infiniop.h"
#include "infiniop/ops/rope.h"

// 1. 创建设备句柄
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_KUNLUN, device_id);

// 2. 定义张量维度
constexpr size_t seqlen = 128;
constexpr size_t nhead = 32;
constexpr size_t dhead = 128;  // 必须是 table_dim * 2
constexpr size_t table_len = 2048;
constexpr size_t table_dim = dhead / 2;

// 3. 创建张量描述符（3D 张量，无 batch 维度）
int64_t shape_3d[3] = {seqlen, nhead, dhead};
int64_t strides_3d[3] = {nhead * dhead, dhead, 1};
infiniopTensorDescriptor_t x_desc, y_desc;
infiniopCreateTensorDescriptor(&x_desc, INFINI_DTYPE_F16, 3, shape_3d, strides_3d);
infiniopCreateTensorDescriptor(&y_desc, INFINI_DTYPE_F16, 3, shape_3d, strides_3d);

// 4. 创建位置 ID 和 sin/cos 表描述符
int64_t pos_shape[1] = {seqlen};
int64_t pos_stride[1] = {1};
infiniopTensorDescriptor_t pos_desc;
infiniopCreateTensorDescriptor(&pos_desc, INFINI_DTYPE_I32, 1, pos_shape, pos_stride);

int64_t table_shape[2] = {table_len, table_dim};
int64_t table_strides[2] = {table_dim, 1};
infiniopTensorDescriptor_t sin_desc, cos_desc;
infiniopCreateTensorDescriptor(&sin_desc, INFINI_DTYPE_F16, 2, table_shape, table_strides);
infiniopCreateTensorDescriptor(&cos_desc, INFINI_DTYPE_F16, 2, table_shape, table_strides);

// 5. 创建 RoPE 描述符（标准模式）
op::rope::kunlun::Descriptor *rope_desc;
infiniStatus_t status = op::rope::kunlun::Descriptor::create(
    handle, &rope_desc, y_desc, x_desc, pos_desc, sin_desc, cos_desc,
    infiniopRoPEAlgo_t::INFINIOP_ROPE_ALGO_DEFAULT);

// 6. 分配设备内存并初始化数据
half *d_x, *d_y, *d_sin, *d_cos;
int32_t *d_pos;
xpu_malloc((void**)&d_x, seqlen * nhead * dhead * sizeof(half));
xpu_malloc((void**)&d_y, seqlen * nhead * dhead * sizeof(half));
xpu_malloc((void**)&d_pos, seqlen * sizeof(int32_t));
xpu_malloc((void**)&d_sin, table_len * table_dim * sizeof(half));
xpu_malloc((void**)&d_cos, table_len * table_dim * sizeof(half));

// 使用 XPU API 初始化输入数据（略）
// ...

// 7. 创建流并执行计算
XPUStream stream;
xpu_stream_create(&stream);

status = rope_desc->calculate(
    nullptr, 0,           // 无需工作空间
    d_y, d_x,             // 输出和输入
    d_pos,                // 位置 ID
    d_sin, d_cos,         // sin/cos 表
    stream);              // 昆仑流

xpu_stream_synchronize(stream);

// 8. 清理资源
xpu_free(d_x); xpu_free(d_y); xpu_free(d_pos);
xpu_free(d_sin); xpu_free(d_cos);
xpu_stream_destroy(stream);
delete rope_desc;
infiniopDestroyHandle(handle);
```

## 6. 实现细节

### 内存管理
- **分块策略**: 使用固定的 `buf_size = 256` 作为局部内存缓冲区大小，每次从全局内存加载一个块的数据到局部内存进行处理
- **局部内存优化**: sin/cos 表数据被缓存在局部内存中，避免重复加载相同位置的数据
- **内存同步**: 使用 `mfence()` 确保局部内存写入在写回全局内存前完成

### 并发与负载均衡
- **静态负载均衡**: 在编译时计算每个线程处理的任务数量（`step_easy` 或 `step_hard`），确保所有核心均匀分配任务
- **核心映射**: 使用 `core_id()`, `cluster_id()`, `core_num()`, `cluster_num()` 获取 XPU 硬件拓扑信息
- **任务分配公式**:
  ```
  total_threads = ncores * cluster_num()
  remain = other_size % total_threads
  step_easy = (other_size - remain) / total_threads
  step_hard = step_easy + 1
  ```
  前 `remain` 个线程分配 `step_hard` 个任务，其余分配 `step_easy` 个任务

### 性能优化
- **数据类型特化**: 针对不同数据类型（F32, F16, BF16）使用不同的计算路径。BF16 通过 `__bfloat162float` 和 `__float2bfloat16` 进行精度转换
- **GPT-J 模式优化**: 顺序处理连续的元素对，减少寻址开销
- **标准模式优化**: 同时读取前半部分和后半部分，减少内存访问次数
- **Kernel 配置**: 固定使用 `<<<8, 64, stream>>>` 启动配置，8 个 cluster，每 cluster 64 个核心

### 错误处理
- **参数验证**: `RoPEInfo::createRoPEInfo` 执行全面的参数检查，包括：
  - 空指针检查（返回 `INFINI_STATUS_NULL_POINTER`）
  - 数据类型一致性检查（返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`）
  - 维度匹配检查（返回 `INFINI_STATUS_BAD_TENSOR_SHAPE`）
  - 步长连续性检查（返回 `INFINI_STATUS_BAD_TENSOR_STRIDES`）
- **运行时检查**: Kernel 中使用 `if (cid >= ncores) return;` 防止越界

### 依赖关系
- **设备层**: 依赖 `device::kunlun::Handle` 和 `device::kunlun::Handle::Internal` 管理设备上下文
- **通用头文件**: 包含 `kunlun_common.h`, `kunlun_handle.h`, `kunlun_kernel_common.h` 获取 XPU API 和工具宏
- **上层接口**: 继承 `InfiniopDescriptor` 基类，实现统一的算子接口

### 设计模式
- **Pimpl 模式**: 使用 `Opaque` 结构体隐藏设备相关的实现细节
- **模板元编程**: 使用 C++ 模板实现数据类型和位置 ID 类型的静态分发
- **RAII**: 使用 `std::shared_ptr` 管理设备句柄的生命周期
- **策略模式**: 通过 `IsGPTJ` 参数选择不同的 RoPE 计算策略

### RoPE 算法细节
- **GPT-J 模式**: 对连续的元素对 `(x[2k], x[2k+1])` 应用 2D 旋转矩阵，适用于交错存储的复数表示
- **标准模式**: 对前半部分和后半部分分别应用旋转，适用于分离存储的复数表示
- **数学公式**:
  ```
  对于复数 z = x + iy，旋转角度 θ 后：
  z' = z * e^(iθ) = (x + iy) * (cos θ + i sin θ)
     = (x * cos θ - y * sin θ) + i(x * sin θ + y * cos θ)
  ```
  在 GPT-J 模式中，`x[2k]` 对应实部，`x[2k+1]` 对应虚部
