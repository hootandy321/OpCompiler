# RoPE (Rotary Position Embedding) Moore/MUSA 后端实现文档

## 模块概述

本模块实现了 RoPE (Rotary Position Embedding) 操作在 Moore 硬件平台（使用 MUSA 框架）上的 CUDA 内核版本。RoPE 是 Transformer 模型中用于编码位置信息的关键技术，通过旋转变换将绝对位置信息注入到注意力机制的查询和键向量中。

该实现提供了两种算法模式：
- **GPT-J 模式**：相邻元素作为复数对进行旋转
- **标准模式**：交替元素进行旋转

核心特性：
- 支持 3D 和 4D 张量（可选批次维度）
- 支持 FP16、BF16、FP32、FP64 数据类型
- 支持多种整数类型的 position IDs
- 通过模板化和 `constexpr` 实现编译期优化
- 为 MUSA 平台适配了 bfloat16 内置函数差异

## 1. 模块结构

- **`rope_moore.h`**：Moore 后端的描述符声明，定义了 `op::rope::moore::Descriptor` 类的接口
- **`rope_kernel_moore.h`**：核心 CUDA 内核实现，包含 `ropeThreadPerItemBlock` 模板函数
- **`rope_moore.mu`**：主机端实现，包括描述符创建、内核调度和类型分发逻辑

## 2. 核心类与数据结构

### `RoPEInfo`
- **位置**：`../rope.h`（父级共享定义）
- **功能**：封装 RoPE 操作的所有张量形状、步长、类型信息
- **关键成员**：
  - `data_type` / `pos_type`：数据类型和位置 ID 类型（infiniDtype_t）
  - `batch`, `seqlen`, `nhead`, `dhead`：张量维度尺寸
  - `table_len`, `table_dim`：sin/cos 查找表的维度
  - `y_stride_*`, `x_stride_*`：各维度的步长（支持非连续内存）
  - `has_batch_dim`：是否有批次维度（3D vs 4D 张量）
  - `pos_has_batch_dim`：position IDs 是否为每批次独立（2D vs 1D）
  - `algo`：算法选择（GPT-J 或标准模式）
- **验证逻辑**：`createRoPEInfo()` 静态方法执行完整的参数验证，包括形状一致性检查、步长连续性验证、数据类型兼容性检查

### `op::rope::moore::Descriptor`
- **位置**：`rope_moore.mu` + `rope_moore.h`（通过 DESCRIPTOR 宏定义）
- **继承**：`InfiniopDescriptor`
- **功能**：Moore 后端的 RoPE 操作描述符，管理设备和操作状态
- **关键成员**：
  - `_opaque`：不透明指针，指向 `Opaque` 结构（持有设备句柄）
  - `_info`：`RoPEInfo` 实例，存储张量布局信息
  - `_workspace_size`：工作空间大小（当前实现固定为 0）
- **生命周期**：
  - 通过 `create()` 静态方法构造，接收 MUSA handle 和张量描述符
  - 析构函数释放 `_opaque` 资源

### `Descriptor::Opaque`
- **位置**：`rope_moore.mu`（内部结构）
- **功能**：封装 Moore 设备句柄的内部状态
- **关键成员**：
  - `internal`：`std::shared_ptr<device::moore::Handle::Internal>`，提供 MUSA 设备能力查询（如最大线程数）

### `ropeThreadPerItemBlock<IsGPTJ, Tdata, Tindex, Tangle>`
- **位置**：`rope_kernel_moore.h`
- **功能**：设备端模板函数，执行单个 block 的 RoPE 变换
- **模板参数**：
  - `IsGPTJ`：`bool`，编译期标志，选择 GPT-J 或标准旋转算法
  - `Tdata`：数据类型（half, cuda_bfloat16, float, double）
  - `Tindex`：位置索引类型（uint8_t, uint16_t, ..., int64_t）
  - `Tangle`：角度表类型（通常为 float）
- **参数**：
  - `y_` / `x_`：输出和输入张量指针
  - `pos_ids`：位置 ID 数组（1D 或 2D）
  - `sin_table` / `cos_table`：预计算的三角函数查找表
  - `table_dim`：旋转维度（dhead/2）
  - `pos_stride_batch`：position IDs 的批次步长
  - `pos_has_batch_dim` / `has_batch_dim`：批次维度标志
  - `*_stride_*`：各维度的步长
- **算法**：
  1. 从 blockIdx 计算批次、序列、注意力头索引
  2. 根据步长计算内存偏移量
  3. 从 pos_ids 读取位置 ID，查找对应的 sin/cos 值
  4. 遍历 table_dim，对每个元素对应用旋转变换
  5. 使用 `if constexpr` 在编译期选择数据类型特化路径

### `ropeThreadPerItemKernel<IsGPTJ, Tdata, Tindex, Tangle>`
- **位置**：`rope_moore.mu`
- **功能**：全局内核包装函数，设置 grid 配置后调用 `ropeThreadPerItemBlock`
- **内核签名**：`INFINIOP_MOORE_KERNEL`（宏展开为 `__global__ void`）
- **Grid 配置**：
  - 3D 张量：`dim3(seqlen, nhead)`（2D grid）
  - 4D 张量：`dim3(seqlen, nhead, batch)`（3D grid）
- **Block 配置**：`max(table_dim, block_size)`，其中 `block_size` 来自设备能力的最大线程数

### `calculateRoPE<Tdata, Tindex>`
- **位置**：`rope_moore.mu`
- **功能**：主机端模板函数，封装内核启动逻辑
- **参数**：
  - `info`：const RoPEInfo&，张量布局信息
  - `block_size`：int，设备最大线程数
  - `y`, `x`, `pos_ids`, `sin_table`, `cos_table`：设备指针
  - `stream`：musaStream_t，CUDA 流
- **执行流程**：
  1. 提取 seqlen, nhead, batch 构建网格维度
  2. 根据 `info.algo` 选择 `IsGPTJ=true` 或 `false` 的内核实例化
  3. 启动 `ropeThreadPerItemKernel` 内核
- **复杂度**：O(batch × seqlen × nhead × table_dim / 并行度)

## 3. API 接口

```cpp
// 创建 Moore 后端 RoPE 描述符
infiniStatus_t op::rope::moore::Descriptor::create(
    infiniopHandle_t handle,                    // Moore 设备句柄
    Descriptor **desc_ptr,                      // [输出] 描述符指针
    infiniopTensorDescriptor_t y_desc,          // 输出张量描述符 [batch?, seqlen, nhead, dhead]
    infiniopTensorDescriptor_t x_desc,          // 输入张量描述符（形状与 y 相同）
    infiniopTensorDescriptor_t pos_desc,        // 位置 ID 描述符 [seqlen] 或 [batch, seqlen]
    infiniopTensorDescriptor_t sin_desc,        // Sin 表描述符 [table_len, table_dim]
    infiniopTensorDescriptor_t cos_desc,        // Cos 表描述符（形状与 sin 相同）
    infiniopRoPEAlgo_t algo                     // 算法：INFINIOP_ROPE_ALGO_GPT_J 或其他
);
// 返回值：成功时返回 INFINI_STATUS_SUCCESS，失败返回错误码（如 INFINI_STATUS_BAD_TENSOR_SHAPE）

// 执行 RoPE 计算
infiniStatus_t op::rope::moore::Descriptor::calculate(
    void *workspace,            // 工作空间（当前未使用，传 nullptr）
    size_t workspace_size,      // 工作空间大小（当前为 0）
    void *y,                    // [输出] 设备端输出张量指针
    const void *x,              // 设备端输入张量指针
    const void *pos_ids,        // 设备端位置 ID 数组指针
    const void *sin_table,      // 设备端 sin 表指针
    const void *cos_table,      // 设备端 cos 表指针
    void *stream                // MUSA 流指针（musaStream_t）
) const;
// 返回值：成功时返回 INFINI_STATUS_SUCCESS，失败返回类型错误码

// 查询工作空间大小
size_t op::rope::moore::Descriptor::workspaceSize() const;
// 返回值：当前实现固定返回 0（无需额外工作空间）
```

## 4. 使用示例

```cpp
// 示例：在 Moore/MUSA 平台上执行 RoPE 操作
#include "infiniop/ops/rope.h"
#include "infiniop/devices/moore/moore_handle.h"

// 1. 初始化 MUSA 设备和句柄
int device_id = 0;
infiniopHandle_t handle;
infiniStatus_t status = infinioCreateMooreHandle(&handle, device_id);
assert(status == INFINI_STATUS_SUCCESS);

// 2. 定义张量形状（4D 张量：带批次维度）
constexpr size_t batch = 8, seqlen = 1024, nhead = 32, dhead = 128;
constexpr size_t table_dim = dhead / 2;  // RoPE 旋转维度为 dhead 的一半
constexpr size_t table_len = 8192;       // 最大序列长度

// 3. 创建张量描述符
int64_t y_shape[4] = {batch, seqlen, nhead, dhead};
int64_t y_strides[4] = {seqlen * nhead * dhead, nhead * dhead, dhead, 1};
infiniopTensorDescriptor_t y_desc, x_desc;
infiniCreateTensorDescriptor(&y_desc, INFINI_DTYPE_F16, 4, y_shape, y_strides);
infiniCreateTensorDescriptor(&x_desc, INFINI_DTYPE_F16, 4, y_shape, y_strides);

// 位置 IDs：2D 张量（每批次独立）
int64_t pos_shape[2] = {batch, seqlen};
int64_t pos_strides[2] = {seqlen, 1};
infiniopTensorDescriptor_t pos_desc;
infiniCreateTensorDescriptor(&pos_desc, INFINI_DTYPE_I32, 2, pos_shape, pos_strides);

// Sin/Cos 表：2D 张量
int64_t table_shape[2] = {table_len, table_dim};
int64_t table_strides[2] = {table_dim, 1};
infiniopTensorDescriptor_t sin_desc, cos_desc;
infiniCreateTensorDescriptor(&sin_desc, INFINI_DTYPE_F32, 2, table_shape, table_strides);
infiniCreateTensorDescriptor(&cos_desc, INFINI_DTYPE_F32, 2, table_shape, table_strides);

// 4. 创建 RoPE 操作描述符
op::rope::moore::Descriptor *rope_desc;
status = op::rope::moore::Descriptor::create(
    handle, &rope_desc, y_desc, x_desc, pos_desc, sin_desc, cos_desc,
    INFINIOP_ROPE_ALGO_DEFAULT  // 或 INFINIOP_ROPE_ALGO_GPT_J
);
assert(status == INFINI_STATUS_SUCCESS);

// 5. 分配设备内存
half *d_x, *d_y;
int32_t *d_pos_ids;
float *d_sin, *d_cos;
size_t x_bytes = batch * seqlen * nhead * dhead * sizeof(half);
size_t pos_bytes = batch * seqlen * sizeof(int32_t);
size_t table_bytes = table_len * table_dim * sizeof(float);
musaMalloc(&d_x, x_bytes);
musaMalloc(&d_y, x_bytes);
musaMalloc(&d_pos_ids, pos_bytes);
musaMalloc(&d_sin, table_bytes);
musaMalloc(&d_cos, table_bytes);

// 6. 上传数据到设备（省略 cudaMemcpy 代码）

// 7. 创建 MUSA 流
musaStream_t stream;
musaStreamCreate(&stream);

// 8. 执行 RoPE 计算
status = rope_desc->calculate(
    nullptr,           // workspace（当前未使用）
    0,                 // workspace_size
    d_y,               // 输出
    d_x,               // 输入
    d_pos_ids,         // 位置 IDs
    d_sin,             // Sin 表
    d_cos,             // Cos 表
    stream             // MUSA 流
);
assert(status == INFINI_STATUS_SUCCESS);

// 9. 同步并清理
musaStreamSynchronize(stream);
musaFree(d_x); musaFree(d_y); musaFree(d_pos_ids);
musaFree(d_sin); musaFree(d_cos);
musaStreamDestroy(stream);
delete rope_desc;
infiniDestroyHandle(handle);
```

## 5. 实现细节

### 内存管理与布局
- **零拷贝设计**：内核直接在输入/输出张量上操作，无需中间缓冲区
- **步长支持**：通过 `*_stride_*` 参数支持任意内存布局（包括转置张量），但要求最后一维连续（stride=1）
- **查找表要求**：sin/cos 表必须完全连续（`isContiguous()` 检查），以优化内存访问模式

### 并行策略
- **Block 分配**：每个 (seqlen, nhead, batch) 组合分配一个独立的 block
- **Thread 并行**：每个 block 内的线程并行处理 table_dim 维度的元素
- **线程数自适应**：`nthreads = max(table_dim, block_size)`，确保充分利用硬件资源

### RoPE 算法细节
- **数学原理**：对向量元素对 $(x_{2i}, x_{2i+1})$ 应用旋转变换：
  ```
  y_{2i}   = x_{2i}   * cos(pos_id) - x_{2i+1} * sin(pos_id)
  y_{2i+1} = x_{2i}   * sin(pos_id) + x_{2i+1} * cos(pos_id)
  ```
- **GPT-J 模式**（`IsGPTJ=true`）：相邻元素 $(x_{2i}, x_{2i+1})$ 作为复数对，使用 half2/bfloat162 向量化加载
- **标准模式**（`IsGPTJ=false`）：交替元素 $(x_i, x_{i+table\_dim})$ 进行旋转，支持非对称维度布局

### MUSA 平台适配
- **类型别名映射**：
  - `cuda_bfloat16` → `mt_bfloat16`（MUSA bfloat16 类型）
  - `cuda_bfloat162` → `mt_bfloat162`（打包 bfloat16 向量类型）
- **内置函数替换**：
  - CUDA 的 `__low2bfloat16` / `__high2bfloat16`（提取 bfloat16 打包值的低/高半部分）在 MUSA 中不可用
  - 替换为 MUSA 特定函数：`__low2float` / `__high2float`（直接转换为 float）
  - 重新打包时使用 `__floats2bfloat162_rn`（舍入到最近值）
- **数学库适配**：通过 `exp_()` 模板函数封装 `expf` / `exp`，解决 MUSA 数学库的类型歧义

### 编译期优化
- **`if constexpr`**：所有数据类型分支在编译期解析，生成无分支的高效机器码
- **模板特化**：为 half/bfloat16/float/double 生成独立内核实例，避免运行期类型判断
- **内联函数**：`__forceinline__` 强制内联设备函数，减少调用开销

### 错误处理
- **类型分发宏**：
  - `ROPE_TYPE(TDATA)`：根据位置 ID 类型（8/16/32/64 位有/无符号整数）分发到特化模板
  - `CALCULATE_ROPE(TDATA, TINDEX)`：进一步根据数据类型（FP16/BF16/FP32/FP64）实例化内核
- **错误码传播**：`INFINI_STATUS_BAD_TENSOR_DTYPE` 表示不支持的类型组合

### 性能特征
- **计算复杂度**：O(N)，其中 N = batch × seqlen × nhead × table_dim
- **内存访问模式**：
  - 每个线程读取 4 个值（x 对, sin, cos），写入 2 个值（y 对）
  - sin/cos 表跨线程重用（同一位置 ID 的所有线程读取相同表项）
  - 输入/输出线性访问，有利于缓存和内存合并
- **优化机会**：
  - 向量化加载（half2/bfloat162）在 GPT-J 模式下提升吞吐量
  - 查找表缓存：sin/cos 值可在 shared memory 中缓存（当前未实现，未来优化方向）

### 依赖项
- **外部库**：
  - `musa.h`：MUSA 运行时 API
  - `musa_fp16_mtgpu.h`：MUSA FP16 支持（Moore 特定）
  - `musa_bf16.h`：MUSA bfloat16 支持
- **内部模块**：
  - `device::moore::Handle`：提供设备能力查询（`maxThreadsPerBlock()`）
  - `RoPEInfo`：父级共享的参数验证和形状推导逻辑
  - `INFINIOP_MOORE_KERNEL` 宏：统一内核声明风格

### 设计模式
- **策略模式**：通过 `algo` 参数选择 GPT-J 或标准旋转算法
- **模板方法模式**：`calculate()` 定义执行框架，`calculateRoPE()` 实现具体内核调度
- **工厂模式**：`Descriptor::create()` 作为工厂函数，封装对象构建逻辑
- **不透明指针模式**：`_opaque` 隐藏设备相关实现细节，保持接口跨平台一致性
