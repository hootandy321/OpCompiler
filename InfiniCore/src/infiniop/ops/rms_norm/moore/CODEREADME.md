# RMSNorm Moore 平台实现核心文档

本文档详细描述了 RMSNorm (Root Mean Square Normalization) 算子在 Moore/MUSA GPU 平台上的实现。Moore 是摩尔线程推出的国产 GPU 架构，MUSA 是其对应的 CUDA 兼容编程模型。

## 1. 模块结构

- **`rms_norm_moore.h`**: Moore 平台的描述符类声明头文件，通过 `DESCRIPTOR(moore)` 宏定义完整的 API 接口
- **`rms_norm_moore.mu`**: Moore 平台的核心实现文件，包含内核调度、类型分发和设备内存管理逻辑

## 2. 核心类与数据结构

### `op::rms_norm::moore::Descriptor`
- **位置**: `rms_norm_moore.h` (声明), `rms_norm_moore.mu` (实现)
- **主要功能**: RMSNorm 算子在 Moore 平台的运行时描述符，负责内核启动配置和设备适配
- **继承关系**: 继承自 `InfiniopDescriptor` 基类
- **关键成员**:
  - `_opaque`: 指向 `Opaque` 结构体的指针，封装 Moore 设备相关上下文
  - `_info`: `RMSNormInfo` 对象，存储张量形状、步长、数据类型等元信息
  - `_workspace_size`: 工作空间大小（当前实现为 0）
- **核心方法**:
  - `create(handle, desc_ptr, y_desc, x_desc, w_desc, epsilon)`: 静态工厂方法，创建并初始化描述符
    - 参数验证：检查张量类型兼容性（FP16/FP32/BF16 组合）
    - 提取 Moore 设备句柄的内部状态 (`device::moore::Handle::Internal`)
    - 构建 `RMSNormInfo` 元信息对象
    - 返回 `INFINI_STATUS_SUCCESS` 或错误码
  - `calculate(workspace, workspace_size, y, x, w, stream)`: 执行 RMSNorm 计算
    - 验证工作空间大小
    - 提取张量步长和维度信息
    - 根据设备能力动态选择 block size (512/1024/2048)
    - 调用类型特化的内核启动函数 `launchKernel<BLOCK_SIZE>`
    - 在 MUSA stream 上异步执行内核

### `Descriptor::Opaque`
- **位置**: `rms_norm_moore.mu` 第 28-30 行
- **主要功能**: 封装 Moore 设备的内部状态，实现 Pimpl (Pointer to Implementation) 模式
- **关键成员**:
  - `internal`: `std::shared_ptr<device::moore::Handle::Internal>`，Moore 设备句柄的共享指针
- **生命周期**: 由 `Descriptor::create` 构造，在 `Descriptor` 析构时释放

### `RMSNormInfo`
- **位置**: `/home/qy/src/Infini/InfiniCore/src/infiniop/ops/rms_norm/info.h`
- **主要功能**: 验证和存储 RMSNorm 操作的张量元信息
- **关键成员**:
  - `atype`: 激活值数据类型 (FP16/FP32/BF16)
  - `wtype`: 权重数据类型 (FP16/FP32/BF16)
  - `epsilon`: 数值稳定项（防止除零）
  - `shape`: 输出张量形状，支持 2D `(batch, dim)` 或 3D `(batch, nhead, dim)`
  - `x_strides`: 输入张量步长数组
  - `y_strides`: 输出张量步长数组
- **核心方法**:
  - `create(y_desc, x_desc, w_desc, epsilon)`: 静态工厂方法，验证张量描述符
    - 类型兼容性检查：
      - FP16/FP32/BF16 激活值对应不同权重类型组合
      - FP32/FP64 激活值要求权重类型相同
    - 形状验证：支持 2D/3D 张量，最后一维必须连续
    - 返回 `utils::Result<RMSNormInfo>` 类型安全结果
  - `dim()`: 返回归一化维度大小（张量的最后一维）

## 3. 核心算法实现

### RMSNorm 数学定义

RMSNorm 对输入张量的最后一维进行归一化：

```
output = input * weight / sqrt(mean(input²) + epsilon)
```

其中：
- `input`: 输入张量 `x`
- `weight`: 可学习的缩放参数 `w`
- `epsilon`: 数值稳定项（通常为 1e-6）
- `mean(input²) = sum(input²) / dim`

### 设备内核实现

#### `rmsnormKernel<BLOCK_SIZE, Tcompute, Tdata, Tweight>`
- **位置**: `rms_norm_moore.mu` 第 11-24 行
- **功能**: RMSNorm 的 MUSA 内核入口函数，封装通用的 `rmsnormBlock` 逻辑
- **模板参数**:
  - `BLOCK_SIZE`: 线程块大小（512/1024/2048）
  - `Tcompute`: 计算类型（通常为 `float`）
  - `Tdata`: 输入/输出数据类型（`half`/`float`/`__mt_bfloat16`）
  - `Tweight`: 权重数据类型（`half`/`float`/`__mt_bfloat16`）
- **参数**:
  - `y`: 输出张量指针
  - `stride_y_batch`, `stride_y_nhead`: 输出张量的 batch 和 head 维度步长
  - `x`: 输入张量指针
  - `stride_x_batch`, `stride_x_nhead`: 输入张量的 batch 和 head 维度步长
  - `w`: 权重指针（1D 张量，长度为 `dim`）
  - `nhead`: 注意力头数量（对于 2D 张量为 1）
  - `dim`: 归一化维度大小
  - `epsilon`: 数值稳定项
- **实现**:
  - 直接调用 `rmsnormBlock<BLOCK_SIZE, Tcompute>` 执行计算
  - 所有计算在设备端完成，使用 CUB 库进行块内归约

#### `rmsnormBlock<BLOCK_SIZE, Tcompute, Tdata, Tweight>`
- **位置**: `/home/qy/src/Infini/InfiniCore/src/infiniop/ops/rms_norm/cuda/kernel.cuh` 第 4-38 行
- **功能**: RMSNorm 的核心设备函数，每个线程块处理一个 batch 中的一个 head
- **并行策略**:
  - **网格维度**: `batch_size * nhead` 个线程块
  - **线程块维度**: `BLOCK_SIZE` 个线程
  - **数据划分**: 每个线程处理每隔 `BLOCK_SIZE` 个元素（stride 访问模式）
- **执行流程**:
  1. **索引计算**:
     - `batch_idx = blockIdx.x / nhead`: 当前 batch 索引
     - `head_idx = blockIdx.x % nhead`: 当前 head 索引
     - 计算输入/输出/权重张量的基地址指针

  2. **平方和计算** (第 26 行):
     ```cpp
     Tcompute ss = op::common_cuda::reduce_op::sumSquared<BLOCK_SIZE, Tdata, Tcompute>(x_ptr, dim);
     ```
     - 每个线程遍历 `x_ptr[threadIdx.x : dim : BLOCK_SIZE]`，计算局部平方和
     - 使用 CUB `BlockReduce` 进行块内归约，得到完整平方和
     - **时间复杂度**: O(dim / BLOCK_SIZE)
     - **空间复杂度**: O(BLOCK_SIZE) 用于共享内存

  3. **RMS 计算** (第 28-33 行):
     ```cpp
     __shared__ Tcompute rms;
     if (threadIdx.x == 0) {
         rms = Tcompute(rsqrtf(ss / Tcompute(dim) + epsilon));
     }
     __syncthreads();
     ```
     - Thread 0 计算均方根的倒数：`rms = 1 / sqrt(ss/dim + epsilon)`
     - 存储到共享内存供所有线程访问
     - `__syncthreads()` 确保所有线程看到正确的 `rms` 值

  4. **归一化与缩放** (第 35-37 行):
     ```cpp
     for (size_t i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
         y_ptr[i] = Tdata(Tcompute(x_ptr[i]) * Tcompute(w_ptr[i]) * rms);
     }
     ```
     - 每个线程更新其负责的元素
     - 计算公式：`y = x * w * rms`
     - 使用 `Tcompute` 精度避免精度损失
     - 最终转换为 `Tdata` 类型存储

### 归约操作实现

#### `op::common_cuda::reduce_op::sumSquared<BLOCK_SIZE, Tdata, Tcompute>`
- **位置**: `/home/qy/src/Infini/InfiniCore/src/infiniop/reduce/cuda/reduce.cuh` 第 17-32 行
- **功能**: 计算连续数组元素的平方和
- **算法**:
  1. **局部累加**: 每个线程遍历 `data_ptr[threadIdx.x : count : BLOCK_SIZE]`，计算 `sum(data[i]²)`
  2. **块内归约**: 使用 CUB 的 `BlockReduce::Sum` 进行 warp 级和块级归约
- **性能特征**:
  - 使用 CUB 库的高效归约原语，充分利用 warp shuffle 指令
  - 时间复杂度: O(count / BLOCK_SIZE + log(BLOCK_SIZE))
  - 空间复杂度: O(BLOCK_SIZE) 共享内存用于临时存储

## 4. API 接口

### 描述符创建接口

```cpp
infiniStatus_t op::rms_norm::moore::Descriptor::create(
    infiniopHandle_t handle,              // MUSA 设备句柄
    Descriptor **desc_ptr,                // 输出：描述符指针
    infiniopTensorDescriptor_t y_desc,    // 输出张量描述符 (batch, nhead, dim) 或 (batch, dim)
    infiniopTensorDescriptor_t x_desc,    // 输入张量描述符，形状与 y 相同
    infiniopTensorDescriptor_t w_desc,    // 权重张量描述符，形状为 (dim,)
    float epsilon                         // 数值稳定项，通常为 1e-6
);
// 返回值: INFINI_STATUS_SUCCESS 或错误码
```

### 计算执行接口

```cpp
infiniStatus_t op::rms_norm::moore::Descriptor::calculate(
    void *workspace,              // 工作空间指针（当前未使用，可传 nullptr）
    size_t workspace_size,        // 工作空间大小（当前需传 0）
    void *y,                      // 输出张量设备指针
    const void *x,                // 输入张量设备指针
    const void *w,                // 权重张量设备指针
    void *stream                  // MUSA stream 句柄
) const;
// 返回值: INFINI_STATUS_SUCCESS 或错误码
```

### 工作空间查询接口

```cpp
size_t op::rms_norm::moore::Descriptor::workspaceSize() const;
// 返回值: 0（当前实现不需要额外工作空间）
```

## 5. 使用示例

```cpp
// 示例：在 Moore GPU 上执行 RMSNorm 操作
#include "infiniop/ops/rms_norm/moore/rms_norm_moore.h"

// 1. 创建张量描述符（假设形状为 [batch_size, num_heads, head_dim]）
size_t batch_size = 32;
size_t num_heads = 16;
size_t head_dim = 128;
std::vector<size_t> shape = {batch_size, num_heads, head_dim};

infiniopTensorDescriptor_t x_desc, y_desc, w_desc;
infiniopCreateTensorDescriptor(x_desc, INFINI_DTYPE_F16, shape, /* strides */);
infiniopCreateTensorDescriptor(y_desc, INFINI_DTYPE_F16, shape, /* strides */);
infiniopCreateTensorDescriptor(w_desc, INFINI_DTYPE_F16, {head_dim}, /* strides */);

// 2. 创建 MUSA 设备句柄
infiniopHandle_t handle;
infiniopCreateHandle(&handle, INFINI_DEVICE_MOORE, /* device_id */ 0);

// 3. 创建 RMSNorm 描述符
op::rms_norm::moore::Descriptor* rmsnorm_desc;
float epsilon = 1e-6f;
auto status = op::rms_norm::moore::Descriptor::create(
    handle, &rmsnorm_desc, y_desc, x_desc, w_desc, epsilon
);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 4. 分配设备内存
void *d_x, *d_y, *d_w;
size_t x_bytes = batch_size * num_heads * head_dim * sizeof(half);
size_t w_bytes = head_dim * sizeof(half);
musaMalloc(&d_x, x_bytes);
musaMalloc(&d_y, x_bytes);
musaMalloc(&d_w, w_bytes);

// 5. 复制数据到设备（示例省略）
// memcpyH2D(d_x, host_x, x_bytes);
// memcpyH2D(d_w, host_w, w_bytes);

// 6. 获取或创建 MUSA stream
musaStream_t stream;
musaStreamCreate(&stream);

// 7. 执行 RMSNorm 计算
status = rmsnorm_desc->calculate(
    /* workspace */ nullptr,
    /* workspace_size */ 0,
    /* y */ d_y,
    /* x */ d_x,
    /* w */ d_w,
    /* stream */ stream
);
if (status != INFINI_STATUS_SUCCESS) {
    // 处理错误
}

// 8. 同步并取回结果（示例省略）
// musaStreamSynchronize(stream);
// memcpyD2H(host_y, d_y, x_bytes);

// 9. 清理资源
delete rmsnorm_desc;
musaFree(d_x);
musaFree(d_y);
musaFree(d_w);
musaStreamDestroy(stream);
infiniopDestroyHandle(handle);
```

## 6. 实现细节

### Moore/MUSA 平台适配

#### 设备架构支持
- **目标硬件**: 摩尔线程 S 系列 GPU（如 MTT S80、S3000 等）
- **编程模型**: MUSA (Moore Unified Shader Architecture)，兼容 CUDA API
- **核心库依赖**:
  - `<musa.h>`: 核心 MUSA 运行时 API
  - `<musa_fp16_mtgpu.h>`: FP16 半精度支持（MTGPU 扩展）
  - `<musa_bf16.h>`: BF16 bfloat16 支持
  - `<mublas.h>`: MUSA BLAS 库
  - `<mudnn.h>`: MUSA Deep Neural Network 库
  - `<cub/block/block_reduce.cuh>`: NVIDIA CUB 库（MUSA 兼容）

#### 线程配置策略
代码根据 Moore GPU 的计算能力（SM 架构）动态选择最优 block size：

```cpp
if (_opaque->internal->maxThreadsPerBlock() == MOORE_BLOCK_SIZE_1024) {
    launchKernel<MOORE_BLOCK_SIZE_1024>(...);
} else if (_opaque->internal->maxThreadsPerBlock() == MOORE_BLOCK_SIZE_512) {
    launchKernel<MOORE_BLOCK_SIZE_512>(...);
} else if (_opaque->internal->maxThreadsPerBlock() == MOORE_BLOCK_SIZE_2048) {
    launchKernel<MOORE_BLOCK_SIZE_2048>(...);
} else {
    return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
}
```

- **MOORE_BLOCK_SIZE_512**: 512 线程/块，适用于低端 GPU 或大维度数据
- **MOORE_BLOCK_SIZE_1024**: 1024 线程/块，标准配置，平衡 occupancy 和寄存器压力
- **MOORE_BLOCK_SIZE_2048**: 2048 线程/块，高端 GPU 优化配置

#### 数据类型支持矩阵

| 激活值类型 (atype) | 权重类型 (wtype) | 计算类型 (Tcompute) | 内核实例化 |
|-------------------|-----------------|-------------------|-----------|
| FP16 (`half`)     | FP16 (`half`)   | float             | 支持 |
| FP16 (`half`)     | BF16 (`__mt_bfloat16`) | float | 支持 |
| FP16 (`half`)     | FP32 (`float`)  | float             | 支持 |
| BF16 (`__mt_bfloat16`) | BF16 (`__mt_bfloat16`) | float | 支持 |
| BF16 (`__mt_bfloat16`) | FP16 (`half`)   | float             | 支持 |
| BF16 (`__mt_bfloat16`) | FP32 (`float`)  | float             | 支持 |
| FP32 (`float`)    | FP32 (`float`)  | float             | 支持 |

**设计说明**:
- 所有计算使用 `float` 精度避免半精度累加误差
- 权重和激活值类型可灵活组合（除了 FP32 激活值要求权重也是 FP32）
- 使用 MUSA 原生类型 `__mt_bfloat16` 表示 BF16

### 内存管理与性能优化

#### 内存访问模式
- **合并访问**: 每个线程处理连续的 `BLOCK_SIZE` 间隔元素，保证跨线程的合并内存访问
- **只读缓存**: 权重张量 `w` 通过 `const __restrict__` 修饰符提示编译器使用只读缓存
- **共享内存**:
  - CUB `BlockReduce` 使用的临时存储（约 `BLOCK_SIZE * sizeof(Tcompute)` 字节）
  - RMS 值广播用的标量共享内存（`sizeof(Tcompute)` 字节）

#### 并行效率
- **Occupancy 优化**: Block size 根据设备能力动态调整，最大化 SM 利用率
- **Warp 原语**: CUB 库使用 warp shuffle 指令实现高效块内归约，避免共享内存竞争
- **负载均衡**: 当 `dim` 不是 `BLOCK_SIZE` 整数倍时，线程间工作分配不均，但影响较小（通常 `dim >> BLOCK_SIZE`）

### 错误处理与边界条件

#### 张量验证逻辑
`RMSNormInfo::create` 执行严格的前置条件检查：

1. **类型兼容性**:
   - FP16/FP32/BF16 激活值允许 FP16/BF16/FP32 权重组合
   - FP32/FP64 激活值要求权重类型相同
   - 不支持的组合返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`

2. **形状一致性**:
   - 输入 `x` 和输出 `y` 形状必须相同
   - 权重 `w` 必须是 1D 张量，长度等于 `dim`
   - 支持 2D `(batch, dim)` 或 3D `(batch, nhead, dim)` 形状
   - 不支持的维度返回 `INFINI_STATUS_BAD_TENSOR_SHAPE`

3. **内存连续性**:
   - 最后一维（归一化维度）必须连续（stride 为 1）
   - 不满足返回 `INFINI_STATUS_BAD_TENSOR_STRIDES`

#### 运行时错误处理
- **工作空间不足**: 返回 `INFINI_STATUS_INSUFFICIENT_WORKSPACE`
- **设备不支持**: 返回 `INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED`
- **类型分发失败**: 返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`

### 设计模式

#### Pimpl (Pointer to Implementation) 模式
`Descriptor::Opaque` 结构体封装了平台相关的实现细节：
- **优点**: 头文件 `rms_norm_moore.h` 不依赖 MUSA 特定类型，保持 API 稳定性
- **实现**: `_opaque` 成员指向 `device::moore::Handle::Internal` 的共享指针

#### 模板元编程
- **类型分发**: `launchKernel` 函数使用模板特化避免运行时分支
  ```cpp
  if (atype == INFINI_DTYPE_F16 && wtype == INFINI_DTYPE_F16) {
      LAUNCH_KERNEL(half, half, float);  // 编译时实例化
  }
  ```
- **性能优势**: 编译器可为每个类型组合生成最优代码，消除虚函数开销

#### RAII 资源管理
- **智能指针**: `Opaque::internal` 使用 `std::shared_ptr` 自动管理设备句柄生命周期
- **析构函数**: `Descriptor::~Descriptor` 自动释放 `_opaque` 资源

### 性能特征

#### 计算复杂度
- **平方和归约**: O(dim / BLOCK_SIZE + log(BLOCK_SIZE)) ≈ O(dim)
- **归一化**: O(dim / BLOCK_SIZE) ≈ O(dim)
- **总复杂度**: O(dim) 每个样本

#### 内存带宽
- **读取**: 2 次 `dim` 元素（输入 `x` 和权重 `w`）
- **写入**: 1 次 `dim` 元素（输出 `y`）
- **算术强度**: 1 次乘法 + 1 次乘法 + 1 次平方 + 1 次加法 / 3 次内存访问 ≈ 1.33 FLOPs/Byte

#### 扩展性
- **Batch 维度**: 完全并行（每个 batch 一个线程块）
- **Head 维度**: 完全并行（每个 head 一个线程块）
- **Dim 维度**: 线程块内并行，使用归约树合并结果

## 7. 依赖关系

### 外部依赖
- **MUSA Runtime**: 摩尔线程 GPU 驱动和运行时环境
- **MUSA Toolkit**: 包含 `musa_fp16_mtgpu.h`, `musa_bf16.h` 等头文件
- **CUB Library**: CUDA 统一块级原语库（MUSA 兼容版本）

### 内部依赖
- **`op::rms_norm::RMSNormInfo`**: 张量元信息验证和存储
- **`op::common_cuda::reduce_op::sumSquared`**: 块内平方和归约
- **`device::moore::Handle::Internal`**: Moore 设备上下文和能力查询
- **`InfiniopDescriptor`**: 基类，提供设备 ID 和类型管理

## 8. 代码质量与维护性

### 代码规范
- **命名约定**: 驼峰命名（`rmsnormKernel`），下划线后缀（`_opaque`, `_info`）表示私有成员
- **常量定义**: 使用宏定义 block size 常量（`MOORE_BLOCK_SIZE_1024` 等）
- **错误处理**: 统一使用 `CHECK_STATUS` 和 `CHECK_RESULT` 宏进行错误传播

### 可扩展性
- **新硬件支持**: 添加新的 block size 常量和对应的 `launchKernel` 分支
- **新数据类型**: 在 `launchKernel` 中添加新的类型组合条件分支
- **新内核优化**: 修改 `rmsnormBlock` 实现即可，无需改动 API

### 测试建议
- **单元测试**: 验证所有支持的数据类型组合
- **正确性验证**: 与 PyTorch `F.rms_norm` 或 NumPy 实现对比结果
- **性能基准**: 测试不同 `dim` 和 `batch_size` 下的吞吐量（elements/s）
- **边界测试**: 验证 `dim` 不是 `BLOCK_SIZE` 整数倍时的正确性
- **数值稳定性**: 测试极端输入（全零、极大值、极小值）下的 `epsilon` 作用
