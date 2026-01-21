# NVIDIA LP-Norm 算子 CUDA 实现文档

## 1. 模块概述

本模块实现了 Lp 范数归一化操作（Lp-Norm Normalization）的 NVIDIA GPU CUDA 后端。该模块针对 NVIDIA GPU 进行了深度优化，支持连续内存和非连续内存（strided）张量，提供了基于 Block 和 Warp 的两种并行归约策略，以适应不同的数据规模和 GPU 架构特性。

## 2. 模块结构

- **`lp_norm_nvidia.cuh`**: 头文件，通过宏 `DESCRIPTOR(nvidia)` 定义 `op::lp_norm::nvidia::Descriptor` 类的接口，继承自通用的 Lp-Norm 操作符框架
- **`lp_norm_nvidia.cu`**: 核心实现文件，包含 CUDA 内核启动逻辑、算子描述符实现以及内核调度策略

## 3. 核心类与组件

### 3.1 `Descriptor` 类
- **位置**: 通过 `lp_norm.h` 的宏展开定义，实现在 `lp_norm_nvidia.cu`
- **主要功能**: 封装 Lp-Norm 操作的 NVIDIA GPU 实现描述符，管理算子生命周期和执行调度

#### 3.1.1 内部结构体 `Opaque`
```cpp
struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};
```
- **用途**: 持有 NVIDIA 设备句柄的内部实现，提供 GPU 硬件能力查询（如最大线程块大小）

#### 3.1.2 关键成员变量
- **`_opaque`**: 不透明句柄，封装 GPU 设备相关信息
- **`_info`**: `LPNormInfo` 实例，存储张量形状、步长、数据类型、归约轴、范数阶数 p 等元数据
- **`_workspace_size`**: 工作空间大小，用于存储设备端的步长和形状数组（`ndim * (2 * sizeof(ptrdiff_t) + sizeof(size))`）

#### 3.1.3 核心方法

**`create(...)` - 算子描述符创建工厂**
```cpp
static infiniStatus_t create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    int axis, int p, float eps);
```
- **功能**: 验证输入/输出张量形状、数据类型、轴参数；计算工作空间大小；初始化描述符
- **关键逻辑**:
  - 调用 `LPNormInfo::createLPNormInfo()` 生成归约元信息
  - 计算工作空间大小为 `ndim * (2 * sizeof(ptrdiff_t) + sizeof(size))`，用于存储设备端的步长和形状数组
- **返回值**: `INFINI_STATUS_SUCCESS` 或相应的错误码

**`calculate(...)` - 执行 Lp-Norm 归一化计算**
```cpp
infiniStatus_t calculate(
    void *workspace, size_t workspace_size,
    void *y, const void *x,
    void *stream_) const;
```
- **功能**: 根据数据类型和 GPU 架构调度相应的 CUDA 内核
- **数据类型支持**:
  - `INFINI_DTYPE_F16` (half)
  - `INFINI_DTYPE_F32` (float)
  - `INFINI_DTYPE_BF16` (__nv_bfloat16)
- **内核选择策略**:
  1. 检查 GPU 的 `maxThreadsPerBlock()` 能力
  2. 根据线程块大小（1024/512/4096）选择对应的 BLOCK_SIZE 模板参数
  3. 调用 `launchKernel<BLOCK_SIZE, Tdata>()` 执行计算

## 4. CUDA 内核实现

### 4.1 内核模板包装器

#### 4.1.1 `blockLPNorm<Tdata, BLOCK_SIZE>`
```cpp
template <typename Tdata, unsigned int BLOCK_SIZE>
INFINIOP_CUDA_KERNEL blockLPNorm(
    Tdata *y, const Tdata *x,
    float p, size_t dimsize,
    ptrdiff_t stride, float eps);
```
- **功能**: Block 级并行归约内核包装器，适用于连续内存张量
- **适用场景**: `dimsize > 1024` 时使用
- **底层实现**: 调用 `../cuda/kernel.cuh` 中的 `blockLPNormKernel()`

#### 4.1.2 `blockLPNormStrides<Tdata, BLOCK_SIZE>`
```cpp
template <typename Tdata, unsigned int BLOCK_SIZE>
INFINIOP_CUDA_KERNEL blockLPNormStrides(
    Tdata *y, const Tdata *x,
    const ptrdiff_t *output_strides,
    const ptrdiff_t *input_strides,
    const size_t *shape, int ndim,
    float p, size_t dimsize, float eps);
```
- **功能**: Block 级并行归约内核包装器，适用于非连续内存（strided）张量
- **约束**: 仅支持 `axis == ndim - 1`（最后一维归约）
- **底层实现**: 调用 `../cuda/kernel.cuh` 中的 `blockLPNormStridesKernel()`

#### 4.1.3 `warpLPNorm<Tdata, BLOCK_SIZE_x, BLOCK_SIZE_y>`
```cpp
template <typename Tdata, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y>
INFINIOP_CUDA_KERNEL warpLPNorm(
    Tdata *y, const Tdata *x,
    float p, size_t othersize,
    size_t dimsize,
    ptrdiff_t stride, float eps);
```
- **功能**: Warp 级并行归约内核包装器，适用于连续内存张量
- **适用场景**: `dimsize <= 1024` 时使用
- **默认配置**: `BLOCK_SIZE_x = 32`, `BLOCK_SIZE_y = 32`（32 个 warp，每个 warp 32 线程）
- **底层实现**: 调用 `../cuda/kernel.cuh` 中的 `warpLPNormKernel()`

#### 4.1.4 `warpLPNormStrides<Tdata, BLOCK_SIZE_x, BLOCK_SIZE_y>`
```cpp
template <typename Tdata, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y>
INFINIOP_CUDA_KERNEL warpLPNormStrides(
    Tdata *y, const Tdata *x,
    const ptrdiff_t *output_strides,
    const ptrdiff_t *input_strides,
    const size_t *shape, int ndim,
    float p, size_t othersize, size_t dimsize, float eps);
```
- **功能**: Warp 级并行归约内核包装器，适用于非连续内存（strided）张量
- **约束**: 仅支持 `axis == ndim - 1`（最后一维归约）
- **默认配置**: `BLOCK_SIZE_x = 32`, `BLOCK_SIZE_y = 32`
- **底层实现**: 调用 `../cuda/kernel.cuh` 中的 `warpLPNormStridesKernel()`

### 4.2 `launchKernel<BLOCK_SIZE, Tdata>()` - 内核调度核心函数
```cpp
template <unsigned int BLOCK_SIZE, typename Tdata>
infiniStatus_t launchKernel(
    const LPNormInfo &info,
    Tdata *y, const Tdata *x,
    cudaStream_t stream, void *workspace);
```

#### 4.2.1 决策树逻辑
```
启动流程
├── 1. 分配工作空间指针
│   ├── input_strides_cuda[0..ndim-1]
│   ├── output_strides_cuda[0..ndim-1]
│   └── shape_cuda[0..ndim-1]
│
├── 2. 异步拷贝元数据到 GPU (cudaMemcpyAsync)
│   ├── 拷贝 input_strides
│   ├── 拷贝 output_strides
│   └── 拷贝 input_shape
│
└── 3. 选择内核策略
    ├── 条件: info.continuous == true
    │   ├── dimsize > 1024 → blockLPNorm (1D grid, BLOCK_SIZE threads)
    │   └── dimsize <= 1024 → warpLPNorm (2D grid, 32x32 threads)
    │
    └── 条件: info.continuous == false (strided)
        ├── info.axis == ndim - 1 (最后一维)
        │   ├── dimsize > 1024 → blockLPNormStrides
        │   └── dimsize <= 1024 → warpLPNormStrides
        │
        └── 否则 → 返回 INFINI_STATUS_BAD_PARAM
```

#### 4.2.2 关键决策点
- **连续内存判断 (`info.continuous`)**: 通过 `LPNormInfo` 预计算，检查输入/输出步长是否符合 C 风格连续内存模式
- **归约维度大小 (`dimsize`)**: 阈值 1024，大于则使用 Block 归约，小于等于则使用 Warp 归约
- **非连续内存约束**: Strided 版本仅支持 `axis == ndim - 1`，其他轴返回错误码
- **线程块配置**:
  - Block 归约: `<<<num_blocks, BLOCK_SIZE, 0, stream>>>`
  - Warp 归约: `<<<num_block_x, (32, 32, 1), 0, stream>>>`

## 5. Lp-Norm 算法实现细节

### 5.1 数学定义
对于输入张量 x，在 axis 维度上计算 Lp 范数并归一化：

```
y = x / (||x||_p + eps)

其中 ||x||_p = (Σ |x_i|^p)^(1/p)
```

**数值稳定性优化**: 采用两阶段归一化
1. 第一阶段: 计算 `max_val = max(|x|)`，得到 `x' = x / max(max_val, eps)`
2. 第二阶段: 计算 `||x'||_p`，得到 `y = x' / (||x'||_p + eps)`

### 5.2 Block 归约内核算法 (kernel.cuh)

#### `blockLPNormKernel<T, BLOCK_SIZE>()`
```cpp
template <typename T, unsigned int BLOCK_SIZE>
__device__ void blockLPNormKernel(
    T const *input, T *output,
    float p, size_t dimsize,
    ptrdiff_t stride, float eps);
```

**算法流程**:
1. **线程索引计算**:
   ```cpp
   int tid = blockIdx.x % stride + (blockIdx.x - blockIdx.x % stride) * dimsize;
   ```
   - `blockIdx.x % stride`: 当前归约组内的偏移
   - `(blockIdx.x - blockIdx.x % stride) * dimsize`: 跳到正确的归约组起始位置

2. **第一阶段：全局最大值归约**
   - 每个线程遍历 `dimsize`，步长为 `BLOCK_SIZE`，计算局部最大值
   - 使用 CUB 的 `BlockReduce::Reduce(cuda::maximum())` 进行块内归约
   - 线程 0 将结果写入共享内存 `global_max`

3. **第二阶段：幂和归约**
   - 所有线程同步，读取 `global_max`
   - 计算归一化因子 `global_max_inv = 1.0 / max(global_max, eps)`
   - 每个线程计算局部幂和 `Σ (input * global_max_inv)^p`
   - 使用 CUB 的 `BlockReduce::Sum()` 进行块内归约
   - 线程 0 计算 `p_total = (p_block)^(1/p)`

4. **第三阶段：输出写回**
   - 计算最终归一化因子 `inv = (1.0 / (p_total + eps)) * global_max_inv`
   - 每个线程将输入乘以 `inv` 并写回输出

**时间复杂度**: O(dimsize / BLOCK_SIZE) 并行度

#### `blockLPNormStridesKernel<T, BLOCK_SIZE>()`
- **关键区别**: 索引计算通过步长数组进行动态地址生成
  ```cpp
  int ind_i = 0; // input id
  int ind_o = 0; // output id
  int tid = blockIdx.x;
  for (int j = ndim - 2; j >= 0; j--) {
      ind_i += (tid % (int)shape[j]) * (int)input_strides[j];
      ind_o += (tid % (int)shape[j]) * (int)output_strides[j];
      tid = tid / (int)shape[j];
  }
  ```
- **约束**: 仅支持 `axis == ndim - 1`，因此最后一维可以直接线性访问 `input[ind_i + ind]`

### 5.3 Warp 归约内核算法

#### `warpLPNormKernel<T, BLOCK_SIZE_x, BLOCK_SIZE_y>()`
```cpp
template <typename T, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y>
__device__ void warpLPNormKernel(
    T const *input, T *output,
    float p, size_t othersize, size_t dimsize,
    ptrdiff_t stride, float eps);
```

**线程组织**:
- **BLOCK_SIZE_x = 32**: 每个 warp 的线程数（warp 固定 32 线程）
- **BLOCK_SIZE_y = 32**: 线程块内的 warp 数量
- **总线程数**: 32 x 32 = 1024

**算法流程**:
1. **2D 线程索引**:
   ```cpp
   int otherIdx = blockIdx.x * blockDim.y + threadIdx.y;  // 选择哪个 warp
   int tid = otherIdx % stride + (otherIdx - otherIdx % stride) * dimsize;
   ```

2. **Warp 级归约 (使用 __shfl_xor_sync)**:
   - 最大值归约: `WarpAllReduce<MaxOp, float, 32>(local_max)`
     ```cpp
     for (int mask = 16; mask > 0; mask /= 2) {
         val = MaxOp<float>()(val, __shfl_xor_sync(0xffffffff, val, mask));
     }
     ```
   - 幂和归约: `WarpAllReduce<SumOp, float, 32>(p_data)`

3. **共享内存优化**:
   - `__shared__ float p_total[BLOCK_SIZE_y]`: 存储每个 warp 的 Lp 范数
   - `__shared__ float p_max[BLOCK_SIZE_y]`: 存储每个 warp 的最大值
   - 每个 warp 的线程 0 负责写回共享内存

**优势**: Warp 内通信无需显式 `__syncthreads()`，减少同步开销

#### `warpLPNormStridesKernel<T, BLOCK_SIZE_x, BLOCK_SIZE_y>()`
- **索引计算**: 与 Block Strides 版本类似，但增加了 `blockIdx.x * blockDim.y + threadIdx.y` 的 2D 索引逻辑
- **约束**: 同样仅支持 `axis == ndim - 1`

## 6. 使用示例

```cpp
// 1. 创建张量描述符
infiniopTensorDescriptor_t x_desc, y_desc;
infiniopCreateTensorDescriptor(handle, &x_desc, ...);
infiniopCreateTensorDescriptor(handle, &y_desc, ...);

// 2. 创建 Lp-Norm 算子描述符
op::lp_norm::nvidia::Descriptor *lp_norm_desc;
infiniStatus_t status = op::lp_norm::nvidia::Descriptor::create(
    handle,           // infiniopHandle_t
    &lp_norm_desc,    // 输出描述符指针
    y_desc,           // 输出张量描述符
    x_desc,           // 输入张量描述符
    -1,               // axis=-1 (最后一维)
    2,                // p=2 (L2 范数)
    1e-5f             // eps=1e-5
);

// 3. 分配工作空间
size_t workspace_size = lp_norm_desc->workspaceSize();
void *workspace;
cudaMalloc(&workspace, workspace_size);

// 4. 分配输入/输出内存
half *x, *y;
cudaMalloc(&x, tensor_size * sizeof(half));
cudaMalloc(&y, tensor_size * sizeof(half));

// 5. 执行计算
cudaStream_t stream;
cudaStreamCreate(&stream);
status = lp_norm_desc->calculate(
    workspace, workspace_size,  // 工作空间
    y, x,                       // 输出/输入指针
    stream                      // CUDA 流
);

// 6. 清理资源
cudaFree(workspace);
cudaFree(x);
cudaFree(y);
delete lp_norm_desc;
cudaStreamDestroy(stream);
```

## 7. 实现细节与优化策略

### 7.1 内存管理
- **工作空间策略**: 动态分配设备端内存存储元数据（步长、形状），避免每次内核调用时的重复传输
- **异步内存拷贝**: 使用 `cudaMemcpyAsync` 在流上重叠数据传输和计算
- **连续内存优化**: 对连续内存张量直接使用固定步长，避免动态地址计算开销

### 7.2 并行计算优化
- **CUB 库集成**: 使用 `cub::BlockReduce` 进行高效的块级归约，充分利用共享内存和 warp 同步
- **双级并行策略**:
  - **Block 归约**: 适用于大维度 (`dimsize > 1024`)，最大化线程级并行
  - **Warp 归约**: 适用于小/中维度 (`dimsize <= 1024`)，减少块间同步开销
- **数值稳定性**: 使用两阶段归一化（先除以最大值，再计算 Lp 范数），避免浮点数溢出

### 7.3 性能关键路径
1. **内存访问模式**: 连续内存版本使用线性访问，strided 版本通过步长数组支持通用张量布局
2. **归约操作**: Block 归约的同步开销为 O(log BLOCK_SIZE)，Warp 归约为零开销（硬件级）
3. **浮点运算**: 使用 `__fdividef(x, y)` 内置函数快速计算倒数（在某些 GPU 架构上比除法快）

### 7.4 GPU 架构适配
- **线程块大小自适应**: 根据 `maxThreadsPerBlock()` 选择 512/1024/4096 线程块配置
- **CUDA 版本兼容**: 对 CUDART_VERSION >= 12.9 使用 `::cuda::maximum()`，否则使用 `cub::Max()`

### 7.5 错误处理
- **数据类型检查**: 仅支持 F16/F32/BF16，其他类型返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`
- **步长验证**: 要求最后一维步长为 1，返回 `INFINI_STATUS_BAD_TENSOR_STRIDES`
- **Strided 约束**: 非最后一维归约返回 `INFINI_STATUS_BAD_PARAM`
- **架构不支持**: 未知线程块大小返回 `INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED`

### 7.6 设计模式
- **策略模式**: 通过 `info.continuous` 和 `dimsize` 动态选择内核实现
- **模板元编程**: 编译期生成不同数据类型和线程块配置的内核特化
- **Pimpl 惯例**: 使用 `Opaque` 结构体隐藏 GPU 设备句柄实现细节
- **RAII 资源管理**: 析构函数自动释放 `_opaque` 内存

## 8. 依赖关系

### 8.1 外部依赖
- **CUDA Toolkit**: `cub/block/block_reduce.cuh` (CUB 库的块级归约原语)
- **Infini 内部模块**:
  - `../../../devices/nvidia/nvidia_common.cuh`: NVIDIA 设备公共定义
  - `../../../devices/nvidia/nvidia_kernel_common.cuh`: CUDA 内核公共宏
  - `../../../reduce/cuda/reduce.cuh`: 通用归约操作（可能用于辅助）
  - `../lp_norm.h`: Lp-Norm 操作符通用接口
  - `../cuda/kernel.cuh`: CUDA 内核算法实现
  - `../../operator.h`: 操作符基类
  - `../../../tensor.h`: 张量描述符

### 8.2 编译依赖
- **C++ 标准**: C++11 或更高（支持 `std::shared_ptr`, `std::vector`）
- **CUDA 标准**: 需要 CUDA Runtime API，支持 CUB 库

## 9. 性能特征

### 9.1 时间复杂度
- **Block 归约**: O(dimsize / BLOCK_SIZE + log(BLOCK_SIZE))
- **Warp 归约**: O(dimsize / 32 + log(32)) = O(dimsize / 32 + 5)

### 9.2 空间复杂度
- **工作空间**: O(ndim)，用于存储设备端元数据
- **共享内存**:
  - Block 归约: `sizeof(typename BlockReduce::TempStorage)`（CUB 自动管理）
  - Warp 归约: `BLOCK_SIZE_y * sizeof(float) * 2 = 32 * 4 * 2 = 256 bytes`

### 9.3 吞吐量优化
- **连续内存**: 高吞吐量（合并内存访问）
- **非连续内存**: 中等吞吐量（需要动态地址计算）
- **小维度**: Warp 归约更高效（减少块同步）
- **大维度**: Block 归约更高效（最大化并行度）

## 10. 约束与限制

1. **数据类型限制**: 仅支持 F16/F32/BF16，不支持整数类型
2. **Strided 张量限制**: 非连续内存版本仅支持 `axis == ndim - 1`（最后一维归约）
3. **步长要求**: 最后一维步长必须为 1（C 风格连续内存）
4. **GPU 架构要求**: 仅适配 NVIDIA GPU，不支持其他硬件后端
5. **轴归一化**: 支持 `axis ∈ [-ndim, ndim-1]`，自动处理负轴索引

---

**文档版本**: 1.0
**最后更新**: 2026-01-14
**适用 CUDA 版本**: 11.0+ (CUB 库集成)
**作者**: Infini 团队
