# Moore Rearrange 算子核心实现文档

本模块实现了针对 Moore (摩尔线程) GPU 架构的张量重排（rearrange）算子，通过高度优化的 MUSA 内核实现高效的数据重排操作。该实现支持任意维度的张量重塑、转置和步长变换，采用智能的块-网格分层策略和约束检查机制，确保内存访问的最优局部性和安全性。

## 1. 模块结构

- **`rearrange_moore.h`**: Moore 平台的算子描述符宏定义，通过 `DESCRIPTOR(moore)` 宏生成 `op::rearrange::moore::Descriptor` 类接口
- **`rearrange_kernel.h`**: MUSA 内核函数的模板化定义和内核选择逻辑，包含 75 个特化内核变体（5×5×3 组合）和运行时分发机制
- **`rearrange_moore.mu`**: 算子的宿主端实现，包含参数准备算法、内核启动调度和 MUSA 平台适配逻辑

## 2. 核心类与数据结构

### `op::rearrange::moore::Descriptor`
- **位置**: `rearrange_moore.h` (宏定义), `rearrange_moore.mu` (实现)
- **主要功能**: Moore 平台的 rearrange 算子描述符，继承自 `InfiniopDescriptor`
- **关键成员**:
  - `_opaque`: 指向 `Opaque` 结构的不透明指针，封装设备内部状态
  - `_meta`: `utils::RearrangeMeta` 类型，存储张量的维度、步长和单元大小等元数据
- **核心方法**:
  - `create(handle, desc_ptr, y_desc, x_desc)`: 静态工厂方法，验证输入/输出张量的形状和类型一致性，通过 `RearrangeMeta::create` 生成元数据，构造描述符实例
  - `calculate(y, x, stream)`: 执行重排计算，调用 `prepareRearrangeParams` 准备内核参数，根据块大小选择合适的模板内核并启动
- **生命周期**: 由工厂方法 `create` 构造，用户负责销毁，析构函数释放 `_opaque` 内存

### `Descriptor::Opaque`
- **位置**: `rearrange_moore.mu:12-14`
- **主要功能**: 封装 Moore 设备句柄的内部状态
- **关键成员**:
  - `internal`: `std::shared_ptr<device::moore::Handle::Internal>`，共享指针管理设备上下文生命周期，提供设备属性查询（如 `maxThreadsPerBlock()`）

### `RearrangeParams`
- **位置**: `rearrange_kernel.h:216-227`
- **主要功能**: 封装传递给 MUSA 内核的所有参数
- **关键成员**:
  - `block_len`: `std::vector<ARRAY_TYPE_SIZE>`，块维度各轴长度数组
  - `src_block_stride` / `dst_block_stride`: `std::vector<ARRAY_TYPE_STRIDE>`，块维度在源/目标张量中的字节步长
  - `grid_len`: `std::vector<ARRAY_TYPE_SIZE>`，网格维度各轴长度数组
  - `src_grid_stride` / `dst_grid_stride`: `std::vector<ARRAY_TYPE_STRIDE>`，网格维度在源/目标张量中的字节步长
  - `block_dim`: `size_t`，块维度数量
  - `block_len_total`: `size_t`，块中元素总数（决定线程块大小）
  - `constraints`: `std::vector<Constraint<ARRAY_TYPE_SIZE>>`，边界约束条件，用于防止越界访问
  - `unit_size`: `size_t`，数据单元大小（1/2/4/8/16/32 字节）

### `Constraint<ElementType>`
- **位置**: `rearrange_kernel.h:20-26`
- **主要功能**: 描述分割维度的边界约束，防止内核访问越界
- **关键成员**:
  - `grid_idx`: `ElementType`，约束条件在 grid 数组中的索引
  - `block_idx`: `ElementType`，约束条件在 block 数组中的索引
  - `grid_div_block`: `ElementType`，grid 维度相对于 block 维度的倍数
  - `total_len`: `ElementType`，该维度的总长度限制
- **使用场景**: 当某个维度无法被块大小整除时，生成约束条件，内核通过 `grid_idx * grid_div_block + block_idx < total_len` 判断是否越界

### `ArrayStruct<ArrSize, ArrayType>`
- **位置**: `rearrange_kernel.h:14-17`
- **主要功能**: 固定大小数组的结构体包装，用于向内核传递可变长度数组参数
- **实现细节**: 模板参数 `ArrSize` 指定数组大小（支持 1-5），`ArrayType` 为元素类型（`ARRAY_TYPE_SIZE` 或 `ARRAY_TYPE_STRIDE`）
- **设计原因**: MUSA 内核不支持 C++ 的 `std::vector`，需要使用静态数组传递参数

## 3. 内核函数接口

### `rearrange_unit_[类型]_block_[N]_grid_[M]_constrain_[C]`
```cpp
extern "C" __global__ void rearrange_unit_[T]_block_[B]_grid_[G]_constrain_[C](
    void *__restrict__ dst,                              // 目标张量数据指针
    const void *__restrict__ src,                        // 源张量数据指针
    const size_t block_dim,                              // 块维度数量
    const size_t block_len_total,                        // 块中元素总数
    const ArrayStruct<B, ARRAY_TYPE_SIZE> block_len,     // 块各轴长度
    const ArrayStruct<B, ARRAY_TYPE_STRIDE> src_block_stride,  // 源块步长（字节）
    const ArrayStruct<B, ARRAY_TYPE_STRIDE> dst_block_stride,  // 目标块步长（字节）
    const ArrayStruct<G, ARRAY_TYPE_SIZE> grid_len,      // 网格各轴长度
    const ArrayStruct<G, ARRAY_TYPE_STRIDE> src_grid_stride,  // 源网格步长（字节）
    const ArrayStruct<G, ARRAY_TYPE_STRIDE> dst_grid_stride,  // 目标网格步长（字节）
    [const ArrayStruct<C, Constraint<...>> constraints]  // 约束条件（C>0 时存在）
);
```
- **功能**: 执行单个数据单元的重排拷贝，每个线程处理一个元素
- **类型变体**: `uchar1`/`uchar2`/`float1`/`float2`/`float4`/`double4`，对应单元大小 1/2/4/8/16/32 字节
- **维度组合**: `block_dim` ∈ [1,5]，`grid_dim` ∈ [1,5]，`constraint_num` ∈ [0,2]，共生成 **75 个特化内核**
- **算法流程**:
  1. **线程 0 计算基础偏移**（第 58-86 行）：遍历 grid 维度，计算当前 block 在 src/dst 中的字节偏移，处理约束条件的 grid 索引倍数
  2. **共享内存同步**（第 90 行）：`__syncthreads()` 确保所有线程可见偏移值
  3. **所有线程计算 block 内偏移**（第 100-115 行）：遍历 block 维度，结合线程索引 `threadIdx.x` 计算最终偏移，检查约束条件防止越界
  4. **执行内存拷贝**（第 128 行）：通过字节偏移指针进行类型化数据拷贝
- **时间复杂度**: O(block_dim + grid_dim)，每个线程的索引计算复杂度与维度数量线性相关
- **空间复杂度**: O(1) 常量额外内存，仅使用少量共享内存变量

### `getRearrangeKernel(const RearrangeParams &params)`
```cpp
utils::Result<void *> getRearrangeKernel(const RearrangeParams &params);
```
- **位置**: `rearrange_kernel.h:229-338`
- **功能**: 根据参数类型选择匹配的内核函数指针
- **选择逻辑**: 嵌套 switch 语句，依次按 `block_num` → `grid_num` → `constraint_num` → `unit_size` 进行分发
- **返回值**: `utils::Result<void *>`，成功时返回内核函数指针，失败时返回 `INFINI_STATUS_BAD_PARAM`
- **性能考虑**: 编译时宏展开，运行时无额外开销

### `launchKernel<BLOCK_SIZE>(...)`
```cpp
template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(
    void *y, const void *x, size_t grid_size,
    const RearrangeParams &params, size_t unit_size, musaStream_t stream
);
```
- **位置**: `rearrange_moore.mu:369-450`
- **功能**: 模板化的内核启动包装器，处理参数打包和 MUSA API 调用
- **关键步骤**:
  1. 调用 `getRearrangeKernel` 获取内核函数
  2. **对齐线程块大小**（第 393 行）：`((block_len_total + 31) / 32) * 32`，向上取整到 32 的倍数（MUSA Warp 大小）
  3. 构建 `void *args[]` 参数数组（第 418-429 行）
  4. 调用 `musaLaunchKernel` 启动内核（第 435-440 行）
  5. 同步设备并检查错误（第 443-447 行）
- **MUSA 特性适配**: Moore 架构要求线程块大小必须是 32 的整数倍，否则可能导致性能下降或执行错误
- **模板参数**: `BLOCK_SIZE` 为 512 或 1024，限制最大线程数以适配不同硬件代际

## 4. 使用示例

```cpp
// 示例：在 Moore GPU 上执行张量转置 (4D Tensor: [N, C, H, W] -> [N, H, W, C])
#include "rearrange_moore.h"

// 1. 创建算子描述符
infiniopHandle_t handle;          // 假设已初始化
infiniTensorDescriptor_t x_desc;  // 输入张量: [32, 64, 224, 224], dtype=FP32
infiniTensorDescriptor_t y_desc;  // 输出张量: [32, 224, 224, 64], dtype=FP32

op::rearrange::moore::Descriptor *rearrange_desc;
infiniStatus_t status = op::rearrange::moore::Descriptor::create(
    handle, &rearrange_desc, y_desc, x_desc
);
// create 内部会调用 RearrangeMeta::create 生成步长信息:
// src_strides: [64*224*224*4, 224*224*4, 224*4, 4]
// dst_strides: [224*224*64*4, 224*64*4, 64*4, 4]

// 2. 分配设备内存
void *d_x, *d_y;
size_t x_bytes = 32 * 64 * 224 * 224 * sizeof(float);
musaMalloc(&d_x, x_bytes);
musaMalloc(&d_y, x_bytes);

// 3. 执行重排计算
musaStream_t stream;
musaStreamCreate(&stream);

status = rearrange_desc->calculate(d_y, d_x, stream);
// calculate 内部流程:
// - prepareRearrangeParams: 选择 block=[C], grid=[N, H, W]，block_len_total=64
// - 对齐 block_size: ceil(64/32)*32 = 64
// - 启动内核: rearrange_unit_float1_block_1_grid_3_constrain_0<<<[32,224,224], 64, ...>>>(...)

// 4. 同步并清理
musaStreamSynchronize(stream);
musaFree(d_x);
musaFree(d_y);
delete rearrange_desc;
```

## 5. 实现细节

### 5.1 维度分割与参数准备算法

**核心函数**: `prepareRearrangeParams(const utils::RearrangeMeta &original_meta, int max_threads)` (rearrange_moore.mu:79-366)

**算法目标**: 将高维张量映射到 GPU 的 block-grid 两级并行架构，最大化内存局部性和计算效率。

**步骤 1: 单元大小优化**
```cpp
auto meta_result = original_meta.distributeUnit({32, 16, 8, 4, 2, 1});
```
- 尝试将数据单元对齐到 2 的幂次方，利用向量化内存访问（如 float4 一次读取 16 字节）
- 优先选择大单元（32 字节），失败时依次降级到 1 字节

**步骤 2: 维度排序**
```cpp
// 按 src_stride 降序排序，确保连续维度优先
std::sort(src_strides_desc_idx.begin(), src_strides_desc_idx.end(),
          [&dims](size_t a, size_t b) {
              return std::abs(dims[a].src_stride) > std::abs(dims[b].src_stride);
          });
```
- 将源张量中步长最大的维度排在前面，优先处理内存连续的维度

**步骤 3: 贪心维度分配**
```cpp
while (src_choose_idx > 0 && dst_choose_idx > 0) {
    // 情况 1: src_idx == dst_idx (源和目标维度相同)
    if (block_elements * len <= block_size) {
        // 完全放入 block
        block_dim_choose[idx] = true;
    } else {
        // 分割维度: num_per_block = block_size / block_elements
        // num_per_grid = ceil(len / num_per_block)
    }

    // 情况 2: src_idx != dst_idx (源和目标维度不同)
    // 使用平方根平衡: sqrt(block_size / block_elements / (src_elem/dst_elem))
}
```
- **Block 维度选择**: 将较小的、相对连续的维度分配给 block，由线程并行处理
- **Grid 维度选择**: 将较大的、分散的维度分配给 grid，由多个 block 串行处理
- **分割策略**: 当维度长度超过 block 容量时，按 `num_per_block` 切分，剩余部分形成 grid 维度

**步骤 4: 约束生成**
```cpp
for (size_t i = 0; i < split_dims.size(); ++i) {
    if (split_dims[i].dim_len % split_dims[i].num_per_block == 0) {
        continue; // 整除，无需约束
    }
    Constraint constraint;
    constraint.grid_idx = split_dims[i].array_struct_idx_grid;
    constraint.block_idx = split_dims[i].array_struct_idx_block;
    constraint.grid_div_block = split_dims[i].num_per_block;
    constraint.total_len = split_dims[i].dim_len;
    constraints.push_back(constraint);
}
```
- 只对无法整除的分割维度生成约束，最多支持 2 个约束条件
- 内核通过约束判断：`grid_multiple + block_idx < total_len`，提前退出越界线程

**算法复杂度**:
- 时间: O(ndim × log(ndim))，主要是排序开销
- 空间: O(ndim)，存储维度和分割信息

### 5.2 内核特化与代码生成

**宏定义层次结构** (rearrange_kernel.h:172-213):
```cpp
DEFINE_KERNELS_BY_CONSTRAINT(block_array_size, grid_array_size)
  └─ DEFINE_KERNELS_BY_TYPE(constraint_num, block_array_size, grid_array_size)
      └─ DEFINE_REARRANGE_KERNEL(Tmem_type, constraint_num, block_array_size, grid_array_size)
```

**生成策略**:
- **类型层面**: 6 种数据类型 × 3 种约束数量 = 18 个内核
- **维度组合**: 5 种 block 维度 × 5 种 grid 维度 = 25 种组合
- **总计**: 18 × 25 = **450 个内核函数**（实际展开为 75 个，因为约束数量作为模板参数编译时展开）

**模板特化示例**:
```cpp
// unit_size=4, block_dim=2, grid_dim=3, constraint_num=0
DEFINE_REARRANGE_KERNEL(float1, 0, 2, 3)
// 展开为:
extern "C" __global__ void rearrange_unit_float1_block_2_grid_3_constrain_0(...)

// unit_size=16, block_dim=5, grid_dim=1, constraint_num=2
DEFINE_REARRANGE_KERNEL(float4, 2, 5, 1)
// 展开为:
extern "C" __global__ void rearrange_unit_float4_block_5_grid_1_constrain_2(
    ..., const ArrayStruct<2, Constraint<ARRAY_TYPE_SIZE>> constraints
)
```

### 5.3 MUSA 平台适配

**1. Warp 对齐** (rearrange_moore.mu:389-399):
```cpp
size_t aligned_block_size = ((block_len_total + 31) / 32) * 32;
```
- Moore GPU 以 32 线程为 Warp（基本调度单位）
- 线程块大小必须是 32 的倍数，否则性能严重下降
- 使用向上取整确保完整 Warp 调度

**2. 线程数限制** (rearrange_moore.mu:495-502):
```cpp
if (block_size <= MOORE_BLOCK_SIZE_512) {
    status = launchKernel<MOORE_BLOCK_SIZE_512>(...);
} else if (block_size <= MOORE_BLOCK_SIZE_1024) {
    status = launchKernel<MOORE_BLOCK_SIZE_1024>(...);
}
```
- 查询设备属性 `maxThreadsPerBlock()`，限制最大并发线程数
- 支持的块大小: 512 / 1024 线程

**3. 字节偏移计算** (rearrange_kernel.h:40-41):
```cpp
const ArrayStruct<block_array_size, ARRAY_TYPE_STRIDE> src_block_stride; // 字节单位
const ArrayStruct<block_array_size, ARRAY_TYPE_STRIDE> dst_block_stride; // 字节单位
```
- 步长以**字节为单位**而非元素个数，简化跨类型数据重排
- 内核中通过 `reinterpret_cast<char *>(ptr) + offset` 实现任意类型访问

**4. 错误处理** (rearrange_moore.mu:443-447):
```cpp
musaError_t err = musaDeviceSynchronize();
if (err != musaSuccess) {
    std::cerr << "[ERROR] musaDeviceSynchronize failed: " << err << std::endl;
    return INFINI_STATUS_INTERNAL_ERROR;
}
```
- 同步设备并检查内核执行错误
- 输出详细错误信息便于调试

**5. 共享内存优化** (rearrange_kernel.h:52-53):
```cpp
__shared__ ptrdiff_t shared_src_offset;
__shared__ ptrdiff_t shared_dst_offset;
```
- 仅让线程 0 计算复杂的多维索引，结果存入共享内存
- 其他线程通过 `__syncthreads()` 同步后直接读取，避免重复计算
- 节省计算资源，降低功耗

### 5.4 性能优化技术

**1. 向量化内存访问**:
```cpp
// unit_size=16 时，使用 float4 一次读取 16 字节
*reinterpret_cast<float4 *>(dst + dst_offset) =
    *reinterpret_cast<const float4 *>(src + src_offset);
```
- 利用 128 位宽内存总线，减少内存事务数量
- 理论带宽提升 4 倍（相比单字节访问）

**2. 索引计算优化**:
- **共享内存复用**: Block 内的线程共享 grid 维度偏移，减少重复计算
- **编译时常量**: `block_array_size` 和 `grid_array_size` 作为模板参数，编译时展开循环
- **字节对齐**: 步长按单元大小对齐，确保跨步访问不会命中同一缓存行

**3. 负载均衡**:
- 通过平方根平衡算法分配 src/dst 维度，避免某个方向过度集中
- 分割大维度时，确保各 block 处理的数据量相近

**4. 提前退出机制**:
```cpp
if (constraints_grid_idx_multiple[j] + idx >= constraints.a[j].total_len) {
    return;  // 越界线程提前退出
}
```
- 对于无法整除的维度，越界线程立即返回，避免无效内存访问

### 5.5 内存管理

**设备端**:
- **零拷贝**: 内核直接通过指针访问全局内存，无需显式分配临时缓冲区
- **共享内存**: 仅使用少量（约 16 字节）共享变量存储索引，开销可忽略

**宿主端**:
- **智能指针**: `Descriptor::Opaque` 使用 `std::shared_ptr` 管理 `Handle::Internal` 生命周期
- **RAII**: `Descriptor` 析构时自动释放 `_opaque`，防止内存泄漏
- **参数拷贝**: `prepareRearrangeParams` 返回值语义的 `RearrangeParams`，避免悬垂引用

### 5.6 错误处理策略

**参数验证**:
```cpp
CHECK_OR_RETURN(x_desc->dtype() == y_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
CHECK_OR_RETURN(x_desc->ndim() == y_desc->ndim(), INFINI_STATUS_BAD_TENSOR_SHAPE);
CHECK_SAME_SHAPE(x_shape, y_shape);
```
- 在内核启动前验证张量形状和类型一致性
- 检查维度数量不超过 `MAX_BLOCK_ARRAY_SIZE` 和 `MAX_GRID_ARRAY_SIZE`（均为 5）

**运行时检查**:
```cpp
if (grid_size == 0) {
    return INFINI_STATUS_BAD_PARAM;
}
if (aligned_block_size > BLOCK_SIZE) {
    aligned_block_size = BLOCK_SIZE;  // 降级到安全值
}
```
- 检查 grid 大小合法性，避免启动零个 block
- 限制最大线程数，防止硬件资源耗尽

**错误传播**:
```cpp
CHECK_RESULT(meta_result);        // 检查 RearrangeMeta::create 结果
CHECK_RESULT(kernel_func_result); // 检查 getRearrangeKernel 结果
CHECK_OR_RETURN(musaLaunchKernel(...) == musaSuccess, INFINI_STATUS_INTERNAL_ERROR);
```
- 使用 `utils::Result<T>` 模式统一错误处理
- 错误信息通过 `std::cerr` 输出到标准错误流

### 5.7 设计模式

**1. 策略模式 (Strategy Pattern)**:
- 不同 `unit_size` / `block_dim` / `grid_dim` / `constraint_num` 组合对应不同的内核策略
- 运行时通过 `getRearrangeKernel` 动态选择最优策略

**2. 模板方法模式 (Template Method Pattern)**:
- `launchKernel<BLOCK_SIZE>` 定义内核启动的骨架流程
- 具体的内核函数指针通过 `getRearrangeKernel` 注入

**3. 工厂模式 (Factory Pattern)**:
- `Descriptor::create` 作为工厂方法，封装复杂的元数据生成和描述符构造逻辑
- 隐藏内部实现细节，用户只需调用 `calculate` 执行计算

**4. 适配器模式 (Adapter Pattern)**:
- `prepareRearrangeParams` 将通用的 `RearrangeMeta` 适配到 Moore 特定的 `RearrangeParams`
- 抽象 MUDA API 差异，提供统一的算子接口

### 5.8 依赖关系

**外部依赖**:
- **MUSA Runtime**: `musaLaunchKernel`, `musaMemcpyAsync`, `musaDeviceSynchronize`
- **MUSA Headers**: `musa_runtime_api.h`, `musa_fp16.h`, `musa_bf16.h`
- **MuBLAS / MuDNN**: 通过 `device::moore::Handle::Internal` 间接使用

**内部依赖**:
- `../rearrange.h`: 算子基类宏定义
- `../../../devices/moore/moore_common.h`: MUDA 类型定义和错误处理宏
- `../../../devices/moore/moore_kernel_common.h`: 内核辅助函数和常量（如 `MOORE_BLOCK_SIZE_*`）
- `../../operator.h`: `InfiniopDescriptor` 基类
- `../../../tensor.h`: 张量描述符接口
- `../../../utils.h`: `RearrangeMeta`, `CHECK_OR_RETURN`, `CHECK_RESULT` 等工具宏

**模块独立性**:
- Moore 特定代码完全隔离在 `moore/` 目录，不影响其他平台（如 CUDA / Ascend）
- 平台差异通过目录名和命名空间区分，便于维护和扩展

## 6. 总结

Moore Rearrange 算子是一个高度优化的 GPU 数据重排实现，核心创新点包括：

1. **智能维度分割**: 通过贪心算法和平方根平衡，自动选择最优 block-grid 分配策略
2. **内核特化**: 75 个编译时特化内核，覆盖常见维度组合和约束场景
3. **MUSA 深度适配**: Warp 对齐、字节偏移、共享内存优化，充分发挥摩尔线程硬件性能
4. **安全性**: 约束检查机制防止越界访问，完善的错误处理和参数验证
5. **通用性**: 支持任意维度重排、转置、重塑操作，兼容多种数据类型（FP8/FP16/FP32/BF16/INT8）

该实现为 Infini 框架在摩尔线程 GPU 上的张量操作提供了底层基础设施，是高性能计算场景的关键组件。
