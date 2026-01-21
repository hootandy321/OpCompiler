# NVIDIA Rearrange 算子核心实现文档

本模块实现了 Infini 框架中 NVIDIA GPU 后端的张量重排（rearrange）操作，通过高度优化的 CUDA Kernel 实现任意维度张量的高效内存重排，支持 transpose、reshape、permute 等多种张量变换场景。

## 1. 模块结构

- **`rearrange_nvidia.cuh`**: NVIDIA 后端描述符头文件，定义 `op::rearrange::nvidia::Descriptor` 类的公共接口
- **`rearrange_kernel.cuh`**: CUDA Kernel 核心实现，包含通过宏生成的 225 个特化 Kernel 和参数调度逻辑
- **`rearrange_nvidia.cu`**: 宿主端实现，包含描述符创建、参数预处理（`prepareRearrangeParams`）和 Kernel 启动逻辑

## 2. 核心类与数据结构

### `Descriptor` 类
- **位置**: `rearrange_nvidia.cuh` (接口定义), `rearrange_nvidia.cu` (实现)
- **主要功能**: 封装张量重排操作的元数据和设备状态，提供创建和执行接口
- **关键成员**:
  - `_meta`: `utils::RearrangeMeta` 类型，存储张量的形状、步长、单元大小等元信息
  - `_opaque`: `Opaque*` 类型，指向包含 NVIDIA 设备句柄内部状态（如 `maxThreadsPerBlock`）的不透明结构
- **核心方法**:
  - `create(handle, desc_ptr, y_desc, x_desc)`: 静态工厂方法，验证输入/输出张量的数据类型和形状一致性，创建 `RearrangeMeta` 和描述符实例
  - `calculate(y, x, stream)`: 执行张量重排计算，根据设备属性选择合适的 Kernel 启动配置（512 或 1024 线程块）
- **生命周期**: 由用户通过 `create` 方法构造，使用完毕后由析构函数自动释放 `_opaque` 资源

### `RearrangeParams` 结构体
- **位置**: `rearrange_kernel.cuh:216-227`
- **主要功能**: 封装预处理后的 CUDA Kernel 启动参数
- **关键成员**:
  - `block_len`: `std::vector<ARRAY_TYPE_SIZE>`，每个 block 维度的长度数组
  - `src_block_stride`, `dst_block_stride`: `std::vector<ARRAY_TYPE_STRIDE>`，每个 block 维度的字节单位步长
  - `grid_len`: `std::vector<ARRAY_TYPE_SIZE>`，每个 grid 维度的长度数组
  - `src_grid_stride`, `dst_grid_stride`: `std::vector<ARRAY_TYPE_STRIDE>`，每个 grid 维度的字节单位步长
  - `block_dim`: `size_t`，block 维度的数量
  - `block_len_total`: `size_t`，block 中线程总数（等于 `block_len` 所有元素乘积）
  - `constraints`: `std::vector<Constraint<ARRAY_TYPE_SIZE>>`，边界约束条件（最多 2 个）
  - `unit_size`: `size_t`，每次内存拷贝的单元大小（1/2/4/8/16/32 字节）

### `Constraint<T>` 模板结构体
- **位置**: `rearrange_kernel.cuh:20-26`
- **主要功能**: 描述维度分割后的边界约束，防止 Kernel 访问越界
- **关键成员**:
  - `grid_idx`: `ElementType`，约束在 `grid_len` 数组中的索引
  - `block_idx`: `ElementType`，约束在 `block_len` 数组中的索引
  - `grid_div_block`: `ElementType`，grid 索引相对于 block 索引的倍数（即 `num_per_block`）
  - `total_len`: `ElementType`，该维度的原始总长度

### `ArrayStruct<T, N>` 模板结构体
- **位置**: `rearrange_kernel.cuh:14-17`
- **主要功能**: 封装固定大小数组，用于传递给 CUDA Kernel
- **关键成员**:
  - `a`: `ArrayType[N]`，C 风格数组，大小为编译时常量 `N`

## 3. API 接口

```cpp
// 创建 Rearrange 描述符
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,                      // Infini 操作句柄
    Descriptor **desc_ptr,                         // 输出：描述符指针
    infiniopTensorDescriptor_t y_desc,            // 输出张量描述符
    infiniopTensorDescriptor_t x_desc             // 输入张量描述符
);
// 返回 INFINI_STATUS_SUCCESS 或错误码（如 dtype/shape 不匹配）

// 执行张量重排计算
infiniStatus_t Descriptor::calculate(
    void *y,                                      // 输出缓冲区（设备内存）
    const void *x,                                // 输入缓冲区（设备内存）
    void *stream                                  // CUDA 流
) const;
// 返回 INFINI_STATUS_SUCCESS 或错误码

// Kernel 选择函数（内部）
utils::Result<void *> getRearrangeKernel(const RearrangeParams &params);
// 根据参数选择对应的特化 Kernel 函数指针

// 参数预处理函数（内部）
utils::Result<RearrangeParams> prepareRearrangeParams(
    const utils::RearrangeMeta &original_meta,    // 原始元数据
    int max_threads                               // 设备最大线程数
);
// 返回处理后的参数结构体
```

## 4. 使用示例

```cpp
// 示例：在 NVIDIA GPU 上执行张量转置 (2, 3) -> (3, 2)

// 1. 准备张量描述符
std::vector<size_t> x_shape = {2, 3};
std::vector<ptrdiff_t> x_strides = {3, 1};  // 行主序
std::vector<size_t> y_shape = {3, 2};
std::vector<ptrdiff_t> y_strides = {1, 2};  // 转置后的步长

infiniopTensorDescriptor_t x_desc, y_desc;
infiniopCreateTensorDescriptor(&x_desc, kInfiniDeviceCUDA, 0,
                               INFINI_DATA_TYPE_FLOAT32,
                               x_shape.size(), x_shape.data(), x_strides.data());
infiniopCreateTensorDescriptor(&y_desc, kInfiniDeviceCUDA, 0,
                               INFINI_DATA_TYPE_FLOAT32,
                               y_shape.size(), y_shape.data(), y_strides.data());

// 2. 创建 Rearrange 描述符
op::rearrange::nvidia::Descriptor *rearrange_desc;
infiniStatus_t status = op::rearrange::nvidia::Descriptor::create(
    handle, &rearrange_desc, y_desc, x_desc);
// 内部调用 RearrangeMeta::create 并验证形状一致性

// 3. 分配设备内存并初始化输入数据
float *d_x, *d_y;
cudaMalloc(&d_x, 2 * 3 * sizeof(float));
cudaMalloc(&d_y, 2 * 3 * sizeof(float));
cudaMemcpy(d_x, host_x, 2 * 3 * sizeof(float), cudaMemcpyHostToDevice);

// 4. 执行重排计算
cudaStream_t stream;
cudaStreamCreate(&stream);
status = rearrange_desc->calculate(d_y, d_x, stream);
// 内部流程：
//   - 调用 prepareRearrangeParams 分配 block/grid 维度
//   - 选择合适的 Kernel（512 或 1024 线程）
//   - 启动 CUDA Kernel

// 5. 同步并获取结果
cudaStreamSynchronize(stream);
cudaMemcpy(host_y, d_y, 2 * 3 * sizeof(float), cudaMemcpyDeviceToHost);

// 6. 清理资源
delete rearrange_desc;
cudaFree(d_x);
cudaFree(d_y);
cudaStreamDestroy(stream);
```

## 5. 实现细节

### 5.1 核心算法：维度贪心分割策略

**问题**: 给定任意维度的张量，如何将维度分配给 CUDA block 和 grid，使得每个 block 处理的数据尽可能在内存中连续，最大化内存访问局部性？

**算法流程**（`prepareRearrangeParams` 函数，`rearrange_nvidia.cu:81-295`）：

1. **单元大小优化**: 调用 `RearrangeMeta::distributeUnit({32, 16, 8, 4, 2, 1})` 将原始单元大小调整为 2 的幂次方，提高内存对齐效率

2. **维度排序**: 按源步长（`src_stride`）升序排序维度，贪心选择内存局部性最好的维度优先分配到 block

3. **贪心选择与分割**:
   - 遍历排序后的维度，如果 `block_elements * dim_len <= max_threads`，则完整维度加入 block
   - 否则，如果 `block_elements > 1` 且 `dim_len > 1`，分割该维度：
     - `num_per_block = min(dim_len, max_threads / block_elements)`
     - `num_per_grid = ceil(dim_len / num_per_block)`
     - 记录 `SplitDim` 信息用于后续边界检查
   - 如果 `block_elements == 1`，强制分割第一个维度（`dim_order[0]`）

4. **参数构建**:
   - 将选中的完整维度和分割维度的 block 部分填充 `block_len` 和 `block_stride`
   - 将未选中的完整维度和分割维度的 grid 部分填充 `grid_len` 和 `grid_stride`
   - 对于 grid stride，乘以 `num_per_block` 跳过 block 已处理的部分

5. **约束生成**: 最多选择 2 个无法整除的分割维度生成 `Constraint`，用于 Kernel 内的越界检查

**复杂度**: O(ndim log ndim)（排序阶段）

**示例**: 张量形状 `[100, 256, 512]`，步长 `[131072, 512, 1]`（最后一个维度最连续）
- 排序后维度顺序: `[2, 1, 0]`（按步长升序）
- 分配过程:
  - 维度 2 (len=512): `1 * 512 <= 1024` → 完整加入 block，`block_elements = 512`
  - 维度 1 (len=256): `512 * 256 > 1024` → 分割，`num_per_block = 2`，`num_per_grid = 128`
  - 维度 0 (len=100): 作为 grid 维度
- 结果: `block_len = [512, 2]`，`grid_len = [100, 128]`，每个 block 处理 2x512 个元素

### 5.2 CUDA Kernel 机制

**宏生成策略**（`rearrange_kernel.cuh:173-213`）：

为了在编译期优化性能，通过宏展开生成 **225 个特化 Kernel**：
- **Block 维度数**: 1-5（`MAX_BLOCK_ARRAY_SIZE`）
- **Grid 维度数**: 1-5（`MAX_GRID_ARRAY_SIZE`）
- **约束条件数**: 0-2
- **内存类型**: 6 种（`uchar1/2`, `float1/2/4`, `double4`），对应单元大小 1/2/4/8/16/32 字节

**命名规则**:
```
rearrange_unit_<type>_block_<bn>_grid_<gn>_constrain_<cn>
```
例如: `rearrange_unit_float4_block_3_grid_2_constrain_1`

**Kernel 核心逻辑**（以有约束版本为例）：

1. **提前退出**: `if (threadIdx.x >= block_len_total) return;`

2. **Block 基础偏移计算**（仅 0 号线程）:
   - 使用 `blockIdx.x` 和 `grid_len` 数组，通过混合基数转换（mixed radix conversion）计算 grid 索引
   - 遍历约束条件，计算 `constraints_grid_idx_multiple[j] = grid_idx * grid_div_block`
   - 将 `src_offset` 和 `dst_offset` 存入共享内存（`__shared__`）

3. **同步**: `__syncthreads()` 确保所有线程看到共享内存的值

4. **Thread 内部偏移计算**:
   - 从共享内存读取 block 基础偏移
   - 使用 `threadIdx.x` 和 `block_len` 数组计算 block 内索引
   - **边界检查**: 对于每个约束，如果 `constraints_grid_idx_multiple[j] + block_idx >= total_len`，提前 `return`
   - 累加字节步长得到最终偏移

5. **内存拷贝**:
   ```cpp
   *reinterpret_cast<Tmem_type *>(reinterpret_cast<char *>(dst) + dst_offset) =
   *reinterpret_cast<const Tmem_type *>(reinterpret_cast<const char *>(src) + src_offset);
   ```

**优化技巧**:
- **向量化加载/存储**: 使用 `float4`（16 字节）或 `double4`（32 字节）类型，一次传输多个标量
- **共享内存优化**: 仅由 0 号线程计算 grid 偏移，通过共享内存广播给其他线程
- **编译期常量**: 所有数组大小均为模板参数，编译器可以展开循环和优化索引计算
- **字节寻址**: 直接使用字节偏移，避免运行时类型转换

### 5.3 内存管理策略

- **设备内存管理**: 调用者负责分配和释放输入/输出缓冲区（通过 `cudaMalloc`/`cudaFree`）
- **元数据存储**: `RearrangeMeta` 在宿主端维护张量形状和步长信息的紧凑表示（`std::vector<ptrdiff_t>`）
- **描述符生命周期**: `_opaque` 持有 `std::shared_ptr<device::nvidia::Handle::Internal>`，引用计数管理设备句柄生命周期
- **Kernel 参数传递**: 所有数组参数通过 `ArrayStruct` 封装为固定大小数组，直接传递给 CUDA Kernel（无需动态分配）

### 5.4 并发与线程安全

- **CUDA 流支持**: `calculate` 方法接受 `stream` 参数，允许与其他操作异步并发执行
- **无共享状态**: Kernel 执行过程中不修改全局内存，仅读取输入并写入输出
- **线程局部性**: 每个线程独立计算偏移并访问内存，无线程间同步（除共享内存广播阶段）
- **原子操作**: 不需要，因为每个线程写入的输出位置互不重叠

### 5.5 性能优化技术

1. **贪心维度选择**: 优先处理源步长最小的维度，最大化 block 内的内存访问局部性
2. **向量化内存访问**: 根据 `unit_size` 选择最优的内存类型（如 16 字节用 `float4`），充分利用 GPU 内存带宽
3. **Kernel 特化**: 通过宏生成 225 个 Kernel 编译期优化，避免运行时分支和动态数组访问
4. **自适应 Block 大小**: 根据 `block_len_total` 选择 512 或 1024 线程块，平衡并行度和资源占用
5. **边界检查优化**: 最多 2 个约束条件，最小化 Kernel 内分支开销
6. **共享内存最小化**: 仅存储 2 个 `ptrdiff_t` 偏移值和约束乘数，降低 SMEM 使用量

### 5.6 错误处理机制

- **输入验证**: `create` 方法检查数据类型一致性、形状一致性（通过 `CHECK_SAME_SHAPE` 宏）
- **参数合法性**: `getRearrangeKernel` 验证 `grid_num <= 5`, `block_num <= 5`, `constraint_num <= 2`
- **CUDA 错误传播**: 所有 CUDA API 调用通过 `CHECK_CUDA` 宏包装，返回 `INFINI_STATUS_INTERNAL_ERROR`
- **Result 类型**: 使用 `utils::Result<T>` 封装可能失败的操作（如 `prepareRearrangeParams`），强制错误检查（通过 `CHECK_RESULT` 宏）

### 5.7 依赖关系

- **外部依赖**:
  - CUDA Toolkit（`cuda_runtime.h` 隐含包含）
  - `cuda_bf16.h`, `cuda_fp16.h`, `cuda_fp8.h`（低精度数据类型支持）
- **内部依赖**:
  - `devices/nvidia/nvidia_common.cuh`: 通用 CUDA 定义和错误检查宏
  - `devices/nvidia/nvidia_kernel_common.cuh`: Kernel 常量（`CUDA_BLOCK_SIZE_*`）和辅助函数
  - `tensor.h`: 张量描述符类型定义
  - `../rearrange.h`: 操作符基类宏（`DESCRIPTOR`）
  - `utils/rearrange.h`: `RearrangeMeta` 类和工具函数
- **设计模式**:
  - **策略模式**: 通过 `getRearrangeKernel` 在运行时选择不同的 Kernel 实现
  - **模板方法模式**: `DESCRIPTOR` 宏定义固定的操作符创建和执行流程
  - **编译期元编程**: 使用宏和模板生成大量特化版本，实现零抽象开销

## 6. 特殊场景处理

### 6.1 标量张量（ndim == 0）
- **检测**: 在 `calculate` 中检查 `_meta.ndim() == 0`
- **处理**: 直接调用 `cudaMemcpyAsync` 进行设备内存拷贝，跳过 Kernel 启动开销

### 6.2 跨步访问与负步长
- **支持**: 通过 `ptrdiff_t` 类型的步长数组支持负步长（反向遍历）
- **计算**: 偏移计算使用有符号整数乘法和加法，自动处理负步长

### 6.3 非对齐内存访问
- **字节寻址**: 所有偏移量以字节为单位，不要求元素级对齐
- **向量化限制**: 如果单元大小不是 2 的幂次方，`distributeUnit` 会调整到合适的对齐大小

### 6.4 大维度分割
- **最多分割 1 个维度**: 如果某个维度被分割，循环会立即 `break`，保证分割的简单性
- **Grid 维度生成**: 被分割维度的 grid 部分乘以 `num_per_block` 作为 stride，跳过 block 已处理部分

## 7. 性能特征

- **时间复杂度**: O(N)，其中 N 为张量元素总数（每个元素恰好处理一次）
- **空间复杂度**: O(1) 额外空间（除输入/输出外，仅使用常数大小共享内存）
- **内存带宽**: 理论上接近设备内存带宽峰值（受向量化访问和合并访问优化）
- **扩展性**: Grid 维度数量随张量形状增长，可充分利用大规模 GPU 并行能力
- **限制因素**:
  - 分割维度的边界检查可能引入分支发散
  - 小张量可能无法充分利用 GPU 并行度
  - 高维张量（>5 维）需要降维处理或分割

---

**总结**: 本模块通过编译期 Kernel 特化、贪心维度分割和向量化内存访问，实现了 NVIDIA GPU 上任意张量变换的高效通用后端，是 Infini 框架实现灵活张量操作的核心基础设施之一。
