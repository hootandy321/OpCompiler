# 目录: rearrange 架构全景

## 1. 子系统职责

`rearrange` 是 InfiniCore 中负责张量重排（rearrange）操作的多硬件后端实现层。该子系统提供了跨多个 GPU 和硬件平台的高效数据重排能力，支持张量的转置（transpose）、重塑（reshape）、排列（permute）等多种数据变换场景。通过维度贪心分割算法和编译期内核特化技术，实现了在保持通用性的同时最大化内存访问局部性和计算效率。

本目录采用硬件隔离的架构设计，每个硬件后端拥有独立的实现子目录，通过统一的抽象接口向上层提供一致的操作体验。这种设计使得框架能够透明地支持多种硬件加速器，包括 NVIDIA GPU、摩尔线程 GPU、华为昇腾 NPU、寒武纪 MLU、百度昆仑 XPU、壁仞科技 GPU 等国产和国际硬件平台。

## 2. 模块导航

- **ascend**:
  - 功能: 文档缺失
  - 职责: 华为昇腾（Ascend）NPU 后端的 rearrange 算子实现

- **bang**:
  - 功能: 文档缺失
  - 职责: 寒武纪（Cambricon）MLU 后端的 rearrange 算子实现

- **cpu**:
  - 功能: 文档缺失
  - 职责: CPU 通用后端的 rearrange 算子实现

- **kunlun**:
  - 功能: 文档缺失
  - 职责: 百度昆仑（Kunlun）XPU 后端的 rearrange 算子实现

- **metax**:
  - 功能: 文档缺失
  - 职责: 壁仞科技（Metax）GPU 后端的 rearrange 算子实现

- **moore**:
  - 功能: 实现针对摩尔线程（Moore）GPU 架构的张量重排算子，通过 MUSA 内核实现高效的数据重排操作。支持任意维度的张量重塑、转置和步长变换，采用智能的块-网格分层策略和约束检查机制，确保内存访问的最优局部性和安全性。
  - 职责: 提供摩尔线程 GPU 平台的 rearrange 操作，包含 75 个特化内核变体（5×5×3 组合）和运行时分发机制，实现 Warp 对齐、向量化内存访问、共享内存优化等性能技术

- **nvidia**:
  - 功能: 实现 NVIDIA GPU 后端的张量重排操作，通过高度优化的 CUDA Kernel 实现任意维度张量的高效内存重排，支持 transpose、reshape、permute 等多种张量变换场景。通过宏生成 225 个特化内核，覆盖不同 block 维度数（1-5）、grid 维度数（1-5）、约束条件数（0-2）和内存类型（6 种）。
  - 职责: 提供 NVIDIA CUDA 平台的 rearrange 操作，采用维度贪心分割策略最大化内存访问局部性，实现向量化加载/存储、自适应 Block 大小、编译期优化等高性能技术

## 3. 架构逻辑图解

rearrange 子系统采用**硬件抽象层（HAL）模式**，整体数据流如下：

### 3.1 统一入口层

上层的 `operator.cc` 和 `rearrange.h` 提供统一的算子注册和调用接口。当用户发起 rearrange 操作时，系统根据设备类型动态分派到对应的后端实现：

```
用户请求 (infiniopRearrange)
    ↓
设备类型检测 (handle->device_type)
    ↓
后端分发
    ├─ NVIDIA GPU → nvidia/ 子目录
    ├─ Moore GPU → moore/ 子目录
    ├─ Ascend NPU → ascend/ 子目录
    ├─ Kunlun XPU → kunlun/ 子目录
    ├─ Metax GPU → metax/ 子目录
    ├─ Bang MLU → bang/ 子目录
    └─ CPU → cpu/ 子目录
```

### 3.2 后端执行流程

每个硬件后端（以已文档化的 nvidia 和 moore 为例）遵循相似的执行模式：

1. **描述符创建阶段**:
   - 验证输入/输出张量的数据类型和形状一致性
   - 调用 `RearrangeMeta::create` 生成元数据（维度、步长、单元大小）
   - 查询设备属性（如 `maxThreadsPerBlock`）

2. **参数预处理阶段** (`prepareRearrangeParams`):
   - **单元大小优化**: 将数据单元对齐到 2 的幂次方（32/16/8/4/2/1 字节），利用向量化内存访问
   - **维度排序**: 按源步长升序或降序排序，贪心选择内存局部性最好的维度
   - **贪心维度分配**:
     - 将较小的、相对连续的维度分配给 **block**（由线程并行处理）
     - 将较大的、分散的维度分配给 **grid**（由多个 block 串行处理）
     - 当维度长度超过 block 容量时，执行维度分割（`num_per_block` × `num_per_grid`）
   - **约束生成**: 对无法整除的分割维度生成边界约束条件（最多 2 个），防止内核访问越界

3. **内核选择与启动阶段**:
   - 根据预处理参数（`block_dim`, `grid_dim`, `constraint_num`, `unit_size`）选择匹配的特化内核
   - 对齐线程块大小到硬件 Warp 大小（NVIDIA/Moore 为 32 的倍数）
   - 将参数打包成 `ArrayStruct` 固定数组（CUDA/MUSA 内核不支持 `std::vector`）
   - 调用 `cudaLaunchKernel` / `musaLaunchKernel` 启动内核执行

4. **内核执行阶段** (GPU 设备端):
   - **线程 0 计算 block 基础偏移**: 遍历 grid 维度，使用混合基数转换计算当前 block 在 src/dst 张量中的字节偏移，结果存入共享内存
   - **同步**: `__syncthreads()` 确保所有线程可见共享内存值
   - **所有线程计算 block 内偏移**: 结合 `threadIdx.x` 和 block 维度数组计算最终偏移，检查约束条件防止越界
   - **向量化内存拷贝**: 使用 `float1/2/4`、`double4` 等类型执行类型化数据拷贝

### 3.3 硬件适配差异

虽然各后端遵循统一的算法框架，但在硬件适配层面存在关键差异：

- **NVIDIA (CUDA)**:
  - 生成 **225 个特化内核**（5 block × 5 grid × 3 constraint × 6 type）
  - 支持 FP8/FP16/FP32/BF16/INT8 等多种数据类型
  - 使用 `cuda_bf16.h`, `cuda_fp16.h`, `cuda_fp8.h` 扩展头文件
  - Warp 大小对齐到 32 线程

- **Moore (MUSA)**:
  - 生成 **75 个特化内核**（5 block × 5 grid × 3 constraint）
  - 同样支持向量化内存访问（`uchar1/2`, `float1/2/4`, `double4`）
  - 强制要求线程块大小必须是 32 的倍数（否则性能严重下降）
  - 采用 `std::shared_ptr` 管理设备句柄生命周期
  - 通过 `musaDeviceSynchronize` 同步并检查内核执行错误

- **其他后端** (ascend/bang/kunlun/metax/cpu):
  - 预期采用类似的三阶段执行模式（创建→预处理→内核启动）
  - 根据硬件特性调整 Warp 大小、向量宽度、同步原语
  - 部分后端（如 CPU）可能使用多线程并行而非 SIMT 执行模型

### 3.4 关键数据结构流动

```
RearrangeMeta (通用元数据)
    ↓
prepareRearrangeParams (贪心分割算法)
    ↓
RearrangeParams (硬件特定参数)
    ├─ block_len / grid_len (数组)
    ├─ src_block_stride / dst_block_stride (字节单位)
    ├─ constraints (边界条件)
    └─ unit_size (1/2/4/8/16/32)
    ↓
ArrayStruct<固定大小> (内核兼容格式)
    ↓
CUDA/MUSA Kernel 参数列表
    ↓
设备端执行 (线程并行计算)
```

### 3.5 性能优化策略

各后端实现共享以下核心优化技术：

1. **编译期内核特化**: 通过宏/模板生成大量特化版本，将运行时决策转移到编译期，实现零抽象开销
2. **向量化内存访问**: 使用 `float4`（16 字节）或 `double4`（32 字节）类型，充分利用 128/256 位宽内存总线
3. **共享内存优化**: 仅由 0 号线程计算复杂的 grid 偏移，通过共享内存广播给 block 内其他线程，避免重复计算
4. **提前退出机制**: 对于无法整除的维度，越界线程立即返回，避免无效内存访问和分支发散
5. **自适应 Block 大小**: 根据设备 `maxThreadsPerBlock` 属性选择 512 或 1024 线程块，平衡并行度和资源占用

### 3.6 安全性保障

- **输入验证**: 在内核启动前检查张量形状、类型一致性，防止运行时崩溃
- **边界约束**: 对分割维度生成约束条件，内核内执行 `grid_idx * grid_div_block + block_idx < total_len` 检查
- **错误传播**: 使用 `utils::Result<T>` 模式统一错误处理，所有 CUDA/MUSA API 调用通过宏包装，自动传播错误码
- **资源管理**: 采用 RAII 和智能指针自动管理设备句柄和描述符生命周期，防止内存泄漏

---

**总结**: rearrange 子系统通过精心设计的硬件抽象层和通用贪心算法，实现了跨多平台的高效张量重排能力。NVIDIA 和 Moore 后端的文档显示，该模块在保持代码复用性的同时，能够深度适配各硬件平台的特性（如 Warp 大小、内存模型、API 差异），为 Infini 框架的跨硬件统一编程模型提供了坚实基础。
