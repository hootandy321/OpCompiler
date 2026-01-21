# RMSNorm 算子多硬件后端架构全景

## 1. 子系统职责

本目录 `rms_norm` 实现了 RMS (Root Mean Square) 归一化算子在不同硬件平台上的多后端支持。RMSNorm 是一种无需中心化的归一化方法，广泛应用于 Transformer 模型（如 GPT、BERT、LLaMA）的层归一化层，计算公式为 `y = x * w / sqrt(mean(x²) + epsilon)`。

该子系统采用**硬件抽象层设计**，为不同的计算设备（NVIDIA GPU、Moore GPU、华为昇腾、寒武纪等）提供统一的算子接口，同时针对每种硬件的架构特性进行专门优化。上层应用通过统一的 API 调用，底层根据运行时设备类型选择最优的实现版本。

## 2. 模块导航

* **📂 ascend**: 文档缺失 - 华为昇腾 AI 处理器后端实现
* **📂 bang**: 文档缺失 - 寒武纪 BANG 算力后端实现
* **📂 cpu**: 文档缺失 - CPU 通用处理器后端实现
* **📂 cuda**: 文档缺失 - CUDA 通用后端实现（可能是与 nvidia 的备用版本）
* **📂 kunlun**: 文档缺失 - 昆仑 AI 芯片后端实现
* **📂 metax**: 文档缺失 - Metax 加速卡后端实现
* **📂 moore**:
    * *功能*: 摩尔线程 Moore/MUSA GPU 平台实现，提供国产 GPU 架构上的高性能 RMSNorm 计算能力
    * *职责*: 基于 MUSA 编程模型（兼容 CUDA API），针对 Moore S 系列 GPU（如 MTT S80、S3000）优化，支持 FP16/BF16/FP32 数据类型，使用 CUB 库进行块级归约优化
* **📂 nvidia**:
    * *功能*: NVIDIA GPU CUDA 平台实现，提供主流 GPU 上的高性能 RMSNorm 计算能力
    * *职责*: 基于 CUDA Runtime API 和 CUB 库，支持 NVIDIA 全系列 GPU，自适应 Block 大小（512/1024/4096），支持 FP16/BF16/FP32 混合精度计算

## 3. 架构逻辑图解

### 3.1 整体架构模式

该子系统采用**策略模式 + 模板元编程**的混合架构：

```
上层调用者
    ↓
统一接口层 (infiniop API)
    ↓
设备分发层 (根据设备类型选择后端)
    ├─→ [缺失] ascend 华为昇腾后端
    ├─→ [缺失] bang 寒武纪后端
    ├─→ [缺失] cpu CPU 后端
    ├─→ [缺失] cuda 通用 CUDA 后端
    ├─→ [缺失] kunlun 昆仑后端
    ├─→ [缺失] metax Metax 后端
    ├─→ moore 摩尔线程 MUSA 后端 ✓
    └─→ nvidia NVIDIA CUDA 后端 ✓
```

### 3.2 Moore 与 NVIDIA 后端的异同分析

#### 相同点

1. **核心算法逻辑**：两个后端实现完全相同的 RMSNorm 数学公式和计算流程
2. **并行策略**：
   - 采用 Block-Head 映射：每个线程块处理一个 batch 中的一个 head
   - 使用 Strided 访问模式：线程以步长 BLOCK_SIZE 遍历数据，保证内存合并访问
   - 块内归约：使用 CUB 库的 `BlockReduce` 原语进行高效归约
3. **计算流程**：
   - Step 1: 块级归约求平方和 `sum(x²)`
   - Step 2: 线程 0 计算均方根倒数 `rms = 1 / sqrt(mean(x²) + epsilon)`
   - Step 3: 共享内存广播 RMS 值
   - Step 4: 并行应用归一化 `y[i] = x[i] * w[i] * rms`
4. **数据类型支持**：均支持 FP16/BF16/FP32 激活值与权重的灵活组合（除了 FP32 激活值要求权重也是 FP32）
5. **精度提升策略**：所有半精度类型（FP16/BF16）的计算均在 `float` 类型中进行，避免精度损失

#### 差异点

| 维度 | Moore 后端 | NVIDIA 后端 |
|-----|-----------|------------|
| **编程模型** | MUSA (Moore Unified Shader Architecture) | CUDA (Compute Unified Device Architecture) |
| **硬件平台** | 摩尔线程 S 系列 GPU（MTT S80、S3000） | NVIDIA 全系列 GPU（如 A100、H100、RTX 系列） |
| **Block 大小配置** | 512 / 1024 / 2048 | 512 / 1024 / 4096 |
| **FP16 头文件** | `<musa_fp16_mtgpu.h>` (MTGPU 扩展) | `<cuda_fp16.h>` (标准 CUDA) |
| **BF16 类型表示** | `__mt_bfloat16` | `__nv_bfloat16` |
| **BLAS 库** | `<mublas.h>` | `<cublas.h>` |
| **DNN 库** | `<mudnn.h>` | `<cudnn.h>` |
| **设备句柄** | `device::moore::Handle::Internal` | `device::nvidia::Handle::Internal` |
| **命名空间** | `op::rms_norm::moore` | `op::rms_norm::nvidia` |

### 3.3 数据流与依赖关系

#### 典型调用流程

```
1. 用户创建张量描述符 (x_desc, y_desc, w_desc)
   ↓
2. 调用 op::rms_norm::{PLATFORM}::Descriptor::create()
   - 验证张量类型兼容性
   - 提取设备句柄内部状态
   - 构建 RMSNormInfo 元信息
   ↓
3. 分配设备内存 (d_x, d_y, d_w)
   ↓
4. 调用 descriptor->calculate(workspace, workspace_size, d_y, d_x, d_w, stream)
   - 验证工作空间大小
   - 根据 GPU 能力选择 Block 大小
   - 提取张量步长和维度信息
   - 类型特化的内核启动 (launchKernel<BLOCK_SIZE, Tcompute, Tdata, Tweight>)
   ↓
5. GPU Kernel 执行
   - Block 内并行计算平方和
   - CUB 归约得到完整 sum(x²)
   - Thread 0 计算 RMS 并广播
   - 所有线程并行应用归一化
   ↓
6. 同步流并获取结果
```

#### 内部依赖关系

**共享依赖**（两个后端都依赖）：
- `../rms_norm.h`: 算子描述符的宏定义声明
- `../rms_norm/info.h`: RMSNormInfo 类，提供张量元信息验证和存储
- `../cuda/kernel.cuh`: RMSNorm CUDA kernel 的设备函数实现（`rmsnormBlock`）

**平台特定依赖**：
- Moore: `../../devices/moore/moore_common.mu`, `../../reduce/cuda/reduce.cuh`
- NVIDIA: `../../devices/nvidia/nvidia_common.cuh`, `../../devices/nvidia/nvidia_kernel_common.cuh`, `../../reduce/cuda/reduce.cuh`

### 3.4 设计模式与优化策略

#### 宏代码生成
通过 `DESCRIPTOR(NAMESPACE)` 宏在不同命名空间（`moore`、`nvidia`）中展开相同的类结构，避免代码重复，同时保持各后端的独立性。

#### 模板元编程
`launchKernel` 函数模板根据数据类型组合在**编译期**生成不同的 kernel 实例，消除运行时分支开销，提高性能。

#### Pimpl (Pointer to Implementation) 模式
`Descriptor::Opaque` 结构体封装了平台相关的实现细节（如设备句柄内部状态），使头文件不依赖特定硬件类型，保持 API 稳定性。

#### 自适应性能调优
- **Block 大小动态选择**: 根据 GPU 的 `maxThreadsPerBlock` 属性选择最优配置
- **内存访问优化**: Strided 访问模式保证跨线程的合并内存访问
- **Warp 原语加速**: CUB 库使用 warp shuffle 指令实现高效块内归约

### 3.5 扩展性与维护性

**新硬件后端添加**：
1. 创建新的子目录（如 `ascend`、`bang`）
2. 实现对应的 `Descriptor` 类（继承 `InfiniopDescriptor`）
3. 实现平台特定的 kernel 函数
4. 在 `RMSNormInfo::create` 中添加新平台的数据类型组合验证
5. 在顶层的 `operator.cc` 中注册新后端

**新数据类型支持**：
在各后端的 `launchKernel` 函数中添加新的类型组合条件分支即可。

**性能优化空间**：
- Tensor Core 加速（针对支持矩阵乘法指令的硬件）
- 多 kernel 并发执行（利用 GPU 的多个流处理器）
- 半精度累加优化（使用硬件支持的原子累加指令）

## 4. 空缺模块说明

当前有 6 个硬件后端的文档缺失（ascend、bang、cpu、cuda、kunlun、metax），这些模块应该提供与 moore/nvidia 类似的实现，但针对各自硬件的架构特性进行优化。建议后续补充这些模块的 CODEREADME.md 文档，以完善整个 RMSNorm 算子的多硬件后端架构视图。
