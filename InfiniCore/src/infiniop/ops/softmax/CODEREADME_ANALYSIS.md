# 📂 目录: softmax 算子架构全景

## 1. 子系统职责

softmax 目录实现了 Infini 框架中 Softmax 算子的多后端支持。Softmax 是深度学习中核心的激活函数，广泛应用于分类任务、注意力机制（如 Transformer 的 self-attention）和概率归一化场景。该算子将输入张量在指定轴上进行指数归一化，使得输出值在 [0, 1] 范围内且和为 1。

该目录通过硬件后端分层设计，为不同 GPU 架构提供优化的实现路径，包括针对 NVIDIA GPU 的专用优化内核以及通用的 CUDA 实现组件。整个子系统在 Infini 算子体系中属于 `src/infiniop/ops/` 下的一元归约算子类别。

## 2. 模块导航 (Module Navigation)

* **📂 cuda**:
    * *功能*: CUDA 通用 kernel 实现库 - 文档缺失
    * *职责*: 提供 CUDA kernel 的底层实现（包含 `kernel.cuh`，定义了 `DataMaxSum` 结构体和 `blockSoftmaxKernel`、`warpSoftmaxKernel` 等核心计算内核），作为不同硬件后端（nvidia、kunlun 等）的共享基础设施

* **📂 nvidia**:
    * *功能*: NVIDIA GPU 后端的完整 Softmax 算子实现，支持 FP16 和 FP32 数据类型，针对不同张量维度提供自适应 kernel 选择策略
    * *职责*: 实现 NVIDIA 特定的算子描述符（`Descriptor`），包含设备适配、kernel 调度逻辑、数值稳定性优化（max-shift 策略）、并行归约优化（CUB BlockReduce 和 Warp Shuffle），以及完整的错误处理机制

## 3. 架构逻辑图解

### 数据流与组件交互关系

```
上层调用流程：
operator.cc (统一算子接口)
    ↓
softmax.h (宏定义的后端分发层)
    ↓
nvidia/softmax_nvidia.cu (NVIDIA 后端实现)
    ├→ Descriptor::create()      [验证参数，提取元数据]
    ├→ Descriptor::calculate()   [调度 kernel 执行]
    └→ cuda/kernel.cuh           [通用 CUDA kernel 库]
        ├→ blockSoftmaxKernel    [大维度：dimsize > 1024]
        └→ warpSoftmaxKernel     [中小维度：dimsize ≤ 1024]
```

### 核心设计模式与执行流程

#### 1. 分层架构设计
整个 softmax 算子系统采用清晰的分层结构：

- **接口层** (`operator.cc`): 提供统一的 C API，屏蔽后端差异
- **抽象层** (`softmax.h`): 通过 `DESCRIPTOR` 宏定义后端无关的算子描述符模板
- **后端层** (`nvidia/`): 针对特定硬件优化的具体实现
- **计算层** (`cuda/kernel.cuh`): 跨后端共享的 CUDA kernel 实现（可能被多个兼容 CUDA 的硬件后端复用）

#### 2. 自适应 Kernel 选择策略

NVIDIA 后端根据输入张量的归约轴维度 (`dimsize`) 动态选择最优 kernel：

- **大维度场景** (`dimsize > 1024`):
  - 调用 `blockSoftmaxKernel`
  - 使用 CUB 库的 `BlockReduce` 进行高效的块内归约
  - 单维线程块（512/1024/4096 线程），每线程处理多个元素
  - 适合序列长度较大的注意力机制场景

- **中等维度场景** (`31 < dimsize ≤ 1024`):
  - 调用 `warpSoftmaxKernel`（32×32 线程块配置）
  - 使用 warp shuffle 指令进行全归约，避免共享内存同步开销
  - 每线程处理 32 个元素，充分利用 warp 内的并行性

- **小维度场景** (`dimsize ≤ 31`):
  - 调用 `warpSoftmaxKernel`（16×32 线程块配置）
  - 每线程处理 2 个元素，减少寄存器压力
  - 适合小批次或小隐藏层的模型

#### 3. 数值稳定性与并行归约

所有 kernel 实现都采用经典的 **max-shift 数值稳定策略**：
```
softmax(x)_i = exp(x_i - max(x)) / Σ exp(x - max(x))
```

这一策略通过两次并行归约实现：
1. **第一次归约**：找到张量中的最大值 `max_val`
2. **第二次归约**：计算指数和 `Σ exp(x_i - max_val)`

**并行归约优化**：
- **Block 级**：使用 `DataMaxSum` 结构体同时追踪最大值和指数和，将两次归约合并为一次遍历
- **Warp 级**：使用 `WarpAllReduce` 模板（基于 `__shfl_xor_sync`），在 warp 内完成全归约，延迟仅 1-2 个周期

#### 4. 设备适配与架构兼容性

NVIDIA 后端通过 `device::nvidia::Handle::Internal` 获取设备能力信息，支持：
- **标准架构**（Compute Capability 5.0+）：最大 1024 线程/块
- **高端架构**（Compute Capability 8.0+）：最大 4096 线程/块
- **兼容平台**（Hygon DCU）：通过 `ENABLE_HYGON_API` 宏适配类型定义差异

#### 5. 错误处理与资源管理

- **创建阶段**：验证数据类型（仅支持 FP16/FP32）、形状匹配、轴参数有效性，使用 `Result<T>` 模式返回错误
- **执行阶段**：检查 GPU 架构兼容性，CUDA 错误通过 `CHECK_STATUS` 宏转换为 `infiniStatus_t`
- **资源管理**：使用 RAII 模式，`Descriptor` 析构时自动释放内部状态；支持 CUDA 流实现异步执行和资源隔离

### 关键优化特性

1. **内存访问合并**：通过精心设计的索引计算（`tid = blockIdx.x % stride + ...`），确保同一 block/warp 内的线程访问连续内存
2. **寄存器复用**：warp softmax 中，`dataPerThreadx` 数组在 max、sum、output 三个阶段复用，减少寄存器压力
3. **最小化全局内存访问**：每个输入元素读取 3 次（max 一次、sum 一次、output 一次），无额外临时内存
4. **零同步开销**：warp softmax 使用 shuffle 指令，完全避免 `__syncthreads()`；block softmax 仅在归约后同步一次

### 依赖关系总结

**内部依赖**：
- `info.h`: 提供计算元数据提取（`SoftmaxInfo` 类）
- `cuda/kernel.cuh`: 通用 CUDA kernel 实现
- `../softmax.h`: 算子描述符宏定义
- `../../../devices/nvidia/`: NVIDIA 设备句柄和工具函数

**外部依赖**：
- CUDA Runtime API
- CUB 库（BlockReduce 原语）
- Infini 核心基础设施（张量描述符、错误处理、工具宏）

该子系统的设计充分体现了"关注点分离"原则：计算逻辑集中在 `cuda/kernel.cuh`，设备适配和调度逻辑集中在 `nvidia/` 后端，通过宏定义实现代码复用和后端扩展性。
