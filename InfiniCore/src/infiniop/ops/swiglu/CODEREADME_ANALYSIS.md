# 📂 目录: swiglu 架构全景

## 1. 子系统职责

SwiGLU (Swish-Gated Linear Unit) 是现代 Transformer 架构（如 LLaMA、GLM、PaLM 等大语言模型）前馈网络中广泛使用的激活函数。该子系统实现了 SwiGLU 操作在多种硬件平台上的后端支持，通过门控机制结合 Swish 激活函数（SiLU），相比传统的 ReLU/GLU 具有更好的性能表现。

数学定义：`output = SiLU(gate) × up = gate × σ(gate) × up`，其中 `up` 和 `gate` 是两个线性变换的输出，`σ(x)` 是 Sigmoid 函数。

该模块位于 Infini 算子库的逐元素操作框架之上，通过统一的 API 接口为上层推理框架提供跨硬件平台的 SwiGLU 加速支持。

## 2. 模块导航

* **📂 ascend**: *文档缺失* - 华为昇腾（Ascend）AI 处理器后端实现
* **📂 bang**: *文档缺失* - 寒武纪（Cambricon）MLU 硬件后端实现（使用 BANG 语言）
* **📂 cpu**: *文档缺失* - CPU 通用处理器后端实现
* **📂 cuda**: *文档缺失* - CUDA 通用后端实现（核函数实现，被 nvidia 后端引用）
* **📂 kunlun**: *文档缺失* - 昆仑（Kunlun）AI 芯片后端实现
* **📂 metax**: *文档缺失* - Metax 硬件平台后端实现
* **📂 moore**:
    * *功能*: Moore (MUSA) 硬件平台后端实现，沐璨 GPU 的 CUDA 兼容平台支持
    * *职责*: 实现 SwiGLU 在 Moore 设备上的加速计算，支持 FP16/BF16/FP32/FP64 数据类型，针对 MUSA 平台进行了特殊优化（如 half 精度 sigmoid 提升、bfloat16 单步转换等）
* **📂 nvidia**:
    * *功能*: NVIDIA CUDA 优化后端实现，针对 NVIDIA GPU 进行深度性能优化
    * *职责*: 提供工业级 CUDA 实现，使用向量化指令（half2/cuda_bfloat162）和快速数学函数（h2rcp/__frcp_rn），支持多数据类型并发执行和流式异步计算

## 3. 架构逻辑图解

### 3.1 分层架构

```
┌─────────────────────────────────────────────────────┐
│          上层推理框架 (InfiniLM/InfiniTrain)         │
└────────────────────┬────────────────────────────────┘
                     │ 调用统一 C API
┌────────────────────▼────────────────────────────────┐
│         SwiGLU 统一接口层 (operator.cc)             │
│  - infiniopCreateSwiGLUDescriptor                   │
│  - infiniopSwiGLU (计算调度)                         │
│  - infiniopGetSwiGLUWorkspaceSize                   │
└────────────────────┬────────────────────────────────┘
                     │ 根据设备类型分发
    ┌────────────────┼────────────────┬──────────────┐
    │                │                │              │
┌───▼────┐  ┌──────▼──────┐  ┌──────▼──────┐  ┌─────▼─────┐
│ Moore  │  │   NVIDIA    │  │   CUDA     │  │  其他硬件  │
│ 后端   │  │   后端      │  │   核函数    │  │  后端      │
│(MUSA)  │  │ (CUDA)      │  │  (共享)    │  │ (ascend/  │
│        │  │             │  │            │  │  bang等)  │
└───┬────┘  └──────┬──────┘  └──────┬──────┘  └───────────┘
    │              │                │
    │         ┌────▼─────────────────▼────┐
    │         │   逐元素操作框架           │
    │         │ (elementwise_moore/nvidia) │
    │         │  - ElementwiseInfo         │
    │         │  - DeviceImpl              │
    │         │  - 通用核函数调度          │
    │         └────────────────────────────┘
    │
┌───▼─────────────────────────────────────────────────┐
│          设备端内核实现                             │
│  - Moore: siwglu_moore_kernel.h (SwiGLUOp functor)  │
│  - NVIDIA: ../cuda/kernel.cuh (SwiGLUOp functor)   │
└─────────────────────────────────────────────────────┘
```

### 3.2 数据流与执行流程

#### 完整计算流程（以 Moore/NVIDIA 后端为例）

1. **描述符创建阶段** (Host 端)
   ```
   用户调用 infiniopCreateSwiGLUDescriptor
   → 验证输入张量形状一致性 (up, gate, output 必须同形)
   → 检查数据类型支持 (F16/BF16/F32/F64)
   → 创建 ElementwiseInfo 元数据（形状、步长、广播标志）
   → 初始化 DeviceImpl (设备实现对象)
   → 计算工作空间大小 = 元数据大小 + 输入指针数组大小
   ```

2. **工作空间分配** (Host 端 → Device 端)
   ```
   查询所需工作空间大小
   → 在设备端分配内存 (musaMalloc/cudaMalloc)
   → 工作空间布局:
     [输入指针数组 (2 * sizeof(void*))]
     [输出形状 (ndim * sizeof(size_t))]
     [输出步长 (ndim * sizeof(ptrdiff_t))]
     [输入形状 (2 * ndim * sizeof(size_t))]
     [输入步长 (2 * ndim * sizeof(ptrdiff_t))]
     [连续标志 (2 * sizeof(bool))]
     [广播标志 (2 * sizeof(bool))]
   ```

3. **计算执行阶段** (Host 端调度)
   ```
   用户调用 infiniopSwiGLU
   → 检查工作空间大小是否充足
   → 根据 dtype 分发到模板实例化 (half/bfloat16/float/double)
   → 异步传输元数据到设备 (musaMemcpyAsync/cudaMemcpyAsync)
   → 计算网格配置:
       block_size = 256 (固定)
       grid_size = min(ceil(output_size / 256), max_grid_dim_x)
   → 启动 CUDA/MUSA 核函数:
       <<<grid_size, 256, 0, stream>>>
   ```

4. **内核执行阶段** (Device 端)
   ```
   每个线程处理一个输出元素:
   → 计算全局索引: idx = blockIdx.x * blockDim.x + threadIdx.x
   → 广播处理: indexToOffset(idx, shape, strides) → 物理偏移
   → 加载数据: up_val = up[up_offset], gate_val = gate[gate_offset]
   → 执行 SwiGLU 操作:
       sig = sigmoid(gate_val)  // 使用向量化 intrinsic
       output_val = gate_val * sig * up_val  // 融合乘加
   → 存储结果: output[idx] = output_val
   ```

5. **同步与清理**
   ```
   流同步: musaStreamSynchronize/cudaStreamSynchronize
   → 释放工作空间和描述符
   ```

### 3.3 平台差异化实现

#### Moore (MUSA) 平台特殊优化
- **Half 精度 Sigmoid**: 因缺少 `hrcp` 支持，提升到 float 计算
  ```cpp
  float xf = __half2float(x);
  float sigf = 1.0f / (1.0f + std::exp(-xf));
  return __float2half(sigf);
  ```
- **BFloat16 向量化**: 使用 `__low2float()` / `__high2float()` 单步提取转换
- **命名空间**: 保持 `op::swiglu::cuda` 兼容性

#### NVIDIA (CUDA) 平台优化
- **Half2 向量化**: `h2rcp(1 + h2exp(-x))` 直接 half 精度计算
- **BFloat16 处理**: 两步转换 `__low2bfloat16` + `__bfloat162float`
- **性能最大化**: 充分利用 Tensor Core 和 CUDA Core 流水线

### 3.4 关键设计模式

- **Pimpl 模式**: `DeviceImpl` 通过 `Opaque` 隐藏实现细节
- **策略模式**: `SwiGLUOp` 作为可插拔函数对象，支持不同逐元素操作
- **工厂模式**: `operator.cc` 中的 `CREATE(CASE, NAMESPACE)` 宏实现多后端分发
- **CRTP 变体**: `ELEMENTWISE_DESCRIPTOR` 宏生成继承层次结构

### 3.5 性能优化技术

- **向量化指令**: half2/cuda_bfloat162 打包处理，吞吐量翻倍
- **快速数学函数**: `h2rcp` / `__frcp_rn` 硬件倒数替代除法
- **融合操作**: `gate * sigmoid(gate) * up` 单次内存访问
- **广播优化**: 连续张量使用线性索引，非连续张量自动计算偏移
- **异步执行**: 元数据传输与计算重叠，隐藏延迟

## 4. 依赖关系

### 4.1 内部依赖
- `infiniop/elementwise/{moore,nvidia}/elementwise_moore_api.h`: 逐元素操作框架宏
- `infiniop/elementwise/{moore,nvidia}/elementwise_moore.h`: 通用设备实现引擎
- `infiniop/devices/{moore,nvidia}/moore_common.h`: 设备通用定义和错误检查宏
- `infiniop/devices/{moore,nvidia}/moore_kernel_common.h`: 内核工具函数（indexToOffset）

### 4.2 外部依赖
- **Moore**: MUSA 运行时 API、MUDA 内置函数（half2、__hmul2 等）
- **NVIDIA**: CUDA Runtime API、CUDA intrinsic（h2rcp、__frcp_rn 等）
- **通用**: C++ 标准库（std::vector、std::shared_ptr、std::enable_if_t）

### 4.3 硬件要求
- Moore 系列 GPU（如 MTT S80），支持 FP16/BF16 硬件加速
- NVIDIA GPU（支持 Tensor Core 的现代架构）
- 其他国产 AI 芯片（ascend、bang、kunlun、metax）

## 5. 当前状态

- **已完成文档**: moore、nvidia (2/8)
- **缺失文档**: ascend、bang、cpu、cuda、kunlun、metax (6/8)
- **文档覆盖率**: 25%

建议优先补充 ascend、bang、kunlun、metax 等国产硬件后端的文档，以完善整体架构视图。
