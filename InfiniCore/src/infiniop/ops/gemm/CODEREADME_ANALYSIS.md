# GEMM 算子架构全景

## 1. 子系统职责

本目录 `./InfiniCore/src/infiniop/ops/gemm` 是 InfiniOp 框架中**通用矩阵乘法（General Matrix Multiply, GEMM）算子的多硬件后端实现层**。作为核心算子之一，GEMM 是深度学习计算（如线性层、注意力机制、卷积等）的基础计算单元。该目录的设计体现了"同一算子接口，多种硬件实现"的架构理念，为不同的计算硬件（NVIDIA GPU、华为昇腾、寒武纪、沐曦、天数智芯、昆仑等）提供统一的高性能矩阵乘法能力。

在 InfiniOp 整体架构中，本目录位于算子实现的硬件抽象层，向上通过统一的 `Descriptor` 接口为算子调度层提供服务，向下调用各硬件厂商提供的数学库（如 cuBLAS、hccl、mublas、bangc 等）。这种设计使得上层框架（如 InfiniLM、InfiniTrain）可以完全屏蔽底层硬件差异，实现"一次编写，多硬件部署"。

## 2. 模块导航

- **nvidia**: NVIDIA CUDA GPU 后端实现，基于 cuBLAS 库提供高性能矩阵乘法。支持 FP16、BF16、FP32 三种数据类型，支持批量 GEMM，通过 Tensor Core 加速，自动处理矩阵转置和内存布局优化。使用 PImpl 模式隐藏 CUDA 特定实现，采用句柄池机制管理 cuBLAS 资源，实现零拷贝转置和流式并发执行。

- **ascend**: 华为昇腾（Ascend）NPU 后端实现，文档缺失。包含 `gemm_ascend.cc/h` 实现文件。

- **bang**: 壁仞科技（Biren）GPU 后端实现，文档缺失。包含 `gemm_bang.cc/h` 实现文件。

- **cpu**: CPU 后端实现，文档缺失。包含 `gemm_cpu.cc/h` 实现文件。

- **kunlun**: 昆仑芯（Kunlun）GPU 后端实现，文档缺失。包含 `gemm_kunlun.cc/h` 实现文件。

- **metax**: 天数智芯（Metax）GPU 后端实现，文档缺失。包含 `gemm_metax.cc/h` 实现文件。

- **moore**: 沐曦（Moore）GPU 后端实现，文档缺失。目录结构较为复杂，包含 `gemm_moore.h` 头文件以及 `mublas`、`mudnn` 两个子目录，可能支持沐曦 BLAS 和 DNN 两种计算路径。

## 3. 架构逻辑图解

### 3.1 统一抽象接口

所有硬件后端共享相同的顶层接口设计（以 NVIDIA 实现为参考）：

```cpp
// 通用接口模式（跨所有硬件后端）
class Descriptor {
    // 工厂方法：创建硬件特定的 GEMM 描述符
    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t c_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc);

    // 查询所需工作空间大小
    size_t workspaceSize() const;

    // 执行矩阵乘法：C = alpha * A * B + beta * C
    infiniopStatus_t calculate(
        void *workspace, size_t workspace_size,
        void *c, float beta,
        const void *a, const void *b, float alpha,
        void *stream) const;
};
```

这种统一接口使得上层算子注册系统可以以多态方式调度不同硬件的实现，而无需关心底层细节。

### 3.2 数据流转路径

从上层调用到硬件执行的完整数据流：

1. **算子创建阶段**：
   - 上层框架（如 InfiniLM）通过 `op::gemm::{backend}::Descriptor::create()` 创建 GEMM 描述符
   - 描述符解析输入张量的形状、步长、数据类型，验证维度兼容性
   - 针对硬件特性进行布局优化（如 NVIDIA 实现中的自动转置检测）
   - 初始化硬件特定的资源（如 cuBLAS 句柄、昇腾 CANN 句柄等）

2. **计算执行阶段**：
   - 上层调用 `calculate()` 接口，传入输入/输出数据指针和标量系数
   - 各后端将数据指针和参数映射到对应硬件库的 API：
     - **NVIDIA**: 调用 `cublasGemmStridedBatchedEx()`，支持 Tensor Core
     - **Ascend**: 调用华为 HCCL/HCOM APIs（推测）
     - **Bang**: 调用壁仞 BANG APIs（推测）
     - **Moore**: 调用沐曦 muBLAS APIs（推测）
   - 计算在硬件流上异步执行（CUDA Stream、昇腾 Stream 等）

3. **结果返回阶段**：
   - `calculate()` 返回成功状态
   - 上层框架负责流同步和结果回传

### 3.3 硬件后端差异处理

尽管接口统一，不同硬件后端在实现细节上存在关键差异：

**NVIDIA (参考实现)**：
- 使用 cuBLAS 库的 `GemmStridedBatchedEx` API
- 支持 Tensor Core 加速（FP16: `CUBLAS_COMPUTE_32F`，F32: `CUBLAS_COMPUTE_32F_FAST_TF32`）
- 自动处理行主序/列主序转换，通过逻辑转置避免内存拷贝
- 句柄池机制支持多流并发

**Ascend (华为昇腾)**：
- 推测使用华为 CANN 框架的 HCOM/HCCL APIs
- 可能针对昇腾的 Cube/CMatrix 单元优化
- 需要适配昇腾的内存布局和流模型

**Bang (壁仞)**：
- 推测使用 BANG C APIs
- 壁仞 GPU 的通用矩阵乘法加速单元优化

**Moore (沐曦)**：
- 目录结构显示存在 `mublas` 和 `mudnn` 两条路径
- 可能支持 muBLAS 库的 GEMM 接口和 muDNN 库的卷积-矩阵乘法转换路径
- 提供了更多实现选择空间

### 3.4 共享基础设施

所有硬件后端共享以下基础设施（位于父目录）：

- **`gemm.h`**: GEMM 算子的通用类型定义和宏
- **`info.h`**: 硬件无关的矩阵乘法信息封装
  - `MatmulInfo`: 封装 GEMM 的维度（M/N/K）、批次、步长等几何信息
  - `BlasMatrix`: 描述单个矩阵的内存布局（行主序/列主序、维度、步长）
  - 提供跨硬件的形状验证和布局优化逻辑

- **`operator.cc`**: 算子注册和分发逻辑，根据硬件类型选择对应后端实现

这种共享基础设施确保了代码复用和一致性，减少各硬件后端的重复开发工作。

### 3.5 设计模式应用

整个 GEMM 子系统采用了多种设计模式：

1. **策略模式（Strategy Pattern）**：
   - 不同硬件后端是不同的矩阵乘法策略
   - 运行时根据硬件类型选择策略（NVIDIA、Ascend、Bang 等）

2. **工厂模式（Factory Pattern）**：
   - `Descriptor::create()` 作为工厂方法，封装复杂的对象构造逻辑
   - 隐藏硬件特定的初始化细节

3. **PImpl 模式（Pointer to Implementation）**：
   - NVIDIA 实现中的 `Opaque` 结构体隐藏 CUDA/cuBLAS 特定类型
   - 保持头文件的平台无关性

4. **RAII（资源获取即初始化）**：
   - 描述符析构时自动释放硬件资源（cuBLAS 句柄等）
   - 使用智能指针管理资源生命周期

### 3.6 性能优化策略

从 NVIDIA 参考实现可推断各后端采用的优化策略：

- **批量计算**：使用 StridedBatched API 一次计算多个矩阵乘法，减少内核启动开销
- **自动布局优化**：在描述符创建阶段检测并适配最优内存布局
- **零拷贝转置**：通过调整 API 参数实现逻辑转置，避免实际内存拷贝
- **硬件加速单元**：
  - NVIDIA: Tensor Core
  - Ascend: Cube/CMatrix 单元
  - Moore/Metax/Kunlun: 各自的矩阵加速器
- **流式并发**：所有计算在硬件流上异步执行，支持流水线重叠

### 3.7 当前文档状态

**已完成文档的子模块**：
- NVIDIA 后端：详细文档覆盖接口、实现、使用示例、性能优化等

**待补充文档的子模块**：
- Ascend、Bang、CPU、Kunlun、Metax、Moore 后端文档缺失
- 这些后端的实现细节、API 差异、性能特性需要在后续文档构建中补充

建议按照智能去重策略，优先完成各硬件后端的代表性实现文档，再覆盖其他后端。
