# 目录: InfiniCore/src/infiniop/ops/sub 架构全景

## 1. 子系统职责

本目录是 Infini 框架中**张量减法运算（Element-wise Subtraction）**的多硬件后端实现层。它向上承接统一的算子接口，向下适配不同的计算设备（NVIDIA GPU、CPU、昆仑、Metax 等），提供高性能的逐元素减法计算能力。该子系统遵循 Infini 的硬件抽象设计理念，通过统一的描述符接口和差异化的内核实现，实现了跨硬件平台的减法运算支持。

## 2. 模块导航

### 硬件后端实现模块

* **📂 nvidia**:
    * *功能*: NVIDIA CUDA GPU 加速的减法算子完整实现，支持 FP16/BF16/FP32/FP64 数据类型，具备广播、步长张量、异步流执行等高级特性
    * *职责*: 基于 CUDA 的 Elementwise 框架实现高性能 GPU 减法内核，包含描述符管理、工作空间计算、内核启动优化等完整功能

* **📂 cpu**:
    * *功能*: 基于 CPU 的减法算子实现（包含 sub_cpu.cc 和 sub_cpu.h）
    * *职责*: 提供通用 CPU 后端的减法运算能力
    * *文档状态*: **文档缺失**

* **📂 cuda**:
    * *功能*: CUDA 通用内核定义（包含 kernel.cuh，定义设备端减法运算函数对象）
    * *职责*: 提供可复用的 CUDA 内核函数对象，支持不同数据类型的减法语义
    * *文档状态*: **文档缺失**

* **📂 kunlun**:
    * *功能*: 昆仑（XPU）加速卡后端实现（包含 sub_kunlun.xpu 和 kernel.h）
    * *职责*: 适配昆仑硬件的减法算子，提供 XPU 特定的内核实现
    * *文档状态*: **文档缺失**

* **📂 metax**:
    * *功能*: Metax 硬件后端实现（包含 sub_metax.maca）
    * *职责*: 提供 Metax 架构的减法运算支持
    * *文档状态*: **文档缺失**

### 顶层编排文件

* **📄 operator.cc**:
    * *功能*: 减法算子的工厂注册和分发逻辑
    * *职责*: 根据设备类型动态选择对应的后端实现（nvidia/cpu/kunlun/metax），统一向上层提供算子创建接口

## 3. 架构逻辑图解

### 数据流向与模块交互

```
上层调用 (InfiniOp API)
    ↓
operator.cc (设备类型分发)
    ↓
    ├─→ NVIDIA GPU → nvidia/sub_nvidia.cu → Elementwise 框架 → CUDA 内核
    ├─→ CPU → cpu/sub_cpu.cc → CPU 向量化实现
    ├─→ Kunlun → kunlun/sub_kunlun.xpu → XPU 内核
    └─→ Metax → metax/sub_metax.maca → Metax 内核
```

### 模块协作机制

1. **统一接口层**：
   - `operator.cc` 作为统一入口，根据 `infiniopHandle_t` 中的设备类型信息，将算子创建请求路由到对应的后端实现
   - 所有后端遵循相同的描述符接口约定（Descriptor 类），确保上层调用的一致性

2. **代码复用层次**：
   - **CUDA 通用层** (`cuda/kernel.cuh`)：定义可复用的设备端函数对象（如 `SubOp`），封装减法运算的语义，通过模板特化支持不同数据类型
   - **NVIDIA 完整实现** (`nvidia/`)：基于 Elementwise 框架实现完整的描述符逻辑，包括元数据管理、工作空间计算、内核调度优化等
   - **其他硬件后端**：各自实现适配其硬件特性的内核和接口（目前文档缺失，具体实现细节未知）

3. **执行流程**（以 NVIDIA 为例）：
   - **创建阶段**：
     - `Descriptor::create()` 验证数据类型（仅支持浮点类型）和形状一致性
     - 初始化 `ElementwiseInfo` 存储张量元数据（形状、步长、广播标志）
     - 创建 `DeviceImpl` 对象，计算并存储工作空间大小
   - **执行阶段**：
     - `calculate()` 验证工作空间大小，异步拷贝元数据到 GPU
     - 根据数据类型分发到对应的模板特化版本
     - 启动 CUDA 内核（BLOCK_SIZE=256），处理大张量的自动分块
     - 内核中通过 `InputIndexer` 处理广播和步长，调用 `SubOp` 函数对象执行实际减法

4. **性能优化策略**：
   - **硬件指令优化**：针对不同数据类型使用专用指令（FP16 用 `__hsub`，FP32 用 `__fsub_rd`）
   - **向量化计算**：通过 `half2` 和 `cuda_bfloat162` 实现 SIMD 指令
   - **并发执行**：固定 256 线程/块，自动分块处理大张量，支持 CUDA 流异步执行
   - **内存优化**：工作空间仅存储元数据和输入指针数组，避免数据冗余拷贝

### 设计模式应用

- **策略模式**：不同硬件后端作为可插拔策略，通过工厂方法动态选择
- **模板方法模式**：Elementwise 框架定义算法骨架，具体运算（减法）通过模板参数特化
- **Pimpl 惯用法**：DeviceImpl 通过 Opaque 结构隐藏硬件实现细节
- **CRTP**：ELEMENTWISE_DESCRIPTOR 宏生成继承自 InfiniopDescriptor 的类，实现编译期多态

### 文档覆盖率说明

- **已文档化**：nvidia 后端（完整详细，包含类、接口、示例、实现细节）
- **未文档化**：cpu, cuda, kunlun, metax 四个后端实现（仅有源代码，缺少技术文档）
- **建议**：需要对未文档化的后端进行代码分析，补全对应的 CODEREADME.md，以建立完整的子系统知识图谱
