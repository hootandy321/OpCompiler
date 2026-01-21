# GELU 算子模块架构全景

## 1. 子系统职责

GELU（Gaussian Error Linear Unit）是 InfiniOp 框架中实现的高斯误差线性单元激活函数算子。该模块负责在多种硬件后端上提供统一的 GELU 激活计算接口，支持不同数据类型（FP16、BF16、FP32、FP64）的逐元素操作。作为深度学习模型中的核心激活函数，GELU 在 Transformer 架构（如 BERT、GPT）中得到广泛应用，相比传统的 ReLU，它提供更平滑的梯度过渡和更好的性能表现。

该模块采用分层架构设计：
- **顶层接口**：提供统一的 C API（`infiniopCreateGeluDescriptor`、`infiniopGelu`、`infiniopDestroyGeluDescriptor`），屏蔽硬件差异
- **设备路由层**：根据设备类型（CPU、NVIDIA GPU、昆仑、MetaX 等）动态分发到对应的后端实现
- **后端实现层**：各硬件平台提供优化的具体实现，基于 InfiniOp 的逐元素操作（elementwise）框架构建

## 2. 模块导航

* **cpu**: CPU 后端实现（文档缺失）
    * **功能**: 基于 CPU 标量计算的 GELU 激活函数实现，适用于无加速器环境
    * **职责**: 提供跨平台的 CPU 通用实现，支持串行和多线程（OpenMP）计算模式

* **cuda**: CUDA 后端通用实现（文档缺失）
    * **功能**: 提供 NVIDIA GPU 的通用 CUDA 内核实现
    * **职责**: 定义 GELU 的 CUDA 设备端计算逻辑（`GeluOp` 函数对象），被 nvidia 子目录引用

* **nvidia**: NVIDIA GPU 算子核心实现（**已归档**）
    * **功能**: 完整的 NVIDIA GPU CUDA 后端实现，基于 InfiniOp 逐元素操作框架
    * **职责**: 提供 NVIDIA GPU（包括兼容的天数智芯、QY 等）的完整描述符和计算实现，支持 FP16、BF16、FP32、FP64 四种浮点数据类型，使用 256 线程/块的配置优化性能
    * **核心组件**:
      - `Descriptor` 类：封装设备信息、元数据和工作空间管理
      - `GeluOp` 函数对象：实现 `GELU(x) = 0.5 * x * (1 + erf(x / √2))` 数学公式
      - `DeviceImpl` 类：继承自 elementwise 框架，负责 CUDA 核函数启动和分段执行
      - `ElementwiseInfo` 结构体：存储张量形状、步幅、连续性标志等元数据

* **kunlun**: 昆仑芯片后端实现（文档缺失）
    * **功能**: 针对昆仑 AI 加速卡优化的 GELU 实现
    * **职责**: 适配昆仑硬件架构（XPU），提供特定优化的内核实现

* **metax**: MetaX 后端实现（文档缺失）
    * **功能**: 针对 MetaX 硬件平台的 GELU 实现
    * **职责**: 支持 MetaX 设备的计算特性

## 3. 架构逻辑图解

### 3.1 调度流程

GELU 算子模块采用**设备类型路由**模式，通过 `operator.cc` 实现统一的设备分发：

```
用户 API 调用
    ↓
infiniopCreateGeluDescriptor()
    ↓
[设备类型检测]
    ↓
    ├─→ INFINI_DEVICE_CPU     → op::gelu::cpu::Descriptor::create()
    ├─→ INFINI_DEVICE_NVIDIA  → op::gelu::nvidia::Descriptor::create()
    ├─→ INFINI_DEVICE_ILUVATAR → op::gelu::nvidia::Descriptor::create() [复用 NVIDIA 实现]
    ├─→ INFINI_DEVICE_QY       → op::gelu::nvidia::Descriptor::create() [复用 NVIDIA 实现]
    ├─→ INFINI_DEVICE_METAX   → op::gelu::metax::Descriptor::create()
    └─→ INFINI_DEVICE_KUNLUN  → op::gelu::kunlun::Descriptor::create()
```

### 3.2 NVIDIA GPU 实现的数据流

基于已归档的 `nvidia` 子目录文档，NVIDIA 后端的计算流程如下：

```
1. 描述符创建阶段
   ├─ 验证数据类型（BF16/F16/F32/F64）
   ├─ 验证输入输出张量形状一致性
   ├─ 创建 ElementwiseInfo（元数据：形状、步幅、连续性）
   ├─ 计算工作空间大小 = 元数据大小 + 输入指针数组大小
   └─ 创建 DeviceImpl 实例（Pimpl 模式）

2. 计算执行阶段
   ├─ 验证工作空间大小
   ├─ 异步拷贝元数据和输入指针到设备（cudaMemcpyAsync）
   ├─ 计算网格维度 = min(元素数 / 256, 设备最大网格 X)
   ├─ [分段执行] 循环启动核函数
   │   ├─ 每个线程处理一个输出元素
   │   ├─ 线程索引 = blockIdx.x * blockDim.x + threadIdx.x + offset
   │   └─ 调用 GeluOp::operator() 计算 GELU(x)
   └─ 返回执行状态

3. 设备端计算逻辑
   ├─ BF16/FP16: 转换为 float → 计算 → 转回原类型
   ├─ FP32: 直接计算 0.5 * x * (1 + erf(x / √2))
   └─ FP64: 双精度计算 0.5 * x * (1 + erf(x / √2))
```

### 3.3 模块间依赖关系

```
operator.cc（设备路由层）
    ↓
    ├─→ cpu/gelu_cpu.h（CPU 实现）
    ├─→ nvidia/gelu_nvidia.cuh（NVIDIA 实现）
    │   ├─→ cuda/kernel.cuh（GeluOp 定义）
    │   ├─→ elementwise/nvidia/elementwise_nvidia.cuh（逐元素框架）
    │   └─→ devices/nvidia/nvidia_common.cuh（设备抽象）
    ├─→ metax/gelu_metax.h（MetaX 实现）
    └─→ kunlun/gelu_kunlun.h（昆仑实现）
```

### 3.4 设计模式应用

1. **策略模式**：`DeviceImpl` 封装不同计算策略（相同输入类型 vs 混合类型）
2. **模板方法模式**：`calculate()` 定义算法骨架，具体实现由子类提供
3. **工厂模式**：`Descriptor::create()` 作为静态工厂方法构造描述符
4. **Pimpl 模式**：`DeviceImpl::Opaque` 隐藏实现细节，减少编译依赖
5. **宏元编程**：使用 `ELEMENTWISE_DESCRIPTOR` 宏自动生成描述符代码，减少重复

### 3.5 性能优化策略

**编译时优化**：
- 模板特化为每种数据类型生成专用核函数
- `if constexpr` 在编译期消除死代码
- `__device__ __forceinline__` 强制内联计算函数

**运行时优化**：
- 256 线程/块的固定配置（平衡寄存器占用和并行度）
- 分段执行支持大型张量（超过网格限制时循环启动）
- 连续张量使用线性索引，避免偏移计算开销
- 异步内存传输（`cudaMemcpyAsync`）隐藏延迟

**数值稳定性**：
- 低精度类型（BF16/FP16）在 float 精度下计算，避免累积误差
- 使用 CUDA 内置优化数学函数（`erf`, `sqrt`）

## 4. 待完善文档

以下子目录尚未完成代码文档归档，需要在后续迭代中补充：
- `cpu/` - CPU 后端的实现细节和性能优化策略
- `cuda/` - CUDA 通用内核的实现（`GeluOp`）和数据类型处理
- `kunlun/` - 昆仑芯片的硬件适配和内核优化
- `metax/` - MetaX 平台的实现细节

## 5. 关键文件路径

- **主接口**: `/home/qy/src/Infini/InfiniCore/src/infiniop/ops/gelu/operator.cc`
- **NVIDIA 实现**:
  - `/home/qy/src/Infini/InfiniCore/src/infiniop/ops/gelu/nvidia/gelu_nvidia.cuh`
  - `/home/qy/src/Infini/InfiniCore/src/infiniop/ops/gelu/nvidia/gelu_nvidia.cu`
- **CUDA 内核**: `/home/qy/src/Infini/InfiniCore/src/infiniop/ops/gelu/cuda/kernel.cuh`
- **逐元素框架**: `/home/qy/src/Infini/InfiniCore/src/infiniop/elementwise/nvidia/elementwise_nvidia.cuh`
