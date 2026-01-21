# 📂 目录: conv (卷积操作) 架构全景

## 1. 子系统职责

`conv` 目录是 Infini 框架中卷积操作（Convolution）的核心实现层，负责在不同硬件后端上提供高性能的卷积计算能力。作为深度学习中最基础且计算密集的操作之一，卷积在卷积神经网络（CNN）中占据核心地位。该子系统通过硬件抽象层，向上层框架提供统一的卷积接口，向下调用不同硬件厂商的优化库（如 cuDNN）或实现自定义算法，支持 1D/2D/3D 卷积、偏置项融合、多种数据类型（FP16/FP32/BF16）以及灵活的卷积参数配置（填充、步长、扩张）。

## 2. 模块导航

* **📂 nvidia**:
    * *功能*: 基于 NVIDIA GPU 的卷积操作实现，通过 cuDNN 库提供高性能的前向传播卷积计算
    * *职责*: 封装 cuDNN API，管理卷积描述符生命周期，提供自动算法选择和工作空间管理，支持带偏置和不带偏置的卷积模式

* **📂 cpu**:
    * *功能*: CPU 后端的卷积操作实现（文档缺失）
    * *职责*: 提供 CPU 设备上的卷积计算能力，具体实现细节需要参考源代码

## 3. 架构逻辑图解

### 数据流与组件交互

`conv` 子系统采用硬件后端隔离的架构设计，不同硬件实现位于独立的子目录中，通过统一的接口向上层提供服务：

1. **接口抽象层** (`conv.h`, `info.h`):
   - 定义卷积操作的通用接口和数据结构
   - 声明跨后端的卷积元数据类型（如 `ConvInfo`）
   - 提供硬件无关的参数验证和形状计算逻辑

2. **硬件实现层** (`nvidia/`, `cpu/`):
   - **NVIDIA 实现**：
     - 核心类 `Descriptor::Opaque` 封装所有 cuDNN 状态（张量描述符、卷积描述符、算法选择）
     - 通过 `Descriptor` 对外暴露统一接口，隐藏 cuDNN 实现细节（Pimpl 模式）
     - 支持两种执行路径：
       - 无偏置模式：调用 `cudnnConvolutionForward()`
       - 有偏置模式：调用 `cudnnConvolutionBiasActivationForward()`（算子融合优化）
     - 自动算法选择：使用 `cudnnFindConvolutionForwardAlgorithm()` 启发式搜索最优算法
     - 工作空间管理：通过 `workspaceSize()` 查询需求，由调用方分配 GPU 内存
   - **CPU 实现**：
     - 文档缺失，从文件结构推测包含 `conv_cpu.cc` 和 `conv_cpu.h`
     - 应提供与 NVIDIA 后端兼容的接口，实现 CPU 上的卷积算法

3. **操作编排层** (`operator.cc`):
   - 注册卷积操作到 Infini 算子库
   - 提供运行时算子分派逻辑，根据设备类型选择对应后端实现

### 执行流程

典型的卷积操作执行流程：

```
用户调用 → Descriptor::create() → 参数验证 → ConvInfo 创建
                                    ↓
                            Opaque::create() → cuDNN 描述符初始化
                                                    ↓
                                            算法选择与工作空间计算
                                                    ↓
                                    返回 Descriptor 对象
                                                    ↓
用户调用 → Descriptor::calculate() → 工作空间绑定
                                    ↓
                            cuDNN 卷积 kernel 启动
                                    ↓
                            异步执行（CUDA Stream）
                                    ↓
                            返回计算结果
```

### 设计特点

- **硬件后端隔离**：每个硬件实现独立维护，通过宏机制（`DESCRIPTOR(nvidia)`）生成统一接口
- **RAII 资源管理**：cuDNN 描述符在对象析构时自动释放，避免资源泄漏
- **算子融合**：偏置加法和激活函数融合到单次 kernel 调用，减少内存访问和中间结果存储
- **性能优化**：自动算法选择、工作空间复用、异步执行（CUDA Stream）
- **错误处理**：使用 `utils::Result<T>` 封装可能失败的操作，统一错误码转换
- **1D 卷积优化**：1D 卷积转换为 2D 张量处理，复用 cuDNN 的高性能 2D 卷积实现

### 依赖关系

- **向上依赖**：依赖 `device::nvidia::Handle` 访问 cuDNN 上下文，依赖 `ConvInfo` 管理卷积元数据
- **向下依赖**：NVIDIA 后端强依赖 cuDNN 库（通过 `ENABLE_CUDNN_API` 宏控制编译）
- **横向协作**：与其他算子系统（如 matmul、pooling）共享设备抽象层和工具类

### 扩展性

新增硬件后端需要：
1. 创建对应子目录（如 `amd/`、`ascend/`）
2. 实现 `Descriptor::Opaque` 类封装硬件特定状态
3. 使用 `DESCRIPTOR(硬件名)` 宏生成统一接口
4. 提供与现有后端兼容的 `create()` 和 `calculate()` 方法
5. 在 `operator.cc` 中注册新后端的算子工厂
