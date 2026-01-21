# InfiniCore/src 架构全景

## 1. 子系统职责

`InfiniCore/src` 是整个 InfiniCore 框架的核心源代码目录，承载着多硬件后端统一计算框架的完整实现。该目录采用**分层架构设计**，自底向上分为四个核心子系统，共同构建了一个从硬件抽象到用户 API 的完整技术栈。

**核心设计理念**:
- **硬件无关性**: 通过运行时抽象层屏蔽底层硬件差异，支持 NVIDIA GPU、华为昇腾、寒武纪、沐曦、天数智芯、昆仑等多种加速设备
- **统一算子接口**: 为相同算子提供跨平台一致的 C 语言接口，实现"一次编写，多硬件部署"
- **高性能内存管理**: 提供从基础主机分配到高性能设备内存池的多策略内存管理系统
- **跨设备通信**: 封装不同硬件的集合通信原语，支持分布式训练场景

在 InfiniCore 整体架构中，本目录位于实现层的核心位置，向上为 Python 前端和上层框架（InfiniLM、InfiniTrain）提供 C/C++ API，向下调用各硬件厂商提供的底层驱动和数学库。

## 2. 模块导航

### 2.1 运行时层 (infinirt)

* **infinirt**:
    * *功能*: 硬件运行时抽象层（Runtime Abstraction Layer）
    * *职责*: 提供跨硬件的统一设备管理、内存分配、流同步、事件同步等底层运行时 API，是整个框架的硬件抽象基础层

    **核心能力**:
    - **设备管理**: 支持 CPU、NVIDIA、Ascend、Bang、Moore、Metax、Kunlun 等 7+ 种硬件后端
    - **内存操作**: `infinirtMalloc/Free`（设备内存）、`infinirtMallocHost/FreeHost`（页锁定内存）、`infinirtMallocAsync/FreeAsync`（流有序异步分配）
    - **流与事件**: `infinirtStreamCreate/Destroy/Synchronize`、`infinirtEventRecord/Wait/Synchronize`
    - **数据传输**: `infinirtMemcpy` 系列 API，支持 Host-to-Device、Device-to-Host、Device-to-Device 拷贝
    - **设备属性查询**: `infinirtGetDeviceCount`、`infinirtGetDevice`、`infinirtSetDevice`

    **架构特点**:
    - 使用 `INFINIRT_CALL_DEVICE_API` 宏实现运行时硬件分发
    - Thread-local 存储管理当前设备上下文（`CURRENT_DEVICE_TYPE`、`CURRENT_DEVICE_ID`）
    - 各硬件后端独立实现于子目录（`cuda/`、`ascend/`、`bang/` 等）
    - PImpl 模式隐藏硬件特定类型，保持头文件平台无关

### 2.2 算子层 (infiniop)

* **infiniop**:
    * *功能*: 统一底层算子框架（Unified Operator Framework）
    * *职责*: 为相同算子在不同平台提供统一的 C 语言多段式接口，包括创建描述符、获取工作空间、执行算子、销毁描述符四个标准阶段

    **核心算子分类**:
    - **矩阵运算**: GEMM（通用矩阵乘法）
    - **逐元素运算**: Add、Sub、Mul、Clip、ReLU、Sigmoid、Tanh、GELU、SiLU、SwiGLU 等
    - **归约运算**: LayerNorm、RMSNorm、LPNorm、Softmax、LogSoftmax、CausalSoftmax
    - **注意力机制**: PagedAttention、PagedAttentionPrefill、TopKRouter、TopKSoftmax
    - **张量操作**: Zeros、Ones、Rearrange、RandomSample
    - **特殊算子**: Conv、DequantizeAWQ、AddRMSNorm

    **架构特点**:
    - **Descriptor 模式**: 每个算子通过描述符封装元数据和硬件特定状态
    - **多硬件后端**: 每个算子在 `ops/[op]/[device]/` 目录下有独立实现
    - **公共代码复用**:
      - `elementwise/`: 逐元素算子的通用实现框架
      - `reduce/`: 归约计算的通用内核
      - `binary/`: 二元运算的公共逻辑
      - `sort/`: 排序算子的通用实现
    - **工作空间机制**: 部分算子（如 GEMM）需要额外临时空间，通过 `GetWorkspaceSize` 接口查询
    - **设备分发机制**: 通过 `operator.cc` 根据设备类型选择对应后端实现

    **代表性算子详解**:
    - **GEMM** (`ops/gemm/`): 支持 NVIDIA cuBLAS、Ascend CANN、Moore muBLAS 等多种数学库，自动处理矩阵转置和布局优化
    - **Elementwise** (`elementwise/`): 基于 `ElementwiseInfo` 封装张量元数据，支持广播机制，通过 `ELEMENTWISE_DESCRIPTOR` 宏生成统一接口

### 2.3 核心库层 (infinicore)

* **infinicore**:
    * *功能*: C++ 高层 API 封装层（High-Level C++ API Layer）
    * *职责*: 基于 C++ 实现张量抽象、算子接口、内存管理、上下文管理等用户可见的核心功能，并通过 pybind11 暴露给 Python 前端

    **核心模块**:
    - **张量系统** (`tensor/`):
      - `Tensor` 类：封装设备内存、形状、数据类型、步长等元数据
      - 支持自动微分（未来扩展）、视图（view）、切片（slice）等高级操作

    - **上下文管理** (`context/`):
      - **设备上下文**: 管理当前活动设备、流、事件
      - **内存分配器** (`allocators/`):
        - `HostAllocator`: 基于 malloc/free 的主机内存分配
        - `DevicePinnedHostAllocator`: 页锁定内存分配，支持跨设备延迟释放
        - `PinnableBlockAllocator`: ⭐ 高性能 Size-Class 内存池，支持固定模式与普通模式切换
        - `StreamOrderedAllocator`: 流有序异步内存分配，与 CUDA Stream 同步

    - **算子系统** (`ops/`):
      - **Dispatcher 机制**: 每个算子通过 `OpDispatcher` 实现运行时 kernel 注册和查找
      - **双重实现路径**:
        - InfiniOP 实现路径：为所有平台注册同一 InfiniOP 算子，利用 InfiniOP 的硬件分发
        - 独立实现路径：为单个或多个设备单独实现 kernel
      - **Python 绑定** (`pybind11/ops/`): 将 C++ 算子暴露给 Python
      - **统一测试框架** (`/test/infinicore/ops/`): 与 PyTorch 对比正确性和性能

    - **神经网络层** (`nn/`): 高层神经网络模块（未来扩展）
    - **计算图** (`graph/`): 计算图构建和优化（未来扩展）
    - **设备抽象** (`device.cc`): 设备类型和设备 ID 的封装

    **开发指南亮点**:
    - 算子定义四步走：定义 schema → 实现 execute → 注册 kernel → 编写测试
    - 支持 inplace 和 out-of-place 两种计算模式
    - 通过 `INFINICORE_ASSERT_TENSORS_SAME_DEVICE` 等宏保证类型安全
    - 使用 thread-local 缓存（`OpCache`）管理算子描述符生命周期

### 2.4 通信层 (infiniccl)

* **infiniccl**:
    * *功能*: 集合通信抽象层（Collective Communication Layer）
    * *职责*: 封装不同硬件的集合通信库（NCCL、HCCL、CCL 等），提供跨硬件的统一通信接口

    **核心通信原语**:
    - **AllReduce**: `infinicclAllReduce` - 所有设备的归约并广播结果
    - **通信域管理**: `infinicclCommInitAll`、`infinicclCommDestroy`
    - **支持设备**: NVIDIA（NCCL）、Ascend（HCCL）、Cambricon、Kunlun、Metax、Moore

    **架构特点**:
    - 使用 `COMM_INIT_ALL`、`ALL_REDUCE` 等宏实现硬件分发
    - 通信域（`infinicclComm_t`）封装设备类型和硬件特定句柄
    - 与 infinirt 紧密集成，使用 `infinirtStream_t` 同步异步通信操作

### 2.5 工具模块 (utils)

* **utils**:
    * *功能*: 通用工具函数和类型定义
    * *职责*: 提供错误检查、自定义类型、数组重排等辅助功能

    **核心文件**:
    - `check.h`: 错误检查宏（`INFINICORE_CHECK_ERROR`、`INFINICORE_ASSERT`）
    - `custom_types.h/cc`: 自定义数据类型封装
    - `rearrange.cc/h`: 数组重排和转置操作
    - `result.hpp`: Result<T> 模式，用于异常安全的错误返回

## 3. 架构逻辑图解

### 3.1 垂直分层架构

```
┌─────────────────────────────────────────────────────────┐
│                    Python Frontend                      │
│              (InfiniLM, InfiniTrain, etc.)              │
└────────────────────┬────────────────────────────────────┘
                     │ pybind11
┌────────────────────▼────────────────────────────────────┐
│                   infinicore (C++ API)                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │  Tensor  │  │  Context │  │   Ops    │  │   NN   │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘  │
│       │              │              │                     │
│       └──────────────┴──────────────┴─────────────────────►
│       Dispatcher (OpDispatcher), Allocators, Graph       │
└────────────────────┬────────────────────────────────────┘
                     │ 统一 C 接口
┌────────────────────▼────────────────────────────────────┐
│                   infiniop (Operator Layer)             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │
│  │  GEMM   │  │Element- │  │  Norms  │  │Attention│   │
│  │         │  │  wise   │  │         │  │         │   │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘   │
│       │            │            │            │          │
│       └────────────┴────────────┴────────────┴──────────►
│       Descriptor Create → GetWorkspace → Calculate     │
└────────────────────┬────────────────────────────────────┘
                     │ 硬件分发
       ┌─────────────┼─────────────┐
       │             │             │
┌──────▼──────┐ ┌───▼────┐ ┌─────▼──────┐
│  infinirt   │ │infini- │ │ infini-    │
│ (Runtime)   │ │  ccl   │ │  test*     │
└──────┬──────┘ └───┬────┘ └─────┬──────┘
       │            │             │
┌──────▼───────────▼─────────────▼──────────┐
│         Hardware Backend Layer            │
│  CUDA | Ascend | Bang | Moore | Metax ... │
└───────────────────────────────────────────┘
```

### 3.2 水平数据流转路径

#### 场景 1: 单设备矩阵乘法 (GEMM)

```
用户调用: infinicore::gemm(tensor_a, tensor_b)
    ↓
infinicore/ops/gemm.cpp::gemm()
    ├─ 创建输出张量 Tensor C
    └─ 调用 Gemm::execute(c, a, b, alpha, beta)
        ↓
Gemm::execute()
    ├─ 检查张量设备一致性 (INFINICORE_ASSERT_TENSORS_SAME_DEVICE)
    ├─ 设置当前设备 (context::setDevice)
    └─ Dispatcher.lookup(device_type)(c, a, b, alpha, beta)
        ↓
infiniop 实现路径 (infinicore/ops/gemm/infiniop/)
    ├─ 查找线程局部缓存: OpCache 查找或创建 Descriptor
    ├─ 获取工作空间: infiniopGetGemmWorkspaceSize
    └─ 调用计算: infiniopGemm(desc, workspace, c, a, b, alpha, beta, stream)
        ↓
infiniop/ops/gemm/nvidia/ (NVIDIA 后端为例)
    ├─ Descriptor::calculate()
    │   ├─ 从 Opaque 结构体获取 cuBLAS 句柄
    │   ├─ 处理矩阵布局（行主序/列主序转置）
    │   └─ 调用 cublasGemmStridedBatchedEx()
    └─ 在 CUDA Stream 上异步执行
        ↓
infinirt/cuda/
    └─ infinirtStreamSynchronize() 等待完成
```

#### 场景 2: 多设备分布式训练 (AllReduce)

```
初始化阶段: infinicclCommInitAll()
    ├─ 检测设备类型（如 NVIDIA）
    └─ 调用 infiniccl::cuda::commInitAll()
        └─ 初始化 NCCL 通信域
            ↓
训练循环: infinicore::op::compute_gradients()
    ↓
调用: infinicclAllReduce(sendbuf, recvbuf, count, dtype, op, comm, stream)
    ├─ 检查通信域有效性
    └─ 根据 comm->device_type 分发
        ↓
infiniccl/cuda/ (NVIDIA 后端)
    └─ infiniccl::cuda::allReduce()
        └─ ncclAllReduce() via NCCL library
            ↓
infinirt/cuda/
    └─ infinirtStreamSynchronize() 等待通信完成
```

#### 场景 3: 内存分配与释放 (PinnableBlockAllocator)

```
推理/训练阶段: 正常模式
    ↓
创建 PinnableBlockAllocator(gpu_device)
    ├─ 初始化 11 个 Size-Class (32KB ~ 256MB)
    └─ pinned_mode_ = false (默认)
        ↓
循环推理: allocator.allocate(16MB)
    ├─ 对齐到 256 字节边界
    ├─ 查找 size_classes_: 命中 16MB 等级
    ├─ 检查 free_blocks: 首次为空
    ├─ 调用 infinirtMalloc(16MB) 分配新块
    ├─ 标记 frozen = false (因为 pinned_mode_ = false)
    └─ 加入 all_blocks_ 索引
        ↓
释放: allocator.deallocate(ptr)
    ├─ 从 all_blocks_ 查找 Block
    ├─ 清除 in_use 标志
    └─ 回收到对应 size_class 的 free_blocks
        ↓
切换到 CUDA Graph 捕获模式
    ↓
allocator.set_pin_mode(true)
    └─ pinned_mode_ = true
        ↓
Graph 捕获: allocator.allocate(4MB)
    ├─ 命中 4MB size_class
    ├─ 分配新块
    └─ 标记 frozen = true (关键！)
        ↓
退出 Graph 模式
    ↓
allocator.set_pin_mode(false)
allocator.trim()
    ├─ 遍历 all_blocks_
    ├─ 跳过 frozen = true 的块（Graph 使用的内存）
    └─ 释放所有 free_blocks 中的非冻结块
```

### 3.3 硬件后端分发机制

所有子系统（infinirt、infiniop、infiniccl）都采用相似的硬件分发策略：

```
用户代码
    ↓
统一的 API 入口 (infinirt*, infiniop*, infiniccl*)
    ↓
switch (device_type) {
    case INFINI_DEVICE_NVIDIA:
        return nvidia::API(...);
    case INFINI_DEVICE_ASCEND:
        return ascend::API(...);
    case INFINI_DEVICE_CAMBRICON:
        return bang::API(...);
    case INFINI_DEVICE_MOORE:
        return musa::API(...);
    case INFINI_DEVICE_METAX:
        return metax::API(...);
    case INFINI_DEVICE_KUNLUN:
        return kunlun::API(...);
    case INFINI_DEVICE_CPU:
        return cpu::API(...);
    }
    ↓
硬件特定实现
    ├─ CUDA: NVIDIA 驱动 + cuBLAS/cuDNN/NCCL
    ├─ Ascend: 华为 CANN 框架 + HCCL
    ├─ Bang: 寒武纪 BANGC + CNCL
    ├─ Moore: 沐曦 MUSA + muBLAS/muDNN
    ├─ Metax: 天数智芯驱动 + 通信库
    └─ Kunlun: 昆仑芯驱动 + 通信库
```

### 3.4 模块间依赖关系

```
infinirt (最底层，无依赖)
    ↑
    └─ 被 infiniop、infiniccl、infinicore 依赖

infiniop (依赖 infinirt)
    ↑
    └─ 被 infinicore 依赖

infiniccl (依赖 infinirt)
    ↑
    └─ 被 infinicore 依赖（分布式训练场景）

infinicore (最高层，依赖 infinirt + infiniop + infiniccl)
    ├─ 上下文管理依赖 infinirt 的设备/流/内存 API
    ├─ 算子系统依赖 infiniop 的底层算子实现
    ├─ 通信模块依赖 infiniccl 的集合通信原语
    └─ 通过 pybind11 暴露给 Python

utils (独立模块，被所有模块依赖)
    └─ 提供错误检查、类型定义等通用功能
```

### 3.5 关键设计模式应用

1. **策略模式**:
   - 不同硬件后端是不同的计算/通信策略
   - 运行时根据设备类型选择策略

2. **工厂模式**:
   - `infiniopCreateXXXDescriptor` 作为工厂方法
   - 封装复杂的对象构造逻辑

3. **PImpl 模式**:
   - NVIDIA 实现中的 `Opaque` 结构体隐藏 CUDA 特定类型
   - 保持头文件的平台无关性

4. **RAII（资源获取即初始化）**:
   - 描述符析构时自动释放硬件资源
   - 使用智能指针管理资源生命周期

5. **Dispatcher 模式**:
   - `infinicore::op::OpDispatcher` 实现 kernel 注册和查找
   - 支持运行时覆盖已有实现

### 3.6 性能优化策略汇总

| 优化维度 | infinirt | infiniop | infinicore | infiniccl |
|---------|----------|----------|------------|-----------|
| **内存管理** | 异步分配、页锁定内存 | 工作空间复用 | Size-Class 内存池、流有序分配 | N/A |
| **计算优化** | N/A | 批量计算、Tensor Core、自动布局优化 | 算子缓存、设备内联执行 | N/A |
| **通信优化** | N/A | N/A | N/A | NCCL/HCCL 硬件加速 |
| **并发执行** | 多流并发、事件同步 | 流式计算、异步内核 | 多线程算子注册 | 异步 AllReduce |

### 3.7 测试架构

```
测试金字塔
    ├─ 单元测试 (infiniop-test, infinirt-test, infinicore-test, infiniccl-test)
    │   └─ 针对单个 API 或算子的白盒测试
    ├─ 集成测试 (test/infiniop/, test/infinicore/ops/)
    │   └─ 与 PyTorch 对比正确性和性能
    └─ 端到端测试 (InfiniLM, InfiniTrain)
        └─ 真实工作负载验证

统一测试框架 (test/infinicore/ops/)
    ├─ BaseOperatorTest: 基础测试类
    ├─ TestCase: 测试用例封装
    ├─ TensorSpec: 张量规格描述
    └─ parse_test_cases: 测试数据解析
```

## 4. 技术亮点与创新

### 4.1 多硬件统一抽象
- **一次编写，多硬件部署**: 通过 infinirt 和 infiniop 两层抽象，实现算子在 7+ 种硬件后端上的无缝切换
- **零运行时开销**: 使用模板和宏实现编译期硬件分发，避免虚函数开销

### 4.2 高性能内存管理
- **Size-Class 内存池**: 类似 jemalloc 的分配策略，减少内存碎片
- **固定模式支持**: 通过 `frozen` 标志支持 CUDA Graph 捕获，防止内存被误释放
- **流有序分配**: 与 CUDA Stream 紧密集成，实现计算与内存操作的流水线重叠

### 4.3 算子缓存机制
- **Thread-local OpCache**: 线程局部缓存避免全局锁竞争
- **RAII 自动清理**: 缓存析构时自动释放所有 Descriptor

### 4.4 广播机制
- **自动步长处理**: `ElementwiseInfo` 自动检测广播维度，内核通过 `broadcasted[]` 标志处理
- **零拷贝语义**: 广播通过逻辑索引实现，无需实际复制数据

## 5. 当前文档状态

### 已完成文档的模块
- `infiniop/elementwise/`: 详细文档覆盖接口、实现、硬件后端、广播机制
- `infiniop/ops/gemm/`: NVIDIA 后端详细文档，接口设计、性能优化、使用示例
- `infinicore/ops/`: 开发指南完整，覆盖算子定义、实现、注册、Python 绑定、测试
- `infinicore/context/allocators/`: 内存分配器模块详细文档，包含算法分析、性能特征、使用示例

### 待补充文档的模块
- `infinirt/`: 运行时层详细文档缺失
- `infiniccl/`: 通信层详细文档缺失
- `infiniop/ops/` 下的多数算子: 仅 NVIDIA/Moore 等部分后端有文档，其他硬件后端文档缺失
- `infinicore/tensor/`: 张量系统详细文档缺失
- `infinicore/graph/`, `infinicore/nn/`: 模块文档缺失

## 6. 依赖关系汇总

### 外部依赖
- **硬件厂商库**:
  - NVIDIA: CUDA Toolkit、cuBLAS、cuDNN、NCCL
  - Ascend: 华为 CANN 框架、HCCL
  - Cambricon: BANGC、CNCL
  - Moore: MUSA、muBLAS、muDNN
  - Metax、Kunlun: 各自的驱动和数学库

- **C++ 标准库**:
  - `<memory>`: 智能指针、std::byte
  - `<mutex>`: 线程同步
  - `<vector>`, `<unordered_map>`, `<queue>`: 容器
  - `<algorithm>`: 算法

- **构建系统**: XMake

### 内部依赖
```
utils (基础工具)
    ↓
infinirt (运行时层)
    ↓
infiniop (算子层)
    ↓
infinicore (核心库层)
    ↓
Python Frontend (Python 前端)
```

---

**文档生成时间**: 2026-01-14
**分析范围**: `InfiniCore/src/` 所有子目录
**文档版本**: v1.0
