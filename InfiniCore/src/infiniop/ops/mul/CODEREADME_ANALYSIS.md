# 目录: mul 乘法操作架构全景

## 1. 子系统职责

`mul` 目录是 Infini 框架中**逐元素张量乘法操作**（Element-wise Multiplication）的多硬件后端实现层。该模块负责在张量级别执行逐元素的乘法运算（C = A ⊙ B），支持广播（broadcasting）和复杂的内存布局（stride）处理。

该模块在整体架构中的位置：
- 上层：接收 `operator.cc` 提供的统一调用接口
- 下层：依赖 `elementwise` 通用基础设施和各硬件加速器的原生指令集
- 横向：与其他算术操作（add、sub、div）共享相同的架构模式

## 2. 模块导航

* **📂 cpu**
  * 功能：*CPU 后端实现（文档待生成）*
  * 职责：提供 x86/ARM 架构下的 CPU 串行向量化乘法实现

* **📂 cuda**
  * 功能：*CUDA 通用后端（文档待生成）*
  * 职责：定义 CUDA 设备端乘法运算符 `MulOp` 的通用实现

* **📂 kunlun**
  * 功能：*昆仑芯片后端（文档待生成）*
  * 职责：适配昆仑 XPU 加速器的乘法 kernel 实现

* **📂 metax**
  * 功能：*Metax 后端（文档待生成）*
  * 职责：支持 Metax MACA 架构的乘法操作实现

* **📂 moore**
  * 功能：*Moore 线程编译器后端（文档待生成）*
  * 职责：使用 Moore 线程编译器生成的专用乘法实现

* **📂 nvidia**
  * 功能：*NVIDIA CUDA 专用后端（已文档化）*
  * 职责：提供高度优化的 NVIDIA GPU 乘法实现，支持 F16/F32/F64/BF16 数据类型，利用 `__hmul`、`__hmul2`、`__fmul_rn` 等专用指令实现向量化计算
  * 核心特性：
    - 模板元编程实现编译期类型分发
    - 完整的广播和 stride 处理能力
    - 自动分块处理超大规模张量
    - 异步执行和流并发模型

## 3. 架构逻辑图解

### 3.1 硬件抽象层次

```
operator.cc (统一接口层)
    ↓
┌─────────────────────────────────────────┐
│     mul 算子多态分发层                  │
├─────────────────────────────────────────┤
│ cpu │ cuda │ kunlun │ metax │ moore │ nvidia │
└─────────────────────────────────────────┘
    ↓         ↓         ↓         ↓         ↓
[x86/ARM] [CUDA]   [XPU]    [MACA]   [MTGPU]  [NVIDIA GPU]
```

### 3.2 数据流与执行流程

#### 3.2.1 NVIDIA 后端完整执行链路

基于已文档化的 `nvidia` 实现，典型执行流程如下：

**阶段 1：描述符创建（初始化）**
```
用户调用 Descriptor::create()
    ↓
1. 数据类型验证 (CHECK_DTYPE)
   └─ 支持：F16, F32, F64, BF16
2. 形状一致性检查 (CHECK_SAME_SHAPE)
   └─ 确保 output.shape == input0.shape == input1.shape
3. 元数据生成 (ElementwiseInfo::create)
   ├─ 提取张量形状、stride、连续性标记
   ├─ 计算广播维度信息
   └─ 紧凑存储到单一 _meta 向量
4. CUDA 设备实现初始化 (DeviceImpl::create)
5. 工作空间大小计算
   └─ workspace_size = 元数据大小 + 输入指针数组大小
```

**阶段 2：计算执行（运行时）**
```
用户调用 Descriptor::calculate()
    ↓
1. 工作空间验证
2. 数据类型分发 (switch-case)
   ├─ F16  → calculate<256, cuda::MulOp, half>
   ├─ F32  → calculate<256, cuda::MulOp, float>
   ├─ F64  → calculate<256, cuda::MulOp, double>
   └─ BF16 → calculate<256, cuda::MulOp, cuda_bfloat16>
    ↓
3. 元数据异步拷贝到 GPU (cudaMemcpyAsync)
4. Grid/Block 维度计算
   ├─ blockDims.x = min(256, maxThreadsPerBlock)
   ├─ gridDims.x = min(ceil_div(output_size, blockDims.x), gridSizeX)
   └─ 支持分块启动处理超大张量
    ↓
5. CUDA Kernel 启动 (elementwiseKernel)
   ├─ 全局索引计算：idx = blockIdx.x * blockDim.x + threadIdx.x + offset
   ├─ 边界检查：if (idx < output_size)
   ├─ 输出索引映射：
   │   └─ 连续张量 → 线性索引
   │   └─ 非连续张量 → indexToOffset(维度映射)
   ├─ 输入索引器构造 (InputIndexer)
   │   └─ 支持广播和 stride 逻辑
   └─ 编译期展开运算：
       └─ output[idx] = MulOp{}(input0[idx], input1[idx])
           └─ 硬件指令优化：
               ├─ half2 → __hmul2 (向量化2元素)
               ├─ half → __hmul (半精度乘法)
               ├─ float → __fmul_rn (IEEE-754舍入)
               └─ double → 原生 * 操作符
```

### 3.3 依赖关系矩阵

#### 3.3.1 横向依赖（跨硬件共享）

| 组件 | 依赖项 | 用途 |
|------|--------|------|
| **所有硬件后端** | `elementwise.h` | 提供宏定义 `ELEMENTWISE_DESCRIPTOR` 和 `ElementwiseInfo` 结构体 |
| **所有硬件后端** | `operator.cc` | 统一的算子注册和分发接口 |
| **nvidia, cuda** | `cuda/kernel.cuh` | 定义 `cuda::MulOp` 设备端运算符 |
| **nvidia** | `elementwise/nvidia/elementwise_nvidia_api.cuh` | CUDA 通用基础设施和 `DeviceImpl` |

#### 3.3.2 NVIDIA 后端内部依赖树

```
mul_nvidia.cuh/cu (乘法操作实现)
    ├─ ELEMENTWISE_DESCRIPTOR 宏 (elementwise.h)
    │   └─ 生成 Descriptor 类框架
    ├─ CREATE_ELEMENTWISE_CUDA_DESCRIPTOR 宏
    │   └─ 调用 ElementwiseInfo::create
    │   └─ 调用 DeviceImpl::create
    ├─ cuda::MulOp (cuda/kernel.cuh)
    │   └─ __hmul, __hmul2, __fmul_rn 指令
    └─ elementwise/nvidia/elementwise_nvidia_api.cuh
        ├─ DeviceImpl::calculate
        │   └─ elementwiseKernel (elementwise_nvidia.cuh)
        │       ├─ indexToOffset (设备端索引映射)
        │       ├─ InputIndexer (广播处理)
        │       └─ unpackInputsAndApply (编译期展开)
        └─ launchElementwiseKernel (kernel 启动调度)
            └─ infoToDevice (元数据 GPU 拷贝)
```

### 3.4 广播机制示例

假设执行形状为 `[3, 1]` ⊙ `[1, 4]` → `[3, 4]` 的乘法：

```
输入 A (shape=[3,1]):
[[1],
 [2],
 [3]]

输入 B (shape=[1,4]):
[[10, 20, 30, 40]]

广播逻辑：
┌─────────────────────────────────────┐
│ InputIndexer 对输入 A 的索引逻辑     │
│ - 维度0: 步长1 (不广播)              │
│ - 维度1: 步长0 (广播，始终索引0)     │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ InputIndexer 对输入 B 的索引逻辑     │
│ - 维度0: 步长0 (广播，始终索引0)     │
│ - 维度1: 步长1 (不广播)              │
└─────────────────────────────────────┘

输出结果 (shape=[3,4]):
[[1*10, 1*20, 1*30, 1*40],     → [10, 20, 30, 40]
 [2*10, 2*20, 2*30, 2*40],     → [20, 40, 60, 80]
 [3*10, 3*20, 3*30, 3*40]]     → [30, 60, 90, 120]
```

CUDA Kernel 执行时：
- 线程 idx=0 (输出位置[0,0]): 读取 A[0] 和 B[0]，计算 1*10
- 线程 idx=1 (输出位置[0,1]): 读取 A[0] 和 B[1]，计算 1*20
- 线程 idx=4 (输出位置[1,0]): 读取 A[1] 和 B[0]，计算 2*10
- 线程 idx=5 (输出位置[1,1]): 读取 A[1] 和 B[1]，计算 2*20

`InputIndexer` 通过 `_meta` 中的广播标记自动处理这种索引映射。

### 3.5 性能关键路径

```
主机端 (Host)                    设备端 (Device)
─────────────────────────────────────────────────
calculate() 调用
    ↓
workspaceSize() 检查
    ↓
类型分发 (编译期优化)
    ↓
元数据拷贝 (cudaMemcpyAsync)
    └─ 隐藏在 kernel 执行背后
    ↓
Kernel 启动开销 (~10μs)
    ↓
┌───────────────────────────────────┐
│ GPU 全局内存带宽 (瓶颈所在)        │
│                                   │
│ 读取 Input A: 2 bytes/element (F16)│
│ 读取 Input B: 2 bytes/element     │
│ 写入 Output:  2 bytes/element     │
│ ──────────────────────────────── │
│ 总计: 6 bytes/element 内存流量     │
│                                   │
│ 理论吞吐量 (A100):                │
│   1.5 TB/s / 6 bytes ≈ 250B elem/s│
│                                   │
│ 实际计算量:                       │
│   2 FLOPs/element (1次乘法)       │
│   A100 算力: 312 TFLOPS (F16)     │
│   计算时间: 2/312 = 0.006 ns/elem │
│                                   │
│ 结论: 内存绑定型操作，             │
│       受限于带宽而非算力           │
└───────────────────────────────────┘
```

## 4. 设计模式与工程实践

### 4.1 模式应用

1. **策略模式 (Strategy Pattern)**
   - `cuda::MulOp` 封装乘法运算逻辑
   - 可替换为 `AddOp`、`SubOp` 等其他策略

2. **工厂模式 (Factory Pattern)**
   - `Descriptor::create()` 静态工厂方法
   - 封装复杂初始化和验证逻辑

3. **Pimpl 模式 (Pointer to Implementation)**
   - `DeviceImpl` 隐藏 CUDA 实现细节
   - 减少头文件编译依赖

4. **CRTP 变体 (宏生成代码)**
   - `ELEMENTWISE_DESCRIPTOR` 宏生成专用 `Descriptor` 类
   - 避免虚函数开销，保持静态多态

### 4.2 零开销抽象技术

- **编译期类型分发**：`if constexpr` + 模板特化，零运行时分支
- **编译期循环展开**：`std::make_index_sequence<N>` 展开输入索引访问
- **强制内联**：`__forceinline__` 消除函数调用开销
- **restrict 指针**：告知编译器无别名，启用激进优化

## 5. 硬件后端实现现状

| 后端 | 文档状态 | 源文件 | 实现特征 |
|------|----------|--------|----------|
| **nvidia** | ✅ 已完成 | `.cuh`, `.cu` | 高度优化，支持向量化指令 |
| **cpu** | ❌ 待生成 | `.cc`, `.h` | x86/ARM 向量化实现 |
| **cuda** | ❌ 待生成 | `.cuh` | 通用 CUDA 运算符定义 |
| **kunlun** | ❌ 待生成 | `.xpu`, `.h` | 昆仑 XPU 专用实现 |
| **metax** | ❌ 待生成 | `.maca`, `.h` | Metax MACA 架构实现 |
| **moore** | ❌ 待生成 | `.mu`, `.h`, `_kernel.h` | Moore 线程编译器实现 |

**注**：根据智能去重策略，`cpu`、`kunlun`、`metax`、`moore` 等硬件后端可作为同级实现参考 `nvidia` 的架构模式，暂不生成重复文档。

## 6. 扩展与维护指南

### 6.1 添加新硬件后端

参考 `nvidia` 实现模式：

1. 创建 `new_hw/` 目录
2. 实现三个核心文件：
   - `mul_new_hw.h`：声明 `Descriptor` 类
   - `mul_new_hw.cc`：实现 `create()` 和 `calculate()`
   - `kernel.h`：定义设备端 `MulOp`
3. 调用 `ELEMENTWISE_DESCRIPTOR(mul, new_hw)` 宏
4. 在 `operator.cc` 中注册新后端

### 6.2 支持新数据类型

在 `Descriptor::calculate()` 中添加分支：

```cpp
case INFINI_DTYPE_F8_E4M3:
    return _device_info->calculate<256, cuda::MulOp, float8_e4m3_t>(
        _info, workspace, workspace_size, output, inputs, stream);
```

并确保 `cuda::MulOp` 中有对应的 `if constexpr` 优化路径。

### 6.3 性能调优检查清单

- [ ] 验证数据类型是否使用专用硬件指令
- [ ] 确认张量连续性以启用快速路径
- [ ] 检查广播模式是否优化内存访问
- [ ] 使用 Nsight Compute 分析内存带宽利用率
- [ ] 确认 grid/block 配置适配硬件限制

---

**文档版本**：v1.0
**最后更新**：2026-01-14
**覆盖范围**：`mul` 目录完整架构视图（基于 nvidia 后端实现）
**未覆盖后端**：cpu, cuda, kunlun, metax, moore（待后续生成文档）
