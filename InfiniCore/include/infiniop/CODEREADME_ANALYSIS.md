# 目录: infiniop 算子接口层架构全景

## 1. 子系统职责

`infiniop` 目录是 **InfiniCore 推理引擎的核心算子接口层**，在整个系统架构中扮演着 **硬件抽象契约层** 的关键角色。该子系统的核心职责包括：

- **统一算子API规范**：为所有计算算子定义跨硬件后端的统一C接口契约，确保上层框架（如模型推理引擎）无需关心底层硬件差异
- **张量抽象与描述**：通过 `tensor_descriptor.h` 提供灵活的张量语义描述（形状、步长、数据类型），支持任意维度和内存布局
- **算子生命周期管理**：规范算子描述符的创建、workspace查询、执行、销毁的标准流程，为编译期优化和运行期调度提供基础
- **设备上下文管理**：通过 `handle.h` 提供全局设备上下文，支持多设备并发执行
- **LLM推理算子全覆盖**：提供大模型推理全流程所需的核心算子，包括注意力机制、KV Cache管理、量化计算、激活函数、归一化等

该层位于 **InfiniCore架构的中枢位置**：
- **向上**：为上层框架（InfiniLM、InfiniTrain）提供稳定的计算接口
- **向下**：约束各硬件后端（CUDA、CPU、Ascend、Kunlun等）的实现规范
- **横向**：通过统一的接口语义，实现计算图在不同硬件间的可移植性

---

## 2. 模块导航

### 2.1 核心基础设施模块

* **`handle.h`**:
    * *功能*: 定义全局设备上下文句柄 `infiniopHandle_t`，提供 `infiniopCreateHandle` 和 `infiniopDestroyHandle` 接口
    * *职责*: 管理设备上下文生命周期，作为所有算子操作的全局环境载体

* **`tensor_descriptor.h`**:
    * *功能*: 张量描述符API，定义 `infiniopTensorDescriptor_t` 类型，支持动态维度、任意步长、多数据类型
    * *职责*: 抽象张量的形状和内存布局，使算子接口与物理内存解耦
    * *接口*: `infiniopCreateTensorDescriptor` / `infiniopDestroyTensorDescriptor`

* **`operator_descriptor.h`**:
    * *功能*: 算子描述符基类定义，提供设备类型和设备ID查询接口
    * *职责*: 统一所有算子描述符的元数据访问，支持运行期设备信息查询

### 2.2 算子接口层（ops 子目录）

* **`ops/`**:
    * *功能*: **LLM推理算子接口集合**，包含33个算子头文件，覆盖线性代数、注意力机制、归一化、激活函数、量化、采样等完整功能域
    * *职责*: 定义每个算子的输入输出张量语义、算法参数、workspace需求，规范算子描述符的四阶段生命周期（创建、查询、执行、销毁）
    * *核心算子分类*:
        - **注意力与KV Cache**: `attention.h`, `paged_attention.h`, `paged_attention_prefill.h`, `paged_caching.h`
        - **线性代数**: `gemm.h`, `int8_gemm.h`, `add.h`, `sub.h`, `mul.h`, `conv.h`
        - **归一化**: `layer_norm.h`, `rms_norm.h`, `add_rms_norm.h`, `lp_norm.h`
        - **激活函数**: `relu.h`, `gelu.h`, `silu.h`, `swiglu.h`, `sigmoid.h`, `tanh.h`, `softplus.h`
        - **位置编码**: `rope.h`
        - **概率与采样**: `softmax.h`, `causal_softmax.h`, `topksoftmax.h`, `topkrouter.h`, `random_sample.h`
        - **量化**: `dequantize_awq.h`, `int8_gemm.h`
        - **内存操作**: `zeros.h`, `ones.h`, `clip.h`, `rearrange.h`
    * *设计模式*: 所有算子遵循统一的四阶段API模式：`CreateXxxDescriptor` → `GetXxxWorkspaceSize` → `Xxx`（执行） → `DestroyXxxDescriptor`

---

## 3. 架构逻辑图解

### 3.1 层次体系结构

```
┌─────────────────────────────────────────────────────────────┐
│          上层框架 (InfiniLM / InfiniTrain)                   │
│                  调用算子接口                                 │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              infiniop 算子接口层 (当前目录)                    │
├─────────────────────────────────────────────────────────────┤
│  基础设施层              │  算子接口层 (ops/)                  │
│  - handle.h             │  - 33个算子头文件                    │
│  - tensor_descriptor.h  │  - 统一的描述符模式                  │
│  - operator_descriptor.h│  - workspace抽象                    │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│          后端实现层 (src/ops/)                               │
│   CUDA    │   CPU    │  Ascend  │  Kunlun  │   ...          │
│   cuBLAS  │ OpenBLAS │  CANN    │   ...    │                │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 算子生命周期标准流程

所有33个算子严格遵循以下统一的生命周期模式：

```
1. 描述符创建阶段（编译期优化机会）
   └─> infiniopCreateXxxDescriptor(handle, &desc, tensor_descs, params)
       - 校验张量形状和数据类型兼容性
       - 根据设备类型选择最优算法（如GEMM的tile大小）
       - 预计算编译期常量（如scale、alignment）

2. Workspace查询阶段（内存规划）
   └─> infiniopGetXxxWorkspaceSize(desc, &size)
       - 返回临时内存需求（用于reduction、中间结果等）
       - 允许调用方预分配全局workspace池，减少分配开销

3. 算子执行阶段（运行期计算）
   └─> infiniopXxx(desc, workspace, workspace_size, outputs, inputs, stream)
       - 异步执行（通过stream参数）
       - 支持batch并发（不同stream并行执行）

4. 描述符销毁阶段（资源释放）
   └─> infiniopDestroyXxxDescriptor(desc)
       - 释放编译期生成的资源（如kernel cache）
```

**设计优势**：
- **编译期/运行期分离**：重量级优化（kernel选择、常量预计算）在create阶段完成一次，执行阶段零开销
- **内存复用**：统一workspace抽象允许不同算子共享内存池
- **异步流式执行**：stream参数支持GPU并行计算，隐藏PCIe传输延迟

### 3.3 核心数据流：LLM推理关键路径

#### 预填充阶段（Prefill Phase）
```
Input Embeddings (Token IDs → Embedding Vectors)
    │
    ▼
┌──────────────────────────────────────────────────┐
│ GEMM (Q/K/V Projection)                          │
│   输入: (seq_len, hidden_dim)                    │
│   输出: Q=(seq_len, num_heads, head_size)        │
│         K=(seq_len, num_kv_heads, head_size)     │
│         V=(seq_len, num_kv_heads, head_size)     │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│ RoPE (Rotary Position Encoding)                  │
│   位置编码注入到Q和K的head维度                    │
│   算法: GPT_J / GPT_NEOX                         │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│ PagedCaching (写入KV Cache)                      │
│   输入: K, V tensors                             │
│   输入: slot_mapping (逻辑位置 → 物理block)       │
│   输出: k_cache, v_cache (分块存储池)             │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│ PagedAttentionPrefill (计算初始注意力)            │
│   输入: Q (完整序列)                              │
│   输入: k_cache, v_cache (刚写入的数据)            │
│   输出: Contextualized Hidden States              │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│ AddRMSNorm (残差连接 + 归一化)                    │
│   融合算子: 减少一次kernel launch                 │
└──────────────────────────────────────────────────┘
```

#### 解码阶段（Decode Phase）
```
Last Token Hidden State
    │
    ▼
┌──────────────────────────────────────────────────┐
│ GEMM (Q Projection for single token)             │
│   输入: (1, hidden_dim)                          │
│   输出: (1, num_heads, head_size)                │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│ RoPE (增量位置编码)                               │
│   只需计算当前位置的sin/cos值                     │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│ PagedAttention (从KV Cache读取 + 注意力)          │
│   输入: Q (单个token)                            │
│   输入: k_cache, v_cache (历史所有tokens)         │
│   输入: block_tables (映射逻辑序列到物理blocks)    │
│   输入: seq_lens (每个序列的当前长度)             │
│   输出: (1, num_heads, head_size)                │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│ AddRMSNorm → GEMM (Output Projection)            │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│ Softmax / TopKSoftmax / RandomSample             │
│   生成下一个token的概率分布或采样结果              │
└──────────────────────────────────────────────────┘
```

### 3.4 关键模块交互模式

#### 模式1: KV Cache管理（PagedAttention系列）
```
PagedCaching (写入)          PagedAttention (读取)
        │                            │
        │    k_cache, v_cache        │
        ├──────────────────────────>│
        │                            │
   slot_mapping               block_tables
   (逻辑→物理映射)              (物理块索引表)
```

**协作机制**：
- `PagedCaching` 根据 `slot_mapping` 将K/V写入分散的物理blocks
- `PagedAttention` 根据 `block_tables` 重组逻辑序列，计算注意力
- 这种设计允许动态batching和variable sequence lengths

#### 模式2: 融合算子优化
```
AddRMSNorm: 融合两个操作
    input_a + input_b → RMSNorm(output)
    相比分离调用，减少:
    - 一次kernel launch
    - 一次全局内存读写

TopKSoftmax: 融合TopK选择 + 归一化
    用于MoE路由或束搜索，一次pass完成两个操作
```

#### 模式3: 量化感知计算
```
DequantizeAWQ + GEMM:
    4-bit权重 → 反量化 → FP16/BF16计算
    或者直接在kernel中融合反量化（性能最优）

Int8Gemm:
    INT8输入 → INT8累加 → FP32输出 → 缩放
    利用Tensor Core加速
```

### 3.5 跨后端可移植性实现

```
上层调用代码（伪代码）:
----------------------------------------
infiniopCreateGemmDescriptor(handle, &desc, c_desc, a_desc, b_desc);
infiniopGetGemmWorkspaceSize(desc, &size);
infiniopGemm(desc, workspace, size, c, a, b, 1.0f, 0.0f, stream);
----------------------------------------

                │
        ┌───────┴───────┐
        ▼               ▼
    CUDA后端        CPU后端
    cuBLAS          OpenBLAS
    (或自定义kernel)  (或oneDNN)
```

**关键设计点**：
- 张量描述符的 `strides` 参数支持任意内存布局（NHWC/NCHW、行主序/列主序）
- `dtype` 参数统一抽象数据类型（FP32/FP16/BF16/INT8/INT4）
- `stream` 参数抽象异步执行上下文（CUDA stream / CPU thread pool）

---

## 4. 设计模式与最佳实践

### 4.1 类型不透明原则（Type Opaqueness）

所有描述符类型使用 `typedef struct InfiniopDescriptor *infiniopXxxDescriptor_t;`，将实现细节完全隐藏。

**优势**：
- 后端可自由定义内部结构（如缓存编译好的kernel）
- 前向兼容性：可在不破坏ABI的情况下扩展结构
- 多态支持：同一类型指针可指向不同后端的实现

### 4.2 Workspace抽象模式

所有算子提供 `GetXxxWorkspaceSize` 接口，而不是在内部分配内存。

**优势**：
- **内存复用**：上层可预分配全局workspace池，多个算子共享
- **避免碎片化**：减少小对象分配，提升内存局部性
- **性能可控**：调用方可选择workspace位置（HBM/HBM+Cache/CPU）

### 4.3 可选参数设计

如 `paged_attention.h` 中的 `alibi_slopes_desc` 参数可为NULL。

**实践**：
- 避免为每种变体创建独立算子（如 `PagedAttentionALiBi`）
- 通过参数可选性支持多种算法（标准注意力 / ALiBi / 其他位置编码）
- 文档中明确标注可选参数及其语义

### 4.4 统一的错误处理

所有接口返回 `infiniStatus_t` 类型，继承自 `infinicore.h` 的状态码系统。

**最佳实践**：
- 编译期错误（如不支持的dtype）在 `CreateDescriptor` 阶段返回
- 运行期错误（如CUDA OOM）在执行阶段返回
- 错误码通过全局 `infiniGetLastErrorString()` 获取详细描述

---

## 5. 性能关键路径与优化机会

### 5.1 计算密集型算子

- **GEMM / Int8Gemm**: 占据LLM推理70%以上计算量
  - 优化：利用Tensor Core、winograd算法、张量并行
- **PagedAttention系列**: 内存带宽密集型
  - 优化：KV Cache数据局部性、block_size调优
- **DequantizeAWQ**: 内存访存密集型
  - 优化：融合反量化到GEMM kernel

### 5.2 Kernel Launch优化

- **融合算子（AddRMSNorm, TopKSoftmax）**: 减少launch开销
- **batch策略**: 将多个小序列打包为单个batch执行
- **stream并发**: 不同独立算子使用不同stream并行执行

### 5.3 内存管理优化

- **Workspace复用**: 全局workspace池避免频繁分配
- **KV Cache预分配**: 根据 `max_num_blocks_per_seq` 预分配固定池
- **数据类型选择**: BF16 vs FP16的权衡（数值稳定性 vs 计算速度）

---

## 6. 扩展指南

### 6.1 添加新算子的标准流程

1. **创建头文件** `ops/xxx.h`，遵循现有命名和格式
2. **定义描述符类型**：`typedef struct InfiniopDescriptor *infiniopXxxDescriptor_t;`
3. **实现四个标准函数**：
   ```c
   infiniopCreateXxxDescriptor(handle, &desc, tensor_descs, params);
   infiniopGetXxxWorkspaceSize(desc, &size);
   infiniopXxx(desc, workspace, size, outputs, inputs, stream);
   infiniopDestroyXxxDescriptor(desc);
   ```
4. **在各后端实现**：
   - CUDA: `src/ops/cuda/xxx.cu` / `xxx.h`
   - CPU: `src/ops/cpu/xxx.cc` / `xxx.h`
   - 其他硬件: 类似目录结构
5. **添加测试**：
   - 单元测试：验证正确性
   - 性能基准：对比竞品（如cuBLAS、cutlass）

### 6.2 支持新硬件后端

1. 在 `src/ops/` 下创建新目录（如 `src/ops/myhardware/`）
2. 实现所有33个算子（或按需逐步实现）
3. 在 `handle.h` 的实现中注册新设备类型
4. 提供设备特定的stream抽象和内存分配器

---

## 7. 相关文档索引

- **上层调用**: `/home/qy/src/Infini/InfiniCore/include/infinicore.h`（全局类型定义）
- **后端实现**: `/home/qy/src/Infini/InfiniCore/src/ops/`（CUDA/CPU/Ascend等具体实现）
- **算子详细说明**: `/home/qy/src/Infini/InfiniCore/include/infiniop/ops/CODEREADME_ANALYSIS.md`（ops子目录的完整架构分析）

---

## 8. 总结

`infiniop` 目录通过 **统一的接口契约**、**灵活的张量抽象**、 **标准化的生命周期管理**，成功构建了一个 **硬件无关的算子API层**。该层的设计使得上层框架可以以一致的方式调用计算能力，而底层后端则可以根据硬件特性进行最优实现，是InfiniCore实现 **跨硬件高性能推理** 的关键基石。
