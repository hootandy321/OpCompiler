# 目录: ops 算子接口层架构全景

## 1. 子系统职责

`ops` 目录是 **InfiniOP 算子接口层**，定义了整个推理引擎的核心计算原语集合。该子系统承担着以下关键职责：

- **统一API抽象**：为所有后端实现（CUDA、CPU、Ascend等）提供统一的C接口规范
- **算子语义定义**：声明每个算子的输入/输出张量形状、数据类型、算法参数
- **生命周期管理**：规范算子描述符的创建、执行、销毁的标准流程
- **资源管理**：通过workspace机制抽象临时内存需求，支持不同后端的内存优化策略
- **模型推理覆盖**：提供LLM推理全流程所需的基础算子（线性层、归一化、激活函数、注意力机制、量化等）

该层是InfiniCore架构中的**契约层**：上层框架通过这些接口调用计算能力，下层后端（如CUDA kernels）负责实现具体逻辑。

---

## 2. 模块导航

该目录包含33个算子头文件，无子目录结构。按功能域分类如下：

### 2.1 基础线性代数算子（BLAS级别）

- **`add.h`**: 张量逐元素加法，支持广播语义
- **`sub.h`**: 张量逐元素减法
- **`mul.h`**: 张量逐元素乘法
- **`gemm.h`**: 通用矩阵乘法（GEMM），支持alpha和beta缩放参数
- **`int8_gemm.h`**: INT8量化矩阵乘法，支持输入和权重的per-channel量化尺度
- **`conv.h`**: 卷积算子，支持padding、stride、dilation等卷积参数

### 2.2 归一化与正则化算子

- **`layer_norm.h`**: Layer Normalization，输出标准化结果、均值、标准差（用于反向传播）
- **`rms_norm.h`**: RMS Normalization（Root Mean Square Layer Norm），LLM常用归一化
- **`add_rms_norm.h`**: **融合算子**：先执行 `a + b`，再对结果执行RMS Norm，可输出残差连接结果
- **`lp_norm.h`**: Lp范数计算

### 2.3 激活函数

- **`relu.h`**: ReLU激活函数
- **`gelu.h`**: GELU激活函数（Gaussian Error Linear Unit）
- **`silu.h`**: SiLU激活函数（Sigmoid Linear Unit，即Swish）
- **`sigmoid.h`**: Sigmoid激活函数
- **`tanh.h`**: Tanh激活函数
- **`softplus.h`**: Softplus激活函数
- **`swiglu.h`**: **SwiGLU激活函数**（LLM核心算子，通常与GEMM融合）

### 2.4 注意力机制算子（核心）

- **`attention.h`**: 标准注意力机制（支持KV Cache输入）
- **`paged_attention.h`**: **Paged Attention v1**（解码阶段），支持：
  - 分块KV Cache（block table映射）
  - ALiBi位置编码（可选）
  - 动态序列长度（seq_lens参数）
- **`paged_attention_prefill.h`**: **Paged Attention Prefill**（预填充阶段），处理初始prompt的长序列注意力
- **`paged_caching.h`**: **KV Cache写入算子**，根据slot_mapping将K/V写入分块缓存池

### 2.5 概率与采样算子

- **`softmax.h`**: Softmax归一化，支持沿指定轴计算
- **`logsoftmax.h`**: Log Softmax
- **`causal_softmax.h`**: 带因果掩码的Softmax（用于自回归模型的注意力分数处理）
- **`topksoftmax.h`**: **融合算子**：TopK选择 + Softmax归一化（用于MoE路由、束搜索等）
- **`topkrouter.h`**: TopK路由算子（MoE专家路由），返回topk的values和indices，支持correction_bias
- **`random_sample.h`**: 随机采样算子（用于生成token采样）

### 2.6 位置编码与变换

- **`rope.h`**: **旋转位置编码（RoPE）**，支持两种算法：
  - `GPT_J`：交错奇偶维度
  - `GPT_NEOX`：前半维度sin，后半维度cos
- **`clip.h`**: 张量裁剪（限制数值范围）
- **`rearrange.h`**: 张量重排（reshape、transpose等操作）

### 2.7 量化相关算子

- **`dequantize_awq.h`**: **AWQ量化方案**的反量化算子，支持：
  - `qweight`：量化权重（4-bit）
  - `scales`：量化尺度
  - `zeros`：零点偏移
- **`int8_gemm.h`**: INT8量化矩阵乘法（已在2.1中列出）

### 2.8 内存初始化算子

- **`zeros.h`**: 张量初始化为零
- **`ones.h`**: 张量初始化为一

---

## 3. 架构逻辑图解

### 3.1 算子层次体系

```
推理流程视角（自上而下）：

1. 模型层组合（如 Transformer Block）
   ├─ 依赖: PagedAttention / PagedAttentionPrefill / PagedCaching（注意力）
   ├─ 依赖: GEMM / Int8Gemm / DequantizeAWQ（线性层）
   ├─ 依赖: LayerNorm / RMSNorm / AddRMSNorm（归一化）
   └─ 依赖: RoPE（位置编码）

2. 算子融合层（性能优化关键）
   ├─ AddRMSNorm: 融合残差连接 + 归一化
   ├─ PagedCaching: 融合KV写入逻辑
   ├─ TopKRouter: 融合TopK选择
   └─ TopKSoftmax: 融合选择 + 归一化

3. 基础算子层
   ├─ 线性代数: Add/Sub/Mul, GEMM, Conv
   ├─ 归一化: LayerNorm, RMSNorm
   ├─ 激活函数: GELU, SiLU, ReLU, etc.
   ├─ 概率: Softmax, LogSoftmax
   └─ 变换: RoPE, Rearrange, Clip
```

### 3.2 数据流关键路径（LLM推理）

**预填充阶段（Prefill）**：
```
Input Embeddings
  ↓
GEMM (Q/K/V投影)
  ↓
RoPE (位置编码)
  ↓
PagedCaching (写入KV Cache)
  ↓
PagedAttentionPrefill (计算初始注意力)
  ↓
AddRMSNorm (残差连接 + 归一化)
```

**解码阶段（Decode）**：
```
Last Token
  ↓
GEMM (Q投影)
  ↓
RoPE (增量位置编码)
  ↓
PagedAttention (从KV Cache读取 + 注意力计算)
  ↓
AddRMSNorm
  ↓
GEMM (输出投影)
  ↓
Softmax / RandomSample (生成下一个token)
```

**MoE推理路径**：
```
Hidden States
  ↓
GEMM (门控网络)
  ↓
TopKRouter (选择TopK专家)
  ↓
[并行] 各专家的 GEMM + SwiGLU
  ↓
加权组合专家输出
```

### 3.3 算子依赖关系

**高内聚模块**：
- `attention.h` 内部依赖 `gemm.h` 和 `swiglu.h`（标准注意力的典型实现）
- `paged_attention.h` 与 `paged_caching.h` 紧密配合（KV Cache管理）
- `paged_attention_prefill.h` 和 `paged_attention.h` 构成完整注意力生命周期

**性能优化策略**：
1. **融合算子优先**：如 `AddRMSNorm` 合并两次kernel launch为一次
2. **内存连续性**：`PagedCaching` 使用slot_mapping确保KV写入物理连续块
3. **量化感知**：`Int8Gemm` 和 `DequantizeAWQ` 支持低精度计算加速

### 3.4 后端实现映射

每个算子头文件定义了统一的接口规范，具体实现由不同后端提供：
- `src/ops/cuda/`：CUDA实现（cuBLAS、cuDNN、自定义kernel）
- `src/ops/cpu/`：CPU实现（OpenBLAS、oneDNN）
- `src/ops/ascend/`：华为昇腾实现
- `src/ops/kunlun/`：昆仑芯实现
- ...

接口统一性确保了上层框架无需感知底层硬件差异，实现了**计算图的可移植性**。

---

## 4. 设计模式与最佳实践

### 4.1 统一的算子描述符模式

所有算子遵循一致的生命周期：
```c
// 1. 创建描述符（编译期优化机会）
infiniopCreateXxxDescriptor(handle, &desc, tensor_descs, params);

// 2. 查询workspace大小（内存规划）
infiniopGetXxxWorkspaceSize(desc, &size);

// 3. 执行计算（运行期）
infiniopXxx(desc, workspace, size, outputs, inputs, stream);

// 4. 销毁描述符
infiniopDestroyXxxDescriptor(desc);
```

这种设计允许：
- **延迟编译**：在create阶段根据张量形状生成最优kernel
- **内存复用**：统一workspace管理，减少分配开销
- **异步执行**：stream参数支持GPU异步计算

### 4.2 张量描述符的灵活性

所有算子使用 `infiniopTensorDescriptor_t` 抽象张量，支持：
- 动态形状（运行期推断）
- 任意数据类型（FP32/FP16/BF16/INT8/INT4）
- 任意内存布局（NHWC/NCHW、行主序/列主序）

### 4.3 可选参数设计

如 `paged_attention.h` 中的 `alibi_slopes_desc` 可为NULL，支持：
- 标准注意力（不使用ALiBi）
- ALiBi注意力（传入slopes张量）

这种设计避免为每种变体创建独立算子。

---

## 5. 关键性能指标

该层算子的性能直接影响推理吞吐量：

- **PagedAttention系列**：影响KV Cache管理效率和显存占用
- **融合算子（AddRMSNorm, PagedCaching）**：减少kernel launch开销
- **量化算子（Int8Gemm, DequantizeAWQ）**：提升计算吞吐，降低显存需求
- **TopKRouter**：决定MoE模型的负载均衡度

---

## 6. 扩展指南

如需添加新算子：

1. 创建 `xxx.h` 头文件，遵循现有命名和模式
2. 定义描述符类型：`typedef struct InfiniopDescriptor *infiniopXxxDescriptor_t;`
3. 实现四个标准函数：
   - `infiniopCreateXxxDescriptor`
   - `infiniopGetXxxWorkspaceSize`
   - `infiniopXxx`（核心计算）
   - `infiniopDestroyXxxDescriptor`
4. 在各后端实现对应kernel（CUDA、CPU等）
5. 添加单元测试和性能基准

---

## 7. 相关文档

- **上层接口**：`../infini.h`（全局上下文和句柄管理）
- **下层实现**：`/src/ops`（各后端的算子实现）
- **张量抽象**：`../operator_descriptor.h`（张量描述符定义）
