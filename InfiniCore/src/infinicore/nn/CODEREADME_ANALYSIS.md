# CODEREADME_ANALYSIS.md: nn 模块架构全景

## 1. 子系统职责

`nn` 目录是 InfiniCore 框架的**神经网络层抽象子系统**，提供高层次的神经网络构建模块。该子系统不直接实现底层算子（由 `ops` 目录负责），而是基于算子和张量操作构建可组合、可训练的神经网络组件。

**核心设计目标**：
- **模块化架构**：提供类似 PyTorch 的模块接口，支持层级嵌套和参数管理
- **分布式支持**：内置张量并行（Tensor Parallel, TP）能力，支持大规模模型训练
- **状态管理**：统一的参数序列化/反序列化机制，支持模型权重加载和保存
- **设备抽象**：透明处理 CPU/GPU 设备切换和内存拷贝

该子系统在整个架构中位于**中间层**：
- **向下依赖**：`ops`（算子库）、`context`（设备与流管理）、Tensor（张量抽象）
- **向上支撑**：用户可直接使用这些模块构建 Transformer、MLP 等复杂网络

---

## 2. 模块导航 (Module Navigation)

由于该目录为**叶子节点**（仅包含源代码，无子目录），以下列出各源文件的职责：

* **module.cc / module.hpp**：
    * *功能*：模块基类的核心实现，提供参数注册、层级管理和状态字典接口
    * *职责*：实现 `state_dict()`（递归收集所有参数）和 `load_state_dict()`（层级加载参数），支持模块树状结构的遍历

* **parameter.cc / parameter.hpp**：
    * *功能*：参数类的张量并行扩展，支持参数分片存储
    * *职责*：在多卡 TP 场景下，仅保存本 rank 负责的参数分片，`load()` 时自动从完整权重中提取对应分片

* **linear.cc / linear.hpp**：
    * *功能*：线性层实现，包含标准版和两种并行变体
    * *职责*：
      - `Linear`：基础线性变换（支持 bias）
      - `ColumnParallelLinear`：列并行线性层（权重按列切分，输出维度分片）
      - `RowParallelLinear`：行并行线性层（权重按行切分，输出需 AllReduce）

* **embedding.cc / embedding.hpp**：
    * *功能*：嵌入层实现，将离散索引映射为连续向量
    * *职责*：支持 `padding_idx`（指定填充位置），通过行拷贝实现高效的查表操作

* **rmsnorm.cc / rmsnorm.hpp**：
    * *功能*：RMS 归一化层（Root Mean Square Normalization）
    * *职责*：对输入张量沿最后一维进行 RMS 归一化，常用于 Transformer 的层后归一化

* **rope.cc / rope.hpp**：
    * *功能*：旋转位置编码（Rotary Positional Embedding, RoPE）模块
    * *职责*：预计算 sin/cos 缓存表，支持 GPT-J 和 GPT-NeoX 两种维度配对算法

---

## 3. 架构逻辑图解

### 3.1 模块层级与参数流转

```
用户构建网络（如 Transformer）
    ↓
Module::register_parameter() 注册权重
    ↓
Module::state_dict() 递归收集参数
    └─> 前序遍历模块树，生成 "layer.sublayer.weight" 风格的键值对
    ↓
保存/加载模型权重（通过 load_state_dict）
    └─> 根据 prefix 递归匹配参数名
```

**关键机制**：
- `Module` 基类维护两个哈希表：`parameters_`（直接参数）和 `submodules_`（子模块）
- 状态字典使用点分隔的层级命名（如 `transformer.h.0.weight`）
- 递归收集/加载时通过 `prefix` 参数传递当前路径

### 3.2 张量并行的线性层协作

```
输入张量 [batch, seq_len, in_features]
    ↓
ColumnParallelLinear::forward()
    ├─> 权重按列分片：[out_features/tp_size, in_features]
    └─> 输出分片：[batch, seq_len, out_features/tp_size]
    ↓
中间计算（注意力层等）
    ↓
RowParallelLinear::forward()
    ├─> 权重按行分片：[out_features, in_features/tp_size]
    ├─> 本地矩阵乘法：[batch, seq_len, out_features]（分片）
    └─> AllReduce 求和：infinicclAllReduce(SUM, communicator)
    ↓
完整输出 [batch, seq_len, out_features]
```

**TP 维度约定**：
- `tp_dim=0`（ColumnParallel）：参数按第 0 维切分（列并行）
- `tp_dim=1`（RowParallel）：参数按第 1 维切分（行并行）
- bias 仅在 `tp_rank=0` 时保存（RowParallel）

### 3.3 RoPE 缓存预计算流程

```
初始化 RoPE 模块
    ↓
initialize_cache()
    ├─> CPU 计算 sin/cos 值（GPT-J 频率公式）
    │   └─> inv_freq = θ^(-2j/head_dim)
    │   └─> angle = pos * inv_freq
    └─> 类型转换（F32 → BF16/F16）
    └─> 拷贝到目标设备（GPU）
    ↓
forward()
    ├─> 读取预计算缓存表
    └─> 调用 op::rope 应用旋转变换
```

**算法差异**：
- `Algo::GPT_J`：相邻两维作为一对 `(cos, sin)`
- `Algo::GPT_NEOX`：间隔两维配对

### 3.4 设备无关的 Embedding 查表

```
输入索引张量 [batch, seq_len]
    ↓
拷贝到 CPU（indices_cpu）
    ↓
遍历每个索引值
    ├─> 类型转换（I32/U32 → I64）
    ├─> 边界检查（0 ≤ idx < num_embeddings）
    └─> 行拷贝：
        ├─> CPU 路径：memcpy
        └─> GPU 路径：context::memcpyD2D（异步流拷贝）
    ↓
输出 [batch, seq_len, embedding_dim]
```

**性能优化**：
- 索引先转到 CPU 避免频繁的 D2H/H2D 同步
- 设备端使用流有序的 D2D 拷贝（无需同步 CPU）

### 3.5 模块间依赖关系

```
Module（基类）
    ├─> 继承 → Parameter（参数类）
    ├─> 组合 → Linear, Embedding, RMSNorm, RoPE
    │   ├─> 调用 → ops::linear, ops::rms_norm, ops::rope
    │   └─> 依赖 → Tensor（张量操作）
    └─> 依赖 → context（设备管理、流同步）
```

**设计原则**：
- 模块只负责参数管理和组合逻辑，计算委托给 `ops` 层
- 设备切换通过 `Tensor::to()` 和 `context::memcpyD2D` 透明处理
- TP 通信通过 `infinicclAllReduce` 原语集成到前向传播中

---

## 4. 技术亮点

1. **零拷贝参数加载**：
   - `Parameter::load_blob()` 直接从二进制块构造张量
   - `Parameter::load()` 使用 `narrow()` 视图提取分片，避免数据复制

2. **流式异步拷贝**：
   - RoPE 缓存和 Embedding 的设备间拷贝通过 CUDA 流异步执行
   - 避免阻塞 CPU，提升吞吐量

3. **类型安全的多态索引读取**：
   - `Embedding::read_index` lambda 函数支持 I32/U32/I64/U64 四种索引类型
   - 编译时分发，运行时零开销

4. **递归状态管理**：
   - `state_dict()` 和 `load_state_dict()` 的递归实现支持任意深度的模块嵌套
   - 前缀机制确保参数名的唯一性和可追溯性

---

## 5. 文档状态说明

该目录为**代码包（Leaf Node）**，所有实现文件已分析完毕。如需查看具体的函数级代码逻辑，请参考各 `.cc` 源文件的注释。
