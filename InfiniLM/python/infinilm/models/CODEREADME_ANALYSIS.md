# models 架构全景

## 1. 子系统职责

`models` 目录是 InfiniLM 框架中预训练大语言模型实现的顶层模块。它作为模型库的统一入口，负责组织和导出各类 Transformer 架构的模型实现。该模块的设计遵循模块化和可扩展原则，每个子目录代表一个独立的模型家族，通过标准化的接口（如 `AutoModel` 工厂类）提供统一的模型加载和使用体验。

当前实现聚焦于 LLaMA 架构（Meta AI 的大语言模型），完整实现了因果语言建模任务所需的所有组件。该子系统直接依赖 InfiniCore 框架进行张量运算和神经网络构建，通过配置驱动的方式支持模型的各种变体和并行策略。

在 InfiniLM 整体架构中，`models` 模块位于核心计算层之上，为上层应用（如训练、推理、部署）提供即用型的模型实例。它与数据处理、训练优化、推理加速等子系统协同工作，共同构成完整的 LLM 工具链。

## 2. 模块导航

* **llama**:
    * *功能*: 实现了基于 InfiniCore 框架的完整 LLaMA 模型，支持因果语言建模任务。包含多头注意力（MHA）、分组查询注意力（GQA）、旋转位置编码（RoPE）、前馈神经网络（MLP）以及 KV Cache 优化机制等所有核心组件。
    * *职责*: 提供 LLaMA 架构的模型定义、配置管理和加载接口，支持从 HuggingFace 格式加载预训练权重，并通过工厂类简化模型创建流程。

## 3. 架构逻辑图解

### 数据流与模块交互

```
models (顶层模块)
    │
    └── __init__.py (统一导出接口)
            │
            └── AutoLlamaModel (工厂类)
                    │
                    ├──> 配置加载阶段
                    │     ├── 读取 config.json
                    │     ├── 创建 LlamaConfig 实例
                    │     └── 验证模型超参数
                    │
                    └──> 模型实例化阶段
                          ├── LlamaForCausalLM (任务头模型)
                          │     ├── LlamaModel (基础 Transformer)
                          │     │     ├── embed_tokens (词嵌入)
                          │     │     ├── layers (解码器层列表)
                          │     │     │     └── LlamaDecoderLayer (×N)
                          │     │     │           ├── input_layernorm
                          │     │     │           ├── LlamaAttention (自注意力)
                          │     │     │           │     ├── q_proj, k_proj, v_proj
                          │     │     │           │     ├── RoPE 位置编码
                          │     │     │           │     ├── grouped_query_attention
                          │     │     │           │     └── o_proj
                          │     │     │           ├── post_attention_layernorm
                          │     │     │           └── LlamaMLP (前馈网络)
                          │     │     │                   ├── gate_proj (门控)
                          │     │     │                   ├── up_proj (上投影)
                          │     │     │                   ├── SiLU 激活
                          │     │     │                   └── down_proj (下投影)
                          │     │     └── norm (最终归一化)
                          │     └── lm_head (语言模型头)
                          │
                          └── 输出: logits [bs, 1, vocab_size]
```

### 执行流程

**模型加载阶段**:
1. 用户调用 `AutoLlamaModel.from_pretrained(model_path, device, dtype)`
2. 工厂类解析 `config.json`，创建 `LlamaConfig` 配置对象
3. 配置对象定义模型架构（层数、头数、隐藏维度等）和并行策略
4. 实例化 `LlamaForCausalLM`，递归创建所有子模块（Attention, MLP, DecoderLayer）
5. 从磁盘加载预训练权重，初始化模型参数

**前向传播阶段** (推理/训练):
1. **输入准备**: 接收 `input_ids` [bs, seq_len] 和 `position_ids` [bs, seq_len]
2. **词嵌入**: `embed_tokens` 将 token ID 映射为连续向量表示 [bs, seq_len, hidden_size]
3. **逐层处理**: 对每个 `LlamaDecoderLayer` (共 N 层):
   - Pre-LayerNorm: 对输入进行 RMS 归一化
   - 自注意力: 计算注意力和 KV Cache 更新
     * Q/K/V 投影: 线性变换到多头表示
     * RoPE 编码: 对 Q 和 K 应用旋转位置编码
     * 注意力计算: 分组查询注意力（支持 MHA/GQA/MQA）
     * 输出投影: 将多头表示融合回单一向量
   - 残差连接: `hidden_states += residual`
   - Post-LayerNorm: 对注意力输出进行 RMS 归一化
   - MLP: SwiGLU 激活函数的前馈网络变换
   - 残差连接: `hidden_states += residual`
4. **最后 token 提取**: 使用 `narrow` 操作提取序列最后一个位置的隐藏状态 [bs, 1, hidden_size]
5. **最终归一化**: 应用全局 RMSNorm
6. **Logits 投影**: `lm_head` 将隐藏状态映射到词表空间 [bs, 1, vocab_size]

**KV Cache 优化** (生成阶段):
- 在自回归生成时，`past_key_values` 缓存历史序列的 Key 和 Value
- 每次生成新 token 时，只需计算当前 token 的注意力，避免重复计算历史 token
- 将时间复杂度从 O(n²) 降低到 O(n)，显著提升推理速度

### 并行策略

**张量并行** (Tensor Parallelism):
- **列并行**: `q_proj`, `k_proj`, `v_proj` (注意力)，`gate_proj`, `up_proj` (MLP)
  - 将权重矩阵按列切分到多个设备，每个设备计算部分输出
- **行并行**: `o_proj` (注意力)，`down_proj` (MLP)
  - 将权重矩阵按行切分，每个设备计算部分结果后进行 All-Reduce 归约

**流水线并行** (Pipeline Parallelism):
- **阶段 1**: `embed_tokens` 处理输入 token ID
- **阶段 2**: `layers` 逐层计算隐藏状态
- **阶段 3**: `norm` 最终归一化和投影
- 不同阶段分配到不同设备，通过微流水线提升设备利用率

### 关键优化技术

1. **内存优化**:
   - 变量复用: `LlamaAttention.attn_output` 张量在推理过程中重用
   - 零拷贝操作: `repeat_kv` 使用 `as_strided` 避免 KV 头的实际内存复制
   - KV Cache: 减少 Key 和 Value 的重复计算

2. **计算优化**:
   - 因果 Softmax: 使用 InfiniCore 的优化实现 `causal_softmax`
   - 最后 token 提取: 使用 `narrow` 操作避免计算整个序列的最终表示
   - 批处理: 重用输出张量内存，减少分配开销

3. **架构优化**:
   - Pre-LayerNorm: 相比 Post-LN 训练更稳定
   - RMSNorm: 替代 LayerNorm，计算更简单，无偏置项
   - SwiGLU: 门控激活函数，提升模型表达能力
   - RoPE: 旋转位置编码，支持扩展长度的位置信息

### 扩展性设计

该模块采用工厂模式和配置分离的设计，易于扩展新模型:

1. **添加新模型**: 在 `models/` 下创建新子目录（如 `gpt/`, `bloom/`），实现对应架构
2. **配置驱动**: 通过 `config.json` 和配置类管理模型超参数
3. **统一接口**: 所有模型通过 `AutoModel` 工厂类加载，保持 API 一致性
4. **并行支持**: 通过 `*_tp_plan` 和 `*_pp_plan` 配置支持分布式训练

当前实现以 LLaMA 为起点，为后续添加更多 Transformer 架构（如 GPT、BLOOM、Falcon 等）提供了清晰的模板和基础设施。
