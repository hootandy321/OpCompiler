# LLaMA 模型核心实现文档

本模块实现了基于 InfiniCore 框架的 LLaMA (Large Language Model Meta AI) 模型，支持因果语言建模 (Causal Language Modeling)。该实现完整包含了 LLaMA 架构的所有核心组件：多头注意力机制 (MHA)、分组查询注意力 (GQA)、旋转位置编码 (RoPE)、前馈神经网络 (MLP) 以及 KV Cache 优化机制。

## 1. 模块结构

- **`__init__.py`**: 提供 `AutoLlamaModel` 工厂类，负责模型加载的统一接口
- **`configuration_llama.py`**: 定义 `LlamaConfig` 配置类，继承自 `PretrainedConfig` 和 C++ 绑定的 `_infinilm.LlamaConfig`
- **`modeling_llama.py`**: 核心模型实现，包含注意力计算、Transformer 层、完整模型和生成接口

## 2. 核心类

### `LlamaConfig`
- **位置**: `configuration_llama.py`
- **主要功能**: 存储 LLaMA 模型的所有超参数配置，管理模型架构的实例化参数
- **关键成员**:
  - `vocab_size` (int): 词表大小，默认 32000
  - `hidden_size` (int): 隐藏层维度，默认 4096
  - `intermediate_size` (int): MLP 中间层维度，默认 11008
  - `num_hidden_layers` (int): Transformer 解码器层数，默认 32
  - `num_attention_heads` (int): 注意力头数，默认 32
  - `num_key_value_heads` (int): KV 头数，用于 GQA，默认等于 `num_attention_heads`
  - `hidden_act` (str): 激活函数类型，默认 "silu"
  - `max_position_embeddings` (int): 最大序列长度，默认 2048
  - `rms_norm_eps` (float): RMSNorm 的 epsilon 值，默认 1e-6
  - `rope_theta` (float): RoPE 基础周期，默认 10000.0
  - `rope_scaling` (Dict): RoPE 扩展配置，支持多种扩展类型 (default, linear, dynamic, yarn, longrope, llama3)
  - `attention_bias` (bool): 是否在注意力投影中使用偏置，默认 True
  - `mlp_bias` (bool): 是否在 MLP 层中使用偏置，默认 False
  - `head_dim` (int): 注意力头维度，计算公式为 `hidden_size // num_attention_heads`
  - `dtype` (infinicore.dtype): 模型数据类型 (float32, bfloat16, float16)
- **核心方法**:
  - `__init__(vocab_size, hidden_size, ...)`: 初始化配置，参数包括词表大小、隐藏维度、层数、注意力头数等；验证 RoPE 参数；设置数据类型；调用父类初始化
- **生命周期**: 工厂模式，通过 `from_pretrained` 或直接实例化创建；不可变配置对象
- **特殊设计**:
  - 双重继承：同时继承 Python 的 `PretrainedConfig` 和 C++ 绑定的 `_infinilm.LlamaConfig`
  - 张量并行计划：定义了 `base_model_tp_plan` 字典，指定各层的列/行并行策略
  - 流水线并行计划：定义了 `base_model_pp_plan` 字典，指定阶段划分和输入/输出张量

### `LlamaRMSNorm`
- **位置**: `modeling_llama.py` (第 92 行)
- **主要功能**: RMS 归一化层，直接使用 InfiniCore 框架的实现
- **实现**: `LlamaRMSNorm = infinicore.nn.RMSNorm`
- **特点**: 替代传统的 LayerNorm，计算更高效，无偏置项，使用 epsilon 防止除零

### `LlamaMLP`
- **位置**: `modeling_llama.py` (第 95-116 行)
- **主要功能**: 实现 LLaMA 的前馈神经网络层，使用 SwiGLU 激活函数
- **关键成员**:
  - `gate_proj` (infinicore.nn.Linear): 门控投影层，`hidden_size -> intermediate_size`
  - `up_proj` (infinicore.nn.Linear): 上投影层，`hidden_size -> intermediate_size`
  - `down_proj` (infinicore.nn.Linear): 下投影层，`intermediate_size -> hidden_size`
  - `act_fn` (callable): SwiGLU 激活函数，使用 `infinicore.nn.functional.silu`
- **核心方法**:
  - `forward(x: infinicore.Tensor) -> infinicore.Tensor`: 前向传播，计算公式：`down_proj(silu(gate_proj(x)) * up_proj(x))`
- **激活函数**: SwiGLU (Swish-Gated Linear Unit)，结合了门控机制和 Swish 激活，提升模型表达能力
- **参数量**: 对于 LLaMA-7B (hidden_size=4096, intermediate_size=11008)，单层 MLP 参数量约 3.6B

### `LlamaAttention`
- **位置**: `modeling_llama.py` (第 119-260 行)
- **主要功能**: 实现多头注意力 (MHA) 或分组查询注意力 (GQA)，支持 RoPE 位置编码和 KV Cache
- **关键成员**:
  - `q_proj` (infinicore.nn.Linear): Query 投影，`hidden_size -> num_attention_heads * head_dim`
  - `k_proj` (infinicore.nn.Linear): Key 投影，`hidden_size -> num_key_value_heads * head_dim`
  - `v_proj` (infinicore.nn.Linear): Value 投影，`hidden_size -> num_key_value_heads * head_dim`
  - `o_proj` (infinicore.nn.Linear): 输出投影，`num_attention_heads * head_dim -> hidden_size`
  - `num_key_value_groups` (int): 每组 KV 头对应的 Query 头数，计算公式 `num_attention_heads // num_key_value_heads`
  - `scaling` (float): 注意力缩放因子，公式为 `head_dim**-0.5`
  - `attn_output` (infinicore.Tensor): 可重用的输出张量，避免重复分配内存
- **核心方法**:
  - `forward(hidden_states, past_key_values, rope_instance, **kwargs)`: 执行注意力计算
    1. 投影 Q/K/V：`hidden_states [bs, seq_len, hidden_size] -> Q/K/V`
    2. 应用 RoPE 位置编码到 Q 和 K
    3. 更新 KV Cache：如果 `past_key_values` 存在，缓存当前层的 K/V
    4. 计算注意力：对 batch 中每个样本调用 `grouped_query_attention`
    5. 输出投影：`o_proj` 映射回 `hidden_size`
- **优化技术**:
  - **变量复用**: `self.attn_output` 张量在不同序列长度间重用，减少内存分配
  - **批处理**: 逐样本循环处理，但重用输出张量内存
  - **因果掩码**: 使用 `infinicore.nn.functional.causal_softmax` 实现自回归掩码

### `LlamaDecoderLayer`
- **位置**: `modeling_llama.py` (第 263-316 行)
- **主要功能**: 实现 Transformer 解码器层，包含 Pre-LN 结构的自注意力和 MLP
- **关键成员**:
  - `self_attn` (LlamaAttention): 自注意力模块
  - `mlp` (LlamaMLP): 前馈神经网络模块
  - `input_layernorm` (LlamaRMSNorm): 注意力前的归一化
  - `post_attention_layernorm` (LlamaRMSNorm): MLP 前的归一化
- **核心方法**:
  - `forward(hidden_states, past_key_values, use_cache, rope_instance, **kwargs)`: 执行解码器层前向传播
    1. 保存残差连接：`residual = hidden_states`
    2. 输入归一化：`hidden_states = input_layernorm(hidden_states)`
    3. 自注意力计算：`hidden_states = self_attn(...)`
    4. 残差连接：`hidden_states += residual`
    5. 保存残差：`residual = hidden_states`
    6. 后注意力归一化：`hidden_states = post_attention_layernorm(hidden_states)`
    7. MLP 计算：`hidden_states = mlp(hidden_states)`
    8. 残差连接：`hidden_states += residual`
- **架构**: Pre-LayerNorm 结构 (先归一化再计算)，相比 Post-LN 训练更稳定
- **残差连接**: 两个残差连接分别用于注意力和 MLP 层，保证梯度流动

### `LlamaModel`
- **位置**: `modeling_llama.py` (第 319-394 行)
- **主要功能**: 完整的 LLaMA Transformer 模型，包含嵌入、多层解码器和最终归一化
- **关键成员**:
  - `embed_tokens` (infinicore.nn.Embedding): 词嵌入层，`vocab_size -> hidden_size`
  - `layers` (infinicore.nn.ModuleList): 解码器层列表，长度为 `num_hidden_layers`
  - `norm` (LlamaRMSNorm): 最终归一化层
  - `rope_instance` (infinicore.nn.RoPE): 旋转位置编码实例
- **核心方法**:
  - `forward(input_ids, position_ids, past_key_values, use_cache, **kwargs)`: 完整前向传播
    1. 初始化 KV Cache：如果 `use_cache=True` 且 `past_key_values` 为空，创建 `DynamicCache`
    2. 词嵌入：`inputs_embeds = embed_tokens(input_ids)`
    3. 逐层处理：遍历所有 `LlamaDecoderLayer`，传入 `hidden_states`、`past_key_values`、`position_ids`、`rope_instance`
    4. 提取最后一个 token：使用 `narrow` 操作获取序列最后一个位置的隐藏状态
    5. 最终归一化：`return self.norm(last_token)`
- **输出**: 形状为 `[bs, 1, hidden_size]` 的张量，仅包含序列最后一个 token 的表示
- **优化**:
  - 仅计算最后一个 token 的隐藏状态，避免不必要的计算
  - KV Cache 在多层间共享，加速推理

### `LlamaForCausalLM`
- **位置**: `modeling_llama.py` (第 397-446 行)
- **主要功能**: 因果语言建模任务的头模型，包含语言模型头和生成能力
- **关键成员**:
  - `model` (LlamaModel): 基础 Transformer 模型
  - `lm_head` (infinicore.nn.Linear): 语言模型头，`hidden_size -> vocab_size`，无偏置
  - `use_cache` (bool): 是否使用 KV Cache，默认 True
  - `device` (infinicore.device): 模型所在设备
- **核心方法**:
  - `forward(input_ids, position_ids, past_key_values, use_cache, **kwargs)`: 计算 logits
    1. 调用基础模型：`last_token = model(input_ids, position_ids, ...)`
    2. 投影到词表：`return lm_head(last_token)`，输出形状 `[bs, 1, vocab_size]`
  - `from_pretrained(model_path, device)`: 类方法，从 HuggingFace 格式加载模型
    1. 读取 `config.json`：解析模型配置
    2. 创建 `LlamaConfig` 实例
    3. 实例化 `LlamaForCausalLM` 对象
- **继承**: 同时继承 `infinicore.nn.Module` 和 `GenerationMixin`，支持文本生成

### `AutoLlamaModel`
- **位置**: `__init__.py` (第 10-36 行)
- **主要功能**: 统一的模型加载工厂类，提供计时和日志输出
- **核心方法**:
  - `from_pretrained(model_path, device, dtype, **kwargs)`: 加载预训练模型
    1. 记录开始时间：`t1 = time.time()`
    2. 打印加载信息：设备、数据类型
    3. 调用 `modeling_llama.LlamaForCausalLM.from_pretrained`
    4. 记录结束时间并打印耗时：`(t2 - t1) * 1000` 毫秒
    5. 返回模型实例
- **用途**: 简化模型加载流程，提供标准化的日志输出

## 3. 核心辅助函数

### `repeat_kv`
- **位置**: `modeling_llama.py` (第 28-44 行)
- **签名**: `repeat_kv(keys: infinicore.Tensor, values: infinicore.Tensor, ngroup: int) -> Tuple[Tensor, Tensor]`
- **功能**: 在分组查询注意力 (GQA) 中，将 KV 头重复 `ngroup` 次以匹配 Q 头数量
- **输入**:
  - `keys`: `[total_seq_len, num_heads, head_dim]`，KV 头的键
  - `values`: `[total_seq_len, num_heads, head_dim]`，KV 头的值
  - `ngroup`: 每组重复次数，等于 `num_attention_heads // num_key_value_heads`
- **输出**:
  - `keys_new`: `[total_seq_len, num_heads * ngroup, head_dim]`
  - `values_new`: `[total_seq_len, num_heads * ngroup, head_dim]`
- **实现细节**:
  - 使用 `as_strided` 创建视图：`(total_seq_len, num_heads, ngroup, head_dim)`，步长为 `(s0, s1, 0, s2)`，其中 `s2=0` 实现零拷贝重复
  - 调用 `contiguous()` 保证内存连续
  - 使用 `view` 重塑张量形状
- **优化**: 零拷贝重复，通过步长操作避免实际的内存复制

### `multi_head_attention`
- **位置**: `modeling_llama.py` (第 47-72 行)
- **签名**: `multi_head_attention(querys, keys, values, scaling) -> Tensor`
- **功能**: 实现标准的多头注意力计算（MHA）
- **输入**:
  - `querys`: `[seq_len, num_heads, head_dim]`
  - `keys`: `[total_seq_len, num_heads, head_dim]`
  - `values`: `[total_seq_len, num_heads, head_dim]`
  - `scaling`: 缩放因子，通常为 `head_dim**-0.5`
- **算法步骤**:
  1. 重排 Q: `[seq_len, num_heads, head_dim] -> [num_heads, seq_len, head_dim]`
  2. 保持 K: `[total_seq_len, num_heads, head_dim]`
  3. 重排 V: `[total_seq_len, num_heads, head_dim] -> [num_heads, total_seq_len, head_dim]`
  4. 计算注意力权重：`attn_weight = matmul(Q, K.T) * scaling`，输出 `[num_heads, seq_len, total_seq_len]`
  5. 因果 softmax：`causal_softmax(attn_weight)`，实现自回归掩码
  6. 加权求和：`out = attn_weight @ V`，输出 `[num_heads, seq_len, head_dim]`
  7. 重排输出：`[num_heads, seq_len, head_dim] -> [seq_len, num_heads, head_dim]`
- **输出**: `[seq_len, num_heads, head_dim]`，经过注意力机制的特征表示

### `grouped_query_attention`
- **位置**: `modeling_llama.py` (第 75-89 行)
- **签名**: `grouped_query_attention(querys, keys, values, scaling) -> Tensor`
- **功能**: 实现分组查询注意力 (GQA)，兼容 MHA 和 MQA
- **输入**:
  - `querys`: `[seq_len, num_attention_heads, head_dim]`
  - `keys`: `[total_seq_len, num_key_value_heads, head_dim]`
  - `values`: `[total_seq_len, num_key_value_heads, head_dim]`
  - `scaling`: 缩放因子
- **算法步骤**:
  1. 计算分组数：`ngroup = num_attention_heads // num_key_value_heads`
  2. 如果 `ngroup > 1`，调用 `repeat_kv` 扩展 KV 头
  3. 调用 `multi_head_attention` 计算注意力
- **兼容性**:
  - MHA: `num_key_value_heads = num_attention_heads`，`ngroup = 1`，不重复 KV
  - MQA: `num_key_value_heads = 1`，`ngroup = num_attention_heads`，单个 KV 头被所有 Q 头共享
  - GQA: `1 < num_key_value_heads < num_attention_heads`，折中方案

## 4. API 接口

```python
# 加载预训练模型
model = AutoLlamaModel.from_pretrained(
    model_path="/path/to/model",
    device=infinicore.device("cuda:0"),
    dtype=infinicore.bfloat16
)

# 前向传播
logits = model.forward(
    input_ids=infinicore.tensor([[1, 1128, 526, 366, 29892]]),  # [bs, seq_len]
    position_ids=infinicore.tensor([[0, 1, 2, 3, 4]]),           # [bs, seq_len]
    past_key_values=cache,                                        # 可选的 KV Cache
    use_cache=True                                               # 是否使用缓存
)  # 输出: [bs, 1, vocab_size]

# 创建配置
config = LlamaConfig(
    vocab_size=32000,
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,  # GQA 配置
    max_position_embeddings=2048,
    rope_theta=10000.0,
    torch_dtype="bfloat16"
)
```

## 5. 使用示例

```python
import infinicore
from infinilm.models.llama import AutoLlamaModel

# 1. 加载模型
model = AutoLlamaModel.from_pretrained(
    model_path="/path/to/llama-2-7b",
    device=infinicore.device("cuda:0"),
    dtype=infinicore.bfloat16
)

# 2. 准备输入
input_ids = infinicore.tensor([[1, 1128, 526, 366, 29892]])  # "<s> Hello world"
position_ids = infinicore.tensor([[0, 1, 2, 3, 4]])

# 3. 首次前向传播 (prefill 阶段)
logits = model(input_ids=input_ids, position_ids=position_ids, use_cache=True)
# logits: [1, 1, 32000] - 最后一个 token 的预测分布

# 4. 生成阶段 (使用 KV Cache)
past_key_values = model.model.layers[0].self_attn.past_key_values  # 假设缓存已保存
new_token = infinicore.tensor([[29892]])  # 上一步生成的 token
new_position = infinicore.tensor([[5]])

# 5. 自回归生成
for _ in range(100):  # 生成 100 个 token
    logits = model(
        input_ids=new_token,
        position_ids=new_position,
        past_key_values=past_key_values,
        use_cache=True
    )
    next_token = infinicore.argmax(logits, dim=-1)  # 贪婪解码
    new_token = next_token.view(1, 1)
    new_position += 1
```

## 6. 实现细节

### 内存管理
- **变量复用**: `LlamaAttention.attn_output` 张量在推理过程中被重用，避免每次前向传播都分配新内存
- **KV Cache**: 使用 `DynamicCache` 类缓存历史计算的 Key 和 Value，减少自回归生成时的重复计算
- **零拷贝操作**: `repeat_kv` 函数使用 `as_strided` 和步长操作实现 KV 头的重复，避免实际的内存复制

### 并行策略
- **张量并行**: `LlamaConfig.base_model_tp_plan` 定义了列并行和行并行策略
  - 列并行: `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`
  - 行并行: `o_proj`, `down_proj`
- **流水线并行**: `LlamaConfig.base_model_pp_plan` 定义了阶段划分
  - 阶段 1: `embed_tokens`，输入 `input_ids`，输出 `inputs_embeds`
  - 阶段 2: `layers`，输入 `hidden_states`，输出 `hidden_states`
  - 阶段 3: `norm`，输入 `hidden_states`，输出 `hidden_states`

### 性能优化
- **注意力计算**: 批量处理时对每个样本单独调用 `grouped_query_attention`，但重用输出张量
- **RoPE 实现**: 使用 InfiniCore 框架的原生 RoPE 实现，支持高效的旋转位置编码
- **因果掩码**: 使用 `infinicore.nn.functional.causal_softmax` 实现优化的因果掩码 softmax
- **最后 token 提取**: 使用 `narrow` 操作高效提取序列最后一个位置的隐藏状态

### 位置编码
- **RoPE (Rotary Position Embeddings)**: 旋转位置编码，通过旋转矩阵注入位置信息
  - 配置参数: `rope_theta` (基础周期，默认 10000.0)，`max_position_embeddings` (最大长度)
  - 支持扩展: `rope_scaling` 配置支持多种扩展策略 (linear, dynamic, yarn, longrope, llama3)
  - 应用方式: 在注意力计算前对 Q 和 K 应用旋转变换

### 归一化策略
- **RMSNorm**: 替代 LayerNorm，计算公式为 `x / sqrt(mean(x^2) + eps)`
  - 优点: 计算更简单，无偏置项，训练更稳定
  - 应用位置: `input_layernorm` (注意力前)，`post_attention_layernorm` (MLP 前)，`norm` (最终输出)
  - epsilon 参数: `rms_norm_eps`，默认 1e-6

### 激活函数
- **SwiGLU**: 门控线性单元，结合了 Swish 激活和门控机制
  - 计算公式: `SwiGLU(x) = Swish(gate_proj(x)) * up_proj(x)`
  - Swish 函数: `SiLU(x) = x * sigmoid(x)`
  - 优势: 提升模型表达能力，在大规模模型上表现优于 ReLU/GLU

### 注意力机制变体
- **MHA (Multi-Head Attention)**: `num_key_value_heads = num_attention_heads`，每个 Query 头有独立的 KV 头
- **MQA (Multi-Query Attention)**: `num_key_value_heads = 1`，所有 Query 头共享单个 KV 头
- **GQA (Grouped-Query Attention)**: `1 < num_key_value_heads < num_attention_heads`，折中方案，在性能和效率间取得平衡

### 错误处理
- **配置验证**: RoPE 参数验证，确保 `rope_scaling` 配置正确
- **数据类型检查**: 只支持 float32, bfloat16, float16，其他类型抛出 `ValueError`
- **向后兼容**: 如果 `num_key_value_heads` 未指定，默认等于 `num_attention_heads`

### 设计模式
- **工厂模式**: `AutoLlamaModel.from_pretrained` 提供统一的模型创建接口
- **模块化设计**: 每个组件（Attention, MLP, DecoderLayer）都是独立的 `nn.Module`
- **配置分离**: 配置与模型实现分离，支持灵活的模型配置
- **继承复用**: `LlamaForCausalLM` 继承 `GenerationMixin`，复用生成逻辑
- **组合模式**: `LlamaModel` 组合多个 `LlamaDecoderLayer`，形成完整的 Transformer

### 依赖关系
- **InfiniCore 框架**: 核心张量运算库，提供 `infinicore.Tensor`, `infinicore.nn`, `infinicore.device`
- **C++ 绑定**: `_infinilm.LlamaConfig` 提供 C++ 实现的配置类
- **缓存工具**: `DynamicCache` 类实现 KV Cache，来自 `...cache_utils`
- **生成工具**: `GenerationMixin` 提供文本生成能力，来自 `...generation.utils`

### 扩展性
- **自定义 RoPE**: 通过 `rope_scaling` 配置支持多种位置编码扩展策略
- **并行训练**: 内置张量并行和流水线并行支持
- **多硬件支持**: 通过 InfiniCore 框架支持 CUDA、CPU 等多种设备
- **可配置架构**: 所有超参数可通过配置对象灵活设置

### 性能特征
- **时间复杂度**:
  - 注意力计算: O(seq_len^2 * hidden_size)，使用 KV Cache 后降为 O(seq_len * hidden_size)
  - MLP 计算: O(hidden_size * intermediate_size)
  - 单层复杂度: O(seq_len * hidden_size * (hidden_size + intermediate_size))
- **空间复杂度**:
  - 模型参数: O(hidden_size^2 * num_layers + vocab_size * hidden_size)
  - KV Cache: O(num_layers * num_key_value_heads * max_seq_len * head_dim)
- **推理加速**: KV Cache 将自回归生成的复杂度从 O(n^2) 降至 O(n)，其中 n 为序列长度
