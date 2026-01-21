# Models 子系统架构全景

## 1. 子系统职责

Models 子系统是 InfiniLM 的核心模型层，负责定义统一的模型接口（`InfinilmModel` 基类）和实现具体的大语言模型架构。该层位于 InfiniLM 架构的中心位置，向上承接推理引擎的调度指令，向下调用 InfiniCore 基础算子库和分布式训练基础设施。其主要职责包括：

- **统一抽象**：通过 `InfinilmModel` 基类定义所有模型的通用接口（`Input`/`Output` 结构体、`forward()` 方法、Cache 管理），实现模型无关的推理流程。
- **模型实例化**：通过 `InfinilmModelFactory` 工厂类根据配置动态创建具体模型实例，支持插件式扩展新架构。
- **具体架构实现**：实现完整的 Transformer 模型（如 Llama），包括注意力机制、前馈网络、层归一化、KV Cache 等核心组件。
- **调试支持**：提供 Hook 机制和张量诊断工具，支持在推理过程中捕获中间结果，便于开发和性能分析。

该子系统采用模块化设计，将模型实现（`llama`）与调试工具（`debug_utils`）分离，同时通过工厂模式和多态性实现架构的可扩展性。

## 2. 模块导航 (Module Navigation)

### 核心框架文件

* **infinilm_model.hpp**:
  - *功能*: 定义 `InfinilmModel` 抽象基类，作为所有具体模型实现的统一接口
  - *职责*: 声明 `Config`、`Input`、`Output` 结构体，规定 `forward()` 和 `reset_cache()` 纯虚函数接口，实现模型的多态调用

* **model_factory.hpp/cpp**:
  - *功能*: 实现 `InfinilmModelFactory` 工厂类，根据配置动态创建模型实例
  - *职责*: 解耦模型创建逻辑，支持运行时根据 `Config::model_type` 字段实例化不同的模型架构（如 Llama、GPT 等），提供统一的模型入口

### 子模块目录

* **llama/**:
  - *功能*: 实现完整的 Llama 大语言模型架构，严格遵循 HuggingFace Transformers 的 Llama 设计
  - *职责*: 提供 `LlamaForCausalLM` 类（继承自 `InfinilmModel`），包含 Grouped Query Attention (GQA)、Rotary Position Embeddings (RoPE)、SwiGLU MLP、KV Cache 等核心特性，支持张量并行训练和推理

* **debug_utils/**:
  - *功能*: 提供模型调试和诊断工具集，包括 Hook 回调机制和张量统计日志
  - *职责*: 实现 `HookRegistry` 类管理模型执行过程中的回调钩子，支持精确匹配和通配符模式捕获中间张量；提供 `log_tensor_stats()` 等工具函数记录张量统计信息（min/max/mean/样本值），用于精度分析和性能调优

## 3. 架构逻辑图解

### 数据流与交互关系

Models 子系统的数据流遵循**自上而下的调用链**和**自下而上的依赖关系**：

#### 3.1 模型创建流程（工厂模式）

```
用户/推理引擎
    ↓ (传入 Config)
InfinilmModelFactory::createModel()
    ↓ (根据 config.model_type 分发)
具体模型构造函数 (如 LlamaForCausalLM::LlamaForCausalLM)
    ↓ (初始化子组件)
├─ 创建 LlamaModel (词嵌入、解码器层、RoPE)
├─ 创建 LM Head (线性投影)
└─ 设置 KV Cache (通过 reset_cache)
    ↓ (返回 shared_ptr<InfinilmModel>)
多态模型实例
```

**关键设计**：工厂类通过 `model_type` 字段（如 "llama"）动态选择实现，用户代码无需感知具体模型类型，仅依赖 `InfinilmModel` 基类接口。

#### 3.2 前向推理流程（Llama 架构为例）

```
Input 结构体 (input_ids, position_ids, cache_lengths, 等)
    ↓
LlamaForCausalLM::forward()
    ├─ 提取输入张量 (input_ids, position_ids)
    ↓
    └─ LlamaModel::forward()
        ├─ 1. 词嵌入: embed_tokens(input_ids) → hidden_states
        ├─ 2. 层堆叠: 遍历 layers_ (32 个 LlamaDecoderLayer)
        │   └─ 每个 LlamaDecoderLayer::forward()
        │       ├─ Pre-LN: input_layernorm(hidden_states)
        │       ├─ 自注意力: LlamaAttention::forward()
        │       │   ├─ QKV 融合投影 (QKVParallelLinear)
        │       │   ├─ Reshape 为多头格式
        │       │   ├─ 应用 RoPE 旋转位置编码
        │       │   ├─ 查询/更新 KV Cache
        │       │   ├─ Grouped Query Attention (GQA) 计算
        │       │   └─ 输出投影 (o_proj)
        │       ├─ 残差连接: hidden_states + attn_output
        │       ├─ Pre-LN: post_attention_layernorm(output)
        │       ├─ MLP: LlamaMLP::forward()
        │       │   ├─ Gate-Up 融合投影 (GateUpParallelLinear)
        │       │   ├─ SwiGLU 激活: SiLU(gate) ⊙ up
        │       │   └─ 下投影 (down_proj)
        │       └─ 残差连接: output + mlp_output
        └─ 3. 最后 token 归一化: RMSNorm(hidden_states[:, -1, :])
    ↓
    └─ LM Head 投影: lm_head(hidden_states) → logits
    ↓
Output 结构体 (logits: [batch, 1, vocab_size])
```

**数据变换**：
- **输入**: Token IDs [batch, seq_len] → 嵌入后 [batch, seq_len, hidden_size]
- **层内变换**: 每层通过注意力（序列间交互）和 MLP（ token 独立变换）交替处理
- **输出**: 最后一个 token 的表示 [batch, 1, hidden_size] → 投影到词汇表 logits [batch, 1, vocab_size]

#### 3.3 调试工具集成（Hook 机制）

```
用户注册回调
    ↓ (register_hook)
HookRegistry::hooks_ ["layer0_q_after_proj" → callback, "layer0_*" → wildcard_callback]
    ↓ (传递到模型各层)
model->set_hook_registry(hook_registry, "model")
    ├─ LlamaModel 传递前缀 "model"
    ├─ 每层 LlamaDecoderLayer 传递前缀 "model_layer0", "model_layer1", ...
    └─ Attention/MLP 传递前缀 "model_layer0_attention", "model_layer0_mlp", ...
    ↓ (模型执行时触发)
CALL_HOOK_LAYER(registry, "q_after_proj", tensor, 0)
    ├─ 检查 has_hooks() (快速路径优化)
    ├─ 构建完整钩子名: "model_layer0_attention_q_after_proj"
    ├─ 精确匹配: hooks_.find("model_layer0_attention_q_after_proj")
    ├─ 通配符匹配: 遍历所有 "model_layer0_*" 模式
    └─ 执行回调: callback(name, tensor, layer_idx)
        └─ log_tensor_stats(tensor, name) (记录 min/max/mean/样本)
```

**Hook 命名层次结构**：
- **顶层**: "model" 前缀
- **层索引**: "model_layer0", "model_layer1", ...
- **模块名**: "model_layer0_attention", "model_layer0_mlp"
- **计算点**: "model_layer0_attention_q_after_proj", "model_layer0_mlp_gate_output"

#### 3.4 分布式训练与 KV Cache 管理

```
RankInfo (tp_rank, tp_size, comm)
    ↓ (传入模型构造函数)
LlamaForCausalLM
    ↓ (传递到所有子模块)
├─ LlamaModel (词嵌入、层归一化)
├─ LlamaDecoderLayer (注意力和 MLP)
│   ├─ LlamaAttention
│   │   ├─ QKVParallelLinear (按 tp_size 分割 Q/K/V 头)
│   │   │   └─ num_attention_heads_ = num_attention_heads / tp_size
│   │   │   └─ num_key_value_heads_ = num_key_value_heads / tp_size
│   │   └─ o_proj (RowParallelLinear 行并行)
│   └─ LlamaMLP
│       ├─ GateUpParallelLinear (按 tp_size 分割中间维度)
│       └─ down_proj (RowParallelLinear 行并行)
└─ KV Cache (StaticKVCache 或 PagedKVCache)
    └─ 按 tp_rank 分割 KV 数据，每个秩存储 num_kv_heads / tp_size 个头
```

**张量并行策略**：
- **注意力头分割**: 每个 TP 秩负责 `num_attention_heads / tp_size` 个头，通过 `ColumnParallelLinear` 分割 Q/K/V 投影
- **MLP 维度分割**: 按 `intermediate_size / tp_size` 分割门控和上投影层
- **通信需求**: 注意力计算需要 AllReduce 聚合各秩的输出（通过 `RowParallelLinear` 自动处理）

#### 3.5 与外部系统的依赖关系

```
上层: 推理引擎/用户代码
    ↓ (调用)
Models 子系统 (InfinilmModel 基类 + 具体实现)
    ├─ 依赖: InfiniCore 框架
    │   ├─ nn/module.hpp (Module 基类、模块注册宏)
    │   ├─ nn/linear.hpp (Linear, RowParallelLinear, ColumnParallelLinear)
    │   ├─ nn/rope.hpp (旋转位置编码)
    │   ├─ nn/rmsnorm.hpp (RMS 归一化)
    │   ├─ ops.hpp (matmul, softmax, swiglu, add 等算子)
    │   └─ tensor.hpp (张量操作、设备管理)
    ├─ 依赖: InfiniLM 内部模块
    │   ├─ cache/kv_cache.hpp (KV Cache 抽象和实现)
    │   ├─ layers/fused_linear.hpp (融合线性层)
    │   └─ engine/distributed/distributed.hpp (RankInfo 分布式信息)
    └─ 依赖: 第三方库
        ├─ spdlog (日志框架，用于 debug_utils)
        └─ 标准库 (optional, any, functional, vector, unordered_map)
```

### 关键设计模式与架构决策

1. **策略模式**：`InfinilmModel` 定义统一接口，`LlamaForCausalLM` 等具体类实现不同架构策略，支持运行时切换。
2. **工厂模式**：`InfinilmModelFactory` 封装对象创建逻辑，解耦模型使用和实现。
3. **模板方法模式**：基类定义算法骨架（`forward()` 流程），子类实现具体步骤（各层的计算细节）。
4. **观察者模式**：`HookRegistry` 实现观察者模式，模型是主题，钩子是观察者，支持动态注册/注销调试回调。
5. **组合模式**：`LlamaModel` 组合多个 `LlamaDecoderLayer`，每个层组合 `LlamaAttention` 和 `LlamaMLP`，形成层次化结构。
6. **依赖注入**：RoPE 模块和 HookRegistry 通过 `set_rotary_emb()` 和 `set_hook_registry()` 注入，而非在构造函数中创建，提升灵活性。

### 性能优化关键点

- **算子融合**：`QKVParallelLinear` 和 `GateUpParallelLinear` 在单次 kernel 调用中完成多个矩阵乘法，减少内存访问和 kernel 启动开销。
- **KV Cache 复用**：增量推理时缓存历史 token 的 K/V，避免重复计算（支持 StaticKVCache 和 PagedKVCache 两种策略）。
- **Grouped Query Attention (GQA)**：通过减少 KV head 数量（如 64 个 query head 对应 8 个 KV head）降低内存占用，尤其适合大模型推理。
- **RoPE 位置编码**：相对位置编码无需训练绝对位置嵌入，支持外推到更长序列长度（通过调整 `rope_theta` 参数）。
- **Pre-LN Transformer**：归一化层放在残差连接之前，提升训练稳定性和推理性能，避免梯度消失/爆炸。

### 扩展性与可维护性

- **新增模型**：继承 `InfinilmModel` 基类，实现 `forward()` 和 `reset_cache()` 方法，在工厂类注册即可（如支持 GPT、BERT 等架构）。
- **新增调试点**：在模型任意位置添加 `CALL_HOOK` 宏，无需修改核心逻辑，支持非侵入式调试。
- **硬件后端扩展**：通过 InfiniCore 的抽象，支持 CUDA、CPU、ROCm、MUSA 等多种设备（模型代码无需修改）。
- **分布式扩展**：当前实现张量并行，架构设计支持未来添加流水线并行和数据并行。
