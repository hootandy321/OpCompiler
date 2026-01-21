# 📂 目录: example 架构全景

## 1. 子系统职责

`example` 目录是 InfiniTrain 框架的示例应用集合，展示了如何使用 InfiniTrain 进行不同类型的深度学习模型训练。该目录提供了从基础的图像分类（MNIST）到复杂的大语言模型（GPT-2、LLaMA 3）的完整训练示例，涵盖了：

- **数据集处理**：自定义数据集实现（MNIST、TinyShakespeare）
- **模型构建**：从简单的 CNN 到复杂的 Transformer 架构
- **并行训练**：数据并行、张量并行、流水线并战的实际应用
- **工具支持**：分词器、工具函数等辅助组件

此目录在整个 InfiniTrain 架构中扮演**示范和教学**的角色，为开发者提供可直接运行的训练模板，同时也作为框架功能的验证测试集。

## 2. 模块导航

### 📂 common
* **功能**: 公共工具和数据集组件库
* **职责**: 提供跨示例共享的基础设施，包括数据集加载、文本分词和通用工具函数
    * `tiny_shakespeare_dataset.*`: Tiny Shakespeare 文本数据集实现，支持 GPT-2 和 LLaMA 3 两种数据格式（UINT16/UINT32）
    * `tokenizer.*`: BPE 分词器实现，用于文本 tokenization
    * `utils.*`: 通用工具函数集合

### 📂 gpt2
* **功能**: GPT-2 模型训练完整示例
* **职责**: 展示如何使用 InfiniTrain 训练 GPT-2 语言模型，支持多种并行策略
    * `net.h/cc`: 完整的 GPT-2 模型定义，包含以下组件：
        * `NewGELU`: GPT-2 使用的 GELU 激活函数变体
        * `CausalSelfAttention`: 因果自注意力层（支持多头注意力）
        * `MLP`: 前馈神经网络层（c_fc → gelu → c_proj）
        * `Block`: Transformer 基础块（ln_1 → attn → ln_2 → mlp）
        * `GPT2FirstStage`: 流水线第一阶段（词嵌入 wte + 位置编码 wpe）
        * `GPT2Chunk`: 流水线中间阶段（包含多个 Transformer Block）
        * `GPT2LastStage`: 流水线最后阶段（层归一化 ln_f + LM Head）
        * `GPT2`: 完整的 GPT-2 模型，支持从预训练权重或 LLMC 格式加载
    * `main.cc`: 训练主程序，支持：
        * 数据并行（DDP）、张量并行（TP）、流水线并行（PP）及其混合并行
        * 梯度累积、混合精度训练
        * 学习率调度、验证评估、文本生成
        * 性能分析（Profiling）
        * 多种模型规格（GPT-2/Medium/Large/XL）

### 📂 llama3
* **功能**: LLaMA 3 模型训练完整示例
* **职责**: 展示如何使用 InfiniTrain 训练 LLaMA 3 大语言模型，采用现代 Transformer 优化技术
    * `net.h/cc`: 完整的 LLaMA 3 模型定义，包含以下组件：
        * `SwiGLU`: SwiGLU 激活函数（比 GELU 更高效）
        * `RMSNorm`: RMS 层归一化（比 LayerNorm 更稳定）
        * `CausalSelfAttention`: 因果自注意力（支持 GQA - 分组查询注意力，n_kv_head < n_head）
        * `MLP`: SwiGLU 前馈网络（c_fc → silu → c_fc2 → c_proj）
        * `Block`: LLaMA 3 Transformer 块（rmsnorm1 → attn → rmsnorm2 → mlp）
        * `LLaMA3FirstStage`: 流水线第一阶段（词嵌入 wte）
        * `LLaMA3Chunk`: 流水线中间阶段（包含 freqs_cis 和多个 Transformer Block）
        * `LLaMA3LastStage`: 流水线最后阶段（rmsnorm + lm_head）
        * `LLaMA3`: 完整的 LLaMA 3 模型，支持多种模型规模（1B/3B/8B/70B）
    * `main.cc`: 训练主程序，架构与 GPT-2 类似，但针对 LLaMA 3 的特性进行了适配

### 📂 mnist
* **功能**: MNIST 手写数字识别训练示例
* **职责**: 展示 InfiniTrain 的基础训练流程，是最简单的入门示例
    * `dataset.h/cc`: MNIST 数据集实现，支持训练集和测试集加载
    * `net.h/cc`: 简单的 CNN 模型定义（卷积层 + 池化层 + 全连接层）
    * `main.cc`: 训练主程序，展示：
        * 数据加载和预处理
        * 模型实例化和设备选择（CPU/CUDA）
        * 训练循环（前向传播、损失计算、反向传播、参数更新）
        * 验证评估

## 3. 架构逻辑图解

### 数据流向

```
┌─────────────────────────────────────────────────────────────┐
│                        训练任务启动                          │
│                 (example/*/main.cc)                          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  1. 数据加载与预处理           │
        │  (common/dataset, tokenizer)  │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  2. 模型构建                   │
        │  (example/*/net)              │
        │  - MNIST: 简单 CNN            │
        │  - GPT2/LLaMA3: Transformer   │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  3. 并行策略配置               │
        │  (DDP/TP/PP/Hybrid)           │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  4. 训练循环                   │
        │  - 前向传播                    │
        │  - 损失计算                    │
        │  - 反向传播                    │
        │  - 参数更新                    │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  5. 验证与生成                 │
        │  (评估损失 / 文本生成)         │
        └───────────────────────────────┘
```

### 模块依赖关系

```
common (公共组件)
  ├── tiny_shakespeare_dataset → 供 gpt2/llama3 使用
  ├── tokenizer                 → 供 gpt2/llama3 使用
  └── utils                     → 供所有示例使用

mnist (独立示例)
  ├── dataset
  ├── net
  └── main

gpt2 (独立示例)
  ├── net (依赖 common)
  └── main (依赖 common + net)

llama3 (独立示例)
  ├── net (依赖 common)
  └── main (依赖 common + net)
```

### 并行策略演进

示例展示了从简单到复杂的并行训练方案：

1. **MNIST**: 单设备训练（CPU/CUDA）
2. **GPT-2**: 完整的并行策略支持
   - **数据并行 (DDP)**: 多卡副本训练
   - **张量并行 (TP)**: 将注意力层和 MLP 切分到多卡
   - **流水线并行 (PP)**: 将模型层分段到不同设备
   - **混合并行**: DDP + TP + PP 组合
3. **LLaMA 3**: 继承 GPT-2 的并行能力，增加：
   - **GQA 支持**: 分组查询注意力优化推理性能
   - **RoPE 位置编码**: 支持长文本训练
   - **KV Cache**: 推理时的缓存优化

### 技术特点对比

| 特性 | MNIST | GPT-2 | LLaMA 3 |
|------|-------|-------|---------|
| 模型类型 | CNN | Transformer | Transformer |
| 激活函数 | ReLU | NewGELU | SwiGLU |
| 归一化 | BatchNorm | LayerNorm | RMSNorm |
| 位置编码 | - | 绝对位置编码 | RoPE |
| 注意力机制 | - | MHA | GQA (可选) |
| 并行支持 | 单设备 | DDP/TP/PP | DDP/TP/PP |
| 数据集 | MNIST | TinyShakespeare | TinyShakespeare |
| 训练目标 | 分类 | 语言建模 | 语言建模 |

### 学习路径建议

1. **入门**: 先阅读 `mnist/`，理解基础训练流程
2. **进阶**: 学习 `gpt2/`，掌握 Transformer 和并行训练
3. **高级**: 研究 `llama3/`，了解现代 LLM 优化技术
4. **复用**: 参考 `common/`，学习如何构建自定义组件

---

**注**: 所有示例均遵循 InfiniTrain 的统一 API 设计，核心概念包括：
- `Dataset/DataLoader`: 数据抽象
- `Module`: 模型组件基类
- `Optimizer`: 优化器接口
- `Device`: 设备管理
- `Parallel`: 并行策略（DDP/TP/PP）
