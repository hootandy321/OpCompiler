# Python NN 模块架构全景

## 1. 子系统职责

本目录 `infinicore.nn` 是 InfiniCore 深度学习推理框架的 Python 神经网络层抽象，提供了与 PyTorch 兼容的高层 API 封装。该子系统采用**双层架构设计**：

* **函数式层（functional）**：提供无状态的纯函数接口，直接调用底层 C++ 算子实现，实现核心神经网络原语（如线性变换、激活函数、归一化、位置编码等）。
* **模块层（modules）**：提供面向对象的有状态组件封装，通过参数管理、状态序列化和模块组合机制，构建可复用的神经网络层（如 Linear、RMSNorm、Embedding、RoPE 等）。

这种设计遵循 PyTorch 的 API 规范，使用户能够使用熟悉的接口定义神经网络模型，同时底层计算通过 InfiniCore 的高性能 C++ 内核执行。该子系统在 InfiniCore 整体架构中扮演**推理应用层**的角色，连接用户模型定义与底层计算引擎。

## 2. 模块导航 (Module Navigation)

### **📂 functional** - 函数式 API 实现层
* **功能**：提供核心神经网络计算的函数式接口，每个函数都是无状态的纯函数，直接绑定到 C++ 算子实现，支持 in-place 操作和可选输出张量参数。
* **职责**：实现底层神经网络原语，包括注意力机制（causal_softmax）、线性变换（linear）、归一化（rms_norm）、激活函数（silu, swiglu）、位置编码（rope）、嵌入查找（embedding）、随机采样（random_sample）。

### **📂 modules** - 面向对象模块封装层
* **功能**：提供 PyTorch 兼容的神经网络层抽象，所有模块继承自 InfiniCoreModule 基类，实现参数注册、状态字典序列化、模块层次管理和前向传播计算。
* **职责**：
  * **module.py**：核心基类 InfiniCoreModule，实现参数/缓冲区注册、state_dict 序列化、模块层次遍历等基础设施。
  * **container.py**：ModuleList 容器，提供类列表接口的模块集合管理。
  * **linear.py**：Linear 层，实现仿射变换 y = xA^T + b。
  * **normalization.py**：RMSNorm 层，实现 RMS 层归一化（Root Mean Square Layer Normalization）。
  * **rope.py**：RoPE 模块，实现旋转位置编码（Rotary Position Embedding），预计算 sin/cos 查找表。
  * **sparse.py**：Embedding 层，实现稀疏查找表操作（词嵌入）。

### **📄 parameter.py** - 参数类型定义（根目录文件）
* **功能**：定义 InfiniCoreParameter 类型，作为 InfiniCore.Tensor 的包装器，用于模块的可训练参数标识。
* **职责**：区分普通张量与可学习参数，参与模块的参数注册和状态序列化机制。

## 3. 架构逻辑图解

### 3.1 双层架构关系

```
┌─────────────────────────────────────────────────────────────┐
│                    用户模型定义层                             │
│  (用户使用 modules 层的 Linear, RMSNorm, Embedding 等)         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   modules 模块封装层                          │
│  InfiniCoreModule (基类)                                      │
│    ├── 参数管理 (_parameters, _buffers, _modules)              │
│    ├── 状态序列化 (state_dict, load_state_dict)               │
│    └── 前向传播 (forward 方法)                                 │
│                                                               │
│  具体模块: Linear, RMSNorm, RoPE, Embedding, ModuleList        │
└──────────────────────────┬──────────────────────────────────┘
                           │ 调用
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  functional 函数式层                          │
│  无状态纯函数接口                                              │
│    ├── 线性变换: linear(input, weight, bias)                  │
│    ├── 归一化: rms_norm(input, normalized_shape, weight)       │
│    ├── 激活函数: silu(input), swiglu(gate, value)            │
│    ├── 位置编码: rope(x, pos_ids, sin_table, cos_table)      │
│    ├── 注意力: causal_softmax(input)                          │
│    ├── 嵌入: embedding(input, weight)                        │
│    └── 采样: random_sample(logits, topp, topk, temperature)   │
└──────────────────────────┬──────────────────────────────────┘
                           │ 绑定
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               C++ 后端算子实现 (_infinicore)                  │
│  高性能计算内核 (支持多硬件后端: CPU, CUDA, MUSA 等)            │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 数据流向

**前向传播（Forward Pass）数据流**：

```
输入数据 (Tensor)
    │
    ▼
┌──────────────────────┐
│ 1. Embedding 层       │
│    modules.Embedding │  ──调用──>  functional.embedding()
└──────────────────────┘                  (查表操作)
    │
    ▼ 输出: (batch, seq_len, hidden_dim)
┌──────────────────────┐
│ 2. Transformer Block  │
│    ├── Linear (QKV)   │  ──调用──>  functional.linear()
│    ├── RoPE           │  ──调用──>  functional.rope()
│    ├── Attention      │  ──调用──>  functional.causal_softmax()
│    ├── Linear (Out)   │  ──调用──>  functional.linear()
│    └── RMSNorm        │  ──调用──>  functional.rms_norm()
└──────────────────────┘
    │
    ▼ 输出: (batch, seq_len, hidden_dim)
┌──────────────────────┐
│ 3. 输出投影层         │
│    modules.Linear    │  ──调用──>  functional.linear()
└──────────────────────┘
    │
    ▼ 最终输出: (batch, seq_len, vocab_size)
┌──────────────────────┐
│ 4. 采样 (生成)         │
│    functional        │  ──调用──>  functional.random_sample()
│    .random_sample    │               (top-p/top-k采样)
└──────────────────────┘
    │
    ▼ 采样 Token ID
```

### 3.3 模块组合与层次结构

```
InfiniCoreModule (根模块)
    │
    ├── 参数 (_parameters): OrderedDict[str, Parameter]
    │   ├── "weight": Parameter(...)
    │   └── "bias": Parameter(...)
    │
    ├── 缓冲区 (_buffers): OrderedDict[str, Tensor]
    │   ├── "_sin_table": Tensor(...)  # RoPE 预计算表
    │   └── "_cos_table": Tensor(...)
    │
    └── 子模块 (_modules): OrderedDict[str, InfiniCoreModule]
        │
        ├── "embedding": Embedding (InfiniCoreModule)
        │       └── _parameters: {"weight": Parameter}
        │
        ├── "layers": ModuleList (InfiniCoreModule)
        │       └── _modules: {"0": TransformerBlock, "1": ...}
        │               │
        │               └── TransformerBlock (InfiniCoreModule)
        │                       ├── _modules: {"attention": ..., "mlp": ...}
        │                       └── _parameters: {"qkv_proj.weight", ...}
        │
        └── "norm": RMSNorm (InfiniCoreModule)
                └── _parameters: {"weight": Parameter}
```

**状态序列化（state_dict）**：

```
model.state_dict() 递归遍历模块树
    │
    ├── "embedding.weight": Tensor([vocab_size, hidden_dim])
    ├── "layers.0.qkv_proj.weight": Tensor([3*hidden, hidden])
    ├── "layers.0.out_proj.weight": Tensor([hidden, hidden])
    ├── "layers.0.norm.weight": Tensor([hidden_dim])
    ├── "layers.0.rope._sin_table": Tensor([max_pos, head_dim//2])
    ├── "layers.0.rope._cos_table": Tensor([max_pos, head_dim//2])
    ├── "layers.1.qkv_proj.weight": ...
    └── ...
```

### 3.4 模块层与函数层协作模式

**模式 1：模块封装函数（典型模式）**

```python
# modules 层：有状态封装
class Linear(InfiniCoreModule):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        # 注册参数（生命周期由模块管理）
        self.weight = Parameter(infinicore.empty([out_features, in_features]))
        if bias:
            self.bias = Parameter(infinicore.empty([out_features]))

    def forward(self, input: Tensor) -> Tensor:
        # 调用 functional 层的无状态函数
        return F.linear(input, self.weight, self.bias)

# functional 层：无状态函数
def linear(input, weight, bias=None, *, out=None):
    # 直接调用 C++ 绑定
    return _infinicore.linear(input._underlying, weight._underlying, ...)
```

**模式 2：模块预计算 + 函数应用（RoPE 案例）**

```python
# modules 层：初始化时预计算查找表
class RoPE(InfiniCoreModule):
    def __init__(self, max_position_embeddings, rope_theta, head_dim):
        super().__init__()
        # 预计算 sin/cos 表（一次性计算，存储为缓冲区）
        sin_table, cos_table = self.create_sin_cos_table(...)
        self.register_buffer("_sin_table", sin_table)
        self.register_buffer("_cos_table", cos_table)

    def forward(self, states, position_ids, algo):
        # 使用预计算表调用 functional 函数
        return F.rope(states, position_ids,
                      self._sin_table, self._cos_table, algo, out=states)

# functional 层：使用查找表应用位置编码
def rope(x, pos_ids, sin_table, cos_table, algo, *, out=None):
    return _infinicore.rope(x._underlying, pos_ids._underlying, ...)
```

**模式 3：容器管理模块列表（ModuleList 案例）**

```python
# 用户定义多层网络
class Transformer(InfiniCoreModule):
    def __init__(self, num_layers, hidden_dim):
        super().__init__()
        # 使用 ModuleList 管理多个子模块
        self.layers = ModuleList([
            TransformerBlock(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        # 遍历 ModuleList 逐层计算
        for layer in self.layers:
            x = layer(x)
        return x

# ModuleList 内部使用 OrderedDict 存储模块
# _modules = {"0": TransformerBlock, "1": TransformerBlock, ...}
```

### 3.5 硬件加速路径

```
用户调用 silu(input)
    │
    ▼
functional.silu(input, inplace=False, out=None)
    │
    ├── 检查加速条件
    │   ├── infinicore.use_ntops == True?
    │   ├── device in ["cuda", "musa"]?
    │   └── out is None?
    │
    ├── 满足条件 → 使用 ntops.torch.silu() (硬件优化路径)
    │
    └── 不满足 → 使用 _infinicore.silu() (通用 C++ 路径)
                │
                ├── inplace == True → _infinicore.silu_() (原地修改)
                └── inplace == False → _infinicore.silu() (新张量)
```

### 3.6 内存优化策略

**In-Place 操作优化链**：

```python
# 内存优化的 FFN 前向传播
def memory_efficient_ffn(x, w_gate, w_up, w_down, norm_weight):
    # 1. 线性变换（必须创建新张量）
    gate = F.linear(x, w_gate)  # 新张量
    up = F.linear(x, w_up)      # 新张量

    # 2. 重用 gate 张量进行 in-place SiLU
    F.silu(gate, inplace=True)  # 原地修改，无新分配

    # 3. SwiGLU 结果写入 gate 张量，重用内存
    F.swiglu(gate, up, out=gate)  # gate 被覆盖

    # 4. 输出投影
    output = F.linear(gate, w_down)  # 新张量

    # 5. RMS 归一化 in-place
    F.rms_norm(output, [output.shape[-1]], norm_weight, out=output)

    return output  # 仅分配 3 个张量（gate, up, output），而非 6 个
```

**参数复用与共享**：

```python
# 权重共享（多个层使用同一参数）
class SharedWeightModel(InfiniCoreModule):
    def __init__(self, hidden_dim):
        super().__init__()
        self.weight = Parameter(...)

        # 多个模块共享同一参数（引用同一对象）
        self.layer1 = Linear(hidden_dim, hidden_dim)
        self.layer1.weight = self.weight  # 共享

        self.layer2 = Linear(hidden_dim, hidden_dim)
        self.layer2.weight = self.weight  # 共享

    # state_dict 仅保存一份权重
    # "layer1.weight" 和 "layer2.weight" 指向同一对象
```

## 4. 设计原则与最佳实践

### 4.1 职责分离

* **functional 层**：专注于计算逻辑，保持无状态、可组合、可测试。不管理参数生命周期，不维护内部状态。
* **modules 层**：专注于状态管理，负责参数注册、模块组合、序列化、前向传播编排。不直接实现计算逻辑，委托给 functional 层。

### 4.2 PyTorch 兼容性

* **API 一致性**：函数签名、参数命名、返回值类型与 PyTorch 对齐（如 `Linear(in_features, out_features, bias=False)`）。
* **状态字典格式**：使用点分隔的层次化键名（如 `layers.0.weight`），与 PyTorch 模型互操作。
* **模块组合模式**：支持嵌套子模块、参数共享、ModuleList 容器等 PyTorch 惯用法。

### 4.3 性能优化策略

* **预计算策略**：RoPE 在初始化时预计算 sin/cos 查找表，避免前向传播重复计算。
* **In-Place 操作**：提供 `inplace=True` 和 `out=` 参数支持内存重用，减少大模型推理的内存占用。
* **硬件加速**：通过 `infinicore.use_ntops` 配置，选择硬件优化算子库（如 NVIDIA/MUSA GPU 的 ntops）。
* **算子融合**：C++ 层可能融合多个操作（如 softmax + 因果掩码融合为 causal_softmax）。

### 4.4 扩展性指南

**添加新函数（functional 层）**：

1. 在 C++ 层实现算子（添加到 `_infinicore` 扩展模块）。
2. 在 `functional/` 目录创建对应 Python 文件，编写包装函数。
3. 遵循命名约定：非 in-place 版本调用 `function()`，in-place 版本调用 `function_()`。
4. 支持可选 `out` 参数用于内存优化。
5. 在 `functional/__init__.py` 中导出函数。

**添加新模块（modules 层）**：

1. 继承 `InfiniCoreModule` 基类。
2. 在 `__init__` 中通过 `self.param_name = Parameter(...)` 注册参数。
3. 通过 `register_buffer()` 注册非参数张量（如预计算表、运行统计）。
4. 实现 `forward()` 方法，调用 `functional` 层的函数完成计算。
5. 实现 `extra_repr()` 返回模块关键配置信息（如 `in_features`, `out_features`）。
6. 在 `modules/__init__.py` 中导出新模块。

## 5. 典型应用场景

### 场景 1：构建 Transformer 语言模型

```python
class LlamaLikeModel(InfiniCoreModule):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads):
        super().__init__()
        self.embedding = Embedding(vocab_size, hidden_dim)
        self.layers = ModuleList([
            TransformerBlock(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_dim)

    def forward(self, input_ids, position_ids):
        # 1. 词嵌入
        hidden = self.embedding(input_ids)

        # 2. 堆叠 Transformer 层
        for layer in self.layers:
            hidden = layer(hidden, position_ids)

        # 3. 最终归一化
        hidden = self.norm(hidden)

        # 4. 投影到词表
        logits = linear(hidden, self.embedding.weight.t())  # 权重共享

        return logits

# 保存/加载模型权重
state_dict = model.state_dict()  # 保存
model.load_state_dict(state_dict)  # 加载
```

### 场景 2：大语言模型推理（含采样）

```python
def generate_text(model, prompt_ids, max_tokens=100):
    """自回归文本生成"""
    generated = prompt_ids.tolist()

    for _ in range(max_tokens):
        input_ids = Tensor(generated)
        position_ids = Tensor.arange(len(generated)).unsqueeze(0)

        # 1. 前向传播获取 logits
        logits = model(input_ids, position_ids)  # [1, seq_len, vocab_size]

        # 2. 取最后一个位置的 logits
        next_token_logits = logits[0, -1, :]  # [vocab_size]

        # 3. nucleus/top-k 采样
        random_val = random.random()
        next_token = random_sample(
            logits=next_token_logits,
            random_val=random_val,
            topp=0.9,
            topk=50,
            temperature=0.8
        )

        # 4. 添加到生成序列
        generated.append(next_token.item())

        # 5. 检查结束符
        if next_token.item() == eos_token_id:
            break

    return generated
```

### 场景 3：内存优化的批量推理

```python
def batch_inference_efficient(model, input_ids_batch, position_ids_batch):
    """使用 in-place 操作优化批量推理内存"""
    batch_outputs = []

    for input_ids, position_ids in zip(input_ids_batch, position_ids_batch):
        # 激活 in-place 模式减少内存分配
        output = model(input_ids, position_ids)

        # 对输出进行 in-place 归一化
        rms_norm(output, [output.shape[-1]], model.norm.weight, out=output)

        batch_outputs.append(output)

    return batch_outputs
```

## 6. 依赖关系图

```
infinicore.nn
    │
    ├── 内部依赖
    │   ├── infinicore.Tensor (张量类型)
    │   ├── infinicore.Parameter (参数类型)
    │   ├── infinicore.device (设备管理)
    │   ├── infinicore.empty, from_numpy (张量构造)
    │   └── infinicore.lib._infinicore (C++ 扩展模块)
    │
    ├── Python 标准库
    │   ├── collections.OrderedDict
    │   ├── typing (类型注解)
    │   ├── itertools.chain
    │   └── numbers.Integral
    │
    └── 外部依赖（条件依赖）
        ├── numpy (RoPE 预计算，可替换)
        └── ntops (硬件加速库，可选)
```

## 7. 性能特征

* **计算复杂度**：
  * Linear: O(batch_size * in_features * out_features)
  * causal_softmax: O(batch_size * num_heads * seq_len^2) - 注意力瓶颈
  * rms_norm: O(batch_size * seq_len * hidden_dim)
  * rope: O(batch_size * seq_len * num_heads * head_dim) - 查找表操作

* **内存占用**：
  * 模块参数：O(total_parameters) - 由模型大小决定
  * RoPE 查找表：O(max_position_embeddings * head_dim) - 固定开销
  * 前向传播中间结果：O(batch_size * seq_len * hidden_dim * num_layers)

* **优化级别**：
  * C++ 内核：使用 SIMD、并行算法、算子融合
  * 硬件加速：针对 CUDA/MUSA 的优化内核（ntops）
  * Python 层：最小化开销，直接转发到 C++ 层
