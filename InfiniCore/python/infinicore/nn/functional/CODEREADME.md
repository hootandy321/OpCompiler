# `functional` 神经网络函数式API实现文档

本模块提供了一套完整的函数式API接口,用于构建神经网络的前向传播计算。这些函数都是无状态的纯函数接口,直接调用底层C++绑定的核心算子实现,支持in-place操作和可选输出张量。

## 1. 模块结构

- **`__init__.py`**: 模块入口,导出所有公共函数式API
- **`causal_softmax.py`**: 因果掩码softmax实现,用于Transformer自注意力机制
- **`embedding.py`**: 嵌入层查找表实现,将离散索引映射到连续向量
- **`linear.py`**: 线性变换层实现:y=xA^T+b
- **`random_sample.py`**: 核采样与top-k采样实现,用于大语言模型生成
- **`rms_norm.py`**: RMS归一化层实现,Root Mean Square Layer Normalization
- **`rope.py`**: 旋转位置编码实现(Rotary Position Embedding),支持GPT-J和GPT-NEOX两种算法
- **`silu.py`**: SiLU激活函数实现,Sigmoid Linear Unit,支持in-place操作和硬件后端加速
- **`swiglu.py`**: SwiGLU激活函数实现,Swish-Gated Linear Unit

## 2. 核心函数

### `causal_softmax(input, out=None) -> Tensor`
- **位置**: `causal_softmax.py`
- **主要功能**: 对输入张量应用因果掩码softmax,确保在计算softmax时当前位置只能关注之前的位置(包括自身)
- **底层绑定**: `_infinicore.causal_softmax()` 和 `_infinicore.causal_softmax_()`
- **参数**:
  - `input: Tensor`: 输入张量,通常是未经归一化的注意力分数
  - `out: Tensor | None`: 可选输出张量,用于in-place操作
- **返回值**: 应用因果softmax后的张量
- **实现细节**:
  - 提供`out`参数时执行in-place操作,调用带下划线的后缀版本函数
  - 未提供`out`时创建新张量返回
- **应用场景**: Transformer解码器的自注意力机制,确保自回归生成的因果性约束

### `embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, *, out=None) -> Tensor`
- **位置**: `embedding.py`
- **主要功能**: 实现词嵌入查找,将离散的token索引映射到连续的稠密向量空间
- **底层绑定**: `_infinicore.embedding()` 和 `_infinicore.embedding_()`
- **参数**:
  - `input: Tensor`: 索引张量,包含要查找的token ID
  - `weight: Tensor`: 嵌入权重矩阵,形状为`[vocab_size, embedding_dim]`
  - `padding_idx: int | None`: 填充索引(当前不支持,必须为None)
  - `max_norm: float | None`: 最大范数归一化(当前不支持,必须为None)
  - `norm_type: float`: 范数类型,默认2.0(当前不支持)
  - `scale_grad_by_freq: bool`: 是否按频率缩放梯度(当前不支持,必须为False)
  - `sparse: bool`: 是否使用稀疏梯度(当前不支持,必须为False)
  - `out: Tensor | None`: 可选输出张量
- **返回值**: 查找得到的嵌入向量
- **约束验证**:
  - `assert`检查确保`padding_idx`、`max_norm`均为`None`
  - `assert`检查确保`scale_grad_by_freq`为`False`
  - `assert`检查确保`sparse`为`False`
  - `assert`检查确保输入张量必须在CPU设备上(`input.device.type == "cpu"`)
- **实现细节**:
  - 当前实现仅支持基本查找功能,不支持高级特性如权重归一化、稀疏梯度
  - 仅支持CPU设备,不支持GPU加速
- **应用场景**: 大语言模型的词嵌入层,将token ID转换为模型可处理的向量表示

### `linear(input, weight, bias=None, *, out=None) -> Tensor`
- **位置**: `linear.py`
- **主要功能**: 应用仿射变换:y = xA^T + b,这是全连接层的核心计算
- **底层绑定**: `_infinicore.linear()` 和 `_infinicore.linear_()`
- **参数**:
  - `input: Tensor`: 输入张量,形状为`[*, in_features]`
  - `weight: Tensor`: 权重矩阵,形状为`[out_features, in_features]`
  - `bias: Tensor | None`: 偏置向量,形状为`[out_features]`,可选
  - `out: Tensor | None`: 可选输出张量
- **返回值**: 变换后的张量,形状为`[*, out_features]`
- **实现细节**:
  - 当`bias`为`None`时,传递`None`到底层C++绑定
  - 支持`out`参数的in-place操作模式
- **数学公式**: `output = input @ weight.T + bias`(其中`@`表示矩阵乘法)
- **应用场景**: 神经网络中的全连接层、Transformer的MLP块、输出投影层

### `random_sample(logits, random_val, topp, topk, temperature, *, out=None) -> Tensor`
- **位置**: `random_sample.py`
- **主要功能**: 从logits分布中采样一个token索引,支持top-p(nucleus)和top-k过滤策略
- **底层绑定**: `_infinicore.random_sample()` 和 `_infinicore.random_sample_()`
- **参数**:
  - `logits: Tensor`: 未归一化的预测分数,形状为`[vocab_size]`
  - `random_val: float`: 随机数,范围[0,1),用于采样决策
  - `topp: float`: top-p(nucleus sampling)阈值,保留累积概率达到p的最小集合
  - `topk: int`: top-k采样参数,仅保留概率最大的k个token
  - `temperature: float`: 温度参数,控制分布的平滑度,值越小越确定性
  - `out: Tensor | None`: 可选输出张量
- **返回值**: 采样的token索引,标量张量
- **实现细节**:
  - 同时支持top-p和top-k两种采样策略
  - 温度参数控制随机性:temperature→0趋于贪心,temperature→∞趋于均匀分布
  - 使用外部提供的随机数值,便于确定性生成和种子控制
- **采样算法**:
  1. 对logits应用温度缩放:`scaled_logits = logits / temperature`
  2. 应用softmax得到概率分布
  3. 根据top-p和top-k过滤候选集
  4. 使用`random_val`在过滤后的分布中采样
- **应用场景**: 大语言模型的文本生成,LLM推理解码过程

### `rms_norm(input, normalized_shape, weight, eps=1e-5, *, out=None) -> Tensor`
- **位置**: `rms_norm.py`
- **主要功能**: 应用RMS归一化(Root Mean Square Layer Normalization),相比LayerNorm更高效
- **底层绑定**: `_infinicore.rms_norm()` 和 `_infinicore.rms_norm_()`
- **参数**:
  - `input: Tensor`: 输入张量
  - `normalized_shape: List[int]`: 要归一化的维度形状
  - `weight: Tensor`: 可学习的缩放参数(γ)
  - `eps: float`: 数值稳定性的小常数,默认1e-5
  - `out: Tensor | None`: 可选输出张量
- **返回值**: 归一化后的张量
- **约束验证**:
  - `assert normalized_shape == weight.shape`: 确保归一化形状与权重形状匹配
- **数学公式**:
  ```
  output = input / sqrt(mean(input^2) + eps) * weight
  ```
  计算每个样本的均方根(RMS),然后归一化并应用可学习的缩放
- **与LayerNorm区别**:
  - RMSNorm不使用均值中心化(不减去均值)
  - 仅使用RMS进行归一化,计算更简单
  - 参数更少(没有β偏置参数)
- **应用场景**: Transformer模型的归一化层(特别是LLaMA等现代大模型),替代LayerNorm

### `RopeAlgo`
- **位置**: `rope.py`
- **主要功能**: 枚举类,定义RoPE的两种主流实现算法变体
- **成员**:
  - `GPT_J`: GPT-J模型的RoPE实现方式
  - `GPT_NEOX`: GPT-NEOX模型的RoPE实现方式(默认)
- **实现细节**:
  - 通过`_infinicore.RoPEAlgo.GPT_J`和`_infinicore.RoPEAlgo.GPT_NEOX`从底层C++绑定导入
  - 两种算法在旋转矩阵的计算方式上有所不同
- **区别**:
  - GPT-J: 将head_dim分为两半,分别应用sin/cos旋转
  - GPT-NEOX: 交错方式应用旋转,相邻维度成对旋转

### `rope(x, pos_ids, sin_table, cos_table, algo=RopeAlgo.GPT_NEOX, *, out=None) -> Tensor`
- **位置**: `rope.py`
- **主要功能**: 应用旋转位置编码(Rotary Position Embedding),将位置信息注入到query和key向量中
- **底层绑定**: `_infinicore.rope()` 和 `_infinicore.rope_()`
- **参数**:
  - `x: Tensor`: 输入张量,通常是query或key,形状为`[batch_size, seq_len, num_heads, head_dim]`
  - `pos_ids: Tensor`: 位置索引张量,形状为`[seq_len]`或`[batch_size, seq_len]`
  - `sin_table: Tensor`: 预计算的sin值查找表
  - `cos_table: Tensor`: 预计算的cos值查找表
  - `algo: RopeAlgo`: RoPE算法变体,默认`RopeAlgo.GPT_NEOX`
  - `out: Tensor | None`: 可选输出张量
- **返回值**: 应用RoPE后的张量
- **数学原理**:
  RoPE通过旋转矩阵将绝对位置编码为相对位置信息:
  ```
  RoPE(x, m) = x * cos(mθ) + rotate(x) * sin(mθ)
  ```
  其中`rotate(x)`将向量维度两两配对并旋转90度
- **实现细节**:
  - 使用预计算的sin/cos查找表提高效率
  - 支持GPT-J和GPT-NEOX两种主流变体
  - 操作是可逆的,且具有相对位置感知能力
- **优势**:
  - 相对位置编码:能自然地编码token之间的相对距离
  - 外推性:在一定程度上可以处理比训练序列更长的输入
  - 无额外参数:不增加模型参数量
- **应用场景**: Transformer的query和key位置编码,特别是大语言模型

### `silu(input, inplace=False, *, out=None) -> Tensor`
- **位置**: `silu.py`
- **主要功能**: 应用SiLU激活函数(Sigmoid Linear Unit,也称Swish-1),定义为`x * sigmoid(x)`
- **底层绑定**: `_infinicore.silu()` 和 `_infinicore.silu_()`
- **参数**:
  - `input: Tensor`: 输入张量
  - `inplace: bool`: 是否原地操作,默认False
  - `out: Tensor | None`: 可选输出张量
- **返回值**: 应用SiLU后的张量
- **数学公式**:
  ```
  SiLU(x) = x * σ(x) = x / (1 + e^{-x})
  ```
  其中σ是sigmoid函数
- **实现逻辑**:
  1. **硬件加速路径**: 当满足以下条件时,使用`ntops.torch.silu`实现加速:
     - `infinicore.use_ntops`为True
     - 设备类型为`cuda`或`musa`
     - 未指定`out`参数
  2. **in-place模式**: 当`inplace=True`时,直接修改输入张量并返回
  3. **out参数模式**: 当指定`out`时,将结果写入out张量
  4. **默认模式**: 创建新张量存储结果
- **优先级顺序**: `inplace` > `out` > 硬件加速 > 默认新张量
- **特性**:
  - 平滑非单调函数,在负值区域有软饱和
  - 自门控特性:输出由输入自身控制
  - 相比ReLU有更好的梯度流
- **应用场景**: 现代Transformer的激活函数(如PaLM、LLaMA),替代ReLU

### `swiglu(input, other, *, out=None) -> Tensor`
- **位置**: `swiglu.py`
- **主要功能**: 应用SwiGLU激活函数(Swish-Gated Linear Unit),一种门控线性单元变体
- **底层绑定**: `_infinicore.swiglu()` 和 `_infinicore.swiglu_()`
- **参数**:
  - `input: Tensor`: 第一个输入张量,应用SiLU激活
  - `other: Tensor`: 第二个输入张量,作为门控值
  - `out: Tensor | None`: 可选输出张量
- **返回值**: 应用SwiGLU后的张量
- **数学公式**:
  ```
  SwiGLU(x, y) = SiLU(x) ⊙ y = (x * σ(x)) ⊙ y
  ```
  其中`⊙`表示逐元素乘法(Hadamard积)
- **实现细节**:
  - 对`input`应用SiLU激活函数
  - 将激活结果与`other`逐元素相乘
  - 支持`out`参数的in-place操作
- **网络结构中的应用**:
  通常在Transformer的FFN(Feed-Forward Network)层中使用:
  ```
  FFN(x) = SwiGLU(xW_g, xW_1)W_2
  ```
  其中`W_g`是门控权重,`W_1`和`W_2`是标准线性变换权重
- **参数量**: 相比标准FFN(2个投影矩阵),SwiGLU需要3个投影矩阵,增加50%参数
- **性能**: 虽然参数量增加,但在大模型中通常带来更好的性能
- **应用场景**: 大语言模型的FFN层(如LLaMA、PaLM),替代传统的ReLU或GELU

## 3. API接口

```python
from infinicore.tensor import Tensor
from infinicore.nn.functional import (
    causal_softmax,
    embedding,
    linear,
    random_sample,
    rms_norm,
    rope,
    RopeAlgo,
    silu,
    swiglu
)

# 1. 因果Softmax(用于自注意力)
attention_scores: Tensor  # shape: [batch_size, num_heads, seq_len, seq_len]
masked_attention = causal_softmax(attention_scores)

# 2. 嵌入查找
token_ids = Tensor([1, 2, 3])  # token索引
embedding_weight = Tensor(...)  # shape: [vocab_size, embedding_dim]
embeddings = embedding(token_ids, embedding_weight)

# 3. 线性变换
input_tensor = Tensor(...)  # shape: [batch_size, in_features]
weight = Tensor(...)  # shape: [out_features, in_features]
bias = Tensor(...)  # shape: [out_features]
output = linear(input_tensor, weight, bias)

# 4. 随机采样(LLM生成)
logits = Tensor(...)  # shape: [vocab_size]
sampled_token = random_sample(
    logits=logits,
    random_val=0.7,  # 伪随机数
    topp=0.9,  # nucleus采样阈值
    topk=50,  # top-k采样
    temperature=0.8  # 温度参数
)

# 5. RMS归一化
hidden_states = Tensor(...)  # shape: [batch_size, seq_len, hidden_dim]
normalized_shape = [hidden_dim]
weight = Tensor(...)  # shape: [hidden_dim]
normalized = rms_norm(hidden_states, normalized_shape, weight, eps=1e-5)

# 6. 旋转位置编码
query = Tensor(...)  # shape: [batch_size, seq_len, num_heads, head_dim]
pos_ids = Tensor([0, 1, 2, 3])  # 位置索引
sin_table = Tensor(...)  # 预计算的sin表
cos_table = Tensor(...)  # 预计算的cos表
rotated_query = rope(
    query,
    pos_ids,
    sin_table,
    cos_table,
    algo=RopeAlgo.GPT_NEOX  # 或 RopeAlgo.GPT_J
)

# 7. SiLU激活
hidden = Tensor(...)
activated = silu(hidden, inplace=False)

# 8. SwiGLU激活
gate = Tensor(...)  # 门控分支
value = Tensor(...)  # 值分支
glu_output = swiglu(gate, value)
```

## 4. 使用示例

### 示例1: 构建Transformer的FFN层

```python
from infinicore.tensor import Tensor
from infinicore.nn.functional import linear, silu, swiglu

def feed_forward_network(x, w_gate, w_up, w_down):
    """
    使用SwiGLU的FFN层
    参数:
        x: 输入张量 [batch_size, seq_len, hidden_dim]
        w_gate: 门控权重 [hidden_dim, intermediate_size]
        w_up: 上投影权重 [hidden_dim, intermediate_size]
        w_down: 下投影权重 [intermediate_size, hidden_dim]
    """
    # 分叉为两个分支
    gate = linear(x, w_gate)  # 门控分支
    up = linear(x, w_up)      # 值分支

    # 应用SwiGLU激活
    activated = swiglu(gate, up)  # [batch_size, seq_len, intermediate_size]

    # 下投影回原始维度
    output = linear(activated, w_down)  # [batch_size, seq_len, hidden_dim]

    return output
```

### 示例2: 实现自注意力机制

```python
from infinicore.tensor import Tensor
from infinicore.nn.functional import linear, rope, causal_softmax

def self_attention(x, w_qkv, w_out, pos_ids, sin_table, cos_table, num_heads):
    """
    带RoPE的自注意力实现
    参数:
        x: 输入张量 [batch_size, seq_len, hidden_dim]
        w_qkv: QKV合并权重 [3*hidden_dim, hidden_dim]
        w_out: 输出权重 [hidden_dim, hidden_dim]
        pos_ids: 位置索引 [seq_len]
        sin_table, cos_table: RoPE查找表
        num_heads: 注意力头数
    """
    batch_size, seq_len, hidden_dim = x.shape
    head_dim = hidden_dim // num_heads

    # 计算QKV
    qkv = linear(x, w_qkv)  # [batch_size, seq_len, 3*hidden_dim]
    q, k, v = qkv.chunk(3, dim=-1)  # 各自 [batch_size, seq_len, hidden_dim]

    # 重塑为多头格式
    q = q.reshape(batch_size, seq_len, num_heads, head_dim)
    k = k.reshape(batch_size, seq_len, num_heads, head_dim)
    v = v.reshape(batch_size, seq_len, num_heads, head_dim)

    # 应用RoPE位置编码
    q = rope(q, pos_ids, sin_table, cos_table, algo=RopeAlgo.GPT_NEOX)
    k = rope(k, pos_ids, sin_table, cos_table, algo=RopeAlgo.GPT_NEOX)

    # 计算注意力分数: Q @ K^T / sqrt(d)
    attn_scores = q @ k.transpose(-2, -1) / (head_dim ** 0.5)
    # [batch_size, num_heads, seq_len, seq_len]

    # 应用因果掩码softmax
    attn_weights = causal_softmax(attn_scores)

    # 加权求和
    attn_output = attn_weights @ v  # [batch_size, seq_len, num_heads, head_dim]

    # 合并多头
    attn_output = attn_output.reshape(batch_size, seq_len, hidden_dim)

    # 输出投影
    output = linear(attn_output, w_out)

    return output
```

### 示例3: RMSNorm归一化层

```python
from infinicore.tensor import Tensor
from infinicore.nn.functional import rms_norm

class RMSNorm:
    """RMSNorm归一化层"""

    def __init__(self, hidden_dim, eps=1e-5):
        self.eps = eps
        self.weight = Tensor.ones(hidden_dim)  # 可学习参数

    def forward(self, x):
        """
        参数:
            x: [batch_size, seq_len, hidden_dim]
        返回:
            归一化后的张量
        """
        return rms_norm(
            input=x,
            normalized_shape=[x.shape[-1]],
            weight=self.weight,
            eps=self.eps
        )
```

### 示例4: LLM文本生成循环

```python
from infinicore.tensor import Tensor
from infinicore.nn.functional import linear, random_sample
import random

def generate_token(model, input_ids, temperature=0.8, topp=0.9, topk=50):
    """
    使用nucleus/top-k采样生成一个token
    参数:
        model: 语言模型
        input_ids: 当前上下文token IDs [seq_len]
        temperature: 采样温度
        topp: nucleus采样阈值
        topk: top-k采样参数
    """
    # 前向传播获取logits
    logits = model.forward(input_ids)  # [vocab_size]

    # 生成随机数
    random_val = random.random()

    # 采样下一个token
    next_token = random_sample(
        logits=logits,
        random_val=random_val,
        topp=topp,
        topk=topk,
        temperature=temperature
    )

    return next_token

def generate_text(model, prompt_ids, max_tokens=100):
    """自回归生成文本"""
    generated = prompt_ids.tolist()

    for _ in range(max_tokens):
        input_ids = Tensor(generated)

        # 采样下一个token
        next_token = generate_token(
            model,
            input_ids,
            temperature=0.8,
            topp=0.9,
            topk=50
        )

        # 添加到生成序列
        generated.append(next_token.item())

        # 可选: 检查结束token
        if next_token.item() == model.eos_token_id:
            break

    return generated
```

### 示例5: 使用in-place操作优化内存

```python
from infinicore.tensor import Tensor
from infinicore.nn.functional import silu, rms_norm, linear

def memory_efficient_ffn(x, w_gate, w_up, w_down, norm_weight):
    """
    使用in-place操作优化内存的FFN实现
    适用于大batch推理场景
    """
    # 1. 线性变换(必须创建新张量)
    gate = linear(x, w_gate)
    up = linear(x, w_up)

    # 2. 可以重用gate张量进行in-place SiLU
    silu(gate, inplace=True)  # gate被原地修改

    # 3. SwiGLU:结果写入gate张量,重用内存
    swiglu(gate, up, out=gate)

    # 4. 输出投影
    output = linear(gate, w_down)

    # 5. RMS归一化可以in-place
    rms_norm(output, [output.shape[-1]], norm_weight, out=output)

    return output
```

## 5. 实现细节

### 底层绑定架构

**Python-C++绑定模式**:
- 所有函数都是对`_infinicore`C++扩展模块的薄包装
- 通过`Tensor._underlying`访问底层C++张量对象
- 遵循PyTorch风格的API设计约定

**函数命名约定**:
- 非in-place版本: `_infinicore.function_name()` - 返回新张量
- in-place版本: `_infinicore.function_name_()` - 带下划线后缀,修改第一个参数

### 输出张量管理

**三种输出模式**:
1. **新张量模式**(`out=None`): 创建并返回新张量
2. **in-place模式**(`inplace=True`): 直接修改输入张量(仅`silu`支持)
3. **输出张量模式**(`out=tensor`): 将结果写入指定张量

**内存优化策略**:
- 在大模型推理中,通过`out`参数重用中间结果张量,减少内存分配
- `silu`的`inplace`模式可直接修改输入,节省内存
- 示例内存优化模式:
  ```python
  # 不推荐: 每步都创建新张量
  x = silu(x)
  y = swiglu(x, other)

  # 推荐: 重用张量
  silu(x, inplace=True)
  swiglu(x, other, out=x)
  ```

### 硬件加速支持

**SiLU的硬件加速路径**:
- 当满足条件时,使用`ntops.torch.silu`实现:
  - `infinicore.use_ntops == True`
  - 设备为`cuda`或`musa`
  - 未指定`out`参数
- `ntops`是针对NVIDIA和沐创GPU的优化算子库
- 提供比纯C++实现更高的性能

**设备约束**:
- `embedding`函数当前仅支持CPU设备
- 其他函数通常支持多种设备(CPU/CUDA/MUSA等)
- 设备检查在Python层进行,通过`tensor.device.type`判断

### 参数验证与错误处理

**断言验证模式**:
- 使用Python `assert`语句检查参数合法性
- 失败时抛出`AssertionError`并显示详细错误信息

**典型验证点**:
1. `embedding`: 检查不支持的参数(`padding_idx`, `max_norm`等)
2. `embedding`: 检查设备类型必须是CPU
3. `rms_norm`: 检查`normalized_shape`与`weight.shape`必须匹配

**错误传播**:
- C++绑定层的错误(如维度不匹配)会作为Python异常传播
- 未捕获的C++异常可能导致段错误(依赖C++层实现)

### 类型注解与文档

**类型注解**:
- 所有函数参数和返回值都有类型注解
- 使用`Tensor`类型(来自`infinicore.tensor`)
- 可选参数使用`| None`或默认值表示

**文档字符串**:
- 每个函数都有简短的docstring说明功能
- 使用原始字符串(`r"""`)避免转义问题
- 文档遵循Google或NumPy风格(较为简洁)

### 性能考虑

**计算复杂度**:
- `linear`: O(batch_size * in_features * out_features) - 矩阵乘法
- `causal_softmax`: O(batch_size * num_heads * seq_len^2) - 注意力机制瓶颈
- `rms_norm`: O(batch_size * seq_len * hidden_dim) - 逐元素操作
- `rope`: O(batch_size * seq_len * num_heads * head_dim) - 查找表操作
- `random_sample`: O(vocab_size * log(vocab_size)) - 需要排序/过滤

**优化技术**:
1. **算子融合**: C++层可能融合多个操作(如softmax+mask)
2. **内存布局优化**: 使用连续内存布局提升缓存命中率
3. **SIMD向量化**: C++实现使用SIMD指令加速逐元素操作
4. **查找表**: RoPE使用预计算的sin/cos表避免重复计算

### 与PyTorch的兼容性

**API兼容性**:
- 函数签名尽可能与PyTorch保持一致
- 参数命名遵循PyTorch约定(`input`, `weight`, `bias`, `out`等)
- 返回值类型和形状与PyTorch对应函数一致

**差异点**:
1. **设备限制**: `embedding`仅支持CPU,PyTorch支持多设备
2. **参数支持**: 部分高级参数未实现(如`embedding`的`sparse`)
3. **精度**: 可能使用不同的默认精度(如float32 vs float16)
4. **自动微分**: 当前实现可能不直接支持自动微分(需检查C++层)

### 依赖关系

**内部依赖**:
- `infinicore.lib._infinicore`: C++扩展模块,提供底层算子实现
- `infinicore.tensor.Tensor`: 张量类型,所有函数的输入输出类型
- `infinicore.use_ntops`: 全局配置,控制是否使用硬件加速

**外部依赖**:
- C++标准库(通过C++绑定)
- 可能的深度学习框架(如CUDA、cuDNN等,通过C++层)
- Python标准库(`typing`用于类型注解)

### 设计模式

**函数式编程风格**:
- 所有函数都是无状态的纯函数(不考虑随机性)
- 不维护内部状态,便于并行和分布式计算
- 与面向对象的模块层(`nn.Module`)形成对比

**工厂模式**:
- `RopeAlgo`枚举类充当算法工厂
- 用户选择算法变体,函数内部路由到不同实现

**策略模式**:
- `random_sample`的`topp`和`topk`参数实现不同的采样策略
- `silu`根据设备和配置选择不同的后端实现

**模板方法模式**:
- 所有函数遵循相同的模式:
  1. 参数验证
  2. 检查`out`参数
  3. 调用C++绑定(带或不带`_`后缀)
  4. 返回结果

### 并发性与线程安全

**GIL影响**:
- Python层受全局解释器锁(GIL)限制
- C++实现在释放GIL后可使用多线程
- 算子内部可能使用并行算法(如并行矩阵乘法)

**线程安全**:
- 函数式API本身无共享状态,天然线程安全
- 张量对象的并发修改需要用户同步
- C++层的线程安全性取决于实现(需检查)

### 可扩展性

**添加新函数**:
1. 在C++层实现算子(添加到`_infinicore`模块)
2. 创建对应的Python包装函数
3. 在`__init__.py`中导出新函数
4. 遵循现有的命名和类型约定

**支持新设备**:
- 在C++层添加设备特定的内核实现
- Python层可能需要添加设备类型检查
- 参考现有函数(如`embedding`的CPU检查)的模式

**优化现有函数**:
- 添加新的后端实现(如参考`silu`的`ntops`路径)
- 使用`infinicore.use_ntops`等全局配置控制
- 保持向后兼容性
