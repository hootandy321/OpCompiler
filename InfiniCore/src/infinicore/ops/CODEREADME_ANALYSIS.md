# ops 目录架构全景

## 1. 子系统职责

`ops` 目录是 InfiniCore 核心计算库的**算子实现层**，位于整个架构的执行前端。该子系统负责将高级神经网络操作（如注意力机制、激活函数、归一化等）封装为统一的 C++ 接口，并通过 `OpDispatcher` 机制实现跨硬件平台（CUDA、CPU、Kunlun 等）的算子分发与调度。

**核心价值**：
- 提供标准化的算子 API，支持张量运算、注意力计算、激活函数、归一化等 LLM 推理与训练的核心操作
- 通过设备类型（Device::Type）动态路由到相应的硬件后端实现（infiniop 或自定义 kernel）
- 为上层 Python 绑定提供底层 C++ 调用接口，实现高性能计算与易用性的平衡

**设计模式**：
- 所有算子采用一致的接口模式：`op_()`（就地写）和 `op()`（返回新张量）
- 使用 `OpDispatcher<Schema>` 实现多态分发，屏蔽不同硬件后端的实现差异
- 支持图模式（Graph Mode）记录，用于计算图优化与执行

---

## 2. 模块导航

本目录包含 **19 个算子子模块**，按功能域分类如下：

### 2.1 基础数学运算
* **add**：逐元素加法算子
    * *功能*：实现张量加法操作的封装与调度
    * *职责*：通过 OpDispatcher 分发到硬件后端的加法实现
    * *文档状态*：✓ 代码存在，文档缺失

* **mul**：逐元素乘法算子
    * *功能*：实现张量乘法操作的封装与调度
    * *职责*：支持元素级张量乘法，路由到对应硬件实现
    * *文档状态*：✓ 代码存在，文档缺失

* **ones**：全 1 张量生成算子
    * *功能*：生成指定形状的全 1 张量
    * *职责*：提供张量初始化能力，用于模型权重填充或掩码构建
    * *文档状态*：✓ 代码存在，文档缺失

### 2.2 线性代数核心
* **gemm**：通用矩阵乘法（GEMM）核心
    * *功能*：实现 C = alpha * A * B + beta * C 的通用矩阵乘法
    * *职责*：作为线性层、注意力计算等操作的基础计算引擎
    * *文档状态*：✓ 代码存在，文档缺失

* **matmul**：矩阵乘法包装器
    * *功能*：提供简化的矩阵乘法接口（alpha=1.0, beta=0.0）
    * *职责*：对 gemm 进行轻量封装，满足常见矩阵乘法场景
    * *文档状态*：✓ 代码存在，文档缺失

* **linear**：全连接层算子
    * *功能*：实现线性变换 output = input @ weight^T + bias
    * *职责*：组合 gemm 与 rearrange，支持 bias 添加，是 MLP 的核心组件
    * *文档状态*：✓ 代码存在，文档缺失

### 2.3 注意力机制
* **attention**：基础注意力算子
    * *功能*：实现标准注意力机制（带 KV 缓存支持）
    * *职责*：计算 Q、K、V 的加权聚合，支持自回归生成的位置索引
    * *参数*：q/k/v 张量、kv_cache、当前位置 pos
    * *文档状态*：✓ 代码存在，文档缺失

* **paged_attention**：分页注意力算子
    * *功能*：实现 vLLM 风格的 PagedAttention 机制
    * *职责*：支持动态批处理的 KV 块表管理，通过 block_tables 映射非连续 KV 缓存
    * *参数*：q/kv_cache、block_tables、kv_lens、alibi_slopes（可选）、scale
    * *文档状态*：✓ 代码存在，文档缺失

* **paged_attention_prefill**：预填充阶段分页注意力
    * *功能*：优化 Prefill 阶段的 PagedAttention 实现
    * *职责*：处理长序列的高效预填充，与 decoding 阶段的 paged_attention 配合
    * *文档状态*：✓ 代码存在，文档缺失

* **causal_softmax**：因果掩码 softmax
    * *功能*：实现带因果掩码的 softmax 归一化
    * *职责*：为注意力机制提供 masked softmax，确保自回归属性
    * *文档状态*：✓ 代码存在，文档缺失

### 2.4 激活与归一化
* **silu**：SiLU 激活函数（Swish）
    * *功能*：实现 SiLU(x) = x * sigmoid(x)
    * *职责*：提供平滑的非线性激活，广泛用于现代 LLM（如 LLaMA）
    * *文档状态*：✓ 代码存在，文档缺失

* **swiglu**：SwiGLU 激活函数
    * *功能*：实现 SwiGLU(a, b) = SiLU(a) ⊗ b
    * *职责*：GLU 变体激活函数，用于 Transformer FFN 层（如 LLaMA 架构）
    * *文档状态*：✓ 代码存在，文档缺失

* **rms_norm**：均方根层归一化
    * *功能*：实现 RMSNorm 归一化操作（不含 bias、无中心化）
    * *职责*：提供 LLaMA 等 LLM 架构的标准归一化层，支持 epsilon 参数
    * *文档状态*：✓ 代码存在，文档缺失

* **add_rms_norm**：融合加法与 RMSNorm
    * *功能*：实现 x + weight * RMSNorm(x) 的融合算子
    * *职责*：优化残差连接与归一化的组合操作，减少内核启动开销
    * *文档状态*：✓ 代码存在，文档缺失

### 2.5 位置编码与变换
* **rope**：旋转位置编码（RoPE）
    * *功能*：应用旋转位置编码到查询或键张量
    * *职责*：实现 RoPE 位置编码机制，支持不同 Algo 策略（如 HF/Llama 变体）
    * *参数*：输入张量 x、位置 pos、sin/cos 查找表、算法模式
    * *文档状态*：✓ 代码存在，文档缺失

* **rearrange**：张量重排与广播
    * *功能*：实现张量视图变换、广播、切片等操作
    * *职责*：提供张量形状操作的基础能力，支持 as_strided 高级索引
    * *文档状态*：✓ 代码存在，文档缺失

### 2.6 缓存与采样
* **paged_caching**：分页 KV 缓存管理
    * *功能*：实现 KV 缓存的分页存储与检索逻辑
    * *职责*：支持动态 KV 块分配，配合 paged_attention 使用
    * *文档状态*：✓ 代码存在，文档缺失

* **random_sample**：核采样与 Top-K 采样
    * *功能*：实现 next token 采样逻辑（支持 top-p、top-k、temperature）
    * *职责*：从 logits 分布中采样下一个 token，支持 nucleus sampling 与 top-k sampling
    * *参数*：logits、随机值、topp 阈值、topk 数量、温度参数
    * *文档状态*：✓ 代码存在，文档缺失

* **embedding**：词嵌入查表算子
    * *功能*：实现索引到词向量的查表操作
    * *职责*：将 token ID 映射为稠密向量，支持 CPU/GPU 内存复制
    * *参数*：input（索引张量，I32/I64）、weight（词表矩阵）
    * *文档状态*：✓ 代码存在，文档缺失

---

## 3. 架构逻辑图解

### 3.1 算子层次与依赖关系

```
┌─────────────────────────────────────────────────────────────┐
│                     Python 绑定层                            │
│  (通过 pybind11 暴露 ops 接口到 Python)                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     ops 目录（本层）                         │
│  ┌──────────────────┬──────────────────┬────────────────┐  │
│  │  线性代数算子    │  注意力算子      │  激活归一化    │  │
│  │  - gemm          │  - attention     │  - silu        │  │
│  │  - matmul        │  - paged_attn    │  - swiglu      │  │
│  │  - linear        │  - causal_softmax│  - rms_norm    │  │
│  │                  │  - paged_cache   │  - add_rms_norm│  │
│  └──────────────────┴──────────────────┴────────────────┘  │
│  ┌──────────────────┬──────────────────┬────────────────┐  │
│  │  基础运算        │  位置编码        │  其他算子      │  │
│  │  - add/mul/ones  │  - rope          │  - rearrange   │  │
│  │                  │                  │  - random_sample│  │
│  │                  │                  │  - embedding   │  │
│  └──────────────────┴──────────────────┴────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                OpDispatcher 分发层                           │
│  根据 device.getType() 路由到对应硬件实现                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
      ┌─────────────┬─────────────┬─────────────┐
      │   CUDA     │    CPU      │   Kunlun    │  ...
      │  Backend   │   Backend   │   Backend   │
      └─────────────┴─────────────┴─────────────┘
```

### 3.2 数据流与执行流程

#### 典型 LLM 推理中的算子调用链

1. **输入处理阶段**
   ```
   token_ids → embedding() → inputs_embeds
   ```
   - `embedding` 将离散 token ID 映射为连续词向量

2. **Transformer 层前向传播**
   ```
   inputs_embeds → rms_norm() → rope()
                    ↓
                 attention()
                    ↓
                 add_rms_norm()
                    ↓
                 swiglu() → linear()
                    ↓
                 add (residual)
   ```
   - `rms_norm`：对输入进行归一化
   - `rope`：注入位置信息到 Q/K
   - `attention`：计算上下文表示（可能调用 `paged_attention`）
   - `add_rms_norm`：残差连接 + 归一化
   - `swiglu` + `linear`：FFN 层计算

3. **输出生成阶段**
   ```
   logits → random_sample() → next_token_id
   ```
   - `random_sample`：从 logits 分布采样下一个 token

#### 关键算子协作案例

**案例 1：线性层的组合实现**
```
linear() = rearrange(bias) + gemm(input, weight^T)
```
- 如果有 bias，先用 `rearrange` 将 bias 广播到输出形状（beta=1.0）
- 然后调用 `gemm` 进行矩阵乘法（alpha=1.0）
- 利用算子融合减少内存拷贝

**案例 2：PagedAttention 的完整流程**
```
1. paged_caching: 分配 KV 块到物理内存
2. paged_attention: 根据 block_tables 查找并计算注意力
3. paged_attention_prefill: 预填充阶段优化
```
- 三个算子协同实现 vLLM 风格的高效推理
- block_tables 作为索引，映射逻辑序列到物理 KV 块

**案例 3：位置编码的注入**
```
Q = rope(Q, positions, sin_table, cos_table, algo)
K = rope(K, positions, sin_table, cos_table, algo)
```
- 在注意力计算前对 Q/K 应用旋转位置编码
- 支持不同的 RoPE 变体（如 Llama 的 HuggingFace 兼容模式）

### 3.3 硬件后端分发机制

每个算子通过 `OpDispatcher<Schema>` 实现跨硬件分发：

```cpp
// 伪代码示例
void SwiGLU::execute(Tensor c, Tensor a, Tensor b) {
    auto device_type = c->device().getType();
    auto func = dispatcher().lookup(device_type);  // 查找后端实现
    func(c, a, b);  // 调用对应硬件的 kernel
}
```

- **CUDA Backend**：调用 *_infiniop.cc 中的实现（通常调用 InfiniOp 库）
- **CPU Backend**：可调用 CPU 优化实现或回退到朴素实现
- **其他硬件**：Kunlun、Ascend 等通过注册对应的 dispatcher 函数支持

### 3.4 算子融合与优化策略

1. **算子融合**：
   - `add_rms_norm`：融合残差连接与归一化，减少内核启动与内存访问
   - `linear`：融合 bias 广播与矩阵乘法，单次 kernel 完成

2. **缓存优化**：
   - `paged_attention` + `paged_caching`：避免 KV 缓存连续分配浪费，支持动态批处理
   - `attention` 支持 KV cache 参数，自回归生成时复用历史键值对

3. **内存布局优化**：
   - `rearrange` 提供零拷贝的视图变换，支持 as_strided 等高级操作
   - 大部分算子支持就地操作（`_` 后缀版本），减少内存分配

---

## 4. 关键技术特性

### 4.1 统一的算子接口模式
所有算子提供两种调用形式：
- `op(...)`：分配输出张量并返回（函数式风格）
- `op_(output, ...)`：就地写入到预分配张量（性能优化）

### 4.2 设备一致性检查
使用 `INFINICORE_ASSERT_TENSORS_SAME_DEVICE` 宏确保所有输入张量位于同一设备，避免隐式数据迁移。

### 4.3 错误处理
对于不支持的后端，抛出 `std::runtime_error` 异常，明确指出缺失的设备类型实现。

### 4.4 图模式支持
部分算子（如 `gemm`）使用 `INFINICORE_GRAPH_OP_RECORD_OR_RUN` 宏，支持计算图记录与优化执行。

---

## 5. 文档缺失说明

**重要提示**：本目录下所有 **19 个子模块均缺少 CODEREADME.md 文档**。以上分析完全基于源代码文件（`*.cc` 和 `*_infiniop.cc`）的静态分析。

**建议后续行动**：
1. 为每个算子子目录创建详细的 CODEREADME.md
2. 补充算子的数学定义、性能特征、使用示例
3. 说明各硬件后端的实现差异与优化策略
