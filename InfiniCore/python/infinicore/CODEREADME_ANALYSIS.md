# Python infinicore 包架构全景

## 1. 子系统职责

本目录 `infinicore` 是 InfiniCore 深度学习推理框架的 Python 前端入口，提供了从 Python API 到底层 C++/CUDA 实现的完整绑定层。该子系统采用**分层架构设计**，实现了从高层神经网络抽象到低层张量计算的全栈覆盖：

* **神经网络层（nn）**：提供与 PyTorch 兼容的高层神经网络模块，包括有状态的组件封装（modules）和无状态的函数式接口（functional），构建用户可直接使用的推理模型。
* **算子层（ops）**：提供底层计算操作的 Python 绑定，包括基础算术运算、张量形状操作、标准注意力和分页注意力（Paged Attention）等高性能算子。
* **核心类型层（根目录）**：提供框架的核心数据结构和管理功能，包括张量（Tensor）、设备（device）、数据类型（dtype）、上下文管理（context）和事件同步（device_event）等基础设施。

这种设计使用户能够使用熟悉的 PyTorch 风格 API 定义神经网络模型，同时底层计算通过 InfiniCore 的高性能 C++ 内核执行，支持多种硬件后端（CPU、CUDA、MUSA 等）。该子系统在 InfiniCore 整体架构中扮演**Python 用户界面层**的角色，是框架与用户交互的桥梁。

## 2. 模块导航 (Module Navigation)

### **📂 nn** - 神经网络模块层
* **功能**：提供 PyTorch 兼容的神经网络层抽象，采用双层架构设计：函数式层（functional）提供无状态的纯函数接口，直接调用底层 C++ 算子；模块层（modules）提供面向对象的有状态组件封装，实现参数管理、状态序列化和模块组合机制。
* **职责**：
  * **functional 子模块**：实现核心神经网络原语，包括注意力机制（causal_softmax）、线性变换（linear）、归一化（rms_norm）、激活函数（silu, swiglu）、位置编码（rope）、嵌入查找（embedding）、随机采样（random_sample）。
  * **modules 子模块**：提供可复用的神经网络层抽象，包括基类 InfiniCoreModule（参数注册、state_dict 序列化）、容器 ModuleList、线性层 Linear、归一化层 RMSNorm、旋转位置编码 RoPE、嵌入层 Embedding。
  * **parameter.py**：定义 InfiniCoreParameter 类型，作为 Tensor 的包装器用于标识可训练参数。

### **📂 ops** - 底层算子绑定层
* **功能**：提供从 Python 到底层 C++/CUDA 实现的算子绑定，实现神经网络推理和训练中的核心计算操作，特别优化了大语言模型（LLM）中的注意力机制和归一化操作。
* **职责**：
  * **基础算术运算**：add（加法）、mul（逐元素乘法）、matmul（矩阵乘法，支持 alpha 缩放因子）。
  * **融合算子**：add_rms_norm（融合加法与 RMS 归一化，减少内存访问和中间结果存储）。
  * **张量形状操作**：narrow（沿指定维度切片）、squeeze（移除大小为 1 的维度）、unsqueeze（插入维度）、rearrange（张量重排）。
  * **注意力算子**：attention（标准注意力机制，支持 KV cache 和位置编码）。
  * **分页注意力算子**：paged_attention_prefill（预填充阶段，处理新 token 的初始注意力计算）、paged_attention（解码阶段，处理自回归生成的增量注意力）、paged_caching（将 KV 对写入分页 KV cache）。

### **📄 tensor.py** - 张量核心类型定义
* **功能**：定义 Tensor 类，作为 InfiniCore 框架的核心数据结构，包装底层 C++ 张量对象，提供张量构造、视图操作、类型转换、设备迁移等接口。
* **职责**：维护底层 C++ 张量对象（_underlying）的生命周期，提供惰性属性访问（shape、dtype、device），支持张量视图操作（view、permute、squeeze、contiguous）、设备间数据传输、与 NumPy/PyTorch 的互操作。

### **📄 device.py** - 设备管理
* **功能**：定义 device 类，表示计算设备（CPU、CUDA、MUSA 等），提供设备类型和设备索引的抽象，支持设备选择和切换。
* **职责**：封装设备类型（type）和设备索引（index），提供与底层 C++ 设备对象的双向转换（_to_infinicore_device/_from_infinicore_device），支持设备字符串解析（如 "cuda:0"）。

### **📄 dtype.py** - 数据类型定义
* **功能**：定义 dtype 类和所有支持的数据类型（float32、float16、bfloat16、int32、int64、bool 等），提供数据类型与底层 C++ 类型的映射。
* **职责**：提供数据类型构造函数和类型转换常量，支持 NumPy/PyTorch 数据类型与 InfiniCore 数据类型的互操作。

### **📄 context.py** - 上下文管理
* **功能**：提供设备上下文、流管理和图录制功能，支持多设备环境下的设备切换、同步和图优化。
* **职责**：
  * 设备管理：get_device（获取当前设备）、set_device（设置当前设备）、get_device_count（获取设备数量）。
  * 同步操作：sync_device（同步设备）、sync_stream（同步流）、get_stream（获取当前流）。
  * 图录制：start_graph_recording（开始图录制）、stop_graph_recording（停止图录制并返回图对象）、is_graph_recording（检查是否在录制）。

### **📄 device_event.py** - 设备事件同步
* **功能**：定义 DeviceEvent 类，提供设备事件记录和同步机制，支持 CUDA 事件风格的异步操作同步。
* **职责**：封装底层 C++ 事件对象，提供事件记录（record）、查询（query）、同步（sync）和等待（wait）接口。

### **📄 graph.py** - 计算图抽象
* **功能**：定义 Graph 类，表示录制的计算图，支持图优化和重放。
* **职责**：封装底层 C++ 计算图对象，提供图操作接口（如执行、保存、加载）。

### **📄 utils.py** - 工具函数
* **功能**：提供数据类型转换工具函数，支持 InfiniCore、NumPy、PyTorch 之间的数据类型映射。
* **职责**：实现 infinicore_to_numpy_dtype、numpy_to_infinicore_dtype、to_infinicore_dtype 等类型转换函数。

### **📄 __init__.py** - 包入口与公共 API
* **功能**：定义包的公共 API，导出所有核心类、函数和子模块，配置可选的硬件加速库（ntops）。
* **职责**：
  * 导出核心类型：Tensor、device、dtype、DeviceEvent。
  * 导出上下文函数：get_device、set_device、sync_device、start_graph_recording 等。
  * 导出数据类型：float32、float16、bfloat16、int32、int64、bool 等。
  * 导出算子函数：add、mul、matmul、attention、paged_attention 系列等。
  * 导出张量构造函数：empty、zeros、ones、from_numpy、from_torch 等。
  * 导出子模块：nn（神经网络）、context（上下文管理）。
  * 配置 use_ntops 全局变量，根据 ntops 库是否可用启用硬件加速路径。

## 3. 架构逻辑图解

### 3.1 分层架构关系

```
┌─────────────────────────────────────────────────────────────┐
│                     用户应用层                                │
│  (用户使用 nn.modules 的 Linear, RMSNorm, Embedding 等定义模型) │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   nn.modules 模块封装层                       │
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
│                  nn.functional 函数式层                       │
│  无状态纯函数接口                                              │
│    ├── 线性变换: linear(input, weight, bias)                  │
│    ├── 归一化: rms_norm(input, normalized_shape, weight)       │
│    ├── 激活函数: silu(input), swiglu(gate, value)            │
│    ├── 位置编码: rope(x, pos_ids, sin_table, cos_table)      │
│    ├── 注意力: causal_softmax(input)                          │
│    ├── 嵌入: embedding(input, weight)                        │
│    └── 采样: random_sample(logits, topp, topk, temperature)   │
└──────────────────────────┬──────────────────────────────────┘
                           │ 调用 / 绑定
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     ops 算子绑定层                            │
│  Python-C++ 绑定接口                                          │
│    ├── 基础算术: add, mul, matmul                            │
│    ├── 融合算子: add_rms_norm                                │
│    ├── 形状操作: narrow, squeeze, unsqueeze, rearrange      │
│    ├── 注意力: attention                                     │
│    └── 分页注意力: paged_attention, paged_attention_prefill,  │
│                  paged_caching                                │
└──────────────────────────┬──────────────────────────────────┘
                           │ 绑定
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Tensor/device/dtype 核心类型层                    │
│  Tensor 类 (张量封装)                                         │
│    ├── 底层 C++ 对象访问 (_underlying)                        │
│    ├── 惰性属性 (shape, dtype, device)                        │
│    ├── 视图操作 (view, permute, squeeze)                      │
│    └── 互操作 (from_numpy, from_torch, to)                    │
│                                                               │
│  device 类 (设备抽象)                                         │
│    ├── 设备类型和索引 (type, index)                          │
│    └── C++ 设备对象转换                                       │
│                                                               │
│  dtype 类 (数据类型)                                          │
│    ├── 类型常量 (float32, int64, etc.)                       │
│    └── 类型转换映射                                           │
└──────────────────────────┬──────────────────────────────────┘
                           │ 绑定
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               C++ 后端实现 (_infinicore)                       │
│  高性能计算内核 (支持多硬件后端: CPU, CUDA, MUSA 等)            │
│  分页注意力算法、算子融合、内存管理、图优化                      │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 数据流向

**典型 LLM 推理流程**：

```
输入 Token IDs (Tensor)
    │
    ▼
┌──────────────────────────────────────┐
│ 1. 嵌入层 (nn.modules.Embedding)      │
│    ↓ 调用 functional.embedding()      │
│    ↓ 查表操作得到嵌入向量              │
└──────────────────────────────────────┘
    │ 输出: (batch, seq_len, hidden_dim)
    ▼
┌──────────────────────────────────────┐
│ 2. Transformer Block (多层堆叠)       │
│    ├── QKV 投影 (nn.modules.Linear)   │
│    │   ↓ 调用 functional.linear()     │
│    │   ↓ ops.matmul() (矩阵乘法)      │
│    │                                 │
│    ├── RoPE 位置编码 (nn.modules.RoPE)│
│    │   ↓ 调用 functional.rope()       │
│    │   ↓ 使用预计算的 sin/cos 表      │
│    │                                 │
│    ├── 自注意力计算                   │
│    │   ↓ causal_softmax()            │
│    │   ↓ 或使用 ops.attention()       │
│    │   ↓ 或使用 ops.paged_attention() │
│    │      (分页注意力，高效 KV cache) │
│    │                                 │
│    ├── FFN (SwiGLU 激活)              │
│    │   ↓ functional.swiglu()         │
│    │   ↓ 内部调用 silu() + 逐元素乘   │
│    │                                 │
│    └── RMS 归一化 (nn.modules.RMSNorm)│
│        ↓ 调用 functional.rms_norm()   │
│        ↓ 或使用 ops.add_rms_norm()    │
│          (融合算子，残差连接优化)       │
└──────────────────────────────────────┘
    │ 输出: (batch, seq_len, hidden_dim)
    ▼
┌──────────────────────────────────────┐
│ 3. 输出投影 (nn.modules.Linear)       │
│    ↓ 投影到词表空间                   │
└──────────────────────────────────────┘
    │ 输出: (batch, seq_len, vocab_size)
    ▼
┌──────────────────────────────────────┐
│ 4. 采样 (生成)                        │
│    ↓ functional.random_sample()      │
│    ↓ top-p/top-k 采样                 │
└──────────────────────────────────────┘
    │ 采样 Token ID
    ▼
输出下一个 Token
```

### 3.3 分页注意力（Paged Attention）优化流程

```
┌─────────────────────────────────────────────────────────────┐
│              阶段 1: 预填充（Prefill）                        │
│  处理初始 Prompt 的所有 Token                                 │
└─────────────────────────────────────────────────────────────┘
    │
    ├── 输入: Prompts Q (形状: [total_tokens, num_heads, head_dim])
    │
    ├── 调用: ops.paged_attention_prefill(
    │         q, k_cache, v_cache,
    │         block_tables,      # 块表: 逻辑页 → 物理页映射
    │         history_lens,      # 每个序列的历史长度
    │         cu_seqlens_q,      # 累积序列长度（CUDA 内核使用）
    │         scale              # 注意力缩放因子
    │       )
    │
    └── 输出: 预填充阶段的注意力输出

┌─────────────────────────────────────────────────────────────┐
│              阶段 2: 解码（Decode）循环                       │
│  自回归生成，每步生成一个新 Token                              │
└─────────────────────────────────────────────────────────────┘
    │
    │ 循环 (每个生成步骤):
    │
    ├── 2.1 计算 new token 的 QKV
    │     │
    │     └── Q, K, V = model.forward(last_token)
    │
    ├── 2.2 写入 KV Cache (分页缓存)
    │     │
    │     ├── slot_mapping = compute_slot_mapping(...)  # 计算物理位置
    │     │
    │     └── ops.paged_caching(
    │            k_cache, v_cache,  # 缓存张量（就地修改）
    │            k, v,              # 新的键值对
    │            slot_mapping       # 槽位映射
    │          )
    │
    ├── 2.3 计算注意力（与历史 Cache）
    │     │
    │     └── ops.paged_attention(
    │            q,                # new token 的查询
    │            k_cache, v_cache, # 完整的 KV cache
    │            block_tables,     # 块表
    │            cache_lens,       # 每个序列的缓存长度
    │            scale             # 缩放因子
    │          )
    │
    └── 2.4 输出 → 下一个 token
```

### 3.4 硬件加速路径

```
用户调用 silu(input) 或 SwiGLU 激活
    │
    ▼
检查全局配置 infinicore.use_ntops
    │
    ├── use_ntops == True
    │   ├── 检查设备类型 (device.type in ["cuda", "musa"]?)
    │   ├── 检查是否指定 out 参数
    │   │
    │   ├── 满足条件 → 使用 ntops.torch.silu() (硬件优化路径)
    │   │   ├── ntops 是针对 NVIDIA/MUSA GPU 的优化算子库
    │   │   └── 提供比纯 C++ 实现更高的性能
    │   │
    │   └── 不满足 → 降级到通用路径
    │
    └── use_ntops == False
        └── 使用 _infinicore.silu() (通用 C++ 路径)
            │
            ├── inplace == True → _infinicore.silu_() (原地修改)
            └── inplace == False → _infinicore.silu() (新张量)
```

### 3.5 上下文管理与图录制流程

```
┌─────────────────────────────────────────────────────────────┐
│              正常执行模式                                     │
└─────────────────────────────────────────────────────────────┘
    │
    ├── 设置设备: context.set_device(device("cuda:0"))
    │   ↓ _infinicore.set_device(device._underlying)
    │
    ├── 执行操作: result = add(a, b)
    │   ↓ 立即执行 C++ 内核
    │
    └── 同步: context.sync_device()
        ↓ 等待设备完成所有操作

┌─────────────────────────────────────────────────────────────┐
│              图录制模式（优化执行）                            │
└─────────────────────────────────────────────────────────────┘
    │
    ├── 1. 开始录制: context.start_graph_recording()
    │   ↓ _infinicore.start_graph_recording()
    │
    ├── 2. 记录操作（不立即执行）
    │   ├── result1 = add(a, b)        # 记录到图
    │   ├── result2 = mul(result1, c)  # 记录到图
    │   └── result3 = rms_norm(...)    # 记录到图
    │
    ├── 3. 停止录制: graph = context.stop_graph_recording()
    │   ↓ _infinicore.stop_graph_recording()
    │   ↓ 返回 Graph 对象
    │
    └── 4. 图优化与重放
        ├── 图优化: 算子融合、内存分配优化
        └── 图执行: 一次性执行优化后的图
```

### 3.6 张量类型系统与互操作

```
┌─────────────────────────────────────────────────────────────┐
│              Tensor 类型层次结构                              │
└─────────────────────────────────────────────────────────────┘
    │
    ├── Python Tensor 类 (tensor.py)
    │   ├── _underlying: _infinicore.Tensor (底层 C++ 对象)
    │   ├── _torch_ref: torch.Tensor (可选的 PyTorch 引用)
    │   ├── 惰性属性: shape, dtype, device (首次访问时缓存)
    │   └── 方法: view(), permute(), to(), copy_()
    │
    ├── InfiniCoreParameter (nn/parameter.py)
    │   └── Tensor 的包装器，标识可训练参数
    │
    └── 底层 C++ Tensor (_infinicore.Tensor)
        ├── 实际数据存储
        ├── 形状、步幅、数据类型信息
        └── 设备亲和性

┌─────────────────────────────────────────────────────────────┐
│              数据类型转换映射                                  │
└─────────────────────────────────────────────────────────────┘
    │
    ├── NumPy → InfiniCore
    │   ├── utils.numpy_to_infinicore_dtype[np.float32] → dtype.float32
    │   └── Tensor.from_numpy(array) → Tensor
    │
    ├── PyTorch → InfiniCore
    │   ├── utils.torch_to_infinicore_dtype[torch.float32] → dtype.float32
    │   └── Tensor.from_torch(tensor) → Tensor
    │
    └── InfiniCore → NumPy/PyTorch
        └── Tensor.to(...) (支持跨框架传输)
```

### 3.7 内存优化策略

**融合算子优化（add_rms_norm 案例）**：

```
传统实现（两次内核启动）:
    │
    ├── 步骤 1: sum = add(a, b)         # 内核 1: 加法
    │           ↓ 中间结果写入内存
    │
    └── 步骤 2: output = rms_norm(sum)  # 内核 2: RMS 归一化
                ↓ 再次读取 sum
    │
    总计: 2 次内核启动, 2 次内存写入, 1 次内存读取

融合算子实现（单次内核启动）:
    │
    └── output, residual = add_rms_norm(a, b, weight)
        │
        ├── 单次内核启动完成: add + rms_norm
        ├── 减少中间结果内存分配
        ├── 减少 PCIe/内存总线传输
        └── 返回: (归一化结果, 残差连接输入)
```

**In-Place 操作优化**：

```
内存优化的 FFN 前向传播:
    │
    ├── 1. gate = linear(x, w_gate)  # 新张量
    ├── 2. up = linear(x, w_up)      # 新张量
    │
    ├── 3. silu(gate, inplace=True)  # 重用 gate 张量，原地修改
    │
    ├── 4. swiglu(gate, up, out=gate)  # 结果写入 gate，重用内存
    │
    ├── 5. output = linear(gate, w_down)  # 新张量
    │
    └── 6. rms_norm(output, ..., out=output)  # 原地归一化

总计: 仅分配 3 个张量（gate, up, output），而非 6 个
```

## 4. 设计原则与最佳实践

### 4.1 PyTorch 兼容性

* **API 一致性**：函数签名、参数命名、返回值类型与 PyTorch 对齐（如 `Linear(in_features, out_features, bias=False)`）。
* **状态字典格式**：使用点分隔的层次化键名（如 `layers.0.weight`），与 PyTorch 模型互操作。
* **模块组合模式**：支持嵌套子模块、参数共享、ModuleList 容器等 PyTorch 惯用法。
* **类型注解**：使用 Python 类型注解增强代码可读性和 IDE 支持。

### 4.2 性能优化策略

* **算子融合**：C++ 层融合多个操作（如 add_rms_norm 融合加法与归一化，causal_softmax 融合 softmax 与因果掩码）。
* **In-Place 操作**：提供 `inplace=True` 和 `out=` 参数支持内存重用，减少大模型推理的内存占用。
* **分页注意力**：通过块表（block table）实现 KV cache 的动态分配，解决显存碎片问题，支持高效的变长序列批处理。
* **硬件加速**：通过 `infinicore.use_ntops` 配置，选择硬件优化算子库（如 NVIDIA/MUSA GPU 的 ntops）。
* **预计算策略**：RoPE 在初始化时预计算 sin/cos 查找表，避免前向传播重复计算。
* **图录制与优化**：支持图模式录制，优化算子执行顺序和内存分配。

### 4.3 多硬件后端支持

* **设备抽象**：device 类提供统一的设备接口，支持 CPU、CUDA、MUSA、NPU 等多种设备类型。
* **设备上下文管理**：context 模块提供设备切换、同步功能，支持多设备环境下的计算。
* **设备约束检查**：部分算子在 Python 层进行设备检查（如 embedding 当前仅支持 CPU），避免无效的内核调用。

### 4.4 模块化设计

* **职责分离**：
  * **functional 层**：专注于计算逻辑，保持无状态、可组合、可测试。
  * **modules 层**：专注于状态管理，负责参数注册、模块组合、序列化。
  * **ops 层**：提供底层算子绑定，直接映射到 C++ 实现。
* **依赖注入**：通过全局配置（如 `infinicore.use_ntops`）控制行为，避免硬编码。
* **薄包装模式**：Python 层仅做参数转换和分发，核心逻辑在 C++ 层，最小化 Python 开销。

### 4.5 扩展性指南

**添加新神经网络层（modules 层）**：

1. 继承 `InfiniCoreModule` 基类。
2. 在 `__init__` 中通过 `self.param_name = Parameter(...)` 注册参数。
3. 通过 `register_buffer()` 注册非参数张量（如预计算表）。
4. 实现 `forward()` 方法，调用 `functional` 层的函数完成计算。
5. 实现 `extra_repr()` 返回模块关键配置信息。
6. 在 `modules/__init__.py` 中导出新模块。

**添加新算子（ops 层）**：

1. 在 C++ 层实现算子（添加到 `_infinicore` 扩展模块）。
2. 在 `ops/` 目录创建对应 Python 文件，编写包装函数。
3. 遵循命名约定：非 in-place 版本调用 `function()`，in-place 版本调用 `function_()`。
4. 支持可选 `out` 参数用于内存优化。
5. 在 `__init__.py` 中导出函数。

## 5. 典型应用场景

### 场景 1：构建 Transformer 语言模型

```python
import infinicore
from infinicore.nn.modules import Module, Linear, RMSNorm, Embedding, RoPE, ModuleList
from infinicore.nn.functional import RopeAlgo

class LlamaLikeModel(Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, max_seq_len):
        super().__init__()
        self.embedding = Embedding(vocab_size, hidden_dim)
        self.layers = ModuleList([
            TransformerBlock(hidden_dim, num_heads, max_seq_len)
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

        return hidden

# 保存/加载模型权重
state_dict = model.state_dict()
model.load_state_dict(state_dict)
```

### 场景 2：使用分页注意力的高效批量推理

```python
from infinicore.ops import paged_attention_prefill, paged_attention, paged_caching

def batch_inference_with_paged_attention(
    model, prompts_batch, k_cache, v_cache, block_tables
):
    """使用分页注意力进行批量推理，支持变长序列"""

    # 1. 预填充阶段：处理所有 prompt
    prefill_output = paged_attention_prefill(
        q=prompts_q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        history_lens=history_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=1.0 / sqrt(head_dim)
    )

    # 2. 解码阶段：自回归生成
    for step in range(max_steps):
        # 计算 new token 的 QKV
        q, k, v = model.forward(last_token)

        # 写入 KV cache
        slot_mapping = compute_slot_mapping(...)
        paged_caching(k_cache, v_cache, k, v, slot_mapping)

        # 计算注意力（与历史 cache）
        token_output = paged_attention(
            q, k_cache, v_cache, block_tables, cache_lens,
            scale=1.0 / sqrt(head_dim)
        )

        last_token = token_output

    return output
```

### 场景 3：使用融合算子优化内存

```python
from infinicore.ops import add_rms_norm

def transformer_layer_with_fusion(hidden, input, weight):
    """使用融合算子优化 Transformer 层"""

    # 传统实现（两次内核调用）:
    # residual = add(hidden, input)
    # normalized = rms_norm(residual, weight)

    # 融合实现（单次内核调用）:
    normalized, residual = add_rms_norm(
        a=hidden,
        b=input,
        weight=weight,
        epsilon=1e-5
    )

    # residual 可直接用于下一层的残差连接
    output = attention(normalized) + mlp(normalized) + residual

    return output
```

## 6. 依赖关系图

```
infinicore (根包)
    │
    ├── 内部依赖
    │   ├── infinicore.lib._infinicore (C++ 扩展模块，核心绑定)
    │   ├── infinicore.nn (神经网络子模块)
    │   │   ├── infinicore.nn.functional (函数式 API)
    │   │   ├── infinicore.nn.modules (模块封装)
    │   │   └── infinicore.nn.parameter (参数类型)
    │   ├── infinicore.ops (算子绑定)
    │   ├── infinicore.tensor (Tensor 类)
    │   ├── infinicore.device (device 类)
    │   ├── infinicore.dtype (dtype 类)
    │   ├── infinicore.context (上下文管理)
    │   ├── infinicore.device_event (事件同步)
    │   ├── infinicore.graph (计算图)
    │   └── infinicore.utils (工具函数)
    │
    ├── Python 标准库
    │   ├── collections.OrderedDict
    │   ├── typing (类型注解)
    │   ├── itertools.chain
    │   ├── numbers.Integral
    │   ├── contextlib
    │   └── ctypes
    │
    └── 外部依赖（条件依赖）
        ├── numpy (数据类型转换、张量互操作)
        ├── torch (可选，PyTorch 张量互操作)
        └── ntops (可选，硬件加速库，用于 CUDA/MUSA GPU)
```

## 7. 性能特征

* **计算复杂度**：
  * Linear: O(batch_size * in_features * out_features)
  * causal_softmax: O(batch_size * num_heads * seq_len^2) - 注意力瓶颈
  * rms_norm: O(batch_size * seq_len * hidden_dim)
  * rope: O(batch_size * seq_len * num_heads * head_dim) - 查找表操作
  * paged_attention: O(batch_size * num_heads * seq_len * cache_len) - 优化的注意力计算

* **内存占用**：
  * 模块参数：O(total_parameters) - 由模型大小决定
  * RoPE 查找表：O(max_position_embeddings * head_dim) - 固定开销
  * 分页 KV Cache：动态分配，基于块表管理，减少内存碎片
  * 前向传播中间结果：通过 in-place 操作和融合算子优化内存占用

* **优化级别**：
  * C++ 内核：使用 SIMD、并行算法、算子融合
  * 硬件加速：针对 CUDA/MUSA 的优化内核（ntops）
  * 分页注意力：高效的 KV cache 管理，支持变长序列批处理
  * Python 层：最小化开销，直接转发到 C++ 层
  * 图优化：支持图录制模式，优化算子执行顺序

## 8. 与 InfiniCore 整体架构的关系

```
InfiniCore 整体架构
    │
    ├── 上层应用
    │   ├── InfiniLM (大语言模型)
    │   ├── InfiniTrain (训练框架)
    │   └── infiniStudio (可视化工具)
    │
    ├── Python 前端层 (当前目录)
    │   ├── infinicore.nn (神经网络 API)
    │   ├── infinicore.ops (算子绑定)
    │   └── infinicore (核心类型)
    │
    ├── C++ API 层
    │   ├── Tensor C++ API
    │   ├── Operator C++ API
    │   └── Runtime C++ API
    │
    └── 硬件后端层
        ├── CPU 后端
        ├── CUDA 后端
        ├── MUSA 后端
        ├── Kunlun 后端
        ├── Ascend 后端
        └── 其他硬件后端
```

**本目录的角色**：Python 前端层，提供用户友好的 Python API，通过 pybind11 绑定到 C++ 实现，是 InfiniCore 框架与用户交互的入口点。
