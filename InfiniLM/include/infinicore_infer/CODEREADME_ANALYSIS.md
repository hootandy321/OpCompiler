# infinicore_infer 架构全景

## 1. 子系统职责

`infinicore_infer` 是 InfiniLM 的核心推理接口层，负责为不同的大语言模型（LLM）架构提供统一的 C 语言推理 API。该子系统封装了模型加载、权重管理、KV Cache 管理和批量推理的底层实现细节，向上层暴露标准化的接口，支持多种硬件加速后端（通过 InfiniRT 抽象层）。

作为 InfiniLM 推理引擎的头文件接口层，该模块的主要职责包括：
- 定义标准化的模型元数据结构（Meta 结构体）
- 提供跨硬件的模型创建、销毁和推理接口
- 支持混合专家（MoE）架构、量化推理（AWQ）等高级特性
- 管理 KV Cache 的生命周期和批量推理请求

## 2. 模块导航

### 核心基础设施模块

* **cache.h**
    * 功能：KV Cache（键值缓存）管理接口
    * 职责：提供跨设备的 KV Cache 创建、复制和销毁功能，支持多层注意力机制的缓存管理

* **weights_loader.h**
    * 功能：通用权重加载器接口
    * 职责：定义统一的模型权重加载接口，支持单机和分布式两种权重加载模式

### 模型特定接口模块

* **models/jiuge.h**
    * 功能：九格（Jiuge）模型推理接口
    * 职责：为标准 Transformer 架构的 Jiuge 模型提供完整的 C API，包括模型创建、批量推理（带采样）和前向传播（输出 logits）

* **models/jiuge_awq.h**
    * 功能：Jiuge 模型的 AWQ 量化推理接口
    * 职责：支持 AWQ（Activation-aware Weight Quantization）量化推理，提供与标准 Jiuge 模型相同的推理接口，但底层使用量化权重以减少显存占用和加速推理

* **models/deepseek.h**
    * 功能：DeepSeek-V3 模型推理接口
    * 职责：为 DeepSeek-V3（混合专家架构）提供完整的推理 API，支持稀疏 MoE 层、密集层、多级注意力（MLA）、负载均衡等高级特性

## 3. 架构逻辑图解

### 数据流向与模块交互关系

```
                     [上层应用/Python 绑定]
                              |
                              v
        +-------------------------------------+
        |     infinicore_infer 接口层         |
        +-------------------------------------+
        |                                     |
   [weights_loader]                    [cache.h]
        |                                     |
        v                                     v
[ModelWeights 抽象]              [KVCache 管理]
        |                                     |
        +------------------+------------------+
                           |
        +------------------+------------------+
        |                  |                  |
        v                  v                  v
 [models/jiuge.h]  [models/jiuge_awq.h]  [models/deepseek.h]
        |                  |                  |
 [JiugeModel]       [JiugeAWQModel]    [DeepSeekV3Model]
        |                  |                  |
        +------------------+------------------+
                           |
                           v
              [InfiniRT / InfiniOP / InfiniCCL]
                           |
                           v
                  [硬件加速后端：CUDA/Ascend 等]
```

### 推理流程解析

**1. 模型初始化阶段**
```
createXXXModel() -> 加载权重 -> 分配显存 -> 初始化算子
     |
     +-> [weights_loader.h]
          |
          +-> loadModelWeight() / loadModelWeightDistributed()
               |
               +-> 将权重从 CPU 传输到指定设备（单卡/多卡）
```

**2. 推理准备阶段**
```
createKVCache() -> 为每个请求分配 KV Cache
     |
     +-> [cache.h]
          |
          +-> 分配多层 KV 缓存（nlayers × max_len × dk/dv）
```

**3. 批量推理阶段**
```
inferBatchXXX() -> 多请求并行处理
     |
     +-> 输入：tokens, ntok, nreq, req_lens, req_pos
     +-> KVCache：每个请求独立的缓存
     +-> 采样参数：temperature, topk, topp
     |
     v
前向计算（Attention + FFN/MoE） -> Logits -> 采样 -> 输出 tokens
     |
     v
更新 KVCache（追加新的 K/V 对）
```

**4. 模型差异化特性**

**Jiuge（标准 Transformer）**：
- 标准 QKV 注意力机制
- FFN（Gate-Up-Down 三层结构）
- 支持多头注意力（MQA）和位置编码（RoPE）

**Jiuge-AWQ（量化版本）**：
- 在 Jiuge 基础上增加权重量化支持
- 量化位宽（nbit）和量化组大小（quant_group_size）配置
- 共享相同的推理接口，底层自动处理量化/反量化

**DeepSeek-V3（混合专家架构）**：
- 多级注意力（MLA）：Q/K/V 通过低秩投影压缩
- 混合专家层（MoE）：包含共享专家（Shared Experts）和路由专家（Routed Experts）
- 稀疏激活：每次推理仅激活 Top-K 个专家（kexperts）
- 负载均衡：通过 routed_scale 参数调节专家负载
- 复杂权重加载：区分全局权重、层权重、线性层权重、专家权重等

### 关键设计模式

**1. 类型擦除与多态**
- 所有模型接口统一使用 `void *` 或自定义结构体指针（如 `struct KVCache *`）
- 通过函数指针类型定义实现灵活的权重加载（DeepSeekV3WeightLoader）

**2. 设备抽象**
- 通过 `infiniDevice_t` 和 `dev_ids` 支持多设备并行
- 权重加载支持分布式模式（`loadModelWeightDistributed`）

**3. 批处理优化**
- 所有推理接口均为批量版本（`inferBatchXXX`, `forwardBatchXXX`）
- 支持不同长度的请求通过 `req_lens` 和 `req_pos` 参数灵活组合

**4. 内存管理**
- KV Cache 支持动态创建（`createKVCache`）、复制（`duplicateKVCache`）和销毁（`dropKVCache`）
- 每个请求维护独立的 KV Cache，支持并发推理

### 依赖关系

**上游依赖**：
- `infinirt.h`：硬件抽象层，提供设备管理和内存操作
- `infiniop.h`：算子库，提供张量计算原语
- `infiniccl.h`：通信库，支持多卡/多节点通信

**下游调用者**：
- InfiniLM 的 Python 绑定层（通过 CFFI 或 Cython 调用）
- 高级推理框架（如 InfiniTrain、InfiniPerf）

### 扩展点

如需添加新模型支持，需实现以下标准接口模式：
1. 定义 `XXXMeta` 结构体（模型元数据和超参数）
2. 定义 `XXXWeights` 结构体或权重加载器（权重组织格式）
3. 实现 `createXXXModel()` / `destroyXXXModel()`
4. 实现 `inferBatchXXX()`（带采样）和 `forwardBatchXXX()`（输出 logits）
5. 可选：实现模型特定的 Cache 结构（如 `DeepSeekV3Cache`）
