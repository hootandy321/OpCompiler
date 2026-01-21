# InfiniLM/include 架构全景

## 1. 子系统职责

`InfiniLM/include` 是 InfiniLM 推理框架的顶层 C 语言接口聚合层，其核心职责是为上层应用（如 Python 绑定、高级推理框架）提供统一、标准的模型推理 API 入口。该目录通过单一的 `infinicore_infer.h` 头文件，将整个推理引擎的所有接口模块化组织，实现了清晰的层次隔离和依赖管理。

作为 InfiniLM 的公共 API 边界，该层的主要功能包括：
- **接口聚合**：通过 `infinicore_infer.h` 统一导出所有子模块接口，为外部调用者提供单一的包含点
- **模块化组织**：将不同功能的接口（KV Cache、权重加载、模型特定接口）分散到独立的子目录和头文件中
- **模型抽象层**：为多种大语言模型架构（DeepSeek-V3、Jiuge、Jiuge-AWQ）提供标准化的推理接口
- **跨硬件抽象**：通过 C 接口封装底层硬件差异（CUDA、Ascend、Bang、Kunlun 等），支持多后端推理

## 2. 模块导航

### 顶层聚合接口

* **infinicore_infer.h**
    * 功能：InfiniLM 推理引擎的主入口头文件
    * 职责：统一包含并导出所有子模块接口，包括缓存管理、权重加载器和各模型特定接口，作为外部应用集成的唯一依赖点

### 核心子模块（infinicore_infer/）

* **cache.h**
    * 功能：KV Cache（键值缓存）的跨设备管理接口
    * 职责：提供 Transformer 注意力机制的键值对缓存管理，支持多层缓存创建、复制和销毁，服务于自回归文本生成任务

* **weights_loader.h**
    * 功能：通用权重加载器抽象接口
    * 职责：定义统一的模型权重加载 API，支持单机和多机分布式两种权重加载模式，屏蔽不同模型的权重组织差异

* **models/** (模型特定接口目录)
    * 功能：各类 LLM 架构的推理接口实现
    * 职责：封装不同模型架构（MoE、量化、标准 Transformer）的推理逻辑，为每种模型提供标准化的创建、推理和销毁接口

    * **models/CODEREADME.md**
        * 功能：models 子目录的核心实现文档
        * 职责：详细记录 DeepSeek-V3、Jiuge、Jiuge-AWQ 三种模型的数据结构、API 接口、使用示例和实现细节

    * **models/deepseek.h**
        * 功能：DeepSeek-V3 混合专家（MoE）大语言模型推理接口
        * 职责：为 DeepSeek-V3 提供完整的 C API，支持稀疏 MoE 层、多级注意力（MLA）、负载均衡等先进特性

    * **models/jiuge.h**
        * 功能：Jiuge 标准版 LLM 推理接口（基于 Transformer 架构）
        * 职责：为标准 Transformer 架构的 Jiuge 模型提供批量推理接口，支持多头注意力（MQA）和分组查询注意力（GQA）

    * **models/jiuge_awq.h**
        * 功能：Jiuge 量化版推理接口（AWQ 量化技术）
        * 职责：支持 AWQ（Activation-aware Weight Quantization）量化推理，提供与标准 Jiuge 相同的接口，但底层使用量化权重以降低显存占用

## 3. 架构逻辑图解

### 接口层次结构

```
[外部应用层]
    Python 绑定 / InfiniTrain / InfiniPerf / 用户 C/C++ 程序
            |
            v
+-----------------------------------------------------------+
|  InfiniLM/include (本层：公共 API 边界)                    |
|                                                           |
|  infinicore_infer.h  <-- 统一入口，聚合所有子模块接口        |
|      |                                                   |
|      +--- cache.h               (KV Cache 管理)           |
|      |                                                   |
|      +--- weights_loader.h     (权重加载抽象)             |
|      |                                                   |
|      +--- models/               (模型特定接口)            |
|            |                                              |
|            +--- deepseek.h      (DeepSeek-V3 MoE)        |
|            |                                              |
|            +--- jiuge.h         (Jiuge 标准)             |
|            |                                              |
|            +--- jiuge_awq.h     (Jiuge 量化)             |
|                                                           |
+-----------------------------------------------------------+
            |
            v
[InfiniRT / InfiniOP / InfiniCCL]  (底层运行时和算子库)
            |
            v
[硬件加速层]  CUDA / Ascend / Bang / Kunlun / Metax
```

### 模块依赖关系

**1. 自顶向下的依赖链**

```
infinicore_infer.h (顶层聚合)
    ├── cache.h
    │   └── 依赖: InfiniRT (设备管理、内存分配)
    │
    ├── weights_loader.h
    │   └── 依赖: InfiniRT (跨设备数据传输)
    │
    └── models/
        ├── deepseek.h
        │   └── 依赖: cache.h, weights_loader.h, InfiniOP, InfiniCCL
        │
        ├── jiuge.h
        │   └── 依赖: cache.h, InfiniOP, InfiniCCL
        │
        └── jiuge_awq.h
            └── 依赖: cache.h, weights_loader.h, InfiniOP
```

**2. 模型接口的差异化特性**

**DeepSeek-V3（混合专家架构）**：
- **复杂性最高**：涉及 MoE 路由、专家激活、负载均衡等逻辑
- **权重组织复杂**：区分全局权重、层权重、共享专家、路由专家等
- **专用 Cache**：使用 `DeepSeekV3Cache` 而非通用 `KVCache`
- **权重加载器**：通过 `DeepSeekV3WeightLoader` 函数指针表提供细粒度加载控制

**Jiuge（标准 Transformer）**：
- **架构简洁**：标准 QKV 注意力 + FFN（Gate-Up-Down 结构）
- **直接权重传递**：使用 `JiugeWeights` 结构体直接包含所有权重指针
- **通用 Cache**：使用标准 `KVCache` 管理键值缓存
- **无量化**：使用 FP16/FP32 权重，无量化开销

**Jiuge-AWQ（量化版本）**：
- **基于 Jiuge**：继承标准 Jiuge 的元数据结构（`JiugeAWQMeta` 包含 `JiugeMeta`）
- **量化优化**：增加 `nbit`、`quant_group_size` 等量化参数
- **通用权重加载**：使用 `ModelWeights` 抽象和 `loadModelWeight()` 通用接口
- **接口兼容**：推理 API 与标准 Jiuge 保持一致，屏蔽量化细节

### 推理执行流程

**阶段 1：模型初始化（一次性）**

```
用户代码
    |
    +-> 配置 Meta 结构体（模型超参数）
    |
    +-> [weights_loader.h] loadModelWeight() / loadModelWeightDistributed()
    |    |
    |    +-> 从磁盘/网络加载权重到 CPU 内存
    |    |
    |    +-> 通过 InfiniRT 传输到 GPU/NPU 显存
    |
    +-> createXXXModel(Meta, Weights)
         |
         +-> 初始化计算图、分配工作空间、编译内核
```

**阶段 2：请求处理（每次推理）**

```
用户代码
    |
    +-> [cache.h] createKVCache() / duplicateKVCache()
    |    |
    |    +-> 为每个请求分配独立的 KV Cache（多层缓存）
    |
    +-> inferBatchXXX() / forwardBatchXXX()
         |
         +-> 输入：tokens, ntok, nreq, req_lens, req_pos, caches[]
         |
         +-> 执行前向计算：
         |    |
         |    +-> Attention（QKV 投影 + RoPE + 缓存读写）
         |    |
         |    +-> FFN/MoE（前馈网络或专家路由）
         |    |
         +-> 输出：Logits 或 采样后的 Token
         |
         +-> 更新 KVCache（追加新的 K/V 对）
```

**阶段 3：资源清理（推理结束）**

```
用户代码
    |
    +-> [cache.h] dropKVCache()  (释放缓存)
    |
    +-> destroyXXXModel()        (释放模型和权重)
```

### 批处理与并发模型

**批处理机制**：
- 所有推理接口均为批量版本（`inferBatchXXX`, `forwardBatchXXX`）
- 支持变长请求：通过 `req_lens` 数组指定每个请求的长度
- 支持随机访问：通过 `req_pos` 数组指定每个请求在 `tokens` 扁平数组中的起始位置
- 隔离采样参数：每个请求可配置独立的 `temperature`、`topk`、`topp`

**并发隔离**：
- 每个请求维护独立的 KVCache，批处理时互不干扰
- 推理 API 本身无状态，可多线程并发调用（需确保模型对象不被同时修改）
- 多卡并行：通过 `ndev` 和 `dev_ids` 参数，自动将计算分布到多个设备

### 设计模式与架构原则

**1. 单一职责原则（SRP）**
- `cache.h`：仅负责 KV Cache 的生命周期管理
- `weights_loader.h`：仅负责权重加载的抽象接口
- `models/`：各模型头文件仅负责该模型的推理逻辑

**2. 开闭原则（OCP）**
- 扩展新模型：在 `models/` 下添加新的 `xxx.h`，实现标准接口模式，无需修改现有代码
- 扩展新硬件：通过 InfiniRT 抽象层，接口层代码无需修改

**3. 依赖倒置原则（DIP）**
- 顶层接口（`infinicore_infer.h`）依赖抽象接口（`cache.h`, `weights_loader.h`），而非具体实现
- 模型接口依赖 InfiniRT/InfiniOP 抽象层，而非特定硬件 API

**4. 接口隔离原则（ISP）**
- 每个模型提供独立的头文件和接口，调用者仅需包含所需模型的头文件
- 通用功能（Cache、权重加载）与特定模型功能分离

**5. 不透明指针模式**
- `DeepSeekV3Weights`, `DeepSeekV3Model`, `KVCache` 等类型为不透明指针
- 隐藏内部实现细节，提供清晰的 C API 边界
- 便于 ABI 兼容性和跨语言调用（如 Python CFFI）

### 扩展指南

**添加新模型支持的标准流程**：

1. **定义元数据结构**：在 `models/xxx.h` 中定义 `XXXMeta`，包含模型超参数
2. **定义权重结构**：定义 `XXXWeights` 或 `XXXWeightLoader`，描述权重组织格式
3. **实现生命周期管理**：
   - `createXXXWeights()` / `createXXXWeightLoader()`
   - `createXXXModel()`
   - `destroyXXXModel()`
4. **实现推理接口**：
   - `inferBatchXXX()`（带采样，返回 token）
   - `forwardBatchXXX()`（输出 logits，支持自定义采样）
5. **更新聚合头文件**：在 `infinicore_infer.h` 中添加 `#include "infinicore_infer/models/xxx.h"`
6. **编写文档**：在 `models/CODEREADME.md` 中添加新模型的详细说明

**关键约束**：
- 必须支持多设备（通过 `infiniDevice_t`, `dev_ids`, `ndev` 参数）
- 必须提供批量推理接口（支持 `nreq > 1`）
- 必须兼容 KV Cache 机制（使用标准 `KVCache` 或提供专用 Cache）
- 必须支持量化（可选，通过元数据中的 `dt_*` 参数控制数据类型）

### 与其他子系统的关系

**上游依赖（被包含层）**：
- **InfiniRT**：提供设备管理（`infiniDevice_t`）、内存分配、数据传输
- **InfiniOP**：提供算子内核（MatMul、LayerNorm、RoPE、量化等）
- **InfiniCCL**：提供多卡通信（AllReduce、AllGather 等）

**下游调用者（使用本层）**：
- **Python 绑定层**：通过 CFFI/Cython 将本层 C 接口封装为 Python API
- **InfiniTrain**：训练框架，可能使用本层进行模型验证和推理基准测试
- **InfiniPerf**：性能分析工具，使用本层接口测试推理吞吐和延迟
- **InfiniStudio**：可视化工具，通过本层接口展示模型推理过程

**同层协作**：
- 与 `InfiniLM/src` 实现层分离：本层仅为接口定义，实现在 `src/` 目录中
- 接口稳定性保证：本层接口保持向后兼容，实现细节可自由演进

### 文件清单

```
InfiniLM/include/
├── infinicore_infer.h              # 顶层聚合头文件（主入口）
└── infinicore_infer/
    ├── cache.h                     # KV Cache 管理接口
    ├── weights_loader.h            # 权重加载器抽象接口
    ├── CODEREADME_ANALYSIS.md      # 本文档：infenicore_infer/ 架构聚合
    └── models/
        ├── CODEREADME.md           # models/ 子目录核心实现文档
        ├── deepseek.h              # DeepSeek-V3 MoE 模型接口
        ├── jiuge.h                 # Jiuge 标准模型接口
        └── jiuge_awq.h             # Jiuge AWQ 量化模型接口
```

### 总结

`InfiniLM/include` 作为 InfiniLM 推理框架的公共 API 边界，成功地实现了多层次的目标：

1. **统一入口**：通过 `infinicore_infer.h` 为外部调用者提供单一的包含点
2. **模块化设计**：将缓存、权重加载、模型接口等功能分离到独立头文件
3. **多模型支持**：为 DeepSeek-V3、Jiuge、Jiuge-AWQ 等不同架构提供标准化接口
4. **跨硬件抽象**：通过 C 接口屏蔽底层硬件差异（CUDA、Ascend、Bang 等）
5. **扩展性**：清晰的设计模式和新模型添加流程，便于未来扩展

该层是 InfiniLM 系统架构中的关键抽象层，连接了底层硬件/算子库与上层应用，为高性能 LLM 推理提供了稳定、高效的 C 语言接口基础。
