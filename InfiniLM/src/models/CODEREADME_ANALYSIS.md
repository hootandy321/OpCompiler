# models 目录架构全景

## 1. 子系统职责

`models` 目录是 InfiniLM 推理引擎的核心模型实现层，负责将通用 Transformer 架构适配到具体的大语言模型。该目录通过模块化设计支持多种模型架构：

1. **模型特异性实现**：针对不同模型（DeepSeek V3、Jiuge、JiugeAWQ）的定制化推理引擎
2. **量化策略支持**：从 FP16 权重到 INT4/INT8 量化的多种精度方案
3. **架构特性封装**：MoE（混合专家）、MLA（多头潜在注意力）、GQA（分组查询注意力）等先进机制
4. **多硬件后端**：统一接口支持 CUDA、CPU、Kunlun 等不同计算设备

该层位于 InfiniLM 架构的中枢位置，向下依赖 InfiniCore 的算子库和内存管理，向上为推理框架提供模型实例化接口。

## 2. 模块导航

* **deepseek_v3**:
    * 功能: DeepSeek V3 混合专家模型（MoE）的高性能推理实现，支持 256 专家路由和 MLA 注意力压缩
    * 职责: 实现 DeepSeek V3 特有的稀疏 MoE 路由、MLA KV Cache 压缩、W8A8 量化和多设备张量并行

* **jiuge**:
    * 功能: Jiuge 大语言模型的标准 Transformer 推理引擎，支持 GQA、RoPE 和批量推理
    * 职责: 提供通用 Transformer 架构的完整推理流程，包括注意力机制、FFN、采样和多设备并行调度

* **jiuge_awq**:
    * 功能: Jiuge 模型的 AWQ（Activation-aware Weight Quantization）INT4 量化推理实现
    * 职责: 实现 INT4 分组量化的反量化计算、量化权重加载和多设备高效推理

## 3. 架构逻辑图解

### 数据流向与模块关系

```
用户请求
    ↓
[模型选择] → deepseek_v3 / jiuge / jiuge_awq
    ↓
[权重加载] → create{Model}Weights()
    ↓
[模型实例化] → create{Model}Model()
    ├── 多设备资源初始化 (DeviceResource)
    ├── 推理线程池启动 (Thread-per-device)
    └── RoPE 表预生成 (sin/cos lookup tables)
    ↓
[批量推理循环]
    ├── 主线程: 准备 InferRequest，通知所有设备
    ├── 设备线程 0-N: 并行执行 inferDeviceBatch()
    │   ├── 词嵌入查找 (Embedding Lookup)
    │   ├── 逐层计算 (Layer Iteration)
    │   │   ├── Attention 部分
    │   │   │   ├── RMSNorm 归一化
    │   │   │   ├── QKV 投影 (jiuge: 线性, deepseek_v3: 量化两阶段, jiuge_awq: INT4 反量化)
    │   │   │   ├── RoPE 位置编码 (旋转 Q/K)
    │   │   │   ├── KV Cache 更新 (deepseek_v3: MLA 压缩存储)
    │   │   │   ├── 注意力计算 (jiuge: GQA, deepseek_v3: 满注意力)
    │   │   │   ├── 输出投影 + 残差
    │   │   │   └── AllReduce (多设备聚合)
    │   │   └── FFN 部分
    │   │       ├── RMSNorm 归一化
    │   │       ├── jiuge: SwiGLU (gate_proj + up_proj)
    │   │       ├── deepseek_v3: MoE 路由
    │   │       │   ├── 共享专家计算 (Shared Expert)
    │   │       │   ├── Top-8 路由选择 (topkrouter)
    │   │       │   └── 路由专家计算 (8 个专家加权求和)
    │   │       ├── jiuge_awq: INT4 量化 SwiGLU
    │   │       ├── 下投影 + 残差
    │   │       └── AllReduce
    │   └── 输出采样 (Rank 0)
    │       ├── RMSNorm + 输出投影
    │       ├── Random Sampling (top-k/top-p/temperature)
    │       └── 返回生成 token
    └── 主线程: 逆序等待所有设备完成
    ↓
[返回结果]
```

### 关键设计模式对比

| 特性 | deepseek_v3 | jiuge | jiuge_awq |
|------|-------------|-------|-----------|
| **模型架构** | MoE (256 专家) | 标准 Transformer | 标准 Transformer |
| **注意力机制** | MLA (低秩 KV 压缩) | GQA (分组查询) | GQA (分组查询) |
| **量化方案** | W8A8 (8-bit 权重+激活) | FP16/FP32 | INT4 分组量化 |
| **KV Cache** | 压缩存储 (r_kv=512) | 标准存储 | 标准存储 |
| **专家路由** | Top-8 路由 + 共享专家 | 无 | 无 |
| **并行策略** | 张量并行 (头数/FFN 分片) | 张量并行 | 张量并行 |
| **线程模型** | Thread-per-device + 条件变量同步 | Thread-per-device + 条件变量同步 | Thread-per-device + 条件变量同步 |
| **内存池** | 128MB 预分配 | 128MB 预分配 | 128MB 预分配 |

### 模块间协作机制

1. **公共基础设施** (共享 InfiniCore 能力):
   - `Tensor`: 所有模块使用统一的张量抽象
   - `MemoryPool`: 推理缓冲区复用策略一致
   - `InferenceContext`: 算子执行上下文管理
   - `InfiniOP`: rmsnorm, linear, rope, causalSoftmax 等算子

2. **差异化实现** (模型特性定制):
   - **deepseek_v3**: 独有的 `topkrouter` 算子、MLA 两阶段投影、MoE 专家加权求和
   - **jiuge**: 标准 GQA 实现、QK-Norm 可选支持
   - **jiuge_awq**: `dequant_linear` 融合算子、INT4 权重解包

3. **统一 API 接口** (对外暴露一致性):
   - `create{Model}Model()`: 模型实例化
   - `inferBatch{Model}()`: 批量推理 + 采样
   - `forwardBatch{Model}()`: 批量前向传播 (无采样)
   - `destroy{Model}Model()`: 资源清理

### 性能优化策略汇总

所有模块共同采用的优化技术：

- **算子融合**: 反量化 + 矩阵乘融合、SiLU + 逐元素乘融合
- **内存复用**: logits_in/logits_out 交替使用、跨层缓冲区共享
- **预计算**: RoPE sin/cos 表初始化生成、避免运行时三角函数计算
- **异步执行**: 权重加载流与计算流分离、NCCL 通信与计算重叠
- **批处理**: 多请求动态形状支持、减少 kernel 启动开销
- **缓存友好**: 权重按层连续存储、提升 GPU L2 Cache 命中率

### 扩展性设计

新增模型实现需遵循的架构约束：

1. **数据结构**: 实现 `{Model}Meta`、`{Model}Weights`、`{Model}DeviceWeight`
2. **资源管理**: 实现 `createDeviceResource()` / `releaseDeviceResource()`
3. **推理内核**: 实现 `inferDeviceBatch()`，遵循"嵌入 → 层迭代 → 输出"流程
4. **线程模型**: 采用 Thread-per-device + 条件变量同步模式
5. **C API 导出**: 使用 `__C` 宏导出 C 兼容接口，确保跨语言调用
6. **依赖注入**: 通过 InfiniCore 算子库实现硬件无关性

通过这种模块化设计，models 目录能够快速支持新模型架构，同时保持代码复用性和性能优化的一致性。
