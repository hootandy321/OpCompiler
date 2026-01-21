# Random Sample 算子架构全景

## 1. 子系统职责

`random_sample` 是 InfiniOp 中负责**随机采样操作**的核心算子模块，广泛应用于大语言模型（LLM）的文本生成场景。该算子支持两种主要采样策略：
- **Top-K 采样**：从概率最高的 K 个候选中选择
- **Top-P（Nucleus）采样**：从累积概率达到阈值 P 的最小候选集中选择

该模块采用**多硬件后端架构**，针对不同厂商的 GPU/加速器提供定制化实现，确保在各种硬件平台上都能获得高性能的采样能力。

## 2. 模块导航

### 已有文档的模块

* **nvidia**：
    * *功能*: NVIDIA CUDA 实现的随机采样算子，利用 CUB 库提供高性能的 GPU 计算能力
    * *职责*: 为 NVIDIA GPU 提供完整的 Top-K/Top-P 采样实现，支持贪心采样（ArgMax）和随机采样（Random）两种模式，通过归约、排序、前缀和等 GPU 原语实现高效的概率分布采样

### 文档缺失的模块

* **ascend**：
    * *功能*: *文档缺失* - 华为昇腾（Ascend）NPU 后端实现
    * *职责*: 为华为昇腾 AI 处理器提供随机采样算子适配

* **bang**：
    * *功能*: *文档缺失* - 寒武纪（Cambricon）MLU Bang 后端实现
    * *职责*: 为寒武纪 MLU 加速卡提供随机采样算子适配

* **cpu**：
    * *功能*: *文档缺失* - CPU 通用处理器后端实现
    * *职责*: 为 CPU 提供基础的随机采样算子实现，用于调试或无 GPU 环境的推理场景

* **kunlun**：
    * *功能*: *文档缺失* - 昆仑（Kunlun）芯片后端实现
    * *职责*: 为昆仑 AI 加速器提供随机采样算子适配

* **metax**：
    * *功能*: *文档缺失* - Metax 后端实现（具体硬件待确认）
    * *职责*: 为 Metax 硬件平台提供随机采样算子适配

* **moore**：
    * *功能*: *文档缺失* - Moore Threads 后端实现
    * *职责*: 为摩尔线程 GPU 提供随机采样算子适配

## 3. 架构逻辑图解

### 3.1 统一抽象接口层

所有硬件后端共享统一的算子接口定义（位于 `random_sample.h` 和 `info.h`）：

```
输入: probs (未归一化的 logits)
      ├─ 随机数 random_val (0~1)
      ├─ Top-P 阈值 topp (0~1)
      ├─ Top-K 范围 topk (整数)
      └─ 温度参数 temperature (浮点数)

输出: result (采样得到的单个整数索引)
```

### 3.2 NVIDIA 后端实现流程（参考文档）

NVIDIA 实现的完整计算流程分为两种模式：

**模式 A：贪心采样（ArgMax）**
```
logits → CUB::ArgMax → 最大值索引 → result
```

**模式 B：随机采样（Random）**
```
logits
  ↓
fillIndices (生成 [0,1,2,...,n-1])
  ↓
CUB::RadixSort (按概率降序排序)
  ↓
partialSoftmax (温度缩放 + exp 归一化)
  ↓
setSoftmaxMax (补齐第一个元素为 1)
  ↓
CUB::InclusiveSum (计算累积概率分布)
  ↓
randomSample (线性扫描采样，应用 Top-K/Top-P 裁剪)
  ↓
result
```

### 3.3 多硬件后端设计模式

该模块采用**策略模式**和**工厂模式**的组合设计：

1. **策略模式**：每个硬件后端（nvidia、ascend、bang 等）都是一个独立的策略实现，封装了该硬件特定的 kernel 调用、内存管理和并行优化逻辑

2. **工厂模式**：通过 `Descriptor::create()` 静态方法，根据传入的设备句柄类型（`infiniopHandle_t`）自动实例化对应硬件的描述符对象

3. **类型特化**：通过模板元编程支持多种数据类型组合（8 种整数类型 × 4 种浮点类型 = 32 种组合），在编译期展开为类型安全的实现

### 3.4 跨后端共性设计

尽管不同硬件的实现细节各异，但所有后端都遵循相同的设计原则：

- **Workspace 管理**：每个后端通过 `minWorkspaceSize()` 报告所需临时存储空间，由调用者统一分配和传递
- **流式执行**：支持异步执行（CUDA Stream、昇腾 Stream 等），允许与其他算子并发执行
- **错误处理**：统一的 `infiniStatus_t` 错误码体系，包括 `INFINI_STATUS_SUCCESS`、`INFINI_STATUS_INSUFFICIENT_WORKSPACE` 等
- **数据类型抽象**：通过 `infinicore.h` 中定义的 `fp16_t`、`bf16_t` 等抽象类型，实现跨硬件的半精度浮点支持

### 3.5 应用场景集成

该算子典型集成在 LLM 推理流程的**解码阶段**：

```
模型输出 (logits)
  ↓
Random Sample 算子
  ↓
采样 Token ID
  ↓
嵌入查找 → 下一轮预测
```

通过调整 `temperature`（温度）、`topk`（Top-K 范围）、`topp`（Top-P 阈值）等参数，可以控制生成文本的多样性和质量。

## 4. 文档完整性状态

- **已完成文档**: 1/7 (14.3%)
  - ✅ nvidia
- **缺失文档**: 6/7 (85.7%)
  - ❌ ascend
  - ❌ bang
  - ❌ cpu
  - ❌ kunlun
  - ❌ metax
  - ❌ moore

**建议**：为保持架构文档的完整性，应优先为其他 6 个硬件后端补充 CODEREADME.md 文档，特别是国内主流的昇腾（ascend）和寒武纪（bang）平台。
