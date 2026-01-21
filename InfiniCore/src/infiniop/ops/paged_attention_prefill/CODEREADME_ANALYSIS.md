# 📂 目录: paged_attention_prefill 架构全景

## 1. 子系统职责

`paged_attention_prefill` 目录实现了**分页注意力预填充（Paged Attention Prefill）算子**，这是大语言模型（LLM）推理服务中的核心计算组件。该子系统的主要职责是在推理阶段高效计算 Query 与分块 KV 缓存（Paged KV Cache）之间的注意力权重，支持增量预填充、多查询注意力（MQA）、分组查询注意力（GQA）以及 ALiBI 位置偏置。其核心价值在于通过分块内存管理实现动态 KV 缓存，显著提升推理服务的批处理能力和内存利用率。

## 2. 模块导航 (Module Navigation)

### 2.1 NVIDIA 后端实现

* **📂 nvidia**:
    * *功能*: NVIDIA CUDA 后端的分页注意力预填充算子实现，提供完整的 CUDA 内核启动逻辑和设备上下文管理
    * *职责*: 封装 CUDA 设备句柄、类型派发逻辑、内核启动配置，并暴露标准化的 InfiniOP API 接口
    * *核心特性*:
        - 支持半精度（FP16/BF16）和单精度（FP32）数据类型
        - 实现因果掩码与增量预填充（支持已有 KV 缓存拼接）
        - 通过分块表（block_tables）实现非连续物理内存的 KV 缓存访问
        - 三遍注意力算法：找最大值 → 求指数和 → 加权求和，确保数值稳定性
        - Grid/Block 配置：`(total_q_tokens, num_heads)` 线程块，每块 `head_size` 线程

### 2.2 缺失文档模块

* **📂 cuda**:
    * *状态*: **文档缺失**
    * *说明*: 该子目录存在但未提供 CODEREADME.md，无法提取功能描述

### 2.3 共享基础设施（父目录级）

* **📄 info.h**:
    * *功能*: 定义 `PagedAttentionPrefillInfo` 类，作为算子元数据容器
    * *职责*: 存储张量形状、步长、数据类型、分块配置等参数，提供 `create()` 工厂方法验证输入合法性

* **📄 paged_attention_prefill.h**:
    * *功能*: 算子描述符的宏定义头文件
    * *职责*: 通过 `DESCRIPTOR` 宏生成后端无关的描述符类结构（如 `nvidia::Descriptor`）

* **📄 operator.cc**:
    * *功能*: C++ 绑定层，将 InfiniOP 算子暴露为高层 API
    * *职责*: 实现符号注册、参数解析、设备调度等胶水逻辑

## 3. 架构逻辑图解

### 3.1 数据流与计算流程

```
用户请求 (推理查询)
    ↓
InfiniOP Handle 初始化
    ↓
TensorDescriptor 创建
    ├── Query: [total_q_tokens, num_heads, head_size]
    ├── K/V Cache: [num_blocks, num_kv_heads, block_size, head_size]
    ├── Block Tables: [num_seqs, max_num_blocks_per_seq]
    ├── Seq Lens: [num_seqs]
    └── Cum Seq Lens Q: [num_seqs + 1]
    ↓
Descriptor::create() (参数验证)
    ├── PagedAttentionPrefillInfo::create()
    │   ├── 检查数据类型一致性 (FP16/BF16/FP32)
    │   ├── 验证张量形状 (3D/4D/2D)
    │   └── 提取步长与配置参数
    └── 分配 Opaque (CUDA 设备句柄)
    ↓
GPU 内存分配与数据拷贝
    ├── cudaMemcpy(H2D) Query 数据
    ├── cudaMemcpy(H2D) K/V 缓存
    ├── cudaMemcpy(H2D) 块表与序列长度
    └── 创建 CUDA 流
    ↓
Descriptor::calculate()
    ├── 类型派发 (dtype → 模板特化)
    │   ├── INFINI_DTYPE_F16 → <half, float>
    │   ├── INFINI_DTYPE_BF16 → <__nv_bfloat16, float>
    │   └── INFINI_DTYPE_F32 → <float, float>
    ↓
    launchPagedAttentionPrefill()
        ├── 配置 Grid: (total_q_tokens, num_heads)
        ├── 配置 Block: (head_size)
        └── 启动 CUDA 内核
    ↓
pagedAttentionPrefillKernel<Tdata, Tcompute> (CUDA 并行执行)
    ├── 每个线程块处理 (token, head) 对
    │   ├── 1. 序列定位: 二分查找 cum_seq_lens_q
    │   ├── 2. 因果掩码: 计算 causal_limit
    │   ├── 3. KV 头映射: num_kv_heads → head_idx
    │   ├── 4. 注意力分数 (第一遍):
    │   │   ├── 遍历 t ∈ [0, causal_limit]
    │   │   ├── 通过 block_tables 解析物理块 ID
    │   │   ├── 计算 Q·K^T 点积 (head_size 维度)
    │   │   ├── 应用 ALiBI 偏置 (可选)
    │   │   └── 记录 max_score
    │   ├── 5. Softmax 归一化 (第二遍):
    │   │   ├── 重新计算分数
    │   │   ├── exp(score - max_score)
    │   │   └── 累加 sum_exp
    │   └── 6. 加权求和 (第三遍):
    │       ├── 重新计算分数与概率
    │       ├── 读取 V 缓存
    │       └── 累加输出维度 (每个线程负责一个 dim)
    ↓
cudaStreamSynchronize() (等待计算完成)
    ↓
cudaMemcpy(D2H) 拷回结果
    ↓
释放资源 (delete/Destroy)
```

### 3.2 模块依赖关系

```
paged_attention_prefill
    │
    ├── [共享基础设施]
    │   ├── info.h (元数据容器)
    │   ├── paged_attention_prefill.h (宏定义)
    │   └── operator.cc (C++ 绑定)
    │
    └── [NVIDIA 后端] ← nvidia/
        ├── paged_attention_prefill_nvidia.cuh
        │   └── DESCRIPTOR 宏生成 Descriptor 类
        ├── paged_attention_prefill_nvidia.cu
        │   ├── Descriptor::create() (工厂方法)
        │   ├── Descriptor::calculate() (类型派发)
        │   └── launchPagedAttentionPrefill() (内核启动)
        └── cuda/kernel.cuh (父目录共享)
            └── pagedAttentionPrefillKernel<Tdata, Tcompute> (CUDA 内核)
                │
                └── [依赖]
                    ├── ../../../devices/nvidia/nvidia_common.cuh
                    └── ../../../devices/nvidia/nvidia_kernel_common.cuh
```

### 3.3 关键设计决策

1. **分页 KV 缓存（Paged KV Cache）**:
   - **动机**: 传统连续 KV 缓存难以处理变长序列和动态批处理
   - **实现**: 通过 `block_tables[seq_idx][b_idx]` 将逻辑块映射到物理块，支持非连续内存布局
   - **优势**: 允许不同序列的 KV 缓存分散存储，提高内存利用率

2. **三遍注意力算法**:
   - **第一遍**: 扫描所有 KV token，记录最大分数 `max_score`（数值稳定性）
   - **第二遍**: 重新计算分数，求 `sum_exp = Σ exp(score - max_score)`
   - **第三遍**: 计算概率 `prob = exp(score - max_score) / sum_exp`，加权求和 V
   - **代价**: 需要三次遍历 KV 缓存，但避免浮点溢出

3. **Grid/Block 并行策略**:
   - **Grid 维度**: `(total_q_tokens, num_heads)` - 每个 (token, head) 对一个线程块
   - **Block 维度**: `(head_size,)` - 每个头维度一个线程
   - **优势**: 完全并行化，无线程间同步（无 `__syncthreads()`）

4. **类型派发（Type Dispatch）**:
   - **策略**: 在 `calculate()` 中根据 `_info.dtype` 选择模板特化
   - **计算类型**: `Tcompute` 固定为 `float`，确保数值精度
   - **存储类型**: `Tdata` 支持 `half`、`__nv_bfloat16`、`float`

### 3.4 性能瓶颈与优化空间

- **瓶颈 1**: 三次遍历 KV 缓存（可优化为单遍缓存分数）
- **瓶颈 2**: 跨步访问 KV 缓存（可能导致缓存行利用率低）
- **瓶颈 3**: 未使用 Tensor Core（无 WMMA 矩阵乘法优化）
- **优化方向**: FlashAttention 风格的分块 tiling、共享内存优化、Tensor Core 加速

## 4. 文档完整性说明

- **已覆盖**: `nvidia` 后端实现（完整 CODEREADME.md）
- **缺失文档**: `cuda` 子目录（无法提取架构信息）
- **建议**: 为 `cuda` 子目录补充 CODEREADME.md，以完善多硬件后端文档体系
