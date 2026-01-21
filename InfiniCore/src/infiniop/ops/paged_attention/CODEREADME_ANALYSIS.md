# Paged Attention 操作目录架构全景

## 1. 子系统职责

本目录实现了 **Paged Attention（分页注意力）** 操作，这是大语言模型推理中的核心优化算子，专门用于处理可变长度序列的 KV Cache 内存管理。通过将 KV Cache 分块存储并使用动态块表映射，该模块实现了高效的长序列推理支持，能够灵活处理不同长度的请求而无需连续内存分配。该子系统采用前后端分离架构，前端提供统一的 C 接口和元数据验证，后端针对不同硬件（NVIDIA/CUDA）提供高性能内核实现。

## 2. 模块导航

### 硬件后端实现子目录

* **`nvidia/`** (NVIDIA GPU 后端)
    * *功能*: NVIDIA 硬件后端的完整实现，包含描述符管理、内核启动逻辑和多态调度
    * *职责*: 提供 NVIDIA GPU 的高性能 Paged Attention 计算，支持 FP16/BF16/FP32 数据类型，实现自适应线程块配置和模板特化优化
    * *文档状态*: 已完成 CODEREADME.md

* **`cuda/`** (CUDA 通用内核)
    * *功能*: CUDA 设备内核的通用实现，提供核心 Paged Attention 计算逻辑
    * *职责*: 实现分块注意力计算的核心 CUDA 设备函数，包含 QK 点积、Softmax 归约和加权值聚合四个阶段，支持 MHA/GQA/MQA 多种注意力模式
    * *文档状态*: 文档缺失，仅有内核实现代码

### 核心接口与元数据文件

* **`paged_attention.h`** (描述符声明宏)
    * *功能*: 通过宏定义生成各硬件后端的 Descriptor 类模板
    * *职责*: 定义统一的描述符接口结构，包含工厂方法 create()、计算方法 calculate() 和工作空间查询方法 workspaceSize()

* **`info.h`** (元数据验证类)
    * *功能*: 定义 PagedAttentionInfo 类，负责验证输入张量的形状、类型和步长
    * *职责*: 在描述符创建阶段进行元数据检查，确保 head_size 为 16/32/64/128/256 之一，验证数据类型一致性（FP16/BF16/FP32），提取关键维度信息

* **`operator.cc`** (C 接口层)
    * *功能*: 提供符合 C 调用约定的外部接口（infiniopCreatePagedAttentionDescriptor 等）
    * *职责*: 实现设备类型路由，根据硬件类型（NVIDIA）分派到对应后端的实现，处理可选参数（如 ALiBI slopes）

## 3. 架构逻辑图解

### 数据流与交互关系

```
外部调用 (C API)
    ↓
operator.cc (设备路由层)
    ├── infiniopCreatePagedAttentionDescriptor  → 选择硬件后端
    ├── infiniopPagedAttention                  → 调度计算
    └── infiniopDestroyPagedAttentionDescriptor  → 资源清理
    ↓
后端选择 (switch-case)
    ↓
[NVIDIA 后端路径]
    ↓
nvidia/Descriptor::create()
    ├── 调用 PagedAttentionInfo::create()     → 元数据验证 (info.h)
    ├── 从 infiniopHandle 提取 NVIDIA 内部句柄
    └── 初始化 Descriptor 对象
    ↓
nvidia/Descriptor::calculate()
    ├── 根据 maxThreadsPerBlock() 选择线程块大小 (512/1024/4096)
    ├── 根据 head_size 和 dtype 进行模板特化
    └── 启动 CUDA 内核
    ↓
cuda/pagedAttentionKernel (核心计算)
    ├── 阶段1: 加载查询向量到共享内存
    ├── 阶段2: 并行计算 QK 点积 → CUB BlockReduce 查找最大值
    ├── 阶段3: 计算稳定 Softmax (exp-sum-normalize)
    └── 阶段4: 加权聚合值向量 → 输出结果
    ↓
返回输出张量 (GPU 内存)
```

### 关键设计模式与优化策略

**1. 前后端分离架构**
- **前端** (`operator.cc` + `paged_attention.h` + `info.h`): 提供统一接口，负责元数据验证和设备路由
- **后端** (`nvidia/`, `cuda/`): 封装硬件特定实现，通过宏 DESCRIPTOR(NAMESPACE) 实现多态

**2. 分层内核设计**
- **NVIDIA 层** (`paged_attention_nvidia.cu`): 薄适配层，处理内核启动配置和模板特化逻辑
- **CUDA 层** (`kernel.cuh`): 通用设备函数，可被不同硬件后端复用（如未来添加 ROCm 后端）

**3. 编译期优化**
- 使用模板参数 `HEAD_SIZE` (16/32/64/128/256) 和 `NUM_THREADS` (512/1024/4096)
- 通过三层宏嵌套 (`SWITCH_HEAD_SIZE` → `LAUNCH_HEADSIZE_BLOCKSIZE`) 在编译期展开所有组合
- QK 点积计算手动循环展开（每 8 个元素），充分利用内存合并访问

**4. 并行计算策略**
- **二维网格布局**: `(num_heads, num_seqs)` - 每个序列的每个注意力头独立处理
- **线程块协作**: 每个线程块内 NUM_THREADS 个线程协作完成单个注意力头的计算
- **共享内存优化**: 查询向量、logits 数组存储在共享内存，减少全局内存访问

**5. 数值稳定性保障**
- **类型分离**: 数据类型 Tdata (FP16/BF16) 与计算类型 Tcompute (FP32) 分离
- **稳定 Softmax**: 先查找全局最大值 (CUB BlockReduce)，再计算 exp，避免数值溢出
- **归一化保护**: 添加 1e-6f 小常数防止除零

### 支持的高级特性

**GQA/MQA 支持**
- 通过 `num_kv_heads` 参数实现分组查询注意力
- KV 头索引计算: `kv_head_idx = head_idx / (num_heads / num_kv_heads)`
- 允许多个查询头共享同一个 KV 头，显著减少显存占用

**ALiBI 位置编码**
- 可选的 `alibi_slopes` 参数，支持线性偏置位置编码
- 计算公式: `qk += alibi_slope * (token_idx - seq_len + 1)`
- 适用于外推长场景（推理长度超过训练长度）

**动态块表映射**
- 每个序列维护独立的块表 (`block_tables`)，将逻辑块号映射到物理块号
- 支持非连续内存分配，提高显存利用率
- 允许动态增删序列而无需重新分配整个 KV Cache

### 依赖关系图

```
paged_attention/
├── operator.cc
│   ├── → ../../operator.h (基础操作接口)
│   ├── → ../../handle.h (设备句柄)
│   └── → paged_attention.h (描述符宏)
│
├── paged_attention.h
│   ├── → ../../operator.h (InfiniopDescriptor 基类)
│   └── → info.h (PagedAttentionInfo)
│
├── info.h
│   ├── → ../../../utils.h (错误处理宏)
│   ├── → ../../tensor.h (张量描述符)
│   └── → <optional> (C++ STL)
│
├── nvidia/paged_attention_nvidia.cu
│   ├── → paged_attention.h (DESCRIPTOR 宏)
│   ├── → info.h (元数据验证)
│   ├── → device/nvidia/Handle (硬件句柄)
│   ├── → ../cuda/kernel.cuh (通用内核)
│   └── → cub/block/block_reduce.cuh (CUB 库)
│
└── cuda/kernel.cuh
    ├── → ../../../reduce/cuda/reduce.cuh (并行归约原语)
    └── → <cuda_fp16.h>, <cuda_bf16.h> (半精度支持)
```

### 性能特征

**时间复杂度**: O(seq_len × head_size)
- 每个注意力头需要遍历所有 token 的 QK 点积计算
- 每个维度需要遍历所有 token 进行加权聚合

**空间复杂度**: O(seq_len + head_size) (每个线程块的共享内存)
- seq_len: 存储 logits 数组
- head_size: 存储查询向量
- 2 × sizeof(float): 存储全局最大值和归一化因子

**并行度**: O(num_heads × num_seqs × NUM_THREADS)
- 所有序列的所有注意力头完全并行
- 每个线程块内 NUM_THREADS 个线程协作

**吞吐量优化点**
1. **内存合并**: QK 点积的手动展开确保全局内存访问合并
2. **共享内存复用**: 查询向量加载一次后对所有 token 复用
3. **避免全局同步**: 线程块之间完全独立，无需跨块同步
4. **Warp 级优化**: 使用 CUB 库的 BlockReduce 利用 Warp Shuffle 指令
