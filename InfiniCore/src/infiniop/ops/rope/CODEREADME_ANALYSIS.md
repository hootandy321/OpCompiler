# 📂 目录: rope (RoPE 旋转位置编码) 架构全景

## 1. 子系统职责

`rope` 目录实现了 **Rotary Position Embedding (RoPE)** 操作，这是 Transformer 模型中用于编码位置信息的关键技术。RoPE 通过旋转变换将绝对位置信息注入到注意力机制的 Query 和 Key 向量中，使得位置编码具备相对位置感知能力（即点积仅依赖于相对位置差 m-n，而非绝对位置）。

该子系统为多种硬件加速平台提供统一的后端实现，包括：
- **NVIDIA GPU** (CUDA)
- **Moore/MUSA 硬件** (沐曦科技)
- **华为 Ascend** (昇腾 NPU)
- **百度昆仑** (昆仑 AI 芯片)
- **寒武纪 Bang** (MLU)
- **天数智芯 Metax** (启明 GPU)
- **x86 CPU** (通用处理器)

每个硬件后端针对其指令集架构和内存层次结构进行了专门优化，同时对外保持一致的 API 接口。

## 2. 模块导航

* **📂 nvidia**:
    * *功能*: NVIDIA CUDA 后端的 RoPE 实现，支持 GPT-J 和 GPT-NeoX 两种主流位置编码算法
    * *职责*: 为 NVIDIA GPU 提供高性能 RoPE 计算内核，通过模板特化支持 FP16/BF16/FP32/FP64 数据类型和 8 种整数类型的 position IDs，采用 half2/bfloat162 向量化指令优化计算吞吐量

* **📂 moore**:
    * *功能*: Moore/MUSA 硬件平台（沐曦科技）的 RoPE CUDA 内核实现
    * *职责*: 适配 MUSA 框架的差异性（如 bfloat16 内置函数映射 `mt_bfloat16`、数学库类型歧义处理），在 GPT-J 和标准模式下提供与 NVIDIA 兼容的功能接口

* **📂 ascend**:
    * *功能*: 华为昇腾 NPU 后端实现
    * *职责*: *文档缺失*（目录存在但未生成 CODEREADME.md）

* **📂 bang**:
    * *功能*: 寒武纪 MLU 后端实现
    * *职责*: *文档缺失*（目录存在但未生成 CODEREADME.md）

* **📂 cpu**:
    * *功能*: x86 CPU 后端实现
    * *职责*: *文档缺失*（目录存在但未生成 CODEREADME.md）

* **📂 cuda**:
    * *功能*: 通用 CUDA 后端实现（可能与 nvidia 目录存在功能重叠）
    * *职责*: *文档缺失*（目录存在但未生成 CODEREADME.md）

* **📂 kunlun**:
    * *功能*: 百度昆仑 AI 芯片后端实现
    * *职责*: *文档缺失*（目录存在但未生成 CODEREADME.md）

* **📂 metax**:
    * *功能*: 天数智芯启明 GPU 后端实现
    * *职责*: *文档缺失*（目录存在但未生成 CODEREADME.md）

## 3. 架构逻辑图解

### 3.1 调用流程层次

RoPE 子系统采用典型的**分层后端架构**，从高层 API 到底层硬件加速器的调用链如下：

```
应用层 (InfiniLM/InfiniTrain)
    ↓
统一接口层 (rope.h + operator.cc)
    - 定义 infiniopCreateRoPEDescriptor()
    - 定义 infiniopRoPE() 执行函数
    - 张量形状验证 (RoPEInfo)
    ↓
硬件后端分发层
    ├─ nvidia::Descriptor      (NVIDIA GPU)
    ├─ moore::Descriptor       (Moore/MUSA)
    ├─ ascend::Descriptor      (昇腾 NPU)     [未文档化]
    ├─ bang::Descriptor        (寒武纪 MLU)   [未文档化]
    ├─ kunlun::Descriptor      (昆仑芯片)     [未文档化]
    ├─ metax::Descriptor       (天数智芯)     [未文档化]
    └─ cpu::Descriptor         (x86 CPU)      [未文档化]
    ↓
设备内核层
    - CUDA/MUSA 内核启动
    - CANN/Ascend 算子调用
    - CPU SIMD 优化路径
```

### 3.2 数据流与并行策略

#### 并行维度映射

基于 nvidia 和 moore 实现的分析，RoPE 采用**三维并行策略**：

```
输入张量形状: [batch, seqlen, nhead, dhead]
                            ↓ 并行划分 ↓
Grid 配置: dim3(seqlen, nhead, batch)
    ├─ blockIdx.x → 序列位置索引 (每个 block 处理一个 token)
    ├─ blockIdx.y → 注意力头索引 (每个 block 处理一个 head)
    └─ blockIdx.z → 批次索引 (每个 block 处理一个样本)

Block 内部:
    └─ threadIdx.x → 旋转维度索引 (并行处理 dhead/2 个角度对)
```

**并行度**: 总共启动 `batch × seqlen × nhead` 个线程块，完全独立执行，无需同步。

#### 内存访问模式

```
位置 ID 查找:
    pos_ids[batch_idx × pos_stride_batch + seq_idx]
            ↓
    sin/cos_table[pos_id × table_dim + angle_idx]
            ↓
    加载旋转角度 (sin__, cos__)

输入/输出访问:
    输入偏移 = batch_idx × x_stride_batch +
               seq_idx × x_stride_seqlen +
               head_idx × x_stride_nhead + elem_idx

    输出偏移 = batch_idx × y_stride_batch +
               seq_idx × y_stride_seqlen +
               head_idx × y_stride_nhead + elem_idx
            ↓
    旋转变换:
    (GPT-J)    y[2i]   =  cos(θ) × x[2i]   - sin(θ) × x[2i+1]
               y[2i+1] =  sin(θ) × x[2i]   + cos(θ) × x[2i+1]

    (GPT-NeoX) y[i]             =  cos(θ) × x[i] - sin(θ) × x[i+table_dim]
               y[i+table_dim]   =  sin(θ) × x[i] + cos(θ) × x[i+table_dim]
            ↓
    写入输出张量
```

**关键特性**:
- **步长支持**: 通过 `*_stride_*` 参数支持非连续内存布局（如转置张量）
- **查找表重用**: 同一位置 ID 的所有线程读取相同的 sin/cos 值，可通过 shared memory 缓存优化（当前未实现）
- **合并访问**: 输入/输出线性访问模式有利于内存合并（coalescing）

### 3.3 算法变体选择

RoPE 支持两种主要的旋转算法，通过 `algo` 参数在描述符创建时指定：

#### GPT-J 模式 (`INFINIOP_ROPE_ALGO_GPT_J`)
```
向量布局: [x0, x1, x2, x3, ..., x_{d-2}, x_{d-1}]
旋转对:   (x0, x1), (x2, x3), ..., (x_{d-2}, x_{d-1})
内存访问: 相邻元素，缓存友好
向量化:   使用 half2/bfloat162 指令一次处理一对
适用模型: GPT-J, LLaMA 系列
```

#### GPT-NeoX 模式 (`INFINIOP_ROPE_ALGO_GPT_NEOX` 或标准模式)
```
向量布局: [x0, x1, ..., x_{table_dim-1} | x_{table_dim}, ..., x_{dhead-1}]
旋转对:   (x0, x_{table_dim}), (x1, x_{table_dim+1}), ...
内存访问: 跨越半个维度，非连续访问
向量化:   需要两次独立加载，性能略低于 GPT-J
适用模型: GPT-NeoX, BLOOM, ChatGLM
```

### 3.4 类型分发与编译优化

#### 双重类型分发机制

为支持 **4 种数据类型** (FP16/BF16/FP32/FP64) × **8 种位置 ID 类型** (u8/u16/u32/u64/i8/i16/i32/i64) = **32 种组合**，系统采用宏生成特化代码：

```cpp
// 外层分发: 数据类型
switch (data_type) {
    case INFINI_DTYPE_F16:  ROPE_TYPE(half); break;
    case INFINI_DTYPE_BF16: ROPE_TYPE(cuda_bfloat16); break;
    case INFINI_DTYPE_F32:  ROPE_TYPE(float); break;
    case INFINI_DTYPE_F64:  ROPE_TYPE(double); break;
}

// 内层分发: 位置 ID 类型
#define ROPE_TYPE(TDATA) \
    switch (pos_type) {
        case INFINI_DTYPE_U8:  CALCULATE_ROPE(TDATA, uint8_t); break;
        case INFINI_DTYPE_I32: CALCULATE_ROPE(TDATA, int32_t); break;
        // ... 其他 6 种整数类型
    }
```

**编译期优化**:
- **模板特化**: 为每种组合生成独立的内核实例，避免运行期类型判断
- **`if constexpr`**: 在编译期选择 GPT-J 或 GPT-NeoX 分支，生成无分支代码
- **向量化指令**: FP16 使用 `half2`，BF16 使用 `cuda_bfloat162`，吞吐量提升 2x

### 3.5 跨平台适配差异

#### Moore/MUSA 平台特殊处理

Moore 后端相对于 NVIDIA 实现的关键适配点：

1. **类型别名映射**:
   ```cpp
   // CUDA → MUSA
   cuda_bfloat16     → mt_bfloat16
   cuda_bfloat162    → mt_bfloat162
   ```

2. **内置函数替换**:
   ```cpp
   // CUDA 原生
   __low2bfloat16()  / __high2bfloat16()  // 提取 bfloat162 的分量

   // MUSA 替代
   __low2float()     / __high2float()     // 转换为 float
   __floats2bfloat162_rn()                // 重新打包（舍入到最近值）
   ```

3. **数学库适配**:
   ```cpp
   // 解决 MUSA 的 exp() 类型歧义
   template <typename T>
   __host__ __device__ T exp_(T x) { return expf(x); }  // float 特化
   template <>
   __host__ __device__ double exp_<double>(double x) { return exp(x); }
   ```

### 3.6 性能特征

#### 计算复杂度
- **理论复杂度**: O(N)，其中 N = batch × seqlen × nhead × table_dim
- **实际并行度**: O(N / blockDim)，每个 block 处理一个 (seq, head, batch) 元素

#### 内存带宽需求
```
每个线程的内存访问量:
- 读取: 2 个输入元素 + 2 个查找表元素 = 4 × sizeof(Tdata)
- 写入: 2 个输出元素 = 2 × sizeof(Tdata)

总数据量: 6 × sizeof(Tdata) × N
```

#### 优化机会
1. **Shared Memory 缓存**: sin/cos 表可在 block 内共享（当前未实现）
2. **向量化加载**: GPT-J 模式下已使用 half2/bfloat162，GPT-NeoX 可优化
3. **多流并行**: 不同 batch 可在不同 stream 上并行执行

### 3.7 统一接口设计

#### 描述符创建流程

```cpp
// 1. 张量验证（所有后端共享）
RoPEInfo::createRoPEInfo()
    ├─ 检查维度数量 (3D 或 4D)
    ├─ 验证数据类型 (浮点张量 + 整数位置 IDs)
    ├─ 检查形状约束 (dhead = table_dim × 2)
    ├─ 验证步长连续性 (最后一维 stride = 1)
    └─ 提取形状信息到 RoPEInfo 结构

// 2. 后端描述符实例化
Backend::Descriptor::create()
    ├─ 调用 createRoPEInfo() 验证参数
    ├─ 创建 Backend::Opaque (持有设备句柄)
    ├─ 存储 RoPEInfo 到 _info 成员
    └─ 计算 workspace_size (当前为 0)
```

#### 执行接口

```cpp
Backend::Descriptor::calculate()
    ├─ 根据 data_type 和 pos_type 双重分发
    ├─ 提取设备能力 (maxThreadsPerBlock)
    ├─ 构建 grid 配置: dim3(seqlen, nhead, batch)
    ├─ 构建 block 配置: max(table_dim, maxThreads)
    └─ 启动内核 <<<grid, block, 0, stream>>>()
```

**跨平台一致性**: 所有后端实现相同的 `Descriptor` 接口（通过 `DESCRIPTOR(backend)` 宏生成），用户代码无需修改即可切换硬件平台。

## 4. 待文档化后端分析

当前仅有 **nvidia** 和 **moore** 两个后端生成了详细文档，其余 6 个硬件后端的实现细节未知：

| 后端目录 | 硬件平台 | 预期特性 | 文档状态 |
|---------|---------|---------|---------|
| `ascend` | 华为昇腾 NPU | 可能使用 CANN ACL 接口，支持 NPCC/Vector 加速 | ❌ 缺失 |
| `bang` | 寒武纪 MLU | 可能使用 BANG 语言或 CNRT 接口 | ❌ 缺失 |
| `cpu` | x86 CPU | 可能使用 AVX-512/AVX2 SIMD 指令优化 | ❌ 缺失 |
| `cuda` | 通用 CUDA | 可能与 nvidia 目录功能重叠或为旧版实现 | ❌ 缺失 |
| `kunlun` | 百度昆仑 | 可能使用 XPU KUNLUN 接口 | ❌ 缺失 |
| `metax` | 天数智芯 | 可能使用 MetaX 扩展 CUDA 接口 | ❌ 缺失 |

**建议**: 为保持架构完整性，应补充这些后端的 CODEREADME.md 文档，重点关注其与 NVIDIA 实现的差异点（如 API 映射、性能优化策略、硬件特性利用）。
