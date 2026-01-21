# ntops 架构全景

## 1. 子系统职责

ntops (Neural Tensor Operations) 是基于 ninetoothed 框架构建的高性能深度学习算子库，位于 Infini 生态系统的计算基础设施层。该目录 `ntops/src/ntops` 作为核心实现节点，通过双层架构设计提供从底层 CUDA 内核到上层 PyTorch 绑定的完整算子实现。

**核心职责**：
- **kernels 层**：实现 38 个基础 CUDA 计算内核，涵盖线性代数、注意力机制、归一化、激活函数等核心算子，基于 ninetoothed 的编译时张量抽象提供高度优化的内存分块和并行计算策略
- **torch 绑定层**：封装底层内核为 PyTorch 兼容的 Python API，提供内核编译缓存、自动调优配置、精度模式检测等运行时优化机制

**架构定位**：ntops 在 Infini 生态中充当算子加速层，向上为 InfiniLM、InfiniTrain 等上层框架提供高性能张量运算支持，向下依赖 ninetoothed 框架进行 CUDA 代码生成和性能优化。

## 2. 模块导航 (Module Navigation)

* **kernels** (核心计算内核层)：
    * *功能*：基于 ninetoothed DSL 实现的 38 个底层 CUDA 计算内核，采用 arrangement → application 两阶段设计模式，提供 Flash Attention、Layer/RMS Normalization、矩阵乘法、逐元素运算等深度学习核心算子
    * *职责*：定义计算逻辑的内存排布策略和具体计算实现，通过编译时分块（tiling）和在线算法优化，为上层提供高性能张量运算原语

* **torch** (PyTorch 绑定接口层)：
    * *功能*：封装 39 个 PyTorch 兼容的张量操作函数，实现内核编译缓存（_cached_make）、全局配置管理（num_warps/num_stages）、矩阵乘法精度检测（TF32/IEEE）等运行时优化机制，支持 GQA、KV 缓存、因果掩码等高级特性
    * *职责*：提供用户友好的 PyTorch API，管理内核编译生命周期和性能调优参数，充当 ninetoothed 内核与 PyTorch 框架之间的适配器

## 3. 架构逻辑图解

### 数据流与交互关系

```
用户代码 (PyTorch)
       ↓
ntops.torch 绑定层
    ├─ 参数验证与标准化
    ├─ 输出张量预分配 (torch.empty_like)
    ├─ 查询内核编译缓存 (_cached_make)
    │   └─ 缓存未命中 → 调用 ninetoothed.make 编译
    └─ 调用 ninetoothed 生成的 CUDA 内核
       ↓
ninetoothed 运行时
    ├─ 加载编译好的 PTX 代码
    ├─ 管理 CUDA 线程块调度
    └─ 执行计算内核
       ↓
ntops.kernels 内核实现
    ├─ arrangement 阶段：内存分块与重排 (tile/expand/squeeze)
    └─ application 阶段：实际计算逻辑 (ntl.dot/ntl.sum/ntl.where)
       ↓
GPU 硬件 (CUDA)
```

### 核心交互流程

**1. 内核编译生命周期**：
- torch 层接收函数调用（如 `scaled_dot_product_attention`）
- 提取张量元数据（ndim/dtype/shape），构造内核配置参数
- 调用 `_cached_make(premake_args, num_warps, num_stages, max_num_configs)`
- functools.cache 检查参数哈希，若命中则直接返回已编译内核
- 未命中时调用 `ntops.kernels.<operator>.premake` 生成 arrangement/application 函数
- 传入 `ninetoothed.make` 编译为 CUDA PTX 代码（耗时约秒级）
- 编译结果缓存至 functools.cache，后续调用直接复用

**2. 计算执行流程**：
- torch 层预分配输出张量（避免内核执行期间动态分配）
- 调用 arrangement 函数对输入/输出张量进行内存重排
  - 将大张量按 block_size 分块（如 64×64）
  - 通过 tile/expand/squeeze 优化内存访问模式
  - 计算张量在源缓冲区的 offsets（用于边界检查）
- 调用 application 函数执行计算逻辑
  - ninetoothed 自动分配 CUDA 线程网格（每个块处理一个 tile）
  - 执行编译时 DSL 描述的计算（ntl.dot/ntl.sum/ntl.where）
  - 写入输出张量的对应 tile

**3. 性能优化协同**：
- **内存局部性**：kernels 层的 tiling 策略将大矩阵分解为适合 L2 Cache 的小块，减少全局内存访问
- **精度控制**：torch 层的 `_get_matmul_input_precision` 检测 PyTorch 设置，在 Ampere GPU 上使用 TF32 加速矩阵乘法（吞吐量提升 8x）
- **流水线并行**：torch 层的 num_stages 参数控制 CUDA pipeline 深度，隐藏全局内存延迟（默认 2-4 阶段）
- **自动调优**：ninetoothed 框架根据 max_num_configs 生成多个内核变体（不同 warp 数/分块大小），运行时基准测试选择最优配置
- **零拷贝广播**：torch 层对 weight/bias 使用 expand_as 创建视图而非复制，节省内存带宽

**4. 高级特性支持**：

**Flash Attention (scaled_dot_product_attention)**：
- kernels 层实现在线 softmax 算法（Welford 变体），避免存储完整的注意力矩阵（从 O(N²) 内存降至 O(N×block_size)）
- 支持因果掩码（UPPER_LEFT/LOWER_RIGHT），通过比较 query/key 的 offsets 实现三角掩码
- torch 层扩展至 GQA（分组查询注意力），允许多个查询头共享键值头（num_heads_q % num_heads_kv == 0）

**KV 缓存**：
- torch 层接收 present_key/present_value 缓存张量和 slot 参数
- 选择带 KV 缓存的内核变体，将新生成的 key/value 写入缓存指定位置
- 支持自回归生成场景（如 GPT 推理），避免重复计算历史 token 的 attention

**旋转位置编码 (RoPE)**：
- kernels 层支持 interleaved（sin/cos 交替）和 non-interleaved（分块存储）两种模式
- 使用 2D 分块和 dilation/strides 实现高效内存访问
- torch 层通过 `[None, :, None, :]` 技巧广播 sin/cos 表至 batch/heads 维度

**归一化层优化**：
- LayerNorm/RMSNorm 使用两遍扫描算法：第一遍计算均值/方差，第二遍应用归一化
- 内部使用 float32 累加器保证精度，最终转换回输出 dtype（如 float16）
- RMSNorm 省略均值计算，减少 30% 计算量（LLaMA 等 LLM 常用）

### 模块依赖关系

**kernels 层依赖**：
- ninetoothed 核心：Tensor 抽象、tile/offsets/squeeze 等 API
- ninetoothed.language (ntl)：编译时 DSL（ntl.dot/ntl.sum/ntl.where/ntl.cast）
- Python 标准库：functools.partial（函数柯里化）、math（数学常量）、enum（枚举类）

**torch 层依赖**：
- ntops.kernels：所有底层内核实现（38 个函数）
- ninetoothed.make：内核编译入口
- torch：张量创建（torch.empty_like/expand_as）、dtype 检测（torch.finfo）、全局设置（torch.get_float32_matmul_precision）
- functools.cache：内核编译缓存

**关键设计模式**：
- **Template Method**：kernels 层的 arrangement → application 两阶段设计，分离数据布局和计算逻辑
- **Strategy**：scaled_dot_product_attention 根据 with_kv_cache/enable_gqa 选择不同内核策略
- **Facade**：torch 层的 matmul 函数隐藏 mm/bmm 的分发逻辑
- **Singleton**：_cached_make_default_config 全局配置单例
- **Cache Aside**：_cached_make 先查 functools.cache，未命中则编译并缓存

**性能关键路径**：
用户调用 → torch 层参数验证 → 查询编译缓存 → 首次调用触发 ninetoothed.make 编译（秒级开销）→ arrangement 内存重排 → application 并行计算 → GPU 执行 → 返回结果。后续调用直接复用编译缓存，仅需微秒级的内核启动开销。
