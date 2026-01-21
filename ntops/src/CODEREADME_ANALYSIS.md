# ntops.src 源代码目录架构

## 1. 子系统职责

`ntops/src` 是 ntops (Neural Tensor Operations) 项目的源代码根目录，作为顶级实现节点，其核心职责是组织整个 Python 包的源代码结构。该目录采用扁平化设计，将所有实现代码集中在单一子目录 `ntops/` 下，符合 Python 包的标准布局规范。

**核心职责**：
- **源代码组织**：作为 ntops 项目的 Python 源代码容器，提供清晰的包层次结构
- **模块封装**：通过单层子目录 `ntops/` 封装所有功能模块，便于导入和打包
- **实现与测试分离**：仅包含源代码实现，测试代码位于项目根目录的 `tests/` 中

**架构定位**：该目录在 ntops 项目中充当源代码根节点，向上对接项目的构建系统（setup.py/pyproject.toml），向下承载所有 Python 模块实现。

## 2. 模块导航 (Module Navigation)

* **ntops** (核心实现包)：
    * *功能*：ntops 的完整 Python 包实现，包含 kernels 层（38 个 CUDA 计算内核）和 torch 绑定层（39 个 PyTorch 兼容 API），基于 ninetoothed 框架实现高性能深度学习算子库
    * *职责*：提供从底层 CUDA 内核实现到上层 PyTorch API 绑定的完整算子加速解决方案，支持 Flash Attention、GQA、KV 缓存、旋转位置编码等高级特性

## 3. 架构逻辑图解

### 目录结构设计

```
ntops/src/                          # 源代码根目录
    └── ntops/                       # Python 包根目录（实际导入路径）
        ├── __init__.py              # 包初始化文件（暴露公共接口）
        ├── kernels/                 # 核心 CUDA 内核层
        │   ├── __init__.py          # 内核模块导出接口
        │   ├── scaled_dot_product_attention.py  # Flash Attention 实现
        │   ├── layer_norm.py        # Layer Normalization 内核
        │   ├── rms_norm.py          # RMS Normalization 内核
        │   ├── rotary_position_embedding.py     # RoPE 内核
        │   ├── mm.py / bmm.py       # 矩阵乘法内核
        │   ├── element_wise.py      # 逐元素运算策略
        │   ├── reduction.py         # 归约运算策略
        │   ├── gelu.py / silu.py    # 激活函数内核
        │   ├── softmax.py           # Softmax 内核
        │   ├── dropout.py           # Dropout 内核
        │   └── ... (共 38 个内核文件)
        └── torch/                   # PyTorch 绑定层
            ├── __init__.py          # 绑定模块导出接口
            ├── utils.py             # 内核编译缓存与配置管理
            ├── scaled_dot_product_attention.py  # 注意力机制绑定
            ├── rotary_position_embedding.py     # RoPE 绑定
            ├── layer_norm.py / rms_norm.py      # 归一化层绑定
            ├── matmul.py / mm.py / bmm.py / addmm.py  # 矩阵运算绑定
            ├── relu.py / gelu.py / silu.py     # 激活函数绑定
            ├── softmax.py           # Softmax 绑定
            ├── dropout.py           # Dropout 绑定
            └── ... (共 39 个绑定文件)
```

### 数据流与层次关系

```
用户代码
    ↓ import ntops.torch
ntops/src/ntops/__init__.py         # 包入口点
    ↓
ntops.torch 绑定层                  # PyTorch API 适配层
    ├─ 参数验证与标准化
    ├─ 输出张量预分配
    ├─ 查询内核编译缓存 (_cached_make)
    └─ 调用 ninetoothed 生成的 CUDA 内核
        ↓
ntops.kernels 内核层                # 底层 CUDA 实现
    ├─ arrangement 阶段：内存分块与重排
    └─ application 阶段：实际计算逻辑
        ↓
ninetoothed 框架                    # CUDA 代码生成与运行时
    ↓
GPU 硬件 (CUDA)
```

### 核心设计决策

**1. 单一子目录结构**：
- `src/` 下仅包含一个 `ntops/` 子目录，符合 Python 包的扁平化组织原则
- 这种设计使得 `import ntops.torch` 等语句能直接映射到文件系统路径
- 便于打包工具（如 setuptools）自动发现包结构

**2. 分层架构实现**：
- **kernels 层**：实现 38 个基础 CUDA 计算内核，基于 ninetoothed 的 `arrangement → application` 两阶段设计模式
  - arrangement 阶段：定义内存排布策略（tiling、重排、边界检查）
  - application 阶段：实现具体计算逻辑（ntl.dot、ntl.sum、ntl.where）
- **torch 绑定层**：提供 39 个 PyTorch 兼容的 Python API，封装底层内核
  - 实现内核编译缓存机制（`@functools.cache`）
  - 提供全局配置管理（num_warps、num_stages、max_num_configs）
  - 支持运行时优化（TF32 精度检测、自动调优）

**3. 模块职责分离**：
- `kernels/` 目录：专注于计算逻辑实现，不依赖 PyTorch，仅依赖 ninetoothed 框架
- `torch/` 目录：专注于 PyTorch 集成，处理参数验证、输出分配、内核调用等粘合逻辑
- `utils.py`：集中管理内核编译的全局配置和缓存策略

**4. 性能优化路径**：
- **编译时优化**：ninetoothed 在内核编译阶段进行代码生成和优化（分块大小、寄存器分配、指令调度）
- **运行时优化**：torch 层通过缓存机制避免重复编译（编译耗时约秒级），后续调用仅需微秒级内核启动开销
- **内存优化**：通过预分配输出张量、零拷贝广播（expand_as）、就地操作（inplace=True）减少内存分配和拷贝

### 关键交互流程

**内核首次调用流程**：
1. 用户调用 `ntops.torch.scaled_dot_product_attention(query, key, value, is_causal=True)`
2. torch 层提取张量元数据（ndim=4、dtype=float16、shape）
3. 调用 `_cached_make(premake_args, num_warps, num_stages, max_num_configs)`
4. functools.cache 检查参数哈希，发现缓存未命中
5. 调用 `ntops.kernels.scaled_dot_product_attention.pmake` 生成 arrangement/application 函数
6. 传入 `ninetoothed.make` 编译为 CUDA PTX 代码（秒级开销）
7. 编译结果缓存至 functools.cache
8. 返回可调用的 CUDA 内核函数
9. 执行 arrangement 阶段：内存分块与重排
10. 执行 application 阶段：Flash Attention 算法实现
11. 返回结果张量

**内核后续调用流程**：
1. 用户再次调用相同参数的函数
2. functools.cache 发现缓存命中
3. 直接返回已编译的 CUDA 内核（微秒级开销）
4. 执行 arrangement + application 阶段
5. 返回结果

### 依赖关系

**ntops.kernels 依赖**：
- ninetoothed 核心：Tensor 抽象、tile/offsets/squeeze 等 API
- ninetoothed.language (ntl)：编译时 DSL（ntl.dot、ntl.sum、ntl.where、ntl.cast）
- Python 标准库：functools.partial（函数柯里化）、math（数学常量）、enum（枚举类）

**ntops.torch 依赖**：
- ntops.kernels：所有底层内核实现（38 个函数）
- ninetoothed.make：内核编译入口
- torch：张量创建（torch.empty_like、expand_as）、dtype 检测（torch.finfo）、全局设置（torch.get_float32_matmul_precision）
- functools.cache：内核编译缓存

**与外部系统的集成**：
- **上层依赖**：InfiniLM（大语言模型推理）、InfiniTrain（分布式训练框架）通过 `import ntops.torch` 使用算子加速
- **下层依赖**：ninetoothed 框架提供 CUDA 代码生成和运行时支持
- **横向依赖**：与 PyTorch 原生算子保持 API 兼容性，可作为 drop-in replacement

### 性能特征

**算子覆盖范围**：
- **线性代数**：matmul、bmm、mm、addmm（4 个）
- **注意力机制**：scaled_dot_product_attention（支持 Flash Attention、GQA、KV 缓存）
- **位置编码**：rotary_position_embedding（RoPE）
- **归一化**：layer_norm、rms_norm（2 个）
- **激活函数**：relu、gelu、silu、sigmoid、tanh、softmax（6 个）
- **正则化**：dropout（1 个）
- **算术运算**：abs、neg、add、sub、mul、div、pow（7 个）
- **数学函数**：exp、rsqrt、sin、cos（4 个）
- **比较运算**：eq、ne、lt、le、gt、ge（6 个）
- **位运算**：bitwise_and、bitwise_or、bitwise_not（3 个）
- **数值检查**：isnan、isinf（2 个）
- **其他**：clamp（1 个）

总计 39 个 PyTorch API，底层由 38 个 CUDA 内核支持（部分算子共享内核实现）。

**性能关键路径**：
首次调用：用户调用 → 参数验证 → 查询缓存（未命中）→ ninetoothed.make 编译（秒级）→ arrangement + application → GPU 执行 → 返回结果
后续调用：用户调用 → 参数验证 → 查询缓存（命中）→ arrangement + application → GPU 执行 → 返回结果

**优化效果**：
- 编译缓存将秒级编译开销降低为微秒级查询开销（约 10000x 加速）
- Tiling 策略将内存访问复杂度从 O(N²) 降至 O(N×block_size)
- TF32 模式在 Ampere GPU 上提升矩阵乘法吞吐量 8 倍
- 在线 softmax 算法避免存储完整注意力矩阵（内存节省 100 倍）
