# test/ 目录架构全景

## 1. 子系统职责

`test/` 目录是 InfiniCore 的综合测试验证层，承载着整个框架的质量保证使命。该目录通过三个互补的测试模块，构建了完整的测试金字塔：

- **核心功能验证**（infinicore/）：对 InfiniCore 的 Python API 进行端到端测试，覆盖张量操作、算子、图执行、神经网络层等核心功能
- **算子性能基准**（infiniop/）：提供基于 GGUF 格式的标准测例生成框架，用于对不同硬件后端（CPU/CUDA 等）的算子实现进行性能和正确性验证
- **自动化测试套件**（infiniop-test/）：C++ 实现的测试执行引擎，负责加载 GGUF 测例文件，执行可配置的预热和性能测试循环

这三个模块共同确保了从高层 API 到底层算子的全栈正确性与性能。

## 2. 模块导航

### 2.1 核心测试模块

* **infinicore/**: *InfiniCore Python API 综合测试套件*
  * *功能*: 提供对 InfiniCore 框架的全面 Python 测试覆盖，包括基础张量操作、算子库（ops/）、神经网络层（nn/）、计算图（graph/）、张量系统（tensor/）、设备事件管理（device_event.py）、框架集成（framework/）等核心组件的单元测试和集成测试
  * *职责*: 验证 InfiniCore 在 Python 层面的 API 正确性、功能完整性和易用性
  * *文档状态*: 文档缺失

* **infiniop/**: *GGUF 测例生成工具集*
  * *功能*: 包含 30+ 个 Python 脚本，用于生成各类算子的标准测例数据并打包为 GGUF 格式。涵盖基础算子（add, sub, mul）、神经网络算子（gelu, relu, sigmoid, silu, swiglu, softmax, layer_norm, rms_norm）、矩阵运算（gemm, scaled_mm_int8）、注意力机制（attention, paged_attention, causal_softmax）、位置编码（rope）、随机采样（random_sample, topksoftmax, topkrouter）、卷积（conv）、量化操作（dequantize_awq）等
  * *职责*: 为 infiniop-test 测试程序生成标准化的、包含输入数据和预期输出的测例文件，确保跨硬件后端的一致性验证
  * *文档状态*: 文档缺失

* **infiniop-test/**: *InfiniOP 性能测试执行引擎*
  * *功能*: 提供基于 GGUF 格式的测例加载与执行框架。支持在 CPU、CUDA 等不同硬件后端上运行测试，提供可配置的预热次数和测试循环次数。定义了严格的 GGUF 文件格式规范，包括元数据（test_count、操作名称、步长参数、alpha/beta 系数）和张量数据（输入、输出、答案）的存储方式。支持零步长（zero-stride）广播测试等高级特性
  * *职责*: 作为算子性能和正确性的终极验证工具，加载由 infiniop/ 生成的测例，执行实际的硬件测试并输出结果
  * *文档状态*: 已有文档（README.md），描述了测例生成流程、运行方式和 GGUF 格式规范

## 3. 架构逻辑图解

```
┌─────────────────────────────────────────────────────────────────┐
│                     InfiniCore 测试生态系统                       │
└─────────────────────────────────────────────────────────────────┘

                        ┌─────────────────┐
                        │   开发者/CI     │
                        └────────┬────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           │                     │                     │
           ▼                     ▼                     ▼
    ┌──────────────┐    ┌──────────────┐      ┌──────────────┐
    │  infinicore  │    │   infiniop   │      │ infiniop-test│
    │   Python API │    │ 测例生成器   │      │   C++引擎    │
    └──────┬───────┘    └──────┬───────┘      └──────┬───────┘
           │                   │                      │
           │                   │                      │
           ▼                   ▼                      ▼
    ┌──────────────┐    ┌──────────────┐      ┌──────────────┐
    │ InfiniCore   │    │  GGUF 文件   │      │  硬件后端    │
    │ 框架功能测试 │    │  (.gguf)     │      │  CPU/CUDA/   │
    └──────────────┘    └──────────────┘      │  其他加速器  │
                       ┌──────┴───────┐      └──────────────┘
                       │              │
                       ▼              │
                ┌──────────────┐     │
                │ infiniop-test│◄────┘
                │ 加载测例并   │
                │ 执行测试     │
                └──────────────┘
```

### 3.1 测试数据流

**Python API 测试流程（infinicore/）**：
1. 测试脚本（如 test.py, run.py, debug.py）导入 InfiniCore Python API
2. 创建张量、构建计算图、配置算子参数
3. 执行计算并验证结果正确性
4. 覆盖框架层、算子层、图层的功能验证

**算子性能测试流程（infiniop/ → infiniop-test/）**：
1. **测例生成阶段**（infiniop/）：
   - Python 脚本（如 gemm.py, attention.py）生成特定算子的测试数据
   - 构造输入张量（a, b）、参数（alpha, beta, strides）
   - 计算预期答案（ans）
   - 将元数据和张量打包为 GGUF 格式文件

2. **测试执行阶段**（infiniop-test/）：
   - C++ 测试程序加载 GGUF 文件
   - 解析元数据（操作类型、张量形状、步长、参数）
   - 在指定硬件后端（CPU/CUDA）上执行算子
   - 对比实际输出与预期答案
   - 执行预热循环（warmup）和性能测试循环（run）
   - 输出测试结果和性能指标

### 3.2 关键依赖关系

- **infiniop/ → infiniop-test/**：infiniop/ 生成的 GGUF 文件是 infiniop-test/ 的输入。两者必须严格遵守 GGUF 格式规范，包括元数据键名（test_count、test.[id].op_name、test.[id].*.strides）、张量命名（test.[id].{a,b,c,ans}）和数据类型约定
- **infinicore/ ↔ InfiniCore**：infinicore/ 测试直接依赖 InfiniCore 的 Python 绑定，测试框架的稳定性依赖于 API 的向后兼容性
- **跨硬件一致性**：infiniop/ 生成硬件无关的测例，infiniop-test/ 可在不同硬件后端上执行同一测例，确保跨平台行为一致性

### 3.3 测试覆盖策略

- **功能正确性**：infinicore/ 通过单元测试验证每个 API 的功能实现是否符合预期
- **算子精度**：infiniop/ + infiniop-test/ 通过参考实现（如 NumPy）生成答案，对比硬件后端的计算结果
- **性能基准**：infiniop-test/ 支持多次预热和测试循环，可用于性能回归测试和优化验证
- **边界条件**：GGUF 测例支持零步长广播、非连续张量、各种数据类型等边界情况测试
- **算子覆盖**：infiniop/ 包含 30+ 个算子测例生成器，覆盖矩阵乘、注意力、归一化、激活函数、量化等关键操作

### 3.4 质量保证机制

整个测试系统形成三道防线：
1. **Python 层测试**（infinicore/）：快速反馈，开发阶段捕获 API 层问题
2. **算子正确性验证**（infiniop/ + infiniop-test/）：确保底层算子实现的数值精度
3. **跨硬件一致性检查**：通过同一测例在不同后端运行，保证行为一致

这种分层测试架构确保了从高层 API 到底层算子的全方位质量保证。
