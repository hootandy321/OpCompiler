# 目录: source 架构全景

## 1. 子系统职责

`source` 目录是 ninetoothed 项目的 **Sphinx 文档源码中心**，承担着将代码自动文档化并生成可读文档的核心职责。该目录通过 Sphinx 框架组织，采用 reStructuredText 格式编写文档，并配置自动化 API 文档生成机制，将 Python 代码的 docstring 自动转化为结构化的 API 参考手册。

在整个系统架构中，`source` 是文档构建的输入层，它定义了文档的结构、内容和生成逻辑，通过 `conf.py` 配置文件驱动 Sphinx 生成最终的 HTML 文档网站。

## 2. 模块导航

* **python_api**:
    * *功能*: Python API 自动化文档生成目录，包含 ninetoothed 库的各个功能模块的 API 参考文档。该目录通过 Sphinx 的自动摘要（autosummary）和自动类（autoclass）指令，从 Python 源代码中提取文档字符串并生成标准化的 API 文档页面。
    * *职责*: 组织并生成 ninetoothed 的核心 API 文档，覆盖符号定义、张量操作、代码生成、调试工具和可视化功能。

* **_static**:
    * *功能*: 静态资源目录，包含文档网站所需的静态文件，如项目标志图片等。
    * *职责*: 提供文档渲染所需的静态媒体资源支持。

## 3. 架构逻辑图解

`source` 目录的文档架构呈现出一个 **层次化的模块组织结构**，反映了 ninetoothed 库的功能划分：

### 文档生成流程
1. **配置层** (`conf.py`): 定义 Sphinx 扩展、主题、路径配置等文档生成规则。
2. **入口层** (`index.rst`): 文档网站的主页，通过 toctree 指令聚合各个子文档。
3. **内容层**:
   - `basics.rst`: 提供基础概念和使用教程（14373 字节，是最大的文档文件）。
   - `installation.rst`: 安装指南。
   - `python_api.rst`: Python API 文档的入口，指向 `python_api/` 子目录。

### Python API 模块结构

`python_api/` 目录按功能划分为五个核心模块，形成 ninetoothed 的完整 API 图谱：

#### 1. **Symbol (符号模块)** - `symbol.rst`
- 定义 `ninetoothed.Symbol` 类
- 这是符号计算的基础抽象层，用于表示张量中的符号变量。

#### 2. **Tensor (张量模块)** - `tensor.rst`
- 核心类：`ninetoothed.Tensor`
- **元操作（Meta-Operations）**：张量形状变换的高级操作
  - `tile`: 张量平铺/重复
  - `expand`: 维度扩展
  - `unsqueeze`/`squeeze`: 维度增减
  - `permute`: 维度排列
  - `flatten`/`ravel`: 展平操作
- **求值（Evaluation）**：符号计算求值
  - `eval`: 计算张量表达式的值
  - `subs`: 符号替换

#### 3. **Code Generation (代码生成模块)** - `code_generation.rst`
- 包含三个核心代码生成函数：
  - `ninetoothed.build`: 构建编译
  - `ninetoothed.jit`: 即时编译
  - `ninetoothed.make`: 制作可执行单元
- 这些函数将高层张量描述转化为底层可执行代码。

#### 4. **Debugging (调试模块)** - `debugging.rst`
- **可选模块**，需要通过 `pip install ninetoothed[debugging]` 单独安装
- 提供调试工具：
  - `ninetoothed.debugging.simulate_arrangement`: 模拟张量排列布局，用于验证内存访问模式

#### 5. **Visualization (可视化模块)** - `visualization.rst`
- **可选模块**，需要通过 `pip install ninetoothed[visualization]` 单独安装
- 提供两类可视化功能：
  - **`visualize`**: 可视化单个张量的内存布局
    - 支持临时显示和保存到文件
    - 支持自定义颜色（Matplotlib 颜色格式）
    - 可批量可视化多个张量
  - **`visualize_arrangement`**: 可视化张量排列的内存布局
    - 需要 CUDA 环境
    - 要求排列和张量参数中不能包含符号，只能是具体值
    - 通过 `functools.partial` 绑定具体参数来构造无符号的排列

### 数据流与依赖关系

```
用户代码
    ↓
Symbol & Tensor 定义
    ↓
Code Generation (build/jit/make)
    ↓
[可选] Debugging.simulate_arrangement 验证布局
    ↓
[可选] Visualization.visualize_arrangement 可视化结果
```

**模块间协作**：
- `Symbol` 和 `Tensor` 是基础抽象层
- `Code Generation` 使用这些抽象生成可执行代码
- `Debugging` 和 `Visualization` 作为辅助工具，帮助开发者理解内存布局和调试性能问题
- 三个可选模块（debugging、visualization）通过 extras_require 机制独立安装，保持核心依赖的精简

### 辅助脚本

- `visualize.py`: 一个 Python 脚本（6064 字节），可能用于自动化可视化流程或批量生成可视化图像。

这个架构设计清晰地体现了 ninetoothed 作为 **张量计算 DSL（领域特定语言）** 的定位，通过符号计算、自动代码生成、调试可视化工具链，提供了从高层描述到底层实现的完整开发体验。
