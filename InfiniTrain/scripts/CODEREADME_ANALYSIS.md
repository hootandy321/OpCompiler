# scripts 目录架构全景

## 1. 子系统职责

`scripts` 目录是 InfiniTrain 项目的**工程化支撑层**，负责提供代码质量保障、自动化测试执行和性能数据管理三大核心能力。该目录不包含核心训练逻辑，而是提供了一套完整的 DevOps 工具链，支撑 InfiniTrain 的开发、测试、性能分析和结果追踪全流程。

作为 InfiniTrain 体系的基础设施层，该目录通过以下方式服务于整体系统：
- **代码质量保障**：通过自动化格式化工具维护多语言代码规范
- **自动化测试编排**：通过配置驱动的测试框架执行多模型、多并行的性能测试
- **数据可视化集成**：通过飞书表格 API 实现性能数据的自动化记录与展示

## 2. 模块导航

### 核心脚本文件

* **format.py**
    * *功能*: 多语言代码格式化工具，支持 C/C++/CUDA/Python 的自动格式检查与修复
    * *职责*: 统一代码风格，通过 clang-format 和 black 维护代码质量标准；支持 Git 增量格式化和指定路径格式化

* **run_models_and_profile.bash**
    * *功能*: 配置驱动的自动化测试执行引擎，支持多构建配置、多测试场景的批量运行
    * *职责*: 根据 JSON 配置文件执行 CMake 构建、运行 GPT2/Llama3 模型训练、收集性能分析数据（支持 profiling 模式）

* **test_config.json**
    * *功能*: 测试执行配置文件，定义构建选项、测试参数和并行策略
    * *职责*: 声明式配置测试矩阵，包括数据类型（float32/bfloat16）、并行模式（张量/流水线/序列并行）、批次大小等参数

* **write_to_feishu_sheet.py**
    * *功能*: 飞书表格数据同步工具，自动解析训练日志和性能报告并写入在线协作表格
    * *职责*: 提取训练指标（延迟、吞吐量）、解析性能分析报告（Top-5 热点）、记录 Git 版本信息，实现实验结果的自动化追踪

## 3. 架构逻辑图解

### 工作流程编排

```
开发阶段 (format.py)
    ↓
    开发者提交代码
    ↓
    format.py 检查暂存区文件风格
    ↓
    调用 clang-format (C/C++/CUDA) 或 black (Python)
    ↓
    通过检查后允许提交

测试阶段 (run_models_and_profile.bash + test_config.json)
    ↓
    读取 test_config.json 配置
    ↓
    遍历 builds 数组，执行 CMake 构建
    ↓
    对每个构建配置，遍历 tests 数组
    ↓
    生成测试命令（GPT2/Llama3 + 并行参数）
    ↓
    执行训练并记录日志到 logs/ 目录
    ↓
    如果 profile=true，收集性能数据到 profile_logs/

数据分析阶段 (write_to_feishu_sheet.py)
    ↓
    读取 logs/ 中的训练日志
    ↓
    parse_training_log() 提取平均延迟和吞吐量
    ↓
    读取 profile_logs/ 中的性能报告
    ↓
    parse_profile_report() 提取 Step_9 的 Top-5 性能热点
    ↓
    获取当前 Git 分支和提交 ID
    ↓
    通过 Feishu API 写入对应模型的在线表格
    ↓
    设置表格样式（交替行颜色、日期格式、合并单元格）
```

### 并行策略覆盖

测试配置覆盖了 InfiniTrain 的所有并行模式：

1. **数据并行 (Test 1-3)**
   - 基础测试：float32/bfloat16 数据类型
   - 批次配置：batch_size × num_processes = total_batch_size

2. **张量并行 (Test 4-5)**
   - `tensor_parallel=4`：模型层内切分到 4 个 GPU
   - 配合 `sequence_parallel=true`：序列维度并行优化

3. **流水线并行 (Test 6-7)**
   - `pipeline_parallel=8`：模型层间切分到 8 个阶段
   - `virtual_pipeline_parallel=2`：虚拟流水线减少气泡

4. **混合并行 (Test 8)**
   - `tensor_parallel=2` + `sequence_parallel=true`
   - `pipeline_parallel=2` + `virtual_pipeline_parallel=2`
   - 验证 4D 并行策略的协同工作

### 性能数据流

```
训练进程 (GPT2/Llama3)
    ↓ stdout/stderr
logs/{model}_{test_id}.log
    ↓ 正则解析
平均延迟 (ms) + 吞吐量 (tok/s)
    ↓
飞书表格前 5 列

性能分析器 (PROFILE_MODE=ON)
    ↓ NSight/Profiler
profile_logs/{model}_{test_id}_profile_*.report.rank0
    ↓ 表格解析
Step_9 阶段的 Top-5 Kernel 热点
    ↓
飞书表格后续列 (设备时间、主机时间、占比)
```

### 关键设计模式

1. **声明式配置驱动**
   - `test_config.json` 通过 JSON 数组声明测试矩阵
   - Bash 脚本解析配置并生成执行计划
   - 易于扩展新的测试场景

2. **工具链解耦**
   - `format.py` 独立于项目，可应用于任何 Git 仓库
   - `write_to_feishu_sheet.py` 通过抽象 FeishuSheetHandler 类实现 API 封装
   - 每个工具都是独立的 CLI 程序

3. **容错与日志**
   - Bash 脚本使用 `set -e` 和 `set -o pipefail` 确保错误传播
   - 所有命令输出重定向到日志文件，便于调试
   - Python 工具提供详细的错误提示和异常处理

4. **版本追踪集成**
   - 自动记录 Git 分支和提交 ID
   - 性能数据与代码版本关联，支持回溯分析
   - 飞书表格的时间戳记录实验时间线

### 测试环境依赖

```
构建工具:
    - CMake (支持 USE_CUDA, USE_NCCL, PROFILE_MODE 选项)
    - Make (并行编译)

代码格式化:
    - clang-format-16 (C/C++/CUDA/MLU/OpenCL)
    - black (Python)

运行时:
    - CUDA 环境 (NCCL 用于集合通信)
    - 数据文件: GPT2/Llama3 权重和训练数据

数据同步:
    - requests (HTTP 客户端)
    - pandas (数据处理)
    - 飞书开放平台权限 (APP_ID, APP_SECRET)
```

## 4. 技术要点总结

### format.py 核心特性
- 支持增量检查：仅处理 Git 暂存区或指定 ref 之后修改的文件
- 多文件类型扩展：通过 `SUPPORTED_FILES` 字典映射后缀到格式化工具
- 检查模式：`--check` 选项仅验证格式，不修改文件
- 自定义工具路径：`--c` 和 `--py` 参数指定格式化程序

### run_models_and_profile.bash 核心特性
- JSON 配置解析：使用 `jq` 命令行工具读取配置
- 环境变量导出：自动导出 `variables` 字段中的所有键值对
- NCCL 优化：检测到 `tensor_parallel` 参数时设置 `NCCL_LAUNCH_MODE=GROUP`
- 日志分类：普通日志存 `logs/`，性能日志存 `profile_logs/`

### write_to_feishu_sheet.py 核心特性
- Token 缓存：访问令牌缓存至过期前 10 分钟，减少 API 调用
- 正则提取：训练日志通过正则匹配 `step N/M | ... (XX ms | YY tok/s)`
- 数据透视：性能报告按 4 个时间维度排序（Device Total/Avg, Host Total/Avg）
- 日期转换：实现 Excel 1900 日期系统到飞书数值日期的转换
- 样式设置：交替行背景色、日期格式化、单元格合并

### 测试配置策略
- 16 个测试用例覆盖 float32/bfloat16 两种数据类型
- 测试编号命名规范：数字越大，并行策略越复杂
- 所有测试保持 `total_batch_size=5120` 不变，验证不同并行策略的效率差异
- 迭代次数统一为 10，确保性能统计的稳定性

---

**架构备注**：该目录体现了现代深度学习框架工程化的最佳实践——通过自动化工具链减少人工操作，通过声明式配置提高可维护性，通过在线协作平台实现实验数据的透明化追踪。这四個脚本共同构成了 InfiniTrain 的"质量保障-性能测试-数据管理"闭环。
