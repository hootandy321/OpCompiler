# InfiniLM-Rust 性能基准测试脚本文档

本模块提供了一套完整的 LLM（大语言模型）性能基准测试工具，用于自动化测试 InfiniLM-Rust 推理引擎在不同 GPU 配置、不同输入长度和不同输出长度下的性能表现。该工具通过矩阵式测试组合，系统性地采集 prefill（预填充）和 decode（解码）阶段的性能指标，并生成结构化的性能报告。

## 1. 模块结构

- **`test.py`**: 核心性能测试驱动脚本，负责执行基准测试、解析日志输出、提取性能指标并生成报告
- **`test.sh`**: Shell 包装脚本，用于自动部署测试环境并执行单卡和多卡推理测试

## 2. 核心类与函数

### `run_command(prompt, max_steps, devices, test_model)`
- **Location**: `test.py` (Lines 11-35)
- **Primary Function**: 执行单次推理测试任务，通过 subprocess 调用 cargo 命令启动 InfiniLM-Rust 推理引擎
- **Parameters**:
  - `prompt` (str): 输入提示文本，长度按单词数控制
  - `max_steps` (int): 最大解码步数（输出 token 数量）
  - `devices` (str): GPU 设备配置，如 "1" 表示单卡，"6,7" 表示多卡
  - `test_model` (str): GGUF 格式模型文件的绝对路径
- **Return Values**:
  - 成功时返回 `(stdout, stderr)` 元组
  - 失败时返回 `(None, None)` 并打印错误信息
- **Command Template**: `cargo gen -p "{prompt}" --gpus {devices} --max-steps {max_steps} {test_model}`
- **Error Handling**: 检测子进程退出码，非零时打印 stderr 并返回 None

### `extract_performance_metrics(log_output)`
- **Location**: `test.py` (Lines 37-71)
- **Primary Function**: 使用正则表达式从推理日志中提取关键性能指标，支持时间单位的智能转换（秒/毫秒）
- **Algorithm**: 双模式正则匹配 + 单位归一化
- **Extracted Metrics**:
  - `prefill_ms`: Prefill 阶段耗时（毫秒），自动从秒转换为毫秒
  - `decode_ms`: Decode 阶段耗时（毫秒），自动从秒转换为毫秒
  - `output_tokens`: 实际生成的 token 数量
  - `ms_per_token`: 平均每 token 生成耗时（毫秒/token）
  - `tokens_per_sec`: 吞吐量（tokens/秒）
- **Regex Patterns**:
  - Pattern 1: `r"prefill = ([\d.]+) ?(\w+), decode = ([\d.]+) ?(\w+)"` - 提取 prefill/decode 时间及单位
  - Pattern 2: `r"n toks = (\d+), perf: ([\d.]+)ms/tok, ([\d.]+)tok/s"` - 提取吞吐量指标
- **Return**: 字典对象（包含所有提取的指标）或 None（匹配失败时）

### `main()`
- **Location**: `test.py` (Lines 74-147)
- **Primary Function**: 测试编排器，执行三层嵌套循环（devices × prompt_lengths × step_lengths）的完整测试矩阵
- **Configuration Constants**:
  - `PROMPT_LENGTHS`: `[32, 64, 128, 256, 512, 1024]` - 输入单词数
  - `BASE_PROMPT_WORD`: `"hello"` - 基础重复词
  - `STEP_LENGTHS`: `[32, 64, 128, 256, 512, 1024, 2048]` - 输出 token 数
- **Command-Line Arguments**:
  - `model_path` (positional): GGUF 模型文件路径（必填）
  - `--gpus` (nargs='+'): GPU 配置列表，默认 `["1", "6,7"]`，支持任意数量配置
- **Loop Order**: `devices → prompt_len → max_steps`（确保同一 GPU 配置下连续测试）
- **Prompt Generation**: `" ".join([BASE_PROMPT_WORD] * prompt_len)` - 生成重复单词组成的测试输入
- **Output**:
  - 实时打印每次测试的 JSON 格式性能数据
  - 测试结束后生成 Pandas DataFrame 表格（如果已安装 pandas）
  - 降级输出：固定宽度格式化表格（pandas 未安装时）

## 3. API 接口

```python
# 命令行调用接口
python test.py <model_path> [--gpus GPU_CONFIG [GPU_CONFIG ...]]

# 参数说明
# model_path: 必填，GGUF 格式模型文件路径
# --gpus: 可选，GPU 配置列表，默认为 ["1", "6,7"]
#         支持单卡（如 "1"）或多卡（如 "6,7,8,9"）

# 示例：测试单卡和双卡配置
python test.py "/path/to/model.gguf" --gpus 1 6,7

# 示例：测试多个多卡配置
python test.py "/path/to/model.gguf" --gpus 2,3 4,5,6,7 8
```

## 4. 使用示例

```bash
# 示例 1：使用 test.sh 自动化脚本（推荐）
# 该脚本会自动复制测试文件、执行单卡和多卡测试、清理临时文件
./test.sh

# 示例 2：手动执行单次 GPU 配置测试
python test.py "/fm9g-7B-sft-v0.0-F16.gguf" --gpus 1

# 示例 3：测试多个 GPU 配置组合
python test.py "/fm9g-7B-sft-v0.0-F16.gguf" --gpus 1 2,3 4,5,6,7

# 示例 4：自定义测试（需要修改脚本中的 PROMPT_LENGTHS 和 STEP_LENGTHS）
# 编辑 test.py，调整：
# PROMPT_LENGTHS = [64, 128, 256]  # 只测试这三种输入长度
# STEP_LENGTHS = [128, 512]        # 只测试这两种输出长度
```

**典型输出示例**：
```
{'devices': '1', 'prompt_len': 32, 'max_steps': 32, 'output_tokens': 32, 'prefill_ms': 123.4567, 'decode_ms': 456.7890, 'ms_per_token': 14.2740, 'tokens_per_sec': 70.0561}

================================================================================ 测试结果汇总 =================================================================================
devices prompt_len max_steps output_tokens     prefill_ms      decode_ms ms_per_token tokens_per_sec
     1         32         32             32      123.4567      456.7890       14.2740        70.0561
     1         32         64             64      123.4567      912.3456       14.2740        70.0561
...
```

## 5. 实现细节

### 测试矩阵设计
- **三层嵌套循环**: 严格按照 `GPU配置 → 输入长度 → 输出长度` 的顺序执行，避免频繁切换 GPU 上下文
- **测试规模**: 默认配置下，单个 GPU 配置执行 6×7=42 次测试，N 个 GPU 配置共执行 42N 次测试
- **Prompt 生成策略**: 使用重复单词（"hello hello hello..."）而非随机文本，确保测试的可重复性和输入长度的精确控制

### 性能指标提取
- **正则表达式鲁棒性**: 两种独立模式匹配不同格式的日志输出
- **单位转换逻辑**: 检测时间单位（`s` 或 `ms`），统一转换为毫秒存储，便于后续计算
- **容错机制**: 单次测试失败不影响整体流程，跳过并继续下一个测试组合

### 输出格式化
- **Pandas 表格**: 优先使用 DataFrame 输出，自动对齐列、格式化浮点数（4 位小数）
- **降级表格**: 使用固定宽度格式化（`{:<12}` 等格式化字符串），保证无 pandas 环境下的可读性
- **实时反馈**: 每次测试完成立即输出 JSON 格式结果，便于监控进度

### Shell 脚本工作流
1. **部署阶段**: 复制 `test.py` 到目标目录 `../../InfiniLM-Rust/`
2. **单卡测试**: 执行 `python test.py "/fm9g-7B-sft-v0.0-F16.gguf" --gpus 1`
3. **多卡测试**: 执行 `python test.py "/fm9g-7B-sft-v0.0-F16.gguf" --gpus 2,3 5,6,7,8`
4. **清理阶段**: 删除复制的 `test.py`，恢复原始目录
5. **路径假设**: Shell 脚本假设当前目录位于 `InfiniPerf/benchmarks/llm/scripts/InfiniLM-Rust/`

### 错误处理策略
- **Subprocess 失败**: 检查 `result.returncode`，非零时打印 stderr 并跳过当前测试
- **正则匹配失败**: `extract_performance_metrics` 返回 None，主循环记录警告但继续执行
- **Pandas 不可用**: 捕获 `ImportError`，切换到降级输出格式
- **部分失败策略**: 单个测试失败不中断整体矩阵，最终汇总所有成功采集的数据

### 依赖项
- **Python 标准库**: `subprocess`, `re`, `json`, `argparse`
- **外部依赖**: `pandas` (可选，用于美化表格输出)
- **系统依赖**: `cargo` 命令（InfiniLM-Rust 构建工具），GGUF 模型文件

### 数据结构
```python
# 提取的指标字典结构
metrics = {
    'devices': str,           # GPU 配置，如 "1" 或 "6,7"
    'prompt_len': int,        # 输入单词数
    'max_steps': int,         # 最大解码步数
    'output_tokens': int,     # 实际生成的 token 数
    'prefill_ms': float,      # Prefill 阶段耗时（毫秒）
    'decode_ms': float,       # Decode 阶段耗时（毫秒）
    'ms_per_token': float,    # 平均每 token 耗时（毫秒/token）
    'tokens_per_sec': float   # 吞吐量（tokens/秒）
}

# 结果列表
all_results: List[Dict]  # 所有成功测试的指标集合
```
