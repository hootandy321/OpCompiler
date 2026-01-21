# InfiniLM Examples 核心实现文档

本模块包含 InfiniLM 框架的三个核心示例程序，展示如何在不同硬件后端（CPU、NVIDIA GPU、Cambricon、Metax、Moore、Iluvatar）上运行大语言模型推理和性能基准测试。这些示例覆盖了 Python 原生模型实现和 C++ 优化推理引擎两种后端。

## 1. 模块结构

- **`bench.py`**: 性能基准测试工具，支持批量大小、输入/输出长度等多维度测试，自动计算 KV Cache 内存占用并按内存使用量排序测试用例
- **`jiuge.py`**: 基于推理引擎（InferEngine）的快速推理示例，使用 C++ 优化后端，支持多种国产 AI 加速卡
- **`llama.py`**: Python 原生模型实现示例，使用 AutoLlamaModel，适合开发和调试

## 2. 核心类与函数

### `TestModel` (bench.py)
- **位置**: `bench.py:193-274`
- **主要功能**: 封装模型实例化、权重加载、tokenizer 初始化和推理执行的完整流程
- **关键成员**:
  - `model: infinicore.nn.Module`: 推理引擎实例（InferEngine）
  - `tokenizer: AutoTokenizer`: HuggingFace 分词器
  - `input_ids_list: list[int]`: 预编码的输入 token 序列
- **核心方法**:
  - `__init__(model_path, infini_device, tp=1, skip_load=False)`: 初始化模型、加载权重、创建 tokenizer、编码提示词。支持跳过权重加载（`skip_load`）用于快速测试
  - `run(batch_size, input_len, output_len)`: 执行单次推理测试，通过 `repeat_prompt` 重复输入序列至目标长度，调用 `model.generate` 进行自回归生成，计算并输出总耗时
- **生命周期**: 按需创建，单例复用于同一模型路径的多次测试

### `get_test_cases()` (bench.py)
- **位置**: `bench.py:70-123`
- **主要功能**: 生成测试用例字典，按 KV Cache 内存占用升序排序
- **算法逻辑**:
  1. 从 `config.json` 提取模型配置（`head_dim`, `num_key_value_heads`, `num_hidden_layers`）
  2. 遍历 batch_size、input_len、output_len 的笛卡尔积
  3. 计算每个测试用例的 KV Cache 内存占用：
     ```
     KV_cache_bytes = data_type_bytes × batch_size × total_seq_len × num_key_value_heads × head_dim × num_hidden_layers
     ```
  4. 按 `kvcache_memory` 字段升序排序，返回 `OrderedDict`（索引映射到测试用例）
- **时间复杂度**: O(n log n)，其中 n = len(batch_size_list) × len(input_len_list) × len(output_len_list)

### `parse_list()` (bench.py)
- **位置**: `bench.py:35-67`
- **主要功能**: 命令行参数解析器，支持单个整数、JSON 列表、逗号分隔列表三种格式
- **支持格式**:
  - `"1"` → `1`
  - `"[1,2,4]"` → `[1, 2, 4]`
  - `"1,2,4"` → `[1, 2, 4]`
- **异常处理**: 解析失败时抛出 `argparse.ArgumentTypeError`

### `test()` (jiuge.py)
- **位置**: `jiuge.py:89-181`
- **主要功能**: 推理引擎示例入口，使用 InferEngine（C++ 后端）执行高效推理
- **核心流程**:
  1. 创建 `InferEngine` 实例，传入 `DistConfig(tp)` 支持张量并行
  2. 调用 `load_model_state_dict_by_file` 加载模型权重
  3. 修复 Llama tokenizer 的 decoder 问题（移除 `Prepend` 和 `Strip`，替换为 `Replace` + `ByteFallback` + `Fuse` 序列）
  4. 应用 chat template 编码输入提示
  5. 调用 `model.reset_cache(batch_size, max_new_tokens + input_len)` 预分配 KV Cache
  6. 执行 `model.generate(input_ids_infini, GenerationConfig(...))`，支持采样参数（`temperature`, `top_k`, `top_p`）
  7. 解码输出 tokens 并打印总耗时
- **性能优化**: 使用 `_measure_and_log_time=True` 启用性能计时

### `test()` (llama.py)
- **位置**: `llama.py:76-162`
- **主要功能**: Python 原生模型示例，使用 `AutoLlamaModel.from_pretrained`
- **与 jiuge.py 的差异**:
  - 使用 `infinilm.AutoLlamaModel` 替代 `InferEngine`
  - 使用 `get_model_state_dict` 获取权重字典
  - 调用 `model.load_state_dict(model_param_infini, strict=True)` 加载权重
  - 使用 Python 实现的 `model.generate()`，参数为 `max_new_tokens` 和 `tokenizer`
  - 不支持张量并行（无 `tp` 参数）
- **Tokenizer 修复**: 与 jiuge.py 相同的 decoder 修复逻辑

### `repeat_prompt()` (bench.py)
- **位置**: `bench.py:187-190`
- **主要功能**: 通过循环重复输入 token 序列至目标长度
- **算法**: `repeat_times = ceil(target_length / len(input_ids))`，返回 `input_ids * repeat_times` 的切片

## 3. API 接口

### 命令行参数接口

```python
# bench.py 专用参数
--batch-size   # int 或 list[int]，批量大小（默认 1）
--input-len    # int 或 list[int]，输入 token 长度（默认 10）
--output-len   # int 或 list[int]，输出 token 长度（默认 20）
--tp           # int，张量并行度（默认 1）
--skip-load    # bool，跳过权重加载（默认 False）

# 通用硬件后端选择（三文件共享）
--cpu          # 使用 CPU 设备
--nvidia       # 使用 NVIDIA CUDA (设备字符串 "cuda")
--cambricon    # 使用寒武纪 MLU (设备字符串 "mlu")
--metax        # 使用 Metax (设备字符串 "cuda")
--moore        # 使用 Moore Threads MUSA (设备字符串 "musa")
--iluvatar     # 使用天数智芯 (设备字符串 "cuda")

# 通用推理参数
--model_path   # str，HuggingFace 模型目录路径（必需）
--max_new_tokens # int，最大生成 tokens 数（默认 100）
--prompt       # str，输入提示文本（默认 "How are you"）
--batch_size   # int，批量大小（默认 1，仅 jiuge.py/llama.py）
--backend      # str，后端选择 "cpp" 或 "python"（仅 jiuge.py/llama.py）
```

### 核心推理 API

```python
# InferEngine 接口（jiuge.py, bench.py）
from infinilm.infer_engine import InferEngine, GenerationConfig

model = InferEngine(
    model_path: str,
    device: infinicore.device,
    distributed_config: DistConfig
)

# 重置 KV Cache
model.reset_cache(batch_size: int, initial_capacity: int)

# 生成文本
output_ids = model.generate(
    input_ids_infini: infinicore.Tensor,
    config: GenerationConfig,  # max_new_tokens, temperature, top_k, top_p
    _measure_and_log_time: bool = False
) -> list[infinicore.Tensor]

# AutoLlamaModel 接口（llama.py）
import infinilm

model = infinilm.AutoLlamaModel.from_pretrained(
    model_path: str,
    device: infinicore.device
)

# 加载权重
model.load_state_dict(state_dict: dict, strict: bool = True)

# 生成文本
model.generate(
    input_ids_infini: infinicore.Tensor,
    max_new_tokens: int,
    tokenizer: AutoTokenizer
)
```

### 权重加载 API

```python
# 推理引擎权重加载（jiuge.py, bench.py）
from infinilm.modeling_utils import load_model_state_dict_by_file

load_model_state_dict_by_file(
    model: InferEngine,
    model_path: str,
    dtype: infinicore.dtype
)

# Python 模型权重加载（llama.py）
from infinilm.modeling_utils import get_model_state_dict

state_dict = get_model_state_dict(
    model_path: str,
    device: infinicore.device,
    dtype: infinicore.dtype
)
model.load_state_dict(state_dict, strict=True)
```

## 4. 使用示例

### 基准测试（bench.py）
```bash
# NVIDIA GPU，多批次多长度测试
python examples/bench.py \
    --nvidia \
    --model=~/TinyLlama-1.1B-Chat-v1.0/ \
    --batch-size="[1,2,4]" \
    --tp=1 \
    --input-len="[32,256]" \
    --output-len="[128,512]"

# CPU 快速测试（跳过权重加载）
python examples/bench.py --cpu --model=~/model/ --skip-load

# 寒武纪 MLU 测试
python examples/bench.py --cambricon --model=~/model/ --batch-size=8
```

### 推理引擎快速推理（jiuge.py）
```bash
# NVIDIA GPU 单次推理
python examples/jiuge.py \
    --nvidia \
    --model_path=~/TinyLlama-1.1B-Chat-v1.0 \
    --max_new_tokens=200 \
    --batch-size=4 \
    --tp=2 \
    --prompt="山东最高的山是？"

# Metax 加速卡
python examples/jiuge.py --metax --model_path=~/model/ --tp=4

# Moore Threads MUSA
python examples/jiuge.py --moore --model_path=~/model/
```

### Python 原生模型（llama.py）
```bash
# CPU 开发调试
python examples/llama.py \
    --cpu \
    --model_path=~/TinyLlama-1.1B-Chat-v1.0 \
    --backend=python \
    --batch_size=1 \
    --prompt="Hello, world!"

# NVIDIA GPU Python 实现
python examples/llama.py --nvidia --model_path=~/model/ --max_new_tokens=50
```

### 代码集成示例
```python
import infinicore
from infinilm.infer_engine import InferEngine, GenerationConfig
from infinilm.distributed import DistConfig
from infinilm.modeling_utils import load_model_state_dict_by_file
from transformers import AutoTokenizer

# 初始化设备
device = infinicore.device("cuda", 0)

# 创建推理引擎（支持张量并行）
model = InferEngine(
    model_path="~/TinyLlama-1.1B-Chat-v1.0",
    device=device,
    distributed_config=DistConfig(tp=2)  # 2 路张量并行
)

# 加载权重
load_model_state_dict_by_file(model, model_path, dtype=model.config.dtype)

# 创建 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 编码输入
input_content = tokenizer.apply_chat_template(
    conversation=[{"role": "user", "content": "什么是人工智能？"}],
    add_generation_prompt=True,
    tokenize=False,
)
input_ids = tokenizer.encode(input_content, return_tensors=None)
input_ids_infini = infinicore.from_list([input_ids])

# 预分配 KV Cache
max_new_tokens = 100
model.reset_cache(batch_size=1, initial_capacity=max_new_tokens + len(input_ids))

# 生成文本（带采样）
output_ids = model.generate(
    input_ids_infini,
    GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )
)

# 解码输出
output_text = tokenizer.decode(output_ids[0].to_numpy()[0], skip_special_tokens=True)
print(output_text)
```

## 5. 实现细节

- **内存管理**:
  - **KV Cache 预分配**: `model.reset_cache(batch_size, initial_capacity)` 根据 batch_size 和最大序列长度预分配 KV Cache，避免动态扩展的开销
  - **KV Cache 容量计算**: `initial_capacity = input_len + output_len`，确保整个序列生成过程中无需重新分配
  - **数据类型**: 默认使用 `bfloat16`（2 bytes/token），相比 `float32` 减少 50% 内存占用

- **并发**:
  - **张量并行**: 通过 `DistConfig(tp)` 支持，tp=2 表示将模型权重切分到 2 个 GPU 上并行计算
  - **设备隔离**: 每个进程绑定单一设备 `infinicore.device(device_str, device_id)`，避免多线程竞争
  - **无锁设计**: 推理阶段无显式锁，依赖单线程执行模型保证

- **性能优化**:
  - **C++ 后端**: `InferEngine` 使用 C++ 实现，相比 Python 后端（`AutoLlamaModel`）有显著性能提升
  - **测试用例排序**: `get_test_cases()` 按 KV Cache 内存升序排序，避免大内存测试用例导致 OOM 后无法测试小用例
  - **Tokenizer 修复**: 针对 Llama tokenizer 的 `Prepend` + `Strip` decoder 问题，替换为 `Replace("▁", " ")` + `ByteFallback` + `Fuse` 序列，确保解码正确性
  - **批量测试**: `bench.py` 支持 `batch_size`、`input_len`、`output_len` 的笛卡尔积测试，使用 `tqdm` 进度条显示

- **错误处理**:
  - **参数验证**: `parse_list()` 对非法参数抛出 `argparse.ArgumentTypeError`
  - **后端检查**: `jiuge.py` 仅支持 `backend="cpp"`，`llama.py` 仅支持 `backend="python"`，否则抛出 `ValueError`
  - **模型类型限制**: `llama.py` 仅支持 `model.config.model_type == "llama"`，否则抛出 `ValueError`

- **依赖**:
  - **内部依赖**: `infininicore`（张量计算库）、`infinilm`（模型实现）、`infinilm.distributed`（分布式）、`infinilm.infer_engine`（推理引擎）
  - **外部依赖**: `transformers`（HuggingFace tokenizer）、`tokenizers`（tokenizer 修复）、`numpy`（数组转换）、`tqdm`（进度条）

- **设计模式**:
  - **工厂模式**: `AutoLlamaModel.from_pretrained()` 根据配置自动创建模型实例
  - **策略模式**: 通过 `--cpu`/`--nvidia`/`--cambricon` 等标志选择设备类型（device_str 策略）
  - **模板方法**: `test()` 函数定义统一的推理流程（创建模型 → 加载权重 → 编码输入 → 生成 → 解码输出），不同后端实现细节不同
