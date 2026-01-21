# `Generation Utils` Core Implementation Documentation

本模块是 InfiniLM 框架的核心文本生成工具包，实现了基于 InfiniCore 张量计算引擎的高效序列生成功能。模块提供了完整的两阶段生成流程（Prefill + Decode），支持 KV Cache 优化、位置编码管理、随机采样策略（Top-K、Top-P、Temperature），并集成了实时性能监控指标统计。

## 1. Module Structure

- **`utils.py`**: 核心生成工具模块，包含张量类型转换函数、位置ID计算、输入准备逻辑以及主要的文本生成循环实现

## 2. Core Classes

### `GenerationMixin`
- **Location**: `utils.py`
- **Primary Function**: 这是一个 Mixin 类，为语言模型提供文本生成能力。它实现了标准的两阶段自回归生成流程（Prefill 阶段处理完整输入序列，Decode 阶段逐 Token 生成），并支持可配置的采样策略和性能统计
- **Key Members**:
  - `config`: 模型配置对象，包含 `eos_token_id` 等生成必需参数
  - `use_cache`: 布尔值，控制是否启用 KV Cache 加速
  - `_model`: 可选的 C++ 后端模型对象（通过 `hasattr` 检测是否存在 `reset_cache` 方法）
- **Core Methods**:
  - `_get_initial_position_ids(bs: int, seq_length: int) -> infinicore.Tensor`: 为 Prefill 阶段生成初始位置编码，返回形状为 `(bs, seq_length)` 的张量，包含 `[0, 1, 2, ..., seq_length-1]` 的序列，用于 RoPE 等位置敏感的注意力机制
  - `prepare_inputs_for_generation(self, past_key_values, **kwargs) -> dict`: 准备每次前向传播的输入字典，核心逻辑包括：处理 KV Cache 传递、计算阶段特定的 `position_ids`（Prefill 生成完整序列，Decode 递增最后位置）、管理 `cache_positions`（记录当前处理的序列长度）、处理 `next_token_ids`（将 Token ID 列表转换为形状 `(bs, 1)` 的张量），以及透传其他 kwargs 参数
  - `generate(self, input_ids, max_new_tokens, tokenizer, stop_on_eos, **kwargs) -> dict`: 生成流程的入口函数，初始化 `DynamicCache`（如果不是 C++ 后端），设置 `use_cache` 标志，然后委托给 `_sample` 方法执行实际生成，返回包含输出 Token IDs、文本内容、延迟统计的字典
  - `_sample(self, input_ids, max_new_tokens, tokenizer, stop_on_eos, **model_kwargs) -> dict`: 核心生成循环，实现完整的自回归采样流程。对于每个生成步骤：调用 `prepare_inputs_for_generation` 准备输入、执行模型前向传播获取 Logits、应用随机采样策略（通过 `infinicore.nn.functional.random_sample` 实现 Top-K/Top-P/Temperature 采样）、解码 Token 为文本、实时打印输出、检测 EOS 终止条件、收集性能指标（Prefill TTFT、Decode ITL、吞吐量统计）。返回包含 `output_token_ids`、`output_content`、`total_latency`、`prefill_latency`、`decode_latency`、`total_input_tokens`、`total_output_tokens` 的字典
- **Lifecycle**: 作为 Mixin 类被继承到语言模型类中，通过 `generate()` 方法实例化调用。每次生成创建新的 `DynamicCache` 对象（如果启用缓存），生成结束后销毁

## 3. API Interface

```python
# 张量转换工具函数
def infini_to_ctype_dtype(infini_dtype):
    """将 InfiniCore 数据类型（int32, float32）映射到 ctypes 类型（c_int32, c_float），
    用于零拷贝 NumPy 转换的类型系统桥接"""

def infini_to_numpy(infini_tensor: infinicore.Tensor) -> np.ndarray:
    """实现 InfiniCore 张量到 NumPy 数组的零拷贝转换。
    工作流程：确保张量在 CPU 设备 → 获取数据指针和形状 → 创建 ctypes 数组共享内存 →
    重塑为原始形状 → 执行深拷贝返回。用于将 GPU 计算结果转移到 Python 环境"""

# 为 infinicore.Tensor 类动态添加 to_numpy 方法
infinicore.Tensor.to_numpy = infini_to_numpy

# Mixin 类公共接口
class GenerationMixin:
    def prepare_inputs_for_generation(
        self,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> dict:
        """准备模型前向传播的输入字典。
        关键参数：
        - past_key_values: KV Cache 对象（DynamicCache 或 C++ 后端缓存）
        - kwargs: 包含 input_ids, position_ids, cache_positions, next_token_ids 等参数
        返回包含以下键的字典：
        - input_ids: 形状 (bs, 1) 的 Token ID 张量（Decode 阶段）或原始输入（Prefill）
        - position_ids: 位置编码张量，Prefill 为完整序列，Decode 为递增位置
        - cache_positions: 当前处理的序列长度（用于 KV Cache 索引）
        - past_key_values: KV Cache 对象"""

    def generate(
        self,
        input_ids: infinicore.Tensor,
        max_new_tokens: int,
        tokenizer,
        stop_on_eos=True,
        **kwargs,
    ) -> dict:
        """执行文本生成的入口函数。
        参数：
        - input_ids: 形状 (bs, seq_len) 的输入 Token ID 张量
        - max_new_tokens: 最大生成 Token 数量
        - tokenizer: 分词器对象，需实现 decode([token_ids]) 方法
        - stop_on_eos: 是否在遇到 EOS Token 时终止生成
        - kwargs: 可包含 random_val, topp, topk, temperature 等采样参数
        返回字典：
        - output_token_ids: List[int]，生成的 Token ID 序列
        - output_content: str，解码后的完整文本
        - total_latency: float，总生成时间（秒）
        - prefill_latency: float，Prefill 阶段延迟（秒）
        - decode_latency: float，Decode 阶段总延迟（秒）
        - total_input_tokens: int，输入 Token 总数（batch_size * seq_len）
        - total_output_tokens: int，实际生成 Token 数量"""

    def _sample(
        self,
        input_ids: infinicore.Tensor,
        max_new_tokens: int,
        tokenizer,
        stop_on_eos=True,
        **model_kwargs,
    ) -> dict:
        """内部生成循环实现，由 generate() 调用。
        采样参数（从 model_kwargs 提取，带默认值）：
        - random_val: float = 0.1，随机采样控制参数
        - topp: float = 0.8，Top-P（核采样）阈值
        - topk: int = 1，Top-K 采样保留的最高概率 Token 数量
        - temperature: float = 1.0，温度参数，控制分布平滑度
        返回格式与 generate() 相同的字典"""
```

## 4. Usage Example

```python
# 示例：使用 GenerationMixin 进行文本生成
import infinicore
from infinilm import ModelForCausalLM  # 假设已继承 GenerationMixin

# 初始化模型和分词器
model = ModelForCausalLM.from_pretrained("path/to/model")
tokenizer = load_tokenizer("path/to/tokenizer")

# 准备输入提示
prompt_text = "Once upon a time"
input_ids = tokenizer.encode(prompt_text, return_tensors="infini")

# 执行生成（启用 KV Cache）
result = model.generate(
    input_ids=input_ids,
    max_new_tokens=100,
    tokenizer=tokenizer,
    stop_on_eos=True,
    # 采样参数（可选）
    temperature=0.8,
    topp=0.9,
    topk=50,
    random_val=0.1
)

# 访问生成结果
print("Generated text:", result["output_content"])
print("Token IDs:", result["output_token_ids"])
print(f"Total latency: {result['total_latency'] * 1000:.2f} ms")
print(f"Prefill TTFT: {result['prefill_latency'] * 1000:.2f} ms")
print(f"Decode throughput: {result['total_output_tokens'] / result['decode_latency']:.2f} tok/s")

# 示例：将 InfiniCore 张量转换为 NumPy 用于后处理
logits = model(infinicore.from_list([[1, 2, 3]]))
logits_np = logits.to_numpy()  # 使用动态添加的 to_numpy 方法
predicted_ids = logits_np.argmax(axis=-1)
```

## 5. Implementation Details

### 两阶段生成策略 (Two-Stage Generation)
- **Prefill 阶段**: 处理完整输入序列（长度 `seq_len`），生成初始位置编码 `[0, 1, ..., seq_len-1]`，一次性计算所有输入 Token 的表示并存入 KV Cache。该阶段延迟称为 TTFT (Time To First Token)，吞吐量计算为 `(batch_size * seq_len) / prefill_latency` tokens/s
- **Decode 阶段**: 自回归循环，每次处理单个新生 Token。位置编码递增（取上一位置 +1），`cache_positions` 累加记录已处理长度。使用 KV Cache 避免重复计算历史 Token。平均每 Token 延迟称为 ITL (Inter-Token Latency)，吞吐量计算为 `(batch_size * generated_tokens) / total_decode_latency` tokens/s

### KV Cache 管理
- **Python 后端**: 使用 `DynamicCache` 类（从 `cache_utils` 导入），通过 `past_key_values` 参数传递。`prepare_inputs_for_generation` 在首次调用时初始化缓存，后续调用传递相同对象
- **C++ 后端**: 通过 `hasattr(self._model, "reset_cache")` 检测，C++ 后端自行管理缓存，无需创建 `DynamicCache` 对象。`cache_positions` 用于同步 C++ 后端的缓存索引

### 位置编码机制 (Position Encoding)
- **Prefill**: `_get_initial_position_ids` 生成完整序列位置 IDs，使用 `infinicore.from_list` 创建 `(bs, seq_length)` 形状的张量，每行为 `[0, 1, 2, ..., seq_length-1]`
- **Decode**: 使用 `narrow(1, seq_len-1, 1)` 提取上一批次最后位置，通过加法操作递增。实现为 `next_position = last_position + ones(bs, 1)`，确保位置 ID 连续性

### 随机采样实现 (Random Sampling)
- 使用 `infinicore.nn.functional.random_sample` 原生函数，支持 Top-K、Top-P、Temperature 三种策略组合
- **批量处理**: 对 Batch 中每个样本独立采样，通过 `narrow(0, i, 1)` 逐个提取 `(vocab_size,)` 形状的 Logits 向量，调用采样函数写入 `next_tokens` 张量的对应位置
- **同步机制**: 采样后调用 `infinicore.sync_stream()` 确保 GPU 计算完成，然后通过 `to_numpy()` 将结果转移到 CPU 进行解码

### 性能监控 (Performance Monitoring)
- **时间戳记录**: 每个生成步骤记录 `start_time` 和 `end_time`，存储到 `time_list`
- **指标计算**:
  - Prefill TTFT: `time_list[0]`（首次前向传播时间）
  - Decode Avg ITL: `sum(time_list[1:]) / (len(time_list) - 1)`（排除 Prefill 步骤）
  - Throughput: `(batch_size * token_count) / latency`，分别计算 Prefill 和 Decode 吞吐量
- **输出格式**: 实时打印生成的文本（使用 `print(output_str, end="", flush=True)`），结束后输出统计摘要和 `</s>` 标记

### 零拷贝张量转换 (Zero-Copy Tensor Conversion)
- **类型桥接**: `infini_to_ctype_dtype` 建立 InfiniCore 类型（`infinicore.int32`, `infinicore.float32`）到 ctypes 类型（`ctypes.c_int32`, `ctypes.c_float`）的映射
- **内存共享**: 使用 `ArrayType.from_address(data_ptr)` 创建 ctypes 数组，直接指向 InfiniCore 张量的内存地址。通过 `np.ctypeslib.as_array` 创建 NumPy 数组视图，避免数据复制
- **深拷贝返回**: 调用 `np.copy()` 创建独立副本，防止 InfiniCore 张量释放后 NumPy 数组访问无效内存。动态添加 `infinicore.Tensor.to_numpy` 方法，提供便捷接口

### 错误处理与边界条件
- **EOS 终止**: 支持 `eos_token_id` 为整数或列表（多 EOS Token 场景），每次生成后检查 `token_id in eos_token_id_list`，满足条件则跳出循环
- **设备兼容**: `infini_to_numpy` 检查张量设备类型，非 CPU 张量通过 `to(infinicore.device("cpu", 0))` 显式转移到 CPU
- **空 Decode 处理**: 性能统计中检查 `len(time_list) > 1`，避免单步生成（仅 Prefill）时 Decode 指标除零错误
- **类型安全**: `infini_to_ctype_dtype` 对不支持的类型抛出 `ValueError`，防止类型映射错误

### 依赖关系
- **InfiniCore**: 核心张量计算引擎，提供 `infinicore.Tensor`、`infinicore.from_list`、`infinicore.empty`、`infinicore.sync_device`、`infinicore.sync_stream` 等底层操作
- **NumPy**: 用于张量转换和后处理，通过 `np.ctypeslib` 实现与 ctypes 的互操作
- **内部模块**: 依赖 `..cache_utils.Cache` 和 `DynamicCache` 实现 KV Cache 机制

### 设计模式
- **Mixin 模式**: `GenerationMixin` 作为独立类，通过多重继承为语言模型类添加生成能力，避免重复代码
- **策略模式**: `random_sample` 函数支持 Top-K、Top-P、Temperature 参数组合，允许运行时配置采样策略
- **模板方法模式**: `generate` 定义生成流程骨架（初始化缓存、调用 `_sample`），`_sample` 实现具体算法逻辑
- **适配器模式**: `infini_to_numpy` 和 `infini_to_ctype_dtype` 将 InfiniCore 类型系统适配到 Python/NumPy 生态系统
