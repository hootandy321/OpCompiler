# `LlamaModel Python Bindings` Core Implementation Documentation

该模块实现了 Llama 模型架构的 Python-C++ 绑定层，使用 pybind11 暴露 C++ 核心实现给 Python 接口。这是 InfiniLM 框架中连接底层高性能 C++ 实现与上层 Python 应用层的关键桥梁。

## 1. Module Structure

- **`llama.hpp`**: Pybind11 绑定定义，包含 `HookRegistry`、`LlamaConfig` 等核心类的 Python 接口映射，实现 Python 可调用对象与 C++ 回调函数的双向转换。

## 2. Core Classes

### `HookRegistry`
- **Location**: `llama.hpp` (lines 23-40), `../../models/debug_utils/hooks.hpp`
- **Primary Function**: 调试工具类，用于注册和管理模型执行过程中的中间张量捕获钩子（hooks）。允许 Python 回调函数在特定计算节点捕获中间结果，用于模型调试、可视化和验证。
- **Key Members**:
  - `hooks_`: `std::unordered_map<std::string, HookCallback>` - 存储钩子名称到回调函数的映射表
- **Core Methods**:
  - `register_hook(name, callback)`: 注册一个钩子，接受 Python 可调用对象并自动转换为 C++ `std::function`。在 Python 异常发生时重新抛出异常以保持调用栈完整性。
  - `clear()`: 清空所有已注册的钩子
  - `has_hooks()`: 返回是否存在活跃钩子的布尔标志
  - `call_hook(name, tensor, layer_idx)`: 触发指定名称的钩子，传入张量数据和层索引
- **Lifecycle**: 通过 `std::shared_ptr` 管理生命周期，支持跨模块共享。在调试场景中手动创建和销毁。

### `LlamaConfig`
- **Location**: `llama.hpp` (lines 42-125), `../../models/llama/llama_config.hpp`
- **Primary Function**: Llama 模型的完整配置结构体，继承自 `InfinilmModel::Config`，包含模型架构的所有超参数。支持 HuggingFace 兼容的配置格式。
- **Key Members**:
  - `dtype`: `infinicore::DataType` - 模型计算数据类型（默认 F32）
  - `vocab_size`: `size_t` - 词表大小（默认 32000）
  - `hidden_size`: `size_t` - 隐藏层维度（默认 4096）
  - `intermediate_size`: `size_t` - MLP 中间层维度（默认 11008）
  - `num_hidden_layers`: `size_t` - Transformer 解码器层数（默认 32）
  - `num_attention_heads`: `size_t` - 注意力头数（默认 32）
  - `num_key_value_heads`: `size_t` - KV 头数（用于 GQA，默认 32）
  - `head_dim`: `size_t` - 每个注意力头的维度（默认 128）
  - `max_position_embeddings`: `size_t` - 最大序列长度（默认 2048）
  - `rope_theta`: `double` - RoPE 基础频率（默认 10000.0）
  - `rms_norm_eps`: `double` - RMSNorm 归一化 epsilon（默认 1e-6）
  - `hidden_act`: `std::string` - 激活函数类型（默认 "silu"）
  - `attention_bias`: `bool` - Q/K/V 投影是否使用偏置（默认 true，兼容 9G7B）
  - `use_cache`: `bool` - 是否启用 KV 缓存（默认 true）
  - `bos_token_id`: `std::vector<int64_t>` - 序列开始 token ID（可变长度列表）
  - `eos_token_id`: `std::vector<int64_t>` - 序列结束 token ID（可变长度列表）
- **Core Methods**:
  - `validate()`: 验证配置参数的数学约束（hidden_size 能被 num_attention_heads 整除、num_attention_heads 能被 num_key_value_heads 整除、head_dim 计算正确）
  - `kv_dim()`: 计算分组查询注意力（GQA）的 KV 投影维度，公式为 `hidden_size * num_key_value_heads / num_attention_heads`
- **Lifecycle**: 值类型，通过 Python 直接构造和修改。支持序列化（通过 HuggingFace config JSON）

## 3. API Interface

```cpp
// HookRegistry Python 绑定接口
py::class_<HookRegistry, std::shared_ptr<HookRegistry>>(m, "HookRegistry")
    .def(py::init<>())  // 默认构造函数
    .def("register_hook", [](HookRegistry &self, const std::string &name, py::object callback) {
        // Python 可调用对象自动转换为 C++ std::function
        self.register_hook(name, [callback](const std::string &hook_name,
                                             const infinicore::Tensor &tensor,
                                             int layer_idx) {
            callback(hook_name, tensor, layer_idx);  // 调用 Python 回调
        });
    }, py::arg("name"), py::arg("callback"))
    .def("clear", &HookRegistry::clear)
    .def("has_hooks", &HookRegistry::has_hooks);

// LlamaConfig Python 绑定接口（部分展示）
py::class_<LlamaConfig, InfinilmModel::Config>(m, "LlamaConfig")
    .def(py::init<>())
    .def_readwrite("vocab_size", &LlamaConfig::vocab_size)
    .def_readwrite("hidden_size", &LlamaConfig::hidden_size)
    .def_property("bos_token_id",
        [](const LlamaConfig &self) { return py::cast(self.bos_token_id); },
        [](LlamaConfig &self, py::object value) {
            // 智能类型转换：支持 int 或 list[int]
            if (py::isinstance<py::int_>(value)) {
                self.bos_token_id = {value.cast<int64_t>()};
            } else if (py::isinstance<py::list>(value)) {
                self.bos_token_id = value.cast<std::vector<int64_t>>();
            }
        })
    .def("validate", &LlamaConfig::validate)
    .def("kv_dim", &LlamaConfig::kv_dim)
    .def("__dir__", [](const LlamaConfig &self) {
        // 自定义属性发现列表，支持 Python dir() 内省
        py::list dir_list;
        dir_list.append("vocab_size");
        dir_list.append("hidden_size");
        // ... 所有可访问属性
        return dir_list;
    });
```

## 4. Usage Example

```python
# Python 端使用示例

from infinilm import LlamaConfig, HookRegistry
import torch

# 1. 创建并配置 Llama 模型
config = LlamaConfig()
config.vocab_size = 128000
config.hidden_size = 4096
config.num_hidden_layers = 32
config.num_attention_heads = 32
config.num_key_value_heads = 8  # 启用 GQA
config.max_position_embeddings = 8192
config.rope_theta = 10000.0

# 2. 验证配置合法性
assert config.validate(), "Invalid configuration"
print(f"KV dimension: {config.kv_dim()}")  # 输出: 4096 * 8 / 32 = 1024

# 3. 设置特殊 token（支持单个或多个）
config.bos_token_id = 1          # 单个 token
config.eos_token_id = [2, 3]     # 多个 token（兼容多轮对话）

# 4. 使用 HookRegistry 捕获中间结果（调试模式）
hook_registry = HookRegistry()

def tensor_callback(hook_name: str, tensor, layer_idx: int):
    """Python 回调函数，接收 C++ 传递的张量"""
    print(f"[Layer {layer_idx}] {hook_name}: shape={tensor.shape}, dtype={tensor.dtype}")
    # 可以进行可视化、统计分析或保存到磁盘

# 注册特定中间节点的钩子
hook_registry.register_hook("layer0_q_after_proj", tensor_callback)
hook_registry.register_hook("layer0_k_after_proj", tensor_callback)
hook_registry.register_hook("layer0_attention_output", tensor_callback)

# 5. 将 hook_registry 传递给 C++ 模型（假设模型已加载）
# model.set_hook_registry(hook_registry, "prefix")

# 6. 运行推理，钩子会自动触发
# output = model.generate(input_ids)

# 7. 清理钩子
hook_registry.clear()
```

## 5. Implementation Details

### Memory Management
- **共享指针管理**: `HookRegistry` 使用 `std::shared_ptr` 管理，允许多个模块（Attention、MLP、DecoderLayer）共享同一钩子注册表，避免生命周期问题
- **Python 对象引用**: Python 回调函数通过 `py::object` 持有引用，pybind11 自动管理引用计数，防止 Python 端回调函数被垃圾回收

### Concurrency
- **GIL 保护**: 所有 Python-C++ 交互都在 Python GIL（Global Interpreter Lock）保护下执行，确保线程安全
- **异常传播**: 使用 `py::error_already_set` 捕获并重新抛出 Python 异常，保留原始异常堆栈信息，避免 C++ 异常跨越语言边界导致未定义行为

### Performance
- **零拷贝设计**: `infinicore::Tensor` 到 Python 的传递通过 pybind11 的缓冲协议实现零拷贝，避免数据复制
- **惰性钩子触发**: `CALL_HOOK` 宏通过 `has_hooks()` 检查避免在无钩子时产生性能开销，生产环境完全无影响

### Error Handling
- **类型安全转换**: `bos_token_id`/`eos_token_id` 的 `def_property` 实现运行时类型检查，支持 `int` 和 `List[int]` 两种输入，抛出 `py::type_error` 提供清晰错误信息
- **配置验证**: `validate()` 方法在 C++ 端执行数学约束检查，防止模型构建时的数值错误（如除零、维度不匹配）

### Dependencies
- **pybind11**: Python-C++ 绑定框架，提供类型安全的外部函数接口
- **InfiniCore**: 底层张量计算库，提供 `Tensor`、`Device`、`Module` 等核心抽象
- **C++ STL**: 使用 `std::unordered_map` 存储钩子，`std::function` 封装回调，`std::vector` 管理可变长度 token ID 列表

### Design Patterns
- **Bridge Pattern**: 绑定层作为 C++ 实现与 Python 接口之间的桥接，分离接口与实现
- **Observer Pattern**: HookRegistry 实现观察者模式，允许 Python 代码订阅模型内部的计算事件
- **Adapter Pattern**: `def_property` 实现了 Python 列表语义与 C++ `std::vector<int64_t>` 的适配，支持灵活的输入类型
- **Type Erasure**: `std::function` 和 `py::object` 实现类型擦除，统一存储不同签名的 Python 可调用对象

### Python Interoperability
- **属性内省**: `__dir__` 方法自定义 Python `dir()` 的输出，使 IDE 和 REPL 能够自动补全配置字段
- **TODO 标记**: 注释中明确指出未来重构方向（将 HookRegistry 从 Llama 特定工具迁移到 InfiniCore 通用工具集），体现架构演进意图
- **HuggingFace 兼容**: 字段命名和语义与 HuggingFace Transformers 的 `LlamaConfig` 完全对齐，支持无缝加载预训练模型配置

### Debug-Only Architecture
- **显式分离**: 钩子系统标记为 "DEBUG ONLY"，通过 `TODO` 注释和文档明确警告不应在生产环境使用
- **宏封装**: `CALL_HOOK` 和 `CALL_HOOK_LAYER` 宏简化钩子调用代码，自动处理空指针检查和 `has_hooks()` 短路逻辑，减少样板代码
- **分层钩子前缀**: 支持层级化钩子命名（如 `layer0_attention_q_proj`），通过 `BUILD_HOOK_PREFIX` 宏自动构建前缀树，避免命名冲突
