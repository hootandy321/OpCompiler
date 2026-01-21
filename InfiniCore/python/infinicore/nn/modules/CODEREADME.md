# InfiniCore Neural Network Modules Core Implementation Documentation

This module provides a PyTorch-compatible neural network layer abstraction for InfiniCore, implementing essential inference components including linear transformations, normalization, rotary position embeddings, embeddings, and module containers. All modules inherit from `InfiniCoreModule`, a base class derived from PyTorch's `torch.nn.Module` with adaptations for the InfiniCore tensor backend.

## 1. Module Structure

- **`module.py`**: Core `InfiniCoreModule` base class implementing parameter/buffer registration, state_dict serialization, and module hierarchy management
- **`container.py`**: `InfiniCoreModuleList` container for holding submodules in a list structure
- **`linear.py`**: `Linear` layer implementing affine transformation `y = xA^T + b`
- **`normalization.py`**: `RMSNorm` layer implementing Root Mean Square Layer Normalization
- **`rope.py`**: `RoPE` (Rotary Position Embedding) for transformer position encoding
- **`sparse.py`**: `Embedding` layer for sparse lookup table operations
- **`__init__.py`**: Public API exports exposing `Module`, `ModuleList`, `Linear`, `RMSNorm`, `Embedding`, and `RoPE`

## 2. Core Classes

### `InfiniCoreModule`
- **Location**: `module.py`
- **Primary Function**: Base class for all InfiniCore neural network modules, providing PyTorch-compatible module abstraction with custom parameter/buffer registration and state_dict management for InfiniCore tensors
- **Key Members**:
  - `_parameters: OrderedDict[str, Optional[Parameter]]`: Registered trainable parameters
  - `_buffers: OrderedDict[str, Optional[Tensor]]`: Registered non-trainable tensors (e.g., running statistics)
  - `_modules: OrderedDict[str, Optional[InfiniCoreModule]]`: Child modules forming tree hierarchy
  - `_non_persistent_buffers_set: Set[str]`: Buffers excluded from state_dict
  - `_version: int`: Module version for serialization compatibility
- **Core Methods**:
  - `__setattr__(name, value)`: Automatic parameter/buffer/module registration based on type (Parameter/Tensor/InfiniCoreModule)
  - `register_parameter(name, param)`: Explicit parameter registration with validation (name cannot contain "." or be empty)
  - `register_buffer(name, tensor, persistent=True)`: Buffer registration with persistent flag controlling state_dict inclusion
  - `add_module(name, module)`: Child module registration with validation
  - `state_dict(prefix="", keep_vars=False) -> OrderedDict`: Recursive serialization of parameters and persistent buffers with dot-notation keys
  - `load_state_dict(state_dict, strict=True) -> _IncompatibleKeys`: Deserialization with shape/dtype/device matching or replacement logic
  - `parameters(recurse=True) -> Iterator[Parameter]`: Yield parameters from module and optionally all submodules
  - `named_parameters(prefix="", recurse=True) -> Iterator[Tuple[str, Parameter]]`: Yield (name, parameter) tuples with full dot-notation paths
  - `buffers(recurse=True) -> Iterator[Tensor]`: Yield persistent buffers
  - `named_buffers(prefix="", recurse=True) -> Iterator[Tuple[str, Tensor]]`: Yield (name, buffer) tuples
  - `modules() -> Iterator[InfiniCoreModule]`: Yield all modules in hierarchy (deduplicated)
  - `named_modules(memo, prefix, remove_duplicate=True)`: Yield (name, module) tuples with full hierarchy paths
  - `children() -> Iterator[InfiniCoreModule]`: Yield immediate child modules only
  - `named_children() -> Iterator[Tuple[str, InfiniCoreModule]]`: Yield (name, child_module) for direct descendants
  - `__call__(*input, **kwargs)`: Forward method invocation via `self.forward(*input, **kwargs)`
  - `eval() -> self`: Evaluation mode setter (currently no-op, returns self)
- **Implementation Details**:
  - **Attribute Access Pattern**: `__getattr__` searches `_parameters`, `_buffers`, then `_modules` in order before raising AttributeError
  - **Registration Cleanup**: `__setattr__` uses `remove_from()` to ensure name exclusivity across parameters/buffers/modules/regular attributes
  - **State Dict Serialization**: `_save_to_state_dict` iterates `_parameters` and `_buffers` (excluding `_non_persistent_buffers_set`), prepending prefix for hierarchy
  - **State Dict Deserialization**: `_load_from_state_dict` performs tensor copy via `param.copy_(input_param)` when shape/dtype/device match, otherwise uses `setattr` replacement
  - **PyTorch Compatibility**: Supports mixed hierarchies with `torch.nn.Module` by checking `isinstance(module, InfiniCoreModule)` vs `isinstance(module, infinicore.nn.Module)` in `named_modules()`
  - **Version Metadata**: Stores `local_metadata = dict(version=self._version)` in state_dict._metadata for backward compatibility
  - **Error Handling**: `_IncompatibleKeys` named tuple returns `missing_keys` and `unexpected_keys` for non-strict loading diagnostics
- **Lifecycle**: Instantiation initializes empty OrderedDicts for `_parameters`, `_buffers`, `_modules`, and empty set for `_non_persistent_buffers_set`. Subclasses call `super().__init__()` then register parameters/modules via `self.weight = Parameter(...)` or `self.add_module("name", module)`

### `InfiniCoreModuleList`
- **Location**: `container.py`
- **Primary Function**: List-like container holding submodules with proper registration, enabling dynamic module collections indexed like Python lists
- **Key Members**:
  - `_modules: OrderedDict`: Stores modules with string-converted integer keys ("0", "1", ...)
  - `ModuleType: TypeVar`: Bound to `Union[Module]` for type hints
- **Core Methods**:
  - `__init__(modules: Optional[Sequence[ModuleType]] = None)`: Initialize with optional iterable, uses `__iadd__` to append modules
  - `__getitem__(idx: Union[int, slice])`: Integer indexing returns module, slice returns new `InfiniCoreModuleList`
  - `__setitem__(idx: int, module: ModuleType)`: Replace module at index via `add_module(str(idx), module)`
  - `__delitem__(idx: Union[int, slice])`: Delete module(s), reconstruct `_modules` with sequential string keys to preserve numbering
  - `__len__() -> int`: Return number of contained modules
  - `__iter__() -> Iterator[ModuleType]`: Yield `_modules.values()`
  - `__add__(other: Union[Sequence, InfiniCoreModuleList])`: Concatenate with list/tuple/ModuleList returning new instance
  - `__iadd__(modules: Sequence) -> self`: In-place extend via `extend()`
  - `append(module: ModuleType) -> self`: Add module to end, returns self for chaining
  - `extend(modules: Sequence) -> self`: Append iterable of modules starting at `offset = len(self)`
  - `insert(index: int, module: ModuleType)`: Insert at index by shifting modules from `len(_modules)` down to `index+1` rightward
  - `pop(idx: int = -1) -> ModuleType`: Remove and return module at index (default -1 for last), uses `__delitem__` for cleanup
  - `_get_abs_string_index(idx: int) -> str`: Convert negative/positive integer to absolute string index with bounds checking
  - `__repr__() -> str`: Multi-line representation showing indexed modules, e.g., `(0): Linear(...)`
  - `__dir__() -> List[str]`: Filter out numeric string keys from attribute listing
- **Implementation Details**:
  - **String Key Storage**: Uses `str(i)` keys in OrderedDict for list-like indexing while maintaining PyTorch's module registration requirements
  - **Index Normalization**: `_get_abs_string_index` converts negative indices (`-1` → last) and validates range `[-len, len-1]`
  - **Slice Reindexing**: `__delitem__` with slice reconstructs entire `_modules` OrderedDict with sequential string keys after deletion
  - **Type Safety**: `__add__` validates other is `list`, `tuple`, or `InfiniCoreModuleList`, raises TypeError otherwise
  - **Method Chaining**: `append()` and `extend()` return `self` for fluent interface
  - **dir() Cleanliness**: `__dir__` filters numeric keys to prevent cluttering IPython/tab-completion
- **Lifecycle**: Calls `super().__init__()` to initialize `_modules` OrderedDict, then `self += modules` if provided to trigger `__iadd__` → `extend()` → `add_module()` registration

### `Linear`
- **Location**: `linear.py`
- **Primary Function**: Affine transformation layer implementing `y = xA^T + b` via functional interface
- **Key Members**:
  - `in_features: int`: Input dimension
  - `out_features: int`: Output dimension
  - `weight: Parameter`: Trainable weight tensor shape `(out_features, in_features)`
  - `bias: Optional[Parameter]`: Optional bias tensor shape `(out_features)`
- **Core Methods**:
  - `__init__(in_features: int, out_features: int, bias: bool = False, device=None, dtype=None)`: Initialize with `infinicore.empty` tensors, defaults to CPU float32
  - `forward(input: Tensor) -> Tensor`: Call `F.linear(input, self.weight, self.bias)`
  - `extra_repr() -> str`: Return `in_features={...}, out_features={...}, bias={True/False}`
- **Implementation Details**:
  - **Factory Pattern**: Uses `factory_kwargs` dict to pass `device` (default `infinicore.device("cpu", 0)`) and `dtype` (default `infinicore.float32`)
  - **Bias Handling**: If `bias=False`, calls `register_parameter("bias", None)` to exclude from state_dict
  - **Functional Backend**: Delegates computation to `infinicore.nn.functional.linear`
  - **Constants**: `__constants__ = ["in_features", "out_features"]` for serialization optimization
- **Lifecycle**: Creates `Parameter(infinicore.empty([out_features, in_features]))` for weight, optionally creates bias parameter, registers via `self.weight =` assignment

### `RMSNorm`
- **Location**: `normalization.py`
- **Primary Function**: Root Mean Square Layer Normalization normalizing over last dimension: `output = input / sqrt(mean(input^2) + eps) * weight`
- **Key Members**:
  - `normalized_shape: tuple[int]`: Shape of dimensions to normalize over (last dimension(s))
  - `eps: float`: Small constant for numerical stability (default 1e-6)
  - `weight: Parameter`: Learnable affine scale parameter shape `normalized_shape`
- **Core Methods**:
  - `__init__(normalized_shape: Union[int, list[int]], eps: float = 1e-6, elementwise_affine: bool = True, device=None, dtype=None)`: Initialize weight parameter, asserts `elementwise_affine=True`
  - `forward(x: Tensor) -> Tensor`: Call `F.rms_norm(x, self.normalized_shape, self.weight, self.eps)`
  - `extra_repr() -> str`: Return `normalized_shape=[...], eps=...`
- **Implementation Details**:
  - **Shape Normalization**: Converts single `int` `normalized_shape` to `[normalized_shape]` list
  - **Elementwise Affine**: Only supports `elementwise_affine=True`, asserts otherwise
  - **Functional Backend**: Delegates to `infinicore.nn.functional.rms_norm`
  - **Constants**: `__constants__ = ["normalized_shape", "eps"]`
- **Lifecycle**: Creates `Parameter(infinicore.empty(normalized_shape))` for weight, registers via `self.weight =`

### `RoPE`
- **Location**: `rope.py`
- **Primary Function**: Rotary Position Embedding applying sinusoidal position encoding to transformer hidden states via rotation operation
- **Key Members**:
  - `max_position_embeddings: int`: Maximum sequence length supported
  - `rope_theta: float`: Base frequency for sinusoidal computation (default 10000.0)
  - `head_dim: int`: Attention head dimension (must be even)
  - `_sin_table: Tensor`: Precomputed sine table shape `(max_position_embeddings, head_dim // 2)`
  - `_cos_table: Tensor`: Precomputed cosine table shape `(max_position_embeddings, head_dim // 2)`
- **Core Methods**:
  - `__init__(max_position_embeddings: int, rope_theta: float, head_dim: int, device=None, dtype=None)`: Precompute sin/cos tables via `create_sin_cos_table()`
  - `forward(states: Tensor, position_ids: Tensor, algo=RopeAlgo.GPT_NEOX) -> Tensor`: Apply RoPE rotation in-place via `F.rope(states, position_ids, self._sin_table, self._cos_table, algo=algo, out=states)`
  - `create_sin_cos_table_numpy(max_position, head_dim, theta=10000.0) -> (np.ndarray, np.ndarray)`: Compute sin/cos using numpy with formula `freqs = 1.0 / (theta^(np.arange(0, head_dim, 2) / head_dim))`
  - `create_sin_cos_table(max_position, head_dim, theta=10000.0, device, dtype) -> (Tensor, Tensor)`: Convert numpy arrays to InfiniCore tensors via `infinicore.from_numpy()`
- **Implementation Details**:
  - **Sinusoidal Formula**: Computes `angles = outer(position, freqs)` where `freqs[i] = theta^(-2i/head_dim)` for `i in [0, head_dim/2)`
  - **Even Dimension Constraint**: `assert head_dim % 2 == 0` for paired sin/cos application
  - **Algorithm Variants**: Supports `RopeAlgo.GPT_NEOX` (default) and other algorithms via `algo` parameter
  - **In-Place Operation**: `forward` uses `out=states` for in-place modification
  - **Precomputation**: Tables computed once at initialization for efficiency
- **Lifecycle**: Computes sin/cos tables during `__init__` using numpy → InfiniCore tensor conversion, stores as buffers via `_sin_table`, `_cos_table` (note: these are Tensors, not registered as Parameters)

### `Embedding`
- **Location**: `sparse.py`
- **Primary Function**: Sparse lookup table mapping integer indices to dense embedding vectors via table lookup
- **Key Members**:
  - `num_embeddings: int`: Vocabulary size / number of embeddings
  - `embedding_dim: int`: Dimension of each embedding vector
  - `weight: Parameter`: Learnable embedding table shape `(num_embeddings, embedding_dim)`
- **Core Methods**:
  - `__init__(num_embeddings: int, embedding_dim: int, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None)`: Initialize weight parameter, asserts all optional features are `None/False`
  - `forward(input: Tensor) -> Tensor`: Call `F.embedding(input, self.weight)` performing table lookup
  - `extra_repr() -> str`: Return `num_embeddings={...}, embedding_dim={...}`
- **Implementation Details**:
  - **Unsupported Features**: Asserts `padding_idx`, `max_norm`, `scale_grad_by_freq`, `sparse`, `_weight`, `_freeze` are all disabled (inference-only design)
  - **Functional Backend**: Delegates to `infinicore.nn.functional.embedding`
  - **Constants**: `__constants__ = ["num_embeddings", "embedding_dim"]`
  - **Shape Flexibility**: Input can be arbitrary shape `(*)`, output is `(*, embedding_dim)` where last dimension is appended
- **Lifecycle**: Creates `Parameter(infinicore.empty([num_embeddings, embedding_dim]))` for weight table, registers via `self.weight =`

## 3. API Interface

```python
from infinicore.nn.modules import (
    Module, ModuleList, Linear, RMSNorm, Embedding, RoPE
)
import infinicore

# Base Module
class InfiniCoreModule:
    def __init__(self) -> None: ...
    def register_parameter(self, name: str, param: Parameter) -> None: ...
    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None: ...
    def add_module(self, name: str, module: Optional["InfiniCoreModule"]) -> None: ...
    def state_dict(self, *, prefix: str = "", keep_vars: bool = False) -> Dict[str, Any]: ...
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> _IncompatibleKeys: ...
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]: ...
    def named_parameters(self, prefix: str = "", recurse: bool = True) -> Iterator[Tuple[str, Parameter]]: ...
    def named_modules(self, memo: Optional[Set] = None, prefix: str = "", remove_duplicate: bool = True) -> Iterator[Tuple[str, "InfiniCoreModule"]]: ...
    def eval(self) -> "InfiniCoreModule": ...

# Linear Layer
class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    # Affine: y = xA^T + b

# RMSNorm Layer
class RMSNorm(Module):
    def __init__(
        self,
        normalized_shape: Union[int, list[int]],
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        device=None,
        dtype=None
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    # Normalize: output = input / sqrt(mean(input^2) + eps) * weight

# RoPE Layer
class RoPE(Module):
    def __init__(
        self,
        max_position_embeddings: int,
        rope_theta: float,
        head_dim: int,
        device=None,
        dtype=None
    ) -> None: ...
    def forward(self, states: Tensor, position_ids: Tensor, algo=RopeAlgo.GPT_NEOX) -> Tensor: ...
    # Rotary position embedding with precomputed sin/cos tables

# Embedding Layer
class Embedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        _freeze=False,
        device=None,
        dtype=None
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    # Sparse lookup: output = weight[input]

# ModuleList Container
class InfiniCoreModuleList(Module):
    def __init__(self, modules: Optional[Sequence[Module]] = None) -> None: ...
    def __getitem__(self, idx: Union[int, slice]) -> Union[Module, "InfiniCoreModuleList"]: ...
    def __setitem__(self, idx: int, module: Module) -> None: ...
    def __delitem__(self, idx: Union[int, slice]) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Module]: ...
    def append(self, module: Module) -> "InfiniCoreModuleList": ...
    def extend(self, modules: Sequence[Module]) -> "InfiniCoreModuleList": ...
    def insert(self, index: int, module: Module) -> None: ...
    def pop(self, idx: int = -1) -> Module: ...
```

## 4. Usage Example

```python
import infinicore
from infinicore.nn.modules import Module, Linear, RMSNorm, Embedding, RoPE, ModuleList

# Define a simple transformer block
class TransformerBlock(Module):
    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        max_seq_len: int = 2048,
        vocab_size: int = 50000,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Token embedding (sparse lookup)
        self.embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_dim,
            device=infinicore.device("cpu", 0),
            dtype=infinicore.float32
        )

        # RoPE for position encoding
        self.rope = RoPE(
            max_position_embeddings=max_seq_len,
            rope_theta=10000.0,
            head_dim=self.head_dim,
            device=infinicore.device("cpu", 0),
            dtype=infinicore.float32
        )

        # Attention projections (QKV)
        self.qkv_proj = Linear(
            in_features=hidden_dim,
            out_features=3 * hidden_dim,  # Q, K, V concatenated
            bias=False,
            device=infinicore.device("cpu", 0),
            dtype=infinicore.float32
        )

        # Output projection
        self.out_proj = Linear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            bias=False,
            device=infinicore.device("cpu", 0),
            dtype=infinicore.float32
        )

        # RMS Normalization
        self.norm = RMSNorm(
            normalized_shape=hidden_dim,
            eps=1e-6,
            elementwise_affine=True,
            device=infinicore.device("cpu", 0),
            dtype=infinicore.float32
        )

    def forward(self, input_ids: infinicore.Tensor, position_ids: infinicore.Tensor):
        # Shape: (batch_size, seq_len)
        batch_size, seq_len = input_ids.shape

        # 1. Embed tokens: (batch_size, seq_len, hidden_dim)
        hidden_states = self.embedding(input_ids)

        # 2. Project to QKV: (batch_size, seq_len, 3 * hidden_dim)
        qkv = self.qkv_proj(hidden_states)

        # 3. Reshape to heads: (batch_size, seq_len, num_heads, head_dim, 3)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(0, 3, 1, 4, 2)  # (batch, num_heads, seq_len, head_dim, 3)
        query, key, value = qkv.unbind(dim=-1)

        # 4. Apply RoPE to Q and K
        query = self.rope(query, position_ids, algo=RopeAlgo.GPT_NEOX)
        key = self.rope(key, position_ids, algo=RopeAlgo.GPT_NEOX)

        # 5. Scaled dot-product attention (simplified)
        # scores = (query @ key.transpose(-2, -1)) / sqrt(self.head_dim)
        # attn_weights = softmax(scores, dim=-1)
        # attn_output = attn_weights @ value
        # For brevity, skipping attention computation details

        # 6. Output projection (placeholder shape)
        attn_output = query.reshape(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output)

        # 7. RMS Normalization
        output = self.norm(output)

        return output

# Usage example
block = TransformerBlock(
    hidden_dim=768,
    num_heads=12,
    max_seq_len=2048,
    vocab_size=50000
)

# Save state dict
state_dict = block.state_dict()
# Returns OrderedDict with keys like:
# 'embedding.weight', 'rope._sin_table', 'rope._cos_table',
# 'qkv_proj.weight', 'out_proj.weight', 'norm.weight'

# Load state dict
block.load_state_dict(state_dict, strict=True)

# Inspect parameters
for name, param in block.named_parameters():
    print(f"{name}: shape={param.shape}, dtype={param.dtype}")

# Inspect modules
for name, module in block.named_modules():
    print(f"{name}: {type(module).__name__}")

# Example inference (requires actual tensor data)
# input_ids = infinicore.randint(0, 50000, (1, 128))
# position_ids = infinicore.arange(128).unsqueeze(0)
# output = block(input_ids, position_ids)

# Using ModuleList for stacking layers
class StackedBlocks(Module):
    def __init__(self, num_layers: int, **kwargs):
        super().__init__()
        self.layers = ModuleList([
            TransformerBlock(**kwargs) for _ in range(num_layers)
        ])

    def forward(self, input_ids, position_ids):
        for layer in self.layers:
            input_ids = layer(input_ids, position_ids)
        return input_ids

stacked = StackedBlocks(num_layers=12, hidden_dim=768, num_heads=12)
print(f"Number of layers: {len(stacked.layers)}")  # 12
```

## 5. Implementation Details

- **Memory Management**: All parameters stored as `InfiniCoreParameter` wrapping `infinicore.Tensor` objects. Tensor allocation uses `infinicore.empty()` for uninitialized memory, then later populated during checkpoint loading or initialization. No automatic memory pooling; relies on InfiniCore runtime's memory management.

- **Concurrency**: No explicit locking mechanisms. Python GIL protects `_parameters`, `_buffers`, `_modules` OrderedDict access during attribute registration/retrieval. Not designed for multi-threaded module construction; should be single-threaded during initialization.

- **Performance**:
  - **Parameter Registration**: O(1) dictionary insertion via OrderedDict with name collision checking
  - **State Dict Serialization**: O(N) traversal over module hierarchy where N = total parameters + buffers, uses recursion with prefix concatenation
  - **State Dict Deserialization**: O(N) traversal with tensor copy via `param.copy_(input_param)` when compatible, otherwise `setattr` replacement (O(1) reassignment)
  - **Module Iteration**: `named_modules()` uses memoization set to deduplicate shared modules, O(V) where V = unique module count
  - **RoPE Precomputation**: Sinusoidal tables computed once at initialization in numpy (CPU), then converted to InfiniCore tensors via `from_numpy()` with O(max_position * head_dim)

- **Error Handling**:
  - **Name Validation**: Parameter/buffer/module names cannot be empty string or contain "." (enforced in registration methods)
  - **Type Checking**: `register_parameter` requires `Parameter` or `None`; `add_module` requires `InfiniCoreModule` or `None`; `register_buffer` requires `Tensor` or `None`
  - **Attribute Collision**: Registration raises `KeyError` if name already exists in another registry (e.g., parameter name conflicts with existing module)
  - **State Dict Mismatch**: `load_state_dict` with `strict=True` raises `RuntimeError` listing missing/unexpected keys, returns `_IncompatibleKeys` with `strict=False`

- **Dependencies**:
  - **External**: `infinicore` core tensor library (`infinicore.Tensor`, `infinicore.device`, `infinicore.empty`, `infinicore.from_numpy`, `infinicore.float32`)
  - **Internal Modules**: `infinicore.nn.Parameter`, `infinicore.nn.functional` (F.linear, F.rms_norm, F.rope, F.embedding), `infinicore.nn.functional.RopeAlgo`
  - **Python Stdlib**: `collections.OrderedDict`, `itertools.chain`, `typing` (TypeVar, Iterator, Dict, List, Tuple, Union, overload), `numbers.Integral`, `warnings`

- **Design Patterns**:
  - **Composite Pattern**: `InfiniCoreModule` forms tree structure via `_modules` dict, enabling recursive `state_dict()` and `named_modules()`
  - **Registry Pattern**: Automatic type-based registration in `__setattr__` routes `Parameter` → `_parameters`, `Tensor` → `_buffers`, `InfiniCoreModule` → `_modules`
  - **Template Method**: `forward()` abstract method (no-op in base class) overridden by subclasses for actual computation
  - **Iterator Pattern**: Multiple generator methods (`parameters()`, `buffers()`, `modules()`, `named_children()`) for traversing module hierarchy
  - **Factory Method**: `factory_kwargs` dict pattern for consistent device/dtype passing to tensor constructors
  - **Facade Pattern**: ModuleList provides list-like interface over OrderedDict storage with string-integer indexing

- **PyTorch Compatibility**:
  - Derives `InfiniCoreModule` implementation from PyTorch v2.4.0 `torch.nn.Module` (BSD 3-Clause License, see header in `module.py`)
  - Adapts parameter/buffer registration for `infinicore.Tensor` instead of `torch.Tensor`
  - Supports mixed `InfiniCoreModule` + `torch.nn.Module` hierarchies via type checking in `named_modules()`
  - State dict format compatible with PyTorch's dot-notation key hierarchy
  - Exports same public API as PyTorch: `Linear`, `RMSNorm`, `Embedding`, `ModuleList`, `Module`
