# ntops/tests Test Suite Implementation Documentation

Comprehensive PyTorch operator compatibility test suite for validating ntops library implementations against PyTorch reference implementations. This test infrastructure ensures numerical correctness across 40+ tensor operations with parametric testing, random seed control, and multi-datatype coverage.

## 1. Module Structure

- **`conftest.py`**: pytest configuration hooks for CUDA precision control and deterministic random seeding
- **`skippers.py`**: Conditional test decorators for hardware availability checks
- **`utils.py`**: Parametric test data generation with random shape/dtype/device combinations
- **`__init__.py`**: Package marker (empty)
- **`test_*.py` (44 files)**: Individual operator test suites covering arithmetic, bitwise, comparison, matrix operations, and neural network primitives

## 2. Core Utilities

### `pytest_configure()` (conftest.py)
- **Location**: `conftest.py:10-14`
- **Primary Function**: Global test suite initialization hook
- **Key Actions**:
  - Disables reduced-precision reduction in CUDA matmul operations (`allow_fp16_reduced_precision_reduction = False`, `allow_bf16_reduced_precision_reduction = False`)
  - Sets default max configurations for ntops kernel auto-tuning via `ntops.torch.utils.set_default_max_num_configs(3)`
- **Lifecycle**: Executed once at pytest session startup

### `_set_random_seed(seed)` (conftest.py)
- **Location**: `conftest.py:35-37`
- **Primary Function**: Ensures reproducible random number generation
- **Implementation Details**:
  - Seeds Python's `random` module and PyTorch's RNG with identical value
  - Called at module level (via `pytest_collectstart`) and per-test (via fixture)
  - Prevents flaky tests through deterministic tensor initialization

### `_hash(string)` (conftest.py)
- **Location**: `conftest.py:48-49`
- **Primary Function**: Generates stable 32-bit integer hash from test path
- **Algorithm**: SHA-256 hash truncated to 32 bits via modulo 2^32
- **Usage**: Converts test module/function names into unique random seeds

### `generate_arguments(use_float=True)` (utils.py)
- **Location**: `utils.py:7-27`
- **Primary Function**: Parametric test data generator
- **Return Format**: Tuple of (param_names_string, test_arguments_list)
- **Parameters Generated**:
  - **Shape**: Random tensor dimensions (1D to 4D) via `_random_shape()`
  - **Dtype**: For `use_float=True`: `(torch.float32, torch.float16)`; for `False`: `(torch.bool, torch.int8, torch.int16, torch.int32)`
  - **Device**: Always `"cuda"`
  - **Tolerances**: `rtol=0.001, atol=0.001` for float32; `rtol=0.01, atol=0.01` for float16
- **Generation Strategy**: Cartesian product of ndim (1-4) and dtype arrays
- **Complexity**: Generates 8 test cases for floating-point mode (4 ndims × 2 dtypes)

### `_random_shape(ndim, min_num_elements=2^8, max_num_elements=2^10)` (utils.py)
- **Location**: `utils.py:34-48`
- **Primary Function**: Generates random tensor shapes with controlled element count
- **Algorithm**:
  1. Randomly select total element count between 256 and 1024
  2. Iteratively decompose into `ndim-1` dimensions using `random.randint(1, sqrt(remaining))`
  3. Assign remaining elements to final dimension
  4. Shuffle dimensions randomly
- **Purpose**: Ensures tests cover varied tensor geometries while controlling memory footprint

### `skip_if_cuda_not_available(func)` (skippers.py)
- **Location**: `skippers.py:5-8`
- **Primary Function**: Conditional test skip decorator
- **Implementation**: Wraps function with `pytest.mark.skipif` checking `torch.cuda.is_available()`
- **Usage Pattern**: Applied to all CUDA-requiring tests

## 3. Test Patterns

### Basic Arithmetic Operations

**Coverage**: `test_add.py`, `test_sub.py`, `test_mul.py`, `test_div.py`

**Canonical Template** (exemplified by `test_add`):
```python
@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_add(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)
    alpha = gauss()  # Random scalar from utils.gauss()

    ninetoothed_output = ntops.torch.add(input, other, alpha=alpha)
    reference_output = torch.add(input, other, alpha=alpha)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
```

**Key Characteristics**:
- Alpha scaling factor generated via `gauss()` (Gaussian distribution μ=0, σ=1)
- Element-wise binary operations with identical shapes
- Division test supports `rounding_mode` parameter (None, "floor"; "trunc" skipped)

### Unary Mathematical Functions

**Coverage**: `test_abs.py`, `test_exp.py`, `test_sin.py`, `test_cos.py`, `test_tanh.py`, `test_sigmoid.py`, `test_neg.py`, `test_rsqrt.py`

**Pattern**:
```python
def test_<op>(shape, dtype, device, rtol, atol):
    if dtype is torch.float16:  # Precision limitations
        return
    input = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.<op>(input)
    reference_output = torch.<op>(input)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
```

**Special Cases**:
- **Exponential/Trig**: Skip float16 due to precision limitations
- **rsqrt/pow**: Use `equal_nan=True` in allclose for NaN handling
- **isinf/isnan**: Use exact equality `torch.equal()` instead of tolerance comparison

### Bitwise Operations

**Coverage**: `test_bitwise_and.py`, `test_bitwise_or.py`, `test_bitwise_not.py`

**Data Generation Strategy**:
```python
if dtype == torch.bool:
    prob = 0.5
    input = torch.rand(shape, dtype=torch.float32, device=device) > prob
else:
    upper_bound = 10
    input = torch.randint(-upper_bound, upper_bound, size=shape, dtype=dtype, device=device)
```

- Boolean tensors generated via thresholding uniform random distribution
- Integer tensors use bounded random generation [-10, 10]
- Exact equality assertion: `torch.equal(ninetoothed_output, reference_output)`

### Comparison Operations

**Coverage**: `test_eq.py`, `test_ne.py`, `test_lt.py`, `test_le.py`, `test_gt.py`, `test_ge.py`

**Implementation**:
```python
def test_<comparison>(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.<comparison>(input, other)
    reference_output = torch.<comparison>(input, other)

    assert torch.equal(ninetoothed_output, reference_output)
```

- Returns boolean tensors
- Requires exact equality (no tolerance)
- Covers all six comparison operators

### Matrix Operations

#### `test_mm.py` - Matrix Multiplication
- **Location**: `test_mm.py:10-32`
- **Custom Generator**: Generates `(m, n, k)` dimensions with random sizes 1-1024
- **Test Cases**: 2 configurations × 2 dtypes = 4 parametric combinations
- **Shapes**: Input `(m, k)`, Other `(k, n)`, Output `(m, n)`

#### `test_bmm.py` - Batched Matrix Multiplication
- **Location**: `test_bmm.py:11-21`
- **Batch Size**: Random integer 4-16
- **Shapes**: Input `(b, m, k)`, Other `(b, k, n)`, Output `(b, m, n)`
- **Imports generator from test_mm**

#### `test_addmm.py` - Matrix Multiplication with Bias Addition
- **Location**: `test_addmm.py:10-22`
- **Formula**: `output = input × beta + (x @ y) × alpha`
- **Parameters**: Random scalars `beta` and `alpha` from Gaussian distribution
- **Shapes**: Input `(m, n)`, X `(m, k)`, Y `(k, n)`

#### `test_matmul.py` - General Matrix Multiplication
- **Location**: `test_matmul.py:9-22`
- **Batch Support**: Tests `b ∈ {None, 1, 2, 3}` for 2D/3D tensor combinations
- **Shapes**:
  - No batch: Input `(m, k)`, Other `(k, n)`
  - With batch: Input `(b, m, k)`, Other `(b, k, n)`

### Neural Network Primitives

#### `test_relu.py` - ReLU Activation
- **Location**: `test_relu.py:10-19`
- **Parameters**: `inplace ∈ {False, True}`
- **Reference**: `torch.nn.functional.relu(input, inplace)`

#### `test_gelu.py` - GELU Activation
- **Location**: `test_gelu.py:10-27`
- **Parameters**: `approximate ∈ {"none", "tanh"}` (tanh skipped)
- **Reference**: `F.gelu(input, approximate=approximate)`

#### `test_silu.py` - SiLU Activation
- **Location**: `test_silu.py:10-19`
- **Reference**: `F.silu(input)`
- **Inplace**: TODO marked for future testing

#### `test_dropout.py` - Dropout Regularization
- **Location**: `test_dropout.py:12-36`
- **Parameters**: `p ∈ [0, 1]` (random uniform)
- **Validation Strategy**:
  1. Shape preservation: `ninetoothed_output.shape == reference_output.shape`
  2. Sparsity ratio: Non-zero element ratio within 10% tolerance
  3. Scaling correctness: Non-zero values scaled by `1/(1-p)`
- **Training/Inplace Modes**: TODO marked

#### `test_softmax.py` - Softmax Normalization
- **Location**: `test_softmax.py:11-21`
- **Parameters**:
  - `dim`: Random dimension 0 to ndim-1
  - `dtype`: Random choice from `(float16, float32, float64)`
- **Reference**: `torch.nn.functional.softmax(input, dim=dim, dtype=dtype)`

#### `test_layer_norm.py` - Layer Normalization
- **Location**: `test_layer_norm.py:11-37`
- **Parameters**:
  - `eps ∈ {1e-8, 1e-5, 1e-3}`
  - `weight_is_none ∈ {False, True}`
  - `bias_is_none ∈ {False, True}`
- **Normalized Shape**: Random suffix of input shape (1 to full length)
- **Reference**: `torch.nn.functional.layer_norm`

#### `test_rms_norm.py` - Root Mean Square Normalization
- **Location**: `test_rms_norm.py:11-30`
- **Parameters**: `eps ∈ {None, 0, 1e-5, 1e-3}`, `weight_is_none ∈ {False, True}`
- **Reference**: `torch.nn.functional.rms_norm`

#### `test_scaled_dot_product_attention.py` - Scaled Dot-Product Attention
- **Location**: `test_scaled_dot_product_attention.py:14-186`
- **Complexity**: Most sophisticated test with 19 parametric dimensions
- **Generator** (`generate_arguments()`):
  - **Shapes**: Random batch (1-4), heads_q (2-32), seq_q (1-512), head_dim (32/64), heads_kv (2 to heads_q), seq_kv (1-512)
  - **Mask Types**: `None`, `torch.bool`, `torch.float32`
  - **Causal**: Boolean flag (mutually exclusive with mask)
  - **Scale**: `None` or random 0.05-0.5
  - **GQA**: Boolean (Grouped Query Attention)
  - **Causal Variant**: `None`, `CausalVariant.LOWER_RIGHT`, `CausalVariant.UPPER_LEFT`
  - **KV Cache**: Boolean with present tensor/slot generation
- **Validation**:
  - Custom causal mask generation via `causal_lower_right()`
  - Reference: `F.scaled_dot_product_attention` with PyTorch-native GQA support
- **Constraints**:
  - Skips combinations with both mask and is_causal=True
  - Skips LOWER_RIGHT variant when seq_q > seq_kv

#### `test_rotary_position_embedding.py` - Rotary Position Embedding (RoPE)
- **Location**: `test_rotary_position_embedding.py:7-89`
- **Parameters**:
  - `batch_size ∈ {1, 4}`
  - `seq_len ∈ {1, 128}`
  - `num_heads ∈ {1, 8}`
  - `emb_dim ∈ {32, 64}`
  - `interleaved ∈ {False, True}`
  - `inplace ∈ {False, True}`
  - `dtype ∈ {float32, float16}`
- **Helper Functions**:
  - `_torch_rotary_position_embedding()`: Reference implementation
    - **Interleaved Mode**: Pairs (x0, x1) at positions 2i, 2i+1
      ```
      x0_rotated = x0 * cos - x1 * sin
      x1_rotated = x0 * sin + x1 * cos
      ```
    - **Non-Interleaved Mode**: Halves at [:dim//2] and [dim//2:]
  - `_generate_sin_and_cos_tables()`: Creates position embeddings
    - Base: 10000
    - Theta calculation: `base^(-2 * arange(emb_dim/2) / emb_dim)`
    - Broadcast multiplication with position indices
- **Validation**: Custom reference implementation vs ntops output

#### `test_clamp.py` - Element-wise Clipping
- **Location**: `test_clamp.py:9-19`
- **Parameters**: Random min/max tensors (same shape as input)
- **Formula**: `output = min(max(input, min), max)`

#### `test_pow.py` - Element-wise Exponentiation
- **Location**: `test_pow.py:9-24`
- **Exponent**: Random tensor (same shape as input)
- **Special Handling**: `equal_nan=True` for NaN edge cases

## 4. Test Execution Flow

### Per-Test Lifecycle
1. **Module Collection** (`pytest_collectstart`):
   - Hash module filename → seed
   - Set global random seed

2. **Test Execution**:
   - Hash `{module_path}::{test_name}` → unique seed
   - Apply seed via fixture
   - Generate parametric arguments (if parametrized)
   - Create tensors with seeded random generators
   - Execute ntops and PyTorch operations
   - Assert numerical equivalence

3. **Assertion Strategies**:
   - **Floating-point**: `torch.allclose(rtol, atol)` with dtype-specific tolerances
   - **Boolean/Integer**: `torch.equal()` for exact matching
   - **Special Cases**: `equal_nan=True` for rsqrt/pow

### Parametric Testing Coverage
- **Shape Space**: 1D to 4D tensors with 256-1024 elements
- **Dtype Coverage**: float32, float16 (float ops); bool, int8, int16, int32 (bitwise ops)
- **Device**: CUDA-only (skipped if unavailable)
- **Combinatorial Explosion**: Each test spawns 8-100+ instances via Cartesian products

## 5. Implementation Details

### Random Seed Determinism
- **Module-level Seed**: Hash of `"tests/test_xxx.py"` string
- **Test-level Seed**: Hash of `"tests/test_xxx.py::test_func[param-id]"` string
- **Collision Resistance**: SHA-256 ensures < 2^-32 probability
- **Thread Safety**: PyTorch RNG operations are thread-hostile; tests run sequentially

### Memory Management
- **Tensor Allocation**: All tensors created on CUDA device via `torch.randn(..., device="cuda")`
- **Cleanup**: Automatic via Python GC and PyTorch's CUDA memory allocator
- **No Explicit Synchronization**: Relies on implicit synchronization in allclose()

### Precision Control
- **FP16/BF16 Matmul**: Disabled reduced-precision reduction for deterministic results
- **Tolerance Hierarchy**:
  - float32: 0.001 relative/absolute
  - float16: 0.01 relative/absolute (10× relaxed)
  - Attention: 0.01 (FP32), 0.025 (FP16)

### Known Limitations (TODOs)
1. **Float16 Precision**: Trigonometric, exponential, and activation functions skip float16
2. **Rounding Modes**: `div()` skips "trunc" mode
3. **GELU Approximation**: "tanh" approximate mode skipped
4. **Inplace Operations**: Dropout and SiLU inplace variants untested
5. **Infinite Masks**: Non-infinite float masks in SDPA cause precision issues

### Auto-Tuning Configuration
- **Max Configs**: 3 kernel configurations per operation
- **Setting**: Applied globally via `ntops.torch.utils.set_default_max_num_configs()`
- **Purpose**: Limits kernel search space during testing for faster execution

## 6. Usage Example

```python
# Example: Running specific test subset
pytest tests/test_mm.py -v  # Verbose mode for matrix multiplication tests

# Run all tests with coverage reporting
pytest tests/ --cov=ntops --cov-report=html

# Filter by CUDA availability (auto-skipped)
pytest tests/test_abs.py  # Skipped if CUDA unavailable

# Parametric test expansion visualization
pytest tests/test_add.py --collect-only
# Output: tests/test_add.py::test_add[shape0-float32-cuda-0.001-0.001]
#         tests/test_add.py::test_add[shape1-float16-cuda-0.01-0.01]
#         ... (8 total combinations)
```

**Reproducing Specific Test Failure**:
```python
# Extract seed from failing test log
# Example: "Using seed 1234567890 from tests/test_exp.py::test_exp[...]"

import random
import torch

seed = 1234567890
random.seed(seed)
torch.manual_seed(seed)

shape = [256, 4]  # From test parameters
dtype = torch.float32
device = "cuda"

input = torch.randn(shape, dtype=dtype, device=device)  # Reproducible tensor
```

## 7. Test Statistics

- **Total Test Files**: 44 test modules
- **Lines of Code**: ~1,800 (excluding utilities)
- **Parametric Combinations**: ~1,500+ test cases when fully expanded
- **Operator Coverage**:
  - Arithmetic: 7 ops (add, sub, mul, div, neg, pow, abs)
  - Bitwise: 3 ops (and, or, not)
  - Comparison: 6 ops (eq, ne, lt, le, gt, ge)
  - Math: 10 ops (sin, cos, exp, rsqrt, clamp, isinf, isnan)
  - Matrix: 4 ops (mm, bmm, addmm, matmul)
  - Neural: 11 ops (relu, gelu, silu, sigmoid, tanh, softmax, layer_norm, rms_norm, dropout, scaled_dot_product_attention, rotary_position_embedding)
