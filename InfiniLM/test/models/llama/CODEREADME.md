# Llama Model Test Suite Core Implementation Documentation

This module provides comprehensive validation testing infrastructure for InfiniLM's Llama model implementation, comparing it against HuggingFace Transformers to ensure numerical correctness and functional equivalence. The test suite validates model inference, intermediate layer outputs, and cross-backend consistency.

## 1. Module Structure

- **`test_forward_validation.py`**: Validates forward pass inference across Python and C++ backends with different data types, testing prefill and decode steps with KV cache management.
- **`test_llama_inference.py`**: Performs end-to-end inference validation comparing InfiniLM model outputs against Transformers reference implementation for single-request scenarios.
- **`test_intermediate_validation.py`**: Systematically validates intermediate layer values (embeddings, attention, MLP, normalization) using hook-based interception and detailed tensor comparisons.
- **`utils.py`**: Provides shared utility functions for tensor type conversion between PyTorch and InfiniCore, parameter name normalization, and numerical comparison algorithms.

## 2. Core Components

### Test Forward Validation (`test_forward_validation.py`)

**Primary Function**: Cross-backend inference validation testing prefill-decode workflows with KV cache.

**Key Functions**:

- **`create_inputs(prompt, tokenizer, device, backend)`** (lines 92-121): Prepares input tensors for forward pass. Tokenizes prompt using chat template, creates position_ids [0, 1, ..., seq_len-1], converts to InfiniCore tensors using `infinicore.from_list()` with appropriate device placement (CPU for Python backend embedding, specified device for C++ backend).

- **`run_forward_pass(model, input_ids, position_ids, backend, dtype, num_decode_steps)`** (lines 124-346): Executes complete prefill + decode workflow. Backend-specific logic:
  - **C++ Backend**: Internal cache management, forward signature `forward(input_ids, position_ids)`
  - **Python Backend**: Explicit `DynamicCache` object, forward signature `forward(input_ids, position_ids, past_key_values, use_cache=True)`
  - Implements auto-regressive decode loop: each step uses previous step's predicted token (argmax of logits) as next input
  - Returns tuple of decode step logits tensors as numpy arrays for comparison

- **`infinicore_to_numpy(tensor)`** (lines 349-418): Converts InfiniCore tensors to numpy arrays. Special handling for bfloat16 dtype: reads raw uint16 data via ctypes.memmove, converts to torch.bfloat16, then to float32, then to numpy. Handles device transfer (GPU->CPU) and contiguity checks. Extensive debug logging for NaN detection during conversion.

- **`test_configuration(model_path, device, backend, dtype, prompt, num_decode_steps)`** (lines 421-504): Orchestrates complete test for a specific backend/dtype configuration. Steps: load tokenizer, create model via `infinilm.AutoLlamaModel.from_pretrained()`, load weights via `get_model_state_dict()`, create inputs, run forward pass, return decode logits tuple.

- **`compare_logits(logits1, logits2, name1, name2, step_name)`** (lines 507-549): Statistical comparison of logits arrays using numpy operations. Computes absolute/relative differences, applies tolerance (rtol=1e-2, atol=1.0 for bfloat16 vs float32), reports top 10 differences with positions.

**Main Execution** (`main` lines 552-631): Runs Python BF16 and C++ BF16 tests, compares decode step logits between backends, generates summary report.

### Test Llama Inference (`test_llama_inference.py`)

**Primary Function**: End-to-end inference validation against Transformers reference.

**Key Functions**:

- **`load_weights_into_infinilm_model(infinilm_model, transformers_model, infini_device, torch_device)`** (lines 65-110): Parameter transfer logic. Iterates through transformers state dict, normalizes parameter names (removes "model." prefix via `normalize_param_name`), finds matching keys in InfiniLM model, clones tensors to target device with `.contiguous()`, converts via `torch_to_infinicore_tensor()`, loads into InfiniLM via `load_state_dict()`. Returns count of matched parameters.

- **`validate_inference(model_dir, prompt, device_type, device_index)`** (lines 113-485): 9-step validation workflow:
  1. Device availability check using `_infinicore.get_device_count()`
  2. Load InfiniLM model via `LlamaForCausalLM.from_pretrained()`
  3. Load Transformers model with `torch.float32` precision
  4. Transfer weights using `load_weights_into_infinilm_model()`
  5. Tokenize prompt, create input_ids and position_ids
  6. Run Transformers inference with `torch.no_grad()`, extract logits and predictions
  7. Run InfiniLM inference via forward method, convert outputs to PyTorch tensors
  8. Compare logits shapes, predicted tokens, and numerical values using `tensor_all_close(rtol=1e-3, atol=1e-3)`
  9. Cleanup and report results

**Validation Logic** (lines 349-457): Comprehensive output comparison including shape verification, token-level prediction matching, logit numerical comparison with detailed statistics (max/mean abs diff, max rel diff), diagnostic warnings for NaN/Inf values, model collapse detection (predicts same token), and sample difference reporting showing top 5 max diff positions with actual values.

**CLI Interface** (`main` lines 488-578): Supports `--prompt` and `--device` arguments with validation.

### Test Intermediate Validation (`test_intermediate_validation.py`)

**Primary Function**: Granular layer-by-layer validation using hook-based intermediate value capture.

**Key Functions**:

- **`validate_infinicore_component(...)`** (imported from utils, lines 372-610 in utils.py): Implements systematic validation pattern for operator correctness:
  - **Test 1**: Run InfiniCore op with InfiniCore input → compare with InfiniLM output
  - **Test 2**: Run InfiniCore op with Transformers input → compare with Transformers output (eliminates input diff)
  - **Analysis**: Compare Test 1 vs Test 2 to quantify input difference impact
  - Returns detailed results dict with match status, statistics, input impact assessment

- **`validate_rope_component(...)`** (lines 69-169): RoPE-specific validation by re-applying RoPE in PyTorch. Handles Q/K head types separately, normalizes batch dimensions, applies `apply_rotary_pos_emb()` from Transformers, compares with relaxed tolerance (1e-5).

- **`validate_infinicore_rope_component(...)`** (lines 212-283): Validates InfiniCore RoPE implementation. Constructs `_infinicore.RoPE` module with head_dim, max_seq_len, rope_theta, GPT_NEOX algorithm (to match Transformers Llama's rotate_half), F32 dtype, delegates to `validate_infinicore_component()` for pattern-based validation.

- **Hook Registration for Transformers** (lines 478-748): Sophisticated hook system to capture intermediate values:
  - **Standard hooks**: `register_forward_hook()` for layer outputs (embed_tokens, layernorms)
  - **Attention wrapper**: Custom `attention_forward_wrapper()` (lines 514-649) intercepts attention forward, captures q_proj output before/after reshape, K/V projections, Q/K tensors before/after RoPE, attention weights, output before o_proj
  - **MLP wrapper**: Custom `mlp_forward_with_hooks()` (lines 677-693) captures gate_proj, up_proj, SwiGLU intermediate, down_proj outputs
  - **Pre-hooks**: Capture inputs to normalization layers

- **Hook Registration for InfiniLM** (lines 754-836): Uses `_infinilm.HookRegistry` with C++ binding support. Registers hook names with wildcard patterns (e.g., "layer0_*") to capture all layer 0 intermediate values. Hook callback converts InfiniCore tensors to PyTorch via `infinicore_to_torch_tensor()` and stores in `infinilm_intermediates` dict.

- **Systematic Validation Loop** (lines 871-1718): Validates 13 intermediate tensor pairs in computation order:
  1. Embedding outputs
  2. Input layernorm
  3. Q projection (before reshape) - validates linear transformation via InfiniCore matmul ops
  4. Q/K/V after reshape - validates projection + reshape via InfiniCore ops
  5. Q/K after RoPE - validates with both PyTorch reference and InfiniCore RoPE module (uses relaxed tolerance 5e-3 for RoPE steps due to numerical precision differences)
  6. Attention output before o_proj
  7. Final attention output
  8. Post-attention layernorm
  9. MLP intermediate values (gate_proj, up_proj, SwiGLU intermediate, final output)
  10. Final normalization

**Validation Features**:
  - Shape normalization (squeezing batch dimensions, permuting transposed layouts)
  - Tolerance adjustment per component type (stricter for linear layers, relaxed for RoPE)
  - Detailed difference analysis (max/mean abs diff, max rel diff, error distribution histograms, top 5 problematic positions with actual values)
  - InfiniCore ops validation for RMSNorm, Q projection, Q projection + reshape
  - RoPE dual validation (PyTorch reference + InfiniCore implementation)
  - MLP step-by-step breakdown to isolate mismatch source

**Validation Results** (lines 1719-1759): Prints summary table showing passed/failed/missing counts, lists each component's status with max diff values for failures. Note about RoPE tolerance (5e-3 acceptable for production use), identifies MLP precision alignment as next priority (max abs diff ~19.4).

### Utils (`utils.py`)

**Primary Function**: Shared tensor conversion and numerical comparison utilities.

**Key Functions**:

- **`normalize_param_name(name)`** (lines 17-21): Removes "model." prefix from parameter names for matching between Transformers and InfiniLM conventions.

- **`to_infinicore_dtype(torch_dtype)`** (lines 24-48): Converts PyTorch dtypes to InfiniCore dtype enum. Maps torch.float32/float16/bfloat16/int8/int16/int32/int64/uint8/bool to corresponding InfiniCore types. Raises ValueError for unsupported types.

- **`torch_to_infinicore_tensor(torch_tensor, infini_device)`** (lines 51-86): Converts PyTorch tensor to InfiniCore tensor sharing memory. Ensures contiguity with `.contiguous()`, converts dtype, uses `infinicore.from_blob()` for contiguous tensors (data_ptr, shape, dtype, device), or `infinicore.strided_from_blob()` for non-contiguous tensors (adds stride information). Zero-copy operation.

- **`to_torch_dtype(infini_dtype)`** (lines 89-127): Converts InfiniCore dtype to PyTorch dtype. Extracts underlying enum value via `_underlying` attribute, compares against `_infinicore.DataType` enum values (F32, F16, BF16, I8, I16, I32, I64, U8, BOOL). Raises ValueError for unsupported types.

- **`infinicore_to_torch_tensor(infini_tensor, torch_reference)`** (lines 130-267): Converts InfiniCore tensor to PyTorch tensor with data copying. Complex device-aware logic:
  - Wraps raw C++ `_infinicore.Tensor` in Python `infinicore.Tensor` wrapper if needed
  - Moves tensor to target device (CPU or GPU) based on reference tensor's device
  - Ensures contiguity via `.contiguous()`
  - Creates empty PyTorch tensor with target shape/dtype/device
  - **CPU path**: Uses `.to()` method for safe device transfer, or falls back to `infinicore.empty()` with `pin_memory=True` for CUDA→CPU copies (pinned memory required for D2H transfers), then uses `copy_()` operation
  - **GPU path**: Direct `copy_()` from InfiniCore tensor to PyTorch tensor
  - Returns PyTorch tensor with copied data

- **`tensor_all_close(tensor1, tensor2, rtol, atol)`** (lines 270-357): Numerical tensor comparison with detailed statistics. Handles NaN/Inf specially: computes stats only on finite values using `torch.isfinite()` mask. Returns tuple of (is_close, stats_dict) where stats contains max_abs_diff, mean_abs_diff, max_rel_diff, is_close, has_nan, has_inf flags, and per-tensor NaN/Indicators. Uses `torch.allclose()` for normal cases.

- **`validate_infinicore_component(...)`** (lines 360-610): Generic validation framework for comparing InfiniCore operations against reference implementations. Implements the dual-test pattern:
  - **Test 1**: Validates InfiniCore op correctness by comparing with InfiniLM output (tests if op reproduces InfiniLM behavior)
  - **Test 2**: Validates InfiniCore op implementation by running with Transformers input and comparing with Transformers output (isolates op correctness from input differences)
  - **Input Impact Analysis**: Compares Test 1 vs Test 2 outputs to quantify how input differences affect results
  - **Input Data Comparison**: Directly compares Transformers vs InfiniCore input tensors
  - **Debug Callback**: Optional callback for custom debugging logic
  - Returns comprehensive results dict with match status, statistics, input impact assessment ("minimal" or "significant")

## 3. API Interface

```python
# Tensor Conversion Utilities
def normalize_param_name(name: str) -> str
# Removes 'model.' prefix for parameter name matching

def to_infinicore_dtype(torch_dtype) -> infinicore.dtype
# Converts PyTorch dtype to InfiniCore dtype enum

def torch_to_infinicore_tensor(torch_tensor: torch.Tensor, infini_device: infinicore.Device) -> infinicore.Tensor
# Zero-copy conversion from PyTorch to InfiniCore tensor sharing memory

def infinicore_to_torch_tensor(infini_tensor: infinicore.Tensor, torch_reference: torch.Tensor) -> torch.Tensor
# Converts InfiniCore to PyTorch with data copying, device-aware

# Tensor Comparison
def tensor_all_close(tensor1: torch.Tensor, tensor2: torch.Tensor,
                    rtol: float = 1e-5, atol: float = 1e-5) -> Tuple[bool, Dict]
# Numerical comparison with statistics, handles NaN/Inf

# Validation Framework
def validate_infinicore_component(
    op_name: str,
    infinicore_op: Callable,
    transformers_input: torch.Tensor,
    transformers_output: torch.Tensor,
    infinicore_input: torch.Tensor,
    infinicore_output: torch.Tensor,
    infini_device: infinicore.Device,
    op_kwargs: Optional[Dict] = None,
    tolerance: float = 1e-5,
    debug_callback: Optional[Callable] = None,
    verbose: bool = True,
) -> Dict[str, Any]
# Dual-test validation pattern for operator correctness

# Inference Validation
def validate_inference(
    model_dir: str,
    prompt: str = "Hello, how are you?",
    device_type: str = "cpu",
    device_index: int = 0,
) -> bool
# End-to-end inference validation against Transformers

def load_weights_into_infinilm_model(
    infinilm_model,
    transformers_model,
    infini_device: infinicore.Device,
    torch_device: torch.device,
) -> int
# Transfers weights from Transformers to InfiniLM model, returns matched parameter count

# Intermediate Validation
def test_intermediate_validation(
    model_dir: str,
    device_type: str = "cpu",
    device_index: int = 0,
) -> bool
# Systematic layer-by-layer validation using hooks

# Forward Pass Validation
def run_forward_pass(
    model,
    input_ids: infinicore.Tensor,
    position_ids: infinicore.Tensor,
    backend: str,
    dtype: str,
    num_decode_steps: int = 2,
) -> Tuple[Tuple[np.ndarray, ...], bool]
# Executes prefill + decode workflow, returns decode logits tuple

def infinicore_to_numpy(tensor: infinicore.Tensor) -> np.ndarray
# Converts InfiniCore tensor to numpy, handles bfloat16 specially

def compare_logits(
    logits1: np.ndarray,
    logits2: np.ndarray,
    name1: str,
    name2: str,
    step_name: str = "logits",
) -> bool
# Statistical comparison of logits arrays
```

## 4. Usage Example

```python
# Example: Running comprehensive Llama model validation
import sys
sys.path.append('/path/to/InfiniLM/test/models/llama')

from test_llama_inference import validate_inference
from test_intermediate_validation import test_intermediate_validation
from test_forward_validation import test_configuration, main as forward_main

# Model path
model_dir = "/path/to/llama-3.2-1b-instruct"

# Test 1: End-to-end inference validation
print("=" * 80)
print("Test 1: End-to-End Inference Validation")
print("=" * 80)
success = validate_inference(
    model_dir=model_dir,
    prompt="Hello, how are you?",
    device_type="cpu",
    device_index=0
)
print(f"Inference validation: {'PASSED' if success else 'FAILED'}")

# Test 2: Intermediate layer validation
print("\n" + "=" * 80)
print("Test 2: Intermediate Layer Validation")
print("=" * 80)
success = test_intermediate_validation(
    model_dir=model_dir,
    device_type="cpu",
    device_index=0
)
print(f"Intermediate validation: {'PASSED' if success else 'FAILED'}")

# Test 3: Cross-backend forward pass validation
print("\n" + "=" * 80)
print("Test 3: Cross-Backend Forward Pass Validation")
print("=" * 80)

# Prepare arguments
import argparse
args = argparse.Namespace(
    model_path=model_dir,
    device="cpu",
    prompt="How are you",
    num_decode_steps=2
)

# Run Python backend test
from test_forward_validation import test_configuration
logits_py_bf16, error_py = test_configuration(
    args.model_path,
    args.device,
    backend="python",
    dtype="bfloat16",
    prompt=args.prompt,
    num_decode_steps=args.num_decode_steps
)

# Run C++ backend test
logits_cpp_bf16, error_cpp = test_configuration(
    args.model_path,
    args.device,
    backend="cpp",
    dtype="bfloat16",
    prompt=args.prompt,
    num_decode_steps=args.num_decode_steps
)

# Compare results if both succeeded
if not error_py and not error_cpp:
    from test_forward_validation import compare_logits
    num_steps = min(len(logits_py_bf16), len(logits_cpp_bf16))
    all_close = True
    for step_idx in range(num_steps):
        is_close = compare_logits(
            logits_py_bf16[step_idx],
            logits_cpp_bf16[step_idx],
            "Python BF16",
            "C++ BF16",
            f"decode step {step_idx + 1}"
        )
        all_close = all_close and is_close

    print(f"\nCross-backend validation: {'PASSED' if all_close else 'FAILED'}")
else:
    print(f"\nCross-backend validation: SKIPPED (errors occurred)")

# Expected Output:
# ================
# Test 1: End-to-End Inference Validation
# ------------------------------------------------------------------------
# This test compares inference outputs between InfiniLM and transformers
# for a single request scenario.
# Device: cpu:0
# Prompt: Hello, how are you?
# ------------------------------------------------------------------------
# ... (9-step validation process)
# ✓ Inference test completed successfully
# Inference outputs match between InfiniLM and transformers models.
# Single request scenario validated.
# ------------------------------------------------------------------------
#
# Test 2: Intermediate Layer Validation
# ------------------------------------------------------------------------
# Intermediate Values Validation Test
# Device: cpu:0
# ------------------------------------------------------------------------
# ... (8-step setup + systematic validation of 13 intermediate tensors)
# Validation Summary
# ------------------
# Note: RoPE validation (steps 9.7 and 9.8) uses relaxed tolerance (5e-3)
#       due to float32 numerical precision differences after refactoring.
#       Max abs diff is ~4e-3, which is acceptable for production use.
#
# Next Focus: MLP precision alignment
#   - layer0_mlp shows significant mismatch (max abs diff: ~19.4)
#   - This is the next priority for precision alignment work.
# ------------------------------------------------------------------------
# Total validations: 13
#   ✓ Passed: 11
#   ✗ Failed: 1
#   ⚠ Missing: 1
#
# Test 3: Cross-Backend Forward Pass Validation
# ------------------------------------------------------------------------
# Testing: Backend=python, Dtype=bfloat16
# ------------------------------------------------------------------------
# ... (5-step test configuration)
#  ✓ Forward pass completed (prefill + 2 decode step(s))
#
# Testing: Backend=cpp, Dtype=bfloat16
# ------------------------------------------------------------------------
# ... (5-step test configuration)
#  ✓ Forward pass completed (prefill + 2 decode step(s))
#
# COMPARISON RESULTS
# ------------------------------------------------------------------------
# Comparing: Python BF16 vs C++ BF16 (decode step 1)
# ------------------------------------------------------------------------
#   Max absolute difference: 0.015625
#   Mean absolute difference: 0.000234
#   ✓ Logits are close (within tolerance)
#
# Comparing: Python BF16 vs C++ BF16 (decode step 2)
# ------------------------------------------------------------------------
#   Max absolute difference: 0.015625
#   Mean absolute difference: 0.000189
#   ✓ Logits are close (within tolerance)
#
# SUMMARY
# ------------------------------------------------------------------------
# python_bf16          : ✓ SUCCESS
# cpp_bf16            : ✓ SUCCESS
#
# Comparisons:
#   Python BF16 vs C++ BF16 (decode step 1):   ✓ CLOSE
#   Python BF16 vs C++ BF16 (decode step 2):   ✓ CLOSE
#
# ✓ All tests completed successfully
```

## 5. Implementation Details

**Memory Management**:
- **Zero-Copy Conversions**: `torch_to_infinicore_tensor()` uses `from_blob()` to share memory between PyTorch and InfiniCore without copying data. Ensures contiguity before wrapping.
- **Explicit Copying**: `infinicore_to_torch_tensor()` performs data copying using `copy_()` operation. For CUDA→CPU transfers, uses `pin_memory=True` to allocate pinned host memory for efficient D2H transfers via `ctypes.memmove`.
- **BFloat16 Special Handling**: `infinicore_to_numpy()` reads raw uint16 data, converts through torch.bfloat16 → float32 → numpy to work around InfiniCore's limited bfloat16 support.
- **Tensor Lifecycle**: Weight loading uses `torch_tensors_keepalive` list to prevent garbage collection during transfer. Clears references after `load_state_dict()` to free memory.

**Concurrency**:
- **Backend-Specific Cache Management**: Python backend uses explicit `DynamicCache` object passed to forward, C++ backend manages internal cache state. Single-threaded execution during validation.
- **Hook System**: PyTorch hooks use `register_forward_hook()` and `register_forward_pre_hook()` with thread-safe callback registration. C++ hooks via `_infinilm.HookRegistry` with string pattern matching.
- **Device Management**: Device availability checked via `_infinicore.get_device_count()`. Device objects created once and reused across conversions.

**Performance**:
- **Batch Processing**: Forward validation runs multiple decode steps sequentially, using previous step's predicted token as next input (auto-regressive generation).
- **Vectorized Operations**: Tensor comparisons use numpy/torch vectorized operations (`.max()`, `.mean()`, `.abs()`, `.flatten()`). Top-k difference finding uses `torch.topk()`.
- **Early Termination**: Validation loops exit early on critical failures (shape mismatches, missing tensors).
- **Tolerance Selection**: Uses relaxed tolerance for bfloat16 (rtol=1e-2, atol=1.0) vs float32 (rtol=1e-3, atol=1e-3). RoPE steps use 5e-3 tolerance due to numerical precision differences.

**Error Handling**:
- **Graduated Error Reporting**: Distinguishes between shape mismatches (configuration error), NaN/Inf (numerical instability), large differences (implementation bug), and small differences (acceptable precision variance).
- **Diagnostic Messages**: Provides actionable diagnostics:
  - "Model collapse - model is predicting same token" → attention mechanism issue
  - "Large logit differences detected" → weight loading/attention/layer processing issues
  - "Input difference causes significant output difference" → upstream problem
  - "InfiniCore ops differs from Transformers even with same input" → operator implementation bug
- **Exception Propagation**: All test functions catch exceptions, print stack trace, return False/None, and clean up resources in finally blocks.
- **Validation Pattern**: The dual-test pattern (Test 1 with InfiniCore input, Test 2 with Transformers input) isolates whether differences come from input data or operator implementation.

**Dependencies**:
- **External**: torch, transformers, infinicore, infinilm, numpy
- **Internal**: Shared `utils.py` for conversion/comparison functions
- **Inter-Module**: Uses `_infinilm.HookRegistry` C++ binding for hook registration, `_infinicore.Device.Type` for device type enums, `_infinicore.RoPE` for RoPE validation, `_infinicore.DataType` for dtype conversion
- **Model Loading**: `transformers.LlamaForCausalLM.from_pretrained()`, `infinilm.AutoLlamaModel.from_pretrained()`, `infinilm.models.llama.LlamaForCausalLM`, `infinilm.models.llama.LlamaConfig`

**Design Patterns**:
- **Template Method**: `test_configuration()` defines test orchestration skeleton, subclasses can override specific steps
- **Strategy Pattern**: Backend-specific logic in `run_forward_pass()` (Python vs C++ cache management)
- **Hook Pattern**: Extensive use of hooks for intermediate value capture without modifying model code
- **Factory Pattern**: `create_inputs()` factory creates appropriate tensor types based on backend
- **Builder Pattern**: `validate_infinicore_component()` builds complex validation logic from op function, inputs, outputs
- **Observer Pattern**: Hook callbacks observe intermediate values and store in dictionaries
- **Adapter Pattern**: Conversion functions adapt between PyTorch and InfiniCore tensor formats
- **Validator Pattern**: Systematic validation loop iterates through pre-defined component list, applies consistent validation logic

**Special Algorithms**:
- **Parameter Name Normalization**: Simple prefix stripping ("model." → "") to handle naming convention differences
- **RoPE Validation**: Dual validation approach using PyTorch reference (`apply_rotary_pos_emb`) and InfiniCore implementation (`_infinicore.RoPE`). Uses GPT_NEOX rotation pairing to match Llama's `rotate_half` behavior
- **Tensor Layout Alignment**: `align_attention_tensor_layout()` detects transposed layouts and permutes to match. `format_rope_tensor_for_module()` converts to [seq_len, num_heads, head_dim] format required by InfiniCore RoPE
- **Numerical Precision Analysis**: Computes error distribution histograms, reports percentage of elements exceeding thresholds (1e-6, 1e-5, 1e-4, 1e-3, 1e-2), identifies top-k problematic positions
- **Weight Transfer**: Iterates through state dicts, normalizes names, matches via normalized comparison, clones tensors to target device, converts dtype, loads into model
