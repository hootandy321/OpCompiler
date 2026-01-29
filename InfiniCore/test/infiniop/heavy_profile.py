"""
Profile benchmark: unfused (separate ops) vs fused (single op) latency.

Compares:
  - add + rms_norm  vs  add_rms_norm (fused)
  - silu + mul      vs  swiglu       (fused)

Outputs a JSON dict to stdout at the end:
{
  "unfused": { "add+rms_norm": { "[1, 512, 4096]": 0.1234, ... }, ... },
  "fused":   { "add+rms_norm": { "[1, 512, 4096]": 0.0987, ... }, ... }
}

Run:
    python profile.py --nvidia --profile
    python profile.py --metax  --profile --num_prerun 20 --num_iterations 100
"""

import json
import math
import time
import ctypes
from ctypes import c_uint64
from typing import Dict, Any
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    get_args,
    TestWorkspace,
    InfiniDtype,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)
from libinfiniop.utils import create_handle, destroy_handle, get_sync_func
from libinfiniop.devices import torch_device_map

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DTYPE = InfiniDtype.F16
EPS = 1e-6

# Accumulated results dict (same structure as test/infinicore/profile.py)
_results: Dict[str, Any] = {"unfused": {}, "fused": {}}

# ---------------------------------------------------------------------------
# Shape generation: 100 shapes per op-combo, log-spaced from low to high load.
#
# Memory budget: 32 GB VRAM.  Peak tensor count during one profile call:
#   ADD_RMS_NORM  ≈ 9 tensors of shape  -> max ≈ 9×131072×4096×2 B ≈ 9.7 GB
#   SILU_MUL      ≈ 7 tensors of shape  -> max ≈ 7× 65536×11008×2 B ≈ 10.1 GB
# Both fit comfortably within 32 GB.
# ---------------------------------------------------------------------------

def _generate_profile_shapes(hidden_dim, min_seqlen, max_seqlen, count=100):
    """Generate `count` shapes (1, seqlen, hidden_dim) with seqlen log-spaced."""
    log_min = math.log(min_seqlen)
    log_max = math.log(max_seqlen)
    shapes = []
    seen = set()
    for i in range(count):
        t = i / (count - 1)
        seqlen = int(round(math.exp(log_min + t * (log_max - log_min))))
        if seqlen not in seen:
            seen.add(seqlen)
            shapes.append((1, seqlen, hidden_dim))
    return shapes

# 100 shapes: seqlen 64 → 131072, hidden_dim = 4096
ADD_RMS_NORM_SHAPES = _generate_profile_shapes(4096, 64, 131072)

# 100 shapes: seqlen 64 → 65536, hidden_dim = 11008
SILU_MUL_SHAPES = _generate_profile_shapes(11008, 64, 65536)

# ---------------------------------------------------------------------------
# Helpers: build descriptor + workspace, return a callable kernel launcher
# ---------------------------------------------------------------------------


def _make_add(handle, device, shape, dtype):
    """Return (a, b, c, run_fn, cleanup_fn) for Add kernel."""
    a = TestTensor(shape, None, dtype, device)
    b = TestTensor(shape, None, dtype, device)
    c = TestTensor(shape, None, dtype, device, mode="zeros")

    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateAddDescriptor(
            handle, ctypes.byref(desc), c.descriptor, a.descriptor, b.descriptor,
        )
    )
    for t in [a, b, c]:
        t.destroy_desc()

    ws_size = c_uint64(0)
    check_error(LIBINFINIOP.infiniopGetAddWorkspaceSize(desc, ctypes.byref(ws_size)))
    ws = TestWorkspace(ws_size.value, device)

    def run():
        check_error(
            LIBINFINIOP.infiniopAdd(
                desc, ws.data(), ws.size(), c.data(), a.data(), b.data(), None,
            )
        )

    def cleanup():
        check_error(LIBINFINIOP.infiniopDestroyAddDescriptor(desc))

    return a, b, c, run, cleanup



def _make_rms_norm(handle, device, shape, w_shape, dtype, x_tensor):
    """Return (y, run_fn, cleanup_fn) for RMSNorm kernel.
    x_tensor: a TestTensor whose data() is used as the input."""
    y = TestTensor(shape, None, dtype, device, mode="zeros")
    w = TestTensor(w_shape, None, dtype, device)

    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateRMSNormDescriptor(
            handle, ctypes.byref(desc), y.descriptor, x_tensor.descriptor, w.descriptor, EPS,
        )
    )
    for t in [y, w, x_tensor]:
        t.destroy_desc()

    ws_size = c_uint64(0)
    check_error(LIBINFINIOP.infiniopGetRMSNormWorkspaceSize(desc, ctypes.byref(ws_size)))
    ws = TestWorkspace(ws_size.value, device)

    def run():
        check_error(
            LIBINFINIOP.infiniopRMSNorm(
                desc, ws.data(), ws_size.value, y.data(), x_tensor.data(), w.data(), None,
            )
        )

    def cleanup():
        check_error(LIBINFINIOP.infiniopDestroyRMSNormDescriptor(desc))

    return y, run, cleanup


def _make_add_rms_norm(handle, device, shape, w_shape, dtype):
    """Return (run_fn, cleanup_fn) for fused AddRMSNorm kernel."""
    a = TestTensor(shape, None, dtype, device, scale=0.01)
    b = TestTensor(shape, None, dtype, device, scale=0.01)
    w = TestTensor(w_shape, None, dtype, device)
    y = TestTensor(shape, None, dtype, device, mode="zeros")
    residual_out = TestTensor(shape, None, dtype, device, mode="zeros")

    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateAddRMSNormDescriptor(
            handle, ctypes.byref(desc),
            y.descriptor, a.descriptor, b.descriptor, w.descriptor,
            EPS, residual_out.descriptor,
        )
    )
    for t in [a, b, w, y, residual_out]:
        t.destroy_desc()

    ws_size = c_uint64(0)
    check_error(LIBINFINIOP.infiniopGetAddRMSNormWorkspaceSize(desc, ctypes.byref(ws_size)))
    ws = TestWorkspace(ws_size.value, device)

    def run():
        check_error(
            LIBINFINIOP.infiniopAddRMSNorm(
                desc, ws.data(), ws_size.value,
                y.data(), a.data(), b.data(), w.data(), residual_out.data(), None,
            )
        )

    def cleanup():
        check_error(LIBINFINIOP.infiniopDestroyAddRMSNormDescriptor(desc))

    return run, cleanup


def _make_silu(handle, device, shape, dtype):
    """Return (input_tensor, output, run_fn, cleanup_fn) for Silu kernel."""
    inp = TestTensor(shape, None, dtype, device)
    out = TestTensor(shape, None, dtype, device, mode="zeros")

    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateSiluDescriptor(
            handle, ctypes.byref(desc), out.descriptor, inp.descriptor,
        )
    )
    # Only destroy inp's descriptor here; out's descriptor is kept alive
    # because _make_mul needs it to create the Mul operator descriptor.
    inp.destroy_desc()

    ws_size = c_uint64(0)
    check_error(LIBINFINIOP.infiniopGetSiluWorkspaceSize(desc, ctypes.byref(ws_size)))
    ws = TestWorkspace(ws_size.value, device)

    def run():
        check_error(
            LIBINFINIOP.infiniopSilu(
                desc, ws.data(), ws.size(), out.data(), inp.data(), None,
            )
        )

    def cleanup():
        check_error(LIBINFINIOP.infiniopDestroySiluDescriptor(desc))

    return inp, out, run, cleanup


def _make_mul(handle, device, shape, dtype, a_tensor):
    """Return (b, c, run_fn, cleanup_fn) for Mul kernel.
    a_tensor: a TestTensor whose data() is used as the first input."""
    b = TestTensor(shape, None, dtype, device)
    c = TestTensor(shape, None, dtype, device, mode="zeros")

    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateMulDescriptor(
            handle, ctypes.byref(desc), c.descriptor, a_tensor.descriptor, b.descriptor,
        )
    )
    for t in [a_tensor, b, c]:
        t.destroy_desc()

    ws_size = c_uint64(0)
    check_error(LIBINFINIOP.infiniopGetMulWorkspaceSize(desc, ctypes.byref(ws_size)))
    ws = TestWorkspace(ws_size.value, device)

    def run():
        check_error(
            LIBINFINIOP.infiniopMul(
                desc, ws.data(), ws_size.value, c.data(), a_tensor.data(), b.data(), None,
            )
        )

    def cleanup():
        check_error(LIBINFINIOP.infiniopDestroyMulDescriptor(desc))

    return b, c, run, cleanup


def _make_swiglu(handle, device, shape, dtype):
    """Return (run_fn, cleanup_fn) for fused SwiGLU kernel."""
    a = TestTensor(shape, None, dtype, device)
    b = TestTensor(shape, None, dtype, device)
    c = TestTensor(shape, None, dtype, device, mode="zeros")

    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateSwiGLUDescriptor(
            handle, ctypes.byref(desc), c.descriptor, a.descriptor, b.descriptor,
        )
    )
    for t in [a, b, c]:
        t.destroy_desc()

    ws_size = c_uint64(0)
    check_error(LIBINFINIOP.infiniopGetSwiGLUWorkspaceSize(desc, ctypes.byref(ws_size)))
    ws = TestWorkspace(ws_size.value, device)

    def run():
        check_error(
            LIBINFINIOP.infiniopSwiGLU(
                desc, ws.data(), ws_size.value, c.data(), a.data(), b.data(), None,
            )
        )

    def cleanup():
        check_error(LIBINFINIOP.infiniopDestroySwiGLUDescriptor(desc))

    return run, cleanup


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def _synchronize(device):
    """Device synchronize for accurate timing."""
    from libinfiniop.utils import synchronize_device
    torch_dev = torch_device_map[device]
    synchronize_device(torch_dev)


def _bench_kernel(func, device, num_prerun, num_iterations):
    """
    Warmup + timed execution. Returns average latency in milliseconds.

    1. Run `num_prerun` warmup iterations (not timed).
    2. Synchronize.
    3. Run `num_iterations` timed iterations.
    4. Synchronize and compute average.
    """
    # Warmup
    for _ in range(num_prerun):
        func()
    _synchronize(device)

    # Timed region
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        func()
    _synchronize(device)
    t1 = time.perf_counter()

    return (t1 - t0) / num_iterations * 1000  # ms


# ---------------------------------------------------------------------------
# Profile entry points
# ---------------------------------------------------------------------------


def profile_add_rms_norm(handle, device, shape, dtype, num_prerun, num_iterations):
    """Profile unfused add+rms_norm vs fused add_rms_norm for a single shape."""
    w_shape = (shape[-1],)
    label = str(list(shape))

    # ---- Unfused: Add then RMSNorm ----
    a, b, c_add, run_add, cleanup_add = _make_add(handle, device, shape, dtype)
    c_add_for_rms = TestTensor(shape, None, dtype, device, mode="zeros")
    y_rms, run_rms, cleanup_rms = _make_rms_norm(handle, device, shape, w_shape, dtype, c_add_for_rms)

    def run_unfused():
        run_add()
        run_rms()

    lat_unfused = _bench_kernel(run_unfused, device, num_prerun, num_iterations)

    # ---- Fused: AddRMSNorm ----
    run_fused, cleanup_fused = _make_add_rms_norm(handle, device, shape, w_shape, dtype)
    lat_fused = _bench_kernel(run_fused, device, num_prerun, num_iterations)

    # Store results
    _results["unfused"].setdefault("add+rms_norm", {})[label] = lat_unfused
    _results["fused"].setdefault("add+rms_norm", {})[label] = lat_fused

    cleanup_add()
    cleanup_rms()
    cleanup_fused()


def profile_silu_mul(handle, device, shape, dtype, num_prerun, num_iterations):
    """Profile unfused silu+mul vs fused swiglu for a single shape."""
    label = str(list(shape))

    # ---- Unfused: Silu then Mul ----
    inp_silu, out_silu, run_silu, cleanup_silu = _make_silu(handle, device, shape, dtype)
    b_mul, c_mul, run_mul, cleanup_mul = _make_mul(handle, device, shape, dtype, out_silu)

    def run_unfused():
        run_silu()
        run_mul()

    lat_unfused = _bench_kernel(run_unfused, device, num_prerun, num_iterations)

    # ---- Fused: SwiGLU ----
    run_fused, cleanup_fused = _make_swiglu(handle, device, shape, dtype)
    lat_fused = _bench_kernel(run_fused, device, num_prerun, num_iterations)

    # Store results
    _results["unfused"].setdefault("silu+mul", {})[label] = lat_unfused
    _results["fused"].setdefault("silu+mul", {})[label] = lat_fused

    cleanup_silu()
    cleanup_mul()
    cleanup_fused()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = get_args()
    num_prerun = args.num_prerun
    num_iterations = args.num_iterations

    for device in get_test_devices(args):
        LIBINFINIOP.infinirtSetDevice(device, ctypes.c_int(0))
        handle = create_handle()

        try:
            for shape in ADD_RMS_NORM_SHAPES:
                profile_add_rms_norm(handle, device, shape, DTYPE, num_prerun, num_iterations)

            for shape in SILU_MUL_SHAPES:
                profile_silu_mul(handle, device, shape, DTYPE, num_prerun, num_iterations)
        finally:
            destroy_handle(handle)

    # Round and output JSON (same format as test/infinicore/profile.py)
    output = {}
    for category, patterns in _results.items():
        output[category] = {}
        for pattern_name, shape_dict in patterns.items():
            output[category][pattern_name] = {
                k: round(v, 4) for k, v in shape_dict.items()
            }
    print("\n" + json.dumps(output, indent=2, ensure_ascii=False))