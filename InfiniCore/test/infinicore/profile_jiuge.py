"""
Pytest benchmark for single and fused operators.

Measures pure kernel execution latency with compilation and warmup
separated from the timed region.

Run:
    pytest profile.py -v -s
    pytest profile.py -v -s -k "fused"
    pytest profile.py -v -s -k "single"
"""

import json
import time
import torch
import torch.nn.functional as TorchF
import pytest
from typing import Dict, Any

from infinicore.fusion.fusion_scheduler import FusionScheduler
from infinicore.fusion.fusion_config import FusionConfig
from infinicore.fusion.subgraph import SubGraph, OpNode

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WARMUP = 20
RUNS = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ---------------------------------------------------------------------------
# Model-based dimension configs (from JiugeMetaFromLlama)
#
#   d  = hidden_size          -- used by add, rms_norm, gelu, add+rms_norm
#   di = intermediate_size    -- used by silu, mul, silu+mul
#
# Three tiers map to real LLM architectures:
#   TinyLlama-1.1B   (d=2048,  di=5632)
#   LLaMA-2-7B       (d=4096,  di=11008)
#   LLaMA-2-13B      (d=5120,  di=13824)
#
# Token count = batch * seq_len (flattened to 2-D before elementwise ops).
# Fixed at 2048 tokens (~batch=4, seq=512) to isolate dimension impact.
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "TinyLlama-1.1B": {"d": 2048, "di": 5632},
    "LLaMA-2-7B":     {"d": 4096, "di": 11008},
    "LLaMA-2-13B":    {"d": 5120, "di": 13824},
}

TOKENS = 2048  # fixed token count across all tiers

MODEL_NAMES = list(MODEL_CONFIGS.keys())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_results: Dict[str, Any] = {"single": {}, "fused": {}}


def _t(*shape):
    return torch.randn(*shape, device=DEVICE, dtype=DTYPE)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench_kernel(func, args, warmup=WARMUP, runs=RUNS):
    """Warmup, sync, then measure *only* kernel execution time."""
    for _ in range(warmup):
        func(*args)
    _sync()

    t0 = time.perf_counter()
    for _ in range(runs):
        func(*args)
    _sync()
    t1 = time.perf_counter()
    return (t1 - t0) / runs * 1000  # ms


def _compile_fused(scheduler, graph, inputs):
    """Trigger compilation + warmup so it is excluded from the timed run."""
    for _ in range(WARMUP):
        scheduler.dispatch(graph, inputs)
    _sync()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def schedulers():
    on = FusionScheduler(FusionConfig(enable_fusion=True))
    off = FusionScheduler(FusionConfig(enable_fusion=False))
    return on, off


@pytest.fixture(scope="session", autouse=True)
def print_results():
    """Print aggregated JSON after all tests finish."""
    yield
    print("\n\n" + "=" * 70)
    print("Aggregated Results")
    print("=" * 70)
    print(json.dumps(_results, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Single-operator tests
# ---------------------------------------------------------------------------

class TestSingleOps:

    @pytest.mark.parametrize("model", MODEL_NAMES)
    def test_silu(self, model):
        di = MODEL_CONFIGS[model]["di"]
        shape = (TOKENS, di)
        x = _t(*shape)
        lat = _bench_kernel(TorchF.silu, (x,))
        _results["single"].setdefault("silu", {})[f"{model} {list(shape)}"] = f"{lat:.4f} ms"
        assert lat > 0

    @pytest.mark.parametrize("model", MODEL_NAMES)
    def test_gelu(self, model):
        d = MODEL_CONFIGS[model]["d"]
        shape = (TOKENS, d)
        x = _t(*shape)
        lat = _bench_kernel(TorchF.gelu, (x,))
        _results["single"].setdefault("gelu", {})[f"{model} {list(shape)}"] = f"{lat:.4f} ms"
        assert lat > 0

    @pytest.mark.parametrize("model", MODEL_NAMES)
    def test_add(self, model):
        d = MODEL_CONFIGS[model]["d"]
        shape = (TOKENS, d)
        a, b = _t(*shape), _t(*shape)
        lat = _bench_kernel(torch.add, (a, b))
        _results["single"].setdefault("add", {})[f"{model} {list(shape)}"] = f"{lat:.4f} ms"
        assert lat > 0

    @pytest.mark.parametrize("model", MODEL_NAMES)
    def test_mul(self, model):
        di = MODEL_CONFIGS[model]["di"]
        shape = (TOKENS, di)
        a, b = _t(*shape), _t(*shape)
        lat = _bench_kernel(torch.mul, (a, b))
        _results["single"].setdefault("mul", {})[f"{model} {list(shape)}"] = f"{lat:.4f} ms"
        assert lat > 0

    @pytest.mark.parametrize("model", MODEL_NAMES)
    def test_rms_norm(self, model):
        d = MODEL_CONFIGS[model]["d"]
        shape = (TOKENS, d)
        x = _t(*shape)
        w = _t(d)
        ns = (d,)

        def fn(x, w):
            return TorchF.rms_norm(x, ns, w, eps=1e-5)

        lat = _bench_kernel(fn, (x, w))
        _results["single"].setdefault("rms_norm", {})[f"{model} {list(shape)}"] = f"{lat:.4f} ms"
        assert lat > 0


# ---------------------------------------------------------------------------
# Fused-operator tests
# ---------------------------------------------------------------------------

class TestFusedOps:

    @pytest.mark.parametrize("model", MODEL_NAMES)
    def test_add_rms_norm(self, model, schedulers):
        sched_on, sched_off = schedulers
        d = MODEL_CONFIGS[model]["d"]
        shape = (TOKENS, d)
        graph = SubGraph(
            nodes=(
                OpNode("add", inputs=("x", "residual"), outputs=("sum",)),
                OpNode("rms_norm", inputs=("sum", "weight"), outputs=("output",)),
            ),
            input_names=("x", "residual", "weight"),
            output_names=("output",),
        )
        inp = {"x": _t(*shape), "residual": _t(*shape), "weight": _t(d)}

        _compile_fused(sched_on, graph, inp)
        _compile_fused(sched_off, graph, inp)

        lat_off = _bench_kernel(sched_off.dispatch, (graph, inp))
        lat_on = _bench_kernel(sched_on.dispatch, (graph, inp))

        label = f"{model} {list(shape)}"
        _results["fused"].setdefault("add+rms_norm", {})[label] = {
            "unfused_ms": f"{lat_off:.4f}",
            "fused_ms": f"{lat_on:.4f}",
            "speedup": f"{lat_off / lat_on:.2f}x",
        }
        assert lat_off > 0 and lat_on > 0

    @pytest.mark.parametrize("model", MODEL_NAMES)
    def test_silu_mul(self, model, schedulers):
        sched_on, sched_off = schedulers
        di = MODEL_CONFIGS[model]["di"]
        shape = (TOKENS, di)
        graph = SubGraph(
            nodes=(
                OpNode("silu", inputs=("gate",), outputs=("gate_activated",)),
                OpNode("mul", inputs=("gate_activated", "up"), outputs=("output",)),
            ),
            input_names=("gate", "up"),
            output_names=("output",),
        )
        inp = {"gate": _t(*shape), "up": _t(*shape)}

        _compile_fused(sched_on, graph, inp)
        _compile_fused(sched_off, graph, inp)

        lat_off = _bench_kernel(sched_off.dispatch, (graph, inp))
        lat_on = _bench_kernel(sched_on.dispatch, (graph, inp))

        label = f"{model} {list(shape)}"
        _results["fused"].setdefault("silu+mul", {})[label] = {
            "unfused_ms": f"{lat_off:.4f}",
            "fused_ms": f"{lat_on:.4f}",
            "speedup": f"{lat_off / lat_on:.2f}x",
        }
        assert lat_off > 0 and lat_on > 0

    @pytest.mark.parametrize("model", MODEL_NAMES)
    def test_gelu_fused(self, model, schedulers):
        sched_on, sched_off = schedulers
        d = MODEL_CONFIGS[model]["d"]
        shape = (TOKENS, d)
        graph = SubGraph(
            nodes=(OpNode("gelu", inputs=("x",), outputs=("output",)),),
            input_names=("x",),
            output_names=("output",),
        )
        inp = {"x": _t(*shape)}

        _compile_fused(sched_on, graph, inp)
        _compile_fused(sched_off, graph, inp)

        lat_off = _bench_kernel(sched_off.dispatch, (graph, inp))
        lat_on = _bench_kernel(sched_on.dispatch, (graph, inp))

        label = f"{model} {list(shape)}"
        _results["fused"].setdefault("gelu", {})[label] = {
            "unfused_ms": f"{lat_off:.4f}",
            "fused_ms": f"{lat_on:.4f}",
            "speedup": f"{lat_off / lat_on:.2f}x",
        }
        assert lat_off > 0 and lat_on > 0
