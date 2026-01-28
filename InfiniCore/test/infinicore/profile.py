"""
Pytest benchmark for fused operators.

Measures pure kernel execution latency with compilation and warmup
separated from the timed region. Outputs clean JSON to stdout.

Run:
    pytest profile.py -v -s
"""

import json
import time
import torch
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

# 3D input shapes: (batch_size, seqlen, hidden_dim)
ADD_RMS_NORM_SHAPES = [
    (1, 512, 4096),
    (4, 1024, 4096),
    (8, 2048, 4096),
    (16, 1024, 4096),
    (32, 512, 4096),
]

SILU_MUL_SHAPES = [
    (1, 512, 11008),
    (4, 1024, 11008),
    (8, 2048, 11008),
    (16, 1024, 11008),
    (32, 512, 11008),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_results: Dict[str, Any] = {"unfused": {}, "fused": {}}


def _t(*shape):
    return torch.randn(*shape, device=DEVICE, dtype=DTYPE)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench_kernel(func, args, warmup=WARMUP, runs=RUNS):
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
    yield
    # Round values for clean output
    output = {}
    for category, patterns in _results.items():
        output[category] = {}
        for pattern_name, shape_dict in patterns.items():
            output[category][pattern_name] = {
                k: round(v, 4) for k, v in shape_dict.items()
            }
    print("\n" + json.dumps(output, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Fused-operator tests
# ---------------------------------------------------------------------------

class TestFusedOps:

    @pytest.mark.parametrize("shape", ADD_RMS_NORM_SHAPES,
                             ids=[str(list(s)) for s in ADD_RMS_NORM_SHAPES])
    def test_add_rms_norm(self, shape, schedulers):
        sched_on, sched_off = schedulers
        graph = SubGraph(
            nodes=(
                OpNode("add", inputs=("x", "residual"), outputs=("sum",)),
                OpNode("rms_norm", inputs=("sum", "weight"), outputs=("output",)),
            ),
            input_names=("x", "residual", "weight"),
            output_names=("output",),
        )
        inp = {"x": _t(*shape), "residual": _t(*shape), "weight": _t(shape[-1])}

        _compile_fused(sched_on, graph, inp)
        _compile_fused(sched_off, graph, inp)

        lat_off = _bench_kernel(sched_off.dispatch, (graph, inp))
        lat_on = _bench_kernel(sched_on.dispatch, (graph, inp))

        label = str(list(shape))
        _results["unfused"].setdefault("add+rms_norm", {})[label] = lat_off
        _results["fused"].setdefault("add+rms_norm", {})[label] = lat_on
        assert lat_off > 0 and lat_on > 0

    @pytest.mark.parametrize("shape", SILU_MUL_SHAPES,
                             ids=[str(list(s)) for s in SILU_MUL_SHAPES])
    def test_silu_mul(self, shape, schedulers):
        sched_on, sched_off = schedulers
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

        label = str(list(shape))
        _results["unfused"].setdefault("silu+mul", {})[label] = lat_off
        _results["fused"].setdefault("silu+mul", {})[label] = lat_on
        assert lat_off > 0 and lat_on > 0
