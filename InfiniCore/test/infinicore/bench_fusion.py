"""
Pytest benchmark for Operator Fusion in InfiniCore.

Compares Latency and Throughput of fused vs. non-fused execution paths.

Run with: 
    pytest bench_fusion.py -v -s
    pytest bench_fusion.py -v -s --batch_size=64 --hidden_dim=2048

Note: Custom command line options (--batch_size, etc.) must be defined in conftest.py
"""

import time
import torch
import numpy as np
import pytest
from typing import Dict, Any, Tuple

from infinicore.fusion.fusion_scheduler import FusionScheduler
from infinicore.fusion.fusion_config import FusionConfig
from infinicore.fusion.patterns.llm_patterns import (
    create_swiglu_pattern,
    create_add_rms_norm_pattern,
    create_gelu_pattern,
    LLM_FUSION_PATTERNS,
)
from infinicore.fusion.subgraph import SubGraph


# Default benchmark parameters
DEFAULT_BATCH_SIZE = 32
DEFAULT_HIDDEN_DIM = 4096
DEFAULT_WARMUP = 50  # Increased for Triton autotuning
DEFAULT_RUNS = 200   # More runs for stable measurements


@pytest.fixture(scope="module")
def benchmark_config(request):
    """Fixture to provide benchmark configuration from command line options or defaults."""
    return {
        "batch_size": request.config.getoption("--batch_size", default=DEFAULT_BATCH_SIZE),
        "hidden_dim": request.config.getoption("--hidden_dim", default=DEFAULT_HIDDEN_DIM),
        "warmup": request.config.getoption("--warmup", default=DEFAULT_WARMUP),
        "runs": request.config.getoption("--runs", default=DEFAULT_RUNS),
    }


@pytest.fixture(scope="module")
def device_info():
    """Fixture to provide device and dtype information."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    return device, dtype


def create_inputs_for_pattern(
    pattern: SubGraph,
    batch_size: int,
    hidden_dim: int,
    device: str,
    dtype: torch.dtype
) -> Dict[str, torch.Tensor]:
    """
    Dynamically create inputs for any pattern based on its input_names.
    
    Args:
        pattern: The SubGraph pattern to create inputs for
        batch_size: Batch size for tensors
        hidden_dim: Hidden dimension for tensors
        device: Device to place tensors on
        dtype: Data type for tensors
        
    Returns:
        Dictionary mapping input names to tensors
    """
    inputs = {}
    for input_name in pattern.input_names:
        # Different input types may need different shapes
        if "weight" in input_name.lower():
            # Weight tensors are typically 1D
            inputs[input_name] = torch.randn(hidden_dim, device=device, dtype=dtype)
        else:
            # Activation tensors are typically 2D (batch_size, hidden_dim)
            inputs[input_name] = torch.randn(batch_size, hidden_dim, device=device, dtype=dtype)
    
    return inputs


def run_benchmark(
    name: str,
    func,
    args: tuple,
    warmup: int = 10,
    runs: int = 100,
    separate_compilation: bool = False
) -> Tuple[float, float]:
    """
    Benchmark a function and return compilation + execution latency.
    
    Args:
        name: Name of the benchmark
        func: Function to benchmark
        args: Arguments to pass to the function
        warmup: Number of warmup iterations
        runs: Number of benchmark runs
        separate_compilation: If True, measure compilation separately
        
    Returns:
        Tuple of (compilation_time_ms, avg_execution_latency_ms)
    """
    compilation_time = 0.0
    
    # Measure compilation time on first call
    if separate_compilation:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        compile_start = time.perf_counter()
        func(*args)  # First call triggers JIT compilation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        compilation_time = (time.perf_counter() - compile_start) * 1000
        print(f"[{name}] Compilation Time: {compilation_time:.4f} ms")
        warmup = max(warmup - 1, 0)  # Already did one call
    
    # Warmup (for GPU kernel autotuning and cache warming)
    for _ in range(warmup):
        func(*args)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    # Actual benchmark
    start_time = time.perf_counter()
    for _ in range(runs):
        func(*args)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    avg_latency = (end_time - start_time) / runs * 1000  # ms
    print(f"[{name}] Avg Execution Latency: {avg_latency:.4f} ms (after warmup)")
    
    return compilation_time, avg_latency


class TestFusionBenchmark:
    """Test class for fusion benchmarking."""
    
    @pytest.mark.parametrize("pattern_factory,pattern_name", [
        (create_swiglu_pattern, "SwiGLU"),
        (create_add_rms_norm_pattern, "Add+RMSNorm"),
        (create_gelu_pattern, "GELU"),
    ])
    def test_fusion_benchmark(self, benchmark_config, device_info, pattern_factory, pattern_name):
        """
        Main benchmark test comparing fused vs non-fused execution for all LLM patterns.
        
        This test separates compilation time from execution time and uses proper warmup.
        """
        device, dtype = device_info
        batch_size = benchmark_config["batch_size"]
        hidden_dim = benchmark_config["hidden_dim"]
        warmup = benchmark_config["warmup"]
        runs = benchmark_config["runs"]
        
        print(f"\n{'='*70}")
        print(f"Benchmarking Fusion Performance - {pattern_name}")
        print(f"Batch Size: {batch_size}, Hidden Dim: {hidden_dim}, Device: {device}")
        print(f"Warmup: {warmup}, Runs: {runs}")
        print(f"{'='*70}")
        
        # Create pattern and corresponding inputs
        graph = pattern_factory()
        test_inputs = create_inputs_for_pattern(graph, batch_size, hidden_dim, device, dtype)
        
        print(f"Pattern inputs: {list(test_inputs.keys())}")
        print(f"Pattern outputs: {graph.output_names}")
        
        # 1. Non-fused Scheduler (baseline)
        config_off = FusionConfig(enable_fusion=False)
        scheduler_off = FusionScheduler(config_off)
        
        # 2. Fused Scheduler
        config_on = FusionConfig(enable_fusion=True)
        scheduler_on = FusionScheduler(config_on)
        
        # Benchmark with separate compilation measurement
        print(f"\n--- Non-fused (Fallback) Execution ---")
        compile_off, lat_off = run_benchmark(
            f"{pattern_name} - Standard (Fallback)", 
            scheduler_off.dispatch, 
            (graph, test_inputs), 
            warmup, 
            runs,
            separate_compilation=False  # Fallback has no compilation
        )
        
        print(f"\n--- Fused (Triton) Execution ---")
        compile_on, lat_on = run_benchmark(
            f"{pattern_name} - Fused (Triton)", 
            scheduler_on.dispatch, 
            (graph, test_inputs), 
            warmup, 
            runs,
            separate_compilation=True  # Measure JIT compilation
        )
        
        improvement = (lat_off - lat_on) / lat_off * 100
        speedup = lat_off / lat_on
        
        print(f"\n{'='*70}")
        print(f"Results for {pattern_name}:")
        print(f"  Non-fused Execution:    {lat_off:.4f} ms")
        print(f"  Fused Compilation:      {compile_on:.4f} ms (one-time cost)")
        print(f"  Fused Execution:        {lat_on:.4f} ms")
        print(f"  Speedup:                {speedup:.2f}x")
        print(f"  Improvement:            {improvement:.2f}%")
        if compile_on > 0:
            print(f"  Amortization Runs:      {int(compile_on / max(lat_off - lat_on, 0.001))}")
        print(f"{'='*70}\n")
        
        # Assert that both executions completed successfully
        assert lat_off > 0, f"{pattern_name}: Non-fused execution failed"
        assert lat_on > 0, f"{pattern_name}: Fused execution failed"
        
    @pytest.mark.parametrize("pattern_factory,pattern_name", [
        (create_swiglu_pattern, "SwiGLU"),
        (create_add_rms_norm_pattern, "Add+RMSNorm"),
        (create_gelu_pattern, "GELU"),
    ])
    @pytest.mark.parametrize("batch_size,hidden_dim", [
        (16, 2048),
        (32, 4096),
        (64, 4096),
    ])
    def test_fusion_various_sizes(self, batch_size, hidden_dim, device_info, pattern_factory, pattern_name):
        """
        Test fusion performance with various input sizes for all LLM patterns.
        
        This parametrized test runs benchmarks for different configurations and patterns.
        """
        device, dtype = device_info
        warmup = 30  # Reduced for parametrized tests
        runs = 100
        
        print(f"\nTesting {pattern_name} with Batch Size: {batch_size}, Hidden Dim: {hidden_dim}")
        
        # Create pattern and inputs dynamically
        graph = pattern_factory()
        inputs = create_inputs_for_pattern(graph, batch_size, hidden_dim, device, dtype)
        
        # Non-fused
        config_off = FusionConfig(enable_fusion=False)
        scheduler_off = FusionScheduler(config_off)
        
        # Fused
        config_on = FusionConfig(enable_fusion=True)
        scheduler_on = FusionScheduler(config_on)
        
        # Benchmark
        _, lat_off = run_benchmark(
            f"{pattern_name} Non-fused (B={batch_size}, H={hidden_dim})", 
            scheduler_off.dispatch, 
            (graph, inputs), 
            warmup, 
            runs,
            separate_compilation=False
        )
        compile_on, lat_on = run_benchmark(
            f"{pattern_name} Fused (B={batch_size}, H={hidden_dim})", 
            scheduler_on.dispatch, 
            (graph, inputs), 
            warmup, 
            runs,
            separate_compilation=True
        )
        
        speedup = lat_off / lat_on
        print(f"{pattern_name} Speedup: {speedup:.2f}x (Compile: {compile_on:.2f}ms, Exec: {lat_on:.4f}ms)")
        
        assert lat_off > 0 and lat_on > 0, f"{pattern_name} execution failed"
    
    def test_all_predefined_patterns(self, benchmark_config, device_info):
        """
        Test all predefined LLM fusion patterns from LLM_FUSION_PATTERNS list.
        
        This test iterates through all patterns defined in llm_patterns.py
        and benchmarks each one with proper compilation time tracking.
        """
        device, dtype = device_info
        batch_size = benchmark_config["batch_size"]
        hidden_dim = benchmark_config["hidden_dim"]
        warmup = benchmark_config["warmup"] // 2  # Reduce for bulk testing
        runs = benchmark_config["runs"] // 2
        
        print(f"\n{'='*70}")
        print(f"Testing All Predefined LLM Fusion Patterns")
        print(f"Total patterns: {len(LLM_FUSION_PATTERNS)}")
        print(f"Batch Size: {batch_size}, Hidden Dim: {hidden_dim}, Device: {device}")
        print(f"{'='*70}\n")
        
        results = []
        
        for idx, pattern in enumerate(LLM_FUSION_PATTERNS, 1):
            pattern_name = f"Pattern-{idx}"
            print(f"\n[{idx}/{len(LLM_FUSION_PATTERNS)}] Testing {pattern_name}")
            print(f"  Inputs: {pattern.input_names}")
            print(f"  Outputs: {pattern.output_names}")
            
            # Create inputs
            inputs = create_inputs_for_pattern(pattern, batch_size, hidden_dim, device, dtype)
            
            # Non-fused
            config_off = FusionConfig(enable_fusion=False)
            scheduler_off = FusionScheduler(config_off)
            
            # Fused
            config_on = FusionConfig(enable_fusion=True)
            scheduler_on = FusionScheduler(config_on)
            
            # Benchmark
            _, lat_off = run_benchmark(
                f"{pattern_name} Non-fused",
                scheduler_off.dispatch,
                (pattern, inputs),
                warmup,
                runs,
                separate_compilation=False
            )
            compile_on, lat_on = run_benchmark(
                f"{pattern_name} Fused",
                scheduler_on.dispatch,
                (pattern, inputs),
                warmup,
                runs,
                separate_compilation=True
            )
            
            speedup = lat_off / lat_on
            improvement = (lat_off - lat_on) / lat_off * 100
            
            results.append({
                "pattern": pattern_name,
                "inputs": pattern.input_names,
                "non_fused_ms": lat_off,
                "compile_ms": compile_on,
                "fused_ms": lat_on,
                "speedup": speedup,
                "improvement_pct": improvement,
            })
            
            print(f"  Speedup: {speedup:.2f}x ({improvement:.2f}% improvement)")
            print(f"  Compilation: {compile_on:.2f}ms")
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"Summary of All Pattern Benchmarks")
        print(f"{'='*70}")
        print(f"{'Pattern':<15} | {'Compile (ms)':<12} | {'Speedup':<8} | {'Improvement':<12}")
        print(f"{'-'*70}")
        for result in results:
            print(f"{result['pattern']:<15} | "
                  f"{result['compile_ms']:>12.2f} | "
                  f"{result['speedup']:>8.2f}x | "
                  f"{result['improvement_pct']:>11.2f}%")
        print(f"{'='*70}\n")
        
        # Assert all patterns executed successfully
        for result in results:
            assert result["non_fused_ms"] > 0 and result["fused_ms"] > 0, \
                f"{result['pattern']} execution failed"

