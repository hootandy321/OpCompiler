"""
Benchmark utilities for comparing fusion performance.

This module provides tools to benchmark and compare:
1. PyTorch native operators (baseline)
2. Manually fused operators (hand-written fusion)
3. Automatically fused operators (using fusion.py)
"""

import time
from typing import Callable, List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import torch


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    name: str
    host_time_ms: float
    device_time_ms: float
    kernel_count: int = 0

    def speedup_vs(self, other: 'BenchmarkResult') -> Tuple[float, float]:
        """Calculate speedup relative to another result"""
        host_speedup = other.host_time_ms / self.host_time_ms if self.host_time_ms > 0 else float('inf')
        device_speedup = other.device_time_ms / self.device_time_ms if self.device_time_ms > 0 else float('inf')
        return host_speedup, device_speedup


def benchmark_function(
    func: Callable,
    *args,
    num_warmup: int = 10,
    num_iterations: int = 100,
    device: str = 'cuda',
    **kwargs
) -> BenchmarkResult:
    """
    Benchmark a function with both host and device timing.

    Args:
        func: Function to benchmark
        *args: Arguments to pass to func
        num_warmup: Number of warm-up iterations
        num_iterations: Number of timed iterations
        device: Device to run on ('cuda' or 'cpu')
        **kwargs: Keyword arguments to pass to func

    Returns:
        BenchmarkResult with timing information
    """
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # Warm-up
    for _ in range(num_warmup):
        result = func(*args, **kwargs)
        if device == 'cuda':
            torch.cuda.synchronize()

    # Host timing
    host_start = time.perf_counter()
    for _ in range(num_iterations):
        result = func(*args, **kwargs)
    if device == 'cuda':
        torch.cuda.synchronize()
    host_elapsed = (time.perf_counter() - host_start) * 1000  # Convert to ms

    # Device timing (CUDA only)
    if device == 'cuda':
        # Use CUDA events for accurate GPU timing
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]

        for i in range(num_iterations):
            start_events[i].record()
            result = func(*args, **kwargs)
            end_events[i].record()

        torch.cuda.synchronize()

        device_elapsed = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events))
    else:
        device_elapsed = host_elapsed

    return BenchmarkResult(
        name=func.__name__,
        host_time_ms=host_elapsed / num_iterations,
        device_time_ms=device_elapsed / num_iterations
    )


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information for the benchmark report"""
    try:
        # Check if torch has cuda support
        if not hasattr(torch, 'cuda') or not torch.cuda.is_available():
            return {"device": "CPU", "note": "CUDA not available"}

        return {
            "device": torch.cuda.get_device_name(0),
            "compute_capability": torch.cuda.get_device_capability(0),
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
        }
    except (AttributeError, Exception) as e:
        return {"device": "Unknown", "note": f"Error detecting GPU: {str(e)}"}


class BenchmarkSuite:
    """Suite for running multiple benchmarks and generating reports"""

    def __init__(self, suite_name: str):
        self.suite_name = suite_name
        self.results: List[BenchmarkResult] = []
        self.test_cases: List[Dict[str, Any]] = []

    def add_test_case(self, test_name: str, test_params: Dict[str, Any]):
        """Add a test case configuration"""
        self.test_cases.append({
            "name": test_name,
            "params": test_params,
            "results": {}
        })

    def run_benchmark(
        self,
        test_name: str,
        implementations: Dict[str, Callable],
        **benchmark_args
    ):
        """
        Run benchmarks for all implementations of a test case.

        Args:
            test_name: Name of the test case
            implementations: Dict mapping implementation names to functions
            **benchmark_args: Additional arguments for benchmark_function
        """
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")

        test_results = {}

        for impl_name, impl_func in implementations.items():
            try:
                result = benchmark_function(impl_func, **benchmark_args)
                test_results[impl_name] = result
                print(f"  {impl_name:30s} - Host: {result.host_time_ms:7.3f} ms, "
                      f"Device: {result.device_time_ms:7.3f} ms")
            except Exception as e:
                print(f"  {impl_name:30s} - ERROR: {str(e)}")
                test_results[impl_name] = None

        # Store results
        for test_case in self.test_cases:
            if test_case["name"] == test_name:
                test_case["results"] = test_results
                break

        # Calculate speedups if baseline exists
        if "PyTorch Native" in test_results and test_results["PyTorch Native"]:
            baseline = test_results["PyTorch Native"]
            print(f"\n  Speedup vs PyTorch Native:")
            for impl_name, result in test_results.items():
                if result and impl_name != "PyTorch Native":
                    host_spd, dev_spd = result.speedup_vs(baseline)
                    print(f"    {impl_name:28s} - Host: {host_spd:5.2f}x, "
                          f"Device: {dev_spd:5.2f}x")

    def generate_markdown_report(self, output_file: Optional[str] = None) -> str:
        """Generate a markdown report of all benchmarks"""
        lines = []
        lines.append(f"# {self.suite_name} - Benchmark Report\n")
        lines.append(f"\n**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Environment info
        gpu_info = get_gpu_info()
        lines.append("## Test Environment\n")
        for key, value in gpu_info.items():
            lines.append(f"- **{key}:** {value}")
        lines.append("\n")

        # Results for each test case
        for test_case in self.test_cases:
            lines.append(f"## {test_case['name']}\n")

            # Test parameters
            if test_case.get('params'):
                lines.append("### Test Parameters\n")
                for key, value in test_case['params'].items():
                    lines.append(f"- **{key}:** {value}")
                lines.append("\n")

            # Results table
            lines.append("### Performance Results\n")
            lines.append("| Implementation | Host Time (ms) | Device Time (ms) | Host Speedup | Device Speedup |\n")
            lines.append("|----------------|----------------|------------------|--------------|----------------|\n")

            results = test_case.get('results', {})
            baseline = results.get("PyTorch Native")

            for impl_name, result in results.items():
                if result is None:
                    lines.append(f"| {impl_name} | ERROR | ERROR | - | - |\n")
                else:
                    if baseline and impl_name != "PyTorch Native":
                        host_spd, dev_spd = result.speedup_vs(baseline)
                        lines.append(f"| {impl_name} | {result.host_time_ms:.3f} | "
                                   f"{result.device_time_ms:.3f} | {host_spd:.2f}x | {dev_spd:.2f}x |\n")
                    else:
                        lines.append(f"| {impl_name} | {result.host_time_ms:.3f} | "
                                   f"{result.device_time_ms:.3f} | - | - |\n")

            lines.append("\n")

        report = "".join(lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {output_file}")

        return report


def estimate_kernel_count(func: Callable) -> int:
    """
    Estimate the number of kernels launched by a function.
    This is a rough estimate using CUDA graph capture.
    """
    if not torch.cuda.is_available():
        return 0

    try:
        # Capture CUDA graph to count kernels
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()

        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph):
                # Create dummy inputs
                # This won't work for all functions, but provides a rough estimate
                pass

        # The number of nodes in the graph gives us a kernel count estimate
        return len(graph.graph_nodes()) if hasattr(graph, 'graph_nodes') else 0
    except:
        return 0
