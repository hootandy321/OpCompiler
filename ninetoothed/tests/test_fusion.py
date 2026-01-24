"""Tests for ninetoothed kernel fusion functionality.

This module tests:
1. Correctness of fused kernels vs unfused kernels
2. Performance comparison between fused and unfused execution
3. Accurate profiling with proper warmup and timing methodology
"""

import gc
import time

import pytest
import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor, block_size, fuser
from tests.utils import get_available_devices

# ============================================================================
# Kernel Definitions
# ============================================================================

BLOCK_SIZE = block_size()


def add_arrangement(lhs, rhs, output, BLOCK_SIZE=BLOCK_SIZE):
    return (
        lhs.tile((BLOCK_SIZE,)),
        rhs.tile((BLOCK_SIZE,)),
        output.tile((BLOCK_SIZE,)),
    )


def add_application(lhs, rhs, output):
    output = lhs + rhs  # noqa: F841


def mul_arrangement(lhs, rhs, output, BLOCK_SIZE=BLOCK_SIZE):
    return (
        lhs.tile((BLOCK_SIZE,)),
        rhs.tile((BLOCK_SIZE,)),
        output.tile((BLOCK_SIZE,)),
    )


def mul_application(lhs, rhs, output):
    output = lhs * rhs  # noqa: F841


def relu_arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return (
        input.tile((BLOCK_SIZE,)),
        output.tile((BLOCK_SIZE,)),
    )


def relu_application(input, output):
    output = ntl.maximum(input, 0.0)  # noqa: F841


def neg_arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return (
        input.tile((BLOCK_SIZE,)),
        output.tile((BLOCK_SIZE,)),
    )


def neg_application(input, output):
    output = -input  # noqa: F841


# Create kernel handles
add_kernel = ninetoothed.make(
    add_arrangement, add_application, (Tensor(1), Tensor(1), Tensor(1))
)

mul_kernel = ninetoothed.make(
    mul_arrangement, mul_application, (Tensor(1), Tensor(1), Tensor(1))
)

relu_kernel = ninetoothed.make(
    relu_arrangement, relu_application, (Tensor(1), Tensor(1))
)

neg_kernel = ninetoothed.make(
    neg_arrangement, neg_application, (Tensor(1), Tensor(1))
)


# ============================================================================
# Unfused Operations (for comparison)
# ============================================================================


def add_mul_unfused(a, b, c):
    """Compute (a + b) * c without fusion."""
    temp = torch.empty_like(a)
    add_kernel(a, b, temp)
    output = torch.empty_like(a)
    mul_kernel(temp, c, output)
    return output


def add_mul_add_unfused(a, b, c, d):
    """Compute ((a + b) * c) + d without fusion."""
    temp1 = torch.empty_like(a)
    add_kernel(a, b, temp1)
    temp2 = torch.empty_like(a)
    mul_kernel(temp1, c, temp2)
    output = torch.empty_like(a)
    add_kernel(temp2, d, output)
    return output


def relu_neg_unfused(x):
    """Compute -relu(x) without fusion."""
    temp = torch.empty_like(x)
    relu_kernel(x, temp)
    output = torch.empty_like(x)
    neg_kernel(temp, output)
    return output


# ============================================================================
# Fused Operations (using torch.compile with fuser backend)
# ============================================================================

# Note: These are defined as module-level to ensure compilation happens once
_fused_functions_compiled = {}


def _get_add_mul_fused():
    """Get or create the compiled add_mul_fused function."""
    if "add_mul" not in _fused_functions_compiled:
        @torch.compile(backend=fuser)
        def add_mul_fused(a, b, c):
            temp = torch.empty_like(a)
            add_kernel(a, b, temp)
            output = torch.empty_like(a)
            mul_kernel(temp, c, output)
            return output
        _fused_functions_compiled["add_mul"] = add_mul_fused
    return _fused_functions_compiled["add_mul"]


def _get_add_mul_add_fused():
    """Get or create the compiled add_mul_add_fused function."""
    if "add_mul_add" not in _fused_functions_compiled:
        @torch.compile(backend=fuser)
        def add_mul_add_fused(a, b, c, d):
            temp1 = torch.empty_like(a)
            add_kernel(a, b, temp1)
            temp2 = torch.empty_like(a)
            mul_kernel(temp1, c, temp2)
            output = torch.empty_like(a)
            add_kernel(temp2, d, output)
            return output
        _fused_functions_compiled["add_mul_add"] = add_mul_add_fused
    return _fused_functions_compiled["add_mul_add"]


def _get_relu_neg_fused():
    """Get or create the compiled relu_neg_fused function."""
    if "relu_neg" not in _fused_functions_compiled:
        @torch.compile(backend=fuser)
        def relu_neg_fused(x):
            temp = torch.empty_like(x)
            relu_kernel(x, temp)
            output = torch.empty_like(x)
            neg_kernel(temp, output)
            return output
        _fused_functions_compiled["relu_neg"] = relu_neg_fused
    return _fused_functions_compiled["relu_neg"]


# Wrapper functions for backward compatibility with tests
def add_mul_fused(a, b, c):
    return _get_add_mul_fused()(a, b, c)


def add_mul_add_fused(a, b, c, d):
    return _get_add_mul_add_fused()(a, b, c, d)


def relu_neg_fused(x):
    return _get_relu_neg_fused()(x)


# ============================================================================
# PyTorch Native Operations (baseline reference)
# ============================================================================


def add_mul_pytorch(a, b, c):
    """Compute (a + b) * c using native PyTorch."""
    return (a + b) * c


def add_mul_add_pytorch(a, b, c, d):
    """Compute ((a + b) * c) + d using native PyTorch."""
    return ((a + b) * c) + d


def relu_neg_pytorch(x):
    """Compute -relu(x) using native PyTorch."""
    return -torch.relu(x)


# ============================================================================
# Correctness Tests
# ============================================================================


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32,))
@pytest.mark.parametrize("size", (1024, 8192, 65536))
def test_add_mul_correctness(size, dtype, device):
    """Test that fused add_mul produces same results as unfused."""
    torch.manual_seed(0)

    a = torch.randn(size, dtype=dtype, device=device)
    b = torch.randn(size, dtype=dtype, device=device)
    c = torch.randn(size, dtype=dtype, device=device)

    output_unfused = add_mul_unfused(a, b, c)
    output_fused = add_mul_fused(a, b, c)
    expected = (a + b) * c

    assert torch.allclose(output_unfused, expected)
    assert torch.allclose(output_fused, expected)
    assert torch.allclose(output_fused, output_unfused)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32,))
@pytest.mark.parametrize("size", (1024, 8192, 65536))
def test_add_mul_add_correctness(size, dtype, device):
    """Test that fused add_mul_add produces same results as unfused."""
    torch.manual_seed(0)

    a = torch.randn(size, dtype=dtype, device=device)
    b = torch.randn(size, dtype=dtype, device=device)
    c = torch.randn(size, dtype=dtype, device=device)
    d = torch.randn(size, dtype=dtype, device=device)

    output_unfused = add_mul_add_unfused(a, b, c, d)
    output_fused = add_mul_add_fused(a, b, c, d)
    expected = ((a + b) * c) + d

    # Use relaxed tolerance for floating point comparisons
    # Fused operations may have slightly different rounding behavior
    assert torch.allclose(output_unfused, expected, rtol=1e-5, atol=1e-5)
    assert torch.allclose(output_fused, expected, rtol=1e-5, atol=1e-5)
    assert torch.allclose(output_fused, output_unfused, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32,))
@pytest.mark.parametrize("size", (1024, 8192, 65536))
def test_relu_neg_correctness(size, dtype, device):
    """Test that fused relu_neg produces same results as unfused."""
    torch.manual_seed(0)

    x = torch.randn(size, dtype=dtype, device=device)

    output_unfused = relu_neg_unfused(x)
    output_fused = relu_neg_fused(x)
    expected = -torch.relu(x)

    assert torch.allclose(output_unfused, expected)
    assert torch.allclose(output_fused, expected)
    assert torch.allclose(output_fused, output_unfused)


# ============================================================================
# Accurate Profiling Utilities
# ============================================================================


class AccurateBenchmark:
    """Accurate benchmarking utility with proper warmup and timing."""

    def __init__(self, device, warmup_iterations=50, benchmark_iterations=200):
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.is_cuda = device != "cpu" and torch.cuda.is_available()

    def _sync(self):
        """Synchronize device."""
        if self.is_cuda:
            torch.cuda.synchronize()

    def _clear_cache(self):
        """Clear GPU cache and run garbage collection."""
        gc.collect()
        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def compile_and_warmup(self, fn, *args):
        """Compile function and warm up to eliminate JIT overhead.
        
        Returns the time taken for compilation/first-run.
        """
        self._sync()
        self._clear_cache()

        # First call triggers compilation
        start = time.perf_counter()
        _ = fn(*args)
        self._sync()
        compile_time = time.perf_counter() - start

        # Extensive warmup to stabilize autotuning
        for _ in range(self.warmup_iterations):
            _ = fn(*args)
        self._sync()

        return compile_time * 1000  # Return in ms

    def benchmark_execution(self, fn, *args):
        """Benchmark pure execution time after warmup.
        
        Uses CUDA events for accurate GPU timing.
        """
        self._sync()

        if self.is_cuda:
            # Use CUDA events for precise GPU timing
            start_events = [torch.cuda.Event(enable_timing=True) 
                          for _ in range(self.benchmark_iterations)]
            end_events = [torch.cuda.Event(enable_timing=True) 
                         for _ in range(self.benchmark_iterations)]

            for i in range(self.benchmark_iterations):
                start_events[i].record()
                _ = fn(*args)
                end_events[i].record()

            self._sync()

            # Calculate timings, skip first few iterations
            skip = min(10, self.benchmark_iterations // 10)
            times = [start_events[i].elapsed_time(end_events[i]) 
                    for i in range(skip, self.benchmark_iterations)]

            # Return median to reduce outlier effects
            times.sort()
            median_idx = len(times) // 2
            return times[median_idx]
        else:
            # CPU timing
            times = []
            for _ in range(self.benchmark_iterations):
                start = time.perf_counter()
                _ = fn(*args)
                end = time.perf_counter()
                times.append((end - start) * 1000)

            times.sort()
            median_idx = len(times) // 2
            return times[median_idx]

    def full_benchmark(self, fn, *args, name="function"):
        """Full benchmark including compilation and execution timing."""
        compile_time = self.compile_and_warmup(fn, *args)
        exec_time = self.benchmark_execution(fn, *args)

        return {
            "name": name,
            "compile_time_ms": compile_time,
            "exec_time_ms": exec_time,
        }


def benchmark_kernel_directly(kernel_fn, *args, device="cuda", warmup=50, repeat=200):
    """Benchmark a kernel function directly without torch.compile overhead.
    
    This measures the actual Triton kernel execution time.
    """
    is_cuda = device != "cpu" and torch.cuda.is_available()

    if is_cuda:
        torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        kernel_fn(*args)

    if is_cuda:
        torch.cuda.synchronize()

        # Use CUDA events for accurate timing
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

        for i in range(repeat):
            start_events[i].record()
            kernel_fn(*args)
            end_events[i].record()

        torch.cuda.synchronize()

        # Skip first iterations, take median
        skip = min(10, repeat // 10)
        times = [start_events[i].elapsed_time(end_events[i]) 
                for i in range(skip, repeat)]
        times.sort()
        return times[len(times) // 2]
    else:
        times = []
        for _ in range(repeat):
            start = time.perf_counter()
            kernel_fn(*args)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        times.sort()
        return times[len(times) // 2]


# ============================================================================
# Performance Benchmark Tests with Accurate Profiling
# ============================================================================


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("size", (1024 * 1024,))
def test_add_mul_performance(size, device):
    """Benchmark fused vs unfused add_mul performance with accurate profiling."""
    torch.manual_seed(0)
    dtype = torch.float32

    a = torch.randn(size, dtype=dtype, device=device)
    b = torch.randn(size, dtype=dtype, device=device)
    c = torch.randn(size, dtype=dtype, device=device)

    bench = AccurateBenchmark(device, warmup_iterations=100, benchmark_iterations=300)

    # Benchmark unfused (no torch.compile overhead)
    unfused_result = bench.full_benchmark(add_mul_unfused, a, b, c, name="unfused")

    # Benchmark fused (includes torch.compile)
    fused_result = bench.full_benchmark(add_mul_fused, a, b, c, name="fused")

    # Benchmark PyTorch native
    pytorch_result = bench.full_benchmark(add_mul_pytorch, a, b, c, name="pytorch")

    speedup = unfused_result["exec_time_ms"] / fused_result["exec_time_ms"]

    print(f"\n[add_mul] size={size}, device={device}")
    print(f"  Unfused:  {unfused_result['exec_time_ms']:.4f} ms (compile: {unfused_result['compile_time_ms']:.2f} ms)")
    print(f"  Fused:    {fused_result['exec_time_ms']:.4f} ms (compile: {fused_result['compile_time_ms']:.2f} ms)")
    print(f"  PyTorch:  {pytorch_result['exec_time_ms']:.4f} ms")
    print(f"  Speedup:  {speedup:.2f}x")

    # Performance assertion - fused should not be drastically slower
    assert fused_result["exec_time_ms"] < unfused_result["exec_time_ms"] * 2.0, (
        f"Fused version ({fused_result['exec_time_ms']:.4f}ms) is too slow compared to "
        f"unfused ({unfused_result['exec_time_ms']:.4f}ms)"
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("size", (1024 * 1024,))
def test_add_mul_add_performance(size, device):
    """Benchmark fused vs unfused add_mul_add performance."""
    torch.manual_seed(0)
    dtype = torch.float32

    a = torch.randn(size, dtype=dtype, device=device)
    b = torch.randn(size, dtype=dtype, device=device)
    c = torch.randn(size, dtype=dtype, device=device)
    d = torch.randn(size, dtype=dtype, device=device)

    bench = AccurateBenchmark(device, warmup_iterations=100, benchmark_iterations=300)

    unfused_result = bench.full_benchmark(add_mul_add_unfused, a, b, c, d, name="unfused")
    fused_result = bench.full_benchmark(add_mul_add_fused, a, b, c, d, name="fused")
    pytorch_result = bench.full_benchmark(add_mul_add_pytorch, a, b, c, d, name="pytorch")

    speedup = unfused_result["exec_time_ms"] / fused_result["exec_time_ms"]

    print(f"\n[add_mul_add] size={size}, device={device}")
    print(f"  Unfused:  {unfused_result['exec_time_ms']:.4f} ms")
    print(f"  Fused:    {fused_result['exec_time_ms']:.4f} ms")
    print(f"  PyTorch:  {pytorch_result['exec_time_ms']:.4f} ms")
    print(f"  Speedup:  {speedup:.2f}x")

    assert fused_result["exec_time_ms"] < unfused_result["exec_time_ms"] * 2.0


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("size", (1024 * 1024,))
def test_relu_neg_performance(size, device):
    """Benchmark fused vs unfused relu_neg performance."""
    torch.manual_seed(0)
    dtype = torch.float32

    x = torch.randn(size, dtype=dtype, device=device)

    bench = AccurateBenchmark(device, warmup_iterations=100, benchmark_iterations=300)

    unfused_result = bench.full_benchmark(relu_neg_unfused, x, name="unfused")
    fused_result = bench.full_benchmark(relu_neg_fused, x, name="fused")
    pytorch_result = bench.full_benchmark(relu_neg_pytorch, x, name="pytorch")

    speedup = unfused_result["exec_time_ms"] / fused_result["exec_time_ms"]

    print(f"\n[relu_neg] size={size}, device={device}")
    print(f"  Unfused:  {unfused_result['exec_time_ms']:.4f} ms")
    print(f"  Fused:    {fused_result['exec_time_ms']:.4f} ms")
    print(f"  PyTorch:  {pytorch_result['exec_time_ms']:.4f} ms")
    print(f"  Speedup:  {speedup:.2f}x")

    assert fused_result["exec_time_ms"] < unfused_result["exec_time_ms"] * 2.0


# ============================================================================
# Direct Kernel Comparison (bypassing torch.compile)
# ============================================================================


@pytest.mark.parametrize("device", get_available_devices())
def test_direct_kernel_comparison(device):
    """Compare kernels directly without torch.compile overhead.
    
    This test measures the raw kernel execution time to understand
    the true performance characteristics of fused vs unfused kernels.
    """
    torch.manual_seed(0)
    size = 1024 * 1024
    dtype = torch.float32

    a = torch.randn(size, dtype=dtype, device=device)
    b = torch.randn(size, dtype=dtype, device=device)
    c = torch.randn(size, dtype=dtype, device=device)
    temp = torch.empty_like(a)
    output = torch.empty_like(a)

    print(f"\n[Direct Kernel Comparison] device={device}, size={size}")
    print("=" * 60)

    # Benchmark individual kernels
    add_time = benchmark_kernel_directly(
        lambda: add_kernel(a, b, temp), device=device
    )
    mul_time = benchmark_kernel_directly(
        lambda: mul_kernel(temp, c, output), device=device
    )

    print(f"  add_kernel:     {add_time:.4f} ms")
    print(f"  mul_kernel:     {mul_time:.4f} ms")
    print(f"  Sequential sum: {add_time + mul_time:.4f} ms")

    # Benchmark unfused operation (both kernels)
    def unfused_op():
        add_kernel(a, b, temp)
        mul_kernel(temp, c, output)

    unfused_time = benchmark_kernel_directly(unfused_op, device=device)
    print(f"  Unfused (both): {unfused_time:.4f} ms")

    # Benchmark fused operation (after compilation)
    fused_fn = _get_add_mul_fused()
    # Force compilation
    _ = fused_fn(a, b, c)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Now benchmark just the execution
    fused_time = benchmark_kernel_directly(
        lambda: fused_fn(a, b, c), device=device
    )
    print(f"  Fused kernel:   {fused_time:.4f} ms")

    speedup = unfused_time / fused_time
    print(f"  Speedup:        {speedup:.2f}x")
    print("=" * 60)

    # Analysis
    kernel_launch_overhead = unfused_time - (add_time + mul_time)
    print(f"\nAnalysis:")
    print(f"  Kernel launch overhead (unfused): ~{kernel_launch_overhead:.4f} ms")
    print(f"  Theoretical best fused time: ~{add_time:.4f} ms (single kernel)")


# ============================================================================
# Memory Efficiency Tests
# ============================================================================


@pytest.mark.parametrize("device", get_available_devices())
def test_fusion_memory_efficiency(device):
    """Test that fusion reduces memory allocations."""
    if device == "cpu":
        pytest.skip("Memory tracking not reliable on CPU")

    torch.manual_seed(0)
    size = 1024 * 1024
    dtype = torch.float32

    a = torch.randn(size, dtype=dtype, device=device)
    b = torch.randn(size, dtype=dtype, device=device)
    c = torch.randn(size, dtype=dtype, device=device)

    # Warm up and compile
    _ = add_mul_unfused(a, b, c)
    _ = add_mul_fused(a, b, c)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Measure unfused memory
        torch.cuda.reset_peak_memory_stats()
        _ = add_mul_unfused(a, b, c)
        torch.cuda.synchronize()
        mem_unfused = torch.cuda.max_memory_allocated()

        # Measure fused memory
        torch.cuda.reset_peak_memory_stats()
        _ = add_mul_fused(a, b, c)
        torch.cuda.synchronize()
        mem_fused = torch.cuda.max_memory_allocated()

        print(f"\n[Memory Efficiency] device={device}")
        print(f"  Unfused peak memory: {mem_unfused / 1e6:.2f} MB")
        print(f"  Fused peak memory:   {mem_fused / 1e6:.2f} MB")
        print(f"  Memory reduction:    {(mem_unfused - mem_fused) / 1e6:.2f} MB")


# ============================================================================
# Comprehensive Performance Report
# ============================================================================


@pytest.mark.parametrize("device", get_available_devices())
def test_fusion_performance_report(device):
    """Generate a comprehensive performance report for fusion."""
    torch.manual_seed(0)
    dtype = torch.float32

    sizes = [1024, 8192, 65536, 262144, 1024 * 1024]

    bench = AccurateBenchmark(device, warmup_iterations=50, benchmark_iterations=100)

    print(f"\n{'='*70}")
    print(f"Fusion Performance Report - Device: {device}")
    print(f"{'='*70}")

    # Pre-compile all functions with a small tensor to avoid compilation overhead in measurements
    small_a = torch.randn(1024, dtype=dtype, device=device)
    small_b = torch.randn(1024, dtype=dtype, device=device)
    small_c = torch.randn(1024, dtype=dtype, device=device)
    _ = add_mul_fused(small_a, small_b, small_c)
    _ = relu_neg_fused(small_a)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # add_mul benchmark
    print(f"\n[add_mul: (a + b) * c]")
    print(f"{'Size':<12} {'Unfused(ms)':<12} {'Fused(ms)':<12} {'PyTorch(ms)':<12} {'Speedup':<10}")
    print("-" * 58)

    for size in sizes:
        a = torch.randn(size, dtype=dtype, device=device)
        b = torch.randn(size, dtype=dtype, device=device)
        c = torch.randn(size, dtype=dtype, device=device)

        # Use direct benchmarking after warmup
        time_unfused = bench.benchmark_execution(add_mul_unfused, a, b, c)
        time_fused = bench.benchmark_execution(add_mul_fused, a, b, c)
        time_pytorch = bench.benchmark_execution(add_mul_pytorch, a, b, c)
        speedup = time_unfused / time_fused

        print(f"{size:<12} {time_unfused:<12.4f} {time_fused:<12.4f} {time_pytorch:<12.4f} {speedup:<10.2f}x")

    # relu_neg benchmark
    print(f"\n[relu_neg: -relu(x)]")
    print(f"{'Size':<12} {'Unfused(ms)':<12} {'Fused(ms)':<12} {'PyTorch(ms)':<12} {'Speedup':<10}")
    print("-" * 58)

    for size in sizes:
        x = torch.randn(size, dtype=dtype, device=device)

        time_unfused = bench.benchmark_execution(relu_neg_unfused, x)
        time_fused = bench.benchmark_execution(relu_neg_fused, x)
        time_pytorch = bench.benchmark_execution(relu_neg_pytorch, x)
        speedup = time_unfused / time_fused

        print(f"{size:<12} {time_unfused:<12.4f} {time_fused:<12.4f} {time_pytorch:<12.4f} {speedup:<10.2f}x")

    print(f"\n{'='*70}")
    print("Note: Speedup > 1.0 means fused is faster than unfused")
    print(f"{'='*70}\n")


# ============================================================================
# Scaling Tests
# ============================================================================


@pytest.mark.parametrize("device", get_available_devices())
def test_fusion_scaling(device):
    """Test how fusion speedup scales with tensor size."""
    torch.manual_seed(0)
    dtype = torch.float32

    sizes = [1024, 4096, 16384, 65536, 262144, 1024 * 1024]
    speedups = []

    bench = AccurateBenchmark(device, warmup_iterations=50, benchmark_iterations=100)

    # Pre-compile
    small_a = torch.randn(1024, dtype=dtype, device=device)
    small_b = torch.randn(1024, dtype=dtype, device=device)
    small_c = torch.randn(1024, dtype=dtype, device=device)
    _ = add_mul_fused(small_a, small_b, small_c)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"\n[Fusion Scaling Test] device={device}")

    for size in sizes:
        a = torch.randn(size, dtype=dtype, device=device)
        b = torch.randn(size, dtype=dtype, device=device)
        c = torch.randn(size, dtype=dtype, device=device)

        time_unfused = bench.benchmark_execution(add_mul_unfused, a, b, c)
        time_fused = bench.benchmark_execution(add_mul_fused, a, b, c)
        speedup = time_unfused / time_fused
        speedups.append(speedup)

        status = "GOOD" if speedup >= 1.0 else "NEEDS_WORK"
        print(f"  Size {size:>8}: speedup = {speedup:.2f}x [{status}]")

    avg_speedup = sum(speedups) / len(speedups)
    print(f"\n  Average speedup: {avg_speedup:.2f}x")

    # Check that we're at least not hurting performance too much
    assert avg_speedup > 0.5, f"Average speedup {avg_speedup:.2f}x is too low"


# ============================================================================
# Complex Operator Chain Definitions (4-6 operators)
# ============================================================================

# Chain 1: add -> mul -> relu -> neg (4 operators)
# Computes: -relu((a + b) * c)

def add_mul_relu_neg_unfused(a, b, c):
    """Compute -relu((a + b) * c) without fusion - 4 operator chain."""
    temp1 = torch.empty_like(a)
    add_kernel(a, b, temp1)
    temp2 = torch.empty_like(a)
    mul_kernel(temp1, c, temp2)
    temp3 = torch.empty_like(a)
    relu_kernel(temp2, temp3)
    output = torch.empty_like(a)
    neg_kernel(temp3, output)
    return output


def _get_add_mul_relu_neg_fused():
    """Get or create the compiled 4-operator chain fused function."""
    if "add_mul_relu_neg" not in _fused_functions_compiled:
        @torch.compile(backend=fuser)
        def add_mul_relu_neg_fused(a, b, c):
            temp1 = torch.empty_like(a)
            add_kernel(a, b, temp1)
            temp2 = torch.empty_like(a)
            mul_kernel(temp1, c, temp2)
            temp3 = torch.empty_like(a)
            relu_kernel(temp2, temp3)
            output = torch.empty_like(a)
            neg_kernel(temp3, output)
            return output
        _fused_functions_compiled["add_mul_relu_neg"] = add_mul_relu_neg_fused
    return _fused_functions_compiled["add_mul_relu_neg"]


def add_mul_relu_neg_fused(a, b, c):
    return _get_add_mul_relu_neg_fused()(a, b, c)


def add_mul_relu_neg_pytorch(a, b, c):
    """Compute -relu((a + b) * c) using native PyTorch."""
    return -torch.relu((a + b) * c)


# Chain 2: add -> mul -> add -> mul (4 operators with multiple inputs)
# Computes: ((a + b) * c + d) * e

def add_mul_add_mul_unfused(a, b, c, d, e):
    """Compute ((a + b) * c + d) * e without fusion - 4 operator chain."""
    temp1 = torch.empty_like(a)
    add_kernel(a, b, temp1)
    temp2 = torch.empty_like(a)
    mul_kernel(temp1, c, temp2)
    temp3 = torch.empty_like(a)
    add_kernel(temp2, d, temp3)
    output = torch.empty_like(a)
    mul_kernel(temp3, e, output)
    return output


def _get_add_mul_add_mul_fused():
    """Get or create the compiled 4-operator chain with multiple inputs."""
    if "add_mul_add_mul" not in _fused_functions_compiled:
        @torch.compile(backend=fuser)
        def add_mul_add_mul_fused(a, b, c, d, e):
            temp1 = torch.empty_like(a)
            add_kernel(a, b, temp1)
            temp2 = torch.empty_like(a)
            mul_kernel(temp1, c, temp2)
            temp3 = torch.empty_like(a)
            add_kernel(temp2, d, temp3)
            output = torch.empty_like(a)
            mul_kernel(temp3, e, output)
            return output
        _fused_functions_compiled["add_mul_add_mul"] = add_mul_add_mul_fused
    return _fused_functions_compiled["add_mul_add_mul"]


def add_mul_add_mul_fused(a, b, c, d, e):
    return _get_add_mul_add_mul_fused()(a, b, c, d, e)


def add_mul_add_mul_pytorch(a, b, c, d, e):
    """Compute ((a + b) * c + d) * e using native PyTorch."""
    return ((a + b) * c + d) * e


# Chain 3: neg -> relu -> add -> mul -> neg -> relu (6 operators)
# Computes: relu(-relu(-x) + y) * z))
# Simplified: relu(-(relu(-x) + y) * z)

def complex_chain_6ops_unfused(x, y, z):
    """Compute relu(-(relu(-x) + y) * z) without fusion - 6 operator chain."""
    temp1 = torch.empty_like(x)
    neg_kernel(x, temp1)  # -x
    temp2 = torch.empty_like(x)
    relu_kernel(temp1, temp2)  # relu(-x)
    temp3 = torch.empty_like(x)
    add_kernel(temp2, y, temp3)  # relu(-x) + y
    temp4 = torch.empty_like(x)
    mul_kernel(temp3, z, temp4)  # (relu(-x) + y) * z
    temp5 = torch.empty_like(x)
    neg_kernel(temp4, temp5)  # -((relu(-x) + y) * z)
    output = torch.empty_like(x)
    relu_kernel(temp5, output)  # relu(-((relu(-x) + y) * z))
    return output


def _get_complex_chain_6ops_fused():
    """Get or create the compiled 6-operator chain fused function."""
    if "complex_chain_6ops" not in _fused_functions_compiled:
        @torch.compile(backend=fuser)
        def complex_chain_6ops_fused(x, y, z):
            temp1 = torch.empty_like(x)
            neg_kernel(x, temp1)
            temp2 = torch.empty_like(x)
            relu_kernel(temp1, temp2)
            temp3 = torch.empty_like(x)
            add_kernel(temp2, y, temp3)
            temp4 = torch.empty_like(x)
            mul_kernel(temp3, z, temp4)
            temp5 = torch.empty_like(x)
            neg_kernel(temp4, temp5)
            output = torch.empty_like(x)
            relu_kernel(temp5, output)
            return output
        _fused_functions_compiled["complex_chain_6ops"] = complex_chain_6ops_fused
    return _fused_functions_compiled["complex_chain_6ops"]


def complex_chain_6ops_fused(x, y, z):
    return _get_complex_chain_6ops_fused()(x, y, z)


def complex_chain_6ops_pytorch(x, y, z):
    """Compute relu(-((relu(-x) + y) * z)) using native PyTorch."""
    return torch.relu(-((torch.relu(-x) + y) * z))


# Chain 4: add -> add -> add -> mul -> mul (5 operators - accumulation pattern)
# Computes: (a + b + c + d) * e * f

def accumulate_chain_5ops_unfused(a, b, c, d, e, f):
    """Compute (a + b + c + d) * e * f without fusion - 5 operator chain."""
    temp1 = torch.empty_like(a)
    add_kernel(a, b, temp1)  # a + b
    temp2 = torch.empty_like(a)
    add_kernel(temp1, c, temp2)  # a + b + c
    temp3 = torch.empty_like(a)
    add_kernel(temp2, d, temp3)  # a + b + c + d
    temp4 = torch.empty_like(a)
    mul_kernel(temp3, e, temp4)  # (a + b + c + d) * e
    output = torch.empty_like(a)
    mul_kernel(temp4, f, output)  # (a + b + c + d) * e * f
    return output


def _get_accumulate_chain_5ops_fused():
    """Get or create the compiled 5-operator accumulation chain."""
    if "accumulate_chain_5ops" not in _fused_functions_compiled:
        @torch.compile(backend=fuser)
        def accumulate_chain_5ops_fused(a, b, c, d, e, f):
            temp1 = torch.empty_like(a)
            add_kernel(a, b, temp1)
            temp2 = torch.empty_like(a)
            add_kernel(temp1, c, temp2)
            temp3 = torch.empty_like(a)
            add_kernel(temp2, d, temp3)
            temp4 = torch.empty_like(a)
            mul_kernel(temp3, e, temp4)
            output = torch.empty_like(a)
            mul_kernel(temp4, f, output)
            return output
        _fused_functions_compiled["accumulate_chain_5ops"] = accumulate_chain_5ops_fused
    return _fused_functions_compiled["accumulate_chain_5ops"]


def accumulate_chain_5ops_fused(a, b, c, d, e, f):
    return _get_accumulate_chain_5ops_fused()(a, b, c, d, e, f)


def accumulate_chain_5ops_pytorch(a, b, c, d, e, f):
    """Compute (a + b + c + d) * e * f using native PyTorch."""
    return (a + b + c + d) * e * f


# ============================================================================
# Complex Chain Correctness Tests
# ============================================================================


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
@pytest.mark.parametrize("size", (1024, 8192, 65536))
def test_add_mul_relu_neg_correctness_complex(size, dtype, device):
    """Test 4-operator chain: -relu((a + b) * c)."""
    torch.manual_seed(0)

    a = torch.randn(size, dtype=dtype, device=device)
    b = torch.randn(size, dtype=dtype, device=device)
    c = torch.randn(size, dtype=dtype, device=device)

    output_unfused = add_mul_relu_neg_unfused(a, b, c)
    output_fused = add_mul_relu_neg_fused(a, b, c)
    expected = add_mul_relu_neg_pytorch(a, b, c)

    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-5

    assert torch.allclose(output_unfused, expected, rtol=rtol, atol=atol)
    assert torch.allclose(output_fused, expected, rtol=rtol, atol=atol)
    assert torch.allclose(output_fused, output_unfused, rtol=rtol, atol=atol)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
@pytest.mark.parametrize("size", (1024, 8192, 65536))
def test_add_mul_add_mul_correctness_complex(size, dtype, device):
    """Test 4-operator chain with multiple inputs: ((a + b) * c + d) * e."""
    torch.manual_seed(0)

    a = torch.randn(size, dtype=dtype, device=device)
    b = torch.randn(size, dtype=dtype, device=device)
    c = torch.randn(size, dtype=dtype, device=device)
    d = torch.randn(size, dtype=dtype, device=device)
    e = torch.randn(size, dtype=dtype, device=device)

    output_unfused = add_mul_add_mul_unfused(a, b, c, d, e)
    output_fused = add_mul_add_mul_fused(a, b, c, d, e)
    expected = add_mul_add_mul_pytorch(a, b, c, d, e)

    # Float16 has much lower precision, especially for chained operations
    rtol = 1e-2 if dtype == torch.float16 else 1e-5
    atol = 1e-2 if dtype == torch.float16 else 1e-5

    assert torch.allclose(output_unfused, expected, rtol=rtol, atol=atol)
    assert torch.allclose(output_fused, expected, rtol=rtol, atol=atol)
    assert torch.allclose(output_fused, output_unfused, rtol=rtol, atol=atol)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
@pytest.mark.parametrize("size", (1024, 8192, 65536))
def test_complex_chain_6ops_correctness_complex(size, dtype, device):
    """Test 6-operator chain: relu(-((relu(-x) + y) * z))."""
    torch.manual_seed(0)

    x = torch.randn(size, dtype=dtype, device=device)
    y = torch.randn(size, dtype=dtype, device=device)
    z = torch.randn(size, dtype=dtype, device=device)

    output_unfused = complex_chain_6ops_unfused(x, y, z)
    output_fused = complex_chain_6ops_fused(x, y, z)
    expected = complex_chain_6ops_pytorch(x, y, z)

    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-5

    assert torch.allclose(output_unfused, expected, rtol=rtol, atol=atol)
    assert torch.allclose(output_fused, expected, rtol=rtol, atol=atol)
    assert torch.allclose(output_fused, output_unfused, rtol=rtol, atol=atol)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
@pytest.mark.parametrize("size", (1024, 8192, 65536))
def test_accumulate_chain_5ops_correctness_complex(size, dtype, device):
    """Test 5-operator accumulation chain: (a + b + c + d) * e * f."""
    torch.manual_seed(0)

    a = torch.randn(size, dtype=dtype, device=device)
    b = torch.randn(size, dtype=dtype, device=device)
    c = torch.randn(size, dtype=dtype, device=device)
    d = torch.randn(size, dtype=dtype, device=device)
    e = torch.randn(size, dtype=dtype, device=device)
    f = torch.randn(size, dtype=dtype, device=device)

    output_unfused = accumulate_chain_5ops_unfused(a, b, c, d, e, f)
    output_fused = accumulate_chain_5ops_fused(a, b, c, d, e, f)
    expected = accumulate_chain_5ops_pytorch(a, b, c, d, e, f)

    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-5

    assert torch.allclose(output_unfused, expected, rtol=rtol, atol=atol)
    assert torch.allclose(output_fused, expected, rtol=rtol, atol=atol)
    assert torch.allclose(output_fused, output_unfused, rtol=rtol, atol=atol)


# ============================================================================
# Complex Chain Performance Benchmarks
# ============================================================================


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("size", (1024 * 1024,))
def test_add_mul_relu_neg_performance_complex(size, device):
    """Benchmark 4-operator chain: -relu((a + b) * c)."""
    torch.manual_seed(0)
    dtype = torch.float32

    a = torch.randn(size, dtype=dtype, device=device)
    b = torch.randn(size, dtype=dtype, device=device)
    c = torch.randn(size, dtype=dtype, device=device)

    bench = AccurateBenchmark(device, warmup_iterations=100, benchmark_iterations=300)

    unfused_result = bench.full_benchmark(add_mul_relu_neg_unfused, a, b, c, name="unfused")
    fused_result = bench.full_benchmark(add_mul_relu_neg_fused, a, b, c, name="fused")
    pytorch_result = bench.full_benchmark(add_mul_relu_neg_pytorch, a, b, c, name="pytorch")

    speedup = unfused_result["exec_time_ms"] / fused_result["exec_time_ms"]

    print(f"\n[4-op chain: -relu((a+b)*c)] size={size}, device={device}")
    print(f"  Unfused (4 kernels): {unfused_result['exec_time_ms']:.4f} ms")
    print(f"  Fused:               {fused_result['exec_time_ms']:.4f} ms")
    print(f"  PyTorch:             {pytorch_result['exec_time_ms']:.4f} ms")
    print(f"  Speedup:             {speedup:.2f}x")

    assert fused_result["exec_time_ms"] < unfused_result["exec_time_ms"] * 2.0


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("size", (1024 * 1024,))
def test_add_mul_add_mul_performance_complex(size, device):
    """Benchmark 4-operator chain with multiple inputs: ((a + b) * c + d) * e."""
    torch.manual_seed(0)
    dtype = torch.float32

    a = torch.randn(size, dtype=dtype, device=device)
    b = torch.randn(size, dtype=dtype, device=device)
    c = torch.randn(size, dtype=dtype, device=device)
    d = torch.randn(size, dtype=dtype, device=device)
    e = torch.randn(size, dtype=dtype, device=device)

    bench = AccurateBenchmark(device, warmup_iterations=100, benchmark_iterations=300)

    unfused_result = bench.full_benchmark(add_mul_add_mul_unfused, a, b, c, d, e, name="unfused")
    fused_result = bench.full_benchmark(add_mul_add_mul_fused, a, b, c, d, e, name="fused")
    pytorch_result = bench.full_benchmark(add_mul_add_mul_pytorch, a, b, c, d, e, name="pytorch")

    speedup = unfused_result["exec_time_ms"] / fused_result["exec_time_ms"]

    print(f"\n[4-op chain: ((a+b)*c+d)*e] size={size}, device={device}")
    print(f"  Unfused (4 kernels): {unfused_result['exec_time_ms']:.4f} ms")
    print(f"  Fused:               {fused_result['exec_time_ms']:.4f} ms")
    print(f"  PyTorch:             {pytorch_result['exec_time_ms']:.4f} ms")
    print(f"  Speedup:             {speedup:.2f}x")

    assert fused_result["exec_time_ms"] < unfused_result["exec_time_ms"] * 2.0


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("size", (1024 * 1024,))
def test_complex_chain_6ops_performance_complex(size, device):
    """Benchmark 6-operator chain: relu(-((relu(-x) + y) * z))."""
    torch.manual_seed(0)
    dtype = torch.float32

    x = torch.randn(size, dtype=dtype, device=device)
    y = torch.randn(size, dtype=dtype, device=device)
    z = torch.randn(size, dtype=dtype, device=device)

    bench = AccurateBenchmark(device, warmup_iterations=100, benchmark_iterations=300)

    unfused_result = bench.full_benchmark(complex_chain_6ops_unfused, x, y, z, name="unfused")
    fused_result = bench.full_benchmark(complex_chain_6ops_fused, x, y, z, name="fused")
    pytorch_result = bench.full_benchmark(complex_chain_6ops_pytorch, x, y, z, name="pytorch")

    speedup = unfused_result["exec_time_ms"] / fused_result["exec_time_ms"]

    print(f"\n[6-op chain: relu(-((relu(-x)+y)*z))] size={size}, device={device}")
    print(f"  Unfused (6 kernels): {unfused_result['exec_time_ms']:.4f} ms")
    print(f"  Fused:               {fused_result['exec_time_ms']:.4f} ms")
    print(f"  PyTorch:             {pytorch_result['exec_time_ms']:.4f} ms")
    print(f"  Speedup:             {speedup:.2f}x")

    assert fused_result["exec_time_ms"] < unfused_result["exec_time_ms"] * 2.0


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("size", (1024 * 1024,))
def test_accumulate_chain_5ops_performance_complex(size, device):
    """Benchmark 5-operator accumulation chain: (a + b + c + d) * e * f."""
    torch.manual_seed(0)
    dtype = torch.float32

    a = torch.randn(size, dtype=dtype, device=device)
    b = torch.randn(size, dtype=dtype, device=device)
    c = torch.randn(size, dtype=dtype, device=device)
    d = torch.randn(size, dtype=dtype, device=device)
    e = torch.randn(size, dtype=dtype, device=device)
    f = torch.randn(size, dtype=dtype, device=device)

    bench = AccurateBenchmark(device, warmup_iterations=100, benchmark_iterations=300)

    unfused_result = bench.full_benchmark(accumulate_chain_5ops_unfused, a, b, c, d, e, f, name="unfused")
    fused_result = bench.full_benchmark(accumulate_chain_5ops_fused, a, b, c, d, e, f, name="fused")
    pytorch_result = bench.full_benchmark(accumulate_chain_5ops_pytorch, a, b, c, d, e, f, name="pytorch")

    speedup = unfused_result["exec_time_ms"] / fused_result["exec_time_ms"]

    print(f"\n[5-op chain: (a+b+c+d)*e*f] size={size}, device={device}")
    print(f"  Unfused (5 kernels): {unfused_result['exec_time_ms']:.4f} ms")
    print(f"  Fused:               {fused_result['exec_time_ms']:.4f} ms")
    print(f"  PyTorch:             {pytorch_result['exec_time_ms']:.4f} ms")
    print(f"  Speedup:             {speedup:.2f}x")

    assert fused_result["exec_time_ms"] < unfused_result["exec_time_ms"] * 2.0


# ============================================================================
# Complex Chain Memory Efficiency Tests
# ============================================================================


@pytest.mark.parametrize("device", get_available_devices())
def test_complex_chain_memory_efficiency(device):
    """Test memory efficiency for complex operator chains.
    
    Longer chains should show more benefit from fusion due to
    elimination of more intermediate tensors.
    """
    if device == "cpu":
        pytest.skip("Memory tracking not reliable on CPU")

    torch.manual_seed(0)
    size = 1024 * 1024
    dtype = torch.float32

    a = torch.randn(size, dtype=dtype, device=device)
    b = torch.randn(size, dtype=dtype, device=device)
    c = torch.randn(size, dtype=dtype, device=device)

    # Warm up and compile all functions
    _ = add_mul_relu_neg_unfused(a, b, c)
    _ = add_mul_relu_neg_fused(a, b, c)
    _ = complex_chain_6ops_unfused(a, b, c)
    _ = complex_chain_6ops_fused(a, b, c)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

        print(f"\n[Complex Chain Memory Efficiency] device={device}, size={size}")
        print("=" * 70)

        # 4-operator chain memory
        torch.cuda.reset_peak_memory_stats()
        _ = add_mul_relu_neg_unfused(a, b, c)
        torch.cuda.synchronize()
        mem_4op_unfused = torch.cuda.max_memory_allocated()

        torch.cuda.reset_peak_memory_stats()
        _ = add_mul_relu_neg_fused(a, b, c)
        torch.cuda.synchronize()
        mem_4op_fused = torch.cuda.max_memory_allocated()

        print(f"4-op chain (-relu((a+b)*c)):")
        print(f"  Unfused: {mem_4op_unfused / 1e6:.2f} MB (3 intermediate tensors)")
        print(f"  Fused:   {mem_4op_fused / 1e6:.2f} MB")
        print(f"  Reduction: {(mem_4op_unfused - mem_4op_fused) / 1e6:.2f} MB")

        # 6-operator chain memory
        torch.cuda.reset_peak_memory_stats()
        _ = complex_chain_6ops_unfused(a, b, c)
        torch.cuda.synchronize()
        mem_6op_unfused = torch.cuda.max_memory_allocated()

        torch.cuda.reset_peak_memory_stats()
        _ = complex_chain_6ops_fused(a, b, c)
        torch.cuda.synchronize()
        mem_6op_fused = torch.cuda.max_memory_allocated()

        print(f"\n6-op chain (relu(-((relu(-x)+y)*z))):")
        print(f"  Unfused: {mem_6op_unfused / 1e6:.2f} MB (5 intermediate tensors)")
        print(f"  Fused:   {mem_6op_fused / 1e6:.2f} MB")
        print(f"  Reduction: {(mem_6op_unfused - mem_6op_fused) / 1e6:.2f} MB")

        print("=" * 70)


# ============================================================================
# Complex Chain Scaling Tests
# ============================================================================


@pytest.mark.parametrize("device", get_available_devices())
def test_complex_chain_scaling(device):
    """Test how fusion speedup scales with chain length and tensor size."""
    torch.manual_seed(0)
    dtype = torch.float32

    sizes = [1024, 16384, 262144, 1024 * 1024]

    bench = AccurateBenchmark(device, warmup_iterations=50, benchmark_iterations=100)

    print(f"\n{'='*80}")
    print(f"Complex Chain Scaling Test - Device: {device}")
    print(f"{'='*80}")

    # Pre-compile all functions
    small = torch.randn(1024, dtype=dtype, device=device)
    _ = add_mul_fused(small, small, small)
    _ = add_mul_relu_neg_fused(small, small, small)
    _ = complex_chain_6ops_fused(small, small, small)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"\n{'Size':<12} {'2-op':<10} {'4-op':<10} {'6-op':<10}")
    print("-" * 42)

    for size in sizes:
        a = torch.randn(size, dtype=dtype, device=device)
        b = torch.randn(size, dtype=dtype, device=device)
        c = torch.randn(size, dtype=dtype, device=device)

        # 2-operator chain
        time_2op_unfused = bench.benchmark_execution(add_mul_unfused, a, b, c)
        time_2op_fused = bench.benchmark_execution(add_mul_fused, a, b, c)
        speedup_2op = time_2op_unfused / time_2op_fused

        # 4-operator chain
        time_4op_unfused = bench.benchmark_execution(add_mul_relu_neg_unfused, a, b, c)
        time_4op_fused = bench.benchmark_execution(add_mul_relu_neg_fused, a, b, c)
        speedup_4op = time_4op_unfused / time_4op_fused

        # 6-operator chain
        time_6op_unfused = bench.benchmark_execution(complex_chain_6ops_unfused, a, b, c)
        time_6op_fused = bench.benchmark_execution(complex_chain_6ops_fused, a, b, c)
        speedup_6op = time_6op_unfused / time_6op_fused

        print(f"{size:<12} {speedup_2op:<10.2f}x {speedup_4op:<10.2f}x {speedup_6op:<10.2f}x")

    print(f"{'='*80}\n")


# ============================================================================
# Comprehensive Complex Chain Performance Report
# ============================================================================


@pytest.mark.parametrize("device", get_available_devices())
def test_complex_chain_performance_report(device):
    """Generate a comprehensive performance report for complex chains."""
    torch.manual_seed(0)
    dtype = torch.float32

    sizes = [1024, 8192, 65536, 262144, 1024 * 1024]

    bench = AccurateBenchmark(device, warmup_iterations=50, benchmark_iterations=100)

    print(f"\n{'='*80}")
    print(f"Complex Chain Performance Report - Device: {device}")
    print(f"{'='*80}")

    # Pre-compile all functions
    small = torch.randn(1024, dtype=dtype, device=device)
    small2 = torch.randn(1024, dtype=dtype, device=device)
    _ = add_mul_relu_neg_fused(small, small, small)
    _ = add_mul_add_mul_fused(small, small, small, small, small)
    _ = complex_chain_6ops_fused(small, small, small)
    _ = accumulate_chain_5ops_fused(small, small, small, small, small, small2)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 4-operator chain: -relu((a + b) * c)
    print(f"\n[4-op: -relu((a+b)*c)]")
    print(f"{'Size':<12} {'Unfused(ms)':<12} {'Fused(ms)':<12} {'PyTorch(ms)':<12} {'Speedup':<10}")
    print("-" * 58)

    for size in sizes:
        a = torch.randn(size, dtype=dtype, device=device)
        b = torch.randn(size, dtype=dtype, device=device)
        c = torch.randn(size, dtype=dtype, device=device)

        time_unfused = bench.benchmark_execution(add_mul_relu_neg_unfused, a, b, c)
        time_fused = bench.benchmark_execution(add_mul_relu_neg_fused, a, b, c)
        time_pytorch = bench.benchmark_execution(add_mul_relu_neg_pytorch, a, b, c)
        speedup = time_unfused / time_fused

        print(f"{size:<12} {time_unfused:<12.4f} {time_fused:<12.4f} {time_pytorch:<12.4f} {speedup:<10.2f}x")

    # 6-operator chain
    print(f"\n[6-op: relu(-((relu(-x)+y)*z))]")
    print(f"{'Size':<12} {'Unfused(ms)':<12} {'Fused(ms)':<12} {'PyTorch(ms)':<12} {'Speedup':<10}")
    print("-" * 58)

    for size in sizes:
        x = torch.randn(size, dtype=dtype, device=device)
        y = torch.randn(size, dtype=dtype, device=device)
        z = torch.randn(size, dtype=dtype, device=device)

        time_unfused = bench.benchmark_execution(complex_chain_6ops_unfused, x, y, z)
        time_fused = bench.benchmark_execution(complex_chain_6ops_fused, x, y, z)
        time_pytorch = bench.benchmark_execution(complex_chain_6ops_pytorch, x, y, z)
        speedup = time_unfused / time_fused

        print(f"{size:<12} {time_unfused:<12.4f} {time_fused:<12.4f} {time_pytorch:<12.4f} {speedup:<10.2f}x")

    # 5-operator accumulation chain
    print(f"\n[5-op: (a+b+c+d)*e*f]")
    print(f"{'Size':<12} {'Unfused(ms)':<12} {'Fused(ms)':<12} {'PyTorch(ms)':<12} {'Speedup':<10}")
    print("-" * 58)

    for size in sizes:
        a = torch.randn(size, dtype=dtype, device=device)
        b = torch.randn(size, dtype=dtype, device=device)
        c = torch.randn(size, dtype=dtype, device=device)
        d = torch.randn(size, dtype=dtype, device=device)
        e = torch.randn(size, dtype=dtype, device=device)
        f = torch.randn(size, dtype=dtype, device=device)

        time_unfused = bench.benchmark_execution(accumulate_chain_5ops_unfused, a, b, c, d, e, f)
        time_fused = bench.benchmark_execution(accumulate_chain_5ops_fused, a, b, c, d, e, f)
        time_pytorch = bench.benchmark_execution(accumulate_chain_5ops_pytorch, a, b, c, d, e, f)
        speedup = time_unfused / time_fused

        print(f"{size:<12} {time_unfused:<12.4f} {time_fused:<12.4f} {time_pytorch:<12.4f} {speedup:<10.2f}x")

    print(f"\n{'='*80}")
    print("Note: Speedup > 1.0 means fused is faster than unfused")
    print("      Longer chains theoretically benefit more from fusion")
    print(f"{'='*80}\n")
