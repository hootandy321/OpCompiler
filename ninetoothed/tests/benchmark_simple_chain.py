"""
Benchmark for simple operator chain fusion.

Tests a common pattern: scale -> add -> activation
This represents a simple neural network layer without matrix multiplication.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../ntops/src'))

import torch
import numpy as np
from benchmark_utils import BenchmarkSuite, get_gpu_info

# Try to import ntops
try:
    import ntops
    import ntops.torch
    NTOPS_AVAILABLE = True
except ImportError:
    NTOPS_AVAILABLE = False
    print("WARNING: ntops not available.")


def pytorch_chain(x, scale, bias):
    """Baseline: PyTorch native - separate operations"""
    # Step 1: Scale
    scaled = x * scale
    # Step 2: Add bias
    biased = scaled + bias
    # Step 3: ReLU activation
    activated = torch.relu(biased)
    return activated


def separate_ops_chain(x, scale, bias):
    """Explicitly separate operations (same as PyTorch but more explicit)"""
    temp1 = x * scale
    temp2 = temp1 + bias
    output = torch.relu(temp2)
    return output


def manual_fused_chain(x, scale, bias):
    """
    Manually fused: combine all operations in one pass
    This is what an ideal fusion would look like
    """
    # Combine: output = relu(x * scale + bias)
    # In a real fused kernel, this would be a single kernel call
    return torch.relu(x * scale + bias)


def run_chain_benchmarks(sizes=[1024, 4096, 16384, 65536]):
    """Run operator chain benchmarks for different tensor sizes"""

    suite = BenchmarkSuite("Simple Operator Chain Fusion")

    # Add test cases
    for size in sizes:
        test_name = f"Chain (size={size})"
        suite.add_test_case(
            test_name,
            {"size": size, "dtype": "float32"}
        )

    # Run benchmarks for each size
    for size in sizes:
        print(f"\n{'#'*70}")
        print(f"# Testing: Operator Chain with size {size}")
        print(f"{'#'*70}")

        # Prepare test data
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float32

        x = torch.randn(size, dtype=dtype, device=device)
        scale = torch.randn(size, dtype=dtype, device=device)
        bias = torch.randn(size, dtype=dtype, device=device)

        test_name = f"Chain (size={size})"

        # Define implementations
        implementations = {
            "PyTorch Native": lambda: pytorch_chain(x, scale, bias),
            "Separate Ops": lambda: separate_ops_chain(x, scale, bias),
            "Manual Fusion (expression)": lambda: manual_fused_chain(x, scale, bias),
        }

        # Run benchmark
        suite.run_benchmark(
            test_name,
            implementations,
            num_warmup=50,
            num_iterations=200,
            device=device
        )

    return suite


def main():
    """Main entry point"""
    print("="*70)
    print("Simple Operator Chain Fusion Benchmark")
    print("="*70)

    # Print environment info
    gpu_info = get_gpu_info()
    print("\nTest Environment:")
    for key, value in gpu_info.items():
        print(f"  {key}: {value}")

    print(f"\n  CUDA: {torch.cuda.is_available()}")
    print(f"  ntops: {NTOPS_AVAILABLE}")

    # Run benchmarks
    suite = run_chain_benchmarks()

    # Generate report
    report_dir = os.path.join(os.path.dirname(__file__), "benchmark_reports")
    os.makedirs(report_dir, exist_ok=True)

    report_file = os.path.join(report_dir, "simple_chain_comparison.md")
    report = suite.generate_markdown_report(report_file)

    print("\n" + "="*70)
    print("Benchmark Complete!")
    print("="*70)

    return suite


if __name__ == "__main__":
    suite = main()
