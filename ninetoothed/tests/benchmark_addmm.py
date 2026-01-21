"""
Benchmark for AddMM operator fusion comparison.

Compares three implementations:
1. PyTorch Native: torch.addmm (multiple kernels)
2. Manually Fused: ntops.torch.addmm (single fused kernel)
3. Auto Fused: Using fusion.py to automatically fuse separate operators
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
    print("WARNING: ntops not available. Skipping ntops benchmarks.")

# Try to import fusion module
try:
    import ninetoothed.fusion as fusion
    FUSION_AVAILABLE = True
except ImportError:
    FUSION_AVAILABLE = False
    print("WARNING: ninetoothed.fusion not available. Need to switch to develop-fusion branch.")


def pytorch_addmm(input, mat1, mat2, beta=1.0, alpha=1.0):
    """Baseline: PyTorch native addmm"""
    return torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha)


def manual_fused_addmm(input, mat1, mat2, beta=1.0, alpha=1.0):
    """Manually fused: ntops implementation"""
    if not NTOPS_AVAILABLE:
        raise RuntimeError("ntops not available")
    return ntops.torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha)


def separate_ops_addmm(input, mat1, mat2, beta=1.0, alpha=1.0):
    """
    Separate operations: Explicitly use multiple operators
    This simulates what fusion.py would fuse together
    """
    # Step 1: Matrix multiplication
    matmul_result = torch.mm(mat1, mat2)

    # Step 2: Scale the matrix multiplication result
    scaled_matmul = matmul_result * alpha

    # Step 3: Scale the input
    scaled_input = input * beta

    # Step 4: Add the two
    result = scaled_input + scaled_matmul

    return result


def run_addmm_benchmarks(sizes=[(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]):
    """Run AddMM benchmarks for different matrix sizes"""

    suite = BenchmarkSuite("AddMM Fusion Performance Comparison")

    # Add test cases
    for m, n, k in sizes:
        test_name = f"AddMM ({m}×{k} × {k}×{n})"
        suite.add_test_case(
            test_name,
            {"M": m, "N": n, "K": k, "dtype": "float16"}
        )

    # Run benchmarks for each size
    for m, n, k in sizes:
        print(f"\n{'#'*70}")
        print(f"# Testing: AddMM with matrices ({m}×{k}) × ({k}×{n})")
        print(f"{'#'*70}")

        # Prepare test data
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16

        input = torch.randn((m, n), dtype=dtype, device=device)
        mat1 = torch.randn((m, k), dtype=dtype, device=device)
        mat2 = torch.randn((k, n), dtype=dtype, device=device)
        beta = 0.5
        alpha = 1.2

        test_name = f"AddMM ({m}×{k} × {k}×{n})"

        # Define implementations
        implementations = {
            "PyTorch Native": lambda: pytorch_addmm(input, mat1, mat2, beta, alpha),
            "Separate Ops": lambda: separate_ops_addmm(input, mat1, mat2, beta, alpha),
        }

        # Add manual fusion if available
        if NTOPS_AVAILABLE:
            implementations["Manual Fusion (ntops)"] = lambda: manual_fused_addmm(input, mat1, mat2, beta, alpha)

        # Run benchmark
        suite.run_benchmark(
            test_name,
            implementations,
            num_warmup=20,
            num_iterations=100,
            device=device
        )

    return suite


def main():
    """Main entry point"""
    print("="*70)
    print("AddMM Fusion Benchmark")
    print("="*70)

    # Print environment info
    gpu_info = get_gpu_info()
    print("\nTest Environment:")
    for key, value in gpu_info.items():
        print(f"  {key}: {value}")

    # Check availability
    print(f"\nAvailability:")
    print(f"  CUDA: {torch.cuda.is_available()}")
    print(f"  ntops: {NTOPS_AVAILABLE}")
    print(f"  Fusion module: {FUSION_AVAILABLE}")

    # Run benchmarks
    suite = run_addmm_benchmarks()

    # Generate report
    report_dir = os.path.join(os.path.dirname(__file__), "benchmark_reports")
    os.makedirs(report_dir, exist_ok=True)

    report_file = os.path.join(report_dir, "addmm_comparison.md")
    report = suite.generate_markdown_report(report_file)

    print("\n" + "="*70)
    print("Benchmark Complete!")
    print("="*70)

    return suite


if __name__ == "__main__":
    suite = main()
