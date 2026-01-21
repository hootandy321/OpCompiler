"""
Demo benchmark that runs without GPU/CUDA to demonstrate the benchmark framework.

This creates synthetic results to show how the benchmark system works.
"""

import os
import sys

# Mock torch for demo purposes
class MockTensor:
    def __init__(self, shape):
        self.shape = shape

class MockTorch:
    cuda = classmethod(lambda cls: False)
    float16 = "float16"
    float32 = "float32"
    randn = classmethod(lambda cls, *args, **kwargs: MockTensor(args[0] if args else (10,)))

sys.modules['torch'] = MockTorch()

# Now import our benchmark utilities
from benchmark_utils import BenchmarkSuite, BenchmarkResult


def create_demo_results():
    """Create a demo benchmark suite with synthetic results"""

    suite = BenchmarkSuite("Demo Fusion Performance (Synthetic Data)")

    # Add demo test cases
    test_cases = [
        ("AddMM Small", {"M": 512, "N": 512, "K": 512}),
        ("AddMM Medium", {"M": 1024, "N": 1024, "K": 1024}),
        ("AddMM Large", {"M": 2048, "N": 2048, "K": 2048}),
        ("Chain Small", {"size": 1024}),
        ("Chain Medium", {"size": 4096}),
        ("Chain Large", {"size": 16384}),
    ]

    for test_name, params in test_cases:
        suite.add_test_case(test_name, params)

        # Create synthetic results
        # Simulate: Manual fusion is 1.2-1.5x faster
        baseline_host = 5.0 if "Small" in test_name else (20.0 if "Medium" in test_name else 80.0)
        baseline_device = baseline_host * 0.9

        # Add some random variation
        import random
        random.seed(hash(test_name))

        variation = random.uniform(0.9, 1.1)

        results = {
            "PyTorch Native": BenchmarkResult(
                name="PyTorch Native",
                host_time_ms=baseline_host * variation,
                device_time_ms=baseline_device * variation,
                kernel_count=3
            ),
            "Separate Ops": BenchmarkResult(
                name="Separate Ops",
                host_time_ms=baseline_host * variation * 1.05,  # Slightly slower
                device_time_ms=baseline_device * variation * 1.02,
                kernel_count=3
            ),
            "Manual Fusion": BenchmarkResult(
                name="Manual Fusion",
                host_time_ms=baseline_host * variation / 1.35,  # 1.35x faster
                device_time_ms=baseline_device * variation / 1.40,  # 1.40x faster
                kernel_count=1
            ),
        }

        # Find and update the test case
        for test_case in suite.test_cases:
            if test_case["name"] == test_name:
                test_case["results"] = results
                break

    return suite


def print_demo_results(suite):
    """Print demo benchmark results"""

    print("\n" + "="*70)
    print("DEMO: Operator Fusion Performance Comparison")
    print("="*70)
    print("\nNOTE: These are SYNTHETIC results for demonstration purposes")
    print("      Real benchmarks require PyTorch with CUDA support\n")

    for test_case in suite.test_cases:
        print(f"\n{'-'*70}")
        print(f"Test: {test_case['name']}")
        print(f"Parameters: {test_case['params']}")
        print(f"{'-'*70}")

        results = test_case.get('results', {})
        baseline = results.get("PyTorch Native")

        if not baseline:
            continue

        print(f"\n  {'Implementation':<30} | {'Host (ms)':>10} | {'Device (ms)':>12} | {'Speedup':>10}")
        print(f"  {'-'*30}-+-{'-'*12}-+-{'-'*14}-+-{'-'*12}")

        for impl_name, result in results.items():
            if result:
                if impl_name != "PyTorch Native" and baseline:
                    host_spd, dev_spd = result.speedup_vs(baseline)
                    print(f"  {impl_name:<30} | {result.host_time_ms:10.3f} | "
                          f"{result.device_time_ms:12.3f} | {dev_spd:10.2f}x")
                else:
                    print(f"  {impl_name:<30} | {result.host_time_ms:10.3f} | "
                          f"{result.device_time_ms:12.3f} | {'baseline':>10}")

        # Calculate improvement
        if baseline and "Manual Fusion" in results:
            fusion_result = results["Manual Fusion"]
            improvement = ((baseline.device_time_ms - fusion_result.device_time_ms) /
                         baseline.device_time_ms * 100)
            print(f"\n  → Manual Fusion improves performance by {improvement:.1f}%")
            print(f"  → Reduces kernel launches from {baseline.kernel_count} to {fusion_result.kernel_count}")


def main():
    """Run demo benchmark"""

    # Create and display demo results
    suite = create_demo_results()
    print_demo_results(suite)

    # Generate reports
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_reports")
    os.makedirs(report_dir, exist_ok=True)

    # Generate individual report
    report_file = os.path.join(report_dir, "demo_report.md")
    suite.generate_markdown_report(report_file)

    print(f"\n{'='*70}")
    print(f"Demo report generated: {report_file}")
    print(f"{'='*70}")

    print("\n" + "="*70)
    print("How to Run Real Benchmarks")
    print("="*70)
    print("""
To run real benchmarks with actual GPU measurements:

1. Install PyTorch with CUDA:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

2. Install ntops (optional):
   cd /path/to/Infini/ntops
   pip install -e .

3. Run the benchmarks:
   cd /path/to/Infini/ninetoothed/tests
   python run_all_benchmarks.py

Or run individual benchmarks:
   python benchmark_addmm.py
   python benchmark_simple_chain.py

4. View reports in benchmark_reports/ directory
    """)

    return suite


if __name__ == "__main__":
    main()
