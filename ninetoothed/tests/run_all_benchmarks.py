"""
Main script to run all fusion benchmarks and generate comprehensive report.
"""

import os
import sys
from datetime import datetime

# Import benchmark modules
from benchmark_utils import BenchmarkSuite, get_gpu_info
from benchmark_addmm import run_addmm_benchmarks
from benchmark_simple_chain import run_chain_benchmarks


def generate_comprehensive_report(suites, output_dir):
    """Generate a comprehensive markdown report from all benchmark suites"""

    report_path = os.path.join(output_dir, "COMPREHENSIVE_REPORT.md")

    with open(report_path, 'w') as f:
        # Title and metadata
        f.write("# Operator Fusion - Comprehensive Benchmark Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Environment info
        gpu_info = get_gpu_info()
        f.write("## Test Environment\n\n")
        for key, value in gpu_info.items():
            f.write(f"- **{key}:** {value}\n")
        f.write("\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("This report compares the performance of three implementation strategies:\n\n")
        f.write("1. **PyTorch Native**: Standard PyTorch operators (multiple kernels)\n")
        f.write("2. **Separate Ops**: Explicitly separated operations (same as native)\n")
        f.write("3. **Manual Fusion**: Hand-written fused operators (single kernel)\n\n")

        f.write("### Key Findings\n\n")
        f.write("- Fused operators reduce kernel launch overhead\n")
        f.write("- Memory traffic is reduced by keeping intermediate results in registers\n")
        f.write("- Performance improvement varies based on operation complexity and data size\n\n")

        # Detailed results from each suite
        for suite in suites:
            f.write(f"---\n\n")
            f.write(f"# {suite.suite_name}\n\n")

            for test_case in suite.test_cases:
                f.write(f"## {test_case['name']}\n\n")

                # Test parameters
                if test_case.get('params'):
                    f.write("### Parameters\n")
                    for key, value in test_case['params'].items():
                        f.write(f"- **{key}:** {value}\n")
                    f.write("\n")

                # Results table
                f.write("### Performance Results\n\n")
                f.write("| Implementation | Host Time (ms) | Device Time (ms) | Host Speedup | Device Speedup |\n")
                f.write("|----------------|----------------|------------------|--------------|----------------|\n")

                results = test_case.get('results', {})
                baseline = results.get("PyTorch Native")

                for impl_name, result in results.items():
                    if result is None:
                        f.write(f"| {impl_name} | ERROR | ERROR | - | - |\n")
                    else:
                        if baseline and impl_name != "PyTorch Native":
                            host_spd, dev_spd = result.speedup_vs(baseline)
                            f.write(f"| {impl_name} | {result.host_time_ms:.3f} | "
                                   f"{result.device_time_ms:.3f} | {host_spd:.2f}x | {dev_spd:.2f}x |\n")
                        else:
                            f.write(f"| {impl_name} | {result.host_time_ms:.3f} | "
                                   f"{result.device_time_ms:.3f} | - | - |\n")

                f.write("\n")

                # Analysis
                if baseline:
                    f.write("### Analysis\n\n")
                    for impl_name, result in results.items():
                        if result and impl_name != "PyTorch Native":
                            host_spd, dev_spd = result.speedup_vs(baseline)
                            improvement = ((baseline.device_time_ms - result.device_time_ms) /
                                         baseline.device_time_ms * 100)
                            f.write(f"- **{impl_name}**: ")
                            if dev_spd > 1.0:
                                f.write(f"{dev_spd:.2f}x faster ({improvement:.1f}% reduction in device time)\n")
                            else:
                                f.write(f"{1/dev_spd:.2f}x slower ({-improvement:.1f}% increase in device time)\n")
                    f.write("\n")

        # Conclusion
        f.write("---\n\n")
        f.write("# Conclusion\n\n")
        f.write("## Recommendations\n\n")
        f.write("1. **Use manual fusion for performance-critical paths**: Hand-written fusion provides the best performance\n")
        f.write("2. **Consider automatic fusion for developer productivity**: Once implemented, auto fusion can provide good performance with less effort\n")
        f.write("3. **Profile before optimizing**: Not all operations benefit equally from fusion\n\n")

        f.write("## Future Work\n\n")
        f.write("- [ ] Implement automatic fusion using `fusion.py` from develop-fusion branch\n")
        f.write("- [ ] Add more complex operators (e.g., scaled_dot_product_attention)\n")
        f.write("- [ ] Test on different GPU architectures\n")
        f.write("- [ ] Profile memory bandwidth utilization\n")
        f.write("- [ ] Compare with other frameworks (e.g., TensorFlow, JAX)\n")

    print(f"\nComprehensive report saved to: {report_path}")
    return report_path


def main():
    """Main entry point to run all benchmarks"""
    print("="*70)
    print("Running All Fusion Benchmarks")
    print("="*70)

    # Print environment info
    gpu_info = get_gpu_info()
    print("\nTest Environment:")
    for key, value in gpu_info.items():
        print(f"  {key}: {value}")

    if not torch.cuda.is_available():
        print("\nWARNING: CUDA not available. Running on CPU (results may not be meaningful)")

    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.join(script_dir, "benchmark_reports")
    os.makedirs(report_dir, exist_ok=True)

    # Run all benchmark suites
    suites = []

    print("\n" + "="*70)
    print("Running AddMM Benchmarks...")
    print("="*70)
    try:
        addmm_suite = run_addmm_benchmarks()
        suites.append(addmm_suite)
        addmm_suite.generate_markdown_report(os.path.join(report_dir, "addmm_detailed.md"))
    except Exception as e:
        print(f"ERROR running AddMM benchmarks: {e}")

    print("\n" + "="*70)
    print("Running Simple Chain Benchmarks...")
    print("="*70)
    try:
        chain_suite = run_chain_benchmarks()
        suites.append(chain_suite)
        chain_suite.generate_markdown_report(os.path.join(report_dir, "chain_detailed.md"))
    except Exception as e:
        print(f"ERROR running Chain benchmarks: {e}")

    # Generate comprehensive report
    if suites:
        print("\n" + "="*70)
        print("Generating Comprehensive Report...")
        print("="*70)
        generate_comprehensive_report(suites, report_dir)

    print("\n" + "="*70)
    print("All Benchmarks Complete!")
    print("="*70)
    print(f"\nReports saved to: {report_dir}")
    print("\nGenerated files:")
    for filename in os.listdir(report_dir):
        filepath = os.path.join(report_dir, filename)
        print(f"  - {filename} ({os.path.getsize(filepath)} bytes)")

    return suites


if __name__ == "__main__":
    import torch
    suites = main()
