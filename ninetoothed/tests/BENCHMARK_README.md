# Operator Fusion Benchmark Suite

This benchmark suite compares the performance of different operator fusion strategies in the ninetoothed ecosystem.

## Overview

The benchmark compares three implementation approaches:

1. **PyTorch Native**: Standard PyTorch operators using multiple kernels
2. **Separate Ops**: Explicitly separated operations (same as native)
3. **Manual Fusion**: Hand-written fused operators using ninetoothed/ntops

## Prerequisites

- Python 3.8+
- PyTorch with CUDA support
- ntops (optional, for manual fusion benchmarks)
- GPU with CUDA support (recommended)

## Installation

```bash
# Install ninetoothed
cd /path/to/Infini/ninetoothed
pip install -e .

# Install ntops (optional, for fusion benchmarks)
cd /path/to/Infini/ntops
pip install -e .
```

## Usage

### Run All Benchmarks

```bash
cd /path/to/Infini/ninetoothed/tests
python run_all_benchmarks.py
```

This will:
1. Run all benchmark suites
2. Generate individual reports for each test
3. Create a comprehensive summary report

All reports are saved to `benchmark_reports/` directory.

### Run Individual Benchmarks

#### AddMM Benchmark
```bash
python benchmark_addmm.py
```

Tests matrix multiplication with scaling and addition:
- Input: (M, N) matrix
- Mat1: (M, K) matrix
- Mat2: (K, N) matrix
- Output: input * beta + matmul(mat1, mat2) * alpha

#### Simple Chain Benchmark
```bash
python benchmark_simple_chain.py
```

Tests a simple operator chain: scale → add → activation

## Benchmark Output

### Console Output

During execution, you'll see real-time results:

```
======================================================================
Running: AddMM (512×512 × 512×512)
======================================================================
  PyTorch Native                 - Host:   1.234 ms, Device:   1.123 ms
  Separate Ops                   - Host:   1.345 ms, Device:   1.234 ms
  Manual Fusion (ntops)          - Host:   0.987 ms, Device:   0.876 ms

  Speedup vs PyTorch Native:
    Manual Fusion (ntops)        - Host:  1.25x, Device:  1.28x
```

### Markdown Reports

Generated reports include:
- Test environment information
- Detailed performance tables
- Speedup calculations
- Analysis and conclusions

Example report structure:
```markdown
# AddMM Fusion Performance Comparison

## Test Environment
- device: NVIDIA GeForce RTX 3090
- compute_capability: (8, 6)
- total_memory_gb: 24.0
- cuda_version: 11.8
- torch_version: 2.0.0

## AddMM (512×512 × 512×512)

### Performance Results
| Implementation | Host Time (ms) | Device Time (ms) | Host Speedup | Device Speedup |
|----------------|----------------|------------------|--------------|----------------|
| PyTorch Native | 1.234 | 1.123 | - | - |
| Manual Fusion  | 0.987 | 0.876 | 1.25x | 1.28x |
```

## Interpreting Results

### Speedup Calculation

Speedup is calculated relative to the PyTorch Native baseline:
- **Speedup > 1.0x**: Faster than baseline (good!)
- **Speedup = 1.0x**: Same as baseline
- **Speedup < 1.0x**: Slower than baseline (overhead!)

### Factors Affecting Performance

1. **Data Size**:
   - Small tensors: Kernel launch overhead dominates
   - Large tensors: Memory bandwidth dominates

2. **Operation Complexity**:
   - Simple ops (element-wise): Less benefit from fusion
   - Complex ops (matrix multiply): More benefit from fusion

3. **GPU Architecture**:
   - Different GPUs have different optimal block sizes
   - Memory bandwidth varies significantly

## Customization

### Modify Test Sizes

Edit the benchmark files to change test parameters:

```python
# In benchmark_addmm.py
sizes = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]
```

### Adjust Iteration Counts

```python
suite.run_benchmark(
    test_name,
    implementations,
    num_warmup=20,      # Increase for more stable results
    num_iterations=100,  # Increase for better averaging
    device='cuda'
)
```

### Add New Benchmarks

1. Create a new benchmark file following the template
2. Use `BenchmarkSuite` for consistency
3. Add to `run_all_benchmarks.py`

## Known Limitations

1. **No Automatic Fusion**: Currently tests only manual fusion
   - Auto fusion from `develop-fusion` branch not yet integrated
   - Plan to add in future updates

2. **Kernel Count Estimation**: Not yet implemented
   - Would be useful to verify actual kernel launches

3. **Memory Profiling**: Not yet included
   - Memory usage comparison would be valuable

## Future Work

- [ ] Integrate automatic fusion from `develop-fusion` branch
- [ ] Add SDPA (scaled dot-product attention) benchmark
- [ ] Profile memory bandwidth and utilization
- [ ] Test on multiple GPU architectures
- [ ] Add CPU baseline for comparison
- [ ] Implement kernel count verification
- [ ] Add visualization of results

## Troubleshooting

### "CUDA not available"
- Ensure you have a CUDA-capable GPU
- Install PyTorch with CUDA support
- Check GPU drivers: `nvidia-smi`

### "ntops not available"
- Install ntops: `pip install -e /path/to/ntops`
- Or skip by commenting out ntops-related code

### Inconsistent Results
- Increase `num_warmup` and `num_iterations`
- Close other GPU-intensive applications
- Ensure GPU is not throttling due to temperature

## Contributing

To add new benchmarks:

1. Follow the existing template in `benchmark_*.py` files
2. Use `BenchmarkSuite` for consistency
3. Add comprehensive documentation
4. Update this README

## License

Same as ninetoothed project.
