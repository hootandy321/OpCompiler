"""
Benchmark script for Operator Fusion in InfiniCore.

Compares Latency and Throughput of fused vs. non-fused execution paths.
"""

import time
import torch
import numpy as np
import argparse
from typing import Dict, Any

from infinicore.fusion.fusion_scheduler import FusionScheduler
from infinicore.fusion.fusion_config import FusionConfig
from infinicore.fusion.patterns.llm_patterns import create_swiglu_pattern

def benchmark(
    name: str,
    func,
    args: tuple,
    warmup: int = 10,
    runs: int = 100
):
    # Warmup
    for _ in range(warmup):
        func(*args)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    for _ in range(runs):
        func(*args)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    avg_latency = (end_time - start_time) / runs * 1000 # ms
    print(f"[{name}] Avg Latency: {avg_latency:.4f} ms")
    return avg_latency

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    
    print(f"Benchmarking with Batch Size: {args.batch_size}, Hidden Dim: {args.hidden_dim}, Device: {device}")
    
    # 准备输入
    gate = torch.randn(args.batch_size, args.hidden_dim, device=device, dtype=dtype)
    up = torch.randn(args.batch_size, args.hidden_dim, device=device, dtype=dtype)
    inputs = {"gate": gate, "up": up}
    
    graph = create_swiglu_pattern()
    
    # 1. Non-fused Scheduler
    config_off = FusionConfig(enable_fusion=False)
    scheduler_off = FusionScheduler(config_off)
    
    # 2. Fused Scheduler
    config_on = FusionConfig(enable_fusion=True)
    scheduler_on = FusionScheduler(config_on)
    
    # Benchmark
    lat_off = benchmark("Standard (Fallback)", scheduler_off.dispatch, (graph, inputs), args.warmup, args.runs)
    lat_on = benchmark("Fused (Triton)", scheduler_on.dispatch, (graph, inputs), args.warmup, args.runs)
    
    improvement = (lat_off - lat_on) / lat_off * 100
    print(f"Speedup: {improvement:.2f}%")

if __name__ == "__main__":
    main()
