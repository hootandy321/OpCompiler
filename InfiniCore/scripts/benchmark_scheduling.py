#!/usr/bin/env python3
"""
Fusion Scheduling Strategy Comparison

对比三种策略：
1. always_fuse: 始终使用融合算子
2. never_fuse: 始终使用分离算子
3. smart_schedule: 根据 profile 数据智能决策

目标：展示调度算法的价值 - 它在不同场景选择最优策略

Usage:
    python scripts/benchmark_scheduling.py --device cuda --dtype float16
"""

import argparse
import json
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

import torch
import infinicore


@dataclass
class ShapeConfig:
    """Shape 配置"""
    hidden_size: int
    batch_size: int
    seq_len: int
    category: str  # "fusion_wins" or "separate_wins" or "neutral"
    
    @property
    def key(self) -> str:
        return f"h{self.hidden_size}_b{self.batch_size}_s{self.seq_len}"
    
    @property
    def total_elements(self) -> int:
        return self.hidden_size * self.batch_size * self.seq_len


# ============================================================
# 扩展 Shape 配置
# 包含融合优势场景和分离优势场景
# ============================================================
SHAPE_CONFIGS = [
    # ========== 融合优势场景 (大 shape, memory-bound) ==========
    # 大序列长度 - kernel launch overhead 被掩盖
    ShapeConfig(hidden_size=4096, batch_size=1, seq_len=2048, category="fusion_wins"),
    ShapeConfig(hidden_size=4096, batch_size=1, seq_len=4096, category="fusion_wins"),
    ShapeConfig(hidden_size=4096, batch_size=4, seq_len=1024, category="fusion_wins"),
    ShapeConfig(hidden_size=8192, batch_size=1, seq_len=1024, category="fusion_wins"),
    ShapeConfig(hidden_size=8192, batch_size=2, seq_len=512, category="fusion_wins"),
    
    # 超大批量
    ShapeConfig(hidden_size=4096, batch_size=64, seq_len=1, category="fusion_wins"),
    ShapeConfig(hidden_size=4096, batch_size=128, seq_len=1, category="fusion_wins"),
    
    # ========== 分离优势场景 (小 shape, compute-bound) ==========
    # 极小序列 - kernel launch overhead 显著
    ShapeConfig(hidden_size=256, batch_size=1, seq_len=1, category="separate_wins"),
    ShapeConfig(hidden_size=512, batch_size=1, seq_len=1, category="separate_wins"),
    ShapeConfig(hidden_size=1024, batch_size=1, seq_len=1, category="separate_wins"),
    
    # 小批量小序列
    ShapeConfig(hidden_size=2048, batch_size=1, seq_len=1, category="separate_wins"),
    ShapeConfig(hidden_size=2048, batch_size=2, seq_len=1, category="separate_wins"),
    
    # ========== 边界场景 (需要实际测量决定) ==========
    ShapeConfig(hidden_size=4096, batch_size=1, seq_len=1, category="neutral"),
    ShapeConfig(hidden_size=4096, batch_size=1, seq_len=8, category="neutral"),
    ShapeConfig(hidden_size=4096, batch_size=1, seq_len=32, category="neutral"),
    ShapeConfig(hidden_size=4096, batch_size=1, seq_len=128, category="neutral"),
    ShapeConfig(hidden_size=4096, batch_size=8, seq_len=1, category="neutral"),
    ShapeConfig(hidden_size=4096, batch_size=16, seq_len=1, category="neutral"),
    ShapeConfig(hidden_size=8192, batch_size=1, seq_len=1, category="neutral"),
]


def sync_device(device: str):
    """同步设备"""
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def time_op(func, device: str, warmup: int = 10, iterations: int = 100) -> float:
    """测量算子执行时间 (ms)"""
    for _ in range(warmup):
        func()
    sync_device(device)
    
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    sync_device(device)
    end = time.perf_counter()
    
    return (end - start) * 1000.0 / iterations


# ============================================================
# Add + RMSNorm 对比
# ============================================================
def benchmark_add_rms_norm(
    shape: ShapeConfig,
    device: str,
    dtype: torch.dtype,
    eps: float = 1e-6,
    warmup: int = 10,
    iterations: int = 100,
) -> Dict[str, float]:
    """
    对比 Add+RMSNorm 三种策略
    
    Returns:
        {
            "separate_time": float,  # add() + rms_norm() 分离执行
            "fused_time": float,     # add_rms_norm() 融合执行
            "best_strategy": str,    # "fuse" or "separate"
            "speedup": float,        # 最优 vs 最差的加速比
        }
    """
    B, S, H = shape.batch_size, shape.seq_len, shape.hidden_size
    
    x = torch.randn(B, S, H, device=device, dtype=dtype)
    residual = torch.randn(B, S, H, device=device, dtype=dtype)
    weight = torch.randn(H, device=device, dtype=dtype)
    
    x_ic = infinicore.from_torch(x)
    residual_ic = infinicore.from_torch(residual)
    weight_ic = infinicore.from_torch(weight)
    
    # 分离执行时间
    def op_separate():
        added = infinicore.op.add(x_ic, residual_ic)
        return infinicore.op.rms_norm(added, weight_ic, eps)
    separate_time = time_op(op_separate, device, warmup, iterations)
    
    # 融合执行时间
    def op_fused():
        return infinicore.op.add_rms_norm(x_ic, residual_ic, weight_ic, eps)
    fused_time = time_op(op_fused, device, warmup, iterations)
    
    # 决策
    if fused_time < separate_time:
        best = "fuse"
        speedup = separate_time / fused_time
    else:
        best = "separate"
        speedup = fused_time / separate_time
    
    return {
        "separate_time": separate_time,
        "fused_time": fused_time,
        "best_strategy": best,
        "speedup": speedup,
    }


# ============================================================
# SwiGLU 对比
# ============================================================
def benchmark_swiglu(
    shape: ShapeConfig,
    device: str,
    dtype: torch.dtype,
    warmup: int = 10,
    iterations: int = 100,
) -> Dict[str, float]:
    """
    对比 SwiGLU 三种策略
    """
    B, S, H = shape.batch_size, shape.seq_len, shape.hidden_size
    
    gate = torch.randn(B, S, H, device=device, dtype=dtype)
    up = torch.randn(B, S, H, device=device, dtype=dtype)
    
    gate_ic = infinicore.from_torch(gate)
    up_ic = infinicore.from_torch(up)
    
    # 分离执行
    def op_separate():
        activated = infinicore.op.silu(gate_ic)
        return infinicore.op.mul(activated, up_ic)
    separate_time = time_op(op_separate, device, warmup, iterations)
    
    # 融合执行
    def op_fused():
        return infinicore.op.swiglu(gate_ic, up_ic)
    fused_time = time_op(op_fused, device, warmup, iterations)
    
    if fused_time < separate_time:
        best = "fuse"
        speedup = separate_time / fused_time
    else:
        best = "separate"
        speedup = fused_time / separate_time
    
    return {
        "separate_time": separate_time,
        "fused_time": fused_time,
        "best_strategy": best,
        "speedup": speedup,
    }


# ============================================================
# 策略模拟
# ============================================================
def simulate_strategies(results: List[Dict]) -> Dict[str, float]:
    """
    模拟三种策略的总时间
    
    Returns:
        {
            "always_fuse": total_time,
            "never_fuse": total_time,
            "smart_schedule": total_time,
            "smart_vs_always_fuse": speedup,
            "smart_vs_never_fuse": speedup,
        }
    """
    always_fuse_time = 0.0
    never_fuse_time = 0.0
    smart_time = 0.0
    
    for r in results:
        always_fuse_time += r["fused_time"]
        never_fuse_time += r["separate_time"]
        smart_time += min(r["fused_time"], r["separate_time"])
    
    return {
        "always_fuse_total": always_fuse_time,
        "never_fuse_total": never_fuse_time,
        "smart_schedule_total": smart_time,
        "smart_vs_always_fuse": always_fuse_time / smart_time if smart_time > 0 else 1.0,
        "smart_vs_never_fuse": never_fuse_time / smart_time if smart_time > 0 else 1.0,
    }


# ============================================================
# Main
# ============================================================
def run_benchmark(
    device: str = "cuda",
    dtype_str: str = "float16",
    warmup: int = 10,
    iterations: int = 100,
):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.float16)
    
    print("=" * 80)
    print("Fusion Scheduling Strategy Comparison")
    print(f"Device: {device}, Dtype: {dtype_str}")
    print("=" * 80)
    
    add_rms_results = []
    swiglu_results = []
    
    # Group by category
    fusion_wins_shapes = [s for s in SHAPE_CONFIGS if s.category == "fusion_wins"]
    separate_wins_shapes = [s for s in SHAPE_CONFIGS if s.category == "separate_wins"]
    neutral_shapes = [s for s in SHAPE_CONFIGS if s.category == "neutral"]
    
    print("\n" + "=" * 80)
    print("【期望融合更优的场景】(大 shape, memory-bound)")
    print("=" * 80)
    for shape in fusion_wins_shapes:
        print(f"\n📊 {shape.key} (elements: {shape.total_elements:,})")
        
        try:
            arm = benchmark_add_rms_norm(shape, device, dtype, warmup=warmup, iterations=iterations)
            add_rms_results.append({**arm, "shape": shape.key, "category": shape.category})
            status = "✅" if arm["best_strategy"] == "fuse" else "❌"
            print(f"  Add+RMSNorm: sep={arm['separate_time']:.4f}ms, fused={arm['fused_time']:.4f}ms → {arm['best_strategy']} {status} ({arm['speedup']:.2f}x)")
        except Exception as e:
            print(f"  Add+RMSNorm: ERROR - {e}")
        
        try:
            sg = benchmark_swiglu(shape, device, dtype, warmup=warmup, iterations=iterations)
            swiglu_results.append({**sg, "shape": shape.key, "category": shape.category})
            status = "✅" if sg["best_strategy"] == "fuse" else "❌"
            print(f"  SwiGLU:      sep={sg['separate_time']:.4f}ms, fused={sg['fused_time']:.4f}ms → {sg['best_strategy']} {status} ({sg['speedup']:.2f}x)")
        except Exception as e:
            print(f"  SwiGLU: ERROR - {e}")
    
    print("\n" + "=" * 80)
    print("【期望分离更优的场景】(小 shape, kernel launch overhead 显著)")
    print("=" * 80)
    for shape in separate_wins_shapes:
        print(f"\n📊 {shape.key} (elements: {shape.total_elements:,})")
        
        try:
            arm = benchmark_add_rms_norm(shape, device, dtype, warmup=warmup, iterations=iterations)
            add_rms_results.append({**arm, "shape": shape.key, "category": shape.category})
            status = "✅" if arm["best_strategy"] == "separate" else "❌"
            print(f"  Add+RMSNorm: sep={arm['separate_time']:.4f}ms, fused={arm['fused_time']:.4f}ms → {arm['best_strategy']} {status} ({arm['speedup']:.2f}x)")
        except Exception as e:
            print(f"  Add+RMSNorm: ERROR - {e}")
        
        try:
            sg = benchmark_swiglu(shape, device, dtype, warmup=warmup, iterations=iterations)
            swiglu_results.append({**sg, "shape": shape.key, "category": shape.category})
            status = "✅" if sg["best_strategy"] == "separate" else "❌"
            print(f"  SwiGLU:      sep={sg['separate_time']:.4f}ms, fused={sg['fused_time']:.4f}ms → {sg['best_strategy']} {status} ({sg['speedup']:.2f}x)")
        except Exception as e:
            print(f"  SwiGLU: ERROR - {e}")
    
    print("\n" + "=" * 80)
    print("【边界场景】(需要实际测量)")
    print("=" * 80)
    for shape in neutral_shapes:
        print(f"\n📊 {shape.key} (elements: {shape.total_elements:,})")
        
        try:
            arm = benchmark_add_rms_norm(shape, device, dtype, warmup=warmup, iterations=iterations)
            add_rms_results.append({**arm, "shape": shape.key, "category": shape.category})
            print(f"  Add+RMSNorm: sep={arm['separate_time']:.4f}ms, fused={arm['fused_time']:.4f}ms → {arm['best_strategy']} ({arm['speedup']:.2f}x)")
        except Exception as e:
            print(f"  Add+RMSNorm: ERROR - {e}")
        
        try:
            sg = benchmark_swiglu(shape, device, dtype, warmup=warmup, iterations=iterations)
            swiglu_results.append({**sg, "shape": shape.key, "category": shape.category})
            print(f"  SwiGLU:      sep={sg['separate_time']:.4f}ms, fused={sg['fused_time']:.4f}ms → {sg['best_strategy']} ({sg['speedup']:.2f}x)")
        except Exception as e:
            print(f"  SwiGLU: ERROR - {e}")
    
    # ========== Summary ==========
    print("\n" + "=" * 80)
    print("📈 Strategy Comparison Summary")
    print("=" * 80)
    
    if add_rms_results:
        arm_summary = simulate_strategies(add_rms_results)
        print(f"\n【Add+RMSNorm】")
        print(f"  Always Fuse:   {arm_summary['always_fuse_total']:.4f}ms")
        print(f"  Never Fuse:    {arm_summary['never_fuse_total']:.4f}ms")
        print(f"  Smart Schedule:{arm_summary['smart_schedule_total']:.4f}ms ⭐")
        print(f"  Smart vs Always-Fuse: {arm_summary['smart_vs_always_fuse']:.2f}x speedup")
        print(f"  Smart vs Never-Fuse:  {arm_summary['smart_vs_never_fuse']:.2f}x speedup")
    
    if swiglu_results:
        sg_summary = simulate_strategies(swiglu_results)
        print(f"\n【SwiGLU】")
        print(f"  Always Fuse:   {sg_summary['always_fuse_total']:.4f}ms")
        print(f"  Never Fuse:    {sg_summary['never_fuse_total']:.4f}ms")
        print(f"  Smart Schedule:{sg_summary['smart_schedule_total']:.4f}ms ⭐")
        print(f"  Smart vs Always-Fuse: {sg_summary['smart_vs_always_fuse']:.2f}x speedup")
        print(f"  Smart vs Never-Fuse:  {sg_summary['smart_vs_never_fuse']:.2f}x speedup")
    
    # Combined
    all_results = add_rms_results + swiglu_results
    if all_results:
        combined = simulate_strategies(all_results)
        print(f"\n【Combined (Add+RMSNorm + SwiGLU)】")
        print(f"  Always Fuse:   {combined['always_fuse_total']:.4f}ms")
        print(f"  Never Fuse:    {combined['never_fuse_total']:.4f}ms")
        print(f"  Smart Schedule:{combined['smart_schedule_total']:.4f}ms ⭐")
        print(f"  Smart vs Always-Fuse: {combined['smart_vs_always_fuse']:.2f}x speedup")
        print(f"  Smart vs Never-Fuse:  {combined['smart_vs_never_fuse']:.2f}x speedup")
    
    print("\n" + "=" * 80)
    print("✅ Benchmark Complete")
    print("=" * 80)
    
    return {
        "add_rms_norm": add_rms_results,
        "swiglu": swiglu_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark fusion scheduling strategies")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()
    
    results = run_benchmark(
        device=args.device,
        dtype_str=args.dtype,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Results saved to: {args.output}")


if __name__ == "__main__":
    main()
