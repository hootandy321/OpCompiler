#!/usr/bin/env python3
"""
Fusion Operator Profiling Script

对比融合算子 vs 分离算子的性能，生成 profile 数据供 FusionScheduler 决策。

Usage:
    python scripts/profile_fusion_ops.py --device cuda --dtype float16 --output profile_result.json
"""

import argparse
import json
import time
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict

import torch
import infinicore


@dataclass
class ShapeConfig:
    """Shape 配置"""
    hidden_size: int
    batch_size: int
    seq_len: int
    
    @property
    def key(self) -> str:
        return f"h{self.hidden_size}_b{self.batch_size}_s{self.seq_len}"


# ============================================================
# 常用 Shape 配置（覆盖典型 LLM 推理场景）
# ============================================================
SHAPE_CONFIGS = [
    # Prefill: 长序列
    ShapeConfig(hidden_size=2048, batch_size=1, seq_len=128),
    ShapeConfig(hidden_size=2048, batch_size=1, seq_len=512),
    ShapeConfig(hidden_size=4096, batch_size=1, seq_len=128),
    ShapeConfig(hidden_size=4096, batch_size=1, seq_len=512),
    ShapeConfig(hidden_size=4096, batch_size=1, seq_len=2048),
    ShapeConfig(hidden_size=8192, batch_size=1, seq_len=128),
    ShapeConfig(hidden_size=8192, batch_size=1, seq_len=512),
    
    # Decode: 短序列，可能批量
    ShapeConfig(hidden_size=2048, batch_size=1, seq_len=1),
    ShapeConfig(hidden_size=4096, batch_size=1, seq_len=1),
    ShapeConfig(hidden_size=4096, batch_size=8, seq_len=1),
    ShapeConfig(hidden_size=4096, batch_size=32, seq_len=1),
    ShapeConfig(hidden_size=8192, batch_size=1, seq_len=1),
    ShapeConfig(hidden_size=8192, batch_size=8, seq_len=1),
]


def sync_device(device: str):
    """同步设备 - 兼容 NVIDIA 和 ILUVATAR"""
    try:
        # 使用 infinicore 的同步，更通用
        infinicore.synchronize()
    except Exception:
        # Fallback: 尝试 torch.cuda
        if "cuda" in device.lower():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass


def time_op(func, device: str, warmup: int = 5, iterations: int = 100) -> float:
    """
    测量算子执行时间
    
    Returns:
        平均执行时间 (ms)
    """
    # Warmup
    print(f"    Warmup ({warmup} iters)...", end="", flush=True)
    for i in range(warmup):
        try:
            func()
        except Exception as e:
            print(f" ERROR at iter {i}: {e}")
            raise
    sync_device(device)
    print(" done", flush=True)
    
    # Timed iterations
    print(f"    Timing ({iterations} iters)...", end="", flush=True)
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    sync_device(device)
    end = time.perf_counter()
    print(" done", flush=True)
    
    return (end - start) * 1000.0 / iterations


# ============================================================
# Add + RMSNorm 测试
# ============================================================
def check_op_availability(op_name: str) -> bool:
    """检查算子是否可用"""
    try:
        op = getattr(infinicore.op, op_name, None)
        return op is not None
    except Exception:
        return False


def profile_add_rms_norm(
    shape: ShapeConfig,
    device: str,
    dtype: torch.dtype,
    eps: float = 1e-6,
    warmup: int = 5,
    iterations: int = 100,
) -> Dict[str, float]:
    """
    Profile Add + RMSNorm 分离 vs 融合
    
    Returns:
        {"add": t1, "rms_norm": t2, "add+rms_norm": t3}
    """
    B, S, H = shape.batch_size, shape.seq_len, shape.hidden_size
    
    print(f"  Creating tensors: ({B}, {S}, {H}) on {device}...", flush=True)
    
    # 创建输入张量
    x = torch.randn(B, S, H, device=device, dtype=dtype)
    residual = torch.randn(B, S, H, device=device, dtype=dtype)
    weight = torch.randn(H, device=device, dtype=dtype)
    
    print(f"  Converting to InfiniCore...", flush=True)
    
    # InfiniCore 张量
    x_ic = infinicore.from_torch(x)
    residual_ic = infinicore.from_torch(residual)
    weight_ic = infinicore.from_torch(weight)
    
    results = {}
    
    # 1. 单独 add
    print(f"  [1/3] Testing add...", flush=True)
    def op_add():
        return infinicore.op.add(x_ic, residual_ic)
    results["add"] = time_op(op_add, device, warmup, iterations)
    
    # 2. 单独 rms_norm
    print(f"  [2/3] Testing rms_norm...", flush=True)
    added = infinicore.op.add(x_ic, residual_ic)
    def op_rms_norm():
        return infinicore.op.rms_norm(added, weight_ic, eps)
    results["rms_norm"] = time_op(op_rms_norm, device, warmup, iterations)
    
    # 3. 融合 add_rms_norm (检查是否可用)
    if check_op_availability("add_rms_norm"):
        print(f"  [3/3] Testing add_rms_norm (fused)...", flush=True)
        def op_fused():
            return infinicore.op.add_rms_norm(x_ic, residual_ic, weight_ic, eps)
        results["add+rms_norm"] = time_op(op_fused, device, warmup, iterations)
    else:
        print(f"  [3/3] add_rms_norm NOT AVAILABLE, skipping", flush=True)
        results["add+rms_norm"] = results["add"] + results["rms_norm"]  # 估计值
    
    return results


# ============================================================
# SwiGLU 测试
# ============================================================
def profile_swiglu(
    shape: ShapeConfig,
    device: str,
    dtype: torch.dtype,
    warmup: int = 5,
    iterations: int = 100,
) -> Dict[str, float]:
    """
    Profile SiLU + Mul 分离 vs SwiGLU 融合
    
    Returns:
        {"silu": t1, "mul": t2, "silu+mul": t3}
    """
    B, S, H = shape.batch_size, shape.seq_len, shape.hidden_size
    
    print(f"  Creating tensors: ({B}, {S}, {H}) on {device}...", flush=True)
    
    # 创建输入张量 (gate 和 up projection 输出)
    gate = torch.randn(B, S, H, device=device, dtype=dtype)
    up = torch.randn(B, S, H, device=device, dtype=dtype)
    
    print(f"  Converting to InfiniCore...", flush=True)
    
    # InfiniCore 张量
    gate_ic = infinicore.from_torch(gate)
    up_ic = infinicore.from_torch(up)
    
    results = {}
    
    # 1. 单独 silu
    print(f"  [1/3] Testing silu...", flush=True)
    def op_silu():
        return infinicore.op.silu(gate_ic)
    results["silu"] = time_op(op_silu, device, warmup, iterations)
    
    # 2. 单独 mul
    print(f"  [2/3] Testing mul...", flush=True)
    activated = infinicore.op.silu(gate_ic)
    def op_mul():
        return infinicore.op.mul(activated, up_ic)
    results["mul"] = time_op(op_mul, device, warmup, iterations)
    
    # 3. 融合 swiglu (检查是否可用)
    if check_op_availability("swiglu"):
        print(f"  [3/3] Testing swiglu (fused)...", flush=True)
        def op_fused():
            return infinicore.op.swiglu(gate_ic, up_ic)
        results["silu+mul"] = time_op(op_fused, device, warmup, iterations)
    else:
        print(f"  [3/3] swiglu NOT AVAILABLE, skipping", flush=True)
        results["silu+mul"] = results["silu"] + results["mul"]  # 估计值
    
    return results


# ============================================================
# Main Profiling
# ============================================================
def run_profiling(
    device: str = "cuda",
    dtype_str: str = "float16",
    shapes: List[ShapeConfig] = None,
    warmup: int = 5,
    iterations: int = 100,
) -> Dict[str, Any]:
    """
    运行完整 profiling
    
    Returns:
        {
            "metadata": {...},
            "profiles": [...]
        }
    """
    if shapes is None:
        shapes = SHAPE_CONFIGS
    
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.float16)
    
    result = {
        "metadata": {
            "device": device,
            "dtype": dtype_str,
            "warmup": warmup,
            "iterations": iterations,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "profiles": [],
    }
    
    for shape in shapes:
        print(f"\n[Profile] Shape: {shape.key}")
        
        profile_entry = {
            "shape": asdict(shape),
            "shape_key": shape.key,
        }
        
        # Add + RMSNorm
        try:
            add_rms = profile_add_rms_norm(shape, device, dtype, warmup=warmup, iterations=iterations)
            profile_entry["add_rms_norm"] = add_rms
            separate = add_rms["add"] + add_rms["rms_norm"]
            fused = add_rms["add+rms_norm"]
            speedup = separate / fused if fused > 0 else 0
            print(f"  Add+RMSNorm: separate={separate:.4f}ms, fused={fused:.4f}ms, speedup={speedup:.2f}x")
        except Exception as e:
            print(f"  Add+RMSNorm: ERROR - {e}")
            profile_entry["add_rms_norm"] = {"error": str(e)}
        
        # SwiGLU
        try:
            swiglu = profile_swiglu(shape, device, dtype, warmup=warmup, iterations=iterations)
            profile_entry["swiglu"] = swiglu
            separate = swiglu["silu"] + swiglu["mul"]
            fused = swiglu["silu+mul"]
            speedup = separate / fused if fused > 0 else 0
            print(f"  SwiGLU: separate={separate:.4f}ms, fused={fused:.4f}ms, speedup={speedup:.2f}x")
        except Exception as e:
            print(f"  SwiGLU: ERROR - {e}")
            profile_entry["swiglu"] = {"error": str(e)}
        
        result["profiles"].append(profile_entry)
    
    return result


def convert_to_legacy_format(profiles: Dict[str, Any]) -> Dict[str, Any]:
    """
    转换为旧版 heuristics.py 兼容格式
    
    取所有 shape 的中位数作为代表值
    """
    single_times = {}
    fused_times = {}
    
    for entry in profiles.get("profiles", []):
        # Add+RMSNorm
        if "add_rms_norm" in entry and "error" not in entry["add_rms_norm"]:
            data = entry["add_rms_norm"]
            single_times.setdefault("add", []).append(data["add"])
            single_times.setdefault("rms_norm", []).append(data["rms_norm"])
            fused_times.setdefault("add+rms_norm", []).append(data["add+rms_norm"])
        
        # SwiGLU
        if "swiglu" in entry and "error" not in entry["swiglu"]:
            data = entry["swiglu"]
            single_times.setdefault("silu", []).append(data["silu"])
            single_times.setdefault("mul", []).append(data["mul"])
            fused_times.setdefault("silu+mul", []).append(data["silu+mul"])
    
    # 取中位数
    def median(lst):
        if not lst:
            return 0
        s = sorted(lst)
        n = len(s)
        return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2
    
    return {
        "single": {k: median(v) for k, v in single_times.items()},
        "fused": {k: median(v) for k, v in fused_times.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Profile fusion operators")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda, cpu)")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output", type=str, default="profile_result.json", help="Output JSON path")
    parser.add_argument("--legacy-output", type=str, default=None, help="Legacy format output (optional)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Timing iterations")
    args = parser.parse_args()
    
    print(f"=== Fusion Operator Profiling ===")
    print(f"Device: {args.device}, Dtype: {args.dtype}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    
    result = run_profiling(
        device=args.device,
        dtype_str=args.dtype,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    
    # 保存完整结果
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[OK] Saved full profile to: {args.output}")
    
    # 保存旧版格式
    if args.legacy_output:
        legacy = convert_to_legacy_format(result)
        with open(args.legacy_output, "w") as f:
            json.dump(legacy, f, indent=2)
        print(f"[OK] Saved legacy format to: {args.legacy_output}")


if __name__ == "__main__":
    main()
