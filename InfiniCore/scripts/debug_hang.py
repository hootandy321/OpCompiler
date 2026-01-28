#!/usr/bin/env python3
"""
最小化调试脚本 - 定位卡死位置
"""
import sys

print("Step 1: Starting...", flush=True)

print("Step 2: Importing torch...", flush=True)
import torch
print(f"  torch version: {torch.__version__}", flush=True)

print("Step 3: Importing infinicore...", flush=True)
import infinicore
print(f"  infinicore loaded", flush=True)

print("Step 4: Creating torch tensor on CPU...", flush=True)
x_cpu = torch.randn(2, 2)
print(f"  x_cpu: {x_cpu.shape}", flush=True)

# 尝试检测可用设备
print("Step 5: Checking CUDA availability...", flush=True)
print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}", flush=True)

if torch.cuda.is_available():
    print("Step 6: Getting CUDA device count...", flush=True)
    print(f"  device_count: {torch.cuda.device_count()}", flush=True)
    
    print("Step 7: Creating torch tensor on CUDA...", flush=True)
    x_cuda = torch.randn(2, 2, device="cuda")
    print(f"  x_cuda: {x_cuda.shape}, device: {x_cuda.device}", flush=True)
    
    print("Step 8: Converting torch to infinicore...", flush=True)
    x_ic = infinicore.from_torch(x_cuda)
    print(f"  x_ic created", flush=True)
    
    print("Step 9: Testing infinicore.op.add...", flush=True)
    y_ic = infinicore.op.add(x_ic, x_ic)
    print(f"  add completed", flush=True)
    
    print("Step 10: Testing infinicore.synchronize...", flush=True)
    try:
        infinicore.synchronize()
        print(f"  synchronize completed", flush=True)
    except Exception as e:
        print(f"  synchronize failed: {e}", flush=True)
    
    print("Step 11: Checking available ops...", flush=True)
    for op_name in ["add", "mul", "silu", "rms_norm", "add_rms_norm", "swiglu"]:
        op = getattr(infinicore.op, op_name, None)
        print(f"  infinicore.op.{op_name}: {'available' if op else 'NOT FOUND'}", flush=True)

print("\n✅ All steps completed!", flush=True)
