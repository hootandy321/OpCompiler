#!/usr/bin/env python3
"""
End-to-End Fusion Profiling Script

控制输入序列长度进行端到端推理 Profile，存储结果为 JSON。

输出格式 (类似 test/infinicore/profile.py):
{
    "config": {...},
    "results": {
        "never_fuse": {
            "[prefill=128, decode=10]": {"total_ms": 100.0, "prefill_ms": 80.0, "decode_ms": 20.0},
            ...
        },
        "always_fuse": {...},
        "smart_schedule": {...}
    }
}

Usage:
    python examples/profile_e2e_fusion.py \
        --nvidia \
        --model_path /path/to/model \
        --output_path ./e2e_profile_result.json \
        --runs 3
"""

import infinicore
from transformers import AutoTokenizer
from tokenizers import decoders as _dec
from infinilm.modeling_utils import load_model_state_dict_by_file
from infinilm.distributed import DistConfig
from infinilm.infer_engine import GenerationConfig, InferEngine
from infinilm.fused_infer_engine import FusedInferEngine
import argparse
import sys
import time
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))

# ============================================================
# Profile 配置: 控制 prefill 和 decode 长度
# ============================================================
PROFILE_CONFIGS = [
    # (prefill_length, decode_length, description)
    # Prefill-heavy: 长输入，短输出
    (16, 8, "tiny"),
    (32, 16, "short_decode"),
    (64, 16, "short_decode"),
    (128, 16, "medium_prefill"),
    (256, 16, "long_prefill"),
    (512, 16, "very_long_prefill"),
    
    # Decode-heavy: 短输入，长输出
    (16, 32, "short_prefill_medium_decode"),
    (16, 64, "short_prefill_long_decode"),
    (32, 64, "medium_prefill_long_decode"),
    
    # Balanced
    (64, 64, "balanced_medium"),
    (128, 128, "balanced_long"),
]

# 基础文本用于生成指定长度的 prompt
BASE_TEXT = """The quick brown fox jumps over the lazy dog. This is a common pangram used in typing practice. 
Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.
Natural language processing enables computers to understand, interpret, and generate human language.
Deep learning uses neural networks with many layers to model complex patterns in data.
Transformers have revolutionized NLP by enabling parallel processing of sequences.
Large language models like GPT and LLaMA have shown remarkable capabilities in various tasks.
"""


def get_args():
    parser = argparse.ArgumentParser(description="E2E Fusion Profiling")
    
    # Device
    parser.add_argument("--cpu", action="store_true", help="Run on CPU")
    parser.add_argument("--nvidia", action="store_true", help="Run on NVIDIA GPU")
    parser.add_argument("--metax", action="store_true", help="Run on MetaX")
    parser.add_argument("--moore", action="store_true", help="Run on Moore")
    
    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism")
    
    # Profile
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per config")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs")
    parser.add_argument("--output_path", type=str, default="./e2e_profile_result.json")
    
    # Strategies to profile
    parser.add_argument("--strategies", type=str, nargs="+", 
                        default=["never_fuse", "always_fuse"],
                        help="Strategies to profile")
    
    return parser.parse_args()


def generate_prompt_with_length(tokenizer, target_tokens: int) -> str:
    """生成指定 token 长度的 prompt"""
    # 扩展 BASE_TEXT 直到达到目标长度
    text = BASE_TEXT
    while True:
        tokens = tokenizer.encode(text)
        if len(tokens) >= target_tokens:
            break
        text = text + " " + BASE_TEXT
    
    # 截断到精确长度
    tokens = tokenizer.encode(text)[:target_tokens]
    return tokenizer.decode(tokens)


def profile_single_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> dict:
    """
    执行单次推理并返回时间测量
    
    Returns:
        {"total_ms": float, "prefill_ms": float, "decode_ms": float, "input_len": int, "output_len": int}
    """
    # Tokenize
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    input_ids_list = tokenizer.batch_encode_plus([input_content])["input_ids"]
    input_len = len(input_ids_list[0])
    
    # Reset cache
    model.reset_cache(1, max_new_tokens + input_len)
    
    # Generate
    input_ids_infini = infinicore.from_list(input_ids_list)
    
    # Prefill timing
    prefill_start = time.perf_counter()
    
    output_ids = model.generate(
        input_ids_infini,
        GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=0,  # Greedy decoding - bypasses random_sample bug
            top_k=1,
            top_p=1.0
        ),
    )
    
    total_end = time.perf_counter()
    total_ms = (total_end - prefill_start) * 1000.0
    
    # 简化: 假设 prefill 时间占第一个 token 生成的大部分
    # 更精确需要修改 generate 函数追踪
    output_len = len(output_ids)
    
    # 粗略估计: prefill_ms ≈ total_ms * (input_len / (input_len + output_len))
    # 这是一个简化估计，实际应该在 generate 中打点
    ratio = input_len / (input_len + output_len) if (input_len + output_len) > 0 else 0.5
    prefill_ms = total_ms * ratio
    decode_ms = total_ms - prefill_ms
    
    return {
        "total_ms": total_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "input_len": input_len,
        "output_len": output_len,
    }


def load_model_with_strategy(model_path: str, device, tp: int, strategy: str):
    """加载模型 (使用 C++ infiniop 融合后端)"""
    model_path = os.path.expanduser(model_path)
    
    if strategy == "always_fuse":
        # 使用 FusedInferEngine，始终融合
        model = FusedInferEngine(
            model_path,
            device=device,
            distributed_config=DistConfig(tp),
            enable_fusion=True,
            fusion_mode="always",
            debug=False,  # Profile 时关闭 debug 输出
        )
        
    elif strategy == "never_fuse":
        # 使用普通 InferEngine，不融合
        model = InferEngine(
            model_path,
            device=device,
            distributed_config=DistConfig(tp),
        )
        
    elif strategy == "smart_schedule":
        # 使用 FusedInferEngine，基于 profile 智能调度
        model = FusedInferEngine(
            model_path,
            device=device,
            distributed_config=DistConfig(tp),
            enable_fusion=True,
            fusion_mode="profile",
            debug=False,
        )
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # 加载权重
    load_model_state_dict_by_file(model, model_path, dtype=model.config.dtype)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 修复 LLaMA tokenizer
    if getattr(model.config, "model_type", "") == "llama":
        backend = getattr(tokenizer, "backend_tokenizer", None)
        target = getattr(backend, "_tokenizer", backend)
        norm = getattr(target, "normalizer", None)
        dec = getattr(target, "decoder", None)
        sn = repr(norm)[:800] if norm is not None else ""
        sd = repr(dec)[:800] if dec is not None else ""
        has_prepend = "Prepend" in sn
        has_strip = "Strip" in sd
        if has_prepend and has_strip:
            target.decoder = _dec.Sequence([
                _dec.Replace("▁", " "),
                _dec.ByteFallback(),
                _dec.Fuse(),
            ])
    
    return model, tokenizer


def profile_strategy(
    model,
    tokenizer,
    strategy: str,
    configs: list,
    runs: int,
    warmup: int,
) -> dict:
    """
    对单个策略进行全面 profile
    
    Returns:
        {"[prefill=X, decode=Y]": {"total_ms": avg, "prefill_ms": avg, ...}, ...}
    """
    results = {}
    
    for prefill_len, decode_len, desc in configs:
        label = f"[prefill={prefill_len}, decode={decode_len}]"
        print(f"  Profiling {label} ({desc})...")
        
        # 生成指定长度的 prompt
        prompt = generate_prompt_with_length(tokenizer, prefill_len)
        
        # Warmup
        for _ in range(warmup):
            try:
                _ = profile_single_inference(model, tokenizer, prompt, decode_len)
            except Exception as e:
                print(f"    [Warning] Warmup failed: {e}")
        
        # Timed runs
        run_results = []
        for i in range(runs):
            try:
                result = profile_single_inference(model, tokenizer, prompt, decode_len)
                run_results.append(result)
            except Exception as e:
                print(f"    [Error] Run {i+1} failed: {e}")
        
        # 计算平均值
        if run_results:
            avg_result = {
                "total_ms": round(np.mean([r["total_ms"] for r in run_results]), 4),
                "prefill_ms": round(np.mean([r["prefill_ms"] for r in run_results]), 4),
                "decode_ms": round(np.mean([r["decode_ms"] for r in run_results]), 4),
                "input_len": run_results[0]["input_len"],
                "output_len": round(np.mean([r["output_len"] for r in run_results]), 1),
                "runs": len(run_results),
            }
            results[label] = avg_result
            print(f"    => total: {avg_result['total_ms']:.2f}ms, "
                  f"prefill: {avg_result['prefill_ms']:.2f}ms, "
                  f"decode: {avg_result['decode_ms']:.2f}ms")
        else:
            print(f"    => [FAILED] No successful runs")
    
    return results


def main():
    args = get_args()
    
    # 确定设备
    if args.nvidia:
        device_str = "cuda"
    elif args.cpu:
        device_str = "cpu"
    elif args.metax:
        device_str = "maca"
    elif args.moore:
        device_str = "musa"
    else:
        print("Please specify device: --cpu, --nvidia, --metax, or --moore")
        sys.exit(1)
    
    device = infinicore.device(device_str, 0)
    
    print("=" * 60)
    print("E2E Fusion Profiling")
    print("=" * 60)
    print(f"Device: {device_str}")
    print(f"Model: {args.model_path}")
    print(f"Strategies: {args.strategies}")
    print(f"Runs per config: {args.runs}")
    print(f"Output: {args.output_path}")
    print("=" * 60)
    
    # 存储所有结果
    all_results = {
        "config": {
            "device": device_str,
            "model_path": args.model_path,
            "runs": args.runs,
            "warmup": args.warmup,
            "strategies": args.strategies,
            "profile_configs": [
                {"prefill": p, "decode": d, "desc": desc}
                for p, d, desc in PROFILE_CONFIGS
            ],
        },
        "results": {}
    }
    
    # Profile 每个策略
    for strategy in args.strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy}")
        print(f"{'='*60}")
        
        # 加载模型
        print("Loading model...")
        model, tokenizer = load_model_with_strategy(
            args.model_path, device, args.tp, strategy
        )
        
        # Profile
        results = profile_strategy(
            model, tokenizer, strategy,
            PROFILE_CONFIGS, args.runs, args.warmup
        )
        
        all_results["results"][strategy] = results
        
        # 释放模型 (如果需要)
        del model
    
    # 保存结果
    print(f"\n{'='*60}")
    print("Saving results...")
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {args.output_path}")
    
    # 打印摘要
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    for strategy, results in all_results["results"].items():
        print(f"\n{strategy}:")
        for label, data in results.items():
            print(f"  {label}: {data['total_ms']:.2f}ms")
    
    # 计算加速比 (如果有多个策略)
    if "never_fuse" in all_results["results"] and "always_fuse" in all_results["results"]:
        print(f"\n{'='*60}")
        print("Speedup (always_fuse vs never_fuse)")
        print(f"{'='*60}")
        
        never = all_results["results"]["never_fuse"]
        always = all_results["results"]["always_fuse"]
        
        for label in never.keys():
            if label in always:
                speedup = never[label]["total_ms"] / always[label]["total_ms"]
                print(f"  {label}: {speedup:.2f}x")


if __name__ == "__main__":
    main()
