#!/usr/bin/env python3
"""
Fusion Strategy End-to-End Comparison

端到端推理对比测试：
1. always_fuse: 始终融合
2. never_fuse: 始终不融合  
3. smart_schedule: 智能调度 (基于 profile 决策)

Usage:
    python examples/benchmark_fusion_e2e.py \
        --iluvatar \
        --model_path /data/liuxingyu/OpCompiler/TinyLlama-1.1B-Chat-v1.0 \
        --prompt "What is the capital of France?" \
        --max_new_tokens 50 \
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
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))


def get_args():
    parser = argparse.ArgumentParser(description="Fusion Strategy E2E Comparison")
    
    # Device
    parser.add_argument("--cpu", action="store_true", help="Run on CPU")
    parser.add_argument("--nvidia", action="store_true", help="Run on NVIDIA GPU")
    parser.add_argument("--iluvatar", action="store_true", help="Run on ILUVATAR GPU")
    parser.add_argument("--metax", action="store_true", help="Run on MetaX")
    parser.add_argument("--moore", action="store_true", help="Run on Moore")
    
    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism")
    
    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=50)
    
    # Benchmark
    parser.add_argument("--runs", type=int, default=2, help="Number of runs per prompt")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs")
    
    return parser.parse_args()


TEST_PROMPTS = [
    # ========== 极短 Prompt (seq_len < 16) ==========
    {
        "name": "tiny_qa",
        "prompt": "Hi",  # ~1-2 tokens
        "max_tokens": 20,
        "category": "tiny",
        "estimated_prefill_len": 2,
        "description": "极短输入 (<16 tokens)，两个算子都不应融合",
        "predicted_swiglu": False,
        "predicted_add_rms_norm": False,
        "predicted_best": "never_fuse",
    },
    {
        "name": "short_question",
        "prompt": "What is 2+2?",  # ~5-6 tokens
        "max_tokens": 30,
        "category": "tiny",
        "estimated_prefill_len": 6,
        "description": "短问题 (<16 tokens)",
        "predicted_swiglu": False,
        "predicted_add_rms_norm": False,
        "predicted_best": "never_fuse",
    },
    
    # ========== 短 Prompt (16 <= seq_len < 64) ==========
    {
        "name": "medium_short",
        "prompt": "Explain the concept of machine learning in simple terms that a beginner can understand.",  # ~20 tokens
        "max_tokens": 80,
        "category": "short",
        "estimated_prefill_len": 20,
        "description": "中短输入 (16-64 tokens)，只有 swiglu 融合",
        "predicted_swiglu": True,
        "predicted_add_rms_norm": False,
        "predicted_best": "smart_schedule",
    },
    {
        "name": "code_request",
        "prompt": "Write a Python function to calculate the nth fibonacci number using dynamic programming.",  # ~15-20 tokens
        "max_tokens": 100,
        "category": "short",
        "estimated_prefill_len": 18,
        "description": "代码请求 (16-64 tokens)",
        "predicted_swiglu": True,
        "predicted_add_rms_norm": False,
        "predicted_best": "smart_schedule",
    },
    {
        "name": "multi_sentence",
        "prompt": "I want to learn programming. What programming language should I start with? Please give me some suggestions and explain why.",  # ~25 tokens
        "max_tokens": 100,
        "category": "short",
        "estimated_prefill_len": 25,
        "description": "多句问题 (16-64 tokens)",
        "predicted_swiglu": True,
        "predicted_add_rms_norm": False,
        "predicted_best": "smart_schedule",
    },
    
    # ========== 中等 Prompt (64 <= seq_len < 128) ==========
    {
        "name": "long_context",
        "prompt": """Here is a story: Once upon a time, in a small village nestled between rolling hills and a sparkling river, there lived a young girl named Aria. She was known throughout the village for her curiosity and kind heart. Every morning, she would wake before dawn. What should Aria do next?""",  # ~65 tokens
        "max_tokens": 80,
        "category": "medium",
        "estimated_prefill_len": 70,
        "description": "中等上下文 (64-128 tokens)，两个算子都融合",
        "predicted_swiglu": True,
        "predicted_add_rms_norm": True,
        "predicted_best": "always_fuse",
    },
    {
        "name": "summarization",
        "prompt": """Please summarize the following text:

Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.

Summary:""",  # ~70 tokens
        "max_tokens": 60,
        "category": "medium",
        "estimated_prefill_len": 75,
        "description": "摘要任务 (64-128 tokens)",
        "predicted_swiglu": True,
        "predicted_add_rms_norm": True,
        "predicted_best": "always_fuse",
    },
    
    # ========== 长 Prompt (seq_len >= 128) ==========
    {
        "name": "very_long_context",
        "prompt": """Here is a detailed technical document about machine learning:

Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers to learn automatically without human intervention or assistance and adjust actions accordingly.

Machine learning algorithms are often categorized as supervised or unsupervised. What are the key differences between these approaches?""",  # ~150 tokens
        "max_tokens": 100,
        "category": "long",
        "estimated_prefill_len": 150,
        "description": "长输入 (128+ tokens)，prefill 主导",
        "predicted_swiglu": True,
        "predicted_add_rms_norm": True,
        "predicted_best": "always_fuse",
    },
]



def run_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device,
) -> tuple:
    """
    运行一次推理 (使用手动 Python 层 sampling 循环来绕过 C++ random_sample bug)

    Returns:
        (output_text, time_ms)
    """
    # Tokenize
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    # Fix: use encode() instead of batch_encode_plus() for newer transformers versions
    input_ids_list = [tokenizer.encode(input_content)]

    # Reset cache
    model.reset_cache(1, max_new_tokens + len(input_ids_list[0]))

    input_ids_infini = infinicore.from_list(input_ids_list, device=device)

    # 静默输出 (通过重定向 stdout)
    import io
    import sys
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    start = time.perf_counter()
    try:
        # 手动实现 generation 循环，使用 Python 层 sampling 绕过 C++ random_sample bug
        batch_size, seq_len = input_ids_infini.shape[:2]

        # 初始化 position_ids 和 cache_lengths
        position_ids = infinicore.from_list(
            [list(range(0, seq_len)) for _ in range(batch_size)],
            dtype=infinicore.int64,
            device=device,
        )
        cache_lengths = infinicore.from_list(
            [0],
            dtype=infinicore.int64,
            device=device,
        )

        output_tokens_list = []
        eos_token_id = model.config.eos_token_id
        eos_token_id_list = [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id

        for _ in range(max_new_tokens):
            # 调用 forward 获取 logits (不传递采样参数，避免触发 C++ 采样)
            logits = model(
                input_ids=input_ids_infini,
                position_ids=position_ids,
                cache_lengths=cache_lengths,
            )
            infinicore.sync_device()

            # Python 层 greedy decoding (使用 argmax)
            # Convert logits to numpy and do argmax there (infinicore doesn't have argmax)
            logits_np = logits.to_numpy()
            next_token_id = int(logits_np.argmax(axis=-1)[0, 0])
            token_id = next_token_id
            output_tokens_list.append(token_id)

            # 检查 EOS
            if token_id in eos_token_id_list:
                break

            # 准备下一轮输入
            seq_len = position_ids.shape[-1]
            input_ids_infini = infinicore.from_list(
                [[token_id] for _ in range(batch_size)],
                dtype=infinicore.int64,
                device=device,
            )
            position_ids = infinicore.from_list(
                [1] * batch_size,
                dtype=infinicore.int64,
                device=device,
            ).view((batch_size, 1)) + position_ids.narrow(1, seq_len - 1, 1)
            cache_lengths = cache_lengths + infinicore.from_list(
                [seq_len],
                dtype=infinicore.int64,
                device=device,
            )

        # 解码输出
        output_text = tokenizer.decode(output_tokens_list, skip_special_tokens=True)
        print(output_text, end="", flush=True)
    finally:
        output_text = sys.stdout.getvalue()
        sys.stdout = old_stdout
    
    end = time.perf_counter()
    time_ms = (end - start) * 1000.0
    
    return output_text.strip(), time_ms


def load_model_with_strategy(
    model_path: str,
    device,
    tp: int,
    strategy: str,
    profile_path: str = None,
    debug: bool = False,
) -> tuple:
    """
    根据策略加载模型 (使用 C++ infiniop 融合后端)
    
    Args:
        model_path: 模型路径
        device: 设备
        tp: 张量并行度
        strategy: 策略 - "always_fuse" | "never_fuse" | "smart_schedule"
        profile_path: profile 数据路径 (仅 smart_schedule 时使用)
        debug: 是否打印调试信息
    """
    model_path = os.path.expanduser(model_path)
    
    if strategy == "always_fuse":
        # 使用 FusedInferEngine，始终融合
        model = FusedInferEngine(
            model_path,
            device=device,
            distributed_config=DistConfig(tp),
            enable_fusion=True,
            fusion_mode="always",
            debug=debug,
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
            profile_path=profile_path,
            debug=debug,
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


def benchmark_strategy(
    model_path: str,
    device,
    tp: int,
    prompt: str,
    max_new_tokens: int,
    strategy: str,
    runs: int,
    warmup: int,
) -> dict:
    """
    对单个策略进行多次测试
    """
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy}")
    print(f"{'='*60}")
    
    # 加载模型
    print(f"Loading model...")
    model, tokenizer = load_model_with_strategy(model_path, device, tp, strategy)
    
    times = []
    
    # Warmup
    print(f"Warmup ({warmup} runs)...")
    for i in range(warmup):
        _, _ = run_inference(model, tokenizer, prompt, max_new_tokens, device)
    
    # Timed runs
    print(f"Benchmark ({runs} runs)...")
    for i in range(runs):
        output_text, time_ms = run_inference(model, tokenizer, prompt, max_new_tokens, device)
        times.append(time_ms)
        print(f"  Run {i+1}: {time_ms:.2f} ms")
    
    # Show sample output
    print(f"Sample output: {output_text[:100]}...")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"Results: avg={avg_time:.2f}ms, min={min_time:.2f}ms, max={max_time:.2f}ms")
    
    # 获取融合统计（如果有）
    fusion_stats = None
    if hasattr(model, 'get_stats'):
        fusion_stats = model.get_stats()
        print(f"Fusion stats: {fusion_stats}")
    
    return {
        "strategy": strategy,
        "times": times,
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "fusion_stats": fusion_stats,
    }


def run_all_prompts_with_strategy(
    model,
    tokenizer,
    prompts: list,
    runs: int,
    warmup: int,
    device,  # 添加 device 参数
) -> dict:
    """对一个策略运行所有 prompts"""
    results = {}
    
    for p in prompts:
        name = p["name"]
        prompt = p["prompt"]
        max_tokens = p["max_tokens"]
        
        times = []
        
        # Warmup
        for _ in range(warmup):
            run_inference(model, tokenizer, prompt, max_tokens, device)
        
        # Timed runs
        for _ in range(runs):
            _, time_ms = run_inference(model, tokenizer, prompt, max_tokens, device)
            times.append(time_ms)
        
        avg_time = sum(times) / len(times)
        results[name] = {
            "avg_time": avg_time,
            "times": times,
            "category": p["category"],
            "description": p["description"],
        }
    
    return results


def main():
    args = get_args()
    
    # 确定设备
    if args.nvidia:
        device_str = "cuda"
    elif args.iluvatar:
        device_str = "cuda"  # ILUVATAR 使用 cuda 接口
    elif args.cpu:
        device_str = "cpu"
    elif args.metax:
        device_str = "maca"
    elif args.moore:
        device_str = "musa"
    else:
        print("Please specify device: --cpu, --nvidia, --iluvatar, --metax, or --moore")
        sys.exit(1)
    
    device = infinicore.device(device_str, 0)
    
    print("=" * 80)
    print("Fusion Strategy E2E Comparison - Multi-Prompt Benchmark")
    print("=" * 80)
    print(f"Device: {device_str}")
    print(f"Model: {args.model_path}")
    print(f"Runs per prompt: {args.runs}, Warmup: {args.warmup}")
    print(f"Test prompts: {len(TEST_PROMPTS)}")
    
    # 测试三种策略
    strategies = ["never_fuse", "always_fuse", "smart_schedule"]
    all_results = {}
    
    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"📌 Strategy: {strategy}")
        print(f"{'='*80}")
        
        try:
            print("Loading model...")
            model, tokenizer = load_model_with_strategy(
                args.model_path, device, args.tp, strategy
            )
            
            print(f"Running {len(TEST_PROMPTS)} prompts...")
            results = run_all_prompts_with_strategy(
                model, tokenizer, TEST_PROMPTS, args.runs, args.warmup, device
            )
            
            all_results[strategy] = results
            
            # 显示该策略的结果
            print(f"\n{'Prompt':<20} {'Category':<15} {'Avg Time (ms)':<15}")
            print("-" * 55)
            for name, r in results.items():
                print(f"{name:<20} {r['category']:<15} {r['avg_time']:<15.2f}")
            
            total = sum(r["avg_time"] for r in results.values())
            print(f"{'TOTAL':<20} {'':<15} {total:<15.2f}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[strategy] = {"error": str(e)}
    
    # ========== Detailed Comparison ==========
    print("\n" + "=" * 80)
    print("📊 PER-PROMPT COMPARISON")
    print("=" * 80)

    valid_strategies = [s for s in strategies if "error" not in all_results.get(s, {})]

    if len(valid_strategies) >= 2:
        # Header
        header = f"{'Prompt':<20}"
        for s in valid_strategies:
            header += f" {s:<12}"
        header += " Best"
        print(header)
        print("-" * (32 + 12 * len(valid_strategies)))

        prompt_winners = {"never_fuse": 0, "always_fuse": 0, "smart_schedule": 0}

        for p in TEST_PROMPTS:
            name = p["name"]

            row = f"{name:<20}"
            times = {}
            for s in valid_strategies:
                if name in all_results[s]:
                    t = all_results[s][name]["avg_time"]
                    times[s] = t
                    row += f" {t:<12.1f}"
                else:
                    row += f" {'N/A':<12}"

            if times:
                best = min(times, key=times.get)
                prompt_winners[best] = prompt_winners.get(best, 0) + 1
                row += f" {best:<12}"

            print(row)

        # Totals
        print("-" * (32 + 12 * len(valid_strategies)))
        row = f"{'TOTAL':<20}"
        totals = {}
        for s in valid_strategies:
            total = sum(all_results[s][p["name"]]["avg_time"] for p in TEST_PROMPTS if p["name"] in all_results[s])
            totals[s] = total
            row += f" {total:<12.1f}"

        best_total = min(totals, key=totals.get)
        row += f" {best_total:<12} ⭐"
        print(row)
        
        # Strategy Summary
        print("\n" + "=" * 80)
        print("📈 STRATEGY SUMMARY")
        print("=" * 80)
        
        baseline = max(totals.values())
        print(f"\n{'Strategy':<20} {'Total (ms)':<15} {'Speedup':<10} {'Wins':<10}")
        print("-" * 60)
        for s in valid_strategies:
            speedup = baseline / totals[s] if totals[s] > 0 else 0
            wins = prompt_winners.get(s, 0)
            marker = "⭐" if s == best_total else ""
            print(f"{s:<20} {totals[s]:<15.2f} {speedup:<10.2f}x {wins:<10} {marker}")
        
        # Category Analysis
        print("\n" + "=" * 80)
        print("📊 CATEGORY ANALYSIS")
        print("=" * 80)
        
        categories = ["decode_heavy", "balanced", "prefill_heavy"]
        for cat in categories:
            cat_prompts = [p for p in TEST_PROMPTS if p["category"] == cat]
            if not cat_prompts:
                continue
            
            print(f"\n【{cat}】({len(cat_prompts)} prompts)")
            cat_totals = {}
            for s in valid_strategies:
                total = sum(
                    all_results[s][p["name"]]["avg_time"] 
                    for p in cat_prompts 
                    if p["name"] in all_results[s]
                )
                cat_totals[s] = total
            
            cat_baseline = max(cat_totals.values())
            for s in valid_strategies:
                speedup = cat_baseline / cat_totals[s] if cat_totals[s] > 0 else 0
                best_marker = "⭐" if cat_totals[s] == min(cat_totals.values()) else ""
                print(f"  {s:<18}: {cat_totals[s]:.2f}ms ({speedup:.2f}x) {best_marker}")
    
    print("\n" + "=" * 80)
    print("✅ Benchmark Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()


