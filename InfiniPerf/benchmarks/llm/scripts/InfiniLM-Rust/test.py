import subprocess
import re
import pandas as pd
import json
import argparse

PROMPT_LENGTHS = [32, 64, 128, 256, 512, 1024]
BASE_PROMPT_WORD = "hello"
STEP_LENGTHS = [32, 64, 128, 256, 512, 1024, 2048]

def run_command(prompt, max_steps, devices, test_model):
    
    # 使用双引号确保带空格的 prompt 被正确处理
    command = f'cargo gen -p "{prompt}" --gpus {devices} --max-steps {max_steps} {test_model}'
    
    print(f"正在运行命令: {command}")
    
    # 执行命令，同时捕获 stdout 和 stderr
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True, 
        shell=True, 
        encoding='utf-8'
    )

    print(result)

    if result.returncode != 0:
        print(f"警告：命令执行失败 (退出码: {result.returncode})")
        print("--- 标准错误信息 ---\n" + result.stderr + "\n--------------------")
        return None, None

    return result.stdout, result.stderr

def extract_performance_metrics(log_output):
    """
    使用正则表达式从日志字符串中提取性能指标。
    能够智能处理时间的单位（ms 或 s），并统一转换为 ms。
    """
    if not log_output:
        return None

    metrics = {}

    # 优化正则表达式，使其可以捕获数值和单位 (ms 或 s)
    # 使用 (\w+) 来捕获单位字符串
    prefill_decode_match = re.search(
        r"prefill = ([\d.]+) ?(\w+), decode = ([\d.]+) ?(\w+)", log_output
    )

    perf_match = re.search(r"n toks = (\d+), perf: ([\d.]+)ms/tok, ([\d.]+)tok/s", log_output)
    
    if prefill_decode_match:
        prefill_val = float(prefill_decode_match.group(1))
        prefill_unit = prefill_decode_match.group(2)
        decode_val = float(prefill_decode_match.group(3))
        decode_unit = prefill_decode_match.group(4)
        
        # 统一转换为毫秒 (ms)
        # 如果单位是 's'，则乘以 1000
        metrics['prefill_ms'] = prefill_val * 1000 if prefill_unit == 's' else prefill_val
        metrics['decode_ms'] = decode_val * 1000 if decode_unit == 's' else decode_val
    
    if perf_match:
        metrics['output_tokens'] = int(perf_match.group(1))
        metrics['ms_per_token'] = float(perf_match.group(2))
        metrics['tokens_per_sec'] = float(perf_match.group(3))
    
    return metrics if metrics else None

# --- 主执行流程 ---
def main():
    parser = argparse.ArgumentParser(
        description="运行大模型性能基准测试。可以灵活配置模型、GPU组合、输入和输出长度。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "model_path", type=str, 
        help="要测试的 GGUF 模型文件的路径。"
    )
    parser.add_argument(
        "--gpus", nargs='+', default=["1", "6,7"],
        help="要测试的GPU配置列表，每个配置用空格隔开。例如: --gpus 1 6,7"
    )
    args = parser.parse_args()
    
    test_model_path = args.model_path
    device_configs = args.gpus
    
    print(f"将使用模型进行测试: {test_model_path}")
    print(f"将测试以下GPU配置: {device_configs}\n")

    all_results = []

    # ======================================================================
    # === 循环顺序为 device -> prompt -> step ===
    # ======================================================================
    for devices in device_configs:
        print(f"\n{'#'*20} 开始测试设备配置: {devices} {'#'*20}")
        for prompt_len in PROMPT_LENGTHS:
            current_prompt = " ".join([BASE_PROMPT_WORD] * prompt_len)
            for max_steps in STEP_LENGTHS:
                print(f"\n{'-'*10} 测试中: Prompt_Len: {prompt_len}, Max_Steps: {max_steps} on Devices: {devices} {'-'*10}")

                model_output, performance_logs = run_command(current_prompt, max_steps, devices, test_model_path)
                
                if model_output is None and performance_logs is None:
                    continue
                
                extracted_metrics = extract_performance_metrics(performance_logs)

                if extracted_metrics:
                    extracted_metrics['devices'] = devices
                    extracted_metrics['prompt_len'] = prompt_len
                    extracted_metrics['max_steps'] = max_steps
                    all_results.append(extracted_metrics)
                    print("成功提取到性能数据:")
                    print(json.dumps(extracted_metrics, indent=2, ensure_ascii=False))
                else:
                    print("警告：未能在此次运行中提取到性能数据。")

    if all_results:
        print(f"\n\n{'='*25} 测试结果汇总 {'='*25}")
        try:
            df = pd.DataFrame(all_results)
            df = df[['devices', 'prompt_len', 'max_steps', 'output_tokens', 'prefill_ms', 'decode_ms', 'ms_per_token', 'tokens_per_sec']]
            pd.options.display.float_format = '{:,.4f}'.format
            print(df.to_string(index=False))
        except ImportError:
            print("Pandas 未安装，将以简单格式打印结果。建议运行 'pip install pandas' 以获得更好的表格输出。")
            print(f"{'prompt_len':<12}{'max_steps':<12}{'output_tokens':<15}{'prefill_ms':<15}{'decode_ms':<15}{'ms_per_token':<18}{'tokens_per_sec':<18}")
            print('-'*105)
            for res in all_results:
                print(f"{res.get('prompt_len', 'N/A'):<12}"
                    f"{res.get('max_steps', 'N/A'):<12}"
                    f"{res.get('output_tokens', 'N/A'):<15}"
                    f"{res.get('prefill_ms', 'N/A'):<15.4f}"
                    f"{res.get('decode_ms', 'N/A'):<15.4f}"
                    f"{res.get('ms_per_token', 'N/A'):<18.4f}"
                    f"{res.get('tokens_per_sec', 'N/A'):<18.4f}")
    else:
        print("\n所有测试运行完毕，但未能收集到任何性能数据。")

if __name__ == "__main__":
    main()
