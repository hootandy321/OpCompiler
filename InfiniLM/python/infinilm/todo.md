# Benchmark 使用指南与 Profile 缺失情况说明

## 1. 使用方法

### 运行命令

```bash
cd /data/liuxingyu/OpCompiler/InfiniLM

# ILUVATAR (天数)
python examples/benchmark_fusion_e2e.py \
    --iluvatar \
    --model_path /data/liuxingyu/OpCompiler/TinyLlama-1.1B-Chat-v1.0 \
    --runs 2
```

### 参数说明
- `--iluvatar` / `--nvidia`: 指定后端设备。
- `--model_path`: 模型路径 (如 TinyLlama)。
- `--runs`: 每个 Prompt 运行次数 (另外包括 1 次 warmup)。

### 测试策略
脚本会对比三种策略：
1. **never_fuse**: 关闭所有融合 (使用 `InferEngine`)。
2. **always_fuse**: 开启融合 (使用 `FusedInferEngine` + `FusionConfig`)。
3. **smart_schedule**: 基于 Profile 智能选择 (使用 `FusionScheduler.should_fuse()`)。

> **Note**: 当前基于 `infinicore.fusion.FusionScheduler` 进行融合调度。
> Benchmark 会传入 `FusionConfig` 来控制融合行为，并启用 `debug_mode=True` 打印融合决策。

---

## 2. 缺失的 Profile 数据

目前的 "smart_schedule" 策略依赖 `infinicore.fusion.FusionScheduler` 进行决策。为了使其真正“智能”，需要以下 Profile 数据支持：

### 2.1 缺失文件
目标路径: `infinicore/fusion/profile_result.json` (或通过 Config 指定的路径)

### 2.2 需要 Profile 的算子
我们需要对以下算子在不同 shape 下进行性能测试：

1.  **SwiGLU Pattern**:
    *   **Fused**: `infinicore.op.swiglu`
    *   **Unfused**: `silu(gate) * up` (PyTorch/InfiniCore 组合实现)
    *   **Shapes**: 覆盖 Prefill (Batch=1, SeqLen=128+) 和 Decode (Batch=1..32, SeqLen=1) 场景。

2.  **Add + RMSNorm Pattern**:
    *   **Fused**: `infinicore.op.add_rms_norm`
    *   **Unfused**: `x + residual` followed by `rms_norm`
    *   **Shapes**: 同上。

### 2.3 端到端 Profile

现在新增了端到端 profile 脚本，可以直接测试完整推理流程：

```bash
cd /data/liuxingyu/OpCompiler/InfiniLM

python examples/profile_e2e_fusion.py \
    --nvidia \
    --model_path /path/to/model \
    --output_path ./e2e_profile_result.json \
    --runs 3
```

输出格式:
```json
{
    "config": {...},
    "results": {
        "never_fuse": {"[prefill=128, decode=16]": {"total_ms": 100.0}},
        "always_fuse": {"[prefill=128, decode=16]": {"total_ms": 80.0}}
    }
}
```


**注意**:
- 当前 `FusionScheduler` 默认可能未加载该 json，需要确认 `FusionConfig` 中是否正确指向了生成的 json 文件。
- 如果没有 Profile 数据，`should_fuse()` 函数可能会默认返回 `True` (always fuse) 或 `False`，导致 smart_schedule 无法展现优势。

## 3. 当前实现局限性

- **Backend**: 使用的是 Python Backend，性能可能不如 C++ Backend (`FusedInferEngine`) 极致，但避开了 `random_sample` 的 stride 问题。
- **采样**: 使用 Greedy Decoding (`temperature=0`) 以确保稳定性。

## 4. 后续工作: Metax 平台验证

需要在沐曦 (Metax) 平台重复同样的实验，以验证跨平台融合效果。

### 运行命令 (Metax)

```bash
cd /data/liuxingyu/OpCompiler/InfiniLM

# METAX (沐曦)
python examples/benchmark_fusion_e2e.py \
    --metax \
    --model_path <path/to/model/on/metax> \
    --runs 2
```

**关注点**:
- `random_sample` 在 Metax 上是否稳定？
- 融合算子 (`swiglu`, `add_rms_norm`) 在 Metax 上是否有性能优势？
