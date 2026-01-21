# Phase 5 & Phase 6 测试操作手册

本文档提供了测试 FusionScheduler 集成到 InfiniLM 以及基准测试的完整操作步骤。

---

## 环境要求

- **Python**: 3.10+
- **CUDA**: 用于 GPU 测试（集成测试和基准测试需要）
- **依赖库**: `infinicore`, `ntops`, `ninetoothed`

---

## Phase 5: InfiniLM 集成测试

### 5.1 快速导入验证

```bash
cd /Users/lxy/lxygit/Infini0120/InfiniLM
python -c "
from infinilm.fusion_utils import create_swiglu_pattern, create_add_rms_norm_pattern, LLMFusionContext
print('✅ fusion_utils imports work!')

from infinilm.models.llama import LlamaConfig
config = LlamaConfig(enable_fusion=True, torch_dtype='float16')
print(f'✅ LlamaConfig enable_fusion = {config.enable_fusion}')
"
```

### 5.2 模型级验证脚本

运行 `test_llama_fusion.py` 来验证 Llama 模型的融合集成：

```bash
cd /Users/lxy/lxygit/Infini0120/InfiniLM
python test_llama_fusion.py
```

---

## Phase 6: 基准测试

### 6.1 运行基准测试

```bash
cd /Users/lxy/lxygit/Infini0120/InfiniCore

# 默认参数
python test/infinicore/bench_fusion.py

# 自定义参数
python test/infinicore/bench_fusion.py --batch_size 64 --hidden_dim 4096 --warmup 20 --runs 100
```

### 6.2 多批量大小对比测试

```bash
for bs in 1 8 32 64 128; do
    echo "=== Batch Size: $bs ==="
    python test/infinicore/bench_fusion.py --batch_size $bs --runs 50
done
```

---

## 完整测试流水线

```bash
# Step 1: 单元测试 (CPU)
cd /Users/lxy/lxygit/Infini0120/InfiniCore
python -m pytest test/infinicore/test_fusion_scheduler.py -v

# Step 2: 集成测试 (GPU)
python -m pytest test/infinicore/test_fusion_ntops.py -v

# Step 3: 基准测试 (GPU)
python test/infinicore/bench_fusion.py --batch_size 32 --runs 100

# Step 4: InfiniLM 集成验证
cd /Users/lxy/lxygit/Infini0120/InfiniLM
python test_llama_fusion.py
```

---

## 预期输出示例

### 基准测试输出
```
Benchmarking with Batch Size: 32, Hidden Dim: 4096, Device: cuda
[Standard (Fallback)] Avg Latency: 0.1234 ms
[Fused (Triton)] Avg Latency: 0.0567 ms
Speedup: 54.03%
```

### 单元测试输出
```
test_fusion_scheduler.py::TestSubGraph::test_opnode_creation PASSED
test_fusion_scheduler.py::TestSubGraph::test_subgraph_hash PASSED
...
```
