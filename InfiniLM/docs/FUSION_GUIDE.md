# InfiniLM Graph 缓存优化

基于 InfiniCore Graph Recording 机制，结合 FusionScheduler 实现算子融合优化。

## 快速开始

```bash
# 启用融合优化
python examples/jiuge.py --nvidia --model_path=<path> --enable-fusion

# 禁用（默认）
python examples/jiuge.py --nvidia --model_path=<path>
```

## 架构

```
FusedInferEngine
├── FusionScheduler (融合决策)
│   ├── FusionHeuristics (启发式规则)
│   └── KernelCompiler (内核编译)
└── Graph Cache (回退路径)
```

## 执行策略

| 阶段 | 融合路径 | 回退路径 |
|------|---------|---------|
| 录制 | Graph Recording | 同左 |
| 转换 | convert_graph_to_subgraph() | 失败 |
| 决策 | FusionScheduler.dispatch() | - |
| 执行 | 融合内核 或 标准算子 | Graph.run() |

## API

```python
from infinilm import FusedInferEngine

engine = FusedInferEngine(
    model_path="path/to/model",
    enable_fusion=True,       # 启用融合
    warmup_iterations=1,      # 预热次数
)

# 推理
output = engine.forward(input_ids=tokens, pos=positions)

# 统计
print(engine.get_stats())
# {
#   "fusion_attempts": 10,
#   "fusion_successes": 8,
#   "fusion_fallbacks": 2,
#   "fusion_modes": {"abc123": "fusion", ...}
# }

# 运行时开关
engine.set_fusion_enabled(False)
engine.clear_cache()
```

## 回退机制

如果 FusionScheduler 不可用或融合失败，自动回退到 Graph 缓存：

1. **FusionScheduler 不可用**：`infinicore.fusion` 模块未安装
2. **SubGraph 转换失败**：C++ Graph 未暴露节点信息
3. **融合编译失败**：ninetoothed 编译错误

## 文件结构

```
InfiniLM/
├── python/infinilm/
│   ├── fused_infer_engine.py    # FusedInferEngine
│   └── fusion_utils.py          # 融合工具
└── examples/
    └── jiuge.py                 # --enable-fusion
```
