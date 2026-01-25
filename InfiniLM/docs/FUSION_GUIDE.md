# InfiniLM Graph 缓存优化

基于 InfiniCore Graph Recording 机制，实现类似 CUDA Graph 的推理加速。

## 快速开始

```bash
# 启用 Graph 缓存
python examples/jiuge.py --nvidia --model_path=<path> --enable-fusion

# 禁用（默认）
python examples/jiuge.py --nvidia --model_path=<path>
```

## 原理

```
首次推理（录制）              后续推理（重放）
┌───────────────────┐       ┌───────────────────┐
│ clone 输入为占位   │       │ 更新占位张量       │
│ start_recording() │  →    │ (copy_ 新输入)    │
│ forward(占位)     │ 缓存  │ Graph.run()       │
│ stop_recording()  │  →    │ 返回缓存输出       │
└───────────────────┘       └───────────────────┘
```

**优势**：跳过每次推理的算子调度开销，对于固定 shape 的推理场景可获得加速。

## API

```python
from infinilm import FusedInferEngine

engine = FusedInferEngine(
    model_path="path/to/model",
    enable_fusion=True,       # 启用 Graph 缓存
    warmup_iterations=1,      # 预热迭代次数
    device=infinicore.device("cuda", 0),
)

# 加载权重
load_model_state_dict_by_file(engine, model_path, dtype=engine.config.dtype)

# 推理（首次录制，后续重放）
output = engine.forward(input_ids=tokens, pos=positions)

# 查看统计
print(engine.get_stats())
# {'cache_hits': 99, 'cache_misses': 1, 'recordings': 1, 'cached_shapes': [...]}

# 运行时开关
engine.set_fusion_enabled(False)  # 禁用
engine.set_fusion_enabled(True)   # 启用

# 清空缓存
engine.clear_cache()
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_fusion` | bool | True | 是否启用 Graph 缓存 |
| `warmup_iterations` | int | 1 | 预热次数，首次录制后重放 |

## 注意事项

1. **Shape 敏感**：不同输入 shape 会录制不同的 Graph
2. **内存占用**：缓存会保持张量引用，占用额外内存
3. **动态 shape**：如果每次输入 shape 都不同，缓存可能无效

## 文件结构

```
InfiniLM/
├── python/infinilm/
│   ├── fused_infer_engine.py    # FusedInferEngine 实现
│   └── __init__.py              # 导出
└── examples/
    └── jiuge.py                 # --enable-fusion 参数
```
