# InfiniCore 算子融合判断逻辑分析

## 概述

InfiniCore 的算子融合系统位于 `InfiniCore/python/infinicore/fusion/` 目录，核心目标是在运行时决定一组算子（SubGraph）是否应该融合为单个内核执行，还是逐个调用标准算子（回退路径）。

融合判断的核心入口是 `FusionHeuristics.should_fuse()` 方法（`heuristics.py:104`），由 `FusionScheduler.dispatch()` 在每次调度时调用。

---

## 判断流程（共两阶段）

### 阶段一：静态规则过滤

按顺序检查以下规则，任一不满足则立即返回 `False`（不融合）：

| 规则 | 检查内容 | 配置参数 | 默认值 |
|------|----------|----------|--------|
| **规则0：总开关** | `FusionConfig.enable_fusion` 是否为 True | `enable_fusion` | `True` |
| **规则1：最小节点数** | 子图节点数 >= `min_nodes_for_fusion` | `min_nodes_for_fusion` | `2` |
| **规则2：最大图大小** | 子图节点数 <= `max_graph_size` | `max_graph_size` | `10` |
| **规则3：最小张量元素数** | 每个输入张量的元素总数 >= `min_tensor_elements` | `min_tensor_elements` | `1024` |
| **规则4：算子类型支持** | 子图中所有算子类型都在支持集合内 | - | 见下方 |

**支持的算子类型**（V1 版本）：
- 激活函数：`silu`, `gelu`, `relu`, `sigmoid`
- 二元逐元素：`add`, `mul`, `sub`, `div`
- 归一化：`rms_norm`, `layer_norm`

算子支持列表优先从 `kernel_compiler.get_supported_fusion_ops()` 动态获取（依赖 ntops 库是否安装），不可用时回退到硬编码的 fallback 列表。

### 阶段二：基于 Profile 数据的性能决策

当静态规则全部通过后，进入基于实际性能数据的决策阶段：

1. **加载 Profile 数据**：从 `profile_result.json` 文件读取预先采集的性能数据，包含：
   - `single`：每个算子单独执行的耗时（毫秒）
   - `fused`：融合后整体执行的耗时（毫秒）

2. **计算分离执行总耗时**：将子图中所有算子的单独执行时间求和：
   ```
   separate_time = sum(single[op] for op in graph.op_types)
   ```

3. **查找融合执行耗时**：依次尝试两种 key 查找融合时间：
   - 精确 key：`graph.cache_key(input_dtypes, input_shapes)`（包含图结构+dtype+shape 的 SHA256 哈希）
   - 兼容 key：算子类型用 `+` 串联，如 `"silu+mul"`

4. **做出决策**：
   ```
   decision = separate_time > fused_time * (1.0 + margin)
   ```
   即：只有当分离执行的总耗时 **显著大于** 融合执行耗时（考虑 margin 裕度）时，才选择融合。

5. **缺失数据的回退策略**：如果 profile 文件中缺少某个算子的单独时间或融合时间，则根据 `config.profile_fallback` 决定行为：
   - `"fuse"`（默认）：缺失数据时倾向于融合
   - `"no_fuse"`：缺失数据时不融合

---

## 调度器完整执行路径

`FusionScheduler.dispatch()`（`fusion_scheduler.py:108`）的完整流程：

```
输入: SubGraph + 输入张量字典
  │
  ├─ 提取 input_shapes, input_dtypes
  │
  ├─ 调用 heuristics.should_fuse(graph, input_shapes)
  │   ├─ False → 回退执行（逐算子调用）
  │   └─ True ↓
  │
  ├─ 检查内核缓存（cache_key = SHA256(图结构+dtype+shape)[:16]）
  │   ├─ 命中 → 直接执行缓存的融合内核
  │   └─ 未命中 ↓
  │
  ├─ 调用 KernelCompiler.compile() 编译融合内核
  │   ├─ 成功 → 缓存并执行融合内核
  │   └─ 失败（FusionError）
  │       ├─ fallback_on_error=True → 回退执行
  │       └─ fallback_on_error=False → 抛出异常
  │
  └─ 返回输出张量字典
```

---

## 预定义的 LLM 融合模式

系统预定义了常见的 LLM 融合模式（`patterns/llm_patterns.py`）：

| 模式 | 算子序列 | 应用场景 |
|------|----------|----------|
| **SwiGLU** | `silu(gate)` → `mul(gate_activated, up)` | LLaMA/Mistral FFN 层 |
| **Add+RMSNorm** | `add(x, residual)` → `rms_norm(sum, weight)` | Transformer 残差连接后处理 |
| **GELU** | `gelu(x)` | 单算子（测试用） |

---

## 模式匹配机制

`graph_converter.py` 还提供了模式匹配能力：

- `match_fusion_pattern(graph, pattern)`：检查子图的算子类型序列是否与模式模板一致（按顺序逐节点比对 `op_type`）
- `find_fusable_subgraphs(graph, patterns)`：在完整计算图中滑动窗口查找所有匹配的可融合子图片段

---

## 总结

融合决策本质上是一个**两级过滤器**：

1. **静态规则**：快速排除不适合融合的场景（图太小/太大、张量太小、含不支持算子）
2. **Profile 驱动**：基于实测性能数据精确判断融合是否有收益，只在 `separate_time > fused_time * (1 + margin)` 时才融合

这种设计确保了融合决策既高效（静态规则 O(1) 排除大量无意义场景），又准确（profile 数据反映真实硬件表现）。
