FusionScheduler 单元测试操作说明
本手册提供了运行和验证 

FusionScheduler 核心功能的完整步骤。1. 环境准备
在开始测试之前，请确保您的开发环境满足以下条件：
核心依赖
InfiniCore: 确保已经安装或在 PYTHONPATH 中。
ntops: 提供 Triton 算子内核。
ninetoothed (develop-fusion 分支): 提供融合编译能力。
硬件要求
NVIDIA GPU: 部分测试（如基准测试和内核编译测试）需要 CUDA 环境。

2. 运行单元测试
测试文件位于 

InfiniCore/test/infinicore/test_fusion_scheduler.py。建议使用 pytest 进行运行。运行所有测试
进入 InfiniCore 目录并运行：
cd /Users/lxy/lxygit/Infini0120/InfiniCore
python -m pytest test/infinicore/test_fusion_scheduler.py -v
运行特定类别的测试
如果您只想验证特定模块，可以使用 -k 参数：
验证子图哈希和缓存键:
python -m pytest test/infinicore/test_fusion_scheduler.py -k TestSubGraph -v
- **验证启发式规则**:
  ```bash
  python -m pytest test/infinicore/test_fusion_scheduler.py -k TestFusionHeuristics -v
验证调度器核心逻辑:
python -m pytest test/infinicore/test_fusion_scheduler.py -k TestFusionScheduler -v
---
## 3. 测试覆盖内容
| 测试类 | 验证重点 |
| :--- | :--- |
| **TestSubGraph** | 检查 [OpNode](file:///Users/lxy/lxygit/Infini0120/InfiniCore/python/infinicore/fusion/subgraph.py#15-43) 和 [SubGraph](file:///Users/lxy/lxygit/Infini0120/InfiniCore/python/infinicore/fusion/subgraph.py#45-104) 是否可哈希，以及 [cache_key](file:///Users/lxy/lxygit/Infini0120/InfiniCore/python/infinicore/fusion/subgraph.py#74-97) 在不同形状/类型下是否唯一。 |
| **TestFusionConfig** | 验证默认配置和自定义配置的正确性。 |
| **TestFusionHeuristics** | 验证静态启发式规则（如最小张量大小、最小节点数、算子白名单）是否正确触发融合决策。 |
| **TestFusionScheduler** | 验证调度器的创建、缓存统计、缓存清空以及自定义算子注册功能。 |
| **TestLLMPatterns** | 验证预定义的 LLM 融合模式（如 SwiGLU, Add+RMSNorm）结构是否正确。 |
---
## 4. 常见问题排查
### `ModuleNotFoundError: No module named 'ntops'` 或 `'ninetoothed'`
- **原因**: 后端依赖未安装。
- **现象**: 调度器会自动切换到 **Fallback** (回退) 模式。如果设置了 `debug_mode=True`，您会在控制台看到 `[KernelCompiler] ntops not available` 类似的提示。
- **解决方法**: 确保这些库在当前 Python 环境中可用。
### 测试跳过或失败
- **原因**: 缺少 GPU。
- **现象**: 如果测试涉及 CUDA 核编译但环境中没有 GPU，部分集成测试可能会失败。
- **解决方法**: 单元测试的大部分逻辑（如哈希、启发式、回退路径）可以在 CPU 上完成，但完整的融合执行需要 GPU。
---
## 5. 后续计划：集成测试
完成上述单元测试后，建议运行集成测试以验证数值一致性：
```bash
# 验证 SiLU + Mul 融合后的数值准确性
python -m pytest test/infinicore/test_fusion_ntops.py -v
