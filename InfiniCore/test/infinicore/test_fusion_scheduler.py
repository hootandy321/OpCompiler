"""
FusionScheduler 单元测试

测试覆盖：
- SubGraph 和 OpNode 的哈希和缓存键
- FusionConfig 配置
- FusionHeuristics 启发式规则
- FusionScheduler 调度逻辑
"""

import pytest
from typing import Dict, Tuple


class TestSubGraph:
    """SubGraph 和 OpNode 测试"""
    
    def test_opnode_creation(self):
        """测试 OpNode 创建"""
        from infinicore.fusion.subgraph import OpNode
        
        node = OpNode(
            op_type="silu",
            inputs=("x",),
            outputs=("y",),
        )
        
        assert node.op_type == "silu"
        assert node.inputs == ("x",)
        assert node.outputs == ("y",)
        assert node.attrs is None
    
    def test_opnode_hash(self):
        """测试 OpNode 可哈希"""
        from infinicore.fusion.subgraph import OpNode
        
        node1 = OpNode("silu", ("x",), ("y",))
        node2 = OpNode("silu", ("x",), ("y",))
        node3 = OpNode("gelu", ("x",), ("y",))
        
        # 相同内容应该有相同的哈希
        assert hash(node1) == hash(node2)
        # 不同内容应该有不同的哈希
        assert hash(node1) != hash(node3)
        
        # 可以放入集合
        nodes = {node1, node2, node3}
        assert len(nodes) == 2  # node1 和 node2 是相同的
    
    def test_subgraph_creation(self):
        """测试 SubGraph 创建"""
        from infinicore.fusion.subgraph import SubGraph, OpNode
        
        graph = SubGraph(
            nodes=(
                OpNode("silu", ("x",), ("y1",)),
                OpNode("mul", ("y1", "x"), ("y2",)),
            ),
            input_names=("x",),
            output_names=("y2",),
        )
        
        assert len(graph) == 2
        assert graph.input_names == ("x",)
        assert graph.output_names == ("y2",)
    
    def test_subgraph_hash(self):
        """测试 SubGraph 可哈希"""
        from infinicore.fusion.subgraph import SubGraph, OpNode
        
        graph1 = SubGraph(
            nodes=(OpNode("silu", ("x",), ("y",)),),
            input_names=("x",),
            output_names=("y",),
        )
        graph2 = SubGraph(
            nodes=(OpNode("silu", ("x",), ("y",)),),
            input_names=("x",),
            output_names=("y",),
        )
        
        assert hash(graph1) == hash(graph2)
    
    def test_subgraph_cache_key(self):
        """测试 SubGraph cache_key 生成"""
        from infinicore.fusion.subgraph import SubGraph, OpNode
        
        graph = SubGraph(
            nodes=(OpNode("silu", ("x",), ("y",)),),
            input_names=("x",),
            output_names=("y",),
        )
        
        # 相同配置应该生成相同的 cache_key
        key1 = graph.cache_key({"x": "float16"}, {"x": (32, 4096)})
        key2 = graph.cache_key({"x": "float16"}, {"x": (32, 4096)})
        assert key1 == key2
        
        # 不同 dtype 应该生成不同的 cache_key
        key3 = graph.cache_key({"x": "float32"}, {"x": (32, 4096)})
        assert key1 != key3
        
        # 不同 shape 应该生成不同的 cache_key
        key4 = graph.cache_key({"x": "float16"}, {"x": (64, 4096)})
        assert key1 != key4


class TestFusionConfig:
    """FusionConfig 测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        from infinicore.fusion.fusion_config import FusionConfig
        
        config = FusionConfig()
        
        assert config.enable_fusion is True
        assert config.enable_cache is True
        assert config.fallback_on_error is True
        assert config.debug_mode is False
        assert config.min_tensor_elements == 1024
        assert config.min_nodes_for_fusion == 2
    
    def test_custom_config(self):
        """测试自定义配置"""
        from infinicore.fusion.fusion_config import FusionConfig
        
        config = FusionConfig(
            enable_fusion=False,
            min_tensor_elements=2048,
            debug_mode=True,
        )
        
        assert config.enable_fusion is False
        assert config.min_tensor_elements == 2048
        assert config.debug_mode is True


class TestFusionHeuristics:
    """FusionHeuristics 测试"""
    
    def test_should_fuse_disabled(self):
        """测试融合关闭时应该返回 False"""
        from infinicore.fusion.heuristics import FusionHeuristics
        from infinicore.fusion.fusion_config import FusionConfig
        from infinicore.fusion.subgraph import SubGraph, OpNode
        
        config = FusionConfig(enable_fusion=False)
        heuristics = FusionHeuristics(config)
        
        graph = SubGraph(
            nodes=(
                OpNode("silu", ("x",), ("y1",)),
                OpNode("mul", ("y1", "x"), ("y2",)),
            ),
            input_names=("x",),
            output_names=("y2",),
        )
        
        result = heuristics.should_fuse(graph, {"x": (32, 4096)})
        assert result is False
    
    def test_should_fuse_too_few_nodes(self):
        """测试节点数不足时应该返回 False"""
        from infinicore.fusion.heuristics import FusionHeuristics
        from infinicore.fusion.fusion_config import FusionConfig
        from infinicore.fusion.subgraph import SubGraph, OpNode
        
        config = FusionConfig(min_nodes_for_fusion=2)
        heuristics = FusionHeuristics(config)
        
        # 只有一个节点
        graph = SubGraph(
            nodes=(OpNode("silu", ("x",), ("y",)),),
            input_names=("x",),
            output_names=("y",),
        )
        
        result = heuristics.should_fuse(graph, {"x": (32, 4096)})
        assert result is False
    
    def test_should_fuse_small_tensor(self):
        """测试小张量时应该返回 False"""
        from infinicore.fusion.heuristics import FusionHeuristics
        from infinicore.fusion.fusion_config import FusionConfig
        from infinicore.fusion.subgraph import SubGraph, OpNode
        
        config = FusionConfig(min_tensor_elements=1024)
        heuristics = FusionHeuristics(config)
        
        graph = SubGraph(
            nodes=(
                OpNode("silu", ("x",), ("y1",)),
                OpNode("mul", ("y1", "x"), ("y2",)),
            ),
            input_names=("x",),
            output_names=("y2",),
        )
        
        # 元素数 = 8 * 64 = 512 < 1024
        result = heuristics.should_fuse(graph, {"x": (8, 64)})
        assert result is False
    
    def test_should_fuse_unsupported_op(self):
        """测试不支持的算子应该返回 False"""
        from infinicore.fusion.heuristics import FusionHeuristics
        from infinicore.fusion.fusion_config import FusionConfig
        from infinicore.fusion.subgraph import SubGraph, OpNode
        
        config = FusionConfig()
        heuristics = FusionHeuristics(config)
        
        graph = SubGraph(
            nodes=(
                OpNode("silu", ("x",), ("y1",)),
                OpNode("unknown_op", ("y1",), ("y2",)),  # 不支持的算子
            ),
            input_names=("x",),
            output_names=("y2",),
        )
        
        result = heuristics.should_fuse(graph, {"x": (32, 4096)})
        assert result is False
    
    def test_should_fuse_success(self):
        """测试满足所有条件时应该返回 True"""
        from infinicore.fusion.heuristics import FusionHeuristics
        from infinicore.fusion.fusion_config import FusionConfig
        from infinicore.fusion.subgraph import SubGraph, OpNode
        
        config = FusionConfig()
        heuristics = FusionHeuristics(config)
        
        graph = SubGraph(
            nodes=(
                OpNode("silu", ("x",), ("y1",)),
                OpNode("mul", ("y1", "x"), ("y2",)),
            ),
            input_names=("x",),
            output_names=("y2",),
        )
        
        # 元素数 = 32 * 4096 = 131072 > 1024
        result = heuristics.should_fuse(graph, {"x": (32, 4096)})
        assert result is True


class TestFusionScheduler:
    """FusionScheduler 测试"""
    
    def test_scheduler_creation(self):
        """测试调度器创建"""
        from infinicore.fusion.fusion_scheduler import FusionScheduler
        from infinicore.fusion.fusion_config import FusionConfig
        
        config = FusionConfig()
        scheduler = FusionScheduler(config)
        
        assert scheduler.config == config
        assert len(scheduler._kernel_cache) == 0
    
    def test_cache_stats(self):
        """测试缓存统计"""
        from infinicore.fusion.fusion_scheduler import FusionScheduler
        
        scheduler = FusionScheduler()
        stats = scheduler.get_cache_stats()
        
        assert stats["size"] == 0
        assert stats["keys"] == []
    
    def test_clear_cache(self):
        """测试清空缓存"""
        from infinicore.fusion.fusion_scheduler import FusionScheduler
        
        scheduler = FusionScheduler()
        scheduler._kernel_cache["test_key"] = "test_value"
        
        assert len(scheduler._kernel_cache) == 1
        
        scheduler.clear_cache()
        
        assert len(scheduler._kernel_cache) == 0
    
    def test_register_op(self):
        """测试注册自定义算子"""
        from infinicore.fusion.fusion_scheduler import FusionScheduler
        
        scheduler = FusionScheduler()
        
        def custom_op(x):
            return x * 2
        
        scheduler.register_op("custom", custom_op)
        
        assert "custom" in scheduler._op_registry
        assert scheduler._op_registry["custom"] == custom_op


class TestLLMPatterns:
    """LLM 融合模式测试"""
    
    def test_swiglu_pattern(self):
        """测试 SwiGLU 模式"""
        from infinicore.fusion.patterns.llm_patterns import create_swiglu_pattern
        
        pattern = create_swiglu_pattern()
        
        assert len(pattern) == 2
        assert pattern.input_names == ("gate", "up")
        assert pattern.output_names == ("output",)
        
        # 第一个节点是 silu
        assert pattern.nodes[0].op_type == "silu"
        # 第二个节点是 mul
        assert pattern.nodes[1].op_type == "mul"
    
    def test_add_rms_norm_pattern(self):
        """测试 Add+RMSNorm 模式"""
        from infinicore.fusion.patterns.llm_patterns import create_add_rms_norm_pattern
        
        pattern = create_add_rms_norm_pattern()
        
        assert len(pattern) == 2
        assert pattern.input_names == ("x", "residual", "weight")
        assert pattern.output_names == ("output",)
        
        # 第一个节点是 add
        assert pattern.nodes[0].op_type == "add"
        # 第二个节点是 rms_norm
        assert pattern.nodes[1].op_type == "rms_norm"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
