"""
静态启发式规则模块 - V1 版本

决定是否值得对给定子图进行融合编译。
融合并不总是带来性能提升，特别是对于小 Shape 或编译开销较大的场景。
"""

from typing import Dict, Tuple, Set

from infinicore.fusion.subgraph import SubGraph
from infinicore.fusion.fusion_config import FusionConfig


def _get_supported_ops() -> Set[str]:
    """获取支持融合的算子集合，与 kernel_compiler 保持同步"""
    # 静态 fallback 列表
    fallback_ops = {
        "silu", "gelu", "relu", "sigmoid",
        "add", "mul", "sub", "div",
        "rms_norm", "layer_norm",
    }
    
    try:
        from infinicore.fusion.kernel_compiler import get_supported_fusion_ops
        ops = get_supported_fusion_ops()
        # 如果 kernel_compiler 返回空集（ntops 不可用），使用 fallback
        return ops if ops else fallback_ops
    except ImportError:
        return fallback_ops


# V1 支持融合的算子类型（延迟初始化）
SUPPORTED_OPS: Set[str] = set()


class FusionHeuristics:
    """
    静态启发式规则 - 决定是否值得融合
    
    V1 实现基于简单规则过滤：
    1. 节点数检查
    2. 张量大小检查
    3. 算子类型检查
    
    后续 V2 版本将添加自动采样机制进行实际性能对比。
    """
    
    def __init__(self, config: FusionConfig):
        self.config = config
        self._supported_ops = None  # 延迟初始化
    
    def _get_ops(self) -> Set[str]:
        """获取支持的算子集合（带缓存）"""
        if self._supported_ops is None:
            self._supported_ops = _get_supported_ops()
        return self._supported_ops
    
    def should_fuse(
        self,
        graph: SubGraph,
        input_shapes: Dict[str, Tuple[int, ...]]
    ) -> bool:
        """
        判断是否应该尝试融合。
        
        Args:
            graph: 待融合的子图
            input_shapes: 输入张量的形状字典
            
        Returns:
            True 如果应该尝试融合，False 则回退到标准执行
        """
        # 规则 0: 总开关检查
        if not self.config.enable_fusion:
            return False
        
        # 规则 1: 节点数检查
        if len(graph.nodes) < self.config.min_nodes_for_fusion:
            if self.config.debug_mode:
                print(f"[Fusion] Skip: node count {len(graph.nodes)} < {self.config.min_nodes_for_fusion}")
            return False
        
        # 规则 2: 图大小上限检查
        if len(graph.nodes) > self.config.max_graph_size:
            if self.config.debug_mode:
                print(f"[Fusion] Skip: node count {len(graph.nodes)} > max {self.config.max_graph_size}")
            return False
        
        # 规则 3: 张量大小检查
        for name, shape in input_shapes.items():
            num_elements = 1
            for dim in shape:
                num_elements *= dim
            if num_elements < self.config.min_tensor_elements:
                if self.config.debug_mode:
                    print(f"[Fusion] Skip: tensor '{name}' elements {num_elements} < {self.config.min_tensor_elements}")
                return False
        
        # 规则 4: 算子类型检查
        supported = self._get_ops()
        for node in graph.nodes:
            if node.op_type not in supported:
                if self.config.debug_mode:
                    print(f"[Fusion] Skip: unsupported op '{node.op_type}'")
                return False
        
        if self.config.debug_mode:
            print(f"[Fusion] Accept: graph with {len(graph.nodes)} nodes")
        
        return True
    
    def get_supported_ops(self) -> Set[str]:
        """返回当前支持融合的算子类型集合"""
        return self._get_ops().copy()

