"""
静态启发式规则模块 - V1 版本

决定是否值得对给定子图进行融合编译。
融合并不总是带来性能提升，特别是对于小 Shape 或编译开销较大的场景。
"""

from typing import Any, Dict, Optional, Tuple, Set
import os,json
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
        self.profile_path = ".\profile_result.json"
        self._profile_cache: Optional[Dict[str, Any]] = None  # 缓存profile
        
    def _get_ops(self) -> Set[str]:
        """获取支持的算子集合（带缓存）"""
        if self._supported_ops is None:
            self._supported_ops = _get_supported_ops()
        return self._supported_ops
    
    def _load_profile(self) -> Dict[str, Dict[str, float]]:
        """
        从 JSON 文件加载 profile（不依赖 FusionConfig）

        Returns:
            {
            "single": {op_type: time},
            "fused": {fused_key: time}
            }
        """
        if self.profile_path is None:
            return {"single": {}, "fused": {}}

        if not isinstance(self.profile_path, str):
            raise TypeError(f"profile_path must be str or None, got {type(self.profile_path)}")

        if not os.path.exists(self.profile_path):
            raise FileNotFoundError(f"Profile json not found: {self.profile_path}")

        with open(self.profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # ---- 归一化结构 ----
        if "single" in data or "fused" in data:
            single = data.get("single", {}) or {}
            fused = data.get("fused", {}) or {}
            return {
                "single": {k: float(v) for k, v in single.items()},
                "fused": {k: float(v) for k, v in fused.items()},
            }

        # ---- 兼容扁平结构 ----
        single: Dict[str, float] = {}
        fused: Dict[str, float] = {}
        for k, v in data.items():
            if not isinstance(v, (int, float)):
                continue
            if k.startswith("single:"):
                single[k[len("single:"):]] = float(v)
            elif k.startswith("fused:"):
                fused[k[len("fused:"):]] = float(v)

        return {"single": single, "fused": fused}

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
        
        # --------------------  基于 profile 决策是否融合 --------------------

        profile = self._load_profile()
        single_t = profile.get("single", {}) or {} # 单个算子独立 kernel 的执行时间
        fused_t = profile.get("fused", {}) or {}
        op_types = [n.op_type for n in graph.nodes]

        # 计算单算子总时间
        missing_single = [op for op in op_types if op not in single_t]
        if missing_single:
            print(f"[Fusion] Profile missing single timings for {missing_single}")
            return True
        
        # 多个单算子 kernel 时间之和
        separate_time = sum(float(single_t[op]) for op in op_types) 

        # 兼容旧 profile：op 串联 key
        fused_key_ops = "+".join(op_types)

        # 依次尝试查融合时间
        fused_time = None
    
        if fused_key_ops in fused_t:
            fused_time = fused_t[fused_key_ops]
            fused_key_used = fused_key_ops
        else:
            print(f"[Fusion] Profile data missing fused timing for key=''{fused_key_ops}'")
            return True


        fused_time = float(fused_time) # 整个子图融合后，一个 kernel 的执行时间

        margin = float(getattr(self.config, "profile_margin", 0.0))
        decision = separate_time > fused_time * (1.0 + margin)

        if self.config.debug_mode:
            print(
                f"[Fusion] Profile decision: separate={separate_time:.6f} ms, "
                f"fused={fused_time:.6f} ms, margin={margin:.3f} => "
                f"{'FUSE' if decision else 'NO_FUSE'} (key='{fused_key_used}')"
            )

        return decision
       


    
    def get_supported_ops(self) -> Set[str]:
        """返回当前支持融合的算子类型集合"""
        return self._get_ops().copy()

