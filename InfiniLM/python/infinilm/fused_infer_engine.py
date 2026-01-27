"""
FusedInferEngine - 集成算子融合的推理引擎

融合执行策略：
1. 对每个预定义模式 + 运行时 shape，调用 should_fuse() 判断
2. should_fuse() 读取 profile 数据做决策
3. Graph 缓存作为通用加速机制
"""

from typing import Optional, Dict, Any, List, Tuple
import torch
import hashlib

from infinilm.infer_engine import InferEngine
import infinicore
from infinicore.graph import Graph

# 融合调度器（可选依赖）
try:
    from infinicore.fusion import (
        FusionScheduler,
        FusionConfig,
        SubGraph,
    )
    from infinicore.fusion.patterns.llm_patterns import (
        create_swiglu_pattern,
        create_add_rms_norm_pattern,
    )
    FUSION_AVAILABLE = True
except ImportError:
    FUSION_AVAILABLE = False


class FusedInferEngine(InferEngine):
    """
    带算子融合优化的推理引擎。
    
    工作流程：
    1. 首次遇到新 shape 时，对每个融合模式调用 should_fuse(pattern, shape)
    2. should_fuse() 读取 profile_result.json 比较融合/非融合性能
    3. 缓存每个 shape 的融合决策
    4. Graph 缓存提供通用加速
    
    融合决策是 **per-shape** 的，不是全局固定的。
    """
    
    def __init__(
        self,
        model_path: str = "",
        enable_fusion: bool = True,
        warmup_iterations: int = 1,
        fusion_config: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(model_path, **kwargs)
        
        self._enable_fusion = enable_fusion
        self._warmup_iterations = warmup_iterations
        
        # 初始化 FusionScheduler
        self._fusion_scheduler: Optional[Any] = None
        self._fusion_patterns: List[Dict[str, Any]] = []
        
        if FUSION_AVAILABLE and enable_fusion:
            config = fusion_config or FusionConfig(
                enable_fusion=True,
                fallback_on_error=True,
                debug_mode=False,
            )
            self._fusion_scheduler = FusionScheduler(config)
            
            # 预定义的融合模式
            self._fusion_patterns = [
                {"name": "swiglu", "pattern": create_swiglu_pattern()},
                {"name": "add_rms_norm", "pattern": create_add_rms_norm_pattern()},
            ]
        
        # 融合决策缓存: shape_key -> {pattern_name: should_fuse}
        self._fusion_decision_cache: Dict[str, Dict[str, bool]] = {}
        
        # Graph 缓存
        self._graph_cache: Dict[str, dict] = {}
        
        # 统计信息
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "recordings": 0,
            "fusion_decisions": 0,
        }
    
    def _set_fusion_context(self, decisions: Dict[str, bool]):
        """设置 C++ FusionContext，传递动态融合决策"""
        try:
            from infinilm.lib import _infinilm
            for op_name, should_fuse in decisions.items():
                _infinilm.FusionContext.set(op_name, should_fuse)
        except (ImportError, AttributeError):
            # FusionContext not available, skip
            pass
    
    def _clear_fusion_context(self):
        """清理 C++ FusionContext"""
        try:
            from infinilm.lib import _infinilm
            _infinilm.FusionContext.clear()
        except (ImportError, AttributeError):
            pass
    
    def _get_shape_key(self, input_ids: torch.Tensor, pos: torch.Tensor) -> str:
        key_str = f"{input_ids.shape}_{input_ids.dtype}_{pos.shape}_{pos.dtype}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def _get_fusion_decisions(
        self,
        shape_key: str,
        hidden_size: int = 4096,  # 从模型配置获取，这里用默认值
    ) -> Dict[str, bool]:
        """
        获取指定 shape 的融合决策。
        
        对每个模式，调用 should_fuse(pattern, input_shapes) 判断。
        should_fuse() 会读取 profile 数据做决策。
        
        Returns:
            {"swiglu": True, "add_rms_norm": False, ...}
        """
        if shape_key in self._fusion_decision_cache:
            return self._fusion_decision_cache[shape_key]
        
        if not self._fusion_scheduler:
            return {}
        
        decisions = {}
        
        for p in self._fusion_patterns:
            pattern = p["pattern"]
            name = p["name"]
            
            # 构建该模式的输入 shape
            # 根据模式类型使用合适的 shape
            if name == "swiglu":
                input_shapes = {
                    "gate": (1, 1, hidden_size),
                    "up": (1, 1, hidden_size),
                }
            elif name == "add_rms_norm":
                input_shapes = {
                    "x": (1, 1, hidden_size),
                    "residual": (1, 1, hidden_size),
                    "weight": (hidden_size,),
                }
            else:
                # 默认 shape
                input_shapes = {n: (1, 1, hidden_size) for n in pattern.input_names}
            
            # 调用 should_fuse - 它会读取 profile 数据
            should_fuse = self._fusion_scheduler._heuristics.should_fuse(
                pattern, 
                input_shapes
            )
            
            decisions[name] = should_fuse
            self._stats["fusion_decisions"] += 1
        
        # 缓存决策
        self._fusion_decision_cache[shape_key] = decisions
        
        return decisions
    
    @property
    def fusion_enabled(self) -> bool:
        return self._enable_fusion
    
    @property
    def fusion_scheduler_available(self) -> bool:
        return self._fusion_scheduler is not None
    
    def set_fusion_enabled(self, enabled: bool):
        self._enable_fusion = enabled
    
    def forward(
        self,
        input_ids: torch.Tensor,
        pos: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        if not self._enable_fusion:
            return super().forward(input_ids=input_ids, pos=pos, **kwargs)
        
        return self._forward_with_fusion(input_ids, pos, **kwargs)
    
    def _forward_with_fusion(
        self,
        input_ids: torch.Tensor,
        pos: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """使用融合优化的前向推理"""
        shape_key = self._get_shape_key(input_ids, pos)
        
        # 获取这个 shape 的融合决策（基于 profile）
        fusion_decisions = self._get_fusion_decisions(shape_key)
        
        # 设置 C++ FusionContext（动态融合决策）
        self._set_fusion_context(fusion_decisions)
        
        try:
            # 检查 Graph 缓存
            if shape_key in self._graph_cache:
                cache_entry = self._graph_cache[shape_key]
                
                if cache_entry["iteration_count"] >= self._warmup_iterations:
                    self._stats["cache_hits"] += 1
                    return self._replay_graph(cache_entry, input_ids, pos)
                else:
                    cache_entry["iteration_count"] += 1
                    self._stats["cache_misses"] += 1
                    return super().forward(input_ids=input_ids, pos=pos, **kwargs)
            
            self._stats["cache_misses"] += 1
            return self._record_and_cache(shape_key, input_ids, pos, fusion_decisions, **kwargs)
        finally:
            # 清理 FusionContext
            self._clear_fusion_context()
    
    def _record_and_cache(
        self,
        shape_key: str,
        input_ids: torch.Tensor,
        pos: torch.Tensor,
        fusion_decisions: Dict[str, bool],
        **kwargs
    ) -> torch.Tensor:
        """录制 Graph 并缓存，同时保存融合决策"""
        self._stats["recordings"] += 1
        
        placeholder_input_ids = input_ids.clone()
        placeholder_pos = pos.clone()
        
        infinicore.start_graph_recording()
        
        try:
            output = super().forward(
                input_ids=placeholder_input_ids,
                pos=placeholder_pos,
                **kwargs
            )
            
            graph = infinicore.stop_graph_recording()
            
            if graph is not None:
                self._graph_cache[shape_key] = {
                    "graph": graph,
                    "iteration_count": 1,
                    "placeholder_input_ids": placeholder_input_ids,
                    "placeholder_pos": placeholder_pos,
                    "output": output,
                    "fusion_decisions": fusion_decisions,  # 保存融合决策
                }
            
            return output
            
        except Exception as e:
            try:
                infinicore.stop_graph_recording()
            except:
                pass
            raise e
    
    def _replay_graph(
        self,
        cache_entry: dict,
        input_ids: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """重放缓存的 Graph"""
        cache_entry["placeholder_input_ids"].copy_(input_ids)
        cache_entry["placeholder_pos"].copy_(pos)
        cache_entry["graph"].run()
        return cache_entry["output"]
    
    def get_fusion_decisions(self, shape_key: Optional[str] = None) -> Dict[str, Any]:
        """
        获取融合决策。
        
        Args:
            shape_key: 可选，指定 shape 的决策。None 返回所有缓存的决策。
            
        Returns:
            {"shape_key": {"swiglu": True, "add_rms_norm": False}, ...}
        """
        if shape_key:
            return self._fusion_decision_cache.get(shape_key, {})
        return self._fusion_decision_cache
    
    def clear_cache(self):
        self._graph_cache.clear()
        self._fusion_decision_cache.clear()
        self._stats = {
            "cache_hits": 0, 
            "cache_misses": 0, 
            "recordings": 0,
            "fusion_decisions": 0,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "enabled": self._enable_fusion,
            "fusion_scheduler_available": self.fusion_scheduler_available,
            "patterns_count": len(self._fusion_patterns),
            "cache_size": len(self._graph_cache),
            "decision_cache_size": len(self._fusion_decision_cache),
            **self._stats,
            "fusion_decisions_by_shape": self._fusion_decision_cache,
        }
    
    def __repr__(self) -> str:
        return (
            f"<FusedInferEngine "
            f"fusion={'ON' if self._enable_fusion else 'OFF'} "
            f"patterns={len(self._fusion_patterns)} "
            f"decisions_cached={len(self._fusion_decision_cache)}>"
        )
