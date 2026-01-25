"""
FusedInferEngine - 集成算子融合的推理引擎

利用 InfiniCore 的 Graph Recording 机制：
1. 首次推理时录制算子调用序列
2. 缓存录制的 Graph
3. 后续推理直接 Graph.run() 重放（跳过开销）
"""

from typing import Optional, Dict, Any, Tuple
import torch
import hashlib

from infinilm.infer_engine import InferEngine
import infinicore
from infinicore.graph import Graph


class FusedInferEngine(InferEngine):
    """
    带 Graph 缓存优化的推理引擎。
    
    工作原理：
    1. 预热阶段：录制算子调用 → 生成 Graph → 缓存
    2. 执行阶段：直接 Graph.run() 重放
    
    这避免了每次推理的算子调度开销，类似 CUDA Graph 的效果。
    
    Usage:
        engine = FusedInferEngine(enable_fusion=True)
        engine.load("path/to/model")
        
        # 第一次调用（录制）
        outputs = engine.forward(input_ids=tokens, pos=positions)
        
        # 后续调用（重放缓存的 Graph）
        outputs = engine.forward(input_ids=tokens2, pos=positions2)
    """
    
    def __init__(
        self,
        model_path: str = "",
        enable_fusion: bool = True,
        warmup_iterations: int = 1,
        **kwargs
    ):
        """
        初始化 FusedInferEngine。
        
        Args:
            model_path: 模型路径
            enable_fusion: 是否启用 Graph 缓存优化
            warmup_iterations: 预热迭代次数（用于录制）
            **kwargs: 传递给父类 InferEngine 的参数
        """
        super().__init__(model_path, **kwargs)
        
        self._enable_fusion = enable_fusion
        self._warmup_iterations = warmup_iterations
        
        # Graph 缓存: shape_key -> (Graph, iteration_count)
        self._graph_cache: Dict[str, Tuple[Graph, int]] = {}
        
        # 统计信息
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "recordings": 0,
        }
    
    @property
    def fusion_enabled(self) -> bool:
        """融合是否启用"""
        return self._enable_fusion
    
    def set_fusion_enabled(self, enabled: bool):
        """运行时开关融合"""
        self._enable_fusion = enabled
    
    def _get_shape_key(self, input_ids: torch.Tensor, pos: torch.Tensor) -> str:
        """
        根据输入 shape 生成缓存 key。
        
        Graph 只能在相同 shape 下重放，因此用 shape 作为缓存 key。
        """
        key_str = f"{input_ids.shape}_{input_ids.dtype}_{pos.shape}_{pos.dtype}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def forward(
        self,
        input_ids: torch.Tensor,
        pos: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        执行前向推理，可选地使用 Graph 缓存优化。
        
        Args:
            input_ids: 输入 token IDs
            pos: 位置信息
            **kwargs: 其他参数
            
        Returns:
            推理输出张量
        """
        if not self._enable_fusion:
            # 融合禁用，直接走原生路径
            return super().forward(input_ids=input_ids, pos=pos, **kwargs)
        
        return self._forward_with_graph_cache(input_ids, pos, **kwargs)
    
    def _record_and_execute(
        self,
        shape_key: str,
        input_ids: torch.Tensor,
        pos: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        录制算子调用并执行。
        
        录制模式（参考 InfiniCore 测试）：
        1. 创建占位输入张量（录制时绑定）
        2. 录制期间执行 forward，算子被捕获
        3. 停止录制，获取 Graph
        4. 将真实输入 copy 到占位张量
        5. Graph.run() 执行
        6. 缓存 (Graph, 占位输入, 输出) 供后续重放
        """
        self._stats["recordings"] += 1
        
        # 创建占位输入张量（与原始张量相同 shape/dtype）
        placeholder_input_ids = input_ids.clone()
        placeholder_pos = pos.clone()
        
        # 开始录制
        infinicore.start_graph_recording()
        
        try:
            # 使用占位张量执行推理，算子调用被录制
            output = super().forward(
                input_ids=placeholder_input_ids, 
                pos=placeholder_pos, 
                **kwargs
            )
            
            # 停止录制，获取 Graph
            graph = infinicore.stop_graph_recording()
            
            if graph is not None:
                # 缓存 Graph 和相关张量引用
                self._graph_cache[shape_key] = {
                    "graph": graph,
                    "iteration_count": 1,
                    "placeholder_input_ids": placeholder_input_ids,
                    "placeholder_pos": placeholder_pos,
                    "output": output,  # 输出张量引用，Graph.run() 会更新它
                }
            
            return output
            
        except Exception as e:
            # 确保异常时停止录制
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
        **kwargs
    ) -> torch.Tensor:
        """
        重放缓存的 Graph。
        
        重放模式：
        1. 将新输入 copy 到录制时的占位张量
        2. Graph.run() 执行（使用更新后的输入）
        3. 返回录制时创建的输出张量（已被 Graph 更新）
        """
        # 1. 更新占位输入为当前输入
        cache_entry["placeholder_input_ids"].copy_(input_ids)
        cache_entry["placeholder_pos"].copy_(pos)
        
        # 2. 重放 Graph
        cache_entry["graph"].run()
        
        # 3. 返回输出（Graph 已更新此张量）
        return cache_entry["output"]
    
    def _forward_with_graph_cache(
        self,
        input_ids: torch.Tensor,
        pos: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        使用 Graph 缓存的前向推理。
        """
        shape_key = self._get_shape_key(input_ids, pos)
        
        if shape_key in self._graph_cache:
            cache_entry = self._graph_cache[shape_key]
            
            if cache_entry["iteration_count"] >= self._warmup_iterations:
                # 预热完成，使用 Graph 重放
                self._stats["cache_hits"] += 1
                return self._replay_graph(cache_entry, input_ids, pos, **kwargs)
            else:
                # 仍在预热中
                cache_entry["iteration_count"] += 1
                self._stats["cache_misses"] += 1
                return super().forward(input_ids=input_ids, pos=pos, **kwargs)
        else:
            self._stats["cache_misses"] += 1
        
        # 录制新 Graph
        return self._record_and_execute(shape_key, input_ids, pos, **kwargs)
    
    def clear_cache(self):
        """清空 Graph 缓存"""
        self._graph_cache.clear()
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "recordings": 0,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "enabled": self._enable_fusion,
            "cache_size": len(self._graph_cache),
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["cache_misses"],
            "recordings": self._stats["recordings"],
            "cached_shapes": list(self._graph_cache.keys()),
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"<FusedInferEngine "
            f"fusion={'ON' if self._enable_fusion else 'OFF'} "
            f"cache_size={stats['cache_size']} "
            f"hits={stats['cache_hits']} misses={stats['cache_misses']}>"
        )
