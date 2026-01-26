"""
FusedInferEngine - 集成算子融合的推理引擎

融合执行策略：
1. 首次推理：录制算子 → 尝试转换为 SubGraph → FusionScheduler 决策
2. 如果 FusionScheduler 可用且融合成功 → 使用融合内核
3. 如果融合失败 → 回退到 Graph 缓存重放
"""

from typing import Optional, Dict, Any, Tuple
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
        convert_graph_to_subgraph,
        SubGraph,
    )
    FUSION_AVAILABLE = True
except ImportError:
    FUSION_AVAILABLE = False


class FusedInferEngine(InferEngine):
    """
    带算子融合优化的推理引擎。
    
    工作流程：
    1. 首次推理时录制算子调用序列
    2. 尝试转换为 SubGraph 并交给 FusionScheduler 决策
    3. FusionScheduler 决定：
       - 融合执行：编译并执行融合内核
       - 回退执行：使用标准算子或 Graph 重放
    4. 后续相同 shape 推理：使用缓存的执行路径
    
    Usage:
        engine = FusedInferEngine(enable_fusion=True)
        engine.load("path/to/model")
        outputs = engine.forward(input_ids=tokens, pos=positions)
    """
    
    def __init__(
        self,
        model_path: str = "",
        enable_fusion: bool = True,
        warmup_iterations: int = 1,
        fusion_config: Optional[Any] = None,
        **kwargs
    ):
        """
        初始化 FusedInferEngine。
        
        Args:
            model_path: 模型路径
            enable_fusion: 是否启用融合优化
            warmup_iterations: 预热迭代次数
            fusion_config: FusionConfig 配置（可选）
            **kwargs: 传递给父类 InferEngine 的参数
        """
        super().__init__(model_path, **kwargs)
        
        self._enable_fusion = enable_fusion
        self._warmup_iterations = warmup_iterations
        
        # 初始化 FusionScheduler（如果可用）
        self._fusion_scheduler: Optional[Any] = None
        if FUSION_AVAILABLE and enable_fusion:
            config = fusion_config or FusionConfig(
                enable_fusion=True,
                fallback_on_error=True,  # 融合失败时回退
                debug_mode=False,
            )
            self._fusion_scheduler = FusionScheduler(config)
        
        # Graph 缓存: shape_key -> cache_entry
        self._graph_cache: Dict[str, dict] = {}
        
        # 统计信息
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "recordings": 0,
            "fusion_attempts": 0,
            "fusion_successes": 0,
            "fusion_fallbacks": 0,
        }
    
    @property
    def fusion_enabled(self) -> bool:
        """融合是否启用"""
        return self._enable_fusion
    
    @property
    def fusion_scheduler_available(self) -> bool:
        """FusionScheduler 是否可用"""
        return self._fusion_scheduler is not None
    
    def set_fusion_enabled(self, enabled: bool):
        """运行时开关融合"""
        self._enable_fusion = enabled
    
    def _get_shape_key(self, input_ids: torch.Tensor, pos: torch.Tensor) -> str:
        """根据输入 shape 生成缓存 key"""
        key_str = f"{input_ids.shape}_{input_ids.dtype}_{pos.shape}_{pos.dtype}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def forward(
        self,
        input_ids: torch.Tensor,
        pos: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        执行前向推理，可选地使用融合优化。
        """
        if not self._enable_fusion:
            return super().forward(input_ids=input_ids, pos=pos, **kwargs)
        
        return self._forward_with_fusion(input_ids, pos, **kwargs)
    
    def _forward_with_fusion(
        self,
        input_ids: torch.Tensor,
        pos: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        使用融合优化的前向推理。
        
        策略：
        1. 检查缓存
        2. 如果有缓存 → 重放（Graph 或融合内核）
        3. 如果无缓存 → 录制 → 尝试融合 → 缓存结果
        """
        shape_key = self._get_shape_key(input_ids, pos)
        
        # 检查缓存
        if shape_key in self._graph_cache:
            cache_entry = self._graph_cache[shape_key]
            
            if cache_entry["iteration_count"] >= self._warmup_iterations:
                self._stats["cache_hits"] += 1
                return self._execute_cached(cache_entry, input_ids, pos, **kwargs)
            else:
                cache_entry["iteration_count"] += 1
                self._stats["cache_misses"] += 1
                return super().forward(input_ids=input_ids, pos=pos, **kwargs)
        
        self._stats["cache_misses"] += 1
        return self._record_and_optimize(shape_key, input_ids, pos, **kwargs)
    
    def _record_and_optimize(
        self,
        shape_key: str,
        input_ids: torch.Tensor,
        pos: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        录制算子调用并尝试优化。
        
        流程：
        1. 创建占位张量
        2. 录制 forward
        3. 尝试转换为 SubGraph 并融合
        4. 如果融合失败，使用 Graph 缓存
        """
        self._stats["recordings"] += 1
        
        # 创建占位输入张量
        placeholder_input_ids = input_ids.clone()
        placeholder_pos = pos.clone()
        
        # 开始录制
        infinicore.start_graph_recording()
        
        try:
            # 使用占位张量执行推理
            output = super().forward(
                input_ids=placeholder_input_ids,
                pos=placeholder_pos,
                **kwargs
            )
            
            # 停止录制
            graph = infinicore.stop_graph_recording()
            
            if graph is None:
                return output
            
            # 尝试融合优化
            cache_entry = self._try_optimize(
                graph=graph,
                placeholder_input_ids=placeholder_input_ids,
                placeholder_pos=placeholder_pos,
                output=output,
            )
            
            # 缓存结果
            self._graph_cache[shape_key] = cache_entry
            
            return output
            
        except Exception as e:
            try:
                infinicore.stop_graph_recording()
            except:
                pass
            raise e
    
    def _try_optimize(
        self,
        graph: Graph,
        placeholder_input_ids: torch.Tensor,
        placeholder_pos: torch.Tensor,
        output: torch.Tensor,
    ) -> dict:
        """
        尝试对录制的 Graph 进行融合优化。
        
        返回缓存条目，包含执行模式（融合/Graph重放）和相关数据。
        """
        cache_entry = {
            "iteration_count": 1,
            "mode": "graph_replay",  # 默认模式：Graph 重放
            "graph": graph,
            "placeholder_input_ids": placeholder_input_ids,
            "placeholder_pos": placeholder_pos,
            "output": output,
            "subgraph": None,
            "fusion_inputs": None,
        }
        
        # 如果 FusionScheduler 不可用，直接返回 Graph 重放模式
        if not self._fusion_scheduler:
            return cache_entry
        
        # 尝试转换为 SubGraph
        self._stats["fusion_attempts"] += 1
        
        if not FUSION_AVAILABLE:
            self._stats["fusion_fallbacks"] += 1
            return cache_entry
        
        subgraph = convert_graph_to_subgraph(graph)
        
        if subgraph is None:
            # 转换失败，回退到 Graph 重放
            self._stats["fusion_fallbacks"] += 1
            return cache_entry
        
        # SubGraph 转换成功，标记为融合模式
        cache_entry["mode"] = "fusion"
        cache_entry["subgraph"] = subgraph
        cache_entry["fusion_inputs"] = self._build_fusion_inputs(
            placeholder_input_ids, placeholder_pos
        )
        self._stats["fusion_successes"] += 1
        
        return cache_entry
    
    def _build_fusion_inputs(
        self,
        input_ids: torch.Tensor,
        pos: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        构建 FusionScheduler.dispatch() 需要的输入字典。
        
        注意：这里的命名需要与 SubGraph 中的 input_names 匹配。
        当前使用简单的命名约定，实际使用时可能需要调整。
        """
        return {
            "input_ids": input_ids,
            "pos": pos,
        }
    
    def _execute_cached(
        self,
        cache_entry: dict,
        input_ids: torch.Tensor,
        pos: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        执行缓存的优化路径。
        
        根据缓存模式选择：
        - fusion: 使用 FusionScheduler 执行（融合内核或回退）
        - graph_replay: 直接 Graph.run() 重放
        """
        mode = cache_entry.get("mode", "graph_replay")
        
        if mode == "fusion" and self._fusion_scheduler:
            return self._execute_fusion(cache_entry, input_ids, pos, **kwargs)
        else:
            return self._execute_graph_replay(cache_entry, input_ids, pos, **kwargs)
    
    def _execute_fusion(
        self,
        cache_entry: dict,
        input_ids: torch.Tensor,
        pos: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        使用 FusionScheduler 执行。
        
        FusionScheduler 会根据启发式规则决定：
        - 使用融合内核
        - 或回退到标准算子执行
        """
        subgraph = cache_entry["subgraph"]
        
        # 更新占位张量
        cache_entry["placeholder_input_ids"].copy_(input_ids)
        cache_entry["placeholder_pos"].copy_(pos)
        
        # 构建输入字典
        inputs = {
            "input_ids": cache_entry["placeholder_input_ids"],
            "pos": cache_entry["placeholder_pos"],
        }
        
        try:
            # 调用 FusionScheduler
            outputs = self._fusion_scheduler.dispatch(subgraph, inputs)
            
            # 取最后一个输出作为结果
            if outputs:
                output_name = list(outputs.keys())[-1]
                return outputs[output_name]
            else:
                # FusionScheduler 没有返回输出，回退到 Graph 重放
                return self._execute_graph_replay(cache_entry, input_ids, pos, **kwargs)
                
        except Exception as e:
            # 融合执行失败，回退到 Graph 重放
            return self._execute_graph_replay(cache_entry, input_ids, pos, **kwargs)
    
    def _execute_graph_replay(
        self,
        cache_entry: dict,
        input_ids: torch.Tensor,
        pos: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        重放缓存的 Graph（回退路径）。
        """
        # 更新占位输入
        cache_entry["placeholder_input_ids"].copy_(input_ids)
        cache_entry["placeholder_pos"].copy_(pos)
        
        # 重放 Graph
        cache_entry["graph"].run()
        
        # 返回输出
        return cache_entry["output"]
    
    def clear_cache(self):
        """清空所有缓存"""
        self._graph_cache.clear()
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "recordings": 0,
            "fusion_attempts": 0,
            "fusion_successes": 0,
            "fusion_fallbacks": 0,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "enabled": self._enable_fusion,
            "fusion_scheduler_available": self.fusion_scheduler_available,
            "cache_size": len(self._graph_cache),
            **self._stats,
            "cached_shapes": list(self._graph_cache.keys()),
            "fusion_modes": {
                k: v.get("mode", "unknown") 
                for k, v in self._graph_cache.items()
            },
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"<FusedInferEngine "
            f"fusion={'ON' if self._enable_fusion else 'OFF'} "
            f"scheduler={'YES' if self.fusion_scheduler_available else 'NO'} "
            f"cache={stats['cache_size']} "
            f"fusion_ok={stats['fusion_successes']} "
            f"fallback={stats['fusion_fallbacks']}>"
        )
