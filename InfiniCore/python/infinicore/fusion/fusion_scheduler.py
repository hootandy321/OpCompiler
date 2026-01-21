"""
融合调度器模块 - 运行时调度核心

接收子图描述，根据配置动态决定执行路径：
1. 融合路径：调用 ninetoothed 编译的融合内核
2. 回退路径：逐个调用标准 InfiniCore 算子
"""

from typing import Dict, Tuple, Optional, Any
import functools

from infinicore.fusion.subgraph import SubGraph, OpNode
from infinicore.fusion.fusion_config import FusionConfig
from infinicore.fusion.heuristics import FusionHeuristics
from infinicore.fusion.kernel_compiler import KernelCompiler, CompiledKernel, FusionError


class FusionScheduler:
    """
    运行时融合调度器
    
    核心职责：
    1. 接收子图和输入张量
    2. 根据启发式规则决定是否融合
    3. 管理编译后内核的缓存
    4. 提供回退到标准执行的能力
    
    Example:
        >>> config = FusionConfig(enable_fusion=True, debug_mode=True)
        >>> scheduler = FusionScheduler(config)
        >>> 
        >>> graph = SubGraph(
        ...     nodes=(OpNode("silu", ("x",), ("y1",)), OpNode("mul", ("y1", "x"), ("y2",))),
        ...     input_names=("x",),
        ...     output_names=("y2",),
        ... )
        >>> 
        >>> outputs = scheduler.dispatch(graph, {"x": tensor_x})
    """
    
    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self._kernel_cache: Dict[str, CompiledKernel] = {}
        self._heuristics = FusionHeuristics(self.config)
        self._compiler = KernelCompiler(self.config)
        self._op_registry: Dict[str, callable] = {}
        self._init_op_registry()
    
    def _init_op_registry(self):
        """初始化算子注册表（用于回退执行）"""
        try:
            import infinicore
            import infinicore.nn.functional as F
            
            self._op_registry = {
                "silu": F.silu,
                "gelu": F.gelu,
                "relu": F.relu,
                "add": infinicore.add,
                "mul": infinicore.mul,
                # TODO: 添加更多算子
            }
            
            # 尝试添加 rms_norm
            if hasattr(F, 'rms_norm'):
                self._op_registry["rms_norm"] = F.rms_norm
                
        except ImportError:
            if self.config.debug_mode:
                print("[FusionScheduler] infinicore not fully available for fallback")
    
    def dispatch(
        self,
        graph: SubGraph,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        调度子图执行。
        
        Args:
            graph: 子图描述（算子序列 + 数据依赖）
            inputs: 输入张量字典，键为张量名，值为张量对象
            
        Returns:
            outputs: 输出张量字典
        """
        # 提取输入形状和类型信息
        input_shapes = self._get_input_shapes(inputs)
        input_dtypes = self._get_input_dtypes(inputs)
        
        # 检查是否应该尝试融合
        if not self._heuristics.should_fuse(graph, input_shapes):
            return self._fallback_execute(graph, inputs)
        
        # 检查缓存
        cache_key = graph.cache_key(input_dtypes, input_shapes)
        
        if self.config.enable_cache and cache_key in self._kernel_cache:
            if self.config.debug_mode:
                print(f"[FusionScheduler] Cache hit: {cache_key}")
            compiled_kernel = self._kernel_cache[cache_key]
            return self._execute_fused(compiled_kernel, inputs, graph)
        
        # 尝试编译融合内核
        try:
            compiled_kernel = self._compiler.compile(graph, input_dtypes, input_shapes)
            
            if self.config.enable_cache:
                self._kernel_cache[cache_key] = compiled_kernel
                
            if self.config.debug_mode:
                print(f"[FusionScheduler] Compilation success: {cache_key}")
                
            return self._execute_fused(compiled_kernel, inputs, graph)
            
        except FusionError as e:
            if self.config.debug_mode:
                print(f"[FusionScheduler] Fusion failed: {e}")
                
            if self.config.fallback_on_error:
                return self._fallback_execute(graph, inputs)
            else:
                raise
    
    def _execute_fused(
        self,
        compiled_kernel: CompiledKernel,
        inputs: Dict[str, Any],
        graph: SubGraph
    ) -> Dict[str, Any]:
        """执行融合内核"""
        # 按照 graph.input_names 的顺序提取输入
        ordered_inputs = [inputs[name] for name in graph.input_names]
        
        # 调用融合内核
        result = compiled_kernel(*ordered_inputs)
        
        # 包装输出
        if len(graph.output_names) == 1:
            return {graph.output_names[0]: result}
        else:
            # 多输出情况
            return dict(zip(graph.output_names, result))
    
    def _fallback_execute(
        self,
        graph: SubGraph,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        回退执行：逐个调用标准算子
        
        按拓扑顺序执行每个节点，中间结果存储在 values 字典中。
        """
        if self.config.debug_mode:
            print(f"[FusionScheduler] Fallback execution for graph with {len(graph.nodes)} nodes")
        
        # 初始化值字典
        values: Dict[str, Any] = dict(inputs)
        
        # 按拓扑顺序执行
        for node in graph.nodes:
            op_func = self._op_registry.get(node.op_type)
            
            if op_func is None:
                raise RuntimeError(f"Operator '{node.op_type}' not registered for fallback")
            
            # 收集输入
            node_inputs = [values[name] for name in node.inputs]
            
            # 解析属性
            kwargs = {}
            if node.attrs:
                kwargs = dict(node.attrs)
            
            # 执行算子
            result = op_func(*node_inputs, **kwargs)
            
            # 存储输出
            if len(node.outputs) == 1:
                values[node.outputs[0]] = result
            else:
                for i, out_name in enumerate(node.outputs):
                    values[out_name] = result[i]
        
        # 返回最终输出
        return {name: values[name] for name in graph.output_names}
    
    def _get_input_shapes(self, inputs: Dict[str, Any]) -> Dict[str, Tuple[int, ...]]:
        """提取输入张量的形状"""
        shapes = {}
        for name, tensor in inputs.items():
            if hasattr(tensor, 'shape'):
                shapes[name] = tuple(tensor.shape)
            else:
                shapes[name] = ()
        return shapes
    
    def _get_input_dtypes(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """提取输入张量的数据类型"""
        dtypes = {}
        for name, tensor in inputs.items():
            if hasattr(tensor, 'dtype'):
                dtypes[name] = str(tensor.dtype)
            else:
                dtypes[name] = "unknown"
        return dtypes
    
    def clear_cache(self):
        """清空内核缓存"""
        self._kernel_cache.clear()
        if self.config.debug_mode:
            print("[FusionScheduler] Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            "size": len(self._kernel_cache),
            "keys": list(self._kernel_cache.keys()),
        }
    
    def register_op(self, op_type: str, op_func: callable):
        """注册自定义算子用于回退执行"""
        self._op_registry[op_type] = op_func
