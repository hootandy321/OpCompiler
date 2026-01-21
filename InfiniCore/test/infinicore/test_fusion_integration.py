"""
GPU 集成测试 - 验证 ninetoothed 融合内核的数值正确性

测试覆盖：
- SwiGLU (silu + mul) 融合模式
- KernelCompiler 直接编译测试
- 回退路径验证
"""

import pytest
from typing import Dict, Tuple


# 尝试导入 CUDA 依赖
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False


class TestKernelCompilerIntegration:
    """KernelCompiler 集成测试（无 GPU 要求）"""
    
    def test_op_registry_init(self):
        """测试算子注册表初始化"""
        from infinicore.fusion.kernel_compiler import (
            get_supported_fusion_ops, _init_op_registry
        )
        
        _init_op_registry()
        ops = get_supported_fusion_ops()
        
        # 验证基础算子已注册
        expected_ops = {"silu", "gelu", "relu", "mul", "add"}
        for op in expected_ops:
            assert op in ops, f"Expected op '{op}' not in registry"
    
    def test_heuristics_sync_with_registry(self):
        """测试 heuristics 与 kernel_compiler 注册表同步"""
        from infinicore.fusion.heuristics import FusionHeuristics
        from infinicore.fusion.fusion_config import FusionConfig
        from infinicore.fusion.kernel_compiler import get_supported_fusion_ops
        
        config = FusionConfig()
        heuristics = FusionHeuristics(config)
        
        heuristics_ops = heuristics.get_supported_ops()
        compiler_ops = get_supported_fusion_ops()
        
        # heuristics 应该包含 compiler 中所有算子
        if compiler_ops:  # 仅当 compiler 可用时
            for op in compiler_ops:
                assert op in heuristics_ops, f"Op '{op}' in compiler but not in heuristics"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestFusionNumericalCorrectness:
    """融合数值正确性测试（需要 GPU）"""
    
    def test_create_handle_for_silu(self):
        """测试为 silu 算子创建 _Handle"""
        from infinicore.fusion.kernel_compiler import KernelCompiler, FusionError
        from infinicore.fusion.fusion_config import FusionConfig
        
        config = FusionConfig(debug_mode=True)
        compiler = KernelCompiler(config)
        
        if not compiler.is_available:
            pytest.skip("ninetoothed/ntops not available")
        
        # 创建 silu 的 handle
        handle = compiler._create_handle_for_op("silu", ndim=2)
        
        assert handle is not None
        assert callable(handle)
        
        # 验证 handle 具有 arrangement 和 application 属性
        assert hasattr(handle, "arrangement")
        assert hasattr(handle, "application")
    
    def test_create_handle_for_mul(self):
        """测试为 mul 算子创建 _Handle"""
        from infinicore.fusion.kernel_compiler import KernelCompiler
        from infinicore.fusion.fusion_config import FusionConfig
        
        config = FusionConfig()
        compiler = KernelCompiler(config)
        
        if not compiler.is_available:
            pytest.skip("ninetoothed/ntops not available")
        
        handle = compiler._create_handle_for_op("mul", ndim=2)
        
        assert handle is not None
        assert callable(handle)
    
    def test_silu_kernel_numerical(self):
        """测试单个 silu 内核的数值正确性"""
        from infinicore.fusion.kernel_compiler import KernelCompiler
        from infinicore.fusion.fusion_config import FusionConfig
        
        config = FusionConfig()
        compiler = KernelCompiler(config)
        
        if not compiler.is_available:
            pytest.skip("ninetoothed/ntops not available")
        
        # 创建 handle
        handle = compiler._create_handle_for_op("silu", ndim=2)
        
        # 创建测试数据
        x = torch.randn(4, 1024, device="cuda", dtype=torch.float16)
        output = torch.empty_like(x)
        
        # 调用内核
        handle(x, output)
        
        # 计算参考值
        expected = torch.nn.functional.silu(x)
        
        # 验证数值
        assert torch.allclose(output, expected, rtol=1e-2, atol=1e-2), \
            f"Max diff: {(output - expected).abs().max()}"
    
    def test_mul_kernel_numerical(self):
        """测试单个 mul 内核的数值正确性"""
        from infinicore.fusion.kernel_compiler import KernelCompiler
        from infinicore.fusion.fusion_config import FusionConfig
        
        config = FusionConfig()
        compiler = KernelCompiler(config)
        
        if not compiler.is_available:
            pytest.skip("ninetoothed/ntops not available")
        
        handle = compiler._create_handle_for_op("mul", ndim=2)
        
        a = torch.randn(4, 1024, device="cuda", dtype=torch.float16)
        b = torch.randn(4, 1024, device="cuda", dtype=torch.float16)
        output = torch.empty_like(a)
        
        handle(a, b, output)
        
        expected = a * b
        
        assert torch.allclose(output, expected, rtol=1e-2, atol=1e-2), \
            f"Max diff: {(output - expected).abs().max()}"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestFallbackExecution:
    """回退执行路径测试"""
    
    def test_fallback_silu_mul(self):
        """测试 silu + mul 的回退执行"""
        from infinicore.fusion import FusionScheduler, FusionConfig, SubGraph, OpNode
        
        # 禁用融合，强制走回退路径
        config = FusionConfig(enable_fusion=False)
        scheduler = FusionScheduler(config)
        
        graph = SubGraph(
            nodes=(
                OpNode("silu", ("x",), ("y1",)),
                OpNode("mul", ("y1", "x"), ("y2",)),
            ),
            input_names=("x",),
            output_names=("y2",),
        )
        
        x = torch.randn(2, 4096, device="cuda", dtype=torch.float16)
        
        outputs = scheduler.dispatch(graph, {"x": x})
        result = outputs["y2"]
        
        # 参考实现
        expected = torch.nn.functional.silu(x) * x
        
        assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2), \
            f"Max diff: {(result - expected).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
