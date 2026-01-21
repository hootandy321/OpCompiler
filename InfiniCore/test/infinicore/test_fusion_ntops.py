"""
Integration tests for ntops operator fusion.

This file tests the actual fusion capability by:
1. Defining common LLM fusion patterns.
2. Running them via FusionScheduler.
3. Comparing outputs between fused and fallback paths.

Note: Requires CUDA environment, ntops, and ninetoothed.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any

from infinicore.fusion.fusion_scheduler import FusionScheduler
from infinicore.fusion.fusion_config import FusionConfig
from infinicore.fusion.subgraph import SubGraph, OpNode
from infinicore.fusion.patterns.llm_patterns import (
    create_swiglu_pattern,
    create_add_rms_norm_pattern
)

# Skip tests if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

@pytest.fixture
def scheduler():
    config = FusionConfig(enable_fusion=True, debug_mode=True)
    return FusionScheduler(config)

@pytest.fixture
def scheduler_no_fusion():
    config = FusionConfig(enable_fusion=False)
    return FusionScheduler(config)

def test_silu_mul_fusion(scheduler, scheduler_no_fusion):
    """验证 SiLU + Mul (SwiGLU) 融合"""
    import infinicore
    
    # 准备输入
    batch_size, hidden_dim = 32, 4096
    gate = torch.randn(batch_size, hidden_dim, device='cuda', dtype=torch.float16)
    up = torch.randn(batch_size, hidden_dim, device='cuda', dtype=torch.float16)
    
    # 将 torch 张量转换为 infinicore 张量 (假设 infinicore.from_torch 可用)
    # 如果不可用，则直接传入 torch 张量（取决于 infinicore 的对接实现）
    inputs = {
        "gate": infinicore.from_torch(gate) if hasattr(infinicore, 'from_torch') else gate,
        "up": infinicore.from_torch(up) if hasattr(infinicore, 'from_torch') else up
    }
    
    # 获取 SwiGLU 模式
    graph = create_swiglu_pattern()
    
    # 1. 运行融合路径
    outputs_fused = scheduler.dispatch(graph, inputs)
    
    # 2. 运行回退路径 (禁用融合)
    outputs_fallback = scheduler_no_fusion.dispatch(graph, inputs)
    
    # 3. 结果验证
    out_fused = outputs_fused["output"]
    out_fallback = outputs_fallback["output"]
    
    # 转换为 torch 以进行对比
    if hasattr(out_fused, 'to_torch'):
        out_fused = out_fused.to_torch()
    if hasattr(out_fallback, 'to_torch'):
        out_fallback = out_fallback.to_torch()
        
    torch.testing.assert_close(out_fused, out_fallback, rtol=1e-3, atol=1e-3)

def test_add_rms_norm_fusion(scheduler, scheduler_no_fusion):
    """验证 Add + RMSNorm 融合"""
    import infinicore
    
    # 准备输入
    batch_size, seq_len, hidden_dim = 1, 128, 4096
    x = torch.randn(batch_size, seq_len, hidden_dim, device='cuda', dtype=torch.float16)
    residual = torch.randn(batch_size, seq_len, hidden_dim, device='cuda', dtype=torch.float16)
    weight = torch.ones(hidden_dim, device='cuda', dtype=torch.float16)
    
    inputs = {
        "x": infinicore.from_torch(x) if hasattr(infinicore, 'from_torch') else x,
        "residual": infinicore.from_torch(residual) if hasattr(infinicore, 'from_torch') else residual,
        "weight": infinicore.from_torch(weight) if hasattr(infinicore, 'from_torch') else weight
    }
    
    # 获取 Add+RMSNorm 模式
    graph = create_add_rms_norm_pattern()
    
    # 1. 运行融合路径
    outputs_fused = scheduler.dispatch(graph, inputs)
    
    # 2. 运行回退路径
    outputs_fallback = scheduler_no_fusion.dispatch(graph, inputs)
    
    # 结果验证
    out_fused = outputs_fused["output"]
    out_fallback = outputs_fallback["output"]
    
    if hasattr(out_fused, 'to_torch'):
        out_fused = out_fused.to_torch()
    if hasattr(out_fallback, 'to_torch'):
        out_fallback = out_fallback.to_torch()
        
    torch.testing.assert_close(out_fused, out_fallback, rtol=1e-3, atol=1e-3)

def test_unsupported_backtrack(scheduler):
    """验证不支持融合时是否正确回退"""
    import infinicore
    
    # 定义一个包含不支持算子的图
    graph = SubGraph(
        nodes=(
            OpNode("unsupported_op_xyz", ("x",), ("y",)),
        ),
        input_names=("x",),
        output_names=("y",),
    )
    
    x = torch.randn(10, device='cuda')
    inputs = {"x": x}
    
    # 预期报错，因为 unsupported_op_xyz 既不能融合也不能回退（未注册）
    with pytest.raises(Exception):
        scheduler.dispatch(graph, inputs)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
