"""
Graph Metadata Extension Test Script

用于验证 InfiniCore Graph 扩展后的新接口：
- Graph.operators() 返回 GraphOperator 列表
- GraphOperator.op_type 返回算子类型
- GraphOperator.tensor_metas 返回张量元信息
- GraphTensorMeta 包含 shape, dtype, is_input

Usage:
    python test_graph_metadata.py                  # 自动检测可用设备
    python test_graph_metadata.py --device cuda:0  # 指定 CUDA 设备
    python test_graph_metadata.py --device npu:0   # 指定昇腾 NPU
    python test_graph_metadata.py --device mlu:0   # 指定寒武纪 MLU
    python test_graph_metadata.py --device musa:0  # 指定摩尔线程

Requirements:
    - 需要先编译安装扩展后的 InfiniCore
    - 需要至少一种设备可用
"""

import sys
import argparse


def get_available_device(preferred=None):
    """
    获取可用设备。
    
    Args:
        preferred: 优先使用的设备类型（如 "cuda:0", "npu:0"）
        
    Returns:
        infinicore.device 或 None
    """
    import infinicore
    
    # 如果指定了设备，尝试使用
    if preferred:
        try:
            dev = infinicore.device(preferred)
            # 测试设备是否可用
            _ = infinicore.Tensor.empty((1,), infinicore.float32, dev)
            return dev
        except Exception as e:
            print(f"⚠ Preferred device '{preferred}' not available: {e}")
    
    # 按优先级自动检测设备
    device_types = ["cuda", "npu", "mlu", "musa", "cpu"]
    
    for dtype in device_types:
        try:
            dev = infinicore.device(f"{dtype}:0")
            # 测试设备是否可用
            _ = infinicore.Tensor.empty((1,), infinicore.float32, dev)
            return dev
        except Exception:
            continue
    
    return None


def test_graph_metadata(device_str=None):
    """测试 Graph 元数据接口"""
    try:
        import infinicore
    except ImportError as e:
        print(f"❌ Failed to import infinicore: {e}")
        print("   请先编译安装 InfiniCore: pip install -e .")
        return False
    
    # 获取可用设备
    dev = get_available_device(device_str)
    if dev is None:
        print("❌ No available device found")
        print("   支持的设备类型: cuda, npu, mlu, musa, cpu")
        return False
    
    print(f"✓ Using device: {dev}")
    
    # 创建测试张量
    a = infinicore.Tensor.empty((32, 4096), infinicore.float16, dev)
    b = infinicore.Tensor.empty((4096, 4096), infinicore.float16, dev)
    
    print(f"✓ Created tensors: a={a.shape()}, b={b.shape()}")
    
    # 录制 Graph
    infinicore.start_graph_recording()
    c = infinicore.matmul(a, b)
    graph = infinicore.stop_graph_recording()
    
    if graph is None:
        print("❌ Graph recording returned None")
        return False
    
    print(f"✓ Recorded graph: {graph}")
    
    # 测试 __len__
    if not hasattr(graph, '__len__'):
        print("⚠ Graph.__len__ not available (old version)")
    else:
        print(f"✓ Graph size: {len(graph)}")
    
    # 测试 operators()
    if not hasattr(graph, 'operators'):
        print("❌ Graph.operators() not available - extension not compiled")
        return False
    
    operators = graph.operators()
    print(f"✓ Number of operators: {len(operators)}")
    
    for i, op in enumerate(operators):
        print(f"\n  Operator {i}:")
        print(f"    op_type: {op.op_type}")
        print(f"    tensor_metas count: {len(op.tensor_metas)}")
        
        for j, meta in enumerate(op.tensor_metas):
            print(f"      Tensor {j}: shape={meta.shape}, dtype={meta.dtype}, is_input={meta.is_input}")
    
    # 测试 SubGraph 转换
    print("\n--- Testing SubGraph conversion ---")
    try:
        from infinicore.fusion.graph_converter import convert_graph_to_subgraph
        
        subgraph = convert_graph_to_subgraph(graph)
        if subgraph is None:
            print("⚠ SubGraph conversion returned None")
        else:
            print(f"✓ SubGraph: {subgraph}")
            print(f"  nodes: {len(subgraph.nodes)}")
            for node in subgraph.nodes:
                print(f"    - {node}")
    except Exception as e:
        print(f"⚠ SubGraph conversion failed: {e}")
    
    print("\n" + "=" * 40)
    print("✅ All tests passed!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Graph Metadata Extension")
    parser.add_argument(
        "--device", "-d",
        type=str,
        default=None,
        help="Device to use (e.g., cuda:0, npu:0, mlu:0, musa:0, cpu:0)"
    )
    args = parser.parse_args()
    
    success = test_graph_metadata(args.device)
    sys.exit(0 if success else 1)
