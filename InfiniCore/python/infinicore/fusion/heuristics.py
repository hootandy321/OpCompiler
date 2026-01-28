"""
静态启发式规则模块 - V1 版本

决定是否值得对给定子图进行融合编译。
融合并不总是带来性能提升，特别是对于小 Shape 或编译开销较大的场景。

当前版本要点：
- profile 从对象属性 self.profile_path 读取（不再作为 should_fuse 入参）
- profile 数据使用 unfused 总时间 vs fused 时间（不再使用 single 逐算子求和）
- shape miss 时使用“全维度多线性插值”（multilinear interpolation）
  - 若 2^D 角点齐全：严格多线性插值
  - 若角点不齐：降级为最近点（避免硬失败）
- margin 改为 should_fuse 的函数入参，默认值 0.0
- 任何 profile 异常/缺失：print 错误信息并提前 return False
"""

import json
import os
from itertools import product
from typing import Dict, Tuple, Set, Optional, Any

from infinicore.fusion.subgraph import SubGraph
from infinicore.fusion.fusion_config import FusionConfig


def _get_supported_ops() -> Set[str]:
    """获取支持融合的算子集合，与 kernel_compiler 保持同步"""
    fallback_ops = {
        "silu", "gelu", "relu", "sigmoid",
        "add", "mul", "sub", "div",
        "rms_norm", "layer_norm",
    }
    try:
        from infinicore.fusion.kernel_compiler import get_supported_fusion_ops
        ops = get_supported_fusion_ops()
        return ops if ops else fallback_ops
    except ImportError:
        return fallback_ops


# V1 支持融合的算子类型（延迟初始化）
SUPPORTED_OPS: Set[str] = set()


class FusionHeuristics:
    """
    静态启发式规则 - 决定是否值得融合

    V1：静态过滤 + profile 决策
    - 静态过滤：节点数、图大小、张量大小、算子支持
    - profile 决策：比较 unfused_total_time vs fused_time

    任何 profile 异常/缺失：打印错误并返回 False（保守不融合）
    """

    def __init__(self, config: FusionConfig, profile_path: Optional[str] = None):
        self.config = config
        self.profile_path = profile_path

        self._supported_ops: Optional[Set[str]] = None

        # profile cache（按路径缓存一次解析结果）
        self._profile_cache: Optional[Dict[str, Any]] = None
        self._profile_path_cached: Optional[str] = None

    # ---------------- basic helpers ----------------

    def _get_ops(self) -> Set[str]:
        if self._supported_ops is None:
            self._supported_ops = _get_supported_ops()
        return self._supported_ops

    def get_supported_ops(self) -> Set[str]:
        return self._get_ops().copy()

    # ---------------- profile helpers ----------------

    def _load_profile(self) -> Dict[str, Any]:
        """从 self.profile_path 加载 profile JSON，并缓存解析结果"""
        if not isinstance(self.profile_path, str) or not self.profile_path:
            raise ValueError("self.profile_path must be a non-empty string")

        profile_path = self.profile_path

        if self._profile_cache is not None and self._profile_path_cached == profile_path:
            return self._profile_cache

        if not os.path.exists(profile_path):
            raise FileNotFoundError(f"Profile json not found: {profile_path}")

        with open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "unfused" not in data or "fused" not in data:
            raise ValueError("Profile JSON must contain top-level keys: 'unfused' and 'fused'")

        self._profile_cache = data
        self._profile_path_cached = profile_path
        return data

    def _shape_to_key(self, shape: Tuple[int, ...]) -> str:
        """tuple -> '[1, 512, 4096]'（与 profile key 格式一致：逗号后有空格）"""
        return "[" + ", ".join(str(int(x)) for x in shape) + "]"

    def _parse_shape_key(self, shape_key: str) -> Tuple[int, ...]:
        """'[1, 512, 4096]' -> (1, 512, 4096)"""
        s = shape_key.strip()
        if not (s.startswith("[") and s.endswith("]")):
            raise ValueError(f"Invalid shape_key format: {shape_key}")
        inner = s[1:-1].strip()
        if not inner:
            return tuple()
        return tuple(int(x.strip()) for x in inner.split(","))

    def _pick_profile_shape_key(
        self,
        graph: SubGraph,
        input_shapes: Dict[str, Tuple[int, ...]]
    ) -> Optional[str]:
        """
        在 input_shapes 中选择一个张量 shape 作为代表执行规模的 profile key。
        优先使用 graph.input_names（子图外部输入）以避免选到权重/bias。
        """
        for name in graph.input_names:
            shape = input_shapes.get(name)
            if shape:
                return self._shape_to_key(shape)

        # 兜底：取任意一个输入 shape
        for _, shape in input_shapes.items():
            if shape:
                return self._shape_to_key(shape)

        return None

    def _fused_op_key(self, graph: SubGraph) -> str:
        """根据拓扑顺序生成子图 op key，例如 'add+rms_norm'"""
        return "+".join(node.op_type for node in graph.nodes)

    # ---------------- interpolation ----------------

    def _interp_time_multidim(
        self,
        target_shape_key: str,
        shape_to_time: Dict[str, Any],
    ) -> Optional[float]:
        """
        在所有维度上做多线性插值（multilinear interpolation）。

        - 仅使用同 rank 的 profile 点
        - 若 2^D 个角点齐全：严格多线性插值
        - 若角点不齐：降级为最近点（L1 距离）
        """
        target = self._parse_shape_key(target_shape_key)
        D = len(target)
        if D == 0:
            return None

        # 收集同 rank 的有效点
        points: Dict[Tuple[int, ...], float] = {}
        for k, v in shape_to_time.items():
            if v is None:
                continue
            try:
                shp = self._parse_shape_key(k)
            except Exception:
                continue
            if len(shp) != D:
                continue
            try:
                points[tuple(int(x) for x in shp)] = float(v)
            except Exception:
                continue

        if not points:
            return None

        target_t = tuple(int(x) for x in target)
        if target_t in points:
            return float(points[target_t])

        # 每维可用坐标集合
        coords_per_dim = [sorted({p[d] for p in points.keys()}) for d in range(D)]

        # 每维 lo/hi（clamp 到边界，避免外推）
        lo = [0] * D
        hi = [0] * D
        for d in range(D):
            xs = coords_per_dim[d]
            x = target[d]

            lo_v = None
            hi_v = None
            for v in xs:
                if v <= x:
                    lo_v = v
                if v >= x and hi_v is None:
                    hi_v = v

            if lo_v is None:
                lo_v = xs[0]
            if hi_v is None:
                hi_v = xs[-1]

            lo[d] = int(lo_v)
            hi[d] = int(hi_v)

        # 构造角点（2^D）
        corner_bits = list(product([0, 1], repeat=D))
        corners = [tuple(lo[d] if bits[d] == 0 else hi[d] for d in range(D)) for bits in corner_bits]

        # 若角点齐全：严格多线性插值
        if all(c in points for c in corners):
            # alpha(d) in [0,1]
            alphas = []
            for d in range(D):
                x = float(target[d])
                x0 = float(lo[d])
                x1 = float(hi[d])
                if x1 == x0:
                    a = 0.0
                else:
                    a = (x - x0) / (x1 - x0)
                    if a < 0.0:
                        a = 0.0
                    elif a > 1.0:
                        a = 1.0
                alphas.append(float(a))

            total = 0.0
            for bits, c in zip(corner_bits, corners):
                w = 1.0
                for d in range(D):
                    a = alphas[d]
                    w *= (1.0 - a) if bits[d] == 0 else a
                total += w * points[c]
            return float(total)

        # 角点不齐：降级为最近点（L1 距离）
        best_t = None
        best_dist = None
        for p, t in points.items():
            dist = 0
            for d in range(D):
                dist += abs(p[d] - target[d])
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_t = t

        return float(best_t) if best_t is not None else None

    def _get_time_with_interp(
        self,
        shape_to_time: Dict[str, Any],
        shape_key: str,
        ctx: str,
    ) -> Optional[float]:
        """
        先精确查表，miss 则做全维度多线性插值。
        """
        v = shape_to_time.get(shape_key, None)
        if v is not None:
            try:
                return float(v)
            except Exception as e:
                print(f"[Fusion][Error] Invalid time value for {ctx} at shape={shape_key}: {v} ({e})")
                return None

        try:
            return self._interp_time_multidim(shape_key, shape_to_time)
        except Exception as e:
            print(f"[Fusion][Error] Interp failed for {ctx} at shape={shape_key}: {e}")
            return None

    # ---------------- main API ----------------

    def should_fuse(
        self,
        graph: SubGraph,
        input_shapes: Dict[str, Tuple[int, ...]],
        margin: float = 0.0,
    ) -> bool:
        """
        判断是否应该尝试融合（静态过滤 + profile 决策）

        决策：
        unfused_total_time(op_key, shape) > fused_time(op_key, shape) * (1 + margin) => True

        任何异常（尤其 profile 相关）提前 return：返回 False
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

        # --- profile 决策（用代表 shape 查表/插值） ---
        shape_key = self._pick_profile_shape_key(graph, input_shapes)
        if shape_key is None:
            print("[Fusion][Error] Cannot pick a representative shape key from input_shapes")
            return False

        op_key = self._fused_op_key(graph)

        try:
            profile = self._load_profile()
        except Exception as e:
            print(f"[Fusion][Error] Failed to load profile from '{self.profile_path}': {e}")
            return False

        try:
            unfused_map = profile["unfused"][op_key]
            fused_map = profile["fused"][op_key]
        except Exception as e:
            print(f"[Fusion][Error] Invalid profile structure for op='{op_key}': {e}")
            return False

        t_unfused = self._get_time_with_interp(unfused_map, shape_key, ctx=f"unfused op='{op_key}'")
        t_fused = self._get_time_with_interp(fused_map, shape_key, ctx=f"fused op='{op_key}'")

        if t_unfused is None or t_fused is None:
            print(
                f"[Fusion][Error] Profile timing unavailable even after interpolation: "
                f"op='{op_key}', shape={shape_key}, unfused={t_unfused}, fused={t_fused}"
            )
            return False

        try:
            margin_f = float(margin)
        except Exception as e:
            print(f"[Fusion][Error] Invalid margin '{margin}': {e}")
            return False

        decision = float(t_unfused) > float(t_fused) * (1.0 + margin_f)

        if self.config.debug_mode:
            print(
                f"[Fusion] Profile decision op='{op_key}', shape={shape_key}: "
                f"unfused={float(t_unfused):.6f} fused={float(t_fused):.6f} margin={margin_f:.3f} => "
                f"{'FUSE' if decision else 'NO_FUSE'}"
            )

        return decision
