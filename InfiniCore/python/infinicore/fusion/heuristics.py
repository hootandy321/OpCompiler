import json
import os
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
    4. profile 决策（unfused 总时间 vs fused 时间）

    profile 缺失/异常：打印错误并返回 False
    """

    def __init__(self, config: FusionConfig, profile_path: Optional[str] = "./profile.json"):
        self.config = config
        self.profile_path = profile_path 
        self._supported_ops: Optional[Set[str]] = None  # 延迟初始化

        # profile cache（按路径缓存一次解析结果）
        self._profile_cache: Optional[Dict[str, Any]] = None
        self._profile_path_cached: Optional[str] = None

    def _get_ops(self) -> Set[str]:
        """获取支持的算子集合（带缓存）"""
        if self._supported_ops is None:
            self._supported_ops = _get_supported_ops()
        return self._supported_ops

    # ---------------- profile helpers ----------------

    def _load_profile(self) -> Dict[str, Any]:
        """
        从 self.profile_path 加载 profile（JSON 文件），并缓存解析结果。

        profile JSON 期望结构：
        {
          "unfused": { "add+rms_norm": { "[1, 512, 4096]": 0.18, ... }, ... },
          "fused":   { "add+rms_norm": { "[1, 512, 4096]": 0.12, ... }, ... }
        }
        """
        if not isinstance(self.profile_path, str) or not self.profile_path:
            raise ValueError("self.profile_path must be a non-empty string")

        profile_path = self.profile_path

        # cache hit
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

    def _parse_shape_key(self, shape_key: str) -> Tuple[int, ...]:
        # "[1, 512, 4096]" -> (1, 512, 4096)
        return tuple(int(x.strip()) for x in shape_key.strip("[]").split(","))

    def _lookup_nearest_shape( # “就近 bucket”查找函数
        self,
        shape_key: str,
        shape_map: Dict[str, Any],
    ) -> Optional[Any]:
        """
        在 shape_map 中查找与 shape_key 最接近的 bucket。
        规则：
        - rank 必须一致
        - B 和 H 必须完全一致
        - 只在 token(S) 维度做 lower-bound
        """
        target = self._parse_shape_key(shape_key)

        best_key = None
        best_s = None

        for k in shape_map.keys():
            try:
                cand = self._parse_shape_key(k)
            except Exception:
                continue

            if len(cand) != len(target):
                continue

            # 固定 B 和 H
            if cand[0] != target[0] or cand[-1] != target[-1]:
                continue

            s = cand[1]
            if s <= target[1]:
                if best_s is None or s > best_s:
                    best_s = s
                    best_key = k

        # 如果所有 bucket 的 S 都 > target.S，则取最小的那个
        if best_key is None:
            for k in shape_map.keys():
                try:
                    cand = self._parse_shape_key(k)
                except Exception:
                    continue
                if len(cand) == len(target) and cand[0] == target[0] and cand[-1] == target[-1]:
                    best_key = k
                    break

        return shape_map.get(best_key) if best_key else None


    def should_fuse(
        self,
        graph: SubGraph,
        input_shapes: Dict[str, Tuple[int, ...]],
        margin: float = 0.0,
    ) -> bool:
        """
        判断是否应该尝试融合（静态过滤 + profile 决策）

        profile 决策：
        unfused_total_time(op_key, shape) > fused_time(op_key, shape) * (1 + margin) => True

        特殊规则：
        - profile 缺失/异常：print 错误信息并返回 False
        """
        return self.config.enable_fusion
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

        # --- profile 决策（用代表 shape 查表） ---
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
        
        t_unfused = unfused_map.get(shape_key)
        if t_unfused is None:
            t_unfused = self._lookup_nearest_shape(shape_key, unfused_map)

        t_fused = fused_map.get(shape_key)
        if t_fused is None:
            t_fused = self._lookup_nearest_shape(shape_key, fused_map)

        if t_unfused is None or t_fused is None:
            print(
                f"[Fusion][Error] Profile missing timing: "
                f"op='{op_key}', shape={shape_key}, unfused={t_unfused}, fused={t_fused}"
            )
            return False

        try:
            t_unfused_f = float(t_unfused)
            t_fused_f = float(t_fused)
        except Exception as e:
            print(
                f"[Fusion][Error] Invalid profile values at op='{op_key}', shape={shape_key}: "
                f"unfused={t_unfused}, fused={t_fused}, err={e}"
            )
            return False

        # margin 从函数入参获取
        try:
            margin_f = float(margin)
        except Exception as e:
            print(f"[Fusion][Error] Invalid margin '{margin}': {e}")
            margin_f = 0.0

        decision = t_unfused_f > t_fused_f * (1.0 + margin_f)

        if self.config.debug_mode:
            print(
                f"[Fusion] Profile decision op='{op_key}', shape={shape_key}: "
                f"unfused={t_unfused_f:.6f} fused={t_fused_f:.6f} margin={margin_f:.3f} => "
                f"{'FUSE' if decision else 'NO_FUSE'}"
            )

        return decision

    def get_supported_ops(self) -> Set[str]:
        """返回当前支持融合的算子类型集合"""
        return self._get_ops().copy()
