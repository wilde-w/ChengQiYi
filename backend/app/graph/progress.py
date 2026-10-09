"""进度计算：把「节点内的完成比例」映射成「整体百分比」。

两条硬约束：
  1. **单调不减**。任何插值、重试、干预导致回退时，对外发出的进度都不许变小——
     进度条往回走是最伤信任的 UI 缺陷。
  2. **节点未完成前不越过带尾**。n1 的带是 2–15，那它在完成前最多只能到 14。
     否则「100% 但还在加载」的观感会出现。

带尾的 100 只有 n7 完成时才可达，且只经由 `finish()`。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.constants import NodeKey
from app.graph.registry import BY_KEY, NODES, NodeSpec


@dataclass(slots=True)
class ProgressTracker:
    """一次运行对应一个实例。非线程安全——运行在单个 asyncio 任务里。"""

    _current: float = 0.0
    _node: NodeKey | None = None
    _node_fraction: float = 0.0
    _completed: set[NodeKey] = field(default_factory=set)

    # ------------------------------------------------------------------
    @property
    def current(self) -> float:
        return self._current

    def _spec(self) -> NodeSpec:
        if self._node is None:
            raise RuntimeError("尚未进入任何节点，先调用 begin()")
        return BY_KEY[self._node]

    def _advance_to(self, value: float) -> float:
        """单调钳制。取 max 而不是赋值——这一个判断就是约束 1 的全部实现。"""
        if value > self._current:
            self._current = min(round(value, 1), 100.0)
        return self._current

    # ------------------------------------------------------------------
    def begin(self, node: NodeKey | str) -> float:
        """进入节点。返回该节点的带起点。"""
        key = NodeKey(node)
        self._node = key
        self._node_fraction = 0.0
        return self._advance_to(BY_KEY[key].start)

    def set(self, fraction: float) -> float:
        """设置节点内完成比例（0.0–1.0），返回映射后的整体进度。

        带尾留 1 个点的余量——这样「节点内部跑到 100%」与
        「节点真的完成了」在界面上是两个可区分的事件。
        """
        spec = self._spec()
        frac = max(0.0, min(1.0, fraction))
        self._node_fraction = max(self._node_fraction, frac)   # 节点内也单调
        ceiling = spec.end - 1.0
        span = ceiling - spec.start
        return self._advance_to(spec.start + span * self._node_fraction)

    def set_absolute(self, fraction: float) -> float:
        """按整体比例推进（用于一个节点内部有多个等权大步骤时）。"""
        return self.set(fraction)

    def finish(self, node: NodeKey | str | None = None) -> float:
        """节点完成，落到带尾。"""
        key = NodeKey(node) if node is not None else self._node
        if key is None:
            raise RuntimeError("尚未进入任何节点，先调用 begin()")
        spec = BY_KEY[key]
        self._completed.add(key)
        self._node_fraction = 1.0
        return self._advance_to(spec.end)

    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, object]:
        return {
            "progress": self._current,
            "node": self._node.value if self._node else None,
            "node_fraction": round(self._node_fraction, 3),
            "completed": [k.value for k in NODES if k.key in self._completed],
        }

    def is_complete(self) -> bool:
        return len(self._completed) == len(NODES)


def empty_snapshot() -> dict[str, object]:
    return {"progress": 0.0, "node": None, "node_fraction": 0.0, "completed": []}
