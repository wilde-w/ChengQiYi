"""节点 → runner 的自定义事件协议。

节点**不知道** seq、进度带、Redis 的存在。它只会说四件事：

    milestone("正在抓取评论…")        面向用户的里程碑文案
    set_progress(0.4)                我在本节点内完成了多少
    partial("comment_page", {...})   一份结构化产物增量（左栏据此实时增长）
    delta("insight", "……")          一段文本增量（右栏据此逐字渲染）

runner 负责把这几件事翻译成带 seq、带整体百分比的 RunEvent 并广播。
这层间接是刻意的：节点因此可以脱离图、脱离 Redis 单测——
喂一个 state，断言返回的 patch 和发出的事件。

`kind` 的取值必须与前端 store/applyEvent.ts 的分派保持一致。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# 自定义事件的 kind
KIND_NODE_START = "node_start"
KIND_MILESTONE = "milestone"
KIND_PROGRESS = "progress"
KIND_PARTIAL = "partial"
KIND_DELTA = "delta"
KIND_WARNING = "warning"


def _noop(_payload: dict[str, Any]) -> None:
    return None


def stream_writer() -> Callable[[dict[str, Any]], None]:
    """拿到 LangGraph 的自定义流写入器；不在图内时退化为空操作。

    退化很重要：节点要能在纯 pytest 里被直接 await，不需要搭一整张图。
    """
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:
        return _noop
    return writer if callable(writer) else _noop


class NodeEmitter:
    """节点内部使用的发射器。每个节点在入口构造一个。"""

    __slots__ = ("_write", "_node")

    def __init__(self, node: str, write: Callable[[dict[str, Any]], None] | None = None) -> None:
        self._node = node
        self._write = write or stream_writer()

    def _send(self, kind: str, **payload: Any) -> None:
        self._write({"kind": kind, "node": self._node, **payload})

    def milestone(self, message: str) -> None:
        """节点**内部**的里程碑，用于过程记录里值得单独一行的事件。

        ⚠️ 不要用它播报节点的开始与结束。那两句文案归注册表
        （registry 的 `entering` / `done`）所有，由 runner 在 node_started /
        node_completed 上发一次——节点再发一遍，过程记录里每个节点就会
        出现两条一模一样的记录，看起来像系统重跑了。
        """
        self._send(KIND_MILESTONE, message=message)

    def set_progress(self, fraction: float) -> None:
        self._send(KIND_PROGRESS, fraction=max(0.0, min(1.0, fraction)))

    def partial(self, data_kind: str, data: dict[str, Any], message: str | None = None) -> None:
        """一份结构化增量。`data_kind` 决定前端把它并进哪一块状态。"""
        payload: dict[str, Any] = {"data_kind": data_kind, "data": data}
        if message:
            payload["message"] = message
        self._send(KIND_PARTIAL, **payload)

    def delta(self, section: str, text: str) -> None:
        """文本增量。section 指明目标段落（见 SectionKey）。"""
        self._send(KIND_DELTA, section=section, text=text)

    def warn(self, code: str, message: str, **extra: Any) -> None:
        self._send(KIND_WARNING, code=code, message=message, **extra)
