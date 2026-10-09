"""图的组装与检查点。

拓扑是**纯线性、无条件边**：n1 → n2 → … → n7 → END。
不做条件分支是刻意的——「重新聚类」「改过滤条件」「排除某条证据」这类需求
应该由**干预**（patch state 后从某个节点重跑）实现，而不是在拓扑里长出岔路。
线性图的可推理性强得多：任何一个产物失效，级联范围就是它之后的所有节点。

每个节点被包一层，在进入时发一个 `node_start` 自定义事件。
这样 runner 不必依赖 `stream_mode="debug"`（噪音极大）就能精确知道节点何时开始。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langgraph.graph import END, StateGraph

from app.constants import NodeKey
from app.graph.emitting import KIND_NODE_START, stream_writer
from app.graph.registry import ORDER
from app.graph.state import AnalysisState
from app.logging_conf import get_logger

log = get_logger(__name__)


def _load(node_key: NodeKey):
    """按需导入节点，避免一次 import 把整条流水线的依赖都拉起来。"""
    from app.graph.nodes import (
        n1_video,
        n2_comments,
        n3_cluster,
        n4_psych,
        n5_retrieval,
        n6_reasoning,
        n7_literary,
    )

    return {
        NodeKey.N1_VIDEO: n1_video.n1_video,
        NodeKey.N2_COMMENTS: n2_comments.n2_comments,
        NodeKey.N3_CLUSTER: n3_cluster.n3_cluster,
        NodeKey.N4_PSYCH: n4_psych.n4_psych,
        NodeKey.N5_RETRIEVAL: n5_retrieval.n5_retrieval,
        NodeKey.N6_REASONING: n6_reasoning.n6_reasoning,
        NodeKey.N7_LITERARY: n7_literary.n7_literary,
    }[node_key]


def _wrap(key: NodeKey, fn):
    """让节点在被调用时先发一个 node_start。"""

    async def wrapped(state: AnalysisState) -> dict[str, Any]:
        write = stream_writer()
        write({"kind": KIND_NODE_START, "node": key.value})
        return await fn(state)

    wrapped.__name__ = f"wrapped_{key.value}"
    return wrapped


def build_graph(*, checkpointer: Any = None, with_checkpointer: bool = False) -> Any:
    """组装图。

    `with_checkpointer=True` 时才异步构建检查点（需要建立数据库连接），
    因为 `AsyncPostgresSaver.from_conn_string` 是异步上下文管理器。
    调用方用 `build_graph_with_checkpointer()` 走这条路。
    """
    graph = StateGraph(AnalysisState)
    for key in ORDER:
        graph.add_node(key.value, _wrap(key, _load(key)))

    graph.set_entry_point(ORDER[0].value)
    for prev, nxt in zip(ORDER, ORDER[1:], strict=False):
        graph.add_edge(prev.value, nxt.value)
    graph.add_edge(ORDER[-1].value, END)

    return graph.compile(checkpointer=checkpointer) if checkpointer else graph.compile()


# ----------------------------------------------------------------------
# 检查点
# ----------------------------------------------------------------------


def psycopg_dsn(asyncpg_dsn: str) -> str:
    """把 SQLAlchemy 的 asyncpg DSN 转成 psycopg 能用的形态。

    AsyncPostgresSaver 走 psycopg（同步驱动跑在线程里），不认 `+asyncpg`；
    SQLAlchemy 也不认 `postgresql://` 以外的裸 scheme，所以两边各留一份。
    """
    return asyncpg_dsn.replace("postgresql+asyncpg://", "postgresql://").replace(
        "postgresql+psycopg://", "postgresql://"
    )


class CheckpointerHandle:
    """一次运行的图 + 检查点。

    检查点必须**可选**：数据库不可达时流水线仍要能跑完（只是失去干预与恢复能力）。
    悄悄失败比明确降级更糟，所以降级时记一条 warning 并让 /health 能看见。
    """

    def __init__(self) -> None:
        self._ctx: Any = None
        self.saver: Any = None
        self.graph: Any = None
        self.degraded: bool = False
        self.reason: str = ""

    async def __aenter__(self) -> CheckpointerHandle:
        from app.config import get_settings

        dsn = psycopg_dsn(get_settings().POSTGRES_DSN)
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            self._ctx = AsyncPostgresSaver.from_conn_string(dsn)
            self.saver = await self._ctx.__aenter__()
            await self.saver.setup()
        except Exception as exc:
            log.warning("checkpointer_unavailable", error=str(exc))
            self.degraded = True
            self.reason = f"{type(exc).__name__}: {exc}"
            await self._close_ctx()
            from langgraph.checkpoint.memory import MemorySaver

            # 内存检查点仍支持同进程内的干预，只是重启后丢失
            self.saver = MemorySaver()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self._close_ctx()

    async def _close_ctx(self) -> None:
        if self._ctx is not None:
            try:
                await self._ctx.__aexit__(None, None, None)
            except Exception as exc:  # pragma: no cover
                log.warning("checkpointer_close_failed", error=str(exc))
            self._ctx = None


@asynccontextmanager
async def graph_context() -> AsyncIterator[CheckpointerHandle]:
    """`async with graph_context() as handle:` → handle.graph 已可用。

    用上下文管理器而不是返回元组：检查点持有数据库连接，
    执行路径上有任何异常都必须走到关闭逻辑，交给 `async with` 最不容易漏。
    """
    handle = CheckpointerHandle()
    await handle.__aenter__()
    try:
        handle.graph = build_graph(checkpointer=handle.saver)
        yield handle
    finally:
        await handle.__aexit__(None, None, None)
