"""Neo4j 客户端与图模型约束。

图只承担一件事：**由情绪反查意象、再由意象找回作品**。
这是纯向量检索做不到的——向量库能回答「这段话和什么相似」，但回答不了
「哀伤这种情绪在古典文学里通常借哪些意象表达」。补上这条路径，
Node 5 的「多路检索」才名副其实。

图里只存 snippet（≤60 字），全文一律回 Qdrant 按 chunk_id 取。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession

from app.config import get_settings
from app.logging_conf import get_logger

log = get_logger(__name__)

SNIPPET_MAX_CHARS = 60

_driver: AsyncDriver | None = None


def init_neo4j(uri: str | None = None) -> AsyncDriver:
    global _driver
    if _driver is None:
        settings = get_settings()
        _driver = AsyncGraphDatabase.driver(
            uri or settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            max_connection_pool_size=20,
            # Neo4j 会对「MATCH (n:Chunk)」这种尚未建过标签的查询逐条回一个
            # UNRECOGNIZED 通知，日志被刷屏且看着像报错。空图是正常状态
            # （还没跑 ingest-kb），不是问题，直接关掉通知。
            notifications_min_severity="OFF",
        )
    return _driver


def get_driver() -> AsyncDriver:
    if _driver is None:
        return init_neo4j()
    return _driver


async def close_neo4j() -> None:
    global _driver
    if _driver is not None:
        try:
            await _driver.close()
        except Exception as exc:  # pragma: no cover
            log.warning("neo4j_close_failed", error=str(exc))
    _driver = None


@asynccontextmanager
async def graph_session() -> AsyncIterator[AsyncSession]:
    async with get_driver().session() as session:
        yield session


async def ping() -> bool:
    try:
        async with graph_session() as session:
            await (await session.run("RETURN 1 AS ok")).single()
        return True
    except Exception:
        return False


# 约束既是正确性保障也是性能保障：MERGE 在无约束时会全表扫描
_CONSTRAINTS: tuple[str, ...] = (
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT work_title IF NOT EXISTS FOR (n:Work) REQUIRE n.title IS UNIQUE",
    "CREATE CONSTRAINT imagery_name IF NOT EXISTS FOR (n:Imagery) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT emotion_name IF NOT EXISTS FOR (n:Emotion) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT character_key IF NOT EXISTS FOR (n:Character) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (n:Concept) REQUIRE n.name IS UNIQUE",
)


async def ensure_constraints() -> None:
    async with graph_session() as session:
        for stmt in _CONSTRAINTS:
            await session.run(stmt)
    log.info("neo4j_constraints_ensured")


def snippet(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= SNIPPET_MAX_CHARS else text[:SNIPPET_MAX_CHARS] + "…"


async def node_counts() -> dict[str, int]:
    """给 /health/deps 与 CLI check 用的图规模概览。"""
    labels = ("Chunk", "Work", "Imagery", "Emotion", "Character", "Concept")
    counts: dict[str, int] = {}
    async with graph_session() as session:
        for label in labels:
            result = await session.run(f"MATCH (n:{label}) RETURN count(n) AS c")  # noqa: S608 - label 来自白名单
            record = await result.single()
            counts[label] = record["c"] if record else 0
    return counts
