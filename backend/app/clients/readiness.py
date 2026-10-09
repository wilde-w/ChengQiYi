"""依赖连通性探测。

`python -m app.cli check` 与 `GET /health/deps` 共用这里。
每个探针都带延迟，且**绝不抛异常**——探测本身失败也是一种结果。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class ProbeResult:
    name: str
    ok: bool
    latency_ms: float
    detail: str = ""
    extra: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "ok": self.ok,
            "latency_ms": round(self.latency_ms, 1),
        }
        if self.detail:
            payload["detail"] = self.detail
        if self.extra:
            payload["extra"] = self.extra
        return payload


async def _timed(name: str, coro) -> ProbeResult:
    started = time.perf_counter()
    try:
        extra = await coro
        elapsed = (time.perf_counter() - started) * 1000
        return ProbeResult(name, True, elapsed, extra=extra if isinstance(extra, dict) else {})
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000
        return ProbeResult(name, False, elapsed, detail=f"{type(exc).__name__}: {exc}")


async def probe_postgres() -> ProbeResult:
    async def run() -> dict[str, object]:
        from sqlalchemy import text

        from app.db.session import get_engine

        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {}

    return await _timed("postgres", run())


async def probe_redis() -> ProbeResult:
    async def run() -> dict[str, object]:
        from app.clients.redis_client import get_redis

        await get_redis().ping()
        return {}

    return await _timed("redis", run())


async def probe_qdrant() -> ProbeResult:
    async def run() -> dict[str, object]:
        from qdrant_client import models as qmodels

        from app.clients.qdrant_client import get_qdrant
        from app.config import get_settings

        settings = get_settings()
        client = get_qdrant()
        await client.get_collections()
        collections = [c.name for c in (await client.get_collections()).collections]
        count = 0
        if settings.QDRANT_COLLECTION in collections:
            info = await client.get_collection(settings.QDRANT_COLLECTION)
            count = info.points_count or 0
        _ = qmodels  # 保持导入，便于后续扩展
        return {"collection": settings.QDRANT_COLLECTION, "points": count}

    return await _timed("qdrant", run())


async def probe_neo4j() -> ProbeResult:
    async def run() -> dict[str, object]:
        from app.clients.neo4j_client import node_counts

        return await node_counts()

    return await _timed("neo4j", run())


async def probe_all() -> dict[str, ProbeResult]:
    """四个依赖并发探测。任何一个挂掉都用时最短——串行探测会让 check 命令卡 30s。"""
    names = ("postgres", "redis", "qdrant", "neo4j")
    coros = (probe_postgres(), probe_redis(), probe_qdrant(), probe_neo4j())
    results = await asyncio.gather(*coros, return_exceptions=True)
    out: dict[str, ProbeResult] = {}
    for name, res in zip(names, results):
        if isinstance(res, ProbeResult):
            out[name] = res
        else:
            out[name] = ProbeResult(name, False, 0.0, detail=f"probe crashed: {res}")
    return out
