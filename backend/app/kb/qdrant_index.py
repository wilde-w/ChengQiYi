"""Qdrant 索引写入与检索。

点 id 由 chunk_id 派生（`clients/qdrant_client.point_id`），所以 upsert 天然幂等：
同一份语料跑一百次，集合里还是那些点。**幂等是这里唯一需要证明的性质**，
其他一切（批量大小、并发）都只是在为它让路。
"""

from __future__ import annotations

from collections.abc import Sequence

from qdrant_client import models

from app.clients.qdrant_client import ensure_collection, get_qdrant, point_id
from app.config import get_settings
from app.constants import Library
from app.kb.schema import KBChunk
from app.logging_conf import get_logger

log = get_logger(__name__)

#: 一次 upsert 多少点。Qdrant 对单请求体积敏感，payload 里带全文时
#: 64 个 chunk 已经是几百 KB 量级，再大就只是把风险堆在一个请求里。
UPSERT_BATCH = 64


async def upsert_chunks(pairs: Sequence[tuple[KBChunk, list[float]]], *, collection: str | None = None) -> int:
    """写入 (chunk, vector) 对。返回写入点数。"""
    if not pairs:
        return 0
    settings = get_settings()
    collection = collection or settings.QDRANT_COLLECTION
    client = get_qdrant()

    written = 0
    for start in range(0, len(pairs), UPSERT_BATCH):
        batch = pairs[start : start + UPSERT_BATCH]
        points = [
            models.PointStruct(id=point_id(chunk.chunk_id), vector=vector, payload=chunk.payload())
            for chunk, vector in batch
        ]
        await client.upsert(collection_name=collection, points=points, wait=True)
        written += len(points)
    log.info("qdrant_upserted", count=written, collection=collection)
    return written


async def delete_chunks(chunk_ids: Sequence[str], *, collection: str | None = None) -> None:
    if not chunk_ids:
        return
    settings = get_settings()
    client = get_qdrant()
    await client.delete(
        collection_name=collection or settings.QDRANT_COLLECTION,
        points_selector=models.PointIdsList(points=[point_id(cid) for cid in chunk_ids]),
        wait=True,
    )


def library_filter(library: Library | None) -> models.Filter | None:
    """按库过滤。`None` 表示不限制——检索时要能一次问遍三个库。"""
    if library is None:
        return None
    return models.Filter(
        must=[models.FieldCondition(key="library", match=models.MatchValue(value=str(library)))]
    )


async def search(
    vector: list[float],
    *,
    library: Library | None = None,
    limit: int = 8,
    score_threshold: float | None = None,
    collection: str | None = None,
) -> list[models.ScoredPoint]:
    """向量检索。返回 ScoredPoint（payload 里带全文）。"""
    settings = get_settings()
    client = get_qdrant()
    # 新版本 qdrant-client 用 query_points 取代了 search；后者仍能跑但会告警。
    result = await client.query_points(
        collection_name=collection or settings.QDRANT_COLLECTION,
        query=vector,
        query_filter=library_filter(library),
        limit=limit,
        score_threshold=score_threshold,
        with_payload=True,
    )
    return list(result.points)


async def fetch_by_ids(
    chunk_ids: Sequence[str], *, collection: str | None = None
) -> list[models.Record]:
    """按 chunk_id 批量取回全文。

    **这是图检索路径的最后一跳**：Neo4j 只存 snippet，走完图之后拿到的
    是 chunk_id 列表，正文必须回 Qdrant 取。少了这一步，证据卡上就只有
    60 个字的摘要，用户点开看不到原文。
    """
    if not chunk_ids:
        return []
    settings = get_settings()
    client = get_qdrant()
    records = await client.retrieve(
        collection_name=collection or settings.QDRANT_COLLECTION,
        ids=[point_id(cid) for cid in chunk_ids],
        with_payload=True,
    )
    return list(records)


async def prepare_collection(*, dim: int, recreate: bool = False) -> None:
    await ensure_collection(dim=dim, recreate=recreate)
