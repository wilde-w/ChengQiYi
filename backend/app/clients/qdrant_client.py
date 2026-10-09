"""Qdrant 客户端与集合管理。

**Qdrant 是知识库全文的唯一归宿**，Neo4j 只存 snippet。这样两个存储不会漂移，
按 id 回查全文永远只有一个真源。

单集合承载三个库（同 embedding 模型、同维度），靠 payload filter 分区——
这样「按库配额的多路检索」可以一次请求返回，也避免三套 collection 各自演化的维护负担。
"""

from __future__ import annotations

import uuid

from qdrant_client import AsyncQdrantClient, models

from app.config import get_settings
from app.logging_conf import get_logger

log = get_logger(__name__)

# 点 id 用 chunk_id 的 UUID5 派生，保证重复摄取天然幂等
_NS = uuid.UUID("6f1b2c3d-4e5f-4a6b-8c9d-0e1f2a3b4c5d")

_client: AsyncQdrantClient | None = None


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NS, chunk_id))


def init_qdrant(url: str | None = None) -> AsyncQdrantClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncQdrantClient(url=url or settings.QDRANT_URL, timeout=30)
    return _client


def get_qdrant() -> AsyncQdrantClient:
    if _client is None:
        return init_qdrant()
    return _client


async def close_qdrant() -> None:
    global _client
    if _client is not None:
        try:
            await _client.close()
        except Exception as exc:  # pragma: no cover
            log.warning("qdrant_close_failed", error=str(exc))
    _client = None


# payload 里需要建索引的字段——不建索引的话按库过滤会退化成全量扫描。
# 字段名与 kb/schema.py 的 payload() 严格对齐：语料磁盘上叫 book，
# 统一模型里叫 work，payload 里落的是 work。索引建在一个永远不存在的
# 字段上不会报错，只会白占一个位置，所以这里必须跟着模型走。
_PAYLOAD_INDEXES: tuple[tuple[str, models.PayloadSchemaType], ...] = (
    ("library", models.PayloadSchemaType.KEYWORD),
    ("type", models.PayloadSchemaType.KEYWORD),
    ("imagery", models.PayloadSchemaType.KEYWORD),
    ("emotion", models.PayloadSchemaType.KEYWORD),
    ("dynasty", models.PayloadSchemaType.KEYWORD),
    ("work", models.PayloadSchemaType.KEYWORD),
    # 导入来源的原始文件名。删除一次导入要按它过滤，不建索引就是全量扫描。
    ("origin", models.PayloadSchemaType.KEYWORD),
)


async def ensure_collection(
    collection: str | None = None, *, dim: int | None = None, recreate: bool = False
) -> None:
    settings = get_settings()
    collection = collection or settings.QDRANT_COLLECTION
    dim = dim or settings.EMBEDDING_DIM
    client = get_qdrant()

    exists = await client.collection_exists(collection)
    if exists and recreate:
        await client.delete_collection(collection)
        exists = False

    if not exists:
        await client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
            # 全文存入 payload，放磁盘避免内存被语料吃满
            on_disk_payload=True,
        )
        log.info("qdrant_collection_created", collection=collection, dim=dim)
    else:
        info = await client.get_collection(collection)
        existing_dim = _vector_size(info)
        if existing_dim is not None and existing_dim != dim:
            raise RuntimeError(
                f"Qdrant collection「{collection}」现有维度 {existing_dim} 与配置的 {dim} 不一致。"
                f" 换过 embedding 模型时必须 `python -m app.cli ingest-kb --reset` 重建。"
            )

    for field_name, schema in _PAYLOAD_INDEXES:
        try:
            await client.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_schema=schema,
            )
        except Exception:
            # 索引已存在时客户端会抛错，这里幂等处理
            pass


def _vector_size(info: object) -> int | None:
    try:
        params = info.config.params.vectors  # type: ignore[attr-defined]
        return getattr(params, "size", None)
    except Exception:
        return None


async def collection_count(collection: str | None = None) -> int:
    settings = get_settings()
    collection = collection or settings.QDRANT_COLLECTION
    client = get_qdrant()
    if not await client.collection_exists(collection):
        return 0
    return (await client.get_collection(collection)).points_count or 0
