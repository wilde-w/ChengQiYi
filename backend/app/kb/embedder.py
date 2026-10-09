"""语料向量化：哈希差分 + 批量嵌入。

**差分存在的理由不是省时间，是省 API 额度。** 语料是人写的，改一条就要重跑
一次摄取；没有差分的世界里，改一个错别字等于重嵌 150 条（接真实 API 时
是 150 次计费调用）。有了 content_hash，改动一条只嵌一条。

哈希的口径必须**包含嵌入模型名与维度**——换模型后同样文本的向量不再可比，
沿用旧向量会得到一组「一半旧模型一半新模型」的垃圾检索结果，而表面上
一切正常。把模型名拌进 hash，换模型自动触发全量重嵌。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from sqlalchemy import select

from app.constants import Library
from app.db.session import get_sessionmaker
from app.kb.schema import KBChunk
from app.logging_conf import get_logger
from app.models.library import KbDocument
from app.providers.base import EmbeddingModel

log = get_logger(__name__)


def content_hash(chunk: KBChunk, model: EmbeddingModel) -> str:
    """嵌入文本 + 模型身份的 sha256。

    用嵌入文本而非 `chunk.text`：意象/情感标签改了，向量就该重算，
    而 text 可能一个字没动。
    """
    material = "\x00".join(
        (
            chunk.text_for_embedding(),
            model.name,
            str(model.dim),
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class DiffResult:
    """三类 chunk：新增、内容变了、没变。

    `skipped` 单独计数而不是从总数里减——摄取完成后打印的数字要能直接
    和「连跑两次 added=0 updated=0 skipped=150」这条验收断言对上。
    """

    added: list[KBChunk] = field(default_factory=list)
    updated: list[KBChunk] = field(default_factory=list)
    skipped: list[KBChunk] = field(default_factory=list)

    @property
    def to_embed(self) -> list[KBChunk]:
        # 顺序稳定（先 added 后 updated），否则两次运行的嵌入批次组成不同，
        # 出问题时无法比对两次日志。
        return [*self.added, *self.updated]


async def load_hashes(libraries: Sequence[Library] | None = None) -> dict[str, str]:
    """读回已登记 chunk 的 content_hash。

    `libraries` 非空时只取这些库——单库重嵌不该把另外两库的登记也拉进内存。
    """
    stmt = select(KbDocument.chunk_id, KbDocument.content_hash)
    if libraries:
        stmt = stmt.where(KbDocument.library.in_([str(lib) for lib in libraries]))
    async with get_sessionmaker()() as session:
        rows = (await session.execute(stmt)).all()
    return {row.chunk_id: row.content_hash for row in rows}


async def diff(chunks: Sequence[KBChunk], model: EmbeddingModel, *, force: bool = False) -> DiffResult:
    """把候选 chunk 分成 新增 / 变更 / 未变。

    `force=True` 时一切视作变更——「向量库被清空了但登记表还在」这种
    状态（有人手工删了 collection）只有强制重嵌能救回来。
    """
    if force:
        return DiffResult(added=list(chunks))

    known = await load_hashes([c.library for c in chunks])
    result = DiffResult()
    for chunk in chunks:
        digest = content_hash(chunk, model)
        previous = known.get(chunk.chunk_id)
        if previous is None:
            result.added.append(chunk)
        elif previous != digest:
            result.updated.append(chunk)
        else:
            result.skipped.append(chunk)
    return result


async def embed_chunks(
    chunks: Sequence[KBChunk],
    model: EmbeddingModel,
    *,
    batch_size: int = 32,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[tuple[KBChunk, list[float]]]:
    """批量嵌入。返回 (chunk, vector) 对。

    **逐条与批次一一对应，且长度必须相等。** 少一条就整体报错而不是
    「跳过」——向量与 chunk 错位是那种不会崩、只会让检索结果缓慢变错的
    故障，宁可当场炸掉。
    """
    if not chunks:
        return []

    out: list[tuple[KBChunk, list[float]]] = []
    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = list(chunks[start : start + batch_size])
        texts = [c.text_for_embedding() for c in batch]
        vectors = await model.embed(texts)
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"embedding 返回 {len(vectors)} 条向量，请求了 {len(batch)} 条。"
                f" 向量与语料错位会让检索结果静默失真，拒绝继续。"
            )
        out.extend(zip(batch, vectors, strict=True))
        if on_progress is not None:
            on_progress(min(start + batch_size, total), total)
    return out


async def record_documents(
    chunks: Sequence[KBChunk],
    model: EmbeddingModel,
    *,
    source_files: dict[Library, str] | None = None,
) -> None:
    """写 kb_document 登记表（存在则更新）。

    用 ORM 的 merge 而不是原生 upsert：登记表在开发机上最多几千行，
    为了它引入方言相关的 ON CONFLICT 语句不划算，而 merge 在任何后端都对。
    """
    if not chunks:
        return
    files = source_files or {}
    async with get_sessionmaker()() as session:
        for chunk in chunks:
            await session.merge(
                KbDocument(
                    chunk_id=chunk.chunk_id,
                    library=str(chunk.library),
                    content_hash=content_hash(chunk, model),
                    dim=model.dim,
                    # 导入的 chunk 自带原始文件名；语料的 `origin` 是 None，
                    # 走 source_files 映射拿到三个语料文件名。
                    # 这个字段是 `_prune_orphans` 区分「语料」与「导入」的
                    # **唯一依据**——它错了，导入的整本书会被下一次 ingest 删掉。
                    source_file=chunk.origin or files.get(chunk.library),
                    text_preview=chunk.text[:120],
                )
            )
        await session.commit()


async def delete_documents(chunk_ids: Sequence[str]) -> int:
    """删除登记行。语料里删掉一条 chunk 时同步清理，否则登记表会越攒越多
    「已经不在语料里的幽灵行」，下次差分时它们仍被认为存在。"""
    if not chunk_ids:
        return 0
    async with get_sessionmaker()() as session:
        rows = (
            await session.execute(select(KbDocument).where(KbDocument.chunk_id.in_(list(chunk_ids))))
        ).scalars()
        count = 0
        for row in rows:
            await session.delete(row)
            count += 1
        await session.commit()
    return count
