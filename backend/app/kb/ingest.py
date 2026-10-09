"""知识库摄取编排：语料 → 向量库 + 图库 + 登记表。

一次摄取要同时改三个存储（Qdrant 点、Neo4j 图、PG 登记表）。**没有事务能
跨越这三个存储**，所以幂等性是唯一可行的正确性策略：每个写入都用确定性的
键（chunk_id / uuid5 / MERGE），跑到一半失败就重跑，不会产生重复或者
半截状态。这也正是差分能成立的前提。

顺序是有讲究的：先写 Qdrant 再写 Neo4j，最后写登记表。登记表是差分的依据，
**它必须在内容真正落库之后才更新**——反过来（先记后写）一旦中途失败，
下次差分就会认为这些 chunk 已经是最新的，永远不再补写。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from sqlalchemy import func, or_, select

from app.config import get_settings
from app.constants import Library
from app.db.session import get_sessionmaker
from app.kb import neo4j_index, qdrant_index
from app.kb.embedder import delete_documents, diff, embed_chunks, load_hashes, record_documents
from app.kb.loader import CORPUS_DIR, CorpusError, load_all, load_graph_edges
from app.kb.schema import KBChunk
from app.logging_conf import get_logger
from app.models.library import KbDocument
from app.providers.base import EmbeddingModel

log = get_logger(__name__)


class EmptyLibraryError(RuntimeError):
    """某个库一条语料都没有。

    这是**最坏的失败模式**：摄取"成功"了，检索却永远查不到古典文学，
    而日志里没有任何异常。宁可让命令失败。
    """


@dataclass(slots=True)
class LibraryStat:
    library: Library
    added: int = 0
    updated: int = 0
    skipped: int = 0
    embed_seconds: float = 0.0

    @property
    def total(self) -> int:
        return self.added + self.updated + self.skipped


@dataclass(slots=True)
class IngestReport:
    stats: dict[Library, LibraryStat] = field(default_factory=dict)
    qdrant_points: int = 0
    graph_edges: int = 0
    removed: int = 0
    collection: str = ""
    embedding_model: str = ""
    dim: int = 0
    is_mock: bool = False

    @property
    def added(self) -> int:
        return sum(s.added for s in self.stats.values())

    @property
    def updated(self) -> int:
        return sum(s.updated for s in self.stats.values())

    @property
    def skipped(self) -> int:
        return sum(s.skipped for s in self.stats.values())

    @property
    def total(self) -> int:
        return sum(s.total for s in self.stats.values())


async def ingest(
    model: EmbeddingModel,
    *,
    libraries: Sequence[Library] | None = None,
    reset: bool = False,
    force: bool = False,
    corpus_dir: object = None,
    on_progress: object = None,
) -> IngestReport:
    """跑一次完整摄取。

    `reset=True` 会重建 Qdrant 集合、清空 Neo4j 图与登记表——**换 embedding
    模型后必须用它**，否则新旧向量混在一起，检索结果会慢慢坏掉且无从察觉。
    """
    settings = get_settings()
    directory = corpus_dir or CORPUS_DIR  # type: ignore[assignment]
    selected = list(libraries) if libraries else list(Library)

    loaded = load_all(directory)  # type: ignore[arg-type]
    chunks: list[KBChunk] = [c for lib in selected for c in loaded[lib]]
    if not chunks:
        raise EmptyLibraryError("选中的库一条语料都没有，检查 corpus 目录")

    # 空库检查放在嵌入之前：花掉 API 额度之后才发现某个库是空的太蠢了。
    for lib in selected:
        if not loaded[lib]:
            raise EmptyLibraryError(f"{lib} 库为空——静默的空库会让检索永远缺一路候选")

    report = IngestReport(
        collection=settings.QDRANT_COLLECTION,
        embedding_model=model.name,
        dim=model.dim,
        is_mock=model.is_mock,
    )

    if reset:
        # recreate 会删掉整个集合，--reset 的语义就是「从零重建」。
        await qdrant_index.prepare_collection(dim=model.dim, recreate=True)
        await neo4j_index.reset_graph()
        await _clear_registry(selected)
    else:
        await qdrant_index.prepare_collection(dim=model.dim)

    # --- 差分 ---
    plan = await diff(chunks, model, force=force or reset)
    log.info(
        "kb_diff",
        added=len(plan.added),
        updated=len(plan.updated),
        skipped=len(plan.skipped),
    )

    # --- 嵌入并写 Qdrant ---
    import time

    started = time.perf_counter()
    pairs = await embed_chunks(plan.to_embed, model, on_progress=on_progress)
    elapsed = time.perf_counter() - started
    await qdrant_index.upsert_chunks(pairs)

    # --- 写 Neo4j ---
    # 图不做差分：节点是 MERGE 出来的，重写一遍是廉价的，而「图缺了几条边」
    # 这种状态极难发现。全量重写是这里唯一值得的奢侈。
    await neo4j_index.write_chunks(chunks)
    edges = load_graph_edges(directory)  # type: ignore[arg-type]
    report.graph_edges = await neo4j_index.write_edges(edges)

    # --- 登记 ---
    # 按 chunk_id 取交集，不要用 `c in plan.to_embed`：KBChunk 是 dataclass，
    # `in` 走的是逐字段相等比较，既慢又可能在两条语料碰巧相同时误判。
    embedded_ids = {c.chunk_id for c in plan.to_embed}
    await record_documents(
        [c for c in chunks if c.chunk_id in embedded_ids],
        model,
        source_files={lib: _FILE_HINT.get(lib, "") for lib in selected},
    )

    # --- 清理语料里已消失的 chunk ---
    report.removed = await _prune_orphans(selected, {c.chunk_id for c in chunks})

    for lib in selected:
        report.stats[lib] = LibraryStat(
            library=lib,
            added=sum(1 for c in plan.added if c.library is lib),
            updated=sum(1 for c in plan.updated if c.library is lib),
            skipped=sum(1 for c in plan.skipped if c.library is lib),
            embed_seconds=elapsed,
        )

    report.qdrant_points = await qdrant_index_count()
    return report


_FILE_HINT: dict[Library, str] = {
    Library.PSYCHOLOGY: "psychology.jsonl",
    Library.LITERATURE: "literature.jsonl",
    Library.POETRY: "poetry.jsonl",
}


def corpus_source_files() -> list[str]:
    """三个语料文件名。

    **这是「语料」与「导入」的分界线**，被 `corpus_chunk_ids`（孤儿清理）
    和 `imported_chunk_count`（reset 警告）共用。两处必须用同一份判据——
    一边认为是语料、另一边认为是导入，后果是导入的内容被当成孤儿删掉。
    """
    return sorted(_FILE_HINT.values())


async def qdrant_index_count() -> int:
    from app.clients.qdrant_client import collection_count

    return await collection_count()


@dataclass(slots=True)
class IndexReport:
    """增量入库的结果。字段名与 `IngestReport` 对齐，便于复用同一套展示。"""

    added: int = 0
    updated: int = 0
    skipped: int = 0
    points: int = 0
    embedding_model: str = ""
    dim: int = 0


async def index_chunks(
    chunks: Sequence[KBChunk],
    model: EmbeddingModel,
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> IndexReport:
    """把一批**指定**的 chunk 写进三个存储，不碰其它任何东西。

    与 `ingest()` 的区别只有两条，但两条都是本质的：

    1. chunk 是**传进来的**，不是从磁盘语料读的。导入的内容没有 jsonl 文件，
       也不该有——它进了库之后，正文由 Qdrant payload 承载。
    2. **不做孤儿清理。** `_prune_orphans` 的前提是「语料文件是这一层内容的
       唯一权威」；导入场景下这个前提不成立，被删的会是整个库的手写语料。

    幂等性照旧：Qdrant 点 id 是 `uuid5(chunk_id)`，Neo4j 是 MERGE，登记表是
    merge。跑到一半失败就重跑。
    """
    if not chunks:
        return IndexReport(embedding_model=model.name, dim=model.dim)

    await qdrant_index.prepare_collection(dim=model.dim)
    plan = await diff(chunks, model)
    pairs = await embed_chunks(plan.to_embed, model, on_progress=on_progress)
    await qdrant_index.upsert_chunks(pairs)
    # 图不做差分——MERGE 幂等且重写廉价，而「图里少了几条边」极难发现。
    await neo4j_index.write_chunks(chunks)

    # 按 chunk_id 取交集，不要用 `c in plan.to_embed`：dataclass 的 `in`
    # 是逐字段相等比较，既慢又可能在两条内容碰巧相同时误判。
    embedded = {c.chunk_id for c in plan.to_embed}
    await record_documents([c for c in chunks if c.chunk_id in embedded], model)

    report = IndexReport(
        added=len(plan.added),
        updated=len(plan.updated),
        skipped=len(plan.skipped),
        points=await qdrant_index_count(),
        embedding_model=model.name,
        dim=model.dim,
    )
    log.info("kb_indexed", added=report.added, updated=report.updated, skipped=report.skipped)
    return report


async def imported_chunk_count(libraries: Sequence[Library] | None = None) -> int:
    """登记表里来自导入（`source_file` 不是语料文件）的 chunk 条数。

    `ingest-kb --reset` 会重建集合、清空图与登记表，导入的内容一并没掉。
    这个行为本身不改（`--reset` 的语义就是「从零重建」），但必须在执行前
    说清楚——否则用户只会在某天发现导入的书不见了，而那天离事故已经很远。
    """
    selected = list(libraries) if libraries else list(Library)
    corpus_files = corpus_source_files()
    stmt = (
        select(func.count())
        .select_from(KbDocument)
        .where(
            KbDocument.library.in_([str(lib) for lib in selected]),
            # NULL 走 `notin_` 会因三值逻辑被漏掉，必须显式并进来
            or_(KbDocument.source_file.is_(None), KbDocument.source_file.notin_(corpus_files)),
        )
    )
    async with get_sessionmaker()() as session:
        return int((await session.execute(stmt)).scalar_one())


async def _clear_registry(libraries: Sequence[Library]) -> None:
    """reset 时清掉登记表——登记表是差分的依据，重建集合却不重建登记表
    会让下一次运行以为所有 chunk 都没变，于是向量库永远空着。"""
    known = await load_hashes(libraries)
    await delete_documents(list(known))


async def corpus_chunk_ids(libraries: Sequence[Library]) -> set[str]:
    """登记表里「来源是语料文件」的 chunk_id。

    **导入进来的 chunk 也在这张表里，但它们不属于任何语料文件，必须排除在
    孤儿判定之外。** 否则下一次 `ingest-kb` 会把用户导入的整本书连同向量、
    图节点一起删掉，而屏幕上只留一行 `kb_pruned count=312` —— 用户看到的是
    「我明明导入过，怎么查不到了」，且没有任何线索指向这里。

    判据是 `source_file`（摄取时写的是三个语料文件名，导入时写的是原文件名）。
    **无法归类的一律当作非语料**：删除不可逆，留一行多余的登记不是。
    """
    stmt = select(KbDocument.chunk_id).where(
        KbDocument.library.in_([str(lib) for lib in libraries]),
        KbDocument.source_file.in_(corpus_source_files()),
    )
    async with get_sessionmaker()() as session:
        return set((await session.execute(stmt)).scalars().all())


async def _prune_orphans(libraries: Sequence[Library], corpus_ids: set[str]) -> int:
    """删掉「登记表里有、语料里没有」的 chunk。

    没有这一步，从语料里删掉一条典故之后，它仍然能被检索到——因为
    Qdrant 里的点不会自己消失。用户看到的是一个已经不存在于知识库里的
    结果，且无从解释。

    只管语料文件里的 chunk（见 `corpus_chunk_ids`），导入的内容不归这里管。
    """
    known = await corpus_chunk_ids(libraries)
    orphans = [cid for cid in known if cid not in corpus_ids]
    if not orphans:
        return 0
    await qdrant_index.delete_chunks(orphans)
    removed = await neo4j_index.delete_chunks(orphans)
    await delete_documents(orphans)
    log.info("kb_pruned", count=len(orphans), graph_nodes=removed)
    return len(orphans)


__all__ = [
    "CorpusError",
    "EmptyLibraryError",
    "IndexReport",
    "IngestReport",
    "LibraryStat",
    "corpus_chunk_ids",
    "corpus_source_files",
    "imported_chunk_count",
    "index_chunks",
    "ingest",
]
