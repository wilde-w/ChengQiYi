"""把用户上传的一个文本文件变成知识库内容。

流程：存盘 → 解码 → 切分 → 打标 → 嵌入 → 入库，全程把进度写进
`kb_import_job`。**进度落库而不是走 SSE**，和 `run_service` 是两套做法，
这是刻意的：导入是「等一两分钟看进度条」，不需要逐 token 的实时性；接
SSE 要新建端点、绕开 `run_event` 的外键约束、改前后端两处的终态集合。
落库还白拿一个好处——刷新页面进度不丢。

三个必须守住的行为：

1. **切分在创建时就跑一遍。** 「这本书多少段」必须在用户点确认之前就知道，
   而切分是纯字符串运算，跑两遍无所谓。真正执行时用同一个纯函数再跑一次，
   结果确定一致。
2. **大文件停下等确认，不静默截断。** 超过阈值时作业停在 `queued` 不动，
   前端弹确认后调 `start` 才真正开跑。停在 `queued` 的作业没有任何副作用，
   用户不确认就等于放弃。
3. **用户提供的字符绝不拼进文件路径。** 落盘名是 `<job_id>.txt`，
   原始文件名只作为 `origin` 的值进库。这条不做，一个叫
   `../../.env` 的上传就能写到任意位置。
4. **epub 在入口处就解包成正文。** 落到磁盘的是抽出来的文本而不是那包
   zip，于是「这个文件读不出正文」永远发生在**建作业之前**——用户当场
   看到原因，而不是作业跑到一半失败；`execute_import` 因此不必认识 epub，
   它做的 decode + split 与纯文本路径完全一致。
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import delete, desc, func, select

from app.config import get_settings
from app.constants import ImportStage, Library, RunStatus
from app.db.session import session_scope
from app.kb import epub, neo4j_index, qdrant_index
from app.kb.chunker import EmptyDocumentError, decode_bytes, split_document
from app.kb.ingest import corpus_source_files, index_chunks
from app.kb.lexical_tags import build_frequency_table
from app.kb.tagging import TaggingCancelled, TaggingError, tag_pieces
from app.logging_conf import get_logger
from app.models.base import new_id
from app.models.library import KbDocument, KbImportJob

log = get_logger(__name__)

#: 原始文件落盘目录。gitignore 掉了——这是用户数据，不是仓库内容。
IMPORT_DIR = Path(__file__).resolve().parents[1] / "kb" / "imports"

_tasks: dict[str, asyncio.Task[Any]] = {}
_semaphore: asyncio.Semaphore | None = None
#: 请求取消的作业 id。和 `run_service` 一样只置标志位，由执行侧在阶段
#: 边界自行退出——强行 cancel() 会跳过终态写入，前端就永远等不到结果。
_cancel_requested: set[str] = set()

#: 进度带。打标占一半以上，因为它是唯一逐块调模型的一步。
_BANDS: dict[ImportStage, tuple[int, int]] = {
    ImportStage.READING: (0, 5),
    ImportStage.CHUNKING: (5, 15),
    ImportStage.TAGGING: (15, 70),
    ImportStage.EMBEDDING: (70, 90),
    ImportStage.INDEXING: (90, 100),
}

_UNSAFE_NAME = re.compile(r"[\x00-\x1f\x7f/\\:*?\"<>|]")


class ImportNotFound(LookupError):
    pass


class ImportRejected(ValueError):
    """用户可修的输入问题。API 应回 400，且消息要能直接展示。"""


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(max(1, get_settings().KB_IMPORT_CONCURRENCY))
    return _semaphore


def safe_filename(raw: str) -> str:
    """清成可安全展示与落库的文件名。

    **这个函数的返回值只用于展示和 `origin` 字段，不参与任何路径拼接。**
    即便如此也要清干净：`origin` 会出现在 API 响应和前端列表里。
    """
    name = os.path.basename(raw.replace("\\", "/")).strip()
    name = _UNSAFE_NAME.sub("", name)
    return name[:200] or "未命名.txt"


def _stored_path(job_id: str) -> Path:
    return IMPORT_DIR / f"{job_id}.txt"


def _read_upload(blob: bytes, *, title: str, author: str) -> tuple[str, str, str, list[str]]:
    """上传字节 → `(正文, 书名, 作者, 附加提示)`。

    epub 走解包抽取，书名/作者留空时用电子书自带的元数据补；其余一律按
    UTF-8/GBK 解码。**这里返回的正文就是落盘的那一份**（见模块 docstring 第 4 条）。
    """
    if not epub.is_zip(blob):
        return decode_bytes(blob), title, author, []

    try:
        doc = epub.read_epub(blob)
    except epub.NotZipError:
        # 头两个字节恰好是 "PK" 的纯文本。按文本再试一次，别误杀。
        return decode_bytes(blob), title, author, []
    except epub.EpubError as exc:
        raise ImportRejected(str(exc)) from exc

    return doc.text, title or doc.title, author or doc.author, _epub_notes(doc)


def _epub_notes(doc: epub.EpubDocument) -> list[str]:
    """把「跳过了什么」说成一句人话。**不静默**：用户该知道正文缺了哪些页。"""
    notes: list[str] = []
    if doc.skipped:
        head = "、".join(doc.skipped[:3]) + ("…" if len(doc.skipped) > 3 else "")
        notes.append(f"跳过 {len(doc.skipped)} 篇封面/目录/版权这类页面（{head}）")
    if doc.empty_docs:
        notes.append(f"另有 {doc.empty_docs} 篇没有文字（纯图片页），已跳过")
    return notes


@dataclass(slots=True)
class CreateResult:
    job: KbImportJob
    #: False 表示作业停在 `queued` 等用户确认（文件太大）。
    auto_started: bool
    warnings: list[str]


# ----------------------------------------------------------------------
# 创建
# ----------------------------------------------------------------------


async def create_import(
    blob: bytes,
    *,
    filename: str,
    library: Library,
    title: str = "",
    author: str = "",
    discipline: str = "",
    use_llm: bool = True,
    allow_partial: bool = False,
) -> CreateResult:
    """校验 + 切分 + 建作业。返回时**可能已经开始跑了**。"""
    settings = get_settings()
    name = safe_filename(filename)

    if not blob:
        raise ImportRejected("文件是空的")
    # 体积闸门按格式选尺子：epub 里大半是图片，用纯文本那条 5MB 量它，
    # 一本正常的书都过不来（见 config.py 里两个上限的注释）。
    zipped = epub.is_zip(blob)
    byte_cap = settings.KB_IMPORT_EPUB_MAX_BYTES if zipped else settings.KB_IMPORT_MAX_BYTES
    if len(blob) > byte_cap:
        raise ImportRejected(
            f"文件 {len(blob) / 1024 / 1024:.1f}MB 超过 "
            f"{byte_cap / 1024 / 1024:.0f}MB 上限。"
            f"{'电子书' if zipped else '纯文本'}一般不会有这么大，请确认没有选错文件。"
        )

    try:
        text, title, author, notes = _read_upload(blob, title=title, author=author)
    except EmptyDocumentError as exc:
        raise ImportRejected(str(exc)) from exc

    try:
        report = split_document(text)
    except EmptyDocumentError as exc:
        raise ImportRejected(str(exc)) from exc

    if len(report.pieces) > settings.KB_IMPORT_MAX_CHUNKS:
        raise ImportRejected(
            f"切出 {len(report.pieces)} 段，超过 {settings.KB_IMPORT_MAX_CHUNKS} 段上限。"
            "请拆分后再导入，或调高 KB_IMPORT_MAX_CHUNKS。"
        )

    job_id = new_id()
    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    # 落盘的是**抽取后的正文**而不是原始上传：epub 已经在这里解包成文本，
    # `execute_import` 重读它时只需 decode + split，与纯文本导入同一条路。
    _stored_path(job_id).write_bytes(text.encode("utf-8"))

    needs_confirm = len(report.pieces) > settings.KB_IMPORT_CONFIRM_CHUNKS
    job = KbImportJob(
        id=job_id,
        library=str(library),
        filename=name,
        file_size=len(blob),
        status=RunStatus.QUEUED.value,
        stage=None,
        progress=0,
        message="等待确认" if needs_confirm else "已排队",
        chunk_total=len(report.pieces),
        warnings=list(report.warnings),
        options={
            "title": title,
            "author": author,
            # 只在建作业时知道的事（这本书跳过了哪几篇封面/目录、有几篇是纯图片），
            # 存进 options 让 `execute_import` 收尾时并进 warnings。**不能只写进
            # 上面的 warnings**：执行侧会用切分报告重写那一栏，建作业时写的东西
            # 活不到用户看到的那一刻。
            "notes": notes,
            "discipline": discipline,
            "use_llm": use_llm,
            "allow_partial": allow_partial,
            "needs_confirm": needs_confirm,
        },
    )
    async with session_scope() as session:
        session.add(job)

    if not needs_confirm:
        _spawn(job_id)
    log.info(
        "kb_import_created",
        job_id=job_id,
        library=str(library),
        chunks=len(report.pieces),
        bytes=len(blob),
        needs_confirm=needs_confirm,
    )
    return CreateResult(job=job, auto_started=not needs_confirm, warnings=list(report.warnings))


def _spawn(job_id: str) -> None:
    existing = _tasks.get(job_id)
    if existing is not None and not existing.done():
        return
    task = asyncio.create_task(_guarded(job_id), name=f"kb-import-{job_id}")
    _tasks[job_id] = task
    task.add_done_callback(lambda _t: _tasks.pop(job_id, None))


async def _guarded(job_id: str) -> None:
    async with _get_semaphore():
        try:
            await execute_import(job_id)
        except asyncio.CancelledError:
            log.info("kb_import_task_cancelled", job_id=job_id)
            raise
        except Exception as exc:
            log.exception("kb_import_task_crashed", job_id=job_id)
            # 兜底：执行器在写终态之前就抛了（作业行不存在、终态写入本身失败）。
            # 不补这一刀，作业行会永远停在 `running`，前端进度条转到天荒地老，
            # 而日志里没有任何一行指向它。
            with contextlib.suppress(Exception):
                await _update(
                    job_id,
                    status=RunStatus.FAILED.value,
                    message="导入中断",
                    error=f"{type(exc).__name__}: {exc}",
                    finished_at=datetime.now(UTC),
                )


async def start_import(job_id: str) -> KbImportJob:
    """放行一个停在 `queued` 的大文件作业。"""
    job = await get_import(job_id)
    if job.status != RunStatus.QUEUED.value:
        raise ImportRejected(f"作业当前状态是 {job.status}，不能重复启动")
    _cancel_requested.discard(job_id)
    _spawn(job_id)
    return job


# ----------------------------------------------------------------------
# 执行
# ----------------------------------------------------------------------


async def _update(job_id: str, **fields: Any) -> None:
    async with session_scope() as session:
        job = await session.get(KbImportJob, job_id)
        if job is None:
            return
        for key, value in fields.items():
            setattr(job, key, value)


def _cancelled(job_id: str) -> bool:
    return job_id in _cancel_requested


def _band(stage: ImportStage, ratio: float) -> int:
    low, high = _BANDS[stage]
    return int(low + (high - low) * min(1.0, max(0.0, ratio)))


#: 阶段内的实时进度，只活在内存里。
#:
#: **`embed_chunks` / `tag_pieces` 的 `on_progress` 是同步回调**，没法 await
#: 一个数据库写。而回调里 `asyncio.create_task(...)` 搞 fire-and-forget 更糟：
#: 那些写会与终态写入竞争，一个晚到的「进度 85%」会盖掉刚刚写好的
#: `status=succeeded`。所以回调只改这个字典，由**唯一的心跳任务**负责落库——
#: 单写入者，没有竞态。
_progress: dict[str, dict[str, Any]] = {}


def _note(job_id: str, stage: ImportStage, ratio: float, message: str) -> None:
    _progress[job_id] = {
        "stage": stage.value,
        "progress": _band(stage, ratio),
        "message": message,
    }


async def _heartbeat(job_id: str) -> None:
    """每 0.8 秒把内存里的进度写一次库。前端 1 秒轮询，这个节奏够用。"""
    try:
        while True:
            await asyncio.sleep(0.8)
            snapshot = _progress.get(job_id)
            if snapshot:
                await _update(job_id, **snapshot)
    except asyncio.CancelledError:
        raise


async def execute_import(job_id: str) -> RunStatus:
    """跑完一次导入。终态一定写回作业行——**失败也必须留下原因**。

    结构上刻意只有一个终态写入点（函数末尾），心跳任务在它之前停下。
    散落各处的 `await _update(status=...)` 迟早会和心跳抢同一行，
    而抢输的那个会静默覆盖结果。
    """
    async with session_scope() as session:
        job = await session.get(KbImportJob, job_id)
        if job is None:
            raise ImportNotFound(job_id)
        library = Library(job.library)
        options = dict(job.options or {})
        filename = job.filename

    title = str(options.get("title") or "")
    author = str(options.get("author") or "")
    discipline = str(options.get("discipline") or "")
    use_llm = bool(options.get("use_llm", True))
    allow_partial = bool(options.get("allow_partial", False))
    # 建作业时就定下、执行侧算不出来的提示（epub 跳过了哪几篇、几篇纯图片）。
    notes = [str(n) for n in (options.get("notes") or [])]

    await _update(
        job_id,
        status=RunStatus.RUNNING.value,
        stage=ImportStage.READING.value,
        progress=0,
        message="读取文件",
        started_at=datetime.now(UTC),
        error=None,
    )
    beat = asyncio.create_task(_heartbeat(job_id), name=f"kb-import-beat-{job_id}")

    status = RunStatus.SUCCEEDED
    outcome: dict[str, Any] = {"stage": ImportStage.INDEXING.value, "message": "已完成"}

    try:
        # --- 1. 取回并解码（重读落盘的文件，不把正文留在内存里跨请求）---
        text = decode_bytes(_stored_path(job_id).read_bytes())
        if _cancelled(job_id):
            raise TaggingCancelled("用户取消了这次导入")

        # --- 2. 切分（与创建时同一份纯函数，结果确定一致）---
        _note(job_id, ImportStage.CHUNKING, 0.0, "切分正文")
        report = split_document(text)
        chunks_total = len(report.pieces)

        # --- 3. 打标 ---
        _note(
            job_id, ImportStage.TAGGING, 0.0,
            "AI 抽取标签" if use_llm else "规则抽取标签（未使用 AI）",
        )
        freq = build_frequency_table(text)

        def _on_tagged(done: int, total: int) -> None:
            _note(job_id, ImportStage.TAGGING, done / max(1, total), f"已打标 {done}/{total} 批")

        chunks, tag_report = await tag_pieces(
            report.pieces,
            library=library,
            model=_chat_model() if use_llm else None,
            work=title,
            author=author,
            discipline=discipline,
            origin=filename,
            freq=freq,
            use_llm=use_llm,
            allow_partial=allow_partial,
            batch_size=get_settings().KB_TAG_BATCH_SIZE,
            on_progress=_on_tagged,
            is_cancelled=lambda: _cancelled(job_id),
        )

        if _cancelled(job_id):
            raise TaggingCancelled("用户取消了这次导入")
        if not chunks:
            raise ImportRejected(
                "切分出的所有片段都没有可用标注，没有内容可以入库。"
                "若是文学/诗词，通常是这段文字里找不到可标注的意象或情绪。"
            )

        # --- 4. 入库（嵌入 + 写三个存储）---
        # `index_chunks` 把嵌入与索引合在一步里，进度回调覆盖的是前半段，
        # 收尾时再推进到 INDEXING 的终点。
        _note(job_id, ImportStage.EMBEDDING, 0.0, "生成向量")

        def _on_embedded(done: int, total: int) -> None:
            _note(job_id, ImportStage.EMBEDDING, done / max(1, total), f"已生成向量 {done}/{total}")

        index_report = await index_chunks(chunks, _embedding_model(), on_progress=_on_embedded)

        warnings = notes + list(report.warnings) + list(tag_report.warnings)
        outcome = {
            "stage": ImportStage.INDEXING.value,
            "chunk_total": chunks_total,
            "chunk_tagged": tag_report.tagged,
            "chunk_indexed": len(chunks),
            "warnings": warnings,
            "message": f"已入库 {len(chunks)} 条",
        }
        log.info(
            "kb_import_done",
            job_id=job_id,
            chunks=len(chunks),
            added=index_report.added,
            skipped=index_report.skipped,
        )

    except TaggingCancelled as exc:
        status = RunStatus.CANCELLED
        outcome = {"message": str(exc)}
    except TaggingError as exc:
        # **打标大面积失败：写任何东西都是错的。**
        # 半本只有词法标签的内容入库后会被每次检索当作噪声证据召回，
        # 而回滚要手工按 work 清。什么都没写是可恢复的。
        status = RunStatus.FAILED
        outcome = {"message": "打标失败，未写入任何内容", "error": str(exc)}
    except (ImportRejected, EmptyDocumentError) as exc:
        status = RunStatus.FAILED
        outcome = {"message": str(exc), "error": str(exc)}
    except Exception as exc:
        log.exception("kb_import_failed", job_id=job_id)
        status = RunStatus.FAILED
        outcome = {"message": "导入失败", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        beat.cancel()
        await asyncio.gather(beat, return_exceptions=True)
        _progress.pop(job_id, None)
        _cancel_requested.discard(job_id)

    terminal: dict[str, Any] = {
        "status": status.value,
        "finished_at": datetime.now(UTC),
        **outcome,
    }
    if status is RunStatus.SUCCEEDED:
        # **只在成功时写 progress。** 这一列是 NOT NULL：失败路径上传
        # `progress=None` 会在提交时炸开，而炸的位置在终态写入里——
        # 作业行就永远停在 `running`，前端的进度条转到天荒地老，
        # 日志里只有一条 IntegrityError。失败时保留最后一次阶段进度即可。
        terminal["progress"] = 100
    await _update(job_id, **terminal)
    return status


def _chat_model() -> Any:
    from app.providers.factory import get_chat_model

    return get_chat_model()


def _embedding_model() -> Any:
    from app.providers.factory import get_embedding_model

    return get_embedding_model()


# ----------------------------------------------------------------------
# 查询
# ----------------------------------------------------------------------


async def reap_stale_imports() -> int:
    """把上一次进程死掉时卡在 `running` 的作业标成失败。**启动时调一次。**

    **进程被杀不会写终态**，那一行就会永远停在 `running`：前端进度条无限转，
    而且没有任何线索指向「服务重启过」。优雅关闭（`shutdown_import_tasks`）
    覆盖不到 kill -9 和崩溃，只有启动时的这一刀能覆盖。

    **不碰 `queued`。** 停在 `queued` 等确认的作业是合法状态，它的文件还在
    磁盘上、还没调用过任何模型，重启后用户点「继续导入」就该照常开跑。
    把它的状态改掉，等于替用户做了「放弃」的决定。
    """
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(KbImportJob).where(KbImportJob.status == RunStatus.RUNNING.value)
            )
        ).scalars()
        count = 0
        for job in rows:
            job.status = RunStatus.FAILED.value
            job.message = "服务重启，导入中断"
            job.error = "进程在导入完成前退出"
            job.finished_at = datetime.now(UTC)
            count += 1
    if count:
        log.warning("kb_import_reaped", count=count, hint="上次运行留下的未完成作业已标记为失败")
    return count


async def get_import(job_id: str) -> KbImportJob:
    async with session_scope() as session:
        job = await session.get(KbImportJob, job_id)
        if job is None:
            raise ImportNotFound(job_id)
        session.expunge(job)
        return job


async def list_imports(*, limit: int = 20) -> list[KbImportJob]:
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(KbImportJob).order_by(desc(KbImportJob.created_at)).limit(limit)
            )
        ).scalars()
        return list(rows)


@dataclass(slots=True)
class SourceSummary:
    source_file: str
    library: str
    chunks: int
    updated_at: datetime | None


async def list_sources() -> list[SourceSummary]:
    """已导入的来源，按 chunk 数聚合。

    **只列导入的来源。** 三个语料文件名被显式排除——它们是随仓库交付的，
    列在这里会让人以为可以删，而删掉之后下一次 `ingest-kb` 又会把它们
    装回来，看上去像删除按钮坏了。
    """
    corpus = corpus_source_files()
    stmt = (
        select(
            KbDocument.source_file,
            KbDocument.library,
            func.count().label("chunks"),
            func.max(KbDocument.updated_at).label("updated_at"),
        )
        .where(
            KbDocument.source_file.is_not(None),
            KbDocument.source_file.notin_(corpus),
        )
        .group_by(KbDocument.source_file, KbDocument.library)
        .order_by(desc("updated_at"))
    )
    async with session_scope() as session:
        rows = (await session.execute(stmt)).all()
    return [
        SourceSummary(
            source_file=str(row.source_file),
            library=str(row.library),
            chunks=int(row.chunks),
            updated_at=row.updated_at,
        )
        for row in rows
    ]


async def source_documents(source_file: str, *, limit: int = 200) -> list[KbDocument]:
    """某个来源切出来的 chunk 预览。

    `kb_document.text_preview` 在入库时就存好了，所以这一步几乎免费。
    这是用户**唯一能直接看到「RAG 到底切出了什么」**的地方——没有它，
    切分质量只能靠猜。
    """
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(KbDocument)
                .where(KbDocument.source_file == source_file)
                .order_by(KbDocument.chunk_id)
                .limit(limit)
            )
        ).scalars()
        return list(rows)


@dataclass(slots=True)
class SourceText:
    """某个来源的原文。``text`` 是按 ``offset`` 切出来的一段，不是全文。"""

    source_file: str
    title: str | None
    author: str | None
    library: str
    #: **全文**字符数（不是本段的），前端据此显示「已加载 N / M 字」。
    char_count: int
    offset: int
    truncated: bool
    text: str


async def source_text(source_file: str, *, offset: int = 0, limit: int = 100_000) -> SourceText:
    """某个来源的**原文全文**（不是切块预览）。

    **读导入时落盘的那份文件**（`IMPORT_DIR/{job_id}.txt`），不从 Qdrant 拼
    chunk：chunk 之间没有任何序号字段，`kb_document` 又是按 chunk_id 哈希排序的，
    拼出来的「原文」顺序是乱的。落盘的那份才是真原文（epub 落的也是抽出来的
    正文，不是 zip），而且零新增存储。

    `filename` 就是 `kb_document.source_file`（见 `models/library.py` 的注释），
    所以「来源」和「作业」之间不需要额外的映射表。同名文件重复导入会有多条
    作业，取**最后一次成功的**：用户看到的是现在这份内容，不是上一版。
    """
    async with session_scope() as session:
        job = (
            await session.execute(
                select(KbImportJob)
                .where(
                    KbImportJob.filename == source_file,
                    KbImportJob.status == RunStatus.SUCCEEDED.value,
                )
                .order_by(desc(KbImportJob.created_at))
                .limit(1)
            )
        ).scalars().first()

    if job is None:
        # 两种「没有」要分开说：一种是这个来源根本不来自界面导入
        #（例如随仓库交付的语料），另一种是导入记录被清理过。
        raise ImportNotFound(
            "这个来源不是从界面导入的，没有留存原文。"
            "只有通过「＋ 导入」上传的文件才能查阅原文。"
        )

    path = _stored_path(job.id)
    if not path.exists():
        raise ImportNotFound(
            f"「{source_file}」的导入记录还在，但落盘的正文文件已经不在了"
            f"（{path.name}）。重新导入一次即可恢复。"
        )

    full = path.read_text(encoding="utf-8")
    options = job.options or {}
    return SourceText(
        source_file=source_file,
        title=_opt_str(options.get("title")),
        author=_opt_str(options.get("author")),
        library=str(job.library),
        char_count=len(full),
        offset=offset,
        truncated=offset + limit < len(full),
        text=full[offset : offset + limit],
    )


def _opt_str(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


# ----------------------------------------------------------------------
# 取消与删除
# ----------------------------------------------------------------------


async def request_cancel(job_id: str) -> bool:
    job = await get_import(job_id)
    status = RunStatus(job.status)
    if status.is_terminal:
        return False
    _cancel_requested.add(job_id)
    if status is RunStatus.QUEUED:
        # 还没开跑的作业直接落终态，不必等一个永远不存在的检查点。
        await _update(
            job_id, status=RunStatus.CANCELLED.value, message="已取消",
            finished_at=datetime.now(UTC),
        )
    return True


async def delete_source(source_file: str) -> int:
    """回滚一次导入：登记表 → Qdrant → Neo4j。

    顺序和摄取时**相反**（先删登记表）。原因和摄取时一样：登记表是权威，
    先删它，中途失败时剩下的孤儿是「登记表里没有、向量库里还有」——
    下一次导入同名文件会按 chunk_id 原地覆盖，能自愈；反过来则会留下
    「登记表说有、实际查不到」的幽灵行，那才会一直骗人。
    """
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(KbDocument.chunk_id).where(KbDocument.source_file == source_file)
            )
        ).scalars()
        chunk_ids = list(rows)
        if not chunk_ids:
            return 0
        await session.execute(delete(KbDocument).where(KbDocument.chunk_id.in_(chunk_ids)))

    await qdrant_index.delete_chunks(chunk_ids)
    await neo4j_index.delete_chunks(chunk_ids)
    log.info("kb_source_deleted", source_file=source_file, chunks=len(chunk_ids))
    return len(chunk_ids)


def active_task_count() -> int:
    return sum(1 for t in _tasks.values() if not t.done())


async def shutdown_import_tasks() -> None:
    """应用关闭时收敛在跑的导入。

    **必须在 `shutdown_tasks()` 之后调用**，也就是在拆连接池之前——
    顺序反了的话，正在提交产物的导入会在关闭连接的过程中抛一串噪音异常，
    把真正的关闭日志淹掉。
    """
    pending = [t for t in _tasks.values() if not t.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    _tasks.clear()
    _cancel_requested.clear()


__all__ = [
    "IMPORT_DIR",
    "CreateResult",
    "ImportNotFound",
    "ImportRejected",
    "SourceSummary",
    "SourceText",
    "active_task_count",
    "create_import",
    "delete_source",
    "execute_import",
    "get_import",
    "list_imports",
    "list_sources",
    "reap_stale_imports",
    "request_cancel",
    "safe_filename",
    "shutdown_import_tasks",
    "source_documents",
    "source_text",
    "start_import",
]
