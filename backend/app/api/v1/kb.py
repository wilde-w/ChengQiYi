"""知识库导入接口：上传、进度、来源列表、回滚。

**用轮询而非 SSE。** 导入是「等一两分钟看进度条」，不需要逐 token 的实时性；
接 SSE 要新建端点、绕开 `run_event` 的外键约束、改前后端两处的终态事件集合。
轮询的代码量约为它的四分之一，而且进度落库之后刷新页面不丢。

路由薄、service 厚，领域异常在这里翻成 HTTP 状态码——和 `runs.py` 一致。
ORM → DTO 的映射函数放在文件底部，一眼能看到线上契约是什么。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from app.config import get_settings
from app.constants import ImportStage, Library, RunStatus
from app.kb import epub
from app.logging_conf import get_logger
from app.schemas.kb import (
    KbDeleteResult,
    KbDocumentOut,
    KbImportCreated,
    KbImportJobOut,
    KbSourceOut,
    KbSourceTextOut,
)
from app.services import kb_import_service

log = get_logger(__name__)
router = APIRouter(tags=["kb"])


@router.post("/kb/imports", response_model=KbImportCreated, status_code=201)
async def create_import(
    file: UploadFile = File(..., description="纯文本文件（UTF-8 / GBK 均可）或 epub 电子书"),
    library: Library = Form(Library.PSYCHOLOGY, description="归入哪个库"),
    title: str = Form("", description="书名 / 篇名，作为 work 落库"),
    author: str = Form(""),
    discipline: str = Form("", description="领域，仅 psychology 库使用"),
    use_llm: bool = Form(True, description="关掉则只做规则抽取"),
    allow_partial: bool = Form(False, description="打标大面积失败时仍入库"),
) -> Any:
    """上传一个文本文件并入库。

    **校验与切分在返回前同步做完**：「这本书多少段」必须在用户点确认之前
    就知道，而且输入不合法应当立刻 400，而不是先建一条作业再让它悄悄失败。
    切分是纯字符串运算，跑两遍无所谓。
    """
    settings = get_settings()
    # **有界读取。** FastAPI 的 UploadFile 超过 1MB 会落到临时文件，所以
    # 这里多读一个字节只是为了判断「超没超」，不会把 500MB 拉进内存——
    # 否则一个误传的视频文件就能把进程撑爆，而那本该是一个 400。
    # 先闻头两个字节只为**选哪把尺子**（epub 的上限比纯文本宽得多），
    # 字节一个不丢，接着读完。
    head = await file.read(2)
    zipped = epub.is_zip(head)
    limit = settings.KB_IMPORT_EPUB_MAX_BYTES if zipped else settings.KB_IMPORT_MAX_BYTES
    blob = head + await file.read(limit + 1 - len(head))
    await file.close()
    if len(blob) > limit:
        # 在这里拦一道而不是全交给 service：service 只看得到被截断的字节，
        # 报出来的会是「文件 5.0MB 超过 5MB 上限」这种自相矛盾的话。
        raise HTTPException(
            status_code=400,
            detail=f"文件超过 {limit / 1024 / 1024:.0f}MB 上限。"
            f"{'电子书' if zipped else '纯文本'}一般不会有这么大，请确认没有选错文件。",
        )

    try:
        result = await kb_import_service.create_import(
            blob,
            filename=file.filename or "",
            library=library,
            title=title.strip(),
            author=author.strip(),
            discipline=discipline.strip(),
            use_llm=use_llm,
            allow_partial=allow_partial,
        )
    except kb_import_service.ImportRejected as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "job": _job(result.job),
        "auto_started": result.auto_started,
        "warnings": result.warnings,
        "demo": settings.is_demo,
    }


@router.get("/kb/imports", response_model=list[KbImportJobOut])
async def list_imports(limit: int = Query(default=20, ge=1, le=100)) -> Any:
    return [_job(job) for job in await kb_import_service.list_imports(limit=limit)]


@router.get("/kb/imports/{job_id}", response_model=KbImportJobOut)
async def get_import(job_id: str) -> Any:
    """作业状态。前端 1 秒轮询一次，`is_terminal` 为真就停。"""
    try:
        return _job(await kb_import_service.get_import(job_id))
    except kb_import_service.ImportNotFound as exc:
        raise HTTPException(status_code=404, detail="导入作业不存在") from exc


@router.post("/kb/imports/{job_id}/start", response_model=KbImportJobOut)
async def start_import(job_id: str) -> Any:
    """放行一个停在 `queued` 的作业——用户在确认框里点了「继续导入」。

    不确认的作业就停在 `queued`，**没有任何副作用**：文件留在磁盘上，
    一次模型调用都不会发生。
    """
    try:
        job = await kb_import_service.start_import(job_id)
    except kb_import_service.ImportNotFound as exc:
        raise HTTPException(status_code=404, detail="导入作业不存在") from exc
    except kb_import_service.ImportRejected as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _job(job)


@router.post("/kb/imports/{job_id}/cancel", status_code=202)
async def cancel_import(job_id: str) -> dict[str, Any]:
    """请求取消。**置标志位而非 kill 任务**，让执行侧在阶段边界收尾。

    立刻强杀会跳过终态写入，作业行就永远停在 `running`。
    """
    try:
        accepted = await kb_import_service.request_cancel(job_id)
    except kb_import_service.ImportNotFound as exc:
        raise HTTPException(status_code=404, detail="导入作业不存在") from exc
    return {
        "accepted": accepted,
        "message": "已请求取消，将在当前阶段结束后停止" if accepted else "作业已结束",
    }


@router.get("/kb/sources", response_model=list[KbSourceOut])
async def list_sources() -> Any:
    """已导入的来源。**只列导入的内容**，三个随仓库交付的语料文件不在内
    ——列在这里会让人以为可以删，而删掉之后下一次 `ingest-kb` 又会装回来。"""
    return [_source(s) for s in await kb_import_service.list_sources()]


@router.get("/kb/sources/{source_file}/documents", response_model=list[KbDocumentOut])
async def source_documents(
    source_file: str,
    limit: int = Query(default=200, ge=1, le=1000),
) -> Any:
    """某个来源切出来的片段预览。这是用户唯一能直接看到「RAG 切出了什么」的地方。"""
    rows = await kb_import_service.source_documents(source_file, limit=limit)
    return [_document(row) for row in rows]


@router.get("/kb/sources/{source_file}/text", response_model=KbSourceTextOut)
async def source_text(
    source_file: str,
    offset: int = Query(default=0, ge=0, description="从第几个字符开始"),
    limit: int = Query(
        default=100_000, ge=1_000, le=500_000, description="这一段最多多少字"
    ),
) -> Any:
    """某个来源的**原文**全文。

    和上面 `/documents` 的分工：那条答的是「RAG 切出了什么」（120 字预览 +
    chunk_id，用来判断切分质量），这条答的是「我导入的到底是本书吗」
    （抽取后的正文本身）。两条都要留着，它们回答的不是同一个问题。

    按**字符**分页而不是按行：原文的行长差异极大（对话一行十几个字，
    叙述一段几百字），按行分页会让每次返回的体积忽大忽小。
    """
    try:
        result = await kb_import_service.source_text(
            source_file, offset=offset, limit=limit
        )
    except kb_import_service.ImportNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "source_file": result.source_file,
        "title": result.title,
        "author": result.author,
        "library": result.library,
        "char_count": result.char_count,
        "offset": result.offset,
        "truncated": result.truncated,
        "text": result.text,
    }


@router.delete("/kb/sources/{source_file}", response_model=KbDeleteResult)
async def delete_source(source_file: str) -> Any:
    """回滚一次导入：登记表 → Qdrant → Neo4j 三处一起删。

    幂等：已经删过的来源再删一次返回 0，不是 404——用户点两下删除按钮
    或者列表页落后一拍，都该得到「现在确实没有了」而不是一个报错。
    """
    removed = await kb_import_service.delete_source(source_file)
    return {
        "source_file": source_file,
        "removed": removed,
        "message": f"已删除 {removed} 条" if removed else "没有找到该来源的内容",
    }


# ----------------------------------------------------------------------
# ORM → DTO
# ----------------------------------------------------------------------


def _job(row: Any) -> dict[str, Any]:
    options = dict(row.options or {})
    status = str(row.status)
    try:
        stage_label = ImportStage(row.stage).label if row.stage else None
    except ValueError:
        # 库里存着代码里已经没有的阶段名（回滚过一版代码）。展示原始值，
        # 不要为了一个标签把整个轮询打成 500。
        stage_label = row.stage
    try:
        library = Library(row.library)
        library_label = library.label
    except ValueError:
        library_label = str(row.library)
    return {
        "id": row.id,
        "library": row.library,
        "library_label": library_label,
        "filename": row.filename,
        "file_size": int(row.file_size or 0),
        "status": status,
        "stage": row.stage,
        "stage_label": stage_label,
        "progress": int(row.progress or 0),
        "message": row.message,
        "chunk_total": int(row.chunk_total or 0),
        "chunk_tagged": int(row.chunk_tagged or 0),
        "chunk_indexed": int(row.chunk_indexed or 0),
        "warnings": [str(w) for w in (row.warnings or [])],
        "error": row.error,
        "is_terminal": _is_terminal(status),
        # 只在还停在 queued 时算「等确认」：用户点了继续之后状态就变了，
        # 而取消掉的作业是终态——两种情况下都不该再弹确认框。
        "needs_confirm": status == RunStatus.QUEUED.value and bool(options.get("needs_confirm")),
        "created_at": row.created_at,
        "started_at": row.started_at,
        "finished_at": row.finished_at,
    }


def _is_terminal(status: str) -> bool:
    try:
        return RunStatus(status).is_terminal
    except ValueError:
        # 认不出的状态一律当终态：让前端停下来，总好过永远轮询。
        return True


def _source(row: Any) -> dict[str, Any]:
    try:
        library_label = Library(row.library).label
    except ValueError:
        library_label = str(row.library)
    return {
        "source_file": row.source_file,
        "library": row.library,
        "library_label": library_label,
        "chunks": int(row.chunks or 0),
        "updated_at": row.updated_at,
    }


def _document(row: Any) -> dict[str, Any]:
    return {
        "chunk_id": row.chunk_id,
        "library": row.library,
        "source_file": row.source_file,
        "text_preview": row.text_preview or "",
        "updated_at": row.updated_at,
    }
