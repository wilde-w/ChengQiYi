"""古典文学知识库查阅读口（另一个仓的 MCP）。

**不进分析流水线。** 这里是顶栏一个平级的「查阅」入口：没有 run_id、
不发事件、不落库，一次请求 = 一次 MCP 调用 = 一段 Markdown。

路由薄、service 厚，领域异常在这里翻成 HTTP 状态码——和 `runs.py` 一致。
唯一的差别是这个模块的失败大多是**环境问题**（对方仓库不在、解释器路径变了），
所以错误码要能区分「配置错」和「对方报错」，见 `_http()`。
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.config import get_settings
from app.logging_conf import get_logger
from app.novel import service as novel_service
from app.novel.mcp_client import NovelMcpError
from app.novel.service import NovelText
from app.schemas.novel import NovelHealthOut, NovelTextOut

log = get_logger(__name__)
router = APIRouter(tags=["novel"])

#: 与对方 MCP 的 `mcp_max_limit` 对齐。前端传大了这边先拦，省一次 1.7s 的子进程。
_MAX_LIMIT = 100


def _http(exc: NovelMcpError) -> HTTPException:
    """MCP 的失败 → HTTP 状态码。

    503/504/502 三档在**前端表现一样**（都显示 `detail`），分档是给运维看的：
    「解释器路径变了」和「对方库锁住了」不该在日志里长得一样。
    """
    status = {"unavailable": 503, "timeout": 504}.get(exc.kind, 502)
    return HTTPException(status_code=status, detail=str(exc))


async def _text(coro: Awaitable[NovelText]) -> dict[str, Any]:
    """service 的结果 → DTO，顺带把 MCP 异常翻成 HTTP。"""
    try:
        result = await coro
    except NovelMcpError as exc:
        log.warning("novel_mcp_failed", kind=exc.kind, error=str(exc))
        raise _http(exc) from exc
    return {"tool": result.tool, "markdown": result.markdown}


# ----------------------------------------------------------------------
# 状态
# ----------------------------------------------------------------------


@router.get("/novel/health", response_model=NovelHealthOut)
async def novel_health() -> Any:
    """这个入口当前能不能用。

    前端开弹窗时先看一眼：不可用就直接显示「为什么 + 怎么修」，
    而不是让用户点了个按钮才看到报错。
    """
    settings = get_settings()
    return {"enabled": settings.NOVEL_MCP_ENABLED, **(await novel_service.health())}


# ----------------------------------------------------------------------
# 找书 → 找人 → 找话
# ----------------------------------------------------------------------


@router.get("/novel/books", response_model=NovelTextOut)
async def novel_books(
    query: str | None = Query(default=None, description="书名关键词，不传则列全部"),
    limit: int = Query(default=20, ge=1, le=_MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
) -> Any:
    """书目。**入口中的入口**：`book_id` 只从这里来。"""
    return await _text(novel_service.books(query=query, limit=limit, offset=offset))


@router.get("/novel/characters", response_model=NovelTextOut)
async def novel_characters(
    name: str = Query(min_length=1, description="人物名或别名，如「宝玉」「宝二爷」"),
    book_id: int | None = Query(default=None, description="限定在某本书内查找"),
    limit: int = Query(default=20, ge=1, le=_MAX_LIMIT),
) -> Any:
    """按人名找人，顺带给出全部别名。"""
    return await _text(
        novel_service.characters(name=name, book_id=book_id, limit=limit)
    )


@router.get("/novel/dialogues", response_model=NovelTextOut)
async def novel_dialogues(
    character_id: int = Query(ge=1, description="来自 /novel/characters"),
    book_id: int | None = Query(default=None),
    chapter_from: int | None = Query(default=None, ge=1, description="起始章号（含）"),
    chapter_to: int | None = Query(default=None, ge=1, description="结束章号（含）"),
    query: str | None = Query(default=None, description="在对话正文里再过滤关键词"),
    limit: int = Query(default=20, ge=1, le=_MAX_LIMIT),
    cursor: int | None = Query(default=None, description="翻页游标"),
) -> Any:
    """某个人物的对话原文。每条的「段N」可以喂给 `/novel/passage-text`。"""
    return await _text(
        novel_service.dialogues(
            character_id=character_id,
            book_id=book_id,
            # 起止章号是**成对**的：只给一个对方会忽略（它的契约如此），
            # 与其在这里悄悄补一个默认值，不如让前端保证成对传。
            chapter_from=chapter_from if chapter_to is not None else None,
            chapter_to=chapter_to if chapter_from is not None else None,
            query=query,
            limit=limit,
            cursor=cursor,
        )
    )


# ----------------------------------------------------------------------
# 读原文：回目 → 整章 → 某一段的前后文
# ----------------------------------------------------------------------


@router.get("/novel/chapters", response_model=NovelTextOut)
async def novel_chapters(
    book_id: int = Query(ge=1, description="来自 /novel/books"),
    query: str | None = Query(default=None, description="回目关键词"),
    limit: int = Query(default=100, ge=1, le=_MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
) -> Any:
    """回目表。`chapter_idx` 只从这里来（它是库内序号，不是原著回目号）。"""
    return await _text(
        novel_service.chapters(book_id=book_id, query=query, limit=limit, offset=offset)
    )


@router.get("/novel/chapter-text", response_model=NovelTextOut)
async def novel_chapter_text(
    book_id: int = Query(ge=1),
    chapter_idx: int = Query(ge=1, description="来自 /novel/chapters"),
    cursor: int | None = Query(default=None, description="翻页游标：上一页的 cursor"),
    limit: int = Query(default=80, ge=1, le=_MAX_LIMIT, description="每页段数"),
) -> Any:
    """某一回的整章原文，一段一行，段首方括号里是全书段序。"""
    return await _text(
        novel_service.chapter_text(
            book_id=book_id, chapter_idx=chapter_idx, cursor=cursor, limit=limit
        )
    )


@router.get("/novel/passage-text", response_model=NovelTextOut)
async def novel_passage_text(
    book_id: int = Query(ge=1),
    global_idx: int = Query(ge=1, description="全书段序，来自「段N」或整章原文的 [N]"),
    before: int = Query(default=2, ge=0, le=20),
    after: int = Query(default=2, ge=0, le=20),
) -> Any:
    """锚点段落 + 前后各 N 段。"""
    return await _text(
        novel_service.passage_text(
            book_id=book_id, global_idx=global_idx, before=before, after=after
        )
    )
