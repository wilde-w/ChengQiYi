"""古典文学检索入口的服务层。

路由薄、service 厚：参数在这里组装成对方 MCP 的工具入参，路由只管 HTTP。

**这个入口是只读的旁观者**：不写观心的任何存储，也不写对方的库，
更不进分析流水线的状态机。它和「＋ 导入」的关系，就是「查」和「存」的关系。
"""

from __future__ import annotations

from typing import Any, NamedTuple

from app.novel.mcp_client import TOOLS, NovelMcpClient

_client: NovelMcpClient | None = None


class NovelText(NamedTuple):
    """一次查阅的结果。

    ``tool`` 是**对方**的工具名（`novel_*`）。透出来是为了排错：同样一句
    「取不到原文」，调 `novel_get_chapter_text` 和调 `novel_list_chapters`
    是两件完全不同的事。
    """

    tool: str
    markdown: str


def get_client() -> NovelMcpClient:
    global _client
    if _client is None:
        _client = NovelMcpClient()
    return _client


def reset_client() -> None:
    """测试用：丢掉缓存的客户端，让下一次调用重新读配置。"""
    global _client
    _client = None


def _args(**kwargs: Any) -> dict[str, Any]:
    """去掉值为 None 的键。

    对方的参数里 `None` 与「不传」语义相同（都是「不限定」），但**只在
    schema 层相同**：pydantic 版本变化时 `null` 可能被判为类型错误。
    少传一个键比传一个 null 稳。
    """
    return {k: v for k, v in kwargs.items() if v is not None}


# ---- 三个「找」的入口 ---------------------------------------------------------


async def books(*, query: str | None = None, limit: int = 20, offset: int = 0) -> NovelText:
    markdown = await get_client().call(
        TOOLS["books"], _args(query=query, limit=limit, offset=offset)
    )
    return NovelText(TOOLS["books"], markdown)


async def characters(
    *, name: str, book_id: int | None = None, limit: int = 20
) -> NovelText:
    markdown = await get_client().call(
        TOOLS["characters"], _args(name=name, book_id=book_id, limit=limit)
    )
    return NovelText(TOOLS["characters"], markdown)


async def dialogues(
    *,
    character_id: int,
    book_id: int | None = None,
    chapter_from: int | None = None,
    chapter_to: int | None = None,
    query: str | None = None,
    limit: int = 20,
    cursor: int | None = None,
) -> NovelText:
    markdown = await get_client().call(
        TOOLS["dialogues"],
        _args(
            character_id=character_id,
            book_id=book_id,
            chapter_from=chapter_from,
            chapter_to=chapter_to,
            query=query,
            limit=limit,
            cursor=cursor,
        ),
    )
    return NovelText(TOOLS["dialogues"], markdown)


# ---- 三个「读原文」的入口 -----------------------------------------------------


async def chapters(
    *, book_id: int, query: str | None = None, limit: int = 100, offset: int = 0
) -> NovelText:
    markdown = await get_client().call(
        TOOLS["chapters"], _args(book_id=book_id, query=query, limit=limit, offset=offset)
    )
    return NovelText(TOOLS["chapters"], markdown)


async def chapter_text(
    *,
    book_id: int,
    chapter_idx: int,
    cursor: int | None = None,
    limit: int = 80,
) -> NovelText:
    markdown = await get_client().call(
        TOOLS["chapter_text"],
        _args(book_id=book_id, chapter_idx=chapter_idx, cursor=cursor, limit=limit),
    )
    return NovelText(TOOLS["chapter_text"], markdown)


async def passage_text(
    *, book_id: int, global_idx: int, before: int = 2, after: int = 2
) -> NovelText:
    markdown = await get_client().call(
        TOOLS["passage_text"],
        _args(book_id=book_id, global_idx=global_idx, before=before, after=after),
    )
    return NovelText(TOOLS["passage_text"], markdown)


async def health() -> dict[str, Any]:
    return await get_client().health()
