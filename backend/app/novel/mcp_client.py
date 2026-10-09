"""通过 MCP 接入古典文学知识库（另一个仓：ClassicalNovelProject）。

## 与 `douyin/mcp_provider.py` 的三处差别

| | 抖音那套 | 这一套 |
|---|---|---|
| 返回 | JSON，要按 `DOUYIN_MCP_FIELD_MAP` 归一化 | **Markdown，原样搬运** |
| 拉起 | `npx`，不需要工作目录 | 对方的 conda 解释器，**必须 cwd 到对方仓库** |
| 环境 | 继承即可 | 必须注入 `DB_URL`（它的库地址是相对路径） |

第三点是真会踩的坑：`DB_URL=sqlite+aiosqlite:///./data/novel.db` 里的 `./`
按**子进程的 cwd** 解析，cwd 不对就是「库是空的」（而不是报错）——
一个静默失败，表现为「所有工具都返回没有数据」。

## 为什么不做载荷解析

对方的渲染层（`app/components/renderer.py`）就是**为 LLM 读而写的 Markdown**，
并且它的 docstring 明确拒绝返回 JSON。在这里把 Markdown 解析回结构体，
等于把对方的展示层当成 API 用：它一次改版（加个「⚠️」前缀、改个分隔符）
我们这边就全线崩，而且崩得很难看。原样透传到前端 `<pre>` 是最稳的。

## 为什么每次调用新建一条会话

和抖音那套一样的理由（见 `mcp_provider.py:79-83`）：MCP 会话绑在 stdio
子进程上，跨请求复用要管理子进程的整个生命周期。实测单次约 1.7s
（几乎全是子进程启动 + 对方 import sqlalchemy），交互式查阅可以接受。
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Final

from app.config import get_settings
from app.logging_conf import get_logger

log = get_logger(__name__)

#: 逻辑名 → 对方 MCP 的工具名。
#: 与抖音那套不同，这里**不做成可配置映射**——这个 MCP 是另一个仓里的自家
#: 代码，工具名是它的对外契约（对方 `tests/test_mcp_stdio.py` 钉死了这 6 个），
#: 让用户能改只会制造「配置写错但看着像没数据」的故障。
TOOLS: Final[dict[str, str]] = {
    "books": "novel_list_books",
    "characters": "novel_search_characters",
    "dialogues": "novel_get_character_dialogues",
    "chapters": "novel_list_chapters",
    "chapter_text": "novel_get_chapter_text",
    "passage_text": "novel_get_passage_text",
}

#: 连不上时统一附带的自救说明。用户看到的是这句话，不是堆栈。
_HINT = (
    "古典文学 MCP 没连上。检查 NOVEL_MCP_COMMAND（对方仓库的 python 解释器）"
    "与 NOVEL_MCP_CWD（对方仓库根目录）是否正确，或把 NOVEL_MCP_ENABLED 设为 false 关掉这个入口。"
)


class NovelMcpError(RuntimeError):
    """调用古典文学 MCP 失败。

    ``kind`` 直接决定 HTTP 状态码，分三档而不是一个布尔 —— 它们的**处置方式**
    不同，而「502/503 都当同一个错误弹给用户」在排错时最费时间：

    - ``unavailable``：这边或对方没准备好（配置错、目录不在）。503。
    - ``timeout``：超时，等一下可能就好。504。
    - ``failed``：对方明确拒绝了这次调用。502。
    """

    def __init__(self, message: str, *, kind: str = "failed") -> None:
        super().__init__(message)
        self.kind = kind

    @property
    def retryable(self) -> bool:
        return self.kind != "unavailable"


class NovelMcpClient:
    name = "novel-mcp"

    def __init__(self) -> None:
        settings = get_settings()
        self._transport = settings.NOVEL_MCP_TRANSPORT
        self._command = settings.NOVEL_MCP_COMMAND
        self._args = _parse_args(settings.NOVEL_MCP_ARGS)
        self._cwd = settings.NOVEL_MCP_CWD
        self._url = settings.NOVEL_MCP_URL
        self._db_url = settings.NOVEL_MCP_DB_URL
        self._timeout = float(settings.NOVEL_MCP_TIMEOUT)

    # ------------------------------------------------------------------
    def prerequisites(self) -> str | None:
        """返回「为什么起不来」的中文说明；一切正常时返回 None。

        在 spawn 之前先查文件，而不是等 `stdio_client` 抛错：子进程启动失败
        会被 anyio 的 task group 包成 ExceptionGroup，剥出来的错误信息
        又长又认不出（`FileNotFoundError` 藏在第三层），而这是一条
        用户自己就能修的配置问题。
        """
        settings = get_settings()
        if not settings.NOVEL_MCP_ENABLED:
            return "古典文学 MCP 已被 NOVEL_MCP_ENABLED=false 关闭。"
        if self._transport == "http":
            if not self._url:
                return "NOVEL_MCP_TRANSPORT=http 但 NOVEL_MCP_URL 是空的。"
            return None
        if not Path(self._command).exists():
            return f"找不到解释器：{self._command}"
        if not Path(self._cwd).is_dir():
            return f"找不到对方仓库目录：{self._cwd}"
        return None

    @asynccontextmanager
    async def _session(self):
        try:
            from mcp import ClientSession
        except ImportError as exc:  # pragma: no cover
            raise NovelMcpError("未安装 mcp 包，无法调用古典文学 MCP。", kind="unavailable") from exc

        if self._transport == "http":
            from mcp.client.streamable_http import streamablehttp_client

            async with streamablehttp_client(self._url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
        else:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=self._command,
                args=self._args,
                cwd=self._cwd,
                # 必须在 os.environ 之上追加：DB_URL 是相对路径，缺了就打开一个
                # **空的**新库（sqlite 不报错），表现为「工具都在，但书都是空的」。
                env={**os.environ, "DB_URL": self._db_url, "PYTHONUTF8": "1"},
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session

    # ------------------------------------------------------------------
    async def call(self, tool: str, arguments: dict[str, Any]) -> str:
        """调一个工具，返回它渲染好的 Markdown。"""
        reason = self.prerequisites()
        if reason:
            raise NovelMcpError(reason, kind="unavailable")

        try:
            return await asyncio.wait_for(
                self._call_once(tool, arguments), timeout=self._timeout
            )
        except TimeoutError as exc:
            raise NovelMcpError(
                f"古典文学 MCP 的 {tool} 超过 {self._timeout:.0f} 秒没有返回。",
                kind="timeout",
            ) from exc

    async def _call_once(self, tool: str, arguments: dict[str, Any]) -> str:
        async with self._session() as session:
            try:
                result = await session.call_tool(tool, arguments)
            except Exception as exc:
                raise NovelMcpError(
                    f"古典文学 MCP 的 {tool} 调用失败：{type(exc).__name__}: {exc}",
                ) from exc
        return _text_of(result, tool)

    async def health(self) -> dict[str, Any]:
        """探测连通性与工具表。**不抛异常**——它是「显示一个状态」用的。"""
        reason = self.prerequisites()
        if reason:
            return {"ok": False, "error": reason, "hint": _HINT, "tools": [], "missing": []}
        try:
            async with self._session() as session:
                listed = await session.list_tools()
        except Exception as exc:  # 状态探测不该有未捕获的异常：它要能返回「为什么不能用」
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "hint": _HINT,
                "tools": [],
                "missing": [],
            }
        names = [t.name for t in listed.tools]
        missing = [v for v in TOOLS.values() if v not in names]
        return {
            "ok": not missing,
            "error": None if not missing else "对方 MCP 缺少工具：" + "、".join(missing),
            "hint": _HINT,
            "tools": names,
            "missing": missing,
        }


def _text_of(result: Any, tool: str) -> str:
    """从 CallToolResult 里取出渲染好的 Markdown。

    对方把工具内部的异常折成 ``is_error=True`` + 一段**给 LLM 读的中文说明**
    （不是协议错误）。这段话要原样透出去——它对用户同样有用，
    而重新包装只会丢掉「哪本书没导入」这类具体信息。
    """
    blocks = getattr(result, "content", None) or []
    parts = [t for b in blocks if (t := getattr(b, "text", None))]
    if getattr(result, "is_error", False) or getattr(result, "isError", False):
        detail = "\n".join(parts) or "（对方没有给出原因）"
        raise NovelMcpError(f"古典文学 MCP 的 {tool} 报错：{detail}")
    if not parts:
        raise NovelMcpError(f"古典文学 MCP 的 {tool} 没有返回文本内容。")
    return "\n".join(parts)


def _parse_args(raw: str) -> list[str]:
    """`NOVEL_MCP_ARGS` 是 JSON 数组。解析不了就按空格切——和抖音那套一致。"""
    try:
        parsed = json.loads(raw or "[]")
        return [str(a) for a in parsed] if isinstance(parsed, list) else []
    except ValueError:
        return [a for a in str(raw).split() if a]
