"""通过 MCP 接入真实抖音数据。

PRD 点名的几个 MCP Server（hhy5562877/douyin_mcp 等）工具名与返回结构各不相同，
所以这一层的真正职责是**映射 + 归一化**，而不是调用：
  工具名映射   DOUYIN_MCP_TOOLS
  字段路径映射 DOUYIN_MCP_FIELD_MAP（点号路径，如 statistics.digg_count）

归一化必须「缺字段不抛错」：真实 MCP 返回的字段随版本变化，
少一个点赞数不该让整次分析失败——视频有元数据、评论为空，仍然是一份可用的部分报告。

前提：该 MCP 需要有效的抖音 cookies（见 README）。这也是它默认不被启用的原因。
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal

from app.config import get_settings
from app.douyin.base import (
    CommentItem,
    CommentPage,
    DouyinError,
    ResolvedRef,
    VideoMeta,
)
from app.logging_conf import get_logger

log = get_logger(__name__)

# 未配置 FIELD_MAP 时使用的默认路径。覆盖 hhy5562877/douyin_mcp 的返回结构——
# 注意它返回的视频与评论都已经是**扁平**结构（不是抖音原始 API 的嵌套结构），
# 所以没有 author./statistics./user. 前缀。评论字段用 comment_ 前缀与视频区分。
DEFAULT_FIELD_MAP: dict[str, str] = {
    "title": "title",
    "caption": "desc",
    "author_name": "nickname",
    "author_id": "sec_uid",
    "author_avatar": "avatar",
    "publish_time": "create_time",
    "duration_ms": "duration",
    "cover_url": "cover_url",
    "share_url": "aweme_url",
    "digg_count": "liked_count",
    "comment_count": "comment_count",
    "share_count": "share_count",
    "collect_count": "collected_count",
    "comment_text": "content",
    "comment_author_name": "nickname",
    "comment_author_id": "sec_uid",
    "comment_like_count": "like_count",
    "comment_reply_count": "sub_comment_count",
    "comment_publish_time": "create_time",
    "comment_id": "comment_id",
}


class McpDouyinProvider:
    """实现 DouyinProvider 协议。"""

    is_mock = False
    name = "mcp-douyin"

    def __init__(self) -> None:
        settings = get_settings()
        self._transport = settings.DOUYIN_MCP_TRANSPORT
        self._command = settings.DOUYIN_MCP_COMMAND or "npx"
        self._args = _parse_args(settings.DOUYIN_MCP_ARGS)
        self._url = settings.DOUYIN_MCP_URL
        self._tools = _parse_tools(settings.DOUYIN_MCP_TOOLS)
        self._field_map = {**DEFAULT_FIELD_MAP, **_parse_json(settings.DOUYIN_MCP_FIELD_MAP)}
        self._timeout = float(settings.DOUYIN_REQUEST_TIMEOUT)

    # ------------------------------------------------------------------
    @asynccontextmanager
    async def _session(self):
        """每次调用建一条 MCP 会话。

        不做长连接池：MCP 会话绑定在 stdio 子进程上，跨请求复用会引入
        进程生命周期管理的一堆边界情况，而这里的调用量很小（每视频 2-3 次）。
        """
        try:
            from mcp import ClientSession
        except ImportError as exc:  # pragma: no cover
            raise DouyinError(
                "未安装 mcp 包。执行 `pip install mcp` 后再启用 DOUYIN_PROVIDER=mcp。",
                retryable=False,
                provider=self.name,
            ) from exc

        if self._transport == "http":
            from mcp.client.streamable_http import streamablehttp_client

            async with streamablehttp_client(self._url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
        else:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(command=self._command, args=self._args, env=None)
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session

    async def _call(self, kind: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(kind)
        if not tool:
            raise DouyinError(
                f"未配置 {kind} 对应的 MCP 工具名（DOUYIN_MCP_TOOLS.{kind}）",
                retryable=False,
                provider=self.name,
            )

        async with self._session() as session:
            try:
                result = await session.call_tool(tool, arguments)
            except Exception as exc:
                raise DouyinError(
                    f"MCP 工具 {tool} 调用失败：{type(exc).__name__}: {exc}",
                    retryable=True,
                    provider=self.name,
                ) from exc

        payload = _extract_payload(result)
        if payload is None:
            raise DouyinError(
                f"MCP 工具 {tool} 返回了无法解析的内容（期望 JSON）", retryable=False, provider=self.name
            )
        return payload

    # ------------------------------------------------------------------
    async def health(self) -> dict[str, Any]:
        try:
            async with self._session() as session:
                tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            missing = [k for k, v in self._tools.items() if v and v not in names]
            return {
                "ok": not missing,
                "provider": self.name,
                "transport": self._transport,
                "tools": names,
                "missing": missing,
                "hint": "若工具名不匹配，请在 .env 的 DOUYIN_MCP_TOOLS 里改正。",
            }
        except DouyinError:
            raise
        except Exception as exc:
            return {"ok": False, "provider": self.name, "error": f"{type(exc).__name__}: {exc}"}

    async def resolve_link(self, raw: str) -> ResolvedRef:
        from app.douyin import link as link_mod

        try:
            local = link_mod.parse_local(raw)
        except link_mod.LinkParseError:
            local = None
        if local is not None:
            return local
        # 只有短链才发外部请求（含 SSRF 守卫）
        return await link_mod.resolve(raw)

    async def get_video(self, aweme_id: str) -> VideoMeta:
        payload = await self._call("video", {"aweme_id": aweme_id})
        _raise_if_failed(payload, self.name)
        node = _unwrap(payload, ("data", "aweme_detail", "video"))
        stats = {
            key: _coerce_int(_dig(node, path))
            for key, path in (
                ("digg_count", self._field_map["digg_count"]),
                ("comment_count", self._field_map["comment_count"]),
                ("share_count", self._field_map["share_count"]),
                ("collect_count", self._field_map["collect_count"]),
            )
        }
        return VideoMeta(
            aweme_id=str(_dig(node, "aweme_id") or aweme_id),
            title=_str_or_none(_dig(node, self._field_map["title"])),
            caption=_str_or_none(_dig(node, self._field_map["caption"])),
            author_name=_str_or_none(_dig(node, self._field_map["author_name"])),
            author_id=_str_or_none(_dig(node, self._field_map["author_id"])),
            author_avatar=_str_or_none(_dig(node, self._field_map["author_avatar"])),
            publish_time=_coerce_dt(_dig(node, self._field_map["publish_time"])),
            duration_ms=_coerce_int(_dig(node, self._field_map["duration_ms"])),
            cover_url=_str_or_none(_dig(node, self._field_map["cover_url"])),
            share_url=_str_or_none(_dig(node, self._field_map["share_url"])),
            stats={k: v for k, v in stats.items() if v is not None},
            raw=payload,
        )

    async def get_comments(
        self,
        aweme_id: str,
        *,
        cursor: str | None = None,
        count: int = 20,
        sort: Literal["hot", "time"] = "hot",
    ) -> CommentPage:
        args: dict[str, Any] = {"aweme_id": aweme_id, "count": count}
        if cursor:
            # 不同 MCP 的游标参数名不一（cursor / offset），两个都给，多余的会被忽略
            args["cursor"] = int(cursor) if str(cursor).isdigit() else cursor
        payload = await self._call("comments", args)
        _raise_if_failed(payload, self.name)

        raw_list = _first_list(payload, ("comments", "data", "items"))
        fm = self._field_map
        items: list[CommentItem] = []
        for c in raw_list:
            if not isinstance(c, dict):
                continue
            text = _str_or_none(_dig(c, fm["comment_text"])) or ""
            if not text:
                continue
            items.append(
                CommentItem(
                    comment_id=str(_dig(c, fm["comment_id"]) or ""),
                    text=text,
                    author_name=_str_or_none(_dig(c, fm["comment_author_name"])),
                    author_id=_str_or_none(_dig(c, fm["comment_author_id"])),
                    like_count=_coerce_int(_dig(c, fm["comment_like_count"])) or 0,
                    reply_count=_coerce_int(_dig(c, fm["comment_reply_count"])) or 0,
                    publish_time=_coerce_dt(_dig(c, fm["comment_publish_time"])),
                    raw=c,
                )
            )

        # 分页字段的位置随 server 变：顶层 / data 下 / metadata 下（hhy5562877 用
        # metadata）。必须用「存在即取」而不是 `or`——游标 0 是合法值，`or` 会把它
        # 当空值跳过去，结果在第一页原地打转。
        has_more = bool(_first_present(payload, "has_more", "data.has_more", "metadata.has_more"))
        next_cursor = _first_present(payload, "cursor", "data.cursor", "metadata.cursor")
        return CommentPage(
            items=items,
            next_cursor=str(next_cursor) if (has_more and next_cursor is not None) else None,
            has_more=has_more,
            total=_coerce_int(_first_present(payload, "total", "metadata.total")),
        )


# ----------------------------------------------------------------------
# 取值与归一化辅助
# ----------------------------------------------------------------------


def _dig(node: Any, path: str | None) -> Any:
    """按点号路径取值，支持列表下标。

    任何一段缺失都返回 None 而不是抛错——真实 MCP 的字段随版本增删，
    少一个字段不该让整次分析失败。
    """
    if not path or not isinstance(node, (dict, list)):
        return None
    current: Any = node
    for part in str(path).split("."):
        if current is None:
            return None
        if isinstance(current, list):
            if not part.isdigit():
                return None
            idx = int(part)
            current = current[idx] if 0 <= idx < len(current) else None
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _unwrap(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    for key in keys:
        node = payload.get(key)
        if isinstance(node, dict):
            return node
    return payload


def _first_list(payload: dict[str, Any], keys: tuple[str, ...]) -> list[Any]:
    for key in keys:
        node = payload.get(key)
        if isinstance(node, list):
            return node
        if isinstance(node, dict):
            nested = _first_list(node, keys)
            if nested:
                return nested
    return []


def _first_present(payload: dict[str, Any], *paths: str) -> Any:
    """按顺序取第一个**存在**的路径值。

    与 `or` 链条的区别在 0 与 False：游标 0、has_more=False 都是合法值，
    `a or b` 会把它们当成「没有」继续往后找，或落到默认值上。
    """
    for path in paths:
        value = _dig(payload, path)
        if value is not None:
            return value
    return None


def _raise_if_failed(payload: dict[str, Any], provider: str) -> None:
    """把 server 自报的失败转成 DouyinError。

    不转的话失败载荷会被当成「字段全缺」的合法数据静默通过，用户在界面上
    看到的是一个没有标题、没有作者的视频——比报错难查得多。
    """
    if payload.get("success") is False:
        raise DouyinError(
            f"MCP 返回失败：{payload.get('error') or '未说明原因'}",
            retryable=True,
            provider=provider,
        )


def _extract_payload(result: Any) -> dict[str, Any] | None:
    """从 MCP CallToolResult 里取出 JSON。"""
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured

    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"data": parsed}
    return None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_dt(value: Any) -> datetime | None:
    """epoch 秒是抖音 API 的常见形态；也接受 ISO 字符串。"""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().isdigit():
        # hhy5562877 的模型把 create_time 统一转成字符串，数字串按 epoch 处理
        value = int(value.strip())
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 1e11:      # 毫秒
            seconds /= 1000.0
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_json(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except ValueError:
        log.warning("mcp_json_config_invalid", raw=raw[:120])
        return {}


def _parse_args(raw: str) -> list[str]:
    try:
        parsed = json.loads(raw or "[]")
        return [str(a) for a in parsed] if isinstance(parsed, list) else []
    except ValueError:
        return [a for a in str(raw).split() if a]


def _parse_tools(raw: str) -> dict[str, str]:
    parsed = _parse_json(raw)
    return {k: str(v) for k, v in parsed.items() if v}
