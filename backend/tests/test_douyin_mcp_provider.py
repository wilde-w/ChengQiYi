"""MCP 抖音源的归一化：默认映射必须真的对上 hhy5562877/douyin_mcp。

这三件事一旦不对，坏都是**静默**的，所以各压一组：

1. **字段路径**：它的返回是扁平结构（`nickname` / `liked_count`），不是抖音
   原始 API 的嵌套结构。路径写错 → 标题、作者、点赞数全变 None，页面照常渲染。
2. **分页位置**：它在 `metadata` 下。找错地方 → `has_more=False`，抓到第一页
   就停，用户以为这个视频只有 20 条评论。
3. **失败载荷**：失败时它返回 `{"success": False, "error": ...}` 而不是抛异常。
   不识别 → 一个字段全缺的「合法」视频混进流水线，比报错难查得多。

载荷形状照抄它的 `src/models.py`（`asdict` 之后的形态），不引入真实网络。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from app.config import Settings
from app.douyin import mcp_provider as mcp_mod
from app.douyin.base import DouyinError
from app.douyin.mcp_provider import McpDouyinProvider

EPOCH = datetime(2024, 10, 4, tzinfo=timezone.utc)  # 1728000000

VIDEO_PAYLOAD: dict[str, Any] = {
    "success": True,
    "video": {
        "aweme_id": "7300000000000000001",
        "title": "标题",
        "desc": "描述文案",
        "create_time": "1728000000",
        "liked_count": "1234",
        "comment_count": "56",
        "share_count": "7",
        "collected_count": "89",
        "aweme_url": "https://www.douyin.com/video/7300000000000000001",
        "cover_url": "https://cover.example/1.jpg",
        "nickname": "作者甲",
        "sec_uid": "MS4wLjABAAAAxxxx",
        "avatar": "https://avatar.example/1.jpg",
    },
}

COMMENT: dict[str, Any] = {
    "comment_id": "c-1",
    "content": "说到心里去了",
    "create_time": "1728000000",
    "sub_comment_count": "3",
    "like_count": "12",
    "nickname": "用户甲",
    "sec_uid": "MS4wLjABAAAAyyyy",
}


def make_provider(monkeypatch: pytest.MonkeyPatch, field_map: str | None = None) -> McpDouyinProvider:
    """绕开本机 .env：映射只来自代码默认值或显式传入的 FIELD_MAP。"""
    overrides: dict[str, Any] = {"DOUYIN_MCP_FIELD_MAP": field_map} if field_map else {}
    monkeypatch.setattr(mcp_mod, "get_settings", lambda: Settings(_env_file=None, **overrides))
    return McpDouyinProvider()


def stub(provider: McpDouyinProvider, payloads: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """替换 `_call`（对外唯一的边界），记录每次调用的参数。"""
    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_call(kind: str, arguments: dict[str, Any]) -> dict[str, Any]:
        calls.append((kind, arguments))
        return payloads[kind]

    provider._call = fake_call  # type: ignore[method-assign]
    return calls


class TestVideoNormalize:
    async def test_扁平字段按默认映射归一化(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = make_provider(monkeypatch)
        calls = stub(provider, {"video": VIDEO_PAYLOAD})

        meta = await provider.get_video("7300000000000000001")

        assert calls == [("video", {"aweme_id": "7300000000000000001"})]
        assert meta.title == "标题"
        assert meta.caption == "描述文案"
        assert meta.author_name == "作者甲"
        assert meta.author_id == "MS4wLjABAAAAxxxx"
        assert meta.author_avatar == "https://avatar.example/1.jpg"
        assert meta.cover_url == "https://cover.example/1.jpg"
        assert meta.share_url == "https://www.douyin.com/video/7300000000000000001"
        # 统计是字符串，必须转成 int 再进库，否则排序/展示全是错的
        assert meta.stats == {
            "digg_count": 1234,
            "comment_count": 56,
            "share_count": 7,
            "collect_count": 89,
        }
        assert meta.publish_time == EPOCH

    async def test_缺字段不抛错(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """真实返回的字段随版本增删，少一个点赞数不该让整次分析失败。"""
        provider = make_provider(monkeypatch)
        stub(provider, {"video": {"success": True, "video": {"aweme_id": "1"}}})

        meta = await provider.get_video("1")

        assert meta.aweme_id == "1"
        assert meta.title is None
        assert meta.stats == {}

    async def test_失败载荷转成异常(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = make_provider(monkeypatch)
        stub(provider, {"video": {"success": False, "error": "Cookie 已失效"}})

        with pytest.raises(DouyinError) as exc:
            await provider.get_video("1")
        assert "Cookie 已失效" in str(exc.value)


class TestCommentsNormalize:
    def _page(self, metadata: dict[str, Any], comments: list[dict[str, Any]] | None = None) -> dict:
        return {
            "success": True,
            "comments": comments if comments is not None else [COMMENT],
            "metadata": metadata,
        }

    async def test_扁平字段与_metadata_分页(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = make_provider(monkeypatch)
        calls = stub(provider, {"comments": self._page({"cursor": 20, "has_more": True, "total": 100})})

        page = await provider.get_comments("7300000000000000001", count=20)

        assert calls == [("comments", {"aweme_id": "7300000000000000001", "count": 20})]
        (item,) = page.items
        assert item.text == "说到心里去了"
        assert item.comment_id == "c-1"
        assert item.author_name == "用户甲"
        assert item.author_id == "MS4wLjABAAAAyyyy"
        assert item.like_count == 12
        assert item.reply_count == 3
        assert item.publish_time == EPOCH
        assert (page.next_cursor, page.has_more, page.total) == ("20", True, 100)

    async def test_最后一页没有游标(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = make_provider(monkeypatch)
        stub(provider, {"comments": self._page({"cursor": 40, "has_more": False, "total": 40})})

        page = await provider.get_comments("1")

        assert page.has_more is False
        assert page.next_cursor is None

    async def test_游标0不被当成空值(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`or` 链会把 0 跳过去 → next_cursor 变 None → 翻页原地打转。"""
        provider = make_provider(monkeypatch)
        stub(provider, {"comments": self._page({"cursor": 0, "has_more": True, "total": 100})})

        page = await provider.get_comments("1")

        assert page.next_cursor == "0"

    async def test_毫秒时间戳字符串按毫秒处理(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = make_provider(monkeypatch)
        comment = {**COMMENT, "create_time": "1728000000000"}
        stub(provider, {"comments": self._page({"cursor": 0, "has_more": False}, comments=[comment])})

        (item,) = (await provider.get_comments("1")).items

        assert item.publish_time == EPOCH

    async def test_映射可被配置覆盖(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """换个 MCP server 时只该改配置，不该改代码。"""
        provider = make_provider(monkeypatch, field_map='{"comment_text": "text"}')
        comment = {**COMMENT, "text": "另一个 server 的字段名", "content": ""}
        stub(provider, {"comments": self._page({"cursor": 0, "has_more": False}, comments=[comment])})

        (item,) = (await provider.get_comments("1")).items

        assert item.text == "另一个 server 的字段名"

    async def test_空评论列表不报错(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = make_provider(monkeypatch)
        stub(provider, {"comments": self._page({"cursor": 0, "has_more": False}, comments=[])})

        page = await provider.get_comments("1")

        assert page.items == []
        assert page.next_cursor is None

    async def test_失败载荷转成异常(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = make_provider(monkeypatch)
        stub(provider, {"comments": {"success": False, "error": "风控拦截"}})

        with pytest.raises(DouyinError) as exc:
            await provider.get_comments("1")
        assert "风控拦截" in str(exc.value)
