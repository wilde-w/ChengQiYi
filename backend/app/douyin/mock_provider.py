"""基于内置样本的数据源。

**这是演示资产，不是桩。** 它真实地分页、真实地带延迟、真实地返回 52 条
情感结构完整的评论——因此前端看到的「已获取 20/40/60 条」是分页驱动的，
而不是硬编码的假进度。换成 MCP provider 时，这条路径一行都不用改。

样本的选取规则：aweme_id 命中样本 → 用该样本；否则按 aweme_id 哈希稳定地
选一个。后者让「随便贴个链接」也能得到自洽的结果，而不是报错——
演示时没人记得住那串 id。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from app.douyin.base import (
    CommentItem,
    CommentPage,
    DouyinError,
    ResolvedRef,
    Transcript,
    VideoMeta,
)
from app.logging_conf import get_logger

log = get_logger(__name__)

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# 默认样本：丧亲哀伤。选它做主推是因为它一次点亮产品的全部能力
# （持续性联结 / 未完成事件 / 双过程模型 × 江城子 / 项脊轩志 / 葬花吟）。
DEFAULT_FIXTURE = "demo_grief_wechat_voice.json"

FIXTURE_ORDER = (
    "demo_grief_wechat_voice.json",
    "demo_burnout_office.json",
    "demo_first_love_summer.json",
)

# 演示用的固定 id 前缀，方便文档与测试引用
DEMO_PREFIX = "730000000000000000"


def fixture_ids() -> list[str]:
    """每个 fixture 暴露一个稳定的演示 aweme_id。"""
    ids: list[str] = []
    for i, name in enumerate(FIXTURE_ORDER):
        ids.append(f"{DEMO_PREFIX}{i + 1}")
    return ids


@lru_cache(maxsize=8)
def load_fixture(name: str) -> dict[str, Any]:
    path = FIXTURE_DIR / name
    if not path.exists():
        raise DouyinError(
            f"演示样本缺失：{path}。请确认 backend/app/douyin/fixtures/ 未被裁剪。",
            retryable=False,
            provider="mock",
        )
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def available_fixtures() -> list[str]:
    return [n for n in FIXTURE_ORDER if (FIXTURE_DIR / n).exists()]


class MockDouyinProvider:
    """实现 DouyinProvider 协议。"""

    is_mock = True
    name = "mock-fixture"

    def __init__(
        self,
        *,
        latency_ms: int = 400,
        page_size: int = 20,
        fail_rate: float = 0.0,
    ) -> None:
        self._latency = max(0, latency_ms) / 1000.0
        self._page_size = max(1, page_size)
        self._fail_rate = fail_rate

    # ------------------------------------------------------------------
    async def health(self) -> dict[str, Any]:
        fixtures = available_fixtures()
        return {
            "ok": bool(fixtures),
            "provider": self.name,
            "fixtures": fixtures,
            "demo_ids": dict(zip(fixtures, fixture_ids())),
        }

    async def resolve_link(self, raw: str) -> ResolvedRef:
        from app.douyin import link as link_mod

        await self._simulate()
        try:
            ref = link_mod.parse_local(raw)
        except link_mod.LinkParseError:
            ref = None

        if ref is not None:
            return ref

        # 无法本地解析的输入（短链、分享口令、任意文本）在 mock 模式下不报错，
        # 而是稳定地映射到一个样本——演示时随手贴个链接也该能跑。
        #
        # 先选样本、再由样本倒推 id，顺序不能反：先散列 raw 得到的 id
        # 与 get_video 里按 id 散列出的样本互不相干，会出现
        # 「解析出的 aweme_id」和「实际返回的视频」对不上这种假 bug。
        aweme_id = fixture_ids()[self._pick_fixture_index(raw)]
        return ResolvedRef(
            aweme_id=aweme_id,
            canonical_url=f"https://www.douyin.com/video/{aweme_id}",
            resolved_via="mock",
        )

    async def get_video(self, aweme_id: str) -> VideoMeta:
        await self._simulate()
        data = load_fixture(self._fixture_for(aweme_id))["video"]
        return VideoMeta(
            aweme_id=str(data.get("aweme_id") or aweme_id),
            title=data.get("title"),
            caption=data.get("caption"),
            author_name=data.get("author_name"),
            author_id=data.get("author_id"),
            author_avatar=data.get("author_avatar"),
            publish_time=_parse_dt(data.get("publish_time")),
            duration_ms=data.get("duration_ms"),
            cover_url=data.get("cover_url"),
            share_url=data.get("share_url"),
            stats=dict(data.get("stats") or {}),
            raw={},
        )

    async def get_comments(
        self,
        aweme_id: str,
        *,
        cursor: str | None = None,
        count: int = 20,
        sort: Literal["hot", "time"] = "hot",
    ) -> CommentPage:
        await self._simulate()
        raw_items = list(load_fixture(self._fixture_for(aweme_id))["comments"])

        if sort == "hot":
            raw_items.sort(key=lambda c: (-int(c.get("like_count") or 0), str(c.get("comment_id"))))
        else:
            raw_items.sort(key=lambda c: (str(c.get("publish_time") or ""), str(c.get("comment_id"))))

        start = int(cursor) if cursor and cursor.isdigit() else 0
        limit = min(max(1, count), self._page_size)
        window = raw_items[start : start + limit]
        next_pos = start + len(window)
        has_more = next_pos < len(raw_items)

        return CommentPage(
            items=[_to_item(c) for c in window],
            next_cursor=str(next_pos) if has_more else None,
            has_more=has_more,
            total=len(raw_items),
        )

    async def get_transcript(self, aweme_id: str) -> Transcript | None:
        await self._simulate()
        data = load_fixture(self._fixture_for(aweme_id)).get("transcript")
        if not data or not data.get("text"):
            return None
        return Transcript(
            text=str(data["text"]),
            source=str(data.get("source") or "mock_asr"),
            segments=list(data.get("segments") or []),
        )

    # ------------------------------------------------------------------
    async def _simulate(self) -> None:
        if self._latency:
            await asyncio.sleep(self._latency)
        if self._fail_rate > 0:
            # 由 aweme_id 无关的调用序号决定不稳定；这里用累计计数实现
            self._calls = getattr(self, "_calls", 0) + 1
            if (self._calls * 7919) % 1000 < self._fail_rate * 1000:
                raise DouyinError("模拟的数据源故障", retryable=True, provider=self.name)

    def _fixture_for(self, aweme_id: str) -> str:
        ids = fixture_ids()
        if aweme_id in ids:
            idx = ids.index(aweme_id)
            fixtures = available_fixtures()
            if idx < len(fixtures):
                return fixtures[idx]
        return self._pick_fixture(aweme_id)

    def _pick_fixture(self, seed: str) -> str:
        return available_fixtures()[self._pick_fixture_index(seed)]

    def _pick_fixture_index(self, seed: str) -> int:
        """把任意输入稳定映射到一个样本下标。

        `resolve_link` 与 `_pick_fixture` 必须共用这一个下标函数——
        两条路径各自散列一次是「解析出的 aweme_id 与实际返回的视频对不上」
        那个 bug 的根源。
        """
        fixtures = available_fixtures()
        if not fixtures:
            raise DouyinError("没有任何可用的演示样本", retryable=False, provider=self.name)
        return _stable_index(seed, len(fixtures))


def _stable_index(seed: str, modulo: int) -> int:
    """稳定散列。用 blake2b 而非内置 hash——后者跨进程不一致。"""
    digest = hashlib.blake2b(seed.strip().encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % modulo


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _to_item(raw: dict[str, Any]) -> CommentItem:
    return CommentItem(
        comment_id=str(raw.get("comment_id") or ""),
        text=str(raw.get("text") or ""),
        author_name=raw.get("author_name"),
        author_id=raw.get("author_id"),
        like_count=int(raw.get("like_count") or 0),
        reply_count=int(raw.get("reply_count") or 0),
        publish_time=_parse_dt(raw.get("publish_time")),
        raw={},
    )
