"""抖音数据源的领域类型与 provider 协议。

协议是**唯一**的接入点。mock provider 与 MCP provider 都归一化成这里定义的
类型，因此流水线节点完全不知道数据从哪来——这正是 mock 能证明流水线是通的
而不是「另走一条分支」的原因。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

LinkKind = Literal["video", "note", "unknown"]
ResolvedVia = Literal["regex", "redirect", "mock"]


@dataclass(slots=True)
class ResolvedRef:
    aweme_id: str
    kind: LinkKind = "video"
    canonical_url: str = ""
    resolved_via: ResolvedVia = "regex"


@dataclass(slots=True)
class VideoMeta:
    aweme_id: str
    title: str | None = None
    caption: str | None = None
    author_name: str | None = None
    author_id: str | None = None
    author_avatar: str | None = None
    publish_time: datetime | None = None
    duration_ms: int | None = None
    cover_url: str | None = None
    share_url: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "aweme_id": self.aweme_id,
            "title": self.title,
            "caption": self.caption,
            "author_name": self.author_name,
            "author_id": self.author_id,
            "author_avatar": self.author_avatar,
            "publish_time": self.publish_time.isoformat() if self.publish_time else None,
            "duration_ms": self.duration_ms,
            "cover_url": self.cover_url,
            "share_url": self.share_url,
            "stats": self.stats,
            "raw": self.raw,
        }


@dataclass(slots=True)
class CommentItem:
    comment_id: str
    text: str
    author_name: str | None = None
    author_id: str | None = None
    like_count: int = 0
    reply_count: int = 0
    publish_time: datetime | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "comment_id": self.comment_id,
            "text": self.text,
            "author_name": self.author_name,
            "author_id": self.author_id,
            "like_count": self.like_count,
            "reply_count": self.reply_count,
            "publish_time": self.publish_time.isoformat() if self.publish_time else None,
            "raw": self.raw,
        }


@dataclass(slots=True)
class CommentPage:
    items: list[CommentItem]
    next_cursor: str | None = None
    has_more: bool = False
    total: int | None = None


@dataclass(slots=True)
class Transcript:
    text: str
    source: str = "unknown"
    segments: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "source": self.source, "segments": self.segments}


class DouyinError(RuntimeError):
    """数据源失败。retryable 决定节点是否值得重试，还是降级继续。"""

    def __init__(self, message: str, *, retryable: bool = False, provider: str = "") -> None:
        super().__init__(message)
        self.retryable = retryable
        self.provider = provider


@runtime_checkable
class DouyinProvider(Protocol):
    """抖音数据源协议。

    `get_transcript` 是可选的——7 个流水线节点里没有 ASR 环节，
    口播文案只在 MCP 同时提供时才有。缺失时左栏渲染明确的空状态，绝不编造。
    """

    name: str
    is_mock: bool

    async def health(self) -> dict[str, Any]: ...

    async def resolve_link(self, raw: str) -> ResolvedRef: ...

    async def get_video(self, aweme_id: str) -> VideoMeta: ...

    async def get_comments(
        self,
        aweme_id: str,
        *,
        cursor: str | None = None,
        count: int = 20,
        sort: Literal["hot", "time"] = "hot",
    ) -> CommentPage: ...
