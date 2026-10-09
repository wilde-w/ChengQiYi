"""一次分析运行及其直接产物：运行、视频、评论、簇、心理画像。

设计要点：
- `analysis_run.revision` 每次干预递增，段落表记 `based_on_revision`，
  两者比对即可判定段落是否过期（stale），无需复杂的依赖图。
- 评论保留 is_ad / is_spam / is_duplicate / filter_reason，
  让「为什么这条被剔除」可被审计——研究者一定会问这个问题。
- **不存 embedding**。向量只活在 Qdrant 与运行期缓存里，PG 不是它的第二个家。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.models.base import Base, TimestampMixin, UUIDPrimaryKey


class AnalysisRun(Base, UUIDPrimaryKey, TimestampMixin):
    __tablename__ = "analysis_run"

    parent_run_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("analysis_run.id", ondelete="SET NULL"), nullable=True
    )
    # LangGraph 检查点的 thread_id，等于 run_id
    thread_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)

    input_raw: Mapped[str] = mapped_column(Text, nullable=False)
    aweme_id: Mapped[str | None] = mapped_column(String(64))
    depth: Mapped[str] = mapped_column(String(16), default="standard")
    kb: Mapped[dict] = mapped_column(JSON, default=dict)
    comment_limit: Mapped[int] = mapped_column(Integer, default=100)
    cluster_params: Mapped[dict] = mapped_column(JSON, default=dict)

    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    current_node: Mapped[str | None] = mapped_column(String(32))
    progress: Mapped[int] = mapped_column(Integer, default=0)
    revision: Mapped[int] = mapped_column(Integer, default=0)

    warnings: Mapped[list] = mapped_column(JSON, default=list)
    error: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    providers: Mapped[dict] = mapped_column(JSON, default=dict)

    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int | None] = mapped_column(Integer)

    video: Mapped["Video | None"] = relationship(  # noqa: F821
        back_populates="run", uselist=False, cascade="all, delete-orphan"
    )
    comments: Mapped[list["Comment"]] = relationship(  # noqa: F821
        back_populates="run", cascade="all, delete-orphan"
    )
    clusters: Mapped[list["Cluster"]] = relationship(  # noqa: F821
        back_populates="run", cascade="all, delete-orphan"
    )


class Video(Base, UUIDPrimaryKey, TimestampMixin):
    __tablename__ = "video"

    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analysis_run.id", ondelete="CASCADE"), unique=True, index=True
    )
    aweme_id: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str | None] = mapped_column(Text)
    author_name: Mapped[str | None] = mapped_column(String(128))
    author_id: Mapped[str | None] = mapped_column(String(128))
    author_avatar: Mapped[str | None] = mapped_column(Text)
    caption: Mapped[str | None] = mapped_column(Text)
    publish_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    cover_url: Mapped[str | None] = mapped_column(Text)
    share_url: Mapped[str | None] = mapped_column(Text)
    stats: Mapped[dict] = mapped_column(JSON, default=dict)
    transcript: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    transcript_source: Mapped[str | None] = mapped_column(String(32))
    raw: Mapped[dict] = mapped_column(JSON, default=dict)

    run: Mapped[AnalysisRun] = relationship(back_populates="video")


class Comment(Base, UUIDPrimaryKey, TimestampMixin):
    __tablename__ = "comment"
    __table_args__ = (
        Index("ix_comment_run_likes", "run_id", "like_count"),
        Index("ix_comment_run_cluster", "run_id", "cluster_id"),
        Index("ix_comment_run_hash", "run_id", "text_hash"),
    )

    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analysis_run.id", ondelete="CASCADE"), index=True
    )
    douyin_comment_id: Mapped[str | None] = mapped_column(String(64))
    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_normalized: Mapped[str] = mapped_column(Text, default="")
    text_hash: Mapped[str] = mapped_column(String(40), default="")

    author_name: Mapped[str | None] = mapped_column(String(128))
    author_id: Mapped[str | None] = mapped_column(String(128))
    like_count: Mapped[int] = mapped_column(Integer, default=0)
    reply_count: Mapped[int] = mapped_column(Integer, default=0)
    publish_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    is_ad: Mapped[bool] = mapped_column(Boolean, default=False)
    is_spam: Mapped[bool] = mapped_column(Boolean, default=False)
    is_duplicate: Mapped[bool] = mapped_column(Boolean, default=False)
    filter_reason: Mapped[str | None] = mapped_column(String(64))

    cluster_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("cluster.id", ondelete="SET NULL"), nullable=True
    )
    raw: Mapped[dict] = mapped_column(JSON, default=dict)

    run: Mapped[AnalysisRun] = relationship(back_populates="comments")


class Cluster(Base, UUIDPrimaryKey, TimestampMixin):
    __tablename__ = "cluster"
    __table_args__ = (UniqueConstraint("run_id", "cluster_key", name="uq_cluster_run_key"),)

    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analysis_run.id", ondelete="CASCADE"), index=True
    )
    cluster_key: Mapped[str] = mapped_column(String(64))
    label: Mapped[str] = mapped_column(Text, default="")
    summary: Mapped[str | None] = mapped_column(Text)

    size: Mapped[int] = mapped_column(Integer, default=0)
    raw_size: Mapped[int] = mapped_column(Integer, default=0)
    is_noise: Mapped[bool] = mapped_column(Boolean, default=False)

    keywords: Mapped[list] = mapped_column(JSON, default=list)
    emotion_tags: Mapped[list] = mapped_column(JSON, default=list)
    topic_tags: Mapped[list] = mapped_column(JSON, default=list)
    need_tags: Mapped[list] = mapped_column(JSON, default=list)
    representative_comment_ids: Mapped[list] = mapped_column(JSON, default=list)
    centroid: Mapped[list | None] = mapped_column(JSON, nullable=True)
    color: Mapped[str] = mapped_column(String(16), default="#8B5CF6")
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    params: Mapped[dict] = mapped_column(JSON, default=dict)

    run: Mapped[AnalysisRun] = relationship(back_populates="clusters")


class PsychProfile(Base, UUIDPrimaryKey, TimestampMixin):
    __tablename__ = "psych_profile"
    __table_args__ = (
        UniqueConstraint("run_id", "revision", name="uq_profile_run_revision"),
    )

    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analysis_run.id", ondelete="CASCADE"), index=True
    )
    revision: Mapped[int] = mapped_column(Integer, default=0)

    emotion_distribution: Mapped[list] = mapped_column(JSON, default=list)
    topic_distribution: Mapped[list] = mapped_column(JSON, default=list)
    need_distribution: Mapped[list] = mapped_column(JSON, default=list)
    global_tags: Mapped[dict] = mapped_column(JSON, default=dict)
    cluster_tags: Mapped[dict] = mapped_column(JSON, default=dict)
    core_tension: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(64))
