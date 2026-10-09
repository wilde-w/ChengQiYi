"""检索证据、推理链与右栏洞察段落。"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from app.models.base import Base, TimestampMixin, UUIDPrimaryKey


class Evidence(Base, UUIDPrimaryKey, TimestampMixin):
    """一条检索到的证据。

    `pinned` / `excluded` 是用户干预的结果，重跑时优先于检索得分——
    这是让使用者能校正检索质量而不必改代码的关键。
    """

    __tablename__ = "evidence"
    __table_args__ = (Index("ix_evidence_run_kind", "run_id", "kind"),)

    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analysis_run.id", ondelete="CASCADE"), index=True
    )
    revision: Mapped[int] = mapped_column(Integer, default=0)

    kind: Mapped[str] = mapped_column(String(16))          # psychology|literature|poetry
    library: Mapped[str] = mapped_column(String(16))
    chunk_id: Mapped[str] = mapped_column(String(128), index=True)

    title: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(Text)
    author: Mapped[str | None] = mapped_column(String(128))
    text: Mapped[str] = mapped_column(Text, default="")

    score: Mapped[float] = mapped_column(Float, default=0.0)
    retrieval_path: Mapped[str] = mapped_column(String(16), default="vector")
    match_reason: Mapped[str | None] = mapped_column(Text)
    cluster_ids: Mapped[list] = mapped_column(JSON, default=list)

    pinned: Mapped[bool] = mapped_column(Boolean, default=False)
    excluded: Mapped[bool] = mapped_column(Boolean, default=False)
    rank: Mapped[int] = mapped_column(Integer, default=0)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)


class ReasoningStep(Base, UUIDPrimaryKey, TimestampMixin):
    """推理链的一步：现象 → 机制 → 典故 → 洞察。

    四个字段里 phenomenon/mechanism/insight 是文字，allusion 体现在 allusion_ids。
    前端据此画确定性分层坐标图，不做力模拟。
    """

    __tablename__ = "reasoning_step"
    __table_args__ = (
        UniqueConstraint("run_id", "revision", "step_index", name="uq_reasoning_run_rev_idx"),
    )

    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analysis_run.id", ondelete="CASCADE"), index=True
    )
    revision: Mapped[int] = mapped_column(Integer, default=0)
    step_index: Mapped[int] = mapped_column(Integer, default=0)

    phenomenon: Mapped[str] = mapped_column(Text, default="")
    mechanism: Mapped[str] = mapped_column(Text, default="")
    insight: Mapped[str] = mapped_column(Text, default="")
    evidence_ids: Mapped[list] = mapped_column(JSON, default=list)
    allusion_ids: Mapped[list] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    model: Mapped[str | None] = mapped_column(String(64))


class InsightSection(Base, UUIDPrimaryKey, TimestampMixin):
    """右栏四个段落之一。

    `stale` 与 `based_on_revision` 配合 analysis_run.revision 使用：
    上游（聚类/证据）变更后，未重新生成的段落会被标记过期，UI 显示 StaleBanner。
    这是「可干预」不变成「结果不可信」的关键——用户永远知道手里这份结论
    是基于哪一版中间产物得出的。
    """

    __tablename__ = "insight_section"
    __table_args__ = (UniqueConstraint("run_id", "key", name="uq_section_run_key"),)

    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analysis_run.id", ondelete="CASCADE"), index=True
    )
    key: Mapped[str] = mapped_column(String(32))          # profile|mechanism|allusion|insight
    title: Mapped[str] = mapped_column(String(64), default="")

    content_md: Mapped[str] = mapped_column(Text, default="")
    content_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    citations: Mapped[list] = mapped_column(JSON, default=list)

    version: Mapped[int] = mapped_column(Integer, default=1)
    stale: Mapped[bool] = mapped_column(Boolean, default=False)
    based_on_revision: Mapped[int] = mapped_column(Integer, default=0)
    edited_by_user: Mapped[bool] = mapped_column(Boolean, default=False)

    citation_coverage: Mapped[float] = mapped_column(Float, default=0.0)
    model: Mapped[str | None] = mapped_column(String(64))
