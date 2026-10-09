"""事件流持久化、案例库与知识库文档登记。"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from app.models.base import Base, TimestampMixin, UUIDPrimaryKey


class RunEvent(Base, UUIDPrimaryKey, TimestampMixin):
    """运行事件的持久化副本——SSE 断线重连的最终回放源。

    Redis Stream 会被 MAXLEN 裁剪，且 Redis 本身可被清空；
    这张表是「刷新页面后日志没有缺口」的兜底保证。
    seq 由 RunBus 分配，每个 run 内单调递增。
    """

    __tablename__ = "run_event"
    __table_args__ = (
        UniqueConstraint("run_id", "seq", name="uq_run_event_seq"),
        Index("ix_run_event_run_seq", "run_id", "seq"),
    )

    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analysis_run.id", ondelete="CASCADE"), index=True
    )
    seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    node: Mapped[str | None] = mapped_column(String(32))
    progress: Mapped[int | None] = mapped_column(Integer)
    message: Mapped[str | None] = mapped_column(Text)
    data: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class Case(Base, UUIDPrimaryKey, TimestampMixin):
    """案例库条目。

    snapshot 冻结运行结果——即使源运行被删除，案例仍能完整渲染。
    这是「沉淀」与「引用一个会消失的东西」的区别。
    """

    __tablename__ = "case"

    title: Mapped[str] = mapped_column(Text, nullable=False)
    run_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("analysis_run.id", ondelete="SET NULL"), nullable=True
    )
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    note: Mapped[str | None] = mapped_column(Text)
    snapshot: Mapped[dict] = mapped_column(JSON, default=dict)


class KbDocument(Base, TimestampMixin):
    """知识库摄取登记表。

    content_hash 是「跳过未变更 chunk」的依据——重复摄取必须是廉价的，
    否则每次改一条语料都要重跑全量 embedding。
    """

    __tablename__ = "kb_document"

    chunk_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    library: Mapped[str] = mapped_column(String(16), index=True)
    content_hash: Mapped[str] = mapped_column(String(80), default="")
    dim: Mapped[int] = mapped_column(Integer, default=0)
    source_file: Mapped[str | None] = mapped_column(String(255))
    text_preview: Mapped[str | None] = mapped_column(Text)


class KbImportJob(Base, UUIDPrimaryKey, TimestampMixin):
    """一次「从 UI 上传一个文本文件并入库」的作业。

    **进度落库而不是走 SSE**，和 `AnalysisRun` 是两套做法，这是刻意的：
    导入是「等一到两分钟看进度条」，不需要逐 token 的实时性；而接 SSE 要
    新建端点、绕开 `run_event` 的外键约束、改前后端两处的终态事件集合。
    落库还有一个白拿的好处：**刷新页面进度不丢**。

    `filename` 是**原始文件名**，同时是 `kb_document.source_file` 的取值，
    也是「删掉这一次导入」的检索键。落盘路径与它无关（见
    `kb_import_service`）——用户提供的字符绝不拼进路径。
    """

    __tablename__ = "kb_import_job"

    library: Mapped[str] = mapped_column(String(16), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, default=0)

    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    stage: Mapped[str | None] = mapped_column(String(16))
    progress: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str | None] = mapped_column(Text)

    chunk_total: Mapped[int] = mapped_column(Integer, default=0)
    chunk_tagged: Mapped[int] = mapped_column(Integer, default=0)
    chunk_indexed: Mapped[int] = mapped_column(Integer, default=0)

    #: 切分/打标阶段的告警原文，逐条展示给用户——
    #: 「硬切了 37 处」这类信息只有用户能判断要不要重来。
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    #: 创建时冻结的选项（书名/作者/领域/是否打标），重跑与审计都靠它。
    options: Mapped[dict] = mapped_column(JSON, default=dict)
    #: 用户勾了「打标失败也导入」时为 True。默认 False 时，打标大面积
    #: 失败会中止且**什么都不写**——写进半垃圾比不写难收拾得多。
    error: Mapped[str | None] = mapped_column(Text)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
