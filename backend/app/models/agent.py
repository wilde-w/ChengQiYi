


"""故事工坊的持久化：会话、消息、事件。

**为什么不复用 `analysis_run` / `run_event`。**
`run_event.run_id` 有一条指向 `analysis_run.id` 的外键（`models/library.py`）。
复用就得为每次写故事伪造一行 AnalysisRun：它会出现在 `/runs` 列表里、会被
`reconcile_orphans` 当成僵尸运行收掉、会让「运行」这个词同时指两种东西。
三张新表换来的是「分析」与「故事工坊」在数据层上也互不干扰。

## 为什么消息与事件要分两张表

`agent_message` 是**模型可见的对话**：下一轮请求要把它逐字重发，一条都不能多、
不能少、不能改。`agent_event` 是**UI 可见的过程**：工具卡片、逐字正文、
轮次提示。两者的读者不同，混成一张表的话，组装历史时得先过滤掉一堆 UI 事件，
而漏掉一条过滤就是一个 400（带 tool_calls 的 assistant 没有配对回复）。

`agent_message` 里**不存 system**——它由代码在每轮重建（工具集、目标字数
都可能变），存下来只会造成「库里的 system 与代码里的不一致」。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
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


class AgentSession(Base, UUIDPrimaryKey, TimestampMixin):
    """一次「写故事」的会话。可以来回改稿，所以它有 n 轮 n 条正文。

    `input_text` 是用户粘贴的原文，**冻结在会话里**：它不随 `analysis_run`
    变化，也不依赖当前正在看的那次分析。这是「故事工坊是独立能力」的落点。
    """

    __tablename__ = "agent_session"

    title: Mapped[str] = mapped_column(Text, default="")
    #: 用户粘贴的评论/任意文本。**不裁剪**：`read_input` 工具要能分段读全文，
    #: 只把开头 3000 字塞进 system 提示词是 `prompts/story.py` 的事。
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    input_chars: Mapped[int] = mapped_column(Integer, default=0)

    status: Mapped[str] = mapped_column(String(16), default="idle", index=True)
    #: 面板上的那句「正在查询知识库…」。status 是机器读的，stage 是给人看的。
    stage: Mapped[str | None] = mapped_column(String(32))
    #: 已完成的轮数（一次 user 消息算一轮）。
    turn: Mapped[int] = mapped_column(Integer, default=0)

    #: 创建时冻结的选项：`allow_novel`（要不要给模型古典文学工具）、
    #: `target_chars`（目标字数）。改稿时要沿用同一份，否则「再长一点」
    #: 会被别的旋钮抵消。
    options: Mapped[dict] = mapped_column(JSON, default=dict)

    #: 累计统计。常驻显示在面板上——这是 agent 唯一可被核查的「自主程度」证据。
    rounds: Mapped[int] = mapped_column(Integer, default=0)
    tool_calls: Mapped[int] = mapped_column(Integer, default=0)

    #: 最近一次的正文。等于最后一条 assistant 消息，冗余一份是为了
    #: 「刷新后重建」不必再扫一遍消息表。
    story: Mapped[str | None] = mapped_column(Text)

    model: Mapped[str | None] = mapped_column(String(64))
    providers: Mapped[dict] = mapped_column(JSON, default=dict)

    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    #: 失败原因。**不把半截稿当成稿子**——出错时 story 保持上一次的值。
    error: Mapped[str | None] = mapped_column(Text)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AgentMessage(Base, UUIDPrimaryKey, TimestampMixin):
    """模型可见的对话。**库里这一串就是下一轮喂给模型的那一串。**

    `seq` 在会话内单调递增且唯一——它同时是 `agent_event` 对账的锚点。
    落库顺序与喂给模型的顺序必须逐字一致：任何一处漏写（最典型的是
    「带 tool_calls 的 assistant 没落库，只落了工具回复」）都会让第二轮
    请求被服务端 400 拒掉，而症状离现场很远。
    """

    __tablename__ = "agent_message"
    __table_args__ = (
        UniqueConstraint("session_id", "seq", name="uq_agent_message_seq"),
        Index("ix_agent_message_session_seq", "session_id", "seq"),
    )

    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("agent_session.id", ondelete="CASCADE"), index=True
    )
    seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    #: 第几轮。同一轮里可以有 消息→工具→工具…→正文 好几条。
    turn: Mapped[int] = mapped_column(Integer, default=1)

    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="")
    #: OpenAI 原样形状的 `[{id, name, arguments}]`。`arguments` 存**原始文本**，
    #: 因为解析失败时要把原文回给模型看。
    tool_calls: Mapped[list] = mapped_column(JSON, default=list)
    #: role="tool" 时必填。一条 assistant 的每次调用都要有一条对得上的回复。
    tool_call_id: Mapped[str | None] = mapped_column(String(64))
    #: role="tool" 时的工具名；role="user" 且为 `force_final` 时表示这条是
    #: 代码生成的收尾指令——前端据此不把它画成用户说的话。
    name: Mapped[str | None] = mapped_column(String(64))


class AgentEvent(Base, UUIDPrimaryKey, TimestampMixin):
    """SSE 事件的持久化副本。制度同 `run_event`：Streams 会被裁剪，这张表是
    「刷新后回放没有缺口」的兜底。

    与 `RunEvent` 的分工一样——`agent_event` 不是给模型看的，是给界面看的，
    所以它可以被裁剪、可以被合成（`_error_frame` 那种），而 `agent_message`
    一个字符都不能歪。
    """

    __tablename__ = "agent_event"
    __table_args__ = (
        UniqueConstraint("session_id", "seq", name="uq_agent_event_seq"),
        Index("ix_agent_event_session_seq", "session_id", "seq"),
    )

    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("agent_session.id", ondelete="CASCADE"), index=True
    )
    seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    message: Mapped[str | None] = mapped_column(Text)
    data: Mapped[dict | None] = mapped_column(JSON, nullable=True)


__all__ = ["AgentEvent", "AgentMessage", "AgentSession"]
