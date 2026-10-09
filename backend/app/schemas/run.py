"""运行的请求与响应 DTO。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.constants import SourceKind, TextMode

Depth = Literal["quick", "standard", "deep"]


class KbSelection(BaseModel):
    psychology: bool = True
    literature: bool = True
    poetry: bool = True

    def enabled(self) -> list[str]:
        return [k for k, v in self.model_dump().items() if v]


class ClusterParams(BaseModel):
    min_cluster_size: int | None = Field(default=None, ge=2, le=50)
    max_clusters: int | None = Field(default=None, ge=2, le=12)
    force_kmeans: bool = False


#: 链接的长度上限。抖音链接、分享口令、裸 aweme_id 都用不到更长。
LINK_INPUT_MAX = 2000

#: 纯文本的长度上限。约等于一百条评论的五十倍，够装一篇长文。
#:
#: 有上限不只是审美：原文会随 state 一起写进每个节点检查点（一次运行 7+ 个），
#: 5 万字 ≈ 每次多写几十 KB 的 JSONB，再往上就该换存储方式了。
TEXT_INPUT_MAX = 50_000


class CreateRunRequest(BaseModel):
    # source 必须声明在 input **之前**：下面的校验器要读到它
    source: SourceKind = SourceKind.DOUYIN
    input: str = Field(
        min_length=1,
        max_length=TEXT_INPUT_MAX,
        description="链接 / 分享口令 / aweme_id；source=text 时是一段纯文本",
    )
    #: 仅 source=text 时有意义。source=douyin 时**接受但忽略**——
    #: 为一个无意义的字段让抖音运行 422，代价和收益不成比例。
    text_mode: TextMode = TextMode.LINE
    depth: Depth = "standard"
    kb: KbSelection = Field(default_factory=KbSelection)
    comment_limit: int = Field(default=100, ge=1, le=2000)
    cluster_params: ClusterParams = Field(default_factory=ClusterParams)

    @field_validator("input")
    @classmethod
    def _strip(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("输入不能为空")
        return text

    @model_validator(mode="after")
    def _limit_by_source(self) -> CreateRunRequest:
        """上限按源分治：文本可以很长，链接不该跟着一起放宽。

        写在字段级 `max_length` 里做不到——那里看不见同请求里的 `source`，
        写死 2000 会把长文挡在门外，写死 50000 又等于对链接一起放宽。
        """
        if self.source == SourceKind.DOUYIN and len(self.input) > LINK_INPUT_MAX:
            raise ValueError(
                f"链接最长 {LINK_INPUT_MAX} 字符；要分析一段长文本，请把 source 设为 text"
            )
        return self


class RunSummary(BaseModel):
    id: str
    status: str
    input_raw: str
    aweme_id: str | None = None
    #: 数据源种类。前端据它决定左栏怎么说话（「N 条评论」还是「手动文本 · N 条」），
    #: 不必自己去猜 `aweme_id is None` 这条隐式规则。默认值让老行照样能序列化。
    source_kind: str = SourceKind.DOUYIN.value
    depth: str
    progress: int
    current_node: str | None = None
    revision: int
    providers: dict[str, Any] = Field(default_factory=dict)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    duration_ms: int | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class VideoOut(BaseModel):
    aweme_id: str
    title: str | None = None
    caption: str | None = None
    author_name: str | None = None
    author_avatar: str | None = None
    publish_time: datetime | None = None
    duration_ms: int | None = None
    cover_url: str | None = None
    share_url: str | None = None
    stats: dict[str, Any] = Field(default_factory=dict)
    transcript: dict[str, Any] | None = None
    transcript_source: str | None = None


class CommentOut(BaseModel):
    # 与 SSE 的 comment_page 同名字段（见 api/v1/runs.py 的 _comment）。
    # 两处不同名会让前端在快照对账后拿到一批无 key 的列表项。
    comment_id: str
    text: str
    author_name: str | None = None
    like_count: int
    reply_count: int
    publish_time: datetime | None = None
    is_ad: bool
    is_spam: bool
    is_duplicate: bool
    filter_reason: str | None = None
    cluster_key: str | None = None


class ClusterOut(BaseModel):
    cluster_key: str
    label: str
    summary: str | None = None
    size: int
    raw_size: int
    is_noise: bool
    emotion_tags: list[str] = Field(default_factory=list)
    topic_tags: list[str] = Field(default_factory=list)
    need_tags: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    color: str
    order_index: int


class ProfileOut(BaseModel):
    emotions: list[dict[str, Any]] = Field(default_factory=list)
    topics: list[dict[str, Any]] = Field(default_factory=list)
    needs: list[dict[str, Any]] = Field(default_factory=list)
    # 按维度分组，不是一个大平表：{emotion: [...], topic: [...], need: [...]}。
    # 侧写卡要分三行渲染，而且 mock 与提示词两边都按这个形状读
    # （`global_tags.get("emotion")`）。声明成 list[str] 时 pydantic 会
    # 把整个 dict 塞进字符串列表，右栏那段情绪列表恒为空。
    global_tags: dict[str, list[str]] = Field(default_factory=dict)
    cluster_tags: dict[str, Any] = Field(default_factory=dict)
    core_tension: str | None = None
    summary: str | None = None
    # 产出这份侧写的模型名，降级时是 `fallback`、无簇时是 `none`。
    # 与簇的 `params.method` 同类：它决定这张卡片该不该挂「兜底」的牌子。
    model: str | None = None


class EvidenceOut(BaseModel):
    id: str
    kind: str
    library: str
    chunk_id: str
    title: str | None = None
    source: str | None = None
    author: str | None = None
    text: str
    score: float
    retrieval_path: str
    match_reason: str | None = None
    cluster_ids: list[str] = Field(default_factory=list)
    pinned: bool
    excluded: bool
    rank: int
    payload: dict[str, Any] = Field(default_factory=dict)


class ReasoningStepOut(BaseModel):
    step_index: int
    phenomenon: str
    mechanism: str
    insight: str
    evidence_ids: list[str] = Field(default_factory=list)
    allusion_ids: list[str] = Field(default_factory=list)
    confidence: float


class CitationOut(BaseModel):
    evidence_id: str
    marker: str
    text: str | None = None


class SectionOut(BaseModel):
    key: str
    title: str
    content_md: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    version: int
    stale: bool
    based_on_revision: int
    edited_by_user: bool
    citation_coverage: float
    model: str | None = None


class RunDetail(RunSummary):
    """右栏与中栏的一次性快照。

    流负责实时性，快照负责真实性——收到 run_completed 后前端会拉一次这个，
    对流式过程中可能出现的错漏做对账。
    """

    video: VideoOut | None = None
    comments: list[CommentOut] = Field(default_factory=list)
    comment_stats: dict[str, Any] = Field(default_factory=dict)
    clusters: list[ClusterOut] = Field(default_factory=list)
    cluster_meta: dict[str, Any] = Field(default_factory=dict)
    profile: ProfileOut | None = None
    evidence: list[EvidenceOut] = Field(default_factory=list)
    reasoning: list[ReasoningStepOut] = Field(default_factory=list)
    sections: list[SectionOut] = Field(default_factory=list)
    stale: dict[str, bool] = Field(default_factory=dict)


class CreateRunResponse(BaseModel):
    run: RunSummary
    resolved: dict[str, Any]
    demo: bool
