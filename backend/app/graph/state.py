"""流水线共享状态。

契约：**全部字段可 JSON 序列化**。这是检查点（AsyncPostgresSaver 把状态写进 PG）
与干预（aupdate_state 要能只 patch 一个字段就重跑）共同的前提。
所以这里不放 ORM 对象、不放 datetime、不放 numpy 数组——一律 dict / list / 标量。

字段分三类：
  输入      run_id / aweme_id / source_kind / source_text / text_mode / depth / kb /
            comment_limit / cluster_params / revision
  节点产物   video / comments / clusters / profile / evidence / reasoning / sections …
  累积字段   errors / warnings 用 operator.add；stale 用合并

节点**不碰数据库**。持久化集中在 graph/persist.py，由 runner 消费 updates 流时驱动。
这样每个节点都能单测——喂一个 state dict 进去，断言返回的 patch。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

Depth = Literal["quick", "standard", "deep"]


def merge_stale(left: dict[str, bool] | None, right: dict[str, bool] | None) -> dict[str, bool]:
    """stale 标记合并。右侧覆盖左侧——新一次干预的判定更权威。"""
    return {**(left or {}), **(right or {})}


def keep_last(left: Any, right: Any) -> Any:
    return right if right is not None else left


class KbSelection(TypedDict, total=False):
    psychology: bool
    literature: bool
    poetry: bool


class ClusterParams(TypedDict, total=False):
    min_cluster_size: int
    max_clusters: int
    force_kmeans: bool


class AnalysisState(TypedDict, total=False):
    # ---------------- 输入 ----------------
    run_id: str
    aweme_id: str
    source_url: str
    # "douyin" | "text"。节点据此选分支，**一律用 == 比较**：
    # state 会经检查点做 JSON 往返，StrEnum 回来时是普通 str，`is` 会静默为假。
    source_kind: str
    # 文本源的原文。与 source_url 分开是有意的：一个装链接、一个装正文。
    # 让 source_url 去装正文，那个名字就成了一句谎话。
    source_text: str
    text_mode: str
    depth: Depth
    kb: KbSelection
    comment_limit: int
    cluster_params: ClusterParams
    # 每次干预递增。段落用它判断自己是否基于旧状态生成（→ stale）
    revision: int

    # ---------------- Node 1：视频 ----------------
    video: dict[str, Any]
    # 口播文案是**可选**的——没有节点产生 ASR，只有 MCP 同时提供时才有。
    # 缺失时前端渲染明确空状态，绝不编造。
    transcript: dict[str, Any] | None

    # ---------------- Node 2：评论 ----------------
    # 已清洗并带 is_ad/is_spam/is_duplicate 标记，不是原始抓取结果
    comments: list[dict[str, Any]]
    comment_stats: dict[str, Any]

    # ---------------- Node 3：聚类 ----------------
    clusters: list[dict[str, Any]]
    cluster_meta: dict[str, Any]

    # ---------------- Node 4：心理语义 ----------------
    profile: dict[str, Any]

    # ---------------- Node 5：多路检索 ----------------
    queries: list[dict[str, Any]]
    evidence: list[dict[str, Any]]

    # ---------------- Node 6：推理 ----------------
    reasoning: list[dict[str, Any]]

    # ---------------- Node 7：文学化生成 ----------------
    sections: dict[str, dict[str, Any]]

    # ---------------- 累积字段（reducer） ----------------
    errors: Annotated[list[dict[str, Any]], operator.add]
    warnings: Annotated[list[dict[str, Any]], operator.add]
    # 段落 key → 是否因上游变更而失效
    stale: Annotated[dict[str, bool], merge_stale]


def initial_state(
    *,
    run_id: str,
    aweme_id: str,
    source_url: str = "",
    depth: Depth = "standard",
    kb: KbSelection | None = None,
    comment_limit: int = 100,
    cluster_params: ClusterParams | None = None,
    source_kind: str = "douyin",
    source_text: str = "",
    text_mode: str = "line",
) -> AnalysisState:
    return AnalysisState(
        run_id=run_id,
        aweme_id=aweme_id,
        source_url=source_url,
        source_kind=source_kind,
        source_text=source_text,
        text_mode=text_mode,
        depth=depth,
        kb=kb or {"psychology": True, "literature": True, "poetry": True},
        comment_limit=comment_limit,
        cluster_params=cluster_params or {},
        revision=1,
        video={},
        transcript=None,
        comments=[],
        comment_stats={},
        clusters=[],
        cluster_meta={},
        profile={},
        queries=[],
        evidence=[],
        reasoning=[],
        sections={},
        errors=[],
        warnings=[],
        stale={},
    )


def warning_message(code: str, message: str, **extra: Any) -> dict[str, Any]:
    """统一的 warning 载荷。code 供测试与前端分类，message 面向用户。"""
    return {"code": code, "message": message, **extra}


def error_message(code: str, message: str, *, node: str = "", **extra: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "node": node, **extra}
