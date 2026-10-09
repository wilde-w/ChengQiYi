"""运行相关的接口：创建、查询、取消、评论分页。

**序列化集中在这里，不在 service 层。** service 返回 ORM 对象，路由把它们
映射成 DTO——这样 service 不依赖任何 HTTP 概念，反过来 DTO 也不必知道
ORM 有几个字段。映射函数都放在文件底部，一眼能看出「线上契约是什么」。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.constants import SectionKey
from app.logging_conf import get_logger
from app.schemas.run import (
    CreateRunRequest,
    CreateRunResponse,
    RunDetail,
    RunSummary,
)
from app.services import run_service
from app.services.comment_service import tally_flags

log = get_logger(__name__)
router = APIRouter(tags=["runs"])


@router.post("/runs", response_model=CreateRunResponse, status_code=201)
async def create_run(payload: CreateRunRequest) -> Any:
    """创建一次分析。立刻返回，本体在后台跑。

    解析链接是同步做的：`input` 不合法应当在这里就报 400，
    而不是先建一条记录再让它悄悄失败。失败原因回给用户，他改一下就能重试。
    """
    from app.config import get_settings

    try:
        run, resolved = await run_service.create_run(payload)
    except run_service.ResolveFailed as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "run": _summary(run),
        "resolved": resolved,
        "demo": get_settings().is_demo,
    }


@router.get("/runs", response_model=list[RunSummary])
async def list_runs(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> Any:
    return [_summary(run) for run in await run_service.list_runs(limit=limit, offset=offset)]


@router.get("/runs/{run_id}", response_model=RunDetail)
async def get_run(run_id: str) -> Any:
    """一次运行的全部产物。

    流负责实时性，这个接口负责真实性：前端收到 `run_completed` 后会拉它做
    终局对账。运行中断线重连失败、或页面在终态之后才打开，也都落到这里。
    """
    try:
        snap = await run_service.run_snapshot(run_id)
    except run_service.RunNotFound as exc:
        raise HTTPException(status_code=404, detail="运行不存在") from exc

    run = snap["run"]
    comments = snap["comments"]
    clusters = snap["clusters"]
    key_by_id = snap["cluster_key_by_id"]

    stats = tally_flags(comments).to_dict()
    # 聚类自身的元信息（算法/参数/是否退化为 KMeans）由 n3 写进 cluster.params。
    # 每个簇各存一份是冗余的——它们同属一次聚类——但省掉一张表，
    # 而且维度分析里「这一簇是在什么参数下分出来的」本来就该跟着簇走。
    cluster_meta = (clusters[0].params or {}) if clusters else {}

    return {
        **_summary(run),
        "video": _video(snap["video"]),
        "comments": [_comment(c, key_by_id) for c in comments],
        "comment_stats": stats,
        "clusters": [_cluster(c) for c in clusters],
        "cluster_meta": cluster_meta,
        "profile": _profile(snap["profile"]),
        "evidence": [_evidence(e) for e in snap["evidence"]],
        "reasoning": [_reasoning(r) for r in snap["reasoning"]],
        "sections": [_section(s) for s in snap["sections"]],
        "stale": {
            s.key: bool(s.stale or s.based_on_revision < int(run.revision or 0))
            for s in snap["sections"]
        },
    }


@router.post("/runs/{run_id}/cancel", status_code=202)
async def cancel_run(run_id: str) -> dict[str, Any]:
    """请求取消。**置标志位而非 kill 任务**，让 runner 在节点边界收尾。

    立刻强杀会跳过事务提交与结束事件，前端就卡在一条永远不结束的流上。
    """
    try:
        accepted = await run_service.request_cancel(run_id)
    except run_service.RunNotFound as exc:
        raise HTTPException(status_code=404, detail="运行不存在") from exc
    return {
        "accepted": accepted,
        "message": "已请求取消，将在当前节点结束后停止" if accepted else "运行已结束",
    }


# ----------------------------------------------------------------------
# ORM → DTO
# ----------------------------------------------------------------------


def _summary(run: Any) -> dict[str, Any]:
    return {
        "id": run.id,
        "status": run.status,
        "input_raw": run.input_raw,
        "aweme_id": run.aweme_id,
        # 走 source_kind_of 而不是直接读 providers 里的键：判别式（含老行的
        # 回落）只有那一处实现，序列化代码抄一份就等于把它变成两处。
        "source_kind": run_service.source_kind_of(run).value,
        "depth": run.depth,
        "progress": int(run.progress or 0),
        "current_node": run.current_node,
        "revision": int(run.revision or 1),
        "providers": run.providers or {},
        "warnings": run.warnings or [],
        "error": run.error,
        "duration_ms": run.duration_ms,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
    }


def _video(row: Any) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "aweme_id": row.aweme_id,
        "title": row.title,
        "caption": row.caption,
        "author_name": row.author_name,
        "author_avatar": row.author_avatar,
        "publish_time": row.publish_time,
        "duration_ms": row.duration_ms,
        "cover_url": row.cover_url,
        "share_url": row.share_url,
        "stats": row.stats or {},
        "transcript": row.transcript,
        "transcript_source": row.transcript_source,
    }


def _comment(row: Any, key_by_id: dict[str, str]) -> dict[str, Any]:
    return {
        # 字段名与 SSE 的 comment_page 保持一致（那边是 douyin_comment_id，
        # 这边是行 id）。两边不同名的话，快照对账后 React 会拿到
        # 一堆 `key=undefined` 的列表项——重复 key 警告只是症状，
        # 真正的后果是列表更新时组件身份错乱。
        "comment_id": row.id,
        "text": row.text,
        "author_name": row.author_name,
        "like_count": int(row.like_count or 0),
        "reply_count": int(row.reply_count or 0),
        "publish_time": row.publish_time,
        "is_ad": bool(row.is_ad),
        "is_spam": bool(row.is_spam),
        "is_duplicate": bool(row.is_duplicate),
        "filter_reason": row.filter_reason,
        "cluster_key": key_by_id.get(row.cluster_id or ""),
    }


def _cluster(row: Any) -> dict[str, Any]:
    return {
        "cluster_key": row.cluster_key,
        "label": row.label,
        "summary": row.summary,
        "size": int(row.size or 0),
        "raw_size": int(row.raw_size or 0),
        "is_noise": bool(row.is_noise),
        "emotion_tags": row.emotion_tags or [],
        "topic_tags": row.topic_tags or [],
        "need_tags": row.need_tags or [],
        "keywords": row.keywords or [],
        "color": row.color or "#8B5CF6",
        "order_index": int(row.order_index or 0),
    }


def _profile(row: Any) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "emotions": row.emotion_distribution or [],
        "topics": row.topic_distribution or [],
        "needs": row.need_distribution or [],
        "global_tags": row.global_tags or {},
        "cluster_tags": row.cluster_tags or {},
        "core_tension": row.core_tension,
        "summary": row.summary,
        # 「这份侧写是谁产出的」与簇的 params.method 一样是可信度线索：
        # `fallback` 意味着它只是各主题标签的汇总，没有张力也没有解读，
        # 而卡片长得和正常侧写一模一样。不给前端这个字段，它就只能装作
        # 两份是同一回事。
        "model": row.model,
    }


def _evidence(row: Any) -> dict[str, Any]:
    return {
        "id": row.id,
        "kind": row.kind,
        "library": row.library,
        "chunk_id": row.chunk_id,
        "title": row.title,
        "source": row.source,
        "author": row.author,
        "text": row.text,
        "score": float(row.score or 0.0),
        "retrieval_path": row.retrieval_path,
        "match_reason": row.match_reason,
        "cluster_ids": row.cluster_ids or [],
        "pinned": bool(row.pinned),
        "excluded": bool(row.excluded),
        "rank": int(row.rank or 0),
        "payload": row.payload or {},
    }


def _reasoning(row: Any) -> dict[str, Any]:
    return {
        "step_index": int(row.step_index or 0),
        "phenomenon": row.phenomenon,
        "mechanism": row.mechanism,
        "insight": row.insight,
        "evidence_ids": row.evidence_ids or [],
        "allusion_ids": row.allusion_ids or [],
        "confidence": float(row.confidence or 0.0),
    }


def _section(row: Any) -> dict[str, Any]:
    try:
        title = SectionKey(row.key).title
    except ValueError:
        title = row.title or row.key
    return {
        "key": row.key,
        "title": row.title or title,
        "content_md": row.content_md or "",
        "citations": row.citations or [],
        "version": int(row.version or 1),
        "stale": bool(row.stale),
        "based_on_revision": int(row.based_on_revision or 0),
        "edited_by_user": bool(row.edited_by_user),
        "citation_coverage": float(row.citation_coverage or 0.0),
        "model": row.model,
    }
