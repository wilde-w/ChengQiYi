"""节点产物的持久化。

**节点自己不碰数据库。** 所有写库集中在这里，由 runner 消费 `updates` 流时驱动。
换来两件事：节点可以脱离数据库单测；从任意节点恢复时，写库逻辑不会成为隐藏状态。

每个节点的写入都是「先删该 run 下的旧行，再插新行」。
干预会重跑节点，幂等替换比增量 diff 简单得多，也不会留下孤儿行。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.constants import NodeKey, SectionKey
from app.logging_conf import get_logger
from app.models.insight import Evidence, InsightSection, ReasoningStep
from app.models.run import Cluster, Comment, PsychProfile, Video
from app.services.comment_service import dedup_key

log = get_logger(__name__)


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


async def apply_patch(
    session: AsyncSession,
    *,
    run_id: str,
    node: str,
    patch: dict[str, Any],
    revision: int,
) -> None:
    """把某个节点返回的 patch 落到库里。未知节点直接忽略——不是错误。"""
    try:
        key = NodeKey(node)
    except ValueError:
        return

    handler = {
        NodeKey.N1_VIDEO: _persist_video,
        NodeKey.N2_COMMENTS: _persist_comments,
        NodeKey.N3_CLUSTER: _persist_clusters,
        NodeKey.N4_PSYCH: _persist_profile,
        NodeKey.N5_RETRIEVAL: _persist_evidence,
        NodeKey.N6_REASONING: _persist_reasoning,
        NodeKey.N7_LITERARY: _persist_sections,
    }[key]

    await handler(session, run_id=run_id, patch=patch, revision=revision)


# ----------------------------------------------------------------------
# n1
# ----------------------------------------------------------------------


async def _persist_video(
    session: AsyncSession, *, run_id: str, patch: dict[str, Any], revision: int
) -> None:
    data = patch.get("video") or {}
    if not data:
        return
    transcript = patch.get("transcript")

    existing = (
        await session.execute(select(Video).where(Video.run_id == run_id))
    ).scalar_one_or_none()
    video = existing or Video(run_id=run_id, aweme_id=str(data.get("aweme_id") or ""))

    video.aweme_id = str(data.get("aweme_id") or video.aweme_id)
    video.title = data.get("title")
    video.author_name = data.get("author_name")
    video.author_id = data.get("author_id")
    video.author_avatar = data.get("author_avatar")
    video.caption = data.get("caption")
    video.publish_time = _parse_dt(data.get("publish_time"))
    video.duration_ms = data.get("duration_ms")
    video.cover_url = data.get("cover_url")
    video.share_url = data.get("share_url")
    video.stats = data.get("stats") or {}
    video.raw = data.get("raw") or {}
    video.transcript = transcript
    video.transcript_source = (transcript or {}).get("source")

    if existing is None:
        session.add(video)


# ----------------------------------------------------------------------
# n2
# ----------------------------------------------------------------------


async def _persist_comments(
    session: AsyncSession, *, run_id: str, patch: dict[str, Any], revision: int
) -> None:
    rows = patch.get("comments")
    if rows is None:
        return

    await session.execute(delete(Comment).where(Comment.run_id == run_id))

    for row in rows:
        text = str(row.get("text") or "")
        session.add(
            Comment(
                run_id=run_id,
                douyin_comment_id=str(row.get("comment_id") or "") or None,
                text=text,
                text_normalized=text,
                text_hash=dedup_key(text),
                author_name=row.get("author_name"),
                author_id=row.get("author_id"),
                like_count=int(row.get("like_count") or 0),
                reply_count=int(row.get("reply_count") or 0),
                publish_time=_parse_dt(row.get("publish_time")),
                is_ad=bool(row.get("is_ad")),
                is_spam=bool(row.get("is_spam")),
                is_duplicate=bool(row.get("is_duplicate")),
                filter_reason=row.get("filter_reason"),
                raw={},
            )
        )


# ----------------------------------------------------------------------
# n3
# ----------------------------------------------------------------------


async def _persist_clusters(
    session: AsyncSession, *, run_id: str, patch: dict[str, Any], revision: int
) -> None:
    rows = patch.get("clusters")
    if rows is None:
        return

    await session.execute(delete(Cluster).where(Cluster.run_id == run_id))

    membership: dict[str, str] = {}
    for index, row in enumerate(rows):
        cluster = Cluster(
            run_id=run_id,
            cluster_key=str(row.get("cluster_key") or f"c{index}"),
            label=str(row.get("label") or ""),
            summary=row.get("summary"),
            size=int(row.get("size") or 0),
            raw_size=int(row.get("raw_size") or row.get("size") or 0),
            is_noise=bool(row.get("is_noise")),
            keywords=row.get("keywords") or [],
            emotion_tags=row.get("emotion_tags") or [],
            topic_tags=row.get("topic_tags") or [],
            need_tags=row.get("need_tags") or [],
            representative_comment_ids=row.get("representative_comment_ids") or [],
            centroid=row.get("centroid"),
            color=str(row.get("color") or ""),
            order_index=int(row.get("order_index", index)),
            params=row.get("params") or {},
        )
        session.add(cluster)
        await session.flush()   # 需要 cluster.id 来回填 comment.cluster_id
        for comment_id in membership_of(row):
            membership[comment_id] = cluster.id

    if membership:
        comments = (
            await session.execute(
                select(Comment).where(
                    Comment.run_id == run_id,
                    Comment.douyin_comment_id.in_(list(membership)),
                )
            )
        ).scalars()
        for comment in comments:
            comment.cluster_id = membership.get(comment.douyin_comment_id or "")


def membership_of(cluster_row: dict[str, Any]) -> list[str]:
    """簇里包含哪些评论。聚类节点把成员 id 放在 `comment_ids`。"""
    return [str(c) for c in (cluster_row.get("comment_ids") or [])]


# ----------------------------------------------------------------------
# n4
# ----------------------------------------------------------------------


async def _persist_profile(
    session: AsyncSession, *, run_id: str, patch: dict[str, Any], revision: int
) -> None:
    data = patch.get("profile")
    if not data:
        return

    existing = (
        await session.execute(select(PsychProfile).where(PsychProfile.run_id == run_id))
    ).scalar_one_or_none()
    profile = existing or PsychProfile(run_id=run_id)

    profile.revision = revision
    profile.emotion_distribution = data.get("emotions") or []
    profile.topic_distribution = data.get("topics") or []
    profile.need_distribution = data.get("needs") or []
    # 兜底必须是 {}：列是 dict，写进去一个空列表的话读出来是 list，
    # `global_tags.get("emotion")` 会 AttributeError。
    profile.global_tags = data.get("global_tags") or {}
    profile.cluster_tags = data.get("cluster_tags") or {}
    profile.core_tension = data.get("core_tension")
    profile.summary = data.get("summary")
    profile.model = data.get("model")

    if existing is None:
        session.add(profile)


# ----------------------------------------------------------------------
# n5
# ----------------------------------------------------------------------


async def _persist_evidence(
    session: AsyncSession, *, run_id: str, patch: dict[str, Any], revision: int
) -> None:
    rows = patch.get("evidence")
    if rows is None:
        return

    # 保留用户的 pin/exclude 决策：干预重跑检索时不该把人工判断抹掉
    prior = (
        await session.execute(
            select(Evidence).where(Evidence.run_id == run_id, Evidence.pinned.is_(True))
        )
    ).scalars()
    pinned_chunks = {e.chunk_id for e in prior}

    await session.execute(delete(Evidence).where(Evidence.run_id == run_id))

    for rank, row in enumerate(rows):
        session.add(
            Evidence(
                run_id=run_id,
                revision=revision,
                kind=str(row.get("kind") or ""),
                library=str(row.get("library") or ""),
                chunk_id=str(row.get("chunk_id") or ""),
                title=row.get("title"),
                source=row.get("source"),
                author=row.get("author"),
                text=str(row.get("text") or ""),
                score=float(row.get("score") or 0.0),
                retrieval_path=str(row.get("retrieval_path") or "vector"),
                match_reason=row.get("match_reason"),
                cluster_ids=row.get("cluster_ids") or [],
                pinned=bool(row.get("pinned")) or str(row.get("chunk_id")) in pinned_chunks,
                excluded=bool(row.get("excluded")),
                rank=int(row.get("rank", rank)),
                payload=row.get("payload") or {},
            )
        )


# ----------------------------------------------------------------------
# n6
# ----------------------------------------------------------------------


async def _persist_reasoning(
    session: AsyncSession, *, run_id: str, patch: dict[str, Any], revision: int
) -> None:
    rows = patch.get("reasoning")
    if rows is None:
        return

    await session.execute(delete(ReasoningStep).where(ReasoningStep.run_id == run_id))

    for index, row in enumerate(rows):
        session.add(
            ReasoningStep(
                run_id=run_id,
                revision=revision,
                step_index=int(row.get("step_index", index)),
                phenomenon=str(row.get("phenomenon") or ""),
                mechanism=str(row.get("mechanism") or ""),
                insight=str(row.get("insight") or ""),
                evidence_ids=row.get("evidence_ids") or [],
                allusion_ids=row.get("allusion_ids") or [],
                confidence=float(row.get("confidence") or 0.0),
                model=row.get("model"),
            )
        )


# ----------------------------------------------------------------------
# n7
# ----------------------------------------------------------------------


async def _persist_sections(
    session: AsyncSession, *, run_id: str, patch: dict[str, Any], revision: int
) -> None:
    sections = patch.get("sections")
    if not sections:
        return

    existing = {
        s.key: s
        for s in (
            await session.execute(
                select(InsightSection).where(InsightSection.run_id == run_id)
            )
        ).scalars()
    }

    for key, data in sections.items():
        if key not in {k.value for k in SectionKey}:
            continue
        try:
            section_key = SectionKey(key)
        except ValueError:
            continue

        section = existing.get(key)
        is_new = section is None
        if is_new:
            section = InsightSection(run_id=run_id, key=key)

        section.title = str(data.get("title") or section_key.title)
        section.content_md = str(data.get("content_md") or "")
        section.content_json = data.get("content_json")
        section.citations = data.get("citations") or []
        section.citation_coverage = float(data.get("citation_coverage") or 0.0)
        section.model = data.get("model")

        # 版本号只在这里递增：干预重跑会生成新版本，而用户手动编辑
        # 走 section_service 的轻路径（也递增）。两者都不覆盖历史。
        if is_new:
            section.version = int(data.get("version") or 1)
            section.based_on_revision = revision
            section.stale = False
            section.edited_by_user = False
            session.add(section)
        elif not section.edited_by_user:
            section.version = int(data.get("version") or section.version or 1) + 1
            section.based_on_revision = revision
            section.stale = False
