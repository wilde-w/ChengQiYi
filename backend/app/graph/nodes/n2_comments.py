"""Node 2 — 评论抓取与清洗。

这是「流式感」最主要的来源：每抓完一页就发一个 partial，
左栏列表因此是**真的在增长**，而不是等全部抓完一次性刷出来。
进度也由真实分页驱动（已获取 / 目标），不是写死的百分比。

清洗（广告/灌水/去重）放在抓取之后、聚类之前，一步完成：
后面的节点只该看到干净数据，不该各自再判一次。
被过滤的评论**保留在结果里**并带 filter_reason——前端「显示被过滤内容」
开关需要它们，可审计性也要求它们还在。
"""

from __future__ import annotations

from typing import Any

from app.constants import SourceKind, TextMode
from app.douyin.base import DouyinError
from app.graph.emitting import NodeEmitter
from app.graph.state import AnalysisState, warning_message
from app.logging_conf import get_logger
from app.services.comment_service import clean_comments
from app.services.text_source import TEXT_ITEM_MAX_CHARS, split_text

log = get_logger(__name__)

NODE = "n2_comments"

# 单次请求条数。抖音一页约 20 条，请求过多会被截断。
PAGE_SIZE = 20
# 上限保护：即使 comment_limit 配得很大，也不无限翻页
MAX_PAGES = 50


async def n2_comments(state: AnalysisState) -> dict[str, Any]:
    emitter = NodeEmitter(NODE)

    # 文本源不抓评论——评论就是用户粘进来的那段文字。分支同样要在
    # provider 导入与 state["aweme_id"] 之前。
    if state.get("source_kind") == SourceKind.TEXT:
        return _from_text(state, emitter)

    from app.douyin.factory import get_douyin_provider

    provider = get_douyin_provider()
    aweme_id = state["aweme_id"]
    limit = max(1, int(state.get("comment_limit") or 100))

    raw_items: list[Any] = []
    cursor: str | None = None
    total: int | None = None
    failure: str | None = None

    for page in range(MAX_PAGES):
        try:
            chunk = await provider.get_comments(aweme_id, cursor=cursor, count=PAGE_SIZE)
        except DouyinError as exc:
            log.warning("n2_comments_failed", page=page, error=str(exc))
            failure = str(exc)
            break

        raw_items.extend(chunk.items)
        total = chunk.total if chunk.total is not None else total
        cursor = chunk.next_cursor

        # 逐页广播，左栏据此实时增长。带上清洗预览以便徽章计数同步更新。
        emitter.partial(
            "comment_page",
            {
                "items": [_raw_payload(c) for c in chunk.items],
                "received": len(raw_items),
                "total": total,
                "page": page + 1,
                "has_more": chunk.has_more,
            },
        )
        emitter.set_progress(min(1.0, len(raw_items) / limit))

        if not chunk.has_more or cursor is None or len(raw_items) >= limit:
            break

    raw_items = raw_items[:limit]

    if not raw_items and failure is not None:
        # 一条都没拿到才算失败。部分失败照常出报告。
        return {
            "comments": [],
            "comment_stats": {"total": 0, "kept": 0},
            "warnings": [warning_message("comments_unavailable", f"未能获取评论：{failure}")],
        }

    cleaned, stats = clean_comments(raw_items)
    payloads = [c.to_payload() for c in cleaned]

    emitter.partial("comment_stats", stats.to_dict())
    if failure is not None:
        emitter.warn("comments_partial", f"评论抓取提前结束：{failure}")

    emitter.set_progress(1.0)

    patch: dict[str, Any] = {"comments": payloads, "comment_stats": stats.to_dict()}
    if failure is not None:
        patch["warnings"] = [warning_message("comments_partial", failure)]
    return patch


def _from_text(state: AnalysisState, emitter: NodeEmitter) -> dict[str, Any]:
    """文本源：切分 → 清洗。

    切完之后与抖音路径**合流**：同一个 `clean_comments`、同样的
    `comment_page` / `comment_stats` 事件、同样的双通道 warning。
    两条路径只在上游不同，下游一个字都不该分叉。

    **不伪造分页节奏**：一次发完，`page=1, has_more=False`。文本已经在手里，
    没有任何东西在「到达」；假装成二十条一页地推给前端，是为了好看而说的一句谎。
    """
    limit = max(1, int(state.get("comment_limit") or 100))
    items, split = split_text(
        str(state.get("source_text") or ""),
        mode=state.get("text_mode") or TextMode.LINE,
        limit=limit,
    )

    if items:
        emitter.partial(
            "comment_page",
            {
                "items": [_raw_payload(c) for c in items],
                "received": len(items),
                "total": split.total,
                "page": 1,
                "has_more": False,
            },
        )

    cleaned, stats = clean_comments(items)
    payloads = [c.to_payload() for c in cleaned]
    emitter.partial("comment_stats", stats.to_dict())
    emitter.set_progress(1.0)

    warnings: list[dict[str, Any]] = []
    if not items:
        # create_run 已经在入口拦过一次，但干预重跑与直接改 limit 都绕得过去。
        # 走到这里就如实说「没东西可分析」，而不是给一份看起来正常的空报告。
        warnings.append(warning_message("empty_text", "没有可分析的文本：内容为空或全是空行"))
    if split.dropped:
        warnings.append(
            warning_message(
                "text_truncated",
                f"文本共 {split.total} 条，超出上限 {limit} 条，已分析前 {split.kept} 条",
            )
        )
    if split.truncated:
        warnings.append(
            warning_message(
                "text_item_truncated",
                f"{split.truncated} 条内容过长，已截断到 {TEXT_ITEM_MAX_CHARS} 字",
            )
        )
    for item in warnings:
        emitter.warn(str(item["code"]), str(item["message"]))

    patch: dict[str, Any] = {"comments": payloads, "comment_stats": stats.to_dict()}
    if warnings:
        patch["warnings"] = warnings
    return patch


def _raw_payload(item: Any) -> dict[str, Any]:
    """逐页广播时的精简载荷。

    这里发的是**未清洗**的原始条目——清洗要等全部抓完才能做去重
    （重复判定是全局的）。所以左栏在流式阶段可能出现稍后会被标记为
    广告的条目，等 comment_stats 到达后再由前端折叠掉。
    """
    return {
        "comment_id": item.comment_id,
        "text": item.text,
        "author_name": item.author_name,
        "like_count": item.like_count,
        "reply_count": item.reply_count,
        "publish_time": item.publish_time.isoformat() if item.publish_time else None,
    }
