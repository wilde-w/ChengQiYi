"""Node 4 — 心理语义抽取。

一次 `extract_psych` 调用拿到两样东西（为什么是一次而不是 k+1 次，
见 `prompts/extract_psych.py` 的模块 docstring）：

    cluster_tags  每个簇的情绪 / 话题 / 需求
    global_tags   全视频的同一组，外加 core_tension 与 summary

**分布不由模型给。** `emotions` / `topics` / `needs` 这三个 `[{label, value}]`
是 Python 按簇大小加权算出来的：把每个簇的标签按它的 `size` 累加，再降序。
理由是可核查——「这个情绪占多少」必须能由落库的字段复算出来。让模型估一个
数字，它既不可核查，又系统性地把最显眼的那个情绪报得过高（模型看到的是
代表评论，而代表评论是按点赞挑的，本身就偏）。

提示词给了 `global_tags`，Python 给了分布。两者会不一致（模型排的第一名
未必是加权后的第一名）。**这是刻意的**：卡片上「模型认为的主导情绪」与
「按人数算的主导情绪」并列显示，不一致本身就是一条信息——它说明最响的
声音和小部分人的声音指向不同的东西。

降级：模型失败时用簇自己的 `emotion_tags` 等字段拼一份 profile，
`method=fallback` 落进 summary 的来源标记，不编造 `core_tension`。
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from app.constants import REPRESENTATIVE_COMMENTS, REPRESENTATIVE_MAX_CHARS
from app.graph.emitting import NodeEmitter
from app.graph.state import AnalysisState, warning_message
from app.logging_conf import get_logger

log = get_logger(__name__)

NODE = "n4_psych"

#: 分布里最多保留几个标签。多出来的长尾在卡片上是几根 1px 的柱子，
#: 不提供信息却把前三名的宽度压小了。
DISTRIBUTION_TOP_N = 6


async def n4_psych(state: AnalysisState) -> dict[str, Any]:
    emitter = NodeEmitter(NODE)
    clusters = list(state.get("clusters") or [])

    if not clusters:
        emitter.set_progress(1.0)
        return {
            "profile": _empty_profile(reason="no_clusters"),
            "warnings": [warning_message("no_clusters_for_profile", "没有聚类结果，未生成心理侧写")],
        }

    profile, failed = await _extract(clusters, state.get("comments") or [])
    if failed:
        profile = _from_cluster_tags(clusters)

    emitter.partial("profile", profile)
    emitter.set_progress(1.0)

    patch: dict[str, Any] = {"profile": profile}
    if failed:
        patch["warnings"] = [
            warning_message("profile_degraded", "心理侧写未能由模型生成，已退回按簇标签汇总")
        ]
    return patch


# ----------------------------------------------------------------------
# 主路径
# ----------------------------------------------------------------------


async def _extract(
    clusters: list[dict[str, Any]], comments: list[dict[str, Any]]
) -> tuple[dict[str, Any], bool]:
    """调模型。返回 (profile, 是否降级)。**不抛异常**——n4 失败不该中断流水线。"""
    from app.graph.nodes._llm import call_json, clean_tags, clean_text
    from app.prompts import extract_psych as prompt
    from app.providers.base import ProviderError
    from app.providers.factory import get_chat_model

    # **代表评论在这里现取，不依赖 n3 留下的内部字段。** n3 的行里确实带过
    # 一份 `_representatives`，但它在 n3 返回前就被剥掉了（那是给 n3 自己的
    # 打标步骤用的，留在 state 里会跟着 checkpoint 写进 PG）。n4 拿到的
    # `clusters` 是**落库的那一份**，只有 `representative_comment_ids`。
    #
    # 曾经直接读 `_representatives`：n3 里剥、n4 里读，两条流水线在同一个
    # state 上错开一步，结果是 n4 拿到空的 comments 数组，模型只能凭关键词
    # 联想——而它照样返回一份看起来完整的 profile，没有任何报错。
    by_id = {str(c.get("comment_id")): str(c.get("text") or "") for c in comments}
    payload_clusters = [
        {
            "cluster_key": str(c.get("cluster_key") or ""),
            "size": int(c.get("size") or 0),
            "comments": _representative_texts(c, by_id),
        }
        for c in clusters
    ]

    model = get_chat_model()
    try:
        payload = await call_json(
            model,
            task=prompt.TASK,
            messages=prompt.build_messages(len(payload_clusters)),
            context=prompt.build_context(payload_clusters),
            temperature=0.2,
            max_tokens=1800,
        )
    except ProviderError as exc:
        log.warning("n4_extract_failed", error=str(exc))
        return _from_cluster_tags(clusters), True

    if not isinstance(payload, dict):
        return _from_cluster_tags(clusters), True

    produced = _parse_cluster_tags(payload.get("cluster_tags"), clusters)

    # **模型漏掉的簇，用 n3 的本地标签补上**，而不是留空。留空会让那个簇
    # 在分布里贡献 0，主导情绪凭空偏向别的簇——用户看到的是一个被扭曲的
    # 分布，且没有任何迹象表明它被扭曲过。
    merged: dict[str, dict[str, list[str]]] = {}
    for cluster in clusters:
        key = str(cluster.get("cluster_key") or "")
        merged[key] = produced.get(key) or _local_tags(cluster)

    from app.kb.lexical_tags import EMOTION_VOCAB

    raw_global = payload.get("global_tags") or {}
    # **全局这一份也要过同一个闭集。** 之前只有 `cluster_tags` 过，
    # 于是同一个越界词在逐簇那边被丢掉、在全局这边留下来，两张卡片
    # 对不上——而用户没有任何办法判断哪一边是对的。
    global_tags = {
        "emotion": clean_tags(raw_global.get("emotion"), allowed=EMOTION_VOCAB, limit=3),
        "topic": clean_tags(raw_global.get("topic"), limit=3),
        "need": clean_tags(raw_global.get("need"), limit=2),
    }

    profile = {
        **_distributions(merged, clusters),
        "global_tags": global_tags,
        "cluster_tags": merged,
        "core_tension": clean_text(payload.get("core_tension"), limit=80) or None,
        "summary": clean_text(payload.get("summary"), limit=300) or None,
        # 落进 PsychProfile.model 列——它记的是「这份侧写是谁产出的」，
        # 与簇的 params.method 一样是可信度线索。
        "model": getattr(model, "name", "model"),
    }
    return profile, False


def _representative_texts(
    cluster: dict[str, Any],
    by_id: dict[str, str],
    *,
    limit: int = REPRESENTATIVE_COMMENTS,
) -> list[str]:
    """按 `representative_comment_ids` 取回原文，取不到就退回关键词。

    退回关键词这一支不是可有可无的：`representative_comment_ids` 是**抖音 id**，
    而 `by_id` 来自 state 里的评论表。只要两者有一处对不上（老数据、
    手工重建的簇、评论被删），整簇就会变成空数组——而空数组会让模型
    按「没有评论」处理，产出的侧写看不出任何异常。给几个关键词，
    至少它知道这一簇在说什么。

    截断与 n3 用的是同一个 `REPRESENTATIVE_MAX_CHARS`：一条 500 字的评论
    会挤掉另外两条的信息量。
    """
    out: list[str] = []
    for comment_id in cluster.get("representative_comment_ids") or []:
        text = by_id.get(str(comment_id))
        if text:
            out.append(text[:REPRESENTATIVE_MAX_CHARS])
        if len(out) >= limit:
            return out
    if out:
        return out
    return [str(k) for k in (cluster.get("keywords") or [])[:limit]]


def _parse_cluster_tags(raw: Any, clusters: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    """把模型给的 `cluster_tags` 收干净。

    **只认识输入里出现过的 cluster_key。** 模型偶尔会自造一个键
    （把 `c0` 写成 `cluster_0`），放进结果里会在分布中多出一个没有成员的
    标签来源，而那个标签的权重无从计算——直接丢弃并记录。
    """
    from app.graph.nodes._llm import clean_tags
    from app.kb.lexical_tags import EMOTION_VOCAB

    known = {str(c.get("cluster_key") or "") for c in clusters}
    out: dict[str, dict[str, list[str]]] = {}
    if not isinstance(raw, dict):
        return out

    for key, value in raw.items():
        if str(key) not in known or not isinstance(value, dict):
            log.debug("n4_unknown_cluster_key", key=str(key))
            continue
        out[str(key)] = {
            "emotion": clean_tags(value.get("emotion"), allowed=EMOTION_VOCAB, limit=3),
            "topic": clean_tags(value.get("topic"), limit=3),
            "need": clean_tags(value.get("need"), limit=2),
        }
    return out


# ----------------------------------------------------------------------
# 分布（Python 算，不问模型）
# ----------------------------------------------------------------------


def _distributions(
    tags_by_cluster: dict[str, dict[str, list[str]]],
    clusters: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """按簇大小加权的分布。

    **权重是簇的 `size`，不是出现次数。** 一个 20 条的簇和一个 3 条的簇
    各说一次「哀伤」，在评论区里的分量当然不同；按次数算的话那个 3 条的
    簇会被放大到和 20 条的一样重，而它只是三个人的说法。

    `cluster_key` 带上是为了让前端能点一根柱子跳到对应的卡片。
    """
    weighting = {str(c.get("cluster_key") or ""): int(c.get("size") or 0) for c in clusters}
    out: dict[str, list[dict[str, Any]]] = {}

    for dimension in ("emotion", "topic", "need"):
        weights: Counter[str] = Counter()
        owner: dict[str, str] = {}
        for key, tags in tags_by_cluster.items():
            # **同一簇里重复的标签只算一次。** 标签在写进 `cluster_tags` 前
            # 已经过 `clean_tags` 去重了，所以这里正常情况下看不出差别；
            # 但分布的唯一职责就是「标签 → 有几个人在说它」，重复三次不等于
            # 三倍的人。哪天多出一个不过 `clean_tags` 的调用方，这里不该跟着错。
            for label in dict.fromkeys(tags.get(dimension) or []):
                weights[label] += weighting.get(key, 0)
                # 首次出现的簇记为归属。并列时前面的簇赢——必须确定，
                # 否则同一份输入两次跑出的分布归属会不一样。
                owner.setdefault(label, key)
        ranked = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))[:DISTRIBUTION_TOP_N]
        # 键名 emotion→emotions / topic→topics / need→needs，与 ORM 的
        # emotion_distribution / topic_distribution / need_distribution 及
        # ProfileOut 的三字段逐一对齐。
        out[f"{dimension}s"] = [
            {"label": label, "value": value, "cluster_key": owner.get(label)}
            for label, value in ranked
        ]
    return out


def _local_tags(cluster: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "emotion": list(cluster.get("emotion_tags") or []),
        "topic": list(cluster.get("topic_tags") or []),
        "need": list(cluster.get("need_tags") or []),
    }


# ----------------------------------------------------------------------
# 降级
# ----------------------------------------------------------------------


def _from_cluster_tags(clusters: list[dict[str, Any]]) -> dict[str, Any]:
    """模型不可用时的侧写：**只用 n3 已经落库的字段**，一个新词都不编。

    分布照常算（它本来就只是簇标签的加权），`core_tension` 留空——
    那是整个报告里最需要判断力的一句，用模板拼出来的一句「A 与 B 同时在场」
    比空着更糟：它看起来像结论，实际是字符串拼接。
    """
    from app.graph.nodes._llm import clean_tags
    from app.kb.lexical_tags import EMOTION_VOCAB

    tags = {str(c.get("cluster_key") or ""): _local_tags(c) for c in clusters}
    flat = {
        "emotion": [e for t in tags.values() for e in t["emotion"]],
        "topic": [t for x in tags.values() for t in x["topic"]],
        "need": [n for x in tags.values() for n in x["need"]],
    }
    return {
        **_distributions(tags, clusters),
        # 与主路径走**同一套参数**：即便来源（n3 的簇标签）已经过了闭集，
        # 两条路也得用同一个收口函数加同一份 `allowed`，改规则时才不会
        # 只改一半。之前这里漏了 `allowed`，全局标签于是放行越界词，
        # 而逐簇那张卡片上同一个词是被丢掉的——两张卡片对不上。
        "global_tags": {
            "emotion": clean_tags(flat["emotion"], allowed=EMOTION_VOCAB, limit=3),
            "topic": clean_tags(flat["topic"], limit=3),
            "need": clean_tags(flat["need"], limit=2),
        },
        "cluster_tags": tags,
        "core_tension": None,
        "summary": "心理侧写由簇标签汇总而成（模型未参与），张力与深层解读缺失。",
        "model": "fallback",
    }


def _empty_profile(*, reason: str) -> dict[str, Any]:
    """没有聚类结果时的空侧写。**形状与正常路径完全一致**，只是全空——
    前端因此不需要为「没有 profile」写第二条渲染分支。

    这里**不塞 `reason`**：正常路径没有这个键，塞了就破坏上面那句「完全一致」，
    而且它只活在 SSE 的那一帧里——落库走的是 `ProfileOut` 的固定字段，
    刷新后这个键就没了。空的原因由 `warnings` 那条 `no_clusters_for_profile`
    承载，它是落库的、刷新后还在的，用户也看得到。`reason` 只用于日志。
    """
    log.info("n4_empty_profile", reason=reason)
    return {
        "emotions": [],
        "topics": [],
        "needs": [],
        "global_tags": {"emotion": [], "topic": [], "need": []},
        "cluster_tags": {},
        "core_tension": None,
        "summary": None,
        "model": "none",
    }


__all__ = ["DISTRIBUTION_TOP_N", "NODE", "n4_psych"]
