"""Node 6 — 推理综合。

一次 `synthesize` 调用产出整条链，然后 **Python 做四件事，一件都不信模型**：

  1. 丢掉引用了检索集之外的 id（并记账）
  2. `allusion_ids` 只留文学与诗词——心理学语料不是典故
  3. **重算 `confidence`**，模型自评整个丢掉
  4. 步数截到 `MAX_REASONING_STEPS`

第 3 条原本打算把模型自评留档到 `payload.model_confidence` 并排显示，落库时
才发现 `reasoning_step` 表**没有 payload 列**——那样它只活在流式那一帧里，
刷新一次就没了，正好是「界面上两个数字对不上」的经典成因。既然它本来就不
参与展示值，索性不留。

## 为什么 confidence 要重算

它是给用户判断可信度用的。模型的自评系统性偏高（它给自己写的每一句都
觉得挺有把握），而这个数字旁边没有任何东西能让用户验证。所以这里用一个
**只能由落库字段复算**的公式：

    support = min(1, 有效 evidence_ids / 2)
    mech    = 1.0 引用了心理学证据且 mechanism 非空
              0.5 池子里根本没有心理学证据（不怪这一步）
              0.0 其余
    allu    = 1.0 引用了典故
              0.5 池子里根本没有文学/诗词证据
              0.0 其余
    confidence = round(0.55*support + 0.30*mech + 0.15*allu, 3)

三项都是「有没有、有几条」，全部可从证据表与步骤本身读出来。用户点开
证据卡就能自己数一遍，这是这个数字敢显示出来的前提。

## 「典故」这一层为什么没有文字

`ReasoningStep` 只有 phenomenon / mechanism / insight 三段，典故靠
`allusion_ids` 指向证据卡。在推理里把典故原文抄一遍的话，同一段文字就有
了两份副本，而用户看到的是「推理里引的那句」和「卡片上的原文」对不上——
文学段落的原文、出处、检索路径只有一份，才不会有漂移。
"""

from __future__ import annotations

from typing import Any

from app.constants import (
    CONFIDENCE_WEIGHTS,
    MAX_REASONING_STEPS,
    SUPPORT_FULL_MARKS,
    Library,
)
from app.graph.emitting import NodeEmitter
from app.graph.nodes._llm import call_json
from app.graph.state import AnalysisState
from app.graph.validators import filter_ids
from app.logging_conf import get_logger
from app.prompts import reasoning as prompt

log = get_logger(__name__)

NODE = "n6_reasoning"

#: 引用了「典故」的库。心理学语料不是典故，写进 allusion_ids 的一律丢弃。
_ALLUSION_LIBRARIES = frozenset({str(Library.LITERATURE), str(Library.POETRY)})


async def n6_reasoning(state: AnalysisState) -> dict[str, Any]:
    emitter = NodeEmitter(NODE)
    evidence = [e for e in (state.get("evidence") or []) if e.get("chunk_id")]
    profile = state.get("profile") or {}
    clusters = state.get("clusters") or []

    # 池子按 chunk_id 建：证据的 id 从 n5 到 n7 一路都是它，不是数据库主键。
    pool = {str(e["chunk_id"]): e for e in evidence}
    psychology = {cid for cid, e in pool.items() if e.get("kind") == str(Library.PSYCHOLOGY)}
    allusions = {cid for cid, e in pool.items() if e.get("kind") in _ALLUSION_LIBRARIES}

    if not evidence:
        # 空池不是故障：知识库关掉了、或者确实一条都没匹配上。仍然调一次
        # 模型——没有证据也得给用户一句「为什么这次没有结论」，
        # 而那句话该由模型结合评论说，不是这里硬编码的。
        emitter.warn("no_evidence_for_reasoning", "没有检索到证据，本次推理只基于评论本身")

    try:
        from app.providers.factory import get_chat_model

        model = get_chat_model()
        payload = await call_json(
            model,
            task=prompt.TASK,
            messages=prompt.build_messages(),
            context=prompt.build_context(evidence, profile=profile, clusters=clusters),
            temperature=0.3,
            max_tokens=2000,
        )
        steps = _sanitize(
            payload.get("steps"),
            valid_ids=set(pool),
            psychology=psychology,
            allusions=allusions,
        )
        model_name = getattr(model, "name", "unknown")
        if not steps:
            raise ValueError("模型没有给出任何可用的推理步骤")
    except Exception as exc:  # noqa: BLE001 — 任何失败都走同一条降级路
        log.warning("n6_fallback", error=f"{type(exc).__name__}: {exc}")
        emitter.warn("reasoning_failed", "推理模型调用失败，已退回只列现象")
        steps = _fallback(clusters, pool=pool, psychology=psychology, allusions=allusions)
        model_name = "fallback"

    for step in steps:
        step["model"] = model_name
        emitter.partial("reasoning", {"reasoning": [step]})

    emitter.set_progress(1.0)
    return {"reasoning": steps}


# ----------------------------------------------------------------------


def _sanitize(
    raw: Any,
    *,
    valid_ids: set[str],
    psychology: set[str],
    allusions: set[str],
) -> list[dict[str, Any]]:
    """把模型给的步骤收进池子里。**这里出去的每一条都必然可追溯。**

    `valid_ids` 是集合而不是 `{id: 证据}` 那份映射：`filter_ids` 认映射时
    把值当「真 id」返回，把证据对象当 id 传出去，错误要漂到很远才炸。
    """
    if not isinstance(raw, list):
        return []

    steps: list[dict[str, Any]] = []
    dropped_evidence = 0
    dropped_allusions = 0

    for item in raw[:MAX_REASONING_STEPS]:
        if not isinstance(item, dict):
            continue
        phenomenon = _text(item.get("phenomenon"))
        if not phenomenon:
            # 没有现象就没有这一步——它是在描述一个不存在的东西。
            continue
        mechanism = _text(item.get("mechanism"))
        insight = _text(item.get("insight"))

        evidence_ids, bad_ev = _keep(item.get("evidence_ids"), valid_ids)
        # 典故只从文学/诗词里取。心理学语料写进 allusion_ids 是模型常见的
        # 串味（它把「引用的东西」都当典故），丢弃比纠正便宜，且不会错。
        allusion_ids, _ = _keep(item.get("allusion_ids"), allusions)
        dropped_evidence += len(bad_ev)
        dropped_allusions += len(_keep(item.get("allusion_ids"), valid_ids)[1])
        # 同一份材料不该同时充当机制与典故：证据优先，典故让位。
        allusion_ids = [cid for cid in allusion_ids if cid not in evidence_ids]

        steps.append(
            {
                "step_index": len(steps) + 1,
                "phenomenon": phenomenon,
                "mechanism": mechanism,
                "insight": insight,
                "evidence_ids": evidence_ids,
                "allusion_ids": allusion_ids,
                "confidence": _confidence(
                    evidence_ids=evidence_ids,
                    allusion_ids=allusion_ids,
                    mechanism=mechanism,
                    psychology=psychology,
                    allusions=allusions,
                ),
            }
        )

    if dropped_evidence or dropped_allusions:
        # 模型抄错 id 这件事必须留下痕迹：不然它只表现为「这次引用的证据
        # 变少了」，数字小了但没人知道为什么。
        log.info(
            "n6_dropped_citations",
            evidence=dropped_evidence,
            allusions=dropped_allusions,
        )
    return steps


def _fallback(
    clusters: list[dict[str, Any]],
    *,
    pool: dict[str, dict[str, Any]],
    psychology: set[str],
    allusions: set[str],
) -> list[dict[str, Any]]:
    """模型不可用时的降级：只列现象，不编机制也不编洞察。

    故意留空的字段会以「尚未解释」的样子显示在卡片上——这比不显示这一步
    诚实，也比用模板编一句话强。用户至少知道系统读到了哪些现象。
    """
    steps: list[dict[str, Any]] = []
    for cluster in clusters:
        if cluster.get("is_noise"):
            continue
        phenomenon = _text(cluster.get("summary")) or _text(cluster.get("label"))
        if not phenomenon:
            continue
        steps.append(
            {
                "step_index": len(steps) + 1,
                "phenomenon": phenomenon,
                "mechanism": "",
                "insight": "",
                "evidence_ids": [],
                "allusion_ids": [],
                "confidence": _confidence(
                    evidence_ids=[],
                    allusion_ids=[],
                    mechanism="",
                    psychology=psychology,
                    allusions=allusions,
                ),
            }
        )
        if len(steps) >= MAX_REASONING_STEPS:
            break
    return steps


def _confidence(
    *,
    evidence_ids: list[str],
    allusion_ids: list[str],
    mechanism: str,
    psychology: set[str],
    allusions: set[str],
) -> float:
    """由落库字段复算的可信度。公式与理由见模块 docstring。"""
    support = min(1.0, len(evidence_ids) / SUPPORT_FULL_MARKS)

    cited = bool(evidence_ids) and any(cid in psychology for cid in evidence_ids)
    if cited and mechanism.strip():
        mech = 1.0
    elif not psychology:
        mech = 0.5
    else:
        mech = 0.0

    if allusion_ids:
        allu = 1.0
    elif not allusions:
        allu = 0.5
    else:
        allu = 0.0

    w_support, w_mech, w_allu = CONFIDENCE_WEIGHTS
    # **留三位小数。** 权重与「半分」都是含 5 的：一位心理学证据的黄金路径
    # 恰好是 0.725，四舍五入到两位会变成 0.72（`round` 的平局规则受二进制
    # 表示影响，0.725 实际略小于 0.725），而这个数字是要显示给用户的，
    # 少掉的 0.005 会让「复算给用户看」这件事第一次演示就对不上。
    return round(w_support * support + w_mech * mech + w_allu * allu, 3)


def _keep(ids: Any, pool: dict[str, Any]) -> tuple[list[str], list[str]]:
    """保序去重地收 id，返回 `(kept, dropped)`。两份都要：见 `_sanitize`。"""
    return filter_ids(ids, pool)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


__all__ = ["NODE", "n6_reasoning"]
