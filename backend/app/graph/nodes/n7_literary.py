"""Node 7 — 文学化生成。

一次 `literary` 调用产出四段，然后 Python 侧做三件事：

  1. `sanitize` 每段正文：合法标记换成脚注记号，池外的连标记带痕迹剥掉
  2. 覆盖率不达标的段落**重生成一次**，两次取覆盖率高的那版
  3. 逐块 `emitter.delta(section, chunk)`，右栏逐字渲染

## 逐字效果为什么在本地产生

`json_mode` 下的流式增量不是合法 JSON 前缀——四段在一个对象里，收到一半
时既解析不出完整的键，也判断不了哪个字属于哪一段。真想流式就得放弃一次
调用产四段，而那会把四段之间的互引关系交给抽样（见 `prompts/literary`）。

所以切块在这里做：正文已经拿到并校验完毕，按标点切成 25ms 一块。
**这不是假装在流式**——它对用户产生的效果与真流式完全相同，而它顺带保证
了一件事：`applyEvent` 的 `delta` 分支是正文进入右栏的**唯一通路**，
不存在「流式渲染一套、快照渲染另一套」的分叉。`content_md` 与逐字拼出来
的结果由构造保证一致（`_chunk_text` 只切不删），测试直接断言这一点。

## 重试判据为什么不是「所有段落」

`REGEN_COVERAGE` 只管**该有引用的段落**（机制 / 类比 / 洞察）。侧写段描述
的是聚类结果本身，不是检索来的知识，它的契约里根本没有引用标记——要求
它达标等于每次运行都白跑一次必然失败的重试，并给一个健康的段落盖上
`retried` 的章。这类「把健康结果报成有问题」的代价比少一次重试大得多，
它会让真正需要重试的那一次淹在噪声里。

## 覆盖率是重试的唯一依据

它按「正文里任意一枚标记能落到证据上的概率」定义（见 `validators.coverage`），
不是「有几段引用了证据」。段落覆盖率与唯一 id 覆盖率作为诊断指标写进
`content_json` 供展开查看，**不驱动重试**——同一条数据上它们是 0.32 与
0.46，都低于阈值，会让每次演示都触发一次注定失败的重试。
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.constants import DELTA_INTERVAL_MS, REGEN_COVERAGE, SectionKey
from app.config import get_settings
from app.graph.emitting import NodeEmitter
from app.graph.nodes._llm import call_json
from app.graph.state import AnalysisState
from app.graph.validators import coverage, sanitize
from app.logging_conf import get_logger
from app.prompts import literary as prompt
from app.text_chunks import chunk_text

log = get_logger(__name__)

NODE = "n7_literary"

#: 需要引用的段落。见模块 docstring：侧写段不在其中。
_CITED_SECTIONS = frozenset(
    {SectionKey.MECHANISM, SectionKey.ALLUSION, SectionKey.INSIGHT}
)

#: 重试时给模型的话。**要指名道姓地说哪里不对**：只说「重写一遍」的话，
#: 模型会写出一段语义相近、引用同样稀薄的文字，这次重试就白花了。
_REGEN_INSTRUCTION = (
    "上一版正文里的引用标记太少，或者标记里的 id 没能落到给定的证据上。"
    "请重写这一段：凡是写了机制或典故的句子，后面紧跟一枚 [[ev:<id>]]，"
    "id 逐字照抄上下文里给出的那些。"
)


async def n7_literary(state: AnalysisState) -> dict[str, Any]:
    emitter = NodeEmitter(NODE)

    evidence = [e for e in (state.get("evidence") or []) if e.get("chunk_id")]
    pool = {str(e["chunk_id"]): e for e in evidence}
    profile = state.get("profile") or {}
    clusters = state.get("clusters") or []
    video = state.get("video") or {}

    drafts, model_name = await _draft(
        evidence, profile=profile, clusters=clusters, video=video, emitter=emitter
    )

    sections: dict[str, dict[str, Any]] = {}
    for key in SectionKey:
        draft = drafts.get(key.value)
        body = await _finalize(
            key,
            # 只认字符串。模型偶尔会回一个列表或数字，`str()` 硬转会把
            # `['不是字符串']` 渲染进右栏给用户看——空段至少是诚实的。
            draft if isinstance(draft, str) else "",
            pool=pool,
            evidence=evidence,
            profile=profile,
            clusters=clusters,
            video=video,
            emitter=emitter,
        )
        body["model"] = model_name
        sections[key.value] = body
        # 先发元信息（标题、引用、覆盖率），再发正文。**正文留空**，
        # 由随后的 `delta` 一块块填进去。
        #
        # 顺序不能反：`applyEvent` 的 `delta` 分支遇到不存在的段落会现场造一个
        # 标题为空的壳，之后再来的元信息把 `content_md` 一起覆盖掉——用户会
        # 看到正文先打完、再整个消失。
        #
        # 元信息先到还顺带解决了另一件事：引用 chip 在逐字打出正文的时候就
        # 已经可点、可 hover 了。等结尾的快照才补上 citations 的话，
        # 正文里会先闪过一串没有下文的 `[^1]`。
        emitter.partial("sections", {"sections": [{**body, "content_md": ""}]})
        await _emit_deltas(emitter, key, body["content_md"])

    emitter.set_progress(1.0)
    return {"sections": sections}


# ----------------------------------------------------------------------


async def _draft(
    evidence: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
    clusters: list[dict[str, Any]],
    video: dict[str, Any],
    emitter: NodeEmitter,
) -> tuple[dict[str, Any], str]:
    """整篇生成。失败时返回空草稿——**不编替代文案**。

    四段都空着，右栏如实显示「本段未生成」，同时一条 warning 已经发出。
    用模板句填空的话，用户读到的是系统写的话，却以为是模型读了评论写的。
    """
    try:
        from app.providers.factory import get_chat_model

        model = get_chat_model()
        payload = await call_json(
            model,
            task=prompt.TASK,
            messages=prompt.build_messages(),
            context=prompt.build_context(
                evidence, profile=profile, clusters=clusters, video=video
            ),
            temperature=0.5,
            max_tokens=prompt.MAX_TOKENS,
        )
        return payload, getattr(model, "name", "unknown")
    except Exception as exc:  # noqa: BLE001 — 任何失败都走同一条降级路
        log.warning("n7_draft_failed", error=f"{type(exc).__name__}: {exc}")
        emitter.warn("literary_failed", "洞察生成失败，右栏本次没有产出")
        return {}, "fallback"


async def _finalize(
    key: SectionKey,
    raw: str,
    *,
    pool: dict[str, dict[str, Any]],
    evidence: list[dict[str, Any]],
    profile: dict[str, Any],
    clusters: list[dict[str, Any]],
    video: dict[str, Any],
    emitter: NodeEmitter,
) -> dict[str, Any]:
    """校验、按需重试、组装落库契约。"""
    clean, citations, report = _sanitize_with_sources(raw, pool)
    best = {
        "content_md": clean,
        "citations": citations,
        "citation_coverage": coverage(report),
        "report": report,
        "retried": False,
    }

    if _should_retry(key, best["citation_coverage"], pool, raw):
        alt_raw = await _regen(
            key, evidence=evidence, profile=profile, clusters=clusters, video=video
        )
        alt_clean, alt_cit, alt_report = _sanitize_with_sources(alt_raw, pool)
        alt_cov = coverage(alt_report)
        # 取覆盖率更高的那一版。相等时保留原版——重试没带来改善，
        # 就不该同时换掉正文（用户读到的东西变了而分数没变，最难解释）。
        if alt_cov > best["citation_coverage"]:
            best.update(
                {
                    "content_md": alt_clean,
                    "citations": alt_cit,
                    "citation_coverage": alt_cov,
                    "report": alt_report,
                }
            )
        best["retried"] = True
        best["regen_coverage"] = alt_cov

    return {
        "key": key.value,
        "title": key.title,
        "content_md": best["content_md"],
        "citations": best["citations"],
        "citation_coverage": best["citation_coverage"],
        "content_json": _diagnostics(best, key),
    }


def _should_retry(key: SectionKey, cov: float, pool: dict[str, Any], raw: str) -> bool:
    if key not in _CITED_SECTIONS:
        return False
    if cov >= REGEN_COVERAGE:
        return False
    if not pool:
        # 池子本来就是空的：无事可引，重试只会拿到同样一段没有标记的正文。
        return False
    return bool(raw.strip())


async def _regen(
    key: SectionKey,
    *,
    evidence: list[dict[str, Any]],
    profile: dict[str, Any],
    clusters: list[dict[str, Any]],
    video: dict[str, Any],
) -> str:
    """重生成单段。失败返回空串——空串会输给原版，于是原版被保留。"""
    try:
        from app.providers.factory import get_chat_model

        model = get_chat_model()
        payload = await call_json(
            model,
            task=prompt.REGEN_TASK,
            messages=prompt.build_regen_messages(key),
            context=prompt.build_regen_context(
                evidence,
                key=key,
                instruction=_REGEN_INSTRUCTION,
                profile=profile,
                clusters=clusters,
                video=video,
            ),
            temperature=0.5,
            max_tokens=prompt.MAX_TOKENS,
        )
    except Exception as exc:  # noqa: BLE001 — 重试失败不该让整段消失
        log.warning("n7_regen_failed", section=key.value, error=f"{type(exc).__name__}: {exc}")
        return ""

    # 两种形状都认：整段重生成要的是 `{key: "…"}`，而 mock 与
    # 「只重写这一段」的提示词都可能回 `{"content_md": "…"}`。
    value = payload.get(key.value)
    if not isinstance(value, str):
        value = payload.get("content_md")
    return value if isinstance(value, str) else ""


def _sanitize_with_sources(
    raw: str, pool: dict[str, dict[str, Any]]
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """`sanitize` + 给每枚引用补上出处。

    `sanitize` 不认识证据对象，只认 id——它要能被 n6 那样没有证据正文的
    调用方复用。出处（`《标题》· 作者`）在这里补，因为 chip 的 hover
    要显示的是人读的东西，而 `chunk_id` 不是。

    所以传进去的是 `set(pool)` 而不是 `pool` 本身：`sanitize` 认映射时
    把**值**当成 id 返回，而这里的值是证据 dict（`validators._resolve`
    会直接抛 TypeError 拦住这种传法）。
    """
    clean, citations, report = sanitize(raw, set(pool))
    for cite in citations:
        item = pool.get(str(cite["evidence_id"])) or {}
        title = item.get("title") or cite["evidence_id"]
        author = item.get("author")
        cite["text"] = f"《{title}》· {author}" if author else f"《{title}》"
        cite["library"] = item.get("library") or item.get("kind") or ""
    return clean, citations, report


def _diagnostics(best: dict[str, Any], key: SectionKey) -> dict[str, Any]:
    """写进 `content_json` 的诊断数据。**不驱动任何行为**，只供展开查看。

    段落覆盖率与唯一 id 覆盖率都在这里——它们是「这段有没有引用」之外
    另外两种口径的读数，口径不同结论会差很远（见模块 docstring），
    所以把原始计数一并留下，谁想复核都能自己算。
    """
    report = best["report"]
    out: dict[str, Any] = {
        "section": key.value,
        "total": int(report.get("total") or 0),
        "valid": int(report.get("valid") or 0),
        "invalid": int(report.get("invalid") or 0),
        "dropped_ids": list(report.get("dropped_ids") or []),
        "coverage": best["citation_coverage"],
        "retried": bool(best["retried"]),
    }
    if "regen_coverage" in best:
        out["regen_coverage"] = best["regen_coverage"]
    return out


# ----------------------------------------------------------------------


async def _emit_deltas(emitter: NodeEmitter, key: SectionKey, text: str) -> None:
    """逐块发正文。`chunk_text` 只切不删，拼回去必须逐字等于 `text`。"""
    chunks = chunk_text(text)
    if not chunks:
        return
    delay = DELTA_INTERVAL_MS / 1000 if get_settings().is_demo else 0.0
    for i, chunk in enumerate(chunks):
        emitter.delta(key.value, chunk)
        if delay and i < len(chunks) - 1:
            await asyncio.sleep(delay)


__all__ = ["NODE", "n7_literary"]
