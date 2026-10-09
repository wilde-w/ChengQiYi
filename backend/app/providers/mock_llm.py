"""规则驱动的 mock LLM。

定位：**演示资产，不是桩**。

一个只返回 "lorem ipsum" 的 mock 无法证明流水线是通的——因为没有任何下游
逻辑会对它产生反应。所以这里做的是：用中文情绪词表对真实评论打分，
产出真的**由本次输入决定**的情绪/主题/需求标签与结构化结论。

代价是它没有语义理解能力（纯词表匹配）。这一点会在所有输出和 UI 徽章上
明确标注为「演示模式」，绝不让 mock 结果被误认为真实分析。

（确定性：同样的 context 必然产出同样的结果，便于写 golden 测试。）
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import AsyncIterator, Sequence
from typing import Any, Callable

from app.providers.base import ChatMessage, ChatResult, ChatUsage, ToolCall

# ----------------------------------------------------------------------
# 情绪 / 主题 / 需求词表
#
# 每条情绪给一组触发词。打分 = 命中词数加权（长词权重更高，因为它更具体）。
# 词表覆盖产品的主要场景（丧失、倦怠、怀旧、孤独、不甘、温暖），
# 足以让三个 demo fixture 产出彼此不同的结论——这正是「无硬编码」的证据。
# ----------------------------------------------------------------------

EMOTION_LEXICON: dict[str, tuple[str, ...]] = {
    "哀伤": ("走", "去世", "没了", "不在了", "想念", "想她", "想他", "哭", "眼泪", "墓地", "遗物", "生离死别", "离开"),
    "愧疚": ("后悔", "没来得及", "对不起", "应该", "最后一次", "没能", "没好好", "亏欠", "要是", "如果当时"),
    "共鸣": ("我也是", "一样", "懂你", "抱抱", "看到这条", "破防", "感同身受", "说不出的", "泪目"),
    "自嘲": ("哈哈", "笑死", "社畜", "打工人", "牛马", "废物", "摆烂", "躺平", "算了"),
    "倦怠": ("累", "加班", "熬夜", "上班", "辞职", "干不动", "透支", "内耗", "疲惫", "撑不住", "通勤"),
    "不甘": ("凭什么", "不公平", "咽不下", "不服", "明明", "努力了", "还是没", "到底为什么"),
    "孤独": ("一个人", "没人", "孤独", "空荡", "睡不着", "半夜", "没人懂", "沉默"),
    "温暖": ("谢谢", "温柔", "善良", "好好生活", "希望你好", "被治愈", "感动", "祝福"),
    "怀念": ("小时候", "以前", "当年", "青春", "旧照", "回不去", "那时候", "记得"),
    "焦虑": ("怎么办", "压力", "害怕", "担心", "未来", "迷茫", "慌", "来不及"),
    "愤怒": ("过分", "恶心", "气死", "太过分", "无语", "离谱", "什么玩意"),
}

TOPIC_LEXICON: dict[str, tuple[str, ...]] = {
    "亲子关系": ("妈", "母亲", "爸", "父亲", "父母", "孩子", "女儿", "儿子", "家人", "回家"),
    "职场压力": ("上班", "加班", "领导", "同事", "公司", "裁员", "绩效", "打工", "上班族", "工作"),
    "亲密关系": ("分手", "恋爱", "男朋友", "女朋友", "结婚", "离婚", "另一半", "异地"),
    "青春怀旧": ("高中", "大学", "同学", "毕业", "初恋", "校园", "青春", "小时候"),
    "自我认同": ("我是谁", "价值", "意义", "自卑", "配不上", "够不够好", "接受自己"),
    "生死与失去": ("去世", "离开", "遗物", "葬礼", "最后一面", "走了", "墓地"),
    "城市生活": ("出租屋", "外卖", "地铁", "合租", "北漂", "沪漂", "漂泊", "城市"),
}

# 需求是「这段情绪背后在要什么」，由情绪映射而来，不直接从词表打分
NEED_BY_EMOTION: dict[str, str] = {
    "哀伤": "被看见的丧失",
    "愧疚": "和解与弥补",
    "共鸣": "确认自己不孤单",
    "自嘲": "有尊严地喘息",
    "倦怠": "停下来休息的权利",
    "不甘": "努力被承认",
    "孤独": "真实的连接",
    "温暖": "把善意传下去",
    "怀念": "留住正在消失的东西",
    "焦虑": "对未来的掌控感",
    "愤怒": "被公平对待",
}

# 核心张力：由「主导情绪 + 次主导情绪」组合出来的句式
TENSION_TEMPLATES: dict[tuple[str, str], str] = {
    ("哀伤", "愧疚"): "想念却再也无法说出口",
    ("哀伤", "温暖"): "失去之后仍想好好活下去",
    ("愧疚", "哀伤"): "想说的太迟，想留的已走",
    ("共鸣", "哀伤"): "各自的悲伤，在同一条评论区相遇",
    ("倦怠", "自嘲"): "明知道该停下，却不敢停",
    ("倦怠", "焦虑"): "身体想逃，理智不敢走",
    ("不甘", "倦怠"): "还在乎结果，却已经没力气",
    ("怀念", "哀伤"): "想回到过去，但过去已经不认我",
    ("孤独", "共鸣"): "一个人，却在这里发现大家也一样",
    ("自嘲", "温暖"): "用玩笑包住自己，又盼着有人听懂",
}


def _score(text: str, lexicon: dict[str, tuple[str, ...]]) -> list[tuple[str, float]]:
    """按命中词打分，长词权重更高（更具体 → 信息量更大）。"""
    scores: dict[str, float] = {}
    for label, words in lexicon.items():
        total = 0.0
        for w in words:
            if w in text:
                total += 1.0 + len(w) * 0.35
        if total > 0:
            scores[label] = total
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def _top_labels(pairs: list[tuple[str, float]], n: int) -> list[str]:
    return [label for label, _ in pairs[:n]]


def _confidence(seed: str, low: float, high: float) -> float:
    """由内容派生的伪随机置信度——确定性，但看起来不呆板。"""
    h = int.from_bytes(hashlib.blake2b(seed.encode("utf-8"), digest_size=4).digest(), "big")
    return round(low + (high - low) * (h / 0xFFFFFFFF), 2)


class MockChatModel:
    """实现 ChatModel 协议。按 task 分派到对应的规则处理器。

    两条路径由 `complete(tools=...)` 区分，**判别式只有这一处**：
    流水线不传（`None`，走 task 分派），故事 agent 传（`[]` 表示收尾轮，
    非空表示这一轮允许调工具）。见 `_agent_reply`。
    """

    is_mock = True
    name = "mock-rule-based"

    def __init__(self, latency_ms: int = 900, agent_script: str = "default") -> None:
        self._latency = max(0, latency_ms) / 1000.0
        self._agent_script = agent_script
        self._handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "cluster_label": self._cluster_label,
            "extract_psych": self._extract_psych,
            "synthesize": self._synthesize,
            "literary": self._literary,
            "section_regen": self._section_regen,
            "tag_chunks": self._tag_chunks,
        }

    # -- 协议实现 ---------------------------------------------------------
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        task: str | None = None,
        context: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> ChatResult:
        if self._latency:
            await asyncio.sleep(self._latency)

        if tools is not None:
            return self._agent_round(messages, tools)

        payload = self._dispatch(task, context or {})
        text = (
            json.dumps(payload, ensure_ascii=False, indent=2)
            if json_mode
            else _payload_to_prose(payload)
        )
        return ChatResult(
            text=text,
            model=self.name,
            usage=ChatUsage(0, 0, 0),
            finish_reason="stop",
        )

    def _agent_round(
        self, messages: Sequence[ChatMessage], tools: Sequence[dict[str, Any]]
    ) -> ChatResult:
        round_no = _turn_round(messages)
        reply = _agent_reply(messages, tools, script=self._agent_script, round_no=round_no)
        calls = [
            ToolCall(id=f"mock_{round_no}_{i}", name=name, arguments=args)
            for i, (name, args) in enumerate(reply["tool_calls"])
        ]
        return ChatResult(
            text=reply["content"],
            model=self.name,
            usage=ChatUsage(0, 0, 0),
            finish_reason="tool_calls" if calls else "stop",
            tool_calls=calls,
        )

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        task: str | None = None,
        context: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        payload = self._dispatch(task, context or {})
        text = _payload_to_prose(payload)
        # 按标点切块流式吐出，模拟真实模型的 token 节奏
        for chunk in _chunk_text(text):
            if self._latency:
                await asyncio.sleep(min(self._latency / 8, 0.06))
            yield chunk

    # -- 分派 -------------------------------------------------------------
    def _dispatch(self, task: str | None, context: dict[str, Any]) -> dict[str, Any]:
        handler = self._handlers.get(task or "")
        if handler is None:
            return {"note": f"mock 未实现 task={task}，返回占位内容", "task": task}
        return handler(context)

    # -- 各 task 的规则实现 ------------------------------------------------
    def _cluster_label(self, ctx: dict[str, Any]) -> dict[str, Any]:
        comments = _as_str_list(ctx.get("comments"))
        text = "\n".join(comments)
        emotions = _score(text, EMOTION_LEXICON)
        topics = _score(text, TOPIC_LEXICON)

        top_emotions = _top_labels(emotions, 2)
        top_topics = _top_labels(topics, 2)
        needs = [NEED_BY_EMOTION[e] for e in top_emotions if e in NEED_BY_EMOTION][:2]

        primary = top_emotions[0] if top_emotions else "复杂情绪"
        topic = top_topics[0] if top_topics else "日常经验"
        size = int(ctx.get("size") or len(comments) or 0)

        label = _CLUSTER_LABELS.get(primary, {}).get(topic) or f"{primary}·{topic}"
        summary = _CLUSTER_SUMMARY.get(
            primary, "这一簇的声音围绕{0}展开，情绪基调是{1}。"
        ).format(topic, primary)

        return {
            "label": label,
            "summary": summary,
            "emotion": top_emotions or [primary],
            "topic": top_topics or [topic],
            "need": needs,
            "size": size,
        }

    def _extract_psych(self, ctx: dict[str, Any]) -> dict[str, Any]:
        clusters = ctx.get("clusters") or []
        cluster_tags: dict[str, dict[str, list[str]]] = {}
        all_emotions: list[str] = []
        all_topics: list[str] = []
        all_needs: list[str] = []

        for c in clusters:
            key = str(c.get("cluster_key", ""))
            comments = _as_str_list(c.get("comments"))
            text = "\n".join(comments)
            emotions = _top_labels(_score(text, EMOTION_LEXICON), 3)
            topics = _top_labels(_score(text, TOPIC_LEXICON), 2)
            needs = [NEED_BY_EMOTION[e] for e in emotions if e in NEED_BY_EMOTION][:2]
            if not emotions:
                emotions = ["平静"]
            if not topics:
                topics = ["日常经验"]
            cluster_tags[key] = {"emotion": emotions, "topic": topics, "need": needs}
            all_emotions.extend(emotions)
            all_topics.extend(topics)
            all_needs.extend(needs)

        ordered_emotions = _rank(all_emotions)
        ordered_topics = _rank(all_topics)
        ordered_needs = _rank(all_needs)

        core_tension = _core_tension(ordered_emotions)
        summary = _profile_summary(ordered_emotions, ordered_topics, ordered_needs, len(clusters))

        return {
            "cluster_tags": cluster_tags,
            "global_tags": {
                "emotion": ordered_emotions,
                "topic": ordered_topics,
                "need": ordered_needs,
            },
            "core_tension": core_tension,
            "summary": summary,
        }

    def _synthesize(self, ctx: dict[str, Any]) -> dict[str, Any]:
        evidence = ctx.get("evidence") or []
        profile = ctx.get("profile") or {}
        clusters = ctx.get("clusters") or []

        by_kind: dict[str, list[dict[str, Any]]] = {}
        for ev in evidence:
            by_kind.setdefault(str(ev.get("kind", "other")), []).append(ev)

        psych = by_kind.get("psychology", [])
        lit = by_kind.get("literature", []) + by_kind.get("poetry", [])

        emotions = (profile.get("global_tags") or {}).get("emotion") or []
        tension = profile.get("core_tension") or "未命名的情绪张力"

        steps: list[dict[str, Any]] = []
        # 每步 = 一个显著簇的现象 + 一条心理学机制 + 一条文学典故
        for i, cluster in enumerate(clusters[:4]):
            if i >= max(len(psych), 1) and i > 0 and not psych:
                break
            phenomenon = str(cluster.get("summary") or cluster.get("label") or "")
            ev_ids = [str(psych[i]["id"])] if i < len(psych) and psych[i].get("id") else []
            allu_ids = [str(lit[i]["id"])] if i < len(lit) and lit[i].get("id") else []

            mechanism = ""
            if ev_ids:
                ev = psych[i]
                mechanism = (
                    f"{ev.get('title') or '相关机制'}（{ev.get('source') or '出处见证据卡'}）。"
                    f"{(ev.get('text') or '')[:110]}"
                )
            insight = _STEP_INSIGHT_TEMPLATES[i % len(_STEP_INSIGHT_TEMPLATES)].format(
                emotion=emotions[i] if i < len(emotions) else "这种情绪",
                tension=tension,
            )

            steps.append(
                {
                    "phenomenon": phenomenon,
                    "mechanism": mechanism or "这一现象尚无已检索到的机制解释。",
                    "insight": insight,
                    "evidence_ids": ev_ids,
                    "allusion_ids": allu_ids,
                    "confidence": _confidence(f"{phenomenon}{mechanism}", 0.6, 0.88),
                }
            )

        if not steps:
            steps.append(
                {
                    "phenomenon": "评论区呈现出一种尚未被命名的共同情绪。",
                    "mechanism": "",
                    "insight": "现有证据不足以形成稳定解释，建议放宽评论采样或开启更多知识库。",
                    "evidence_ids": [],
                    "allusion_ids": [],
                    "confidence": 0.4,
                }
            )

        return {"steps": steps}

    def _literary(self, ctx: dict[str, Any]) -> dict[str, Any]:
        return _compose_sections(ctx, instruction=None)

    def _section_regen(self, ctx: dict[str, Any]) -> dict[str, Any]:
        key = str(ctx.get("section_key") or "insight")
        instruction = ctx.get("instruction")
        composed = _compose_sections(ctx, instruction=instruction)
        return {"content_md": composed.get(key, ""), "section_key": key}

    def _tag_chunks(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """知识库导入的打标。

        **刻意复用降级路径的同一份实现**（`app.kb.lexical_tags`）。如果 mock
        另写一套打标逻辑，它就只能证明 mock 自己是通的；复用之后，每次离线
        演示都在把校验器、对齐检查和闭集词表真跑一遍——这正是 mock 作为
        演示资产而非桩的意义所在。

        频次表按整批的正文现算：单批约 3200 字，足够让「此在」这种反复出现
        的术语浮上来，而不必把整本书的频次表塞进 context。
        """
        from app.constants import Library
        from app.kb.lexical_tags import build_frequency_table, lexical_tags

        library = Library(str(ctx.get("library") or "psychology"))
        raw_items = [it for it in (ctx.get("items") or []) if isinstance(it, dict)]
        freq = build_frequency_table("\n".join(str(it.get("text") or "") for it in raw_items))

        items: list[dict[str, Any]] = []
        for item in raw_items:
            text = str(item.get("text") or "")
            tags = lexical_tags(
                text,
                library=library,
                work=str(ctx.get("work") or ""),
                discipline_hint=str(ctx.get("discipline") or ""),
                freq=freq,
            )
            items.append(
                {
                    "i": item.get("i"),
                    "quote": tags.quote,
                    "concept": tags.concept,
                    "keywords": tags.keywords,
                    "imagery": tags.imagery,
                    "emotion": tags.emotion,
                    "type": tags.type,
                }
            )
        return {"items": items}


# ----------------------------------------------------------------------
# 文本组装
# ----------------------------------------------------------------------

_CLUSTER_LABELS: dict[str, dict[str, str]] = {
    "哀伤": {
        "生死与失去": "失去之后·持续的联结",
        "亲子关系": "与父母有关的、说不出口的想念",
        "城市生活": "在异乡独自消化的丧失",
    },
    "愧疚": {
        "生死与失去": "来不及说的话",
        "亲子关系": "关于父母的亏欠感",
        "职场压力": "为工作让渡掉的陪伴",
    },
    "共鸣": {"生死与失去": "原来大家都一样", "职场压力": "打工人之间的相互确认"},
    "自嘲": {"职场压力": "用自嘲换来的一点体面", "城市生活": "勉强站住的年轻人"},
    "倦怠": {"职场压力": "被透支之后的疲惫", "城市生活": "撑不住的日常"},
    "不甘": {"职场压力": "努力过却仍未被承认"},
    "孤独": {"城市生活": "一个人的深夜", "亲密关系": "在关系里也感到孤单"},
    "温暖": {"亲子关系": "被善意托住的人"},
    "怀念": {"青春怀旧": "回不去的那几年", "亲子关系": "记忆里的旧时光"},
    "焦虑": {"职场压力": "对未来的持续担心"},
    "愤怒": {"职场压力": "对不被公平对待的愤怒"},
}

_CLUSTER_SUMMARY: dict[str, str] = {
    "哀伤": "这一簇讲述具体的失去：某个人不在了，某个号码还在通讯录里。{0}的细节反复出现。",
    "愧疚": "这一簇的核心是回望——反复检索自己「本可以做得更好」的那个瞬间。",
    "共鸣": "这一簇不是叙事而是应答：用自己的经历回应视频里的情绪，形成密集的情绪确认。",
    "自嘲": "这一簇用玩笑包裹处境。自嘲是防御，也是唯一被允许的表达方式。",
    "倦怠": "这一簇描述持续的消耗状态：睡不够、停不下来、也看不到尽头。",
    "不甘": "这一簇在追问公平——付出与回报之间的落差没有被解释。",
    "孤独": "这一簇描述无人回应的处境，且多发生在夜里。",
    "温暖": "这一簇是善意的传递，试图为前面的情绪提供一个出口。",
    "怀念": "这一簇把当下与过去对折——通过回忆来重新定义此刻的自己。",
    "焦虑": "这一簇指向尚未发生的事，担心多于事实。",
    "愤怒": "这一簇表达对不公的即时反应，情绪强度高但叙事简短。",
}

_STEP_INSIGHT_TEMPLATES = (
    "当{emotion}被反复书写，它就不再是私人经验，而成为一种可以被共同承认的处境——{tension}。",
    "这里的{emotion}不指向解决，而指向被承认：先说出口，才谈得上安放。",
    "重复的叙述本身构成了仪式，把{tension}从一个人的事变成一群人的事。",
    "{emotion}在此处既是症状也是黏合剂，让分散的个体在评论区里短暂地成为共同体。",
)


def _rank(items: list[str]) -> list[str]:
    """按出现频次排序，保留首次出现的顺序作为次序（稳定输出，便于测试）。"""
    counts: dict[str, int] = {}
    order: dict[str, int] = {}
    for i, item in enumerate(items):
        counts[item] = counts.get(item, 0) + 1
        order.setdefault(item, i)
    return sorted(counts, key=lambda k: (-counts[k], order[k]))


def _core_tension(emotions: list[str]) -> str:
    if not emotions:
        return "尚未命名"
    if len(emotions) == 1:
        return f"单一的{emotions[0]}在扩散"
    for i, a in enumerate(emotions):
        for b in emotions[i + 1 :]:
            hit = TENSION_TEMPLATES.get((a, b)) or TENSION_TEMPLATES.get((b, a))
            if hit:
                return hit
    return f"{emotions[0]}与{emotions[1]}同时在场"


def _profile_summary(
    emotions: list[str], topics: list[str], needs: list[str], cluster_count: int
) -> str:
    if not emotions:
        return "未能从现有评论中提取稳定的情绪结构。"
    e = "、".join(emotions[:3])
    t = "、".join(topics[:2]) or "日常经验"
    n = "、".join(needs[:2]) or "被理解"
    return (
        f"在 {cluster_count} 个语义簇中，情绪以{e}为主，话题集中在{t}。"
        f"表层诉求指向{n}——但真正被反复书写的是{_core_tension(emotions)}。"
    )


def _compose_sections(ctx: dict[str, Any], *, instruction: str | None) -> dict[str, str]:
    """组装右栏四段。

    引用以 `[[ev:<id>]]` 标记内联写入，由 graph/validators.py 抽取、
    校验、剥离无效引用并计算覆盖率。真实模型走同一套契约。
    """
    profile = ctx.get("profile") or {}
    evidence = ctx.get("evidence") or []
    clusters = ctx.get("clusters") or []
    video = ctx.get("video") or {}

    global_tags = profile.get("global_tags") or {}
    emotions: list[str] = global_tags.get("emotion") or []
    topics: list[str] = global_tags.get("topic") or []
    needs: list[str] = global_tags.get("need") or []
    tension = profile.get("core_tension") or "尚未命名的张力"

    psych = [e for e in evidence if e.get("kind") == "psychology"]
    lit = [e for e in evidence if e.get("kind") in ("literature", "poetry")]

    cluster_lines = []
    for c in clusters[:5]:
        label = c.get("label") or ""
        size = c.get("size") or 0
        emo = "、".join((c.get("emotion_tags") or [])[:2])
        if label:
            cluster_lines.append(f"- **{label}**（{size} 条）——情绪基调：{emo or '未标注'}")

    cite = lambda evs, n: " ".join(f"[[ev:{e.get('id')}]]" for e in evs[:n] if e.get("id"))

    # ---- 心理侧写 ----
    profile_md = "\n".join(
        [
            f"评论区在 {len(clusters)} 个语义簇上聚集。",
            "",
            *cluster_lines,
            "",
            f"主导情绪为 **{'、'.join(emotions[:3]) or '未识别'}**，"
            f"话题集中在 **{'、'.join(topics[:2]) or '日常经验'}**。"
            f"表层诉求是「{'、'.join(needs[:2]) or '被理解'}」。",
            "",
            f"但把所有簇放在一起看，真正反复出现的是**{tension}**。"
            "这不是一个可以被建议解决的问题，而是一种需要被承认的处境。",
        ]
    )

    # ---- 科学机制 ----
    mech_bits = [
        f"从心理学角度看，「{tension}」并非例外状态，它有可描述的机制。"
    ]
    for ev in psych[:3]:
        mech_bits.append(
            f"\n**{ev.get('title') or '相关机制'}**（{ev.get('source') or ''}）"
            f"{cite([ev], 1)}\n\n{_trim(ev.get('text') or '', 180)}"
        )
    if not psych:
        mech_bits.append("\n本次分析未检索到匹配的心理学证据——可尝试开启心理学库或放宽评论采样。")
    mechanism_md = "\n".join(mech_bits)

    # ---- 文学类比 ----
    allu_bits = ["这种心境在古典文本里有更早、也更凝练的写法。"]
    for ev in lit[:3]:
        # 作者与出处都可能缺（语料里红楼梦这类佚名条目就没有作者），
        # 用 `·` 拼接前先滤掉空的——否则会渲染成「（·红楼梦）」。
        byline = "·".join(p for p in (ev.get("author"), ev.get("source")) if p)
        allu_bits.append(
            f"\n**{ev.get('title') or ''}**"
            f"（{byline}）"
            f"{cite([ev], 1)}\n\n> {_trim(ev.get('text') or '', 150).replace(chr(10), ' ')}"
        )
    if not lit:
        allu_bits.append("\n本次未启用文学库，或未检索到匹配典故。")
    allusion_md = "\n".join(allu_bits)

    # ---- 最终洞察 ----
    title = _trim(video.get("title") or video.get("caption") or "这条视频", 40)
    insight_md = "\n".join(
        [
            f"《{title}》的评论区，本质上是一次集体书写。",
            "",
            f"人们不是来讨论视频的，是来认领一个位置——"
            f"{'、'.join(emotions[:2]) or '某种情绪'}在这里被反复确认。"
            f"{cite(psych[:2] + lit[:2], 3)}",
            "",
            f"值得注意的是，这类评论几乎不寻求解决方案。"
            f"它们更像一种仪式：先把「{tension}」说出来，"
            f"让它从一个人的私事变成一群人的共同处境。"
            f"古典文学早就懂得这件事——"
            f"{_allusion_echo(lit)}",
            "",
            "所以这条视频真正的功能，不是内容本身，"
            "而是提供了一个让分散情绪得以汇合的场所。"
            "理解了这一点，也就理解了它为什么会被反复转发。",
        ]
    )

    if instruction:
        profile_md = f"（按「{instruction}」重新生成）\n\n{profile_md}"
        insight_md = f"（按「{instruction}」重新生成）\n\n{insight_md}"

    return {
        "profile": profile_md,
        "mechanism": mechanism_md,
        "allusion": allusion_md,
        "insight": insight_md,
    }


def _allusion_echo(lit: list[dict[str, Any]]) -> str:
    if not lit:
        return "只是我们这一代把悼亡写在了评论区。"
    first = lit[0]
    return (
        f"{first.get('author') or ''}写「{_trim(first.get('title') or '', 20)}」时，"
        f"处理的正是同一件事——把失去转写成可以留存的文字。"
    )


def _trim(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "…"


def _as_str_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            out.append(str(item.get("text") or item.get("comment") or ""))
        else:
            out.append(str(item))
    return [s for s in out if s]


def _payload_to_prose(payload: dict[str, Any]) -> str:
    """非 JSON 模式下把结构化结果摊平成文本（用于流式段落生成）。"""
    if "content_md" in payload:
        return str(payload["content_md"])
    if "steps" in payload:
        return "\n\n".join(
            f"{i + 1}. 现象：{s.get('phenomenon', '')}\n   机制：{s.get('mechanism', '')}\n   洞察：{s.get('insight', '')}"
            for i, s in enumerate(payload["steps"])
        )
    if "label" in payload:
        return f"{payload.get('label')}：{payload.get('summary', '')}"
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _chunk_text(text: str, size: int = 12) -> list[str]:
    """按标点/长度切块。标点优先，让流式断点落在自然位置。"""
    chunks: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if len(buf) >= size and (ch in "，。！？；：\n、" or len(buf) >= size * 2):
            chunks.append(buf)
            buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def sample_confidence(seed: str) -> float:
    """供测试复用的确定性取值。"""
    return _confidence(seed, 0.0, 1.0)


# ----------------------------------------------------------------------
# 故事 agent 的确定性假循环（MockChatModel._agent_round 调用）
#
# 硬要求：**调用参数从输入派生**。换一段评论，query 与 key 就跟着变——
# 演示时面板上的工具卡片才算证据；写死一段 JSON 会让它看起来像个摆设。
#
# 默认三轮：查知识库 → 查古典文学 → 交稿。修改轮只查一次再交稿，
# 好让「再暗一点」也在面板上留下一组新的工具卡片。
# ----------------------------------------------------------------------

#: 场景句按话题出，避免每段故事都从「她坐在窗前」开始
_TOPIC_SCENE: dict[str, str] = {
    "亲子关系": "她每周六给家里打电话，话题总在「吃了吗」之后停住。",
    "职场压力": "凌晨一点，办公室只剩他那一盏灯，屏幕上是还没改完的表格。",
    "亲密关系": "分开后的第一个月，她把两个人的聊天记录翻到了最上面。",
    "青春怀旧": "旧校服还压在箱底，口袋里是半张没写完的同学录。",
    "自我认同": "他在地铁里看玻璃上的自己，觉得那张脸有点陌生。",
    "生死与失去": "遗物是一只没上满发条的旧手表，停在某个下午三点。",
    "城市生活": "出租屋的窗朝北，一年到头照不进太阳，他却习惯了。",
}

#: 第二段按主导情绪出
_EMOTION_LINE: dict[str, str] = {
    "哀伤": "想念是不讲道理的，它不挑时间，只在最忙的时候忽然涌上来。",
    "愧疚": "他反复想那句没说出口的话——当时以为还有下一次。",
    "共鸣": "他以为自己是一个人，直到看见评论区里排着队说「我也是」。",
    "自嘲": "他学会了先把话说成笑话，这样别人就来不及笑话他。",
    "倦怠": "累不是身体的事，是每天早上睁眼都要重新说服自己一遍。",
    "不甘": "他不是没努力，只是努力和结果之间始终隔着一层看不见的玻璃。",
    "孤独": "手机屏幕亮了又暗，没有一条是想收到的那个人的。",
    "温暖": "有人隔着屏幕说了一句「好好生活」，他居然真的信了。",
    "怀念": "那时候什么都缺，回忆里却连空气都是亮的。",
    "焦虑": "他不敢算日子，一算就发现来不及的事情已经排到了明年。",
    "愤怒": "他气的不是这件事本身，是所有人都觉得这很正常。",
}

#: 关键词 → (插在第二段前的句子, 追加到末尾的句子)。命中即打断，顺序即优先级
_TWEAKS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (
        ("暗", "冷", "沉", "压抑", "悲", "狠"),
        "灯只开了一半，影子比人先到。",
        "雨到后半夜才停，谁也没等到天亮。",
    ),
    (
        ("亮", "暖", "温", "治愈", "轻", "希望"),
        "那天太阳很好，晾在竹竿上的衣服全是暖的。",
        "他忽然觉得，日子大概还是能过下去的。",
    ),
    (
        ("快", "节奏", "紧", "利落", "干脆"),
        "他只记得几个画面。",
        "然后就没有然后了。",
    ),
    (
        ("细", "长", "慢", "缓", "从容"),
        "他把那天的事从头到尾想了一遍，连风里的气味都还在。",
        "这些没有人知道，只有他自己一遍遍回放。",
    ),
)

_WORK_RE = re.compile(r"《([^》]{1,24})》")


def _args(payload: dict[str, Any]) -> str:
    """工具参数一律走 json.dumps——真模型返回的也是 JSON 文本。"""
    return json.dumps(payload, ensure_ascii=False)


def _tool_names(tools: Sequence[dict[str, Any]]) -> set[str]:
    return {
        str((t.get("function") or {}).get("name") or "")
        for t in tools
        if isinstance(t, dict)
    }


def _last_user_text(messages: Sequence[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


def _turn_round(messages: Sequence[ChatMessage]) -> int:
    """本回合内的第几轮（**不跨回合累加**）。

    历史里躺着上一回合交的稿，按整串数 assistant 会把修改轮的第一轮
    数成第二轮，脚本就走岔了。
    """
    last_user = max((i for i, m in enumerate(messages) if m.role == "user"), default=-1)
    return sum(1 for m in messages[last_user + 1 :] if m.role == "assistant") + 1


def _is_revision(messages: Sequence[ChatMessage]) -> bool:
    """最后一条 user 之前已经有过 assistant → 这是改稿轮，不是首次生成。"""
    last_user = max((i for i, m in enumerate(messages) if m.role == "user"), default=-1)
    return any(m.role == "assistant" for m in messages[:last_user])


def _agent_source_text(messages: Sequence[ChatMessage]) -> str:
    """打分用的原文：system（内含首轮原文片段）+ read_input 的结果。

    刻意**不并入其它工具结果**——知识库与古籍的文本会把情绪打分带偏，
    而用户粘贴的那段评论才是要写的东西。
    """
    parts = [m.content for m in messages if m.role == "system"]
    parts += [m.content for m in messages if m.role == "tool" and (m.name or "") == "read_input"]
    return "\n".join(parts)


def _work_names(messages: Sequence[ChatMessage]) -> list[str]:
    """从工具结果里抠出《书名》——证明「查过的东西真的进了稿子」。"""
    out: list[str] = []
    for m in messages:
        if m.role != "tool":
            continue
        for name in _WORK_RE.findall(m.content or ""):
            if name not in out:
                out.append(name)
    return out


def _story_probe(messages: Sequence[ChatMessage]) -> dict[str, str]:
    """从原文派生出的情绪/话题/检索词。换一段文本，这些值全变。"""
    source = _agent_source_text(messages)
    emotions = _top_labels(_score(source, EMOTION_LEXICON), 2)
    topics = _top_labels(_score(source, TOPIC_LEXICON), 1)
    e1 = emotions[0] if emotions else "哀伤"
    e2 = emotions[1] if len(emotions) > 1 else "温暖"
    return {
        "emotion": e1,
        "second": e2,
        "topic": topics[0] if topics else "城市生活",
        "query": _core_tension([e1, e2]),
    }


def _apply_tweak(paras: list[str], instruction: str) -> tuple[list[str], str]:
    """按用户的修改意见改稿。返回（段落, 改稿说明）。

    每一条分支都要产生**肉眼可见**的差异——用户看到的若是同一段文字，
    会以为按钮坏了。
    """
    out = list(paras)
    notes: list[str] = []

    if "短" in instruction or "精简" in instruction:
        out = out[:2] + out[-1:]
        notes.append("砍掉了中间一段")
    if "视角" in instruction or "人称" in instruction:
        out = [p.replace("他", "你").replace("她", "你") for p in out]
        notes.append("换成了第二人称")
    for keys, head, tail in _TWEAKS:
        if any(k in instruction for k in keys):
            out.insert(1, head)
            out.append(tail)
            notes.append(f"按「{keys[0]}」调了明暗与收尾")
            break

    if not notes:
        out.insert(1, "他把窗帘拉上了一半。")
        out.append("稿子改到这里，还留着一点没说完的。")
        notes.append("按你的要求重写了一版")
    return out, "；".join(notes)


def _mock_story(messages: Sequence[ChatMessage], *, instruction: str, version: int) -> str:
    probe = _story_probe(messages)
    scene = _TOPIC_SCENE.get(
        probe["topic"], "他把那段话读了三遍，第三遍时停了下来。"
    )
    body = _EMOTION_LINE.get(probe["emotion"], "有些话说不出口，就变成了叹气。")
    turn = f"说到底，他要的不多：{NEED_BY_EMOTION.get(probe['second'], '被听见')}。"
    works = _work_names(messages)
    closing = (
        f"《{_trim(works[0], 12)}》里也有过这样的夜晚——{probe['query']}。"
        if works
        else f"这样的夜晚古人也写过：{probe['query']}。"
    )

    paras = [scene, body, turn, closing]
    note = ""
    if instruction:
        paras, note = _apply_tweak(paras, instruction)

    title = f"《{probe['topic']}》" + (f"（第 {version} 稿）" if version > 1 else "")
    text = title + "\n\n" + "\n\n".join(paras)
    return f"{text}\n\n——改稿说明：{note}" if note else text


def _submit(
    messages: Sequence[ChatMessage], *, instruction: str, revising: bool
) -> dict[str, Any]:
    """收尾轮：交稿。"""
    prior = next(
        (m.content for m in reversed(messages) if m.role == "assistant" and m.content), ""
    )
    text = _mock_story(messages, instruction=instruction if revising else "", version=2 if revising else 1)
    if revising and text.strip() == prior.strip():
        # 兜底：宁可多一行改动说明，也不让用户看到一模一样的第二版
        text += "\n\n——改稿说明：这一版把语气又压低了一点。"
    return {"content": text, "tool_calls": []}


def _agent_reply(
    messages: Sequence[ChatMessage],
    tools: Sequence[dict[str, Any]],
    *,
    script: str = "default",
    round_no: int = 1,
) -> dict[str, Any]:
    """假模型的一轮：返回 {"content": str, "tool_calls": [(name, arguments_json)]}。

    `tools=[]`（强制作答轮，或工具被全部撤掉）时**必须**交稿——
    这里再要一次工具调用会撞上循环的未知工具分支，白烧一轮。
    """
    names = _tool_names(tools)
    instruction = _last_user_text(messages)
    revising = _is_revision(messages)

    if not tools:
        return _submit(messages, instruction=instruction, revising=revising)
    if script == "empty_answer":
        return {"content": "", "tool_calls": []}
    if script == "never_calls_tools":
        return _submit(messages, instruction=instruction, revising=revising)
    if script == "loops_forever":
        name = "kb_search" if "kb_search" in names else next(iter(sorted(names)), "")
        if not name:
            return _submit(messages, instruction=instruction, revising=revising)
        return {
            "content": "",
            "tool_calls": [(name, _args({"query": f"再查一次（第 {round_no} 轮）", "limit": 2}))],
        }
    if script == "bad_arguments":
        if "kb_search" in names:
            # 故意不是 JSON —— 真模型最常见的错法。连续两次后循环会把
            # kb_search 撤掉，那时这里落到交稿分支，正好证明撤销生效了。
            return {"content": "", "tool_calls": [("kb_search", "query=想念，limit=3")]}
        return _submit(messages, instruction=instruction, revising=revising)

    if revising:
        if round_no == 1 and "kb_search" in names:
            query = _trim(" ".join(instruction.split()), 24) or "同一段情绪的另一种说法"
            return {
                "content": "先照你说的方向再翻一遍知识库。",
                "tool_calls": [("kb_search", _args({"query": query, "limit": 3}))],
            }
        return _submit(messages, instruction=instruction, revising=True)

    if round_no == 1:
        probe = _story_probe(messages)
        calls: list[tuple[str, str]] = []
        if "kb_search" in names:
            calls.append(("kb_search", _args({"query": probe["query"], "limit": 3})))
        if "kb_graph" in names:
            calls.append(
                ("kb_graph", _args({"mode": "emotion", "key": probe["emotion"], "limit": 3}))
            )
        if calls:
            return {"content": "先去知识库里找相近的表达。", "tool_calls": calls}
        return _submit(messages, instruction=instruction, revising=False)

    if round_no == 2 and "novel_lookup" in names:
        return {
            "content": "再去古典文学库里找一段能对上的原文。",
            "tool_calls": [("novel_lookup", _args({"action": "books"}))],
        }

    return _submit(messages, instruction=instruction, revising=False)
