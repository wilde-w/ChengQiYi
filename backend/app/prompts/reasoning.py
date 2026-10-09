"""把「评论现象 + 检索到的证据」综合成一条条推理链。

**一次调用产出全部步骤**，不是每步一次。理由不是省调用：四步推理强互引
（第三步的机制要接得住第一步的现象，第四步的洞察要回收第二步的典故），
分四次调用等于把整条链的一致性交给运气，而模型每次只看得见自己那一段。

输出契约（顶层必须是 JSON **对象**——`parse_json_response` 只截最外层 `{}`，
DeepSeek 的 json_object 同样要求对象）：

```json
{"steps": [{"phenomenon": "…", "mechanism": "…", "insight": "…",
            "evidence_ids": ["…"], "allusion_ids": ["…"], "confidence": 0.7}]}
```

**四条硬约束，写完正文后由 Python 复核，不靠模型自觉**（见 `n6_reasoning`）：

  1. `evidence_ids` / `allusion_ids` 只能引用上下文里出现过的 id
  2. `allusion_ids` **只取文学与诗词**——心理学语料不是「典故」
  3. `confidence` 由 `constants.CONFIDENCE_WEIGHTS` 重算，模型自评只留档
  4. 最多 `MAX_REASONING_STEPS` 步

前两条是提示词与代码重复约束，这是刻意的：提示词让模型**大概率**写对，
代码让剩下的**必然**不会落库。产品的立身之本是「每条结论都能追溯」，
只靠提示词约束等于把这句话交给抽样。

**没有「典故」这一层文字。** `ReasoningStep` 只有 phenomenon / mechanism /
insight 三段，典故由 `allusion_ids` 指向证据卡。这不是偷懒：一旦在推理里
把典故的原文抄一遍，它就与证据卡各存了一份，两份迟早不一致，而用户看到
的是推理里引的那句和卡片上的对不上。
"""

from __future__ import annotations

from typing import Any

from app.constants import MAX_REASONING_STEPS
from app.graph.nodes._llm import evidence_brief
from app.providers.base import ChatMessage

TASK = "synthesize"

#: 送进提示词的证据条数上限。再多也只是把真正要对照的那几条挤出上下文窗口。
MAX_EVIDENCE = 12

#: 每条证据正文截断到多少字。同 `_llm.evidence_brief` 的默认值。
EVIDENCE_CHARS = 400

SYSTEM = f"""你在为一个中文内容分析系统做「因果链综合」。上游已经给出：
一批聚类后的评论主题、一份心理侧写、一批从知识库里检索到的证据
（心理学机制 / 古典文学 / 诗词）。你的任务是**把它们接成一条推理链**。

输出 JSON，形如：

{{"steps": [{{"phenomenon": "…", "mechanism": "…", "insight": "…",
  "evidence_ids": ["…"], "allusion_ids": ["…"], "confidence": 0.7}}]}}

每一步的结构：

- **`phenomenon`**：评论里观察到的现象，一到两句，写具体的行为或说法，
  不要写成「用户存在情绪困扰」这类空洞概括。最多 {MAX_REASONING_STEPS} 步。
- **`mechanism`**：解释这个现象背后的心理机制。必须落在**已给出的心理学
  证据**上，并写明出处（作者或理论名）。证据里没有能解释它的，就如实写
  「这一现象尚无已检索到的机制解释」，**不要用你的先验知识补**——用户点开
  证据卡是为了核对，编出来的机制在那里对不上任何一条。
- **`insight`**：这一步的落点，一句能让人停下来想一想的话。不要复述现象。
- **`evidence_ids`**：这一步引用的证据 id，**必须逐字照抄上下文里给出的
  id**。抄错一个字符，那条引用就会被丢弃，而卡片上只会少一个数字，
  没有人知道为什么。
- **`allusion_ids`**：这一步用到的典故 id，**只能从文学与诗词类证据里取**。
  心理学语料不是典故。没有贴切的就留空数组——硬凑一个不相关的典故，
  比没有典故更伤。
- **`confidence`**：你对这一步的把握（0–1）。这个数字只作留档，
  系统会用证据支撑度、机制完备度、典故有无三项重算一个对外展示的值。

四条准则：

1. **每步至少一条 `evidence_ids`。** 一条证据都引不到的步骤，说明它
   只是常识复述，不配出现在这条链上。
2. **不要重复同一个现象。** 五步说的是五件事，不是一件事的五个说法。
3. **`evidence_ids` 与 `allusion_ids` 不重叠**——同一份材料不该同时充当
   「机制」与「典故」。
4. 上下文里没有的证据，等于不存在。不要凭印象写 id。"""


def build_messages() -> list[ChatMessage]:
    """只放指令，材料通过 `context` 传（真实 provider 渲染到 user 消息尾部）。"""
    return [
        ChatMessage(role="system", content=SYSTEM),
        ChatMessage(
            role="user",
            content=f"请综合出最多 {MAX_REASONING_STEPS} 步推理链，按 JSON 输出，不要任何额外说明。",
        ),
    ]


def build_context(
    evidence: list[dict[str, Any]],
    *,
    profile: dict[str, Any] | None,
    clusters: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """构造传给 `complete(context=...)` 的结构。

    键名与 mock 处理器读取的一致（`ctx["evidence"]` / `ctx["profile"]` /
    `ctx["clusters"]`）——真实与 mock 两条路共用同一份形状。

    `evidence` 里每项的 `id` 就是 `chunk_id`（见 `_llm.evidence_brief`），
    所以模型抄进 `evidence_ids` 的天然就是跨库唯一、重跑稳定的那把键。
    """
    return {
        "evidence": evidence_brief(evidence[:MAX_EVIDENCE], limit=EVIDENCE_CHARS),
        "profile": _profile_digest(profile or {}),
        "clusters": _cluster_digest(clusters or []),
    }


def _profile_digest(profile: dict[str, Any]) -> dict[str, Any]:
    """只留下「做推理要用到的那几项」。

    整份 profile 塞进去是几百行分布数据，模型读不出重点，调用还更贵。
    """
    return {
        "core_tension": profile.get("core_tension"),
        "summary": profile.get("summary"),
        "global_tags": profile.get("global_tags") or {},
    }


def _cluster_digest(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """簇只留标签与摘要。代表评论与关键词在这里没用——现象该由
    `phenomenon` 概括，而不是让模型再读一遍原始评论。"""
    return [
        {
            "label": c.get("label"),
            "summary": c.get("summary"),
            "size": c.get("size"),
            "is_noise": bool(c.get("is_noise")),
        }
        for c in clusters[:8]
        if not c.get("is_noise")
    ]


__all__ = ["EVIDENCE_CHARS", "MAX_EVIDENCE", "TASK", "build_context", "build_messages"]
