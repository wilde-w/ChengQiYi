"""把聚类、侧写与证据写成右栏的四段洞察。

**一次调用产出四段**，不是每段一次。三个理由，缺一条都还值得争论：

1. 四段强互引——机制段要复述侧写给出的张力，洞察段要回收文学段的典故。
   分四次调用等于把整段的互文关系交给运气，模型每次只看得见自己那一段。
2. 四次调用就是四倍延迟，而右栏是用户唯一会逐字读下去的地方。
3. `json_mode` 下的流式增量不是合法 JSON 前缀，边收边解析根本不可能。
   真流式换不来收益，逐字效果改由本地切块产生（见 `n7_literary`）。

输出契约是**顶层 JSON 对象**，四个键各是一段 markdown：

```json
{"profile": "…", "mechanism": "…", "allusion": "…", "insight": "…"}
```

键名与 `SectionKey` 一一对应。`parse_json_response` 只截最外层 `{}`，
所以四段必须是同一个对象的键，不能是数组——数组会被截成一堆碎片。

**只允许一个 markdown 子集**：`##` 标题、段落、`**粗体**`、`> 引用块`、
`- 列表`。这不是审美偏好，是安全边界：前端的渲染器是手写的（不引 markdown
库，理由见 `Markdown.tsx`），子集之外的东西会按**纯文本原样显示**——
模型写 `<script>` 就真的会显示成 `<script>` 这六个字符。约束在这里是为了
让「显示成什么样」和「模型想表达什么」保持一致。

引用一律写成内联的 `[[ev:<id>]]`，id 必须逐字照抄上下文里给的那些。
写完由 Python 复核并剥离（`graph/validators.py`），不靠模型自觉。
"""

from __future__ import annotations

from typing import Any

from app.constants import SectionKey
from app.graph.nodes._llm import evidence_brief
from app.providers.base import ChatMessage

TASK = "literary"

#: 单段重生成的 task 名。**必须是独立的名字**：它和整篇生成的提示词不同、
#: 缓存键也不同，共用一个 task 会让「重试」拿到整篇生成时留下的缓存。
REGEN_TASK = "section_regen"

#: 四段加起来的上限。mock 输出约 1200 字，真实模型写得更散，留一倍余量。
MAX_TOKENS = 2400

#: 送进提示词的证据条数。与 n6 一致——两处看到的证据不一致的话，
#: 推理链里引得到的东西，正文里会写不出来。
MAX_EVIDENCE = 12

EVIDENCE_CHARS = 400

#: 每段正文的软上限，用来说明「这段该有多长」。
SECTION_CHARS = 400

SYSTEM = f"""你在为一个中文内容分析系统写「洞察报告」。上游已经给出：
一批聚类后的评论主题、一份心理侧写、一批从知识库里检索到的证据
（心理学机制 / 古典文学 / 诗词）。

输出 JSON，四个键各是一段 markdown：

{{"profile": "…", "mechanism": "…", "allusion": "…", "insight": "…"}}

四段的分工：

- **profile（心理侧写）**：这批评论在说什么。先写他们表面上在谈论什么，
  再写反复出现的那个真正的处境。**这一段不引用证据**——它描述的是聚类
  结果本身，不是检索来的知识。每段 {SECTION_CHARS} 字左右。
- **mechanism（科学机制）**：用心理学证据解释上面那个处境。每条机制都要
  写明出处（作者或理论名），并紧跟一枚引用标记。不要罗列证据摘要，
  要让读者看懂「这个机制如何解释我看到的那些评论」。
- **allusion（文学类比）**：同一处境在古典文本里的更早、更凝练的写法。
  引用原文，说明它和当下评论的相似处。**注意分寸**：类比是为了照亮，
  不是为了显得有文化，牵强的对应比没有类比更伤。
- **insight（最终洞察）**：落点。一句话能让读者停下来想一想，并回收
  前三段——它不是总结，是让前面所有材料突然变得有意义的那句话。

## 引用

写了机制或典故的句子后面紧跟 `[[ev:<id>]]`，id **必须逐字照抄**上下文里
给出的那些。抄错一个字符，这条引用就会被丢弃，而读者只会看到某个数字
变小了，没有人知道为什么。上下文里没有的证据等于不存在，不要凭印象写。

## 排版

只允许这几种写法，其余的一律按纯文本原样显示：

- 段落之间空一行
- `**粗体**` 用于关键概念，**不要整句加粗**
- `> ` 开头表示引用原文（文学段引用诗句时用）
- `- ` 开头表示列表项
- `## ` 开头表示小标题

不要写 HTML、不要写表格、不要写代码块、不要写链接语法。
不要输出 JSON 之外的任何说明文字。"""


def build_messages() -> list[ChatMessage]:
    """只放指令，材料通过 `context` 传（真实 provider 渲染到 user 消息尾部）。"""
    return [
        ChatMessage(role="system", content=SYSTEM),
        ChatMessage(
            role="user",
            content="请按 JSON 输出这四段，键名必须是 profile / mechanism / allusion / insight。",
        ),
    ]


def build_regen_messages(key: SectionKey) -> list[ChatMessage]:
    """单段重生成。指令里点名是哪一段、要写成什么样。"""
    return [
        ChatMessage(role="system", content=SYSTEM),
        ChatMessage(
            role="user",
            content=(
                f"只重写 **{key.value}**（{key.title}）这一段，"
                f'按 JSON 输出 {{"{key.value}": "…"}}，不要输出其他键。'
            ),
        ),
    ]


def build_context(
    evidence: list[dict[str, Any]],
    *,
    profile: dict[str, Any] | None,
    clusters: list[dict[str, Any]] | None,
    video: dict[str, Any] | None,
) -> dict[str, Any]:
    """构造传给 `complete(context=...)` 的结构。

    键名与 mock 处理器读的一致（`ctx["evidence"]` / `ctx["profile"]` /
    `ctx["clusters"]` / `ctx["video"]`）——真实与 mock 两条路共用同一份形状。
    """
    return {
        "evidence": evidence_brief(evidence[:MAX_EVIDENCE], limit=EVIDENCE_CHARS),
        "profile": _profile_digest(profile or {}),
        # 与 n6 不同，这里**保留 `emotion_tags`**：侧写段要写出「哪个簇是什么
        # 情绪基调」，砍掉它模型就只能对着簇标签猜。
        "clusters": _cluster_digest(clusters or []),
        "video": _video_digest(video or {}),
    }


def build_regen_context(
    evidence: list[dict[str, Any]],
    *,
    key: SectionKey,
    instruction: str,
    profile: dict[str, Any] | None,
    clusters: list[dict[str, Any]] | None,
    video: dict[str, Any] | None,
) -> dict[str, Any]:
    """重生成用的 context。在 `build_context` 之上加两个键。

    `section_key` 与 `instruction` 是 mock 处理器读的（它按 `section_key`
    挑要重写哪一段）；`attempt` 由 `_llm.call_json` 注入，这里不写——
    写了会和它注入的那份打架。
    """
    ctx = build_context(evidence, profile=profile, clusters=clusters, video=video)
    ctx["section_key"] = key.value
    ctx["instruction"] = instruction
    return ctx


def _profile_digest(profile: dict[str, Any]) -> dict[str, Any]:
    """只留写作要用到的几项。整份分布数据塞进去，模型读不出重点。"""
    return {
        "core_tension": profile.get("core_tension"),
        "summary": profile.get("summary"),
        "global_tags": profile.get("global_tags") or {},
    }


def _cluster_digest(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "label": c.get("label"),
            "summary": c.get("summary"),
            "size": c.get("size"),
            "emotion_tags": list(c.get("emotion_tags") or [])[:3],
        }
        for c in clusters[:8]
        if not c.get("is_noise")
    ]


def _video_digest(video: dict[str, Any]) -> dict[str, Any]:
    """标题与作者。洞察段要写出「这条视频是什么」，不给出处就只能写「这个视频」。"""
    return {
        "title": video.get("title") or video.get("caption"),
        "author": video.get("author_name") or video.get("author"),
    }


__all__ = [
    "MAX_EVIDENCE",
    "MAX_TOKENS",
    "REGEN_TASK",
    "SECTION_CHARS",
    "TASK",
    "build_context",
    "build_messages",
    "build_regen_context",
    "build_regen_messages",
]
