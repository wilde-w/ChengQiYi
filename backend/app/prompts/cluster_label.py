"""给一个评论簇命名。

**一次只给一个簇。** 提示词里不放别的簇——簇之间的差别是「这一簇和别的不
一样」，一旦把邻居也塞进来，模型会开始做对比，产出「其他簇更偏向…」这类
描述，而我们需要的是一个能贴在卡片标题上的短语。

输出契约（顶层必须是 JSON **对象**，`parse_json_response` 只截最外层 `{}`，
DeepSeek 的 json_object 同样要求对象）：

```json
{"label": "…", "summary": "…", "emotion": ["…"], "topic": ["…"], "need": ["…"]}
```

system 里必须出现 "json" 字样——DeepSeek 开 json_object 模式时强制要求，
否则直接 400。这句话看着像废话，但删掉它的后果只有在接真实 provider 的
那几天才出现，而那时离现在很远。

**注意这与 `extract_psych` 的分工**：这里产出的是「人话标签」，是给用户看的
卡片；`extract_psych` 产出的是结构化的三组标签，用于聚合与检索。两者的
`emotion` 词表是同一张闭集表，否则同一簇在两个界面显示成两个情绪。
"""

from __future__ import annotations

from app.kb.lexical_tags import EMOTION_VOCAB
from app.providers.base import ChatMessage

TASK = "cluster_label"

MAX_EMOTION = 3
MAX_TOPIC = 3
MAX_NEED = 2

_EMOTION_LIST = "、".join(EMOTION_VOCAB)

SYSTEM = f"""你在为一个中文内容分析系统读一条抖音视频的评论区。系统已经把评论
按语义分成了若干簇，现在需要你**给其中一簇命名**。

输出 JSON，形如：

{{"label": "…", "summary": "…", "emotion": ["…"], "topic": ["…"], "need": ["…"]}}

各字段的要求：

- **`label`**：8 字以内，能直接贴在卡片标题上。要具体——「失恋后的反复查看」
  比「情感困扰」有用，「加班到深夜的自我怀疑」比「职场压力」有用。
  **不许出现** ｜ 《 》 ： 、 换行 这些字符（它们会破坏前端拼接的结构）。
- **`summary`**：一到两句话（40–70 字），说明这簇人在说什么、语气是什么。
  写给人看，不要写成标签的堆砌。
- **`emotion`**：最多 {MAX_EMOTION} 个，**只能从下面这张表里选，不许自创**：
  {_EMOTION_LIST}
  表里没有贴切的就给空数组。**宁缺毋滥**：编一个不存在的情绪，会让它在
  聚合时对不上任何一个别处的标签，前端分布条上长出一根永远只有一条的柱子。
- **`topic`**：最多 {MAX_TOPIC} 个话题词，2–6 字的名词短语（例：亲密关系、
  职场倦怠、代际沟通）。同样不许自创长句。
- **`need`**：最多 {MAX_NEED} 个**未被说出口的**需求，2–6 字（例：被看见、
  被允许休息、确认自己没错）。这是表层抱怨底下的东西——如果评论都在骂
  加班，需求多半不是「少加班」，而是「被允许说不」。

**判断依据是评论本身，不是你的先验。** 这一簇如果没有明显的情绪，就
如实给空数组，不要为了让卡片好看而硬凑。用户在卡片上还会看到这一簇的
原始关键词与代表评论，编出来的标签会被当场看穿。"""


def build_messages(comment_count: int) -> list[ChatMessage]:
    """只放指令，评论通过 `context` 传（真实 provider 渲染到 user 消息尾部）。"""
    return [
        ChatMessage(role="system", content=SYSTEM),
        ChatMessage(
            role="user",
            content=f"这一簇有 {comment_count} 条评论，为它命名，按 JSON 输出，不要任何额外说明。",
        ),
    ]


def build_context(
    comments: list[str],
    *,
    size: int,
    keywords: list[str] | None = None,
) -> dict[str, object]:
    """构造传给 `complete(context=...)` 的结构。

    `keywords` 是本地 n-gram 统计算出来的高频词，与评论一起给模型：
    统计量能指出「这簇反复出现哪几个词」，而模型负责说这些词意味着什么。
    两者给出的东西不同，不是冗余。

    `comments` 键名与 mock 处理器读取的键名一致（`ctx.get("comments")`）——
    真实与 mock 两条路共用同一份 context 形状，改一处就两条都改。
    """
    ctx: dict[str, object] = {
        "size": size,
        "comments": list(comments),
    }
    if keywords:
        ctx["keywords"] = list(keywords)
    return ctx


__all__ = ["MAX_EMOTION", "MAX_NEED", "MAX_TOPIC", "TASK", "build_context", "build_messages"]
