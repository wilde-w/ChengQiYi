"""给导入的长文打标。

**一套 schema 服务三个库**，而不是写三套提示词。差异只在「哪些字段对
这个库有意义」上，用一段按库渲染的说明表达就够；写三套提示词意味着
三份会各自漂移的契约，而漂移的表现是某个库的标签悄悄变少。

输出契约（顶层必须是 JSON **对象**，`parse_json_response` 只截最外层
`{}`，DeepSeek 的 json_object 同样要求对象）：

```json
{"items": [{"i": 0, "quote": "…", "concept": "…", "keywords": ["…"],
            "imagery": ["…"], "emotion": ["…"], "type": "…"}]}
```

system 里必须出现 "json" 字样——DeepSeek 开 json_object 模式时强制要求，
否则直接 400。这句话看着像废话，但删掉它的后果只有在接真实 provider
的那天才出现，而那天离现在很远。
"""

from __future__ import annotations

from app.constants import Library
from app.kb.lexical_tags import EMOTION_VOCAB
from app.providers.base import ChatMessage

TASK = "tag_chunks"

#: 一个块最多给几个关键词。与词法兜底保持一致。
MAX_KEYWORDS = 6
MAX_IMAGERY = 3
MAX_EMOTION = 3

#: 每批几条。8 条 × 约 400 字 ≈ 3200 字上下文，DeepSeek 完全吃得下，
#: 而单次调用失败只损失 8 条（切半重试的粒度也不至于太细）。
BATCH_SIZE = 8

_EMOTION_LIST = "、".join(EMOTION_VOCAB)

_COMMON = f"""你在为一个中文内容分析系统的知识库做标注。系统会把标注拼进被
检索的文本里（形如「关键词：此在、时间性｜正文…」），所以标注的质量直接
决定这段文字将来能不能被查出来。

输出 JSON，形如：

{{"items": [{{"i": 0, "quote": "…", "concept": "…", "keywords": ["…"],
  "imagery": ["…"], "emotion": ["…"], "type": "…"}}]}}

`items` 里每一条对应输入里编号相同的那一段，`i` 必须原样抄回来。

硬性规则，逐条都会被程序校验，违反的那一条会被整批作废：

1. **`quote` 必须是该段正文里逐字出现的一段话**（8–30 字）。它的用途是
   核对「你标注的确实是这一段」。从正文里复制，不要改写。
2. **`keywords`、`imagery` 里的每一项都必须是该段正文里逐字出现的词。**
   不要给一些文外的漂亮词——那会让这段文字被它根本没回答的查询召回。
3. **`emotion` 只能从下面这张表里选，不许自创**：
   {_EMOTION_LIST}
   表里没有贴切的，就给空数组。**宁缺毋滥**：编一个不存在的情绪，会让
   图谱里长出一个永远查不到对应内容的孤立节点。
4. 所有标签里**不许出现** ｜ 《 》 ： 、 换行 这些字符——它们会破坏上面
   那个拼接模板的结构。
5. 每段最多 {MAX_KEYWORDS} 个 `keywords`、{MAX_IMAGERY} 个 `imagery`、
   {MAX_EMOTION} 个 `emotion`。

标注要**具体**：「时间性」比「哲学」有用，「此在」比「人」有用。
抽象到任何一段文字都能用的词，等于没有信息。"""


_LIBRARY_RULES: dict[Library, str] = {
    Library.PSYCHOLOGY: """这些是**心理学/哲学/神经科学的论述文本**。

- `concept`：**必填**，这段在讲哪个概念或mechanism，2–10 字的名词术语
  （例：情绪粒度、反事实思维、此在、时间性）。给不出就抄一个正文里的核心名词。
- `keywords`：这段的 3–6 个核心术语，全部取自正文。
- `imagery`：**必须是空数组 `[]`。** 这类论述文本没有意象标注，
  填了会把它们混进文学检索的结果里。
- `emotion`：只在这段确实在讨论某种情绪时才填（多数段落应该是空的）。
- `type`：从 论述 / 实证 / 综述 / 评论 里选一个。""",
    Library.LITERATURE: """这些是**文学文本**（小说、散文、史传、议论）。

- `imagery`：这段借哪些**具体物象**在说话（例：落花、明月、坟、旧物）。
  只收在文里真实出现、且承载情绪的物象，不要收纯粹的字面景物。
- `emotion`：这段传递的情绪，从闭集表里选。
- `keywords`：3–6 个核心词，全部取自正文。
- `concept`：**留空字符串。**
- `type`：从 散文 / 议论 / 史传 / 小说 / 诗词 里选一个。

意象和情绪**至少要有一样**。两样都空的话，这段文字在检索里够不着任何
查询，会被直接丢弃——所以如果你觉得这段确实有内容，请认真找一找。""",
    Library.POETRY: """这些是**诗词文本**。

- `imagery`：诗里出现的**意象**（例：明月、落花、孤舟、白发）。
- `emotion`：诗的情绪基调，从闭集表里选。
- `keywords`：2–4 个关键词，取自正文。
- `concept`：**留空字符串。**
- `type`：从 诗 / 词 / 曲 / 赋 里选一个。

意象和情绪**至少要有一样**，否则这段会被丢弃。""",
}


def build_system_prompt(library: Library) -> str:
    """按库拼出 system prompt。"""
    return f"{_COMMON}\n\n---\n\n{_LIBRARY_RULES[library]}"


def build_messages(library: Library, batch_size: int) -> list[ChatMessage]:
    """只放指令，正文通过 `context` 传（真实 provider 会渲染到 user 消息尾部）。"""
    return [
        ChatMessage(role="system", content=build_system_prompt(library)),
        ChatMessage(
            role="user",
            content=f"为下面 {batch_size} 段正文打标，按 JSON 输出，不要有任何额外说明。",
        ),
    ]


def render_item(index: int, text: str) -> str:
    """把一条正文渲染成给模型看的一行。

    **序号紧贴文本**（`[3] 正文：…`）是刻意的：模型抄 40 字符的 chunk_id
    一定会改数字，抄一个短整数可靠得多。这个序号是回传配对的唯一依据——
    绝不能改成按数组下标 zip。
    """
    return f"[{index}] 正文：{text}"


def build_context(
    library: Library,
    items: list[tuple[int, str]],
    *,
    work: str = "",
    author: str = "",
    discipline: str = "",
) -> dict[str, object]:
    """构造传给 `complete(context=...)` 的结构。

    书名/作者/领域由用户填一次即可，**不让模型对每一段重复猜**：省 token，
    也更准。它们只作为背景出现在这里，`discipline` 不要求模型回填。
    """
    return {
        "library": str(library),
        "work": work,
        "author": author,
        "discipline": discipline,
        "items": [{"i": i, "text": text} for i, text in items],
    }


__all__ = [
    "BATCH_SIZE",
    "TASK",
    "build_context",
    "build_messages",
    "build_system_prompt",
    "render_item",
]
