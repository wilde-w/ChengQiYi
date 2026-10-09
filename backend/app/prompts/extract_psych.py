"""从簇标签里抽出心理结构。

**一次调用覆盖全部簇 + 全局**，不是「k 次逐簇 + 1 次聚合」。

这条与最初的设想不同，改的原因是 mock 侧的契约：`mock_llm._extract_psych`
读的是一个 `clusters` 数组、返回的是 `{cluster_tags, global_tags,
core_tension, summary}` 四件套——它把「逐簇」和「聚合」写在同一个处理器里。
拆成 k+1 次调用就要么让真实模型和 mock 读不同形状的返回值（`providers/base.py`
开篇第一条明确反对这种分叉），要么让每簇那 k 次调用也返回全域结构再丢掉一半。

改成一次调用还顺带解决两件事：① 全局视图不再依赖「把 k 份局部标签再拼一遍」，
模型能同时看到所有簇，张力与主导情绪的判断更连贯；② k=5 时省下 4 次往返，
mock 每次 900ms，演示时这是实打实的 4 秒。

代价是单簇的评论会与其他簇共享注意力。以每簇 3 条代表评论、每条 120 字算，
5 簇约 1800 字，远在上下文之内，不构成问题。

输出契约：

```json
{"cluster_tags": {"<cluster_key>": {"emotion": [...], "topic": [...], "need": [...]}},
 "global_tags": {"emotion": [...], "topic": [...], "need": [...]},
 "core_tension": "…", "summary": "…"}
```

**分布（每条标签的权重）不由模型给**——它在 Python 侧按簇大小加权算出来，
因为「这个情绪占多少」是个可复算的量，让模型估一个数字既不可核查又系统性
偏高。模型只负责「哪些标签属于哪一簇」。

system 里必须出现 "json" 字样（DeepSeek json_object 模式的硬要求）。
"""

from __future__ import annotations

from typing import Any

from app.kb.lexical_tags import EMOTION_VOCAB
from app.providers.base import ChatMessage

TASK = "extract_psych"

MAX_TAGS = 3

_EMOTION_LIST = "、".join(EMOTION_VOCAB)

SYSTEM = f"""你在为一个中文内容分析系统做心理语义抽取。系统已经把一条抖音视频的
评论分成了若干簇（每簇附若干条代表评论），需要你回答两个层次的问题。

输出 JSON，形如：

{{"cluster_tags": {{"<簇的 cluster_key>": {{"emotion": ["…"], "topic": ["…"], "need": ["…"]}}}},
 "global_tags": {{"emotion": ["…"], "topic": ["…"], "need": ["…"]}},
 "core_tension": "…", "summary": "…"}}

**逐簇（`cluster_tags`）**：键必须原样抄回输入里的 `cluster_key`，一个不多
一个不少。每个簇最多 {MAX_TAGS} 个 emotion、{MAX_TAGS} 个 topic、2 个 need。
某个簇确实读不出某种标签，就给空数组——**空数组比编一个强**。

**全局（`global_tags`）**：整条视频评论区的主导情绪 / 话题 / 需求，各最多
{MAX_TAGS} 个。按「多少人在说」排序，不是按「你说得最顺口」。

**`core_tension`**：一句话（20–40 字）说明这些情绪之间**互相拉扯**的地方。
这是整个报告最重要的一句。它不该是「用户很焦虑」这种复述，而要写出矛盾：
「想念却再也无法说出口」「想被看见又怕被打扰」。给不出张力时，就描述
最主导的那一种情绪如何自我加强。

**`summary`**：两到三句话（60–120 字）概述这个评论区在经历什么。

硬性规则：

1. **`emotion` 只能从下面这张表里选，不许自创**：
   {_EMOTION_LIST}
   表里没有贴切的就给空数组。**宁缺毋滥**——图谱里的情绪节点是合并出来的，
   自创词会让「焦虑」「不安」「忧惧」裂成三个节点，检索只命中三分之一。
2. 所有标签里**不许出现** ｜ 《 》 ： 、 换行 这些字符。
3. `topic` / `need` 都是 2–6 字的名词短语，不是句子，也不是标签的堆砌。
4. `need` 写的是**没说出口**的那一层。评论都在骂加班时，需求多半不是
   「少加班」，而是「被允许说不」。

最后一条，也是最重要的：**依据是评论本身，不是你的先验知识。**
这是一条特定的视频，不是一个「关于失恋的案例」。如果代表评论里读不出
某个情绪，那就是读不出。"""


#: 每个簇送几条代表评论进提示词。与 constants.REPRESENTATIVE_COMMENTS
#: 保持一致——那边是聚类节点取数的依据，这里是提示词侧的说明。
def build_messages(cluster_count: int) -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content=SYSTEM),
        ChatMessage(
            role="user",
            content=(
                f"下面是 {cluster_count} 个评论簇，逐簇抽取并给出全局判断，"
                "按 JSON 输出，不要任何额外说明。"
            ),
        ),
    ]


def build_context(
    clusters: list[dict[str, Any]],
    *,
    comment_stats: dict[str, Any] | None = None,
) -> dict[str, object]:
    """构造传给 `complete(context=...)` 的结构。

    `clusters` 里每一项的形状（`cluster_key` / `comments`）与
    `mock_llm._extract_psych` 读取的键名一致——真实与 mock 共用同一份
    context 形状，改一处就两条都改。

    `comment_stats` 只作为背景（总共多少条、多少条被判为广告），
    不要求模型回填：它影响的是「这个分布可信到什么程度」，而那是
    前端该显示的事，不是模型该复述的事。
    """
    ctx: dict[str, object] = {
        "clusters": [
            {
                "cluster_key": str(c.get("cluster_key") or ""),
                "size": int(c.get("size") or 0),
                "comments": list(c.get("comments") or []),
            }
            for c in clusters
        ]
    }
    if comment_stats:
        ctx["comment_stats"] = comment_stats
    return ctx


__all__ = ["MAX_TAGS", "TASK", "build_context", "build_messages"]
