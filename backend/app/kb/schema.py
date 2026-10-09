"""知识库 chunk 的统一模型。

**为什么要有这一层。** 三个库的磁盘格式是各自领域自然长出来的，字段名互不相同：
心理学讲 discipline/concept/source/author，古典文学讲 book/chapter/character，
诗词讲 poem_title/dynasty。直接把它们塞进同一个 Qdrant payload 会得到一堆
「有的文档有 book 没有 poem_title」的异构数据，检索时无从下手。

所以：**磁盘上保留各自的原始字段名**（改语料文件是高风险动作，而且人写语料时
按领域习惯写最不容易出错），**加载时映射成这一个模型**。`library` 是判别字段，
payload 里所有可选字段一律给出（缺失为 None / 空列表），下游不用再做存在性判断。

`text_for_embedding()` 把元数据烘进被嵌入的文本——这是检索质量最大的杠杆，
理由见该方法的注释。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.constants import Library

# Neo4j 里 Work 节点的名字来源：心理学用文献名，文学用书名，诗词用诗题。
# 它同时是图检索里「由意象找作品」那一跳的落点。


@dataclass(slots=True)
class KBChunk:
    chunk_id: str
    library: Library
    text: str

    # --- 分类与标签 ---
    type: str | None = None
    imagery: list[str] = field(default_factory=list)
    emotion: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    # --- 出处 ---
    author: str | None = None
    year: int | None = None
    #: 心理学：文献名；文学：书名；诗词：诗题。三者语义相同，都是「作品」。
    work: str | None = None
    #: 文学专有
    chapter: str | None = None
    character: str | None = None
    dynasty: str | None = None
    # --- 主题 ---
    discipline: str | None = None
    concept: str | None = None
    #: 文学条目给出的一段现代处境解读，不参与嵌入（它是元评论，不是原文）
    context: str | None = None
    #: 导入来源的原始文件名。语料条目为 None。
    #: **不进 `text_for_embedding()`**——文件名不该被嵌入，那是行政信息不是内容。
    #: 它是「删掉这一次导入」的检索键；`work` 是第二条路径（文件改了名
    #: 仍能按书清理），两条都留着不是冗余，是两种不同的回滚语义。
    origin: str | None = None

    def text_for_embedding(self) -> str:
        """构造被送进 embedding 模型的文本。

        **这不是「把 text 原样嵌进去」那么简单，也不该是。**
        用户查询是「落花 哀伤」这类短标签组合，而原文是「花谢花飞花满天」——
        如果只嵌原文，两者在向量空间里离得很远，检索会退化成靠运气。
        把 discipline/意象/情感这些标签一并嵌进去，等于给每条语料
        预先挂上一串用户真会用的检索词，短查询才够得着长原文。

        模板按库区分而不是取并集：诗词没有「人物」，硬留一个空的
        「人物：」段只会稀释掉有效信号，让同一库内的向量彼此更接近。
        """
        if self.library is Library.PSYCHOLOGY:
            head = "｜".join(x for x in (self.discipline, self.concept, self.author) if x)
            parts = [head]
            if self.work:
                parts.append(f"《{self.work}》")
            if self.keywords:
                parts.append("关键词：" + "、".join(self.keywords))
            return f"{''.join(parts)}：{self.text}"

        if self.library is Library.LITERATURE:
            # 空值要**整段略去**，不能留下空壳：`f"《{work or ''}》"` 在
            # work 为空时渲染出「《》」，那对向量是纯粹的噪声，还会让
            # 所有缺书名的条目彼此更像。
            parts: list[str] = []
            head = f"《{self.work}》" if self.work else ""
            if self.chapter:
                head += self.chapter
            if head:
                parts.append(head)
            if self.character:
                parts.append(f"人物：{self.character}")
            if self.imagery:
                parts.append("意象：" + "、".join(self.imagery))
            if self.emotion:
                parts.append("情感：" + "、".join(self.emotion))
            prefix = "｜".join(parts)
            return f"{prefix}｜{self.text}" if prefix else self.text

        # 诗词
        head = f"《{self.work}》" if self.work else ""
        if self.dynasty or self.author:
            head += "·".join(x for x in (self.dynasty, self.author) if x)
        parts = [head] if head else []
        if self.imagery:
            parts.append("意象：" + "、".join(self.imagery))
        if self.emotion:
            parts.append("情感：" + "、".join(self.emotion))
        prefix = "｜".join(parts)
        return f"{prefix}｜{self.text}" if prefix else self.text

    def payload(self) -> dict[str, Any]:
        """进 Qdrant 的 payload。

        **全文存在这里**（放磁盘），Neo4j 只存 snippet。检索命中后不必回
        关系库取正文，一次请求就够；也让「两个存储漂移」在结构上不可能发生。
        值为 None 的字段一律不写——Qdrant 的 filter 对「字段不存在」与
        「字段为 null」处理不同，少写就少一种边界情况。
        """
        data: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "library": str(self.library),
            "text": self.text,
        }
        for key in (
            "type", "author", "year", "work", "chapter",
            "character", "dynasty", "discipline", "concept", "context", "origin",
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        for key in ("imagery", "emotion", "keywords"):
            value = getattr(self, key)
            if value:
                data[key] = value
        return data

    def snippet_payload(self) -> dict[str, Any]:
        """给 Neo4j 的轻量字段：**不含全文**，只留书名/作者/短摘。"""
        return {
            "chunk_id": self.chunk_id,
            "library": str(self.library),
            "work": self.work,
            "author": self.author,
            "dynasty": self.dynasty,
            "character": self.character,
        }


@dataclass(slots=True)
class GraphEdge:
    """语料里手写的聚合边。

    自动从 chunk 里生成的边只能表达「这条语料同时提到落花和哀伤」，
    而 `落花 -[ASSOCIATED_WITH]-> 哀伤` 是一个关于**文化惯例**的断言，
    权重也是人评出来的。两者都进图，但来源要能区分，所以带 `weight`
    和显式的 source_type/target_type。
    """

    source: str
    source_type: str
    target: str
    target_type: str
    relation: str
    weight: float = 1.0


def dumps(obj: Any) -> str:
    """写回语料时的统一序列化（ensure_ascii=False，中文可读）。"""
    return json.dumps(obj, ensure_ascii=False)
