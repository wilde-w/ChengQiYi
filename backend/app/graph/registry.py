"""流水线节点的元数据注册表——**进度文案与分带的唯一真源**。

为什么要集中一处：`GET /meta/pipeline` 会把这张表原样下发给前端，
前端不再硬编码任何一句中文。改文案只改这里。

另一件必须由这里决定的事是**进度分带**。PRD 给的是一组固定百分比，
但真实进度是数据相关的（「已获取 523 条评论」跟「已获取 40 条」不该走到同一个刻度）。
解法：每个节点占一段固定区间，节点内部按实际完成比例插值。
这样既有稳定的整体节奏，又不会在数据量变化时说谎。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.constants import NodeKey, SourceKind


@dataclass(frozen=True, slots=True)
class NodeSpec:
    key: NodeKey
    label: str
    start: float  # 进入该节点时的整体进度
    end: float  # 该节点完成时的整体进度
    entering: str  # 节点开始时的里程碑文案
    done: str  # 节点完成时的里程碑文案，可含 {占位符}
    weight_hint: str = ""  # 给前端排序/说明用的补充


# 分带刻意不均匀：n1/n2 是网络 IO（快但不可控），n3–n7 是计算与生成。
# n7 独占 10 个点，因为它是唯一逐字流式输出的节点，需要留足观感空间。
NODES: tuple[NodeSpec, ...] = (
    NodeSpec(
        key=NodeKey.N1_VIDEO,
        label="视频引入",
        start=2.0,
        end=15.0,
        entering="正在解析视频链接…",
        done="已获取视频信息",
    ),
    NodeSpec(
        key=NodeKey.N2_COMMENTS,
        label="评论抓取",
        start=15.0,
        end=35.0,
        entering="正在抓取评论…",
        done="已获取 {count} 条评论",
    ),
    NodeSpec(
        key=NodeKey.N3_CLUSTER,
        label="评论聚类",
        start=35.0,
        end=50.0,
        entering="正在对评论进行向量化与聚类…",
        done="评论聚类完成：{k} 个簇",
    ),
    NodeSpec(
        key=NodeKey.N4_PSYCH,
        label="心理语义抽取",
        start=50.0,
        end=63.0,
        entering="正在抽取情感/主题/需求…",
        done="心理语义抽取完成",
    ),
    NodeSpec(
        key=NodeKey.N5_RETRIEVAL,
        label="多路检索",
        start=63.0,
        end=78.0,
        entering="正在检索心理学与文学证据…",
        done="检索到 {psychology} 条心理学证据、{literature} 条文学典故",
    ),
    NodeSpec(
        key=NodeKey.N6_REASONING,
        label="推理综合",
        start=78.0,
        end=90.0,
        entering="正在构建推理链…",
        done="推理完成：{steps} 步",
    ),
    NodeSpec(
        key=NodeKey.N7_LITERARY,
        label="文学化生成",
        start=90.0,
        end=100.0,
        entering="正在生成洞察…",
        done="洞察生成完成",
    ),
)

BY_KEY: dict[NodeKey, NodeSpec] = {spec.key: spec for spec in NODES}
ORDER: tuple[NodeKey, ...] = tuple(spec.key for spec in NODES)

# 第一个节点的带起点即整体起始进度（2.0，而非 0——0 是「已受理」的意思）
START_PROGRESS = NODES[0].start
END_PROGRESS = NODES[-1].end


def spec_for(key: NodeKey | str) -> NodeSpec:
    return BY_KEY[NodeKey(key)]


#: 文本源专用的节点文案：key → (entering, done)。
#:
#: n1/n2 的默认文案说的是「解析视频链接」「抓取评论」——文本源里这两件事
#: 一件都没有发生。让它们照常播报，等于当着用户的面说一句假话，而过程记录
#: 恰恰是他用来看「它到底做了什么」的地方。
#:
#: 只覆盖这两句：n3–n7 的文案对文本源同样成立（评论聚类、检索、推理、生成）。
TEXT_SOURCE_COPY: dict[NodeKey, tuple[str, str]] = {
    NodeKey.N1_VIDEO: ("正在读取手动文本…", "已接收手动文本"),
    NodeKey.N2_COMMENTS: ("正在切分文本…", "已切出 {count} 条"),
}


def entering_for(key: NodeKey | str, *, source_kind: str = SourceKind.DOUYIN.value) -> str:
    spec = spec_for(key)
    if source_kind == SourceKind.TEXT and spec.key in TEXT_SOURCE_COPY:
        return TEXT_SOURCE_COPY[spec.key][0]
    return spec.entering


def done_template_for(key: NodeKey | str, *, source_kind: str = SourceKind.DOUYIN.value) -> str:
    """节点完成文案的模板（可能带 {count} 这类占位符）。"""
    spec = spec_for(key)
    if source_kind == SourceKind.TEXT and spec.key in TEXT_SOURCE_COPY:
        return TEXT_SOURCE_COPY[spec.key][1]
    return spec.done


def predecessor(key: NodeKey | str) -> NodeKey | None:
    """该节点的前驱。干预（aupdate_state 的 as_node）需要它来定位重跑起点。"""
    idx = ORDER.index(NodeKey(key))
    return ORDER[idx - 1] if idx > 0 else None


def successors(key: NodeKey | str, *, inclusive: bool = False) -> tuple[NodeKey, ...]:
    """该节点之后的全部节点。用于判断一次干预会级联影响哪些产物。"""
    idx = ORDER.index(NodeKey(key))
    return ORDER[idx:] if inclusive else ORDER[idx + 1 :]


def describe() -> list[dict[str, object]]:
    """下发给前端的形态。字段名即前端类型定义，改动需同步 api/types.ts。"""
    return [
        {
            "key": spec.key.value,
            "label": spec.label,
            "start": spec.start,
            "end": spec.end,
            "entering": spec.entering,
            "done": spec.done,
        }
        for spec in NODES
    ]
