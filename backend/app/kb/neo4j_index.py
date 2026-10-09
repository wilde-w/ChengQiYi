"""Neo4j 索引写入。

**图里存什么、不存什么，是这一层唯一要守住的事。** 只有 snippet（≤60 字）
与几个定位字段，正文一律回 Qdrant 按 chunk_id 取。这样两个存储不可能漂移：
它们不是同一份数据的两个副本，而是「索引」与「内容」。全量重刷图也不会有
一致性问题，因为图里本来就没有会过期的内容。

写入用 UNWIND 批量展开。逐条 `session.run` 在 150 条语料上要多跑 600 次
网络往返，Windows 上的 bolt 连接会让这个数字变成分钟级。
"""

from __future__ import annotations

from collections.abc import Sequence

from app.clients.neo4j_client import ensure_constraints, graph_session, snippet
from app.constants import Library
from app.kb.schema import GraphEdge, KBChunk
from app.logging_conf import get_logger

log = get_logger(__name__)

BATCH = 100


async def reset_graph() -> None:
    """清空整张图。只在 `ingest-kb --reset` 时调用。

    先删关系再删节点，避免 DETACH DELETE 在大图上一次性持有过多事务内存。
    语料量级（几百节点）其实无所谓，但这条路径将来会被 expand-poetry 撑大。
    """
    async with graph_session() as session:
        await session.run("MATCH ()-[r]->() DELETE r")
        await session.run("MATCH (n) DELETE n")
    log.info("neo4j_graph_reset")


async def write_chunks(chunks: Sequence[KBChunk]) -> None:
    """写节点与边。整批幂等——MERGE 按约束字段匹配，重复跑不会长出新节点。"""
    if not chunks:
        return
    await ensure_constraints()

    rows = [_row(c) for c in chunks]
    for start in range(0, len(rows), BATCH):
        batch = rows[start : start + BATCH]
        async with graph_session() as session:
            await session.run(_CHUNK_CYPHER, rows=batch)
            await session.run(_IMAGERY_CYPHER, rows=batch)
            await session.run(_EMOTION_CYPHER, rows=batch)
            await session.run(_CHARACTER_CYPHER, rows=batch)
            await session.run(_WORK_CYPHER, rows=batch)
            await session.run(_CONCEPT_CYPHER, rows=batch)
    log.info("neo4j_chunks_written", count=len(rows))


async def delete_chunks(chunk_ids: Sequence[str]) -> int:
    """删除语料里已移除的 chunk 节点。

    只删 Chunk 本身（DETACH），不动 Imagery / Emotion 这些共享节点——
    它们属于整张图，删掉一条语料不该让「落花」这个意象消失。
    真正孤立的聚合节点会留在图上，这是可接受的：它们是文化词汇表，
    不是某条语料的附属品。
    """
    if not chunk_ids:
        return 0
    async with graph_session() as session:
        result = await session.run(
            "UNWIND $ids AS id MATCH (c:Chunk {chunk_id: id}) DETACH DELETE c RETURN count(*) AS n",
            ids=list(chunk_ids),
        )
        record = await result.single()
    return int(record["n"]) if record else 0


def _row(chunk: KBChunk) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "library": str(chunk.library),
        "snippet": snippet(chunk.text),
        "work": chunk.work,
        "author": chunk.author,
        "dynasty": chunk.dynasty,
        "character": chunk.character,
        "chapter": chunk.chapter,
        "imagery": chunk.imagery,
        "emotion": chunk.emotion,
        "keywords": chunk.keywords,
        # 心理学的 concept 是图里的检索入口（Concept → 意象 → 语料）
        "concept": chunk.concept if chunk.library is Library.PSYCHOLOGY else None,
    }


# Cypher 里的标签不能参数化，所以这六条语句的标签与关系名是硬编码的。
# 值全部走 $ 参数。graph_edges.jsonl 那边的类型白名单（loader.GRAPH_NODE_TYPES）
# 是同一套名字——语料里写错的类型会在加载时报错，不会流到这里拼进查询。
_CHUNK_CYPHER = """
UNWIND $rows AS row
MERGE (c:Chunk {chunk_id: row.chunk_id})
SET c.library = row.library,
    c.snippet = row.snippet,
    c.chapter = row.chapter
"""

_IMAGERY_CYPHER = """
UNWIND $rows AS row
MATCH (c:Chunk {chunk_id: row.chunk_id})
UNWIND row.imagery AS name
MERGE (i:Imagery {name: name})
MERGE (c)-[:MENTIONS_IMAGERY]->(i)
"""

_EMOTION_CYPHER = """
UNWIND $rows AS row
MATCH (c:Chunk {chunk_id: row.chunk_id})
UNWIND row.emotion AS name
MERGE (e:Emotion {name: name})
MERGE (c)-[:EVOKES_EMOTION]->(e)
"""

# Character 节点用 `key` 而不是 `name`：约束就是这么建的，而且人物名在不同
# 作品里会重（「宝玉」在《红楼梦》与别处不是同一个人），将来加作品前缀时
# key 是那个可组合的字段，name 是纯展示。
_CHARACTER_CYPHER = """
UNWIND $rows AS row
MATCH (c:Chunk {chunk_id: row.chunk_id})
WITH c, row WHERE row.character IS NOT NULL
MERGE (ch:Character {key: row.character})
ON CREATE SET ch.name = row.character
MERGE (ch)-[:FEATURES_CHARACTER]->(c)
"""

# Work 与 Concept 共用 HAS_CHUNK：都是「这个实体有哪些语料」。
# 关系名复用让 Cypher 的词汇表保持在个位数，代价是查询时得带上源标签。
_WORK_CYPHER = """
UNWIND $rows AS row
MATCH (c:Chunk {chunk_id: row.chunk_id})
WITH c, row WHERE row.work IS NOT NULL
MERGE (w:Work {title: row.work})
ON CREATE SET w.author = row.author, w.dynasty = row.dynasty
MERGE (w)-[:HAS_CHUNK]->(c)
"""

# 心理学语料没有意象/情感标注，却有 concept——它是心理侧写与文学典故之间
# 唯一的桥：侧写抽出的概念 → 该概念「由哪些意象佐证」→ 那些意象出现在
# 哪些诗词里。没有这条边，右栏的文学类比就只能靠向量碰运气。
_CONCEPT_CYPHER = """
UNWIND $rows AS row
MATCH (c:Chunk {chunk_id: row.chunk_id})
WITH c, row WHERE row.concept IS NOT NULL
MERGE (cp:Concept {name: row.concept})
MERGE (cp)-[:HAS_CHUNK]->(c)
"""


async def write_edges(edges: Sequence[GraphEdge]) -> int:
    """写语料里手写的聚合边。

    这些边是**关于文化惯例的断言**（落花 常与 无常 相连，权重 0.9），
    与「某条语料同时提到落花和无常」不是一回事，所以不自动从 chunk 推导。
    权重落在关系属性上，检索时可按权重排序。
    """
    if not edges:
        return 0
    await ensure_constraints()

    rows = [
        {
            "source": e.source,
            "target": e.target,
            "weight": e.weight,
            # 关系名与标签来自 loader 的白名单校验，此处只做拼接不做转义。
            "source_label": e.source_type,
            "target_label": e.target_type,
            "relation": e.relation,
        }
        for e in edges
    ]

    # 按 (源标签, 关系, 目标标签) 分组后逐组发一条语句——Cypher 的标签与
    # 关系名无法参数化，唯一安全的做法是让它们来自白名单并在代码里穷举。
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["source_label"]), str(row["relation"]), str(row["target_label"]))
        grouped.setdefault(key, []).append(row)

    written = 0
    async with graph_session() as session:
        for (source_label, relation, target_label), group in grouped.items():
            if not _allowed(source_label, relation, target_label):
                raise ValueError(f"未登记的图模式：(:{source_label})-[:{relation}]->(:{target_label})")
            # **MERGE 的键必须与约束建在同一个属性上**，否则写 chunk 时建的
            # `(:Imagery {name:'落花'})` 和这里建的 `(:Imagery {__key:'落花'})`
            # 会成为两个节点：「由意象找语料」那一跳永远查不到东西，
            # 而整张图看起来完全正常。所以键属性按标签查表，不写死。
            src_key = _KEY_PROPERTY[source_label]
            dst_key = _KEY_PROPERTY[target_label]
            stmt = (
                f"UNWIND $rows AS row "
                f"MERGE (a:{source_label} {{{src_key}: row.source}}) "
                f"MERGE (b:{target_label} {{{dst_key}: row.target}}) "
                f"MERGE (a)-[r:{relation}]->(b) SET r.weight = row.weight"
            )
            await session.run(stmt, rows=group)
            written += len(group)
    log.info("neo4j_edges_written", count=written)
    return written


#: 每种标签的「身份属性」。与 clients/neo4j_client.py 的约束一一对应：
#: Chunk.chunk_id / Work.title / Imagery.name / Emotion.name /
#: Character.key / Concept.name。改这里就必须同步改约束。
_KEY_PROPERTY: dict[str, str] = {
    "Chunk": "chunk_id",
    "Work": "title",
    "Imagery": "name",
    "Emotion": "name",
    "Character": "key",
    "Concept": "name",
}

#: 允许写入的图模式（源标签, 关系, 目标标签）。与 corpus/graph_edges.jsonl
#: 的实际取值一一对应；语料里出现新组合时必须先在这里登记。
_ALLOWED_PATTERNS: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("Imagery", "ASSOCIATED_WITH", "Emotion"),
        ("Imagery", "CO_OCCURS", "Imagery"),
        ("Character", "FEATURES_CHARACTER", "Imagery"),
        ("Concept", "EVIDENCED_BY", "Imagery"),
    }
)


def _allowed(source_label: str, relation: str, target_label: str) -> bool:
    return (source_label, relation, target_label) in _ALLOWED_PATTERNS


# ----------------------------------------------------------------------
# 查询：图检索路径（Node 5 用，kb-search 也用它证明图是活的）
# ----------------------------------------------------------------------

#: 由情绪找语料：Emotion ← ASSOCIATED_WITH ← Imagery ← MENTIONS_IMAGERY ← Chunk。
#:
#: **这是纯向量检索做不到的事。** 向量能回答「这段话和什么相似」，
#: 回答不了「哀伤在古典文学里通常借哪些意象说出来」。走到意象之后，
#: 还会顺着 CO_OCCURS 多跳一步——「落花」单独看太窄，它常与「春」「泪」
#: 同现，把同现意象的语料一并捞上来，召回才不至于被单个标签卡死。
#: 权重沿路径相乘，让「意象与情绪强相关」的语料排在前面。
_CHUNKS_BY_EMOTION_CYPHER = """
MATCH (e:Emotion {name: $emotion})<-[r:ASSOCIATED_WITH]-(i:Imagery)
OPTIONAL MATCH (i)-[co:CO_OCCURS]-(j:Imagery)
WITH e, i, r, co, j
MATCH (c:Chunk)-[:MENTIONS_IMAGERY]->(i)
WITH c, r.weight AS w
RETURN c.chunk_id AS chunk_id, c.snippet AS snippet, c.library AS library,
       max(w) AS weight
ORDER BY weight DESC, c.chunk_id
LIMIT $limit
"""

#: 由意象直接找语料。检索时簇的情绪标签落到意象上（比如「落花」既是意象
#: 也可能是用户直接给出的词），这条路比绕情绪更直接。
_CHUNKS_BY_IMAGERY_CYPHER = """
MATCH (c:Chunk)-[:MENTIONS_IMAGERY]->(i:Imagery {name: $imagery})
RETURN c.chunk_id AS chunk_id, c.snippet AS snippet, c.library AS library,
       $bias AS weight
ORDER BY c.chunk_id
LIMIT $limit
"""

#: 由心理概念找意象，再由意象找文学语料。右栏「文学类比」这一段的
#: 落点主要靠它——把「未完成事件」这种抽象概念翻成具体的物象。
#:
#: `ORDER BY` 的第二个键不能省：强度并列时 Neo4j 不保证返回顺序，
#: 同一次分析重跑两次会捞出不同的意象，下游的文学段跟着变——而这是
#: 一条只读查询，看起来「什么都没改」。
_IMAGERY_BY_CONCEPT_CYPHER = """
MATCH (cp:Concept {name: $concept})-[:EVIDENCED_BY]->(i:Imagery)
RETURN i.name AS name, count(*) AS strength
ORDER BY strength DESC, i.name
LIMIT $limit
"""

#: 模糊版：概念名不完全相等时（提示词抽出的说法与语料标注的用词差一两个字）
#: 仍能接上。**包含匹配往两个方向都试**——语料里是「未完成事件」而查询给的是
#: 「未完成的哀伤」，单向的 `CONTAINS` 接不住。
#:
#: 精度换召回是有意为之：这一跳的产物是「意象」这种短名词，错配的代价是
#: 多几张不贴切的证据卡，而漏配的代价是整段「文学类比」没有落点。
_IMAGERY_BY_CONCEPT_FUZZY_CYPHER = """
MATCH (cp:Concept)
WHERE cp.name CONTAINS $concept OR $concept CONTAINS cp.name
MATCH (cp)-[:EVIDENCED_BY]->(i:Imagery)
RETURN i.name AS name, count(*) AS strength
ORDER BY strength DESC, i.name
LIMIT $limit
"""


async def chunks_by_emotion(emotion: str, *, limit: int = 12) -> list[dict[str, object]]:
    """情绪 → 关联意象 → 语料。返回 [{chunk_id, snippet, library, weight}]。"""
    async with graph_session() as session:
        result = await session.run(_CHUNKS_BY_EMOTION_CYPHER, emotion=emotion, limit=limit)
        return [dict(record) async for record in result]


async def chunks_by_imagery(imagery: str, *, limit: int = 12, bias: float = 0.8) -> list[dict[str, object]]:
    async with graph_session() as session:
        result = await session.run(
            _CHUNKS_BY_IMAGERY_CYPHER, imagery=imagery, limit=limit, bias=bias
        )
        return [dict(record) async for record in result]


async def imagery_by_concept(
    concept: str, *, limit: int = 6, fuzzy: bool = False
) -> list[dict[str, object]]:
    """概念 → 意象。`fuzzy=True` 走包含匹配，用于提示词抽出的说法与语料
    标注用词不完全一致的情况（见 `_IMAGERY_BY_CONCEPT_FUZZY_CYPHER`）。"""
    if not concept.strip():
        return []
    cypher = _IMAGERY_BY_CONCEPT_FUZZY_CYPHER if fuzzy else _IMAGERY_BY_CONCEPT_CYPHER
    async with graph_session() as session:
        result = await session.run(cypher, concept=concept, limit=limit)
        return [dict(record) async for record in result]
