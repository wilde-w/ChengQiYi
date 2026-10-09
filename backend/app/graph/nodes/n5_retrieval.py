"""Node 5 — 多路检索（向量 + 图谱）。

两条路径各自解决对方解决不了的事：

  路径 A（向量）  按簇的标签/情绪/关键词问「哪段话和这个主题像」
  路径 B（图谱）  按情绪问「这种情绪在古典文学里借哪些意象说出来」——
                  向量能回答相似度，回答不了这个问题

融合用 RRF，**只在库内融合**。理由是实测的量纲差：本机向量分落在
0.06–0.29，图谱权重落在 0.7–0.95，任何加权求和都得先拍一个归一化系数，
而那个系数没有任何依据。名次没有量纲。

融合完还要**按库配额削**。939 个点里绝大多数是文学与诗词，纯按分数排
会把 6 条心理学证据挤出榜单——而心理学才是结论的支柱。配额与回填见
`graph/retrieval.py:allocate`。

图谱那条路的最后一跳必须是 `fetch_by_ids()`：Neo4j 里只有 60 字 snippet，
正文在 Qdrant。少了这一步，证据卡上就只剩摘要，用户点开看不到原文。

## 逐条发事件

每挑出一条证据就发一帧 `partial("evidence", {"evidence": [item]})`。
**批量形态是硬要求**（见 `runner.py` 里那段注释）：证据项自带一个
`kind` 字段，逐条发的话它会和判别键撞名，前端一张卡都认不出来。
间隔 `EVIDENCE_STAGGER_MS` 只在 demo 模式加——真实检索本来就是一串
连续到达的事件，仪式感是给演示看的，不是给用户等的。
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from app.config import get_settings
from app.constants import (
    CONCEPT_BRIDGE_LIMIT,
    EVIDENCE_STAGGER_MS,
    EVIDENCE_TEXT_CAP,
    GRAPH_TOP_K,
    LIBRARY_QUOTA,
    MAX_CONCEPT_BRIDGES,
    MAX_GRAPH_QUERIES,
    QUERY_TOP_K,
    Library,
    RetrievalPath,
)
from app.graph.emitting import NodeEmitter
from app.graph.retrieval import (
    LIBRARY_ORDER,
    allocate,
    build_queries,
    rrf_fuse,
)
from app.graph.retrieval import graph_queries as build_graph_queries
from app.graph.state import AnalysisState
from app.logging_conf import get_logger

log = get_logger(__name__)

NODE = "n5_retrieval"

#: 证据正文截断到多少字。payload 里存的是全文（有的上千字），
#: 证据卡上放全文等于让人在卡片里读论文。
TEXT_CAP = EVIDENCE_TEXT_CAP


async def n5_retrieval(state: AnalysisState) -> dict[str, Any]:
    emitter = NodeEmitter(NODE)
    depth = str(state.get("depth") or "standard")
    kb = state.get("kb") or {}
    enabled = [name for name in LIBRARY_ORDER if kb.get(name, True)]

    if not enabled:
        emitter.warn("no_kb_enabled", "未启用任何知识库，本次不做检索")
        emitter.set_progress(1.0)
        return {"queries": [], "evidence": []}

    queries = build_queries(
        state.get("profile"), state.get("clusters"), depth=depth, enabled=enabled
    )
    if not queries:
        emitter.warn("no_queries", "没有可用于检索的主题或侧写，跳过检索")
        emitter.set_progress(1.0)
        return {"queries": [], "evidence": []}

    emitter.milestone(f"正在检索 {len(queries)} 条查询…")
    collector = _Collector(queries, enabled=enabled)
    await _vector_pass(collector, queries, emitter)
    await _graph_pass(collector, state.get("clusters"), emitter)

    fused = collector.fused()
    quota = LIBRARY_QUOTA.get(depth, LIBRARY_QUOTA["standard"])
    selected = allocate(
        {name: [cid for cid, _ in fused.get(name, [])] for name in LIBRARY_ORDER},
        quota=quota,
        cap=sum(quota),
    )

    evidence = await _materialize(collector, fused, selected, emitter)
    if not evidence:
        emitter.warn("no_evidence", "知识库里没有匹配到可用证据，报告将只基于评论本身")

    emitter.set_progress(1.0)
    return {"queries": queries, "evidence": evidence}


# ----------------------------------------------------------------------
# 采集：两条路径往同一个篮子里放「有序候选列表」
# ----------------------------------------------------------------------


class _Collector:
    """把两条路径的候选攒起来，最后统一融合。

    每个库一份「有序列表的列表」——RRF 的输入单位是**一条查询的一个有序
    结果**，而不是一条合并过的榜单。合并是 RRF 自己的事。

    `enabled` 从入口一路带到这里：图谱命中的 chunk 自带 library，不受
    查询上的 library 约束，不在这里再挡一道的话，被用户关掉的库照样会
    从图谱那条路漏进证据列表。
    """

    __slots__ = ("_queries", "enabled", "rankings", "hits", "payloads")

    def __init__(self, queries: list[dict[str, Any]], *, enabled: Sequence[str]) -> None:
        self._queries = queries
        self.enabled = list(enabled)
        self.rankings: dict[str, list[list[str]]] = {name: [] for name in self.enabled}
        #: chunk_id → 最好的一次命中（名次最靠前，并列时取更早的查询）
        self.hits: dict[str, dict[str, Any]] = {}
        self.payloads: dict[str, dict[str, Any]] = {}

    def add_query(self, query: dict[str, Any]) -> str:
        """追加一条运行期才产生的查询（图谱/概念桥），返回它的 id。

        图谱查询不在 `build_queries` 里：它问的不是「哪段话像」，而是
        「这个情绪在图上连到哪些语料」，文本就是情绪词本身。硬塞进那个
        纯函数只是为了形状统一，反而会让它同时承担两种语义。
        """
        query_id = f"q{len(self._queries) + 1}"
        query["id"] = query_id
        self._queries.append(query)
        return query_id

    def add_ranking(
        self,
        library: str,
        ranked: Sequence[tuple[str, dict[str, Any]]],
        *,
        query_id: str,
        path: str,
        cluster_key: Any = None,
        concept_bridge: str | None = None,
    ) -> None:
        """记下一条查询在一个库内的有序命中。

        `ranked` 是 `[(chunk_id, 附加信息)]`，顺序即名次。附加信息里的
        `vector_score` / `graph_weight` 只用于展示与溯源——融合看的是名次。
        """
        if library not in self.rankings or not ranked:
            return
        self.rankings[library].append([cid for cid, _ in ranked])
        for rank, (chunk_id, extra) in enumerate(ranked, start=1):
            self._remember(
                chunk_id,
                rank=rank,
                query_id=query_id,
                path=path,
                cluster_key=cluster_key,
                concept_bridge=concept_bridge,
                **extra,
            )

    def _remember(
        self,
        chunk_id: str,
        *,
        rank: int,
        query_id: str,
        path: str,
        cluster_key: Any,
        concept_bridge: str | None,
        **extra: Any,
    ) -> None:
        previous = self.hits.get(chunk_id)
        # 并列时取更早的查询：`q3` 与 `q1` 都排第一，那条证据是随 `q1`
        # 被检索到的——溯源要指向第一次让它出现的那条。
        if previous is not None and (previous["rank"], previous["query"]) <= (rank, query_id):
            return
        self.hits[chunk_id] = {
            "rank": rank,
            "query": query_id,
            "path": path,
            "cluster_key": cluster_key,
            "concept_bridge": concept_bridge,
            **extra,
        }

    def fused(self) -> dict[str, list[tuple[str, float]]]:
        return {name: rrf_fuse(self.rankings[name]) for name in self.enabled}

    def top_concepts(self, *, limit: int) -> list[str]:
        """按名次取最靠前的若干个心理学概念，供概念桥使用。"""
        ordered = sorted(
            (hit for hit in self.hits.items() if hit[1]["path"] == RetrievalPath.VECTOR),
            key=lambda item: (item[1]["rank"], item[1]["query"]),
        )
        out: list[str] = []
        for chunk_id, _ in ordered:
            payload = self.payloads.get(chunk_id, {})
            if str(payload.get("library") or "") != Library.PSYCHOLOGY:
                continue
            concept = str(payload.get("concept") or "").strip()
            if concept and concept not in out:
                out.append(concept)
            if len(out) >= limit:
                break
        return out


async def _vector_pass(
    collector: _Collector,
    queries: list[dict[str, Any]],
    emitter: NodeEmitter,
) -> None:
    """路径 A：每条查询一次向量检索，按 `library` 过滤。"""
    vectors = await _embed([str(q["text"]) for q in queries])
    for index, (query, vector) in enumerate(zip(queries, vectors)):
        points = await _search(vector, query.get("library"), QUERY_TOP_K)
        # 一条 `library=None` 的查询会一次问遍三个库，而 RRF 只在库内融合，
        # 所以先按候选自己的库拆开——拆的依据是 payload 里的 library，
        # 不是查询上写的那个（图谱命中尤其如此）。
        buckets: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for point in points:
            payload = dict(getattr(point, "payload", None) or {})
            chunk_id = str(payload.get("chunk_id") or "")
            library = str(payload.get("library") or "")
            if not chunk_id or library not in collector.rankings:
                continue
            collector.payloads.setdefault(chunk_id, payload)
            buckets.setdefault(library, []).append(
                (
                    chunk_id,
                    {"vector_score": round(float(getattr(point, "score", 0.0) or 0.0), 4)},
                )
            )
        for library, ranked in buckets.items():
            collector.add_ranking(
                library,
                ranked,
                query_id=str(query["id"]),
                path=str(RetrievalPath.VECTOR),
                cluster_key=query.get("cluster_key"),
            )
        emitter.set_progress(0.1 + 0.5 * (index + 1) / max(1, len(queries)))


async def _graph_pass(
    collector: _Collector,
    clusters: Any,
    emitter: NodeEmitter,
) -> None:
    """路径 B：情绪 → 意象 → 语料，以及概念桥。

    图上 `Emotion` 只有 24 个节点，而本机情绪词表里的 11 个词只有 6 个在
    图里——查不到是常态，不是故障。所以**空召回只进日志不进 warning**：
    为每一个「图里没这个词」的提示去打扰用户，会让他习惯性忽略真正
    要紧的那条。
    """
    skipped = 0
    for query in build_graph_queries(clusters, limit=MAX_GRAPH_QUERIES):
        rows = await _graph_by_emotion(str(query["text"]), GRAPH_TOP_K)
        if not rows:
            skipped += 1
            continue
        query_id = collector.add_query(query)
        _absorb(collector, rows, query_id=query_id, cluster_key=query.get("cluster_key"))
    if skipped:
        log.info("n5_graph_emotions_missing", skipped=skipped)

    for concept in collector.top_concepts(limit=MAX_CONCEPT_BRIDGES):
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for imagery in await _imagery_by_concept(concept, CONCEPT_BRIDGE_LIMIT):
            name = str(imagery.get("name") or "")
            if not name:
                continue
            for row in await _chunks_by_imagery(name, GRAPH_TOP_K):
                chunk_id = str(row.get("chunk_id") or "")
                if chunk_id and chunk_id not in seen:
                    seen.add(chunk_id)
                    rows.append(row)
        if not rows:
            continue
        query_id = collector.add_query(
            {
                "text": concept,
                "library": None,
                "cluster_key": None,
                "path": str(RetrievalPath.GRAPH),
                "kind": str(Library.LITERATURE),
                "weight": 1.0,
            }
        )
        _absorb(collector, rows, query_id=query_id, concept_bridge=concept)
    emitter.set_progress(0.75)


def _absorb(
    collector: _Collector,
    rows: Sequence[dict[str, Any]],
    *,
    query_id: str,
    cluster_key: Any = None,
    concept_bridge: str | None = None,
) -> None:
    """把图谱返回的行按库分桶后喂给采集器。

    库别取自行自己的 `library` 列（两条 Cypher 都返回了它），而不是回问
    Qdrant：概念桥捞上来的语料很可能一条都不在向量命中里，那时 payload
    表里根本没有它，靠回问会把这些证据整批丢掉——而它们恰恰是图谱路径
    独有的产出。
    """
    buckets: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for row in rows:
        chunk_id = str(row.get("chunk_id") or "")
        library = str(row.get("library") or "")
        if not chunk_id or library not in collector.rankings:
            continue
        buckets.setdefault(library, []).append(
            (chunk_id, {"graph_weight": round(float(row.get("weight") or 0.0), 4)})
        )
    for library, ranked in buckets.items():
        collector.add_ranking(
            library,
            ranked,
            query_id=query_id,
            path=str(RetrievalPath.GRAPH),
            cluster_key=cluster_key,
            concept_bridge=concept_bridge,
        )


# ----------------------------------------------------------------------
# 落成证据
# ----------------------------------------------------------------------


async def _materialize(
    collector: _Collector,
    fused: dict[str, list[tuple[str, float]]],
    selected: Sequence[dict[str, Any]],
    emitter: NodeEmitter,
) -> list[dict[str, Any]]:
    """取回全文，组装成证据卡，逐条发。

    正文一律回 Qdrant 取：图谱与向量命中的都可能是 payload 不全的副本，
    而证据卡上要显示原文。取不回来的**直接丢弃而不是塞个空壳**——
    一张点开只有标题的卡片比少一张卡片更让人困惑。
    """
    if not selected:
        return []
    records = await _fetch_by_ids([str(item["chunk_id"]) for item in selected])
    full = {
        str((getattr(record, "payload", None) or {}).get("chunk_id") or ""): dict(
            getattr(record, "payload", None) or {}
        )
        for record in records
    }
    scores = {cid: score for ranked in fused.values() for cid, score in ranked}

    stagger = _stagger_seconds()
    evidence: list[dict[str, Any]] = []

    for item in selected:
        chunk_id = str(item["chunk_id"])
        fetched = full.get(chunk_id)
        if not fetched:
            log.info("n5_evidence_missing_text", chunk_id=chunk_id)
            continue
        payload = {**collector.payloads.get(chunk_id, {}), **fetched}
        rank = len(evidence) + 1
        evidence.append(_evidence(payload, item, collector, scores, rank))
        emitter.partial("evidence", {"evidence": [evidence[-1]]})
        if stagger:
            await asyncio.sleep(stagger)

    return evidence


def _evidence(
    payload: dict[str, Any],
    item: dict[str, Any],
    collector: _Collector,
    scores: dict[str, float],
    rank: int,
) -> dict[str, Any]:
    chunk_id = str(item["chunk_id"])
    hit = collector.hits.get(chunk_id, {})
    library = str(item["library"])
    rrf = round(float(scores.get(chunk_id, 0.0)), 6)

    # 9 个 trace 键**平铺**在 payload 里，不另起一层 `trace`：
    # 证据卡展开时是逐行读的，多一层嵌套只是多一次解引用；
    # 而且它们本来就该和 chunk 自己的元数据一起被看到。
    trace = {
        "ev_id": f"e{rank}",
        "query": hit.get("query"),
        "path": str(hit.get("path") or RetrievalPath.VECTOR),
        "vector_score": hit.get("vector_score"),
        "graph_weight": hit.get("graph_weight"),
        "rrf_score": rrf,
        "rank_in_library": int(item["rank_in_library"]) + 1,
        "quota_slot": item.get("quota_slot"),
        "concept_bridge": hit.get("concept_bridge"),
    }
    cluster_key = hit.get("cluster_key")

    return {
        "kind": library,
        "library": library,
        "chunk_id": chunk_id,
        "title": _first(payload, "work", "title", "chapter"),
        "source": _source(payload),
        "author": payload.get("author"),
        "text": str(payload.get("text") or "")[:TEXT_CAP],
        "score": rrf,
        "retrieval_path": trace["path"],
        "match_reason": _reason(trace, payload),
        # 证据服务哪个主题：存 cluster_key 而不是数据库主键，理由同
        # `evidence_brief` 里的 id——主键在节点跑的时候还不存在。
        "cluster_ids": [str(cluster_key)] if cluster_key else [],
        "pinned": False,
        "excluded": False,
        "rank": rank,
        "payload": {**payload, **trace},
    }


def _reason(trace: dict[str, Any], payload: dict[str, Any]) -> str:
    """一句话说清「它怎么被选中的」。证据卡上最该回答的就是这个。"""
    if trace.get("concept_bridge"):
        return f"概念桥：{trace['concept_bridge']} → 意象"
    if trace["path"] == RetrievalPath.GRAPH:
        return "图谱：情绪 → 意象 → 语料"
    score = trace.get("vector_score")
    if score is None:
        return "向量检索"
    imagery = payload.get("imagery")
    if isinstance(imagery, list) and imagery:
        return f"向量相似 {score:.2f}｜意象：{'、'.join(str(i) for i in imagery[:3])}"
    return f"向量相似 {score:.2f}"


def _first(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _source(payload: dict[str, Any]) -> str | None:
    """出处：能拼成「作者·作品」就拼，否则退回作品名或来源。"""
    author = payload.get("author")
    work = payload.get("work") or payload.get("title")
    if author and work:
        return f"{author}·{work}"
    return str(work or payload.get("source") or payload.get("origin") or "") or None


def _stagger_seconds() -> float:
    return EVIDENCE_STAGGER_MS / 1000 if get_settings().is_demo else 0.0


# ----------------------------------------------------------------------
# 对外部世界的几次调用。包成函数是为了单测能整段替换掉。
# ----------------------------------------------------------------------


async def _embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    from app.providers.factory import get_embedding_model

    return list(await get_embedding_model().embed(texts))


async def _search(vector: list[float], library: Any, limit: int) -> list[Any]:
    from app.kb.qdrant_index import search

    # **不设 score_threshold**：本机实测向量分只有 0.06–0.29，
    # 按常识设个 0.5 会把全部命中一刀切光，而界面上只会看到「没有证据」。
    return await search(vector, library=library, limit=limit)


async def _fetch_by_ids(chunk_ids: Sequence[str]) -> list[Any]:
    from app.kb.qdrant_index import fetch_by_ids

    return await fetch_by_ids(chunk_ids)


async def _graph_by_emotion(emotion: str, limit: int) -> list[dict[str, Any]]:
    from app.kb.neo4j_index import chunks_by_emotion

    return await chunks_by_emotion(emotion, limit=limit)


async def _imagery_by_concept(concept: str, limit: int) -> list[dict[str, Any]]:
    from app.kb.neo4j_index import imagery_by_concept

    return await imagery_by_concept(concept, limit=limit, fuzzy=True)


async def _chunks_by_imagery(imagery: str, limit: int) -> list[dict[str, Any]]:
    from app.kb.neo4j_index import chunks_by_imagery

    return await chunks_by_imagery(imagery, limit=limit)


__all__ = ["NODE", "n5_retrieval"]
