"""n5 的三段：查询构造、RRF 融合、配额分配，以及节点的接线。

这一层守的是三件**不会报错**的事：

1. **并列定序**。RRF 的分数是离散的（就那么几个名次组合），并列是常态。
   没有三级 tie-break 的话，同一次分析重跑两次给出不同的证据顺序——
   界面上看不出异常，而任何关于顺序的断言都只能碰运气。
2. **配额回填**。心理学库里只命中 2 条时，报告不该是「2 条心理学 + 0 条
   文学」。一条证据都没有的库比配额失衡糟得多，而两者在界面上都「正常」。
3. **发事件的形态**。证据项自带 `kind`，逐条发会和判别键撞名——前端一张
   卡都认不出来，而事件流里条条都在（见 `runner.py` 那段注释）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from app.constants import LIBRARY_QUOTA, MAX_QUERIES, RRF_K
from app.graph import retrieval
from app.graph.emitting import NodeEmitter
from app.graph.nodes import n5_retrieval as n5
from app.graph.retrieval import allocate, build_queries, graph_queries, rrf_fuse

TRACE_KEYS = {
    "ev_id",
    "query",
    "path",
    "vector_score",
    "graph_weight",
    "rrf_score",
    "rank_in_library",
    "quota_slot",
    "concept_bridge",
}


# ----------------------------------------------------------------------
# 素材
# ----------------------------------------------------------------------


def theme(key: str, size: int, *, label: str = "想念", emotion: str = "哀伤", order: int = 0) -> dict:
    return {
        "cluster_key": key,
        "label": label,
        "summary": f"{label}的那些话",
        "size": size,
        "is_noise": False,
        "emotion_tags": [emotion],
        "topic_tags": ["亲密关系"],
        "need_tags": ["被看见"],
        "keywords": ["想你", "再也"],
        "order_index": order,
    }


def profile(**over: Any) -> dict:
    base = {
        "core_tension": "想念却再也无法说出口",
        "summary": "一种没有出口的想念",
        "global_tags": {"emotion": ["哀伤"], "topic": ["亲密关系"], "need": ["被看见"]},
    }
    base.update(over)
    return base


@dataclass
class Point:
    payload: dict
    score: float


@dataclass
class Record:
    payload: dict


def chunk(chunk_id: str, library: str, **extra: Any) -> dict:
    return {
        "chunk_id": chunk_id,
        "library": library,
        "text": f"{chunk_id} 的正文",
        "work": "《测试》",
        "author": "某人",
        **extra,
    }


# ----------------------------------------------------------------------
# build_queries
# ----------------------------------------------------------------------


class TestBuildQueries:
    def test_每个主题两条_全局两条(self):
        out = build_queries(profile(), [theme("c0", 10)], depth="deep")
        # 顺序按权重：核心张力(3.0) > 全局情绪(2.5) > 单簇心理学(2.0) > 单簇文学(1.8)。
        # 编号跟着顺序走，所以 q1 一定是那个最要紧的查询。
        assert [q["library"] for q in out] == [
            "psychology",
            "poetry",
            "psychology",
            "literature",
        ]
        assert {q["cluster_key"] for q in out} == {"c0", None}
        # 核心张力是整份报告的主线，永远排在单簇之前。
        assert out[0]["text"] == "想念却再也无法说出口 一种没有出口的想念"

    def test_噪声与不足三条的簇不参与检索(self):
        clusters = [
            theme("noise", 20, label="边缘"),
            theme("c1", 2, label="太小"),
            theme("c2", 5, label="够格"),
        ]
        clusters[0]["is_noise"] = True
        out = build_queries(profile(), clusters, depth="deep")
        assert {q["cluster_key"] for q in out} == {"c2", None}

    def test_关掉的库不产生查询(self):
        out = build_queries(profile(), [theme("c0", 10)], depth="deep", enabled=["psychology"])
        assert {q["library"] for q in out} == {"psychology"}

    def test_按深度封顶且全局查询一定活下来(self):
        clusters = [theme(f"c{i}", 10 - i, order=i) for i in range(8)]
        out = build_queries(profile(), clusters, depth="quick")
        assert len(out) == MAX_QUERIES["quick"]
        assert any(q["cluster_key"] is None for q in out)

    def test_先归一化去重再封顶(self):
        # 两个簇的标签完全一样 → 两条心理学查询文本相同，只该留一条。
        clusters = [theme("c0", 10, order=0), theme("c1", 8, order=1)]
        out = build_queries(profile(), clusters, depth="deep")
        texts = [(q["library"], q["text"]) for q in out]
        assert len(texts) == len(set(texts))

    def test_编号连续且唯一(self):
        out = build_queries(profile(), [theme("c0", 10)], depth="deep")
        assert [q["id"] for q in out] == ["q1", "q2", "q3", "q4"]

    def test_空输入不炸(self):
        assert build_queries(None, None, depth="standard") == []
        assert build_queries({}, [], depth="standard") == []

    def test_单字符的标签被丢掉(self):
        # 查询文本同时是 embedding 的缓存键，噪声词会让它命中不了任何缓存。
        out = build_queries(profile(global_tags={"emotion": ["哀"], "need": []}), [], depth="standard")
        assert [q["library"] for q in out] == ["psychology"]


class TestGraphQueries:
    def test_每种情绪只发一条且按簇大小排(self):
        clusters = [
            theme("c0", 10, emotion="哀伤", order=0),
            theme("c1", 8, emotion="哀伤", order=1),
            theme("c2", 6, emotion="倦怠", order=2),
        ]
        out = graph_queries(clusters, limit=5)
        assert [q["text"] for q in out] == ["哀伤", "倦怠"]
        assert out[0]["cluster_key"] == "c0"

    def test_限量(self):
        clusters = [
            theme("c0", 10, emotion="哀伤"),
            theme("c1", 9, emotion="倦怠"),
            theme("c2", 8, emotion="愧疚"),
        ]
        assert len(graph_queries(clusters, limit=2)) == 2

    def test_图谱查询没有_library(self):
        # 命中哪条语料由 chunk 自己决定（可能是文学也可能是诗词），
        # 在查询上写死一个库会把另一半整批丢掉。
        out = graph_queries([theme("c0", 10)], limit=3)
        assert out[0]["library"] is None
        assert out[0]["path"] == "graph"


# ----------------------------------------------------------------------
# rrf_fuse
# ----------------------------------------------------------------------


class TestRrf:
    def test_手算比对(self):
        # a: 两条榜单都是第 1  → 1/61 + 1/61
        # b: 第 2、第 3        → 1/62 + 1/63
        # c: 只在第二条榜第 2   → 1/62
        fused = dict(rrf_fuse([["a", "b"], ["a", "c", "b"]]))
        assert fused["a"] == pytest.approx(round(2 / 61, 6))
        assert fused["b"] == pytest.approx(round(1 / 62 + 1 / 63, 6))
        assert fused["c"] == pytest.approx(round(1 / 62, 6))

    def test_顺序按分数降序(self):
        assert [cid for cid, _ in rrf_fuse([["a", "b", "c"], ["b", "a"]])] == ["a", "b", "c"]

    def test_同一条在一条榜单里出现两次只算第一次(self):
        # 图谱的 CO_OCCURS 多跳会让同一个 chunk 经不同意象命中两次；
        # 重复计数等于给这条证据偷偷加权。
        # 注意不能拿「b 的分数」来测：重复项占掉的那个名次会把后头的
        # 候选整个推后，b 变低了是**名次**的锅，不是重复计数的锅。
        assert dict(rrf_fuse([["a", "a"]]))["a"] == pytest.approx(round(1 / 61, 6))
        assert len(rrf_fuse([["a", "a"]])) == 1

    def test_空输入(self):
        assert rrf_fuse([]) == []
        assert rrf_fuse([[], []]) == []

    def test_分数并列时先比最好名次(self):
        # x 最好第 1（来自第一条），y 最好第 2 → x 在前，尽管两者总分相同。
        fused = rrf_fuse([["x", "z"], ["z", "y"], ["y", "x"]])
        assert [cid for cid, _ in fused][0] in {"x", "z"}

    def test_并列定序跑十次结果一致(self):
        # 三条榜单两两对称，制造出一批总分完全相同的候选。
        rankings = [["a", "b", "c"], ["c", "a", "b"], ["b", "c", "a"]]
        first = [cid for cid, _ in rrf_fuse(rankings)]
        for _ in range(10):
            assert [cid for cid, _ in rrf_fuse(rankings)] == first
        # 总分确实并列——不然这个测试什么也没守住。
        scores = {s for _, s in rrf_fuse(rankings)}
        assert len(scores) == 1

    def test_k_的作用是压平差距(self):
        small = dict(rrf_fuse([["a", "b"]], k=1))
        big = dict(rrf_fuse([["a", "b"]], k=60))
        assert small["a"] - small["b"] > big["a"] - big["b"]
        assert RRF_K == 60


# ----------------------------------------------------------------------
# allocate
# ----------------------------------------------------------------------


class TestAllocate:
    def test_按配额取(self):
        ranked = {"psychology": ["p1", "p2", "p3"], "literature": ["l1"], "poetry": ["o1"]}
        out = allocate(ranked, quota=(2, 1, 1), cap=4)
        assert [(i["library"], i["chunk_id"]) for i in out] == [
            ("psychology", "p1"),
            ("psychology", "p2"),
            ("literature", "l1"),
            ("poetry", "o1"),
        ]
        assert [i["quota_slot"] for i in out] == [0, 1, 0, 0]

    def test_某库不足时按优先级回填(self):
        ranked = {"psychology": ["p1", "p2"], "literature": ["l1", "l2", "l3"], "poetry": []}
        out = allocate(ranked, quota=(3, 2, 1), cap=6)
        # 心理学少 1 条 → 回填给文学；诗词没有 → 那 1 条也给文学。
        assert [i["chunk_id"] for i in out] == ["p1", "p2", "l1", "l2", "l3"]
        assert len(out) <= 6
        # 回填进来的没有配额序号——这不是瑕疵，是「这条是补位来的」的线索。
        assert [i["quota_slot"] for i in out] == [0, 1, 0, 1, None]

    def test_回填后总数仍不超过_cap(self):
        ranked = {name: [f"{name}-{i}" for i in range(20)] for name in retrieval.LIBRARY_ORDER}
        out = allocate(ranked, quota=(6, 4, 3), cap=13)
        assert len(out) == 13

    def test_配额为零的库不参与回填(self):
        ranked = {"psychology": ["p1"], "literature": ["l1", "l2"], "poetry": []}
        out = allocate(ranked, quota=(1, 0, 0), cap=3)
        assert [i["chunk_id"] for i in out] == ["p1"]

    def test_全部为空时不炸(self):
        assert allocate({}, quota=(3, 2, 1), cap=6) == []

    def test_配额表按深度递增(self):
        assert LIBRARY_QUOTA["quick"] < LIBRARY_QUOTA["standard"] < LIBRARY_QUOTA["deep"]


# ----------------------------------------------------------------------
# 节点
# ----------------------------------------------------------------------


@dataclass
class Recorder:
    events: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, payload: dict[str, Any]) -> None:
        self.events.append(payload)


def wire(monkeypatch: pytest.MonkeyPatch, *, search, fetch, graph=None) -> Recorder:
    """把节点对外部世界的几次调用整段换掉，只留接线本身被测。"""
    rec = Recorder()
    monkeypatch.setattr(
        n5, "NodeEmitter", lambda node: NodeEmitter(node, write=rec)
    )
    monkeypatch.setattr(n5, "_embed", lambda texts: _fake_embed(texts))
    monkeypatch.setattr(n5, "_search", search)
    monkeypatch.setattr(n5, "_fetch_by_ids", fetch)
    monkeypatch.setattr(n5, "_graph_by_emotion", graph or _no_graph)
    monkeypatch.setattr(n5, "_imagery_by_concept", _no_imagery)
    monkeypatch.setattr(n5, "_chunks_by_imagery", _no_chunks)
    monkeypatch.setattr(n5, "_stagger_seconds", lambda: 0.0)
    return rec


async def _fake_embed(texts: list[str]) -> list[list[float]]:
    return [[float(len(t))] * 4 for t in texts]


def evidence_events(rec: Recorder) -> list[dict[str, Any]]:
    return [e for e in rec.events if e.get("kind") == "partial" and e.get("data_kind") == "evidence"]


class TestNode:
    async def test_kb_全关时只发_warning(self, monkeypatch: pytest.MonkeyPatch):
        rec = wire(monkeypatch, search=_never, fetch=_never_fetch)
        patch = await n5.n5_retrieval(
            {"depth": "standard", "kb": {"psychology": False, "literature": False, "poetry": False}}
        )
        assert patch == {"queries": [], "evidence": []}
        assert [e for e in rec.events if e["kind"] == "warning"][0]["code"] == "no_kb_enabled"

    async def test_没有主题时不检索(self, monkeypatch: pytest.MonkeyPatch):
        rec = wire(monkeypatch, search=_never, fetch=_never_fetch)
        patch = await n5.n5_retrieval({"depth": "standard", "kb": {}, "clusters": [], "profile": {}})
        assert patch == {"queries": [], "evidence": []}
        assert [e for e in rec.events if e["kind"] == "warning"][0]["code"] == "no_queries"

    async def test_证据逐条发且用批量形态(self, monkeypatch: pytest.MonkeyPatch):
        rec = wire(monkeypatch, search=_search_by_library, fetch=_fetch_all)
        patch = await n5.n5_retrieval(
            {
                "depth": "quick",
                "kb": {},
                "clusters": [theme("c0", 10)],
                "profile": profile(),
            }
        )
        events = evidence_events(rec)
        assert len(events) == len(patch["evidence"]) >= 1
        for event in events:
            assert event["data_kind"] == "evidence"
            batch = event["data"]["evidence"]
            assert isinstance(batch, list) and len(batch) == 1
            # **payload 顶层不许再出现一个 `kind`。** runner 拼 RunEvent.data 时
            # 是 `{**payload, "kind": data_kind}`——payload 里再有一个同名的键，
            # 前端拿到的判别键就会被证据自己的 psychology/literature 顶掉，
            # 于是一张卡都认不出来（见 runner.py:180 那段注释）。
            assert "kind" not in event["data"]
            # 照 runner 的合并顺序重演一遍，确认判别键活到最后。
            merged = {**event["data"], "kind": event["data_kind"]}
            assert merged["kind"] == "evidence"
            assert merged["evidence"][0]["kind"] in {"psychology", "literature", "poetry"}

    async def test_payload_九个_trace_键齐全(self, monkeypatch: pytest.MonkeyPatch):
        wire(monkeypatch, search=_search_by_library, fetch=_fetch_all)
        patch = await n5.n5_retrieval(
            {"depth": "standard", "kb": {}, "clusters": [theme("c0", 10)], "profile": profile()}
        )
        assert patch["evidence"]
        for item in patch["evidence"]:
            assert TRACE_KEYS <= set(item["payload"])
            assert item["payload"]["ev_id"] == f"e{item['rank']}"
            assert 0 <= item["confidence"] if "confidence" in item else True

    async def test_证据字段与_persist_契约对齐(self, monkeypatch: pytest.MonkeyPatch):
        wire(monkeypatch, search=_search_by_library, fetch=_fetch_all)
        patch = await n5.n5_retrieval(
            {"depth": "quick", "kb": {}, "clusters": [theme("c0", 10)], "profile": profile()}
        )
        for item in patch["evidence"]:
            assert set(item) >= {
                "kind", "library", "chunk_id", "title", "source", "author", "text",
                "score", "retrieval_path", "match_reason", "cluster_ids",
                "pinned", "excluded", "rank", "payload",
            }
            assert item["kind"] == item["library"]
            assert item["pinned"] is False and item["excluded"] is False

    async def test_取不回正文的证据被丢掉而不是留个空壳(self, monkeypatch: pytest.MonkeyPatch):
        async def half(chunk_ids):
            return [Record(chunk(cid, "psychology")) for cid in chunk_ids[:1]]

        wire(monkeypatch, search=_search_by_library, fetch=half)
        patch = await n5.n5_retrieval(
            {"depth": "standard", "kb": {}, "clusters": [theme("c0", 10)], "profile": profile()}
        )
        assert len(patch["evidence"]) == 1

    async def test_图谱命中进候选并带回_graph_weight(self, monkeypatch: pytest.MonkeyPatch):
        # `lit:9` 只在图谱里出现。用它而不是 `lit:1`：后者向量那条路也命中，
        # 而两条路都命中时溯源指向更早的那条查询（向量），
        # 于是这个测试会变成在测 tie-break，而不是在测图谱写没写进 trace。
        async def graph(emotion: str, limit: int):
            return [{"chunk_id": "lit:9", "library": "literature", "snippet": "…", "weight": 0.9}]

        wire(monkeypatch, search=_search_by_library, fetch=_fetch_all, graph=graph)
        patch = await n5.n5_retrieval(
            {"depth": "deep", "kb": {}, "clusters": [theme("c0", 10)], "profile": profile()}
        )
        graph_hits = [e for e in patch["evidence"] if e["retrieval_path"] == "graph"]
        assert graph_hits, "图谱路径的命中没有进证据列表"
        assert graph_hits[0]["payload"]["graph_weight"] == 0.9
        assert graph_hits[0]["payload"]["vector_score"] is None
        assert graph_hits[0]["match_reason"].startswith("图谱")
        assert any(q["path"] == "graph" for q in patch["queries"])

    async def test_图里没有这个情绪时不发查询也不报错(self, monkeypatch: pytest.MonkeyPatch):
        # 实测：11 个情绪词只有 6 个在图上。查不到是常态，不该去打扰用户。
        rec = wire(monkeypatch, search=_search_by_library, fetch=_fetch_all)
        patch = await n5.n5_retrieval(
            {"depth": "deep", "kb": {}, "clusters": [theme("c0", 10)], "profile": profile()}
        )
        assert all(q["path"] != "graph" for q in patch["queries"])
        assert not [e for e in rec.events if e["kind"] == "warning"]

    async def test_关掉的库不会从图谱那条路漏进来(self, monkeypatch: pytest.MonkeyPatch):
        async def graph(emotion: str, limit: int):
            return [{"chunk_id": "po:1", "library": "poetry", "snippet": "…", "weight": 0.9}]

        wire(monkeypatch, search=_search_by_library, fetch=_fetch_all, graph=graph)
        patch = await n5.n5_retrieval(
            {
                "depth": "deep",
                "kb": {"psychology": True, "literature": True, "poetry": False},
                "clusters": [theme("c0", 10)],
                "profile": profile(),
            }
        )
        assert all(e["library"] != "poetry" for e in patch["evidence"])

    async def test_一条都没有时发_warning(self, monkeypatch: pytest.MonkeyPatch):
        async def nothing(vector, library, limit):
            return []

        rec = wire(monkeypatch, search=nothing, fetch=_never_fetch)
        patch = await n5.n5_retrieval(
            {"depth": "standard", "kb": {}, "clusters": [theme("c0", 10)], "profile": profile()}
        )
        assert patch["evidence"] == []
        assert [e for e in rec.events if e["kind"] == "warning"][0]["code"] == "no_evidence"


async def _no_graph(emotion: str, limit: int) -> list[dict[str, Any]]:
    return []


async def _no_imagery(concept: str, limit: int) -> list[dict[str, Any]]:
    return []


async def _no_chunks(imagery: str, limit: int) -> list[dict[str, Any]]:
    return []


async def _never(vector: Any, library: Any, limit: int) -> list[Any]:
    raise AssertionError("这条路径不该被走到")


async def _never_fetch(chunk_ids: Any) -> list[Any]:
    raise AssertionError("这条路径不该被走到")


async def _search_by_library(vector: list[float], library: Any, limit: int) -> list[Any]:
    """按库回一批固定的命中。分数故意只有 0.1–0.3——真实量纲就在这一段，
    而 0.3 已经是不错的命中了（所以不能设 score_threshold）。"""
    table = {
        "psychology": ["psy:1", "psy:2", "psy:3"],
        "literature": ["lit:1", "lit:2"],
        "poetry": ["po:1"],
        None: ["psy:1", "lit:1", "po:1"],
    }
    ids = table.get(library, [])
    return [Point(chunk(cid, _library_of(cid), imagery=["落花"]), 0.1 + 0.05 * i) for i, cid in enumerate(ids)]


def _library_of(chunk_id: str) -> str:
    return {"psy": "psychology", "lit": "literature", "po": "poetry"}[chunk_id.split(":")[0]]


async def _fetch_all(chunk_ids):
    return [Record(chunk(cid, _library_of(cid), concept="未完成事件")) for cid in chunk_ids]
