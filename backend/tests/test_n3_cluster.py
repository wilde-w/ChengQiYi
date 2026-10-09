"""评论聚类：退化链的每一个分支。

**纯离线，向量自造。** 这个文件不碰 PG / Qdrant / Redis，也不调模型——
`cluster_only` 本来就是纯函数，那正是把它从 `embed_comments` 里拆出来的
理由（见 n3_cluster 的模块 docstring）。

守的性质不是「分得好不好看」，而是**每一条分支都真的走得到、且走在它该在
的时候**：mock 嵌入没有语义几何，真实嵌入也不是每次都能过主题闸门，
这两条路都会在生产里出现，所以两条都得有测试压着。

向量用 8 维正交基构造：三个中心两两余弦为 0，簇内因为抖动而落在 0.85 附近。
抖动这个数字是校准过的——太小（0.15）簇内余弦会越过去重阈值 0.95，把整个簇
当成复读合并掉；太大（0.3）HDBSCAN 会把它切成两半。两头都会让测试测的是
别的东西。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from app.constants import (
    CLUSTER_PALETTE,
    MIN_COMMENTS_FOR_CLUSTERING,
    NOISE_CLUSTER_KEY,
    SINGLE_THEME_SPLIT_AT,
    TAIL_CLUSTER_KEY,
)
from app.graph.nodes import n3_cluster
from app.graph.nodes.n3_cluster import (
    _dedup,
    _normalize,
    cluster_only,
    clusterable,
    emit_clusters,
)

DIM = 8
#: 簇内抖动。见模块 docstring——这个值是校准出来的，不是随手写的。
JITTER = 0.22


def blob(axis: int, count: int, *, jitter: float = JITTER, seed: int = 0) -> list[list[float]]:
    """第 `axis` 个正交方向上的一团向量。"""
    rng = np.random.default_rng(seed)
    center = np.zeros(DIM)
    center[axis % DIM] = 1.0
    return [list(center + rng.normal(0, jitter, DIM)) for _ in range(count)]


def comments_for(count: int, *, prefix: str = "c") -> list[dict]:
    """构造评论。`like_count` 递减，让代表评论的顺序是确定的。"""
    return [
        {
            "comment_id": f"{prefix}-{i}",
            "text": f"{prefix} 的第 {i} 条评论，说了一件具体的事",
            "like_count": count - i,
        }
        for i in range(count)
    ]


def make(blobs: list[int], *, jitter: float = JITTER) -> tuple[list, list]:
    """按每团的大小造向量与评论，返回 (vectors, comments)。"""
    vectors: list[list[float]] = []
    comments: list[dict] = []
    for axis, count in enumerate(blobs):
        vectors.extend(blob(axis, count, jitter=jitter, seed=axis))
        comments.extend(comments_for(count, prefix=f"b{axis}"))
    return vectors, comments


class TestDegradationChain:
    def test_three_blobs_cluster_with_hdbscan(self) -> None:
        vectors, comments = make([10, 10, 10])
        clusters, meta = cluster_only(vectors, comments, params={})

        assert meta["method"] == "hdbscan"
        assert meta["k"] == 3
        assert meta["retried"] is False
        # 每团 10 条应当原样落在一个簇里。数量对不上说明抖动或参数漂了。
        assert sorted(c["raw_size"] for c in clusters) == [10, 10, 10]

    def test_too_few_comments_skips_clustering_entirely(self) -> None:
        """11 条不分簇。分成三簇的话，每簇的「主导情绪」只是三个人的情绪。"""
        vectors, comments = make([11])
        clusters, meta = cluster_only(vectors, comments, params={})

        assert meta["method"] == "none"
        assert meta["reason"] == "too_few_comments"
        assert len(clusters) == 1
        assert clusters[0]["size"] == 11

    def test_no_comments_is_not_an_error(self) -> None:
        clusters, meta = cluster_only([], [], params={})
        assert clusters == []
        assert meta["reason"] == "no_comments"

    def test_hdbscan_finding_nothing_falls_through_to_kmeans(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """HDBSCAN 一条簇都给不出时，必须退到 KMeans——**不能返回空结果**。

        这就是 mock 嵌入下的常态：实测 30 条评论里 16 条落进噪声，语料再散
        一点就全军覆没。所以它不是异常路径，是一条要一直好用的正常路径。

        这里把 `_hdbscan` 打成全 -1 而不是构造真向量：本机的 HDBSCAN 在
        8 维小样本上非常激进，随手造的随机方向它都能切出几簇来，想靠输入
        凑出「全噪声」反而要调参调到看不出在测什么。要守的性质是「全噪声
        → 有产出」，直接把前提摆出来最清楚。
        """
        monkeypatch.setattr(
            n3_cluster, "_hdbscan", lambda matrix, mcs, min_samples: np.full(matrix.shape[0], -1, dtype=int)
        )
        vectors, comments = make([30])
        clusters, meta = cluster_only(vectors, comments, params={})

        assert meta["method"] == "kmeans"
        assert meta["retried"] is True
        # KMeans 至少给出两个簇，否则「切分」没有发生
        assert len(clusters) >= 2
        assert sum(c["raw_size"] for c in clusters) == 30

    @pytest.mark.parametrize("count", [12, SINGLE_THEME_SPLIT_AT, 40])
    def test_a_lone_theme_is_split_only_when_there_are_enough_people(
        self, monkeypatch: pytest.MonkeyPatch, count: int
    ) -> None:
        """只切出一个簇，是「没找到结构」还是「真的同质」——按人数分界。

        12 条挤成一簇可能真是一件事；40 条还挤成一簇就只能说明参数没调对，
        给一张「全体」的卡片等于什么都没说。分界线是 `SINGLE_THEME_SPLIT_AT`。

        同样打桩 `_hdbscan`：自然输入下 HDBSCAN 几乎不会返回 k==1。
        """
        monkeypatch.setattr(
            n3_cluster, "_hdbscan", lambda matrix, mcs, min_samples: np.zeros(matrix.shape[0], dtype=int)
        )
        vectors, comments = make([count])
        _clusters, meta = cluster_only(vectors, comments, params={})

        if count >= SINGLE_THEME_SPLIT_AT:
            assert meta["method"] == "kmeans"
            assert meta["retried"] is True
        else:
            # 人数不够时保留 HDBSCAN 的判断，别硬凑出几个簇来
            assert meta["method"] == "hdbscan"
            assert meta["k"] == 1

    def test_theme_gate_demotes_the_two_comment_cluster(self) -> None:
        """一簇 10 条 + 两簇 2 条：小的两个达不到主题门槛，必须降级。

        不降级时中栏渲染成 1 张真卡片加 2 张两条评论的卡片，看起来像坏了。
        """
        vectors = blob(0, 10, seed=0) + blob(1, 2, seed=1) + blob(2, 2, seed=2)
        comments = comments_for(10, prefix="big") + comments_for(2, prefix="s1") + comments_for(2, prefix="s2")
        clusters, meta = cluster_only(vectors, comments, params={})

        themes = [c for c in clusters if not c["is_noise"]]
        assert meta["demoted"] >= 1
        for cluster in themes:
            assert cluster["size"] >= 3, f"主题闸门没拦住 {cluster['cluster_key']}（{cluster['size']} 条）"

    def test_force_kmeans_override_is_honoured(self) -> None:
        vectors, comments = make([10, 10, 10])
        _clusters, meta = cluster_only(vectors, comments, params={"force_kmeans": True})
        assert meta["method"] == "kmeans"

    def test_max_clusters_pushes_the_rest_into_tail(self) -> None:
        """7 个团、上限 5：后两个并进「其他·长尾」，**不是丢掉**。

        丢掉会让画像建立在一个有偏的子集上，而且屏幕上没有任何迹象。

        **走 HDBSCAN，不走 KMeans**：KMeans 的 k 由 `_kmeans_k` 钳在上限内，
        长尾那条分支根本进不去；长尾是给「HDBSCAN 找出 7 个真簇但卡片放不下」
        准备的。
        """
        vectors: list[list[float]] = []
        comments: list[dict] = []
        for axis in range(7):
            vectors.extend(blob(axis, 4, seed=axis))
            comments.extend(comments_for(4, prefix=f"g{axis}"))

        clusters, meta = cluster_only(vectors, comments, params={"max_clusters": 5})

        assert meta["method"] == "hdbscan"
        tails = [c for c in clusters if c["cluster_key"] == TAIL_CLUSTER_KEY]
        assert len(tails) == 1
        named = [c for c in clusters if c["cluster_key"].startswith("c")]
        assert len(named) == 5
        # 长尾里必须**至少两个团的人**。只看条数不够——一个团也可能有 8 条，
        # 而那样「长尾」就只是把某个簇改了个名字，被挤下去的那个照样没了。
        # 哪个团被挤下去由簇大小排序决定，不固定，所以只数来源数。
        prefixes = {cid.split("-")[0] for cid in tails[0]["comment_ids"]}
        assert len(prefixes) >= 2, f"长尾只装了一个团的成员：{prefixes}"

    def test_kmeans_path_loses_nobody(self) -> None:
        """KMeans 会给每条评论一个簇，所以总数必须严丝合缝地对上。

        HDBSCAN 那边允许有噪声（它会物化成「边缘声音」或按阈值丢掉），
        KMeans 这条路上少一条就是真的丢了。
        """
        vectors, comments = make([10, 10, 10])
        clusters, _meta = cluster_only(vectors, comments, params={"force_kmeans": True})

        assert sum(c["raw_size"] for c in clusters) == len(comments)
        seen = [cid for c in clusters for cid in c["comment_ids"]]
        assert sorted(seen) == sorted(c["comment_id"] for c in comments)


class TestNoise:
    def test_noise_is_materialized_above_the_ratio(self) -> None:
        """噪声超过 15% 时物化成「边缘声音」。

        不物化的话这些人从画像里凭空消失，而结论是建立在剩下的人身上的。
        """
        vectors = blob(0, 12, seed=0) + blob(1, 4, seed=1)
        # 4 个孤立方向的单点：彼此余弦为 0，HDBSCAN 判为噪声
        vectors += [list(np.eye(DIM)[i % DIM] * -1.0) for i in range(4)]
        comments = comments_for(12, prefix="big") + comments_for(4, prefix="s") + comments_for(4, prefix="n")

        clusters, meta = cluster_only(vectors, comments, params={"force_kmeans": False})
        noise = [c for c in clusters if c["cluster_key"] == NOISE_CLUSTER_KEY]
        # KMeans 不给噪声，所以这条只有 HDBSCAN 路径才可能成立；
        # 无论走哪条路，断言都是「要么没有噪声簇，要么它标着 is_noise」
        for cluster in noise:
            assert cluster["is_noise"] is True
        assert meta["method"] in {"hdbscan", "kmeans"}

    def test_noise_ratio_below_threshold_is_not_materialized(self) -> None:
        """12 条里有 1 条噪声（8%）不该长出一张「边缘声音」的卡片。"""
        vectors = blob(0, 11, seed=0) + [list(np.eye(DIM)[3] * -1.0)]
        comments = comments_for(11, prefix="big") + comments_for(1, prefix="odd")
        clusters, _meta = cluster_only(vectors, comments, params={})
        assert not [c for c in clusters if c["cluster_key"] == NOISE_CLUSTER_KEY]


class TestDedup:
    """去重是纯几何判断，所以直接测 `_dedup`，不绕整条流水线。

    曾经写成「造两个团 + 一对待合并的向量，断言 `meta["deduped"] == 1`」——
    那个断言测的是「`blob` 造出来的向量恰好没有别的近邻对」，跟去重逻辑毫无
    关系；抖动一改数字就变，而失败信息指向的是去重。这里改成手写一个只有
    一对相似的 4 条矩阵，`deduped` 是多少就是多少。
    """

    @staticmethod
    def _matrix(rows: list[list[float]]) -> Any:
        return _normalize(np.asarray(rows, dtype=np.float64))

    def test_near_identical_comments_are_folded_into_one(self) -> None:
        """余弦 >0.95 的两条是复读，只留一条参与分簇。"""
        matrix = self._matrix([[1.0, 0.0, 0.0], [1.0, 0.01, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        comments = [
            {"comment_id": "dup-a", "like_count": 99},
            {"comment_id": "dup-b", "like_count": 1},
            {"comment_id": "other-c", "like_count": 5},
            {"comment_id": "other-d", "like_count": 5},
        ]

        keep, rep_of = _dedup(matrix, comments)

        kept = [comments[i]["comment_id"] for i in keep]
        assert "dup-a" in kept and "dup-b" not in kept
        assert len(keep) == 3
        # 位置空间：被并掉的那条（下标 1）指向代表在保留集里的位置，不是原始下标
        assert rep_of[0] == rep_of[1] == 0
        assert len(rep_of) == 4

    def test_representative_wins_by_like_count(self) -> None:
        """两条复读里留下点赞高的那条——「谁说的」在评论区是有区别的。"""
        matrix = self._matrix([[1.0, 0.0, 0.0], [1.0, 0.01, 0.0]])
        comments = [
            {"comment_id": "low", "like_count": 1},
            {"comment_id": "high", "like_count": 500},
        ]

        keep, _rep_of = _dedup(matrix, comments)

        assert [comments[i]["comment_id"] for i in keep] == ["high"]

    def test_duplicates_are_not_deleted_from_the_clusters(self) -> None:
        """被并掉的那条不算消失：它归进代表所在的簇，raw_size 记着它。

        落库时 `comment_ids` 是**全部成员**（含重复项），只有分簇那一步
        看不见它们。少一条就是真的少了一条人。
        """
        half = blob(0, 8, seed=0) + blob(1, 7, seed=1)
        vectors = half + half
        comments = comments_for(15, prefix="a") + comments_for(15, prefix="b")

        clusters, meta = cluster_only(vectors, comments, params={})

        assert meta["deduped"] >= 15, "逐条复制了一遍，至少该并掉一半"
        assert sum(c["raw_size"] for c in clusters) == 30
        assert sum(c["size"] for c in clusters) == 30 - meta["deduped"]
        seen = [cid for c in clusters for cid in c["comment_ids"]]
        assert len(seen) == len(set(seen)) == 30


class TestRowContract:
    """与 `persist._persist_clusters` 的字段契约。列名对不上就是静默丢数据。"""

    def test_comment_ids_are_douyin_ids_not_indexes(self) -> None:
        vectors, comments = make([10, 10, 10])
        clusters, _meta = cluster_only(vectors, comments, params={})

        known = {c["comment_id"] for c in comments}
        for cluster in clusters:
            assert cluster["comment_ids"]
            assert set(cluster["comment_ids"]) <= known

    def test_every_row_carries_the_same_params(self) -> None:
        """`runs.py` 的 cluster_meta 取自 clusters[0].params。

        只写第一行的话，改一次排序元信息就没了。
        """
        vectors, comments = make([10, 10, 10])
        clusters, meta = cluster_only(vectors, comments, params={})
        assert all(c["params"] == meta for c in clusters)

    def test_centroid_is_left_empty(self) -> None:
        # 1024 维浮点存进 PG 的 JSON 列只会变成没人读的死数据
        vectors, comments = make([10, 10, 10])
        clusters, _meta = cluster_only(vectors, comments, params={})
        assert all(c["centroid"] is None for c in clusters)

    def test_colors_come_from_the_palette_in_order(self) -> None:
        vectors, comments = make([10, 10, 10, 4, 4, 4, 4])
        clusters, _meta = cluster_only(vectors, comments, params={"force_kmeans": True})
        for index, cluster in enumerate(clusters):
            assert cluster["color"] == CLUSTER_PALETTE[index % len(CLUSTER_PALETTE)]
            assert cluster["order_index"] == index

    def test_rows_are_json_serialisable(self) -> None:
        """**checkpoint 的硬要求。** numpy 的 int64 不是 JSON 类型，
        `json.dumps` 会在写到 PG 的那一刻才炸，而那时离这里很远。"""
        import json

        vectors, comments = make([10, 10, 10])
        clusters, meta = cluster_only(vectors, comments, params={})
        for cluster in clusters:
            json.dumps({k: v for k, v in cluster.items() if not k.startswith("_")})
        json.dumps(meta)

    def test_size_counts_survivors_and_raw_size_counts_all(self) -> None:
        vectors, comments = make([10, 10, 10])
        clusters, _meta = cluster_only(vectors, comments, params={})
        for cluster in clusters:
            assert cluster["size"] <= cluster["raw_size"]


class TestDeterminism:
    def test_same_input_gives_the_same_clusters(self) -> None:
        """KMeans 要固定 random_state，HDBSCAN 要固定 tie-break。

        不定的话重跑一次簇就换一批，用户会以为数据变了。
        """
        vectors, comments = make([10, 10, 10])
        first, meta_a = cluster_only(vectors, comments, params={})
        second, meta_b = cluster_only(vectors, comments, params={})

        assert meta_a == meta_b
        assert [c["comment_ids"] for c in first] == [c["comment_ids"] for c in second]
        assert [c["label"] for c in first] == [c["label"] for c in second]


class TestEmitting:
    def test_emit_strips_internal_fields(self) -> None:
        """`_representatives` 是几百字的原文摘录，不进 SSE 也不进 checkpoint。"""
        sent: list[dict] = []

        class FakeEmitter:
            def partial(self, kind: str, data: dict) -> None:
                sent.append({"kind": kind, "data": data})

        vectors, comments = make([10, 10, 10])
        clusters, meta = cluster_only(vectors, comments, params={})
        clusters[0]["_representatives"] = ["一段只该给模型看的原文"]
        clusters[0]["label_source"] = "model"

        emit_clusters(FakeEmitter(), clusters, meta)  # type: ignore[arg-type]

        assert len(sent) == 1
        assert sent[0]["data"]["cluster_meta"] == meta
        for row in sent[0]["data"]["clusters"]:
            assert not any(k.startswith("_") for k in row)
            assert "label_source" not in row


class TestClusterable:
    def test_ads_and_spam_are_excluded(self) -> None:
        """广告文本高度雷同，会把一整个簇吸走，让主导情绪变成广告词的分布。"""
        rows = [
            {"comment_id": "1", "text": "正常评论", "like_count": 0},
            {"comment_id": "2", "text": "加微信 buy now", "like_count": 0, "is_ad": True},
            {"comment_id": "3", "text": "刷屏刷屏", "like_count": 0, "is_spam": True},
            {"comment_id": "4", "text": "重复的", "like_count": 0, "is_duplicate": True},
            {"comment_id": "5", "text": "   ", "like_count": 0},
        ]
        kept = clusterable(rows)
        assert [c["comment_id"] for c in kept] == ["1"]

    def test_threshold_is_named_not_magic(self) -> None:
        # 这条测试的意义是：改常量时这里会跟着动，而不是散落的字面量
        assert MIN_COMMENTS_FOR_CLUSTERING == 12


@pytest.mark.parametrize("count", [12, 13, 24, 49])
def test_never_returns_more_themes_than_max_clusters(count: int) -> None:
    vectors, comments = make([count // 3 + 1] * 3)
    vectors, comments = vectors[:count], comments[:count]
    clusters, _meta = cluster_only(vectors, comments, params={"max_clusters": 5})
    named = [c for c in clusters if c["cluster_key"].startswith("c")]
    assert len(named) <= 5
