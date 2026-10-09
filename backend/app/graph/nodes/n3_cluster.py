"""Node 3 — 评论聚类。

分三段，边界是**代价**而不是逻辑：

    embed_comments  昂贵（每条约一次网络往返）且可缓存 → 按文本哈希缓存在 Redis
    cluster_only    纯函数，无 IO 无模型，~50ms → 换参数就能重跑
    label_clusters  昂贵（每簇一次 LLM 调用）→ 逐张卡片浮现

拆开是为了让「调整聚类参数」不必重跑 embedding——1024 维 × 100 条浮点写进
checkpoint 是白烧，所以向量只在函数间传递，不进 state。

**但要说清楚：这个收益本轮是零。** 干预入口（重新聚类按钮）没有做，而且
mock 模式下 `factory.py` 也没给 `MockEmbeddingModel` 包 `CachedEmbeddingModel`。
拆分的价值要等下一轮接上入口才兑现。

## 退化链

1. `n < 12` → 单簇，`method=none`。评论太少时任何分法都是噪声，不如老实说
2. 语义去重（cosine > 0.95，保留点赞高的那条）→ 重复的并入代表所在簇
3. `HDBSCAN(min_cluster_size=clamp(round(n/20),3,8), min_samples=2, metric=cosine)`
4. 一个非噪声簇都没有 → `min_cluster_size=2` 重试一次
5. 仍没有，或只出 1 簇而 n≥20 → KMeans
6. **主题质量闸门**：成员 < `MIN_THEME_SIZE(3)` 的簇降级为噪声；降级后合格簇
   不足 2 个 → 改走 KMeans
7. 超 `max_clusters` 的尾部并入「其他·长尾」
8. 噪声占比 > 15% → 物化成「边缘声音」簇

**第 6 条不在原始设计里，是实测加的。** mock 嵌入没有语义几何（本机三个
fixture 的两两余弦上限 0.34–0.40，真实嵌入下同类文本通常 0.7+），HDBSCAN 会
把评论切成「一个大块 + 一堆 2 条小簇」。不设闸门时中栏渲染成 1 张真卡片加
4 张两条评论的卡片，看起来像坏了。加闸门后三个 fixture 都落到 KMeans 兜底，
每张卡片都有 ≥3 条。真实嵌入下 HDBSCAN 通常能过闸门，KMeans 只作兜底。

## 配色与 centroid

`centroid` **刻意留空**：1024 维浮点存进 PG 的 JSON 列只会变成没人读的死数据。
`color` 由后端按下标从 `CLUSTER_PALETTE` 取——它是流水线的产物，前端只负责画。
"""

from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.constants import (
    CLUSTER_PALETTE,
    DUPLICATE_COSINE_THRESHOLD,
    HDBSCAN_MIN_SIZE_CEIL,
    HDBSCAN_MIN_SIZE_FLOOR,
    HDBSCAN_MIN_SIZE_RETRY,
    KEYWORDS_TOP_N,
    MIN_COMMENTS_FOR_CLUSTERING,
    MIN_THEMES_KEPT,
    MIN_THEME_SIZE,
    NOISE_CLUSTER_KEY,
    NOISE_CLUSTER_LABEL,
    NOISE_MATERIALIZE_RATIO,
    REPRESENTATIVE_COMMENTS,
    REPRESENTATIVE_MAX_CHARS,
    REPRESENTATIVE_STORED,
    SINGLE_THEME_SPLIT_AT,
    TAIL_CLUSTER_KEY,
    TAIL_CLUSTER_LABEL,
    ClusterMethod,
)
from app.graph.emitting import NodeEmitter
from app.graph.state import AnalysisState, warning_message
from app.kb.lexical_tags import keyphrases_across
from app.logging_conf import get_logger

log = get_logger(__name__)

NODE = "n3_cluster"

#: 一次嵌入多少条。与 provider 自己的批大小无关——那个在 openai_compat 里
#: 按厂商上限切；这里切是为了**进度条能走**，一次调 100 条的话 0% 会停很久。
EMBED_CHUNK = 32


# ----------------------------------------------------------------------
# 第一段：嵌入
# ----------------------------------------------------------------------


def clusterable(comments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """能参与聚类的评论。

    广告与灌水必须排除：它们的文本高度雷同（同一个模板刷几百条），会把
    一整个簇吸走，让「主导情绪」变成广告词的分布。空文本同理。
    被排除的评论仍在 state 里，左栏靠 `filter_reason` 展示它们。
    """
    out: list[dict[str, Any]] = []
    for c in comments:
        if c.get("is_ad") or c.get("is_spam") or c.get("is_duplicate"):
            continue
        if not str(c.get("text") or "").strip():
            continue
        out.append(c)
    return out


async def embed_comments(
    texts: list[str],
    *,
    emitter: NodeEmitter | None = None,
    model: Any = None,
) -> list[list[float]]:
    """把评论文本变成向量。**分段调用**，让进度条真的在动。

    `model` 可注入是为了单测能塞一个假模型；生产路径从工厂取，那里按
    `EMBEDDING_MODE` 决定是 mock 还是真实 provider。
    """
    if model is None:
        from app.providers.factory import get_embedding_model

        model = get_embedding_model()

    vectors: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, EMBED_CHUNK):
        batch = texts[start : start + EMBED_CHUNK]
        vectors.extend(await model.embed(batch))
        if emitter is not None:
            # 嵌入占本节点 0–60% 的带宽：聚类本身是毫秒级的，而标签生成
            # 才是剩下的 40%。这样分带，进度条的形状与实际耗时一致。
            emitter.set_progress(0.6 * min(1.0, (start + len(batch)) / max(1, total)))
    return vectors


# ----------------------------------------------------------------------
# 第二段：聚类（纯函数）
# ----------------------------------------------------------------------


def cluster_only(
    vectors: list[list[float]],
    comments: list[dict[str, Any]],
    *,
    params: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """向量 → 簇。**无 IO、无模型、无 emitter**，所以能直接喂假向量测。

    返回 `(clusters, meta)`。`meta` 与每个簇的 `params` 是同一份内容——
    persist 每行写一遍，`runs.py` 的 `cluster_meta` 取自 `clusters[0].params`。
    """
    import numpy as np

    overrides = dict(params or {})
    settings = get_settings()
    n = len(comments)

    if n == 0:
        return [], _meta(ClusterMethod.NONE, reason="no_comments", n=0)

    if n < MIN_COMMENTS_FOR_CLUSTERING:
        # 不硬凑：12 条评论分成 3 个簇，每个簇的「主导情绪」就是 4 个人的情绪，
        # 而卡片上会写得像一条结论。整体当一簇，方法如实写 none。
        meta = _meta(ClusterMethod.NONE, reason="too_few_comments", n=n, k=1)
        labels = np.zeros(n, dtype=int)
        # 没有去重，所以「代表」就是自己，下标即位置。
        clusters = _build(labels, list(range(n)), set(range(n)), comments, {}, meta)
        return clusters, meta

    matrix = _normalize(np.asarray(vectors, dtype=np.float64))
    if matrix.shape[0] != n:
        raise ValueError(f"向量数 {matrix.shape[0]} 与评论数 {n} 不一致")

    keep, rep_of = _dedup(matrix, comments)
    kept = matrix[keep]
    survivors = set(keep)
    deduped = n - len(keep)
    # 每个保留点代表多少条原始评论。主题闸门与 k==1 判定都按**原始条数**看，
    # 理由见 `_theme_gate`；`rep_of` 是位置列表，bincount 直接给出每个位置
    # 被代表了多少次。
    raw_counts = np.bincount(np.asarray(rep_of, dtype=int), minlength=len(keep))

    max_clusters = int(overrides.get("max_clusters") or settings.CLUSTER_MAX_DEFAULT)
    min_samples = int(settings.CLUSTER_MIN_SAMPLES)
    mcs = max(
        HDBSCAN_MIN_SIZE_FLOOR,
        min(HDBSCAN_MIN_SIZE_CEIL, round(len(keep) / 20)),
    )

    if overrides.get("force_kmeans"):
        labels = _kmeans(kept, _kmeans_k(len(keep), max_clusters))
        method, retried = ClusterMethod.KMEANS, False
    else:
        labels, method, retried = _fit(kept, raw_counts, mcs, min_samples, max_clusters)

    labels, demoted = _theme_gate(labels, raw_counts)
    # 闸门把 HDBSCAN 的结果砍到不足 2 个合格簇 → 它没找到主题，改走 KMeans。
    # 已经在 KMeans 上就不再退——KMeans 给的是等分，再跑一次结果一样。
    if (
        demoted
        and method is ClusterMethod.HDBSCAN
        and _theme_count(labels, raw_counts) < MIN_THEMES_KEPT
    ):
        log.info("n3_fallback_to_kmeans", n=len(keep), demoted=demoted, mcs=mcs)
        labels = _kmeans(kept, _kmeans_k(len(keep), max_clusters))
        labels, _ = _theme_gate(labels, raw_counts)
        method, retried = ClusterMethod.KMEANS, True

    meta = _meta(
        method,
        reason="ok",
        n=n,
        k=_theme_count(labels, raw_counts),
        deduped=deduped,
        retried=retried,
        demoted=demoted,
        min_cluster_size=mcs if method is ClusterMethod.HDBSCAN else None,
        max_clusters=max_clusters,
    )
    clusters = _build(labels, rep_of, survivors, comments, overrides, meta)
    # 每行写同一份 meta：`runs.py` 从 clusters[0].params 取元信息，
    # 只写第一行的话换一次排序就丢了。
    for row in clusters:
        row["params"] = meta
    return clusters, meta


def _meta(method: ClusterMethod, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"method": str(method), "k": 0}
    out.update(extra)
    return out


def _normalize(matrix: Any) -> Any:
    """按行 L2 归一化。余弦距离在归一化后的向量上就是欧氏距离。"""
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # 零向量的范数是 0，除下去会得到 nan 并污染整个距离矩阵
    return matrix / np.where(norms == 0, 1.0, norms)


def _dedup(matrix: Any, comments: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    """语义去重。返回 (保留的下标, 每条评论 → 其代表**在保留集里的位置**)。

    n2 已经用哈希去过一次重复（逐字相同的），这里补的是**语义级**的复读：
    「我也有同感」「我也是」这类不同字但同义的。它们会在任何聚类算法里
    抱成独立的一团，然后被当成一个「主题」摆上卡片。

    去重只影响**谁参与分簇**，不把重复项删掉——每个重复项照样归到它代表的
    那一簇里，`raw_size` 记总数、`size` 记去重后的数。

    返回的下标空间容易搞混，这里定死：`keep[i]` 是第 i 条**保留**评论在原始
    数组里的位置；`rep_of[j]` 是第 j 条**原始**评论的代表在 `keep` 里的**位置**
    （不是原始下标）。聚类算法跑在 `keep` 上，它的 labels 是按位置索引的，
    所以摊回原始评论时必须用位置，用原始下标会错位。
    """
    n = matrix.shape[0]
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # 代表选点赞高的；点赞相同取下标小的。**必须完全确定**，否则
        # 每次重跑的代表不同，簇的成员构成就会漂移，测试没法写。
        ka = (int(comments[ra].get("like_count") or 0), -ra)
        kb = (int(comments[rb].get("like_count") or 0), -rb)
        if ka >= kb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    similarity = matrix @ matrix.T
    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] > DUPLICATE_COSINE_THRESHOLD:
                union(i, j)

    roots = [find(i) for i in range(n)]
    keep = sorted(set(roots))
    position = {origin: index for index, origin in enumerate(keep)}
    return keep, [position[root] for root in roots]


def _fit(
    matrix: Any, raw_counts: Any, mcs: int, min_samples: int, max_clusters: int
) -> tuple[Any, ClusterMethod, bool]:
    """先 HDBSCAN，不行再降 min_cluster_size 重试一次。返回 (labels, 方法, 是否降级过)。"""
    labels = _hdbscan(matrix, mcs, min_samples)
    themes = _theme_count(labels, raw_counts)

    if themes == 0:
        retry = _hdbscan(matrix, HDBSCAN_MIN_SIZE_RETRY, min_samples)
        if _theme_count(retry, raw_counts) > 0:
            return retry, ClusterMethod.HDBSCAN, True
        return _kmeans(matrix, _kmeans_k(matrix.shape[0], max_clusters)), ClusterMethod.KMEANS, True

    if themes == 1 and int(raw_counts.sum()) >= SINGLE_THEME_SPLIT_AT:
        # **只找到一个簇等于什么都没找到。** HDBSCAN 在这种输入上会返回
        # 「全部属于 0 号簇」，看起来像成功（k=1 而非 k=0），但它没有切分
        # 任何东西。评论够多时改走 KMeans——等分至少给出可比较的几组。
        return _kmeans(matrix, _kmeans_k(matrix.shape[0], max_clusters)), ClusterMethod.KMEANS, True

    return labels, ClusterMethod.HDBSCAN, False


def _hdbscan(matrix: Any, min_cluster_size: int, min_samples: int) -> Any:
    """一簇 HDBSCAN。样本不够时**直接返回全噪声**，不让 sklearn 抛。

    用 `sklearn.cluster.HDBSCAN` 而不是 `hdbscan` 包：后者在 Windows 上要
    C 工具链，而本项目的目标是「克隆下来就能跑」。
    """
    import numpy as np
    from sklearn.cluster import HDBSCAN

    n = matrix.shape[0]
    if n < max(2, min_cluster_size):
        return np.full(n, -1, dtype=int)
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="cosine",
        # 显式给 copy：sklearn 1.9 起默认值要改，不写会刷 DeprecationWarning
        copy=True,
    )
    return np.asarray(model.fit_predict(matrix), dtype=int)


def _kmeans(matrix: Any, k: int) -> Any:
    import numpy as np
    from sklearn.cluster import KMeans

    n = matrix.shape[0]
    k = max(1, min(k, n))
    if k == 1:
        return np.zeros(n, dtype=int)
    model = KMeans(n_clusters=k, n_init=10, random_state=0)
    return np.asarray(model.fit_predict(matrix), dtype=int)


def _kmeans_k(n: int, max_clusters: int) -> int:
    """KMeans 的簇数：每 10 条评论一个簇，钳在 [2, max_clusters]。

    这个比率是实测出来的：本机三个 fixture（49 / 37 / 27 条）在它下面分别
    得到 5 / 4 / 3 个簇，每个簇 5–15 条，卡片上放得下也读得出趋势。
    """
    return max(2, min(max_clusters, round(n / 10)))


def _theme_count(labels: Any, raw_counts: Any) -> int:
    """非噪声、且**原始条数**够 `MIN_THEME_SIZE` 的簇有几个。"""
    import numpy as np

    return int(
        sum(
            1
            for value in np.unique(labels)
            if value >= 0 and int(raw_counts[labels == value].sum()) >= MIN_THEME_SIZE
        )
    )


def _theme_gate(labels: Any, raw_counts: Any) -> tuple[Any, int]:
    """原始条数不足 `MIN_THEME_SIZE` 的簇降级为噪声。返回 (新 labels, 降级了几个)。

    「2 条评论」不是一个主题，是一个巧合。把它标成噪声而不是删掉：噪声
    仍会在占比超阈值时物化成「边缘声音」，那才是它该出现的地方。

    **判据是原始条数，不是去重后的条数。** 这两个数在复读多的簇上差得很远：
    30 条「我也是」会被去重成 1 个代表点，按代表点数判就会被当成一个只有
    1 条的小簇降级成噪声——而它恰恰是评论区里最一致的那群人。门槛问的是
    「有几个人在说这件事」，那就得用人的数量去量。
    """
    import numpy as np

    out = np.array(labels, dtype=int, copy=True)
    demoted = 0
    for value in np.unique(out):
        if value >= 0 and int(raw_counts[out == value].sum()) < MIN_THEME_SIZE:
            out[out == value] = -1
            demoted += 1
    return out, demoted


def _build(
    labels: Any,
    rep_of: Any,
    survivors: set[int],
    comments: list[dict[str, Any]],
    overrides: dict[str, Any],
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    """把「保留集上的标签」摊回全部评论，并组装成 persist 认识的形状。

    `labels[i]` 是第 i 条保留评论的簇号；`rep_of[j]` 是第 j 条原始评论的代表
    在保留集里的**位置**（见 `_dedup`）。重复项因此自动跟随它代表的那一簇。

    `survivors` 是去重后仍作为代表的原始下标。它决定每簇的 `size`（去重后）
    与 `raw_size`（含重复）——两个数字不同的时候，卡片上那句「含 N 条重复表达」
    才有东西可显示。
    """
    settings = get_settings()
    max_clusters = int(overrides.get("max_clusters") or settings.CLUSTER_MAX_DEFAULT)

    groups: dict[int, list[int]] = {}
    for index in range(len(comments)):
        label = int(labels[int(rep_of[index])])
        groups.setdefault(label, []).append(index)

    themes = sorted(
        (label for label in groups if label >= 0),
        key=lambda lb: (-len(groups[lb]), lb),
    )

    # 超上限的尾部并入「其他·长尾」。**不是丢弃**：那些评论确实存在，
    # 删掉它们等于让画像建立在一个有偏的子集上，而且没人会发现少了什么。
    tail: list[int] = []
    if len(themes) > max_clusters:
        for label in themes[max_clusters:]:
            tail.extend(groups.pop(label))
        themes = themes[:max_clusters]

    noise = groups.pop(-1, [])
    rows: list[dict[str, Any]] = []

    for order, label in enumerate(themes):
        rows.append(
            _row(f"c{label}", groups[label], comments, survivors=survivors, is_noise=False, meta=meta)
        )

    if tail:
        rows.append(
            _row(TAIL_CLUSTER_KEY, tail, comments, survivors=survivors, is_noise=False, meta=meta)
        )

    # 噪声占比不够高时不做成簇——那会把一两句无关的吐槽摆成一个「主题」。
    # 它们只是不参与画像，仍在左栏看得见。
    if noise and len(noise) / max(1, len(comments)) > NOISE_MATERIALIZE_RATIO:
        rows.append(
            _row(NOISE_CLUSTER_KEY, noise, comments, survivors=survivors, is_noise=True, meta=meta)
        )

    for order, row in enumerate(rows):
        row["order_index"] = order
        row["color"] = CLUSTER_PALETTE[order % len(CLUSTER_PALETTE)]
    return rows


def _row(
    key: str,
    indices: list[int],
    comments: list[dict[str, Any]],
    *,
    survivors: set[int],
    is_noise: bool,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """一个簇的行。字段名与 `persist._persist_clusters` 逐一对齐。"""
    members = [comments[i] for i in indices]
    # 代表评论按点赞降序。索引作次级键，保证同赞数时的顺序也是确定的。
    ranked = sorted(indices, key=lambda i: (-int(comments[i].get("like_count") or 0), i))

    texts = [str(comments[i].get("text") or "") for i in ranked]
    representatives = [text[:REPRESENTATIVE_MAX_CHARS] for text in texts[:REPRESENTATIVE_COMMENTS]]

    return {
        "cluster_key": key,
        "label": NOISE_CLUSTER_LABEL if is_noise else "",
        "summary": None,
        # size 是**去重后**的条数，raw_size 是包括复读在内的总数。两个数字
        # 不同的时候，说明这一簇里有 N 条是在把同一句话换个说法讲——
        # 那会显著影响「这簇有多少人」的观感，所以卡片上有义务显示出来。
        "size": sum(1 for i in indices if i in survivors),
        "raw_size": len(members),
        "is_noise": is_noise,
        # 本地算的关键词，不调模型——卡片上除了模型给的标签，还该有一条
        # 可核查的原始线索，而且它在模型没接上时也算得出来。
        "keywords": keyphrases_across(texts, KEYWORDS_TOP_N),
        "emotion_tags": [],
        "topic_tags": [],
        "need_tags": [],
        "representative_comment_ids": [
            str(comments[i].get("comment_id")) for i in ranked[:REPRESENTATIVE_STORED]
        ],
        # 1024 维浮点存进 JSON 列只会变成没人读的死数据
        "centroid": None,
        "color": "",
        "order_index": 0,
        "params": meta,
        # persist 的 membership_of 读这个键把评论挂到簇上。
        # **用抖音 comment_id**，不是数据库主键——节点根本不知道后者。
        "comment_ids": [str(c.get("comment_id")) for c in members],
        # 给提示词用的代表评论原文。不进 DB（persist 不读这个键），
        # 只在节点内部流转。
        "_representatives": representatives,
    }


# ----------------------------------------------------------------------
# 第三段：标签
# ----------------------------------------------------------------------


async def label_clusters(
    clusters: list[dict[str, Any]],
    *,
    emitter: NodeEmitter | None = None,
    model: Any = None,
) -> list[dict[str, Any]]:
    """逐簇调一次 `cluster_label`，就地补上 label/summary/三种标签。

    **任何一簇失败都不抛出去。** 标签是卡片上最好看的部分，但簇本身
    （成员、关键词、代表评论）才是分析结果；为了一个名字让整条流水线
    在 n3 断掉，代价与收益完全不成比例。失败的那一簇退回本地标签，
    调用方据此计一条 warning，用户看得见「这些名字是兜底的」。
    """
    from app.prompts import cluster_label as prompt
    from app.providers.base import ProviderError
    from app.providers.factory import get_chat_model

    if not clusters:
        return clusters
    if model is None:
        model = get_chat_model()

    meta = clusters[0].get("params") or {}
    total = len(clusters)
    for index, cluster in enumerate(clusters):
        comments = list(cluster.get("_representatives") or [])
        try:
            payload = await _call(model, prompt, cluster, comments)
        except ProviderError as exc:
            log.warning("n3_label_failed", cluster=cluster.get("cluster_key"), error=str(exc))
            payload = None

        _apply_label(cluster, payload)
        if emitter is not None:
            emitter.set_progress(0.6 + 0.4 * (index + 1) / total)
            # 每标完一簇就发一次**完整列表**：前端 `clusters` 分支是整体替换，
            # 发单条会被当成「只有这一簇」把别的卡片抹掉。
            emit_clusters(emitter, clusters, meta)

    return clusters


def emit_clusters(
    emitter: NodeEmitter, clusters: list[dict[str, Any]], meta: dict[str, Any]
) -> None:
    """发一次 clusters 增量，**剥掉内部字段**。

    `_representatives` 是给提示词用的原文摘录，每簇几百字。不剥的话它会
    跟着 SSE 帧发给浏览器（前端用不上）、跟着 checkpoint 写进 PG（每簇一份
    重复），而这两处都不会有人发现——只会在某天奇怪「运行记录怎么这么大」。
    """
    payload = [
        {k: v for k, v in row.items() if not k.startswith("_") and k != "label_source"}
        for row in clusters
    ]
    emitter.partial("clusters", {"clusters": payload, "cluster_meta": meta})


async def _call(model: Any, prompt: Any, cluster: dict[str, Any], comments: list[str]) -> dict[str, Any]:
    from app.graph.nodes._llm import call_json

    return await call_json(
        model,
        task=prompt.TASK,
        messages=prompt.build_messages(len(comments)),
        context=prompt.build_context(
            comments, size=int(cluster.get("size") or 0), keywords=cluster.get("keywords")
        ),
        temperature=0.3,
        max_tokens=600,
    )


def _apply_label(cluster: dict[str, Any], payload: dict[str, Any] | None) -> None:
    """把模型产出写进簇行。`payload is None` 时给本地兜底，绝不抛。"""
    from app.graph.nodes._llm import clean_tags, clean_text
    from app.kb.lexical_tags import EMOTION_VOCAB

    if payload is None:
        cluster["label"] = _fallback_label(cluster)
        cluster.setdefault("summary", None)
        cluster["label_source"] = "fallback"
        return

    label = clean_text(payload.get("label"), limit=40) or _fallback_label(cluster)
    cluster["label"] = label
    cluster["summary"] = clean_text(payload.get("summary"), limit=200) or None
    # emotion 走**闭集**过滤，topic/need 是开放词表（提示词里给了示例，
    # 但没有一张能穷尽的表——硬造一张只会让贴切的词落选）。
    cluster["emotion_tags"] = clean_tags(payload.get("emotion"), allowed=EMOTION_VOCAB, limit=3)
    cluster["topic_tags"] = clean_tags(payload.get("topic"), limit=3)
    cluster["need_tags"] = clean_tags(payload.get("need"), limit=2)
    cluster["label_source"] = "model"


def _fallback_label(cluster: dict[str, Any]) -> str:
    """没有模型时的簇名。

    用关键词拼，而不是编一个「情绪·主题」——关键词是这一簇真实的字，
    用户一眼能看出这是兜底而不是模型说的。拼不出来就退回「一组声音」，
    至少不是空字符串（空 label 在卡片上是一片空白，看起来像渲染坏了）。
    """
    if cluster.get("is_noise"):
        return NOISE_CLUSTER_LABEL
    if cluster.get("cluster_key") == TAIL_CLUSTER_KEY:
        return TAIL_CLUSTER_LABEL
    keywords = [k for k in (cluster.get("keywords") or []) if k][:2]
    if keywords:
        return "·".join(keywords)
    return f"未命名的一组（{int(cluster.get('size') or 0)} 条）"


# ----------------------------------------------------------------------
# 节点入口
# ----------------------------------------------------------------------


async def n3_cluster(state: AnalysisState) -> dict[str, Any]:
    from app.providers.base import ProviderError

    emitter = NodeEmitter(NODE)
    comments = clusterable(state.get("comments") or [])

    if not comments:
        emitter.set_progress(1.0)
        return {
            "clusters": [],
            "cluster_meta": _meta(ClusterMethod.NONE, reason="no_comments", n=0),
            "warnings": [warning_message("no_comments_to_cluster", "没有可用于聚类的评论")],
        }

    texts = [str(c.get("text") or "") for c in comments]
    warnings: list[dict[str, Any]] = []

    try:
        vectors = await embed_comments(texts, emitter=emitter)
    except ProviderError as exc:
        # 嵌入挂了就没有聚类可言。**不静默返回空**——空簇在前端会渲染成
        # 「分析完成，0 个主题」，而真实情况是这一步根本没跑成。
        log.error("n3_embed_failed", error=str(exc))
        emitter.set_progress(1.0)
        return {
            "clusters": [],
            "cluster_meta": _meta(ClusterMethod.NONE, reason="embed_failed", n=len(comments)),
            "errors": [{"code": "embed_failed", "message": f"生成向量失败：{exc}", "node": NODE}],
            "warnings": [warning_message("embed_failed", f"聚类未执行：{exc}")],
        }

    clusters, meta = cluster_only(
        vectors, comments, params=state.get("cluster_params") or {}
    )

    if meta.get("method") == ClusterMethod.KMEANS.value and meta.get("k", 0) > 1:
        # 如实告知：KMeans 是等分，簇的形状不如 HDBSCAN 可信。用户有权知道
        # 自己看到的卡片是兜底结果。
        emitter.milestone("语义分簇未达门槛，已改用等分聚类")
    if meta.get("deduped"):
        emitter.milestone(f"合并了 {meta['deduped']} 条重复表达")

    # 先占位：结构（成员、关键词、代表评论）已经全了，只有名字还没生成。
    # 用户体验上这一段是「卡片先长出来，名字随后填上」，而不是等所有模型
    # 调用跑完一次性刷出五张卡。
    emit_clusters(emitter, clusters, meta)
    emitter.set_progress(0.6)

    await label_clusters(clusters, emitter=emitter)

    if any(c.get("label_source") == "fallback" for c in clusters):
        warnings.append(
            warning_message("cluster_label_fallback", "部分主题未能生成名称，已使用本地关键词兜底")
        )

    # 内部字段出栈前清掉——它们只有提示词那一步用得上，留着会跟着
    # checkpoint 一起写进 PG。
    for row in clusters:
        row.pop("_representatives", None)
        row.pop("label_source", None)

    emitter.set_progress(1.0)
    # **收尾再发一次。** label_clusters 里每簇发的是「当时」的列表，
    # 而落库走的是下面 return 的 patch——两者必须是同一份数据。多这一条，
    # 流的末态与快照的末态就永远一致，不必再去推理「最后那条 partial 到底
    # 覆盖到第几簇」。
    emit_clusters(emitter, clusters, meta)

    patch: dict[str, Any] = {"clusters": clusters, "cluster_meta": meta}
    if warnings:
        patch["warnings"] = warnings
    return patch


# 保持 `ClusterMethod` / `NOISE_MATERIALIZE_RATIO` 等常量在本模块可读——
# 测试与后续的重聚类入口都要按名字引用它们。
__all__ = [
    "EMBED_CHUNK",
    "NODE",
    "cluster_only",
    "clusterable",
    "embed_comments",
    "label_clusters",
    "n3_cluster",
]
