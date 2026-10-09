"""检索的纯函数部分：查询构造、RRF 融合、配额分配。

**这里没有 IO，也不认识 Qdrant 与 Neo4j。** 抽出来的理由不是「解耦」这种
空话，而是这三步恰好是检索里最容易悄悄算错的地方：

  - 查询少构造一条 → 证据少一张卡，界面上看不出来少的是什么
  - RRF 的并列定序不确定 → 同一次分析重跑两次给出不同结果，测都没法测
  - 配额不回填 → 「6 条心理学 + 2 条文学」或者更偏，而报告看起来完全正常

三者都不报错。放在纯函数里，每条性质都能用手算的数字钉住。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.constants import (
    LIBRARY_QUOTA,
    MAX_QUERIES,
    MIN_THEME_SIZE,
    QUERY_TOP_K,
    RRF_K,
    Library,
    RetrievalPath,
)

#: 库的固定优先级。配额、回填、最终排序都按它。
LIBRARY_ORDER: tuple[str, ...] = tuple(str(lib) for lib in Library)

#: 全局查询的权重。刻意定得比任何单簇都高——「核心张力」是整份报告的主线，
#: 簇再多也不该把它挤掉。单簇权重落在 (0, 2.0]，见 `_theme_weight`。
_WEIGHT_TENSION = 3.0
_WEIGHT_GLOBAL = 2.5


def build_queries(
    profile: Mapping[str, Any] | None,
    clusters: Sequence[Mapping[str, Any]] | None,
    *,
    depth: str = "standard",
    enabled: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """把侧写与簇翻成一组检索查询。

    每个合格主题出 2 条（心理学 1 + 文学 1），全局出 2 条（核心张力 → 心理学，
    top 情绪 → 诗词），再按 `MAX_QUERIES[depth]` 砍。砍的判据是权重：全局查询
    永远排在单簇之前，簇与簇之间按原始条数——**一个 20 条的主题比 3 条的
    更该被检索到**，这条直觉不需要模型来判断。

    `enabled` 是启用的库；`None` 表示全开。关掉的库不会产生查询，它连
    「发出去了但没人要」的空转都不该有。
    """
    profile = profile or {}
    clusters = clusters or []
    allowed = {str(x) for x in enabled} if enabled is not None else set(LIBRARY_ORDER)

    out: list[dict[str, Any]] = []
    themes = [
        c
        for c in clusters
        if not c.get("is_noise") and int(c.get("size") or 0) >= MIN_THEME_SIZE
    ]
    # 大簇在前。并列时按 order_index——它已经是确定性的，不再另找判据。
    themes.sort(key=lambda c: (-int(c.get("size") or 0), int(c.get("order_index") or 0)))
    top_size = max((int(c.get("size") or 0) for c in themes), default=1)

    for theme in themes:
        weight = _theme_weight(theme, top_size)
        key = theme.get("cluster_key")
        # 心理学那条问的是「这个主题在说什么」——标签与需求比情绪更贴近语义。
        _append(
            out,
            allowed,
            library=Library.PSYCHOLOGY,
            text=_join(theme.get("label"), theme.get("topic_tags"), theme.get("need_tags")),
            cluster_key=key,
            path=RetrievalPath.VECTOR,
            weight=weight,
        )
        # 文学那条问的是「这种情绪在古典文学里长什么样」——情绪与关键词
        # （n3 用 n-gram 统计出来的原始词）比抽象标签更容易碰到意象。
        _append(
            out,
            allowed,
            library=Library.LITERATURE,
            text=_join(theme.get("emotion_tags"), theme.get("keywords")),
            cluster_key=key,
            path=RetrievalPath.VECTOR,
            weight=weight * 0.9,
        )

    tags = profile.get("global_tags") or {}
    emotions = [e for e in (tags.get("emotion") or []) if isinstance(e, str)]
    needs = [n for n in (tags.get("need") or []) if isinstance(n, str)]

    _append(
        out,
        allowed,
        library=Library.PSYCHOLOGY,
        text=_join(profile.get("core_tension"), profile.get("summary")),
        cluster_key=None,
        path=RetrievalPath.VECTOR,
        weight=_WEIGHT_TENSION,
    )
    if emotions:
        _append(
            out,
            allowed,
            library=Library.POETRY,
            text=_join(emotions[:2], needs[:1]),
            cluster_key=None,
            path=RetrievalPath.VECTOR,
            weight=_WEIGHT_GLOBAL,
        )

    return _finalize(out, depth)


def graph_queries(
    clusters: Sequence[Mapping[str, Any]] | None,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """由簇情绪出发的图谱查询。**只补典故**，所以固定打在诗词/文学上。

    情绪词先过一遍去重再限量：三个簇都在说「哀伤」时，图上那条
    `Emotion {name:'哀伤'}` 的路径只该走一次——第二、三次返回的是同一批
    chunk_id，融合时只会把同一条证据的分数抬高，让它看起来比实际更受支持。
    """
    clusters = clusters or []
    themes = [
        c
        for c in clusters
        if not c.get("is_noise") and int(c.get("size") or 0) >= MIN_THEME_SIZE
    ]
    themes.sort(key=lambda c: (-int(c.get("size") or 0), int(c.get("order_index") or 0)))

    seen: set[str] = set()
    top_size = max((int(c.get("size") or 0) for c in themes), default=1)
    out: list[dict[str, Any]] = []
    for theme in themes:
        for emotion in theme.get("emotion_tags") or []:
            if not isinstance(emotion, str) or emotion in seen:
                continue
            seen.add(emotion)
            out.append(
                {
                    "text": emotion,
                    "library": None,
                    "cluster_key": theme.get("cluster_key"),
                    "path": str(RetrievalPath.GRAPH),
                    # 图谱返回的库由 chunk 自己决定（可能是文学也可能是诗词），
                    # 这里的 kind 只表示「这条查询在找典故」。
                    "kind": str(Library.LITERATURE),
                    "weight": _theme_weight(theme, top_size) * 0.8,
                }
            )
            if len(out) >= limit:
                return out
    return out


def rrf_fuse(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """倒数排名融合。`Σ 1/(k + rank)`，rank 从 1 开始。

    融的是名次不是分数：本机向量分 0.06–0.29 与图谱权重 0.7–0.95 量纲
    不可比，加权求和前得先拍一个归一化系数，而那个系数没有任何依据。

    **同一条候选在一条查询里出现两次只算第一次**：Qdrant 不会这么干，
    但图谱的 CO_OCCURS 多跳会（同一 chunk 经不同意象命中），而重复计数
    等价于给这条证据偷偷加权。

    并列的定序是 `(分数降序, 最好名次升序, chunk_id 字典序)`——三级都
    确定，所以同输入必然同输出，测试才敢断言顺序。
    """
    scores: dict[str, float] = {}
    best: dict[str, int] = {}

    for ranking in rankings:
        seen: set[str] = set()
        for index, chunk_id in enumerate(ranking):
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            rank = index + 1
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in best or rank < best[chunk_id]:
                best[chunk_id] = rank

    ordered = sorted(scores, key=lambda cid: (-scores[cid], best[cid], cid))
    return [(cid, round(scores[cid], 6)) for cid in ordered]


def allocate(
    ranked: Mapping[str, Sequence[str]],
    *,
    quota: Sequence[int],
    cap: int,
) -> list[dict[str, Any]]:
    """按库配额挑证据，不足的额度按同优先级**回填**。

    回填不是锦上添花：心理学库里只有 2 条命中时，那份报告不该只给
    「2 条心理学 + 0 条文学」——一条证据都没有的库比配额失衡糟得多。

    返回按 `(库优先级, 该库内的融合名次)` 排好序的选中项，每项带
    `rank_in_library` 与 `quota_slot`（回填进来的 `quota_slot` 为 `None`，
    这不是瑕疵而是线索：它告诉你这条证据是「补位」来的）。

    `quota` 为 0 的库不参与回填——配额为 0 的语义是「这个库不要」，
    而不是「先要 0 条、再靠回填补上」。
    """
    limits = {
        name: max(0, int(quota[i])) if i < len(quota) else 0
        for i, name in enumerate(LIBRARY_ORDER)
    }
    cursor = dict.fromkeys(LIBRARY_ORDER, 0)
    picked: dict[str, list[dict[str, Any]]] = {name: [] for name in LIBRARY_ORDER}

    def _take(name: str, quota_slot: int | None) -> bool:
        candidates = ranked.get(name) or []
        index = cursor[name]
        if index >= len(candidates):
            return False
        picked[name].append(
            {
                "library": name,
                "chunk_id": candidates[index],
                "rank_in_library": index,
                "quota_slot": quota_slot,
            }
        )
        cursor[name] = index + 1
        return True

    for name in LIBRARY_ORDER:
        while len(picked[name]) < limits[name] and _take(name, len(picked[name])):
            pass

    total = sum(len(v) for v in picked.values())
    progressed = True
    while total < cap and progressed:
        progressed = False
        for name in LIBRARY_ORDER:
            if total >= cap:
                break
            if not limits[name]:
                continue
            if _take(name, None):
                total += 1
                progressed = True

    out: list[dict[str, Any]] = []
    for name in LIBRARY_ORDER:
        out.extend(picked[name])
    return out[:cap]


# ----------------------------------------------------------------------


def _theme_weight(theme: Mapping[str, Any], top_size: int) -> float:
    """单簇权重落在 (0, 2.0]，永远低于全局查询的 2.5。"""
    size = int(theme.get("size") or 0)
    return 2.0 * size / max(1, top_size)


def _join(*parts: Any) -> str:
    """把若干字段拼成一条查询文本，**顺手做归一化**。

    归一化（去空白、去重、丢弃单字符）不是洁癖：查询文本同时是 embedding
    的缓存键（见 `providers/cache.py`），「想念 ｜ 遗憾」与「想念｜遗憾」
    是两个键、两次网络往返、两份向量，而它们的语义一模一样。
    """
    words: list[str] = []
    for part in parts:
        values = part if isinstance(part, (list, tuple)) else [part]
        for value in values:
            if not isinstance(value, str):
                continue
            word = " ".join(value.split())
            if len(word) < 2 or word in words:
                continue
            words.append(word)
    return " ".join(words)


def _append(
    out: list[dict[str, Any]],
    allowed: set[str],
    *,
    library: Library,
    text: str,
    cluster_key: Any,
    path: RetrievalPath,
    weight: float,
) -> None:
    if str(library) not in allowed or not text:
        return
    out.append(
        {
            "text": text,
            "library": str(library),
            "cluster_key": cluster_key,
            "path": str(path),
            "kind": str(library),
            "weight": round(weight, 3),
        }
    )


def _finalize(out: list[dict[str, Any]], depth: str) -> list[dict[str, Any]]:
    """去重 → 封顶 → 编号。

    **先去重再封顶**，不是反过来：两条一模一样的查询占掉两个名额，
    等于白白少检索一个主题——而去重恰恰是在封顶前才看得出重复。
    """
    seen: set[tuple[Any, Any, str]] = set()
    unique: list[dict[str, Any]] = []
    for query in out:
        # 同一条文本打进两个库是两条不同的查询（库不同、候选不同），
        # 所以去重键带上 library。
        key = (query["path"], query["library"], query["text"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(query)

    cap = MAX_QUERIES.get(depth, MAX_QUERIES["standard"])
    unique.sort(
        key=lambda q: (
            -float(q["weight"]),
            str(q["path"]),
            str(q.get("cluster_key") or ""),
            str(q["text"]),
        )
    )
    kept = unique[:cap]
    for index, query in enumerate(kept, start=1):
        query["id"] = f"q{index}"
    return kept


__all__ = [
    "LIBRARY_ORDER",
    "QUERY_TOP_K",
    "allocate",
    "build_queries",
    "graph_queries",
    "rrf_fuse",
]
