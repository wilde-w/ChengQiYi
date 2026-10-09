"""语料加载与校验。

**这里唯一重要的设计决定：校验失败必须硬失败，并指出文件与行号。**

语料是一份人写的、会不断增补的东西，出错是常态——漏个逗号、意象写成字符串
而不是数组、chunk_id 手抖打重。如果加载层选择「跳过坏行继续」，后果是
**知识库静默地少了几条**：检索偶尔找不到本该找到的典故，没有任何人会发现，
因为一切看起来都正常。这类失败比崩溃昂贵得多。

所以坏行一律抛 `CorpusError`，带上 `文件:行号` 与具体哪里不对。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.constants import Library
from app.kb.schema import GraphEdge, KBChunk

CORPUS_DIR = Path(__file__).parent / "corpus"

#: 每个库的必填字段。**只校验真正必需的**——把可选字段也列进来会逼着
#: 语料作者填无意义的占位值，反而降低数据质量。
REQUIRED: dict[Library, tuple[str, ...]] = {
    Library.PSYCHOLOGY: ("chunk_id", "discipline", "concept", "source", "author", "text"),
    Library.LITERATURE: ("chunk_id", "book", "character", "imagery", "emotion", "text"),
    Library.POETRY: ("chunk_id", "poem_title", "author", "dynasty", "imagery", "emotion", "text"),
}

_FILE_OF: dict[Library, str] = {
    Library.PSYCHOLOGY: "psychology.jsonl",
    Library.LITERATURE: "literature.jsonl",
    Library.POETRY: "poetry.jsonl",
}

GRAPH_EDGES_FILE = "graph_edges.jsonl"

#: 图里允许出现的节点类型。白名单而非自由字符串——Cypher 里标签不能参数化，
#: 只能拼字符串，没有白名单就是一个注入口子。
GRAPH_NODE_TYPES = frozenset({"Imagery", "Emotion", "Character", "Concept", "Chunk", "Work"})
GRAPH_RELATIONS = frozenset({"ASSOCIATED_WITH", "CO_OCCURS", "FEATURES_CHARACTER", "EVIDENCED_BY"})


class CorpusError(RuntimeError):
    """语料格式错误。消息里带文件与行号，直接可定位。"""


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    if not path.exists():
        raise CorpusError(f"语料文件不存在：{path}")
    rows: list[tuple[int, dict[str, Any]]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorpusError(f"{path.name}:{lineno} JSON 解析失败：{exc.msg}") from exc
        if not isinstance(obj, dict):
            raise CorpusError(f"{path.name}:{lineno} 每行必须是一个 JSON 对象")
        rows.append((lineno, obj))
    return rows


def _require_str(row: dict[str, Any], key: str, where: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CorpusError(f"{where} 字段 `{key}` 必须是非空字符串，实际是 {value!r}")
    return value


def _str_list(row: dict[str, Any], key: str, where: str, *, required: bool) -> list[str]:
    value = row.get(key)
    if value is None:
        if required:
            raise CorpusError(f"{where} 缺少字段 `{key}`")
        return []
    # 单个字符串自动升级成单元素列表：语料是手写的，只有一个意象时
    # 写成 "落花" 而不是 ["落花"] 太自然了，为此丢一条语料不值得。
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise CorpusError(f"{where} 字段 `{key}` 必须是字符串或字符串数组，实际是 {value!r}")
    return [v for v in (s.strip() for s in value) if v]


def load_library(library: Library, corpus_dir: Path | None = None) -> list[KBChunk]:
    """加载单个库，返回校验过的 chunk 列表。"""
    directory = corpus_dir or CORPUS_DIR
    path = directory / _FILE_OF[library]
    rows = _read_jsonl(path)
    required = REQUIRED[library]
    chunks: list[KBChunk] = []
    seen: dict[str, int] = {}

    for lineno, row in rows:
        where = f"{path.name}:{lineno}"
        for key in required:
            if key not in row:
                raise CorpusError(f"{where} 缺少必填字段 `{key}`")
        chunk_id = _require_str(row, "chunk_id", where)
        if chunk_id in seen:
            raise CorpusError(
                f"{where} chunk_id `{chunk_id}` 与第 {seen[chunk_id]} 行重复。"
                f" 重复 id 会让两条语料在 Qdrant 里互相覆盖，其中一条永远查不到。"
            )
        seen[chunk_id] = lineno
        if ":" not in chunk_id:
            raise CorpusError(f"{where} chunk_id `{chunk_id}` 应形如 `{{lib}}:{{slug}}:{{seq}}`")

        text = _require_str(row, "text", where)
        if len(text) < 8:
            raise CorpusError(f"{where} text 过短（{len(text)} 字），疑似误填")

        chunks.append(_to_chunk(library, chunk_id, text, row, where))

    return chunks


def _to_chunk(
    library: Library, chunk_id: str, text: str, row: dict[str, Any], where: str
) -> KBChunk:
    """把领域字段名映射成统一模型。映射表就是这一处，改格式只改这里。

    `where` 一路带下来（形如 `poetry.jsonl:42`），让**字段类型错误也报行号**。
    只报 chunk_id 的话，编辑器里还得先搜一遍才知道是第几行——
    报错信息的存在意义就是省掉这一步。
    """
    if library is Library.PSYCHOLOGY:
        source = row.get("source")
        source = source.strip() if isinstance(source, str) else None
        # 语料里的 source 自带书名号（《依恋与失落》第一卷·依恋），
        # 去掉后存进 work——展示层按需再加，避免出现《《…》》
        work = source.strip("《》") if source else None
        if work:
            work = work.split("·")[0].split("》")[0]
        return KBChunk(
            chunk_id=chunk_id,
            library=library,
            text=text,
            author=row.get("author"),
            year=row.get("year") if isinstance(row.get("year"), int) else None,
            work=work,
            discipline=row.get("discipline"),
            concept=row.get("concept"),
            keywords=_str_list(row, "keywords", where, required=False),
        )

    if library is Library.LITERATURE:
        return KBChunk(
            chunk_id=chunk_id,
            library=library,
            text=text,
            type=row.get("type"),
            work=row.get("book"),
            chapter=row.get("chapter"),
            character=row.get("character"),
            imagery=_str_list(row, "imagery", where, required=True),
            emotion=_str_list(row, "emotion", where, required=True),
            context=row.get("context"),
        )

    return KBChunk(
        chunk_id=chunk_id,
        library=library,
        text=text,
        type=row.get("type"),
        work=row.get("poem_title"),
        author=row.get("author"),
        dynasty=row.get("dynasty"),
        imagery=_str_list(row, "imagery", where, required=True),
        emotion=_str_list(row, "emotion", where, required=True),
    )


def load_all(corpus_dir: Path | None = None) -> dict[Library, list[KBChunk]]:
    """加载全部三个库。

    跨库查重：同一个 chunk_id 出现在两个库里，说明 slug 前缀写错了
    （`lit:` 打成 `poem:`）。查出来当场报错，别等 Qdrant 里少一条再回头找。
    """
    directory = corpus_dir or CORPUS_DIR
    result = {lib: load_library(lib, directory) for lib in Library}
    owner: dict[str, Library] = {}
    for lib, chunks in result.items():
        for chunk in chunks:
            if chunk.chunk_id in owner:
                raise CorpusError(
                    f"chunk_id `{chunk.chunk_id}` 同时出现在 {owner[chunk.chunk_id]} 与 {lib} 中"
                )
            owner[chunk.chunk_id] = lib
    return result


def load_graph_edges(corpus_dir: Path | None = None) -> list[GraphEdge]:
    directory = corpus_dir or CORPUS_DIR
    path = directory / GRAPH_EDGES_FILE
    edges: list[GraphEdge] = []
    for lineno, row in _read_jsonl(path):
        where = f"{path.name}:{lineno}"
        for key in ("source", "source_type", "target", "target_type", "relation"):
            _require_str(row, key, where)
        source_type = row["source_type"]
        target_type = row["target_type"]
        relation = row["relation"]
        if source_type not in GRAPH_NODE_TYPES or target_type not in GRAPH_NODE_TYPES:
            raise CorpusError(
                f"{where} 节点类型 {source_type}/{target_type} 不在白名单内"
                f"（{sorted(GRAPH_NODE_TYPES)}）"
            )
        if relation not in GRAPH_RELATIONS:
            raise CorpusError(f"{where} 关系 `{relation}` 不在白名单内（{sorted(GRAPH_RELATIONS)}）")
        weight = row.get("weight", 1.0)
        if not isinstance(weight, (int, float)):
            raise CorpusError(f"{where} weight 必须是数字，实际是 {weight!r}")
        edges.append(
            GraphEdge(
                source=row["source"],
                source_type=source_type,
                target=row["target"],
                target_type=target_type,
                relation=relation,
                weight=float(weight),
            )
        )
    return edges
