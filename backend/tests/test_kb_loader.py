"""语料加载与差分。

**这里断言的核心只有一句：坏语料必须炸，而且炸在正确的位置上。**
语料加载层的失败模式是「静默少几条」——跳过一行坏数据，命令照样成功，
只是知识库里永远查不到那条典故。这类 bug 不会有人发现，所以它必须
由测试守着，而不是靠 code review 时的注意力。

第二组是 content_hash 差分：改一条只应重嵌一条。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.constants import Library
from app.kb.embedder import content_hash
from app.kb.loader import CorpusError, load_all, load_graph_edges, load_library
from app.kb.schema import KBChunk
from app.providers.mock_embedding import MockEmbeddingModel

CORPUS = Path(__file__).resolve().parents[1] / "app" / "kb" / "corpus"

PSY: dict[str, Any] = {
    "chunk_id": "psy:test:001",
    "discipline": "依恋理论",
    "concept": "安全基地",
    "source": "《依恋》第一卷·导论",
    "author": "John Bowlby",
    "text": "这是一条用于测试的心理学语料，长度足够。",
}
LIT: dict[str, Any] = {
    "chunk_id": "lit:test:001",
    "book": "红楼梦",
    "chapter": "第二十七回",
    "character": "林黛玉",
    "imagery": ["落花"],
    "emotion": ["哀伤"],
    "text": "花谢花飞花满天，红消香断有谁怜？",
}
POEM: dict[str, Any] = {
    "chunk_id": "poem:test:001",
    "poem_title": "锦瑟",
    "author": "李商隐",
    "dynasty": "唐",
    "imagery": ["明月"],
    "emotion": ["哀伤"],
    "text": "锦瑟无端五十弦，一弦一柱思华年。",
}


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """一份最小可用的三库语料，每库一条。"""
    (tmp_path / "psychology.jsonl").write_text(json.dumps(PSY, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "literature.jsonl").write_text(json.dumps(LIT, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "poetry.jsonl").write_text(json.dumps(POEM, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "graph_edges.jsonl").write_text(
        json.dumps(
            {
                "source": "落花",
                "source_type": "Imagery",
                "target": "哀伤",
                "target_type": "Emotion",
                "relation": "ASSOCIATED_WITH",
                "weight": 0.9,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return tmp_path


def _rewrite(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )


class TestHardFailures:
    """坏语料一律抛 CorpusError，消息里带文件与行号。"""

    def test_broken_json_reports_line(self, corpus: Path) -> None:
        path = corpus / "psychology.jsonl"
        path.write_text(
            json.dumps(PSY, ensure_ascii=False) + "\n" + '{"chunk_id": "x",\n',
            encoding="utf-8",
        )
        with pytest.raises(CorpusError, match=r"psychology\.jsonl:2"):
            load_library(Library.PSYCHOLOGY, corpus)

    def test_missing_required_field_reports_line(self, corpus: Path) -> None:
        row = {k: v for k, v in PSY.items() if k != "discipline"}
        _rewrite(corpus / "psychology.jsonl", [PSY, row])
        with pytest.raises(CorpusError, match=r"psychology\.jsonl:2 缺少必填字段 `discipline`"):
            load_library(Library.PSYCHOLOGY, corpus)

    def test_wrong_type_reports_line_not_just_id(self, corpus: Path) -> None:
        # 类型错误发生在字段映射阶段，也必须带上行号——只给 chunk_id 的话
        # 还得回编辑器里再搜一遍才知道是第几行。
        # chunk_id 必须换掉：同 id 会让重复检查先命中，测不到类型分支
        row = dict(LIT, chunk_id="lit:test:002", imagery={"落花": 1})
        _rewrite(corpus / "literature.jsonl", [LIT, row])
        with pytest.raises(CorpusError, match=r"literature\.jsonl:2.*`imagery`"):
            load_library(Library.LITERATURE, corpus)

    def test_duplicate_chunk_id_fails(self, corpus: Path) -> None:
        # 重复 id 会让两条语料在 Qdrant 里互相覆盖——其中一条永远查不到。
        _rewrite(corpus / "poetry.jsonl", [POEM, dict(POEM, text="另一条同 id 的诗句文本")])
        with pytest.raises(CorpusError, match="chunk_id `poem:test:001` 与第 1 行重复"):
            load_library(Library.POETRY, corpus)

    def test_missing_file_fails(self, tmp_path: Path) -> None:
        with pytest.raises(CorpusError, match="语料文件不存在"):
            load_library(Library.POETRY, tmp_path)

    def test_unknown_graph_relation_fails(self, corpus: Path) -> None:
        path = corpus / "graph_edges.jsonl"
        path.write_text(
            path.read_text(encoding="utf-8").replace("ASSOCIATED_WITH", "DROP TABLE"),
            encoding="utf-8",
        )
        # 关系名要拼进 Cypher，白名单是唯一防线
        with pytest.raises(CorpusError, match="关系"):
            load_graph_edges(corpus)

    def test_duplicate_across_libraries_fails(self, corpus: Path) -> None:
        _rewrite(corpus / "literature.jsonl", [dict(LIT, chunk_id=POEM["chunk_id"])])
        with pytest.raises(CorpusError, match="同时出现在"):
            load_all(corpus)


class TestTolerance:
    """该宽容的地方要宽容——手写语料常见的无害写法不该被判死。"""

    def test_single_string_upgraded_to_list(self, corpus: Path) -> None:
        _rewrite(corpus / "literature.jsonl", [dict(LIT, imagery="落花", emotion="哀伤")])
        chunk = load_library(Library.LITERATURE, corpus)[0]
        assert chunk.imagery == ["落花"]
        assert chunk.emotion == ["哀伤"]

    def test_blank_lines_and_comments_skipped(self, corpus: Path) -> None:
        path = corpus / "poetry.jsonl"
        path.write_text(
            "\n// 这是一行注释\n" + json.dumps(POEM, ensure_ascii=False) + "\n\n", encoding="utf-8"
        )
        assert len(load_library(Library.POETRY, corpus)) == 1

    def test_psychology_source_becomes_work_without_brackets(self, corpus: Path) -> None:
        chunk = load_library(Library.PSYCHOLOGY, corpus)[0]
        # 语料写的是「《依恋》第一卷·导论」，落库应是干净的书名
        assert chunk.work == "依恋"


class TestRealCorpus:
    """随仓库交付的语料必须始终可用——它是无密钥演示的全部依据。"""

    def test_all_libraries_load(self) -> None:
        data = load_all()
        assert len(data[Library.PSYCHOLOGY]) >= 50
        assert len(data[Library.LITERATURE]) >= 30
        assert len(data[Library.POETRY]) >= 40

    def test_every_chunk_has_imagery_or_concept(self) -> None:
        # 缺标注的语料检索不到（嵌入文本靠标签把短查询接上长原文），
        # 等于往库里灌了一条永远查不到的数据。
        for lib, chunks in load_all().items():
            for chunk in chunks:
                assert chunk.imagery or chunk.emotion or chunk.concept, (
                    f"{lib} 的 {chunk.chunk_id} 既无意象/情感也无概念，检索不到它"
                )

    def test_graph_edges_reference_known_vocabulary(self) -> None:
        # 聚合边里的意象/情绪标签必须真的出现在某条语料里，否则那条边
        # 在检索时永远接不上语料——图上看着有，实际是死路。
        data = load_all()
        imagery = {i for chunks in data.values() for c in chunks for i in c.imagery}
        emotion = {e for chunks in data.values() for c in chunks for e in c.emotion}
        for edge in load_graph_edges():
            if edge.source_type == "Imagery" and edge.relation == "ASSOCIATED_WITH":
                assert edge.source in imagery, f"意象「{edge.source}」不在任何语料里"
                assert edge.target in emotion, f"情绪「{edge.target}」不在任何语料里"


class TestContentHash:
    """差分是「改一条只重嵌一条」的全部依据。"""

    def test_same_chunk_same_hash(self) -> None:
        model = MockEmbeddingModel(dim=64)
        a = KBChunk(chunk_id="x", library=Library.POETRY, text="甲", imagery=["月"])
        b = KBChunk(chunk_id="x", library=Library.POETRY, text="甲", imagery=["月"])
        assert content_hash(a, model) == content_hash(b, model)

    def test_tag_change_changes_hash(self) -> None:
        # 原文一个字没动，但标注变了 → 向量必须重算。
        # 嵌入文本里含标签，所以 hash 也必须跟着变。
        model = MockEmbeddingModel(dim=64)
        a = KBChunk(chunk_id="x", library=Library.POETRY, text="甲", imagery=["月"])
        b = KBChunk(chunk_id="x", library=Library.POETRY, text="甲", imagery=["月", "花"])
        assert content_hash(a, model) != content_hash(b, model)

    def test_model_identity_changes_hash(self) -> None:
        # 换 embedding 模型后旧向量不再可比。把模型身份拌进 hash，
        # 换模型自动触发全量重嵌，而不是新旧向量混在一起慢慢坏掉。
        chunk = KBChunk(chunk_id="x", library=Library.POETRY, text="甲")
        assert content_hash(chunk, MockEmbeddingModel(dim=64)) != content_hash(
            chunk, MockEmbeddingModel(dim=128)
        )
