"""故事工坊的工具：边界、截断、失败信息。

每个工具**只 monkeypatch 它自己的那一个边界**（`qdrant_index.search` /
`neo4j_index.*` / `kb_import_service.*` / `novel.service.*`），
这样测试断言的是「我们怎么用那个边界」，而不是「边界自己对不对」——
后者已经由 test_n5_retrieval / test_kb_import / test_novel_mcp 管着了。

两条最容易在真机上坏掉、而 mock 路径看不见的事，各有一个用例：

1. **图检索必须回 Qdrant 取正文。** Neo4j 里只有 60 字 snippet，少了那一跳，
   卡片上永远只有半句话——而它看起来完全正常。
2. **失败要带可用清单。** 模型看不到文件系统也看不到表结构，只说「失败」
   等于让它下一轮再撞一次。
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.agent.tools import (
    MAX_RESULT_CHARS,
    ToolRegistry,
    build_registry,
    parse_arguments,
)
from app.constants import Library
from app.providers.base import ToolCall

INPUT = "妈妈走了以后我才发现，她留下的那件毛衣我一直没舍得洗。想她的时候我就抱着坐一会儿。" * 3


def registry(*, allow_novel: bool = True) -> ToolRegistry:
    return build_registry(input_text=INPUT, allow_novel=allow_novel)


def call(name: str, **args: Any) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=json.dumps(args, ensure_ascii=False))


def point(text: str, *, work: str = "饮水词", author: str = "纳兰性德", score: float = 0.21, cid: str = "abc12345"):
    """一个假的 ScoredPoint。字段名按 `KBChunk.payload()`。"""
    return SimpleNamespace(
        score=score,
        payload={
            "chunk_id": cid,
            "library": str(Library.POETRY),
            "text": text,
            "work": work,
            "author": author,
            "origin": "纳兰词选.txt",
        },
    )


def record(text: str, *, cid: str = "abc12345", work: str = "饮水词"):
    return SimpleNamespace(payload={"chunk_id": cid, "text": text, "work": work, "author": "纳兰性德"})


# ---------------------------------------------------------------- 参数


def test_参数解析_空字符串等价于空对象():
    """无参工具在 DeepSeek 上会回 `arguments=""`——这里统一成 `{}`。"""
    assert parse_arguments(ToolCall(id="1", name="x", arguments="")) == ({}, None)
    assert parse_arguments(ToolCall(id="1", name="x", arguments="  ")) == ({}, None)


def test_参数解析_坏JSON保留原文给模型看():
    args, err = parse_arguments(ToolCall(id="1", name="x", arguments="query=想念"))
    assert args == {}
    assert err and "不是合法 JSON" in err
    assert "query=想念" in err


def test_参数解析_数组不是对象():
    _args, err = parse_arguments(ToolCall(id="1", name="x", arguments="[1,2]"))
    assert err and "JSON 对象" in err


async def test_缺必填参数返回文本不抛():
    reg = registry()
    out = await reg.dispatch(call("kb_search"))

    assert out.ok is False
    assert out.reason == "bad_arguments"
    assert "缺少必填参数" in out.text
    assert "query" in out.text


# ---------------------------------------------------------------- 检索


async def test_kb_search_不受低分影响并带上出处(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, Any] = {}

    class FakeEmbedding:
        async def embed(self, texts):
            seen["texts"] = list(texts)
            return [[0.1, 0.2, 0.3]]

    async def fake_search(vector, *, library=None, limit=8, **kw):
        seen["vector"] = vector
        seen["library"] = library
        seen["limit"] = limit
        seen["threshold"] = kw.get("score_threshold")
        return [point("被酒莫惊春睡重，赌书消得泼茶香。")]

    monkeypatch.setattr("app.providers.factory.get_embedding_model", lambda **kw: FakeEmbedding())
    monkeypatch.setattr("app.kb.qdrant_index.search", fake_search)

    out = await registry().dispatch(call("kb_search", query="想念却说不出口", limit=3))

    assert out.ok
    assert seen["texts"] == ["想念却说不出口"]  # 查的就是模型写的那句话
    assert seen["limit"] == 3
    # 不设阈值：本机实测向量分只有 0.06–0.29，设 0.5 会把命中全切光。
    assert seen["threshold"] is None
    assert "《饮水词》" in out.text and "纳兰性德" in out.text
    assert "赌书消得泼茶香" in out.text


async def test_kb_search_指定库时透传枚举(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, Any] = {}

    class FakeEmbedding:
        async def embed(self, texts):
            return [[0.0]]

    async def fake_search(vector, *, library=None, limit=8, **kw):
        seen["library"] = library
        return []

    monkeypatch.setattr("app.providers.factory.get_embedding_model", lambda **kw: FakeEmbedding())
    monkeypatch.setattr("app.kb.qdrant_index.search", fake_search)

    out = await registry().dispatch(call("kb_search", query="x", library="poetry"))
    assert seen["library"] is Library.POETRY
    assert "没有检索到" in out.text


async def test_库名写错时不报错只是不限库(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, Any] = {}

    class FakeEmbedding:
        async def embed(self, texts):
            return [[0.0]]

    async def fake_search(vector, *, library=None, limit=8, **kw):
        seen["library"] = library
        return []

    monkeypatch.setattr("app.providers.factory.get_embedding_model", lambda **kw: FakeEmbedding())
    monkeypatch.setattr("app.kb.qdrant_index.search", fake_search)

    await registry().dispatch(call("kb_search", query="x", library="我瞎写的"))
    assert seen["library"] is None


async def test_kb_graph_必须回取正文而不是只给snippet(monkeypatch: pytest.MonkeyPatch):
    """图库只存 60 字摘要，正文必须回 Qdrant 取——少了这一跳卡片上只有半句话。"""
    asked: dict[str, Any] = {}
    full_text = "原来姹紫嫣红开遍，似这般都付与断井颓垣。" * 3

    async def fake_emotion(emotion, *, limit=12):
        asked["emotion"] = emotion
        return [{"chunk_id": "abc12345", "snippet": "原来姹紫嫣红开遍", "library": "literature", "weight": 2.0}]

    async def fake_fetch(ids, **kw):
        asked["ids"] = list(ids)
        return [record(full_text)]

    monkeypatch.setattr("app.kb.neo4j_index.chunks_by_emotion", fake_emotion)
    monkeypatch.setattr("app.kb.qdrant_index.fetch_by_ids", fake_fetch)

    out = await registry().dispatch(call("kb_graph", mode="emotion", key="哀伤"))

    assert asked["emotion"] == "哀伤"
    assert asked["ids"] == ["abc12345"]
    assert out.ok and "断井颓垣" in out.text


async def test_kb_graph_意象走另一条边(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, Any] = {}

    async def fake_imagery(imagery, *, limit=12, **kw):
        seen["imagery"] = imagery
        return []

    monkeypatch.setattr("app.kb.neo4j_index.chunks_by_imagery", fake_imagery)
    out = await registry().dispatch(call("kb_graph", mode="imagery", key="旧毛衣"))

    assert seen["imagery"] == "旧毛衣"
    assert "没有和意象" in out.text
    assert "换一个更常见的词" in out.text  # 失败信息里带下一步


async def test_kb_graph_模式写错算参数问题():
    out = await registry().dispatch(call("kb_graph", mode="心情", key="哀伤"))
    assert out.ok is False and out.reason == "bad_arguments"


# ---------------------------------------------------------------- 来源


async def test_kb_sources_列出文件与段数(monkeypatch: pytest.MonkeyPatch):
    async def fake_list():
        return [
            SimpleNamespace(source_file="纳兰词选.txt", library="poetry", chunks=42, updated_at=None),
            SimpleNamespace(source_file="红楼梦.txt", library="literature", chunks=7702, updated_at=None),
        ]

    monkeypatch.setattr("app.services.kb_import_service.list_sources", fake_list)
    out = await registry().dispatch(call("kb_sources"))

    assert out.ok
    assert "2 个来源" in out.text
    assert "纳兰词选.txt" in out.text and "42" in out.text
    assert "诗词" in out.text  # 库里给人看的是中文标签


async def test_kb_source_text_读不到时带可用清单(monkeypatch: pytest.MonkeyPatch):
    from app.services.kb_import_service import ImportNotFound

    async def fake_text(source_file, *, offset=0, limit=100_000):
        raise ImportNotFound("没有这个来源")

    async def fake_list():
        return [SimpleNamespace(source_file="纳兰词选.txt", library="poetry", chunks=42, updated_at=None)]

    monkeypatch.setattr("app.services.kb_import_service.source_text", fake_text)
    monkeypatch.setattr("app.services.kb_import_service.list_sources", fake_list)

    out = await registry().dispatch(call("kb_source_text", source_file="不存在的.txt"))

    assert out.ok is False and out.reason == "failed"
    assert "没有名为" in out.text
    assert "纳兰词选.txt" in out.text  # ← 这段就是模型自纠所需的全部信息


async def test_kb_source_text_正常读取带字数(monkeypatch: pytest.MonkeyPatch):
    async def fake_text(source_file, *, offset=0, limit=100_000):
        return SimpleNamespace(
            source_file=source_file, title="纳兰词选", author="纳兰性德", library="poetry",
            char_count=5000, offset=offset, truncated=True, text="人生若只如初见。",
        )

    monkeypatch.setattr("app.services.kb_import_service.source_text", fake_text)
    out = await registry().dispatch(call("kb_source_text", source_file="纳兰词选.txt", offset=100))

    assert out.ok
    assert "《纳兰词选》" in out.text and "全文 5000 字" in out.text
    assert "可继续读" in out.text


# ---------------------------------------------------------------- 材料


async def test_read_input_分段读并说清还剩多少():
    reg = registry()
    out = await reg.dispatch(call("read_input", offset=0, limit=20))

    assert out.ok
    assert out.text.splitlines()[0].startswith("原文第 0–20 字")
    assert f"共 {len(INPUT)} 字" in out.text
    assert "后面还有" in out.text


async def test_read_input_越界只说明不报错():
    out = await registry().dispatch(call("read_input", offset=99999))
    assert out.ok and "已经超出末尾" in out.text


# ---------------------------------------------------------------- 古典文学


async def test_关掉开关时工具根本不进schema():
    reg = registry(allow_novel=False)
    assert "novel_lookup" not in reg.names()
    assert all(s["function"]["name"] != "novel_lookup" for s in reg.schemas())


async def test_关掉开关时调用它算未知工具():
    out = await registry(allow_novel=False).dispatch(call("novel_lookup", action="books"))
    assert out.ok is False and out.reason == "unknown"


async def test_novel_透传markdown不解析(monkeypatch: pytest.MonkeyPatch):
    md = "# 书目\n\n- 《红楼梦》 `book_id=2` 119 回\n"

    async def fake_books(*, query=None, limit=20, offset=0):
        return SimpleNamespace(tool="novel_list_books", markdown=md)

    monkeypatch.setattr("app.novel.service.books", fake_books)
    out = await registry().dispatch(call("novel_lookup", action="books"))

    assert out.ok and out.text == md  # 原样透出，一个字符都不改


async def test_novel_缺id时告诉它去哪拿():
    out = await registry().dispatch(call("novel_lookup", action="dialogues"))
    assert out.ok is False and out.reason == "bad_arguments"
    assert "character_id" in out.text
    assert "action=characters" in out.text


async def test_novel_服务没连上时是红色卡片且劝它换路(monkeypatch: pytest.MonkeyPatch):
    from app.novel.mcp_client import NovelMcpError

    async def fake_books(*, query=None, limit=20, offset=0):
        raise NovelMcpError("古典文学 MCP 没连上。", kind="unavailable")

    monkeypatch.setattr("app.novel.service.books", fake_books)
    out = await registry().dispatch(call("novel_lookup", action="books"))

    assert out.ok is False and out.reason == "failed"
    assert "unavailable" in out.text
    assert "不要再调它" in out.text


async def test_novel_超时建议重试一次(monkeypatch: pytest.MonkeyPatch):
    from app.novel.mcp_client import NovelMcpError

    async def fake_chapters(*, book_id, query=None, limit=100, offset=0):
        raise NovelMcpError("超过 30 秒没有返回。", kind="timeout")

    monkeypatch.setattr("app.novel.service.chapters", fake_chapters)
    out = await registry().dispatch(call("novel_lookup", action="chapters", book_id=2))

    assert out.ok is False
    assert "重试一次" in out.text


# ---------------------------------------------------------------- 截断


async def test_单条结果截断并显式标注(monkeypatch: pytest.MonkeyPatch):
    async def fake_list():
        return [
            SimpleNamespace(source_file=f"书{i}.txt", library="poetry", chunks=i, updated_at=None)
            for i in range(200)
        ]

    monkeypatch.setattr("app.services.kb_import_service.list_sources", fake_list)
    out = await registry().dispatch(call("kb_sources"))

    assert len(out.text) <= MAX_RESULT_CHARS + len("…（已截断）")
    assert out.text.endswith("（已截断）")
    assert len(out.preview) <= 400 + len("…（已截断）")


async def test_逐条摘录也截断(monkeypatch: pytest.MonkeyPatch):
    class FakeEmbedding:
        async def embed(self, texts):
            return [[0.0]]

    async def fake_search(vector, *, library=None, limit=8, **kw):
        return [point("句子。" * 200)]

    monkeypatch.setattr("app.providers.factory.get_embedding_model", lambda **kw: FakeEmbedding())
    monkeypatch.setattr("app.kb.qdrant_index.search", fake_search)

    out = await registry().dispatch(call("kb_search", query="x"))
    assert "句子。" * 200 not in out.text
    assert "…" in out.text


# ---------------------------------------------------------------- 注册表


async def test_撤销之后调用它变成未知工具():
    reg = registry()
    reg.retire("kb_search")
    assert "kb_search" not in reg.names()
    assert all(s["function"]["name"] != "kb_search" for s in reg.schemas())

    out = await reg.dispatch(call("kb_search", query="x"))
    assert out.ok is False and out.reason == "unknown"
    assert "当前可用的工具" in out.text


def test_schema_是openai原样形状():
    schema = registry().get("kb_search").schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "kb_search"
    assert schema["function"]["parameters"]["required"] == ["query"]
