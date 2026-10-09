"""古典文学 MCP 客户端：工具名、入参、Markdown 透传、失败分档。

被 stub 掉的是 `NovelMcpClient.call` —— **对外唯一的边界**。它下面（stdio 会话、
`session.call_tool`）是 SDK 和子进程，上面（`service`、路由）才是本仓的代码；
前者只能真起一个进程去验（见文件末尾的 live 用例），后者用 stub 就能精确钉住。

**为什么入参要逐条断言**：对方六个工具的入参是各自的契约，`chapter_idx`（库内
章序号）与 `global_idx`（全书段序）名字相近但坐标系不同，而且 `_args()` 会悄悄
丢掉 `None`。写错一个键，在参数校验宽松的工具上不会报错，返回的是一份看着正常
的错结果——这类错误只能靠把参数钉死来防。
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import pytest

from app.api.v1 import novel as novel_api
from app.config import Settings
from app.novel import mcp_client as mod
from app.novel import service as svc
from app.novel.mcp_client import (
    TOOLS,
    NovelMcpClient,
    NovelMcpError,
    _parse_args,
    _text_of,
)

#: 对方渲染层的产物长这样：**Markdown，不是 JSON**。原样透传是契约。
MARKDOWN = "- 《红楼梦》 · `book_id=2` · 120 回 · 1,078,715 字"


# ----------------------------------------------------------------------
# 夹具
# ----------------------------------------------------------------------


def make_client(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> NovelMcpClient:
    """用代码默认值构造客户端，绕开本机 .env —— 换台机器结果要一样。"""
    monkeypatch.setattr(mod, "get_settings", lambda: Settings(_env_file=None, **overrides))
    return NovelMcpClient()


def stub_call(
    monkeypatch: pytest.MonkeyPatch,
    *,
    markdown: str = MARKDOWN,
    error: Exception | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """替换 service 手上那个客户端的 `call`，记录每次 (工具名, 入参)。"""
    calls: list[tuple[str, dict[str, Any]]] = []
    client = svc.get_client()

    async def fake_call(tool: str, arguments: dict[str, Any]) -> str:
        calls.append((tool, arguments))
        if error is not None:
            raise error
        return markdown

    monkeypatch.setattr(client, "call", fake_call)
    return calls


def result_of(*texts: str, **flags: Any) -> SimpleNamespace:
    """伪造 `CallToolResult`。content 块只需要有 `.text`。"""
    return SimpleNamespace(content=[SimpleNamespace(text=t) for t in texts], **flags)


# ----------------------------------------------------------------------
# service → 工具名与入参
# ----------------------------------------------------------------------


class TestServiceArguments:
    async def test_书目传查询与分页并原样带回_markdown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = stub_call(monkeypatch)

        out = await svc.books(query="红楼", limit=5, offset=10)

        assert calls == [(TOOLS["books"], {"query": "红楼", "limit": 5, "offset": 10})]
        assert out.tool == TOOLS["books"]
        # 一个字节都不动：解析它等于把对方的展示层当 API 用
        assert out.markdown == MARKDOWN

    async def test_可选参数为_None_时不进参数表(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = stub_call(monkeypatch)

        await svc.books()

        # `offset=0` 必须留着——用 `or` 链筛空值时 0 会被一起吞掉，
        # 表现是翻页永远回到第一页。
        assert calls == [(TOOLS["books"], {"limit": 20, "offset": 0})]

    async def test_人物带书名限定(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = stub_call(monkeypatch)

        await svc.characters(name="宝玉", book_id=2, limit=3)

        assert calls == [(TOOLS["characters"], {"name": "宝玉", "book_id": 2, "limit": 3})]

    async def test_人物不限书时不传_book_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = stub_call(monkeypatch)

        await svc.characters(name="宝玉")

        assert calls == [(TOOLS["characters"], {"name": "宝玉", "limit": 20})]

    async def test_对话传全部筛选条件(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = stub_call(monkeypatch)

        await svc.dialogues(
            character_id=7,
            book_id=2,
            chapter_from=3,
            chapter_to=9,
            query="好",
            limit=10,
            cursor=40,
        )

        assert calls == [
            (
                TOOLS["dialogues"],
                {
                    "character_id": 7,
                    "book_id": 2,
                    "chapter_from": 3,
                    "chapter_to": 9,
                    "query": "好",
                    "limit": 10,
                    "cursor": 40,
                },
            )
        ]

    async def test_对话首页不传游标(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = stub_call(monkeypatch)

        await svc.dialogues(character_id=7)

        assert calls == [(TOOLS["dialogues"], {"character_id": 7, "limit": 20})]

    async def test_回目表(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = stub_call(monkeypatch)

        await svc.chapters(book_id=2, query="甄士隐", limit=20, offset=40)

        assert calls == [
            (TOOLS["chapters"], {"book_id": 2, "query": "甄士隐", "limit": 20, "offset": 40})
        ]

    async def test_整章原文首页不传游标(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = stub_call(monkeypatch)

        await svc.chapter_text(book_id=2, chapter_idx=8, limit=3)

        assert calls == [
            (TOOLS["chapter_text"], {"book_id": 2, "chapter_idx": 8, "limit": 3})
        ]

    async def test_整章原文的游标是段序号不是页号(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`cursor` 直接就是上一页最后一段的 `passage.idx`，原样回传。"""
        calls = stub_call(monkeypatch)

        await svc.chapter_text(book_id=2, chapter_idx=8, cursor=79)

        assert calls == [
            (TOOLS["chapter_text"], {"book_id": 2, "chapter_idx": 8, "cursor": 79, "limit": 80})
        ]

    async def test_段落原文的前后段数原样传(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = stub_call(monkeypatch)

        await svc.passage_text(book_id=2, global_idx=1234, before=1, after=3)

        assert calls == [
            (TOOLS["passage_text"], {"book_id": 2, "global_idx": 1234, "before": 1, "after": 3})
        ]

    async def test_失败原样往上抛不被_service_吞掉(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        boom = NovelMcpError("对方库锁住了", kind="timeout")
        stub_call(monkeypatch, error=boom)

        with pytest.raises(NovelMcpError) as exc:
            await svc.chapter_text(book_id=2, chapter_idx=1)

        # kind 是路由决定状态码的唯一依据，翻包一次就可能丢掉
        assert exc.value is boom
        assert exc.value.kind == "timeout"


# ----------------------------------------------------------------------
# 起子进程之前的前置检查
# ----------------------------------------------------------------------


class TestPrerequisites:
    def test_关掉开关时说明原因(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reason = make_client(monkeypatch, NOVEL_MCP_ENABLED=False).prerequisites()

        assert reason is not None and "NOVEL_MCP_ENABLED" in reason

    def test_解释器不存在时点名路径(self, monkeypatch: pytest.MonkeyPatch) -> None:
        missing = "D:/nope/python.exe"

        reason = make_client(monkeypatch, NOVEL_MCP_COMMAND=missing).prerequisites()

        assert reason is not None and missing in reason

    def test_对方仓库目录不存在时点名路径(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        exe = tmp_path / "python.exe"
        exe.write_bytes(b"")
        missing_dir = str(tmp_path / "没有这个仓")

        reason = make_client(
            monkeypatch, NOVEL_MCP_COMMAND=str(exe), NOVEL_MCP_CWD=missing_dir
        ).prerequisites()

        assert reason is not None and missing_dir in reason

    def test_http_模式缺_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reason = make_client(
            monkeypatch, NOVEL_MCP_TRANSPORT="http", NOVEL_MCP_URL=""
        ).prerequisites()

        assert reason is not None and "NOVEL_MCP_URL" in reason

    def test_一切就位时返回_None(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
        exe = tmp_path / "python.exe"
        exe.write_bytes(b"")

        client = make_client(
            monkeypatch, NOVEL_MCP_COMMAND=str(exe), NOVEL_MCP_CWD=str(tmp_path)
        )

        assert client.prerequisites() is None

    async def test_起不来时不_spawn_就报_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """起不来是配置问题：给 503（不可重试），别让它伪装成「对方拒绝了」。"""
        client = make_client(monkeypatch, NOVEL_MCP_COMMAND="D:/nope/python.exe")

        with pytest.raises(NovelMcpError) as exc:
            await client.call(TOOLS["books"], {})

        assert exc.value.kind == "unavailable"
        assert exc.value.retryable is False

    async def test_状态探测不抛异常只给原因(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """前端开弹窗时先看这一眼，它必须永远有返回值。"""
        health = await make_client(monkeypatch, NOVEL_MCP_ENABLED=False).health()

        assert health["ok"] is False
        assert health["tools"] == []
        assert "NOVEL_MCP_ENABLED" in (health["error"] or "")
        assert health["hint"]  # 附上「怎么修」，用户看到的是这句话


# ----------------------------------------------------------------------
# CallToolResult → Markdown
# ----------------------------------------------------------------------


class TestTextOf:
    def test_多个文本块按换行拼接(self) -> None:
        assert _text_of(result_of("第一行", "第二行"), "t") == "第一行\n第二行"

    def test_is_error_时把对方的中文说明原样带出来(self) -> None:
        """对方把业务失败折成 is_error + 一段说明，那段话比我们的转述具体。"""
        said = "book_id=9 不存在，用 novel_list_books 取有效值。"
        with pytest.raises(NovelMcpError) as exc:
            _text_of(result_of(said, is_error=True), "t")

        assert "book_id=9 不存在" in str(exc.value)
        assert exc.value.kind == "failed"

    def test_兼容驼峰_isError(self) -> None:
        """mcp 1.x 是 `isError`，2.x 是 `is_error`。两个都认，升级不会静默变哑。"""
        with pytest.raises(NovelMcpError):
            _text_of(result_of("出错了", isError=True), "t")

    def test_没有文本块时报错(self) -> None:
        with pytest.raises(NovelMcpError) as exc:
            _text_of(SimpleNamespace(content=[]), "novel_list_books")

        assert "novel_list_books" in str(exc.value)


# ----------------------------------------------------------------------
# 路由：错误分档与参数配对
# ----------------------------------------------------------------------


class TestRoutes:
    @pytest.mark.parametrize(
        ("kind", "status"),
        [("unavailable", 503), ("timeout", 504), ("failed", 502), ("没见过的档位", 502)],
    )
    def test_kind_决定状态码(self, kind: str, status: int) -> None:
        exc = NovelMcpError("坏掉了", kind=kind)

        assert novel_api._http(exc).status_code == status
        assert novel_api._http(exc).detail == "坏掉了"

    async def test_返回体只有工具名与_markdown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub_call(monkeypatch)

        payload = await novel_api.novel_books(query=None, limit=20, offset=0)

        assert payload == {"tool": TOOLS["books"], "markdown": MARKDOWN}

    async def test_起止章号成对时都传下去(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = stub_call(monkeypatch)

        await novel_api.novel_dialogues(
            character_id=7,
            book_id=2,
            chapter_from=3,
            chapter_to=9,
            query="好",
            limit=20,
            cursor=None,
        )

        assert calls == [
            (
                TOOLS["dialogues"],
                {
                    "character_id": 7,
                    "book_id": 2,
                    "chapter_from": 3,
                    "chapter_to": 9,
                    "query": "好",
                    "limit": 20,
                },
            )
        ]

    async def test_起止章号只给一个就两个都不传(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """对方的契约是成对才生效。只给一个的话，悄悄补默认值会静默改变筛选范围。"""
        calls = stub_call(monkeypatch)

        await novel_api.novel_dialogues(
            character_id=7,
            book_id=None,
            chapter_from=5,
            chapter_to=None,
            query=None,
            limit=20,
            cursor=None,
        )

        assert calls == [(TOOLS["dialogues"], {"character_id": 7, "limit": 20})]


# ----------------------------------------------------------------------
# 参数解析
# ----------------------------------------------------------------------


class TestParseArgs:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ('["-m","app.mcp.server"]', ["-m", "app.mcp.server"]),
            ("", []),
            ("-m app.mcp.server", ["-m", "app.mcp.server"]),  # 不是 JSON，按空格切
            ('{"module":"app.mcp.server"}', []),  # 是 JSON 但不是数组
        ],
    )
    def test_解析(self, raw: str, expected: list[str]) -> None:
        assert _parse_args(raw) == expected


# ----------------------------------------------------------------------
# 真起子进程
# ----------------------------------------------------------------------


class TestLive:
    """真起 stdio 子进程跑一遍。

    **前提不满足就 skip**：对方仓库、它的 conda 解释器、`data/novel.db` 都是
    另一台机器上可能不存在的东西，而这个测试的价值恰恰是「本机真能连通」——
    把它做成硬失败只会让换台机器的人去删测试。
    """

    async def test_书目到整章原文走一遍(self) -> None:
        client = svc.get_client()
        reason = client.prerequisites()
        if reason:
            pytest.skip(f"本机没有可用的古典文学 MCP：{reason}")

        books = await svc.books(limit=5)
        assert books.markdown.strip()
        found = re.search(r"book_id=(\d+)", books.markdown)
        if not found:
            pytest.skip("对方库里还没有书（未导入），无可读的原文")
        book_id = int(found.group(1))

        chapters = await svc.chapters(book_id=book_id, limit=3)
        assert chapters.markdown.strip()
        first = re.search(r"chapter_idx=(\d+)", chapters.markdown)
        if not first:
            pytest.skip(f"book_id={book_id} 没有回目，读不到整章原文")

        text = await svc.chapter_text(book_id=book_id, chapter_idx=int(first.group(1)), limit=3)

        assert text.tool == TOOLS["chapter_text"]
        # 段号是回到原文的唯一坐标：整章原文必须带着它，否则对话/段落之间串不起来
        assert re.search(r"\[\d+\]", text.markdown)
        assert "没有正文" not in text.markdown
