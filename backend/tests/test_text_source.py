"""手动文本数据源：切分规则，以及 n1 / n2 的文本分支。

**纯离线。** 这条路径的全部价值在于「文本源不碰抖音、但下游与抖音完全合流」，
所以测试要压的就是这件事的两头：
  - n1 不该因为「没有视频」就去问 provider（空 aweme_id 会被判非法链接）；
  - n2 切出来的条目必须**原样**流进 `clean_comments`——广告/灌水/去重的判定
    只有一套，文本源不另立标准。

判定结果按 `filter_reason` 与 `comment_stats` 的计数断言，不按文本内容查表：
正文在清洗阶段会过 `normalize()`（合并空白），拿原串当字典键会测成别的东西。
"""

from __future__ import annotations

from typing import Any

import pytest

from app.constants import SourceKind, TextMode
from app.douyin.base import DouyinError
from app.graph.emitting import NodeEmitter
from app.graph.nodes import n1_video, n2_comments
from app.graph.state import initial_state
from app.services.text_source import TEXT_ITEM_MAX_CHARS, TEXT_SOURCE_TITLE, split_text


class Recorder:
    """接住节点发的事件。

    `NodeEmitter` 允许注入 write，节点因此可以在没有图、没有总线的情况下直调
    ——这正是这些测试不搭 Redis 也能测出「事件发对了没有」的原因。
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def __call__(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def partials(self, data_kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("data_kind") == data_kind]

    def codes(self, kind: str) -> list[str]:
        return [str(e.get("code")) for e in self.events if e.get("kind") == kind]


def wire(monkeypatch: pytest.MonkeyPatch, module: Any) -> Recorder:
    rec = Recorder()
    monkeypatch.setattr(module, "NodeEmitter", lambda node: NodeEmitter(node, write=rec))
    return rec


def text_state(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "run_id": "r-text",
        "aweme_id": "",
        "source_kind": SourceKind.TEXT.value,
    }
    return initial_state(**{**base, **overrides})


# ----------------------------------------------------------------------
# split_text
# ----------------------------------------------------------------------


class TestSplitText:
    def test_行模式逐行切并丢掉空行(self) -> None:
        items, stats = split_text("  第一条  \n\n第二条\n   \n第三条\n")
        assert [i.text for i in items] == ["第一条", "第二条", "第三条"]
        assert (stats.total, stats.kept, stats.dropped) == (3, 3, 0)
        assert stats.mode == TextMode.LINE.value

    def test_换行符三种写法等价(self) -> None:
        crlf, _ = split_text("一\r\n二\r\n三")
        cr, _ = split_text("一\r二\r三")
        assert [i.text for i in crlf] == [i.text for i in cr] == ["一", "二", "三"]

    def test_段落模式以空行为界_段内换行原样保留(self) -> None:
        items, _ = split_text("第一段第一行\n第一段第二行\n\n\n第二段", mode=TextMode.PARAGRAPH)
        # 段内换行不在这里压平：那是 normalize() 的活，切分只负责切
        assert [i.text for i in items] == ["第一段第一行\n第一段第二行", "第二段"]

    def test_段落模式没有空行时整篇是一条(self) -> None:
        items, stats = split_text("一\n二\n三", mode=TextMode.PARAGRAPH)
        assert len(items) == 1 and stats.total == 1

    def test_条数上限取前N条并如实计数(self) -> None:
        text = "\n".join(f"第{i}条" for i in range(1, 11))
        items, stats = split_text(text, limit=4)
        assert [i.text for i in items] == ["第1条", "第2条", "第3条", "第4条"]
        assert (stats.total, stats.kept, stats.dropped) == (10, 4, 6)

    def test_单条过长被截断并计数(self) -> None:
        items, stats = split_text("啊" * (TEXT_ITEM_MAX_CHARS + 10))
        assert len(items[0].text) == TEXT_ITEM_MAX_CHARS
        assert stats.truncated == 1

    def test_恰好等于上限不算截断(self) -> None:
        _, stats = split_text("啊" * TEXT_ITEM_MAX_CHARS)
        assert stats.truncated == 0

    @pytest.mark.parametrize("text", ["", "   ", "\n\n\n", "\r\n \t\r\n"])
    def test_空或纯空白切不出东西(self, text: str) -> None:
        items, stats = split_text(text)
        assert items == []
        assert (stats.total, stats.kept, stats.dropped) == (0, 0, 0)

    def test_id确定_唯一_且字典序等于序号序(self) -> None:
        text = "\n".join(f"第{i}条" for i in range(1, 13))
        first, _ = split_text(text)
        second, _ = split_text(text)
        ids = [i.comment_id for i in first]

        assert ids == [i.comment_id for i in second]  # 可复现：重跑给出同一批
        assert len(set(ids)) == len(ids)
        assert ids[0] == "txt-0001" and ids[-1] == "txt-0012"
        # 零填充的意义就在这里：快照按 douyin_comment_id 升序做次级排序，
        # 字典序一旦不等于序号序，第 10 条就会排到第 2 条前面
        assert ids == sorted(ids)
        # 与抖音的 19 位纯数字 id 永不互撞
        assert all(i.startswith("txt-") for i in ids)

    def test_条目不带作者与互动数据(self) -> None:
        items, _ = split_text("一条")
        assert items[0].like_count == 0 and items[0].reply_count == 0
        assert items[0].author_name is None and items[0].publish_time is None


# ----------------------------------------------------------------------
# n1：文本源不解析视频
# ----------------------------------------------------------------------


class TestN1Text:
    async def test_文本源不碰抖音provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*_: Any, **__: Any) -> Any:
            raise AssertionError("文本源不该去问抖音 provider")

        monkeypatch.setattr("app.douyin.factory.get_douyin_provider", boom)
        rec = wire(monkeypatch, n1_video)

        patch = await n1_video.n1_video(text_state())

        assert patch["transcript"] is None
        video = patch["video"]
        assert video["title"] == TEXT_SOURCE_TITLE
        # aweme_id 非空：Video.aweme_id 是非空列，且各次运行之间不能混成同一个值
        assert video["aweme_id"] == "text:r-text"
        # 「取失败了」与「这个源本来就没有视频」是两回事，前端要能分辨
        assert "unavailable" not in video
        assert video["author_name"] is None and video["stats"] == {}
        assert "warnings" not in patch

        sent = rec.partials("video")
        assert len(sent) == 1 and sent[0]["data"]["title"] == TEXT_SOURCE_TITLE

    async def test_不同运行的合成id互不相同(self, monkeypatch: pytest.MonkeyPatch) -> None:
        wire(monkeypatch, n1_video)
        a = await n1_video.n1_video(text_state(run_id="run-a"))
        b = await n1_video.n1_video(text_state(run_id="run-b"))
        assert a["video"]["aweme_id"] != b["video"]["aweme_id"]

    async def test_抖音源仍然走provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """反向哨兵：分支判别不能把抖音路径也吃掉。"""

        class FakeProvider:
            async def get_video(self, aweme_id: str) -> Any:
                raise DouyinError("取不到")

        monkeypatch.setattr("app.douyin.factory.get_douyin_provider", lambda: FakeProvider())
        wire(monkeypatch, n1_video)

        patch = await n1_video.n1_video(initial_state(run_id="r1", aweme_id="7311"))

        assert patch["video"] == {"aweme_id": "7311", "unavailable": True}
        assert patch["warnings"][0]["code"] == "video_unavailable"


# ----------------------------------------------------------------------
# n2：切分后与抖音路径合流
# ----------------------------------------------------------------------


SAMPLE = "\n".join(
    [
        "好喝",
        "好喝",  # 复读
        "买它买它加微信 abc123 www.x.com",  # 广告
        "😀😀😀",  # 没有可见内容
        "普通的一条",
    ]
)


class TestN2Text:
    async def test_切分后走同一套清洗(self, monkeypatch: pytest.MonkeyPatch) -> None:
        wire(monkeypatch, n2_comments)

        patch = await n2_comments.n2_comments(text_state(source_text=SAMPLE))

        comments = patch["comments"]
        stats = patch["comment_stats"]
        assert len(comments) == stats["total"] == 5

        # 与抖音路径同一套判定：被标记的条目**留在结果里**并带原因，
        # 不做物理删除——前端还有「显示被过滤内容」开关
        reasons = {c["filter_reason"] for c in comments}
        assert reasons == {None, "duplicate", "ad", "empty"}
        assert (stats["duplicates"], stats["ads"], stats["empty"]) == (1, 1, 1)
        assert stats["kept"] == 2

    async def test_一次发完不分页(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rec = wire(monkeypatch, n2_comments)

        await n2_comments.n2_comments(text_state(source_text=SAMPLE))

        pages = rec.partials("comment_page")
        assert len(pages) == 1
        assert pages[0]["data"]["page"] == 1
        assert pages[0]["data"]["has_more"] is False
        assert pages[0]["data"]["received"] == len(pages[0]["data"]["items"]) == 5
        assert len(rec.partials("comment_stats")) == 1

    async def test_段落模式一段一条(self, monkeypatch: pytest.MonkeyPatch) -> None:
        wire(monkeypatch, n2_comments)
        text = "第一段第一行\n第一段第二行\n\n第二段\n\n第三段"

        patch = await n2_comments.n2_comments(
            text_state(source_text=text, text_mode=TextMode.PARAGRAPH)
        )

        assert len(patch["comments"]) == 3
        # 段内换行到这里已经被 normalize 压平（切分只切不改，压平是清洗的活）
        assert patch["comments"][0]["text"] == "第一段第一行 第一段第二行"

    async def test_超出上限如实告警(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rec = wire(monkeypatch, n2_comments)
        text = "\n".join(f"第{i}条" for i in range(1, 21))

        patch = await n2_comments.n2_comments(text_state(source_text=text, comment_limit=5))

        assert patch["comment_stats"]["total"] == 5
        codes = [w["code"] for w in patch["warnings"]]
        assert codes == ["text_truncated"]
        assert "text_truncated" in rec.codes("warning")

    async def test_没有文本时说明白而不是给空报告(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rec = wire(monkeypatch, n2_comments)

        patch = await n2_comments.n2_comments(text_state(source_text="\n\n   \n"))

        assert patch["comments"] == []
        assert patch["warnings"][0]["code"] == "empty_text"
        # 没有条目就没有分页事件，前端不该收到一页空的
        assert rec.partials("comment_page") == []
        assert "empty_text" in rec.codes("warning")
