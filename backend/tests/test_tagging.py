"""打标：校验、对齐防御、降级。

**这个文件守的是一类不会报错的错误：标签张冠李戴。**
第 3 块的正文配第 5 块的标签，程序照样跑完、照样入库、日志一行不差；
坏掉的是检索——用户查「此在」查出一段讲落花的东西，而没有人会去查标签
是怎么分配错的。

所以下面的断言几乎都是「坏输入必须被挡住」，而不是「好输入能通过」：
好输入跑通只是基本盘，坏输入悄悄通过才是这个模块唯一真正的失败模式。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pytest

from app.constants import Library
from app.kb.chunker import Piece
from app.kb.lexical_tags import EMOTION_VOCAB
from app.kb.tagging import (
    TaggingCancelled,
    TaggingError,
    TagReport,
    _is_misaligned,
    _validate_item,
    tag_pieces,
)
from app.providers.base import ChatMessage, ChatResult, ProviderError
from app.providers.mock_llm import MockChatModel

PHIL = [
    "操心是此在存在的整体结构，它先行于自身，并且已经在世界之中。此在的时间性构成了操心的意义。",
    "我们每一个人都已经处在对存在的某种理解之中，这种理解先于一切专题化的认识。",
    "因此，此在的分析必须从日常状态出发，从平均的、无差别的常人出发。",
    "畏揭示了虚无，而虚无并不是某个存在者的缺席，它是存在本身的遮蔽。",
    "语言是存在之家，人在语言中居住，并且以此方式回应存在的呼唤。",
]


def _pieces(texts: list[str] | None = None) -> list[Piece]:
    return [
        Piece(index=i, text=t, body=t, paragraph=i) for i, t in enumerate(texts or PHIL)
    ]


# ----------------------------------------------------------------------
# 假模型：这里要构造的是「模型答错了」，而 mock 恰好总是答对
# ----------------------------------------------------------------------


@dataclass
class ScriptedChat:
    reply: Callable[[dict[str, Any]], dict[str, Any]]
    finish_reason: str | None = "stop"
    name: str = "scripted"
    is_mock: bool = True
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        task: str | None = None,
        context: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        ctx = dict(context or {})
        self.calls.append(ctx)
        payload = self.reply(ctx)
        return ChatResult(
            text=json.dumps(payload, ensure_ascii=False),
            model=self.name,
            finish_reason=self.finish_reason,
        )


@dataclass
class FailingChat:
    """一律抛错。`retryable` 决定调用方该重试还是该立刻止损。"""

    retryable: bool = True
    error: str = "boom"
    name: str = "failing"
    is_mock: bool = True
    calls: int = 0

    async def complete(self, *args: Any, **kwargs: Any) -> ChatResult:
        self.calls += 1
        raise ProviderError(self.error, retryable=self.retryable)


def _row(item: dict[str, Any], quote: str | None = None, **overrides: Any) -> dict[str, Any]:
    """以某条正文为准，造一条合法返回项。

    `quote` 单独可换：错位场景要的就是「i 写的是这一条、正文抄的是另一条」。
    """
    text = item["text"]
    row = {
        "i": item["i"],
        "quote": (quote or text)[:12],
        "concept": "操心",
        "keywords": [k for k in ("操心", "此在", "存在") if k in text][:2],
        "imagery": [],
        "emotion": [],
        "type": "论述",
    }
    row.update(overrides)
    return row


def _shifted_reply(ctx: dict[str, Any]) -> dict[str, Any]:
    """每条都把**文档里的下一条**抄去当 quote。

    按全局位置而不是批内位置错位是刻意的：批内互换在切半重试时会
    自行变回正确（半批里只有一条，`(pos+1) % 1 == 0`），于是「整批作废」
    变成「重试一次就好了」——测不到想测的东西。
    """
    out = []
    for item in ctx["items"]:
        nxt = PHIL[(PHIL.index(item["text"]) + 1) % len(PHIL)]
        out.append(_row(item, quote=nxt))
    return {"items": out}


# ----------------------------------------------------------------------
# 校验器
# ----------------------------------------------------------------------


class TestValidator:
    TEXT = "操心是此在存在的整体结构，此在的时间性构成了操心的意义。"

    def _check(self, **row: Any) -> tuple[Any, TagReport]:
        report = TagReport()
        item = {"i": 0, "quote": self.TEXT[:12], "concept": "操心"}
        item.update(row)
        return _validate_item(item, self.TEXT, library=Library.PSYCHOLOGY, report=report), report

    def test_valid_row_passes(self) -> None:
        tags, _ = self._check()
        assert tags is not None
        assert tags.quote == self.TEXT[:12]
        assert tags.source == "llm"

    def test_quote_not_in_text_voids_the_whole_item(self) -> None:
        # 这是**唯一**一条会让整条作废的校验：它是身份问题，不是质量问题。
        # 剩下的字段再漂亮也不能信——它们可能是另一段的。
        tags, report = self._check(quote="畏揭示了虚无，而虚无并不是某个存在者的缺席")
        assert tags is None
        assert report.dropped_tags["quote 对不上"] == 1

    def test_missing_quote_voids_the_item(self) -> None:
        tags, report = self._check(quote="")
        assert tags is None
        assert report.dropped_tags["缺 quote"] == 1

    @pytest.mark.parametrize("tag", ["存在｜情感：虚无", "《存在与时间》", "此在：时间性", "操\n心"])
    def test_forbidden_characters_are_rejected_not_cleaned(self, tag: str) -> None:
        # 这些字符会破坏 `text_for_embedding()` 的模板结构——**一个标签就能
        # 伪造出「｜情感：x」**。清洗成「情感x」既不合法也不像话，
        # 而且掩盖了模型正在试图注入结构这件事。
        tags, report = self._check(keywords=[tag])
        assert tags is not None
        assert tags.keywords == []
        assert report.dropped_tags["禁用字符"] == 1

    def test_non_substring_keyword_dropped(self) -> None:
        # 顺口编出来的词，会让这段文字被它根本没回答的查询召回。
        tags, report = self._check(keywords=["操心", "现象学还原"])
        assert tags is not None
        assert tags.keywords == ["操心"]
        assert report.dropped_tags["非原文子串"] == 1

    def test_emotion_outside_the_closed_set_dropped(self) -> None:
        # 自由词表会把「焦虑/不安/忧惧」裂成三个图节点，
        # `chunks_by_emotion("焦虑")` 只命中三分之一。
        tags, report = self._check(emotion=["忧惧", "焦虑"])
        assert tags is not None
        assert tags.emotion == ["焦虑"]
        assert report.dropped_tags["emotion 越界"] == 1

    def test_emotion_need_not_be_a_substring(self) -> None:
        # 正文说「心慌」，该标的是「焦虑」——两者本来就不会互为子串。
        text = "他感到一阵心慌，说不出原因，只好反复地走来走去。"
        report = TagReport()
        tags = _validate_item(
            {"i": 0, "quote": text[:8], "emotion": ["焦虑"]},
            text,
            library=Library.PSYCHOLOGY,
            report=report,
        )
        assert tags is not None and tags.emotion == ["焦虑"]

    def test_type_outside_the_vocab_falls_back_to_default(self) -> None:
        tags, report = self._check(type="论文")
        assert tags is not None
        assert tags.type == ""
        assert report.dropped_tags["type 越界"] == 1

    def test_psychology_imagery_is_forced_empty(self) -> None:
        # 哲学块被打上「大地」，`chunks_by_imagery("大地")` 就会从古典文学
        # 那条图路径里把它返回出来。
        tags, report = self._check(imagery=["操心"])
        assert tags is not None
        assert tags.imagery == []
        assert report.dropped_tags["psychology 误标意象"] == 1

    def test_literature_keeps_imagery(self) -> None:
        text = "他站在悬崖边上，望着深渊，心里充满了恐惧。"
        tags = _validate_item(
            {"i": 0, "quote": text[:6], "imagery": ["深渊"], "emotion": ["恐惧"]},
            text,
            library=Library.LITERATURE,
            report=TagReport(),
        )
        assert tags is not None
        assert tags.imagery == ["深渊"] and tags.emotion == ["恐惧"]

    def test_non_dict_row_is_ignored(self) -> None:
        assert _validate_item("不是对象", self.TEXT, library=Library.PSYCHOLOGY, report=TagReport()) is None


class TestMisalignmentDetector:
    TEXTS: ClassVar[dict[int, str]] = {0: "甲甲甲甲甲甲甲甲", 1: "乙乙乙乙乙乙乙乙"}

    def test_quote_from_another_piece_is_evidence(self) -> None:
        rows = {0: {"quote": "乙乙乙乙"}, 1: {"quote": "甲甲甲甲"}}
        assert _is_misaligned(rows, self.TEXTS) is True

    def test_own_quotes_are_not_evidence(self) -> None:
        rows = {0: {"quote": "甲甲甲"}, 1: {"quote": "乙乙乙"}}
        assert _is_misaligned(rows, self.TEXTS) is False

    def test_shared_sentence_is_not_evidence(self) -> None:
        # 两条正文里恰好都含同一句话（引用、重复段落）时不能作数，
        # 否则会把一批正常的返回判死。
        texts = {0: "共同的一句话甲乙", 1: "共同的一句话丙丁"}
        rows = {0: {"quote": "共同的一句话"}, 1: {"quote": "共同的一句话"}}
        assert _is_misaligned(rows, texts) is False


# ----------------------------------------------------------------------
# mock 路径：它必须真的过一遍校验器，否则离线演示就是假通过
# ----------------------------------------------------------------------


class TestMockPath:
    async def _tag(self, library: Library = Library.PSYCHOLOGY, **kwargs: Any):
        kwargs.setdefault("model", MockChatModel(latency_ms=0))
        return await tag_pieces(_pieces(), library=library, **kwargs)

    async def test_every_piece_is_tagged_without_fallback(self) -> None:
        chunks, report = await self._tag()
        assert report.total == len(chunks) == len(PHIL)
        assert report.tagged == len(PHIL)
        assert report.fallback == 0
        assert report.failed_batches == 0
        assert not report.dropped_tags, f"mock 的产出被校验器挡了：{report.dropped_tags}"

    async def test_concept_is_never_empty_in_psychology(self) -> None:
        chunks, _ = await self._tag()
        assert all(c.concept for c in chunks)

    async def test_emotion_stays_inside_the_closed_set(self) -> None:
        chunks, _ = await self._tag()
        for chunk in chunks:
            assert set(chunk.emotion) <= set(EMOTION_VOCAB)
            assert chunk.imagery == []

    async def test_every_tag_is_a_substring_of_its_own_text(self) -> None:
        # 「绝不张冠李戴」最直接的表述：每个标签都能在**它自己那一条**
        # 正文里指出来。
        chunks, _ = await self._tag()
        for chunk in chunks:
            for tag in [*chunk.keywords, *chunk.imagery]:
                assert tag in chunk.text

    async def test_origin_and_work_are_carried_through(self) -> None:
        chunks, _ = await self._tag(work="哲学导论", author="周濂", origin="哲学导论.txt")
        assert all(c.origin == "哲学导论.txt" for c in chunks)
        assert all(c.work == "哲学导论" and c.author == "周濂" for c in chunks)

    async def test_use_llm_false_is_not_a_fallback(self) -> None:
        # 「这次不用 AI」是用户的显式选择，不该在报告里显示成降级。
        chunks, report = await self._tag(model=None, use_llm=False)
        assert len(chunks) == len(PHIL)
        assert report.fallback == 0 and report.tagged == 0

    async def test_model_is_required_when_use_llm(self) -> None:
        with pytest.raises(ValueError, match="必须提供 model"):
            await tag_pieces(_pieces(), library=Library.PSYCHOLOGY, model=None, use_llm=True)


# ----------------------------------------------------------------------
# 对齐防御
# ----------------------------------------------------------------------


class TestAlignmentDefence:
    async def test_array_order_changed_is_fine(self) -> None:
        """数组倒序返回**不该**作废：我们按 `i` 配对，不按下标 zip。"""

        def reply(ctx: dict[str, Any]) -> dict[str, Any]:
            items = list(reversed(ctx["items"]))
            return {"items": [_row(it) for it in items]}

        model = ScriptedChat(reply=reply)
        chunks, report = await tag_pieces(
            _pieces(PHIL[:3]), library=Library.PSYCHOLOGY, model=model
        )
        assert report.tagged == 3
        assert report.fallback == 0
        assert chunks[0].concept == "操心"

    async def test_relabelled_quotes_void_the_batch(self) -> None:
        """`i` 写着自己、正文抄的是下一条 → 整批作废、切半重试、绝不张冠李戴。"""
        model = ScriptedChat(reply=_shifted_reply)
        chunks, report = await tag_pieces(
            _pieces(PHIL[:2]),
            library=Library.PSYCHOLOGY,
            model=model,
            allow_partial=True,
        )

        # 整批作废 → 两个半批各自重试一次 → 仍对不上 → 全部走词法降级。
        assert report.tagged == 0
        assert report.fallback == 2
        assert report.failed_batches == 1
        assert any("切半重试" in w for w in report.warnings)
        assert len(model.calls) == 3, "应是一次整批 + 两次半批"
        # 降级产物同样是「每个标签都能在自己那条正文里指出来」。
        for chunk in chunks:
            for tag in [*chunk.keywords, *chunk.imagery]:
                assert tag in chunk.text

    async def test_missing_item_voids_the_batch(self) -> None:
        def reply(ctx: dict[str, Any]) -> dict[str, Any]:
            return {"items": [_row(it) for it in ctx["items"][:-1]]}  # 永远少一条

        model = ScriptedChat(reply=reply)
        _, report = await tag_pieces(
            _pieces(PHIL[:2]), library=Library.PSYCHOLOGY, model=model, allow_partial=True
        )
        assert report.tagged == 0
        assert report.fallback == 2

    async def test_out_of_range_index_voids_the_batch(self) -> None:
        def reply(ctx: dict[str, Any]) -> dict[str, Any]:
            rows = [_row(it) for it in ctx["items"][1:]]
            rows.append(_row(ctx["items"][0], i=999))  # 越界的 i
            return {"items": rows}

        model = ScriptedChat(reply=reply)
        _, report = await tag_pieces(
            _pieces(PHIL[:1]), library=Library.PSYCHOLOGY, model=model, allow_partial=True
        )
        assert report.tagged == 0 and report.fallback == 1

    async def test_truncated_response_voids_the_batch(self) -> None:
        # JSON 能解析不等于内容完整：被 max_tokens 砍掉的那个 item
        # 往往正好是半截的。
        def reply(ctx: dict[str, Any]) -> dict[str, Any]:
            return {"items": [_row(it) for it in ctx["items"]]}

        model = ScriptedChat(reply=reply, finish_reason="length")
        _, report = await tag_pieces(
            _pieces(PHIL[:2]), library=Library.PSYCHOLOGY, model=model, allow_partial=True
        )
        assert report.tagged == 0 and report.fallback == 2

    async def test_retry_changes_the_request(self) -> None:
        # 重试必须改变请求：缓存模型的 key 含 context，原样重试会命中
        # 上一次那条坏缓存，原样失败——看起来像「重试逻辑没生效」。
        def reply(ctx: dict[str, Any]) -> dict[str, Any]:
            return {}

        model = ScriptedChat(reply=reply)
        await tag_pieces(
            _pieces(PHIL[:2]), library=Library.PSYCHOLOGY, model=model, allow_partial=True
        )
        attempts = [c.get("attempt") for c in model.calls]
        assert attempts[0] == 0
        assert any(a != 0 for a in attempts[1:]), f"重试没有改变请求：{attempts}"


# ----------------------------------------------------------------------
# 失败降级
# ----------------------------------------------------------------------


class TestFailureHandling:
    async def test_batch_failure_falls_back_to_lexical(self) -> None:
        model = FailingChat(retryable=True)
        chunks, report = await tag_pieces(
            _pieces(PHIL[:2]), library=Library.PSYCHOLOGY, model=model, allow_partial=True
        )
        assert report.fallback == 2
        assert all(c.concept for c in chunks), "降级也要保证 concept 非空"

    async def test_majority_failure_aborts_with_nothing_written(self) -> None:
        """过半批次失败 → 硬失败，什么都不写。

        600 条只有词法标签的块入库后会被每次检索当作候选召回成噪声证据，
        而回滚要手工按 work 清——什么都没写是可恢复的，写进去是慢性中毒。
        """
        model = FailingChat(retryable=True)
        with pytest.raises(TaggingError, match="未写入任何内容"):
            await tag_pieces(_pieces(), library=Library.PSYCHOLOGY, model=model)

    async def test_allow_partial_lets_it_through(self) -> None:
        model = FailingChat(retryable=True)
        chunks, report = await tag_pieces(
            _pieces(), library=Library.PSYCHOLOGY, model=model, allow_partial=True
        )
        assert len(chunks) == len(PHIL)
        assert report.failure_ratio == 1.0

    async def test_non_retryable_errors_stop_after_three(self) -> None:
        # 鉴权失败、模型名写错、额度耗尽：配置错误不会自愈，
        # 硬跑下去只是把时间浪费掉，还会在日志里刷几百行同样的错误。
        model = FailingChat(retryable=False, error="401 invalid api key")
        with pytest.raises(TaggingError, match="配置问题"):
            await tag_pieces(
                _pieces(), library=Library.PSYCHOLOGY, model=model, batch_size=1
            )
        assert model.calls == 3, "第 3 次不可重试的错误之后就该停手"

    async def test_cancellation_is_honoured(self) -> None:
        with pytest.raises(TaggingCancelled):
            await tag_pieces(
                _pieces(),
                library=Library.PSYCHOLOGY,
                model=MockChatModel(latency_ms=0),
                is_cancelled=lambda: True,
            )

    async def test_progress_is_reported_monotonically(self) -> None:
        seen: list[tuple[int, int]] = []
        await tag_pieces(
            _pieces(),
            library=Library.PSYCHOLOGY,
            model=MockChatModel(latency_ms=0),
            batch_size=2,
            on_progress=lambda done, total: seen.append((done, total)),
        )
        assert seen[-1] == (3, 3)
        assert [d for d, _ in seen] == [1, 2, 3]


# ----------------------------------------------------------------------
# 组装
# ----------------------------------------------------------------------


class TestAssembly:
    async def test_duplicate_bodies_are_merged_and_counted(self) -> None:
        # 同一份文件里出现两次完全相同的正文（重复页、刻意重复的格言）。
        # 合并成一条，但**不静默**——「少了一块」必须是可见的。
        texts = [PHIL[0], PHIL[0], PHIL[1]]
        chunks, report = await tag_pieces(
            _pieces(texts), library=Library.PSYCHOLOGY, model=MockChatModel(latency_ms=0)
        )
        assert report.deduped == 1
        assert len({c.chunk_id for c in chunks}) == len(chunks) == 2
        assert any("完全相同" in w for w in report.warnings)

    async def test_literature_drops_pieces_without_imagery_or_emotion(self) -> None:
        # 抽象文本在文学模板下产不出意象和情感，嵌进去就是一段没有
        # 检索钩子的裸文本——占一次嵌入、一个候选位，永远不被命中。
        chunks, report = await tag_pieces(
            _pieces(["的了着过是在"]),
            library=Library.LITERATURE,
            model=MockChatModel(latency_ms=0),
            use_llm=False,
        )
        assert chunks == []
        assert report.dropped == 1
        assert any("标注全空" in w for w in report.warnings)

    async def test_chunk_id_is_derived_from_body_not_overlap_text(self) -> None:
        # 用 `text` 派生的话，改一次 OVERLAP_CHARS 就会让整本书换一批新 id。
        body = PHIL[0]
        plain = [Piece(index=0, text=body, body=body, paragraph=0)]
        overlapped = [Piece(index=0, text="前文重叠" + body, body=body, paragraph=0)]
        kwargs = {"library": Library.PSYCHOLOGY, "use_llm": False}
        a, _ = await tag_pieces(plain, **kwargs)  # type: ignore[arg-type]
        b, _ = await tag_pieces(overlapped, **kwargs)  # type: ignore[arg-type]
        assert a[0].chunk_id == b[0].chunk_id
        assert a[0].text != b[0].text
