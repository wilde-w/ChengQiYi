"""切分器与解码。

**守的核心只有一句：不丢字。** 切分器的失败模式是「悄悄少了一段」——检索时
少一条证据，不抛异常、日志正常，也没有人会去数。所以下面对每一类输入都跑
同一条等式：**各块 body 拼接后去空白 == 源文本去空白**。这一条必须由测试
守着，不能靠读代码时的那点注意力。

第二组守的是「不能被顺手做到的事」：中文全角标点绝不能被 NFKC 归一成半角。
`comment_service.normalize()` 就是这么干的，那是为抖音短评论写的；沿用到
长文上会把「，」变成「,」，和已有语料不一致，而且没人会发现。
"""

from __future__ import annotations

import pytest

from app.constants import Library
from app.kb.chunker import (
    MAX_CHARS,
    MIN_KEEP_CHARS,
    EmptyDocumentError,
    SplitConfig,
    decode_bytes,
    make_chunk_id,
    split_document,
)

PHIL = """\
此在的存在方式问题，在海德格尔那里并不是一个可以从外部观察的对象。

我们每一个人都已经处在对存在的某种理解之中，这种理解先于一切专题化的认识。
当哲学追问存在的意义时，它问的不是某个存在者的属性，而是存在本身。
因此，此在的分析必须从日常状态出发，从平均的、无差别的常人出发。

操心是此在存在的整体结构。它包含三个环节：先行于自身的存在、已经在世界之中、
寓于世界内存在者的存在。这三者不是并列的三个部分，而是一个整体结构的三个面向。
"""


def _flat(text: str) -> str:
    """去掉全部空白。换行、缩进、行尾空格都不该被算作「正文」。"""
    return "".join(text.split())


def _assert_no_loss(raw: str, pieces_text: list[str]) -> None:
    assert _flat("".join(pieces_text)) == _flat(raw)


# ----------------------------------------------------------------------
# 解码
# ----------------------------------------------------------------------


class TestDecode:
    def test_utf8_and_gbk_yield_identical_text(self) -> None:
        # 中文 Windows 上记事本存的 txt 默认是 GBK——这不是边角情况。
        # 两条路必须得到**逐字相同**的正文，否则同一个文件导入两次会被
        # 当成两份不同的内容，全都算「新增」。
        assert decode_bytes(PHIL.encode("utf-8")) == decode_bytes(PHIL.encode("gb18030"))

    def test_bom_is_stripped_not_kept_as_text(self) -> None:
        assert decode_bytes(b"\xef\xbb\xbf" + PHIL.encode("utf-8")) == PHIL

    def test_undecodable_bytes_fail_loudly(self) -> None:
        # **不许 errors="ignore"。** 静默丢字节会让同一个文件第二次导入切出
        # 不同的正文，而两次的差别只在于几个被丢掉的字节，谁也不会去看那里。
        with pytest.raises(EmptyDocumentError, match="无法解码"):
            decode_bytes(b"\xff\xff\xff\xff")


# ----------------------------------------------------------------------
# 空文档
# ----------------------------------------------------------------------


class TestEmptyDocument:
    """这些一律硬失败。**「切出 0 块然后报成功」是最坏的结果**——
    它在界面上和真的成功长得一模一样。"""

    @pytest.mark.parametrize(
        "raw",
        ["", "   \n\t \n  ", "﻿", "​​", "短", "不到八字"],
    )
    def test_rejected(self, raw: str) -> None:
        with pytest.raises(EmptyDocumentError):
            split_document(raw)

    def test_min_keep_chars_is_the_boundary(self) -> None:
        with pytest.raises(EmptyDocumentError):
            split_document("字" * (MIN_KEEP_CHARS - 1))
        assert len(split_document("字" * MIN_KEEP_CHARS).pieces) == 1


# ----------------------------------------------------------------------
# 清洗
# ----------------------------------------------------------------------


class TestClean:
    def test_crlf_and_lone_cr_normalized(self) -> None:
        raw = "第一段有足够多的字来通过下限。\r\n\r\n第二段也有足够多的字。\r结尾。"
        report = split_document(raw)
        assert all("\r" not in p.body for p in report.pieces)
        # 归一化若没发生，空行正则匹配不上，两段会粘成一段。
        assert report.paragraphs == 2

    def test_full_width_punctuation_survives(self) -> None:
        report = split_document("存在是什么？这不是一个可以回答的问题，但必须问。")
        body = report.pieces[0].body
        assert "？" in body and "，" in body and "。" in body
        assert "," not in body and "?" not in body

    def test_zero_width_chars_removed(self) -> None:
        report = split_document("此在​的存在‌方式，需要足够的字数。")
        assert "​" not in report.pieces[0].body


# ----------------------------------------------------------------------
# 切分
# ----------------------------------------------------------------------


class TestSplitting:
    def test_normal_document_keeps_paragraph_boundaries(self) -> None:
        report = split_document(PHIL)
        assert report.paragraphs == 3
        assert len(report.pieces) == 3
        assert report.hard_splits == 0
        _assert_no_loss(PHIL, [p.body for p in report.pieces])

    def test_hard_wrapped_lines_are_joined_back(self) -> None:
        # 每 12 字折一行、段间才有空行的书极常见。折行不接回去的话，
        # 这本书会变成一个巨型段落，后面的按句装箱与硬切全线开花。
        line = "此在的存在方式问题需要足够的字数"
        raw = "\n".join([line] * 4) + "\n\n" + "\n".join([line] * 4)
        report = split_document(raw)
        assert report.paragraphs == 2
        assert report.hard_splits == 0

    def test_long_paragraph_respects_max_chars(self) -> None:
        sentence = "操心是此在存在的整体结构，它先行于自身并且已经在世界之中。"
        paragraph = sentence * 30  # 约 870 字，超过 MAX_CHARS
        report = split_document(paragraph)
        assert len(report.pieces) >= 2
        assert all(len(p.body) <= MAX_CHARS for p in report.pieces)
        # 边界落在句末，没有句子被拦腰截断——所以不需要硬切。
        assert report.hard_splits == 0
        _assert_no_loss(paragraph, [p.body for p in report.pieces])

    def test_no_punctuation_text_is_hard_cut_and_warns(self) -> None:
        # 5 万字无标点：宁可带警告入库，也不要静默丢字。
        raw = "无标点长串" * 12_500
        report = split_document(raw)
        assert report.hard_splits > len(report.pieces) * 0.2
        assert any("硬切" in w for w in report.warnings)
        _assert_no_loss(raw, [p.body for p in report.pieces])

    def test_hard_cut_overlap_only_touches_text_not_body(self) -> None:
        # 重叠是为了让被拦腰截断的句子在下一块里读得通。它**不能进 body**：
        # 「不丢字」等式比的是 body，而重叠是正文的副本。
        raw = "无标点长串" * 300
        report = split_document(raw)
        assert all(p.hard_split for p in report.pieces)
        # 第一块没有前文可回退，所以没有重叠；从第二块起每块都带前缀。
        assert report.pieces[0].overlap_chars == 0
        assert all(p.overlap_chars > 0 for p in report.pieces[1:])
        assert all(len(p.text) > len(p.body) for p in report.pieces[1:])
        _assert_no_loss(raw, [p.body for p in report.pieces])

    def test_short_tail_merged_into_previous(self) -> None:
        cfg = SplitConfig(target_chars=60, max_chars=100, min_chars=40)
        paragraph = "甲" * 30 + "。" + "乙" * 30 + "。" + "丙" * 20 + "。"
        report = split_document(paragraph, cfg)
        assert report.too_short_pieces == 0
        assert all(len(p.body) >= cfg.min_chars for p in report.pieces)
        _assert_no_loss(paragraph, [p.body for p in report.pieces])

    def test_tail_kept_when_merge_would_exceed_max(self) -> None:
        # 并进前一块会超上限时的两难：宁可留一个短块，也不能丢字。
        cfg = SplitConfig(target_chars=60, max_chars=100, min_chars=40)
        paragraph = "甲" * 95 + "。" + "乙" * 10 + "。"
        report = split_document(paragraph, cfg)
        assert report.too_short_pieces == 1
        assert report.pieces[-1].body.endswith("。")
        _assert_no_loss(paragraph, [p.body for p in report.pieces])

    def test_english_text_warns_about_cjk_ratio(self) -> None:
        report = split_document("This is an English paragraph, long enough to be kept.")
        assert any("中文字符" in w for w in report.warnings)

    def test_short_document_warns(self) -> None:
        report = split_document("只有一句话，字数不多，但也收下。")
        assert any("请确认文件是否正确" in w for w in report.warnings)

    def test_pieces_are_indexed_from_zero(self) -> None:
        report = split_document(PHIL)
        assert [p.index for p in report.pieces] == list(range(len(report.pieces)))


class TestNoTextLoss:
    """逐类输入跑同一条等式。

    每个用例都显式给 `id`：Windows 的环境变量上限是 32767 字符，而 pytest
    会把**整个 node id**（含参数值）写进 `PYTEST_CURRENT_TEST`——几千字的
    参数会让测试在 setup 阶段就炸掉，报一个和测试毫无关系的错。
    """

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param(PHIL, id="plain"),
            pytest.param(PHIL.replace("\n", "\r\n"), id="crlf"),
            pytest.param(PHIL.replace("\n\n", "\n"), id="no-blank-lines"),
            pytest.param("。".join(["此在的存在方式"] * 200) + "。", id="long-single-paragraph"),
            pytest.param("甲" * 5000, id="no-punctuation"),
            pytest.param((PHIL.strip() + "\n") * 30, id="repeated"),
        ],
    )
    def test_bodies_cover_the_document(self, raw: str) -> None:
        report = split_document(raw)
        _assert_no_loss(raw, [p.body for p in report.pieces])


class TestDeterminism:
    def test_same_text_yields_same_ids(self) -> None:
        a, b = split_document(PHIL), split_document(PHIL)
        assert [p.body for p in a.pieces] == [p.body for p in b.pieces]
        assert [make_chunk_id(Library.PSYCHOLOGY, "哲学导论", p.body) for p in a.pieces] == [
            make_chunk_id(Library.PSYCHOLOGY, "哲学导论", p.body) for p in b.pieces
        ]

    def test_different_paragraph_order_changes_ids(self) -> None:
        # 用户往中间插一段之后，序号派生的 id 会整体错位；内容派生的不会。
        # 这里断言的是「正文变了 → id 才变」，而不是「位置变了 → id 就变」。
        first, second = PHIL.split("\n\n")[0], PHIL.split("\n\n")[1]
        assert make_chunk_id(Library.PSYCHOLOGY, "书", first) != make_chunk_id(
            Library.PSYCHOLOGY, "书", second
        )


class TestChunkId:
    """id 的每一条性质都对应一个具体的失败场景。"""

    def test_work_normalized_so_second_import_matches(self) -> None:
        # 第二次把书名写成「《存在与时间》」时 id 必须不变，
        # 否则同一本书会被当成两份各导一遍。
        a = make_chunk_id(Library.PSYCHOLOGY, "存在与时间", PHIL)
        b = make_chunk_id(Library.PSYCHOLOGY, "《存在与时间》", PHIL)
        c = make_chunk_id(Library.PSYCHOLOGY, " 存在与时间 ", PHIL)
        assert a == b == c

    def test_library_is_part_of_the_identity(self) -> None:
        # 同一段文字导进两个库会撞 id，且导入路径不经 load_all 的跨库查重。
        assert make_chunk_id(Library.PSYCHOLOGY, "书", PHIL) != make_chunk_id(
            Library.LITERATURE, "书", PHIL
        )

    def test_work_is_part_of_the_identity(self) -> None:
        # 二手研究引原著：两本书引用同一段原文，不含 work 会撞成一条，
        # 后导入的把前一条的书名作者一并改写，原著那条静默消失。
        assert make_chunk_id(Library.PSYCHOLOGY, "甲书", PHIL) != make_chunk_id(
            Library.PSYCHOLOGY, "乙书", PHIL
        )

    def test_prefix_follows_library(self) -> None:
        assert make_chunk_id(Library.POETRY, "书", PHIL).startswith("poem:imp:")
        assert make_chunk_id(Library.LITERATURE, "书", PHIL).startswith("lit:imp:")

    def test_id_fits_the_column(self) -> None:
        # kb_document.chunk_id 是 String(128)。
        assert len(make_chunk_id(Library.PSYCHOLOGY, "书" * 100, PHIL)) < 128
