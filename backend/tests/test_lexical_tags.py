"""不依赖模型的标签抽取。

它是两条路的共用实现：mock 模式下的 `tag_chunks`，以及真实 LLM 打标失败时的
降级。所以这里守的性质不是「标签好看」，而是**两条路都不会分叉的那几条**：

1. 所有标签都是原文的子串——这条同时让 mock 产出天然通过校验器，
   并让「关键词：此在」与用户查询「此在」在 n-gram 桶里真的共享片段；
2. `concept` 对 psychology 永不为空——它是嵌入模板的头部，空了这条就检索不到；
3. `emotion` 恒在闭集内——图里的 Emotion 节点是 MERGE 出来的，
   自由词表会把「焦虑/不安/忧惧」裂成三个节点，`chunks_by_emotion("焦虑")`
   只命中三分之一。**闭集必须覆盖已交付语料用到的每一个情绪**，
   否则语料里那个情绪一旦被 LLM 复述，会被校验器当成越界值丢掉。
"""

from __future__ import annotations

import pytest

from app.constants import Library
from app.kb.lexical_tags import (
    EMOTION_VOCAB,
    MAX_KEYWORDS,
    PROSE_IMAGERY_LEXICON,
    ChunkTags,
    build_frequency_table,
    extract_keyphrases,
    lexical_tags,
)
from app.kb.loader import load_all

PHIL = (
    "操心是此在存在的整体结构，它先行于自身，并且已经在世界之中。"
    "此在的存在方式问题不能从外部观察，因为追问存在的那个存在者，"
    "本身就是它所追问的东西。此在的时间性构成了操心的意义。"
)


class TestShippedCorpusCoverage:
    """**闭集必须覆盖随仓库交付的语料。**

    `tagging._match_vocab` 会把不在闭集里的情绪一律丢掉。语料用了某个情绪
    而闭集没有它，表现是「LLM 老老实实照着语料回了一个词，被程序判成越界」
    ——检索命中率莫名偏低，而且没人会想到去看这里。
    """

    def test_every_corpus_emotion_is_in_the_closed_vocab(self) -> None:
        used = {e for chunks in load_all().values() for c in chunks for e in c.emotion}
        missing = sorted(used - set(EMOTION_VOCAB))
        assert not missing, f"语料在用的情绪不在闭集里，会被校验器丢掉：{missing}"

    def test_vocab_has_no_duplicates(self) -> None:
        # 闭集是手工拼出来的（查询侧词表 + 语料补充 + 哲学补充），
        # 重复项会原样出现在 prompt 的词表里。
        assert len(EMOTION_VOCAB) == len(set(EMOTION_VOCAB))

    def test_vocab_survives_the_validators_own_filter(self) -> None:
        # 校验器会拒绝含禁用字符的标签。闭集里出现这类值 = 这个情绪永远标不上。
        from app.kb.tagging import _FORBIDDEN

        bad = [e for e in EMOTION_VOCAB if any(ch in _FORBIDDEN for ch in e)]
        assert not bad, f"闭集里的这些值永远过不了校验器：{bad}"

    def test_imagery_lexicon_is_multi_char_only(self) -> None:
        # 单字词表（「月」）在散文里会从「岁月」「月色」误命中，
        # 而误命中比落空危害大——错误标签持续污染检索，且没人会去查。
        single = [f for forms in PROSE_IMAGERY_LEXICON.values() for f in forms if len(f) < 2]
        assert not single, f"散文意象词表里不该有单字形：{single}"


class TestKeyphrases:
    def test_every_phrase_is_a_substring(self) -> None:
        for phrase in extract_keyphrases(PHIL):
            assert phrase in PHIL

    def test_no_more_than_k(self) -> None:
        assert len(extract_keyphrases(PHIL, k=3)) <= 3

    def test_no_fragments_starting_with_function_words(self) -> None:
        # 「的存在方式」「所以就」这类碎片在测语料时是最常见的坏产出。
        for phrase in extract_keyphrases(PHIL):
            assert phrase[0] not in "的了着过是在和与而也就都还又很把被"

    def test_document_frequency_promotes_the_recurring_term(self) -> None:
        # 单块内每个 n-gram 都只出现一次时，长度加成是唯一判据，选出来的是
        # 最长的那个碎片。文档级频次把「此在」这种全书反复出现的术语顶上来。
        #
        # 构造要点：术语出现在**各不相同**的上下文里（每行的后四个字互不相同），
        # 否则重复整行会让所有 n-gram 同样高频，长度加成照样赢。
        def block(i: int) -> str:
            return "".join(chr(0x4E00 + i * 4 + k) for k in range(4))

        lines = [f"此在{block(i)}" for i in range(40)]
        freq = build_frequency_table("\n".join(lines))
        chunk = "\n".join(lines[:2])

        assert extract_keyphrases(chunk, k=1, freq=freq) == ["此在"]
        assert extract_keyphrases(chunk, k=1) != ["此在"]


class TestAssembledTags:
    def test_psychology_is_概念_only(self) -> None:
        tags = lexical_tags(PHIL, library=Library.PSYCHOLOGY, discipline_hint="存在主义哲学")
        assert tags.concept, "psychology 的 concept 是嵌入模板的头部，不许为空"
        assert tags.imagery == [], "给哲学块打意象会污染古典文学那条图路径"
        assert tags.type == "论述"
        assert tags.source == "lexical"
        assert tags.quote and tags.quote in PHIL

    def test_literature_uses_prose_imagery(self) -> None:
        text = "他站在悬崖边上，望着深渊，心里充满了恐惧与孤独。"
        tags = lexical_tags(text, library=Library.LITERATURE)
        assert "深渊" in tags.imagery
        assert "恐惧" in tags.emotion
        assert tags.type == "散文"

    def test_poetry_type_is_诗(self) -> None:
        assert lexical_tags(PHIL, library=Library.POETRY).type == "诗"

    def test_emotion_always_inside_the_closed_set(self) -> None:
        for text in (PHIL, "他心里充满愤怒与不甘。", "春江潮水连海平，海上明月共潮生。"):
            for lib in Library:
                tags = lexical_tags(text, library=lib)
                assert set(tags.emotion) <= set(EMOTION_VOCAB)

    def test_single_char_imagery_is_not_matched(self) -> None:
        # 「岁月」「水平」里的「月」「水」都不是意象。散文词表只认多字形。
        tags = lexical_tags("他的水平很高，岁月也没有磨掉他的锐气。", library=Library.LITERATURE)
        assert tags.imagery == []

    def test_deterministic(self) -> None:
        a = lexical_tags(PHIL, library=Library.PSYCHOLOGY, freq=build_frequency_table(PHIL))
        b = lexical_tags(PHIL, library=Library.PSYCHOLOGY, freq=build_frequency_table(PHIL))
        assert a == b


class TestConceptFallbackChain:
    """`concept` 对 psychology 是必填的，所以它有一条绝不落空的兜底链。"""

    FAINT = "的了着过是在"  # 每个 n-gram 都被首/尾虚词表挡掉

    def test_falls_back_to_discipline_hint(self) -> None:
        tags = lexical_tags(self.FAINT, library=Library.PSYCHOLOGY, discipline_hint="存在主义")
        assert tags.concept == "存在主义"

    def test_falls_back_to_work(self) -> None:
        tags = lexical_tags(self.FAINT, library=Library.PSYCHOLOGY, work="存在与时间")
        assert tags.concept == "存在与时间"

    def test_last_resort_is_never_empty(self) -> None:
        assert lexical_tags(self.FAINT, library=Library.PSYCHOLOGY).concept == "未命名概念"

    def test_keyphrases_capped(self) -> None:
        tags = lexical_tags(PHIL, library=Library.PSYCHOLOGY)
        assert len(tags.keywords) <= MAX_KEYWORDS


class TestEmptyFor:
    """文学/诗词要求意象或情感至少占一样；psychology 要求 concept。"""

    def test_psychology_requires_concept(self) -> None:
        assert ChunkTags().is_empty_for(Library.PSYCHOLOGY) is True
        assert ChunkTags(concept="此在").is_empty_for(Library.PSYCHOLOGY) is False

    def test_literature_accepts_either(self) -> None:
        assert ChunkTags().is_empty_for(Library.LITERATURE) is True
        assert ChunkTags(imagery=["落花"]).is_empty_for(Library.LITERATURE) is False
        assert ChunkTags(emotion=["哀伤"]).is_empty_for(Library.LITERATURE) is False
        # 抽象文本产出的 concept 不算数：文学模板不渲染它，检索够不着。
        assert ChunkTags(concept="此在").is_empty_for(Library.LITERATURE) is True


@pytest.mark.parametrize("library", list(Library))
def test_quote_anchor_is_usable(library: Library) -> None:
    # quote 是打标器的身份锚点：它必须真的出现在对应的正文里，否则
    # 校验器会把整条判成「身份对不上」，降级率凭空飙高。
    tags = lexical_tags(PHIL, library=library)
    assert tags.quote
    assert tags.quote in PHIL
