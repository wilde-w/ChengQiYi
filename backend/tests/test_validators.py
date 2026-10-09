"""引用标记的抽取、剥离与覆盖率。

这里守的是 n7 的重试阈值唯一的依据。覆盖率算错的方向有两种，都有害：

  - **算高了** → 一段引用了不存在证据的正文被判定合格，用户点开 chip
    什么也看不到，而徽章上写着「引用可追溯 9/9」。
  - **算低了** → 每次演示都触发一次注定失败的重生成，把一个健康的结果
    报成有问题的。段落覆盖率（0.32）与唯一 id 覆盖率（0.46）在黄金 mock
    路径上就是这样，所以它们只做诊断指标，不驱动重试。
"""

from __future__ import annotations

import pytest

from app.graph.validators import (
    ATTEMPT_RE,
    MARKER_RE,
    coverage,
    extract_markers,
    filter_ids,
    sanitize,
)

POOL = {"psy:klass:001", "lit:shen-yuan:002", "po:jiang-cheng-zi:003"}


class TestExtract:
    def test_按出现顺序取出且不去重(self):
        text = "甲[[ev:psy:klass:001]]乙[[ev:lit:shen-yuan:002]]丙[[ev:psy:klass:001]]"
        assert extract_markers(text) == ["psy:klass:001", "lit:shen-yuan:002", "psy:klass:001"]

    def test_容忍多余空白(self):
        assert extract_markers("[[ ev : psy:klass:001 ]]") == ["psy:klass:001"]

    def test_畸形标记不被认作合法标记(self):
        # 形态不对的，`extract_markers` 一只都不认；剥离那是 `ATTEMPT_RE` 的事。
        for bad in ("[[ev:两条词 a b]]", "[[ev:]]", "[[ev:没闭合]", "[[ev:"):
            assert extract_markers(bad) == []

    def test_多余括号时只认出其中合法的那一枚(self):
        # `[[ev:x]]]` 里前两个括号就是一个完整的标记，多出来的是残渣。
        # 这里如实承认：合法标记的定义是形态，不是「上下文里没有多余字符」。
        assert extract_markers("[[ev:x]]]") == ["x"]

    def test_正则与剥离正则的口径差(self):
        # 两条正则故意不同：一条管「合法」，一条管「看起来像」。
        # 合并成一条的话，要么畸形标记漏进正文，要么合法标记被误判。
        text = "[[ev:psy:klass:001]][[ev:坏 id]]"
        assert len(MARKER_RE.findall(text)) == 1
        assert len(ATTEMPT_RE.findall(text)) == 2


class TestSanitize:
    def test_合法标记换成脚注记号并记进_citations(self):
        clean, citations, report = sanitize("想念是未完成的事[[ev:psy:klass:001]]。", POOL)
        assert clean == "想念是未完成的事[^1]。"
        assert citations == [{"evidence_id": "psy:klass:001", "marker": "[^1]"}]
        assert coverage(report) == 1.0

    def test_剥离后不残留任何半个括号(self):
        messy = "甲[[ev:psy:klass:001]]乙[[EV: 乱写 ]]丙[[ev:没闭合]丁"
        clean, _, _ = sanitize(messy, POOL)
        assert "[[" not in clean
        assert "]]" not in clean or clean.count("]]") == 0
        assert "ev:" not in clean

    def test_池外的_id_丢弃且计入分母(self):
        clean, citations, report = sanitize("甲[[ev:psy:klass:001]]乙[[ev:查无此条]]", POOL)
        assert clean == "甲[^1]乙"
        assert len(citations) == 1
        # 分母记的是**出现次数**：读者在那个位置确实看到过一次引用。
        assert report["total"] == 2
        assert coverage(report) == 0.5

    def test_同一个_id_出现三次分母计三(self):
        text = "甲[[ev:psy:klass:001]]乙[[ev:psy:klass:001]]丙[[ev:psy:klass:001]]"
        clean, citations, report = sanitize(text, POOL)
        assert report["total"] == 3
        assert coverage(report) == 1.0  # 三次都有效
        # 每一次出现各给一枚 chip：它们指向同一条证据，但在正文里是三个位置。
        assert [c["marker"] for c in citations] == ["[^1]", "[^2]", "[^3]"]

    def test_全部无效时为_0(self):
        _, citations, report = sanitize("甲[[ev:甲]]乙[[ev:乙]]", POOL)
        assert citations == []
        assert coverage(report) == 0.0

    def test_一次都没出现时为_0_而不是_1(self):
        # 空集不算满分。是「池子空」还是「模型忘了写」，由调用方判断——
        # 两者该有不同处置，而这个函数看不到池子。
        _, _, report = sanitize("这段话一个标记都没有。", POOL)
        assert report["total"] == 0
        assert coverage(report) == 0.0

    def test_空输入不抛(self):
        assert sanitize("", POOL) == ("", [], {"total": 0, "valid": 0, "invalid": 0, "dropped_ids": []})

    def test_超长的_id_照样能被认出来(self):
        # 真实 chunk_id 长这样：`psy:continuing-bonds:klass:001`。
        # 正则是非贪婪的，写到什么长度都得完整吃下去。
        long_id = "psy:" + "x" * 120 + ":001"
        clean, citations, _ = sanitize(f"甲[[ev:{long_id}]]", {long_id})
        assert citations[0]["evidence_id"] == long_id
        assert clean == "甲[^1]"

    def test_括号里带下划线或点也认得(self):
        pool = {"a.b_c-d"}
        _, citations, _ = sanitize("[[ev:a.b_c-d]]", pool)
        assert citations[0]["evidence_id"] == "a.b_c-d"


class TestFilterIds:
    def test_保序去重并分出被丢的(self):
        kept, dropped = filter_ids(["psy:klass:001", "查无", "psy:klass:001", "lit:shen-yuan:002"], POOL)
        assert kept == ["psy:klass:001", "lit:shen-yuan:002"]
        assert dropped == ["查无"]

    def test_非列表输入不抛(self):
        assert filter_ids(None, POOL) == ([], [])
        assert filter_ids("psy:klass:001", POOL) == (["psy:klass:001"], [])
        assert filter_ids({"a": 1}, POOL) == ([], [])
        assert filter_ids([1, 2, None], POOL) == ([], [])

    def test_limit_在去重之后生效(self):
        kept, _ = filter_ids(["psy:klass:001", "psy:klass:001", "lit:shen-yuan:002"], POOL, limit=2)
        assert kept == ["psy:klass:001", "lit:shen-yuan:002"]


class TestCoverage:
    @pytest.mark.parametrize(
        ("total", "valid", "expected"),
        [(9, 9, 1.0), (9, 3, 0.3333), (2, 1, 0.5), (3, 0, 0.0), (0, 0, 0.0)],
    )
    def test_定义(self, total: int, valid: int, expected: float):
        assert coverage({"total": total, "valid": valid}) == expected
