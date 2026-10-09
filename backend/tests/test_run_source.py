"""请求契约与运行判别式：source=text 怎么进来、老行怎么被判成抖音。

**纯离线。** 判别式落在 `AnalysisRun.providers` 这个 JSON 列上（不新增数据库列，
理由见 constants.SourceKind），代价是三种历史状态并存：新行带键、更老的行不带键、
以及理论上的未知取值。这个文件把三种都压住——判别式只此一处，它错了，
前端左栏会对着一次文本分析说「N 条评论」。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from app.constants import SourceKind, TextMode
from app.models.run import AnalysisRun
from app.schemas.run import LINK_INPUT_MAX, TEXT_INPUT_MAX, CreateRunRequest
from app.services import run_service
from app.services.run_service import ResolveFailed, _resolve_input, source_kind_of, text_mode_of


def fake_run(**overrides: Any) -> AnalysisRun:
    """脱离会话造一行。判别式只读两个字段，不需要数据库。"""
    base: dict[str, Any] = {
        "id": "r1",
        "thread_id": "r1",
        "input_raw": "x",
        "aweme_id": "7311",
        "depth": "standard",
        "providers": {},
    }
    return AnalysisRun(**{**base, **overrides})


# ----------------------------------------------------------------------
# 请求契约
# ----------------------------------------------------------------------


class TestCreateRunRequest:
    def test_默认是抖音源(self) -> None:
        req = CreateRunRequest(input="https://v.douyin.com/abc/")
        assert req.source == SourceKind.DOUYIN
        assert req.text_mode == TextMode.LINE

    def test_文本源可以很长(self) -> None:
        req = CreateRunRequest(source="text", input="啊" * 30_000)
        assert len(req.input) == 30_000

    def test_链接沿用旧上限(self) -> None:
        with pytest.raises(ValidationError):
            CreateRunRequest(source="douyin", input="x" * (LINK_INPUT_MAX + 1))

    def test_文本也有上限(self) -> None:
        with pytest.raises(ValidationError):
            CreateRunRequest(source="text", input="啊" * (TEXT_INPUT_MAX + 1))

    def test_文本源认段落模式(self) -> None:
        req = CreateRunRequest(source="text", input="一\n\n二", text_mode="paragraph")
        assert req.text_mode == TextMode.PARAGRAPH

    @pytest.mark.parametrize("mode", ["line", "paragraph"])
    def test_抖音源带text_mode不报错(self, mode: str) -> None:
        """这个字段对抖音没意义，但为一个无意义的值让抖音运行 422 不值。"""
        req = CreateRunRequest(source="douyin", input="https://v.douyin.com/abc/", text_mode=mode)
        assert req.source == SourceKind.DOUYIN

    def test_纯空白输入被拒(self) -> None:
        with pytest.raises(ValidationError):
            CreateRunRequest(source="text", input="   \n  \n")

    def test_未知source被拒(self) -> None:
        with pytest.raises(ValidationError):
            CreateRunRequest(source="weibo", input="x")


# ----------------------------------------------------------------------
# _resolve_input：入口探针
# ----------------------------------------------------------------------


class TestResolveInput:
    async def test_文本源不解析链接(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*_: Any, **__: Any) -> Any:
            raise AssertionError("文本源不该去解析链接")

        monkeypatch.setattr("app.douyin.factory.get_douyin_provider", boom)

        ref, resolved = await _resolve_input(
            CreateRunRequest(source="text", input="第一条\n第二条"), SourceKind.TEXT
        )

        assert ref is None
        assert resolved == {"aweme_id": None, "canonical_url": "", "resolved_via": "text"}

    async def test_切不出条目当场报错(self) -> None:
        """入口拦一次，用户就不必先等一条注定失败、只能去读日志的运行。

        注意这里绕开 `CreateRunRequest` 直接喂裸对象：schema 的 `min_length` +
        strip 校验已经挡掉了纯空白，所以这个探针**不是**给 HTTP 路径兜底的
        ——它兜的是干预重跑、脚本直调这类不经过请求模型进来的调用。
        """
        payload = SimpleNamespace(input="\n\n   \n", text_mode="line", comment_limit=100)
        with pytest.raises(ResolveFailed):
            await _resolve_input(payload, SourceKind.TEXT)

    async def test_超长文本在入口不算空(self) -> None:
        """单条被截断不改变「有没有东西」——截断发生在切分里，不是拒绝。"""
        req = CreateRunRequest(source="text", input="啊" * 5000)
        ref, _ = await _resolve_input(req, SourceKind.TEXT)
        assert ref is None

    async def test_抖音源仍走provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeProvider:
            async def resolve_link(self, _: str) -> Any:
                from app.douyin.base import ResolvedRef

                return ResolvedRef(
                    aweme_id="7311", canonical_url="https://x/", resolved_via="regex"
                )

        monkeypatch.setattr("app.douyin.factory.get_douyin_provider", lambda: FakeProvider())

        ref, resolved = await _resolve_input(
            CreateRunRequest(input="https://v.douyin.com/abc/"), SourceKind.DOUYIN
        )

        assert ref.aweme_id == "7311"
        # 抖音路径照旧原样透传 ResolvedRef 的字段（文本源那条用的是 "text"）
        assert resolved["resolved_via"] == "regex"
        assert resolved["canonical_url"] == "https://x/"

    async def test_抖音解析失败转成ResolveFailed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.douyin.base import DouyinError

        class FakeProvider:
            async def resolve_link(self, _: str) -> Any:
                raise DouyinError("这不是链接")

        monkeypatch.setattr("app.douyin.factory.get_douyin_provider", lambda: FakeProvider())

        with pytest.raises(ResolveFailed):
            await _resolve_input(CreateRunRequest(input="随便一段话"), SourceKind.DOUYIN)


# ----------------------------------------------------------------------
# 判别式
# ----------------------------------------------------------------------


class TestSourceKindOf:
    def test_带键的文本行(self) -> None:
        run = fake_run(aweme_id=None, providers={"source_kind": "text"})
        assert source_kind_of(run) == SourceKind.TEXT

    def test_带键的抖音行(self) -> None:
        run = fake_run(providers={"source_kind": "douyin"})
        assert source_kind_of(run) == SourceKind.DOUYIN

    @pytest.mark.parametrize(
        ("aweme_id", "expected"),
        [("7311", SourceKind.DOUYIN), (None, SourceKind.TEXT)],
    )
    def test_老行按aweme_id回落(self, aweme_id: str | None, expected: SourceKind) -> None:
        """这个功能上线前的行没有 source_kind 键——它们全是抖音。"""
        run = fake_run(aweme_id=aweme_id, providers={"douyin": "mcp"})
        assert source_kind_of(run) == expected

    def test_providers为None也不炸(self) -> None:
        assert source_kind_of(fake_run(aweme_id=None, providers=None)) == SourceKind.TEXT

    def test_未知取值回落而不是抛错(self) -> None:
        """将来加了新源、又被回滚，旧代码不该在反序列化时把整个接口带崩。"""
        assert source_kind_of(fake_run(providers={"source_kind": "weibo"})) == SourceKind.DOUYIN
        assert source_kind_of(fake_run(aweme_id=None, providers={"source_kind": "weibo"})) == (
            SourceKind.TEXT
        )


class TestTextModeOf:
    def test_读出段落模式(self) -> None:
        run = fake_run(aweme_id=None, providers={"text_mode": "paragraph"})
        assert text_mode_of(run) == TextMode.PARAGRAPH

    @pytest.mark.parametrize("providers", [{}, None, {"text_mode": None}, {"text_mode": "x"}])
    def test_缺失或非法一律取按行(self, providers: dict[str, Any] | None) -> None:
        """按行是弹窗里的默认值，也是唯一能在任何输入上都切出东西的模式。"""
        assert text_mode_of(fake_run(providers=providers)) == TextMode.LINE


def test_判别式与DTO用的是同一处实现() -> None:
    """`_summary` 不能自己去读 providers 里的键——抄一份就是立了第二处判别式。"""
    import inspect

    from app.api.v1 import runs

    source = inspect.getsource(runs._summary)
    assert "source_kind_of" in source
    assert "providers.get" not in source


def test_文本源不与抖音共享运行身份() -> None:
    """文本源的 aweme_id 是 NULL 而非空串：空串能过非空约束，
    却会让所有文本运行在任何按 aweme_id 聚合的地方混成同一坨。"""
    run = fake_run(aweme_id=None, providers={"source_kind": "text"})
    assert run.aweme_id is None
    assert run_service.source_kind_of(run) == SourceKind.TEXT
