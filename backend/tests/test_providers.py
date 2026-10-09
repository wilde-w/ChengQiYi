"""Provider 抽象层：模式解析与 mock 向量的质量。

两组断言各自守护一条产品承诺：
  1. **空 .env 必须能跑** —— 所以「没配密钥 → mock」必须在所有组合下成立。
  2. **离线检索必须合理** —— mock 向量若只是随机数，整个无密钥演示就是假的。
     所以这里断言的是「词法相近的文本余弦更高」，而不只是「两次结果相同」。
"""

from __future__ import annotations

import math
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from app.config import ConfigurationError, Settings
from app.providers.base import ChatMessage, ChatResult, ToolCall
from app.providers.mock_embedding import deterministic_embed

# 清空这些变量后构造 Settings，才能测到「空 .env」的真实行为
_ENV_KEYS = (
    "MOCK_MODE",
    "LLM_PROVIDER",
    "DEEPSEEK_API_KEY",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_API_KEY",
    "DOUYIN_PROVIDER",
    "DOUYIN_MCP_URL",
    "DOUYIN_MCP_COMMAND",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    # 同时屏蔽 .env 文件：否则「没配密钥 → mock」这条承诺会在任何配好 .env 的
    # 机器上失败——测的就是这台机器的配置，不是代码。
    monkeypatch.setitem(Settings.model_config, "env_file", None)


def make_settings(**overrides: object) -> Settings:
    """绕开 .env 文件与真实环境变量，只测解析逻辑本身。"""
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# 模式解析：空 .env 必须能跑
# ----------------------------------------------------------------------


class TestModeResolution:
    def test_empty_env_falls_back_to_mock_everywhere(self, clean_env: None) -> None:
        s = make_settings()
        assert s.llm_mode == "mock"
        assert s.embedding_mode == "mock"
        assert s.douyin_mode == "mock"
        assert s.is_demo is True

    def test_subsystems_resolve_independently(self, clean_env: None) -> None:
        """最常见的开发配置：真 DeepSeek + mock 抖音 + 真 embedding。"""
        s = make_settings(DEEPSEEK_API_KEY="sk-x", EMBEDDING_API_KEY="sk-y")
        assert s.llm_mode == "deepseek"
        assert s.embedding_mode == "api"
        assert s.douyin_mode == "mock"       # 没配 MCP，独立降级
        assert s.is_demo is True             # 只要有一个 mock 就是演示模式

    def test_key_presence_flips_mode(self, clean_env: None) -> None:
        assert make_settings().llm_mode == "mock"
        assert make_settings(DEEPSEEK_API_KEY="sk-x").llm_mode == "deepseek"

    def test_whitespace_key_counts_as_missing(self, clean_env: None) -> None:
        assert make_settings(DEEPSEEK_API_KEY="   ").llm_mode == "mock"

    def test_explicit_mock_wins_over_configured_key(self, clean_env: None) -> None:
        """写了 mock 就该得到 mock，哪怕密钥配好了。"""
        s = make_settings(LLM_PROVIDER="mock", DEEPSEEK_API_KEY="sk-x")
        assert s.llm_mode == "mock"

    def test_mock_mode_always_overrides_everything(self, clean_env: None) -> None:
        s = make_settings(
            MOCK_MODE="always",
            DEEPSEEK_API_KEY="sk-x",
            EMBEDDING_API_KEY="sk-y",
            DOUYIN_MCP_URL="http://localhost:9999",
        )
        assert (s.llm_mode, s.embedding_mode, s.douyin_mode) == ("mock", "mock", "mock")

    def test_mock_mode_never_raises_when_key_missing(self, clean_env: None) -> None:
        """绝不静默降级——真实配置下悄悄返回假数据是最危险的失败模式。"""
        s = make_settings(MOCK_MODE="never")
        with pytest.raises(ConfigurationError) as exc:
            _ = s.llm_mode
        assert "DEEPSEEK_API_KEY" in str(exc.value)

    def test_mock_mode_never_succeeds_when_configured(self, clean_env: None) -> None:
        s = make_settings(MOCK_MODE="never", DEEPSEEK_API_KEY="sk-x")
        assert s.llm_mode == "deepseek"

    def test_explicit_real_provider_without_key_raises(self, clean_env: None) -> None:
        """点名要真 provider 但没密钥 → 硬失败，不降级。"""
        s = make_settings(LLM_PROVIDER="deepseek")
        with pytest.raises(ConfigurationError):
            _ = s.llm_mode

    @pytest.mark.parametrize(
        "override,expected",
        [
            ({"DOUYIN_MCP_COMMAND": "python -m douyin_mcp"}, "mcp"),
            ({"DOUYIN_MCP_URL": "http://localhost:8080/mcp"}, "mcp"),
            ({}, "mock"),
        ],
    )
    def test_douyin_requires_mcp_config(
        self, clean_env: None, override: dict[str, str], expected: str
    ) -> None:
        assert make_settings(**override).douyin_mode == expected

    def test_provider_summary_shape(self, clean_env: None) -> None:
        summary = make_settings().provider_summary()
        assert summary == {
            "llm": "mock",
            "embedding": "mock",
            "douyin": "mock",
            "mock_mode": "auto",
            "is_demo": True,
        }

    @pytest.mark.parametrize(
        "depth,expected",
        [("quick", (3, 2, 1)), ("standard", (6, 4, 3)), ("deep", (9, 6, 5))],
    )
    def test_retrieval_quota(self, depth: str, expected: tuple[int, int, int]) -> None:
        assert make_settings().retrieval_quota(depth) == expected

    def test_unknown_depth_falls_back_to_standard(self) -> None:
        assert make_settings().retrieval_quota("nonexistent") == (6, 4, 3)


# ----------------------------------------------------------------------
# mock 向量
# ----------------------------------------------------------------------


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class TestDeterministicEmbed:
    DIM = 1024

    def test_same_text_same_vector(self) -> None:
        assert deterministic_embed("我想我妈妈了", self.DIM) == deterministic_embed("我想我妈妈了", self.DIM)

    def test_different_text_different_vector(self) -> None:
        a = deterministic_embed("我想我妈妈了", self.DIM)
        b = deterministic_embed("今天股市大涨", self.DIM)
        assert a != b
        assert cosine(a, b) < 0.9

    def test_is_l2_normalized(self) -> None:
        vec = deterministic_embed("落花有意流水无情", self.DIM)
        assert math.isclose(math.sqrt(sum(v * v for v in vec)), 1.0, abs_tol=1e-9)

    def test_dimension_is_respected(self) -> None:
        assert len(deterministic_embed("测试", 256)) == 256

    def test_empty_text_returns_unit_vector_not_zero(self) -> None:
        """零向量会让下游的余弦相似度全部变成 NaN，必须避免。"""
        vec = deterministic_embed("", self.DIM)
        assert any(v != 0 for v in vec)
        assert math.isclose(math.sqrt(sum(v * v for v in vec)), 1.0, abs_tol=1e-9)

    def test_punctuation_only_returns_unit_vector(self) -> None:
        vec = deterministic_embed("。。。！！！", self.DIM)
        assert math.isclose(math.sqrt(sum(v * v for v in vec)), 1.0, abs_tol=1e-9)

    def test_lexical_similarity_is_meaningful(self) -> None:
        """这是 mock 向量存在的全部理由：离线检索要返回合理近邻而非随机结果。"""
        query = deterministic_embed("想念妈妈 哀伤", self.DIM)
        near = deterministic_embed("我非常想念我的妈妈，很难过", self.DIM)
        far = deterministic_embed("股票基金今天大涨了三个点", self.DIM)
        assert cosine(query, near) > cosine(query, far) + 0.2

    def test_whitespace_and_case_do_not_matter(self) -> None:
        """归一化在哈希之前——否则「落花 哀伤」与「落花哀伤」检索不到彼此。"""
        assert deterministic_embed("落花 哀伤", self.DIM) == deterministic_embed("落花哀伤", self.DIM)
        assert deterministic_embed("Hello", self.DIM) == deterministic_embed("hello", self.DIM)

    def test_stable_across_processes(self) -> None:
        """跨进程确定是硬约束：摄取进程与查询进程算出的向量必须一致，
        否则表现就是「明明入库了却检索不到」。用内置 hash() 会随机加盐而失败。
        """
        backend = Path(__file__).resolve().parents[1]
        code = (
            "import sys; sys.path.insert(0, r'%s');"
            "from app.providers.mock_embedding import deterministic_embed;"
            "print(sum(deterministic_embed('十年生死两茫茫', 64)))" % backend
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
        )
        assert proc.returncode == 0, proc.stderr
        expected = sum(deterministic_embed("十年生死两茫茫", 64))
        assert math.isclose(float(proc.stdout.strip()), expected, rel_tol=1e-12)


class TestMockEmbeddingModel:
    async def test_implements_protocol_and_batches(self) -> None:
        from app.providers.mock_embedding import MockEmbeddingModel

        model = MockEmbeddingModel(dim=128)
        assert model.is_mock is True
        vectors = await model.embed(["甲", "乙", "丙"])
        assert len(vectors) == 3
        assert all(len(v) == 128 for v in vectors)

    async def test_empty_batch(self) -> None:
        from app.providers.mock_embedding import MockEmbeddingModel

        assert await MockEmbeddingModel(dim=64).embed([]) == []


# ----------------------------------------------------------------------
# 工厂
# ----------------------------------------------------------------------


class TestFactory:
    def test_factory_returns_mock_without_keys(self, clean_env: None, monkeypatch) -> None:
        from app import config, providers
        from app.config import reset_settings_cache
        from app.providers import factory

        reset_settings_cache()
        factory.reset_provider_cache()
        try:
            assert factory.get_chat_model().is_mock is True
            assert factory.get_embedding_model().is_mock is True
            assert factory.get_embedding_model().dim == 1024
        finally:
            factory.reset_provider_cache()
            reset_settings_cache()

    def test_factory_is_cached(self, clean_env: None) -> None:
        from app.providers import factory

        factory.reset_provider_cache()
        try:
            assert factory.get_embedding_model() is factory.get_embedding_model()
            assert factory.get_embedding_model(fresh=True) is not factory.get_embedding_model()
        finally:
            factory.reset_provider_cache()


# ----------------------------------------------------------------------
# 工具调用的 wire format
#
# 这里是唯一「写错了不会立刻报错、而是第三轮突然 400」的地方，所以逐字段断言，
# 而不是断言「能跑通」。三条硬约束见 `openai_compat._to_openai_messages`。
# ----------------------------------------------------------------------


class TestToolWireFormat:
    def test_带tool_calls的助手消息_content不能是null(self) -> None:
        from app.providers.openai_compat import _to_openai_messages

        out = _to_openai_messages(
            [ChatMessage("assistant", "", tool_calls=[ToolCall("c1", "kb_search", '{"query":"x"}')])],
            None,
        )
        assert out[0]["content"] == ""  # 空串两边都收，null 会被拒
        assert out[0]["tool_calls"][0]["id"] == "c1"
        assert out[0]["tool_calls"][0]["type"] == "function"
        assert out[0]["tool_calls"][0]["function"] == {"name": "kb_search", "arguments": '{"query":"x"}'}

    def test_tool角色必须带tool_call_id(self) -> None:
        from app.providers.openai_compat import _to_openai_messages

        out = _to_openai_messages([ChatMessage("tool", "结果", tool_call_id="c1", name="kb_search")], None)
        assert out[0] == {"role": "tool", "tool_call_id": "c1", "content": "结果"}

    def test_空arguments补成空对象(self) -> None:
        """无参工具在 DeepSeek 上回 ""，直接发回去会被拒。"""
        from app.providers.openai_compat import _to_openai_messages

        out = _to_openai_messages([ChatMessage("assistant", "x", tool_calls=[ToolCall("c1", "kb_sources")])], None)
        assert out[0]["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_解析返回的tool_calls(self) -> None:
        from types import SimpleNamespace

        from app.providers.openai_compat import _tool_calls_of

        message = SimpleNamespace(
            tool_calls=[
                SimpleNamespace(id="c1", function=SimpleNamespace(name="kb_search", arguments='{"a":1}')),
                SimpleNamespace(id=None, function=SimpleNamespace(name="kb_sources", arguments=None)),
            ]
        )
        calls = _tool_calls_of(message)
        assert [(c.id, c.name, c.arguments) for c in calls] == [
            ("c1", "kb_search", '{"a":1}'),
            ("call_1", "kb_sources", "{}"),
        ]

    def test_没有tool_calls时是空列表(self) -> None:
        from types import SimpleNamespace

        from app.providers.openai_compat import _tool_calls_of

        assert _tool_calls_of(SimpleNamespace(tool_calls=None)) == []
        assert _tool_calls_of(SimpleNamespace()) == []

    def test_渲染context时不会破坏tool消息(self) -> None:
        """流水线的数据块挂到最后一条 user 上；agent 的历史里有 tool 消息，
        它必须被跳过，不能被追加成 `role="tool"` 的上下文。"""
        from app.providers.openai_compat import _to_openai_messages

        out = _to_openai_messages(
            [
                ChatMessage("user", "写故事"),
                ChatMessage("assistant", "", tool_calls=[ToolCall("c1", "kb_search", "{}")]),
                ChatMessage("tool", "结果", tool_call_id="c1", name="kb_search"),
            ],
            {"k": 1},
        )
        assert "<<<DATA" in out[0]["content"]
        assert out[2]["content"] == "结果"


class CountingChat:
    """只数调用次数的最小 ChatModel。`is_mock = False` 才能让缓存真的生效。"""

    is_mock = False

    def __init__(self) -> None:
        self.name = "counting"
        self.calls = 0

    async def complete(self, messages, *, task=None, context=None, **kw) -> ChatResult:
        self.calls += 1
        return ChatResult(text="回答", model=self.name, finish_reason="stop")

    def stream(self, messages, **kw):  # pragma: no cover — 本文件用不到
        raise NotImplementedError


def _fresh_text() -> str:
    """每次都不一样。否则上一轮测试留在 Redis 里的同一条记录会让
    「第一次必定未命中」这个前提不成立，用例就会随环境偶发失败。"""
    return f"缓存用例 {uuid.uuid4().hex}"


class TestToolCaching:
    """带工具一律绕开缓存。判据是「传了 tools」而不是真值判断——
    收尾轮传的是空列表，它同样属于 agent 路径。"""

    async def test_带tools时不写缓存(self) -> None:
        from app.providers.cache import CachedChatModel

        inner = CountingChat()
        model = CachedChatModel(inner, enabled=True)
        text = _fresh_text()
        for _ in range(3):
            await model.complete([ChatMessage("user", text)], task="agent_turn", tools=[])

        assert inner.calls == 3

    async def test_收尾轮的空列表也算带tools(self) -> None:
        """同一串消息：不传 tools 时第二次命中缓存，传空列表就一次都不命中。
        少写这一个分支，收尾轮会把「上次这个世界里的答案」当成这次的稿子。"""
        from app.providers.cache import CachedChatModel

        inner = CountingChat()
        model = CachedChatModel(inner, enabled=True)
        messages = [ChatMessage("user", _fresh_text())]

        await model.complete(messages, task="agent_turn", tools=None)
        await model.complete(messages, task="agent_turn", tools=None)
        assert inner.calls == 1  # 流水线路径：照旧命中

        await model.complete(messages, task="agent_turn", tools=[])
        await model.complete(messages, task="agent_turn", tools=[])
        assert inner.calls == 3  # agent 路径：两次都打过去

    async def test_不带tools时照旧走缓存(self) -> None:
        """流水线的六个 task 一行没改，缓存也照旧。"""
        from app.providers.cache import CachedChatModel

        inner = CountingChat()
        model = CachedChatModel(inner, enabled=True)
        messages = [ChatMessage("user", _fresh_text())]
        first = await model.complete(messages, task="literary")
        second = await model.complete(messages, task="literary")

        assert inner.calls == 1
        assert second.finish_reason == "cache"
        assert second.text == first.text
