"""LLM / Embedding 的 provider 协议。

两个刻意的设计决定：

1. **`task` 与 `context` 是显式参数，不塞进提示词里让 mock 去解析。**
   真实 provider 把 context 渲染进 prompt；mock provider 直接读它。
   这样节点代码不需要 `if isinstance(llm, MockLLM)` 这种分支——
   分支一旦出现，mock 路径就会和真实路径分叉，mock 也就不再能证明流水线是通的。

2. **协议只要 `complete` / `stream` / `embed` 三个方法。**
   provider 越薄，替换成本越低。重试、缓存、限流放在包装层而不是实现里。

3. **工具调用是 `complete` 的一个参数，不是第二个协议。**
   另开一个 `ToolChatModel` 协议的话，mock 就得分成两半，而 mock 路径一分叉，
   它就再也证明不了真实路径是通的——第 1 条的理由在这里同样成立。
   代价是流水线的那六个 task 也得看得到 `tools` 这个参数，但它们不传，
   于是行为与从前逐字一致（`None` 表示「这次调用与工具无关」，
   与空列表 `[]` 的语义不同：后者是 agent 的收尾轮，明确要求不许再调工具）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class ToolCall:
    """模型发起的一次工具调用。

    `arguments` 保持**原始 JSON 文本**，不在这里解析：解析失败时要把原文
    原样回给模型看（「你给的这段不是合法 JSON」），在这里解析就等于把
    那条最有用的错误信息丢掉了。
    """

    id: str
    name: str
    arguments: str = ""


@dataclass(slots=True)
class ChatMessage:
    role: Role
    content: str
    #: role="assistant" 且模型决定调工具时非空。
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: role="tool" 时必填：应答的是哪一次调用。少一个或对不上，
    #: 下一轮请求会被服务端 400 拒掉。
    tool_call_id: str | None = None
    #: role="tool" 时的工具名。协议上不需要，日志、前端与 mock 需要。
    name: str | None = None


@dataclass(slots=True)
class ChatUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: ChatUsage) -> ChatUsage:
        return ChatUsage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.total_tokens + other.total_tokens,
        )


@dataclass(slots=True)
class ChatResult:
    text: str
    model: str
    usage: ChatUsage = field(default_factory=ChatUsage)
    finish_reason: str | None = None
    #: 非空表示模型这一轮要求调工具，`text` 只是它的中场话（可能为空串）。
    tool_calls: list[ToolCall] = field(default_factory=list)


class ProviderError(RuntimeError):
    """provider 调用失败。retryable 决定调用方是否值得重试。"""

    def __init__(self, message: str, *, retryable: bool = True, provider: str = "") -> None:
        super().__init__(message)
        self.retryable = retryable
        self.provider = provider


@runtime_checkable
class ChatModel(Protocol):
    """对话模型。"""

    name: str
    is_mock: bool

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        task: str | None = None,
        context: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> ChatResult: ...

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        task: str | None = None,
        context: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]: ...


@runtime_checkable
class EmbeddingModel(Protocol):
    """向量模型。dim 必须与 Qdrant collection 的 size 一致。"""

    name: str
    dim: int
    is_mock: bool

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def render_context(context: dict[str, Any] | None) -> str:
    """把结构化 context 渲染成给真实模型看的文本块。

    用显式分隔符而不是裸 JSON——模型更容易把注意力放在标记内的数据上，
    也便于排查「究竟是哪一段数据喂错了」。
    """
    if not context:
        return ""
    import json

    body = json.dumps(context, ensure_ascii=False, indent=2, default=str)
    return f"\n\n<<<DATA\n{body}\nDATA>>>\n"
