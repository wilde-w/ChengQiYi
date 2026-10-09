"""OpenAI 兼容的 Chat / Embedding 实现。

一份代码同时服务 DeepSeek 与 SiliconFlow（以及任何 OpenAI 兼容端点），
差别全部通过 base_url / model / key 表达——这是选 OpenAI 协议作为抽象面的全部理由。

DeepSeek 的两个坑（都在配置里规避，代码里加了断言）：
1. 不接受 `developer` 角色，只认 `system`（OpenAI 新协议里的 developer 会被拒）。
2. `response_format={"type":"json_object"}` 要求提示词里出现 "json" 字样，
   否则直接报错。我们把这条写进了 prompts/system.py 的输出契约里。
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncIterator, Sequence
from typing import Any

from openai import APIStatusError, AsyncOpenAI, APIConnectionError, APITimeoutError

from app.logging_conf import get_logger
from app.providers.base import (
    ChatMessage,
    ChatResult,
    ChatUsage,
    ProviderError,
    ToolCall,
    render_context,
)

log = get_logger(__name__)

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _to_openai_messages(
    messages: Sequence[ChatMessage], context: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """把内部消息转成 OpenAI 的 wire format。

    三条硬约束（违反任何一条都是 400，且症状离现场很远）：

    1. `role="tool"` 的消息**必须**带 `tool_call_id`，且它与上一轮 assistant
       的某次调用一一对应。少答一个调用，下一轮请求就被拒——症状是「第三轮
       突然报错」，最难查。
    2. 带 `tool_calls` 的 assistant 消息，`content` 不能是 null（模型可以只调
       工具不说话）。空串两边都接受，null 会被拒。
    3. `arguments` 不能是 null，至少要是 `"{}"`。
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m.tool_call_id or "",
                    "content": m.content,
                }
            )
            continue

        item: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_calls:
            item["content"] = m.content or ""
            item["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments or "{}"},
                }
                for c in m.tool_calls
            ]
        out.append(item)

    block = render_context(context)
    if block:
        # 数据块挂到最后一条 user 消息上，避免破坏 system 的输出契约。
        # agent 路径一律不传 context，所以这里不会去动 tool 消息。
        for msg in reversed(out):
            if msg["role"] == "user":
                msg["content"] = msg["content"] + block
                break
        else:
            out.append({"role": "user", "content": block.strip()})
    return out


def _tool_calls_of(message: Any) -> list[ToolCall]:
    """取出这一轮的工具调用。

    无参工具在 DeepSeek 上可能回 `arguments=""` 而不是 `"{}"`，这里统一成
    `"{}"` —— 下游按 JSON 解析时不必再为这个特例写分支。
    """
    out: list[ToolCall] = []
    for i, call in enumerate(getattr(message, "tool_calls", None) or []):
        function = getattr(call, "function", None)
        out.append(
            ToolCall(
                id=getattr(call, "id", None) or f"call_{i}",
                name=getattr(function, "name", None) or "",
                arguments=getattr(function, "arguments", None) or "{}",
            )
        )
    return out


class OpenAICompatChat:
    """ChatModel 协议的 OpenAI 兼容实现。"""

    is_mock = False

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 90.0,
        max_retries: int = 2,
        default_temperature: float = 0.7,
        provider_label: str = "openai_compat",
    ) -> None:
        self.name = model
        self._model = model
        self._provider = provider_label
        self._temperature = default_temperature
        self._max_retries = max(0, max_retries)
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    # ------------------------------------------------------------------
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
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": _to_openai_messages(messages, context),
            "temperature": self._temperature if temperature is None else temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if tools is not None:
            # 工具 schema 由调用方按 OpenAI 的形状组好，这里不改写。
            # **不设 tool_choice**：默认 auto 就是「该调就调、该说就说」；
            # 显式 required 会把收尾轮也逼着调工具，而那一轮的 tools 是空列表。
            payload["tools"] = list(tools)

        data = await self._with_retry(
            lambda: self._client.chat.completions.create(**payload), task=task
        )
        choice = data.choices[0]
        message = choice.message
        usage = data.usage
        return ChatResult(
            text=message.content or "",
            model=data.model or self._model,
            usage=ChatUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
            ),
            finish_reason=choice.finish_reason,
            tool_calls=_tool_calls_of(message),
        )

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        task: str | None = None,
        context: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": _to_openai_messages(messages, context),
            "temperature": self._temperature if temperature is None else temperature,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            stream = await self._client.chat.completions.create(**payload)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except (APIConnectionError, APITimeoutError) as exc:
            raise ProviderError(
                f"{self._provider} 流式连接失败：{exc}", retryable=True, provider=self._provider
            ) from exc
        except APIStatusError as exc:
            raise ProviderError(
                f"{self._provider} 流式请求失败 [{exc.status_code}]：{exc}",
                retryable=exc.status_code in _RETRYABLE_STATUS,
                provider=self._provider,
            ) from exc

    # ------------------------------------------------------------------
    async def _with_retry(self, call, *, task: str | None):
        last: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await call()
            except APIStatusError as exc:
                last = exc
                if exc.status_code not in _RETRYABLE_STATUS:
                    raise ProviderError(
                        f"{self._provider} 请求被拒 [{exc.status_code}]：{exc}",
                        retryable=False,
                        provider=self._provider,
                    ) from exc
                wait = _backoff(attempt)
                log.warning(
                    "llm_retry",
                    provider=self._provider,
                    task=task,
                    status=exc.status_code,
                    attempt=attempt + 1,
                    wait_s=round(wait, 2),
                )
            except (APIConnectionError, APITimeoutError) as exc:
                last = exc
                wait = _backoff(attempt)
                log.warning(
                    "llm_retry",
                    provider=self._provider,
                    task=task,
                    reason=type(exc).__name__,
                    attempt=attempt + 1,
                    wait_s=round(wait, 2),
                )
            if attempt < self._max_retries:
                await asyncio.sleep(wait)

        raise ProviderError(
            f"{self._provider} 重试 {self._max_retries} 次后仍失败：{last}",
            retryable=True,
            provider=self._provider,
        )


class OpenAICompatEmbedding:
    """EmbeddingModel 协议的 OpenAI 兼容实现。"""

    is_mock = False

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        dim: int,
        batch_size: int = 32,
        timeout: float = 60.0,
        max_retries: int = 2,
        provider_label: str = "openai_compat",
    ) -> None:
        self.name = model
        self.dim = dim
        self._model = model
        self._batch = max(1, batch_size)
        self._provider = provider_label
        self._max_retries = max(0, max_retries)
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            batch = list(texts[start : start + self._batch])
            data = await self._with_retry(batch)
            # 服务端不保证返回顺序，按 index 字段重排
            ordered = sorted(data.data, key=lambda d: d.index)
            out.extend([list(item.embedding) for item in ordered])

        for vec in out:
            if len(vec) != self.dim:
                raise ProviderError(
                    f"embedding 维度不符：期望 {self.dim}，实际 {len(vec)}。"
                    f" 请核对 EMBEDDING_DIM 与模型是否匹配；若换了模型需 `ingest-kb --reset`。",
                    retryable=False,
                    provider=self._provider,
                )
        return out

    async def _with_retry(self, batch: list[str]):
        last: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await self._client.embeddings.create(model=self._model, input=batch)
            except APIStatusError as exc:
                last = exc
                if exc.status_code not in _RETRYABLE_STATUS:
                    raise ProviderError(
                        f"{self._provider} embedding 被拒 [{exc.status_code}]：{exc}",
                        retryable=False,
                        provider=self._provider,
                    ) from exc
            except (APIConnectionError, APITimeoutError) as exc:
                last = exc
            if attempt < self._max_retries:
                await asyncio.sleep(_backoff(attempt))
        raise ProviderError(
            f"{self._provider} embedding 重试 {self._max_retries} 次后仍失败：{last}",
            retryable=True,
            provider=self._provider,
        )


def _backoff(attempt: int) -> float:
    """指数退避 + 抖动。抖动是必要的：并发批次同时失败时不应同时重试。"""
    return min(8.0, 0.5 * (2**attempt)) * (0.7 + 0.6 * random.random())


def parse_json_response(text: str) -> dict[str, Any]:
    """从模型输出里抠出 JSON。

    即便用了 json_object 模式，也仍可能遇到模型在前后加说明文字的情况，
    或输出被 max_tokens 截断。这里的策略是：能修就修，修不了就抛出可重试错误，
    绝不返回半个 JSON 让下游静默产生错误结果。
    """
    text = text.strip()
    if not text:
        raise ProviderError("模型返回空内容", retryable=True)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试剥离 markdown 代码围栏
    if text.startswith("```"):
        body = text.split("```")[1] if "```" in text[3:] else text[3:]
        body = body.removeprefix("json").strip()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            pass

    # 最后尝试截取最外层的花括号
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"模型输出不是合法 JSON（可能被截断）：{exc}", retryable=True
            ) from exc

    raise ProviderError("模型输出中找不到 JSON 对象", retryable=True)
