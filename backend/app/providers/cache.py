"""向量与 LLM 结果的缓存包装。

存在的核心理由：**重新聚类不该重跑 embedding**。
500 条评论的向量算一次要 10-30 秒，而用户调整簇数时期望的是立刻看到结果。
按文本哈希缓存后，重聚类只需重跑 HDBSCAN（~50ms）。

Redis 不可用时降级为进程内字典——缓存是优化，不是正确性依赖，
绝不能因为缓存后端挂了就让分析失败。
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Sequence
from typing import Any

from app.logging_conf import get_logger
from app.providers.base import ChatMessage, EmbeddingModel

log = get_logger(__name__)

_EMBED_PREFIX = "guanxin:emb:"
_CHAT_PREFIX = "guanxin:chat:"
_MEMORY_LIMIT = 20_000


def text_key(namespace: str, text: str) -> str:
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
    return f"{namespace}:{h}"


def _pack(vec: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack(blob: bytes) -> list[float]:
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


class CachedEmbeddingModel:
    """EmbeddingModel 的缓存包装。

    只缓存**逐条**结果，批量请求会被拆成缓存命中 + 未命中两批，
    未命中的部分才真正打到 API。这样离线重跑几乎不花钱。
    """

    def __init__(self, inner: EmbeddingModel, *, namespace: str = "") -> None:
        self._inner = inner
        self.name = f"cached({inner.name})"
        self.dim = inner.dim
        self.is_mock = inner.is_mock
        self._ns = f"{_EMBED_PREFIX}{namespace}" if namespace else _EMBED_PREFIX
        self._memory: dict[str, list[float]] = {}

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        keys = [text_key(self._ns, t) for t in texts]
        found = await self._mget(keys)

        missing_idx = [i for i, k in enumerate(keys) if k not in found]
        if missing_idx:
            fresh = await self._inner.embed([texts[i] for i in missing_idx])
            to_store: dict[str, list[float]] = {}
            for slot, vec in zip(missing_idx, fresh):
                key = keys[slot]
                found[key] = vec
                to_store[key] = vec
            await self._mset(to_store)

        return [found[k] for k in keys]

    # ------------------------------------------------------------------
    async def _mget(self, keys: list[str]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for key in keys:
            if key in self._memory:
                out[key] = self._memory[key]

        pending = [k for k in keys if k not in out]
        if not pending:
            return out

        try:
            from app.clients.redis_client import get_redis

            values = await get_redis().mget(pending)
            for key, blob in zip(pending, values):
                if blob is None:
                    continue
                try:
                    vec = _unpack(bytes.fromhex(blob) if isinstance(blob, str) else blob)
                except Exception:
                    continue
                out[key] = vec
                self._remember(key, vec)
        except Exception as exc:
            log.debug("embedding_cache_read_skipped", error=str(exc))
        return out

    async def _mset(self, mapping: dict[str, list[float]]) -> None:
        for key, vec in mapping.items():
            self._remember(key, vec)
        if not mapping:
            return
        try:
            from app.clients.redis_client import get_redis

            payload = {k: _pack(v).hex() for k, v in mapping.items()}
            # 向量内容不变，给个较长的 TTL 即可；换模型时 namespace 会变
            await get_redis().mset(payload)
        except Exception as exc:
            log.debug("embedding_cache_write_skipped", error=str(exc))

    def _remember(self, key: str, vec: list[float]) -> None:
        if len(self._memory) >= _MEMORY_LIMIT:
            # 简单淘汰：字典保序，丢掉最早的一批
            for old in list(self._memory)[: _MEMORY_LIMIT // 4]:
                self._memory.pop(old, None)
        self._memory[key] = vec


class CachedChatModel:
    """按 (task, 提示词, 上下文) 哈希缓存 LLM 结果。

    只对**非流式**调用生效。流式段落生成刻意不缓存——用户点「重新生成」
    时期望看到新的结果，命中缓存反而像是按钮坏了。
    """

    def __init__(self, inner, *, namespace: str = "", enabled: bool = True) -> None:
        self._inner = inner
        self._enabled = enabled and not getattr(inner, "is_mock", False)
        self.name = f"cached({inner.name})"
        self.is_mock = inner.is_mock
        self._ns = f"{_CHAT_PREFIX}{namespace}" if namespace else _CHAT_PREFIX
        self._memory: dict[str, str] = {}

    def _key(self, messages: Sequence[ChatMessage], task: str | None, ctx: dict[str, Any] | None) -> str:
        blob = json.dumps(
            {
                "task": task,
                "m": [(m.role, m.content) for m in messages],
                "c": ctx or {},
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return text_key(self._ns, blob)

    async def complete(self, messages, *, task=None, context=None, **kw):
        # 工具调用一律绕开缓存。判据是「**传了** tools」而不是真值判断：
        # 收尾轮传的是空列表（语义是「这一轮不许调工具」），它同样属于
        # agent 路径。理由有两条——
        #   1. 缓存键里没有工具的 schema，命中一次就等于把「当时那个知识库
        #      状态下做的决定」复制到之后所有会话上，而 agent 恰恰是靠重跑
        #      观察世界的；
        #   2. 带 tool_calls 的回复根本不是 `text` 能表达的东西（缓存只存文本）。
        if kw.get("tools") is not None:
            return await self._inner.complete(messages, task=task, context=context, **kw)

        if not self._enabled:
            return await self._inner.complete(messages, task=task, context=context, **kw)

        key = self._key(messages, task, context)
        cached = await self._get(key)
        if cached is not None:
            from app.providers.base import ChatResult

            return ChatResult(text=cached, model=f"cache({self._inner.name})", finish_reason="cache")

        result = await self._inner.complete(messages, task=task, context=context, **kw)
        await self._put(key, result.text)
        return result

    def stream(self, messages, *, task=None, context=None, **kw):
        return self._inner.stream(messages, task=task, context=context, **kw)

    async def _get(self, key: str) -> str | None:
        if key in self._memory:
            return self._memory[key]
        try:
            from app.clients.redis_client import get_redis

            value = await get_redis().get(key)
            if value:
                self._memory[key] = value
            return value
        except Exception:
            return None

    async def _put(self, key: str, value: str) -> None:
        if len(self._memory) >= _MEMORY_LIMIT:
            for old in list(self._memory)[: _MEMORY_LIMIT // 4]:
                self._memory.pop(old, None)
        self._memory[key] = value
        try:
            from app.clients.redis_client import get_redis

            await get_redis().set(key, value, ex=86_400)
        except Exception:
            pass
