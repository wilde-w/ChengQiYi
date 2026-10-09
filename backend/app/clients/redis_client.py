"""Redis 客户端。

用途有三：事件总线（Streams）、embedding/LLM 结果缓存、运行取消标志。
Redis 不可达时不应让整个应用起不来——降级路径见 graph/bus.py 的 InProcessBus。
"""

from __future__ import annotations

import redis.asyncio as aioredis

from app.config import get_settings
from app.logging_conf import get_logger

log = get_logger(__name__)

_client: aioredis.Redis | None = None


def init_redis(url: str | None = None) -> aioredis.Redis:
    global _client
    if _client is None:
        settings = get_settings()
        _client = aioredis.from_url(
            url or settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            health_check_interval=30,
        )
    return _client


def get_redis() -> aioredis.Redis:
    if _client is None:
        return init_redis()
    return _client


async def close_redis() -> None:
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        except Exception as exc:  # pragma: no cover - 关闭失败无需中断退出
            log.warning("redis_close_failed", error=str(exc))
    _client = None


async def ping() -> bool:
    try:
        return bool(await get_redis().ping())
    except Exception:
        return False
