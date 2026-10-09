"""抖音数据源工厂。"""

from __future__ import annotations

from app.config import get_settings
from app.douyin.base import DouyinProvider
from app.douyin.mock_provider import MockDouyinProvider
from app.logging_conf import get_logger

log = get_logger(__name__)

_provider: DouyinProvider | None = None


def get_douyin_provider(*, fresh: bool = False) -> DouyinProvider:
    global _provider
    if _provider is not None and not fresh:
        return _provider

    settings = get_settings()
    if settings.douyin_mode == "mcp":
        from app.douyin.mcp_provider import McpDouyinProvider

        provider: DouyinProvider = McpDouyinProvider()
        log.info("douyin_provider_ready", mode="mcp", transport=settings.DOUYIN_MCP_TRANSPORT)
    else:
        provider = MockDouyinProvider(
            latency_ms=settings.MOCK_DOUYIN_LATENCY_MS,
            page_size=settings.MOCK_DOUYIN_PAGE_SIZE,
            fail_rate=settings.MOCK_DOUYIN_FAIL_RATE,
        )
        log.info("douyin_provider_ready", mode="mock", page_size=settings.MOCK_DOUYIN_PAGE_SIZE)

    _provider = provider
    return provider


def reset_douyin_provider() -> None:
    global _provider
    _provider = None
