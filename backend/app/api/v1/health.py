"""健康检查与依赖探测。"""

from __future__ import annotations

import time

from fastapi import APIRouter

from app.clients.readiness import probe_all
from app.config import get_settings

router = APIRouter(tags=["health"])

_STARTED_AT = time.time()


@router.get("/health")
async def health() -> dict[str, object]:
    """轻量探活。不触碰任何依赖，供容器 healthcheck 高频调用。"""
    settings = get_settings()
    return {
        "status": "ok",
        "version": "1.0.0",
        "uptime_seconds": round(time.time() - _STARTED_AT, 1),
        "env": settings.APP_ENV,
        "providers": settings.provider_summary(),
    }


@router.get("/health/deps")
async def health_deps() -> dict[str, object]:
    """逐依赖探测。会真实建立连接，因此不要用作高频探活。"""
    results = await probe_all()
    # Redis 挂了不算致命——事件总线会降级到进程内实现，分析仍可运行
    critical = ("postgres", "qdrant", "neo4j")
    degraded = [name for name in critical if not results[name].ok]
    return {
        "status": "ok" if not degraded else "degraded",
        "degraded": degraded,
        "deps": {name: res.as_dict() for name, res in results.items()},
    }
