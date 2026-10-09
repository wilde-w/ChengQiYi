"""SQLAlchemy 异步引擎与会话。

引擎是进程级单例，在 FastAPI lifespan 里初始化与释放——每次请求新建引擎
会把连接池的意义完全抵消掉。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def init_engine(dsn: str | None = None) -> AsyncEngine:
    global _engine, _sessionmaker
    if _engine is not None:
        return _engine

    settings = get_settings()
    _engine = create_async_engine(
        dsn or settings.POSTGRES_DSN,
        echo=False,
        pool_pre_ping=True,   # Docker 里的 PG 重启后连接会失效
        pool_size=5,
        max_overflow=10,
    )
    _sessionmaker = async_sessionmaker(
        _engine, class_=AsyncSession, expire_on_commit=False
    )
    return _engine


async def dispose_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None


def get_engine() -> AsyncEngine:
    if _engine is None:
        return init_engine()
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    if _sessionmaker is None:
        init_engine()
    assert _sessionmaker is not None
    return _sessionmaker


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """脚本/后台任务用的事务边界。异常回滚，正常提交。"""
    async with get_sessionmaker()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI 依赖注入用的会话。"""
    async with get_sessionmaker()() as session:
        yield session
