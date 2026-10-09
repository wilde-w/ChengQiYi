"""开发期的建表入口。

生产用 Alembic；这里只是为了 `cli init-db` 能一条命令把开发库拉起来。
"""

from __future__ import annotations

from app.db.session import get_engine
from app.logging_conf import get_logger
from app.models import Base  # noqa: F401 - 导入即注册所有表

log = get_logger(__name__)


async def create_all() -> list[str]:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return sorted(Base.metadata.tables.keys())


async def drop_all() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
