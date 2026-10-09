"""开发用启动器：`python -m app.server`。

存在的唯一理由是 Windows 上的事件循环类型。

uvicorn 0.54 在 win32 且 `use_subprocess=False` 时把事件循环**写死**成
`ProactorEventLoop`（见 `uvicorn/loops/asyncio.py`），而 psycopg3 的异步模式
只支持 `SelectorEventLoop`。后果不是报错退出，而是 LangGraph 的
`AsyncPostgresSaver` 连接失败后**静默降级为内存检查点**：分析照样跑完，
但刷新页面无法恢复、中断后不能续跑。这种「能跑但少一半能力」的失败最难查，
所以宁可换个入口。

设 `asyncio.set_event_loop_policy()` 是没用的——uvicorn 显式传 `loop_factory`
给 `asyncio.run()`，根本不走 policy。它的扩展点是 `loop` 参数接受一个
`"module:attr"` 形式的工厂函数，这里就用那条路径。

`reload` 仍然可用：reloader 在父进程里做文件监听，应用跑在子进程，
子进程沿用同一个 `loop` 配置。
"""

from __future__ import annotations

import asyncio
import sys


def selector_loop_factory() -> asyncio.AbstractEventLoop:
    """给 uvicorn 的事件循环工厂。

    Selector 而非 Proactor 的两个代价，都不影响我们：
      - 不能用 asyncio 的子进程 API（本项目不 spawn 子进程）；
      - 单进程 fd 上限 512（V1 是单人工作台，够用）。
    换来的是一条能用的持久化检查点链路，值。
    """
    return asyncio.SelectorEventLoop()


def main() -> None:
    from app.config import get_settings

    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        # V1 强制单 worker：Redis 降级时事件总线是进程内的、运行任务是
        # 进程内的 asyncio.Task，多 worker 会让两者都失效。
        workers=1,
        loop="app.server:selector_loop_factory" if sys.platform == "win32" else "auto",
        reload=settings.API_RELOAD,
    )


if __name__ == "__main__":
    main()
