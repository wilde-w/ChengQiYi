"""FastAPI 应用入口。

启动策略：**基础设施不可达不应阻止应用启动**。
Redis/Qdrant/Neo4j 的连接都是惰性建立的，这里只做初始化与日志，
真正的健康状态由 /api/v1/health/deps 暴露。
这样前端能起来并明确告诉你「Qdrant 没连上」，而不是看到一个起不来的后端。
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.clients.neo4j_client import close_neo4j, init_neo4j
from app.clients.qdrant_client import close_qdrant, init_qdrant
from app.clients.redis_client import close_redis, init_redis
from app.config import ConfigurationError, get_settings
from app.db.session import dispose_engine, init_engine
from app.logging_conf import configure_logging, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        settings = get_settings()
    except ConfigurationError as exc:
        # 配置非法时快速失败并说清原因，不要带着半吊子配置跑起来
        raise RuntimeError(f"配置错误，无法启动：{exc}") from exc

    configure_logging(settings.LOG_LEVEL, settings.LOG_JSON)
    summary = settings.provider_summary()
    log.info(
        "guanxin_starting",
        env=settings.APP_ENV,
        **summary,
    )
    if summary["is_demo"]:
        log.warning(
            "demo_mode_active",
            hint="部分或全部子系统使用 mock 数据，结果仅供演示。配置对应 API key 后自动切换。",
            providers=summary,
        )

    init_engine()
    init_redis()
    init_qdrant()
    init_neo4j()

    # 上次进程被杀时留下的导入作业还挂在 `running` 上，而那一行不会自己
    # 变成终态——前端会对着它无限轮询。收一次，但**收不动也要能启动**：
    # 基础设施不可达不该阻止应用起来，这正是本文件的启动策略。
    from app.services.kb_import_service import reap_stale_imports
    from app.services.run_service import reconcile_orphans

    try:
        await reap_stale_imports()
    except Exception:
        log.warning("kb_import_reap_skipped", hint="数据库不可达，未清理上次残留的导入作业")

    # 分析运行同理，但后果更显眼：一条停在 running 的运行会让前端的开始
    # 按钮永远置灰。同样收不动也要能启动。
    try:
        await reconcile_orphans()
    except Exception:
        log.warning("run_reconcile_skipped", hint="数据库不可达，未清理上次残留的分析运行")

    # 故事工坊的会话是第三种「不会自己变终态」的行。它比前两种更能藏：面板
    # 是开着才看得见的，用户下次打开时那一场已经挂了一整天，界面上写着
    # 「正在写…」而根本没有任务在跑。**必须在这个进程开始接请求之前收掉**，
    # 否则新起的进程会把它当成一个活着的会话。
    from app.services.agent_service import reap_stale_agent_sessions

    try:
        await reap_stale_agent_sessions()
    except Exception:
        log.warning("agent_reap_skipped", hint="数据库不可达，未清理上次残留的故事工坊会话")

    # 对话工坊同理。它的「正在演」尤其能藏：一场 3 人 × 3 轮的戏要跑好几分钟，
    # 进程被杀时后台任务一起没了，而那一行会停在 running；下次打开面板看到的
    # 是「正在演…」，其实台上一个演员都没有。
    from app.services.scene_service import reap_stale_scene_sessions

    try:
        await reap_stale_scene_sessions()
    except Exception:
        log.warning("scene_reap_skipped", hint="数据库不可达，未清理上次残留的对话工坊会话")

    yield

    # 先收敛在跑的后台任务再拆连接池——顺序反了的话，正在提交产物的任务
    # 会在关闭连接的过程中抛一串噪音异常
    from app.services.kb_import_service import shutdown_import_tasks
    from app.services.run_service import shutdown_tasks

    await shutdown_tasks()
    # 导入任务同样要在拆连接池之前收掉：它在跑的时候正握着 Qdrant/Neo4j/PG
    # 三个客户端，此时拆池子会让收尾写入全部失败。
    await shutdown_import_tasks()
    # 故事工坊的一轮最长能跑 180 秒，且中途一直在查库、查向量库。任务被
    # cancel 掉时循环会在下一个轮边界退出并写下终态——所以这一步必须在拆
    # 连接池**之前**，否则收尾那一条 UPDATE 正好撞上已经关掉的引擎。
    from app.services.agent_service import shutdown_agent_tasks

    await shutdown_agent_tasks()
    # 对话工坊的一场戏最长能跑 600 秒（SHOW_TIMEOUT），中途一直在落库、发事件。
    # 同一条理由：必须在拆连接池**之前**收，否则收尾那一条 UPDATE 正好撞上
    # 已经关掉的引擎。
    from app.services.scene_service import shutdown_scene_tasks

    await shutdown_scene_tasks()

    await close_neo4j()
    await close_qdrant()
    await close_redis()
    await dispose_engine()
    log.info("guanxin_stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="观心 — 抖音内容心理洞察工作台",
        description="输入抖音视频链接，输出对大众心理活动的深度分析。",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Last-Event-ID"],
    )

    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()
