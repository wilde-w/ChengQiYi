"""/api/v1 路由聚合。

新模块在这里挂载；保持 api_router 是唯一的装配点，
让 main.py 不必随功能增长而反复修改。
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import agent, events, health, kb, meta, novel, runs, scene

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(meta.router)
api_router.include_router(runs.router)
api_router.include_router(kb.router)
api_router.include_router(novel.router)
api_router.include_router(events.router)
# 故事工坊。**与流水线并列**，不共用端点也不共用表——见 agent.py 的模块注释。
api_router.include_router(agent.router)
# 对话工坊。三条链路各自独立，见 scene.py 与 `app/scene/__init__.py`。
api_router.include_router(scene.router)
