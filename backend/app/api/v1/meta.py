"""运行时的元信息：进度分带、枚举字典、能力开关。

这些内容前端**一处都不该硬编码**：
  - 改了节点文案却忘记改前端，界面上就会出现两套说法；
  - 枚举值（库名、段落名、状态）散在 TS 里，加一个库要改两个仓库；
  - 能力开关由服务端决定（MCP 是否可用、四个库是否已摄取），
    前端据此决定哪些控件置灰，而不是乐观地假设一切就绪。

启动时抓一次即可——这些内容在一次部署内不会变。
"""

from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings
from app.constants import EventType, Library, RunStatus, SectionKey
from app.graph.registry import describe

router = APIRouter(tags=["meta"])


@router.get("/meta/pipeline")
async def pipeline() -> dict[str, object]:
    """七个节点及其进度分带。字段名即前端契约（见 api/types.ts）。"""
    return {"nodes": describe()}


@router.get("/meta/enums")
async def enums() -> dict[str, object]:
    """枚举字典。`label` 是给用户看的，`value` 是代码里比较用的。"""
    return {
        "run_status": [{"value": s.value} for s in RunStatus],
        "section": [{"value": k.value, "title": k.title} for k in SectionKey],
        "library": [{"value": lib.value, "label": lib.label} for lib in Library],
        "event_type": [{"value": e.value} for e in EventType],
    }


@router.get("/meta/capabilities")
async def capabilities() -> dict[str, object]:
    """服务端能做什么。

    `mock` 标志是刻意独立于 `is_demo` 暴露的：真实 LLM + mock 抖音
    是开发时最常见的组合，此时结果「一半真一半假」，
    徽章必须说清楚是哪一半，不能让 mock 输出被当成真实抓取。
    """
    settings = get_settings()
    providers = settings.provider_summary()
    return {
        "is_demo": settings.is_demo,
        "providers": providers,
        "douyin_modes": ["mock", "mcp"],
        "depth_levels": ["quick", "standard", "deep"],
    }
