"""Node 1 — 视频引入。

取回视频元数据，以及**可能不存在的**口播文案。

关于口播文案：7 个节点里没有任何一个会做 ASR。只有数据源（MCP）顺带提供时才存在。
所以它是一个明确可选的字段——缺失时前端渲染空状态「未获取到口播文案」。
绝不编造，也绝不因为拿不到就失败。

数据源故障一律降级为 warning 继续：有元数据没评论仍是一份可用的部分报告，
而直接失败会让用户面对一个空界面，什么信息都拿不到。
"""

from __future__ import annotations

from typing import Any

from app.constants import SourceKind
from app.douyin.base import DouyinError
from app.graph.emitting import NodeEmitter
from app.graph.state import AnalysisState, warning_message
from app.logging_conf import get_logger
from app.services.text_source import TEXT_SOURCE_TITLE

log = get_logger(__name__)

NODE = "n1_video"


async def n1_video(state: AnalysisState) -> dict[str, Any]:
    emitter = NodeEmitter(NODE)
    emitter.set_progress(0.15)

    # 分支必须在 provider 导入与 state["aweme_id"] 之前：文本源根本没有 aweme_id，
    # 走到下面就是拿着空串去问抖音。
    if state.get("source_kind") == SourceKind.TEXT:
        return _from_text(state, emitter)

    from app.douyin.factory import get_douyin_provider

    provider = get_douyin_provider()
    aweme_id = state["aweme_id"]

    patch: dict[str, Any] = {}

    try:
        video = await provider.get_video(aweme_id)
    except DouyinError as exc:
        log.warning("n1_video_failed", aweme_id=aweme_id, error=str(exc))
        emitter.warn("video_unavailable", f"未能获取视频信息：{exc}")
        patch["video"] = {"aweme_id": aweme_id, "unavailable": True}
        patch["transcript"] = None
        return {**patch, "warnings": [warning_message("video_unavailable", str(exc))]}

    emitter.set_progress(0.6)
    patch["video"] = video.to_dict()
    emitter.partial("video", patch["video"])

    # 口播文案：拿不到是完全正常的，不是错误
    transcript = await _try_transcript(provider, aweme_id, emitter)
    patch["transcript"] = transcript

    emitter.set_progress(1.0)
    return patch


def _from_text(state: AnalysisState, emitter: NodeEmitter) -> dict[str, Any]:
    """文本源：没有视频可解析，但**要有一条来源行**。

    为什么不干脆不写（`_persist_video` 对空 patch 会早返回，确实不会建行）：
    n7 的来源描述、mock 的正文、将来左栏的视频卡都读 `state["video"]`，
    不写就得让**每一个**读它的地方各自知道「这次没有视频，请显示手动文本」。
    写一行 `title="手动文本"` 让这份知识只存在于一个地方。

    `aweme_id` 是合成的唯一值，不是空串：`Video.aweme_id` 是非空列，
    空串能过约束却会让所有文本运行在任何按 aweme_id 聚合的查询里混成同一坨。
    `text:<run_id>` 一眼看得出不是抖音 id（抖音的是 19 位数字），且各次运行互不相同。

    `stats/author/cover` 一概留空——它们是视频的互动数据，这一行没有视频，
    填 0 或占位符就是在左栏和画像里造一批假数字。前端渲染的是空状态。
    """
    video: dict[str, Any] = {
        "aweme_id": f"text:{state.get('run_id') or ''}",
        "title": TEXT_SOURCE_TITLE,
        "author_name": None,
        "author_id": None,
        "caption": None,
        "stats": {},
        "raw": {},
    }
    emitter.partial("video", video)
    emitter.set_progress(1.0)
    return {"video": video, "transcript": None}


async def _try_transcript(provider: Any, aweme_id: str, emitter: NodeEmitter) -> dict | None:
    """只有 provider 实现了 get_transcript 才尝试。

    用 hasattr 而不是给协议加必选方法：ASR 是可选能力，
    把它塞进必选接口会逼着每个 provider 写一个假实现。
    """
    getter = getattr(provider, "get_transcript", None)
    if getter is None:
        return None
    try:
        transcript = await getter(aweme_id)
    except DouyinError as exc:
        emitter.warn("transcript_unavailable", f"未能获取口播文案：{exc}")
        return None
    except Exception as exc:  # 第三方 ASR 的异常类型无法穷举
        log.warning("n1_transcript_error", error=str(exc))
        return None
    return transcript.to_dict() if transcript is not None else None
