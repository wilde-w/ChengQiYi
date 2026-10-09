"""纯文本数据源：把用户粘贴的一段文字切成可分析的条目。

这是文本源**唯一的入口**——切分方式、条数上限、超长策略、合成 id 的方案
都收在这里，与 `app/douyin/*` 把「抖音这个源怎么取数」收在一处是同一个分层。
调用方（`create_run` 的入口探针、`n2_comments` 的正式切分）都只调这一个函数，
所以「弹窗里数出 42 条、库里也是 42 条」不需要靠两处实现保持一致。

**切分只切不改**：丢掉空行、去掉每条首尾空白，正文一个字都不动。
广告/灌水/去重的判定全归 `comment_service.clean_comments`——这里再判一次
就等于立了第二套规则，而两套规则在「这条到底算不算数」上迟早会给出不同答案。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.constants import TextMode
from app.douyin.base import CommentItem

#: 单条文本的长度上限。
#:
#: 不是审美取舍。按段落切的时候，一整篇**没有空行**的长文会变成一条几万字的
#: 条目，它进 n3 会被 embedding 接口按输入长度拒掉，整个节点降级成「0 个簇」
#: ——一次完全合法的输入换来一份空报告。截断并如实告知，比这好。
TEXT_ITEM_MAX_CHARS = 2000

#: 文本源的来源名。固定文案、不分份次：用户没有为这次粘贴起过名，
#: 编一个「第 3 次粘贴」之类的东西只会让报告里多出一个他不认识的词。
TEXT_SOURCE_TITLE = "手动文本"

#: 合成评论 id 的位宽。
#:
#: 必须零填充。快照按 `douyin_comment_id` 升序做次级排序（文本源点赞数全为 0，
#: 顺序全靠它），而字典序下 `txt-10` < `txt-2`——第 10 条会排到第 2 条前面。
_ID_WIDTH = 4

#: 段落分隔：一个空行（允许带空格/制表符）。连续多个空行与一个等价。
_BLANK_LINE = re.compile(r"\n[ \t]*\n+")


@dataclass(slots=True)
class SplitStats:
    """切分结果的口径。**每个数字都要能对上左栏看到的东西。**"""

    #: 切出的条目总数（条数上限**之前**）
    total: int = 0
    #: 实际进入分析的条数
    kept: int = 0
    #: 因条数上限被丢弃的条数
    dropped: int = 0
    #: 因单条过长被截断的条数
    truncated: int = 0
    mode: str = TextMode.LINE.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "kept": self.kept,
            "dropped": self.dropped,
            "truncated": self.truncated,
            "mode": self.mode,
        }


def split_text(
    text: str,
    *,
    mode: TextMode | str = TextMode.LINE,
    limit: int = 100,
) -> tuple[list[CommentItem], SplitStats]:
    """把一段文本切成评论条目。返回 (条目, 统计)。

    取前 N 条而不是随机抽样：抖音那边 `raw_items[:limit]` 也是取前 N，
    两条路径的语义应当一致；而且用户粘贴的顺序**是有意义的**
    （它多半是从评论区一路复制的），随机抽样会把这份顺序打散。
    """
    resolved = TextMode(mode)
    parts = _split_parts(text, mode=resolved)
    stats = SplitStats(total=len(parts), mode=resolved.value)

    items: list[CommentItem] = []
    for index, body in enumerate(parts[: max(0, limit)]):
        if len(body) > TEXT_ITEM_MAX_CHARS:
            body = body[:TEXT_ITEM_MAX_CHARS]
            stats.truncated += 1
        items.append(
            CommentItem(
                # 前缀 `txt-` + 序号：抖音的评论 id 是 19 位纯数字，
                # 两者永远不会撞；确定性也让它可复现（重跑给出同一批 id）。
                comment_id=f"txt-{index + 1:0{_ID_WIDTH}d}",
                text=body,
                # 作者与互动数据一概不编：文本源里它们根本不存在，
                # 编一个出来就会在左栏和画像里变成假的「点赞数」。
                like_count=0,
                reply_count=0,
            )
        )

    stats.kept = len(items)
    stats.dropped = stats.total - stats.kept
    return items, stats


def _split_parts(text: str, *, mode: TextMode) -> list[str]:
    """切成原始片段。空片段在这里就被丢掉，后面的索引因此是干净的。"""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    raw = _BLANK_LINE.split(normalized) if mode == TextMode.PARAGRAPH else normalized.split("\n")
    return [part.strip() for part in raw if part.strip()]
