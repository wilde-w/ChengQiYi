"""正文切块：把一段已经写好的文字切成小块，供前端逐字渲染。

**这不是流式**，是把「已经拿到的完整正文」重放成流式的样子。两处需要它：

- `graph/nodes/n7_literary`：四段洞察一次调用产出（`json_mode` 下流式增量
  不是合法 JSON 前缀，见该模块 docstring）；
- `agent/loop`：`ChatModel.stream()` 不支持工具调用，而 agent 的等待时间
  几乎全花在工具调用上，正文本身的流式收益接近于零。

两处共用同一份实现，是为了让「逐字渲染出来的正文」和「落库/快照里的正文」
不可能分叉——一旦分成两份代码，迟早会出现「刷新一下，文字变了」。

模块本身零依赖（只读 `constants`），因此不 import 任何 graph/agent 代码。
"""

from __future__ import annotations

from app.constants import DELTA_CHARS, DELTA_FLOOR

#: 可以断行的字符。句末标点优先——在逗号处断句会让逐字效果看起来像在跳词。
BREAK_CHARS = "。！？…；\n"


def chunk_text(
    text: str, *, size: int = DELTA_CHARS, floor: int = DELTA_FLOOR
) -> list[str]:
    """按标点切块。**只切不删**——`"".join(out) == text` 恒成立。

    逐字比较比循环本身重要：逐字效果与落库正文一旦来自两条路径，
    它们迟早会不一致，而那种不一致表现为「刷新一下，文字变了」。

    `**` 计数为奇数时不在标点处断——那会把一对粗体标记劈成两半，
    用户会在正文里看到孤零零的两个星号。硬上限到了仍然断，因为
    不成对的 `**` 比整段卡死好。
    """
    if not text:
        return []

    out: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        bold_open = buf.count("**") % 2
        # 到了标点且够长就断；`**` 计数为奇数时不断（那会劈开一对粗体标记）。
        # 硬上限到了无论如何都断——不成对的 `**` 比整段卡死好。
        if (ch in BREAK_CHARS and len(buf) >= floor and not bold_open) or len(buf) >= size:
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out


__all__ = ["BREAK_CHARS", "chunk_text"]
