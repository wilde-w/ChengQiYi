"""故事工坊的请求与响应 DTO。

两个刻意的形状决定：

1. **快照里带 `messages`，不带事件。** 刷新页面时，界面要重建的是「用户说了
   什么 / 模型答了什么」（这是对话），而工具卡片那些过程从 `agent_event`
   回放。两者混在一个数组里，前端就得靠 `type` 再分拣一遍，而漏分拣的表现
   是对话里冒出一堆 JSON。
2. **`is_terminal` / `is_running` 由后端算。** 前端不该自己拿 `status` 字符串
   去推——终态集合改一次就要前后端各改一次，而漏改的表现是「永远在转圈」。
   这与 `KbImportJobOut` 是同一条经验。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

#: 粘贴文本的上限。与流水线共用同一个数字——「一段能粘进来的长文本有多长」
#: 不该因为入口不同而变。
from app.schemas.run import TEXT_INPUT_MAX

#: 一段改稿指令的上限。这不是文本材料，几百字足够说清「再暗一点」。
INSTRUCTION_MAX = 2000

#: 目标字数的可选档位与默认值。写成常量而不是散在代码里，是因为前端要拿它
#: 渲染选择器，且提示词里的「{target} 字左右」必须与它同源。
TARGET_CHARS_DEFAULT = 900
TARGET_CHARS_CHOICES = (300, 600, 900, 1500, 2400)


class CreateAgentSessionRequest(BaseModel):
    """开一次写故事。**与当前分析运行无关**——输入就是用户自己粘的东西。"""

    input: str = Field(
        min_length=1,
        max_length=TEXT_INPUT_MAX,
        description="材料原文，通常是一段网友评论",
    )
    #: 关掉之后 `novel_lookup` 根本不进工具 schema。这不是「看得见但会失败」：
    #: 后者只会让模型把轮数浪费在撞墙上。
    allow_novel: bool = Field(default=True, description="允许模型查古典文学作品")
    target_chars: int | None = Field(default=None, ge=300, le=3000)
    #: 开场那句指令。**默认由后端给**（`prompts/story.FIRST_INSTRUCTION`），
    #: 前端不硬编码它：那句话是「这个 agent 被要求做什么」的定义，是提示词的
    #: 一部分，改它的地方应该和 `SYSTEM` 在同一处。前端另存一份的下场是
    #: 「后端改了开场白，跑起来还是旧的那句」——而两边都不报错。
    #: 留这个字段是为了以后能有「让用户自己说第一句」的界面。
    instruction: str | None = Field(default=None, max_length=INSTRUCTION_MAX)

    @field_validator("input")
    @classmethod
    def _strip(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("材料不能只有空白字符")
        return stripped


class SendAgentMessageRequest(BaseModel):
    """接着说一句。「再暗一点」「短一些」「换成第二人称」——都是这样进来的。"""

    text: str = Field(min_length=1, max_length=INSTRUCTION_MAX)

    @field_validator("text")
    @classmethod
    def _strip(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("要说的话不能只有空白字符")
        return stripped


class AgentToolCallOut(BaseModel):
    """assistant 消息里的一次工具调用。落库的是模型给的原始形状。"""

    id: str
    name: str
    #: **原始 JSON 文本**，不是解析后的对象。参数写坏时，界面上要能看到模型
    #: 到底写了什么——那是排查「它为什么老失败」的唯一线索。
    arguments: str = ""


class AgentMessageOut(BaseModel):
    """一条对话消息。前端按 `role` 决定气泡长相。"""

    seq: int
    turn: int
    role: str
    content: str = ""
    tool_calls: list[AgentToolCallOut] = Field(default_factory=list)
    tool_call_id: str | None = None
    #: `role="tool"` 时是工具名；`role="user"` 且为 `force_final` 时，表示这条
    #: 是代码生成的收尾指令——**它不是用户说的话**，前端不能画成他说过。
    name: str | None = None


class AgentSessionOut(BaseModel):
    """会话概要。SSE 的每一次 `agent_*` 事件之后，前端都会重新拉一次它。"""

    id: str
    title: str = ""
    input_chars: int = 0

    status: str
    #: 中文阶段名，直接展示。前端不维护状态映射表。
    stage: str | None = None
    turn: int = 0

    #: 最近一版的正文。等于最后一条 assistant 消息，冗余一份省得前端回扫。
    story: str | None = None

    #: 常驻显示在面板上：这是这个 agent「自主程度」的唯一可核查证据。
    rounds: int = 0
    tool_calls: int = 0

    model: str | None = None
    providers: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)

    error: str | None = None
    is_running: bool = False
    is_terminal: bool = False

    created_at: datetime
    updated_at: datetime


class AgentSessionDetail(AgentSessionOut):
    """打开面板时拉的那一份：会话 + 原文 + 全部对话。"""

    input_text: str
    messages: list[AgentMessageOut] = Field(default_factory=list)
    #: 演示模式徽章。模型是 mock 时必须显示——否则用户会把模板文字当成模型写的。
    demo: bool = False


class LibraryOut(BaseModel):
    """键值对形状与 `/meta` 里那份一致——同一个东西不摆出两种长相。"""

    value: str
    label: str


class AgentCapabilitiesOut(BaseModel):
    """面板能做什么。

    **不探 MCP。** 探一次要起子进程、等两秒，而 `capabilities` 是打开面板就会
    调的接口；把首屏卡在无关的依赖上，是最没必要的一处等待。所以这里只做
    配置层检查（`prerequisites()` 只查文件是否存在），真连不上时那一轮工具
    调用会失败，界面上是一张红卡片——**那是特性**，它演示了 agent 处理失败
    工具的能力。
    """

    input_max: int = TEXT_INPUT_MAX
    instruction_max: int = INSTRUCTION_MAX

    target_chars: int = TARGET_CHARS_DEFAULT
    target_chars_choices: list[int] = Field(default_factory=lambda: list(TARGET_CHARS_CHOICES))

    allow_novel: bool = True
    #: 古典文学 MCP 的配置自检：None 表示看起来没问题，字符串是「为什么起不来」。
    novel_hint: str | None = None

    model: str = ""
    demo: bool = False
    #: 当前有哪些库可供检索，给面板上的说明文案用。
    libraries: list[LibraryOut] = Field(default_factory=list)


__all__ = [
    "INSTRUCTION_MAX",
    "TARGET_CHARS_CHOICES",
    "TARGET_CHARS_DEFAULT",
    "AgentCapabilitiesOut",
    "AgentMessageOut",
    "AgentSessionDetail",
    "AgentSessionOut",
    "AgentToolCallOut",
    "CreateAgentSessionRequest",
    "LibraryOut",
    "SendAgentMessageRequest",
]
