"""古典文学查阅入口的响应 DTO。

**没有请求 DTO**：全是 GET + 查询参数，FastAPI 直接从函数签名生成 OpenAPI。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class NovelTextOut(BaseModel):
    """一次查阅的结果：对方 MCP 渲染好的 Markdown。

    刻意**不解析成结构体**——理由见 `app/novel/mcp_client.py` 的模块注释。
    前端把它整段放进 `<pre>`，不做二次加工。
    """

    tool: str
    #: 对方工具的原始输出。可能很长（整章原文），前端自己决定怎么滚。
    markdown: str


class NovelHealthOut(BaseModel):
    """这个入口当前能不能用。"""

    #: 配置里的总开关。False 时前端应当收起这个 Tab，而不是显示一个报错。
    enabled: bool
    ok: bool
    error: str | None = None
    #: 出问题时**怎么修**。前端把这句显示出来，用户不用翻文档。
    hint: str = ""
    tools: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
