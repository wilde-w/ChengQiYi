"""知识库导入的请求与响应 DTO。

请求侧只有 multipart 表单（没有 JSON body），所以这里只有响应模型。
表单字段在路由签名里逐一声明——FastAPI 需要它们在函数签名上才能生成
OpenAPI 的 multipart schema。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class KbImportJobOut(BaseModel):
    """一次导入作业的状态。

    **这是前端轮询的唯一载荷**，所以「前端要拿来判断的东西」必须在里面，
    不能让前端自己从 `status` 字符串推。`is_terminal` / `needs_confirm`
    就是为此存在的：终态集合改一次，前端就得跟着改一次，而漏改的表现是
    进度条永远转下去——没人会想到去看后端的状态枚举。
    """

    id: str
    library: str
    library_label: str
    filename: str
    file_size: int

    status: str
    stage: str | None = None
    #: 阶段的中文名。前端直接展示，不必自己维护一份映射。
    stage_label: str | None = None
    progress: int = 0
    message: str | None = None

    chunk_total: int = 0
    chunk_tagged: int = 0
    chunk_indexed: int = 0

    warnings: list[str] = Field(default_factory=list)
    error: str | None = None

    is_terminal: bool = False
    #: True 表示作业停在 `queued` 等用户确认（文件太大）。前端据此弹确认框。
    needs_confirm: bool = False

    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class KbImportCreated(BaseModel):
    job: KbImportJobOut
    #: False 表示停在 `queued` 等确认，需要再调一次 `/start`。
    auto_started: bool
    warnings: list[str] = Field(default_factory=list)
    #: 演示模式下前端要显示「标签由规则抽取」。绝不让 mock 产物被误认作真实分析。
    demo: bool = False


class KbSourceOut(BaseModel):
    """一个已导入的来源（一次上传的文件）。"""

    source_file: str
    library: str
    library_label: str
    chunks: int
    updated_at: datetime | None = None


class KbDocumentOut(BaseModel):
    """某个来源切出来的一条 chunk 预览。

    这是用户**唯一能直接看到「RAG 到底切出了什么」**的地方——没有它，
    切分质量只能靠猜。
    """

    chunk_id: str
    library: str
    source_file: str | None = None
    text_preview: str = ""
    updated_at: datetime | None = None


class KbDeleteResult(BaseModel):
    source_file: str
    removed: int
    message: str


class KbSourceTextOut(BaseModel):
    """某个来源的**原文**（不是切块预览）。

    一次不一定给全：整本红楼梦上百万字，一次性塞进 DOM 会卡，
    所以按字符分页。前端用 `char_count` 与 `offset + len(text)` 算「还有多少没加载」，
    **不要**用 `truncated` 反推位置——它只说「后面还有」。
    """

    source_file: str
    title: str | None = None
    author: str | None = None
    library: str
    #: 全文长度（不是本段的）
    char_count: int
    offset: int
    truncated: bool = False
    text: str
