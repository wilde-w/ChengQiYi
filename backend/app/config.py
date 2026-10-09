"""应用配置。

设计要旨：**没有任何密钥是必需的**。空 .env 必须能启动整个系统，三个子系统
（LLM / Embedding / 抖音数据源）各自独立解析，密钥缺失时降级为 mock。

独立解析是刻意的——最常见的开发配置是「真 DeepSeek + mock 抖音 + API embedding」，
这三者互不相关，不该被绑成一个开关。

MOCK_MODE=never 时缺密钥会在启动阶段直接抛错，绝不静默降级——真实配置下
悄悄返回 mock 结果是最危险的失败模式。
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Mode = Literal["mock", "deepseek", "api", "mcp"]

#: 仓库根（ChengQiYi/）。config.py 在 backend/app/ 下，往上三层。
#: 只用来推「同级的另一个仓库」这类默认值，不参与运行期路径解析。
_REPO_ROOT = Path(__file__).resolve().parents[2]


class ConfigurationError(RuntimeError):
    """配置非法，无法安全启动。"""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # backend/ 下运行时要能读到仓库根目录的 .env
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- 应用 ----------
    APP_ENV: Literal["dev", "test", "prod"] = "dev"
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    API_RELOAD: bool = False
    CORS_ORIGINS: str = "http://localhost:5173"
    RUN_MAX_CONCURRENCY: int = 2
    SSE_HEARTBEAT_SECONDS: int = 15
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False

    # ---------- mock 降级 ----------
    MOCK_MODE: Literal["auto", "always", "never"] = "auto"
    MOCK_LLM_LATENCY_MS: int = 900
    MOCK_DOUYIN_LATENCY_MS: int = 400
    MOCK_DOUYIN_PAGE_SIZE: int = 20
    MOCK_DOUYIN_FAIL_RATE: float = 0.0
    MOCK_MIN_STEP_DELAY_MS: int = 120

    # ---------- LLM ----------
    LLM_PROVIDER: Literal["auto", "deepseek", "mock"] = "auto"
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-chat"
    LLM_TEMPERATURE: float = 0.7
    LLM_TIMEOUT_SECONDS: int = 90
    LLM_MAX_RETRIES: int = 2

    # ---------- Embedding ----------
    EMBEDDING_PROVIDER: Literal["auto", "api", "mock"] = "auto"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = "https://api.siliconflow.cn/v1"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIM: int = 1024
    EMBEDDING_BATCH_SIZE: int = 32

    # ---------- 抖音 ----------
    DOUYIN_PROVIDER: Literal["auto", "mcp", "mock"] = "auto"
    DOUYIN_MCP_TRANSPORT: Literal["stdio", "http"] = "stdio"
    DOUYIN_MCP_COMMAND: str = ""
    DOUYIN_MCP_ARGS: str = "[]"
    DOUYIN_MCP_URL: str = ""
    DOUYIN_MCP_TOOLS: str = '{"video":"get_video_detail","comments":"get_video_comments","asr":null}'
    DOUYIN_MCP_FIELD_MAP: str = "{}"
    DOUYIN_REQUEST_TIMEOUT: int = 20

    # ---------- 古典文学 MCP ----------
    # 这是**另一个仓**里的知识库（ClassicalNovelProject），通过它自己的
    # MCP Server 接入，只做「查阅原文」这一件事，不进分析流水线。
    #
    # 与抖音那套的关键差别：那边是第三方 MCP、用 npx 拉起、返回 JSON；
    # 这边是我们自己的、要用对方的 conda 解释器（`pip install -e` 装在那个
    # 环境里）、并且**必须 cwd 到对方仓库**（它的 DB_URL 是相对路径）。
    #
    # 默认值指向本机实测可用的路径，所以**不改 .env 也能跑**；
    # 换机器时在 .env 里把这几个键覆盖掉即可（见 .env.example）。
    NOVEL_MCP_ENABLED: bool = True
    NOVEL_MCP_TRANSPORT: Literal["stdio", "http"] = "stdio"
    NOVEL_MCP_COMMAND: str = "D:/Program Files/miniconda3/envs/pythonartproject/python.exe"
    NOVEL_MCP_ARGS: str = '["-m","app.mcp.server"]'
    #: 对方仓库的根目录。默认取 workspace 下与观心同级的那一个。
    NOVEL_MCP_CWD: str = str(_REPO_ROOT.parent / "ClassicalNovelProject")
    NOVEL_MCP_URL: str = ""
    #: 对方库的相对路径，按上面 cwd 解析。
    NOVEL_MCP_DB_URL: str = "sqlite+aiosqlite:///./data/novel.db"
    #: 每次调用都要起一个子进程 + 对方 import sqlalchemy，实测约 1.7s。
    #: 超时给到 30s 是为了容忍冷启动和杀毒软件扫新进程。
    NOVEL_MCP_TIMEOUT: int = 30

    # ---------- 基础设施 ----------
    # 一律写 127.0.0.1 而不是 localhost。Windows 上 localhost 会优先解析到 IPv6
    # 的 ::1，而 Docker Desktop 的端口代理只监听 IPv4：redis 会直接连接超时
    # （不重试），postgres/neo4j 则会先耗掉 ~20s 再回退到 IPv4。
    # 实测 redis://localhost 4.02s 超时 vs redis://127.0.0.1 0.01s 连通。
    POSTGRES_DSN: str = "postgresql+asyncpg://guanxin:guanxin@127.0.0.1:55432/guanxin"
    REDIS_URL: str = "redis://127.0.0.1:56379/0"
    QDRANT_URL: str = "http://127.0.0.1:56333"
    QDRANT_COLLECTION: str = "guanxin_kb_v1"
    NEO4J_URI: str = "bolt://127.0.0.1:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "guanxin12345"

    # ---------- 流水线 ----------
    CLUSTER_MIN_SIZE_AUTO: bool = True
    CLUSTER_MIN_SAMPLES: int = 2
    CLUSTER_MAX_DEFAULT: int = 5
    RETRIEVAL_QUOTA_QUICK: str = "3,2,1"
    RETRIEVAL_QUOTA_STANDARD: str = "6,4,3"
    RETRIEVAL_QUOTA_DEEP: str = "9,6,5"
    CITATION_MIN_COVERAGE: float = 0.6

    # ---------- 知识库导入 ----------
    #: 上传文件大小上限（纯文本）。5MB 纯文本约 250 万字，一本《存在与时间》
    #: 中文版是 40 万字左右——上限设在这里，是为了挡住「误传了一个日志文件」
    #: 这类事故，而不是真的要限制书的长度。
    KB_IMPORT_MAX_BYTES: int = 5 * 1024 * 1024
    #: epub 的体积上限单独放宽。电子书里大半体积是墨迹图片，正文反而只有
    #: 几十万字——用纯文本那条尺子量它，一本 5.4MB 的哈维就进不来。
    #: 解压总量与段数另有各自的闸门（见 `app/kb/epub.py`、下面的块数上限）。
    KB_IMPORT_EPUB_MAX_BYTES: int = 20 * 1024 * 1024
    #: 段数硬上限。红楼梦 epub 切出 8107 段（同书 txt 版 7854 段），
    #: 上限取 12000 让这类大部头能一次进来——真正的误操作闸门是下面那条
    #: 确认阈值，这条只是防荒唐值的兜底。
    KB_IMPORT_MAX_CHUNKS: int = 12000
    #: 超过这个段数就先停下来等用户确认。**不静默截断**——
    #: 截掉一本书然后显示「成功」，比失败糟糕得多。
    KB_IMPORT_CONFIRM_CHUNKS: int = 800
    #: 打标批大小。改它会让 mock 的标签随之变化（频次表按批现算），
    #: 但对真实模型无影响。
    KB_TAG_BATCH_SIZE: int = 8
    #: 同时进行的导入作业数。和 RUN_MAX_CONCURRENCY 同量级——
    #: 导入本身会并发调打标 API，开太多会撞上 LLM 侧的限流，
    #: 表现是一批批的 429 而不是变快。
    KB_IMPORT_CONCURRENCY: int = 2

    # ------------------------------------------------------------------
    # 派生配置
    # ------------------------------------------------------------------

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def is_dev(self) -> bool:
        return self.APP_ENV in ("dev", "test")

    def _resolve(
        self,
        subsystem: str,
        explicit: str,
        *,
        configured: bool,
        real: Mode,
        required_env: str,
    ) -> Mode:
        """把 (显式选择, 全局 MOCK_MODE, 是否已配置) 解析成最终 provider 名。

        显式指定 mock 永远优先——用户在 .env 里写了 mock 就该得到 mock，
        哪怕密钥已经配好。
        """
        if explicit == "mock":
            return "mock"

        if self.MOCK_MODE == "always":
            return "mock"

        # MOCK_MODE=never，或用户点名要真实 provider：此时缺密钥必须硬失败
        if self.MOCK_MODE == "never" or explicit == real:
            if not configured:
                raise ConfigurationError(
                    f"{subsystem} 被要求使用真实 provider「{real}」，但 {required_env} 未配置。"
                    f" 要么补上该变量，要么设 MOCK_MODE=auto/always。"
                    f"（不会静默降级为 mock——真实配置下偷偷返回假数据是最危险的失败模式）"
                )
            return real

        # auto：配了就真的用，没配就 mock
        return real if configured else "mock"

    @property
    def llm_mode(self) -> Mode:
        return self._resolve(
            "LLM",
            self.LLM_PROVIDER,
            configured=bool(self.DEEPSEEK_API_KEY.strip()),
            real="deepseek",
            required_env="DEEPSEEK_API_KEY",
        )

    @property
    def embedding_mode(self) -> Mode:
        return self._resolve(
            "Embedding",
            self.EMBEDDING_PROVIDER,
            configured=bool(self.EMBEDDING_API_KEY.strip()),
            real="api",
            required_env="EMBEDDING_API_KEY",
        )

    @property
    def douyin_mode(self) -> Mode:
        configured = bool(self.DOUYIN_MCP_URL.strip() or self.DOUYIN_MCP_COMMAND.strip())
        return self._resolve(
            "抖音数据源",
            self.DOUYIN_PROVIDER,
            configured=configured,
            real="mcp",
            required_env="DOUYIN_MCP_URL 或 DOUYIN_MCP_COMMAND",
        )

    @property
    def is_demo(self) -> bool:
        """任一子系统处于 mock 即视为演示模式，前端据此显示「演示模式」徽章。"""
        return "mock" in (self.llm_mode, self.embedding_mode, self.douyin_mode)

    def provider_summary(self) -> dict[str, object]:
        return {
            "llm": self.llm_mode,
            "embedding": self.embedding_mode,
            "douyin": self.douyin_mode,
            "mock_mode": self.MOCK_MODE,
            "is_demo": self.is_demo,
        }

    def retrieval_quota(self, depth: str) -> tuple[int, int, int]:
        """按深度返回 (心理学, 文学, 唐诗) 的检索配额。"""
        raw = {
            "quick": self.RETRIEVAL_QUOTA_QUICK,
            "standard": self.RETRIEVAL_QUOTA_STANDARD,
            "deep": self.RETRIEVAL_QUOTA_DEEP,
        }.get(depth, self.RETRIEVAL_QUOTA_STANDARD)
        parts = [int(p) for p in raw.split(",")]
        return tuple(parts[:3])  # type: ignore[return-value]


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    # 触发一次模式解析，让配置错误在启动阶段就暴露，而不是等到第一次请求
    settings.provider_summary()
    return settings


def reset_settings_cache() -> None:
    """测试用：改完环境变量后清缓存。"""
    get_settings.cache_clear()
