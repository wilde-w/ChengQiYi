"""provider 工厂。

配置 → 实现 的唯一映射点。整个项目「空 .env 即可运行」的能力由这个文件兑现：
mode 解析在 config.py，实例构造在这里，其余代码只依赖协议。
"""

from __future__ import annotations

from app.config import get_settings
from app.logging_conf import get_logger
from app.providers.base import ChatModel, EmbeddingModel
from app.providers.cache import CachedChatModel, CachedEmbeddingModel

log = get_logger(__name__)

_chat: ChatModel | None = None
_embedding: EmbeddingModel | None = None


def get_chat_model(*, fresh: bool = False) -> ChatModel:
    global _chat
    if _chat is not None and not fresh:
        return _chat

    settings = get_settings()
    mode = settings.llm_mode

    if mode == "mock":
        from app.providers.mock_llm import MockChatModel

        inner: ChatModel = MockChatModel(latency_ms=settings.MOCK_LLM_LATENCY_MS)
        log.info("chat_model_ready", mode="mock", model=inner.name)
        model: ChatModel = inner
    else:
        from app.providers.openai_compat import OpenAICompatChat

        inner = OpenAICompatChat(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
            model=settings.DEEPSEEK_MODEL,
            timeout=float(settings.LLM_TIMEOUT_SECONDS),
            max_retries=settings.LLM_MAX_RETRIES,
            default_temperature=settings.LLM_TEMPERATURE,
            provider_label="deepseek",
        )
        log.info("chat_model_ready", mode="deepseek", model=inner.name)
        model = CachedChatModel(inner, namespace=settings.DEEPSEEK_MODEL)

    if not fresh:
        _chat = model
    return model


def get_embedding_model(*, fresh: bool = False) -> EmbeddingModel:
    global _embedding
    if _embedding is not None and not fresh:
        return _embedding

    settings = get_settings()
    mode = settings.embedding_mode

    if mode == "mock":
        from app.providers.mock_embedding import MockEmbeddingModel

        inner: EmbeddingModel = MockEmbeddingModel(dim=settings.EMBEDDING_DIM)
        log.info("embedding_model_ready", mode="mock", model=inner.name, dim=inner.dim)
        model: EmbeddingModel = inner
    else:
        from app.providers.openai_compat import OpenAICompatEmbedding

        inner = OpenAICompatEmbedding(
            api_key=settings.EMBEDDING_API_KEY,
            base_url=settings.EMBEDDING_BASE_URL,
            model=settings.EMBEDDING_MODEL,
            dim=settings.EMBEDDING_DIM,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            provider_label="embedding-api",
        )
        log.info(
            "embedding_model_ready",
            mode="api",
            model=inner.name,
            dim=inner.dim,
        )
        # namespace 带模型名与维度：换模型后不会错误命中旧向量
        model = CachedEmbeddingModel(inner, namespace=f"{settings.EMBEDDING_MODEL}:{settings.EMBEDDING_DIM}")

    if not fresh:
        _embedding = model
    return model


def reset_provider_cache() -> None:
    """测试用：改完配置后重建 provider。"""
    global _chat, _embedding
    _chat = None
    _embedding = None
