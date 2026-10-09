"""确定性 mock 向量。

这不是一个「随便返回随机数」的桩。它的目标是：**离线时检索结果依然合理**。

做法是把中文字符 n-gram 哈希进固定维度并 L2 归一化。效果是没有语义理解，
但有**词法相似度**——查询「落花 哀伤」会真的命中带这些字样的语料，
而不是返回随机结果。知识库的 `text_for_embedding` 把 imagery/emotion 标签
烘进了文本，所以按情绪/意象检索在离线状态下仍然成立。

关键约束：必须**跨进程确定**。因此用 blake2b 而绝不能用 Python 内置的
hash()——后者对 str 默认加盐随机化，会导致摄取进程与查询进程算出不同向量，
表现为「明明入库了却检索不到」。
"""

from __future__ import annotations

import hashlib
import math
import unicodedata
from collections.abc import Sequence

# n-gram 权重：越长的 n-gram 越具体，区分度越高
_NGRAM_WEIGHTS: tuple[tuple[int, float], ...] = ((1, 0.5), (2, 1.0), (3, 0.8))


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in text if not ch.isspace())


def _grams(text: str) -> list[tuple[str, float]]:
    chars = _normalize(text)
    out: list[tuple[str, float]] = []
    for n, weight in _NGRAM_WEIGHTS:
        if len(chars) < n:
            continue
        for i in range(len(chars) - n + 1):
            out.append((chars[i : i + n], weight))
    return out


def _bucket(gram: str, dim: int) -> tuple[int, float]:
    """把 n-gram 映射到 (维度下标, 符号)。

    带符号累加是必要的：无符号时不同 n-gram 的碰撞会单向累加，
    让所有向量趋同；带符号后碰撞近似相消，向量间才有区分度。
    """
    digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    sign = 1.0 if (value >> 63) & 1 else -1.0
    return value % dim, sign


def deterministic_embed(text: str, dim: int) -> list[float]:
    vec = [0.0] * dim
    for gram, weight in _grams(text):
        idx, sign = _bucket(gram, dim)
        vec[idx] += sign * weight

    norm = math.sqrt(sum(v * v for v in vec))
    if norm < 1e-12:
        # 空文本或全是标点：返回一个固定的单位向量，保证下游不用处理零向量
        vec = [0.0] * dim
        vec[0] = 1.0
        return vec
    return [v / norm for v in vec]


class MockEmbeddingModel:
    """实现 EmbeddingModel 协议。"""

    is_mock = True

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.name = f"mock-hash-ngram-{dim}"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [deterministic_embed(t, self.dim) for t in texts]
