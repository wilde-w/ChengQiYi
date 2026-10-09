"""n3 的第三段：给簇起名，以及起不出来的时候怎么办。

聚类本身（`cluster_only`）在 `test_n3_cluster.py` 里按分支测过了。这里守的是
**模型那一侧**的三件事，它们都不报错、只在屏幕上表现为「卡片怪怪的」：

1. 模型挂了/答非所问时，簇**必须有名字**。为了一个名字让整条流水线在 n3
   断掉，代价与收益完全不成比例——所以 `label_clusters` 不抛，退回关键词。
2. 重试必须真的换一把缓存钥匙。`CachedChatModel` 按 `(task, messages, context)`
   哈希，`_llm.call_json` 往 context 里塞 `attempt` 就是为了这个。塞漏了的话
   重试拿回的是同一份坏输出，日志里两次调用参数一模一样——最容易归因成
   「模型不稳定」，而其实根本没重试。
3. 每次 partial 发的是**完整列表**。前端 `clusters` 分支是整体替换，发单条
   会把已经渲染出来的卡片抹掉。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from app.constants import ClusterMethod
from app.graph.emitting import NodeEmitter
from app.graph.nodes import n3_cluster
from app.graph.nodes.n3_cluster import cluster_only, emit_clusters, label_clusters, n3_cluster as run_n3
from app.providers.base import ChatMessage, ChatResult, ProviderError

DIM = 8
JITTER = 0.22


# ----------------------------------------------------------------------
# 假模型
# ----------------------------------------------------------------------


@dataclass
class ScriptedChat:
    """按脚本回话。默认回一段**不是 JSON 的废话**——那正是要测的输入。"""

    text: str = "这一簇大概是关于想念的吧，我也说不太清楚。"
    name: str = "scripted"
    is_mock: bool = True
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        task: str | None = None,
        context: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        self.calls.append(dict(context or {}))
        return ChatResult(text=self.text, model=self.name, finish_reason="stop")


@dataclass
class EchoChat:
    """回一份合法 JSON。`calls` 记下每次的 context，用来看重试换了什么。"""

    name: str = "echo"
    is_mock: bool = True
    calls: list[dict[str, Any]] = field(default_factory=list)
    fail_times: int = 0

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        task: str | None = None,
        context: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        ctx = dict(context or {})
        self.calls.append(ctx)
        if len(self.calls) <= self.fail_times:
            raise ProviderError("第一次不行", retryable=True)
        payload = {
            "label": "说不出口的想念",
            "summary": "这一簇反复提到想联系却不敢开口。",
            "emotion": ["哀伤"],
            "topic": ["亲子关系"],
            "need": ["被看见的丧失"],
        }
        return ChatResult(text=json.dumps(payload, ensure_ascii=False), model=self.name)


# ----------------------------------------------------------------------
# 夹具
# ----------------------------------------------------------------------


def blob(axis: int, count: int, *, seed: int = 0) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    center = np.zeros(DIM)
    center[axis % DIM] = 1.0
    return [list(center + rng.normal(0, JITTER, DIM)) for _ in range(count)]


def comments_for(count: int, *, prefix: str = "c") -> list[dict]:
    return [
        {"comment_id": f"{prefix}-{i}", "text": f"{prefix} 第 {i} 条，说了一件具体的事", "like_count": count - i}
        for i in range(count)
    ]


def three_clusters() -> tuple[list[dict], dict]:
    vectors: list[list[float]] = []
    comments: list[dict] = []
    for axis in range(3):
        vectors.extend(blob(axis, 10, seed=axis))
        comments.extend(comments_for(10, prefix=f"b{axis}"))
    return cluster_only(vectors, comments, params={})


class Recorder:
    """把节点发出去的事件收下来。`NodeEmitter` 允许注入 write，用它。"""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emitter(self) -> NodeEmitter:
        return NodeEmitter("n3_cluster", write=self.events.append)

    def partials(self, data_kind: str) -> list[dict[str, Any]]:
        return [
            e["data"]
            for e in self.events
            if e["kind"] == "partial" and e.get("data_kind") == data_kind
        ]


# ----------------------------------------------------------------------
# 起不出名字时
# ----------------------------------------------------------------------


class TestFallback:
    async def test_garbage_reply_does_not_raise(self) -> None:
        """模型回一段散文 → 解析失败 → 重试用尽 → 兜底名字，不抛。"""
        clusters, meta = three_clusters()
        for row in clusters:
            row["_representatives"] = ["听到这个消息我哭了"]

        out = await label_clusters(clusters, model=ScriptedChat())

        assert len(out) == len(clusters)
        for row in out:
            assert row["label"], "空 label 在卡片上是一片空白，看起来像渲染坏了"
            assert row["label_source"] == "fallback"
            assert row["emotion_tags"] == []

    async def test_provider_error_does_not_raise(self) -> None:
        """provider 直接抛错（没网、超时）也走同一条兜底路。"""
        clusters, _meta = three_clusters()
        out = await label_clusters(clusters, model=BoomChat())

        assert all(row["label_source"] == "fallback" for row in out)

    def test_fallback_label_uses_local_keywords(self) -> None:
        """兜底名字用**这一簇真实的字**，不是编一个「情绪·主题」。

        编出来的名字看起来和模型给的一模一样，用户没有任何办法分辨
        卡片上哪几个名字是兜底的。
        """
        clusters, _meta = three_clusters()
        clusters[0]["keywords"] = ["我妈的微信", "语音", "她自己"]
        n3_cluster._apply_label(clusters[0], None)
        assert clusters[0]["label"] == "我妈的微信·语音"

    def test_noise_and_tail_get_their_own_names(self) -> None:
        clusters, _meta = three_clusters()
        clusters[0].update({"is_noise": True, "keywords": ["随便", "说说"]})
        clusters[1].update({"cluster_key": "tail", "keywords": []})
        n3_cluster._apply_label(clusters[0], None)
        n3_cluster._apply_label(clusters[1], None)
        assert clusters[0]["label"] == "边缘声音"
        assert clusters[1]["label"] == "其他·长尾"


class BoomChat:
    """一律抛错，连一次都不成功。"""

    name = "boom"
    is_mock = True

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, *args: Any, **kwargs: Any) -> ChatResult:
        self.calls += 1
        raise ProviderError("网络不通", retryable=True)


# ----------------------------------------------------------------------
# 模型正常时
# ----------------------------------------------------------------------


class TestHappyPath:
    async def test_tags_are_written_into_the_row(self) -> None:
        clusters, _meta = three_clusters()
        out = await label_clusters(clusters, model=EchoChat())

        for row in out:
            assert row["label"] == "说不出口的想念"
            assert row["emotion_tags"] == ["哀伤"]
            assert row["topic_tags"] == ["亲子关系"]
            assert row["need_tags"] == ["被看见的丧失"]
            assert row["label_source"] == "model"

    async def test_off_vocabulary_emotion_is_dropped(self) -> None:
        """情绪走闭集。自创词在图谱里长成孤立节点，且没人会想到去查这里。"""
        clusters, _meta = three_clusters()
        model = ScriptedChat(
            text=json.dumps({"label": "x", "emotion": ["悲怆", "哀伤"], "topic": ["a"], "need": []})
        )
        await label_clusters(clusters, model=model)
        assert clusters[0]["emotion_tags"] == ["哀伤"]


class TestRetryKey:
    async def test_two_attempts_carry_different_context(self) -> None:
        """**这是 `_llm.py` 存在的唯一理由。**

        `CachedChatModel` 的键是 `(task, messages, context)` 的哈希。两次
        context 一样 = 第二次拿回第一次那份坏输出，重试等于没重试。
        """
        model = EchoChat(fail_times=1)
        clusters, _meta = three_clusters()
        out = await label_clusters(clusters[:1], model=model)

        assert len(model.calls) == 2, "第一次失败后应当再试一次"
        assert model.calls[0] != model.calls[1], "重试的 context 没变，缓存会命中上一次的坏结果"
        assert {model.calls[0]["attempt"], model.calls[1]["attempt"]} == {0, 1}
        assert out[0]["label_source"] == "model", "第二次成功就该用模型的名字"

    async def test_failure_after_both_attempts_falls_back(self) -> None:
        model = EchoChat(fail_times=99)
        clusters, _meta = three_clusters()
        out = await label_clusters(clusters[:1], model=model)

        assert len(model.calls) == 2, "不能无限重试"
        assert out[0]["label_source"] == "fallback"


# ----------------------------------------------------------------------
# 发出去的东西
# ----------------------------------------------------------------------


class TestEmitting:
    def test_each_partial_carries_the_whole_list(self) -> None:
        """前端 `clusters` 分支是整体替换——发单条会把别的卡片抹掉。"""
        clusters, meta = three_clusters()
        rec = Recorder()
        emit_clusters(rec.emitter(), clusters, meta)
        emit_clusters(rec.emitter(), clusters, meta)

        for payload in rec.partials("clusters"):
            assert len(payload["clusters"]) == len(clusters)
            assert payload["cluster_meta"] == meta

    async def test_label_clusters_emits_after_every_cluster(self) -> None:
        clusters, meta = three_clusters()
        rec = Recorder()
        # label_clusters 收的是 emitter 对象；这里用真的 Recorder 包一层。
        await label_clusters(clusters, emitter=rec.emitter(), model=EchoChat())

        assert len(rec.partials("clusters")) == len(clusters)


# ----------------------------------------------------------------------
# 整节点
# ----------------------------------------------------------------------


class TestNode:
    async def test_end_to_end_with_mock_model_cleans_internal_fields(self, monkeypatch) -> None:
        """走真工厂（conftest 把 MOCK_MODE 钉成 always）跑一遍 n3。

        顺带守住一处**季节性的错位**：`cluster_label` 这个 task 名必须与
        `mock_llm._handlers` 的键一致。对不上时 mock 返回一段占位 JSON，
        节点照样跑完、照样落库——只是所有簇的名字都变成同一个词。
        """
        vectors = blob(0, 10, seed=0) + blob(1, 10, seed=1) + blob(2, 10, seed=2)
        comments = comments_for(10, prefix="b0") + comments_for(10, prefix="b1") + comments_for(10, prefix="b2")

        async def fake_embed(texts, *, emitter=None, model=None):
            return vectors

        monkeypatch.setattr(n3_cluster, "embed_comments", fake_embed)
        rec = Recorder()
        monkeypatch.setattr(n3_cluster, "NodeEmitter", lambda node: rec.emitter())

        patch = await run_n3({"comments": comments, "cluster_params": {}})

        assert patch["cluster_meta"]["method"] in {ClusterMethod.HDBSCAN, ClusterMethod.KMEANS}
        assert patch["clusters"]
        for row in patch["clusters"]:
            # 内部字段不能跟着 checkpoint 写进 PG
            assert "_representatives" not in row
            assert "label_source" not in row
            assert row["label"]
            assert not row["label"].startswith("mock 未实现")

        # 流的末态与落库的末态必须是同一份数据
        last = rec.partials("clusters")[-1]["clusters"]
        assert [r["comment_ids"] for r in last] == [r["comment_ids"] for r in patch["clusters"]]
        assert all("_representatives" not in r for r in last)

    async def test_embed_failure_is_not_a_silent_empty(self, monkeypatch) -> None:
        """嵌入挂了要发 error，**不能**返回空簇。

        空簇在前端渲染成「分析完成，0 个主题」——用户会以为这个视频真的
        没人讨论，而实际是这一步根本没跑成。
        """

        async def boom(texts, *, emitter=None, model=None):
            raise ProviderError("额度用尽", retryable=False)

        monkeypatch.setattr(n3_cluster, "embed_comments", boom)
        monkeypatch.setattr(n3_cluster, "NodeEmitter", lambda node: Recorder().emitter())

        patch = await run_n3({"comments": comments_for(30), "cluster_params": {}})

        assert patch["clusters"] == []
        assert patch["cluster_meta"]["reason"] == "embed_failed"
        assert patch["errors"][0]["code"] == "embed_failed"

    async def test_no_comments_returns_warning_not_error(self, monkeypatch) -> None:
        monkeypatch.setattr(n3_cluster, "NodeEmitter", lambda node: Recorder().emitter())
        patch = await run_n3({"comments": [], "cluster_params": {}})

        assert patch["clusters"] == []
        assert patch["cluster_meta"]["reason"] == "no_comments"
        assert patch["warnings"][0]["code"] == "no_comments_to_cluster"
        assert "errors" not in patch
