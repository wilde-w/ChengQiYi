"""心理侧写：加权、降级、以及字段形状。

这个文件的重点是**分布**。`emotions` / `topics` / `needs` 三个列表是 Python
按簇大小算出来的，不是模型给的——所以它必须能由落库字段复算，也必须是
按**人数**加权而不是按出现次数。一个 22 条的簇和一个 3 条的簇各说一次
「倦怠」，在评论区里的分量当然不同。

另一件事是**代表评论从哪来**。n4 拿到的 `clusters` 是落库的那一份，
里面只有 `representative_comment_ids`（抖音 id），原文要靠 state 里的评论表
去取。曾经直接读 n3 留下的 `_representatives`，而那个字段在 n3 返回前就被
剥掉了——于是 n4 永远跑在空的 comments 数组上，模型只能凭关键词联想，
**照样返回一份看起来完整的 profile，没有任何报错**。第一类测试守的就是它。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from app.graph.emitting import NodeEmitter
from app.graph.nodes import n4_psych as mod
from app.graph.nodes.n4_psych import DISTRIBUTION_TOP_N, _distributions, _representative_texts, n4_psych
from app.providers.base import ChatMessage, ChatResult, ProviderError


# ----------------------------------------------------------------------
# 假模型
# ----------------------------------------------------------------------


@dataclass
class ScriptedChat:
    """回一份固定载荷。`payload` 是 dict 就原样 JSON 出去，是 str 就原样吐。"""

    payload: Any = None
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
        body = self.payload if isinstance(self.payload, str) else json.dumps(self.payload, ensure_ascii=False)
        return ChatResult(text=body, model=self.name)


@dataclass
class BoomChat:
    name: str = "boom"
    is_mock: bool = True
    calls: int = 0

    async def complete(self, *args: Any, **kwargs: Any) -> ChatResult:
        self.calls += 1
        raise ProviderError("模型挂了", retryable=True)


def use_model(monkeypatch: pytest.MonkeyPatch, model: Any) -> Any:
    """把 `_extract` 里那句 `get_chat_model()` 换掉。

    节点是在函数体内 import 工厂的，所以补丁必须打在**工厂模块**的属性上。
    """
    monkeypatch.setattr("app.providers.factory.get_chat_model", lambda: model)
    return model


# ----------------------------------------------------------------------
# 夹具
# ----------------------------------------------------------------------


def cluster(key: str, size: int, **overrides: Any) -> dict[str, Any]:
    row = {
        "cluster_key": key,
        "size": size,
        "raw_size": size,
        "keywords": [f"{key}-关键词"],
        "emotion_tags": [],
        "topic_tags": [],
        "need_tags": [],
        "representative_comment_ids": [],
        "is_noise": False,
    }
    row.update(overrides)
    return row


def envelope(channel: str) -> dict[str, Any]:
    """一个捕获 emitter 事件的假节点外壳。`NodeEmitter` 允许注入 write。"""

    class Env:
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        def emitter(self) -> NodeEmitter:
            return NodeEmitter(channel, write=self.events.append)

        def partials(self, kind: str) -> list[dict[str, Any]]:
            return [e["data"] for e in self.events if e.get("data_kind") == kind]

    return Env()  # type: ignore[return-value]


def patch_payload(cluster_tags: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "cluster_tags": cluster_tags,
        "global_tags": {"emotion": ["哀伤"], "topic": ["亲子关系"], "need": ["被看见的丧失"]},
        "core_tension": "想念却再也无法说出口",
        "summary": "这一条视频底下的人都在说同一件事。",
        **extra,
    }


# ----------------------------------------------------------------------
# 加权
# ----------------------------------------------------------------------


class TestDistributions:
    def test_weight_is_cluster_size_not_mention_count(self) -> None:
        """22 条的簇说一次「倦怠」，胜过 3 条的簇说三次同一件事。

        按出现次数算的话，那个 3 条的簇会被放大到和 22 条的一样重——
        而它只是三个人的说法。手工算一遍：倦怠 = 22 + 4 = 26，哀伤 = 3*3 = 9，
        所以 26 必须排在 9 前面。
        """
        clusters = [cluster("c0", 22), cluster("c1", 4), cluster("c2", 3)]
        tags = {
            "c0": {"emotion": ["倦怠"], "topic": [], "need": []},
            "c1": {"emotion": ["倦怠"], "topic": [], "need": []},
            "c2": {"emotion": ["哀伤", "哀伤", "哀伤"], "topic": [], "need": []},
        }

        out = _distributions(tags, clusters)

        assert out["emotions"] == [
            {"label": "倦怠", "value": 26, "cluster_key": "c0"},
            {"label": "哀伤", "value": 3, "cluster_key": "c2"},
        ]

    def test_duplicate_labels_within_one_cluster_count_once(self) -> None:
        """同一个簇里重复的标签只算一次——模型重复三次不代表三倍的人。"""
        clusters = [cluster("c0", 10)]
        tags = {"c0": {"emotion": ["哀伤", "哀伤", "哀伤"], "topic": [], "need": []}}
        assert _distributions(tags, clusters)["emotions"] == [
            {"label": "哀伤", "value": 10, "cluster_key": "c0"}
        ]

    def test_owner_is_deterministic_on_a_tie(self) -> None:
        """并列时归属固定给先出现的簇。

        不固定的话同一份输入两次跑出的分布归属不一样，前端点柱子跳到的
        卡片每次都在换。
        """
        clusters = [cluster("c0", 5), cluster("c1", 5)]
        tags = {
            "c0": {"emotion": ["哀伤"], "topic": [], "need": []},
            "c1": {"emotion": ["哀伤"], "topic": [], "need": []},
        }
        first = _distributions(tags, clusters)
        second = _distributions(tags, clusters)
        assert first == second
        assert first["emotions"][0]["cluster_key"] == "c0"

    def test_top_n_is_capped(self) -> None:
        clusters = [cluster(f"c{i}", 5) for i in range(DISTRIBUTION_TOP_N + 3)]
        tags = {
            f"c{i}": {"emotion": [f"词{i}"], "topic": [], "need": []}
            for i in range(DISTRIBUTION_TOP_N + 3)
        }
        assert len(_distributions(tags, clusters)["emotions"]) == DISTRIBUTION_TOP_N

    def test_three_dimensions_are_always_present(self) -> None:
        """三个键一个都不能少，且键名与 ORM 的三列逐一对齐。"""
        out = _distributions({}, [])
        assert set(out) == {"emotions", "topics", "needs"}
        assert all(v == [] for v in out.values())


# ----------------------------------------------------------------------
# 代表评论
# ----------------------------------------------------------------------


class TestRepresentatives:
    def test_texts_are_recovered_from_the_comment_table(self) -> None:
        """**这是那个静默 bug 的回归闸门。**

        n4 拿到的簇里只有抖音 id，原文要靠 state 的评论表取。取不到就退回
        关键词——绝不能是一个空数组然后照常发给模型。
        """
        row = cluster("c0", 3, representative_comment_ids=["a", "b", "c"], keywords=["兜底词"])
        by_id = {"a": "听到这个消息我哭了", "b": "我也经历过", "c": "谢谢你说出来"}

        assert _representative_texts(row, by_id) == ["听到这个消息我哭了", "我也经历过", "谢谢你说出来"]

    def test_missing_ids_fall_back_to_keywords(self) -> None:
        """id 对不上（老数据、手工重建的簇）时给几个关键词。

        给空数组的话模型按「没有评论」处理，产出的侧写看不出任何异常。
        """
        row = cluster("c0", 3, representative_comment_ids=["x", "y"], keywords=["想念", "语音", "微信"])
        assert _representative_texts(row, {}) == ["想念", "语音", "微信"]

    def test_long_comments_are_truncated(self) -> None:
        """一条 500 字的评论会挤掉另外两条的信息量。"""
        row = cluster("c0", 1, representative_comment_ids=["a"])
        out = _representative_texts(row, {"a": "字" * 500})
        assert len(out[0]) == 120

    async def test_model_receives_the_actual_comment_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """整条链：评论表 → 提示词 context。"""
        model = use_model(monkeypatch, ScriptedChat(payload=patch_payload({})))
        clusters = [cluster("c0", 2, representative_comment_ids=["a", "b"])]
        comments = [
            {"comment_id": "a", "text": "听到这个消息我哭了"},
            {"comment_id": "b", "text": "我也经历过"},
        ]

        await n4_psych({"clusters": clusters, "comments": comments})

        sent = model.calls[0]["clusters"]
        assert sent[0]["comments"] == ["听到这个消息我哭了", "我也经历过"]


# ----------------------------------------------------------------------
# 主路径
# ----------------------------------------------------------------------


class TestProfileShape:
    async def test_global_tags_is_a_dict_with_three_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """**缺陷 3 的回归闸门。**

        `global_tags` 曾经声明成 `list[str]`，而 persist 存的是 dict、
        mock 读的是 `global_tags.get("emotion")`。三者对不上时右栏的情绪列表
        恒为空——接口不报错，只是永远没内容。
        """
        use_model(monkeypatch, ScriptedChat(payload=patch_payload({"c0": {"emotion": ["哀伤"]}})))
        patch = await n4_psych({"clusters": [cluster("c0", 5)], "comments": []})

        tags = patch["profile"]["global_tags"]
        assert isinstance(tags, dict)
        assert set(tags) == {"emotion", "topic", "need"}
        assert all(isinstance(v, list) for v in tags.values())

    async def test_shape_matches_with_or_without_clusters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """空侧写与正常侧写**键完全相同**，前端因此不需要第二条渲染分支。"""
        normal = (await n4_psych({"clusters": [cluster("c0", 5)], "comments": []}))["profile"]
        empty_patch = await n4_psych({"clusters": [], "comments": []})

        use_model(monkeypatch, ScriptedChat(payload=patch_payload({"c0": {"emotion": ["哀伤"]}})))
        assert set(empty_patch["profile"]) == set(normal)

    async def test_missing_cluster_is_filled_from_local_tags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """模型漏掉的簇用 n3 的本地标签补上，**不能留空**。

        留空会让那个簇在分布里贡献 0，主导情绪凭空偏向别的簇——用户看到
        的是一个被扭曲的分布，且没有任何迹象表明它被扭曲过。
        """
        payload = patch_payload({"c0": {"emotion": ["哀伤"]}})  # 没有 c1
        use_model(monkeypatch, ScriptedChat(payload=payload))
        clusters = [
            cluster("c0", 5),
            cluster("c1", 5, emotion_tags=["倦怠"], topic_tags=["职场压力"], need_tags=["停下来休息的权利"]),
        ]

        profile = (await n4_psych({"clusters": clusters, "comments": []}))["profile"]

        assert profile["cluster_tags"]["c1"]["emotion"] == ["倦怠"]
        weights = {e["label"]: e["value"] for e in profile["emotions"]}
        assert weights == {"哀伤": 5, "倦怠": 5}, "被漏掉的簇在分布里必须仍然有分量"

    async def test_unknown_cluster_key_is_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """模型自造键（把 c0 写成 cluster_0）要丢掉：它的权重无从计算。"""
        payload = patch_payload(
            {"c0": {"emotion": ["哀伤"]}, "cluster_0": {"emotion": ["倦怠"], "topic": [], "need": []}}
        )
        use_model(monkeypatch, ScriptedChat(payload=payload))

        profile = (await n4_psych({"clusters": [cluster("c0", 5)], "comments": []}))["profile"]

        assert set(profile["cluster_tags"]) == {"c0"}
        assert [e["label"] for e in profile["emotions"]] == ["哀伤"]

    async def test_summary_and_tension_are_cleaned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        use_model(monkeypatch, ScriptedChat(payload=patch_payload({}, core_tension="  想念却再也无法说出口  ")))
        profile = (await n4_psych({"clusters": [cluster("c0", 5)], "comments": []}))["profile"]
        assert profile["core_tension"] == "想念却再也无法说出口"
        assert profile["model"] == "scripted"


# ----------------------------------------------------------------------
# 降级
# ----------------------------------------------------------------------


class TestDegradation:
    async def test_model_failure_degrades_without_raising(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = use_model(monkeypatch, BoomChat())
        clusters = [cluster("c0", 5, emotion_tags=["哀伤"], need_tags=["被看见的丧失"])]

        patch = await n4_psych({"clusters": clusters, "comments": []})

        assert model.calls == 2, "`call_json` 应当试满两次再降级"
        profile = patch["profile"]
        assert profile["model"] == "fallback"
        assert profile["core_tension"] is None, "张力是报告里最需要判断力的一句，不许用模板拼"
        assert [e["label"] for e in profile["emotions"]] == ["哀伤"]
        assert patch["warnings"][0]["code"] == "profile_degraded"

    async def test_unparseable_reply_degrades(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """模型回一段散文也是降级，不是崩溃。"""
        use_model(monkeypatch, ScriptedChat(payload="这一簇大概是关于想念的吧。"))
        patch = await n4_psych({"clusters": [cluster("c0", 5, emotion_tags=["哀伤"])], "comments": []})
        assert patch["profile"]["model"] == "fallback"

    async def test_degraded_global_tags_still_pass_the_closed_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """降级路径的标签与主路径走**同一个**闭集收口。

        曾经只有逐簇那边过闭集，全局这边放行，于是同一个越界词在两张卡片上
        一张有一张没有——用户没有任何办法判断哪边是对的。
        """
        use_model(monkeypatch, BoomChat())
        clusters = [cluster("c0", 5, emotion_tags=["悲怆", "哀伤"], topic_tags=["亲子关系"])]

        profile = (await n4_psych({"clusters": clusters, "comments": []}))["profile"]

        assert profile["global_tags"]["emotion"] == ["哀伤"], "越界词漏进了全局标签"


# ----------------------------------------------------------------------
# 发出的事件
# ----------------------------------------------------------------------


class TestEmitting:
    async def test_profile_partial_replaces_the_whole_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`profile` 分支在前端是整包替换，所以发出去的就该是完整的 profile。"""
        use_model(monkeypatch, ScriptedChat(payload=patch_payload({"c0": {"emotion": ["哀伤"]}})))
        env = envelope("n4_psych")
        monkeypatch.setattr(mod, "NodeEmitter", lambda node: env.emitter())

        patch = await n4_psych({"clusters": [cluster("c0", 5)], "comments": []})

        payloads = env.partials("profile")
        assert len(payloads) == 1
        assert payloads[0] == patch["profile"]
        assert set(payloads[0]) == {"emotions", "topics", "needs", "global_tags", "cluster_tags", "core_tension", "summary", "model"}
