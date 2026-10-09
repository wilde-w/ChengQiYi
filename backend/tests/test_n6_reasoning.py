"""n6：把模型的输出收进证据池，以及 confidence 重算。

三件事都**不能靠提示词**：

1. `evidence_ids` / `allusion_ids` 必须落在检索集里。模型抄错一个字符，
   那条引用就该被丢掉——留着它，用户点开 chip 什么也看不到，
   而卡片上只是一个数字小了一点。
2. `allusion_ids` 只能取文学与诗词。把心理学语料当典故是模型常见的串味。
3. `confidence` 由落库字段重算。它是给用户判断可信度用的，不能是一个
   谁也没法验证的模型自评。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from app.constants import MAX_REASONING_STEPS
from app.graph.emitting import NodeEmitter
from app.graph.nodes import n6_reasoning as n6
from app.providers.base import ChatMessage, ChatResult


@dataclass
class JsonChat:
    """回一份可控的 JSON。`fail_times` 用来演练模型挂掉的那条路。"""

    payload: Any = None
    name: str = "json-chat"
    is_mock: bool = True
    fail_times: int = 0
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
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("provider 挂了")
        text = json.dumps(self.payload, ensure_ascii=False)
        return ChatResult(text=text, model=self.name, finish_reason="stop")


def use_model(monkeypatch: pytest.MonkeyPatch, model: Any) -> None:
    monkeypatch.setattr("app.providers.factory.get_chat_model", lambda: model)


def evidence(chunk_id: str, kind: str, **extra: Any) -> dict:
    return {
        "chunk_id": chunk_id,
        "kind": kind,
        "library": kind,
        "title": f"《{chunk_id}》",
        "source": f"{chunk_id} 的出处",
        "author": "某人",
        "text": "正文",
        **extra,
    }


POOL = [
    evidence("psy:1", "psychology"),
    evidence("psy:2", "psychology"),
    evidence("lit:1", "literature"),
    evidence("po:1", "poetry"),
]


def step(**over: Any) -> dict:
    base = {
        "phenomenon": "反复点开对话框又关掉",
        "mechanism": "持续性联结（Klass）。哀伤不是切断关系，而是把它换个形式留住。",
        "insight": "真正难的不是告别，是不再需要告别。",
        "evidence_ids": ["psy:1"],
        "allusion_ids": ["lit:1"],
        "confidence": 0.95,
    }
    base.update(over)
    return base


def cluster(**over: Any) -> dict:
    base = {
        "cluster_key": "c0",
        "label": "想念",
        "summary": "明明知道说不了了，还是把话打出来又删掉。",
        "size": 12,
        "is_noise": False,
    }
    base.update(over)
    return base


async def run(state: dict[str, Any], monkeypatch: pytest.MonkeyPatch, model: Any) -> tuple[dict, list]:
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(n6, "NodeEmitter", lambda node: NodeEmitter(node, write=events.append))
    use_model(monkeypatch, model)
    patch = await n6.n6_reasoning(state)
    return patch, events


def base_state(**over: Any) -> dict[str, Any]:
    state = {
        "evidence": list(POOL),
        "profile": {
            "core_tension": "想念却再也无法说出口",
            "global_tags": {"emotion": ["哀伤"], "topic": [], "need": []},
        },
        "clusters": [cluster()],
    }
    state.update(over)
    return state


# ----------------------------------------------------------------------


class TestCitations:
    async def test_引用了池外的_id_被丢掉(self, monkeypatch: pytest.MonkeyPatch):
        model = JsonChat(payload={"steps": [step(evidence_ids=["psy:1", "编的", "psy:9"])]})
        patch, _ = await run(base_state(), monkeypatch, model)

        assert patch["reasoning"][0]["evidence_ids"] == ["psy:1"]

    async def test_心理学语料不算典故(self, monkeypatch: pytest.MonkeyPatch):
        # 模型常把「一切引用的东西」都当典故。丢弃比纠正便宜，也不会错。
        model = JsonChat(payload={"steps": [step(allusion_ids=["psy:2", "po:1"])]})
        patch, _ = await run(base_state(), monkeypatch, model)

        assert patch["reasoning"][0]["allusion_ids"] == ["po:1"]

    async def test_同一份材料不同时充当机制与典故(self, monkeypatch: pytest.MonkeyPatch):
        model = JsonChat(payload={"steps": [step(evidence_ids=["psy:1"], allusion_ids=["psy:1"])]})
        patch, _ = await run(base_state(), monkeypatch, model)

        assert patch["reasoning"][0]["evidence_ids"] == ["psy:1"]
        assert patch["reasoning"][0]["allusion_ids"] == []

    async def test_重复引用去重(self, monkeypatch: pytest.MonkeyPatch):
        model = JsonChat(payload={"steps": [step(evidence_ids=["psy:1", "psy:1", "psy:2"])]})
        patch, _ = await run(base_state(), monkeypatch, model)

        assert patch["reasoning"][0]["evidence_ids"] == ["psy:1", "psy:2"]

    async def test_没有现象的那一步不产出(self, monkeypatch: pytest.MonkeyPatch):
        # 没有现象就是在描述一个不存在的东西，留着它等于凭空多一张卡。
        model = JsonChat(payload={"steps": [step(phenomenon="  "), step(evidence_ids=["psy:2"])]})
        patch, _ = await run(base_state(), monkeypatch, model)

        assert len(patch["reasoning"]) == 1
        assert patch["reasoning"][0]["evidence_ids"] == ["psy:2"]

    async def test_步数截到上限(self, monkeypatch: pytest.MonkeyPatch):
        model = JsonChat(payload={"steps": [step() for _ in range(9)]})
        patch, _ = await run(base_state(), monkeypatch, model)

        assert len(patch["reasoning"]) == MAX_REASONING_STEPS
        assert [s["step_index"] for s in patch["reasoning"]] == list(range(1, MAX_REASONING_STEPS + 1))

    async def test_模型给的怪形状不炸(self, monkeypatch: pytest.MonkeyPatch):
        for payload in ({"steps": "不是列表"}, {"steps": [1, 2]}, {}, {"steps": None}):
            patch, _ = await run(base_state(), monkeypatch, JsonChat(payload=payload))
            # 解析不出步骤时退回「只列现象」，而不是抛出去让整条流水线断掉。
            assert patch["reasoning"][0]["mechanism"] == ""
            assert patch["reasoning"][0]["phenomenon"] == cluster()["summary"]


class TestConfidence:
    async def test_黄金路径手算(self, monkeypatch: pytest.MonkeyPatch):
        # support = min(1, 1/2) = 0.5   → 0.55 * 0.5 = 0.275
        # mech    = 1.0（引了心理学证据且机制非空） → 0.30
        # allu    = 1.0（引了典故）                → 0.15
        #                                   合计 = 0.725
        patch, _ = await run(base_state(), monkeypatch, JsonChat(payload={"steps": [step()]}))
        assert patch["reasoning"][0]["confidence"] == 0.725

    async def test_没有引用就没有支撑(self, monkeypatch: pytest.MonkeyPatch):
        # 机制、洞察都写着，但一条证据都没引 → support 0，mech 0（池子里
        # 明明有心理学证据却没引），allu 0（明明有文学证据却没引）。
        patch, _ = await run(
            base_state(),
            monkeypatch,
            JsonChat(payload={"steps": [step(evidence_ids=[], allusion_ids=[])]}),
        )
        assert patch["reasoning"][0]["confidence"] == 0.0

    async def test_池子里根本没有这类证据时给半分(self, monkeypatch: pytest.MonkeyPatch):
        # 没有文学证据不是这一步的错——库本来就没有，不该按「没引用」扣分。
        pool = [evidence("psy:1", "psychology"), evidence("psy:2", "psychology")]
        patch, _ = await run(
            base_state(evidence=pool),
            monkeypatch,
            JsonChat(payload={"steps": [step(evidence_ids=["psy:1", "psy:2"], allusion_ids=[])]}),
        )
        # support 1.0 → 0.55；mech 1.0 → 0.30；allu 0.5 → 0.075
        assert patch["reasoning"][0]["confidence"] == 0.925

    async def test_分数只由落库字段决定(self, monkeypatch: pytest.MonkeyPatch):
        # 模型自评 0.95 与 0.1 给出同一个展示值——它不参与计算。
        a, _ = await run(base_state(), monkeypatch, JsonChat(payload={"steps": [step(confidence=0.95)]}))
        b, _ = await run(base_state(), monkeypatch, JsonChat(payload={"steps": [step(confidence=0.1)]}))
        assert a["reasoning"][0]["confidence"] == b["reasoning"][0]["confidence"] == 0.725


class TestDegradation:
    async def test_空证据池仍然给一句解释(self, monkeypatch: pytest.MonkeyPatch):
        model = JsonChat(
            payload={
                "steps": [
                    {
                        "phenomenon": "评论区呈现出一种尚未被命名的共同情绪。",
                        "mechanism": "",
                        "insight": "现有证据不足以形成稳定解释。",
                        "evidence_ids": [],
                        "allusion_ids": [],
                    }
                ]
            }
        )
        patch, events = await run(base_state(evidence=[]), monkeypatch, model)

        assert len(patch["reasoning"]) == 1
        assert [e for e in events if e["kind"] == "warning"][0]["code"] == "no_evidence_for_reasoning"
        # 池子空 → mech 与 allu 各给半分，support 0。
        assert patch["reasoning"][0]["confidence"] == 0.225

    async def test_模型挂掉时退回只列现象(self, monkeypatch: pytest.MonkeyPatch):
        patch, events = await run(base_state(), monkeypatch, JsonChat(fail_times=2))

        assert [e for e in events if e["kind"] == "warning"][0]["code"] == "reasoning_failed"
        assert patch["reasoning"][0]["phenomenon"] == cluster()["summary"]
        # 不编机制也不编洞察：留空比模板句诚实，用户至少知道系统读到了现象。
        assert patch["reasoning"][0]["mechanism"] == ""
        # 0.0，而不是「没有心理证据给半分」那 0.45：池子里**有**心理学证据，
        # 只是这一步一条都没引。半分只留给「库里本来就没有这类证据」的情况。
        assert patch["reasoning"][0]["confidence"] == 0.0

    async def test_降级时跳过噪声簇(self, monkeypatch: pytest.MonkeyPatch):
        clusters = [cluster(is_noise=True, cluster_key="noise"), cluster(cluster_key="c1", summary="够格的那条")]
        patch, _ = await run(base_state(clusters=clusters), monkeypatch, JsonChat(fail_times=2))

        assert [s["phenomenon"] for s in patch["reasoning"]] == ["够格的那条"]


class TestEmitting:
    async def test_逐步发且带判别键(self, monkeypatch: pytest.MonkeyPatch):
        model = JsonChat(payload={"steps": [step(), step(evidence_ids=["psy:2"])]})
        patch, events = await run(base_state(), monkeypatch, model)

        frames = [e for e in events if e.get("kind") == "partial" and e.get("data_kind") == "reasoning"]
        assert len(frames) == len(patch["reasoning"]) == 2
        for frame in frames:
            assert "kind" not in frame["data"]
            assert len(frame["data"]["reasoning"]) == 1
        assert [f["data"]["reasoning"][0]["step_index"] for f in frames] == [1, 2]

    async def test_每步都写了产出它的模型(self, monkeypatch: pytest.MonkeyPatch):
        patch, _ = await run(base_state(), monkeypatch, JsonChat(payload={"steps": [step()]}))
        assert patch["reasoning"][0]["model"] == "json-chat"

    async def test_降级步骤也标明是降级(self, monkeypatch: pytest.MonkeyPatch):
        # `reasoning_step` 表有 model 列但接口不下发它——留档是为了排查时
        # 能一眼看出这一版推理是哪来的。
        patch, _ = await run(base_state(), monkeypatch, JsonChat(fail_times=2))
        assert patch["reasoning"][0]["model"] == "fallback"


class TestPromptContext:
    async def test_证据清单里的_id_就是_chunk_id(self, monkeypatch: pytest.MonkeyPatch):
        # 提示词里给的 id 与正文里引用、库里存的必须是同一把键。
        # 用数据库主键的话，节点跑的时候它还不存在（要 flush 后才有）。
        model = JsonChat(payload={"steps": [step()]})
        await run(base_state(), monkeypatch, model)

        brief = model.calls[0]["evidence"]
        assert {item["id"] for item in brief} == {"psy:1", "psy:2", "lit:1", "po:1"}
        assert brief[0]["kind"] == "psychology"

    async def test_重试换了缓存钥匙(self, monkeypatch: pytest.MonkeyPatch):
        model = JsonChat(payload={"steps": [step()]})
        # 先失败一次再成功：两次调用的 context 必须不同，否则 CachedChatModel
        # 会从缓存里把同一份坏输出再端回来。
        model.fail_times = 1
        await run(base_state(), monkeypatch, model)

        assert len(model.calls) == 2
        assert model.calls[0] != model.calls[1]
