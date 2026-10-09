"""n7：生成 → 校验 → 重试 → 落库这条闭环。

三个不变量，每一个都曾经在别处以「看起来正常」的方式出过错：

1. **正文里不能有标记残渣。** 库里存的是模型的原话的话，用户就会在洞察
   正文里读到 `[[ev:`。剥离必须彻底，包括没闭合的那种。
2. **逐字拼出来的结果 == 落库正文。** 两者来自两条路径的话，症状是
   「刷新一下，文字变了」。
3. **重试有界且不制造噪声。** 侧写段没有引用契约、空池无事可引，这两处
   都不该重试——否则每一次健康运行都会盖上一个 `retried` 的章。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from app.constants import SectionKey
from app.graph.emitting import NodeEmitter
from app.graph.nodes import n7_literary as n7
from app.providers.base import ChatMessage, ChatResult
from app.text_chunks import chunk_text


@dataclass
class ScriptedChat:
    """按 task 分发脚本化的回复。`fail` 用来演练整篇生成挂掉。"""

    sections: dict[str, str] = field(default_factory=dict)
    regen: dict[str, str] = field(default_factory=dict)
    name: str = "scripted-chat"
    is_mock: bool = True
    fail: bool = False
    fail_on_regen: bool = False
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
        ctx = dict(context or {})
        self.calls.append({"task": task, **ctx})
        if self.fail:
            raise RuntimeError("provider 挂了")
        if self.fail_on_regen and task == "section_regen":
            raise RuntimeError("重试这一次挂了")
        if task == "section_regen":
            key = str(ctx.get("section_key") or "")
            text = json.dumps({"content_md": self.regen.get(key, "")}, ensure_ascii=False)
        else:
            text = json.dumps(self.sections, ensure_ascii=False)
        return ChatResult(text=text, model=self.name, finish_reason="stop")


def use_model(monkeypatch: pytest.MonkeyPatch, model: Any) -> None:
    monkeypatch.setattr("app.providers.factory.get_chat_model", lambda: model)


def evidence(chunk_id: str, kind: str) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "kind": kind,
        "library": kind,
        "title": chunk_id,
        "source": f"{chunk_id} 的出处",
        "author": "某人",
        "text": "正文",
    }


POOL = [
    evidence("psy:1", "psychology"),
    evidence("lit:1", "literature"),
]

MARKED = "机制说明 [[ev:psy:1]]"


def base_state(**over: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "evidence": list(POOL),
        "profile": {"core_tension": "想念却再也无法说出口", "global_tags": {"emotion": ["哀伤"]}},
        "clusters": [{"cluster_key": "c0", "label": "想念", "size": 12, "is_noise": False}],
        "video": {"title": "一条视频"},
    }
    state.update(over)
    return state


async def run(
    state: dict[str, Any], monkeypatch: pytest.MonkeyPatch, model: Any
) -> tuple[dict, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(n7, "NodeEmitter", lambda node: NodeEmitter(node, write=events.append))
    use_model(monkeypatch, model)
    patch = await n7.n7_literary(state)
    return patch, events


def deltas(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in events if e["kind"] == "delta"]


def joined(events: list[dict[str, Any]], key: str) -> str:
    return "".join(e["text"] for e in deltas(events) if e["section"] == key)


# ----------------------------------------------------------------------


class TestShape:
    async def test_四段齐全且键名固定(self, monkeypatch: pytest.MonkeyPatch):
        model = ScriptedChat(sections={k.value: f"{k.title}的正文。" for k in SectionKey})
        patch, _ = await run(base_state(), monkeypatch, model)

        assert set(patch["sections"]) == {k.value for k in SectionKey}
        assert patch["sections"]["profile"]["title"] == "心理侧写"
        assert patch["sections"]["insight"]["content_md"] == "最终洞察的正文。"

    async def test_模型少给一段也只是那一段空着(self, monkeypatch: pytest.MonkeyPatch):
        model = ScriptedChat(sections={"insight": "只有这一段。"})
        patch, _ = await run(base_state(), monkeypatch, model)

        assert patch["sections"]["insight"]["content_md"] == "只有这一段。"
        assert patch["sections"]["profile"]["content_md"] == ""

    async def test_模型给的怪形状不炸(self, monkeypatch: pytest.MonkeyPatch):
        model = ScriptedChat(sections={"profile": ["不是字符串"], "insight": None})
        patch, _ = await run(base_state(), monkeypatch, model)

        # 非字符串一律当空段，不 str() 硬转——那会把 `['不是字符串']`
        # 原样渲染进右栏给用户看。
        assert patch["sections"]["profile"]["content_md"] == ""
        assert patch["sections"]["insight"]["content_md"] == ""


class TestSanitize:
    async def test_正文里没有标记残渣(self, monkeypatch: pytest.MonkeyPatch):
        model = ScriptedChat(
            sections={
                "mechanism": f"{MARKED}\n\n还有没闭合的 [[ev:psy:1",
                "allusion": "池外的 [[ev:不存在]] 也要消失。",
            }
        )
        patch, _ = await run(base_state(), monkeypatch, model)

        for section in patch["sections"].values():
            assert "[[" not in section["content_md"]
            assert "]]" not in section["content_md"]
        assert "还有没闭合的" in patch["sections"]["mechanism"]["content_md"]

    async def test_合法标记换成脚注记号(self, monkeypatch: pytest.MonkeyPatch):
        patch, _ = await run(
            base_state(), monkeypatch, ScriptedChat(sections={"mechanism": MARKED})
        )

        section = patch["sections"]["mechanism"]
        assert section["content_md"] == "机制说明 [^1]"
        assert section["citations"] == [
            {
                "evidence_id": "psy:1",
                "marker": "[^1]",
                "text": "《psy:1》· 某人",
                "library": "psychology",
            }
        ]

    async def test_池外引用拉低覆盖率(self, monkeypatch: pytest.MonkeyPatch):
        # 分母按「出现次数」算：一枚好的加一枚坏的 → 0.5。
        model = ScriptedChat(sections={"insight": "甲 [[ev:psy:1]] 乙 [[ev:不存在]]"})
        patch, _ = await run(base_state(), monkeypatch, model)

        assert patch["sections"]["insight"]["citation_coverage"] == 0.5
        assert patch["sections"]["insight"]["content_json"]["dropped_ids"] == ["不存在"]


class TestRetry:
    async def test_覆盖率不足时重生成并取更优者(self, monkeypatch: pytest.MonkeyPatch):
        model = ScriptedChat(
            sections={"mechanism": "没有引用的机制说明。"},
            regen={"mechanism": MARKED},
        )
        patch, _ = await run(base_state(), monkeypatch, model)

        section = patch["sections"]["mechanism"]
        assert section["content_md"] == "机制说明 [^1]"
        assert section["citation_coverage"] == 1.0
        assert section["content_json"]["retried"] is True
        assert section["content_json"]["regen_coverage"] == 1.0
        assert [c["task"] for c in model.calls].count("section_regen") == 1

    async def test_重生成更差时保留原版(self, monkeypatch: pytest.MonkeyPatch):
        model = ScriptedChat(
            sections={"mechanism": "甲 [[ev:psy:1]] 乙 [[ev:不存在]]"},
            regen={"mechanism": "全是坏的 [[ev:假]] [[ev:假2]]"},
        )
        patch, _ = await run(base_state(), monkeypatch, model)

        section = patch["sections"]["mechanism"]
        # 原版 0.5，重生成 0.0 → 原版留下，但重试这件事要如实记下来。
        assert section["citation_coverage"] == 0.5
        assert section["content_md"] == "甲 [^1] 乙 "
        assert section["content_json"]["retried"] is True

    async def test_两次都不达标也接受(self, monkeypatch: pytest.MonkeyPatch):
        model = ScriptedChat(sections={"allusion": "没有引用。"}, regen={"allusion": "还是没有。"})
        patch, _ = await run(base_state(), monkeypatch, model)

        section = patch["sections"]["allusion"]
        assert section["content_md"] == "没有引用。"
        assert section["citation_coverage"] == 0.0
        assert section["content_json"]["retried"] is True
        # 有界：只重试一次，不因为还是不达标就再来一轮。
        assert [c["task"] for c in model.calls].count("section_regen") == 1

    async def test_侧写段不重试(self, monkeypatch: pytest.MonkeyPatch):
        # 侧写段的契约里没有引用标记，它永远达不到 0.6。为它重试等于
        # 每次运行都白跑一次必然失败的调用，还给健康结果盖上 retried 的章。
        model = ScriptedChat(sections={k.value: "没有引用。" for k in SectionKey})
        patch, _ = await run(base_state(), monkeypatch, model)

        assert patch["sections"]["profile"]["content_json"]["retried"] is False
        assert "profile" not in {c.get("section_key") for c in model.calls if c["task"] == "section_regen"}

    async def test_空证据池不重试(self, monkeypatch: pytest.MonkeyPatch):
        model = ScriptedChat(sections={"mechanism": "没有引用。"})
        patch, _ = await run(base_state(evidence=[]), monkeypatch, model)

        assert patch["sections"]["mechanism"]["content_json"]["retried"] is False
        assert [c["task"] for c in model.calls] == ["literary"]

    async def test_空段落不重试(self, monkeypatch: pytest.MonkeyPatch):
        # 模型整段没写：重试它等于赌一次它愿意写，而它连第一次都没写。
        model = ScriptedChat(sections={"insight": ""})
        patch, _ = await run(base_state(), monkeypatch, model)

        assert patch["sections"]["insight"]["content_json"]["retried"] is False

    async def test_重试换了缓存钥匙(self, monkeypatch: pytest.MonkeyPatch):
        # `CachedChatModel` 按 (task, messages, context) 哈希。重试若沿用
        # 同一把钥匙，拿回来的还是上次那份没有引用的正文，重试看起来没生效。
        model = ScriptedChat(sections={"mechanism": "没有引用。"}, regen={"mechanism": MARKED})
        await run(base_state(), monkeypatch, model)

        first = model.calls[0]
        regen = [c for c in model.calls if c["task"] == "section_regen"][0]
        assert regen["task"] != first["task"]
        assert regen["section_key"] == "mechanism"

    async def test_重生成挂掉时保留原版(self, monkeypatch: pytest.MonkeyPatch):
        model = ScriptedChat(
            sections={"mechanism": "甲 [[ev:psy:1]] 乙 [[ev:不存在]]"}, fail_on_regen=True
        )
        patch, _ = await run(base_state(), monkeypatch, model)

        section = patch["sections"]["mechanism"]
        # 重试失败不该把整段一起弄丢：空串输给原版，原版留下。
        assert section["content_md"] == "甲 [^1] 乙 "
        assert section["content_json"]["retried"] is True


class TestDeltas:
    async def test_拼接结果等于落库正文(self, monkeypatch: pytest.MonkeyPatch):
        body = (
            "第一段。这里有一句很长的话，长到需要被切开，"
            "因为它超过了四十个字符的上限，切块器必须处理它。\n\n"
            "**加粗的一句。**后面还有。\n> 引用一行。\n- 列表一项。"
        )
        model = ScriptedChat(sections={"insight": body})
        patch, events = await run(base_state(), monkeypatch, model)

        assert joined(events, "insight") == patch["sections"]["insight"]["content_md"] == body

    async def test_每段各自成串且顺序固定(self, monkeypatch: pytest.MonkeyPatch):
        model = ScriptedChat(sections={k.value: f"{k.title}。" for k in SectionKey})
        _, events = await run(base_state(), monkeypatch, model)

        seen = [e["section"] for e in deltas(events)]
        assert seen == [k.value for k in SectionKey]  # 每段一块，段间不穿插

    async def test_空段落不发_delta(self, monkeypatch: pytest.MonkeyPatch):
        model = ScriptedChat(sections={"insight": "有内容。"})
        _, events = await run(base_state(), monkeypatch, model)

        assert "profile" not in {e["section"] for e in deltas(events)}

    async def test_元信息先于正文且正文留空(self, monkeypatch: pytest.MonkeyPatch):
        # 顺序反了的话，元信息里的 content_md 会把已经逐字打出来的正文
        # 整个覆盖掉——用户看到正文写完又消失。而元信息里的 citations
        # 必须在正文之前到，否则逐字打出时会先闪过一串没有下文的 `[^1]`。
        model = ScriptedChat(sections={"mechanism": MARKED, "insight": "有内容。"})
        _, events = await run(base_state(), monkeypatch, model)

        stream = [e for e in events if e["kind"] in ("partial", "delta")]
        first_delta = next(i for i, e in enumerate(stream) if e["kind"] == "delta")
        meta = stream[first_delta - 1]

        assert meta["kind"] == "partial" and meta["data_kind"] == "sections"
        section = meta["data"]["sections"][0]
        assert section["key"] == "mechanism"
        assert section["title"] == "科学机制"
        assert section["content_md"] == ""
        assert section["citations"][0]["marker"] == "[^1]"


class TestChunking:
    """切块实现已搬到 `app/text_chunks`（n7 与故事 agent 共用一份）。"""

    def test_只切不删(self):
        for text in ["", "短", "。", "**粗**体", "a" * 200, "第一句。第二句！第三句？"]:
            assert "".join(chunk_text(text)) == text

    def test_句末优先断(self):
        chunks = chunk_text("第一句到此为止。第二句也到此为止。第三句。", size=40, floor=6)
        assert chunks[0].endswith("。")
        assert all(len(c) <= 40 for c in chunks)

    def test_标点处不断开粗体标记(self):
        # 句号落在一对 `**` 中间：在这一刀切下去，正文里就会留下
        # 孤零零的两个星号，而它们永远不会被闭合。
        text = "甲" * 15 + "**" + "乙" * 20 + "。" + "**" + "丙" * 10
        chunks = chunk_text(text, size=40, floor=12)

        assert "".join(chunks) == text
        assert all(c.count("**") % 2 == 0 for c in chunks)
        # 证明这一刀确实是「忍住」而不是碰巧没切到：句号落在第 38 个字，
        # 没有护栏的话第一块会停在 38；停到 40 说明它在句号处等了两个字符，
        # 硬上限到了才断。
        assert len(chunks[0]) == 40
        assert not chunks[0].endswith("。")

    def test_没有标点的长文本按长度硬切(self):
        chunks = chunk_text("啊" * 100, size=40, floor=12)
        assert [len(c) for c in chunks] == [40, 40, 20]


class TestDegradation:
    async def test_整篇生成挂掉时四段为空且发warning(self, monkeypatch: pytest.MonkeyPatch):
        patch, events = await run(base_state(), monkeypatch, ScriptedChat(fail=True))

        assert [e for e in events if e["kind"] == "warning"][0]["code"] == "literary_failed"
        assert all(s["content_md"] == "" for s in patch["sections"].values())
        assert all(s["model"] == "fallback" for s in patch["sections"].values())
        # 没有模型就没有重试——重试的还是同一个挂掉的 provider。
        assert [e for e in events if e["kind"] == "delta"] == []


class TestContext:
    async def test_证据清单里的_id_就是_chunk_id(self, monkeypatch: pytest.MonkeyPatch):
        model = ScriptedChat(sections={"insight": "正文。"})
        await run(base_state(), monkeypatch, model)

        assert {e["id"] for e in model.calls[0]["evidence"]} == {"psy:1", "lit:1"}

    async def test_簇保留情绪标签(self, monkeypatch: pytest.MonkeyPatch):
        # 侧写段要写出「哪个簇是什么情绪基调」，砍掉它模型只能对着簇标签猜。
        state = base_state(
            clusters=[
                {"cluster_key": "c0", "label": "想念", "size": 12, "emotion_tags": ["哀伤", "愧疚"]}
            ]
        )
        model = ScriptedChat(sections={"insight": "正文。"})
        await run(state, monkeypatch, model)

        assert model.calls[0]["clusters"][0]["emotion_tags"] == ["哀伤", "愧疚"]

    async def test_噪声簇不进提示词(self, monkeypatch: pytest.MonkeyPatch):
        state = base_state(
            clusters=[
                {"cluster_key": "n", "label": "杂音", "size": 4, "is_noise": True},
                {"cluster_key": "c0", "label": "想念", "size": 12},
            ]
        )
        model = ScriptedChat(sections={"insight": "正文。"})
        await run(state, monkeypatch, model)

        assert [c["label"] for c in model.calls[0]["clusters"]] == ["想念"]

    async def test_带上视频标题与作者(self, monkeypatch: pytest.MonkeyPatch):
        state = base_state(video={"title": "夜里的语音", "author_name": "某某"})
        model = ScriptedChat(sections={"insight": "正文。"})
        await run(state, monkeypatch, model)

        assert model.calls[0]["video"] == {"title": "夜里的语音", "author": "某某"}
