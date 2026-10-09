"""故事工坊的接口走查（对应计划里的「curl 走一遍」）。

真跑一轮：建会话 → 发消息 → 读 SSE → 拉快照 → 撞 404/409 → 停止。
打印每步的状态码与关键字段，输出必须是「一眼能看出对不对」的形状。
"""
import json
import sys
import time

import httpx

BASE = "http://127.0.0.1:8000/api/v1"
INPUT = (
    "我妈走的那年我十九岁。她走之前一直说等好了要去看海，"
    "最后也没去成。现在每次看到海我都想，要是能带她来一次就好了。"
    "我不敢跟人说这些，说了他们只会让我别想了。"
)

fail = []


def check(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        fail.append(name)


def parse(block):
    out = []
    for line in block.splitlines():
        if line.startswith("data: "):
            with_ = json.loads(line[6:])
            out.append(with_)
    return out


#: 这一轮结束了。**不是「这条流结束了」**——两者不一样。
TURN_END = {"agent_completed", "agent_cancelled", "error"}


def collect(resp, *, limit=None, stop=None, deadline=None):
    """读到该停的地方为止。

    **不能读到 EOF（`for chunk in resp.iter_text()` 跑干净）。** 这条流的寿命是
    **会话**的，不是一轮的：交稿之后连接仍然开着，等着用户说下一句。第一版这里
    读到 EOF，等后端把流改成跨回合常驻，脚本就整条挂死在第一轮上了。读 SSE 的
    客户端都该有个自己知道什么时候停的判据，而不是等对端关连接。
    """
    got, buf = [], ""
    for chunk in resp.iter_text():
        buf += chunk
        while "\n\n" in buf:
            block, buf = buf.split("\n\n", 1)
            got.extend(parse(block))
            if limit and len(got) >= limit:
                return got
            if stop and got and stop(got[-1]):
                return got
        if deadline and time.time() > deadline:
            return got
    return got


# ---- 能力 ---------------------------------------------------------------
r = httpx.get(f"{BASE}/agent/capabilities", timeout=20)
cap = r.json()
check("capabilities 200", r.status_code == 200, f"model={cap.get('model')} demo={cap.get('demo')} "
      f"allow_novel={cap.get('allow_novel')} novel_hint={cap.get('novel_hint')}")
check("档位与库", cap.get("target_chars_choices") == [300, 600, 900, 1500, 2400]
      and len(cap.get("libraries") or []) == 3, str(cap.get("libraries")))

# ---- 404 ---------------------------------------------------------------
r = httpx.get(f"{BASE}/agent/sessions/nope", timeout=20)
check("不存在的会话 GET 404", r.status_code == 404, r.text[:60])
r = httpx.post(f"{BASE}/agent/sessions/nope/messages", json={"text": "喂"}, timeout=20)
check("不存在的会话 POST 404", r.status_code == 404, r.text[:60])
r = httpx.post(f"{BASE}/agent/sessions/nope/cancel", timeout=20)
check("不存在的会话 cancel 404", r.status_code == 404, r.text[:60])
r = httpx.post(f"{BASE}/agent/sessions", json={"input": "   "}, timeout=20)
check("纯空白材料 422", r.status_code == 422)

# ---- 建会话 -------------------------------------------------------------
r = httpx.post(f"{BASE}/agent/sessions", json={"input": INPUT, "allow_novel": True}, timeout=30)
check("建会话 201", r.status_code == 201, r.text[:80])
sess = r.json()
sid = sess["id"]
# 建会话**就把第一轮排上了**：返回的是 running / turn=1，不是 idle / turn=0。
# 面板按下的那一下「开始写」必须真的开始写——曾经这里返回 idle，前端也没有
# 补一条开场消息，于是会话静静地挂着，模型一次都没跑。
check("新会话：已经在写、带原文", sess["status"] == "running" and sess["turn"] == 1
      and sess["input_text"] == INPUT and sess["input_chars"] == len(INPUT),
      f"status={sess['status']} turn={sess['turn']}")
print(f"     session={sid} title={sess['title']!r}")

# ---- 第一轮不用再发消息（建会话时后端已经把开场白发了） -------------------
t0 = time.time()
r = httpx.post(f"{BASE}/agent/sessions/{sid}/messages", json={"text": "抢一句"}, timeout=30)
check("正在写时第二条 409", r.status_code == 409, r.text[:80])

print("     读 SSE…")
frames = []
with httpx.stream("GET", f"{BASE}/agent/sessions/{sid}/events", timeout=600) as resp:
    frames = collect(resp, stop=lambda f: f.get("type") in TURN_END,
                     deadline=time.time() + 300)
elapsed = time.time() - t0
kinds = [f.get("type") for f in frames]
cards = {}
for f in frames:
    if f.get("type") == "agent_tool_call":
        cards[f["data"]["call_id"]] = f["data"]["name"]
results = [f for f in frames if f.get("type") == "agent_tool_result"]
check("SSE 有终态且是 agent_completed", kinds[-1] == "agent_completed", f"最后 {kinds[-1]}")
seqs = [f["seq"] for f in frames]
want = list(range(1, len(frames) + 1))
ok = seqs == want
detail = f"{len(frames)} 帧"
if not ok:
    seen = set()
    dup = [s for s in seqs if s in seen or seen.add(s)]
    missing = sorted(set(want) - set(seqs))
    detail += f" 重复={dup} 缺失={missing}"
    first_bad = next((i for i, (a, b) in enumerate(zip(seqs, want, strict=False)) if a != b), None)
    detail += f" 首个不同处={first_bad}: {seqs[max(0,(first_bad or 0)-2):(first_bad or 0)+3]}"
check("帧 seq 从 1 连续", ok, detail)
check("工具卡片与结果一一对应", len(cards) >= 2 and len(results) == len(cards),
      f"{len(cards)} 张卡 {list(cards.items())}")
check("工具结果带耗时", all((f["data"].get("elapsed_ms") or 0) > 0 for f in results),
      str([f["data"].get("summary") for f in results])[:120])
deltas = [f for f in frames if f.get("type") == "delta"]
check("正文逐字下发", len(deltas) > 3, f"{len(deltas)} 块")
print(f"     用时 {elapsed:.1f}s，工具调用 {len(results)} 次：{list(cards.values())}")

# ---- 停止后的对话快照 ----------------------------------------------------
r = httpx.get(f"{BASE}/agent/sessions/{sid}", timeout=30)
detail = r.json()
roles = [m["role"] for m in detail["messages"]]
story = detail.get("story") or ""
check("快照：终态 idle", detail["status"] == "idle", detail["status"])
check("快照：有正文", len(story) > 50, f"{len(story)} 字")
check("快照：对话含工具往返", "assistant" in roles and "tool" in roles, str(roles))
check("快照：轮次与工具次数已记账",
      detail["rounds"] >= 2 and detail["tool_calls"] == len(cards),
      f"rounds={detail['rounds']} tool_calls={detail['tool_calls']}")
check("改稿指令不出现在对话里（收尾指令是代码生成的）",
      all("直接交稿" not in m["content"] for m in detail["messages"] if m["role"] == "user"),
      str([m["content"][:12] for m in detail["messages"] if m["role"] == "user"]))

# ---- 改稿 ---------------------------------------------------------------
t1 = time.time()
r = httpx.post(f"{BASE}/agent/sessions/{sid}/messages", json={"text": "再暗一点，短一些。"}, timeout=30)
check("改稿 202", r.status_code == 202)
with httpx.stream("GET", f"{BASE}/agent/sessions/{sid}/events",
                  params={"since": frames[-1]["seq"]}, timeout=600) as resp:
    again = collect(resp, stop=lambda f: f.get("type") in TURN_END,
                    deadline=time.time() + 300)
check("第二轮也有终态", (again[-1].get("type") if again else "") == "agent_completed",
      f"{len(again)} 帧，{time.time() - t1:.1f}s")
detail2 = httpx.get(f"{BASE}/agent/sessions/{sid}", timeout=30).json()
check("第二版与第一版不同", detail2["story"] != story, f"{len(detail2['story'])} 字")
check("turn 累加到 2", detail2["turn"] == 2)
check("seq 跨轮连续", [m["seq"] for m in detail2["messages"]]
      == list(range(1, len(detail2["messages"]) + 1)))

# ---- 刷新重建：从头回放 ------------------------------------------------
with httpx.stream("GET", f"{BASE}/agent/sessions/{sid}/events",
                  params={"since": 0}, timeout=20) as resp:
    replay = collect(resp, limit=1, deadline=time.time() + 5)
check("从头回放接到第 1 帧", bool(replay) and replay[0]["seq"] == 1,
      f"{replay[0]['type'] if replay else '空'}")

# ---- 停止 ---------------------------------------------------------------
r = httpx.post(f"{BASE}/agent/sessions/{sid}/cancel", timeout=30)
check("待命时 cancel 不炸", r.status_code == 202, r.text[:80])
print("     待命时 cancel:", r.json())

# 第四轮：抢一个 409，然后在中途停
r = httpx.post(f"{BASE}/agent/sessions/{sid}/messages", json={"text": "接着写"}, timeout=30)
check("第四轮 202", r.status_code == 202, str(r.status_code))
r = httpx.post(f"{BASE}/agent/sessions/{sid}/messages", json={"text": "抢一句"}, timeout=30)
check("正在写时第二条 409", r.status_code == 409, r.text[:80])
rc = httpx.post(f"{BASE}/agent/sessions/{sid}/cancel", timeout=30)
print("     中途 cancel:", rc.status_code, rc.json())

with httpx.stream("GET", f"{BASE}/agent/sessions/{sid}/events",
                  params={"since": 0}, timeout=600) as resp:
    tail = collect(resp)                    # 终态会话会把流的最后一段补齐后关掉
check("停止后流以终态收尾", bool(tail) and tail[-1]["type"] in
      ("agent_cancelled", "error", "agent_completed"),
      f"{len(tail)} 帧，末帧 {tail[-1]['type'] if tail else '空'}")

st = httpx.get(f"{BASE}/agent/sessions/{sid}", timeout=30).json()
if rc.json().get("accepted"):
    check("会话停在 cancelled", st["status"] == "cancelled", st["status"])
    check("终态会话回放完立刻关流（不会挂住）",
          bool(tail) and tail[-1]["type"] == "agent_cancelled")
    r2 = httpx.post(f"{BASE}/agent/sessions/{sid}/messages", json={"text": "再来"}, timeout=30)
    check("已停止的会话 409（终态单向门）", r2.status_code == 409, r2.text[:80])
    check("取消的这一轮没有覆盖上一版正文", st["story"] == detail2["story"],
          f"{len(st['story'] or '')} 字")
else:
    print(f"     本轮在 cancel 到达前就结束了（status={st['status']}），"
          f"终态 409 由 pytest 的 test_终态会话不能再被喂消息 兜住")

print()
print("失败项：", fail or "无")
sys.exit(1 if fail else 0)
