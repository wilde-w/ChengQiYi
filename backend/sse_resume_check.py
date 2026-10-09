"""M4 验收：模拟「进度到一半时刷新页面」，确认从 since 精确续上、无缺口。

用法：`python sse_resume_check.py [链接或分享文本]`。默认那条短链是**假的**
（`DEMO1234`），解析要真的走一次网络跳转——抖音早就把它跳回首页了，于是
「短链跳转后仍未找到 aweme_id」400。要真跑就传一个能解析的链接，例如上一次
成功运行里用过的那个。
"""
import json, sys, time
import httpx

BASE = "http://127.0.0.1:8000/api/v1"
INPUT = sys.argv[1] if len(sys.argv) > 1 else "https://v.douyin.com/DEMO1234/"

def parse(block):
    out = []
    for line in block.splitlines():
        if line.startswith("data: "):
            out.append(json.loads(line[6:]))
    return out

def frames(resp, limit=None, deadline=None):
    got = []
    buf = ""
    for chunk in resp.iter_text():
        buf += chunk
        while "\n\n" in buf:
            block, buf = buf.split("\n\n", 1)
            for e in parse(block):
                got.append(e)
                if limit and len(got) >= limit:
                    return got
        if deadline and time.time() > deadline:
            return got
    return got

resp = httpx.post(f"{BASE}/runs", json={
    "input": INPUT, "depth": "standard", "comment_limit": 100,
}, timeout=30)
if resp.status_code >= 400:
    # 链接解析失败也要一眼看出原因，而不是一串 httpx 的堆栈
    print("建运行失败:", resp.status_code, resp.text[:300])
    sys.exit(2)
resp.raise_for_status()
run_id = resp.json()["run"]["id"]
print("run:", run_id)

# --- 第一次连接：看到一半就断（等价于用户刷新页面） ---
first = []
with httpx.stream("GET", f"{BASE}/runs/{run_id}/events", timeout=60) as r:
    first = frames(r, limit=18)
last_seq = first[-1]["seq"]
print(f"第一次连接收到 {len(first)} 帧，最后 seq={last_seq}（断开）")

# --- 第二次连接：带 since 续上 ---
second = []
with httpx.stream("GET", f"{BASE}/runs/{run_id}/events",
                  params={"since": last_seq}, timeout=120) as r:
    second = frames(r)
print(f"第二次连接收到 {len(second)} 帧，seq {second[0]['seq']}..{second[-1]['seq']}")

allseq = [e["seq"] for e in first] + [e["seq"] for e in second]
ok_contiguous = allseq == list(range(1, allseq[-1] + 1))
# `.get` 而不是 `[...]`：`to_sse()` 是 `exclude_none=True`，没有进度的帧
# （比如终态那条）整个键都不在，取它直接 KeyError。
prog = [e.get("progress") for e in first + second if e.get("progress") is not None]
ok_monotonic = all(b >= a for a, b in zip(prog, prog[1:]))
last = second[-1]

print()
print("无缺口（seq 1..N 连续）:", ok_contiguous)
print("进度单调不减        :", ok_monotonic)
print("终态                :", last["type"], last["progress"])
print("第二次首帧 = 断点+1 :", second[0]["seq"] == last_seq + 1)

# 终态运行再连一次：应当回放完历史就关闭
with httpx.stream("GET", f"{BASE}/runs/{run_id}/events", params={"since": 0}, timeout=30) as r:
    replay = frames(r)
print("终态运行回放后自动关闭:", replay[-1]["seq"] == last["seq"], f"({len(replay)} 帧)")

sys.exit(0 if (ok_contiguous and ok_monotonic and last["type"] == "run_completed"
               and second[0]["seq"] == last_seq + 1) else 1)
