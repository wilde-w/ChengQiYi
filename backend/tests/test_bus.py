"""总线订阅：**一条事件只能送一次**。

这条不变量在 SSE 那边是肉眼可见的：`applyEvent` 按 `seq` 推进游标，
多送一帧就表现为「界面上某一帧的内容出现两遍」，而流的另一种坏法
（少送一帧）会让续传的游标停在中间——刷新之后接不上。

两种实现各有一个「回放与订阅之间的窗口」，而且两个窗口**方向相反**：

- `InProcessBus`：先注册后回放。窗口里到达的事件在**两边都有**（队列里一份、
  回放快照里一份）→ 不去重就送两遍。
- `RedisStreamBus`：先回放后 `xread`。`xread` 的起点是**不含**语义，如果起点
  停在调用方给的游标上而不是「刚回放到的那一条」，窗口里的那几条同样送两遍。

两条都在真机上演过：agent 会话的一轮里出现了一帧重复的 `delta`。
"""

from __future__ import annotations

import asyncio
from contextlib import suppress

import pytest

from app.constants import EventType
from app.graph.bus import InProcessBus, RedisStreamBus

RUN = "bus-test-run"


def _ev(seq: int) -> object:
    from app.graph.bus import RunEvent

    return RunEvent(seq=seq, run_id=RUN, type=EventType.DELTA, data={"i": seq})


async def _take(source: object, n: int, timeout: float = 3.0) -> list[int]:
    """收满 n 条就收工。超时/结束都算数——断言的是收到的那一串，不是「收满」。"""
    out: list[int] = []
    it = source.__aiter__()
    try:
        while len(out) < n:
            event = await asyncio.wait_for(it.__anext__(), timeout)
            out.append(event.seq)
    except (TimeoutError, StopAsyncIteration):
        pass
    finally:
        with suppress(Exception):
            await source.aclose()
    return out


# ----------------------------------------------------------------------
# 进程内实现
# ----------------------------------------------------------------------


async def test_进程内总线_订阅窗口里到达的事件只送一次(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """把窗口撑开成确定的：回放**那次调用里**发一条。

    注册先于回放，所以这条事件同时落进队列和回放快照——正是真机上的那个窗口，
    只是这里不再靠运气撞上它。
    """
    bus = InProcessBus()
    await bus.publish(_ev(1))
    real_replay = bus.replay

    async def replay_publishing(run_id: str, since_seq: int = 0) -> list[object]:
        await bus.publish(_ev(2))
        return await real_replay(run_id, since_seq)

    monkeypatch.setattr(bus, "replay", replay_publishing)

    # 只等 0.5s：这两条是内存里立刻可达的，等满了就是没等到（也就是没有第三份）
    assert await _take(bus.subscribe(RUN, 0), 3, timeout=0.5) == [1, 2]


async def test_进程内总线_订阅能看到之后发布的事件() -> None:
    """去重不能顺手把「真的新事件」也拦掉——那会变成另一种坏法（丢帧）。"""
    bus = InProcessBus()
    await bus.publish(_ev(1))

    async def feed() -> None:
        await asyncio.sleep(0.05)
        for seq in (2, 3):
            await bus.publish(_ev(seq))

    task = asyncio.create_task(feed())
    assert await _take(bus.subscribe(RUN, 1), 3) == [2, 3]
    await task


async def test_进程内总线_起点之后的事件才送() -> None:
    bus = InProcessBus()
    for seq in (1, 2, 3):
        await bus.publish(_ev(seq))
    assert await _take(bus.subscribe(RUN, 1), 3) == [2, 3]


# ----------------------------------------------------------------------
# Redis Streams
# ----------------------------------------------------------------------


@pytest.fixture
async def redis_bus() -> object:
    """Redis 不可达就跳过并说明原因（与 PG 那边的口径一致）。"""
    from app.clients.redis_client import get_redis, ping

    try:
        reachable = await ping()
    except Exception as exc:
        pytest.skip(f"需要 Redis：{type(exc).__name__}: {exc}")
    if not reachable:
        pytest.skip("需要 Redis：ping 失败")

    bus = RedisStreamBus(get_redis(), namespace="test")
    await bus.clear(RUN)
    yield bus
    await bus.clear(RUN)


async def test_redis_订阅不把已有的事件送两遍(redis_bus: RedisStreamBus) -> None:
    """对着已经有的 1,2,3 从 1 订阅。**修复前这里是 [2,3,2,3]**。

    这一条是确定性的红：不需要任何竞态，起点判断写错就必然重复——因为
    `xread` 从 1-0 起读会把 2、3 再读一遍。
    """
    for seq in (1, 2, 3):
        await redis_bus.publish(_ev(seq))
    assert await _take(redis_bus.subscribe(RUN, 1), 4, timeout=2.0) == [2, 3]


async def test_redis_边订阅边发布_每条只到一次(redis_bus: RedisStreamBus) -> None:
    """真机上的形状：订阅挂在半路，事件一条条来。seq 必须严格递增、无重复。

    这一个窗口是抢出来的，命中率不保证——所以守这条不变量的主力是上面那条
    确定性的用例，这里只负责覆盖「真的边订边发」这条路径。
    """

    async def feed() -> None:
        await asyncio.sleep(0.05)
        for seq in (1, 2, 3, 4, 5):
            await redis_bus.publish(_ev(seq))
            await asyncio.sleep(0.02)

    task = asyncio.create_task(feed())
    got = await _take(redis_bus.subscribe(RUN, 0), 5, timeout=5.0)
    await task
    assert got == sorted(set(got)), f"有重复或乱序：{got}"
    assert got[0] == 1 and len(got) == 5, got
