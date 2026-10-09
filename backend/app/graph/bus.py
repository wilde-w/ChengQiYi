"""事件总线：把流水线的进展送到 SSE 订阅者手里。

**为什么不是 pub/sub。** Redis 的 pub/sub 是即发即忘的：浏览器在 40% 时刷新，
重连上来只能收到之后的事件，前面 40% 的产物全部丢失，界面会退回到半空状态。
所以这里用 Streams——每条事件持久在流里，`since=seq` 就能精确续上。
流被 MAXLEN 裁剪掉的那一段，由调用方回退查 PG 的 run_event 表补齐。

两个实现：
  RedisStreamBus  生产路径。跨进程，可重放。
  InProcessBus    Redis 不可达时的降级 + 单测。保留有界内存缓冲，同样支持 since。

`seq` 由发布方单调分配，同时用作 Redis Stream 的 ID（`{seq}-0`），
因此 `XRANGE` 的起点判断不需要额外索引。

## 命名空间

流水线用 `namespace="run"`（默认），故事工坊用 `"agent"`。两边的 seq 各自
从 1 开始，Stream 键也分开（`guanxin:run:…` / `guanxin:agent:…`）。这不是
洁癖：如果共用一个流，agent 的 `seq` 会被流水线的事件推着走，而前端的游标
是按会话存的——刷新一次就会把「重新回放整场对话」当成「接到了新事件」。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.constants import EventType
from app.logging_conf import get_logger

log = get_logger(__name__)

STREAM_MAXLEN = 2000
#: 一次阻塞读等多久。**这个数被实测卡死在 5 秒以下，不是随手挑的。**
#:
#: 本来写的是 15000，而在这台机器上每一条空闲的事件流都在刷
#: `bus_read_failed error='Timeout reading from 127.0.0.1:56379'`：那个 15 秒的
#: 阻塞读在 5.0 秒整被掐断，**抛异常**而不是返回空。实测（同一台 Redis，只改
#: block 值）：2000/3000/4000/4500 都正常返回 `[]`，5000 与 6000 一律在 5.02s
#: 抛 `TimeoutError`。客户端侧 `socket_timeout` / `socket_connect_timeout` /
#: `health_check_interval` 全加全不加都一样——**不是客户端设的**。查服务端：
#: `timeout 0`、端口 6379，而我们是连到 56379 的转发口。所以那 5 秒是中间那层
#: 端口转发对空闲连接的上限（WSL/docker 的 localhost 转发都有这个毛病）。
#:
#: 代价原来很具体：读永远走异常分支 → 每个客户端每 6 秒（5 秒超时 + 1 秒
#: `sleep`）一条 warning，事件最坏多等 1 秒才被取走。取 3000 留出足够余量，
#: 空读一次只是一个来回，比每 6 秒一条假的错误日志便宜得多。
BLOCK_MS = 3000
# 进程内缓冲上限。与 Streams 的 MAXLEN 同量级，保证降级时行为一致。
MEMORY_BUFFER = 2000


class RunEvent(BaseModel):
    """SSE 事件的统一信封。前端 store/applyEvent.ts 按 `type` 分派。"""

    seq: int
    run_id: str
    type: EventType
    node: str | None = None
    progress: float | None = None
    message: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    ts: float = Field(default_factory=time.time)

    def to_sse(self) -> str:
        """SSE 帧。event 名用 type，前端可以 addEventListener 直接监听。"""
        payload = self.model_dump_json(exclude_none=True)
        return f"id: {self.seq}\nevent: {self.type.value}\ndata: {payload}\n\n"


#: 默认命名空间。流水线用它，且**只有它**——现有调用点一个字都不用改。
NAMESPACE_RUN = "run"


def stream_key(run_id: str, *, namespace: str = NAMESPACE_RUN) -> str:
    """Streams 里的键。`namespace` 让故事工坊与流水线共用这一份实现而不串台。"""
    return f"guanxin:{namespace}:{run_id}:events"


class RunBus(Protocol):
    async def publish(self, event: RunEvent) -> None: ...

    def subscribe(self, run_id: str, since_seq: int = 0) -> AsyncIterator[RunEvent]: ...


# ----------------------------------------------------------------------
# 进程内实现
# ----------------------------------------------------------------------


class InProcessBus:
    """单进程内的事件分发。Redis 不可用时自动接管，也让流水线单测不需要 Redis。"""

    name = "in-process"

    def __init__(self) -> None:
        self._buffers: dict[str, list[RunEvent]] = {}
        self._subscribers: dict[str, set[asyncio.Queue[RunEvent | None]]] = {}
        self._lock = asyncio.Lock()

    async def publish(self, event: RunEvent) -> None:
        async with self._lock:
            buf = self._buffers.setdefault(event.run_id, [])
            buf.append(event)
            if len(buf) > MEMORY_BUFFER:
                del buf[: len(buf) - MEMORY_BUFFER]
            queues = list(self._subscribers.get(event.run_id, ()))
        for q in queues:
            # 队列满时丢最旧的：SSE 客户端落后太多时，宁可丢中间帧也不要阻塞流水线
            if q.full():
                with suppress(asyncio.QueueEmpty):
                    q.get_nowait()
            with suppress(asyncio.QueueFull):
                q.put_nowait(event)

    async def replay(self, run_id: str, since_seq: int = 0) -> list[RunEvent]:
        async with self._lock:
            buf = self._buffers.get(run_id, [])
            return [e for e in buf if e.seq > since_seq]

    async def subscribe(self, run_id: str, since_seq: int = 0) -> AsyncIterator[RunEvent]:
        queue: asyncio.Queue[RunEvent | None] = asyncio.Queue(maxsize=1024)
        async with self._lock:
            self._subscribers.setdefault(run_id, set()).add(queue)
        try:
            # 回放与订阅之间可能漏事件：publish 在 replay 之后、注册之前发生。
            # 上面的注册先于 replay，所以这里再补一次差集。
            #
            # 差集的判据是 `seen`（**已经送出去的最后一条**）而不是调用方给的
            # `since_seq`：注册先于回放，窗口里到达的事件**两边都有**——回放里
            # 一份、队列里一份。用 since_seq 当判据就会把它送两遍。
            seen = since_seq
            for event in await self.replay(run_id, since_seq):
                seen = max(seen, event.seq)
                yield event
            while True:
                event = await queue.get()
                if event is None:
                    break
                if event.seq > seen:
                    seen = event.seq
                    yield event
        finally:
            async with self._lock:
                subs = self._subscribers.get(run_id)
                if subs is not None:
                    subs.discard(queue)
                    if not subs:
                        self._subscribers.pop(run_id, None)

    async def close_run(self, run_id: str) -> None:
        """通知订阅者流已结束。"""
        async with self._lock:
            queues = list(self._subscribers.get(run_id, ()))
        for q in queues:
            with suppress(asyncio.QueueFull):
                q.put_nowait(None)

    async def clear(self, run_id: str) -> None:
        async with self._lock:
            self._buffers.pop(run_id, None)

    async def aclose(self) -> None:
        async with self._lock:
            run_ids = list(self._subscribers)
        for run_id in run_ids:
            await self.close_run(run_id)


# ----------------------------------------------------------------------
# Redis Streams 实现
# ----------------------------------------------------------------------


class RedisStreamBus:
    """生产路径。事件持久在 Stream 里，`since` 可精确续传。

    `namespace` 只影响 Stream 的键名，不影响行为——故事工坊用
    `namespace="agent"` 拿到另一组流，两边的 seq 各自从 1 开始。
    """

    name = "redis-streams"

    def __init__(self, client: Any, *, namespace: str = NAMESPACE_RUN) -> None:
        self._redis = client
        self._ns = namespace

    async def publish(self, event: RunEvent) -> None:
        key = stream_key(event.run_id, namespace=self._ns)
        try:
            await self._redis.xadd(
                key,
                {"e": event.model_dump_json()},
                # ID 直接用 seq：这样 XRANGE 的起点判断不需要额外索引。
                # 代价是 seq 必须严格单调——由 RunEmitter 保证。
                id=f"{event.seq}-0",
                maxlen=STREAM_MAXLEN,
                approximate=True,
            )
        except Exception as exc:
            # 总线故障绝不能让分析中断——退回日志，SSE 侧仍可从 PG 回放
            log.warning("bus_publish_failed", run_id=event.run_id, seq=event.seq, error=str(exc))

    async def replay(self, run_id: str, since_seq: int = 0) -> list[RunEvent]:
        key = stream_key(run_id, namespace=self._ns)
        start = f"({since_seq}-0" if since_seq > 0 else "-"
        try:
            rows = await self._redis.xrange(key, min=start, max="+")
        except Exception as exc:
            log.warning("bus_replay_failed", run_id=run_id, error=str(exc))
            return []
        return [e for e in (_decode(raw) for _id, raw in rows) if e is not None]

    async def subscribe(self, run_id: str, since_seq: int = 0) -> AsyncIterator[RunEvent]:
        # **游标要跟着「刚回放到的那一条」走，不能停在调用方给的 `since_seq`。**
        # 回放与 `xread` 之间新到的事件两边都有，而 xread 的起点是**不含**语义，
        # 于是那几条会被送两遍。实测：对着已经有的 1,2,3 从 1 订阅会收到
        # `[2,3,2,3]`。正常路径上这个窗口只有零点几毫秒（回放里紧跟着查一次
        # PG），所以症状是「偶尔多一帧」——SSE 那条不变式（seq 从 1 起连续）
        # 一帧就破，而内容看着完全正常，最难查。
        last = f"{since_seq}-0"
        for event in await self.replay(run_id, since_seq):
            last = f"{event.seq}-0"
            yield event

        key = stream_key(run_id, namespace=self._ns)
        while True:
            try:
                resp = await self._redis.xread({key: last}, count=64, block=BLOCK_MS)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("bus_read_failed", run_id=run_id, error=str(exc))
                await asyncio.sleep(1.0)
                continue

            if not resp:
                # 阻塞超时。交给调用方发心跳——这里不产生事件，避免
                # 把「没有进展」伪装成「有事件」。
                continue

            for _stream, rows in resp:
                for raw_id, fields in rows:
                    last = raw_id if isinstance(raw_id, str) else raw_id.decode()
                    event = _decode(fields)
                    if event is not None:
                        yield event

    async def clear(self, run_id: str) -> None:
        with suppress(Exception):
            await self._redis.delete(stream_key(run_id, namespace=self._ns))

    async def aclose(self) -> None:  # 连接由 redis_client 统一管理
        return None


def _decode(fields: dict[Any, Any]) -> RunEvent | None:
    raw = fields.get("e") or fields.get(b"e")
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return RunEvent.model_validate(json.loads(raw))
    except Exception as exc:
        log.warning("bus_decode_failed", error=str(exc))
        return None


# ----------------------------------------------------------------------
# 发射器：分配 seq 并同时落总线与 PG
# ----------------------------------------------------------------------


class RunEmitter:
    """发布事件的唯一入口。

    同时负责：seq 分配（单调）、进度写入、PG 持久化调度。
    节点通过 `get_stream_writer()` 发自定义事件，runner 把它们交给这里。
    """

    def __init__(
        self,
        bus: RunBus,
        run_id: str,
        *,
        start_seq: int = 0,
        on_emit: Callable[[RunEvent], Awaitable[None]] | None = None,
    ) -> None:
        self._bus = bus
        self._run_id = run_id
        self._seq = start_seq
        self._on_emit = on_emit
        self.progress = 0.0
        self._lock = asyncio.Lock()

    @property
    def seq(self) -> int:
        return self._seq

    async def emit(
        self,
        event_type: EventType,
        *,
        node: str | None = None,
        message: str | None = None,
        progress: float | None = None,
        data: dict[str, Any] | None = None,
    ) -> RunEvent:
        async with self._lock:
            self._seq += 1
            seq = self._seq
            if progress is not None:
                # 单调钳制放在这里而不是调用方——所有事件都经过这个入口，
                # 这是唯一能保证「进度绝不回退」的地方。
                progress = max(progress, self.progress)
                self.progress = progress
            event = RunEvent(
                seq=seq,
                run_id=self._run_id,
                type=event_type,
                node=node,
                progress=progress,
                message=message,
                data=data or {},
            )
        await self._bus.publish(event)
        if self._on_emit is not None:
            # 落 PG 是回放兜底（Streams 会被 MAXLEN 裁剪）。
            # 它失败不能影响主流程——发布已经成功，前端照样收得到。
            with suppress(Exception):
                await self._on_emit(event)
        return event


# ----------------------------------------------------------------------
# 全局总线
# ----------------------------------------------------------------------

_buses: dict[str, RunBus] = {}


async def get_bus(namespace: str = NAMESPACE_RUN) -> RunBus:
    """选一条总线。**每个命名空间一条**，各自缓存。

    Redis 可达就用 Streams（跨进程 + 可重放）；否则退到进程内实现。
    降级是**明确且被记录**的，不静默——进程内总线只对同进程的 SSE 订阅者有效，
    而 V1 强制 `--workers 1`，所以这条降级路径实际是可用的。

    「每个命名空间一条」不只是为了 Stream 键名：进程内总线也要分开。否则
    故事工坊的会话 id 会落进流水线那张缓冲区表里，`close_run` 一广播，
    连错的订阅者都会收到哨兵而收工。
    """
    cached = _buses.get(namespace)
    if cached is not None:
        return cached

    try:
        from app.clients.redis_client import get_redis, ping

        if await ping():
            bus: RunBus = RedisStreamBus(get_redis(), namespace=namespace)
            log.info("bus_ready", kind="redis-streams", ns=namespace)
        else:
            bus = InProcessBus()
            log.warning("bus_degraded", reason="redis ping failed", kind="in-process", ns=namespace)
    except Exception as exc:
        bus = InProcessBus()
        log.warning("bus_degraded", reason=str(exc), kind="in-process", ns=namespace)
    _buses[namespace] = bus
    return bus


def reset_bus(namespace: str | None = None) -> None:
    if namespace is None:
        _buses.clear()
    else:
        _buses.pop(namespace, None)


def set_bus(bus: RunBus, namespace: str = NAMESPACE_RUN) -> None:
    """测试用：直接注入。"""
    _buses[namespace] = bus
