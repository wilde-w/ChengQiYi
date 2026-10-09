"""看最近几次会话的正文开头：模型有没有把「我先去查查」这类前言印进正文。

只读，不改任何东西。

跑法：

    cd backend && .venv/Scripts/python.exe scripts/story_head.py

（要在**建脚本之前**把 `backend/` 塞进 `sys.path`：`python scripts/xxx.py`
时 sys.path[0] 是脚本自己所在的 `scripts/`，不是 `backend/`。）
"""
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from app.db.session import session_scope
from app.models.agent import AgentSession


async def main() -> None:
    async with session_scope() as db:
        rows = (
            await db.execute(
                select(AgentSession).order_by(AgentSession.created_at.desc()).limit(5)
            )
        ).scalars().all()

    for row in rows:
        story = row.story or ""
        head = story[:60].replace("\n", " ")
        # 正文开头 200 字里的 ASCII 字母：中文故事不该有连续的英文单词
        latin = re.findall(r"[A-Za-z]{2,}", story[:200])
        print(f"--- {row.id}  turn={row.turn} {len(story)} 字")
        print(f"    开头：{head}")
        print(f"    前 200 字里的英文词：{latin or '无'}")
        print(f"    末 40 字：{story[-40:].replace(chr(10), ' ')}")


asyncio.run(main())
