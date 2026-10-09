"""测试进程的公共配置。

**在 import 期就把 mock 钉死。** 开发机的 `.env` 里配了真的
`DEEPSEEK_API_KEY` 时，`MOCK_MODE=auto` 会让打标与嵌入这些测试真的发出去
网络请求——测试于是变成「今天有没有额度和网速」的函数，而且会往用户账上
扣钱。测试跑在哪条路径上，不该取决于跑测试的人 `.env` 里恰好有什么。

延迟一律设 0：mock 的 900ms 是给演示看的节奏，在测试里只是白等。
（`MOCK_MODE=never` 的配置错误路径由 `test_providers.py` 自己构造
`Settings` 测，不走这里。）
"""

from __future__ import annotations

import os

os.environ["MOCK_MODE"] = "always"
os.environ["MOCK_LLM_LATENCY_MS"] = "0"
os.environ["MOCK_DOUYIN_LATENCY_MS"] = "0"
os.environ["MOCK_MIN_STEP_DELAY_MS"] = "0"
