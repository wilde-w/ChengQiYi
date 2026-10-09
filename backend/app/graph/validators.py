"""引用标记的抽取、剥离与覆盖率。**n6 与 n7 共用。**

模型在正文里写 `[[ev:<chunk_id>]]` 来声明「这句话有证据」。但提示词约束从来
不是保证——它会写错 id、写一个池子里没有的 id、把括号写歪。这个模块的职责
就是在落库前把这些标记**收干净**：合法的规范化成脚注记号，非法的连痕迹一起
剥掉。库里永远不存悬空引用，因为库里存的根本不是模型的原话。

## 为什么记号和正文要分开存

`content_md` 里存的是规范化后的 `[^1]`，`citations` 里存 `{evidence_id,
marker, text}`。两个理由：
  1. 模型可能写 `[[EV: x ]]`、`[[ev:x]]`、`[[ ev : x ]]`——三种写法在读者
     眼里是一回事。规范化成一种，前端只需要认识一种。
  2. 剥离掉的畸形标记会留下一个洞（「这句话后头本来有个尾巴」），而规范化
     则保证只要正文里出现 `[^n]`，就一定能翻到第 n 条证据。**可点性由构造
     保证，不靠约定。**
"""

from __future__ import annotations

import re
from collections.abc import Container, Iterable, Mapping, Sequence
from typing import Any

#: 形态正确的标记，捕获组是证据 id。
MARKER_RE = re.compile(r"\[\s*\[\s*ev\s*:\s*([^\s\[\]]+?)\s*\]\s*\]")

#: 一切「看着像标记」的东西，**含畸形**（id 里有空格、大小写混写）。
#: 剥离用它而不是用 `MARKER_RE`：只剥合法标记的话，畸形标记会原样落进
#: 正文，用户在洞察正文里读到 `[[ev:` 这样一串机器残渣。
#:
#: 捕获组故意宽松到 `[^\]]*`——收得紧就只能抓合法的，而不合法的那些
#: 恰恰是这里最需要看见的。判它合不合法是 `pool` 的事，不是正则的事。
ATTEMPT_RE = re.compile(r"\[\s*\[\s*ev\s*:\s*([^\]]*?)\s*\]\s*\]", re.IGNORECASE)

#: 收尾用的：`ATTEMPT_RE` 要求两个闭合括号，而模型偶尔只写一个
#: （`[[ev:xxx]`）。那一条会整段活下来，用户就在正文里读到 `[[ev:`。
#: 逮的是「`[[ev:` 之后到第一个 `]` 或行尾」——停在 `]` 而不是行尾，
#: 是为了不把标记后面那半句正常的话一起吃掉。
LEFTOVER_RE = re.compile(r"\[\s*\[\s*ev\s*:[^\]]*\]?", re.IGNORECASE)

#: 规范化后的记号模板。用脚注语法而不是自造符号，因为它在纯文本里
#: 也不刺眼——正文被复制到别处时，`[^3]` 至少还是一句人话。
MARKER_TEMPLATE = "[^{n}]"

#: 正文里最多留多少枚记号。防的是模型把证据列表整个抄进正文——
#: 那已经不是在引用了，是在贴公告。
MAX_CITATIONS = 40


def extract_markers(text: str) -> list[str]:
    """按出现顺序取出所有**形态正确**的标记里的 id。不去重。

    不去重是为了让它和 `coverage()` 的分母口径一致：同一个 id 出现三次
    就是三枚 chip，读者要点三次。
    """
    if not text:
        return []
    return [m.group(1).strip() for m in MARKER_RE.finditer(text)]


def sanitize(
    text: str,
    pool: Container[str] | Mapping[str, str],
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """把正文里的标记收干净。

    `pool` 是**合法 id 的集合**（也可以是 {别名: 真 id} 的映射，为短别名
    方案预留——现在两处都只传集合）。凡是落到池子外的 id 一律丢弃并计数，
    调用方据此决定要不要重生成。

    返回 `(clean, citations, report)`：
      - `clean`  —— 正文，所有标记（含畸形）都已不在，合法的换成了 `[^n]`
      - `citations` —— `[{evidence_id, marker, text}]`，`text` 留给调用方
        补出处；这里不认识证据对象，只认 id
      - `report` —— `{total, valid, invalid, dropped_ids}`，`coverage()` 的输入
    """
    if not text:
        return "", [], _report(0, 0, [])

    citations: list[dict[str, Any]] = []
    dropped: list[str] = []
    total = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal total
        total += 1
        raw = match.group(1)
        # 畸形标记同样计入分母：读者看到的是一处「本该有引用」的位置，
        # 它没落到证据上，就拉低覆盖率。只统计合法标记等于把错误
        # 从分母里摘出去，覆盖率会虚高。
        resolved = _resolve(raw, pool)
        if resolved is None or len(citations) >= MAX_CITATIONS:
            dropped.append(raw)
            return ""
        citations.append(
            {"evidence_id": resolved, "marker": MARKER_TEMPLATE.format(n=len(citations) + 1)}
        )
        return citations[-1]["marker"]

    clean = ATTEMPT_RE.sub(_replace, text)

    # 第二遍收拾没闭合的那些。它们同样计入分母：位置上确实出现过一次引用，
    # 只是它连自己的 id 都没写完——这种位置最该拉低覆盖率。
    leftover = LEFTOVER_RE.findall(clean)
    if leftover:
        clean = LEFTOVER_RE.sub("", clean)
        total += len(leftover)
        dropped.extend(part.strip() for part in leftover)

    return clean, citations, _report(total, len(citations), dropped)


def coverage(report: Mapping[str, Any]) -> float:
    """`valid / total`，**每次出现计一次**；一次都没出现时返回 0.0。

    为什么是「出现次数」而不是段落数或唯一 id 数——读者看到的是一枚枚
    chip，这个数字的含义就是「正文里任意一枚 chip 都能落到证据上的概率」。
    另外两种口径在黄金 mock 路径上分别只有 0.32 与 0.46，都低于重试阈值
    0.6，会让每次演示都触发一次注定失败的重试，把健康结果报成有问题；
    而同一条数据上本口径是 1.0。

    `total == 0` 返回 0.0 而不是 1.0：**空集不算满分**。至于这 0.0 是
    「池子本来就是空的」还是「池子有货但模型没写标记」，由调用方判断——
    两者该有不同的处置，而这个函数看不到池子。
    """
    total = int(report.get("total") or 0)
    if total <= 0:
        return 0.0
    return round(int(report.get("valid") or 0) / total, 4)


def filter_ids(
    ids: Any,
    pool: Container[str] | Mapping[str, str],
    *,
    limit: int | None = None,
) -> tuple[list[str], list[str]]:
    """把模型给的一串 id 收进池子里。返回 `(kept, dropped)`。

    两份都要返回：`kept` 进产物，`dropped` 进日志与 `payload`。只丢不报的话，
    「模型抄错 chunk_id」这件事会以「这次推理只引用了 1 条证据」的样子
    呈现出来——数字变小了，但没人知道为什么。
    """
    kept: list[str] = []
    dropped: list[str] = []
    for item in _as_list(ids):
        resolved = _resolve(item, pool)
        if resolved is None:
            dropped.append(item)
            continue
        if resolved in kept:
            continue
        kept.append(resolved)
        if limit is not None and len(kept) >= limit:
            break
    return kept, dropped


# ----------------------------------------------------------------------


def _resolve(raw: str, pool: Container[str] | Mapping[str, str]) -> str | None:
    token = raw.strip()
    if not token:
        return None
    if isinstance(pool, Mapping):
        resolved = pool.get(token)
        if resolved is not None and not isinstance(resolved, str):
            # 映射的语义是「别名 → 真 id」。传一份 `{id: 证据对象}` 进来
            # 会让这个函数把 dict 当成 id 返回，错误一路漂到 `cid in 集合`
            # 才炸，栈里已经看不出是谁传错了。在这里就拦住。
            raise TypeError(f"pool 映射的值必须是 id 字符串，收到 {type(resolved).__name__}")
        return resolved
    return token if token in pool else None


def _as_list(value: Any) -> list[str]:
    """只认「字符串或字符串序列」。**刻意不认任意可迭代对象**：
    dict 是可迭代的，把它的键当 id 收进来会把 `{"a": 1}` 这种调用方的
    笔误伪装成「模型给了个查不到的 id」，于是错误在日志里长得像模型的问题。
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [v for v in value if isinstance(v, str)]
    return []


def _report(total: int, valid: int, dropped: list[str]) -> dict[str, Any]:
    return {
        "total": total,
        "valid": valid,
        "invalid": total - valid,
        # 只留去重后的前若干个：这个字段是给人看的诊断线索，
        # 不是用来还原模型全部错误的（畸形标记可能重复几百次）。
        "dropped_ids": list(dict.fromkeys(dropped))[:10],
    }


__all__ = [
    "ATTEMPT_RE",
    "MARKER_RE",
    "MARKER_TEMPLATE",
    "coverage",
    "extract_markers",
    "filter_ids",
    "sanitize",
]
