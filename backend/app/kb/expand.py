"""从 chinese-poetry 扩充诗词语料。

**这个脚本做的是「打标」而不是「下载」。** 上游数据只有诗句、作者、词牌，
没有意象与情感标注——而这两样恰恰是检索的全部依据（见 `schema.text_for_embedding`）。
原样灌进去只会得到一堆查不到的文本，还会把 `kb-search` 的结果稀释成噪声。

所以规则是：**意象与情感都必须认得出，才收下这首。** 宁可少收，不要灌垃圾。

两个细节值得说明：

1. **标注词表里每个词都同时列出简繁两种写法。** 上游《全宋词》是简体、
   《全唐诗》是繁体，而我们自己的语料是简体、查询也是简体。给词表加繁体变体，
   比引入一张繁简转换表便宜得多，也不会把繁体原文改写成混合字形。
   **原文一律原样保存**——它是要给人读的。
2. **chunk_id 由内容哈希派生**，不依赖上游的 id 字段（有的数据集没有，
   有的会在上游更新时变化）。这样重跑天然幂等：同一首诗永远得到同一个 id，
   第二次跑会被识别为「已存在」而不是追加一份副本。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from app.logging_conf import get_logger

log = get_logger(__name__)

_BASE = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/"
_CI = "%E5%AE%8B%E8%AF%8D"       # 宋词
_TANG = "%E5%85%A8%E5%94%90%E8%AF%97"  # 全唐诗
_SANBAI = "%E5%AE%8B%E8%AF%8D%E4%B8%89%E7%99%BE%E9%A6%96"  # 宋词三百首


@dataclass(slots=True, frozen=True)
class Source:
    name: str
    urls: tuple[str, ...]
    #: 标题字段：词用 `rhythmic`（词牌），诗用 `title`
    title_field: str


# 顺序即优先级：三百首是精选且带 tags，先取它；不够再往全宋词、全唐诗扩。
# 每个数据集单文件 1000 首，取够就停，不会把整个仓库拖下来。
SOURCES: tuple[Source, ...] = (
    Source("宋词三百首", (_BASE + _SANBAI + "/%E5%AE%8B%E8%AF%8D%E4%B8%89%E7%99%BE%E9%A6%96.json",), "rhythmic"),
    Source(
        "全宋词",
        tuple(_BASE + _CI + f"/ci.song.{i}.json" for i in (0, 1000, 2000, 3000, 4000, 5000)),
        "rhythmic",
    ),
    Source(
        "全唐诗",
        tuple(_BASE + _TANG + f"/poet.tang.{i}.json" for i in (0, 1000, 2000, 3000, 4000, 5000)),
        "title",
    ),
)

#: 意象词表：规范标签 → 在原文里找的字面形式（简繁并列）。
IMAGERY_LEXICON: dict[str, tuple[str, ...]] = {
    "明月": ("明月", "月明", "月光", "蟾", "月"),
    "落花": ("落花", "花谢", "飞花", "残红", "落红", "花"),
    "风": ("风", "風"),
    "雨": ("雨",),
    "雪": ("雪",),
    "霜": ("霜",),
    "柳": ("柳",),
    "酒": ("酒", "樽", "尊", "觞", "觴"),
    "舟": ("舟", "船", "帆", "孤篷"),
    "雁": ("雁", "鴻", "鸿"),
    "灯": ("灯", "燈", "烛", "燭", "残烛"),
    "夜": ("夜", "宵"),
    "秋": ("秋",),
    "春": ("春",),
    "水": ("江水", "流水", "湘水", "水", "波"),
    "山": ("山", "峰", "嵩"),
    "云": ("云", "雲", "煙", "烟"),
    "泪": ("泪", "淚", "泣", "啼"),
    "楼": ("楼", "樓", "阑", "闌", "栏", "欄"),
    "窗": ("窗", "戶", "户", "帘", "簾"),
    "梦": ("梦", "夢"),
    "白发": ("白发", "白髮", "华发", "華髮", "鬓", "鬢"),
    "梅": ("梅",),
    "竹": ("竹",),
    "草": ("草", "苔"),
    "桥": ("桥", "橋",),
    "琴": ("琴", "弦", "絃", "笛", "箫", "簫"),
    "钟": ("钟", "鐘", "钟声", "砧"),
}

#: 情感词表。**只收能对应到心理语义的**——「愁」「恨」这类模糊词单独看
#: 没有分析价值，所以归到更具体的一类里，或者干脆不收。
EMOTION_LEXICON: dict[str, tuple[str, ...]] = {
    "思念": ("思君", "思乡", "思故", "相思", "忆", "憶", "念", "怀人", "懷人", "远望", "遠望"),
    "孤独": ("独", "獨", "孤", "寂", "无人", "無人", "空自"),
    "哀伤": ("断肠", "斷腸", "肠断", "愁", "悲", "哀", "泪", "淚", "哭", "泣", "销魂", "消魂"),
    "无常": ("无常", "無常", "浮生", "如梦", "如夢", "转瞬", "轉瞬", "几度", "幾度", "流年"),
    "怅惘": ("惆怅", "惆悵", "惘然", "黯然", "空留", "无奈", "無奈", "何堪"),
    "离愁": ("别离", "別離", "离别", "離別", "送君", "分手", "南浦", "长亭", "長亭", "归期", "歸期"),
    "倦怠": ("倦", "懒", "懶", "慵", "憔悴", "衰"),
    "旷达": ("放歌", "纵酒", "縱酒", "何妨", "一笑", "且尽", "且盡", "天地间", "天地間"),
    "回忆": ("当时", "當時", "旧时", "舊時", "昨夜", "曾记", "曾記", "记得", "記得"),
    "希冀": ("相逢", "重见", "重見", "归来", "歸來", "共", "愿", "願"),
}

#: 每首最多留几个标签。标签越多，嵌入文本越像一份标签清单、
#: 越不像一段文本，反而会把原诗的信号淹掉。
MAX_TAGS = 3

#: 诗句长度上下限。太短（「春。」）没有语义，太长（长调几页）会
#: 在嵌入时被截断，标签与原文的对应关系也就断了。
MIN_CHARS, MAX_CHARS = 14, 160

_WS = re.compile(r"\s+")


class ExpansionError(RuntimeError):
    """抓取或解析上游数据失败。"""


@dataclass(slots=True)
class ExpansionResult:
    scanned: int = 0
    accepted: int = 0
    duplicate: int = 0
    written: int = 0
    path: str = ""
    samples: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def _fetch(url: str, *, timeout: float = 25.0, attempts: int = 3) -> list[dict[str, Any]]:
    """取一个 JSON 文件，失败重试几次。

    GitHub raw 在部分网络下握手会间歇性超时，一次失败就放弃会让
    `--count 300` 这种需要连取好几个文件的命令几乎必然失败。
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last = exc
            if attempt < attempts - 1:
                time.sleep(1.5 * (attempt + 1))
            continue
        if not isinstance(payload, list):
            raise ExpansionError(f"{url} 返回的不是数组，上游结构可能变了")
        return [row for row in payload if isinstance(row, dict)]
    raise ExpansionError(f"抓取失败 {url}：{last}")


def _tag(text: str, lexicon: dict[str, tuple[str, ...]], limit: int) -> list[str]:
    """按出现位置打标，先命中先得。

    不按词表顺序（那是字典序的一种，没有意义），而按词在诗里出现的先后：
    一首词里先写到的意象通常是有意铺陈的那一个，前面三个标签配上它更贴。
    """
    hits: list[tuple[int, str]] = []
    for tag, forms in lexicon.items():
        positions = [text.find(f) for f in forms]
        found = [p for p in positions if p >= 0]
        if found:
            hits.append((min(found), tag))
    hits.sort()
    return [tag for _, tag in hits[:limit]]


def _chunk_id(author: str, title: str, text: str) -> str:
    digest = hashlib.sha1(f"{author}|{title}|{text}".encode()).hexdigest()[:12]
    return f"poem:ext:{digest}"


def _to_row(poem: dict[str, Any], title_field: str) -> dict[str, Any] | None:
    paragraphs = poem.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        return None
    text = _WS.sub("", "".join(str(p) for p in paragraphs))
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return None
    # 缺字用 □ 占位，这类残篇不该进库
    if "□" in text:
        return None

    imagery = _tag(text, IMAGERY_LEXICON, MAX_TAGS)
    emotion = _tag(text, EMOTION_LEXICON, MAX_TAGS)
    if not imagery or not emotion:
        return None

    author = str(poem.get("author") or "").strip()
    title = str(poem.get(title_field) or poem.get("title") or "").strip()
    if not author or not title:
        return None

    return {
        "chunk_id": _chunk_id(author, title, text),
        "poem_title": title,
        "author": author,
        "dynasty": "宋" if title_field == "rhythmic" else "唐",
        "imagery": imagery,
        "emotion": emotion,
        "type": "词" if title_field == "rhythmic" else "诗",
        "text": text,
    }


def _existing_ids(path: object) -> set[str]:
    from pathlib import Path

    file = Path(str(path))
    if not file.exists():
        return set()
    ids: set[str] = set()
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.add(json.loads(line)["chunk_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return ids


async def expand_poetry(
    *,
    count: int = 200,
    source: str | None = None,
    dry_run: bool = False,
    corpus_dir: object = None,
) -> ExpansionResult:
    """抓取 → 打标 → 追加到 poetry.jsonl。返回统计。"""
    from pathlib import Path

    from app.kb.loader import CORPUS_DIR

    path = Path(str(corpus_dir)) if corpus_dir else CORPUS_DIR
    target = path / "poetry.jsonl"
    result = ExpansionResult(path=str(target))
    known = _existing_ids(target)
    rows: list[dict[str, Any]] = []

    sources = SOURCES
    if source:
        # 自定义地址时不猜标题字段：只有一个文件，按「词牌优先、退回 title」处理。
        sources = (Source("自定义来源", (source,), "rhythmic"),)

    ok_urls = 0
    for src in sources:
        if len(rows) >= count:
            break
        for url in src.urls:
            if len(rows) >= count:
                break
            try:
                poems = _fetch(url)
            except ExpansionError as exc:
                # **单个文件失败不该葬送整个命令。** 已经抓到的几页是有效的，
                # 扔掉它们只会让弱网下的用户反复重试。记下来，继续下一个。
                result.failed.append(f"{src.name}：{exc}")
                continue
            ok_urls += 1
            result.scanned += len(poems)
            for poem in poems:
                row = _to_row(poem, src.title_field)
                if row is None:
                    continue
                result.accepted += 1
                if row["chunk_id"] in known:
                    result.duplicate += 1
                    continue
                known.add(row["chunk_id"])
                rows.append(row)
                if len(rows) >= count:
                    break

    # 一个文件都没取到才算真失败——此时不是「不够多」，是网络或上游结构变了。
    if ok_urls == 0:
        detail = result.failed[0] if result.failed else "没有可用的数据源"
        raise ExpansionError(f"所有数据源都取不到：{detail}")

    if dry_run:
        result.samples = [
            f"{r['dynasty']}·{r['author']}《{r['poem_title']}》"
            f"  意象={'/'.join(r['imagery'])}  情感={'/'.join(r['emotion'])}"
            f"  {r['text'][:24]}…"
            for r in rows[: max(0, 8)]
        ]
        return result

    if rows:
        # 原文原样落盘，ensure_ascii=False 让中文可读、可 diff。
        blob = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
        with target.open("a", encoding="utf-8") as handle:
            handle.write(blob)
        result.written = len(rows)
        log.info("poetry_expanded", written=len(rows), path=str(target))

    return result
