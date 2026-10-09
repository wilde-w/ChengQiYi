"""评论清洗：归一化、广告/水军识别、哈希去重。

三个判断都**只产出标记，不产出删除**。`is_ad` / `is_spam` / `is_duplicate`
连同 `filter_reason` 一起入库，前端「隐藏广告」开关只是切换显示——
关掉开关能原样看到被判定的内容。可审计是刻意的：
把一个真实用户误判成广告却让他永久消失，比留着几条广告糟得多。

因此阈值一律取保守值：宁漏勿误。真正的近重复清洗交给 M5 的余弦去重
（0.95 阈值，见 clustering_service），那里有向量，判断比这里准。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from app.douyin.base import CommentItem

FilterReason = Literal["ad", "spam", "duplicate", "empty"]

# ----------------------------------------------------------------------
# 正则：清洗
# ----------------------------------------------------------------------

# 零宽字符与 BOM。抖音评论里常见，会污染去重键。
_ZERO_WIDTH = re.compile(r"[​-‏⁠﻿­]")

_EMOJI = re.compile(
    "["
    "\U0001f000-\U0001faff"   # 各类 emoji / 符号
    "\U00002600-\U000027bf"   # 杂项符号与装饰符
    "\U0001f1e6-\U0001f1ff"   # 区域指示符（国旗）
    "⬀-⯿"
    "️"                  # 变体选择符
    "‍"                  # 零宽连接符
    "]+"
)

_URL = re.compile(r"(?:https?://|www\.)[^\s，。、）)】\]]+", re.I)
_MENTION = re.compile(r"@[\w一-鿿\-.]+")
# 归一化后只剩标点/空白 → 视为无内容
_PUNCT_ONLY = re.compile(r"^[\s\W_]*$", re.UNICODE)

# ----------------------------------------------------------------------
# 正则：广告信号
# ----------------------------------------------------------------------

# 强信号：命中一条即判广告。这些词在正常情感评论里几乎不可能出现。
_AD_STRONG = re.compile(
    r"("
    r"加\s*(?:我|一下)?\s*(?:微信|vx|v信|威信|薇信|wx|v\b|q\b|qq)"
    # 「加V」「+V」是中文互联网里最固定的引流话术。上面那条的 v\b 只在
    # V 后面跟非字母时成立，而「加V情感疗愈」这类后面直接接汉字，
    # 所以单独补一条：加/+/＋ 后面跟 v，且 v 后面不是拉丁字母。
    r"|[加+＋]\s*[vV](?![a-zA-Z])"
    r"|(?:微信|vx|wx|qq)\s*[:：]?\s*[a-z0-9_-]{5,}"
    r"|私\s*(?:我|聊|信)"
    r"|(?:看|点)\s*(?:我)?\s*主页"
    r"|(?:点|戳)\s*(?:击)?\s*(?:下方)?\s*(?:链接|小黄车|购物车|橱窗)"
    r"|(?:领|抢)\s*(?:取|购)?\s*(?:优惠券|福利|红包|免费)"
    r"|(?:加|进)\s*(?:群|粉丝群)"
    r"|(?:刷单|代购|招代理|招兼职|日结|包邮|现货|批发|一手货源)"
    # 引流漏斗：把用户导到主页/简介再成交。不带「看/点」前缀也算。
    r"|主页\s*(?:简介|签名)?\s*(?:自取|有链接|看简介|领取)"
    # 收入承诺。刻意要求带 w/万/k 量级——「我月入5000」是真实评论，不该误杀。
    r"|月\s*入\s*\d+\s*[wW万kK]"
    r"|(?:0|零)\s*基础\s*(?:转行|学|做|也能|月入)"
    r"|名额\s*(?:只剩|有限|不多)"
    r"|1[3-9]\d{9}"                        # 手机号
    r"|[qQ]\s*[qQ]?\s*[:：]?\s*\d{6,}"     # QQ 号
    r")",
    re.I,
)

# 弱信号：需累计到 AD_KEYWORD_THRESHOLD 才判广告。
# 刻意不收「优惠 / 下单 / 同款 / 购买」——这些在真实用户的追问里太常见
# （「这个有优惠吗我想下单」是顾客，不是广告），收进来必然误杀。
# 留下的是几乎只出现在带货话术里的词。
_AD_WEAK = re.compile(
    r"(优惠券|限时|秒杀|福利价|橱窗|小黄车|一手货源|招代理|返还现金|免费送|点击购买|链接在下方)"
)

# 长数字串多为联系方式。但纯数字的评论（1111111111）是灌水不是广告，
# 由 detect_spam 处理，这里排除掉。
_LONG_DIGITS = re.compile(r"\d{8,}")

AD_KEYWORD_THRESHOLD = 2

_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_REPEAT_RUN = re.compile(r"(.)\1{9,}")    # 同一字符连续 10 次以上
REPEAT_MIN_RUN = 10


@dataclass(slots=True)
class CleanedComment:
    """清洗后的评论。字段与 ORM 的 Comment 模型一一对应。"""

    comment_id: str
    text: str                 # 归一化后的显示文本
    dedup_key: str            # 用于哈希去重的指纹
    author_name: str | None = None
    author_id: str | None = None
    like_count: int = 0
    reply_count: int = 0
    publish_time: object | None = None
    is_ad: bool = False
    is_spam: bool = False
    is_duplicate: bool = False
    filter_reason: str | None = None

    @property
    def is_visible_by_default(self) -> bool:
        return not (self.is_ad or self.is_spam or self.is_duplicate)

    def to_payload(self) -> dict:
        return {
            "comment_id": self.comment_id,
            "text": self.text,
            "author_name": self.author_name,
            "like_count": self.like_count,
            "reply_count": self.reply_count,
            "publish_time": self.publish_time.isoformat() if self.publish_time else None,
            "is_ad": self.is_ad,
            "is_spam": self.is_spam,
            "is_duplicate": self.is_duplicate,
            "filter_reason": self.filter_reason,
        }


@dataclass(slots=True)
class CleanStats:
    total: int = 0
    kept: int = 0
    ads: int = 0
    spam: int = 0
    duplicates: int = 0
    empty: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "kept": self.kept,
            "ads": self.ads,
            "spam": self.spam,
            "duplicates": self.duplicates,
            "empty": self.empty,
            "reasons": dict(self.reasons),
        }


# ----------------------------------------------------------------------
# 归一化
# ----------------------------------------------------------------------


def normalize(text: str) -> str:
    """显示用归一化。NFKC 把全角转半角（'！' → '!'），保留 emoji 与标点。

    NFKC 是必须的：抖音评论里全角/半角混用极常见，
    「太好看了！」和「太好看了!」不归一化就是两条不同的评论。
    """
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", str(text))
    out = _ZERO_WIDTH.sub("", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def dedup_key(text: str) -> str:
    """去重指纹：剥离 @、链接、emoji、标点与全部空白后取 blake2b。

    剥离是刻意的——「太好看了😭😭」和「太好看了！！」表达同一件事，
    归一化后应当是同一条。用 blake2b 而非内置 hash：后者跨进程随机化，
    会让「同一次运行内去重」和「重启后复现」结果不一致。
    """
    base = normalize(text)
    base = _URL.sub("", base)
    base = _MENTION.sub("", base)
    base = _EMOJI.sub("", base)
    base = re.sub(r"[\s\W_]+", "", base, flags=re.UNICODE)
    return hashlib.blake2b(base.encode("utf-8"), digest_size=16).hexdigest()


def _has_visible_content(text: str) -> bool:
    """剥离 emoji 后是否还有可读文字。

    刻意**不**剥离 URL：一条裸链接是有内容的（它是广告，交给 detect_ad），
    不是「什么都没说」。
    """
    stripped = _EMOJI.sub("", normalize(text))
    return bool(stripped.strip()) and not _PUNCT_ONLY.match(stripped)


# ----------------------------------------------------------------------
# 单条判定
# ----------------------------------------------------------------------


def detect_ad(text: str) -> bool:
    """广告判定。强信号一条即中；弱信号需累计。

    优先判 spam：纯数字串是灌水不是广告，不属于这里。
    """
    body = normalize(text)
    if not body:
        return False
    if _URL.search(body):
        return True
    if _AD_STRONG.search(body):
        return True
    # 纯数字（或数字+标点）的长串是刷屏，交给 detect_spam
    if not re.fullmatch(r"[\d\s\W_]+", body) and _LONG_DIGITS.search(body):
        return True
    return len(set(_AD_WEAK.findall(body))) >= AD_KEYWORD_THRESHOLD


def detect_spam(text: str) -> bool:
    """水军/灌水判定。只处理「显然没有语义」的情况。

    刻意不处理两件事：
    - 超短评论（「好」「+1」）。它们是低信息量，不是恶意，
      交给 M5 让 HDBSCAN 归入噪声簇即可——那是信号判断，不是价值判断。
    - 中文叠字。「哈哈哈哈哈哈」「呜呜呜呜」是最常见的真实情绪表达，
      按字符重复一律过滤会误杀一大批高赞评论。
      只有非中文（字母/数字）的长重复才是灌水——那才是机器人和刷屏的形态。
    """
    body = normalize(text)
    if not body:
        return True
    if not _has_visible_content(body):
        return True                       # 纯 emoji / 纯标点
    for match in _REPEAT_RUN.finditer(body):
        if not _CJK.match(match.group(1)):
            return True                   # aaaaaaaa / 1111111111
    return False


# ----------------------------------------------------------------------
# 批量清洗
# ----------------------------------------------------------------------


def clean_comments(
    items: Iterable[CommentItem],
    *,
    drop_duplicates: bool = True,
) -> tuple[list[CleanedComment], CleanStats]:
    """清洗一批评论。返回 (全部评论含标记, 统计)。

    返回**全部**评论而非只返回保留的：被过滤的也要入库，
    否则前端「显示被过滤内容」开关就没有东西可显示。
    """
    # 第一趟：逐条打标记。去重单独放第二趟——重复判定要看全局，
    # 混在一起会让「后出现的重复项比先出现的更热」这种替换逻辑
    # 与统计计数纠缠不清（早期版本就在这里算错过 kept/duplicates）。
    cleaned: list[CleanedComment] = []
    for raw in items:
        text = normalize(raw.text)
        item = CleanedComment(
            comment_id=raw.comment_id,
            text=text,
            dedup_key=dedup_key(text),
            author_name=raw.author_name,
            author_id=raw.author_id,
            like_count=raw.like_count,
            reply_count=raw.reply_count,
            publish_time=raw.publish_time,
        )

        if not _has_visible_content(text):
            item.is_spam = True
            item.filter_reason = "empty"
        elif detect_ad(text):
            item.is_ad = True
            item.filter_reason = "ad"
        elif detect_spam(text):
            item.is_spam = True
            item.filter_reason = "spam"

        cleaned.append(item)

    if drop_duplicates:
        _mark_duplicates(cleaned)

    return cleaned, tally_flags(cleaned)


def _mark_duplicates(cleaned: list[CleanedComment]) -> None:
    """就地标记重复项：同一指纹只保留点赞最高的那条。

    只考虑已经干净的评论——广告文案高度雷同，若不排除，
    一条广告会把后面所有同文案的**真实评论**一起拖成重复。
    """
    best: dict[str, int] = {}
    for idx, item in enumerate(cleaned):
        if item.is_ad or item.is_spam:
            continue
        key = item.dedup_key
        prior = best.get(key)
        if prior is None:
            best[key] = idx
            continue
        if item.like_count > cleaned[prior].like_count:
            cleaned[prior].is_duplicate = True
            cleaned[prior].filter_reason = "duplicate"
            best[key] = idx
        else:
            item.is_duplicate = True
            item.filter_reason = "duplicate"


def tally_flags(items: Iterable[Any]) -> CleanStats:
    """按标记重新统计。

    输入的判定顺序是**有意的**（重复 > 广告 > 灌水 > 保留），且与
    `_mark_duplicates` 里「先排除广告再判重」的顺序配套：同一条评论只计一次，
    归到最高优先级的那一类里。

    鸭子类型而非只收 `CleanedComment`：ORM 的 Comment 行有同样的四个字段，
    快照接口用它从库里重建出与流式期间**完全一致**的统计数字。
    （快照与流报出不同的数字，是用户对系统失去信任的最快方式。）
    """
    stats = CleanStats()
    for item in items:
        stats.total += 1
        if item.is_duplicate:
            stats.duplicates += 1
        elif item.is_ad:
            stats.ads += 1
        elif item.is_spam:
            if item.filter_reason == "empty":
                stats.empty += 1
            else:
                stats.spam += 1
        else:
            stats.kept += 1
        if item.filter_reason:
            stats.reasons[item.filter_reason] = stats.reasons.get(item.filter_reason, 0) + 1
    return stats


def visible_comments(items: Iterable[CleanedComment]) -> list[CleanedComment]:
    """默认展示的评论：去掉广告、灌水、重复。"""
    return [c for c in items if c.is_visible_by_default]
