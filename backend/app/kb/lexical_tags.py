"""不依赖模型的标签抽取。

两处用它：mock 模式下的 `tag_chunks`，以及真实 LLM 打标失败时的降级路径。
**必须是同一份实现**——降级路径只有日常被跑到才会保持可用，而 mock 每次
演示都在跑它。

设计上只有一条铁律：**所有标签都必须是原文的子串。**

这一条同时买到三样东西：

1. `text_for_embedding()` 里的「关键词：此在、时间性」与用户查询「此在」
   在 n-gram 桶里必然共享片段——mock 的向量相似度是真实抬升，不是碰运气；
2. mock 的产出天然通过打标器的「必须是子串」校验，两条路径不会分叉；
3. 标签永远可解释，用户能在原文里指出它出现在哪。

**词表按语境分开，不复用 `expand.py` 的。** 那套是给古典诗词写的：它用
`text.find("月")` 匹配单字，在散文里会从「岁月」「月色」误命中，「水」会从
「水平」「水货」误命中。误命中比落空危害大得多——错误标签会持续污染检索，
而且没人会去查。所以散文版的意象词表**只认多字形**，单字一律不认。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.constants import Library

#: 一个块最多留几个关键词。标签越多，嵌入文本越像标签清单而非文本，
#: 反而把原文信号淹掉。
MAX_KEYWORDS = 6
MAX_IMAGERY = 3
MAX_EMOTION = 3

#: n-gram 长度。1-gram 在中文里几乎全是虚词或单字词，噪声远大于信号。
MIN_GRAM, MAX_GRAM = 2, 4


@dataclass(slots=True)
class ChunkTags:
    """一个块的全部标签。字段与 LLM 返回的 JSON 一一对应。"""

    quote: str = ""
    concept: str = ""
    keywords: list[str] = field(default_factory=list)
    imagery: list[str] = field(default_factory=list)
    emotion: list[str] = field(default_factory=list)
    type: str = ""
    #: "llm" 或 "lexical"——降级产物必须能和陈旧产物区分开。
    source: str = "lexical"

    def is_empty_for(self, library: Library) -> bool:
        """是否该整块丢弃。

        文学/诗词的嵌入模板只渲染意象与情感，抽象文本产不出稳定意象时
        嵌进去的是一段没有检索钩子的裸文本——它占一次嵌入、一个检索候选位，
        却永远不会被命中。沿用 `expand.py` 已写下的教条：宁可少收，不要灌垃圾。
        """
        if library is Library.PSYCHOLOGY:
            return not self.concept
        return not (self.imagery or self.emotion)


# ----------------------------------------------------------------------
# 关键词抽取
# ----------------------------------------------------------------------

#: 切分成「语段」的标点。**只在语段内取 n-gram**——跨标点的组合
#: （「…存在，时间…」→「存在时间」）在原文里根本不是一个词。
_RUN_SPLIT = re.compile(r"[。！？；…，、,.!?;:：\s「」『』“”‘’（）()《》〈〉【】\[\]—–\-]+")

#: **首字**虚词表。中文的碎片几乎总是以虚词开头（「的存在方式」「和与此」
#: 「所以就」），所以这一条卡得严，误杀率很低。
_LEAD_BAD = frozenset(
    "的了着过是在和与而也就都还又很把被从对为以之其所以因故乃且夫盖若则虽但或及于乎焉矣兮者们个每该"
    "一二两这那几并"
)

#: **尾字**虚词表。刻意比 `_LEAD_BAD` 小得多，因为汉字在词尾的分布完全不同：
#: 「存在」「此在」「现在」「行为」「认为」「面对」「温和」全都是正常的词，
#: 而把「在/为/对/和」一律当尾字碎片否决，等于先把最该选中的那批词杀掉，
#: 剩下能通过的就只有跨词界的垃圾。**尾字只留纯语法助词。**
_TAIL_BAD = frozenset("的了着过是而也就都还又很把被之其所以因故乃且夫盖若则虽但或及于乎焉矣兮个每该")

#: 整词停用词。两字以上，边界判据管不到它们。
_STOP_WORDS = frozenset(
    {
        "这个", "那个", "我们", "你们", "他们", "它们", "什么", "怎么", "因为",
        "所以", "但是", "如果", "已经", "可以", "这样", "那样", "一个", "一种",
        "一些", "就是", "不是", "没有", "这些", "那些", "自己", "以及", "并且",
        "或者", "而且", "然后", "于是", "对于", "关于", "由于", "通过", "根据",
        "时候", "地方", "东西", "问题", "情况", "方面", "一定的", "某种",
    }
)

#: 名词后缀加成。中文的抽象概念几乎都带这些尾巴，命中一个就基本可以
#: 断定这是个术语而不是半句话。
_NOUN_SUFFIXES = (
    "主义", "论", "学", "性", "化", "观", "关系", "存在", "自由", "时间",
    "意义", "结构", "现象", "经验", "意识", "概念", "价值", "主体", "客体",
    "理性", "意志", "情感", "认知", "精神", "世界", "真理", "本质", "语言",
    "权力", "制度", "实践", "状态", "过程", "能力", "理论", "模型", "机制",
)


def _is_candidate(gram: str) -> bool:
    if gram[0] in _LEAD_BAD or gram[-1] in _TAIL_BAD:
        return False
    return gram not in _STOP_WORDS


def _grams(text: str) -> list[str]:
    """语段内的 2–4-gram 候选。"""
    out: list[str] = []
    for run in _RUN_SPLIT.split(text):
        n = len(run)
        if n < MIN_GRAM:
            continue
        for size in range(MIN_GRAM, min(MAX_GRAM, n) + 1):
            for start in range(n - size + 1):
                gram = run[start : start + size]
                if _is_candidate(gram):
                    out.append(gram)
    return out


def build_frequency_table(text: str) -> Counter[str]:
    """整篇文档的 n-gram 频次。

    **必须在文档级统计，不能只看单块。** 一个 350 字的块里几乎每个 n-gram
    都只出现一次，长度加成于是成了唯一判据，而越长的 n-gram 越容易跨词界
    ——选出来的全是「一种对存」「够追问存」这种碎片，真正反复出现的
    「此在」「时间性」反而落选。文档级频次把 200 次和 1 次的差距拉到两个
    数量级，碎片自动沉底。
    """
    return Counter(_grams(text))


def _score(gram: str, freq: int) -> float:
    """log(频次) × 长度加成 × 名词后缀加成。

    频次取对数是为了**压缩**：不然最高频的两字词会垄断全部名额，而
    「存在论」这种更长更具体的术语永远选不上。取完对数，长度与后缀
    才有机会把长术语顶上来（`存在` 12.2 分 vs `存在论` 22.9 分）。
    """
    value = math.log(freq + 1) * (len(gram) ** 1.2)
    if gram.endswith(_NOUN_SUFFIXES):
        value *= 1.8
    return value


def extract_keyphrases(
    text: str, k: int = MAX_KEYWORDS, *, freq: Counter[str] | None = None
) -> list[str]:
    """从正文里抽出最多 k 个关键词，全部是原文子串。

    `freq` 传整篇文档的频次表（见 `build_frequency_table`）；不传就退回
    块内统计——那在短文本上还行，在长文上会退化成一堆碎片。
    """
    counts = Counter(_grams(text))
    weights = freq or counts

    ranked = sorted(
        counts,
        key=lambda g: (-_score(g, weights.get(g, counts[g])), -len(g), g),
    )

    picked: list[str] = []
    for gram in ranked:
        if len(picked) >= k:
            break
        # 只做包含去重，不做「字符重叠率」去重：后者会把「存在」和「在此」
        # 判成同一个（共享一个「在」字），而它们是两个不同的概念。
        if any(gram in chosen or chosen in gram for chosen in picked):
            continue
        picked.append(gram)
    return picked


def keyphrases_across(texts: Sequence[str], k: int = 5, *, min_docs: int = 2) -> list[str]:
    """一组**短文本**共有的关键词。判据是「出现在至少 `min_docs` 条里」。

    `extract_keyphrases` 服务的是 350 字的书页，它的判据是「这个词在这一大段
    里出现了几次」。换成一组三条评论就失效了：文档级频次退化成块内频次，
    长度加成变成唯一的排序依据，于是选出「天都在加」「己像个工」这种跨句
    碎片——它们连任何一条评论的子串都不是。

    短文本的正确判据是**跨条数**：反复出现的是主题，只出现一次的是一句
    具体的抱怨。所以这里的频次单位是「几条评论里出现过」，不是「出现几次」。

    一条都凑不满 `min_docs` 时返回**空列表**，不退回按长度排序。卡片上没有
    关键词是可以接受的，关键词是「己像个工」则会让整张卡看着像坏了。
    """
    if not texts:
        return []
    doc_freq: Counter[str] = Counter()
    for text in texts:
        # 用 set 而不是逐次累加：同一条评论里说三遍「加班」仍然只算一条，
        # 否则一个复读的人能凭一己之力定义整簇的关键词。
        doc_freq.update(set(_grams(text)))

    ranked = sorted(
        (g for g, n in doc_freq.items() if n >= min_docs),
        key=lambda g: (-_score(g, doc_freq[g]), -len(g), g),
    )

    picked: list[str] = []
    for gram in ranked:
        if len(picked) >= k:
            break
        # 与 extract_keyphrases 同一套去重理由：只做包含去重。
        if any(gram in chosen or chosen in gram for chosen in picked):
            continue
        picked.append(gram)
    return picked


# ----------------------------------------------------------------------
# 意象
# ----------------------------------------------------------------------

#: 散文/哲学意象词表：规范标签 → 原文里的字面形式。**全部是多字形**。
#: 收录的是思辨文本反复借用的那些意象（深渊、道路、镜子、大地…），
#: 不是自然景物清单——后者在散文里几乎全是字面义，没有检索价值。
PROSE_IMAGERY_LEXICON: dict[str, tuple[str, ...]] = {
    "深渊": ("深渊", "悬崖", "无底"),
    "道路": ("道路", "路途", "路径", "旅途", "歧路"),
    "镜子": ("镜子", "镜中", "明镜", "倒影"),
    "大地": ("大地", "土地", "泥土"),
    "星辰": ("星辰", "星空", "群星"),
    "沉默": ("沉默", "缄默", "无言", "默然"),
    "影子": ("影子", "阴影", "投影"),
    "火焰": ("火焰", "火光", "燃烧", "灰烬"),
    "河流": ("河流", "河水", "长河", "洪流"),
    "海洋": ("海洋", "大海", "海水", "海面"),
    "山峰": ("山峰", "高山", "群山", "山顶"),
    "天空": ("天空", "苍穹", "天际"),
    "石头": ("石头", "岩石", "巨石", "石块"),
    "桥梁": ("桥梁", "渡口"),
    "夜晚": ("夜晚", "深夜", "黑夜", "夜里", "黄昏"),
    "黎明": ("黎明", "破晓", "天亮", "清晨"),
    "树木": ("树木", "大树", "树枝", "树根", "森林"),
    "花朵": ("花朵", "鲜花", "落花", "花开"),
    "雨水": ("雨水", "大雨", "暴雨", "雨滴"),
    "雪": ("雪花", "大雪", "积雪", "冰雪"),
    "风": ("风声", "狂风", "微风", "风吹"),
    "光": ("光芒", "光线", "光明", "光亮", "阳光"),
    "水域": ("水流", "水面", "溺水", "沉入"),
    "镜子碎片": ("碎片", "裂缝", "裂隙"),
    "面容": ("面容", "面孔", "脸庞", "凝视"),
    "身体": ("身体", "肉身", "躯体", "血肉"),
}

# ----------------------------------------------------------------------
# 情绪
# ----------------------------------------------------------------------

#: 散文情绪触发词：规范标签 → 原文里的字面形式。**全部是多字形**
#: （「累」会从「积累」「拖累」误命中）。
PROSE_EMOTION_TRIGGERS: dict[str, tuple[str, ...]] = {
    "哀伤": ("悲伤", "悲哀", "哀伤", "哀恸", "痛苦", "痛哭", "哭泣", "流泪", "眼泪", "伤心", "难过", "哀悼"),
    "愧疚": ("愧疚", "内疚", "悔恨", "后悔", "罪责", "罪恶感", "自责", "亏欠"),
    "共鸣": ("共鸣", "感同身受", "也是如此", "同样感到"),
    "自嘲": ("自嘲", "自讽", "反讽自己"),
    "倦怠": ("倦怠", "疲惫", "疲劳", "厌倦", "精疲力竭", "无力", "消耗殆尽"),
    "不甘": ("不甘", "不服", "愤懑", "怨恨", "委屈", "凭什么"),
    "孤独": ("孤独", "孤寂", "寂寞", "孤单", "孤立"),
    "温暖": ("温暖", "温情", "温柔", "慰藉", "安慰", "善意", "亲切"),
    "怀念": ("怀念", "想念", "追忆", "魂牵梦萦"),
    "焦虑": ("焦虑", "不安", "担忧", "忧心", "惶惑", "紧张"),
    "愤怒": ("愤怒", "怒气", "怒火", "愤慨", "憎恨", "暴怒"),
    "悲悯": ("悲悯", "怜悯", "同情", "慈悲", "恻隐"),
    "无常": ("无常", "生灭", "转瞬即逝", "消逝", "瞬息"),
    "豁达": ("豁达", "旷达", "从容", "泰然", "坦荡"),
    "虚无": ("虚无", "空虚", "空洞", "无意义"),
    "释然": ("释然", "解脱", "放下", "释怀", "超脱"),
    "自怜": ("自怜", "可怜", "顾影自怜"),
    "思念": ("思念", "相思", "牵挂", "惦记"),
    "怅惘": ("怅惘", "惆怅", "迷茫", "惘然", "若有所失", "失落"),
    "回忆": ("回忆", "往事", "曾经", "当年", "记忆"),
    "希冀": ("希望", "期待", "盼望", "憧憬", "期盼"),
    "离愁": ("离别", "分别", "离愁", "送别", "告别", "诀别"),
    "恐惧": ("恐惧", "害怕", "惊恐", "畏惧", "恐怖"),
    "荒诞": ("荒诞", "荒谬", "荒唐"),
    "疏离": ("疏离", "隔阂", "格格不入", "异化"),
}


def _closed_emotions() -> tuple[str, ...]:
    """情绪闭集。

    **查询侧那张簇情绪词表按引用取用，不复制。** 复制一份一定会漂移，
    而漂移的表现是「检索命中率莫名偏低」——最难查的那类 bug。这里直接
    import `mock_llm.EMOTION_LEXICON`，它是目前仓库里唯一一份「查询侧会
    产出哪些情绪标签」的可执行规格。

    **这个保证只在 mock 模式下是完整的。** 接真实 LLM 之后，心理侧的情绪
    标签变成自由文本，届时需要在检索层做归一（M6 的事）。在那之前，
    这至少保证了两件事：库里不会长出重复的 Emotion 节点，以及导入的
    文学/哲学内容与已有语料用同一套词。

    语料侧那份是**现有三个库已经在用的标签**，由
    `tests/test_lexical_tags.py::test_vocab_covers_shipped_corpus` 守着——
    语料一旦用了新词而这里没同步，测试会红，而不是悄悄裂成两个节点。
    """
    from app.providers.mock_llm import EMOTION_LEXICON

    return (
        *EMOTION_LEXICON,
        # 现有语料里在用的、查询侧词表没有的（多来自古典文学）
        "悲悯", "无常", "豁达", "虚无", "释然", "自怜", "思念", "怅惘",
        "回忆", "希冀", "离愁", "恐惧",
        # 哲学散文的底色：存在主义那一支的核心词。
        # 只补两个——补多了就是在给自己造一批永远查不到的图节点。
        "荒诞", "疏离",
        # 「平静」不在上面任何一处，但 `mock_llm._extract_psych` 把它当作
        # 「这一簇读不出任何情绪」的兜底值往外发。不在闭集里就会被校验器
        # 悄悄丢掉——表现是某一簇的 emotion_tags 空着，而分布图上那一块
        # 什么都没有，看不出是被过滤了还是本来就没有。
        # 加进来还有个副作用是好的：闭集因此能表达「没有强烈情绪」，
        # 而在此之前那只能用空数组表达，两者在图上长得一样。
        "平静",
    )


EMOTION_VOCAB: tuple[str, ...] = _closed_emotions()


# ----------------------------------------------------------------------
# 组装
# ----------------------------------------------------------------------


def _first_sentence(text: str, limit: int = 40) -> str:
    """取作 `quote` 锚点用。它必须是正文子串——校验器会查这一条。"""
    match = re.search(r"[^。！？；…!?;]+[。！？；…!?;]?", text)
    snippet = (match.group(0) if match else text)[:limit]
    return snippet.strip()


def _match_lexicon(text: str, lexicon: dict[str, tuple[str, ...]], limit: int) -> list[str]:
    """按出现位置打标，先命中先得。

    不按词表顺序（那是字典序，没有意义）而按词在文里出现的先后：
    先写到的意象通常是有意铺陈的那一个。
    """
    hits: list[tuple[int, str]] = []
    for tag, forms in lexicon.items():
        positions = [text.find(f) for f in forms]
        found = [p for p in positions if p >= 0]
        if found:
            hits.append((min(found), tag))
    hits.sort()
    return [tag for _, tag in hits[:limit]]


def _match_emotion(text: str) -> list[str]:
    """闭集匹配。**落空就是空列表，绝不编。**

    编一个会让图里多一个孤立的 Emotion 节点：`chunks_by_emotion` 永远
    只返回这一条，看起来像检索坏了，实际是标签是假的。
    """
    return _match_lexicon(text, PROSE_EMOTION_TRIGGERS, MAX_EMOTION)


def _concept(text: str, keyphrases: list[str], discipline_hint: str, work: str) -> str:
    """`concept` 兜底链——psychology 的嵌入模板把它放在最前面，
    空了等于这一条的头部全空。所以它**永远不许为空**。"""
    for phrase in keyphrases:
        if phrase.endswith(_NOUN_SUFFIXES):
            return phrase
    if keyphrases:
        return keyphrases[0]
    if discipline_hint:
        return discipline_hint
    if work:
        return work[:8]
    return "未命名概念"


def lexical_tags(
    text: str,
    *,
    library: Library,
    work: str = "",
    discipline_hint: str = "",
    freq: Counter[str] | None = None,
) -> ChunkTags:
    """纯规则的兜底标签。纯函数——同一个块 + 同一张频次表永远得到同一份标签。

    `freq` 传整篇文档的频次表，见 `build_frequency_table`。
    """
    keyphrases = extract_keyphrases(text, freq=freq)
    tags = ChunkTags(
        quote=_first_sentence(text),
        keywords=keyphrases,
        emotion=_match_emotion(text),
        source="lexical",
    )

    if library is Library.PSYCHOLOGY:
        # **强制空，不是「落空也无所谓」。** `neo4j_index.write_chunks`
        # 对所有库都写 MENTIONS_IMAGERY；给哲学块打上「大地」「深渊」，
        # `chunks_by_imagery("大地")` 就会从古典文学那条图路径里返回哲学块。
        # 现有 psychology 语料 60 条一条 imagery 都没有，保持这个状态。
        tags.imagery = []
        tags.concept = _concept(text, keyphrases, discipline_hint, work)
        tags.type = "论述"
    else:
        tags.imagery = _match_lexicon(text, PROSE_IMAGERY_LEXICON, MAX_IMAGERY)
        tags.type = "散文" if library is Library.LITERATURE else "诗"

    return tags


__all__ = [
    "EMOTION_VOCAB",
    "MAX_EMOTION",
    "MAX_IMAGERY",
    "MAX_KEYWORDS",
    "PROSE_EMOTION_TRIGGERS",
    "PROSE_IMAGERY_LEXICON",
    "ChunkTags",
    "build_frequency_table",
    "extract_keyphrases",
    "keyphrases_across",
    "lexical_tags",
]
