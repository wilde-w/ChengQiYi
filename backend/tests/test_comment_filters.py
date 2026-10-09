"""评论清洗：广告识别、灌水识别、去重。

这里的断言分两类，意图完全不同：
  - **误杀测试**（必须放行）：真实的高赞情绪评论，尤其叠字与超短评论。
    把用户的声音当成广告删掉，比留下几条广告严重得多。
  - **漏放测试**（必须拦截）：内置样本里那几条种子广告与纯符号评论。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.douyin.base import CommentItem
from app.services.comment_service import (
    clean_comments,
    dedup_key,
    detect_ad,
    detect_spam,
    normalize,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "app" / "douyin" / "fixtures"
FIXTURES = sorted(FIXTURE_DIR.glob("*.json"))


# ----------------------------------------------------------------------
# 必须放行
# ----------------------------------------------------------------------

MUST_KEEP = [
    "看哭了，我妈也走了三年了",
    "你到家了给我说一声",
    "哈哈哈哈哈哈哈哈哈哈",      # 叠字是最常见的真实情绪表达
    "呜呜呜呜呜呜呜呜呜呜",
    "太好看了！！！",
    "好",                        # 低信息量 ≠ 恶意，交给聚类判噪声
    "+1",
    "妈妈做的饭最好吃",
    "这个价格有优惠吗我想下单",   # 顾客在提问，不是广告
    "想你想你想你想你想你",
    "哈哈哈哈哈哈哈广告狗",       # 含「广告」二字但不是广告
]


@pytest.mark.parametrize("text", MUST_KEEP)
def test_real_comments_are_not_filtered(text: str) -> None:
    assert detect_ad(text) is False, f"误判为广告：{text}"
    assert detect_spam(text) is False, f"误判为灌水：{text}"


# ----------------------------------------------------------------------
# 必须拦截
# ----------------------------------------------------------------------

MUST_BLOCK_AD = [
    "加微信 xxx666 领取同款",
    "加V:gwjy_668 专业情感疗愈",
    "加V情感疗愈，一小时走出丧亲之痛",   # V 后直接接汉字
    "私我，主页有链接",
    "http://spam.example.com",
    "同款老人机 9.9包邮，点我主页，支持货到付款",
    "限时秒杀 优惠券点这里领",
    "联系电话 13800138000",
]


@pytest.mark.parametrize("text", MUST_BLOCK_AD)
def test_ads_are_detected(text: str) -> None:
    assert detect_ad(text) is True, f"漏放广告：{text}"


MUST_BLOCK_SPAM = [
    "😭😭😭😭😭😭😭😭😭😭",
    "。。。。。。。。。。。。",
    "aaaaaaaaaaaa",              # 非中文长重复才是灌水
    "111111111111",
    "。。",
]


@pytest.mark.parametrize("text", MUST_BLOCK_SPAM)
def test_spam_is_detected(text: str) -> None:
    assert detect_spam(text) is True, f"漏放灌水：{text}"


# ----------------------------------------------------------------------
# 归一化与指纹
# ----------------------------------------------------------------------


class TestNormalize:
    def test_nfkc_folds_fullwidth(self) -> None:
        """全角/半角混用在抖音评论里极常见，不归一化就是两条不同评论。"""
        assert normalize("太好看了！") == "太好看了!"

    def test_strips_zero_width_and_collapses_space(self) -> None:
        assert normalize("好看​的   很") == "好看的 很"

    def test_keeps_emoji(self) -> None:
        assert "😭" in normalize("哭了😭")


class TestDedupKey:
    def test_emoji_and_punctuation_do_not_change_identity(self) -> None:
        assert dedup_key("太好看了！！") == dedup_key("太好看了😭😭") == dedup_key("太好看了")

    def test_at_mention_and_url_are_ignored(self) -> None:
        assert dedup_key("想你了 @小明") == dedup_key("想你了")
        assert dedup_key("看这个 http://a.com") == dedup_key("看这个")

    def test_different_text_differs(self) -> None:
        assert dedup_key("想你了") != dedup_key("想他了")

    def test_stable_across_calls(self) -> None:
        """blake2b 而非内置 hash——后者跨进程随机化，会让去重不可复现。"""
        assert dedup_key("妈妈我想你") == dedup_key("妈妈我想你")


# ----------------------------------------------------------------------
# 批量清洗
# ----------------------------------------------------------------------


class TestCleanComments:
    def test_keeps_highest_liked_duplicate(self) -> None:
        items = [
            CommentItem("c1", "太好看了！！", like_count=10),
            CommentItem("c2", "太好看了😭😭", like_count=99),
            CommentItem("c3", "太好看了", like_count=1),
        ]
        cleaned, stats = clean_comments(items)
        assert [c.is_duplicate for c in cleaned] == [True, False, True]
        assert cleaned[1].comment_id == "c2"
        assert stats.duplicates == 2
        assert stats.kept == 1

    def test_ads_do_not_absorb_real_comments(self) -> None:
        """广告文案高度雷同，若参与去重会把真实评论一起拖下水。"""
        items = [
            CommentItem("a1", "加微信 abc123"),
            CommentItem("a2", "加微信 abc123"),
            CommentItem("r1", "妈妈我想你了", like_count=5),
        ]
        cleaned, stats = clean_comments(items)
        assert stats.ads == 2 and stats.duplicates == 0 and stats.kept == 1

    def test_filtered_comments_are_retained_not_dropped(self) -> None:
        """被过滤的也要返回——前端「显示被过滤内容」开关需要它们。"""
        items = [CommentItem("c1", "加微信 abc123"), CommentItem("c2", "真实评论")]
        cleaned, _ = clean_comments(items)
        assert len(cleaned) == 2
        assert cleaned[0].is_ad is True
        assert cleaned[0].filter_reason == "ad"

    def test_filter_reason_is_set_for_every_filtered_item(self) -> None:
        items = [
            CommentItem("c1", "加微信 abc123"),
            CommentItem("c2", "😭😭😭😭😭😭😭😭😭😭"),
            CommentItem("c3", "重复内容"),
            CommentItem("c4", "重复内容"),
            CommentItem("c5", "正常"),
        ]
        cleaned, stats = clean_comments(items)
        for c in cleaned:
            if not c.is_visible_by_default:
                assert c.filter_reason, f"{c.comment_id} 被过滤但没有 reason"
        assert stats.total == 5
        assert stats.kept + stats.ads + stats.spam + stats.duplicates + stats.empty == 5

    def test_stats_reconcile_with_marks(self) -> None:
        """统计必须与标记一致——早期版本在「后出现的重复更热」时会算错。"""
        items = [
            CommentItem("c1", "同一句话", like_count=1),
            CommentItem("c2", "同一句话", like_count=100),   # 后出现的更热 → 顶替
            CommentItem("c3", "广告 http://x.com"),
            CommentItem("c4", "正常评论"),
        ]
        cleaned, stats = clean_comments(items)
        counted = len([c for c in cleaned if c.is_duplicate])
        assert counted == stats.duplicates == 1
        assert len([c for c in cleaned if c.is_ad]) == stats.ads == 1
        assert len([c for c in cleaned if c.is_visible_by_default]) == stats.kept == 2


# ----------------------------------------------------------------------
# 内置样本：真实数据上的验收
# ----------------------------------------------------------------------


@pytest.mark.skipif(not FIXTURES, reason="演示样本不存在")
class TestAgainstFixtures:
    """主验收：所有种子噪声都被抓到，且没有一条真实评论被误杀。

    这组断言的期望值是**照样本标注写的**，改动过滤规则若误伤会立刻失败。
    """

    SEED_NOISE = {
        "demo_grief_wechat_voice.json": {"c012", "c026", "c040"},
        "demo_burnout_office.json": {"c012", "c024", "c033"},
        "demo_first_love_summer.json": {"c008", "c015", "c025"},
    }

    @pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
    def test_seed_noise_is_caught_and_clean_comments_survive(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = [
            CommentItem(c["comment_id"], c["text"], like_count=c.get("like_count", 0))
            for c in data["comments"]
        ]
        cleaned, stats = clean_comments(items)

        flagged = {c.comment_id for c in cleaned if not c.is_visible_by_default}
        expected = self.SEED_NOISE[path.name]
        missed = expected - flagged
        assert not missed, f"{path.name} 未识别出种子噪声：{sorted(missed)}"

        # 阴阳怪气那几条是真实（虽不友善）的观点，属于信号不是噪声——
        # 过滤它们就等于删掉用户的声音。HDBSCAN 会把它们收进噪声簇。
        assert stats.kept >= len(items) - len(expected) - 1, (
            f"{path.name} 误杀过多：kept={stats.kept}，仅应过滤 {len(expected)} 条"
        )

    @pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
    def test_every_filtered_item_has_a_reason(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = [CommentItem(c["comment_id"], c["text"]) for c in data["comments"]]
        cleaned, _ = clean_comments(items)
        for c in cleaned:
            if not c.is_visible_by_default:
                assert c.filter_reason in {"ad", "spam", "duplicate", "empty"}
