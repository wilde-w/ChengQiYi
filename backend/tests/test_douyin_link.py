"""链接解析与 SSRF 守卫。

`link.py` 是整个后端唯一对用户输入发起外部请求的地方，
所以这里的断言重点是**拒绝**：拒绝非抖音域名、拒绝跳转链中途换主机、
拒绝无限跳转。放行逻辑反而简单。
"""

from __future__ import annotations

import httpx
import pytest

from app.douyin import link as link_mod
from app.douyin.base import DouyinError


class TestParseLocal:
    """不发起请求的解析路径。"""

    @pytest.mark.parametrize(
        "raw",
        [
            "7300000000000000001",                                  # 裸 id
            "https://www.douyin.com/video/7300000000000000001",     # 标准
            "https://www.douyin.com/video/7300000000000000001?x=1",  # 带查询串
            "https://m.douyin.com/video/7300000000000000001",       # 移动端
            "https://www.iesdouyin.com/share/video/7300000000000000001/?a=b",
            "https://www.douyin.com/?modal_id=7300000000000000001",  # 查询串形式
            "https://www.douyin.com/note/7300000000000000001",
            "  7300000000000000001  ",                              # 前后空白
        ],
    )
    def test_accepts_known_shapes(self, raw: str) -> None:
        ref = link_mod.parse_local(raw)
        assert ref is not None
        assert ref.aweme_id == "7300000000000000001"

    def test_extracts_url_from_share_text(self) -> None:
        """用户粘贴的通常是整段分享口令，不是裸链接。"""
        share = "7.65 abc:/ 复制打开抖音，看看【某某的作品】 https://www.douyin.com/video/7300000000000000001"
        ref = link_mod.parse_local(share)
        assert ref is not None and ref.aweme_id == "7300000000000000001"

    def test_short_link_defers_to_network(self) -> None:
        """短链本地解析不了，必须返回 None 让调用方去请求跳转。"""
        assert link_mod.parse_local("https://v.douyin.com/iRNBho6u/") is None

    @pytest.mark.parametrize(
        "raw",
        [
            "https://www.douyin.com/user/MS4wLjABAAAAxxxx",   # 主页不是视频
            "https://www.douyin.com/",                        # 首页
            "随便打的一段话",                                   # 无链接
            "",                                               # 空
        ],
    )
    def test_rejects_non_video_input(self, raw: str) -> None:
        with pytest.raises(link_mod.LinkParseError):
            link_mod.parse_local(raw)


class TestSsrfGuard:
    """域名白名单。这一组全是安全断言，不能放宽。"""

    @pytest.mark.parametrize(
        "url",
        [
            "https://evil-douyin.com/video/7300000000000000001",  # 后缀混淆
            "https://douyin.com.evil.com/video/7300000000000000001",
            "https://www.douyin.com@evil.com/video/7300000000000000001",  # userinfo 混淆
            "http://127.0.0.1:8000/video/7300000000000000001",
            "http://169.254.169.254/latest/meta-data/",           # 云元数据
            "http://[::1]/video/7300000000000000001",
            "http://192.168.1.1/video/7300000000000000001",
            "file:///etc/passwd",
        ],
    )
    def test_rejects_non_whitelisted_hosts(self, url: str) -> None:
        with pytest.raises(link_mod.LinkParseError):
            link_mod.parse_local(url)

    def test_evil_suffix_host_is_not_accepted_by_allowed_host(self) -> None:
        """精确匹配而非后缀匹配——这是白名单唯一正确的实现方式。"""
        with pytest.raises(link_mod.LinkParseError):
            link_mod.assert_allowed_host("https://evil-douyin.com/x")
        assert link_mod.assert_allowed_host("https://www.douyin.com/x") == "www.douyin.com"

    def test_userinfo_does_not_spoof_host(self) -> None:
        with pytest.raises(link_mod.LinkParseError):
            link_mod.assert_allowed_host("https://www.douyin.com@evil.com/x")


class _MockTransport(httpx.AsyncBaseTransport):
    """按 URL 返回预设响应，顺便记录请求次数。"""

    def __init__(self, routes: dict[str, httpx.Response]) -> None:
        self.routes = routes
        self.hits: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.hits.append(url)
        if url not in self.routes:
            return httpx.Response(404, request=request)
        return self.routes[url]


async def _resolve_with(routes: dict[str, httpx.Response], raw: str):
    transport = _MockTransport(routes)
    async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
        return await link_mod.resolve(raw, client=client), transport


class TestShortLinkResolution:
    async def test_follows_redirect_to_canonical_url(self) -> None:
        # extract_url 会剥掉结尾斜杠，所以 mock 里也按无斜杠注册
        routes = {
            "https://v.douyin.com/abc123": httpx.Response(
                302,
                headers={"location": "https://www.douyin.com/video/7300000000000000001"},
            ),
        }
        ref, _ = await _resolve_with(routes, "https://v.douyin.com/abc123/")
        assert ref.aweme_id == "7300000000000000001"
        assert ref.resolved_via == "redirect"

    async def test_rejects_redirect_escaping_whitelist(self) -> None:
        """白名单域名 302 到内网——没有这条守卫，白名单形同虚设。"""
        routes = {
            "https://v.douyin.com/abc123": httpx.Response(
                302, headers={"location": "http://169.254.169.254/latest/meta-data/"}
            ),
        }
        with pytest.raises(link_mod.LinkParseError):
            await _resolve_with(routes, "https://v.douyin.com/abc123/")

    async def test_stops_after_max_redirects(self) -> None:
        """自跳转链不能无限跟随。"""
        loop = httpx.Response(302, headers={"location": "https://v.douyin.com/loop/"})
        routes = {"https://v.douyin.com/loop/": loop}
        for i in range(link_mod.MAX_REDIRECTS + 2):
            routes[f"https://v.douyin.com/hop{i}/"] = httpx.Response(
                302, headers={"location": f"https://v.douyin.com/hop{i + 1}/"}
            )
        with pytest.raises(link_mod.LinkParseError):
            await _resolve_with(routes, "https://v.douyin.com/hop0/")

    async def test_timeout_is_retryable(self) -> None:
        class _Timeout(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                raise httpx.ReadTimeout("timed out", request=request)

        async with httpx.AsyncClient(transport=_Timeout()) as client:
            with pytest.raises(DouyinError) as exc:
                await link_mod.resolve("https://v.douyin.com/slow/", client=client)
        assert exc.value.retryable is True

    async def test_local_input_does_not_hit_network(self) -> None:
        """能本地解析就绝不发请求——少一次外部依赖。"""
        ref, transport = await _resolve_with({}, "https://www.douyin.com/video/7300000000000000001")
        assert ref.aweme_id == "7300000000000000001"
        assert transport.hits == []


class TestExtractUrl:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("看这个 https://v.douyin.com/abc/ 好玩", "https://v.douyin.com/abc"),
            ("https://www.douyin.com/video/7300000000000000001，好看", "https://www.douyin.com/video/7300000000000000001"),
            ("（https://v.douyin.com/abc/）", "https://v.douyin.com/abc"),
            ("没有链接", None),
        ],
    )
    def test_strips_trailing_cjk_punctuation(self, raw: str, expected: str | None) -> None:
        assert link_mod.extract_url(raw) == expected
