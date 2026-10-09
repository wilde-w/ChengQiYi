"""抖音链接解析与 SSRF 守卫。

**这是整个后端唯一一处会对用户输入发起外部请求的地方**，因此防护写在这里
而不是散落在调用点：短链必须跳转才能拿到 aweme_id，而跳转目标完全由用户控制。

防护四条：主机白名单、跳转次数上限、超时、以及禁止跳转链中途换到白名单外的主机。
没有最后一条，攻击者可以用一个白名单域名 302 到内网地址。
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import httpx

from app.douyin.base import DouyinError, ResolvedRef
from app.logging_conf import get_logger

log = get_logger(__name__)

# 只允许这些主机。刻意精确匹配而不做后缀匹配——
# `evil-douyin.com` 会通过后缀匹配，但绝不该被放行。
ALLOWED_HOSTS: frozenset[str] = frozenset(
    {
        "v.douyin.com",
        "www.douyin.com",
        "douyin.com",
        "m.douyin.com",
        "iesdouyin.com",
        "www.iesdouyin.com",
    }
)

SHORT_LINK_HOSTS: frozenset[str] = frozenset({"v.douyin.com"})

MAX_REDIRECTS = 3
RESOLVE_TIMEOUT_SECONDS = 5.0

# 19 位数字（抖音 aweme_id 的实际形态）
_BARE_ID = re.compile(r"^\d{15,25}$")
_ID_IN_PATH = re.compile(r"/(?:video|note|share/video|share/note)/(\d{15,25})")
_ID_IN_QUERY = re.compile(r"[?&](?:modal_id|aweme_id|vid)=(\d{15,25})")
_URL = re.compile(r"https?://[^\s，。、）)】\]]+")
# 分享口令形如「7.65 复制打开抖音，看看【...】 https://v.douyin.com/xxxx/」
_SHARE_TEXT = re.compile(r"^\s*(?P<code>[A-Za-z0-9]{4,12})\s*$")


class LinkParseError(DouyinError):
    pass


def extract_url(raw: str) -> str | None:
    """从分享文案里抠出 URL。用户粘贴的通常是整段口令而不是裸链接。"""
    match = _URL.search(raw)
    return match.group(0).rstrip("/") if match else None


def _host_of(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    return host


def assert_allowed_host(url: str) -> str:
    host = _host_of(url)
    if not host:
        raise LinkParseError(f"无法解析链接主机：{url[:80]}", retryable=False)
    if host not in ALLOWED_HOSTS:
        raise LinkParseError(
            f"不接受该域名（{host}）。仅支持抖音官方域名：{', '.join(sorted(ALLOWED_HOSTS))}",
            retryable=False,
        )
    return host


def parse_local(raw: str) -> ResolvedRef | None:
    """不发起网络请求的解析。能解析出来就别请求——少一次外部依赖。"""
    text = raw.strip()
    if not text:
        raise LinkParseError("输入为空", retryable=False)

    if _BARE_ID.match(text):
        return ResolvedRef(aweme_id=text, canonical_url=_canonical(text), resolved_via="regex")

    url = extract_url(text)
    if url is None:
        match = _SHARE_TEXT.match(text)
        if match:
            # 分享口令本身不含 aweme_id，需要走短链跳转
            return None
        raise LinkParseError(
            "没有识别到抖音链接或 aweme_id。可直接粘贴视频链接、分享口令，或 19 位 aweme_id。",
            retryable=False,
        )

    host = assert_allowed_host(url)

    for pattern in (_ID_IN_PATH, _ID_IN_QUERY):
        m = pattern.search(url)
        if m:
            return ResolvedRef(
                aweme_id=m.group(1), canonical_url=_canonical(m.group(1)), resolved_via="regex"
            )

    if host in SHORT_LINK_HOSTS:
        return None  # 短链必须请求跳转

    raise LinkParseError(
        "链接里找不到视频 id。请确认这是视频页链接（含 /video/ 或 modal_id），而不是主页或直播。",
        retryable=False,
    )


async def resolve(raw: str, *, client: httpx.AsyncClient | None = None) -> ResolvedRef:
    """把任意形式的输入解析成 ResolvedRef。只有短链才会发起请求。"""
    local = parse_local(raw)
    if local is not None:
        return local

    url = extract_url(raw)
    if url is None:
        raise LinkParseError("未找到可解析的链接", retryable=False)

    target = await _follow_short_link(url, client=client)
    for pattern in (_ID_IN_PATH, _ID_IN_QUERY):
        m = pattern.search(target)
        if m:
            return ResolvedRef(
                aweme_id=m.group(1), canonical_url=_canonical(m.group(1)), resolved_via="redirect"
            )
    raise LinkParseError(
        f"短链跳转后仍未找到 aweme_id（最终地址：{target[:120]}）", retryable=False
    )


async def _follow_short_link(url: str, *, client: httpx.AsyncClient | None) -> str:
    """跟随跳转，但每一跳都重新校验主机。"""
    current = url
    owns_client = client is None
    client = client or httpx.AsyncClient(
        timeout=RESOLVE_TIMEOUT_SECONDS,
        follow_redirects=False,   # 手动跟跳，才能逐跳校验主机
        headers={"User-Agent": "Mozilla/5.0"},
    )
    try:
        for hop in range(MAX_REDIRECTS + 1):
            assert_allowed_host(current)
            try:
                resp = await client.get(current)
            except httpx.TimeoutException as exc:
                raise LinkParseError(f"解析短链超时：{exc}", retryable=True) from exc
            except httpx.HTTPError as exc:
                raise LinkParseError(f"解析短链失败：{exc}", retryable=True) from exc

            if resp.is_redirect:
                location = resp.headers.get("location", "")
                if not location:
                    raise LinkParseError("跳转响应缺少 Location 头", retryable=False)
                current = str(httpx.URL(current).join(location))
                log.debug("short_link_hop", hop=hop + 1, host=_host_of(current))
                continue

            # 非跳转即终点；抖音短链最终会 200 返回含 aweme_id 的页面
            _inject_location = resp.headers.get("location")
            final = str(resp.url)
            if _inject_location:
                final = str(httpx.URL(final).join(_inject_location))
            return final

        raise LinkParseError(f"短链跳转超过 {MAX_REDIRECTS} 次仍未落地", retryable=False)
    finally:
        if owns_client:
            await client.aclose()


def _canonical(aweme_id: str) -> str:
    return f"https://www.douyin.com/video/{aweme_id}"
