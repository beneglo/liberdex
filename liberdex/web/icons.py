"""The favicon proxy.

A SERP row wants the site's icon, and `include_favicon` gives us its URL. But
letting the browser load those URLs directly would hand every site in the SERP a
request saying which query you ran and when, which is the leak this UI exists to
close. So the icon comes through here instead: liberdex fetches it, caches it,
and serves it from the same origin as the page.

The URL comes out of third-party HTML (`extract.favicon_of`), so this endpoint
is an SSRF sink and is written as one. `Fetcher` is deliberately not reused:
`Fetcher.get_raw` drops the body of anything that is not textual, which is every
image there is.
"""
from __future__ import annotations

import asyncio
import ipaddress
import socket
import time
from collections import OrderedDict
from typing import Optional
from urllib.parse import urlsplit

import httpx

MAX_BYTES = 64 * 1024
TIMEOUT = 3.0
MAX_REDIRECTS = 2
MAX_URL = 2048
CACHE_MAX = 512
CACHE_TTL = 24 * 3600

# An icon is a picture. Anything else (an HTML error page served with a 200, a
# login redirect, a tracking pixel's text/plain) is dropped rather than passed
# through to the page.
_ALLOWED = "image/"

_cache: "OrderedDict[str, tuple[float, bytes, str]]" = OrderedDict()
_client: Optional[httpx.AsyncClient] = None
_lock = asyncio.Lock()


# ---------------------------------------------------------------- SSRF fencing
def target(url: str) -> str:
    """The host of a URL we would consider fetching, or "" for one we would not.

    Shape only: scheme, length, a host at all. Where that host points is
    `public_host`'s question, and it needs the network to answer it.
    """
    if not url or len(url) > MAX_URL:
        return ""
    try:
        parts = urlsplit(url)
    except ValueError:
        return ""
    if parts.scheme not in ("http", "https"):
        return ""
    return parts.hostname or ""


async def public_host(host: str) -> bool:
    """Does every address this host resolves to sit on the public internet?

    Every address, not any: a host that resolves to one routable address and one
    loopback address is still a way into this machine, and DNS picks which one
    the fetch uses. Resolution goes through the loop's executor because a stalled
    resolver must not stop the search that is running alongside it.

    httpx resolves the name again when it connects, so the check here is one
    lookup ahead of the connection rather than pinned to it. Pinning would cost
    the SNI and certificate handling that TLS to a raw address needs; what
    remains is a GET whose body must be under 64 KB and must be an image before
    anything is returned.
    """
    if not host:
        return False
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(
            host, None, proto=socket.IPPROTO_TCP)
    except (OSError, UnicodeError):
        return False
    if not infos:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_multicast or ip.is_reserved or ip.is_unspecified):
            return False
    return True


async def safe_url(url: str) -> bool:
    """Is this a URL we are willing to fetch on the caller's behalf?"""
    return await public_host(target(url))


# -------------------------------------------------------------------- fetching
async def client() -> httpx.AsyncClient:
    global _client
    async with _lock:
        if _client is None:
            _client = httpx.AsyncClient(
                follow_redirects=False,  # each hop is re-fenced by hand
                timeout=TIMEOUT,
                limits=httpx.Limits(max_connections=16),
                headers={"Accept": "image/*,*/*;q=0.5",
                         "User-Agent": "Mozilla/5.0 (compatible; liberdex)"},
            )
    return _client


async def aclose() -> None:
    global _client
    c, _client = _client, None
    if c is not None:
        await c.aclose()
    _cache.clear()


async def _fetch(url: str) -> Optional[tuple[bytes, str]]:
    """(body, content_type), or None for anything that is not a small image."""
    http = await client()
    for _ in range(MAX_REDIRECTS + 1):
        if not await safe_url(url):
            return None
        try:
            async with http.stream("GET", url) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    nxt = resp.headers.get("location")
                    if not nxt:
                        return None
                    url = str(resp.url.join(nxt))
                    await resp.aclose()
                    continue
                if resp.status_code != 200:
                    return None
                ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
                if not ctype.startswith(_ALLOWED):
                    return None
                body = bytearray()
                async for chunk in resp.aiter_bytes():
                    body += chunk
                    if len(body) > MAX_BYTES:
                        return None
                return (bytes(body), ctype) if body else None
        except Exception:
            return None
    return None


async def get(url: str) -> Optional[tuple[bytes, str]]:
    """A cached icon. None means the caller should draw the monogram instead."""
    now = time.time()
    hit = _cache.get(url)
    if hit is not None:
        stamp, body, ctype = hit
        if now - stamp < CACHE_TTL:
            _cache.move_to_end(url)
            return (body, ctype) if body else None
        del _cache[url]

    got = await _fetch(url)
    # A miss is cached too, as an empty body: a site with no reachable icon must
    # not be re-fetched on every search that returns it.
    _cache[url] = (now, got[0] if got else b"", got[1] if got else "")
    _cache.move_to_end(url)
    while len(_cache) > CACHE_MAX:
        _cache.popitem(last=False)
    return got


# ------------------------------------------------------------------- fallbacks
_MONOGRAM = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16">'
    '<rect width="16" height="16" rx="4" fill="#0B1A31"/>'
    '<text x="8" y="11.5" text-anchor="middle" fill="#6FC5FF" '
    'font-family="Helvetica,Arial,sans-serif" font-size="9" '
    'font-weight="600">{ch}</text></svg>'
)


def monogram(site: str) -> bytes:
    """A stand-in icon: the site's initial, so a row never shows a broken image."""
    ch = (site or "").removeprefix("www.").lstrip(".")[:1].upper() or "?"
    if ch in "<>&\"'":
        ch = "?"
    return _MONOGRAM.format(ch=ch).encode()
