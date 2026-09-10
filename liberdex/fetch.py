"""Async page fetcher.

curl_cffi is the backend: libcurl's multi handle handles a hundred concurrent
URLs comfortably, and Chrome TLS-fingerprint impersonation walks through the
bot walls that 403 a plain Python client.

Latency first: one shared session, high global concurrency, low per-host
concurrency so one slow origin cannot own the pool or earn a 429, a byte cap,
and a global deadline. Docs are yielded as they land so downstream stages can
start before the slowest fetch returns.
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Iterable, Optional
from urllib.parse import urlsplit

from curl_cffi.requests import AsyncSession

from .types import Candidate, Deadline, Doc

IMPERSONATE = "chrome"

BASE_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
              "application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

TEXTUAL = ("text/html", "application/xhtml", "text/plain", "application/json",
           "application/xml", "text/xml", "application/rss", "application/atom",
           "application/ld+json")

# Not text, but a document. For statistics offices, ministries and preprint
# servers the PDF is the authoritative version and the HTML an abstract of it.
DOCUMENT = ("application/pdf",)

MAX_BYTES = 1_500_000
# A PDF is a container: 1.5 MB of HTML is a huge page, 1.5 MB of PDF is a
# routine annual report, and truncating one yields an unparseable document,
# since the cross-reference table is at the end of the file. The extractor's
# page limit bounds the parsing work. Both caps slice a body already received
# in full, so they are chosen against resident memory at `total_concurrency`
# documents in flight; `max_request_timeout` is what bounds the transfer.
MAX_DOC_BYTES = 8_000_000


def host_of(url: str) -> str:
    try:
        return urlsplit(url).hostname or ""
    except ValueError:
        return ""


# Suffixes under which every subdomain is a different publisher. Collapsing
# rust-lang.github.io and py-free-threading.github.io into one "host" makes the
# per-host SERP cap throw away an independent source.
MULTI_TENANT = frozenset({
    "github.io", "gitlab.io", "codeberg.page", "sourceforge.net",
    "readthedocs.io", "rtfd.io", "gitbook.io", "notion.site",
    "pages.dev", "workers.dev", "netlify.app", "vercel.app", "surge.sh",
    "herokuapp.com", "appspot.com", "azurewebsites.net", "onrender.com",
    "fly.dev", "glitch.me", "repl.co", "replit.app",
    "blogspot.com", "wordpress.com", "tumblr.com", "weebly.com",
    "wixsite.com", "neocities.org", "bearblog.dev", "itch.io",
    "substack.com", "medium.com", "notion.so", "obsidian.md",
})


# Second-level labels a country registry operates as a public suffix, so what
# sits under one is a site and not a subdomain of one. Missing one collapses
# every institution under that label into a single host and SERP diversity
# suppresses all but a couple of them. Not the full public suffix list, which
# is a 200 KB data file; a label wrongly here only shows more sources rather
# than fewer, so erring towards coverage is the safe direction.
_REGISTRY_SLD = frozenset({
    "co", "com", "org", "net", "edu", "ac", "gov", "mil", "int",
    "gv", "go", "gob", "gouv", "govt", "or", "ne", "lg", "ad", "priv",
    "sch", "nhs", "res", "asn", "nom", "web", "info", "biz",
})


def accept_language(lang: str) -> Optional[dict[str, str]]:
    """Per-request Accept-Language for a non-English plan; None keeps the
    session default. A site that negotiates language must serve the edition
    whose words the query and the ranker share."""
    lang = (lang or "").strip().lower()
    if not lang or lang == "en":
        return None
    return {"Accept-Language": f"{lang},en;q=0.5"}


def registrable(host: str) -> str:
    """Cheap eTLD+1 approximation, good enough for per-host fairness."""
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    if parts[-2] in _REGISTRY_SLD and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    two = ".".join(parts[-2:])
    if two in MULTI_TENANT:
        return ".".join(parts[-3:])
    return two


def host_matches(host: str, domains: "set[str]") -> bool:
    """Does `host` fall under any of `domains`?

    An entry matches the host itself and everything beneath it, so
    "apache.org" covers kafka.apache.org while "kafka.apache.org" does not
    cover the rest of apache.org. A leading "www." is noise on both sides.
    """
    if not domains:
        return False
    h = (host or "").lower().rstrip(".")
    if h.startswith("www."):
        h = h[4:]
    if not h:
        return False
    for d in domains:
        d = (d or "").lower().strip().rstrip(".")
        if d.startswith("www."):
            d = d[4:]
        if not d:
            continue
        if h == d or h.endswith("." + d):
            return True
    return False


class Fetcher:
    """Shared session. Construct once per process, reuse across queries."""

    def __init__(
        self,
        *,
        total_concurrency: int = 96,
        per_host_concurrency: int = 6,
        timeout: float = 6.0,
        max_request_timeout: float = 3.0,
        impersonate: str = IMPERSONATE,
    ) -> None:
        self.total_concurrency = total_concurrency
        self.per_host = per_host_concurrency
        self.timeout = timeout
        # A single stalling origin must not be allowed to eat the whole query
        # budget just because the budget happens to be large.
        self.max_request_timeout = max_request_timeout
        self.impersonate = impersonate
        self._session: Optional[AsyncSession] = None
        self._gate: Optional[asyncio.Semaphore] = None
        self._host_gates: dict[str, asyncio.Semaphore] = {}

    async def __aenter__(self) -> "Fetcher":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def start(self) -> None:
        if self._session is None:
            self._session = AsyncSession(
                impersonate=self.impersonate,
                timeout=self.timeout,
                max_clients=self.total_concurrency,
                headers=BASE_HEADERS,
                verify=False,  # the long tail runs on expired and mismatched certificates
            )
            self._gate = asyncio.Semaphore(self.total_concurrency)

    async def aclose(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            try:
                await session.close()
            except Exception:
                pass
            # Give libcurl's multi handle a loop tick to finish teardown before
            # the event loop itself goes away.
            await asyncio.sleep(0)

    def _host_gate(self, host: str) -> asyncio.Semaphore:
        gate = self._host_gates.get(host)
        if gate is None:
            gate = asyncio.Semaphore(self.per_host)
            self._host_gates[host] = gate
        return gate

    async def get_raw(
        self,
        url: str,
        headers: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> tuple[int, bytes, str, str]:
        """Fetch a URL, returning (status, body, final_url, content_type)."""
        assert self._session is not None, "Fetcher not started"
        resp = await self._session.get(
            url,
            headers=headers,
            timeout=timeout or self.timeout,
            allow_redirects=True,
            max_redirects=5,
        )
        ctype = (resp.headers.get("content-type") or "").lower()
        body = resp.content or b""
        cap = MAX_BYTES
        if resp.status_code == 200 and ctype and not any(t in ctype for t in TEXTUAL):
            if any(t in ctype for t in DOCUMENT):
                cap = MAX_DOC_BYTES
            else:
                body = b""  # images, archives: keep the status, drop the payload
        return resp.status_code, body[:cap], str(resp.url), ctype

    async def _fetch_one(self, cand: Candidate, deadline: Deadline, reserve: float) -> Doc:
        url = cand.url
        t0 = time.perf_counter()
        doc = Doc(url=url, final_url=url, status=0, candidate=cand)
        if deadline.remaining - reserve <= 0.15:
            doc.error = "deadline"
            return doc
        gate, host_gate = self._gate, self._host_gate(registrable(host_of(url)))
        assert gate is not None
        try:
            async with gate, host_gate:
                budget = min(self.max_request_timeout, deadline.remaining - reserve)
                if budget <= 0.15:
                    doc.error = "deadline"
                    return doc
                status, body, final_url, ctype = await asyncio.wait_for(
                    self.get_raw(url, headers=accept_language(cand.lang),
                                 timeout=budget),
                    timeout=budget + 0.25
                )
            doc.status = status
            doc.final_url = final_url
            doc.content_type = ctype
            doc.bytes_in = len(body)
            doc.body = body
        except asyncio.TimeoutError:
            doc.error = "timeout"
        except asyncio.CancelledError:
            doc.error = "cancelled"
        except Exception as e:
            doc.error = type(e).__name__
        doc.fetch_ms = (time.perf_counter() - t0) * 1000
        return doc

    async def stream(
        self, cands: Iterable[Candidate], deadline: Deadline, reserve: float = 0.3
    ) -> AsyncIterator[Doc]:
        """Yield Docs as they complete. `reserve` is time held back for ranking."""
        tasks = [asyncio.create_task(self._fetch_one(c, deadline, reserve)) for c in cands]
        if not tasks:
            return
        try:
            for fut in asyncio.as_completed(tasks):
                try:
                    yield await fut
                except Exception:
                    continue
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()

    async def gather(
        self, cands: Iterable[Candidate], deadline: Deadline, reserve: float = 0.3
    ) -> list[Doc]:
        return [d async for d in self.stream(cands, deadline, reserve)]
