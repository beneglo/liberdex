"""The site's own index, borrowed at query time.

An index-free engine still has to answer the question an index answers: what
pages does this host have? The planner is good at naming hosts and bad at
naming paths, so the common failure is the right building and the wrong door.

Almost every host publishes the list of its own doors, for crawlers, under
`robots.txt` or `/sitemap.xml`. Reading it is not maintaining an index:
nothing is crawled, nothing is stored beyond a per-host cache with a TTL, and
the file is the host's own current statement about itself rather than a stale
copy of its content.

Cost is one or two requests per host, once per TTL, and only for hosts whose
guessed URL failed. On a warm cache choosing among the slugs is a string
operation.
"""
from __future__ import annotations

import asyncio
import gzip
import re
from typing import Optional
from urllib.parse import urlsplit

from .fetch import registrable
from .rank import fold, path_fit, path_terms

# <loc> is the only element in the schema that carries a URL, in both the
# urlset and the sitemapindex document, which is why one pattern reads both.
_LOC = re.compile(rb"<loc>\s*([^<\s]{4,2048}?)\s*</loc>", re.I)
_IS_INDEX = re.compile(rb"<sitemapindex", re.I)
_ROBOTS_SITEMAP = re.compile(rb"(?im)^[ \t]*sitemap:[ \t]*(\S+)")

# Past this we are downloading someone's whole catalogue to pick three links
# out of it.
MAX_SITEMAP_BYTES = 4_000_000
# Enough to hold a mid-sized publisher's whole site. Beyond it the marginal URL
# is not going to be the answer, and the scoring pass stops being free.
MAX_LOCS = 20_000
# A sitemap index points at child sitemaps. Following all of them is a crawl;
# following the few whose own address best fits the query is a lookup.
INDEX_CHILDREN = 3
# How well a child sitemap's own address must fit the query before it jumps
# the index's own ordering. High: these names are rarely topical.
INDEX_CHILD_FIT = 0.20
# Requests spent per host, total, across discovery and children.
MAX_FETCHES = 5


def _decompress(body: bytes) -> bytes:
    if body[:2] == b"\x1f\x8b":
        try:
            return gzip.decompress(body)
        except Exception:
            return b""
    return body


def parse(body: bytes) -> tuple[list[str], bool]:
    """(<loc> values, is this a sitemap index)."""
    body = _decompress(body)[:MAX_SITEMAP_BYTES]
    if not body:
        return [], False
    out: list[str] = []
    for m in _LOC.finditer(body):
        try:
            u = m.group(1).decode("utf-8", "ignore")
        except Exception:
            continue
        # `&amp;` is the only entity a sitemap is required to escape.
        u = u.replace("&amp;", "&")
        if u.startswith("http"):
            out.append(u)
        if len(out) >= MAX_LOCS:
            break
    return out, bool(_IS_INDEX.search(body[:4096]))


def robots_sitemaps(body: bytes, host: str) -> list[str]:
    out: list[str] = []
    want = registrable(host)
    for m in _ROBOTS_SITEMAP.finditer(body or b""):
        u = m.group(1).decode("utf-8", "ignore").strip()
        # A `Sitemap:` line is defined to be absolute, but plenty are not, and
        # one pointing at another host is somebody else's index. Compared on the
        # registrable domain, so a CDN subdomain counts as this site.
        if not u.startswith("http"):
            continue
        if registrable((urlsplit(u).hostname or "").lower()) != want:
            continue
        out.append(u)
        if len(out) >= 4:
            break
    return out


# `/de/`, `/en-us/`, `/zh-tw/` as the first path segment: the one piece of URL
# structure that means the same thing on nearly every multinational site. The
# slug fit cannot tell fourteen editions of one report apart; the locale can.
_LOCALE = re.compile(r"^/([a-z]{2})(?:[-_]([a-z]{2}))?/", re.I)


def locale_of(path: str) -> str:
    m = _LOCALE.match(path)
    return m.group(1).lower() if m else ""


# A query word this host puts in more than this share of its own addresses
# says nothing about which of them was wanted: a regional portal writes its
# region's name into a third of its slugs. Only judged on a catalogue big
# enough for a share to mean something.
GENERIC_SHARE = 0.15
GENERIC_MIN_LOCS = 40
_YEAR = re.compile(r"^(19|20)\d\d$")
YEAR_BONUS = 0.05


def _host_specific(locs: list[str], vocabs: list[set[str]]) -> list[set[str]]:
    """The query's vocabularies less the words this host uses everywhere."""
    if len(locs) < GENERIC_MIN_LOCS:
        return vocabs
    df: dict[str, int] = {}
    for u in locs:
        try:
            _, split = path_terms(urlsplit(u).path)
        except ValueError:
            continue
        for t in {fold(x) for x in split}:
            df[t] = df.get(t, 0) + 1
    cut = GENERIC_SHARE * len(locs)
    out: list[set[str]] = []
    for v in vocabs:
        kept = {t for t in v if df.get(fold(t), 0) <= cut}
        out.append(kept or v)
    return out


def pick(locs: list[str], vocabs: list[set[str]], n: int,
         *, min_fit: float = 0.12, lang: str = "",
         tie_shortest: bool = True) -> list[str]:
    """The n addresses in this host's catalogue that best fit the query.

    Scored on the path alone, with the same fit function the link expander
    uses, so a slug is judged by exactly the criteria an anchor is. `min_fit`
    is what stops a host with nothing relevant from contributing its front
    page: a catalogue that does not answer should return nothing rather than
    its least irrelevant entry.
    """
    vocabs = _host_specific(locs, vocabs)
    # A year is a tie-breaker, not a subject: scored as a word it is an exact
    # hit a slug can carry about anything. The words decide and the year adds
    # a little on top, when both the query and the address name one.
    years = {t for v in vocabs for t in v if _YEAR.match(t)}
    vocabs = [{t for t in v if t not in years} or v for v in vocabs]
    scored: list[tuple[float, str]] = []
    for u in locs:
        parts = urlsplit(u)
        fit = path_fit(vocabs, parts.path)
        if years and fit >= min_fit:
            _whole, split = path_terms(parts.path)
            if years & split:
                fit += YEAR_BONUS
        if fit < min_fit:
            continue
        if lang:
            loc = locale_of(parts.path)
            # An unlocalised path is the site's default and stays eligible. An
            # explicit foreign edition goes behind everything else rather than
            # out, since some hosts publish only under locale prefixes.
            if loc and loc != lang:
                fit *= 0.3
                if fit < min_fit:
                    continue
        scored.append((fit, u))
    # Among equal fits the shorter address is the more canonical one, except
    # among the children of a sitemap index, whose names ("pages.xml",
    # "offices.xml") say nothing about any query and all tie at zero. There
    # document order is the only information, and it is real: an index
    # conventionally lists its main content sitemap first.
    if tie_shortest:
        scored.sort(key=lambda t: (-t[0], len(t[1])))
    else:
        scored.sort(key=lambda t: -t[0])   # stable: ties keep document order
    return [u for _, u in scored[:n]]


class SitemapReader:
    """Fetches and caches one host's catalogue.

    `store` is any object with `get(host) -> list[str]` and `put(host, locs)`;
    None disables caching.
    """

    def __init__(self, fetcher, store=None, *, timeout: float = 3.0) -> None:
        self.fetcher = fetcher
        self.store = store
        self.timeout = timeout
        self._inflight: dict[str, asyncio.Task] = {}

    async def _get(self, url: str) -> bytes:
        try:
            status, body, _final, _ct = await self.fetcher.get_raw(
                url, timeout=self.timeout)
        except Exception:
            return b""
        return body if status == 200 else b""

    async def locs(self, host: str, vocabs: Optional[list[set[str]]] = None
                   ) -> list[str]:
        """Every URL this host publishes, from cache or from the host."""
        if self.store is not None:
            cached = self.store.get(host)
            if cached is not None:
                return cached
        task = self._inflight.get(host)
        if task is None:
            task = asyncio.ensure_future(self._fetch(host, vocabs or []))
            self._inflight[host] = task
            task.add_done_callback(lambda _t, h=host: self._inflight.pop(h, None))
        try:
            return await asyncio.shield(task)
        except Exception:
            return []

    async def _fetch(self, host: str, vocabs: list[set[str]]) -> list[str]:
        """Never raises: a query cancelled mid-recovery leaves this task running
        to finish populating the store, and an exception it left behind would
        surface as an unretrieved one long after the search."""
        try:
            return await self._read(host, vocabs)
        except asyncio.CancelledError:
            raise
        except Exception:
            return []

    async def _read(self, host: str, vocabs: list[set[str]]) -> list[str]:
        # robots.txt and the conventional address are asked for together: one
        # round trip instead of two. Where both answer, robots wins, since it is
        # the host's own statement.
        base = f"https://{host}"
        robots, direct = await asyncio.gather(
            self._get(f"{base}/robots.txt"), self._get(f"{base}/sitemap.xml"))
        fetches = 2
        locs: list[str] = []
        declared = robots_sitemaps(robots, host)
        # A host that serves its script shell for every unknown path answers
        # /sitemap.xml with 200 and HTML. Parsing finds no <loc> and it falls
        # through to whatever robots.txt declared.
        direct_locs, direct_index = parse(direct)
        candidates: list[tuple[list[str], bool]] = []
        if declared:
            for u in declared[: MAX_FETCHES - fetches]:
                if fetches >= MAX_FETCHES:
                    break
                body = await self._get(u)
                fetches += 1
                got, is_index = parse(body)
                if got:
                    candidates.append((got, is_index))
                    break
        if not candidates and direct_locs:
            candidates.append((direct_locs, direct_index))
        for got, is_index in candidates:
            if not is_index:
                locs.extend(got)
                continue
            # A sitemap index: its entries are addresses of more sitemaps. Their
            # names are structural, not topical ("pages.xml", "offices.xml"), so a
            # child has to fit the query convincingly to jump the queue; otherwise
            # the index's own order stands, main content sitemap first.
            named = pick(got, vocabs, INDEX_CHILDREN, min_fit=INDEX_CHILD_FIT,
                         tie_shortest=False) if vocabs else []
            kids = named + [u for u in got if u not in named]
            kids = kids[:INDEX_CHILDREN]
            for k in kids[: max(0, MAX_FETCHES - fetches)]:
                body = await self._get(k)
                fetches += 1
                more, _ = parse(body)
                locs.extend(more)
                if len(locs) >= MAX_LOCS:
                    break
        locs = locs[:MAX_LOCS]
        if self.store is not None:
            self.store.put(host, locs)
        return locs
