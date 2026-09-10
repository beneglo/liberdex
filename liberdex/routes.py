"""Site-native search endpoints.

Not a web index: each route is one site's own public search API. The planner
picks which routes apply and writes the query string for each one, since it
knows what a given site's search box wants far better than a generic
term-trimming heuristic does. The engine also fires a small speculative set at
t=0, on a heuristic intent guess, so the network is busy while the LLM is still
generating; those speculative firings are the only ones that fall back to the
heuristic trim.

Routes return Candidates carrying title/snippet, so a hit is useful for ranking
even if the underlying page later fails to fetch.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import quote, quote_plus

import orjson

from .query import keyword_query
from .types import Candidate

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Backends reject or silently truncate very long query strings, and a pathological
# query must never be able to build a multi-kilobyte request URL.
MAX_QUERY_CHARS = 240

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_ENT = {
    "&quot;": '"', "&#39;": "'", "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&nbsp;": " ", "&#x27;": "'",
}


def strip_html(s: str) -> str:
    s = _TAG.sub(" ", s or "")
    for k, v in _ENT.items():
        s = s.replace(k, v)
    return _WS.sub(" ", s).strip()


def _json(b: bytes) -> Any:
    try:
        return orjson.loads(b)
    except Exception:
        return None


@dataclass(slots=True)
class Route:
    name: str
    # (query, lang) -> (url, headers). lang is the ISO 639-1 code the planner
    # asked for; almost every backend ignores it.
    build: Callable[[str, str], tuple[str, dict[str, str]]]
    parse: Callable[[bytes], list[Candidate]]
    # One line, shown to the planner so it can choose routes and phrase each
    # route's query. This is the only description of the route that exists.
    help: str = ""
    # Alternate backends for the same site, tried in order when the primary
    # answers anything but 200. Reddit serves its own JSON to almost no
    # datacentre IP any more, so the route needs somewhere else to go.
    fallbacks: tuple[Callable[[str, str], tuple[str, dict[str, str]]], ...] = ()
    # Intents for which this route fires speculatively, before the LLM plan.
    speculative_for: tuple[str, ...] = ()
    enabled: bool = True
    # Backends differ wildly in how many ANDed terms they tolerate before
    # returning nothing at all. Only used when the planner gave us no query of
    # its own. 0 = pass the query through untouched.
    max_terms: int = 0
    # Set when the backend tells us to go away (429/quota); checked before firing.
    cooldown_until: float = 0.0
    blocks: int = 0
    # How long a first block keeps the route off. GitHub's search limit resets
    # every 60s, so parking it for five minutes throws away a working route.
    cooldown_seconds: float = 300.0

    def query_for(self, query: str, explicit: str = "", lang: str = "en") -> str:
        """The planner's own phrasing wins; otherwise trim the user's words."""
        if explicit:
            return explicit[:MAX_QUERY_CHARS]
        q = (keyword_query(query, self.max_terms, lang=lang)
             if self.max_terms else query)
        return q[:MAX_QUERY_CHARS]

    @property
    def available(self) -> bool:
        return self.enabled and time.monotonic() >= self.cooldown_until

    def penalize(self, status: int, seconds: float = 0.0) -> None:
        """A backend that rate-limits us stays off for a while, engine-wide.

        Backing off doubles each consecutive time. Reddit's search JSON, for
        one, serves a handful of requests from an IP and then 403s everything
        for a long while; retrying it once a query is pure wasted latency.
        """
        if status in (400, 401, 403, 420, 429, 451, 503):
            self.blocks = min(self.blocks + 1, 5)
            base = seconds or self.cooldown_seconds
            self.cooldown_until = time.monotonic() + base * (2 ** (self.blocks - 1))
        elif status == 200:
            self.blocks = 0


# ---------------------------------------------------------------- wikipedia
# A shape check is not enough: "zz" looks like a language code and resolves to
# nothing, and a DNS failure costs the whole route. These are the editions with
# enough articles to be worth answering from; anything else falls back to en.
WIKI_LANGS = frozenset("""
ar az be bg bn bs ca ceb cs cy da de el en eo es et eu fa fi fr ga gl he hi hr
hu hy id is it ja ka kk ko la lt lv mk ml mr ms my nl nn no pl pt ro ru sh simple
sk sl sq sr sv sw ta te th tl tr uk ur uz vi war zh
""".split())


def _wiki_host(lang: str) -> str:
    lang = (lang or "en").lower()
    return f"{lang}.wikipedia.org" if lang in WIKI_LANGS else "en.wikipedia.org"


def _wikipedia_build(q: str, lang: str = "en") -> tuple[str, dict[str, str]]:
    host = _wiki_host(lang)
    return (
        f"https://{host}/w/api.php?action=query&list=search"
        f"&srsearch={quote_plus(q)}&srlimit=6&srprop=snippet%7Ctimestamp"
        "&format=json&formatversion=2",
        {"User-Agent": UA, "Accept": "application/json"},
    )


def _wikipedia_parse_for(host: str):
    def parse(b: bytes) -> list[Candidate]:
        d = _json(b) or {}
        out: list[Candidate] = []
        for i, hit in enumerate((d.get("query") or {}).get("search") or []):
            title = hit.get("title") or ""
            if not title:
                continue
            out.append(Candidate(
                url=f"https://{host}/wiki/" + quote(title.replace(" ", "_")),
                title=title,
                snippet=strip_html(hit.get("snippet", "")),
                reason="wikipedia full-text search",
                prior=0.78 - i * 0.04,
                source="route:wikipedia",
            ))
        return out
    return parse


def _wikipedia_parse(b: bytes) -> list[Candidate]:
    # Default host; the engine swaps in a language-aware parser when the plan
    # asks for one.
    return _wikipedia_parse_for("en.wikipedia.org")(b)


# ----------------------------------------------------------------- hn
def _hn_build(q: str, lang: str = "en") -> tuple[str, dict[str, str]]:
    return (
        f"https://hn.algolia.com/api/v1/search?query={quote_plus(q)}"
        "&hitsPerPage=12&tags=story",
        {"User-Agent": UA},
    )


def _hn_parse(b: bytes) -> list[Candidate]:
    d = _json(b) or {}
    out: list[Candidate] = []
    for i, hit in enumerate(d.get("hits") or []):
        url = hit.get("url") or hit.get("story_url")
        title = hit.get("title") or hit.get("story_title") or ""
        if not url or not title:
            continue
        pts = hit.get("points") or 0
        out.append(Candidate(
            url=url,
            title=title,
            snippet=strip_html(hit.get("story_text") or ""),
            reason=f"HN submission, {pts} points",
            prior=min(0.82, 0.42 + pts / 400.0) - i * 0.01,
            source="route:hn",
        ))
    return out


# ------------------------------------------------------------ stackoverflow
def _so_build(q: str, lang: str = "en") -> tuple[str, dict[str, str]]:
    # Anonymous callers get 300 requests per IP per day; a (free, no-signup)
    # app key raises that to 10,000 and is the difference between the route
    # working all day and dying before lunch.
    key = os.environ.get("STACKEXCHANGE_KEY") or os.environ.get("STACKAPPS_KEY")
    return (
        "https://api.stackexchange.com/2.3/search/advanced"
        f"?order=desc&sort=relevance&q={quote_plus(q)}&site=stackoverflow"
        "&pagesize=8&filter=withbody" + (f"&key={quote_plus(key)}" if key else ""),
        {"User-Agent": UA, "Accept-Encoding": "gzip"},
    )


def _so_parse(b: bytes) -> list[Candidate]:
    d = _json(b) or {}
    # The API asks, in-band, to be left alone for N seconds; ignoring it is
    # what earns a hard block rather than a throttle.
    backoff = d.get("backoff")
    if isinstance(backoff, (int, float)) and backoff > 0:
        r = ROUTES.get("stackoverflow")
        if r is not None:
            r.cooldown_until = max(r.cooldown_until, time.monotonic() + float(backoff))
    out: list[Candidate] = []
    for i, it in enumerate(d.get("items") or []):
        link = it.get("link")
        if not link:
            continue
        score = it.get("score") or 0
        answered = bool(it.get("is_answered"))
        body = strip_html(it.get("body") or "")
        out.append(Candidate(
            url=link,
            title=strip_html(it.get("title", "")),
            snippet=body[:400],
            content=body[:12000],
            # stackoverflow.com 403s every non-browser client, so the API body
            # is the only content we will ever get for these.
            skip_fetch=bool(body),
            reason=f"Stack Overflow, score {score}" + (", answered" if answered else ""),
            prior=min(0.86, 0.5 + score / 200.0) + (0.05 if answered else 0.0) - i * 0.01,
            source="route:stackoverflow",
        ))
    return out


# ------------------------------------------------------------------- github
def _gh_headers() -> dict[str, str]:
    h = {"User-Agent": UA, "Accept": "application/vnd.github+json"}
    # Unauthenticated code search is 10 requests/minute for the whole host,
    # which a handful of concurrent queries exhausts. A token raises it to 30.
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _gh_build(q: str, lang: str = "en") -> tuple[str, dict[str, str]]:
    return (
        f"https://api.github.com/search/repositories?q={quote_plus(q)}&per_page=6",
        _gh_headers(),
    )


def _gh_parse(b: bytes) -> list[Candidate]:
    d = _json(b) or {}
    out: list[Candidate] = []
    for i, it in enumerate(d.get("items") or []):
        url = it.get("html_url")
        if not url:
            continue
        stars = it.get("stargazers_count") or 0
        # Repo search returns hobby projects whose README keyword-matches the
        # query. Popularity is the only cheap signal separating the project
        # everyone uses from someone's weekend script, so the floor is here and
        # the prior's curve below is steep.
        if stars < 25:
            continue
        out.append(Candidate(
            url=url,
            title=it.get("full_name") or url,
            snippet=it.get("description") or "",
            reason=f"GitHub repo, {stars} stars",
            prior=0.22 + 0.60 * min(1.0, (stars ** 0.5) / 70.0) - i * 0.01,
            source="route:github",
        ))
    return out


# -------------------------------------------------------------------- arxiv
_ARX_ENTRY = re.compile(r"<entry>(.*?)</entry>", re.S)
_ARX_ID = re.compile(r"<id>(.*?)</id>", re.S)
_ARX_TITLE = re.compile(r"<title>(.*?)</title>", re.S)
_ARX_SUM = re.compile(r"<summary>(.*?)</summary>", re.S)
_ARX_PUB = re.compile(r"<published>(.*?)</published>", re.S)


def _arxiv_build(q: str, lang: str = "en") -> tuple[str, dict[str, str]]:
    return (
        "https://export.arxiv.org/api/query?search_query=all:"
        f"{quote_plus(q)}&start=0&max_results=6&sortBy=relevance",
        {"User-Agent": UA},
    )


def _arxiv_parse(b: bytes) -> list[Candidate]:
    s = b.decode("utf-8", "ignore")
    out: list[Candidate] = []
    for i, ent in enumerate(_ARX_ENTRY.findall(s)):
        mid = _ARX_ID.search(ent)
        if not mid:
            continue
        mt, ms, mp = _ARX_TITLE.search(ent), _ARX_SUM.search(ent), _ARX_PUB.search(ent)
        out.append(Candidate(
            url=mid.group(1).strip().replace("http://", "https://", 1),
            title=strip_html(mt.group(1)) if mt else "",
            snippet=strip_html(ms.group(1))[:600] if ms else "",
            reason="arXiv relevance search" + (f" ({mp.group(1)[:10]})" if mp else ""),
            prior=0.70 - i * 0.03,
            source="route:arxiv",
            published=mp.group(1)[:10] if mp else "",
        ))
    return out


# ------------------------------------------------------- semantic scholar
def _s2_build(q: str, lang: str = "en") -> tuple[str, dict[str, str]]:
    return (
        "https://api.semanticscholar.org/graph/v1/paper/search?query="
        f"{quote_plus(q)}&limit=6&fields=title,abstract,year,url,citationCount",
        {"User-Agent": UA},
    )


def _s2_parse(b: bytes) -> list[Candidate]:
    d = _json(b) or {}
    out: list[Candidate] = []
    for i, p in enumerate(d.get("data") or []):
        url = p.get("url")
        if not url:
            continue
        cites = p.get("citationCount") or 0
        out.append(Candidate(
            url=url,
            title=p.get("title") or url,
            snippet=(p.get("abstract") or "")[:600],
            reason=f"Semantic Scholar, {cites} citations ({p.get('year')})",
            prior=min(0.82, 0.5 + (cites ** 0.5) / 200.0) - i * 0.02,
            source="route:semanticscholar",
            published=str(p.get("year") or ""),
        ))
    return out


# ------------------------------------------------------------------- pubmed
def _pubmed_build(q: str, lang: str = "en") -> tuple[str, dict[str, str]]:
    return (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term="
        f"{quote_plus(q)}&retmode=json&retmax=6&sort=relevance",
        {"User-Agent": UA},
    )


def _pubmed_parse(b: bytes) -> list[Candidate]:
    d = _json(b) or {}
    ids = ((d.get("esearchresult") or {}).get("idlist")) or []
    return [
        Candidate(url=f"https://pubmed.ncbi.nlm.nih.gov/{i}/",
                  reason="PubMed relevance search", prior=0.66, source="route:pubmed")
        for i in ids[:6]
    ]


# ---------------------------------------------------------------------- mdn
def _mdn_build(q: str, lang: str = "en") -> tuple[str, dict[str, str]]:
    return (
        f"https://developer.mozilla.org/api/v1/search?q={quote_plus(q)}&locale=en-US",
        {"User-Agent": UA},
    )


def _mdn_parse(b: bytes) -> list[Candidate]:
    d = _json(b) or {}
    out: list[Candidate] = []
    for i, doc in enumerate((d.get("documents") or [])[:6]):
        slug = doc.get("mdn_url")
        if not slug:
            continue
        out.append(Candidate(
            url="https://developer.mozilla.org" + slug,
            title=doc.get("title") or slug,
            snippet=doc.get("summary") or "",
            reason="MDN web docs search",
            prior=0.74 - i * 0.03,
            source="route:mdn",
        ))
    return out


# --------------------------------------------------------- package registries
def _npm_build(q: str, lang: str = "en") -> tuple[str, dict[str, str]]:
    return (f"https://registry.npmjs.org/-/v1/search?text={quote_plus(q)}&size=6",
            {"User-Agent": UA})


def _npm_parse(b: bytes) -> list[Candidate]:
    d = _json(b) or {}
    out: list[Candidate] = []
    for i, o in enumerate(d.get("objects") or []):
        p = o.get("package") or {}
        name = p.get("name")
        if not name:
            continue
        out.append(Candidate(
            url=(p.get("links") or {}).get("npm") or f"https://www.npmjs.com/package/{name}",
            title=f"{name} - npm",
            snippet=p.get("description") or "",
            reason="npm registry search",
            prior=0.62 - i * 0.03,
            source="route:npm",
        ))
    return out


_PKG_STOP = {
    "python", "pypi", "package", "library", "module", "install", "how", "to",
    "the", "in", "for", "with", "use", "using", "a", "an", "of", "and", "docs",
}


def _pkg_token(q: str) -> str:
    toks = re.findall(r"[A-Za-z0-9_.\-]{2,}", q)
    cands = [t for t in toks if t.lower() not in _PKG_STOP]
    return cands[0] if cands else (toks[0] if toks else q)


def _pypi_build(q: str, lang: str = "en") -> tuple[str, dict[str, str]]:
    # PyPI retired its search API; resolve the most package-like token instead.
    # The planner is told to pass the bare distribution name.
    return (f"https://pypi.org/pypi/{quote(_pkg_token(q))}/json", {"User-Agent": UA})


def _pypi_parse(b: bytes) -> list[Candidate]:
    info = (_json(b) or {}).get("info") or {}
    name = info.get("name")
    if not name:
        return []
    return [Candidate(
        url=f"https://pypi.org/project/{name}/",
        title=f"{name} - PyPI",
        snippet=(info.get("summary") or "")[:400],
        reason="PyPI package metadata",
        prior=0.66,
        source="route:pypi",
    )]


def _crates_build(q: str, lang: str = "en") -> tuple[str, dict[str, str]]:
    return (f"https://crates.io/api/v1/crates?q={quote_plus(q)}&per_page=5",
            {"User-Agent": UA})


def _crates_parse(b: bytes) -> list[Candidate]:
    d = _json(b) or {}
    out: list[Candidate] = []
    for i, c in enumerate(d.get("crates") or []):
        n = c.get("name")
        if not n:
            continue
        out.append(Candidate(
            url=f"https://crates.io/crates/{n}",
            title=f"{n} - crates.io",
            snippet=c.get("description") or "",
            reason="crates.io search",
            prior=0.62 - i * 0.03,
            source="route:crates",
        ))
    return out


# --------------------------------------------------------------- openlibrary
def _openlibrary_build(q: str, lang: str = "en") -> tuple[str, dict[str, str]]:
    return (
        f"https://openlibrary.org/search.json?q={quote_plus(q)}&limit=5"
        "&fields=title,author_name,key,first_publish_year",
        {"User-Agent": UA},
    )


def _openlibrary_parse(b: bytes) -> list[Candidate]:
    d = _json(b) or {}
    out: list[Candidate] = []
    for i, doc in enumerate(d.get("docs") or []):
        key = doc.get("key")
        if not key:
            continue
        auth = ", ".join((doc.get("author_name") or [])[:2])
        out.append(Candidate(
            url="https://openlibrary.org" + key,
            title=doc.get("title") or key,
            snippet=f"{auth} ({doc.get('first_publish_year', '')})",
            reason="Open Library search",
            prior=0.60 - i * 0.03,
            source="route:openlibrary",
        ))
    return out


# ------------------------------------------------------------------- reddit
_SUBREDDIT = re.compile(r"^\s*(?:/?r/)([A-Za-z0-9_]{2,24})\b[:,]?\s*(.*)$")


def _reddit_date(created: Any) -> str:
    try:
        return time.strftime("%Y-%m-%d", time.gmtime(float(created)))
    except (TypeError, ValueError, OSError, OverflowError):
        return ""


def _reddit_scope(q: str) -> tuple[str, str]:
    """Split a leading `r/<sub>` off the query, which is how the planner
    narrows a broad question to the right community."""
    m = _SUBREDDIT.match(q)
    if m and m.group(2).strip():
        return m.group(1), m.group(2).strip()
    return "", q


def _reddit_build(q: str, lang: str = "en") -> tuple[str, dict[str, str]]:
    # old.reddit.com/search.json redirects to a login wall, and www serves
    # this JSON to some clients from some networks and 403s the rest. When it
    # answers it is the freshest source there is, so it stays first and the
    # PullPush fallback catches the refusal.
    scope, terms = _reddit_scope(q)
    base = (f"https://www.reddit.com/r/{quote(scope)}/search.json?restrict_sr=1&"
            if scope else "https://www.reddit.com/search.json?")
    return (
        f"{base}q={quote_plus(terms)}&limit=10&sort=relevance&raw_json=1",
        {"User-Agent": UA, "Accept": "application/json"},
    )


def _reddit_pullpush_build(q: str, lang: str = "en") -> tuple[str, dict[str, str]]:
    """PullPush, the surviving Pushshift mirror: the same post objects, from an
    archive, served to anyone. It lags live Reddit by months, so results carry
    real dates and let the freshness prior discount them honestly."""
    scope, terms = _reddit_scope(q)
    return (
        "https://api.pullpush.io/reddit/search/submission/"
        f"?q={quote_plus(terms)}&size=10&sort=desc&sort_type=score"
        + (f"&subreddit={quote_plus(scope)}" if scope else ""),
        {"User-Agent": UA, "Accept": "application/json"},
    )


def _reddit_posts(d: Any) -> list[dict]:
    """Both backends hand back the same post objects, wrapped differently:
    Reddit nests them under data.children[].data, PullPush lists them flat."""
    data = (d or {}).get("data")
    if isinstance(data, list):
        return [p for p in data if isinstance(p, dict)]
    if isinstance(data, dict):
        return [
            ch.get("data") or {}
            for ch in (data.get("children") or []) if isinstance(ch, dict)
        ]
    return []


def _reddit_parse(b: bytes) -> list[Candidate]:
    d = _json(b) or {}
    out: list[Candidate] = []
    for i, p in enumerate(_reddit_posts(d)):
        perma = p.get("permalink")
        if not perma:
            continue
        ups = p.get("ups") or p.get("score") or 0
        body = (p.get("selftext") or "").strip()
        title = p.get("title") or ""
        # www.reddit.com 403s a datacentre client on the HTML page, so the
        # listing JSON is usually the only text we will ever have for a thread.
        text = (title + ". " + body).strip() if body else ""
        out.append(Candidate(
            url="https://www.reddit.com" + perma,
            title=title,
            snippet=body[:400] or (p.get("link_flair_text") or ""),
            content=text[:12000],
            skip_fetch=bool(text) and len(text) > 300,
            reason=f"Reddit r/{p.get('subreddit', '')}, {ups} upvotes",
            prior=min(0.72, 0.40 + ups / 3000.0) - i * 0.01,
            source="route:reddit",
            published=_reddit_date(p.get("created_utc")),
        ))
    return out


# ------------------------------------------------------------------ youtube
_YT_DATA = re.compile(r"ytInitialData\s*=\s*(\{.*?\})\s*;\s*</script>", re.S)


def _youtube_build(q: str, lang: str = "en") -> tuple[str, dict[str, str]]:
    return (
        f"https://www.youtube.com/results?search_query={quote_plus(q)}",
        {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"},
    )


def _yt_runs(node: Any) -> str:
    if isinstance(node, dict):
        if "simpleText" in node:
            return str(node["simpleText"])
        runs = node.get("runs")
        if isinstance(runs, list):
            return "".join(str(r.get("text", "")) for r in runs if isinstance(r, dict))
    return ""


def _youtube_parse(b: bytes) -> list[Candidate]:
    m = _YT_DATA.search(b.decode("utf-8", "ignore"))
    if not m:
        return []
    try:
        data = orjson.loads(m.group(1))
    except Exception:
        return []
    out: list[Candidate] = []
    stack: list[Any] = [data]
    seen: set[str] = set()
    while stack and len(out) < 6:
        cur = stack.pop()
        if isinstance(cur, dict):
            vr = cur.get("videoRenderer")
            if isinstance(vr, dict):
                vid = vr.get("videoId")
                if isinstance(vid, str) and vid not in seen:
                    seen.add(vid)
                    title = _yt_runs(vr.get("title"))
                    desc = " ".join(
                        _yt_runs(s.get("snippetText"))
                        for s in (vr.get("detailedMetadataSnippets") or [])
                        if isinstance(s, dict)
                    ) or _yt_runs(vr.get("descriptionSnippet"))
                    chan = _yt_runs((vr.get("ownerText") or {}))
                    views = _yt_runs(vr.get("viewCountText"))
                    when = _yt_runs(vr.get("publishedTimeText"))
                    length = _yt_runs(vr.get("lengthText"))
                    meta = " · ".join(x for x in (chan, views, when, length) if x)
                    body = ". ".join(x for x in (title, desc, meta) if x)
                    out.append(Candidate(
                        url=f"https://www.youtube.com/watch?v={vid}",
                        title=title,
                        snippet=(desc or meta)[:400],
                        # The watch page is ~1.5 MB of player JSON with almost no
                        # prose in it; the search result already carries every
                        # word we would get from fetching it.
                        content=body[:2000],
                        skip_fetch=True,
                        reason=f"YouTube search{(' · ' + meta) if meta else ''}",
                        prior=0.55 - len(out) * 0.03,
                        source="route:youtube",
                    ))
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return out


ROUTES: dict[str, Route] = {r.name: r for r in [
    Route("wikipedia", _wikipedia_build, _wikipedia_parse, max_terms=3,
          help="encyclopedia full-text search; give it the article title, 2-4 words, "
               "not a question. Follows L for the language edition.",
          # Not `product` or `local`: the encyclopedia has an article on the
          # category and never an answer to "which one should I buy" or "what is
          # near me". The planner can still ask for it by name.
          speculative_for=("informational", "reference", "navigational", "academic",
                           "news")),
    Route("hn", _hn_build, _hn_parse, max_terms=3,
          help="Hacker News stories: practitioner discussion, launches, postmortems. "
               "Title keywords only.",
          speculative_for=("code", "news")),
    Route("stackoverflow", _so_build, _so_parse, max_terms=6,
          help="Stack Overflow Q&A: concrete programming problems and error messages. "
               "Include the exact API or error text.",
          speculative_for=("code",)),
    Route("github", _gh_build, _gh_parse, max_terms=3, cooldown_seconds=60.0,
          help="GitHub repository search (repos, not code or issues): finds the "
               "project itself. Give the tool or library name.",
          speculative_for=("code",)),
    Route("arxiv", _arxiv_build, _arxiv_parse, max_terms=5,
          help="arXiv preprints: physics, maths, CS, quantitative biology. Use paper "
               "vocabulary, e.g. method and task names.",
          speculative_for=("academic",)),
    # Semantic Scholar 429s hard without an API key; opt in explicitly.
    Route("semanticscholar", _s2_build, _s2_parse, max_terms=6, enabled=False,
          help="Semantic Scholar: peer-reviewed papers across all fields."),
    Route("pubmed", _pubmed_build, _pubmed_parse, max_terms=5,
          help="PubMed: biomedical and clinical literature. Use MeSH-style terms."),
    Route("mdn", _mdn_build, _mdn_parse, max_terms=4,
          help="MDN: the reference for HTML, CSS, JavaScript and browser APIs. "
               "Give the API or property name."),
    Route("npm", _npm_build, _npm_parse, max_terms=3,
          help="npm registry: JavaScript packages. Give the package name or purpose."),
    Route("pypi", _pypi_build, _pypi_parse,
          help="PyPI: pass the bare Python distribution name and nothing else."),
    Route("crates", _crates_build, _crates_parse, max_terms=3,
          help="crates.io: Rust crates. Give the crate name or purpose."),
    Route("openlibrary", _openlibrary_build, _openlibrary_parse, max_terms=4,
          help="Open Library: books by title or author."),
    Route("reddit", _reddit_build, _reddit_parse, max_terms=5,
          fallbacks=(_reddit_pullpush_build,),
          help="Reddit thread search: lived experience, recommendations, "
               "troubleshooting. Prefix with `r/<subreddit>` to scope it, e.g. "
               "`r/rust async trait`.",
          # "best X", "X vs Y", "is X worth it" is the one intent where a
          # forum thread beats every encyclopedia and every vendor page, and
          # it is worth having before the plan lands rather than after.
          speculative_for=("product",)),
    Route("youtube", _youtube_build, _youtube_parse, max_terms=5,
          help="YouTube video search: use when the answer is genuinely a video "
               "(a demo, a talk, a physical how-to, a walkthrough)."),
]}

ROUTE_HELP = "\n".join(
    f"    {r.name:<16}{r.help}"
    for r in sorted(ROUTES.values(), key=lambda x: x.name) if r.enabled
)


def route_parser(name: str, lang: str = "en"):
    """The parser for a route, specialised to the plan's language where it matters."""
    if name == "wikipedia":
        return _wikipedia_parse_for(_wiki_host(lang))
    return ROUTES[name].parse


def speculative_routes(intent_guess: str) -> list[Route]:
    return [r for r in ROUTES.values() if r.available and intent_guess in r.speculative_for]
