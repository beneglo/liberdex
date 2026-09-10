"""Core data types for liberdex."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Intent = Literal[
    "navigational",   # user wants a specific site/page
    "informational",  # general knowledge / how-does-X-work
    "code",           # programming, APIs, errors, libraries
    "academic",       # papers, research
    "news",           # recent events
    "product",        # shopping / product specs
    "local",          # places, businesses
    "reference",      # definitions, stats, docs lookup
]


@dataclass(slots=True)
class Candidate:
    """A URL the planner believes may contain the answer."""

    url: str
    # Why we think this URL is relevant; used as a weak ranking prior.
    reason: str = ""
    # Planner's own 0-1 guess at relevance.
    prior: float = 0.5
    # Where the candidate came from: "llm", "route:wikipedia", "expand", ...
    source: str = "llm"
    # Route candidates carry the title/snippet the API already gave us,
    # which lets us rank them even if the page fetch fails.
    title: str = ""
    snippet: str = ""
    published: str = ""
    # The plan's language; sets Accept-Language on the fetch so a site that
    # negotiates serves the reader's edition. Empty means English.
    lang: str = ""
    # Full text the route already returned. When a route hands back the whole
    # answer, fetching the page is pure latency, and for a site that blocks
    # datacentre clients it is the only way to get the content at all.
    content: str = ""
    skip_fetch: bool = False
    # A fixed SERP position, 1-based; 0 means ranked.
    rank: int = 0


@dataclass(slots=True)
class Plan:
    """The planner's answer to 'where does this live on the web?'"""

    query: str
    intent: Intent = "informational"
    # ISO 639-1 code of the language the answer should be in. Selects the
    # Wikipedia edition, the lexical stemmer, and the stopword list.
    lang: str = "en"
    # Keyword expansions used for lexical scoring, not for fetching.
    expansions: list[str] = field(default_factory=list)
    # Named site-native search routes to run (see liberdex.routes).
    routes: list[str] = field(default_factory=list)
    # route name -> the query string the planner wrote for that specific site.
    # A site's own search box wants different words than the user typed, and the
    # planner knows those words better than any generic trimming heuristic.
    route_queries: dict[str, str] = field(default_factory=dict)
    candidates: list[Candidate] = field(default_factory=list)
    # Pages likely to *link to* the answer; crawled one hop deep.
    hubs: list[str] = field(default_factory=list)
    raw: str = ""

    def as_dict(self) -> dict[str, Any]:
        """The plan in the shape a caller may hand back (see planner.plan_from_dict).

        Routes and their queries fold into one mapping, because to a caller a
        route without its query is half a fact.
        """
        return {
            "query": self.query,
            "intent": self.intent,
            "lang": self.lang,
            "candidates": [
                {"url": c.url, "prior": c.prior,
                 **({"reason": c.reason} if c.reason else {})}
                for c in self.candidates
            ],
            "routes": {name: self.route_queries.get(name, "") for name in self.routes},
            "hubs": list(self.hubs),
            "expansions": list(self.expansions),
        }


@dataclass(slots=True)
class Findings:
    """What one round of a search came back with, as the planner sees it.

    The first plan is written blind, from memory alone. A second one need not
    be: by the time it is asked for, the engine has fetched and read the first
    round's pages and knows which of them were about the query, which addresses
    were dead, and on a later page of results which pages the user has already
    been shown. This is that report, and it is the difference between a deep
    search and a standard one that waited longer.
    """

    # (title, url, relevance 0..1, excerpt), best first.
    found: list[tuple[str, str, float, str]] = field(default_factory=list)
    # Addresses the plan named that did not resolve to a readable page.
    dead: list[str] = field(default_factory=list)
    # Pages delivered on earlier result pages; never to be named again.
    shown: list[str] = field(default_factory=list)
    # Which page of results this round is for. 1 is the deep second pass.
    page: int = 1


@dataclass(slots=True)
class Doc:
    """A fetched + extracted document."""

    url: str
    final_url: str
    status: int
    title: str = ""
    text: str = ""
    description: str = ""
    site: str = ""
    published: str = ""
    lang: str = ""
    # The page's own `<link rel=canonical>`, absolute. Empty when it has none.
    canonical: str = ""
    content_type: str = ""
    fetch_ms: float = 0.0
    bytes_in: int = 0
    error: str = ""
    body: bytes = b""
    consensus: int = 1
    candidate: Optional[Candidate] = None
    # (url, anchor, context), where context is the text of the block the link
    # sits in. It predicts the target far better than the anchor alone.
    links: list[tuple[str, str, str]] = field(default_factory=list)
    # Parsed only when asked for: another pass over the tree per page.
    favicon: str = ""
    images: list[str] = field(default_factory=list)
    # The site's own search endpoint, as a `{}` template, read off this page's
    # search form. The nearest thing to an index that an arbitrary host offers.
    search_url: str = ""

    @property
    def ok(self) -> bool:
        return self.status == 200 and bool(self.text or self.title)


@dataclass(slots=True)
class Result:
    """One SERP row.

    `passages` are the query-selected windows of the page, each scored by the
    fraction of the query's distinct terms it covers. `snippet` is the same
    windows joined into one string.
    """
    url: str
    title: str
    snippet: str
    site: str
    score: float
    # `score` is the fusion output the SERP is ordered by: comparable within
    # one query, but a weighted sum, not a probability. `relevance` is the
    # absolute reading, a cross-encoder probability where the cross-encoder
    # ran and a cosine elsewhere, always in 0..1. Threshold on this one.
    relevance: float = 0.0
    published: str = ""
    source: str = ""
    passages: list[str] = field(default_factory=list)
    passage_scores: list[float] = field(default_factory=list)
    # Set only when the caller asked for them; extraction is not free.
    text: str = ""
    # Length of the whole page as read, whatever was shipped. A consumer
    # holding a cut `text` or passages alone learns from this what reading
    # the page in full would cost before it asks for it.
    text_chars: int = 0
    favicon: str = ""
    images: list[str] = field(default_factory=list)
    status: int = 0
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResponse:
    query: str
    intent: str
    results: list[Result]
    timings: dict[str, float] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    plan: Optional[Plan] = None


class Deadline:
    """Wall-clock budget shared across pipeline stages."""

    __slots__ = ("t0", "budget")

    def __init__(self, budget: float) -> None:
        self.t0 = time.perf_counter()
        self.budget = budget

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    @property
    def remaining(self) -> float:
        return max(0.0, self.budget - self.elapsed)

    def expired(self, reserve: float = 0.0) -> bool:
        return self.remaining <= reserve

    def extend(self, seconds: float) -> None:
        """Move the deadline by `seconds`; negative brings it forward.

        The clock keeps its origin, so `elapsed` is untouched and every reader
        of `remaining` sees the move on its next call.
        """
        self.budget += seconds
