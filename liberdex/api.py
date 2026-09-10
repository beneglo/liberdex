"""The HTTP wire shape, and the one translation away from it.

liberdex's own vocabulary is the native shape: a query produces a plan, the
plan names candidates, routes and hubs, those become pages, and the ranker cuts
each page into passages. Everything a caller sees is one of those words, because
they are what the engine does. A caller that wants different field names maps
them on its own side.
"""
from __future__ import annotations

import datetime
import time
import uuid
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from .types import Plan, Result, SearchResponse

Depth = Literal["fast", "standard", "deep"]
Format = Literal["text", "markdown"]
AnswerMode = Literal["extract", "synthesize"]
Intent = Literal[
    "navigational", "informational", "code", "academic",
    "news", "product", "local", "reference",
]
Freshness = Literal["day", "week", "month", "year"]

# The three depth tiers. `fast` reads fewer planner URLs, skips hub expansion
# and stops waiting sooner: a real trade of a few points of answer recall for
# half the latency on factoid queries. `deep` is standard plus a second
# planning pass (`refine`) over the first round's report, which is one more
# LLM call and the reason it may take twice as long; a longer wait alone buys
# nothing, since the candidate supply runs out before the budget does. `budget`
# is the fetch window, which starts at the planner's first URL (PLAN_GRACE).
DEPTH: dict[str, dict[str, Any]] = {
    "fast": dict(budget=2.5, max_llm_candidates=6, expand_hubs=False,
                 enough_docs=16, fetch_tail_window=0.9, max_fetch=28),
    "standard": dict(budget=6.0, max_llm_candidates=10, expand_hubs=True,
                     enough_docs=26, fetch_tail_window=1.6, max_fetch=44),
    "deep": dict(budget=24.0, max_llm_candidates=0, expand_hubs=True,
                 enough_docs=48, fetch_tail_window=4.0, max_fetch=72,
                 refine=True),
}
# A page past the first is a new round: the planner is asked, over what has
# been shown, for what comes next. It gets the time that round costs on top of
# the tier's own budget, or the second pass is declined for lack of headroom.
PAGE_EXTRA_BUDGET = 10.0
MAX_PAGE = 10

# How long each tier waits for the planner's first URL before the fetch window
# starts. The tier's `budget` is that window. The grace is granted up front, on
# top of it, and handed back the moment the first URL lands: a cache hit at 4ms
# costs nothing, a cold hosted planner gets its seconds to first token without
# them coming out of the fetch wave. Beside DEPTH rather than in it: DEPTH's
# other keys are engine arguments that callers splat as they are.
PLAN_GRACE: dict[str, float] = {"fast": 1.0, "standard": 4.5, "deep": 6.0}


def size_budget(kw: dict[str, Any], planner: Any, *, depth: str,
                explicit: bool, supplied: bool = False) -> dict[str, Any]:
    """The deadline of one search: tier budget, planner grace, planner floor.

    One rule for the CLI, the server, the MCP server and the host plugins. An
    explicit budget is the caller's wall clock and gets no grace. A supplied
    plan runs no planner and gets none either. A whole-reply planner declares
    `min_budget`, the time before its first URL can exist at all: the budget
    grows to it, and grace on top would count the same wait twice.
    """
    kw["plan_grace"] = 0.0
    if explicit or supplied:
        return kw
    floor = float(getattr(planner, "min_budget", 0.0) or 0.0)
    if floor > kw["budget"]:
        kw["budget"] = floor
    else:
        kw["plan_grace"] = PLAN_GRACE.get(depth, 0.0)
    return kw

_FRESHNESS_DAYS = {"day": 1, "week": 7, "month": 31, "year": 366}


def request_id() -> str:
    return uuid.uuid4().hex[:16]


def freshness_floor(window: Optional[str]) -> str:
    """`"week"` -> the ISO date one week ago, or "" for no window."""
    days = _FRESHNESS_DAYS.get(window or "")
    if not days:
        return ""
    return (datetime.date.today() - datetime.timedelta(days=days)).isoformat()


# --------------------------------------------------------------------- search
class PlanIn(BaseModel):
    """A plan the caller wrote, in place of the one liberdex would ask for.

    The agent on the other side of an MCP call is already a model with the web
    in its memory; when it writes the plan, the search costs no LLM call and
    starts fetching at t=0. Fields are the plan protocol in liberdex's words.
    Every URL guard the parser applies to a streamed plan applies here.
    """

    candidates: list[Union[str, dict[str, Any]]] = Field(default_factory=list,
                                                        max_length=40)
    # route name -> the query to send that site's own search box.
    routes: Union[dict[str, str], list[str]] = Field(default_factory=dict)
    hubs: list[str] = Field(default_factory=list, max_length=6)
    expansions: list[str] = Field(default_factory=list, max_length=8)
    intent: Optional[Intent] = None
    lang: Optional[str] = Field(default=None, max_length=8)

    def to_plan(self, query: str) -> Plan:
        from .planner.base import plan_from_dict
        return plan_from_dict(query, self.model_dump(exclude_none=True))


class SearchRequest(BaseModel):
    """One search. `query` may be a list: liberdex runs them concurrently."""

    query: Union[str, list[str], None] = None
    # Seed the plan from a page instead of a phrase. Not embedding-neighbour
    # retrieval: there is no index to find neighbours in. The page is fetched,
    # read, and turned into a query.
    like: Optional[str] = None

    top_k: int = Field(default=10, ge=1, le=25)
    depth: Depth = "standard"
    budget: Optional[float] = Field(default=None, ge=0.5, le=60.0)
    # Result pages. There is no result set to slice: a page past the first is
    # a new round that never dispatches or returns `exclude_urls`, the pages the
    # caller already has, and asks the planner over them for what comes next.
    # Send every URL from every earlier page.
    page: int = Field(default=1, ge=1, le=MAX_PAGE)
    exclude_urls: list[str] = Field(default_factory=list, max_length=250)

    # --- what comes back ---
    include_text: bool = False
    include_plan: bool = False
    include_debug: bool = False
    include_favicon: bool = False
    include_images: bool = False
    format: Format = "text"
    # `passage_chars` is per passage, so total shipped text is
    # passages_per_page * passage_chars. Three windows put the answer in the
    # shipped SERP far more often than one, and three is what comparable APIs
    # default to.
    passages_per_page: int = Field(default=3, ge=1, le=8)
    passage_chars: int = Field(default=500, ge=80, le=8000)

    # --- what gets fetched ---
    include_domains: list[str] = Field(default_factory=list, max_length=300)
    exclude_domains: list[str] = Field(default_factory=list, max_length=300)
    published_after: Optional[str] = None
    published_before: Optional[str] = None
    freshness: Optional[Freshness] = None
    require_date: bool = False
    intent: Optional[Intent] = None

    # --- how much of it ---
    token_budget: int = Field(default=0, ge=0, le=200_000)
    token_budget_per_page: int = Field(default=0, ge=0, le=100_000)

    # --- how it is ranked ---
    # Named overrides on top of the intent's own ranking weights. Not a rule
    # language: it can reweight the signals the fusion already has, not add one.
    weights: dict[str, float] = Field(default_factory=dict)

    # --- the answer layer ---
    # The answer model reads the top pages whole and lists the ones about the
    # question; the SERP is ordered by that list (see answer.judged).
    answer: Union[bool, AnswerMode] = False
    answer_model: Optional[str] = None
    # A JSON Schema the answer must satisfy, instead of prose.
    output_schema: Optional[dict[str, Any]] = None

    # --- the plan, if the caller wrote it ---
    # Given, the configured planner is not called: the candidates, routes and
    # hubs here are what gets fetched. Site-native routes the heuristics pick
    # still fire.
    plan: Optional[PlanIn] = None

    @model_validator(mode="after")
    def _settle(self) -> "SearchRequest":
        # Markdown is a rendering of the page text, so asking for it without
        # asking for text is a request for nothing. Passages are selected
        # during ranking, before the conversion, and stay plain either way.
        if self.format == "markdown":
            self.include_text = True
        # A query of spaces is no query; caught here so the engine never
        # answers it with an empty 200.
        if isinstance(self.query, str):
            self.query = self.query.strip()
        if isinstance(self.like, str):
            self.like = self.like.strip() or None
        if not self.query and not self.like:
            raise ValueError("one of `query` or `like` is required")
        if isinstance(self.query, list):
            self.query = [q for q in self.query if q and q.strip()][:8]
            if not self.query:
                raise ValueError("`query` list was empty")
        if self.freshness and not self.published_after:
            self.published_after = freshness_floor(self.freshness)
        # A date bound the engine cannot read would be dropped, and a filter
        # that silently stops filtering is worse than a refusal.
        from .rank import parse_date
        for field in ("published_after", "published_before"):
            raw = getattr(self, field)
            if raw and parse_date(raw) is None:
                raise ValueError(f"{field}: not a date: {raw!r} (use YYYY-MM-DD)")
        if self.weights:
            from .rank import WEIGHT_FIELDS
            unknown = sorted(set(self.weights) - WEIGHT_FIELDS)
            if unknown:
                raise ValueError(
                    f"unknown ranking weights: {', '.join(unknown)}; "
                    f"known: {', '.join(sorted(WEIGHT_FIELDS))}")
        if self.output_schema and not self.answer:
            # A schema with no answer mode is a request for typed nothing.
            self.answer = "extract"
        if self.plan is not None and isinstance(self.query, list):
            raise ValueError("`plan` goes with one `query`, not a list")
        return self

    def queries(self) -> list[str]:
        if isinstance(self.query, list):
            return self.query
        return [self.query] if self.query else []

    def engine_kwargs(self, planner: Any = None) -> dict[str, Any]:
        """Everything `Liberdex.search` takes, resolved from the tier up.

        `planner` is the one that will run, so the deadline can be sized for
        it (see `size_budget`); None sizes for a streaming hosted planner.
        """
        tier = dict(DEPTH[self.depth])
        if self.budget:
            tier["budget"] = self.budget
        elif self.page > 1:
            tier["budget"] = tier["budget"] + PAGE_EXTRA_BUDGET
        size_budget(tier, planner, depth=self.depth, explicit=bool(self.budget),
                    supplied=self.plan is not None)
        return dict(
            tier,
            top_k=self.top_k,
            page=self.page,
            exclude_urls=[u for u in self.exclude_urls if u and u.strip()],
            # The answer model reads the page, not a window of it, so text is kept
            # whenever an answer is wanted; the reply still omits it unless asked.
            keep_text=self.include_text or bool(self.answer),
            include_plan=self.include_plan,
            # The ranker splits one character budget across the windows,
            # so a per-passage size becomes a total here.
            snippet_chars=self.passage_chars * self.passages_per_page,
            snippet_windows=self.passages_per_page,
            include_domains=self.include_domains,
            exclude_domains=self.exclude_domains,
            published_after=self.published_after,
            published_before=self.published_before,
            require_date=self.require_date,
            intent=self.intent,
            want_media=self.include_favicon or self.include_images,
            markdown=self.format == "markdown",
            token_budget=self.token_budget,
            token_budget_per_page=self.token_budget_per_page,
            weights=self.weights or None,
            plan=(self.plan.to_plan(self.query if isinstance(self.query, str) else "")
                  if self.plan is not None else None),
        )


class SearchResult(BaseModel):
    url: str
    title: str
    site: str
    # Fusion score: what the SERP is ordered by, comparable within one query.
    score: float
    # Absolute relevance in 0..1, comparable across queries. Threshold on this.
    relevance: float = 0.0
    # The query-selected windows of the page, in document order, each scored by
    # the fraction of the query's distinct terms it covers.
    passages: list[str] = Field(default_factory=list)
    passage_scores: list[float] = Field(default_factory=list)
    published: str = ""
    # Where this row came from: the planner's memory, a site-native route, or a
    # hub's outbound links. No index engine can report this.
    source: str = ""
    status: int = 0
    text: Optional[str] = None
    # The whole page as read, in characters, whether or not `text` was
    # shipped or was cut to fit a budget. What reading it in full would cost.
    text_chars: int = 0
    favicon: Optional[str] = None
    images: Optional[list[str]] = None
    debug: Optional[dict[str, Any]] = None


class PlanView(BaseModel):
    intent: str = ""
    lang: str = ""
    expansions: list[str] = Field(default_factory=list)
    routes: list[str] = Field(default_factory=list)
    route_queries: dict[str, str] = Field(default_factory=dict)
    hubs: list[str] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)


class SearchReply(BaseModel):
    request_id: str
    query: str
    intent: str
    page: int = 1
    lang: str = ""
    answer: Optional[str] = None
    results: list[SearchResult] = Field(default_factory=list)
    plan: Optional[PlanView] = None
    timings: dict[str, float] = Field(default_factory=dict)
    stats: dict[str, Any] = Field(default_factory=dict)


def plan_view(plan: Optional[Plan]) -> Optional[PlanView]:
    if plan is None:
        return None
    # A streamed plan can repeat a URL: the model restates one, or a route and
    # the model name the same page. The engine dedupes at dispatch, so the view
    # has to as well or it reports work that never happens twice.
    seen: set[str] = set()
    candidates = []
    for c in plan.candidates:
        if c.url in seen:
            continue
        seen.add(c.url)
        candidates.append({"url": c.url, "prior": c.prior, "source": c.source,
                           "reason": c.reason})
    return PlanView(
        intent=plan.intent, lang=plan.lang, expansions=plan.expansions,
        routes=plan.routes, route_queries=plan.route_queries, hubs=plan.hubs,
        candidates=candidates,
    )


def result_view(r: Result, req: SearchRequest) -> SearchResult:
    return SearchResult(
        url=r.url, title=r.title, site=r.site, score=r.score,
        relevance=r.relevance,
        passages=r.passages or ([r.snippet] if r.snippet else []),
        passage_scores=r.passage_scores,
        published=r.published, source=r.source, status=r.status,
        text=(r.text or "") if req.include_text else None,
        text_chars=r.text_chars,
        favicon=r.favicon if req.include_favicon else None,
        images=(r.images or []) if req.include_images else None,
        debug=r.debug if req.include_debug else None,
    )


def reply(resp: SearchResponse, req: SearchRequest, *, rid: str = "",
          answer: Optional[str] = None) -> SearchReply:
    return SearchReply(
        request_id=rid or request_id(),
        query=resp.query,
        intent=resp.intent,
        page=req.page,
        lang=str(resp.stats.get("lang") or ""),
        answer=answer,
        results=[result_view(r, req) for r in resp.results],
        # Gated on the request, not on whether the engine happens to have one:
        # a caller who did not ask for the plan must not be handed it.
        plan=plan_view(resp.plan) if req.include_plan else None,
        timings=resp.timings,
        stats=resp.stats,
    )


# -------------------------------------------------------------------- extract
class ExtractRequest(BaseModel):
    """Fetch and read pages the caller already has URLs for."""

    urls: Union[str, list[str]]
    format: Format = "text"
    # Given a query, the page is cut into the same query-selected passages a
    # search result gets, instead of being returned whole.
    query: Optional[str] = None
    passages_per_page: int = Field(default=3, ge=1, le=8)
    # Per passage, as on /search. Larger than a search passage because a caller
    # who names a URL wants to read it, not to skim a ranked list of them.
    passage_chars: int = Field(default=1200, ge=80, le=20_000)
    include_favicon: bool = False
    include_images: bool = False
    include_links: bool = False
    token_budget_per_page: int = Field(default=0, ge=0, le=100_000)
    timeout: float = Field(default=15.0, ge=1.0, le=60.0)

    @model_validator(mode="after")
    def _settle(self) -> "ExtractRequest":
        urls = [self.urls] if isinstance(self.urls, str) else list(self.urls)
        urls = [u.strip() for u in urls if u and u.strip()]
        if not urls:
            raise ValueError("`urls` is required")
        if len(urls) > 20:
            raise ValueError("at most 20 urls per request")
        bad = [u for u in urls if not u.lower().startswith(("http://", "https://"))]
        if bad:
            # Otherwise a typo is dispatched and comes back as a DNS error,
            # which says the host is down rather than that this was never a URL.
            raise ValueError(f"not http(s) URLs: {', '.join(bad[:3])}")
        self.urls = urls
        return self


class ExtractedPage(BaseModel):
    url: str
    title: str = ""
    site: str = ""
    text: str = ""
    # The whole page as read, whether `text` was cut to `token_budget_per_page`
    # or replaced by passages. The budget that reads it whole follows from it.
    text_chars: int = 0
    passages: list[str] = Field(default_factory=list)
    passage_scores: list[float] = Field(default_factory=list)
    published: str = ""
    lang: str = ""
    status: int = 0
    favicon: Optional[str] = None
    images: Optional[list[str]] = None
    links: Optional[list[dict[str, str]]] = None


class FailedPage(BaseModel):
    url: str
    status: int = 0
    error: str = ""


class ExtractReply(BaseModel):
    request_id: str
    results: list[ExtractedPage] = Field(default_factory=list)
    failed: list[FailedPage] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)


async def extract(eng: Any, req: ExtractRequest) -> ExtractReply:
    """Fetch and read the pages, the same way for HTTP and for MCP."""
    from .budget import apply as budget_fit
    from .query import content_terms
    from .rank import select_windows

    timer = Timer()
    docs = await eng.extract_urls(
        req.urls, budget=req.timeout, want_links=req.include_links,
        want_media=req.include_favicon or req.include_images,
        markdown=req.format == "markdown",
    )
    terms = content_terms(req.query) if req.query else []
    ok: list[ExtractedPage] = []
    bad: list[FailedPage] = []
    for d in docs:
        if not d.ok:
            bad.append(FailedPage(url=d.url, status=d.status,
                                  error=d.error or f"HTTP {d.status}"))
            continue
        passages: list[str] = []
        scores: list[float] = []
        text = d.text or ""
        if terms:
            # With a query, a page comes back as the same query-selected passages
            # a search result gets. select_windows splits one character budget
            # across the windows, so a per-passage size becomes a total here, as
            # in SearchRequest.engine_kwargs.
            parts, _, _ = select_windows(
                text, d.description, terms,
                req.passage_chars * req.passages_per_page, req.passages_per_page)
            passages = [p for p, _ in parts]
            scores = [sc for _, sc in parts]
        page = ExtractedPage(
            url=d.final_url, title=d.title, site=d.site,
            # The passages are what was asked for; the page they were cut from
            # travels only when none of it was about the query. `text_chars`
            # still says what the whole page would cost.
            text="" if passages else text,
            text_chars=len(text), passages=passages, passage_scores=scores,
            published=d.published, lang=d.lang, status=d.status,
            favicon=d.favicon if req.include_favicon else None,
            images=(d.images or []) if req.include_images else None,
            links=([{"url": u, "anchor": a, "context": c} for u, a, c in d.links]
                   if req.include_links else None),
        )
        if req.token_budget_per_page:
            row = Result(url=page.url, title=page.title, snippet="", site=page.site,
                         score=0.0, passages=page.passages,
                         passage_scores=page.passage_scores, text=page.text)
            budget_fit([row], token_budget_per_page=req.token_budget_per_page)
            page.passages, page.passage_scores, page.text = (
                row.passages, row.passage_scores, row.text)
        ok.append(page)
    return ExtractReply(request_id=request_id(), results=ok, failed=bad,
                        timings={"total_ms": timer.ms()})


# ----------------------------------------------------------------------- plan
class PlanRequest(BaseModel):
    """Where does this live on the web? No fetching, no ranking."""

    query: str = Field(min_length=1, max_length=800)
    budget: float = Field(default=6.0, ge=0.5, le=60.0)
    max_candidates: int = Field(default=14, ge=1, le=40)


class PlanReply(BaseModel):
    request_id: str
    query: str
    plan: PlanView
    timings: dict[str, float] = Field(default_factory=dict)
    # Carries `planner_error` when the plan came back empty for a reason. An
    # empty plan is a legitimate answer; an empty plan because the planner was
    # never configured, or returned a 401, is not, and the two have to be
    # distinguishable without reading the server's logs.
    stats: dict[str, Any] = Field(default_factory=dict)


def plan_stats(plan: Optional[Plan]) -> dict[str, Any]:
    raw = (plan.raw or "") if plan else ""
    if raw.startswith("planner error: "):
        return {"planner_error": raw[len("planner error: "):]}
    return {}


def sse(rid: str, event: str, data: Any) -> str:
    """One SSE frame. The OpenAI-shaped chunk is what every agent framework
    already parses, so matching it saves every consumer a parser."""
    import orjson
    payload = {"id": rid, "object": "search.chunk",
               "created": int(time.time()), "event": event, "data": data}
    return "data: " + orjson.dumps(payload).decode() + "\n\n"


class Timer:
    """Wall-clock for one request, in the same milliseconds `timings` uses."""

    __slots__ = ("t0",)

    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def ms(self) -> float:
        return round((time.perf_counter() - self.t0) * 1000, 1)
