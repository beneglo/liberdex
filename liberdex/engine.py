"""The liberdex engine.

Every stage overlaps. Site-native routes fire at t=0 off a heuristic intent
guess, so the network is busy while the planner is still generating; planner
URLs are dispatched as each line completes; extraction runs in a thread pool as
pages land. At the deadline the engine ranks what arrived and returns a thinner
SERP rather than overrunning the budget.
"""
from __future__ import annotations

import asyncio
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Sequence
from urllib.parse import quote_plus, unquote, urlsplit

from .budget import apply as budget_fit
from .extract import decode, extract, to_markdown
from .fetch import Fetcher, host_matches, registrable
from .query import _informativeness, content_terms, guess_lang, keyword_query
from .rank import (
    Models,
    Prescore,
    Ranker,
    _f1,
    best_f1,
    date_fit,
    normalize_url,
    parse_date,
    path_fit,
    query_period,
    select_windows,
    title_bag,
    tokens,
)
from .routes import ROUTES, Route, route_parser, speculative_routes
from .sitemap import pick as sitemap_pick
from .types import Candidate, Deadline, Doc, Findings, Plan, Result, SearchResponse

# Longer than this is a paste, not a question. The planner prompt and every
# route URL are built from the query, so it is clamped once, at the door.
MAX_QUERY_CHARS = 2000

# --- hub expansion ----------------------------------------------------------
# Minimum anchor fit to follow a link off a hub page, scored with the ranker's
# recall-dominant F1 against the query and each expansion separately.
HUB_LINK_FIT = 0.22
# Lower bar for a link that stays inside the hub's own section. A hub indexes
# its descendants and everything else on the page is site chrome, so in section
# a link only has to not contradict the query; outside it the anchor carries
# the whole case.
HUB_SECTION_FIT = 0.06
# Fetches spent following one hub, in section and in total. Past this the
# marginal link displaces a real result.
HUB_SECTION_LINKS = 5
HUB_LINKS = 5
# Hosts asked to search their own content, one request each.
SITE_SEARCH_HOSTS = 3
# A site's own results page is a denser hub than an index page: every link on
# it is a document the site believes matches the query.
SITE_SEARCH_LINKS = 6
# Discovering an endpoint costs an unbudgeted round trip, so it only happens
# with this much of the deadline left.
SITE_SEARCH_RESERVE = 0.9

# --- sitemap recall ---------------------------------------------------------
# The planner names the right host far more often than the right path. When a
# URL 404s, returns a script shell, or is a homepage the planner fell back to,
# the host is still right, and hosts publish their page list under robots.txt
# or /sitemap.xml. Nothing is crawled: the cache holds the host's own statement
# about itself, not its content. Spent only after a failure, so a plan whose
# URLs resolve pays nothing.
SITEMAP_HOSTS = 8
# Extra slots reserved for the homepage case. A 404 is a guess that missed; a
# homepage for a topical query is the planner saying it knew the site and not
# the page, which is exactly what a sitemap answers.
SITEMAP_ROOT_EXTRA = 3
SITEMAP_LINKS = 3
# Two requests, sometimes four, then a page fetch on top.
SITEMAP_RESERVE = 1.3
# Below this many characters a 200 is not a page: a JavaScript shell, a cookie
# interstitial or a soft 404. All three are recoverable through the sitemap.
THIN_TEXT = 400
# Minimum slug fit to spend a fetch on. On real catalogues the pages a query is
# about score 0.10-0.35, a host's careers and contact pages 0.00-0.03.
SITEMAP_FIT = 0.06

# --- route hygiene ----------------------------------------------------------
# A route candidate arrives with its backend's title, so a hopeless fetch can be
# declined before it is spent. This refuses a request, it does not rank, so the
# bar sits at the floor.
ROUTE_TITLE_FIT = 0.03
# Cap on one backend's share of the pool. A search backend cannot tell which
# query word is the subject, and unbounded, one that matches the other words
# well fills the whole top of the SERP with near misses.
ROUTE_MAX_CANDIDATES = 4
# Candidates admitted past the plan's URL cap (see `run_planner`), and lines
# scanned looking for them. Only specific addresses qualify: the tail of a plan
# is its least confident half and costs seconds to wait for.
LLM_TAIL_SPECIFIC = 3
LLM_TAIL_SCAN = 5
# Ways the planner says a route does not apply. Taken literally they would be
# searched for.
_NO_ROUTE = frozenset({"none", "n/a", "na", "-", "--", "—", "null", "nil",
                       "keine", "kein", "nothing", "skip"})

# --- deep links -------------------------------------------------------------
# The planner recalls which site holds an answer far better than which page. It
# is told never to guess an opaque slug, so where it cannot recall the address
# it names the section root instead. Every fetched page is therefore also read
# as a directory of its own site, and a link that describes the question better
# than the page it sits on is fetched.
#
# Higher than HUB_LINK_FIT: an H line is the planner saying the answer is one
# hop away, an ordinary candidate is not.
DEEP_LINK_FIT = 0.34
# How much better than its own page a link has to fit. This is what keeps the
# pass silent on pages the planner got right.
DEEP_MARGIN = 0.05
# Links followed per query, sized to fit in the slack the first wave leaves
# inside the deadline rather than open a second one.
DEEP_LINKS = 4
# Per page, so one index cannot fill the budget with its own back catalogue.
DEEP_LINKS_PER_PAGE = 2
# Headroom before a deep link is dispatched, on top of the ranking reserve.
# Below it the fetch cannot come back in time to be ranked.
DEEP_RESERVE = 0.45
# How long the one-hop expansion may hold the deadline open after the first
# wave drains. Its fetches are the newest and most specific the search has
# made, so they get a window of their own, but a bounded one.
HOP2_WINDOW = 2.5

# --- second pass -------------------------------------------------------------
# Deep is a second plan, written by the same planner over the first round's
# report: which pages came back and how well they fit, which addresses were
# dead, what has already been shown. A page turn is the same mechanism with
# the pages on screen as the shown list. Both cost one LLM call.
#
# Headroom before a second pass starts. Below it the planner cannot answer and
# the wave cannot land in time to be ranked.
REFINE_RESERVE = 2.5
# How long the second wave may hold the deadline open after its plan streamed.
REFINE_WINDOW = 4.0
# Fetches the second wave may spend past the first wave's cap.
REFINE_FETCH = 12
# Size of the report: pages, dead addresses, shown URLs, excerpt per page. Paid
# for on every deep search and page turn, so it stays short.
FINDINGS_PAGES = 8
FINDINGS_DEAD = 6
FINDINGS_SHOWN = 40
FINDINGS_EXCERPT = 220
# ---------------------------------------------------------------- intent guess
_CODE = re.compile(
    r"\b(python|javascript|typescript|rust|golang|java|c\+\+|sql|regex|api|npm|pip|"
    r"cargo|docker|kubernetes|git|bash|shell|error|exception|traceback|stacktrace|"
    r"compile|install|import|function|async|await|library|framework|package|"
    r"segfault|nullpointer|undefined|syntax|debug|deprecated|typeerror|df\.|\.py|"
    r"\.js|\.ts|\.rs|\.go|npm i |pip install)\b|[{}();]|=>|::", re.I)
_ACADEMIC = re.compile(
    r"\b(paper|papers|arxiv|preprint|study|studies|meta-analysis|citation|cited|"
    r"journal|doi|clinical trial|dataset benchmark|state of the art|sota|"
    r"et al|proceedings|thesis)\b", re.I)
_NEWS = re.compile(
    r"\b(today|yesterday|latest|breaking|news|202[4-9]|announce[ds]?|launch(ed)?|"
    r"election|earnings|acquired|acquisition|resigned|died|update|recall|"
    r"this week|right now|current(ly)?|live)\b", re.I)
_PRODUCT = re.compile(
    r"\b(price|cheap|buy|review|vs|versus|best|compare|deal|specs?|"
    r"alternative[s]?|pricing|cost)\b", re.I)
_LOCAL = re.compile(r"\b(near me|nearby|in [A-Z][a-z]+|restaurant|hotel|hours|open now|directions)\b")
_NAV = re.compile(r"^(?:go to |open )?[\w-]+(?:\.[\w-]+)+(?:/.*)?$|^\w+ (login|homepage|website|docs)$", re.I)
_REFERENCE = re.compile(
    r"^(what is|who is|who was|define|definition of|meaning of|when did|when was|"
    r"where is|how many|how much|how tall|how old|capital of|population of)\b", re.I)


def _specific(url: str) -> bool:
    """Does this address name a page, rather than a site or a section?

    A homepage or a one-word section is the planner saying it knows the site;
    a slug that reads like a title, or a path three levels down, is it saying
    it knows the page.
    """
    parts = [p for p in urlsplit(url).path.split("/") if p]
    if not parts:
        return False
    last = parts[-1]
    return len(parts) >= 3 or len(last) >= 12 or "-" in last or "." in last


def _with_lang(cand: Candidate, plan: Plan) -> Candidate:
    if not cand.lang:
        cand.lang = plan.lang
    return cand


def _title_fit(title: str, vocabs: list[frozenset[str]]) -> float:
    """How much of the query any one of its phrasings shares with a title."""
    bag = title_bag(title, latin_min=2)
    return best_f1(vocabs, bag) if bag else 0.0


def _atlas_host(url: str) -> str:
    """The key a site-search endpoint is filed under: the host, minus www.

    Not the registrable domain. Search endpoints are per host, and `registrable`
    would file one `*.gv.at` site under another's.
    """
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def guess_intent(query: str) -> str:
    q = query.strip()
    if _NAV.match(q) and len(q.split()) <= 3:
        return "navigational"
    if _ACADEMIC.search(q):
        return "academic"
    if _CODE.search(q):
        return "code"
    if _PRODUCT.search(q):
        return "product"
    if _NEWS.search(q):
        return "news"
    if _REFERENCE.match(q):
        return "reference"
    if _LOCAL.search(q):
        return "local"
    return "informational"



def _default_commons():
    """The shared plan cache client an engine gets when none is handed in."""
    try:
        from .commons import Commons
        return Commons()
    except Exception:
        return None


class Liberdex:
    """Reusable engine. One instance per process; `search()` is concurrency-safe."""

    def __init__(
        self,
        planner=None,
        *,
        fetcher: Optional[Fetcher] = None,
        ranker: Optional[Ranker] = None,
        extract_workers: int = 6,
        max_fetch: int = 44,
        max_llm_candidates: int = 10,
        enough_docs: int = 26,
        fetch_tail_window: float = 1.6,
        speculate: bool = True,
        expand_hubs: bool = True,
        cache=None,
        atlas=None,
        site_search: bool = True,
        site_search_discover: bool = True,
        sitemap: bool = True,
        sitemaps=None,
        commons=None,
    ) -> None:
        self.planner = planner
        self.fetcher = fetcher or Fetcher()
        self.ranker = ranker or Ranker()
        self.pool = ThreadPoolExecutor(max_workers=extract_workers,
                                       thread_name_prefix="liberdex-extract")
        self.max_fetch = max_fetch
        # Stop reading the planner stream once this many URLs have arrived. The
        # tail of a plan is its least confident half and costs real seconds.
        self.max_llm_candidates = max_llm_candidates
        # Once this many pages have come back with content, the stragglers are not
        # worth the tail latency they cost.
        self.enough_docs = enough_docs
        self.fetch_tail_window = fetch_tail_window
        self.speculate = speculate
        self.expand_hubs = expand_hubs
        self.cache = cache
        # Ask a host's own search box for the document when the plan named the
        # host but not the page. Reading the endpoint off a page that was going to
        # be fetched anyway is free, so it is always done and always remembered.
        # Using it costs a request, so it is spent only on a host whose endpoint is
        # already known. `site_search_discover` additionally lets a query pay a
        # round trip to learn an endpoint and use it on the same query.
        self.site_search = site_search
        self.site_search_discover = site_search_discover
        if atlas is None and site_search:
            try:
                from .cache import SiteSearchAtlas
                atlas = SiteSearchAtlas()
            except Exception:
                atlas = None
        self.atlas = atlas
        # Recall of last resort: when the plan's URL for a host fails, ask the host
        # what pages it has. See SITEMAP_HOSTS.
        self.sitemap = sitemap
        if sitemaps is None and sitemap:
            try:
                from .cache import SitemapStore
                sitemaps = SitemapStore()
            except Exception:
                sitemaps = None
        self.sitemaps = sitemaps
        self._sitemap_reader = None
        if sitemap:
            from .sitemap import SitemapReader
            self._sitemap_reader = SitemapReader(self.fetcher, sitemaps)
        # The shared plan cache, asked before the planner runs and told what
        # became of each plan's URLs. None builds the client; False leaves the
        # engine without one.
        if commons is None:
            commons = _default_commons()
        self.commons = commons or None
        # The cached planner does the lookups. It is built before the engine, so
        # the engine hands it the client it owns.
        if self.commons is not None and getattr(planner, "commons", 1) is None:
            planner.commons = self.commons
        self._started = False

    async def start(self) -> None:
        if not self._started:
            await self.fetcher.start()
            if self.commons is not None:
                # Register now, off the critical path, so the first search's
                # lookup finds a key instead of waiting for one.
                try:
                    self.commons.kick()
                except Exception:
                    pass
            self._started = True

    async def aclose(self) -> None:
        # Drain extraction threads before tearing down libcurl: curl_cffi's cleanup
        # racing a running worker aborts the process.
        self.pool.shutdown(wait=True, cancel_futures=True)
        await self.fetcher.aclose()
        # The planner owns an HTTP client of its own; nothing else closes it.
        closer = getattr(self.planner, "aclose", None)
        if callable(closer):
            try:
                await closer()
            except Exception:
                pass
        if self.commons is not None:
            try:
                await self.commons.aclose()
            except Exception:
                pass
        await asyncio.sleep(0)
        self._started = False

    async def __aenter__(self) -> "Liberdex":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    def warmup(self) -> None:
        """Load ranking models so the first query does not pay for it."""
        Models.preload()

    # ------------------------------------------------------------------- plan
    async def plan(self, query: str, *, budget: float = 6.0,
                   max_candidates: int = 14) -> Plan:
        """Just the plan: where does this live on the web?

        No fetching and no ranking: one LLM call, back in about a second.
        Everything liberdex knows about the web comes out of here.
        """
        query = " ".join((query or "").split())[:MAX_QUERY_CHARS]
        out = Plan(query=query, intent=guess_intent(query), lang=guess_lang(query))
        if not query:
            return out
        if self.planner is None:
            # Say why the plan is empty. A server with no planner configured would
            # otherwise look like a planner that found nothing.
            out.raw = "planner error: no planner configured"
            return out
        dl = Deadline(budget)
        try:
            async for kind, payload in self.planner.stream(query, dl):
                if kind in ("meta", "done") and isinstance(payload, Plan):
                    out = payload
                    if len(out.candidates) >= max_candidates and kind == "meta":
                        break
                elif kind == "error":
                    out.raw = out.raw or f"planner error: {payload}"
        except Exception as e:
            out.raw = out.raw or f"planner error: {type(e).__name__}: {e}"
        out.candidates = out.candidates[:max_candidates]
        if not out.intent:
            out.intent = guess_intent(query)
        if not out.lang:
            out.lang = guess_lang(query)
        return out

    # ---------------------------------------------------------------- extract
    async def extract_urls(
        self,
        urls: Sequence[str],
        *,
        budget: float = 15.0,
        want_links: bool = False,
        want_media: bool = False,
        markdown: bool = False,
    ) -> list[Doc]:
        """Fetch and read pages the caller already has URLs for.

        The same fetcher and the same extractor a search uses, with no planner
        and no ranking in front of them.
        """
        await self.start()
        dl = Deadline(budget)
        cands = [Candidate(url=u, source="explicit") for u in urls]
        docs = await self.fetcher.gather(cands, dl, reserve=0.0)
        loop = asyncio.get_running_loop()

        def read(d: Doc) -> Doc:
            if d.body:
                extract(d, want_links=want_links, want_media=want_media)
                if markdown:
                    md = to_markdown(decode(d.body, d.content_type), d.final_url)
                    if md:
                        d.text = md
            return d

        return list(await asyncio.gather(*(
            loop.run_in_executor(self.pool, read, d) for d in docs
        )))

    async def query_from_url(self, url: str, *, budget: float = 8.0
                             ) -> tuple[str, str]:
        """Turn a page into a query. Returns (query, its registrable domain).

        There is no index to look up neighbours in, so the page is read and
        turned into the phrase a planner would have been handed instead.
        """
        docs = await self.extract_urls([url], budget=budget)
        if not docs:
            return "", ""
        d = docs[0]
        host = registrable((urlsplit(d.final_url).hostname or "").lower())
        lang = d.lang[:2] or "en"
        # The title is the publisher's own one-line summary, so it is the query,
        # minus the site name most CMSs append to it.
        title = _strip_site_suffix(" ".join((d.title or "").split()), d.site)
        if not title:
            title = keyword_query(d.description or "", 8, lang=lang)
        # The description adds the terms a title had no room for. The body does
        # not: its first 2000 characters are navigation and boilerplate as often as
        # subject matter. Terms are stripped of punctuation and compared as whole
        # words against the title's words.
        seen = {w for w in re.findall(r"[\w-]+", title.lower())}
        extra: list[str] = []
        for t in content_terms(d.description or "", lang):
            t = t.strip(".,;:!?()[]{}\"'\u2018\u2019\u201c\u201d")
            key = t.lower()
            if len(t) < 3 or key in seen:
                continue
            seen.add(key)
            extra.append(t)
        extra.sort(key=lambda t: -_informativeness(t))
        # A site search chokes on the typographic dashes a CMS puts in a title.
        title = re.sub(r"\s*[\u2013\u2014|]\s*", " ", title)
        query = " ".join(re.sub(r"\s+", " ", " ".join([title] + extra[:4])).split())
        return query.strip()[:400], host

    # ------------------------------------------------------------------ search
    async def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        budget: float = 4.0,
        # How long to wait for the planner's first URL on top of `budget`, which
        # is then the fetch window. Granted up front, handed back unused the
        # moment the first URL lands (see run_planner). api.PLAN_GRACE per tier.
        plan_grace: float = 0.0,
        rank_reserve: float = 0.35,
        include_plan: bool = False,
        keep_text: bool = False,
        snippet_chars: int = 340,
        snippet_windows: int = 1,
        max_llm_candidates: Optional[int] = None,
        max_fetch: Optional[int] = None,
        enough_docs: Optional[int] = None,
        fetch_tail_window: Optional[float] = None,
        expand_hubs: Optional[bool] = None,
        # --- retrieval controls ---
        include_domains: Optional[Sequence[str]] = None,
        exclude_domains: Optional[Sequence[str]] = None,
        published_after: Optional[str] = None,
        published_before: Optional[str] = None,
        require_date: bool = False,
        intent: Optional[str] = None,
        # --- context controls ---
        want_media: bool = False,
        markdown: bool = False,
        token_budget: int = 0,
        token_budget_per_page: int = 0,
        passages_per_page: int = 0,
        # Named ranking-weight overrides, on top of the intent's own. They can
        # reweight the signals the fusion has, not add a new one.
        weights: Optional[dict] = None,
        # --- rounds ---
        # A second planning pass over the first round's report (see REFINE_*).
        # What makes `deep` deep.
        refine: bool = False,
        # Which page of results this is. Past the first the search is a new round:
        # `exclude_urls` are never dispatched and never returned, and the planner
        # is asked, over them, for what comes next.
        page: int = 1,
        exclude_urls: Optional[Sequence[str]] = None,
        # Called synchronously as the pipeline advances: ("candidate", url),
        # ("page", Doc), ("plan", Plan). Must not block or raise; see _emit.
        on_event: Optional[Callable[[str, Any], None]] = None,
        # A plan written by the caller (see planner.plan_from_dict). The configured
        # planner is not called; site-native routes still fire.
        plan: Optional[Plan] = None,
        # A planner for this call only, in place of the engine's own: the model
        # that writes the plan belongs to the caller (a host plugin's, say).
        planner: Optional[Any] = None,
    ) -> SearchResponse:
        await self.start()
        lent = planner is not None and planner is not self.planner
        if planner is None:
            planner = self.planner
        elif lent and self.commons is not None and getattr(planner, "commons", 1) is None:
            planner.commons = self.commons
        supplied = plan is not None
        if supplied:
            from .planner.base import SuppliedPlanner
            planner = SuppliedPlanner(plan)
        # Per-request overrides. Instance attributes are only the defaults, so
        # concurrent searches with different depths cannot clobber each other.
        max_llm_candidates = (self.max_llm_candidates if max_llm_candidates is None
                              else max_llm_candidates)
        max_fetch = self.max_fetch if max_fetch is None else max_fetch
        enough_docs = self.enough_docs if enough_docs is None else enough_docs
        fetch_tail_window = (self.fetch_tail_window if fetch_tail_window is None
                             else fetch_tail_window)
        expand_hubs = self.expand_hubs if expand_hubs is None else expand_hubs
        allow = {d for d in (include_domains or []) if d and d.strip()}
        deny = {d for d in (exclude_domains or []) if d and d.strip()}
        after = parse_date(published_after or "")
        before = parse_date(published_before or "")
        query = " ".join((query or "").split())[:MAX_QUERY_CHARS]
        # A supplied plan has its first URL at t=0; there is nothing to wait for.
        plan_grace = 0.0 if supplied or planner is None else max(0.0, plan_grace)
        dl = Deadline(budget + plan_grace)
        timings: dict[str, float] = {}
        stats: dict[str, object] = {}
        if supplied:
            stats["planner"] = "supplied"
        elif lent:
            stats["planner"] = getattr(planner, "name", "caller")
        if not query:
            return SearchResponse(query="", intent="informational", results=[],
                                  timings={"total_ms": 0.0},
                                  stats={"error": "empty query"})

        plan = Plan(query=query, intent=intent or guess_intent(query),
                    lang=guess_lang(query))
        # Does the query constrain when? If so a page's date is part of the answer
        # and worth reading out of the body when the metadata lacks it.
        q_years, q_recent = query_period(query)
        want_date = bool(q_years or q_recent or plan.intent == "news")
        # An explicit intent outranks the planner's guess (see absorb).
        pinned_intent = intent or ""
        # Pages the caller has already been given. Seeding `seen` with them keeps a
        # page turn from re-fetching the page it just showed; the filter after
        # ranking catches a redirect that lands on one under another address.
        page = max(1, int(page or 1))
        shown = [u for u in (exclude_urls or []) if u and u.strip()][:FINDINGS_SHOWN * 2]
        excluded = {normalize_url(u) for u in shown}
        seen: set[str] = set(excluded)
        # Fetches the second pass may spend past the first wave's cap.
        extra_cap = [0]
        docs: list[Doc] = []
        fetch_tasks: set[asyncio.Task] = set()
        cand_q: asyncio.Queue = asyncio.Queue()
        first_url_at: Optional[float] = None
        routes_fired: set[tuple[str, str]] = set()
        route_status: dict[str, int] = {}
        producer_tasks: list[asyncio.Task] = []
        hub_docs: list[Doc] = []
        hub_tasks: set[asyncio.Task] = set()
        hubs_grabbed: set[str] = set()
        # Documents already reached by a one-hop expansion, keyed by `_unversioned`,
        # so the eager pass and the closing sweep cannot fetch one page twice under
        # two addresses.
        deep_picked: set[str] = set()
        # Second-hop fetches, tracked apart from the first wave. They are the
        # newest and most specific requests the search has made, and cutting them
        # off on a tail window that started before they existed throws away the
        # answer page. They get HOP2_WINDOW instead.
        hop2_tasks: set[asyncio.Task] = set()
        counts = {"llm": 0, "route": 0, "expand": 0, "deep": 0, "fetched": 0,
                  "ok": 0, "filtered": 0, "sitesearch": 0, "sitemap": 0, "route_dropped": 0,
                  "refine": 0}

        loop = asyncio.get_running_loop()

        def emit(kind: str, payload: Any) -> None:
            """Tell a listener the pipeline moved. A listener that throws is not the
            query's problem."""
            if on_event is None:
                return
            try:
                on_event(kind, payload)
            except Exception:
                pass

        # The cross-encoder reads each page as it lands, off the loop, so the
        # ranker finds its scores waiting instead of computing them last.
        prescore = Prescore(self.ranker, query)

        # --------------------------------------------------------- dispatching
        async def fetch_and_extract(cand: Candidate, want_links: bool = False) -> None:
            if cand.skip_fetch and cand.content:
                host = (urlsplit(cand.url).hostname or "").lower()
                docs.append(Doc(
                    url=cand.url, final_url=cand.url, status=200,
                    title=cand.title, text=cand.content, description=cand.snippet,
                    site=host[4:] if host.startswith("www.") else host,
                    published=cand.published, candidate=cand,
                ))
                counts["fetched"] += 1
                counts["ok"] += 1
                return
            doc = await self.fetcher._fetch_one(_with_lang(cand, plan), dl, rank_reserve)
            counts["fetched"] += 1
            if doc.body:
                try:
                    await loop.run_in_executor(
                        self.pool,
                        lambda: extract(doc, want_links=want_links,
                                        want_media=want_media,
                                        want_date=want_date),
                    )
                except Exception:
                    pass
            if doc.ok:
                counts["ok"] += 1
                emit("page", doc)
                prescore.submit(doc, plan.lang)
            docs.append(doc)
            # The plan named this host and could not name its page. Ask the host.
            if cand.source in ("llm", "hub"):
                why = unusable(doc)
                if why:
                    fire_sitemap(doc.final_url or doc.url, declared=why == "root")
            # One hop deeper, now rather than after the wave drains. Only pages we
            # asked for links from, which is the first wave: a page reached by an
            # expansion does not expand again.
            if (doc.ok and expand_hubs and want_links
                    and counts["deep"] < DEEP_LINKS
                    and not dl.expired(rank_reserve + DEEP_RESERVE)):
                try:
                    counts["deep"] += self._follow_section(
                        query, plan, doc, dispatch, deep_picked,
                        min(DEEP_LINKS_PER_PAGE, DEEP_LINKS - counts["deep"]))
                except Exception:
                    pass

        def dispatch(cand: Candidate, want_links: bool = False) -> bool:
            nonlocal first_url_at
            key = normalize_url(cand.url)
            # While a sitemap recovery is in flight the last few fetches are held for
            # it. Expansion follows links off pages already in hand; recovery is the
            # only path to a host whose page was never read, and it arrives late by
            # construction, since the failure that triggers it has to happen first.
            cap = max_fetch + extra_cap[0]
            if recovered and cand.source in ("expand", "deep"):
                cap -= SITEMAP_LINKS * 2
            if key in seen or len(seen) >= cap or dl.expired(rank_reserve):
                return False
            # Gate before the fetch, not after the rank. With no index there is no
            # result set to filter down, so an allowlist has to shape what is fetched.
            if allow or deny:
                host = (urlsplit(cand.url).hostname or "").lower()
                if allow and not host_matches(host, allow):
                    counts["filtered"] += 1
                    return False
                if deny and host_matches(host, deny):
                    counts["filtered"] += 1
                    return False
            seen.add(key)
            if first_url_at is None:
                first_url_at = dl.elapsed
            # A host with a known search endpoint needs no discovery round trip, so its
            # search goes out in this wave rather than after the pages come back.
            if cand.source == "llm":
                fire_site_search(_atlas_host(cand.url))
            emit("candidate", cand)
            t = asyncio.create_task(fetch_and_extract(cand, want_links))
            fetch_tasks.add(t)
            t.add_done_callback(fetch_tasks.discard)
            if cand.source in ("deep", "expand"):
                hop2_tasks.add(t)
                t.add_done_callback(hop2_tasks.discard)
            return True

        # ------------------------------------------------------------- routes
        def claim(route: Route, explicit: str = "") -> Optional[str]:
            """Reserve a (route, query) firing, or None if it is redundant.

            Synchronous, before the task is created, so the planner, which emits a
            meta event on every token, cannot queue the same route a hundred times.
            """
            if not route.available:
                return None
            # The planner writes each site's own search query; the heuristic trim is
            # only the fallback for routes fired before the plan arrives.
            q = route.query_for(query, explicit or plan.route_queries.get(route.name, ""),
                                plan.lang)
            if not q.strip() or q.strip().lower() in _NO_ROUTE:
                return None
            # Keyed on the query, not just the route: a route fired speculatively on a
            # guessed query is worth firing again on the one the planner wrote.
            key = (route.name, q)
            if key in routes_fired:
                return None
            routes_fired.add(key)
            return q

        async def run_route(route: Route, q: str) -> None:
            # A route may name more than one backend for the same site, tried in turn
            # until one answers, so a site that blocks one address is not absent.
            body, status = b"", 0
            for build in (route.build, *route.fallbacks):
                try:
                    url, headers = build(q, plan.lang)
                    status, body, _final, _ct = await self.fetcher.get_raw(
                        url, headers=headers,
                        # A search API answers in well under a second; a search page can
                        # need the slack, and is only fetched when the planner asked for it.
                        timeout=min(3.0, max(0.4, dl.remaining - rank_reserve)),
                    )
                except Exception:
                    status, body = 0, b""
                    continue
                route_status[route.name] = status
                if status == 200 and body:
                    break
            if status != 200 or not body:
                # Every backend refused us. The route stays off for a while rather
                # than burning requests on every query.
                route.penalize(status)
                return
            try:
                kept = 0
                parse = (route.parse if route.name not in ROUTES
                         else route_parser(route.name, plan.lang))
                for cand in parse(body):
                    if kept >= ROUTE_MAX_CANDIDATES:
                        break
                    kept += 1
                    counts["route"] += 1
                    await cand_q.put(cand)
            except Exception:
                return

        async def speculative() -> None:
            routes = speculative_routes(plan.intent)
            if not routes:
                routes = [ROUTES["wikipedia"]]
            fired = [(r, claim(r)) for r in routes]
            await asyncio.gather(
                *(run_route(r, q) for r, q in fired if q), return_exceptions=True
            )

        # --------------------------------------------------------------- hubs
        async def grab_hub(url: str) -> None:
            c = Candidate(url=url, reason="hub page", prior=0.35, source="hub")
            d = await self.fetcher._fetch_one(_with_lang(c, plan), dl, rank_reserve)
            if not d.body:
                return
            try:
                await loop.run_in_executor(
                    self.pool, lambda: extract(d, want_links=True)
                )
            except Exception:
                return
            hub_docs.append(d)

        # ------------------------------------------------------- site search
        searched_hosts: set[str] = set()

        async def grab_site_search(host: str, template: str, terms: str = "") -> None:
            """Ask one host's own search box for the query.

            The result is treated as a hub, not a result: a search page is a list of
            links and its own title says nothing about the query, so it goes through
            the same anchor-fit filter as any other index page. Nothing trusts the
            site's ranking, only its knowledge of what it holds.
            """
            terms = terms or keyword_query(query, 8, lang=plan.lang)
            try:
                url = template.replace("{}", quote_plus(terms))
            except Exception:
                return
            if normalize_url(url) in seen:
                return
            seen.add(normalize_url(url))
            c = Candidate(url=url, reason=f"{host} site search", prior=0.3,
                          source="sitesearch")
            counts["sitesearch"] += 1
            d = await self.fetcher._fetch_one(_with_lang(c, plan), dl, rank_reserve)
            if not d.body:
                return
            try:
                await loop.run_in_executor(
                    self.pool, lambda: extract(d, want_links=True)
                )
            except Exception:
                return
            # Its own text is a results listing; only its links are wanted.
            d.text = ""
            hub_docs.append(d)

        # ---------------------------------------------------- sitemap recall
        recovered: set[str] = set()

        def vocabs_for_links() -> list[frozenset[str]]:
            """The query's vocabularies as they stand right now.

            Recomputed rather than cached because the plan is still streaming: a
            recovery that fires before the X line has arrived would otherwise score
            every slug against the user's words alone.
            """
            return self._vocabs(query, plan)

        rooted: set[str] = set()
        free: set[str] = set()

        async def grab_sitemap(host: str, seed: str) -> None:
            """Ask a host that failed us what pages it actually has.

            The plan named this host and got the path wrong. Its catalogue is public
            and its slugs are descriptive, so the same fit function that picks a link
            off a hub picks the address off the list.
            """
            assert self._sitemap_reader is not None
            try:
                locs = await self._sitemap_reader.locs(host, vocabs_for_links())
            except Exception:
                return
            if not locs:
                return
            picks = sitemap_pick(locs, vocabs_for_links(), SITEMAP_LINKS,
                                 min_fit=SITEMAP_FIT, lang=plan.lang)
            for u in picks:
                if normalize_url(u) == normalize_url(seed):
                    continue
                c = Candidate(url=u, reason=f"{host} sitemap", prior=0.45,
                              source="sitemap")
                if dispatch(c):
                    counts["sitemap"] += 1

        def fire_sitemap(url: str, declared: bool = False) -> bool:
            """Recover a host whose page we could not read. One host, once."""
            host = (urlsplit(url).hostname or "").lower()
            if (not self.sitemap or self._sitemap_reader is None or not host
                    or host in recovered
                    or len(seen) >= max_fetch
                    or dl.expired(rank_reserve + SITEMAP_RESERVE)):
                return False
            # The quota bounds requests, not work. A host already in the store costs
            # a string scan over cached slugs, not a slot.
            cached = (self.sitemaps is not None
                      and self.sitemaps.get(host) is not None)
            if not cached and len(recovered) - len(free) >= SITEMAP_HOSTS:
                if not declared or len(rooted) >= SITEMAP_ROOT_EXTRA:
                    return False
                rooted.add(host)
            if cached:
                free.add(host)
            recovered.add(host)
            t = asyncio.create_task(grab_sitemap(host, url))
            hub_tasks.add(t)
            t.add_done_callback(hub_tasks.discard)
            return True

        def unusable(doc: Doc) -> str:
            """Did this fetch fail to produce a page worth ranking?

            Three failures that look different on the wire and identical to a reader:
            the address does not exist, it serves a script loader, or it is the front
            door of a site asked a topical question. The planner uses a homepage only
            as a last resort, so one is an admission that it could not recall the
            page, which is the case the host's own catalogue answers.
            """
            if not doc.ok:
                return "dead"
            if len(doc.text or "") < THIN_TEXT:
                return "thin"
            if (plan.intent != "navigational"
                    and urlsplit(doc.final_url or doc.url).path.strip("/") == ""):
                return "root"
            return ""

        def fire_site_search(host: str) -> bool:
            """Search a host we already know the endpoint for. Free of a probe."""
            if (not self.site_search or self.atlas is None
                    or host in searched_hosts
                    or len(searched_hosts) >= SITE_SEARCH_HOSTS
                    or dl.expired(rank_reserve + SITE_SEARCH_RESERVE)):
                return False
            template = self.atlas.get(host)
            if not template:
                return False
            searched_hosts.add(host)
            t = asyncio.create_task(grab_site_search(host, template))
            hub_tasks.add(t)
            t.add_done_callback(hub_tasks.discard)
            return True

        def start_hubs() -> None:
            """Fetch hub pages the moment the planner names them.

            The protocol puts the H lines before the bulk of the U lines for this
            reason: waiting for the whole plan would put the one-hop expansion on the
            critical path.
            """
            if not expand_hubs:
                return
            for u in plan.hubs[:4]:
                if u in hubs_grabbed or dl.expired(rank_reserve + 0.6):
                    continue
                hubs_grabbed.add(u)
                t = asyncio.create_task(grab_hub(u))
                hub_tasks.add(t)
                t.add_done_callback(hub_tasks.discard)

        # ------------------------------------------------------------ planner
        plan_shape: list = [None]

        def absorb(p: Plan) -> None:
            """Fold whatever the planner has emitted so far into the live plan."""
            # A caller-supplied intent outranks the planner's guess.
            if p.intent and not pinned_intent:
                plan.intent = p.intent
            if p.lang:
                plan.lang = p.lang
            if p.expansions:
                plan.expansions = p.expansions
            if p.hubs:
                plan.hubs = p.hubs
                start_hubs()
            if p.candidates:
                plan.candidates = p.candidates
            if p.raw:
                plan.raw = p.raw
            # absorb() runs on every token delta, so emit only when the plan gained
            # something a listener has not already been told.
            shape = (plan.intent, plan.lang, tuple(plan.routes), tuple(plan.hubs),
                     tuple(plan.expansions))
            if shape != plan_shape[0]:
                plan_shape[0] = shape
                emit("plan", plan)
            for name in p.routes:
                if name in ROUTES and name not in plan.routes:
                    plan.routes.append(name)
            plan.route_queries.update(p.route_queries)
            for name in p.routes:
                route = ROUTES.get(name)
                if route is None:
                    continue
                q = claim(route)
                if q:
                    producer_tasks.append(asyncio.create_task(run_route(route, q)))

        async def run_planner() -> None:
            if planner is None:
                return
            t0 = time.perf_counter()
            first = True
            # (specific candidates taken past the cap, lines scanned past it)
            tail = [0, 0]
            try:
                async for kind, payload in planner.stream(query, dl):
                    if kind == "candidate":
                        if first:
                            first = False
                            timings["plan_first_url_ms"] = (time.perf_counter() - t0) * 1000
                            if plan_grace:
                                # The fetch window starts here. The grace was granted
                                # up front so a slow planner is not cut before its
                                # first URL; what it did not use goes back now, and
                                # every later reader of the deadline sees that.
                                used = min(dl.elapsed, plan_grace)
                                dl.extend(used - plan_grace)
                                stats["plan_grace_ms"] = used * 1000
                        # Past the cap the tail is still read, but only for candidates worth
                        # what they cost. A homepage low in the list is the planner admitting
                        # it could not recall the page; a deep path low in the list is a
                        # recollection it was merely unsure of.
                        if max_llm_candidates and counts["llm"] >= max_llm_candidates:
                            if (tail[0] >= LLM_TAIL_SPECIFIC
                                    or tail[1] >= LLM_TAIL_SCAN):
                                break
                            tail[1] += 1
                            if not _specific(payload.url):
                                continue
                            tail[0] += 1
                        counts["llm"] += 1
                        await cand_q.put(payload)
                    elif kind in ("meta", "done"):
                        p: Plan = payload  # type: ignore[assignment]
                        # Copy on every tick, not only at the end: the planner is routinely
                        # cut off by the deadline or the URL cap, and the expansions and
                        # hubs that arrived before the cut still count.
                        absorb(p)
                    elif kind == "source":
                        # A cached planner says where the plan came from ("cache" or
                        # "commons"). No LLM ran for it.
                        stats["planner"] = str(payload)
                    elif kind == "error":
                        # A refused request: a bad key, a rate limit, a replay cache miss.
                        # Recorded so the routes-only floor does not pass for a working
                        # search that found little.
                        stats["planner_error"] = str(payload)[:300]
            except asyncio.CancelledError:
                # Cut past the URL cap, while scanning the tail for a specific
                # page worth taking, is not a plan lost: the engine was about to
                # stop reading anyway. Below the cap it is.
                if not (max_llm_candidates and counts["llm"] >= max_llm_candidates):
                    stats["planner_error"] = "cancelled"
            except BaseException as e:
                # BaseException: an async generator closed early raises GeneratorExit,
                # and that has to be recorded too.
                stats["planner_error"] = f"{type(e).__name__}: {e}"
            timings["plan_ms"] = (time.perf_counter() - t0) * 1000

        # ------------------------------------------------------------ consumer
        # Routes the planner asks for start mid-stream and are producers too; the
        # consumer must not decide everything is done the moment the planner is.
        producer_tasks.append(asyncio.create_task(run_planner()))
        if self.speculate:
            producer_tasks.append(asyncio.create_task(speculative()))

        async def consume() -> None:
            while True:
                try:
                    cand = await asyncio.wait_for(cand_q.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    if all(p.done() for p in producer_tasks) and cand_q.empty():
                        return
                    if dl.expired(rank_reserve):
                        return
                    continue
                admit(cand)

        def admit(cand: Candidate) -> bool:
            """A candidate off the queue: judged on its route title, then dispatched
            with link extraction on."""
            if cand.source.startswith("route:") and cand.title:
                # A query of nothing but stopwords has no vocabulary to judge against,
                # and judging against an empty one rejects everything.
                vocabs = vocabs_for_links()
                if vocabs and _title_fit(cand.title, vocabs) < ROUTE_TITLE_FIT:
                    counts["route_dropped"] += 1
                    return False
            # Every landed page is also read as a directory of its own site: link
            # extraction is cheap and is what makes a deep link findable.
            return dispatch(cand, want_links=expand_hubs)

        async def drain_producers() -> bool:
            """True once every producer has finished, False when the deadline
            cut them. The deadline is re-read each slice, not fixed up front:
            the planner's first URL hands back the grace it did not use."""
            while True:
                pend = [p for p in producer_tasks if not p.done()]
                if not pend:
                    return True
                if dl.expired(rank_reserve):
                    return False
                await asyncio.wait(pend, timeout=0.05)

        consumer = asyncio.create_task(consume())
        if not await drain_producers():
            for p in producer_tasks:
                p.cancel()
        try:
            await asyncio.wait_for(consumer, timeout=max(0.05, dl.remaining - rank_reserve))
        except (asyncio.TimeoutError, asyncio.CancelledError):
            consumer.cancel()
        timings["gather_ms"] = dl.elapsed * 1000

        # ------------------------------------------------- one-hop hub expansion
        if expand_hubs and hub_tasks and dl.remaining > rank_reserve + 0.4:
            # The hub fetches have been in flight since the planner named them; give
            # the stragglers a short grace period, then follow their links.
            try:
                await asyncio.wait_for(
                    asyncio.gather(*hub_tasks, return_exceptions=True),
                    timeout=max(0.05, min(0.8, dl.remaining - rank_reserve - 0.4)),
                )
            except asyncio.TimeoutError:
                pass
        # Every page that came back may have stated its host's search endpoint.
        # Recording it makes the next query on this host free; using it now is
        # what makes this one work when the plan named the site and not the page.
        if self.site_search and self.atlas is not None:
            self.atlas.put_many(
                (d.site, d.search_url)
                for d in list(docs) + list(hub_docs)
                if d.search_url and d.site)
            if (self.site_search_discover
                    and dl.remaining > rank_reserve + SITE_SEARCH_RESERVE):
                # Which host to ask: the one whose pages look most like the subject,
                # not the one that returned the most pages. A site that is on topic
                # and did not hand us the document is the one worth searching.
                vocabs = self._vocabs(query, plan)
                by_host: dict[str, float] = {}
                for d in docs:
                    if not d.ok or not d.site:
                        continue
                    h = d.site
                    fit = max(best_f1(vocabs, set(tokens(d.title or ""))),
                              path_fit(vocabs, urlsplit(d.final_url).path))
                    by_host[h] = max(by_host.get(h, 0.0), fit)
                fired = [h for h, _fit in sorted(by_host.items(),
                                                 key=lambda kv: -kv[1])
                         if fire_site_search(h)]
                if fired:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*hub_tasks, return_exceptions=True),
                            timeout=max(0.05, min(
                                SITE_SEARCH_RESERVE,
                                dl.remaining - rank_reserve - 0.3)),
                        )
                    except asyncio.TimeoutError:
                        pass
        if expand_hubs and hub_docs and dl.remaining > rank_reserve + 0.3:
            # A site's own results page earns a larger share of the expansion budget
            # than an editorial index: every link on it is the site's answer to this
            # query.
            self._follow_hubs(query, plan, hub_docs, dispatch, counts,
                              limit=HUB_LINKS + (SITE_SEARCH_LINKS
                                                 if searched_hosts else 0))
        if (expand_hubs and docs and counts["deep"] < DEEP_LINKS
                and dl.remaining > rank_reserve + DEEP_RESERVE):
            counts["deep"] += self._follow_deep(
                query, plan, list(docs), dispatch, counts, deep_picked,
                DEEP_LINKS - counts["deep"])

        # ------------------------------------------------------- second pass
        # Started now, before the first wave's tail is waited on, so the planner's
        # seconds to first token overlap the stragglers. What it plans over is
        # what has come back by this point, which is nearly everything that will.
        second: Optional[asyncio.Task] = None
        refiner = getattr(planner, "refine", None) if planner else None
        if ((refine or page > 1) and refiner is not None
                and dl.remaining > rank_reserve + REFINE_RESERVE):
            second = asyncio.create_task(self._second_pass(
                query, plan, docs, prescore, refiner, dl, rank_reserve,
                page=page, shown=shown, unusable=unusable, dispatch=dispatch,
                admit=admit, claim=claim, run_route=run_route, cand_q=cand_q,
                fetch_tasks=fetch_tasks, extra_cap=extra_cap, counts=counts,
                timings=timings, stats=stats, expand_hubs=expand_hubs))
        elif refine or page > 1:
            stats["refine_skipped"] = ("no second pass on this planner"
                                       if refiner is None else "no time left")

        if fetch_tasks:
            window = min(fetch_tail_window,
                         max(0.05, dl.remaining - rank_reserve * 0.5))
            cutoff = time.perf_counter() + window
            # The first wave only: what the second pass dispatches is its own.
            pending = set(fetch_tasks)
            # Deep links are dispatched after `enough_docs` is reached, so each one
            # raises the bar, or they would be cancelled the moment they are created.
            enough = enough_docs + counts["deep"]
            while pending and counts["ok"] < enough:
                left = cutoff - time.perf_counter()
                if left <= 0:
                    break
                _done, pending = await asyncio.wait(
                    pending, timeout=min(0.08, left),
                    return_when=asyncio.FIRST_COMPLETED,
                )
            # The one-hop expansion gets its own window rather than the first
            # wave's leftovers.
            hop2 = {t for t in pending if t in hop2_tasks}
            if hop2:
                left = max(0.05, min(HOP2_WINDOW, dl.remaining - rank_reserve))
                cutoff2 = time.perf_counter() + left
                while hop2:
                    left = cutoff2 - time.perf_counter()
                    if left <= 0:
                        break
                    _done, hop2 = await asyncio.wait(
                        hop2, timeout=min(0.08, left),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                pending = {t for t in pending if not t.done()}
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            stats["fetch_cancelled"] = len(pending)

        # The second pass, if any, has been planning while the first wave's
        # stragglers were waited on; now it gets its own wave.
        if second is not None:
            try:
                await second
            except Exception as e:
                stats["refine_error"] = f"{type(e).__name__}: {e}"
        # Hub fetches start mid-plan and are not in fetch_tasks. Nothing may
        # outlive the query that started it: a fetch still running at aclose()
        # races libcurl teardown and aborts the process.
        if hub_tasks:
            for t in list(hub_tasks):
                t.cancel()
            await asyncio.gather(*list(hub_tasks), return_exceptions=True)
        timings["fetch_ms"] = dl.elapsed * 1000

        # ------------------------------------------------------------- ranking
        t_rank = time.perf_counter()
        results = self.ranker.rank(
            query, docs, plan, top_k=top_k,
            time_budget=max(0.05, dl.remaining), keep_text=keep_text,
            snippet_chars=snippet_chars, snippet_windows=snippet_windows,
            weights=weights, prescore=prescore,
        )
        prescore.close()
        stats["prescored"] = len(prescore._futures)
        if excluded:
            results = [r for r in results if normalize_url(r.url) not in excluded]
        if not results and plan.candidates:
            # Nothing fetched, or the deadline landed mid-flight. The recalled URLs
            # are worth more than nothing, as long as they are marked unverified.
            stats["degraded"] = "no fetched pages; returning unverified candidates"
            seen_deg: set[str] = set()
            for c in plan.candidates[:top_k]:
                key = normalize_url(c.url)
                if key in seen_deg:
                    continue
                seen_deg.add(key)
                host = (urlsplit(c.url).hostname or "").lower()
                results.append(Result(
                    url=c.url, title=c.title or c.url, snippet=c.snippet,
                    site=host[4:] if host.startswith("www.") else host,
                    score=0.0, source=c.source, debug={"unverified": True},
                ))
        if after or before:
            results = _filter_by_date(results, after, before, require_date)
        if markdown or want_media:
            await _enrich(results, docs, self.pool,
                          markdown=markdown, want_media=want_media)
        if token_budget or token_budget_per_page or passages_per_page:
            results, over = budget_fit(
                results, token_budget=token_budget,
                token_budget_per_page=token_budget_per_page,
                passages_per_page=passages_per_page)
            if over:
                stats["dropped_over_budget"] = over
        timings["rank_ms"] = (time.perf_counter() - t_rank) * 1000
        timings["total_ms"] = dl.elapsed * 1000
        if first_url_at is not None:
            timings["first_dispatch_ms"] = first_url_at * 1000

        stats.update({
            "candidates_llm": counts["llm"],
            "candidates_route": counts["route"],
            "candidates_expand": counts["expand"],
            **({"candidates_deep": counts["deep"]} if counts["deep"] else {}),
            **({"candidates_refine": counts["refine"]} if counts["refine"] else {}),
            **({"page": page, "excluded": len(excluded)} if page > 1 else {}),
            **({"site_searches": counts["sitesearch"]} if counts["sitesearch"] else {}),
            **({"sitemap_urls": counts["sitemap"],
                "sitemap_hosts": sorted(recovered)} if recovered else {}),
            **({"route_dropped": counts["route_dropped"]}
               if counts["route_dropped"] else {}),
            "dispatched": len(seen),
            "fetched": counts["fetched"],
            "fetched_ok": counts["ok"],
            "routes_fired": sorted({n for n, _ in routes_fired}),
            "route_status": route_status,
            "intent": plan.intent,
            "lang": plan.lang,
            "domains": len({registrable(r.site) for r in results}),
            **({"filtered_by_domain": counts["filtered"]} if counts["filtered"] else {}),
        })
        if self.commons is not None and self.commons.enabled:
            self._share(query, plan, supplied, seen, docs, results)
        return SearchResponse(
            query=query, intent=plan.intent, results=results,
            timings={k: round(v, 1) for k, v in timings.items()},
            stats=stats, plan=plan if include_plan else None,
        )

    def _share(self, query: str, plan: Plan, supplied: bool, dispatched,
               docs, results) -> None:
        """Report what became of the plan's URLs and offer a plan the caller
        wrote. Both only queue; nothing here waits on the network."""
        try:
            from .commons import render_plan
            c = self.commons
            if supplied:
                c.offer_plan(query, render_plan(plan), model="supplied", planner="supplied")
            # `dispatched` is keyed the way the fetch wave keys it, so every
            # address here is compared in that same normalised form.
            fetched: set[str] = set()
            for d in docs:
                for u in (getattr(d, "url", ""), getattr(d, "final_url", "")):
                    if u:
                        fetched.add(normalize_url(u))
                cand = getattr(d, "candidate", None)
                if cand is not None and getattr(cand, "url", ""):
                    fetched.add(normalize_url(cand.url))
            top = {normalize_url(r.url) for r in results[:10]}
            urls = []
            for cand in plan.candidates:
                if cand.source != "llm":
                    continue
                key = normalize_url(cand.url)
                if key not in dispatched:
                    continue
                urls.append({"url": cand.url, "ok": key in fetched, "serp": key in top})
            if urls:
                c.report(query, urls)
        except Exception:
            pass

    # ----------------------------------------------------------- second pass
    async def _second_pass(
        self, query: str, plan: Plan, docs: list[Doc], prescore: Prescore,
        refiner, dl: Deadline, rank_reserve: float, *, page: int,
        shown: Sequence[str], unusable, dispatch, admit, claim, run_route,
        cand_q: asyncio.Queue, fetch_tasks: set, extra_cap: list,
        counts: dict, timings: dict, stats: dict, expand_hubs: bool,
    ) -> None:
        """Plan again, over what the first round found, and fetch the gap.

        The first plan was written from memory alone. This one is written over
        the round's report: the pages that came back with their scores, the
        addresses that died, the pages already shown. What it names is fetched as
        a second wave and ranked with the first. Everything it yields goes through
        round one's gates, `dispatch` for URLs and `admit` for route hits.
        """
        t0 = time.perf_counter()
        # What is in flight now is the first wave's and is already being waited
        # on. This pass waits only on what it dispatches itself.
        before = set(fetch_tasks)
        findings = self._findings(query, plan, docs, prescore, unusable,
                                  shown=shown, page=page)
        extra_cap[0] = REFINE_FETCH
        route_tasks: list[asyncio.Task] = []
        hubs_seen: set[str] = set()
        first = True
        try:
            async for kind, payload in refiner(query, findings, dl):
                if kind == "candidate":
                    if first:
                        timings["refine_first_url_ms"] = (time.perf_counter() - t0) * 1000
                        first = False
                    payload.source = "refine"
                    if dispatch(payload, want_links=expand_hubs):
                        counts["refine"] += 1
                elif kind in ("meta", "done"):
                    p2: Plan = payload  # type: ignore[assignment]
                    # A hub named here is fetched as a page with its links read, so the
                    # eager one-hop expansion follows them the moment it lands.
                    for u in p2.hubs:
                        if u in hubs_seen or dl.expired(rank_reserve + 0.6):
                            continue
                        hubs_seen.add(u)
                        c = Candidate(url=u, reason="second-pass hub",
                                      prior=0.35, source="refine")
                        if dispatch(c, want_links=True):
                            counts["refine"] += 1
                    plan.route_queries.update(p2.route_queries)
                    for name in p2.routes:
                        route = ROUTES.get(name)
                        if route is None:
                            continue
                        q = claim(route)
                        if q:
                            route_tasks.append(asyncio.create_task(run_route(route, q)))
                elif kind == "error":
                    stats["refine_error"] = str(payload)[:300]
                if dl.expired(rank_reserve + 0.3):
                    break
        except asyncio.CancelledError:
            stats["refine_error"] = "cancelled"
        except BaseException as e:
            stats["refine_error"] = f"{type(e).__name__}: {e}"
        timings["refine_ms"] = (time.perf_counter() - t0) * 1000
        # Route hits land on the queue the consumer has already left; they are
        # read here, through the same gate.
        if route_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*route_tasks, return_exceptions=True),
                    timeout=max(0.05, min(1.5, dl.remaining - rank_reserve - 0.5)))
            except asyncio.TimeoutError:
                for t in route_tasks:
                    t.cancel()
        while not cand_q.empty():
            if admit(cand_q.get_nowait()):
                counts["refine"] += 1
        # The second wave gets a window of its own, measured from now.
        wave = {t for t in fetch_tasks if t not in before and not t.done()}
        if wave:
            window = max(0.05, min(REFINE_WINDOW, dl.remaining - rank_reserve))
            cutoff = time.perf_counter() + window
            while wave:
                left = cutoff - time.perf_counter()
                if left <= 0:
                    break
                _done, wave = await asyncio.wait(
                    wave, timeout=min(0.08, left),
                    return_when=asyncio.FIRST_COMPLETED)
            for t in wave:
                t.cancel()
            if wave:
                await asyncio.gather(*wave, return_exceptions=True)
                stats["fetch_cancelled"] = stats.get("fetch_cancelled", 0) + len(wave)
        stats["rounds"] = 2
        timings["round2_ms"] = (time.perf_counter() - t0) * 1000

    @staticmethod
    def _findings(query: str, plan: Plan, docs: list[Doc], prescore: Prescore,
                  unusable, *, shown: Sequence[str], page: int) -> Findings:
        """The first round as a report the planner can read.

        Found pages are ordered by the cross-encoder score the prescorer computed
        as they landed (title fit where it has not), each with the window the
        ranker would have quoted, so the planner sees not only which pages came
        back but whether they answer.
        """
        vocabs = Liberdex._vocabs(query, plan)
        scored: list[tuple[float, Doc]] = []
        for d in docs:
            if not d.ok:
                continue
            rel = prescore.get(d, 0.0)
            if rel is None:
                rel = _title_fit(d.title or "", vocabs) if vocabs else 0.0
            scored.append((float(rel), d))
        scored.sort(key=lambda x: -x[0])
        terms = prescore.terms(plan.lang)
        found: list[tuple[str, str, float, str]] = []
        for rel, d in scored[:FINDINGS_PAGES]:
            excerpt = ""
            try:
                parts, _, _ = select_windows(d.text or "", d.description or "",
                                             terms, FINDINGS_EXCERPT, 1)
                if parts:
                    excerpt = parts[0][0]
            except Exception:
                excerpt = ""
            if not excerpt:
                excerpt = (d.description or d.text or "")[:FINDINGS_EXCERPT]
            found.append((" ".join((d.title or "").split())[:120],
                          d.final_url or d.url, round(rel, 2), excerpt))
        dead: list[str] = []
        for d in docs:
            if len(dead) >= FINDINGS_DEAD:
                break
            src = d.candidate.source if d.candidate else ""
            if src in ("llm", "hub") and unusable(d):
                dead.append(d.url)
        return Findings(found=found, dead=dead,
                        shown=list(shown)[:FINDINGS_SHOWN], page=page)

    # ------------------------------------------------------------- expansion
    @staticmethod
    def _vocabs(query: str, plan: Plan) -> list[frozenset[str]]:
        """The query and each expansion as separate bags of content words.

        Separate, because an anchor is a handful of words and the query plus
        every expansion is dozens: recall against the union asks one anchor to
        contain a third of the whole vocabulary. Content words only, so two
        how-to titles do not match on {how, to, a}.
        """
        def vocab(text: str) -> frozenset[str]:
            # Frozen so the n-gram expansion is computed once per query, not once
            # per link scored against it.
            return frozenset(t.lower() for t in content_terms(text, plan.lang))
        vocabs = [vocab(query)] + [vocab(e) for e in plan.expansions[:4]]
        return [v for v in vocabs if v]

    @staticmethod
    def _section(url: str) -> str:
        """The part of a site a page belongs to: its path, minus a filename.

        `/insights` is a section. `/research/italy.aspx` is a page inside
        `/research`, and treating its own address as the section would let
        nothing be a descendant of it.
        """
        try:
            path = unquote(urlsplit(url).path)
        except ValueError:
            return ""
        segs = [x for x in path.split("/") if x]
        if segs and "." in segs[-1]:
            segs.pop()
        return "/" + "/".join(segs) if segs else ""

    def _link_fit(self, url: str, anchor: str, vocabs: list[set[str]],
                  lang: str, years, context: str = "") -> float:
        """How well a link says it is the page the query wanted.

        Three things say what a link is. The anchor is what the site calls it.
        The address is what the site filed it under, and often names the city,
        the document type and the quarter where the anchor says "Q2". The period,
        when the query named one, is a constraint the link meets or fails: a
        report and its predecessor from years earlier sit side by side under
        near-identical anchors, and only the address tells them apart.
        """
        bag = {t.lower() for t in content_terms(anchor, lang)}
        try:
            path = unquote(urlsplit(url).path).lower()
        except ValueError:
            path = ""
        # best_f1 expands the anchor's n-grams once for all vocabularies.
        fit = best_f1(vocabs, bag) if bag else 0.0
        # The address is weaker evidence than the anchor, since a section root
        # inherits its parent's words: it can raise a score but never carry one.
        fit = max(fit, 0.85 * path_fit(vocabs, path))
        # The sentence around the link is weaker still, but it is the only
        # evidence there is when the anchor is a brand name.
        if context:
            cbag = {t.lower() for t in content_terms(context, lang)}
            if cbag:
                fit = max(fit, 0.6 * best_f1(vocabs, cbag))
        if years:
            d = date_fit("", path, anchor, years)
            # Multiplicative going up, additive going down. A year is a filter on
            # documents that are already plausible, so it must not promote a page
            # on the number alone; failing the period is a fact about the link
            # regardless of how well it reads.
            fit = fit * (1.0 + 0.6 * d) if d > 0 else fit + 0.30 * d
        return fit

    def _follow(self, query: str, plan: Plan, sources: list[Doc], dispatch,
                counts: dict, *, reason: str, cross_host: float,
                limit: int) -> None:
        """Follow the links whose anchor text matches the query best."""
        vocabs = self._vocabs(query, plan)
        if not vocabs:
            return
        years, _ = query_period(query)
        scored: list[tuple[float, str, str, bool, str]] = []
        per_src: dict[str, int] = {}
        seen_link: set[str] = set()
        for src in sources:
            shost = registrable(src.site)
            section = self._section(src.final_url)
            for url, anchor, ctx in src.links:
                if url in seen_link:
                    continue
                seen_link.add(url)
                off_site = registrable(
                    (url.split("/")[2] if "://" in url else "")) != shost
                inside = (
                    not off_site and len(section) > 1
                    and self._section(url).startswith(section + "/")
                )
                fit = self._link_fit(url, anchor, vocabs, plan.lang, years, ctx)
                if fit < (HUB_SECTION_FIT if inside else HUB_LINK_FIT):
                    continue
                per_src[src.final_url] = per_src.get(src.final_url, 0) + int(inside)
                scored.append((fit + (0.12 if inside else 0.0)
                               + (cross_host if off_site else 0.0),
                               url, anchor, inside, src.final_url))
        # A page that indexes its own section has said what it is for. Its
        # remaining links are chrome (the country picker, the services menu, the
        # branch offices), so an off-section link off such a page has to beat one
        # off a page with nothing else to offer.
        scored = [x for x in scored
                  if x[3] or per_src.get(x[4], 0) < 2 or x[0] >= HUB_LINK_FIT + 0.12]
        scored.sort(key=lambda x: -x[0])
        n_inside = 0
        for score, url, anchor, inside, _src in scored:
            if counts["expand"] >= limit:
                break
            if inside:
                if n_inside >= HUB_SECTION_LINKS:
                    continue
                n_inside += 1
            counts["expand"] += 1
            dispatch(Candidate(
                url=url, title=anchor, reason=f"{reason} ({score:.2f})",
                prior=min(0.75, 0.4 + score / 2), source="expand",
            ), want_links=False)

    def _follow_section(self, query: str, plan: Plan, doc: Doc, dispatch,
                        picked: set[str], budget: int) -> int:
        """Follow one landed page's links into its own section, immediately.

        Scoring every link after the first wave drains puts the fetch that finds
        the answer at the end of the deadline, where a slow origin never makes it.
        The section rule is precise enough to act on the moment a page lands: a
        page listing its own descendants is a directory of them.
        """
        section = self._section(doc.final_url)
        if not doc.links or len(section) <= 1 or budget <= 0:
            return 0
        vocabs = self._vocabs(query, plan)
        if not vocabs:
            return 0
        years, _ = query_period(query)
        shost = registrable(doc.site)
        ident = max(
            max((_f1(v, {t.lower() for t in content_terms(doc.title or "",
                                                          plan.lang)})
                 for v in vocabs), default=0.0),
            path_fit(vocabs, urlsplit(doc.final_url).path.lower()),
        )
        bar = max(HUB_SECTION_FIT, ident * 0.5)
        scored: list[tuple[float, str, str]] = []
        for url, anchor, ctx in doc.links:
            if registrable(url.split("/")[2] if "://" in url else "") != shost:
                continue
            if not self._section(url).startswith(section + "/"):
                continue
            fit = self._link_fit(url, anchor, vocabs, plan.lang, years, ctx)
            if fit < bar:
                continue
            scored.append((fit, url, anchor))
        if not scored:
            return 0
        scored.sort(key=lambda x: -x[0])
        fired = 0
        for fit, url, anchor in scored:
            if fired >= budget:
                break
            key = _unversioned(url)
            if key in picked:
                continue
            picked.add(key)
            if dispatch(Candidate(
                url=url, title=anchor, reason=f"section link ({fit:.2f})",
                prior=min(0.8, 0.45 + fit / 2), source="deep",
            ), want_links=False):
                fired += 1
        return fired

    def _follow_hubs(
        self, query: str, plan: Plan, hubs: list[Doc], dispatch, counts: dict,
        limit: int = HUB_LINKS,
    ) -> None:
        # A hub exists to point elsewhere, so leaving its domain is the point.
        self._follow(query, plan, hubs, dispatch, counts,
                     reason="linked from hub", cross_host=0.05,
                     limit=limit)

    def _follow_deep(self, query: str, plan: Plan, docs: list[Doc], dispatch,
                     counts: dict, picked: Optional[set[str]] = None,
                     budget: int = DEEP_LINKS) -> int:
        """Follow the links of pages that point at the question better than they
        answer it.

        A page whose own title and address describe the query is the page that
        was wanted; a page carrying a link whose anchor describes it better is a
        directory one hop in front of the answer. Nothing is classified as a hub
        and no domain needs a rule: the comparison is between what a page says it
        is and what its links say they are. On the page the planner got right,
        nothing clears the bar.
        """
        vocabs = self._vocabs(query, plan)
        if not vocabs:
            return 0
        years, _ = query_period(query)
        # A substring pre-filter: tokenising every anchor on forty pages is
        # expensive, testing a few substrings first is not. Built from the query
        # and the planner's expansions, since the expansions are the vocabulary
        # an authoritative page prints and the right anchor is written in.
        keys = {t.lower() for v in vocabs for t in v if len(t) >= 4}
        keys |= {y for y in years}
        if not keys:
            return 0

        def vocab(text: str) -> frozenset[str]:
            # Frozen so the n-gram expansion is computed once per query, not once
            # per link scored against it.
            return frozenset(t.lower() for t in content_terms(text, plan.lang))

        scored: dict[str, tuple[float, str, bool]] = {}
        for d in docs:
            if not d.links or not d.ok:
                continue
            # How well the page identifies itself as the answer. Its links have to
            # clear this.
            ident = max(
                max((_f1(v, vocab(d.title or "")) for v in vocabs), default=0.0),
                path_fit(vocabs, urlsplit(d.final_url).path.lower()),
            )
            floor = max(DEEP_LINK_FIT, ident + DEEP_MARGIN)
            shost = registrable(d.site)
            section = self._section(d.final_url)
            for url, anchor, ctx in d.links:
                low = f"{anchor} {ctx}".lower()
                off_site = registrable(
                    url.split("/")[2] if "://" in url else "") != shost
                # A page linking deeper into its own section is a directory of that
                # section, whatever it calls itself. Inside it the anchor does not
                # have to carry the evidence.
                inside = (not off_site and len(section) > 1
                          and self._section(url).startswith(section + "/"))
                if not inside and not any(k in low for k in keys) and not (
                        years and any(y in url for y in years)):
                    continue
                fit = self._link_fit(url, anchor, vocabs, plan.lang, years, ctx)
                # Unlike a hub, staying on the site is the point: the planner already
                # established which site holds the answer.
                if off_site:
                    fit -= 0.06
                if fit < (max(HUB_SECTION_FIT, ident * 0.5) if inside else floor):
                    continue
                if inside:
                    fit += 0.12
                best = scored.get(url)
                if best is None or fit > best[0]:
                    scored[url] = (fit, anchor, inside)
        if not scored:
            return 0
        fired = 0
        # Seeded with what is already in hand, so a page already read is not
        # fetched again under a different release number.
        picked = set() if picked is None else picked
        picked |= {_unversioned(d.final_url) for d in docs}
        for url, (fit, anchor, _inside) in sorted(scored.items(),
                                                  key=lambda kv: -kv[1][0]):
            if fired >= budget:
                break
            # Documentation sites carry the same page under every release they
            # have published. They are one page.
            key = _unversioned(url)
            if key in picked:
                continue
            picked.add(key)
            if dispatch(Candidate(
                url=url, title=anchor, reason=f"deep link ({fit:.2f})",
                prior=min(0.8, 0.45 + fit / 2), source="deep",
            ), want_links=False):
                fired += 1
        return fired


# A release number, or a name that stands in for one. Every documentation site
# carries the same page under all of them.
_VERSION_SEG = re.compile(
    r"^(v?\d+(?:[._]\d+)*|devel|latest|stable|current|master|main|next|"
    r"release-[\w.]+)$", re.I)


def _unversioned(url: str) -> str:
    """The URL with release segments removed, as an identity for one document."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    segs = [x for x in parts.path.split("/") if x and not _VERSION_SEG.match(x)]
    host = (parts.hostname or "").lower()
    return f"{host}/{'/'.join(segs)}"


# Most CMSs append the site's name to the <title>. It names the publisher, not
# the subject, and as a query term it pulls every result back toward the site
# the query is trying to leave.
_SITE_SUFFIX = re.compile(r"\s*[|\u2013\u2014:-]\s*[^|\u2013\u2014:-]{1,40}$")


def _strip_site_suffix(title: str, site: str) -> str:
    if not title:
        return ""
    m = _SITE_SUFFIX.search(title)
    if not m:
        return title
    tail = m.group(0).strip(" |-\u2013\u2014:").lower()
    # The registrable label: en.wikipedia.org is named "wikipedia", not "en".
    label = registrable((site or "").lower()).split(".")[0]
    # Only drop the tail when it names the site; plenty of titles end in a
    # dash and a word.
    if label and label in re.findall(r"[a-z0-9]+", tail):
        return title[:m.start()].strip() or title
    return title


def _filter_by_date(results: list[Result], after, before,
                    require_date: bool) -> list[Result]:
    """Drop rows outside the window.

    A page with no parseable date is kept unless `require_date` says otherwise.
    Most primary sources (docs.python.org, postgresql.org) publish no date at
    all, so dropping the undated by default would empty the SERP for the pages
    liberdex is best at finding.
    """
    out: list[Result] = []
    for r in results:
        d = parse_date(r.published)
        if d is None:
            if not require_date:
                out.append(r)
            continue
        if after and d < after:
            continue
        if before and d > before:
            continue
        out.append(r)
    return out


async def _enrich(results: list[Result], docs: list[Doc], pool, *,
                  markdown: bool, want_media: bool) -> None:
    """Second-pass extraction, on the pages that made the SERP and no others.

    Markdown conversion is slow on a large page, so it runs in the pool over
    the ten ranked pages rather than over everything fetched.
    """
    by_url: dict[str, Doc] = {}
    for d in docs:
        by_url.setdefault(d.final_url, d)
        by_url.setdefault(d.url, d)

    pairs = [(r, by_url[r.url]) for r in results if r.url in by_url]
    if want_media:
        for r, d in pairs:
            r.favicon = d.favicon
            r.images = list(d.images)
    if not markdown:
        return
    loop = asyncio.get_running_loop()
    todo = [(r, d) for r, d in pairs if d.body]

    def convert(d: Doc) -> str:
        return to_markdown(decode(d.body, d.content_type), d.final_url)

    for (r, _d), md in zip(todo, await asyncio.gather(*(
        loop.run_in_executor(pool, convert, d) for _r, d in todo
    ))):
        if md:
            r.text = md
            r.text_chars = len(md)


async def search(
    query: str, *, planner=None, top_k: int = 10, budget: float = 4.0, **kw
) -> SearchResponse:
    """One-shot convenience wrapper. Prefer reusing a Liberdex instance."""
    async with Liberdex(planner=planner, **kw) as eng:
        return await eng.search(query, top_k=top_k, budget=budget)
