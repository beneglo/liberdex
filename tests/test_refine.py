"""The second pass: what makes deep deep, and what a page turn is.

Against a stub planner and a stub fetcher: no network, no LLM. The ranking
models load, as they do everywhere else in the suite.
"""
from __future__ import annotations

import asyncio

import pytest

from liberdex.cache import CachedPlanner, PlanStore
from liberdex.engine import Liberdex
from liberdex.planner.base import PlanParser
from liberdex.planner.prompt import build_refine_prompt
from liberdex.types import Candidate, Deadline, Doc, Findings

FIRST = ["I reference", "L en",
         "U 0.9 https://a.test/one", "X one thing | other thing",
         "U 0.8 https://b.test/two", "U 0.7 https://c.test/dead"]
SECOND = ["U 0.9 https://d.test/three", "U 0.8 https://e.test/four",
          "U 0.7 https://g.test/five", "H https://f.test/hub"]


class StubPlanner:
    model = "stub"
    n_urls = 14
    effort = "low"

    def __init__(self, second=SECOND) -> None:
        self.second = second
        self.refined: list[Findings] = []

    async def _emit(self, query, lines):
        parser = PlanParser(query)
        for line in lines:
            for cand in parser.feed(line + "\n"):
                yield ("candidate", cand)
            yield ("meta", parser.plan)
        for cand in parser.finish():
            yield ("candidate", cand)
        yield ("done", parser.plan)

    async def stream(self, query: str, deadline: Deadline):
        async for ev in self._emit(query, FIRST):
            yield ev

    async def refine(self, query: str, findings: Findings, deadline: Deadline):
        self.refined.append(findings)
        async for ev in self._emit(query, self.second):
            yield ev


class StubFetcher:
    """Every address resolves to a page about the query, except a dead one."""

    def __init__(self) -> None:
        self.fetched: list[str] = []

    async def start(self) -> None: ...
    async def aclose(self) -> None: ...

    async def _fetch_one(self, cand: Candidate, dl, reserve) -> Doc:
        self.fetched.append(cand.url)
        await asyncio.sleep(0.01)
        host = cand.url.split("/")[2]
        if "dead" in cand.url:
            return Doc(url=cand.url, final_url=cand.url, status=404,
                       site=host, candidate=cand, error="not found")
        slug = cand.url.rsplit("/", 1)[-1]
        return Doc(url=cand.url, final_url=cand.url, status=200, site=host,
                   title=f"Capital of Australia at {host}",
                   text=(f"The capital of Australia is Canberra, {host} notes "
                         f"in its article {slug}. " * 8
                         + f"Page {slug} on {host} has its own words: "
                         + " ".join(f"{slug}{i}" for i in range(60))),
                   candidate=cand)

    async def get_raw(self, *a, **kw):
        return 0, b"", "", ""


def engine(planner) -> Liberdex:
    return Liberdex(planner=planner, fetcher=StubFetcher(), speculate=False,
                    site_search=False, sitemap=False)


def search(eng: Liberdex, budget: float = 6.0, **kw):
    async def go():
        async with eng:
            return await eng.search("capital of australia", budget=budget,
                                    expand_hubs=False, **kw)
    return asyncio.run(go())


def test_standard_never_plans_twice():
    planner = StubPlanner()
    resp = search(engine(planner))
    assert planner.refined == []
    assert "rounds" not in resp.stats and "candidates_refine" not in resp.stats
    assert {r.url for r in resp.results} == {"https://a.test/one", "https://b.test/two"}


def test_deep_plans_again_over_the_first_round_and_ranks_both_waves():
    planner = StubPlanner()
    eng = engine(planner)
    resp = search(eng, refine=True)
    # The planner was handed the first round: the pages that came back, best
    # first with their fit and an excerpt, and the address that died.
    assert len(planner.refined) == 1
    f = planner.refined[0]
    assert {u for _, u, _, _ in f.found} == {"https://a.test/one", "https://b.test/two"}
    assert all(0.0 <= rel <= 1.0 for _, _, rel, _ in f.found)
    assert all("Canberra" in ex for _, _, _, ex in f.found)
    assert f.dead == ["https://c.test/dead"] and f.page == 1 and f.shown == []
    # Its answer was fetched, as a second wave, and ranked with the first.
    urls = {r.url for r in resp.results}
    assert {"https://d.test/three", "https://e.test/four"} <= urls
    assert resp.stats["rounds"] == 2
    assert resp.stats["candidates_refine"] == 4     # three pages and the hub
    assert "https://f.test/hub" in eng.fetcher.fetched
    assert {r.source for r in resp.results if r.url.startswith("https://d")} == {"refine"}
    assert resp.timings["refine_ms"] > 0 and resp.timings["round2_ms"] > 0


def test_a_page_turn_excludes_what_was_shown_and_tells_the_planner():
    planner = StubPlanner()
    eng = engine(planner)
    shown = ["https://a.test/one", "https://d.test/three"]
    resp = search(eng, page=2, exclude_urls=shown)
    # Never fetched, never returned.
    assert "https://a.test/one" not in eng.fetcher.fetched
    urls = [r.url for r in resp.results]
    assert not set(shown) & set(urls)
    assert "https://b.test/two" in urls and "https://e.test/four" in urls
    # The planner was asked, over the shown pages, for what comes next.
    f = planner.refined[0]
    assert f.page == 2 and f.shown == shown
    assert resp.stats["page"] == 2 and resp.stats["excluded"] == 2


def test_a_planner_without_a_second_pass_says_so():
    class OneShot(StubPlanner):
        refine = None
    resp = search(engine(OneShot()), refine=True)
    assert resp.stats["refine_skipped"] == "no second pass on this planner"
    assert "rounds" not in resp.stats


def test_no_time_left_means_no_second_pass():
    planner = StubPlanner()
    eng = engine(planner)

    async def go():
        async with eng:
            return await eng.search("capital of australia", budget=1.2,
                                    expand_hubs=False, refine=True)
    resp = asyncio.run(go())
    assert planner.refined == [] and resp.stats["refine_skipped"] == "no time left"


@pytest.mark.asyncio
async def test_the_second_pass_is_cached_per_query_and_page(tmp_path):
    store = PlanStore(str(tmp_path / "plans.sqlite3"))
    inner = StubPlanner()
    cp = CachedPlanner(inner, store=store, model="stub")
    f = Findings(found=[("t", "https://a.test/one", 0.9, "x")], page=2)

    async def urls(findings):
        out = []
        async for kind, payload in cp.refine("q", findings, Deadline(2.0)):
            if kind == "candidate":
                out.append(payload.url)
        return out

    first = await urls(f)
    await cp.aclose()
    assert first == ["https://d.test/three", "https://e.test/four",
                     "https://g.test/five"]
    assert len(inner.refined) == 1
    # Same query, same page: served from the store, the planner not asked.
    assert await urls(f) == first and len(inner.refined) == 1
    # Another page is another plan.
    await urls(Findings(page=3))
    await cp.aclose()
    assert len(inner.refined) == 2
    assert store.get("q", "stub", 0, cp.variant) is None       # first-pass key untouched
    assert store.get("q", "stub", 0, cp.variant + "|refine2") is not None


def test_the_second_pass_prompt_carries_the_report():
    f = Findings(found=[("Title", "https://a.test/one", 0.83, "an excerpt")],
                 dead=["https://c.test/dead"], shown=["https://s.test/seen"], page=2)
    p = build_refine_prompt("capital of australia", f, n_urls=5)
    assert "Emit 5 in total" in p
    assert "0.83  Title  <https://a.test/one>" in p and "an excerpt" in p
    assert "https://c.test/dead" in p and "https://s.test/seen" in p
    assert "page 2" in p and "Query: capital of australia" in p
    # No few-shot: the report replaces it, and the prompt is paid for per round.
    assert "asyncio" not in p and len(p) < 6000


# ---------------------------------------------------------------- plan grace
class SlowFirstPlanner(StubPlanner):
    """A planner whose first line takes `ttft` seconds, like a cold hosted one."""

    def __init__(self, ttft: float) -> None:
        super().__init__()
        self.ttft = ttft

    async def stream(self, query: str, deadline: Deadline):
        await asyncio.sleep(self.ttft)
        async for ev in self._emit(query, FIRST):
            yield ev


def test_the_fetch_window_starts_at_the_planners_first_url():
    """Budget 0.5s, planner 0.4s to its first URL: without grace the producer
    drain cancels the planner at 0.15s and nothing of the plan is read. With a
    grace the wait is granted up front, the unused part is handed back at the
    first URL, and the whole plan is fetched inside its own window."""
    p = SlowFirstPlanner(ttft=0.4)
    resp = search(engine(p), budget=0.5, plan_grace=1.0)
    assert "planner_error" not in resp.stats
    assert resp.stats["candidates_llm"] == 3
    assert resp.timings["plan_first_url_ms"] >= 350
    # What the planner used of its grace, not the whole grant.
    assert 350 <= resp.stats["plan_grace_ms"] <= 900
    assert len(resp.results) == 2 and resp.timings["total_ms"] < 1500

    cut = search(engine(SlowFirstPlanner(ttft=0.4)), budget=0.5, plan_grace=0.0)
    assert cut.stats["planner_error"] == "cancelled"
    assert cut.stats["candidates_llm"] == 0


def test_a_cache_hit_hands_the_whole_grace_back():
    resp = search(engine(StubPlanner()), budget=0.5, plan_grace=1.0)
    assert "planner_error" not in resp.stats
    assert resp.stats["plan_grace_ms"] < 100
    assert resp.timings["total_ms"] < 1000


class StallsAfterPlanner(StubPlanner):
    """Emits the plan's U lines at once, then stalls before saying done."""

    async def stream(self, query: str, deadline: Deadline):
        async for ev in self._emit(query, FIRST):
            if ev[0] == "done":
                await asyncio.sleep(5.0)
            yield ev


def test_a_cut_past_the_url_cap_is_not_a_lost_plan():
    # Cap 2 of 3 U lines: the third is read as tail, the stall is cut by the
    # deadline, and the cut is not reported as an error since the engine was
    # about to stop reading anyway.
    resp = search(engine(StallsAfterPlanner()), budget=0.6, max_llm_candidates=2)
    assert "planner_error" not in resp.stats
    # The tail line is a four-letter slug, which the tail rule declines.
    assert resp.stats["candidates_llm"] == 2
    # Uncapped (deep), the same cut is a plan lost.
    resp = search(engine(StallsAfterPlanner()), budget=0.6, max_llm_candidates=0)
    assert resp.stats["planner_error"] == "cancelled"


def test_pinned_rows_take_their_position():
    from liberdex.answer import judged
    from liberdex.rank import pin
    from liberdex.types import Result

    def row(u, **debug):
        return Result(url=u, title=u, snippet="", site="x", score=0.5, debug=debug)

    rows = [row("a"), row("b"), row("c", rank=2), row("d", rank=1)]
    assert [r.url for r in pin(rows)] == ["d", "c", "a", "b"]
    assert [r.url for r in pin([row("a"), row("b", rank=9)])] == ["a", "b"]
    # The answer judge read pages 0-3 and listed only "a": the pinned rows stay.
    out = judged(rows, [0], read=4, keep=3)
    assert [r.url for r in out] == ["d", "c", "a"]
