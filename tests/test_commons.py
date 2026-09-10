import asyncio
import json
import time

import httpx
import pytest

from liberdex.cache import CachedPlanner, PlanStore
from liberdex.commons import Commons, bag, commons_keys, render_plan
from liberdex.planner.base import plan_from_dict
from liberdex.types import Deadline

PLAN = ("I reference\nL de\nR wikipedia: Bochum\n"
        "U 0.9 https://www.city.example/\nX Bochum Stadt | Bochum Ruhrgebiet\n"
        "H https://de.encyclopedia.example/wiki/Bochum\n"
        "U 0.8 https://de.encyclopedia.example/wiki/Bochum\nU 0.7 https://www.tourism.example/\n"
        "U 0.6 https://www.uni.example/\nU 0.5 https://www.club.example/\n")


@pytest.fixture(autouse=True)
def _commons_on(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.delenv("LIBERDEX_COMMONS_KEY", raising=False)


# ------------------------------------------------------------------ keys
def test_the_same_question_worded_differently_is_one_key():
    assert commons_keys("How do I cancel an asyncio task in Python?")[0] == \
        commons_keys("python asyncio cancel task")[0]
    assert commons_keys("BOCHUM.") == commons_keys("bochum")
    assert commons_keys("Köln Sehenswürdigkeiten") == commons_keys("koln sehenswurdigkeiten")
    assert bag("北京 天气") == ["北京", "天气"]


def test_a_generic_word_relaxes_to_the_subject_and_never_past_it():
    stadt, bochum = commons_keys("Stadt Bochum"), commons_keys("Bochum")
    assert len(stadt) == 2 and len(bochum) == 1
    assert stadt[1] == bochum[0]           # "Stadt Bochum" falls back to "Bochum"
    assert stadt[0] != bochum[0]           # but is stored under its own key
    # Two real words: nothing to drop, so nothing to fall back to.
    assert len(commons_keys("bochum wetter")) == 1
    # Everything generic: no relaxed key that would match every question.
    assert len(commons_keys("official website")) == 1


def test_a_quoted_phrase_is_one_word():
    assert '"exact error text"' in bag('"exact error text" python')
    assert commons_keys('"exact error text" python') != commons_keys("exact error text python")


def test_a_caller_written_plan_renders_to_the_line_protocol():
    plan = plan_from_dict("bochum", {
        "candidates": [{"url": "https://www.city.example/", "prior": 0.9},
                       {"url": "https://de.encyclopedia.example/wiki/Bochum", "prior": 0.8}],
        "routes": {"wikipedia": "Bochum"}, "hubs": ["https://de.encyclopedia.example/wiki/Bochum"],
        "expansions": ["Bochum Stadt"], "intent": "reference", "lang": "de"})
    raw = render_plan(plan)
    lines = raw.splitlines()
    assert lines[0] == "I reference" and lines[1] == "L de"
    assert lines[2].startswith("R wikipedia: Bochum")
    assert lines[3].startswith("U 0.90 https://www.city.example/")
    assert "X Bochum Stadt" in lines and "H https://de.encyclopedia.example/wiki/Bochum" in lines
    assert lines[-1].startswith("U 0.80 https://de.encyclopedia.example/wiki/Bochum")


# ---------------------------------------------------------------- client
class Server:
    """A commons that answers from a dict and remembers what it was sent."""

    def __init__(self, plans=None, status=200, register=True, register_delay=0.0):
        self.plans = plans or {}
        self.status = status
        self.register = register
        self.register_delay = register_delay
        self.calls: list[tuple[str, object]] = []
        self.agents: list[str] = []
        # Fields a newer server adds to a lookup reply, beside `raw`.
        self.extra: dict = {}

    def transport(self):
        async def handle(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content or b"{}")
            self.calls.append((request.url.path, body))
            self.agents.append(request.headers.get("user-agent", ""))
            if request.url.path == "/commons/register":
                if self.register_delay:
                    await asyncio.sleep(self.register_delay)
                return (httpx.Response(200, json={"key": "cx_test"}) if self.register
                        else httpx.Response(429, json={"detail": "later"}))
            if request.headers.get("authorization") != "Bearer cx_test":
                return httpx.Response(401)
            if self.status != 200:
                return httpx.Response(self.status)
            if request.url.path == "/commons/lookup":
                for k in body["keys"]:
                    if k in self.plans:
                        return httpx.Response(200, json={"key": k, "raw": self.plans[k],
                                                         **self.extra})
                return httpx.Response(404)
            return httpx.Response(200, json={"ok": True})
        return httpx.MockTransport(handle)


def test_the_first_use_registers_and_keeps_the_key(tmp_path):
    srv = Server(plans={commons_keys("bochum")[0]: PLAN})
    c = Commons(transport=srv.transport())
    raw = asyncio.run(c.lookup(commons_keys("Stadt Bochum")))
    assert raw == PLAN and c.key == "cx_test"
    from liberdex import envfile
    assert envfile.read()["LIBERDEX_COMMONS_KEY"] == "cx_test"
    assert srv.calls[0][0] == "/commons/register"
    assert srv.calls[1][1]["keys"] == commons_keys("Stadt Bochum")
    assert c.stats["hits"] == 1
    asyncio.run(c.aclose())


def test_a_miss_is_quiet_and_a_refusal_mutes():
    srv = Server()
    c = Commons(key="cx_test", transport=srv.transport())
    assert asyncio.run(c.lookup(commons_keys("nothing here"))) is None
    assert c.stats["misses"] == 1 and c.enabled
    bad = Commons(key="cx_wrong", transport=srv.transport())
    assert asyncio.run(bad.lookup(commons_keys("x y"))) is None
    assert not bad.enabled and "refused" in bad.error
    down = Commons(key="cx_test", transport=Server(status=503).transport())
    assert asyncio.run(down.lookup(commons_keys("x y"))) is None
    assert not down.enabled
def test_offers_and_reports_go_out_in_one_batch():
    srv = Server()

    async def run():
        c = Commons(key="cx_test", transport=srv.transport())
        assert c.offer_plan("Stadt Bochum", PLAN, model="opus", planner="claude-cli")
        assert c.report("Stadt Bochum", [{"url": "https://www.city.example/", "ok": True, "serp": True}])
        await c.aclose()   # drains the queue without waiting for the timer
    asyncio.run(run())
    paths = [p for p, _ in srv.calls]
    assert paths == ["/commons/offer", "/commons/report"]
    offer = srv.calls[0][1][0]
    assert offer["key"] == commons_keys("Stadt Bochum")[0]
    assert offer["key"].endswith("|bochum stadt") and offer["raw"] == PLAN
    assert offer["model"] == "opus" and offer["planner"] == "claude-cli"
    assert srv.calls[1][1][0]["urls"][0]["ok"] is True


# ------------------------------------------------------- cached planner
class Inner:
    name = "inner"
    model = "m"
    n_urls = 14
    effort = "low"

    def __init__(self):
        self.calls = 0

    async def stream(self, query, deadline):
        self.calls += 1
        from liberdex.planner.base import PlanParser
        p = PlanParser(query)
        for line in PLAN.splitlines(keepends=True):
            for cand in p.feed(line):
                yield ("candidate", cand)
        for cand in p.finish():
            yield ("candidate", cand)
        yield ("done", p.plan)


async def _events(planner, query):
    out = []
    async for ev in planner.stream(query, Deadline(5.0)):
        out.append(ev)
    return out


def test_a_commons_hit_skips_the_planner_and_lands_in_the_local_store(tmp_path):
    srv = Server(plans={commons_keys("bochum")[0]: PLAN})
    c = Commons(key="cx_test", transport=srv.transport())
    inner = Inner()
    store = PlanStore(str(tmp_path / "plans.sqlite3"))
    cp = CachedPlanner(inner, store=store, commons=c)
    evs = asyncio.run(_events(cp, "Stadt Bochum"))
    assert evs[0] == ("source", "commons")
    assert sum(1 for k, _ in evs if k == "candidate") >= 4
    assert inner.calls == 0
    # The next search of the same question is a local hit; the commons is not asked.
    evs = asyncio.run(_events(cp, "Stadt Bochum"))
    assert evs[0] == ("source", "cache") and inner.calls == 0
    assert sum(1 for p, _ in srv.calls if p == "/commons/lookup") == 1


def test_a_commons_miss_runs_the_planner_and_offers_what_it_wrote(tmp_path):
    srv = Server()
    c = Commons(key="cx_test", transport=srv.transport())
    inner = Inner()
    store = PlanStore(str(tmp_path / "plans.sqlite3"))

    async def run():
        cp = CachedPlanner(inner, store=store, commons=c)
        evs = await _events(cp, "python asyncio cancel task")
        assert evs[0][0] == "candidate" and inner.calls == 1
        await cp.aclose()
        await c.aclose()
    asyncio.run(run())
    offers = [b for p, b in srv.calls if p == "/commons/offer"]
    assert len(offers) == 1 and offers[0][0]["key"] == commons_keys("python asyncio cancel task")[0]
    assert offers[0][0]["planner"] == "inner" and offers[0][0]["model"] == "m"
def test_the_engine_hands_its_commons_to_the_cached_planner(monkeypatch):
    from liberdex.engine import Liberdex
    cp = CachedPlanner(Inner(), store=PlanStore(":memory:"))
    eng = Liberdex(planner=cp, commons=Commons(key="cx_test", register=False))
    assert cp.commons is eng.commons
    off = Liberdex(planner=CachedPlanner(Inner(), store=PlanStore(":memory:")), commons=False)
    assert off.commons is None


def test_forget_deletes_upstream_and_drops_the_key(tmp_path):
    calls = []

    async def handle(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, request.headers.get("authorization")))
        return httpx.Response(200, json={"forgotten": {}})
    from liberdex import envfile
    envfile.write({"LIBERDEX_COMMONS_KEY": "cx_test"})
    c = Commons(transport=httpx.MockTransport(handle))
    assert asyncio.run(c.forget()) is True
    assert calls == [("DELETE", "/commons/me", "Bearer cx_test")]
    assert c.key == "" and "LIBERDEX_COMMONS_KEY" not in envfile.read()


def test_the_commons_is_asked_with_the_local_cache_off(monkeypatch):
    """`--no-cache` / LIBERDEX_PLAN_CACHE=0 switch off the local store only:
    the commons is still looked up, and still offered what the planner wrote."""
    from liberdex.cache import NoStore
    from liberdex.planner.factory import make_planner
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    p = make_planner("openai", "openrouter/some/model", cache=False)
    assert isinstance(p, CachedPlanner) and isinstance(p.store, NoStore)
    assert p.store.get("q", "m") is None and p.store.count() == 0
    p.store.put("q", "m", PLAN, 1.0)
    assert p.store.get("q", "m") is None


def test_a_lookup_does_not_wait_for_a_slow_registration():
    """Registration sits before the planner on the first search of a fresh
    install. The lookup waits its own timeout for it and no more."""
    import time
    srv = Server(plans={commons_keys("bochum")[0]: PLAN}, register_delay=1.0)

    async def go():
        c = Commons(transport=srv.transport())
        t0 = time.perf_counter()
        first = await c.lookup(commons_keys("bochum"))
        took = time.perf_counter() - t0
        await c._registering
        second = await c.lookup(commons_keys("bochum"))
        await c.aclose()
        return first, took, second

    first, took, second = asyncio.run(go())
    assert first is None and took < 0.6
    assert second == PLAN and srv.calls[0][0] == "/commons/register"


def test_the_engine_registers_at_start_off_the_critical_path():
    from liberdex.engine import Liberdex

    class NoFetcher:
        async def start(self): ...
        async def aclose(self): ...

    srv = Server()

    async def go():
        eng = Liberdex(planner=None, fetcher=NoFetcher(), speculate=False,
                       commons=Commons(transport=srv.transport()))
        await eng.start()
        assert eng.commons._registering is not None
        await eng.commons._registering
        await eng.aclose()

    asyncio.run(go())
    assert [c[0] for c in srv.calls] == ["/commons/register"]


# ------------------------------------------------------- the report
def test_the_report_compares_plan_urls_the_way_the_fetch_wave_keys_them():
    """`dispatched` holds normalised keys; a plan URL with a trailing slash or a
    www. host must still be found there, or nothing is ever reported."""
    from liberdex.engine import Liberdex
    from liberdex.rank import normalize_url
    from liberdex.types import Candidate, Doc, Plan, Result

    class Stub:
        writes = True
        reports: list = []

        def report(self, query, urls):
            self.reports.append(urls)
            return True

        def offer_plan(self, *a, **k):
            return True

    eng = Liberdex.__new__(Liberdex)
    eng.commons = Stub()
    plan = Plan(query="Stadt Bochum", candidates=[
        Candidate(url="https://www.city.example/", source="llm"),
        Candidate(url="https://example.org/a/", source="llm"),
        Candidate(url="https://never.example/", source="llm"),
    ])
    dispatched = {normalize_url("https://www.city.example/"), normalize_url("https://example.org/a/")}
    docs = [Doc(url="https://city.example", final_url="https://city.example/", status=200)]
    results = [Result(url="https://city.example/", title="t", snippet="", site="city.example", score=1.0)]
    eng._share("Stadt Bochum", plan, False, dispatched, docs, results)
    assert eng.commons.reports == [[
        {"url": "https://www.city.example/", "ok": True, "serp": True},
        {"url": "https://example.org/a/", "ok": False, "serp": False},
    ]]


# ------------------------------------------------------ a newer server
def test_every_call_names_the_client_version():
    """The server may one day answer an old client differently from a new
    one. It can only do that if every call says which it is."""
    from liberdex.commons import USER_AGENT
    assert USER_AGENT.startswith("liberdex-commons/1 liberdex/")
    srv = Server(plans={})
    c = Commons(key="cx_test", transport=srv.transport())
    asyncio.run(c.lookup(commons_keys("bochum")))
    assert srv.agents and all(a == USER_AGENT for a in srv.agents)


def test_a_newer_server_may_say_more_than_the_client_reads(tmp_path):
    """A lookup reply with fields this client does not know, and a plan with
    a line tag the parser does not know, are a hit and a plan: the client
    takes what it understands and drops the rest, so a server can grow
    without every installed client moving first."""
    newer = PLAN + "Q something a later protocol says\n"
    srv = Server(plans={commons_keys("bochum")[0]: newer})
    srv.extra = {"since": "commons2", "note": {"kind": "x"}}
    c = Commons(key="cx_test", transport=srv.transport())
    inner = Inner()
    cp = CachedPlanner(inner, store=PlanStore(str(tmp_path / "plans.sqlite3")), commons=c)
    evs = asyncio.run(_events(cp, "Stadt Bochum"))
    assert evs[0] == ("source", "commons") and inner.calls == 0
    plan = [p for k, p in evs if k == "done"][0]
    assert len(plan.candidates) == 4 and plan.lang == "de"
    assert not any(cand.url.startswith("Q") for cand in plan.candidates)


def test_failures_in_a_row_back_off_and_an_answer_resets():
    from liberdex import commons as mod
    srv = Server(status=503)
    c = Commons(key="cx_test", transport=srv.transport())
    waits = []
    for _ in range(8):
        c._muted_until = 0.0
        assert asyncio.run(c.lookup(commons_keys("x y"))) is None
        waits.append(c._muted_until - time.monotonic())
    assert 55 < waits[0] <= mod.MUTE
    assert all(b > a * 1.5 for a, b in zip(waits, waits[1:4]))
    assert waits[-1] <= mod.MUTE_MAX < waits[-1] + 5
    srv.status = 200
    c._muted_until = 0.0
    assert asyncio.run(c.lookup(commons_keys("x y"))) is None
    assert c.enabled and not c.error and c._strikes == 0
    srv.status = 503
    assert asyncio.run(c.lookup(commons_keys("x y"))) is None
    assert c._muted_until - time.monotonic() <= mod.MUTE
