"""The plan cache must never persist a plan the planner did not finish."""
from __future__ import annotations

import asyncio

import pytest

from liberdex.cache import CachedPlanner, PlanStore
from liberdex.planner.base import PlanParser
from liberdex.types import Deadline

PLAN = ["I reference", "L en", "R wikipedia: x",
        "U 0.9 https://a.example/1", "X a | b | c", "H https://a.example/",
        *[f"U 0.{9 - i} https://site{i}.example/page" for i in range(1, 9)]]


class SlowPlanner:
    """Emits one plan line every `pace` seconds; finishes when they are out."""
    model = "fake"
    n_urls = 14
    effort = "low"

    def __init__(self, pace: float, min_budget: float = 0.0) -> None:
        self.pace = pace
        self.min_budget = min_budget
        self.calls = 0

    async def stream(self, query: str, deadline: Deadline):
        self.calls += 1
        parser = PlanParser(query)
        for line in PLAN:
            if deadline.expired():
                break
            await asyncio.sleep(self.pace)
            for cand in parser.feed(line + "\n"):
                yield ("candidate", cand)
            yield ("meta", parser.plan)
        for cand in parser.finish():
            yield ("candidate", cand)
        yield ("done", parser.plan)


def _urls(raw: str) -> int:
    return sum(1 for ln in raw.splitlines() if ln.startswith("U "))


@pytest.mark.asyncio
async def test_a_plan_the_caller_could_not_wait_for_still_lands_whole(tmp_path):
    """Caller leaves after 0.15s with a few URLs; the store gets all nine."""
    store = PlanStore(str(tmp_path / "plans.sqlite3"))
    inner = SlowPlanner(pace=0.03, min_budget=2.0)
    cp = CachedPlanner(inner, store=store, model="fake")
    got = []
    async for kind, payload in cp.stream("q", Deadline(0.15)):
        if kind == "candidate":
            got.append(payload)
    assert 0 < len(got) < 9, len(got)
    assert store.get("q", "fake", 0, cp.variant) is None  # not yet
    await cp.aclose()
    hit = store.get("q", "fake", 0, cp.variant)
    assert hit is not None and _urls(hit[0]) == 9
    # And the next search is served from the store, not the planner.
    async for _ in cp.stream("q", Deadline(1.0)):
        pass
    assert inner.calls == 1


@pytest.mark.asyncio
async def test_a_plan_cut_by_its_own_deadline_is_not_stored(tmp_path):
    store = PlanStore(str(tmp_path / "plans.sqlite3"))
    inner = SlowPlanner(pace=0.05, min_budget=0.0)   # needs ~0.7s, gets 0.2
    # No finish floor: the point is a planner cut by its own clock.
    cp = CachedPlanner(inner, store=store, model="fake", finish_floor=0.0)
    async for _ in cp.stream("q", Deadline(0.2)):
        pass
    await cp.aclose()
    assert store.get("q", "fake", 0, cp.variant) is None


@pytest.mark.asyncio
async def test_a_caller_that_stops_reading_early_does_not_lose_the_plan(tmp_path):
    """The engine closes the generator once it has enough URLs."""
    store = PlanStore(str(tmp_path / "plans.sqlite3"))
    inner = SlowPlanner(pace=0.01, min_budget=1.0)
    cp = CachedPlanner(inner, store=store, model="fake")
    n = 0
    async for kind, _ in cp.stream("q", Deadline(2.0)):
        if kind == "candidate":
            n += 1
            if n == 2:
                break
    await cp.aclose()
    hit = store.get("q", "fake", 0, cp.variant)
    assert hit is not None and _urls(hit[0]) == 9


@pytest.mark.asyncio
async def test_the_finish_floor_lets_a_short_search_still_store_a_whole_plan(tmp_path):
    """A streaming planner declares no `min_budget`; the store still gets the
    whole plan because the detached reader has a floor of its own."""
    store = PlanStore(str(tmp_path / "plans.sqlite3"))
    inner = SlowPlanner(pace=0.03, min_budget=0.0)   # needs ~0.4s
    cp = CachedPlanner(inner, store=store, model="fake", finish_floor=2.0)
    got = [ev async for ev in cp.stream("q", Deadline(0.1))]
    assert len(got) < 20
    await cp.aclose()
    raw, _ = store.get("q", "fake", 0, cp.variant)
    assert raw.count("\nU ") + raw.startswith("U ") == 9


def test_the_floor_comes_from_the_inner_planner_when_it_declares_one():
    class Declares(SlowPlanner):
        finish_floor = 7.5
    assert CachedPlanner(Declares(pace=0.0), store=PlanStore(":memory:"),
                         model="fake").finish_floor == 7.5
    assert CachedPlanner(SlowPlanner(pace=0.0), store=PlanStore(":memory:"),
                         model="fake").finish_floor == 15.0
