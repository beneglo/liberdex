"""Plan caching and replay, plus the two small per-host stores.

CachedPlanner makes a repeated query cheap: the plan is the expensive part
and it is stable, so a hit turns a four-second search into a sub-second one.
ReplayPlanner reads the same store and re-emits a recorded plan with its
recorded time to first token, so a search reruns without a planner. SiteSearchAtlas and SitemapStore remember what a host has
said about itself.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
import threading
import time
import zlib
from typing import AsyncIterator, Iterable, Optional

from .commons import commons_keys
from .planner.base import PlanParser
from .types import Deadline, Findings

DEFAULT_PATH = os.path.expanduser("~/.cache/liberdex/plans.sqlite3")

# Bumped whenever the line protocol changes. Plans recorded under an older
# protocol parse fine but are missing whatever the new one added, so they are
# kept under a separate key rather than silently serving a downgrade.
PROTOCOL_VERSION = "v3"


class _Sqlite:
    """One connection per thread over a WAL database at `path`."""

    def __init__(self, path: str, schema: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._local = threading.local()
        self._conn().execute(schema)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn


class PlanStore(_Sqlite):
    def __init__(self, path: str = DEFAULT_PATH) -> None:
        super().__init__(path, (
            "CREATE TABLE IF NOT EXISTS plans ("
            " key TEXT PRIMARY KEY, query TEXT, model TEXT,"
            " raw TEXT, created REAL, gen_ms REAL)"))

    @staticmethod
    def key(query: str, model: str, variant: str = "") -> str:
        # `variant` carries every planner setting that changes what the plan
        # contains rather than how it is worded, above all how many URLs were
        # asked for, so a settings change cannot be served a stale plan.
        return hashlib.sha1(
            f"{PROTOCOL_VERSION}\x00{model}\x00{variant}\x00"
            f"{query.strip().lower()}".encode()
        ).hexdigest()

    def get(self, query: str, model: str, ttl: float = 0.0,
            variant: str = "") -> Optional[tuple[str, float]]:
        row = self._conn().execute(
            "SELECT raw, created, gen_ms FROM plans WHERE key=?",
            (self.key(query, model, variant),),
        ).fetchone()
        if not row:
            return None
        raw, created, gen_ms = row
        if ttl and time.time() - created > ttl:
            return None
        return raw, gen_ms or 0.0

    def put(self, query: str, model: str, raw: str, gen_ms: float,
            variant: str = "") -> None:
        self._conn().execute(
            "INSERT OR REPLACE INTO plans(key, query, model, raw, created, gen_ms)"
            " VALUES(?,?,?,?,?,?)",
            (self.key(query, model, variant), query, model, raw, time.time(), gen_ms),
        )

    def count(self) -> int:
        return self._conn().execute("SELECT COUNT(*) FROM plans").fetchone()[0]


class NoStore:
    """The local plan cache switched off. Every search calls the planner; the
    shared cache is still asked first and still offered what the planner
    wrote."""

    path = ""

    def get(self, query: str, model: str, ttl: float = 0.0,
            variant: str = "") -> Optional[tuple[str, float]]:
        return None

    def put(self, query: str, model: str, raw: str, gen_ms: float,
            variant: str = "") -> None:
        return None

    def count(self) -> int:
        return 0


async def _emit(raw: str, query: str, ttft: float, tok_per_s: float
                ) -> AsyncIterator[tuple[str, object]]:
    """Replay recorded plan text with realistic streaming timing."""
    parser = PlanParser(query)
    if ttft > 0:
        await asyncio.sleep(ttft)
    # ~4 chars per token is close enough for pacing purposes.
    per_line = 0.0 if tok_per_s <= 0 else 1.0 / tok_per_s
    for line in raw.splitlines(keepends=True):
        if per_line:
            await asyncio.sleep(per_line * max(1.0, len(line) / 4.0))
        for cand in parser.feed(line):
            yield ("candidate", cand)
        yield ("meta", parser.plan)
    for cand in parser.finish():
        yield ("candidate", cand)
    yield ("done", parser.plan)


def _worth_caching(raw: str, *, refine: bool = False) -> bool:
    """A sanity floor on what may be stored: real URLs plus the metadata line.

    Completeness is decided by the producer (below), not here. This only
    refuses a plan that is malformed even when complete: a planner that emitted
    two lines and stopped has not produced a plan. A second pass is
    asked for fewer URLs and no X line, so its floor is lower.
    """
    if not raw or not raw.strip():
        return False
    lines = raw.splitlines()
    urls = sum(1 for ln in lines if ln[:2].upper() == "U ")
    if refine:
        return urls >= 3
    meta = any(ln[:2].upper() in ("X ", "H ") for ln in lines)
    return urls >= 5 and meta


# How long `aclose` waits for plans still finishing in the background. A local
# CLI planner can need most of this; nothing in a request waits on it.
FINISH_WAIT = 20.0
# The least a detached producer gets to finish a plan for the store, whatever
# the search that started it had left. A planner that knows it needs more says
# so with `finish_floor`. The request never waits on this.
FINISH_FLOOR = 15.0


class CachedPlanner:
    """Wraps any planner with a persistent plan cache.

    The inner planner is read on a task of its own, with a deadline of its
    own, and the caller is handed events off a queue for as long as it wants
    them. The two lifetimes are separate on purpose: the engine stops reading
    the moment it has enough URLs or its budget runs out, and a plan cut off
    there must not be what every later search replays. So the producer keeps
    reading after the caller has gone, for as long as the planner itself
    needs (`finish_floor` at least), and stores a plan only once the planner
    said it was done before its own deadline. The first search of a query is
    served what arrived in time; the second is served the whole plan.
    """
    name = "cached"

    def __init__(
        self,
        inner,
        *,
        store: Optional[PlanStore | NoStore] = None,
        model: Optional[str] = None,
        ttl: float = 30 * 86400,
        replay_ttft: float = 0.0,
        replay_tok_per_s: float = 0.0,
        read_only: bool = False,
        commons=None,
        finish_floor: Optional[float] = None,
    ) -> None:
        self.inner = inner
        self.finish_floor = (finish_floor if finish_floor is not None
                             else float(getattr(inner, "finish_floor", FINISH_FLOOR)))
        # The shared tier behind the local one: asked on a local miss, offered
        # every complete first-pass plan the inner planner writes. The engine
        # sets it, since it owns the client; None means local only.
        self.commons = commons
        self.store = store or PlanStore()
        self.model = model or getattr(inner, "model", "") or getattr(inner, "name", "?")
        self.ttl = ttl
        self.replay_ttft = replay_ttft
        self.replay_tok_per_s = replay_tok_per_s
        self.read_only = read_only
        n_urls = getattr(inner, "n_urls", 0)
        effort = getattr(inner, "effort", "") or ""
        self.variant = f"u{n_urls}e{effort}"
        # A wrapper must not hide what it wraps: the engine sizes the deadline
        # from this.
        self.min_budget = getattr(inner, "min_budget", 0.0)
        # Plans still being read after their caller left. `aclose` drains them.
        self._pending: set[asyncio.Task] = set()

    async def stream(
        self, query: str, deadline: Deadline
    ) -> AsyncIterator[tuple[str, object]]:
        async for ev in self._serve(query, deadline, self.variant,
                                    lambda dl: self.inner.stream(query, dl)):
            yield ev

    async def refine(
        self, query: str, findings: Findings, deadline: Deadline
    ) -> AsyncIterator[tuple[str, object]]:
        """The second pass, cached like the first.

        Keyed on the query and the page, not on the report: the report is what
        the first round happened to fetch and varies run to run, while the plan
        written over it is a set of pages for this query, as stable as the first
        plan. A page turn that reads a cached second pass is free.
        """
        inner = getattr(self.inner, "refine", None)
        if inner is None:
            yield ("error", "planner has no second pass")
            yield ("done", PlanParser(query).plan)
            return
        variant = f"{self.variant}|refine{findings.page}"
        async for ev in self._serve(query, deadline, variant,
                                    lambda dl: inner(query, findings, dl)):
            yield ev

    async def _serve(self, query: str, deadline: Deadline, variant: str,
                     open_stream) -> AsyncIterator[tuple[str, object]]:
        hit = self.store.get(query, self.model, self.ttl, variant)
        if hit is not None:
            yield ("source", "cache")
            async for ev in _emit(hit[0], query, self.replay_ttft,
                                  self.replay_tok_per_s):
                yield ev
            return
        commons = self.commons
        if commons is not None and commons.enabled and "|refine" not in variant:
            raw = await commons.lookup(commons_keys(query))
            if raw and _worth_caching(raw):
                # Kept locally under this planner's key, so the next search of this
                # question is a local hit whatever the commons does.
                if not self.read_only:
                    try:
                        self.store.put(query, self.model, raw, 0.0, variant)
                    except Exception:
                        pass
                yield ("source", "commons")
                async for ev in _emit(raw, query, 0.0, 0.0):
                    yield ev
                return
        q: asyncio.Queue = asyncio.Queue()
        # The planner gets the time a whole plan takes, whatever the search has
        # left. If the caller leaves first, the plan still finishes and lands
        # in the store for the next search of this query.
        own = Deadline(max(deadline.remaining, self.finish_floor, self.min_budget))
        task = asyncio.create_task(
            self._produce(query, own, q, variant, open_stream(own)))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        while True:
            remaining = deadline.remaining
            if remaining <= 0:
                return
            try:
                ev = await asyncio.wait_for(q.get(), timeout=remaining)
            except asyncio.TimeoutError:
                return
            if ev is None:
                return
            yield ev

    async def _produce(self, query: str, deadline: Deadline,
                       q: asyncio.Queue, variant: str, events) -> None:
        raw = ""
        complete = False
        t0 = time.perf_counter()
        try:
            async for kind, payload in events:
                if kind in ("meta", "done"):
                    raw = getattr(payload, "raw", "") or raw
                if kind == "error":
                    # A refused request is not a plan, whatever else arrived.
                    raw = ""
                    complete = False
                    q.put_nowait((kind, payload))
                    break
                if kind == "done":
                    # A planner cut off by the deadline still says "done" on
                    # the way out; what tells the two apart is the clock.
                    complete = not deadline.expired()
                q.put_nowait((kind, payload))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # the caller sees the error, not a hang
            q.put_nowait(("error", f"{type(e).__name__}: {e}"))
        finally:
            q.put_nowait(None)
        if complete and _worth_caching(raw, refine="|refine" in variant):
            if not self.read_only:
                try:
                    self.store.put(query, self.model, raw,
                                   (time.perf_counter() - t0) * 1000, variant)
                except Exception:
                    pass
            if self.commons is not None and "|refine" not in variant:
                try:
                    self.commons.offer_plan(query, raw, model=self.model,
                                            planner=getattr(self.inner, "name", ""))
                except Exception:
                    pass

    async def aclose(self) -> None:
        """Let plans that outlived their callers finish, then close the inner."""
        pend = [t for t in self._pending if not t.done()]
        if pend:
            await asyncio.wait(pend, timeout=FINISH_WAIT)
            for t in pend:
                if not t.done():
                    t.cancel()
        closer = getattr(self.inner, "aclose", None)
        if callable(closer):
            await closer()


# The variant of the default ClaudeCLIPlanner (n_urls=14, effort="low"). A
# replay reading under any other variant misses every row.
DEFAULT_VARIANT = "u14elow"


class ReplayPlanner:
    """Cache-only planner: a recorded plan, replayed with its recorded timing.

    A miss yields an empty plan and an error event, so a variant mismatch
    reads as an error rather than as the routes-only floor.
    """
    name = "replay"

    def __init__(
        self,
        *,
        store: Optional[PlanStore] = None,
        model: str = "opus",
        ttft: float = 0.35,
        tok_per_s: float = 130.0,
        variant: str = DEFAULT_VARIANT,
    ) -> None:
        self.store = store or PlanStore()
        self.model = model
        self.ttft = ttft
        self.tok_per_s = tok_per_s
        # Must match the variant a CachedPlanner wrote under, or every read misses.
        self.variant = variant
        self.hits = 0
        self.misses = 0

    async def stream(
        self, query: str, deadline: Deadline
    ) -> AsyncIterator[tuple[str, object]]:
        hit = self.store.get(query, self.model, 0.0, self.variant)
        if hit is None:
            self.misses += 1
            yield ("error", f"no cached plan for model={self.model!r} "
                            f"variant={self.variant!r}")
            yield ("done", PlanParser(query).plan)
            return
        self.hits += 1
        async for ev in _emit(hit[0], query, self.ttft, self.tok_per_s):
            yield ev

    async def refine(
        self, query: str, findings: Findings, deadline: Deadline
    ) -> AsyncIterator[tuple[str, object]]:
        """A recorded second pass, under the key CachedPlanner wrote it."""
        variant = f"{self.variant}|refine{findings.page}"
        hit = self.store.get(query, self.model, 0.0, variant)
        if hit is None:
            self.misses += 1
            yield ("error", f"no cached second pass for model={self.model!r} "
                            f"variant={variant!r}")
            yield ("done", PlanParser(query).plan)
            return
        self.hits += 1
        async for ev in _emit(hit[0], query, self.ttft, self.tok_per_s):
            yield ev


ATLAS_PATH = os.path.expanduser("~/.cache/liberdex/atlas.sqlite3")
# Sites redesign and endpoints go away. Long enough that a host learned once
# stays learned across a working session, short enough that a stale endpoint
# expires on its own.
ATLAS_TTL = 30 * 24 * 3600.0


class SiteSearchAtlas(_Sqlite):
    """Learned site-search endpoints, keyed by host.

    The nearest thing to an index that exists for an arbitrary site is the
    search box it runs over its own content, and the address of that box is
    stated in the page's search form. Reading it costs nothing once a page
    is fetched; fetching a page from the host first is a round trip a query
    cannot afford before it has any results. So the first query that lands
    on a host writes the template down, and every later query on that host
    fires its search at t=0.

    Confirmations are counted rather than overwritten: some pages carry a
    section-scoped search box rather than the site-wide one, and the
    endpoint seen from the most distinct pages is the site-wide one.
    """

    def __init__(self, path: str = ATLAS_PATH) -> None:
        super().__init__(path, (
            "CREATE TABLE IF NOT EXISTS sites ("
            " host TEXT NOT NULL, template TEXT NOT NULL,"
            " seen INTEGER NOT NULL DEFAULT 1, updated REAL NOT NULL,"
            " PRIMARY KEY (host, template))"))

    def get(self, host: str, now: float = 0.0) -> str:
        now = now or time.time()
        try:
            row = self._conn().execute(
                "SELECT template FROM sites WHERE host = ? AND updated > ?"
                " ORDER BY seen DESC, updated DESC LIMIT 1",
                (host, now - ATLAS_TTL),
            ).fetchone()
        except sqlite3.Error:
            return ""
        return row[0] if row else ""

    def put(self, host: str, template: str) -> None:
        if not host or not template or "{}" not in template:
            return
        try:
            self._conn().execute(
                "INSERT INTO sites (host, template, seen, updated)"
                " VALUES (?, ?, 1, ?)"
                " ON CONFLICT(host, template) DO UPDATE SET"
                "  seen = seen + 1, updated = excluded.updated",
                (host, template[:600], time.time()),
            )
        except sqlite3.Error:
            pass

    def put_many(self, pairs: Iterable[tuple[str, str]]) -> None:
        """One transaction for a whole query's worth of endpoints."""
        rows = [(h, t[:600], time.time()) for h, t in pairs
                if h and t and "{}" in t]
        if not rows:
            return
        try:
            conn = self._conn()
            with conn:
                conn.executemany(
                    "INSERT INTO sites (host, template, seen, updated)"
                    " VALUES (?, ?, 1, ?)"
                    " ON CONFLICT(host, template) DO UPDATE SET"
                    "  seen = seen + 1, updated = excluded.updated", rows)
        except sqlite3.Error:
            pass

    def count(self) -> int:
        try:
            return int(self._conn().execute(
                "SELECT COUNT(DISTINCT host) FROM sites").fetchone()[0])
        except sqlite3.Error:
            return 0


SITEMAP_PATH = os.path.expanduser("~/.cache/liberdex/sitemaps.sqlite3")
# A publisher's catalogue changes on the timescale of its publishing schedule,
# not its traffic.
SITEMAP_TTL = 3 * 24 * 3600.0
# A host that answered with nothing usable is remembered for a shorter time: a
# 403 from a bot wall is a property of the hour, not of the host.
SITEMAP_MISS_TTL = 6 * 3600.0


class SitemapStore(_Sqlite):
    """One host's published URL list, gzipped, with a TTL.

    The fetch is the whole cost: scoring twenty thousand slugs is a
    millisecond. A miss is stored too, as an empty list, so a host that
    blocks us is asked once per `SITEMAP_MISS_TTL` rather than once per
    query.
    """

    def __init__(self, path: str = SITEMAP_PATH) -> None:
        super().__init__(path, (
            "CREATE TABLE IF NOT EXISTS sitemaps ("
            " host TEXT PRIMARY KEY, locs BLOB NOT NULL,"
            " n INTEGER NOT NULL, updated REAL NOT NULL)"))

    def get(self, host: str, now: float = 0.0) -> Optional[list[str]]:
        """The host's URLs, [] if it has none to give, None if never asked."""
        now = now or time.time()
        try:
            row = self._conn().execute(
                "SELECT locs, n, updated FROM sitemaps WHERE host = ?", (host,)
            ).fetchone()
        except sqlite3.Error:
            return None
        if not row:
            return None
        blob, n, updated = row
        ttl = SITEMAP_TTL if n else SITEMAP_MISS_TTL
        if updated <= now - ttl:
            return None
        if not n:
            return []
        try:
            return zlib.decompress(blob).decode("utf-8").split("\n")
        except Exception:
            return None

    def put(self, host: str, locs: list[str]) -> None:
        if not host:
            return
        blob = zlib.compress("\n".join(locs).encode("utf-8"), 6) if locs else b""
        try:
            self._conn().execute(
                "INSERT INTO sitemaps (host, locs, n, updated) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(host) DO UPDATE SET"
                " locs = excluded.locs, n = excluded.n, updated = excluded.updated",
                (host, blob, len(locs), time.time()),
            )
        except sqlite3.Error:
            pass

    def count(self) -> int:
        try:
            row = self._conn().execute(
                "SELECT COUNT(*) FROM sitemaps WHERE n > 0").fetchone()
        except sqlite3.Error:
            return 0
        return int(row[0]) if row else 0
