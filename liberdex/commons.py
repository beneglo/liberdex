"""The shared plan cache: a hosted store of plans keyed on the question,
asked before the planner runs.

What leaves the machine: a lookup sends the question's key (`commons_keys`),
its content words casefolded, accents and function words gone, sorted; an
offer sends a complete first-pass plan the local planner wrote, under the same
key; a report says, for a plan's URLs, whether each one fetched and whether it
made the SERP. The key is issued on first use and kept in
`~/.config/liberdex/env`.
"""
from __future__ import annotations

import asyncio
import os
import re
import time
import unicodedata
from typing import Any, Iterable, Optional

import httpx
import orjson

from .query import NOISE, content_terms, guess_lang

DEFAULT_URL = "https://commons.liberdex.net"


def _user_agent() -> str:
    """`liberdex-commons/1 liberdex/<version>`: the client protocol, then the
    package."""
    try:
        from importlib.metadata import version
        v = version("liberdex")
    except Exception:
        v = "0"
    return f"liberdex-commons/1 liberdex/{v}"


USER_AGENT = _user_agent()
# Bumped when `bag` or `commons_keys` change what they compute: every client
# must turn the same question into the same key.
KEY_VERSION = "commons1"
MAX_KEY_CHARS = 400
# The lookup sits between the request and the planner. Routes fire at t=0
# regardless, so this is not dead time, but it is on the critical path.
LOOKUP_TIMEOUT = 0.3
# Offers and reports leave in batches, off the request path.
FLUSH_EVERY = 2.0
FLUSH_BATCH = 20
QUEUE_MAX = 200
# How long `aclose` waits for the last batch to go out.
CLOSE_WAIT = 3.0
# After a failure the commons is left alone for MUTE seconds, doubling with
# each failure in a row up to MUTE_MAX, so a commons that is not running
# costs one probe an hour per process rather than one a minute.
MUTE = 60.0
MUTE_MAX = 3600.0

# Words that name the kind of thing wanted, not the thing: "Stadt Bochum" and
# "Bochum" want the same plan. Dropping only these, never a word that might be
# the subject, is what makes the relaxed key safe to serve.
GENERIC = frozenset({
    "stadt", "gemeinde", "city", "town", "ville", "ciudad", "cidade", "città",
    "info", "infos", "information", "informationen", "informacion", "información",
    "website", "webseite", "webseiten", "site", "seite", "page", "homepage",
    "official", "offiziell", "offizielle", "officiel", "officielle", "oficial",
    "meaning", "bedeutung", "definition", "erklärung", "erklarung", "explained",
    "overview", "übersicht", "ubersicht", "summary", "zusammenfassung",
}) | NOISE


def normalise(text: str) -> str:
    """Casefold, strip accents, keep letters and digits of every script."""
    s = unicodedata.normalize("NFKD", text or "").casefold()
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w]+", " ", s).replace("_", " ")
    return " ".join(s.split())


def _terms(query: str, lang: str = "") -> list[tuple[str, bool]]:
    """(normalised word, is it generic) for each content word, sorted.

    No stemming. A stemmer needs the language, the language of a three-noun
    query is a guess, and the English rules applied to "aktuelle" and
    "aktueller" give two different stems, which is two keys for one question
    and worse than one key each for "city" and "cities".
    """
    query = " ".join((query or "").split())
    if not query:
        return []
    lang = lang or guess_lang(query)
    out: dict[str, bool] = {}
    for tok in content_terms(query, lang):
        if tok.startswith('"'):
            inner = normalise(tok.strip('"'))
            if inner:
                out[f'"{inner}"'] = False
            continue
        low = tok.casefold()
        for piece in normalise(low).split():
            if len(piece) < 2 and not piece.isdigit():
                continue
            generic = piece in GENERIC or low in GENERIC
            out[piece] = out.get(piece, True) and generic
    if not out:
        n = normalise(query)
        return [(n, False)] if n else []
    return sorted(out.items())


def bag(query: str, lang: str = "") -> list[str]:
    """The question as a sorted set of normalised content words.

    Word order, case, accents, punctuation and function words do not change
    what is being asked; this is the part that does. A quoted phrase stays
    one word, so `"exact error text"` is not the same question as its words.
    """
    return [w for w, _ in _terms(query, lang)]


def _key(words: Iterable[str]) -> str:
    from .cache import PROTOCOL_VERSION
    return f"{KEY_VERSION}.{PROTOCOL_VERSION}|" + " ".join(words)


def commons_keys(query: str, lang: str = "") -> list[str]:
    """Keys to look up, best first: the question, then the question without
    its generic words. A plan is only ever *stored* under the first."""
    terms = _terms(query, lang)
    if not terms:
        return []
    words = [w for w, _ in terms]
    if len(" ".join(words)) > MAX_KEY_CHARS:
        return []          # a question this long is nobody else's question
    keys = [_key(words)]
    relaxed = [w for w, generic in terms if not generic]
    if relaxed and len(relaxed) < len(words):
        keys.append(_key(relaxed))
    return keys


def render_plan(plan: Any) -> str:
    """A `Plan` object as the line protocol: what a caller-written plan looks
    like once it is on the wire. `Plan.raw` where the planner wrote it."""
    raw = getattr(plan, "raw", "") or ""
    if raw.strip():
        return raw
    lines: list[str] = []
    if getattr(plan, "intent", ""):
        lines.append(f"I {plan.intent}")
    if getattr(plan, "lang", ""):
        lines.append(f"L {plan.lang}")
    rq = getattr(plan, "route_queries", {}) or {}
    routes = [f"{r}: {rq[r]}" if rq.get(r) else r for r in (getattr(plan, "routes", []) or [])]
    if routes:
        lines.append("R " + " | ".join(routes))
    cands = list(getattr(plan, "candidates", []) or [])
    if cands:
        c = cands[0]
        lines.append(f"U {c.prior:.2f} {c.url}")
    exps = [x for x in (getattr(plan, "expansions", []) or []) if x]
    if exps:
        lines.append("X " + " | ".join(exps))
    for h in getattr(plan, "hubs", []) or []:
        lines.append(f"H {h}")
    for c in cands[1:]:
        lines.append(f"U {c.prior:.2f} {c.url}")
    return "\n".join(lines) + ("\n" if lines else "")


class Commons:
    """One commons, one key, a small outbound queue. Safe to share across
    searches. Never raises: the commons being down is a search that plans
    for itself, not a failed search."""

    def __init__(self, url: Optional[str] = None, key: Optional[str] = None,
                 *, timeout: float = LOOKUP_TIMEOUT, register: bool = True,
                 transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        self.url = (url or os.environ.get("LIBERDEX_COMMONS_URL") or DEFAULT_URL).rstrip("/")
        self.key = key if key is not None else (os.environ.get("LIBERDEX_COMMONS_KEY") or "")
        self.timeout = timeout
        self.auto_register = register
        self._transport = transport
        self._http: Optional[httpx.AsyncClient] = None
        self._muted_until = 0.0
        self._strikes = 0
        self._registering: Optional[asyncio.Task] = None
        self._queue: list[tuple[str, dict]] = []
        self._flusher: Optional[asyncio.Task] = None
        self.error = ""
        self.stats = {"hits": 0, "misses": 0, "offered": 0, "reported": 0, "dropped": 0}

    # ------------------------------------------------------------- state
    @property
    def enabled(self) -> bool:
        """False while muted after a failure; the commons being down is a
        search that plans for itself."""
        return time.monotonic() >= self._muted_until

    def state(self) -> dict[str, Any]:
        return {"url": self.url, "registered": bool(self.key),
                "muted": time.monotonic() < self._muted_until, **self.stats}

    def _mute(self, seconds: float, why: str) -> None:
        """Stop asking for a while. Failures in a row double the wait, up to
        MUTE_MAX; an answer of any kind (`_heard`) resets the run."""
        self.error = why
        self._strikes += 1
        self._muted_until = time.monotonic() + min(MUTE_MAX, seconds * 2 ** (self._strikes - 1))

    def _heard(self) -> None:
        """The commons answered as it should: forget the failures before it."""
        self._strikes = 0
        self.error = ""

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self.url, timeout=self.timeout, transport=self._transport,
                headers={"user-agent": USER_AGENT})
        return self._http

    def _headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self.key}", "content-type": "application/json"}

    async def _post(self, path: str, body: Any, timeout: Optional[float] = None
                    ) -> Optional[httpx.Response]:
        try:
            c = await self._client()
            return await c.post(path, content=orjson.dumps(body), headers=self._headers(),
                                timeout=timeout if timeout is not None else self.timeout)
        except Exception as e:
            self._mute(MUTE, f"commons unreachable: {type(e).__name__}")
            return None

    # ---------------------------------------------------------- register
    async def register(self) -> bool:
        """A key of our own, once. The server issues them to anyone."""
        if self.key:
            return True
        if not self.auto_register:
            return False
        try:
            c = await self._client()
            r = await c.post("/commons/register", content=b"{}",
                             headers={"content-type": "application/json"}, timeout=3.0)
        except Exception as e:
            self._mute(MUTE, f"commons unreachable: {type(e).__name__}")
            return False
        if r.status_code != 200:
            self._mute(MUTE_MAX, f"commons would not issue a key ({r.status_code})")
            return False
        self._heard()
        try:
            key = str(orjson.loads(r.content).get("key") or "")
        except Exception:
            key = ""
        if not key.startswith("cx_"):
            self._mute(MUTE_MAX, "commons issued no key")
            return False
        self.key = key
        try:
            from .envfile import write
            write({"LIBERDEX_COMMONS_KEY": key})
        except Exception:
            os.environ["LIBERDEX_COMMONS_KEY"] = key
        return True

    def kick(self) -> None:
        """Start registering now, off any request's critical path, so the first
        lookup finds a key rather than waiting for one. A no-op with a key."""
        if not self.enabled or self.key or not self.auto_register:
            return
        if self._registering is None or self._registering.done():
            self._registering = asyncio.create_task(self.register())

    async def _ready(self, wait: Optional[float] = None) -> bool:
        """True with a key in hand. Without one, registration is started (or
        joined) and waited on for `wait` seconds: the lookup's own timeout by
        default, since it sits between the request and the planner."""
        if not self.enabled:
            return False
        if self.key:
            return True
        self.kick()
        if self._registering is None:
            return False
        try:
            return await asyncio.wait_for(
                asyncio.shield(self._registering),
                timeout=self.timeout if wait is None else wait)
        except Exception:
            return False

    # ------------------------------------------------------------ lookup
    async def lookup(self, keys: list[str]) -> Optional[str]:
        """The served plan for the first key that has one, or None."""
        if not keys or not await self._ready():
            return None
        r = await self._post("/commons/lookup", {"keys": keys[:4]})
        if r is None:
            return None
        if r.status_code == 404:
            self._heard()
            self.stats["misses"] += 1
            return None
        if r.status_code in (401, 403):
            self._mute(MUTE_MAX, "commons refused the key")
            return None
        if r.status_code != 200:
            self._mute(MUTE, f"commons answered {r.status_code}")
            return None
        self._heard()
        try:
            raw = orjson.loads(r.content).get("raw") or ""
        except Exception:
            return None
        if not isinstance(raw, str) or not raw.strip():
            self.stats["misses"] += 1
            return None
        self.stats["hits"] += 1
        return raw

    # ----------------------------------------------------- offers, reports
    def _enqueue(self, kind: str, item: dict) -> None:
        if len(self._queue) >= QUEUE_MAX:
            self.stats["dropped"] += 1
            return
        self._queue.append((kind, item))
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._flusher is None or self._flusher.done():
            self._flusher = loop.create_task(self._flush_later())

    def offer_plan(self, query: str, raw: str, *, model: str = "", planner: str = "") -> bool:
        """Queue a complete first-pass plan for this question. Returns whether
        it was queued."""
        if not self.enabled or not raw or not raw.strip():
            return False
        keys = commons_keys(query)
        if not keys:
            return False
        self._enqueue("offer", {"key": keys[0], "raw": raw,
                                "model": model[:80], "planner": planner[:40]})
        self.stats["offered"] += 1
        return True

    def report(self, query: str, urls: list[dict]) -> bool:
        """Queue what became of a plan's URLs: `{url, ok, serp}` each."""
        if not self.enabled or not urls:
            return False
        keys = commons_keys(query)
        if not keys:
            return False
        self._enqueue("report", {"key": keys[0], "urls": urls[:40]})
        self.stats["reported"] += 1
        return True

    async def _flush_later(self) -> None:
        try:
            await asyncio.sleep(FLUSH_EVERY)
        except asyncio.CancelledError:
            raise
        await self.flush()

    async def flush(self) -> None:
        """Send what is queued. Called by the timer and by `aclose`."""
        if not self._queue or not await self._ready(wait=3.0):
            self._queue.clear()
            return
        while self._queue and self.enabled:
            batch, self._queue = self._queue[:FLUSH_BATCH], self._queue[FLUSH_BATCH:]
            for kind in ("offer", "report"):
                items = [it for k, it in batch if k == kind]
                if not items:
                    continue
                r = await self._post(f"/commons/{kind}", items, timeout=5.0)
                if r is None:
                    return
                if r.status_code in (401, 403):
                    self._mute(MUTE_MAX, "commons refused the key")
                    return
                if r.status_code >= 500:
                    self._mute(MUTE, f"commons answered {r.status_code}")
                    return
                self._heard()

    async def forget(self) -> bool:
        """Ask the commons to delete everything this key sent, then drop the
        key here. What `liberdex commons forget` runs."""
        if not self.key:
            return False
        try:
            c = await self._client()
            r = await c.delete("/commons/me", headers=self._headers(), timeout=10.0)
        except Exception as e:
            self._mute(MUTE, f"commons unreachable: {type(e).__name__}")
            return False
        if r.status_code not in (200, 403):
            return False
        self._heard()
        self.key = ""
        try:
            from .envfile import write
            write({"LIBERDEX_COMMONS_KEY": None})
        except Exception:
            os.environ.pop("LIBERDEX_COMMONS_KEY", None)
        return True

    async def aclose(self) -> None:
        if self._flusher is not None and not self._flusher.done():
            self._flusher.cancel()
        if self._queue:
            try:
                await asyncio.wait_for(self.flush(), timeout=CLOSE_WAIT)
            except Exception:
                pass
        if self._http is not None:
            await self._http.aclose()
            self._http = None
