"""The provider hermes-agent calls.

hermes hands a provider `search(query, limit)` and `extract(urls)`, nothing
else: no plan, no intent. So the plan is written here, by hermes's own model
through `ctx.llm`, and handed to the engine as this call's planner. The engine
runs on a loop thread of its own for the life of the hermes process; hermes
calls `search` from a worker thread and awaits `extract`, and both are bridged
onto that loop.
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import threading
from typing import Any, Callable, Optional

try:  # hermes's base class, present only inside its process
    from agent.web_search_provider import WebSearchProvider  # type: ignore
except ImportError:  # pragma: no cover - exercised by the tests, never by hermes
    class WebSearchProvider:  # type: ignore[no-redef]
        pass

from ...api import ExtractRequest, SearchRequest, reply
from ...api import extract as api_extract

# How deep hermes's searches go. `fast` is a single fact, `deep` a research
# question; hermes has no knob of its own for it, so it is an environment choice.
DEPTH_ENV = "LIBERDEX_HERMES_DEPTH"


class _Loop:
    """One asyncio loop on a daemon thread; the engine lives on it."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever,
                                       name="liberdex-hermes", daemon=True)
        self.thread.start()

    def run(self, coro, timeout: Optional[float] = None):
        """From a plain thread: run it on the loop and wait for the result."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    async def call(self, coro):
        """From another loop: run it on this one and await the result."""
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return await asyncio.wrap_future(fut)

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        self.loop.close()


class LiberdexProvider(WebSearchProvider):
    """liberdex behind hermes's `web_search` and `web_extract`."""

    def __init__(self, ctx: Any = None, *,
                 engine_factory: Optional[Callable[[], Any]] = None,
                 depth: Optional[str] = None) -> None:
        self.ctx = ctx
        self.engine_factory = engine_factory
        self.depth = depth or os.environ.get(DEPTH_ENV, "standard")
        self._loop: Optional[_Loop] = None
        self._engine: Any = None
        self._planner: Any = None
        self._fallback: Any = None
        self._fallback_why = ""
        self._lock = threading.Lock()

    # ---------------------------------------------------------- hermes's ABC
    @property
    def name(self) -> str:
        return "liberdex"

    @property
    def display_name(self) -> str:
        return "liberdex (index-free, full-page passages, no key)"

    def is_available(self) -> bool:
        # Registration-time and on every `hermes tools` paint: no engine, no I/O.
        return importlib.util.find_spec("liberdex.engine") is not None

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def is_keyless_available(self) -> bool:
        # The plan comes from hermes's own model; there is no key to have.
        return True

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "free",
            "tag": "No key. Your own model writes the plan; liberdex fetches, "
                   "reads and ranks the pages on this machine.",
            "env_vars": [],
        }

    # ------------------------------------------------------------- plumbing
    def _loop_(self) -> _Loop:
        with self._lock:
            if self._loop is None:
                self._loop = _Loop()
            return self._loop

    async def _engine_(self) -> Any:
        """Built once, on the loop thread, the first time it is needed."""
        if self._engine is None:
            if self.engine_factory is not None:
                eng = self.engine_factory()
            else:
                from ...engine import Liberdex
                from ...rank import Models
                eng = Liberdex(planner=None)
                await asyncio.to_thread(Models.preload)
            await eng.start()
            self._engine = eng
        return self._engine

    def planner(self) -> Any:
        if self._planner is None:
            from ...planner.reply import ReplyPlanner
            self._planner = ReplyPlanner(self._complete, name="hermes")
        return self._planner

    async def _complete(self, messages: list[dict[str, str]], budget: float) -> str:
        """hermes's model, borrowed. Its `complete` is synchronous and owns its
        own HTTP, so it runs on a worker thread rather than on our loop."""
        try:
            llm = self.ctx.llm
            result = await asyncio.to_thread(
                llm.complete, messages=messages, max_tokens=1500, temperature=0,
                timeout=budget, purpose="liberdex plan", task="web_search")
            text = getattr(result, "text", None)
            if not text:
                raise RuntimeError("empty reply")
            return text
        except Exception as e:
            host_err = f"hermes model: {type(e).__name__}: {e}"
        # liberdex's own planner, if this machine has one configured.
        chat = self._own_planner()
        if chat is None:
            raise RuntimeError(f"{host_err}; liberdex: {self._fallback_why}")
        rep = await chat.complete(messages, budget=budget)
        if rep.error:
            raise RuntimeError(f"{host_err}; liberdex: {rep.error}")
        return rep.text

    def _own_planner(self) -> Any:
        if self._fallback is None and not self._fallback_why:
            from ...llm import Chat
            from ...models import auto_planner
            profile, model, why = auto_planner()
            if profile is None:
                self._fallback_why = why
            else:
                self._fallback = Chat(profile, model, max_retries=1)
        return self._fallback

    async def _search(self, query: str, limit: int) -> dict[str, Any]:
        eng = await self._engine_()
        planner = self.planner()
        req = SearchRequest(query=query, top_k=max(1, min(25, limit)), depth=self.depth)
        # The deadline is sized for the planner that will run: a whole-reply
        # one grows it to its `min_budget` (api.size_budget, one rule everywhere).
        kw = req.engine_kwargs(planner=planner)
        resp = await eng.search(query, planner=planner, **kw)
        out = reply(resp, req)
        err = (out.stats or {}).get("planner_error")
        if err and not out.results:
            return {"success": False,
                    "error": f"liberdex could not plan: {err}. Configure a fallback "
                             "with `liberdex install --model openrouter/google/"
                             "gemini-3.5-flash-lite`."}
        web = []
        for i, r in enumerate(out.results, 1):
            web.append({
                "title": r.title or r.url,
                "url": r.url,
                "description": "\n".join(r.passages),
                "position": i,
                "relevance": r.relevance,
                "published": r.published,
                "source": r.source,
            })
        return {"success": True, "data": {"web": web}}

    async def _extract(self, urls: list[str], fmt: str, max_chars: int,
                       include_raw: bool) -> list[dict[str, Any]]:
        eng = await self._engine_()
        req = ExtractRequest(urls=urls, format=fmt, token_budget_per_page=0, timeout=15)
        out = await api_extract(eng, req)
        by_url: dict[str, dict[str, Any]] = {}
        for p in out.results:
            text = p.text or "\n".join(p.passages)
            row: dict[str, Any] = {
                "url": p.url,
                "title": p.title,
                "content": text[:max_chars] if max_chars else text,
                "metadata": {"text_chars": p.text_chars, "published": p.published,
                             "lang": p.lang, "site": p.site},
            }
            if include_raw:
                row["raw_content"] = text
            by_url[p.url] = row
        for f in out.failed:
            by_url.setdefault(f.url, {"url": f.url, "title": "", "content": "",
                                      "error": f.error or f"status {f.status}"})
        # Input order, one row per URL asked for.
        return [by_url.get(u, {"url": u, "title": "", "content": "", "error": "not read"})
                for u in urls]

    # --------------------------------------------------------- what hermes calls
    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        try:
            return self._loop_().run(self._search(query, limit), timeout=120)
        except Exception as e:
            return {"success": False, "error": f"liberdex: {type(e).__name__}: {e}"}

    async def extract(self, urls: list, **kwargs: Any) -> list[dict[str, Any]]:
        urls = [u if isinstance(u, str) else str(u.get("url", "")) for u in urls]
        fmt = "markdown" if kwargs.get("format") == "markdown" else "text"
        max_chars = int(kwargs.get("max_chars") or 0)
        include_raw = bool(kwargs.get("include_raw"))
        try:
            return await self._loop_().call(
                self._extract(urls, fmt, max_chars, include_raw))
        except Exception as e:
            return [{"url": u, "title": "", "content": "",
                     "error": f"liberdex: {type(e).__name__}: {e}"} for u in urls]

    def close(self) -> None:
        if self._loop is not None:
            if self._engine is not None and hasattr(self._engine, "aclose"):
                try:
                    self._loop.run(self._engine.aclose(), timeout=10)
                except Exception:
                    pass
            self._loop.close()
            self._loop = None
