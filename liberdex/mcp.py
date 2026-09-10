"""liberdex as an MCP server, for the agent CLIs.

    liberdex mcp            # stdio; what `liberdex install <host>` wires up

Two tools, `search` and `extract`, in liberdex's own words. The host on the
other side is already a model with the web in its memory, so the tool asks it
to write the plan: the URLs it recalls, the per-site queries, the hub pages.
liberdex then fetches, reads and ranks: no second LLM call, no key, and the
first fetch leaves at t=0. A host that would rather not plan omits `plan` and
the configured planner is used, if there is one.

The same server is mounted at `/mcp` by `liberdex serve`, so a hosted
liberdex is a connector for claude.ai, Claude Desktop and ChatGPT.
"""
from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal, Optional

from mcp.server.mcpserver import MCPServer
from mcp_types import CallToolResult, TextContent

from . import __version__
from .api import ExtractRequest, SearchReply, SearchRequest, reply
from .api import extract as api_extract

Depth = Literal["fast", "standard", "deep"]
Freshness = Literal["day", "week", "month", "year"]
Intent = Literal["navigational", "informational", "code", "academic",
                 "news", "product", "local", "reference"]

INSTRUCTIONS = (
    "liberdex is an index-free web search engine. There is no index: an LLM "
    "names where the answer lives and liberdex fetches, reads and ranks those "
    "pages. You are that LLM. Before calling `search`, write the plan yourself "
    "and pass it as `plan`; the search then costs no LLM call and starts "
    "fetching immediately."
)

SEARCH_DESC = """\
Web search with full-page passages, not snippets. Write the plan yourself and \
pass it as `plan`: {"candidates": [{"url": ..., "prior": 0-1, "reason": ...}], \
"routes": {"<site>": "<query for that site's own search>"}, "hubs": [...], \
"expansions": [...], "intent": ..., "lang": ...}.
- candidates: 8-14 exact page URLs you recall from memory that most likely hold \
the answer. Prefer primary sources and the specific page over a homepage; never \
invent an opaque identifier (a numeric id, a hash, a dated news slug); give \
the host's section page instead. `prior` is your own confidence.
- routes: sites with a search box worth firing with a short query written for \
that site: wikipedia, hn, stackoverflow, github, arxiv, pubmed, mdn, npm, pypi, \
crates, openlibrary, reddit, youtube. Name a documentation site as a hub instead.
- hubs: 2-3 index or category pages whose links lead to the answer.
- expansions: 2-3 other phrasings of the query, for lexical scoring.
Omit `plan` only if you cannot write one; liberdex then plans with its own \
configured model, if it has one.
Results come back numbered, each with its URL, `relevance` in 0..1, where it \
came from, the size of the whole page in chars, and the passages about the \
query. `include_text` ships the pages themselves within `token_budget`; a page \
that was cut says so and names the `extract` call that reads the rest.\
"""

EXTRACT_DESC = (
    "Fetch and read pages you already have URLs for, as clean text or markdown. "
    "Each page comes back with its title, URL, size in chars, and its text within "
    "`token_budget_per_page` (default 4000); a page cut to fit ends with a note "
    "saying how much was shown and the budget that reads it whole. With `query`, "
    "each page comes back as the passages about that query instead of whole, "
    "the cheaper way to read a long page. Up to 20 URLs; PDFs are read too."
)


class Service:
    """The engine behind the tools, built once per server process."""

    def __init__(self, engine: Any = None, *, planner_error: str = "",
                 shared: bool = False) -> None:
        self.engine = engine
        self.planner_error = planner_error
        # Mounted inside `liberdex serve`, the engine belongs to the HTTP
        # server: it is started and closed there, not by the MCP lifespan.
        self.shared = shared
        self._warm: Optional[asyncio.Task] = None

    @classmethod
    def from_env(cls, backend: str = "auto", spec: Optional[str] = None) -> "Service":
        from .engine import Liberdex
        from .planner.factory import make_planner
        try:
            planner = make_planner(backend, spec)
            err = ""
        except LookupError as e:
            planner, err = None, str(e)
        return cls(Liberdex(planner=planner), planner_error=err)

    async def start(self) -> None:
        from .rank import Models
        if self.shared:
            return
        await self.engine.start()
        # Loading the ranking models takes a second or two and `initialize`
        # must not wait for it; the first search does.
        self._warm = asyncio.create_task(asyncio.to_thread(Models.preload))

    async def ready(self) -> None:
        if self._warm is not None:
            await self._warm

    async def aclose(self) -> None:
        if self.shared:
            return
        if self._warm is not None and not self._warm.done():
            self._warm.cancel()
        await self.engine.aclose()


def cut_note(url: str, shown: int, whole: int) -> str:
    """What a model needs when a page did not fit: how much it saw, and the
    exact call that reads the rest. Empty when nothing was cut.

    The budget named is the whole page at three chars a token, rounded up to
    the thousand: an over-estimate on purpose, since the budget is a cap and a
    second cut page is the one outcome this line exists to prevent.
    """
    if not whole or shown >= whole:
        return ""
    tokens = -(-whole // 3)
    tokens = -(-tokens // 1000) * 1000
    return (f"[cut: {shown:,} of {whole:,} chars shown. "
            f'extract(urls=["{url}"], token_budget_per_page={tokens}) reads it '
            f"whole; add `query` for the passages about it instead]")


def render(out: SearchReply) -> str:
    """The SERP as text a model reads well: numbered, sourced, with passages."""
    lines = [f"# {out.query}  ({out.intent}, {len(out.results)} pages)"]
    if out.stats.get("planner_error"):
        lines.append(f"planner: {out.stats['planner_error']}")
    for i, r in enumerate(out.results, 1):
        lines.append(f"\n## {i}. {r.title or r.url}")
        meta = f"{r.url}  relevance={r.relevance:.2f}  via {r.source}"
        if r.published:
            meta += f"  published {r.published}"
        if r.text_chars:
            meta += f"  page {r.text_chars:,} chars"
        lines.append(meta)
        for p in r.passages:
            lines.append(f"> {p}")
        if r.text:
            lines.append("")
            lines.append(r.text)
            note = cut_note(r.url, len(r.text), r.text_chars)
            if note:
                lines.append(note)
    if out.stats.get("dropped_over_budget"):
        lines.append(f"\n{out.stats['dropped_over_budget']} more pages ranked "
                     f"but dropped for `token_budget`")
    if out.answer:
        lines.append(f"\nanswer: {out.answer}")
    t = out.timings
    lines.append(f"\n{t.get('total_ms', 0):.0f} ms, "
                 f"{out.stats.get('fetched_ok', 0)}/{out.stats.get('dispatched', 0)} "
                 f"pages read")
    return "\n".join(lines)


def build_server(service: Service) -> MCPServer:
    @asynccontextmanager
    async def lifespan(server: MCPServer) -> AsyncIterator[dict[str, Any]]:
        await service.start()
        try:
            yield {"service": service}
        finally:
            await service.aclose()

    server = MCPServer("liberdex", version=__version__, instructions=INSTRUCTIONS,
                       lifespan=lifespan)

    @server.tool(name="search", description=SEARCH_DESC)
    async def search(
        query: str,
        plan: Optional[dict[str, Any]] = None,
        top_k: int = 8,
        depth: Depth = "standard",
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
        freshness: Optional[Freshness] = None,
        intent: Optional[Intent] = None,
        token_budget: int = 6000,
        include_text: bool = False,
        page: int = 1,
        exclude_urls: Optional[list[str]] = None,
    ) -> CallToolResult:
        if service.engine is None:
            return _error("liberdex is still starting; try again in a moment")
        try:
            req = SearchRequest(
                query=query, plan=plan, top_k=max(1, min(25, top_k)), depth=depth,
                include_domains=include_domains or [],
                exclude_domains=exclude_domains or [],
                freshness=freshness, intent=intent,
                token_budget=max(0, min(200_000, token_budget)),
                include_text=include_text, page=max(1, min(10, page)),
                exclude_urls=exclude_urls or [],
            )
        except ValueError as e:
            return _error(str(e))
        if plan is None and service.engine.planner is None:
            why = service.planner_error or "no planner configured"
            return _error(
                f"no plan given and liberdex has no planner of its own ({why}). "
                "Write the plan yourself and pass it as `plan` (see the tool "
                "description) or configure one: `liberdex install --model "
                "openrouter/google/gemini-3.5-flash-lite` with OPENROUTER_API_KEY set.")
        await service.ready()
        # The deadline is sized for the planner that will run (api.size_budget).
        kwargs = req.engine_kwargs(planner=service.engine.planner)
        resp = await service.engine.search(query, **kwargs)
        out = reply(resp, req)
        return CallToolResult(
            content=[TextContent(type="text", text=render(out))],
            structuredContent=out.model_dump(exclude_none=True),
        )

    @server.tool(name="extract", description=EXTRACT_DESC)
    async def extract(
        urls: list[str],
        query: Optional[str] = None,
        format: Literal["text", "markdown"] = "markdown",
        token_budget_per_page: int = 4000,
        include_links: bool = False,
    ) -> CallToolResult:
        try:
            req = ExtractRequest(
                urls=urls, query=query, format=format,
                token_budget_per_page=max(0, min(100_000, token_budget_per_page)),
                include_links=include_links,
            )
        except ValueError as e:
            return _error(str(e))
        if service.engine is None:
            return _error("liberdex is still starting; try again in a moment")
        await service.ready()
        out = await api_extract(service.engine, req)
        parts = []
        for p in out.results:
            head = f"# {p.title or p.url}\n{p.url}"
            if p.text_chars:
                head += f"  page {p.text_chars:,} chars"
            parts.append(head)
            if p.passages:
                parts.extend(f"> {x}" for x in p.passages)
            elif p.text:
                parts.append(p.text)
                note = cut_note(p.url, len(p.text), p.text_chars)
                if note:
                    parts.append(note)
            if p.links:
                parts.append("links:")
                parts.extend(f"- {ln['anchor']}: {ln['url']}" for ln in p.links[:60])
        for f in out.failed:
            parts.append(f"# failed: {f.url}\n{f.error}")
        return CallToolResult(
            content=[TextContent(type="text", text="\n\n".join(parts) or "nothing read")],
            structuredContent=out.model_dump(exclude_none=True),
        )

    return server


def _error(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=True)


async def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="liberdex mcp",
                                 description="liberdex as an MCP server on stdio")
    ap.add_argument("-p", "--planner", default=os.environ.get("LIBERDEX_PLANNER", "auto"),
                    choices=["auto", "claude", "openai", "replay", "none"])
    ap.add_argument("--planner-model", default=os.environ.get("LIBERDEX_PLANNER_MODEL"),
                    help="profile/model for searches that arrive without a plan")
    args = ap.parse_args(argv)
    service = Service.from_env(args.planner, args.planner_model)
    if service.planner_error:
        print(f"liberdex mcp: no planner of its own ({service.planner_error}); "
              f"searches need a `plan` from the host", file=sys.stderr)
    server = build_server(service)
    await server.run_stdio_async()
    return 0
