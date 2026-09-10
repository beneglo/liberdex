"""liberdex command line.

    liberdex "how does the TCP three way handshake work"
    liberdex search --json --sites python.org "dict.get default"
    liberdex serve --open
    liberdex install claude-code
    liberdex models check
"""
from __future__ import annotations

import argparse
import asyncio
import sys

import orjson

from .api import DEPTH, freshness_floor, size_budget
from .types import SearchResponse

USAGE = """\
usage: liberdex <query>                search from the terminal (same as `liberdex search`)
       liberdex search [options] <query>
       liberdex serve [--port 8080] [--open]   the search page and the HTTP API
       liberdex mcp                            MCP server on stdio, for agent CLIs
       liberdex install <host>                 wire liberdex into claude-code | codex |
                                               opencode | cursor | gemini | hermes |
                                               openclaw | t3code
       liberdex models check [profile ...]     which models this machine can reach
       liberdex commons forget                 delete what this machine sent to the shared plan cache
       liberdex warmup                         download the ranking models now

`liberdex <command> --help` for the options of each.
"""


def build_planner(args):
    from .planner.factory import make_planner
    try:
        return make_planner(
            args.planner, args.planner_model or args.model,
            n_urls=args.n_urls, effort=args.effort or None,
            reasoning_effort=args.reasoning_effort or None,
            base_url=args.base_url, cache=not args.no_cache)
    except LookupError as e:
        print(f"\x1b[31m{e}\x1b[0m", file=sys.stderr)
        print("  (or pass --planner none for a routes-only search)", file=sys.stderr)
        raise SystemExit(2)


def render(resp: SearchResponse, *, as_json: bool, debug: bool,
           answer: str = "") -> str:
    if as_json:
        return orjson.dumps(
            {
                "query": resp.query,
                "intent": resp.intent,
                **({"answer": answer} if answer else {}),
                "results": [
                    {"title": r.title, "url": r.url, "site": r.site,
                     "score": r.score,
                     "passages": r.passages or ([r.snippet] if r.snippet else []),
                     "passage_scores": r.passage_scores,
                     "published": r.published, "source": r.source,
                     "status": r.status,
                     **({"text": r.text} if r.text else {}),
                     **({"favicon": r.favicon} if r.favicon else {}),
                     **({"images": r.images} if r.images else {}),
                     **({"debug": r.debug} if debug else {})}
                    for r in resp.results
                ],
                "timings": resp.timings,
                "stats": resp.stats,
            },
            option=orjson.OPT_INDENT_2,
        ).decode()
    lines = [f"\n\x1b[1m{resp.query}\x1b[0m  ({resp.intent})"]
    if answer:
        lines.append(f"\n\x1b[1;33m{answer}\x1b[0m")
    for i, r in enumerate(resp.results, 1):
        lines.append(f"\n{i:2d}. \x1b[1;34m{r.title}\x1b[0m")
        lines.append(f"    \x1b[32m{r.url}\x1b[0m")
        for p in (r.passages or ([r.snippet] if r.snippet else [])):
            lines.append(f"    {p}")
        meta = f"    \x1b[90m{r.site}  score={r.score}"
        if r.published:
            meta += f"  {r.published}"
        if r.source:
            meta += f"  via {r.source}"
        lines.append(meta + "\x1b[0m")
        if debug:
            lines.append(f"    \x1b[90m{r.debug}\x1b[0m")
    t = resp.timings
    lines.append(
        f"\n\x1b[90m{t.get('total_ms', 0):.0f}ms total  "
        f"(plan first url {t.get('plan_first_url_ms', 0):.0f}ms, "
        f"gather {t.get('gather_ms', 0):.0f}ms, rank {t.get('rank_ms', 0):.0f}ms)  "
        f"{resp.stats.get('fetched_ok')}/{resp.stats.get('dispatched')} pages  "
        f"routes={','.join(resp.stats.get('routes_fired') or [])}\x1b[0m"
    )
    if resp.stats.get("planner_error"):
        lines.append(f"\x1b[31mplanner: {resp.stats['planner_error']}\x1b[0m")
    return "\n".join(lines)


# ------------------------------------------------------------------ models check
async def models_check(names: list[str]) -> int:
    """Probe every configured profile and report reachability and latency.

    Many built-in profiles and a config file is a lot of ways to be
    misconfigured; this turns "which one is wrong" into one screen.
    """
    import time as _time

    from .llm import Chat
    from .models import auto_planner, profiles, resolve_role

    known = profiles()
    wanted = names or sorted(known)
    roles = {}
    profile, model, why = auto_planner()
    roles["planner"] = (f"{profile.name}/{model}  \x1b[90m({why})\x1b[0m"
                        if profile else f"\x1b[31m{why}\x1b[0m")
    try:
        p, m = resolve_role("answer")
        roles["answer"] = f"{p.name}/{m}"
    except Exception as e:
        roles["answer"] = f"unresolved ({e})"
    for role, spec in roles.items():
        print(f"\x1b[1m{role:<8}\x1b[0m {spec}")
    print()

    worst = 0
    for name in wanted:
        profile = known.get(name)
        if profile is None:
            print(f"  {name:<13} \x1b[31munknown profile\x1b[0m")
            worst = 1
            continue
        chat = Chat(profile, profile.default_model, max_tokens=5, timeout=10.0)
        t0 = _time.perf_counter()
        try:
            ok, detail = await chat.probe()
        finally:
            await chat.aclose()
        ms = (_time.perf_counter() - t0) * 1000
        mark = "\x1b[32mok\x1b[0m  " if ok else "\x1b[31mno\x1b[0m  "
        where = profile.base_url or (" ".join(profile.argv[:1]) if profile.argv
                                     else profile.kind)
        print(f"  {name:<13} {mark} {ms:6.0f}ms  {where}  \x1b[90m{detail}\x1b[0m")
    return worst


def warmup() -> int:
    import os

    from .rank import Models
    print("downloading the ranking models (about 130 MB the first time) ...")
    Models.preload()
    hf = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    print(f"ok  embeddings under {hf}, cross-encoder under "
          f"{os.path.expanduser('~/.cache/liberdex/flashrank')}")
    return 0


# ------------------------------------------------------------------------ search
def search_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="liberdex search")
    # `*` rather than `+`: `--answer <question>` parks the question in
    # `answer`, and search_main moves it back before checking for one.
    ap.add_argument("query", nargs="*")
    ap.add_argument("-k", "--top-k", type=int, default=10)
    ap.add_argument("-b", "--budget", type=float, default=0.0,
                    help="override the depth tier's time budget, seconds")
    ap.add_argument("-d", "--depth", default="standard",
                    choices=["fast", "standard", "deep"])
    ap.add_argument("-p", "--planner", default="auto",
                    choices=["auto", "claude", "openai", "replay", "none"],
                    help="auto: your configured model, else a key in the "
                         "environment, else the claude binary")
    ap.add_argument("--no-cache", action="store_true",
                    help="skip the plan cache and always call the planner")
    ap.add_argument("-m", "--model", default=None,
                    help="planner model; defaults to the backend's own default")
    ap.add_argument("--planner-model", default=None,
                    help="profile/model for the planner, e.g. ollama/qwen3:8b")
    ap.add_argument("--answer-model", default=None,
                    help="profile/model for --answer; defaults to the planner's")
    ap.add_argument("--reasoning-effort", default=None,
                    choices=["none", "minimal", "low", "medium", "high"],
                    help="openai backend: thinking budget. 'none' disables it")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--n-urls", type=int, default=14)
    ap.add_argument("--no-speculate", action="store_true")
    ap.add_argument("--no-expand", action="store_true")
    ap.add_argument("--plan-file", default=None,
                    help="a JSON plan written by someone else; the planner is not called")

    ap.add_argument("--sites", default="",
                    help="only fetch these domains, comma separated")
    ap.add_argument("--not-sites", default="",
                    help="never fetch these domains, comma separated")
    ap.add_argument("--after", default=None, help="published on or after, YYYY-MM-DD")
    ap.add_argument("--before", default=None, help="published on or before, YYYY-MM-DD")
    ap.add_argument("--freshness", default=None,
                    choices=["day", "week", "month", "year"])
    ap.add_argument("--require-date", action="store_true",
                    help="drop pages with no parseable publication date")
    ap.add_argument("--intent", default=None,
                    choices=["navigational", "informational", "code", "academic",
                             "news", "product", "local", "reference"])
    ap.add_argument("--format", default="text", choices=["text", "markdown"])
    ap.add_argument("--passages", type=int, default=1,
                    help="how many query-selected windows per page")
    ap.add_argument("--passage-chars", type=int, default=500)
    ap.add_argument("--token-budget", type=int, default=0,
                    help="cap the whole SERP at this many tokens")
    ap.add_argument("--token-budget-per-page", type=int, default=0)
    ap.add_argument("--text", action="store_true", help="include full page text")
    ap.add_argument("--media", action="store_true", help="include favicon and images")
    ap.add_argument("--answer", nargs="?", const="extract", default=None,
                    metavar="{extract,synthesize}",
                    help="answer the question from the ranked pages")

    ap.add_argument("--json", action="store_true")
    ap.add_argument("--debug", action="store_true")
    return ap


async def search_main(argv: list[str]) -> int:
    from .engine import Liberdex
    from .rank import Models

    args = search_parser().parse_args(argv)
    # `--answer` takes an optional mode, so `--answer "the question"` hands
    # argparse the question as the mode. Anything that is not a mode is the
    # start of the query, and the mode is the default.
    if args.answer is not None and args.answer not in ("extract", "synthesize"):
        args.query.insert(0, args.answer)
        args.answer = "extract"
    query = " ".join(args.query).strip()
    if not query:
        search_parser().error("the following arguments are required: query")

    plan = None
    if args.plan_file:
        from .planner.base import plan_from_dict
        with open(args.plan_file, "rb") as fh:
            plan = plan_from_dict(query, orjson.loads(fh.read()))

    Models.preload()
    tier = dict(DEPTH[args.depth])
    planner = None if plan is not None else build_planner(args)
    if args.budget:
        tier["budget"] = args.budget
    # Grace for the planner's first URL, or the floor a whole-reply planner
    # declares: the same rule the server and the MCP server apply.
    size_budget(tier, planner, depth=args.depth, explicit=bool(args.budget),
                supplied=plan is not None)
    if args.no_expand:
        tier["expand_hubs"] = False

    after = args.after or (freshness_floor(args.freshness) or None)

    eng = Liberdex(planner=planner, speculate=not args.no_speculate)
    async with eng:
        resp = await eng.search(
            query, top_k=args.top_k,
            keep_text=args.text or args.format == "markdown" or bool(args.answer),
            snippet_chars=args.passage_chars, snippet_windows=args.passages,
            include_domains=[d for d in args.sites.split(",") if d.strip()],
            exclude_domains=[d for d in args.not_sites.split(",") if d.strip()],
            published_after=after, published_before=args.before,
            require_date=args.require_date, intent=args.intent,
            want_media=args.media, markdown=args.format == "markdown",
            token_budget=args.token_budget,
            token_budget_per_page=args.token_budget_per_page,
            plan=plan,
            **tier,
        )
        text = ""
        if args.answer:
            from .answer import judged, respond
            from .planner.factory import answer_default
            # The answer role follows the planner: a planner that is a local
            # process has no key the answer could inherit, so the answer runs on
            # the same binary rather than failing on a hosted provider.
            answer_model = args.answer_model or answer_default(planner)
            a = await respond(query, resp.results, mode=args.answer,
                              model=answer_model)
            text, rep = a.text, a.reply
            if a.pages:
                resp.results = judged(resp.results, a.pages)
                resp.stats["judged"] = len(a.pages)
            if args.debug:
                print(f"answer pages: {a.pages}", file=sys.stderr)
            if not args.text and args.format != "markdown":
                for r in resp.results:
                    r.text = ""
            if rep.error:
                print(f"\x1b[31manswer failed: {rep.error}\x1b[0m", file=sys.stderr)
    print(render(resp, as_json=args.json, debug=args.debug, answer=text))
    return 0


# ------------------------------------------------------------------------ main
async def amain(argv: list[str]) -> int:
    head = argv[0] if argv else ""
    if head == "models":
        if argv[1:2] == ["check"]:
            return await models_check(argv[2:])
        print("usage: liberdex models check [profile ...]", file=sys.stderr)
        return 2
    if head == "commons":
        if argv[1:2] == ["forget"]:
            from .commons import Commons
            c = Commons(register=False)
            ok = await c.forget()
            await c.aclose()
            print("forgotten; the key is gone from ~/.config/liberdex/env" if ok
                  else f"nothing to forget ({c.error or 'no key'})", file=sys.stderr)
            return 0 if ok else 1
        print("usage: liberdex commons forget", file=sys.stderr)
        return 2
    if head == "mcp":
        from .mcp import main as mcp_main
        return await mcp_main(argv[1:])
    if head == "search":
        argv = argv[1:]
    return await search_main(argv)


def main() -> int:
    argv = list(sys.argv[1:])
    from .envfile import load
    load()
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE, end="")
        return 0 if argv else 2
    if argv[0] in ("-V", "--version"):
        from . import __version__
        print(f"liberdex {__version__}")
        return 0
    if argv[0] == "serve":
        # uvicorn owns the event loop here, so this one stays synchronous.
        from .serve import main as serve_main
        return serve_main(argv[1:])
    if argv[0] == "install":
        from .install import main as install_main
        return install_main(argv[1:])
    if argv[0] == "warmup":
        if argv[1:]:
            print("usage: liberdex warmup    download the ranking models now, "
                  "about 130 MB the first time")
            return 0 if argv[1] in ("-h", "--help") else 2
        return warmup()
    return asyncio.run(amain(argv))


if __name__ == "__main__":
    sys.exit(main())
