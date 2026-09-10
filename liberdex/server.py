"""HTTP API for liberdex.

    uv run uvicorn liberdex.server:app --port 8080
    curl -s localhost:8080/search -d '{"query":"how does TLS 1.3 work"}' \
         -H 'content-type: application/json' | jq

The response speaks liberdex's own vocabulary: plan, page, passage, source.
"""
from __future__ import annotations

import asyncio
import hmac
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

import orjson
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import __version__, web
from . import settings as settings_mod
from .api import (
    DEPTH,
    MAX_PAGE,
    Depth,
    ExtractReply,
    ExtractRequest,
    Format,
    Freshness,
    Intent,
    PlanReply,
    PlanRequest,
    SearchReply,
    SearchRequest,
    Timer,
    plan_stats,
    plan_view,
    reply,
    request_id,
    sse,
)
from .api import extract as api_extract
from .cache import PlanStore
from .engine import Liberdex
from .rank import Models
from .types import Result

_engine: Optional[Liberdex] = None

__all__ = ["app", "DEPTH"]


def _build_planner():
    """The planner named by the environment, resolved through model profiles.

    `LIBERDEX_PLANNER` is `auto` (default), `claude`, `openai`, `replay` or
    `none`; `LIBERDEX_PLANNER_MODEL` is a `profile/model` spec and outranks
    the backend name. When nothing on the machine can plan the server still
    starts, because the page has to come up to say so, and `/health` carries
    the reason.
    """
    from .planner.factory import make_planner
    backend = os.environ.get("LIBERDEX_PLANNER", "auto").lower()
    spec = os.environ.get("LIBERDEX_PLANNER_MODEL", "")
    try:
        planner = make_planner(
            backend, spec,
            base_url=os.environ.get("LIBERDEX_BASE_URL"),
            reasoning_effort=os.environ.get("LIBERDEX_REASONING_EFFORT"),
            cache=os.environ.get("LIBERDEX_PLAN_CACHE", "1") != "0",
            store=PlanStore() if os.environ.get("LIBERDEX_PLAN_CACHE", "1") != "0" else None)
    except LookupError as e:
        app.state.planner_error = str(e)
        return None
    app.state.planner_error = ""
    return planner


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    from .envfile import load
    load()
    Models.preload()
    _engine = Liberdex(planner=_build_planner())
    await _engine.start()
    _mcp_service.engine = _engine
    try:
        async with _mcp.session_manager.run():
            yield
    finally:
        _mcp_service.engine = None
        await _engine.aclose()
        _engine = None
        await web.aclose()


app = FastAPI(title="liberdex", version=__version__, lifespan=lifespan)

# The search page, on the same origin as the API it calls. Adds `/` and a few
# assets; every route above is untouched.
web.mount(app)

# The same two MCP tools the stdio server offers, over streamable HTTP at
# /mcp, so a hosted liberdex is a connector for claude.ai, Claude Desktop and
# ChatGPT. The engine is this process's; the MCP lifespan does not own it.
from .mcp import Service as _McpService  # noqa: E402
from .mcp import build_server as _build_mcp  # noqa: E402

_mcp_service = _McpService(None, shared=True)
_mcp = _build_mcp(_mcp_service)
app.mount("/mcp", _mcp.streamable_http_app(
    streamable_http_path="/", stateless_http=True,
    host=os.environ.get("LIBERDEX_HOST", "127.0.0.1")))


# ------------------------------------------------------------------- the door
# Off by default, so a local server is exactly what it was. Set, they make
# the server safe to bind beyond loopback: keys callers must present, the
# origins a browser may call from, and how many searches run at once.
# Only these need a key. The page, /health, the icon proxy and the OpenAPI
# views stay open, so a keyed server is still a server someone can look at.
PROTECTED = ("/search", "/extract", "/plan", "/mcp")

_inflight: Optional[asyncio.Semaphore] = None
_inflight_size = 0


def _keys() -> list[str]:
    return [k.strip() for k in os.environ.get("LIBERDEX_KEYS", "").split(",") if k.strip()]


def _origins() -> list[str]:
    return [o.strip().rstrip("/") for o in os.environ.get("LIBERDEX_CORS", "").split(",")
            if o.strip()]


def _presented(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key", "").strip()


def _protected(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in PROTECTED)


def _cors_headers(request: Request) -> dict[str, str]:
    origin = request.headers.get("origin", "").rstrip("/")
    allowed = _origins()
    if not origin or not allowed or (origin not in allowed and "*" not in allowed):
        return {}
    return {
        "access-control-allow-origin": origin,
        "access-control-allow-methods": "GET, POST, PUT, OPTIONS",
        "access-control-allow-headers": "authorization, content-type, x-api-key, "
                                        "mcp-session-id, mcp-protocol-version",
        "access-control-expose-headers": "x-request-id, mcp-session-id",
        "access-control-max-age": "600",
        "vary": "origin",
    }


def _semaphore() -> asyncio.Semaphore:
    global _inflight, _inflight_size
    n = max(1, int(os.environ.get("LIBERDEX_MAX_INFLIGHT", "8") or 8))
    if _inflight is None or n != _inflight_size:
        _inflight, _inflight_size = asyncio.Semaphore(n), n
    return _inflight


@app.middleware("http")
async def door(request: Request, call_next):
    path = request.url.path
    if path == "/mcp":
        # The MCP app is mounted under /mcp and answers at its root; without
        # this a connector configured as .../mcp would be sent a redirect.
        request.scope["path"] = "/mcp/"
    cors = _cors_headers(request)
    if request.method == "OPTIONS" and "origin" in request.headers:
        return Response(status_code=204, headers=cors)
    keys = _keys()
    if keys and _protected(path):
        given = _presented(request)
        if not any(hmac.compare_digest(given, k) for k in keys):
            return JSONResponse({"detail": "an API key is required: "
                                 "Authorization: Bearer <key>"},
                                status_code=401,
                                headers={"www-authenticate": "Bearer", **cors})
    if _protected(path) and request.method in ("POST", "GET"):
        sem = _semaphore()
        if sem.locked():
            return JSONResponse({"detail": "liberdex is busy; try again shortly"},
                                status_code=503,
                                headers={"retry-after": "2", **cors})
        async with sem:
            response = await call_next(request)
    else:
        response = await call_next(request)
    for k, v in cors.items():
        response.headers[k] = v
    return response


def engine() -> Liberdex:
    if _engine is None:
        raise HTTPException(503, "engine not ready")
    return _engine


@app.get("/health")
async def health() -> dict[str, Any]:
    out: dict[str, Any] = {"ok": _engine is not None, "version": app.version}
    err = getattr(app.state, "planner_error", "")
    if err:
        out["planner_error"] = err
    commons = getattr(_engine, "commons", None)
    if commons is not None:
        out["commons"] = commons.state()
    return out


# ---------------------------------------------------------------------- search
async def _run_one(eng: Liberdex, req: SearchRequest, query: str,
                   *, exclude_host: str = "") -> Any:
    kwargs = req.engine_kwargs(planner=getattr(eng, "planner", None))
    if exclude_host:
        # A page is never its own best neighbour, and nor is the rest of its
        # site: the point of seeding from a URL is to find what else says this.
        kwargs["exclude_domains"] = list(kwargs.get("exclude_domains") or []) + [exclude_host]
    return await eng.search(query, **kwargs)


async def _answer_for(req: SearchRequest, query: str, results: list[Result]
                      ) -> tuple[Optional[str], dict[str, Any]]:
    """The answer, plus what it cost. Failures degrade to no answer, never 500.

    `results` is reordered in place when the answer model, having read the
    pages, says which were about the question: see `answer.judged`.
    """
    mode = req.answer if isinstance(req.answer, str) else "extract"
    # The question was answered on page one. A later page is more sources
    # for the same question, and the answer call is the one paid stage that
    # a page turn does not need to repeat.
    if not req.answer or not results or req.page > 1:
        return None, {}
    info: dict[str, Any] = {"answer_mode": "schema" if req.output_schema else mode}
    from .planner.factory import answer_default
    model = req.answer_model or answer_default(engine().planner)
    try:
        if req.output_schema:
            from .answer import structured
            obj, rep = await structured(query, results, req.output_schema, model=model)
            text = None if obj is None else orjson.dumps(obj).decode()
        else:
            from .answer import judged, respond
            a = await respond(query, results, mode=mode, model=model)
            rep, text = a.reply, a.text or None
            if a.pages:
                results[:] = judged(results, a.pages)
                info["judged"] = len(a.pages)
    except Exception as e:
        # A model that cannot even be resolved (no profile, bad spec) is the
        # same story as one that answered with an error: the search stands.
        info["answer_error"] = f"{type(e).__name__}: {e}"[:300]
        return None, info
    if rep.error:
        info["answer_error"] = rep.error
    else:
        info["answer_model"] = rep.model
        info["answer_tokens"] = rep.prompt_tokens + rep.completion_tokens
    return text, info


@app.post("/search")
async def search(req: SearchRequest):
    eng = engine()
    timer = Timer()
    rid = request_id()

    queries = req.queries()
    exclude_host = ""
    if req.like:
        query, exclude_host = await eng.query_from_url(req.like)
        if not query:
            raise HTTPException(422, f"could not read a query out of {req.like!r}")
        queries = [query]

    if len(queries) > 1:
        # Concurrent, sharing the fetcher's global semaphore, so N queries cost
        # far less than N sequential searches.
        responses = await asyncio.gather(*(
            _run_one(eng, req, q) for q in queries
        ))
        replies = []
        for q, resp in zip(queries, responses):
            text, info = await _answer_for(req, q, resp.results)
            r = reply(resp, req, rid=f"{rid}-{len(replies)}", answer=text)
            r.stats.update(info)
            replies.append(r)
        body = {"request_id": rid,
                "results": [r.model_dump() for r in replies],
                "response_ms": timer.ms()}
        return JSONResponse(body)

    resp = await _run_one(eng, req, queries[0], exclude_host=exclude_host)
    text, info = await _answer_for(req, queries[0], resp.results)
    out = reply(resp, req, rid=rid, answer=text)
    out.stats.update(info)
    if req.like:
        out.stats["seeded_from"] = req.like
    return out


@app.get("/search", response_model=SearchReply)
async def search_get(
    # `query` is the field name everywhere else in this API, so it is the field
    # name here too. `q` stays accepted because a querystring search is the one
    # people type by hand and `q` is what a hand types.
    query: str = Query(default="", max_length=800),
    q: str = Query(default="", max_length=800),
    top_k: int = Query(default=10, ge=1, le=25),
    depth: Depth = Query(default="standard"),
    format: Format = Query(default="text"),
    intent: Optional[Intent] = Query(default=None),
    freshness: Optional[Freshness] = Query(default=None),
    passages_per_page: int = Query(default=3, ge=1, le=8),
    passage_chars: int = Query(default=500, ge=80, le=8000),
    include_text: bool = False,
    include_plan: bool = False,
    include_domains: str = Query(default="", description="comma separated"),
    exclude_domains: str = Query(default=""),
    token_budget: int = Query(default=0, ge=0, le=200_000),
    page: int = Query(default=1, ge=1, le=MAX_PAGE),
    exclude_urls: str = Query(default="", description="comma separated; the "
                              "URLs already shown on earlier pages"),
):
    """Querystring form, for a browser or a one-line curl."""
    text = (query or q).strip()
    if not text:
        raise HTTPException(422, "`query` is required")
    req = SearchRequest(
        query=text, top_k=top_k, depth=depth, format=format,
        intent=intent, freshness=freshness,
        passages_per_page=passages_per_page, passage_chars=passage_chars,
        include_text=include_text, include_plan=include_plan,
        token_budget=token_budget,
        include_domains=[d for d in include_domains.split(",") if d.strip()],
        exclude_domains=[d for d in exclude_domains.split(",") if d.strip()],
        page=page,
        exclude_urls=[u for u in exclude_urls.split(",") if u.strip()],
    )
    resp = await _run_one(engine(), req, text)
    return reply(resp, req)


@app.post("/search/stream")
async def search_stream(req: SearchRequest):
    """The same search, but the pipeline reports as it runs.

    Nothing is faked: the first URL is dispatched long before the SERP is
    ready and the consume loop is already event-driven. Events follow the
    pipeline's own stages: `candidate` as each URL is dispatched, `plan` as
    the planner names routes and hubs, `page` as each one comes back
    readable, then `results`.
    """
    eng = engine()
    rid = request_id()
    query, exclude_host = req.queries()[0], ""
    if req.like:
        query, exclude_host = await eng.query_from_url(req.like)
        if not query:
            raise HTTPException(422, f"could not read a query out of {req.like!r}")

    events: asyncio.Queue = asyncio.Queue()

    def on_event(kind: str, payload: Any) -> None:
        # Called from the engine's own loop, so it must not block. A slow or
        # vanished client backs up here and nowhere else.
        if kind == "candidate":
            events.put_nowait(("candidate", {"url": payload.url,
                                             "source": payload.source}))
        elif kind == "page":
            events.put_nowait(("page", {"url": payload.final_url,
                                        "title": payload.title,
                                        "site": payload.site,
                                        "status": payload.status,
                                        "chars": len(payload.text or "")}))
        elif kind == "plan":
            events.put_nowait(("plan", {"intent": payload.intent,
                                        "lang": payload.lang,
                                        "routes": list(payload.routes),
                                        "hubs": list(payload.hubs),
                                        "expansions": list(payload.expansions)}))

    async def run():
        kwargs = req.engine_kwargs(planner=getattr(eng, "planner", None))
        if exclude_host:
            kwargs["exclude_domains"] = list(kwargs.get("exclude_domains") or []) \
                + [exclude_host]
        return await eng.search(query, on_event=on_event, **kwargs)

    async def frames():
        task = asyncio.create_task(run())
        try:
            while True:
                drain = asyncio.create_task(events.get())
                done, _ = await asyncio.wait(
                    {drain, task}, return_when=asyncio.FIRST_COMPLETED)
                if drain in done:
                    kind, data = drain.result()
                    yield sse(rid, kind, data)
                    continue
                drain.cancel()
                break
            # Whatever the engine queued before finishing still belongs to the
            # client; draining it after the task is the difference between a
            # complete stream and one that stops mid-fetch.
            while not events.empty():
                kind, data = events.get_nowait()
                yield sse(rid, kind, data)
            resp = await task
            text, info = await _answer_for(req, query, resp.results)
            out = reply(resp, req, rid=rid, answer=text)
            out.stats.update(info)
            yield sse(rid, "results", out.model_dump())
        except Exception as e:
            yield sse(rid, "error", {"message": f"{type(e).__name__}: {e}"})
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        yield "data: [DONE]\n\n"

    return StreamingResponse(frames(), media_type="text/event-stream",
                             headers={"cache-control": "no-store",
                                      "x-request-id": rid})


# -------------------------------------------------------------------- settings
# The drawer on the page. Loopback only, unless told otherwise: it writes API
# keys to this machine's disk, which a page on the open internet must not.
def _settings_or_404() -> None:
    if not settings_mod.enabled():
        raise HTTPException(404, "settings are not served on this bind")


async def rebuild() -> None:
    """Swap the engine's planner for what the settings now say."""
    eng = engine()
    old = eng.planner
    eng.planner = _build_planner()
    closer = getattr(old, "aclose", None)
    if closer is not None:
        try:
            await closer()
        except Exception:
            pass


@app.get("/settings", include_in_schema=False)
async def settings_get() -> dict[str, Any]:
    _settings_or_404()
    return settings_mod.snapshot(getattr(app.state, "planner_error", ""))


@app.put("/settings", include_in_schema=False)
async def settings_put(body: dict[str, Any]) -> dict[str, Any]:
    _settings_or_404()
    try:
        written = settings_mod.update(body)
    except ValueError as e:
        raise HTTPException(422, str(e))
    await rebuild()
    out = settings_mod.snapshot(getattr(app.state, "planner_error", ""))
    out["written"] = written
    return out


@app.get("/settings/models", include_in_schema=False)
async def settings_models(profile: str = Query(default="", max_length=64)) -> dict[str, Any]:
    _settings_or_404()
    return {"profile": profile, "models": await settings_mod.list_models(profile)}


# --------------------------------------------------------------------- extract
@app.post("/extract", response_model=ExtractReply)
async def extract_pages(req: ExtractRequest) -> ExtractReply:
    return await api_extract(engine(), req)


# ------------------------------------------------------------------------ plan
@app.post("/plan", response_model=PlanReply)
async def plan_only(req: PlanRequest) -> PlanReply:
    """Where does this live on the web? One LLM call, nothing fetched."""
    eng = engine()
    timer = Timer()
    plan = await eng.plan(req.query, budget=req.budget,
                          max_candidates=req.max_candidates)
    view = plan_view(plan)
    assert view is not None
    return PlanReply(request_id=request_id(), query=req.query, plan=view,
                     timings={"total_ms": timer.ms()}, stats=plan_stats(plan))
