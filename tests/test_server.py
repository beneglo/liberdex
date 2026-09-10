"""HTTP routes, against a stub engine: no network, no models, no LLM."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from starlette.responses import Response

import liberdex.server as srv
from liberdex.rank import WINDOW_JOIN
from liberdex.types import Candidate, Doc, Plan, Result, SearchResponse


class StubEngine:
    """Records what the routes asked for and answers with fixed data."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def search(self, query, **kw):
        self.calls.append((query, kw))
        cb = kw.get("on_event")
        if cb is not None:
            cb("plan", Plan(query=query, intent="reference", lang="en",
                            routes=["wikipedia"], hubs=["https://h.test"]))
            cb("candidate", Candidate(url="https://x.test/a", source="llm"))
            cb("page", Doc(url="https://x.test/a", final_url="https://x.test/a",
                           status=200, title="A", site="x.test", text="alpha beta"))
        return SearchResponse(
            query=query, intent="reference",
            results=[Result(
                url="https://x.test/a", title="A", snippet="alpha" + WINDOW_JOIN + "beta",
                site="x.test", score=0.9, published="2024-05-01", source="llm",
                passages=["alpha", "beta"], passage_scores=[0.7, 0.3], status=200,
                text="whole page", text_chars=len("whole page"),
                favicon="https://x.test/f.ico",
                images=["https://x.test/i.png"],
            )],
            timings={"total_ms": 1930.0}, stats={"lang": "en"},
            plan=Plan(query=query, intent="reference",
                      candidates=[Candidate(url="https://x.test/a", prior=0.9)]),
        )

    async def plan(self, query, *, budget=6.0, max_candidates=14):
        self.calls.append((query, {"budget": budget}))
        return Plan(query=query, intent="code", lang="en", routes=["github"],
                    route_queries={"github": "tokio"}, hubs=["https://h.test"],
                    candidates=[Candidate(url="https://d.test/a", prior=0.8)])

    async def extract_urls(self, urls, **kw):
        self.calls.append(("extract", {"urls": list(urls), **kw}))
        out = []
        for i, u in enumerate(urls):
            if "bad" in u:
                out.append(Doc(url=u, final_url=u, status=404, error="not found"))
                continue
            out.append(Doc(url=u, final_url=u, status=200, title=f"T{i}",
                           text="alpha beta gamma " * 200, site="x.test",
                           favicon="https://x.test/f.ico",
                           images=["https://x.test/i.png"],
                           links=[("https://x.test/b", "next", "see also")]))
        return out

    async def query_from_url(self, url, *, budget=8.0):
        self.calls.append(("like", {"url": url}))
        return ("seeded query", "x.test")


@pytest.fixture()
def client(monkeypatch):
    stub = StubEngine()
    monkeypatch.setattr(srv, "_engine", stub)
    c = TestClient(srv.app)
    c.stub = stub
    return c


def test_health_reports_the_engine_and_version(client):
    body = client.get("/health").json()
    assert body["ok"] is True and body["version"] == srv.app.version


def test_search_returns_the_native_shape(client):
    body = client.post("/search", json={"query": "q"}).json()
    assert body["query"] == "q" and body["intent"] == "reference"
    assert body["lang"] == "en" and body["request_id"]
    row = body["results"][0]
    assert row["passages"] == ["alpha", "beta"]
    assert row["passage_scores"] == [0.7, 0.3]
    assert row["source"] == "llm" and row["status"] == 200
    # Not requested, so absent.
    assert row["text"] is None and row["favicon"] is None
    # The page's size travels even when its text does not.
    assert row["text_chars"] == len("whole page")
    assert body["plan"] is None


def test_search_honours_the_optional_blocks(client):
    body = client.post("/search", json={
        "query": "q", "include_text": True, "include_favicon": True,
        "include_images": True, "include_plan": True,
    }).json()
    row = body["results"][0]
    assert row["text"] == "whole page"
    assert row["favicon"] == "https://x.test/f.ico"
    assert row["images"] == ["https://x.test/i.png"]
    assert body["plan"]["candidates"][0]["url"] == "https://x.test/a"


def test_retrieval_controls_reach_the_engine(client):
    client.post("/search", json={
        "query": "q", "include_domains": ["a.test"], "exclude_domains": ["b.test"],
        "published_after": "2024-01-01", "intent": "news", "format": "markdown",
        "token_budget": 500, "passages_per_page": 3,
    })
    _q, kw = client.stub.calls[-1]
    assert kw["include_domains"] == ["a.test"]
    assert kw["exclude_domains"] == ["b.test"]
    assert kw["published_after"] == "2024-01-01"
    assert kw["intent"] == "news" and kw["markdown"] is True
    assert kw["token_budget"] == 500 and kw["snippet_windows"] == 3


def test_a_query_list_runs_them_all(client):
    body = client.post("/search", json={"query": ["one", "two"]}).json()
    assert [r["query"] for r in body["results"]] == ["one", "two"]
    assert {q for q, _ in client.stub.calls} == {"one", "two"}


def test_seeding_from_a_url_excludes_the_page_s_own_site(client):
    body = client.post("/search", json={"like": "https://x.test/a"}).json()
    assert body["query"] == "seeded query"
    assert body["stats"]["seeded_from"] == "https://x.test/a"
    _q, kw = client.stub.calls[-1]
    assert "x.test" in kw["exclude_domains"]


def test_search_needs_a_query_or_a_seed(client):
    assert client.post("/search", json={}).status_code == 422


def test_get_search_takes_comma_separated_domains(client):
    body = client.get("/search", params={"q": "q", "include_domains": "a.test,b.test",
                                         "top_k": 3}).json()
    assert body["query"] == "q"
    _q, kw = client.stub.calls[-1]
    assert kw["include_domains"] == ["a.test", "b.test"] and kw["top_k"] == 3


def test_the_stream_reports_the_pipeline_then_the_results(client):
    with client.stream("POST", "/search/stream", json={"query": "q"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(resp.iter_text())
    frames = [json.loads(line[6:]) for line in body.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    kinds = [f["event"] for f in frames]
    # The pipeline's own stages, in the order it runs them.
    assert kinds == ["plan", "candidate", "page", "results"]
    plan_frame = frames[0]["data"]
    assert plan_frame["routes"] == ["wikipedia"] and plan_frame["hubs"] == ["https://h.test"]
    assert frames[2]["data"]["url"] == "https://x.test/a"
    assert frames[2]["data"]["status"] == 200
    assert body.rstrip().endswith("data: [DONE]")
    final = frames[-1]["data"]
    assert final["results"][0]["passages"] == ["alpha", "beta"]
    # One request id across every frame, so a client can correlate them.
    assert len({f["id"] for f in frames}) == 1


def test_the_stream_is_shaped_like_an_openai_chunk(client):
    with client.stream("POST", "/search/stream", json={"query": "q"}) as resp:
        body = "".join(resp.iter_text())
    first = json.loads(next(l for l in body.splitlines()
                            if l.startswith("data: {"))[6:])
    assert first["object"] == "search.chunk"
    assert {"id", "created", "event", "data"} <= set(first)


def test_ranking_weights_reach_the_engine(client):
    client.post("/search", json={"query": "q", "weights": {"ce": 0.5, "rare": 0.9}})
    _q, kw = client.stub.calls[-1]
    assert kw["weights"] == {"ce": 0.5, "rare": 0.9}


def test_an_unknown_ranking_weight_is_a_422_not_a_silent_no_op(client):
    """Ignoring a typo would let a caller believe they had reweighted the SERP."""
    r = client.post("/search", json={"query": "q", "weights": {"nope": 1.0}})
    assert r.status_code == 422
    assert "unknown ranking weights" in r.text


def test_extract_splits_readable_pages_from_dead_ones(client):
    body = client.post("/extract", json={
        "urls": ["https://x.test/a", "https://x.test/bad"]}).json()
    assert [r["url"] for r in body["results"]] == ["https://x.test/a"]
    assert body["failed"][0]["status"] == 404
    # No query, so the page comes back whole rather than cut into passages.
    assert body["results"][0]["passages"] == []


def test_extract_with_a_query_returns_passages(client):
    body = client.post("/extract", json={
        "urls": ["https://x.test/a"], "query": "beta gamma",
        "passages_per_page": 2, "passage_chars": 400}).json()
    page = body["results"][0]
    assert 1 <= len(page["passages"]) <= 2
    assert len(page["passage_scores"]) == len(page["passages"])
    # The passages stand in for the page; its size still says what whole costs.
    assert page["text"] == "" and page["text_chars"] == len("alpha beta gamma " * 200)


def test_extract_reports_the_whole_page_size_when_the_text_is_cut(client):
    body = client.post("/extract", json={
        "urls": ["https://x.test/a"], "token_budget_per_page": 50}).json()
    page = body["results"][0]
    assert page["text_chars"] == len("alpha beta gamma " * 200)
    assert 0 < len(page["text"]) < page["text_chars"] and page["text"].endswith(" ...")


def test_extract_optional_blocks(client):
    body = client.post("/extract", json={
        "urls": ["https://x.test/a"], "include_favicon": True,
        "include_images": True, "include_links": True}).json()
    page = body["results"][0]
    assert page["favicon"] == "https://x.test/f.ico"
    assert page["links"] == [{"url": "https://x.test/b", "anchor": "next",
                             "context": "see also"}]


def test_extract_rejects_an_empty_or_oversized_url_list(client):
    assert client.post("/extract", json={"urls": []}).status_code == 422
    assert client.post("/extract", json={
        "urls": [f"https://x.test/{i}" for i in range(21)]}).status_code == 422


def test_plan_returns_the_plan_and_fetches_nothing(client):
    body = client.post("/plan", json={"query": "tokio select"}).json()
    assert body["plan"]["routes"] == ["github"]
    assert body["plan"]["route_queries"] == {"github": "tokio"}
    assert body["plan"]["hubs"] == ["https://h.test"]
    assert body["plan"]["candidates"][0]["url"] == "https://d.test/a"


def test_every_route_refuses_before_the_engine_is_up(monkeypatch):
    monkeypatch.setattr(srv, "_engine", None)
    c = TestClient(srv.app)
    assert c.post("/search", json={"query": "q"}).status_code == 503
    assert c.post("/plan", json={"query": "q"}).status_code == 503
    assert c.post("/extract", json={"urls": ["https://x.test"]}).status_code == 503


def test_the_querystring_search_is_spelled_the_same_as_every_other_query(client):
    """`query` is the field name in the body, so it is the field name here."""
    assert client.get("/search", params={"query": "tungsten"}).status_code == 200
    assert client.get("/search", params={"q": "tungsten"}).status_code == 200
    assert client.get("/search", params={"top_k": 3}).status_code == 422


# ----------------------------------------------------------------- rounds
def test_a_page_turn_reaches_the_engine_and_skips_the_answer(client):
    body = client.post("/search", json={
        "query": "q", "page": 2, "answer": True,
        "exclude_urls": ["https://x.test/shown"],
    }).json()
    _q, kw = client.stub.calls[-1]
    assert kw["page"] == 2 and kw["exclude_urls"] == ["https://x.test/shown"]
    assert body["page"] == 2
    # Page one answered the question. A later page never calls the model.
    assert body["answer"] is None
    assert "answer_error" not in body["stats"] and "answer_mode" not in body["stats"]


def test_the_querystring_form_takes_a_page_too(client):
    body = client.get("/search", params={
        "q": "q", "page": 3, "exclude_urls": "https://x.test/1,https://x.test/2",
    }).json()
    _q, kw = client.stub.calls[-1]
    assert kw["page"] == 3
    assert kw["exclude_urls"] == ["https://x.test/1", "https://x.test/2"]
    assert body["page"] == 3


# ---------------------------------------------------------------- the door
@pytest.fixture()
def keyed(client, monkeypatch):
    monkeypatch.setenv("LIBERDEX_KEYS", "k-one, k-two")
    return client


def test_without_keys_everything_is_open(client, monkeypatch):
    monkeypatch.delenv("LIBERDEX_KEYS", raising=False)
    assert client.post("/search", json={"query": "q"}).status_code == 200


def test_keys_guard_the_engine_and_leave_the_page_open(keyed):
    assert keyed.post("/search", json={"query": "q"}).status_code == 401
    assert keyed.get("/search", params={"q": "q"}).status_code == 401
    assert keyed.post("/extract", json={"urls": ["https://x.test/a"]}).status_code == 401
    assert keyed.post("/plan", json={"query": "q"}).status_code == 401
    r = keyed.post("/search", json={"query": "q"}, headers={"authorization": "Bearer nope"})
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert keyed.post("/search", json={"query": "q"},
                      headers={"authorization": "Bearer k-two"}).status_code == 200
    assert keyed.post("/search", json={"query": "q"},
                      headers={"x-api-key": "k-one"}).status_code == 200
    assert keyed.get("/health").status_code == 200
    assert keyed.get("/").status_code == 200
    assert keyed.get("/opensearch.xml").status_code == 200


def test_cors_answers_only_the_origins_it_was_given(client, monkeypatch):
    monkeypatch.setenv("LIBERDEX_CORS", "https://app.example")
    r = client.options("/search", headers={"origin": "https://app.example",
                                            "access-control-request-method": "POST"})
    assert r.status_code == 204
    assert r.headers["access-control-allow-origin"] == "https://app.example"
    assert "authorization" in r.headers["access-control-allow-headers"]
    r = client.post("/search", json={"query": "q"}, headers={"origin": "https://app.example"})
    assert r.headers["access-control-allow-origin"] == "https://app.example"
    r = client.post("/search", json={"query": "q"}, headers={"origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers
    monkeypatch.delenv("LIBERDEX_CORS")
    r = client.post("/search", json={"query": "q"}, headers={"origin": "https://app.example"})
    assert "access-control-allow-origin" not in r.headers


def test_the_inflight_cap_says_busy_instead_of_queueing(client, monkeypatch):
    monkeypatch.setenv("LIBERDEX_MAX_INFLIGHT", "1")
    sem = srv._semaphore()
    # Hold the one slot from outside, as a running search would.
    sem._value = 0
    try:
        r = client.post("/search", json={"query": "q"})
        assert r.status_code == 503 and r.headers["retry-after"] == "2"
        assert client.get("/health").status_code == 200
    finally:
        sem._value = 1
    assert client.post("/search", json={"query": "q"}).status_code == 200


def test_mcp_is_mounted_on_the_same_server(keyed):
    # No key: the door answers before the MCP app does. (The MCP app itself
    # needs the lifespan, which the stub client does not run.)
    assert keyed.post("/mcp", json={}).status_code == 401
    assert keyed.post("/mcp/", json={}).status_code == 401
    assert any(getattr(r, "path", "") == "/mcp" for r in srv.app.routes)


@pytest.mark.asyncio
async def test_the_mounted_mcp_offers_the_two_tools():
    from mcp.client.client import Client
    async with Client(srv._mcp) as c:
        names = sorted(t.name for t in (await c.list_tools()).tools)
    assert names == ["extract", "search"]


def test_mcp_without_the_trailing_slash_is_not_a_redirect(client, monkeypatch):
    # A connector is configured as https://host/mcp; the mounted app lives at
    # /mcp/. The door rewrites the path so no client has to follow a 307.
    monkeypatch.delenv("LIBERDEX_KEYS", raising=False)
    seen = {}

    async def spy(scope, receive, send):
        seen["path"] = scope["path"]
        await Response(status_code=204)(scope, receive, send)

    from starlette.routing import Mount
    for r in srv.app.routes:
        if isinstance(r, Mount) and r.path == "/mcp":
            monkeypatch.setattr(r, "app", spy)
    r = client.post("/mcp", json={}, follow_redirects=False)
    assert r.status_code == 204 and seen["path"] == "/mcp/"


# ------------------------------------------------------------- plan grace
def test_the_server_sizes_the_deadline_for_its_planner_like_the_cli_does(client):
    """The page was the one entry point that cut a slow planner: the CLI and
    the MCP server grew the budget to `min_budget`, `/search` did not."""
    from types import SimpleNamespace

    from liberdex.api import DEPTH, PLAN_GRACE
    client.post("/search", json={"query": "q"})
    kw = client.stub.calls[-1][1]
    assert kw["budget"] == DEPTH["standard"]["budget"]
    assert kw["plan_grace"] == PLAN_GRACE["standard"]

    client.post("/search", json={"query": "q", "budget": 4.0})
    kw = client.stub.calls[-1][1]
    assert kw["budget"] == 4.0 and kw["plan_grace"] == 0.0

    client.stub.planner = SimpleNamespace(min_budget=15.0)
    client.post("/search", json={"query": "q"})
    kw = client.stub.calls[-1][1]
    assert kw["budget"] == 15.0 and kw["plan_grace"] == 0.0
    with client.stream("POST", "/search/stream", json={"query": "q"}) as resp:
        resp.read()
    kw = client.stub.calls[-1][1]
    assert kw["budget"] == 15.0 and kw["plan_grace"] == 0.0
