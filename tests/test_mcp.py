"""The MCP tools, over an in-memory transport, against the stub engine."""
from __future__ import annotations

import pytest
from mcp.client.client import Client

from liberdex.api import SearchReply, SearchResult
from liberdex.mcp import Service, build_server, cut_note, render
from test_server import StubEngine


class McpStub(StubEngine):
    planner = None

    async def start(self):
        pass

    async def aclose(self):
        pass


class Planned(McpStub):
    planner = object()


@pytest.fixture()
def service():
    svc = Service(McpStub(), planner_error="nothing on this machine can plan")
    svc.ready = _noop  # models never load in tests
    return svc


async def _noop():
    return None


async def _call(service, name, **args):
    async with Client(build_server(service)) as c:
        return await c.call_tool(name, args)


@pytest.mark.asyncio
async def test_the_tools_are_search_and_extract(service):
    async with Client(build_server(service)) as c:
        tools = await c.list_tools()
    names = sorted(t.name for t in tools.tools)
    assert names == ["extract", "search"]
    search = next(t for t in tools.tools if t.name == "search")
    assert "plan" in search.input_schema["properties"]
    assert "never invent an opaque identifier" in search.description.lower()


@pytest.mark.asyncio
async def test_a_supplied_plan_reaches_the_engine_and_is_rendered(service):
    r = await _call(service, "search", query="tcp handshake", plan={
        "candidates": [{"url": "https://en.encyclopedia.example/wiki/Handshake", "prior": 0.9}],
        "routes": {"wikipedia": "handshake"}})
    assert r.is_error is False
    _query, kw = service.engine.calls[-1]
    plan = kw["plan"]
    assert [c.url for c in plan.candidates] == ["https://en.encyclopedia.example/wiki/Handshake"]
    assert plan.routes == ["wikipedia"]
    text = r.content[0].text
    assert "https://x.test/a" in text and "> alpha" in text
    assert r.structured_content["results"][0]["url"] == "https://x.test/a"
    assert r.structured_content["results"][0]["relevance"] is not None


@pytest.mark.asyncio
async def test_no_plan_and_no_planner_is_an_error_that_says_how_to_fix_it(service):
    r = await _call(service, "search", query="tcp handshake")
    assert r.is_error is True
    text = r.content[0].text
    assert "nothing on this machine can plan" in text
    assert "`plan`" in text and "liberdex install --model" in text
    assert service.engine.calls == []


@pytest.mark.asyncio
async def test_no_plan_with_a_planner_runs_the_engines_own():
    svc = Service(Planned())
    svc.ready = _noop
    r = await _call(svc, "search", query="tcp handshake", depth="fast", top_k=3)
    assert r.is_error is False
    _q, kw = svc.engine.calls[-1]
    assert kw["plan"] is None and kw["top_k"] == 3 and kw["budget"] == 2.5


@pytest.mark.asyncio
async def test_extract_reads_pages_and_reports_failures(service):
    r = await _call(service, "extract", urls=["https://x.test/a", "https://x.test/bad"],
                    query="alpha")
    assert r.is_error is False
    text = r.content[0].text
    assert "# T0" in text and "> alpha" in text
    assert "failed: https://x.test/bad" in text
    assert r.structured_content["failed"][0]["status"] == 404


@pytest.mark.asyncio
async def test_a_bad_request_is_an_error_not_a_crash(service):
    r = await _call(service, "extract", urls=["not a url"])
    assert r.is_error is True and "not http(s) URLs" in r.content[0].text


def test_render_names_planner_errors_so_a_thin_serp_is_not_mistaken_for_a_full_one():
    out = SearchReply(request_id="r", query="q", intent="reference",
                      results=[SearchResult(url="https://a.test", title="A",
                                            site="a.test", score=1.0, relevance=0.5,
                                            passages=["p"], source="route:x")],
                      stats={"planner_error": "cancelled"})
    text = render(out)
    assert "planner: cancelled" in text and "relevance=0.50" in text


def test_render_says_how_big_the_page_is_and_how_to_read_a_cut_one():
    row = SearchResult(url="https://a.test/p", title="A", site="a.test", score=1.0,
                       relevance=0.5, passages=["p"], source="llm",
                       text="head of the page ...", text_chars=44_800)
    text = render(SearchReply(request_id="r", query="q", intent="reference",
                              results=[row], stats={"dropped_over_budget": 2}))
    assert "page 44,800 chars" in text
    assert "[cut: 20 of 44,800 chars shown" in text
    # Three chars a token, rounded up to the thousand: a cap, so over rather than under.
    assert 'extract(urls=["https://a.test/p"], token_budget_per_page=15000)' in text
    assert "2 more pages ranked but dropped for `token_budget`" in text

    whole = row.model_copy(update={"text": "x" * 44_800})
    text = render(SearchReply(request_id="r", query="q", intent="reference",
                              results=[whole]))
    assert "[cut:" not in text
    assert cut_note("u", 10, 0) == "" and cut_note("u", 10, 10) == ""


@pytest.mark.asyncio
async def test_extract_reports_the_cut_and_the_page_size(service):
    r = await _call(service, "extract", urls=["https://x.test/a"],
                    token_budget_per_page=100)
    text = r.content[0].text
    page = r.structured_content["results"][0]
    assert page["text_chars"] == len("alpha beta gamma " * 200)
    assert len(page["text"]) < page["text_chars"]
    assert "page 3,400 chars" in text
    assert "[cut:" in text and "token_budget_per_page=2000" in text
    # The tool descriptions say what comes back, not only what goes in.
    async with Client(build_server(service)) as c:
        tools = (await c.list_tools()).tools
    assert "cut" in next(t for t in tools if t.name == "extract").description
    assert "size of the whole page" in next(t for t in tools if t.name == "search").description


@pytest.mark.asyncio
async def test_a_bad_request_is_reported_before_the_missing_planner(service):
    r = await _call(service, "search", query="   ")
    assert r.is_error is True
    assert "query" in r.content[0].text and "plan" not in r.content[0].text.split("`")[0]
    assert service.engine.calls == []
