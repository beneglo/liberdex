"""`liberdex.client.Client` against the server in-process, no socket."""
from __future__ import annotations

import httpx
import pytest

import liberdex.server as srv
from liberdex.client import Client, ServerError
from test_server import StubEngine


@pytest.fixture()
def lx(monkeypatch):
    monkeypatch.setattr(srv, "_engine", StubEngine())
    monkeypatch.delenv("LIBERDEX_KEYS", raising=False)
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=srv.app),
                             base_url="http://test")
    return Client("http://test", http=http)


@pytest.mark.asyncio
async def test_search_returns_the_servers_own_types(lx):
    reply = await lx.search("q", top_k=3, plan={"candidates": ["https://x.test/a"]})
    assert reply.results[0].url == "https://x.test/a"
    assert reply.results[0].relevance is not None
    _q, kw = srv._engine.calls[-1]
    assert kw["top_k"] == 3 and kw["plan"].candidates[0].url == "https://x.test/a"


@pytest.mark.asyncio
async def test_extract_and_plan(lx):
    ex = await lx.extract(["https://x.test/a", "https://x.test/bad"])
    assert len(ex.results) == 1 and ex.failed[0].status == 404
    pl = await lx.plan("q")
    assert pl.plan.candidates[0]["url"] == "https://d.test/a"


@pytest.mark.asyncio
async def test_stream_yields_the_pipelines_events(lx):
    kinds = []
    async for kind, data in lx.stream("q"):
        kinds.append(kind)
        if kind == "results":
            assert data["results"][0]["url"] == "https://x.test/a"
    assert kinds[:3] == ["plan", "candidate", "page"] and kinds[-1] == "results"


@pytest.mark.asyncio
async def test_a_refusal_is_an_error_with_the_status(lx, monkeypatch):
    monkeypatch.setenv("LIBERDEX_KEYS", "secret")
    with pytest.raises(ServerError) as e:
        await lx.search("q")
    assert e.value.status == 401
    lx.key = "secret"
    assert (await lx.search("q")).results
