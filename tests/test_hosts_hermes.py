"""The hermes-agent provider: hermes's shapes in, hermes's shapes out, the plan
written by the model hermes lends."""
from __future__ import annotations

import pytest

from liberdex.hosts.hermes import register
from liberdex.hosts.hermes.provider import LiberdexProvider
from test_mcp import McpStub

PLAN = ("I reference\nR wikipedia: handshake\n"
        "U 0.9 https://en.encyclopedia.example/wiki/Handshake :: the article\n")


class _Result:
    def __init__(self, text):
        self.text = text


class _Llm:
    def __init__(self, text=PLAN, fail=False):
        self.text, self.fail, self.calls = text, fail, []

    def complete(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise RuntimeError("no credits")
        return _Result(self.text)


class _Ctx:
    def __init__(self, llm=None):
        self.llm = llm or _Llm()
        self.providers = []

    def register_web_search_provider(self, p):
        self.providers.append(p)


class _Consuming(McpStub):
    """An engine that runs the planner it is handed, like the real one."""

    async def search(self, query, **kw):
        planner = kw["planner"]
        from liberdex.types import Deadline
        errors = []
        async for kind, payload in planner.stream(query, Deadline(5.0)):
            if kind == "error":
                errors.append(payload)
        resp = await super().search(query, **kw)
        if errors:
            resp.results = []
            resp.stats["planner_error"] = errors[0]
        return resp


@pytest.fixture()
def provider():
    ps = []

    def make(ctx=None, engine=None):
        p = LiberdexProvider(ctx or _Ctx(), engine_factory=lambda: engine or McpStub())
        ps.append(p)
        return p

    yield make
    for p in ps:
        p.close()


def test_register_hands_hermes_the_provider():
    ctx = _Ctx()
    register(ctx)
    (p,) = ctx.providers
    assert p.name == "liberdex" and p.is_available() and p.supports_extract()
    assert p.is_keyless_available()
    assert p.get_setup_schema()["env_vars"] == []


def test_search_returns_hermes_shape_with_passages_as_description(provider):
    ctx = _Ctx()
    stub = McpStub()
    p = provider(ctx, stub)
    out = p.search("tcp handshake", limit=3)
    assert out["success"] is True
    row = out["data"]["web"][0]
    assert row["url"] == "https://x.test/a" and row["title"] == "A"
    assert row["position"] == 1 and "alpha" in row["description"]
    assert row["relevance"] is not None and row["source"] == "llm"
    query, kw = stub.calls[-1]
    assert query == "tcp handshake" and kw["top_k"] == 3
    assert kw["planner"] is p.planner() and kw["budget"] >= 15.0


def test_the_plan_is_written_by_hermes_model(provider):
    ctx = _Ctx()
    p = provider(ctx, _Consuming())
    out = p.search("tcp handshake")
    assert out["success"] is True
    (call,) = ctx.llm.calls
    assert call["purpose"] == "liberdex plan"
    assert "tcp handshake" in call["messages"][1]["content"]


def test_no_model_anywhere_is_a_failure_that_names_both(provider, monkeypatch):
    monkeypatch.setattr("liberdex.models.auto_planner", lambda: (None, "", "no planner here"))
    p = provider(_Ctx(_Llm(fail=True)), _Consuming())
    out = p.search("tcp handshake")
    assert out["success"] is False
    assert "no credits" in out["error"] and "no planner here" in out["error"]
    assert "liberdex install --model" in out["error"]


@pytest.mark.asyncio
async def test_extract_keeps_order_and_reports_failures(provider):
    p = provider()
    rows = await p.extract(["https://x.test/a", "https://x.test/bad"], format="markdown")
    assert [r["url"] for r in rows] == ["https://x.test/a", "https://x.test/bad"]
    assert rows[0]["content"] and "error" not in rows[0]
    assert rows[0]["metadata"]["text_chars"] > 0 and "raw_content" not in rows[0]
    assert rows[1]["content"] == "" and rows[1]["error"]
    _name, kw = p._engine.calls[-1]
    assert kw["markdown"] is True


@pytest.mark.asyncio
async def test_extract_honours_max_chars_and_include_raw(provider):
    p = provider()
    rows = await p.extract([{"url": "https://x.test/a"}], max_chars=3, include_raw=True)
    assert len(rows[0]["content"]) == 3
    assert len(rows[0]["raw_content"]) > 3


def test_an_engine_that_blows_up_is_an_error_not_an_exception(provider):
    class Boom(McpStub):
        async def search(self, query, **kw):
            raise ValueError("bad day")

    out = provider(engine=Boom()).search("x")
    assert out["success"] is False and "bad day" in out["error"]
