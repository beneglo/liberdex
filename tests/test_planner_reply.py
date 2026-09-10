"""`ReplyPlanner`: a plan from any whole-reply completion, and `CLIPlanner` on it."""
from __future__ import annotations

import pytest

from liberdex.llm import Reply
from liberdex.models import BUILTIN
from liberdex.planner.cli import CLIPlanner
from liberdex.planner.prompt import SYSTEM
from liberdex.planner.reply import ReplyPlanner
from liberdex.types import Deadline, Findings

PLAN = ("I reference\nL en\nR wikipedia: handshake\n"
        "U 0.9 https://en.encyclopedia.example/wiki/Handshake :: the article\n")


async def events(planner, method="stream", *args):
    out = []
    async for ev in getattr(planner, method)("tcp handshake", *args, Deadline(8.0)):
        out.append(ev)
    return out


@pytest.mark.asyncio
async def test_a_reply_becomes_candidates_and_a_plan():
    seen = {}

    async def complete(messages, budget):
        seen["messages"] = messages
        seen["budget"] = budget
        return PLAN

    p = ReplyPlanner(complete, name="host")
    evs = await events(p)
    assert p.name == "host"
    assert evs[-1][0] == "done"
    plan = evs[-1][1]
    assert [c.url for c in plan.candidates] == ["https://en.encyclopedia.example/wiki/Handshake"]
    assert plan.routes == ["wikipedia"]
    assert [e[0] for e in evs[:-1]] == ["candidate"]
    assert seen["messages"][0] == {"role": "system", "content": SYSTEM}
    assert "tcp handshake" in seen["messages"][1]["content"]
    assert 0 < seen["budget"] <= 8.0


@pytest.mark.asyncio
async def test_a_failing_completion_is_an_error_then_an_empty_plan():
    async def complete(messages, budget):
        raise RuntimeError("model down")

    evs = await events(ReplyPlanner(complete))
    assert evs[0] == ("error", "RuntimeError: model down")
    assert evs[-1][0] == "done" and evs[-1][1].candidates == []


@pytest.mark.asyncio
async def test_refine_hands_the_findings_to_the_model():
    seen = {}

    async def complete(messages, budget):
        seen["user"] = messages[1]["content"]
        return PLAN

    f = Findings(found=[("t", "https://a.test/one", 0.9, "x")], page=2)
    evs = await events(ReplyPlanner(complete), "refine", f)
    assert evs[-1][0] == "done"
    assert "https://a.test/one" in seen["user"]


@pytest.mark.asyncio
async def test_the_cli_planner_is_a_reply_planner_on_a_chat(monkeypatch):
    calls = []

    async def fake_complete(self, messages, *, response_format=None, budget=None):
        calls.append((self.profile.name, self.model, budget))
        return Reply(text=PLAN)

    monkeypatch.setattr("liberdex.llm.Chat.complete", fake_complete)
    p = CLIPlanner(BUILTIN["codex-cli"], "gpt-5.5")
    assert p.name == "cli" and p.model == "gpt-5.5" and p.min_budget == 15.0
    evs = await events(p)
    assert evs[-1][1].routes == ["wikipedia"]
    assert calls and calls[0][:2] == ("codex-cli", "gpt-5.5")


@pytest.mark.asyncio
async def test_the_cli_planner_reports_a_reply_error(monkeypatch):
    async def fake_complete(self, messages, *, response_format=None, budget=None):
        return Reply(text="", error="exit 1: no subscription")

    monkeypatch.setattr("liberdex.llm.Chat.complete", fake_complete)
    evs = await events(CLIPlanner(BUILTIN["codex-cli"]))
    assert evs[0][0] == "error" and "no subscription" in evs[0][1]
