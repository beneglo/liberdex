"""Planner on a whole-reply completion: any callable that returns the text.

`OpenAIPlanner` streams, so the engine fetches the first URL while the rest of
the plan is still being written. Not every model reaches liberdex through a
stream: a subscription CLI answers as one block, and a host plugin (hermes,
say) lends liberdex its own model through a `complete(messages)` call. This is
the planner for those. The reply lands whole, so the first URL arrives when
the last one does, and `min_budget` tells the engine to wait that long rather
than cut the plan.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Awaitable, Callable

from ..types import Deadline, Findings
from .base import PlanParser
from .prompt import SYSTEM, build_refine_prompt, build_user_prompt

# (messages, budget in seconds) -> the reply text. Raises when it cannot.
Complete = Callable[[list[dict[str, str]], float], Awaitable[str]]


class ReplyPlanner:
    name = "reply"
    # What a whole-reply model needs before the plan can exist at all. The
    # engine grows the deadline by this much rather than truncating the plan.
    min_budget = 15.0

    def __init__(
        self,
        complete: Complete,
        *,
        name: str = "reply",
        few_shot: bool = True,
        n_urls: int = 14,
        refine_urls: int = 8,
        min_budget: float = 15.0,
    ) -> None:
        self.complete = complete
        self.name = name
        self.few_shot = few_shot
        self.n_urls = n_urls
        self.refine_urls = refine_urls
        self.min_budget = min_budget

    async def aclose(self) -> None:
        return None

    async def stream(
        self, query: str, deadline: Deadline
    ) -> AsyncIterator[tuple[str, object]]:
        prompt = build_user_prompt(query, few_shot=self.few_shot, n_urls=self.n_urls)
        async for ev in self._run(query, prompt, deadline):
            yield ev

    async def refine(
        self, query: str, findings: Findings, deadline: Deadline
    ) -> AsyncIterator[tuple[str, object]]:
        prompt = build_refine_prompt(query, findings, n_urls=self.refine_urls)
        async for ev in self._run(query, prompt, deadline):
            yield ev

    async def _run(
        self, query: str, prompt: str, deadline: Deadline
    ) -> AsyncIterator[tuple[str, object]]:
        parser = PlanParser(query)
        budget = max(1.0, deadline.remaining)
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt}]
        try:
            text = await asyncio.wait_for(self.complete(messages, budget), budget)
        except asyncio.TimeoutError:
            yield ("error", f"no reply within {budget:.0f}s")
        except Exception as e:  # the caller's model, the caller's failure modes
            yield ("error", f"{type(e).__name__}: {e}")
        else:
            for cand in parser.feed(text + "\n"):
                yield ("candidate", cand)
        for cand in parser.finish():
            yield ("candidate", cand)
        yield ("done", parser.plan)
