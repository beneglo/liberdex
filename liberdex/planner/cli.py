"""Planner on a subscription CLI: codex, gemini, opencode, or one of your own.

Same prompt as `OpenAIPlanner`, one subprocess, no streaming: the reply lands
whole, so the first URL arrives when the last one does. That is a plan ten to
twenty seconds after the query, which is fine at `deep` and is a routes-only
search at `fast`. `min_budget` says so to the engine; the search page and the
CLI say so to the user.

Everything about *which* command is in the `Profile` (see models.cli_argv);
nothing in this file names a vendor. The mechanics are `ReplyPlanner`'s.
"""
from __future__ import annotations

from ..llm import Chat
from ..models import Profile
from .reply import ReplyPlanner


class CLIPlanner(ReplyPlanner):
    name = "cli"

    def __init__(
        self,
        profile: Profile,
        model: str = "",
        *,
        few_shot: bool = True,
        n_urls: int = 14,
        refine_urls: int = 8,
    ) -> None:
        super().__init__(self._complete, name="cli", few_shot=few_shot,
                         n_urls=n_urls, refine_urls=refine_urls)
        self.profile = profile
        self.model = model or profile.default_model

    async def _complete(self, messages: list[dict[str, str]], budget: float) -> str:
        chat = Chat(self.profile, self.model, max_retries=1, timeout=budget)
        reply = await chat.complete(messages, budget=budget)
        if reply.error:
            raise RuntimeError(reply.error)
        return reply.text
