"""liberdex as a hermes-agent web search and extract backend.

    hermes plugins enable liberdex
    hermes config set web.backend liberdex

hermes then serves its `web_search` and `web_extract` tools from liberdex,
and the plan behind every search is written by hermes's own model, borrowed
through `ctx.llm`: no key, no second provider. Discovered through the
`hermes_agent.plugins` entry point in liberdex's own package; `liberdex install
hermes` does the three steps above.

This file is part of liberdex (AGPL-3.0-or-later). It is loaded by the user's
hermes process because the user enabled it; hermes ships none of it.
"""
from __future__ import annotations


def register(ctx) -> None:
    from .provider import LiberdexProvider
    ctx.register_web_search_provider(LiberdexProvider(ctx))
