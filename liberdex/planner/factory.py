"""One place that turns "which planner" into a planner object.

The CLI, the server and the MCP server answer the same question from the same
inputs: a backend name, a `profile/model` spec, the environment.
"""
from __future__ import annotations

from typing import Any, Optional

from ..models import Profile, auto_planner, resolve_role


def make_planner(
    backend: str = "auto",
    spec: Optional[str] = None,
    *,
    n_urls: int = 14,
    effort: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    base_url: Optional[str] = None,
    cache: bool = True,
    store: Any = None,
) -> Any:
    """A planner, or None for `backend == "none"`.

    `backend` is `auto`, `claude`, `openai`, `replay` or `none`. `auto` picks
    from the machine: the configured spec, else a key in the environment, else
    the claude binary. `spec` is `profile/model` and outranks the backend
    name: `codex-cli/gpt-5.5` on any backend means the codex CLI.

    Raises `LookupError` when nothing can plan. The engine would otherwise run
    routes-only and return a thin SERP that looks like a working search.
    """
    backend = (backend or "auto").lower()
    if backend == "none":
        return None
    if backend == "replay":
        from ..cache import ReplayPlanner
        return ReplayPlanner(model=spec or "opus")

    profile: Optional[Profile]
    if spec:
        profile, model = resolve_role("planner", spec)
    elif backend == "claude":
        profile, model = resolve_role("planner", "claude-cli")
    elif backend == "openai":
        profile, model = resolve_role("planner")
    else:
        profile, model, why = auto_planner()
        if profile is None:
            raise LookupError(why)

    if profile.kind == "claude_cli":
        from .claude_cli import ClaudeCLIPlanner
        inner = ClaudeCLIPlanner(model=model or "opus", effort=effort or None,
                                 n_urls=n_urls)
    elif profile.kind == "cli":
        from .cli import CLIPlanner
        inner = CLIPlanner(profile, model, n_urls=n_urls)
    else:
        from .openai_compat import OpenAIPlanner
        inner = OpenAIPlanner(model=model, profile=profile, base_url=base_url,
                              n_urls=n_urls, reasoning_effort=reasoning_effort or None)
    from ..cache import CachedPlanner, NoStore
    # With the local cache off the wrapper stays, over a store that never
    # answers: it is also what asks the shared cache before the planner runs.
    if not cache:
        store = NoStore()
    return CachedPlanner(inner, store=store, model=getattr(inner, "model", ""))


def answer_default(planner: Any) -> Optional[str]:
    """The answer model to use when the caller named none.

    The answer role follows the planner by configuration, but a planner that
    is a local process has no key the answer could inherit: on the claude CLI
    the answer runs on the same binary, and on any other CLI the same command.
    """
    inner = getattr(planner, "inner", planner)
    profile = getattr(inner, "profile", None)
    if profile is None:
        if type(inner).__name__ == "ClaudeCLIPlanner":
            return "claude-cli/sonnet"
        return None
    if profile.kind == "cli":
        return f"{profile.name}/{inner.model}" if inner.model else profile.name
    return None
