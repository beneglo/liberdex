"""Planner backed by the local `claude` CLI.

The no-key path: a subscription already on the machine, reached through a
subprocess. It costs ~3s before the first token, which the engine partly hides
by running speculative site-native routes concurrently. `OpenAIPlanner` is the
intended hosted backend; this one is what `auto` falls back to when no key is
in the environment.
"""
from __future__ import annotations

import asyncio
import os
import shutil
from typing import AsyncIterator, Optional

import orjson

from ..types import Deadline, Findings
from .base import PlanParser
from .prompt import SYSTEM, build_refine_prompt, build_user_prompt


class ClaudeCLIPlanner:
    name = "claude-cli"

    # A local `claude` process spends 3-4s before its first token and 9-12s on
    # a whole plan. It streams, so the tier's plan grace covers the wait for
    # the first URL (api.PLAN_GRACE) and the search itself needs no floor. The
    # plan cache's detached reader does: this is what a whole plan takes, so
    # the store gets all of it however early the search stopped reading.
    finish_floor = 15.0

    def __init__(
        self,
        model: str = "opus",
        *,
        binary: Optional[str] = None,
        cwd: Optional[str] = None,
        few_shot: bool = True,
        effort: Optional[str] = "low",
        n_urls: int = 14,
        refine_urls: int = 8,
    ) -> None:
        self.model = model
        self.binary = binary or shutil.which("claude") or "claude"
        self.cwd = cwd or os.path.expanduser("~")
        self.few_shot = few_shot
        self.effort = effort
        self.n_urls = n_urls
        # Fewer than the first pass asks for: the second pass is filling gaps
        # in a plan that already exists, and every line is paid for.
        self.refine_urls = refine_urls

    def _argv(self, prompt: str) -> list[str]:
        agents = orjson.dumps({
            "liberdex-recall": {
                "description": "index-free search URL recall",
                "prompt": SYSTEM,
                # No tools: the planner answers from parametric memory only,
                # and an empty tool list also cuts input tokens.
                "tools": [],
            }
        }).decode()
        return [
            self.binary, "-p", prompt,
            "--model", self.model,
            "--system-prompt", SYSTEM,
            "--agents", agents,
            "--agent", "liberdex-recall",
            "--output-format", "stream-json",
            "--include-partial-messages",
            "--no-session-persistence",
            "--strict-mcp-config",
            # The agent already replaces the system prompt, but user settings,
            # hooks and CLAUDE.md load regardless of it. A plan is a line
            # protocol; anything the operator has told their editor to do is
            # noise in it at best.
            "--restricted",
            "--verbose",
        ] + (["--effort", self.effort] if self.effort else [])

    async def stream(
        self, query: str, deadline: Deadline
    ) -> AsyncIterator[tuple[str, object]]:
        prompt = build_user_prompt(query, few_shot=self.few_shot, n_urls=self.n_urls)
        async for ev in self._run(query, prompt, deadline):
            yield ev

    async def refine(
        self, query: str, findings: Findings, deadline: Deadline
    ) -> AsyncIterator[tuple[str, object]]:
        """The second pass. Same process, same protocol, a shorter prompt: no
        few-shot, and the first round's report in place of it."""
        prompt = build_refine_prompt(query, findings, n_urls=self.refine_urls)
        async for ev in self._run(query, prompt, deadline):
            yield ev

    async def _run(
        self, query: str, prompt: str, deadline: Deadline
    ) -> AsyncIterator[tuple[str, object]]:
        parser = PlanParser(query)
        proc = await asyncio.create_subprocess_exec(
            *self._argv(prompt),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            stdin=asyncio.subprocess.PIPE,
            cwd=self.cwd,
        )
        assert proc.stdout is not None
        if proc.stdin is not None:
            # The CLI waits 3s for stdin unless the pipe is closed outright.
            proc.stdin.close()
        try:
            while True:
                remaining = deadline.remaining
                if remaining <= 0:
                    break
                try:
                    raw = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if not raw:
                    break
                try:
                    evt = orjson.loads(raw)
                except Exception:
                    continue
                text = _delta_text(evt)
                if not text:
                    continue
                for cand in parser.feed(text):
                    yield ("candidate", cand)
                yield ("meta", parser.plan)
            for cand in parser.finish():
                yield ("candidate", cand)
            yield ("done", parser.plan)
        finally:
            # The consumer may stop early (the engine caps how many URLs it
            # reads), which runs this block inside the generator's aclose().
            # Awaiting there raises, so the process is killed synchronously and
            # reaped by a detached task.
            _terminate(proc)


def _terminate(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        try:
            proc.kill()
        except (ProcessLookupError, RuntimeError):
            pass
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = loop.create_task(_reap(proc))
    _PENDING_REAPS.add(task)
    task.add_done_callback(_PENDING_REAPS.discard)


_PENDING_REAPS: set = set()


async def _reap(proc: asyncio.subprocess.Process) -> None:
    try:
        await asyncio.wait_for(proc.wait(), timeout=2.0)
    except (asyncio.TimeoutError, ProcessLookupError, asyncio.CancelledError):
        pass


def _delta_text(evt: dict) -> str:
    if evt.get("type") != "stream_event":
        return ""
    ev = evt.get("event") or {}
    if ev.get("type") != "content_block_delta":
        return ""
    delta = ev.get("delta") or {}
    if delta.get("type") != "text_delta":
        return ""
    return delta.get("text") or ""
