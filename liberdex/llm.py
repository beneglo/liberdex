"""One non-streaming chat call against a `Profile`.

The planner streams and lives on the search deadline; this is the other shape.
Answer synthesis and schema extraction each want one complete reply, off the
critical path, with retries. Different tradeoffs, so a different client rather
than a flag on the streaming one.

Everything provider-specific comes from the Profile: `headers`, `extra_body`,
which spelling of `max_tokens` the endpoint accepts, and how it expresses a
thinking budget.
"""
from __future__ import annotations

import asyncio
import os
import random
import shutil
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import orjson

from .models import Profile, reasoning_payload

RETRY_STATUS = (408, 409, 425, 429, 500, 502, 503, 504)


@dataclass(slots=True)
class Reply:
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    model: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


@dataclass(slots=True)
class Usage:
    """What the LLM calls on one request cost, in tokens and calls."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, r: Reply) -> None:
        self.calls += 1
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens

    def as_dict(self) -> dict[str, int]:
        return {"llm_calls": self.calls,
                "llm_prompt_tokens": self.prompt_tokens,
                "llm_completion_tokens": self.completion_tokens}


class Chat:
    """A profile plus a model id, callable once per request."""

    def __init__(
        self,
        profile: Profile,
        model: str = "",
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        max_retries: int = 3,
        reasoning_effort: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.profile = profile
        self.model = model or profile.default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        self.usage = Usage()
        self._client = client
        self._owned = client is None

    async def aclose(self) -> None:
        if self._owned and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                http2=True,
                timeout=httpx.Timeout(connect=5.0, read=self.timeout,
                                      write=15.0, pool=15.0),
                limits=httpx.Limits(max_connections=16),
            )
        return self._client

    def _payload(self, messages: list[dict[str, str]],
                 response_format: Optional[dict]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            self.profile.max_tokens_param: self.max_tokens,
        }
        payload.update(reasoning_payload(self.profile.reasoning_style,
                                         self.reasoning_effort))
        payload.update(self.profile.extra_body)
        if response_format:
            payload["response_format"] = response_format
        return payload

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: Optional[dict] = None,
        budget: Optional[float] = None,
    ) -> Reply:
        """One reply. Never raises: a failed call returns a Reply with `error`.

        The caller is a search response that already has results to return, so
        a dead answer model should cost the answer, not the search.
        """
        if self.profile.kind == "claude_cli":
            return await self._claude_cli(messages, budget=budget)
        if self.profile.kind == "cli":
            return await self._cli(messages, budget=budget)
        if not self.model:
            return Reply(error="no model configured")
        key = self.profile.key()
        if not key:
            return Reply(error=f"no API key for profile {self.profile.name!r} "
                               f"(set {' or '.join(self.profile.api_key_env) or 'a key'})")

        url = f"{self.profile.base_url.rstrip('/')}/chat/completions"
        headers = self.profile.merged_headers()
        payload = self._payload(messages, response_format)
        deadline = None if budget is None else (asyncio.get_running_loop().time() + budget)
        last = ""
        for attempt in range(self.max_retries):
            if deadline is not None and asyncio.get_running_loop().time() >= deadline:
                return Reply(error=f"answer budget exhausted; last error: {last}")
            try:
                r = await self._http().post(url, json=payload, headers=headers)
            except httpx.HTTPError as e:
                last = f"{type(e).__name__}: {e}"
            else:
                if r.status_code == 200:
                    reply = _parse(r.content, self.model)
                    if reply.ok:
                        self.usage.add(reply)
                        return reply
                    last = reply.error
                elif r.status_code in RETRY_STATUS:
                    last = f"{r.status_code}: {r.text[:200]}"
                else:
                    return Reply(error=f"{r.status_code}: {r.text[:300]}")
            if attempt + 1 < self.max_retries:
                await asyncio.sleep(min(8.0, 2 ** attempt) * (0.5 + random.random()))
        return Reply(error=f"failed after {self.max_retries} tries: {last}")

    async def _claude_cli(self, messages: list[dict[str, str]], *,
                          budget: Optional[float]) -> Reply:
        """The `claude` binary, one shot. No API key, ~1.2s of startup."""
        binary = shutil.which("claude") or "claude"
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        prompt = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
        # --restricted ignores the user's own CLAUDE.md, settings, output styles
        # and hooks. Without it the operator's editor instructions steer an
        # answer this API presents as grounded in the pages liberdex fetched.
        argv = [binary, "-p", prompt, "--model", self.model or "sonnet",
                "--output-format", "text", "--no-session-persistence",
                "--strict-mcp-config", "--restricted"]
        if system:
            argv += ["--system-prompt", system]
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL, stdin=asyncio.subprocess.DEVNULL,
                cwd=os.path.expanduser("~"),
            )
        except (FileNotFoundError, OSError) as e:
            return Reply(error=f"claude CLI not runnable: {e}")
        try:
            out, _ = await asyncio.wait_for(proc.communicate(),
                                            timeout=budget or self.timeout)
        except asyncio.TimeoutError:
            if proc.returncode is None:
                proc.kill()
            return Reply(error="claude CLI timed out")
        reply = Reply(text=out.decode("utf-8", "replace").strip(),
                      model=self.model, finish_reason="stop")
        self.usage.add(reply)
        return reply

    async def _cli(self, messages: list[dict[str, str]], *,
                   budget: Optional[float]) -> Reply:
        """Any CLI that reads a prompt and prints a reply. See models.cli_argv."""
        from .models import cli_argv
        argv = cli_argv(self.profile, self.model)
        if not argv:
            return Reply(error=f"profile {self.profile.name!r} has no argv")
        binary = shutil.which(argv[0])
        if binary is None:
            return Reply(error=f"{argv[0]} not on PATH")
        argv[0] = binary
        # System and user turns become one prompt: none of these CLIs takes a
        # system prompt of its own on every invocation, and the recall prompt
        # is written to survive that.
        prompt = "\n\n".join(m["content"] for m in messages if m.get("content"))
        stdin_prompt = "" in argv
        if stdin_prompt:
            argv = [prompt if a == "" else a for a in argv]
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                stdin=asyncio.subprocess.DEVNULL if stdin_prompt else asyncio.subprocess.PIPE,
                cwd=os.path.expanduser("~"),
            )
        except (FileNotFoundError, OSError) as e:
            return Reply(error=f"{argv[0]} not runnable: {e}")
        try:
            out, _ = await asyncio.wait_for(
                proc.communicate(None if stdin_prompt else prompt.encode("utf-8")),
                timeout=budget or self.timeout)
        except asyncio.TimeoutError:
            if proc.returncode is None:
                proc.kill()
            return Reply(error=f"{self.profile.name} timed out")
        text = out.decode("utf-8", "replace").strip()
        if proc.returncode not in (0, None) and not text:
            return Reply(error=f"{self.profile.name} exited {proc.returncode}")
        reply = Reply(text=text, model=self.model, finish_reason="stop")
        self.usage.add(reply)
        return reply

    async def probe(self) -> tuple[bool, str]:
        """Is this profile reachable? Used by `liberdex models check`."""
        if self.profile.kind == "claude_cli":
            return (shutil.which("claude") is not None,
                    "claude binary on PATH" if shutil.which("claude") else "not on PATH")
        if self.profile.kind == "cli":
            head = self.profile.argv[0] if self.profile.argv else ""
            if not head:
                return False, "no argv configured"
            found = shutil.which(head) is not None
            return found, (f"{head} on PATH; no streaming, plans take ~10-20s"
                           if found else f"{head} not on PATH")
        if not self.profile.key():
            return False, "no API key"
        try:
            r = await self._http().get(
                f"{self.profile.base_url.rstrip('/')}/models",
                headers=self.profile.merged_headers(), timeout=8.0)
        except httpx.HTTPError as e:
            return False, f"{type(e).__name__}: {e}"
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        try:
            n = len((r.json().get("data") or []))
        except Exception:
            n = 0
        return True, f"{n} models" if n else "reachable"


def _parse(body: bytes, model: str) -> Reply:
    try:
        data = orjson.loads(body)
    except Exception as e:
        return Reply(error=f"unparseable response: {e}")
    if not data.get("choices"):
        return Reply(error=str(data.get("error") or data)[:300])
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    return Reply(
        text=(choice.get("message") or {}).get("content") or "",
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        finish_reason=str(choice.get("finish_reason") or ""),
        model=str(data.get("model") or model),
    )
