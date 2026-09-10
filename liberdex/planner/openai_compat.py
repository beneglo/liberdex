"""Planner backed by any OpenAI-compatible chat-completions endpoint.

This is the intended production backend, and in 2026 it reaches nearly
everything: OpenRouter, OpenAI, Groq, Together, vLLM, and the local runtimes
(Ollama, llama.cpp, LM Studio) all speak this wire format. It streams, so the
engine starts fetching the first URL while the rest of the plan is still being
generated.

Everything provider-specific arrives in a `Profile` (see liberdex.models):
attribution headers, a sampler flag, how the endpoint spells "think less".
Nothing in this file names a vendor.
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, AsyncIterator, Optional

import httpx
import orjson

from ..models import BUILTIN, Profile, reasoning_payload
from ..types import Deadline, Findings
from .base import PlanParser
from .prompt import SYSTEM, build_refine_prompt, build_user_prompt

DEFAULT_MODEL = "google/gemini-3.5-flash-lite"
# The planner is a recall task, not a reasoning one, and it is bottlenecked on
# output throughput: the plan is a few hundred tokens and the engine cannot
# finish fetching until the last U line lands. So the default is the fastest
# adequate model, with thinking pinned to the floor, since some endpoints
# cannot switch it off.
DEFAULT_REASONING_EFFORT = "minimal"


def _profile_for(base_url: Optional[str], api_key: Optional[str]) -> Profile:
    """A Profile from a base URL, for callers that pass one instead of a
    profile: a built-in one when the URL matches, else a custom one."""
    url = (
        base_url
        or os.environ.get("LIBERDEX_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or BUILTIN["openrouter"].base_url
    ).rstrip("/")
    for name, p in BUILTIN.items():
        if p.base_url and p.base_url.rstrip("/") == url:
            return replace(p, api_key=api_key or p.api_key)
    return Profile(
        name="custom", base_url=url, api_key=api_key or "",
        api_key_env=("LIBERDEX_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"),
        headers={"HTTP-Referer": "https://liberdex.net", "X-Title": "liberdex"},
    )


class OpenAIPlanner:
    name = "openai-compat"

    def __init__(
        self,
        model: str = "",
        *,
        profile: Optional[Profile] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        few_shot: bool = True,
        n_urls: int = 14,
        refine_urls: int = 8,
        temperature: float = 0.2,
        # Reasoning tokens bill against max_tokens, so a budget sized for the
        # plan alone starves it.
        max_tokens: int = 3000,
        reasoning_effort: Optional[str] = None,
        extra_headers: Optional[dict[str, str]] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.profile = profile or _profile_for(base_url, api_key)
        self.model = model or self.profile.default_model or DEFAULT_MODEL
        self.few_shot = few_shot
        self.n_urls = n_urls
        self.refine_urls = refine_urls
        self.temperature = temperature
        self.max_tokens = max_tokens
        # A planner recalls URLs and has nothing to reason about, so every
        # reasoning token is latency before the first URL is dispatched. The
        # default only applies where the endpoint understands the field at all;
        # a strict OpenAI-compatible server would 400 on an unknown one.
        self.reasoning_effort = (
            reasoning_effort
            or os.environ.get("LIBERDEX_REASONING_EFFORT")
            or (DEFAULT_REASONING_EFFORT
                if self.profile.reasoning_style != "none" else None)
        )
        self.extra_headers = extra_headers or {}
        self._client = client
        self._owned_client = client is None

    # Views of the profile, for callers that configured the planner by URL.
    @property
    def base_url(self) -> str:
        return self.profile.base_url

    @property
    def api_key(self) -> str:
        return self.profile.key()

    async def aclose(self) -> None:
        if self._owned_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                http2=True,
                timeout=httpx.Timeout(connect=2.0, read=30.0, write=5.0, pool=5.0),
            )
        return self._client

    def _payload(self, prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": True,
            "temperature": self.temperature,
            self.profile.max_tokens_param: self.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
        }
        payload.update(reasoning_payload(self.profile.reasoning_style,
                                         self.reasoning_effort))
        payload.update(self.profile.extra_body)
        return payload

    async def stream(
        self, query: str, deadline: Deadline
    ) -> AsyncIterator[tuple[str, object]]:
        prompt = build_user_prompt(query, few_shot=self.few_shot, n_urls=self.n_urls)
        async for ev in self._run(query, prompt, deadline):
            yield ev

    async def refine(
        self, query: str, findings: Findings, deadline: Deadline
    ) -> AsyncIterator[tuple[str, object]]:
        """The second pass: the first round's report in place of the few-shot."""
        prompt = build_refine_prompt(query, findings, n_urls=self.refine_urls)
        async for ev in self._run(query, prompt, deadline):
            yield ev

    async def _run(
        self, query: str, prompt: str, deadline: Deadline
    ) -> AsyncIterator[tuple[str, object]]:
        parser = PlanParser(query)
        headers = self.profile.merged_headers(self.extra_headers)
        payload = self._payload(prompt)
        try:
            async with self._http().stream(
                "POST", f"{self.profile.base_url.rstrip('/')}/chat/completions",
                headers=headers, json=payload,
                timeout=httpx.Timeout(connect=2.0, read=max(1.0, deadline.remaining),
                                      write=5.0, pool=5.0),
            ) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    yield ("error", f"{resp.status_code}: {resp.text[:200]}")
                    yield ("done", parser.plan)
                    return
                async for line in resp.aiter_lines():
                    if deadline.expired():
                        break
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = orjson.loads(data)
                    except Exception:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0].get("delta") or {}).get("content") or ""
                    if not delta:
                        continue
                    for cand in parser.feed(delta):
                        yield ("candidate", cand)
                    yield ("meta", parser.plan)
        except Exception as e:
            yield ("error", f"{type(e).__name__}: {e}")
        for cand in parser.finish():
            yield ("candidate", cand)
        yield ("done", parser.plan)
