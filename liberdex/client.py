"""A client for a liberdex server, for programs that would rather not run one.

    from liberdex.client import Client

    async with Client("http://127.0.0.1:8080", key="...") as lx:
        reply = await lx.search("how does the TCP three way handshake work", top_k=5)
        for r in reply.results:
            print(r.relevance, r.url, r.passages[0])

The replies are the server's own pydantic models (`api.SearchReply`,
`api.ExtractReply`, `api.PlanReply`), so a program that moves from calling a
server to importing the engine keeps its types. Calling a liberdex server over
HTTP does not make the caller a derivative work; see the licence.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Optional, Sequence

import httpx
import orjson

from .api import ExtractReply, PlanReply, SearchReply


class ServerError(RuntimeError):
    def __init__(self, status: int, detail: Any) -> None:
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


class Client:
    def __init__(
        self,
        url: str = "http://127.0.0.1:8080",
        key: Optional[str] = None,
        *,
        timeout: float = 90.0,
        http: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.key = key or ""
        self._http = http
        self._owned = http is None
        self._timeout = timeout

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self.url,
                timeout=httpx.Timeout(connect=5.0, read=self._timeout,
                                      write=15.0, pool=15.0))
        return self._http

    def _headers(self) -> dict[str, str]:
        h = {"content-type": "application/json"}
        if self.key:
            h["authorization"] = f"Bearer {self.key}"
        return h

    async def aclose(self) -> None:
        if self._owned and self._http is not None:
            await self._http.aclose()
            self._http = None

    async def __aenter__(self) -> "Client":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def _post(self, path: str, body: dict[str, Any]) -> Any:
        r = await self._client().post(self.url + path, content=orjson.dumps(body),
                                      headers=self._headers())
        if r.status_code != 200:
            try:
                detail = r.json().get("detail")
            except Exception:
                detail = r.text[:300]
            raise ServerError(r.status_code, detail)
        return r.json()

    async def search(self, query: str, **kwargs: Any) -> SearchReply:
        """`POST /search` with liberdex's own fields: `top_k`, `depth`,
        `answer`, `plan`, `include_domains`, `freshness`, `token_budget`, ..."""
        return SearchReply.model_validate(await self._post("/search", {"query": query, **kwargs}))

    async def extract(self, urls: Sequence[str] | str, **kwargs: Any) -> ExtractReply:
        return ExtractReply.model_validate(await self._post("/extract", {"urls": urls, **kwargs}))

    async def plan(self, query: str, **kwargs: Any) -> PlanReply:
        return PlanReply.model_validate(await self._post("/plan", {"query": query, **kwargs}))

    async def stream(self, query: str, **kwargs: Any) -> AsyncIterator[tuple[str, Any]]:
        """`POST /search/stream`: yields ("candidate"|"plan"|"page"|"results"|
        "error", data) as the pipeline runs. The `results` event's data is the
        full reply."""
        async with self._client().stream(
            "POST", self.url + "/search/stream",
            content=orjson.dumps({"query": query, **kwargs}), headers=self._headers(),
        ) as r:
            if r.status_code != 200:
                await r.aread()
                raise ServerError(r.status_code, r.text[:300])
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    frame = orjson.loads(data)
                except Exception:
                    continue
                yield frame.get("event", ""), frame.get("data")


__all__ = ["Client", "ServerError"]
