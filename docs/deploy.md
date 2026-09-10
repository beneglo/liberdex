# Running liberdex for other programs

The server is one process: the search page at `/`, the JSON API, streaming,
`/extract`, `/plan`, and MCP at `/mcp`.

```bash
uvx --from git+https://github.com/beneglo/liberdex liberdex serve   # http://127.0.0.1:8080
docker run -p 8080:8080 -e OPENROUTER_API_KEY=… ghcr.io/beneglo/liberdex
docker compose up                                    # docker-compose.yml in the repo
```

The container has the ranking models baked in, so `/health` is green seconds
after start. Mount `/home/liberdex/.cache/liberdex` to keep the plan cache,
the site-search atlas and the sitemap cache between runs.

## Beyond localhost

Everything below is off by default, so a local server is exactly what it was.

| setting | flag | what it does |
| --- | --- | --- |
| `LIBERDEX_KEYS` | `--keys k1,k2` | callers must send `Authorization: Bearer <key>` (or `x-api-key`) to `/search*`, `/extract`, `/plan`, `/mcp`. The page, `/health` and the icon proxy stay open; the page asks for the key once and keeps it in `localStorage`. |
| `LIBERDEX_CORS` | `--cors https://app.example` | the origins a browser may call the API from. Exactly those; no wildcard unless you write `*`. |
| `LIBERDEX_MAX_INFLIGHT` | | searches running at once, default 8. Past it, `503` with `Retry-After: 2` instead of a queue that grows. |
| `LIBERDEX_SETTINGS` | `--settings on\|off` | the page's settings drawer, which writes model choices and keys to the server's disk. Default: on for a loopback bind, off otherwise. |

`liberdex serve --host 0.0.0.0` with no `--keys` prints a warning and serves
anyway; it is your port.

The fetcher makes outbound requests on behalf of whoever calls the API. A
public liberdex is a public fetcher; read [Responsible use](../README.md#responsible-use)
before you open one.

## From Python, without the engine

```python
from liberdex.client import Client

async with Client("https://search.example", key="k1") as lx:
    reply = await lx.search("how does the TCP three way handshake work", top_k=5,
                            answer="extract")
    print(reply.answer)
    for r in reply.results:
        print(r.relevance, r.url, r.passages[0][:80])

    async for event, data in lx.stream("rust borrow checker lifetimes"):
        ...                      # "plan", "candidate", "page", then "results"

    pages = await lx.extract(["https://example.com/a", "https://example.com/b"])
```

The replies are the same pydantic models the server uses (`SearchReply`,
`ExtractReply`, `PlanReply`), so a program that later imports the engine keeps
its types. Calling a liberdex server over HTTP does not make the caller a
derivative work; importing `liberdex` does. See the licence.

## As a connector

The MCP endpoint speaks streamable HTTP at `https://search.example/mcp`. Add it
as a custom connector in claude.ai or Claude Desktop, or as an MCP server in
ChatGPT's developer settings, with the key as a bearer token. The two tools are
the same `search` and `extract` the agent CLIs get, and the host model writes
the plan the same way; see [agents.md](agents.md).

## Health and version

`GET /health` answers `{"ok": true, "version": "..."}` and, when nothing on the
machine can plan, `"planner_error"` saying why. `"commons"` is the shared
plan cache: whether this machine holds a key, whether it is muted after a
failure, and its hit and miss counts. `GET /docs` is the OpenAPI page.
