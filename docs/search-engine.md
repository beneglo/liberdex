# The search page

```bash
uvx --from git+https://github.com/beneglo/liberdex liberdex serve --open
# http://127.0.0.1:8080, and opens it
```

## Settings

The gear opens a drawer with the two model roles, a key field for whichever
provider needs one, and the shared plan cache. It is served only when the server is bound to
loopback, or when `--settings on` says otherwise, because it writes to the
machine running liberdex: model choices to `~/.config/liberdex/models.toml`, key
values to `~/.config/liberdex/env` (mode 0600, never into the TOML). A save
takes effect on the next search; nothing restarts.

The provider list includes the four subscription CLIs, `claude-cli`,
`codex-cli`, `gemini-cli` and `opencode-cli`, each marked *on this machine* or
*not installed*. Pick one and no key is needed. Only the Claude CLI streams; the
others deliver the plan whole, ten to twenty seconds in. The deadline grows to
what they declare, on the page as on the command line, but `deep` gets more
out of them. With no model chosen at all, the home view says so in one line and offers
the drawer rather than running a routes-only search that would look like a thin
result.

For a hosted model, pick `openrouter`, paste the key, and choose a model from
the list the endpoint reports. `ollama`, `lmstudio`, `llamacpp` and `vllm` are
there for local runtimes, and a `[providers.*]` block in `models.toml` adds any
other OpenAI-compatible endpoint; see [models.md](models.md).

## The page

The same server serves a search page at `/`. Open it, type, get ten links: the
shape everyone already knows, over an engine that has no index behind it.

It is plain HTML, CSS and JavaScript with no build step and no npm, served from
the same origin as the API it calls. Nothing on the page reaches a third party.
Archivo is self-hosted, result favicons are fetched by liberdex and served from
`/icon` rather than loaded from the sites you searched, links carry
`rel="noreferrer"`, and the response sets a Content-Security-Policy of
`default-src 'self'` so the browser enforces all of that rather than taking the
claim on trust. Open the network panel on a search: `localhost` is the only
origin listed.

`/opensearch.xml` offers liberdex to the browser, so Chrome and Firefox can make
it the default engine and the address bar becomes the search box.

The page shows what the API knows and an index engine does not. Each row names
its `source` (`llm`, `route:<site>`, `hub`, `expand`, `deep`, `sitesearch`,
`sitemap`, `refine`) and carries a
rule whose lit fraction is that row's `relevance`, so a page the ranker was
unsure about looks unsure. A page that could not be read says so instead of
showing the hostname as a summary, and a search that fetched nothing is labelled
unverified, answer included. The depth toggle is `fast` / `standard` / `deep`,
`site:example.com` in the query becomes an `include_domains` allowlist, and the
answer is on by default, since it is one more LLM call and the toggle turns it
off.

There is no result set to page through, so a page turn is a new round. The pages
already shown go back to the server as `exclude_urls`; they are never fetched or
returned again, and the planner is asked, over them, for what comes next. Every
page costs one planner call the first time and nothing on repeat.

## Deep

`deep` plans twice. After the first wave lands, the planner is handed the
round's report, meaning the pages that came back with their fit, the addresses
that were dead and what has been shown, and is asked for what it missed: the
kind of source the SERP has none of, the specific page on a host that answered
with its front door, the constraint of the query the found pages do not cover.
Its answer is fetched as a second wave and ranked with the first. One more LLM
call, cached per query and page like the first plan, and its wait overlaps the
first wave's tail.

```bash
curl -s localhost:8080/search -d '{"query":"...","depth":"deep"}' -H 'content-type: application/json'
curl -s localhost:8080/search -d '{"query":"...","page":2,"exclude_urls":["https://..."]}' -H 'content-type: application/json'
```

`stats.rounds` is 2 when the second pass ran, `stats.candidates_refine` is how
many pages it named, and a row it found says `source: refine`.

Dark is the default because the brand is; the theme toggle, the depth choice and
the answer toggle persist in `localStorage`. Model choices live on the server
(above), so they follow you across browsers. There are no cookies and no
telemetry.
