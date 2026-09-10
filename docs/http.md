# The HTTP API

`liberdex serve` (or the container) serves this on port 8080. Every route below is also reachable through `liberdex.client.Client` from Python, and the same two tools are served as MCP at `/mcp`; see [deploy.md](deploy.md).

| endpoint | what it does |
| --- | --- |
| `POST /search` | the SERP. Also `GET /search?q=...` for a browser or a one-line curl |
| `POST /search/stream` | the same search, reporting as the pipeline runs |
| `POST /extract` | fetch and read pages you already have URLs for, up to 20 |
| `POST /plan` | *where does this live on the web?* One LLM call, nothing fetched |
| `GET /health` | liveness |
| `GET /` | the search page. `/?q=...` runs a search |
| `GET /icon?u=` | a result's favicon, fetched by liberdex so your browser never asks the site for it |
| `GET /opensearch.xml` | what a browser reads to offer liberdex as its search engine |

```bash
curl -s localhost:8080/search -H 'content-type: application/json' -d '{
  "query": "create index concurrently locking",
  "top_k": 10, "depth": "fast",
  "include_domains": ["db.example"],
  "format": "markdown", "token_budget": 4000,
  "answer": "extract"
}' | jq
```

## The response

The response speaks liberdex's own vocabulary, because those words are what the
engine actually does. A query produces a plan; the plan names candidates,
routes and hubs; those become pages; the ranker cuts each page into passages.

```jsonc
{
  "request_id": "…", "query": "…", "intent": "code", "lang": "en",
  "answer": "…",                        // when `answer` is set
  "results": [{
    "url": "…", "title": "…", "site": "db.example",
    "score": 2.45,                      // fusion; orders the SERP, unbounded
    "relevance": 0.94,                  // absolute, 0..1; threshold on this
    "passages": ["…", "…", "…"],        // query-selected windows, in page order
    "passage_scores": [0.83, 0.5, 0.4], // the fraction of query terms each covers
    "text": "…",                        // the whole page, when include_text
    "text_chars": 44800,                // the page as read, even when text was cut or not shipped
    "published": "2026-08-13", "favicon": "…", "images": [],
    "source": "llm",                    // see below
    "status": 200
  }],
  "plan": null,                         // when include_plan
  "timings": {…}, "stats": {…}
}
```

`source` says where the row came from: `llm` for the planner's parametric
memory, `route:<site>` for a site's own search, `hub` for a page the plan named
as an index, `expand` for a link followed off one, `deep` for a link that
described the query better than the page it sat on, `sitesearch` and `sitemap`
for the two recoveries attempted after a guessed address failed, `refine` for
the second planning round, and `explicit` for a URL the caller handed in.

There are two numbers on purpose. `score` is the fusion output and orders the
SERP; it is a weighted sum, so it runs past 2.5 and 2.5 on one query does not
mean what 2.5 means on another. `relevance` is a cross-encoder probability where
the cross-encoder ran and a cosine elsewhere, always in 0..1 and always meaning
the same thing, so it is the one to threshold on. A page can carry the answer
and still read near zero when the surrounding text is thin, so the cut is yours
to choose rather than one liberdex applies for you.

| knob | |
| --- | --- |
| `top_k`, `depth` | `fast` (2.5s window, no hub expansion), `standard` (6s), `deep` (24s, second planning round). The window starts at the planner's first URL, waited for up to 1s / 4.5s / 6s; a cached plan waits nothing. `stats.plan_grace_ms` is what the wait cost |
| `budget` | a wall clock of your own, in seconds, in place of the tier's window and its wait |
| `include_domains` / `exclude_domains` | applied at dispatch, not after ranking; see below |
| `published_after` / `published_before`, `freshness` | `day\|week\|month\|year` is sugar over the former |
| `intent` | override the heuristic guess with any of the eight intents |
| `format` | `text` or `markdown` |
| `passages_per_page`, `passage_chars` | how many windows per page, and how big each one is. Defaults to 3 x 500, so 1500 characters per page |
| `token_budget`, `token_budget_per_page` | cap the response; rows past the cap are dropped, not emptied. A cut `text` ends in ` ...` and is shorter than `text_chars` |
| `include_text`, `include_plan`, `include_favicon`, `include_images` | |
| `answer` | `extract` (the fact alone) or `synthesize` (composed, with citations). Either also orders the SERP: the model read the top pages whole and lists the ones about the question; `stats.judged` says how many |
| `output_schema` | a JSON Schema the answer fills instead of prose |
| `weights` | named overrides on the intent's ranking weights |
| `query` as a list | up to 8, run concurrently against one fetcher |
| `like` | seed the plan from a URL instead of a phrase |

## Streaming

`POST /search/stream` emits Server-Sent Events in the chunk shape every agent
framework already parses. The events are the pipeline's own stages rather than
a token trickle: the plan, then each candidate and page as it lands, then the
results.

```
data: {"id":"…","object":"search.chunk","created":…,"event":"plan",
       "data":{"intent":"code","routes":["stackoverflow","github"],
               "hubs":["https://pkgdocs.example/tokio/latest/tokio/"],"expansions":[…]}}
data: {… "event":"candidate","data":{"url":"…","source":"llm"}}
data: {… "event":"page","data":{"url":"…","title":"…","status":200,"chars":4502}}
data: {… "event":"results","data":{ …the full response… }}
data: [DONE]
```

`plan` fires as the planner names routes, hubs and expansions, and only when
something changed: the planner emits on every token delta, and a listener does
not need to be told the same plan twelve times.

## Typed answers

`output_schema` takes a JSON Schema and fills it from the ranked pages, adding a
`_grounding` array: which pages support each field, and how confidently.
Per-field grounding beats a flat citation list, because a caller can act on the
field that was well supported and re-check the one that was not.

```jsonc
{"melting_point_celsius": 3422, "atomic_number": 74, "symbol": "W",
 "_grounding": [{"field": "melting_point_celsius", "pages": [1], "confidence": "high"},
                {"field": "symbol", "pages": [1, 2], "confidence": "high"}]}
```

## Reweighting

`weights` overrides named terms of the ranking fusion on top of the intent's
own: `{"rare": 0.8, "authority": 0.5}` leans harder on literal term matches and
less on the curated host list. It reweights the signals the fusion already has
and cannot add one. An unknown name is a 422 rather than a silent no-op, because
ignoring a typo would let a caller believe they had reweighted the SERP when they
had not.

Domain filters gate the fetch. With no index there is no result set to filter
down to, so an allowlist spends the entire fetch budget inside itself. The cost
is that a tight allowlist returns a short SERP rather than padding to `top_k`,
the same bargain the relevance floor already makes.

Undated pages survive a date filter unless you pass `require_date`. Most primary
sources, language and database manuals among them, publish no parseable date at
all, and dropping them would empty the SERP for the pages liberdex is best at
finding.
