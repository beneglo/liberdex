# The plan, in full

A plan is what liberdex's own planner would have written, in JSON instead of
the line protocol it streams. Every guard the streamed form gets applies to a
supplied one: a web search engine's own result pages are dropped, URLs over
400 characters are dropped, unknown route names are ignored, and `prior` is
clamped to 0.05..1.

## Fields

| field | type | limit | notes |
| --- | --- | --- | --- |
| `candidates` | list of `{url, prior?, reason?}` or bare URL strings | 40 | `prior` 0..1, default 0.55 |
| `routes` | `{route: query}` (a bare list of names also works) | 8 | query is trimmed to 200 chars |
| `hubs` | list of URLs | 6 | |
| `expansions` | list of strings | 8 | |
| `intent` | one of the eight intents | | see below |
| `lang` | ISO 639-1, optionally with region (`pt-BR`) | | sets `Accept-Language` and the Wikipedia edition |

## Intents

| intent | what it changes |
| --- | --- |
| `navigational` | the user wants a specific site; its homepage ranks well |
| `informational` | how does X work; authority and coverage weigh most |
| `code` | APIs, errors, libraries; docs and Q&A sites rank up |
| `academic` | papers; arxiv, publisher pages, university hosts |
| `news` | recent events; freshness matters, undated pages sink |
| `product` | specs, prices; maker pages and retailers |
| `local` | places and businesses |
| `reference` | definitions, statistics, standards |

## Writing candidates well

The hosts in this file end in `.example`, a reserved name that never resolves.
They stand in for real sites; a plan names real pages.

1. Exact page first. If you remember `https://docs.lang.example/3/library/dict.html`, give that, not `docs.lang.example`.
2. Section page when unsure. For a tail entity give the host's listing or category page and let hub expansion find the leaf. `https://www.wire.example/world/middle-east/` is fine; `https://www.wire.example/world/middle-east/some-guessed-slug-2026-05-01/` is a wasted fetch.
3. No opaque identifiers. Numeric ids, hashes, Stack Overflow question numbers, YouTube video ids, dated news slugs, `?p=1234`. If the address has a part you would have to guess, stop at the part you know.
4. Prefer the canonical host. The encyclopedia over its mirrors; the project's docs over a tutorial that quotes them; the standard over a blog about it.
5. Match the language. A question in German about a German institution wants German hosts and the German Wikipedia; set `lang: "de"`.
6. Spread hosts. Ten pages from one host is one source. Aim for 5+ hosts.
7. Mind the date. For `news`, name the outlet's section pages and set `intent: "news"`; freshness does the rest.

## Routes

A route is a site's own search endpoint that liberdex knows how to call and
parse. The query you give it is sent to that site's search box, so write it the
way that box wants it: short, the key nouns, no question words.

The routes: `wikipedia`, `hn` (Hacker News), `stackoverflow`, `github`,
`arxiv`, `pubmed`, `mdn`, `npm`, `pypi`, `crates`, `openlibrary`, `reddit`,
`youtube`. Any other name is ignored. A host's own search box is found by
liberdex itself, off the pages it fetches, so name the host as a candidate or a
hub rather than as a route.

## Example plans

### Factoid: "melting point of tungsten"
```json
{"candidates": [
  {"url": "https://en.encyclopedia.example/wiki/Tungsten", "prior": 0.95},
  {"url": "https://www.chemsociety.example/periodic-table/element/74/tungsten", "prior": 0.85},
  {"url": "https://webbook.standards.example/cgi/cbook.cgi?ID=C7440337", "prior": 0.5},
  {"url": "https://compounds.example/element/Tungsten", "prior": 0.7}],
 "routes": {"wikipedia": "tungsten"},
 "expansions": ["tungsten melting temperature", "wolfram melting point"],
 "intent": "reference", "lang": "en"}
```

### Code: "asyncio gather return_exceptions cancel remaining"
```json
{"candidates": [
  {"url": "https://docs.lang.example/3/library/asyncio-task.html", "prior": 0.95, "reason": "gather() reference"},
  {"url": "https://docs.lang.example/3/library/asyncio-task.html#asyncio.gather", "prior": 0.9},
  {"url": "https://tutorials.example/asyncio-gather-return_exceptions/", "prior": 0.5}],
 "routes": {"stackoverflow": "asyncio gather return_exceptions cancel", "github": "asyncio.gather return_exceptions"},
 "hubs": ["https://docs.lang.example/3/library/asyncio.html"],
 "expansions": ["gather exceptions cancel other tasks", "TaskGroup vs gather"],
 "intent": "code", "lang": "en"}
```

### News: "ECB rate decision this week"
```json
{"candidates": [
  {"url": "https://www.centralbank.example/press/pr/html/index.en.html", "prior": 0.9, "reason": "press releases index"},
  {"url": "https://www.centralbank.example/press/govcdec/mopo/html/index.en.html", "prior": 0.85},
  {"url": "https://www.wire.example/markets/rates-bonds/", "prior": 0.6},
  {"url": "https://www.financialpaper.example/central-banks", "prior": 0.5}],
 "routes": {"hn": "ECB rate decision"},
 "hubs": ["https://www.centralbank.example/press/html/index.en.html"],
 "expansions": ["European Central Bank interest rate announcement", "ECB deposit facility rate"],
 "intent": "news", "lang": "en"}
```

### Non-English: "Bundeskanzler Amtszeit Grundgesetz"
```json
{"candidates": [
  {"url": "https://de.encyclopedia.example/wiki/Bundeskanzler_(Deutschland)", "prior": 0.95},
  {"url": "https://www.gesetze.example/gg/art_63.html", "prior": 0.85},
  {"url": "https://www.parlament.example/aufgaben/kanzlerwahl", "prior": 0.7}],
 "routes": {"wikipedia": "Bundeskanzler Amtszeit"},
 "hubs": ["https://www.gesetze.example/gg/"],
 "expansions": ["Amtsdauer Bundeskanzler", "Wahl des Bundeskanzlers Artikel 63"],
 "intent": "reference", "lang": "de"}
```
