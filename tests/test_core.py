"""Invariants for the parts of liberdex that are easy to break silently."""
from __future__ import annotations

import asyncio

import pytest

from liberdex.engine import Liberdex, guess_intent
from liberdex.extract import decode, extract
from liberdex.fetch import registrable
from liberdex.planner.base import PlanParser, valid_url
from liberdex.query import guess_lang, keyword_query
from liberdex.rank import (
    Ranker,
    authority,
    normalize_url,
    select_passages,
    split_passages,
    tokens,
    weights_for,
)
from liberdex.routes import ROUTES, Route, route_parser
from liberdex.types import Candidate, Doc, Plan


# ------------------------------------------------------------------ plan parse
def test_parser_streams_candidates_incrementally():
    p = PlanParser("q")
    assert p.feed("I code\nR github, stackoverflow\n") == []
    got = p.feed("U 0.9 https://docs.lang.example/3/x.html\n")
    assert [c.url for c in got] == ["https://docs.lang.example/3/x.html"]
    assert got[0].prior == pytest.approx(0.9)
    assert p.plan.intent == "code"
    assert p.plan.routes == ["github", "stackoverflow"]


def test_parser_handles_split_lines():
    p = PlanParser("q")
    assert p.feed("U 0.8 https://exam") == []
    got = p.feed("ple.com/a\n")
    assert [c.url for c in got] == ["https://example.com/a"]


def test_parser_rejects_search_engines_and_junk():
    p = PlanParser("q")
    p.feed("U 0.9 https://www.google.com/search?q=x\n")
    p.feed("U 0.9 https://bad\n")
    p.feed("U 0.9 not-a-url\n")
    assert p.plan.candidates == []


def test_parser_tolerates_missing_confidence():
    p = PlanParser("q")
    got = p.feed("U https://example.com/a :: because\n")
    assert len(got) == 1 and got[0].reason == "because"


def test_parser_pinned_line():
    p = PlanParser("q")
    got = p.feed("P 1 https://example.com/a\nP x https://example.com/b\nU 0.5 https://example.com/c\n")
    assert [(c.url, c.rank, c.prior) for c in got] == [
        ("https://example.com/a", 1, 1.0), ("https://example.com/c", 0, 0.5)]


def test_parser_dedups_and_collects_hubs_expansions():
    p = PlanParser("q")
    p.feed("U 0.9 https://a.example/x\nU 0.8 https://a.example/x\n"
           "H https://hub.example/i\nX one thing | another thing\n")
    p.finish()
    assert len(p.plan.candidates) == 1
    assert p.plan.hubs == ["https://hub.example/i"]
    assert p.plan.expansions == ["one thing", "another thing"]


def test_valid_url():
    assert valid_url("https://example.com/a")
    assert not valid_url("https://google.com/search?q=a")
    assert not valid_url("ftp://example.com")


# ---------------------------------------------------------------- query rewrite
def test_keyword_query_pins_subject_term():
    assert keyword_query("rust borrow checker lifetime elision", 3).startswith("rust")
    assert "HNSW" in keyword_query("how do vector databases implement HNSW indexes", 3)


def test_keyword_query_strips_question_words():
    assert keyword_query("what is the capital of australia") == "capital australia"


# --------------------------------------------------------------------- ranking
def test_normalize_url_strips_tracking_and_case():
    assert (normalize_url("https://WWW.Example.com/a/b/?utm_source=x&z=1#f")
            == "https://example.com/a/b?z=1")
    assert normalize_url("http://example.com") == "https://example.com/"


def test_split_passages_overlaps_and_caps():
    chunks = split_passages("Sentence number one. " * 300)
    assert 1 < len(chunks) <= 6
    assert all(len(c) > 100 for c in chunks)


def test_dedup_merges_consensus_and_keeps_richer_copy():
    thin = Doc(url="https://a.example/x", final_url="https://a.example/x", status=200, text="hi")
    fat = Doc(url="https://www.a.example/x/", final_url="https://www.a.example/x/",
              status=200, text="a much longer body")
    out = Ranker()._dedup([thin, fat])
    assert len(out) == 1
    assert out[0].text == "a much longer body"
    assert out[0].consensus == 2


def test_rank_survives_docs_with_no_text():
    docs = [Doc(url="https://a.example/x", final_url="https://a.example/x", status=200,
                title="Capital of Australia",
                candidate=Candidate(url="https://a.example/x", snippet="Canberra"))]
    res = Ranker().rank("capital of australia", docs, Plan(query="q"), top_k=5)
    assert res and res[0].url == "https://a.example/x"


# -------------------------------------------------------------------- extract
def test_extract_picks_prose_over_navigation():
    html = b"""<html><head><title>T</title>
      <meta name="description" content="D"></head><body>
      <nav><a href="/1">one</a><a href="/2">two</a><a href="/3">three</a></nav>
      <div class="sidebar"><a href="/4">four</a></div>
      <article><p>""" + b"The real content of the page is here. " * 30 + b"""</p></article>
      </body></html>"""
    d = Doc(url="https://e.example/a", final_url="https://e.example/a", status=200,
            content_type="text/html", body=html)
    extract(d, want_links=True)
    assert d.title == "T" and d.description == "D"
    assert "real content" in d.text
    assert "four" not in d.text
    assert any(u.endswith("/1") for u, _, _ in d.links)


def test_decode_handles_declared_charset():
    assert decode("café".encode("latin-1"), "text/html; charset=latin-1") == "café"


# --------------------------------------------------------------------- intent
@pytest.mark.parametrize("query,intent", [
    ("how do i fix TypeError in python", "code"),
    ("what is the tallest mountain in japan", "reference"),
    ("attention is all you need paper", "academic"),
    ("kafka vs rabbitmq", "product"),
    ("latest news on the election", "news"),
])
def test_guess_intent(query, intent):
    assert guess_intent(query) == intent


# ---------------------------------------------------------------- protocol
def test_parser_reads_per_route_queries_and_language():
    p = PlanParser("q")
    p.feed("I code\nL de\n"
           "R wikipedia: Bundeskanzler Deutschland | reddit: r/rust async trait "
           "| github\n")
    assert p.plan.lang == "de"
    assert p.plan.routes == ["wikipedia", "reddit", "github"]
    assert p.plan.route_queries == {
        "wikipedia": "Bundeskanzler Deutschland",
        "reddit": "r/rust async trait",
    }


def test_parser_still_accepts_v1_bare_route_list():
    p = PlanParser("q")
    p.feed("R github, stackoverflow, hn\n")
    assert p.plan.routes == ["github", "stackoverflow", "hn"]
    assert p.plan.route_queries == {}


def test_parser_ignores_bad_language_codes():
    p = PlanParser("q")
    p.feed("L Deutsch\n")
    assert p.plan.lang == "en"


def test_parser_strips_markdown_skin():
    p = PlanParser("q")
    got = p.feed("```\n- **U** 0.9 https://example.com/a\n```\n")
    assert [c.url for c in got] == ["https://example.com/a"]


def test_parser_keeps_balanced_parentheses_in_urls():
    p = PlanParser("q")
    got = p.feed("U 0.9 https://de.encyclopedia.example/wiki/Bundeskanzler_(Deutschland).\n")
    assert got[0].url == "https://de.encyclopedia.example/wiki/Bundeskanzler_(Deutschland)"
    got = p.feed("U 0.9 https://example.com/a)\n")
    assert got[0].url == "https://example.com/a"


# -------------------------------------------------------------- language
@pytest.mark.parametrize("query,lang", [
    ("how does the tcp handshake work", "en"),
    ("wer ist der aktuelle bundeskanzler von deutschland", "de"),
    ("comment fonctionne le protocole tcp", "fr"),
    ("日本で一番高い山は", "ja"),
    ("как работает tcp", "ru"),
])
def test_guess_lang(query, lang):
    assert guess_lang(query) == lang


def test_keyword_query_uses_language_stopwords():
    q = keyword_query("wer ist der aktuelle bundeskanzler von deutschland", 3,
                      lang="de")
    assert q == "aktuelle bundeskanzler deutschland"


# ----------------------------------------------------------------- hosts
def test_registrable_splits_multi_tenant_hosts():
    assert registrable("rust-lang.github.io") == "rust-lang.github.io"
    assert registrable("py-free-threading.github.io") == "py-free-threading.github.io"
    assert registrable("en.encyclopedia.example") == "encyclopedia.example"
    assert registrable("www.bbc.co.uk") == "bbc.co.uk"


def test_authority_walks_suffixes_and_falls_back_to_tld_class():
    assert authority("en.wikipedia.org") == authority("wikipedia.org")
    assert authority("towardsdatascience.medium.com") < 0
    assert authority("www.nist.gov") > 0
    assert authority("example.com") == 0.0


# ------------------------------------------------------------------ routes
def test_route_query_prefers_the_planner_phrasing():
    r = ROUTES["wikipedia"]
    assert r.query_for("what is the capital of australia", "Canberra") == "Canberra"
    assert r.query_for("what is the capital of australia") == "capital australia"


def test_reddit_route_scopes_to_a_subreddit():
    url, _ = ROUTES["reddit"].build("r/rust async trait", "en")
    assert "/r/rust/search.json" in url and "restrict_sr=1" in url
    url, _ = ROUTES["reddit"].build("async trait", "en")
    assert "/r/" not in url


def test_reddit_route_falls_back_to_the_pushshift_mirror():
    r = ROUTES["reddit"]
    assert r.fallbacks, "reddit must have somewhere to go when www 403s us"
    url, _ = r.fallbacks[0]("r/rust async trait", "en")
    assert "api.pullpush.io" in url and "subreddit=rust" in url and "q=async+trait" in url


def test_reddit_parse_reads_both_backend_shapes():
    listing = (b'{"data":{"children":[{"kind":"t3","data":{"permalink":"/r/a/comments/1/x/",'
               b'"title":"T","ups":40,"subreddit":"a","created_utc":1700000000}}]}}')
    mirror = (b'{"data":[{"permalink":"/r/a/comments/1/x/","title":"T","score":40,'
              b'"subreddit":"a","created_utc":1700000000}]}')
    for payload in (listing, mirror):
        c = ROUTES["reddit"].parse(payload)
        assert len(c) == 1
        assert c[0].url == "https://www.reddit.com/r/a/comments/1/x/"
        assert "40 upvotes" in c[0].reason
        assert c[0].published == "2023-11-14"


def test_wikipedia_route_ignores_a_language_that_is_not_an_edition():
    # A plausible-looking code that resolves to nothing costs a DNS failure and
    # the whole route, so it must never reach the network.
    for bogus in ("zz", "xx", "", "EN-US"):
        url, _ = ROUTES["wikipedia"].build("x", bogus)
        assert url.startswith("https://en.wikipedia.org/")


def test_wikipedia_route_follows_the_plan_language():
    url, _ = ROUTES["wikipedia"].build("Bundeskanzler", "de")
    assert url.startswith("https://de.wikipedia.org/")
    assert route_parser("wikipedia", "de")(
        b'{"query":{"search":[{"title":"Bundeskanzler (Deutschland)"}]}}'
    )[0].url.startswith("https://de.wikipedia.org/wiki/")


def test_route_backs_off_exponentially_when_blocked():
    r = Route("t", lambda q, l: ("", {}), lambda b: [])
    assert r.available
    r.penalize(403, seconds=10.0)
    assert not r.available and r.blocks == 1
    first = r.cooldown_until
    r.penalize(429, seconds=10.0)
    assert r.cooldown_until - first > 5.0


def test_youtube_parse_reads_search_results():
    payload = (b'<script>var ytInitialData = {"c":[{"videoRenderer":'
               b'{"videoId":"abc123","title":{"runs":[{"text":"Fix a chain"}]},'
               b'"ownerText":{"runs":[{"text":"Bike Co"}]},'
               b'"viewCountText":{"simpleText":"1M views"}}}]};</script>')
    cands = ROUTES["youtube"].parse(payload)
    assert cands[0].url == "https://www.youtube.com/watch?v=abc123"
    assert cands[0].skip_fetch and "Fix a chain" in cands[0].content


# ----------------------------------------------------------------- ranking
def _doc(url, text, status=200, title=""):
    return Doc(url=url, final_url=url, status=status, title=title or url, text=text,
               site=url.split("/")[2], candidate=Candidate(url=url))


def test_rank_returns_a_descending_serp():
    docs = [_doc(f"https://s{i}.example/p", f"capital of australia canberra {i} " * 20)
            for i in range(6)]
    res = Ranker().rank("capital of australia", docs, Plan(query="q"), top_k=10)
    assert res == sorted(res, key=lambda r: -r.score)
    # Text is not kept unless asked for, but its size always travels.
    assert res[0].text == "" and res[0].text_chars == len(docs[0].text)


def test_rank_collapses_pages_that_render_identical_text():
    shell = "crates io serves as a central registry for sharing crates " * 6
    docs = [_doc(f"https://registry.example/crates/pkg{i}", shell) for i in range(4)]
    docs.append(_doc("https://real.example/a", "async trait object safety in rust " * 20))
    res = Ranker().rank("rust async trait object safety", docs, Plan(query="q"),
                        top_k=10)
    assert sum(1 for r in res if "registry.example" in r.url) == 1


def test_rank_drops_pages_that_are_gone():
    good = [_doc(f"https://s{i}.example/p",
                 f"canberra is the capital of australia, source {i}. " * 20)
            for i in range(4)]
    dead = _doc("https://gone.example/p", "Error 404 page not found " * 20, status=404)
    res = Ranker().rank("capital of australia", good + [dead], Plan(query="q"),
                        top_k=10)
    assert all("gone.example" not in r.url for r in res)


def test_rank_keeps_a_minimum_serp_even_below_the_floor():
    docs = [_doc(f"https://s{i}.example/p", "unrelated boilerplate text " * 20)
            for i in range(5)]
    res = Ranker().rank("quantum chromodynamics lattice", docs, Plan(query="q"),
                        top_k=10)
    assert len(res) >= 1


# ------------------------------------------------------------------ engine
def test_search_rejects_an_empty_query_without_calling_the_planner():
    async def boom(*a, **k):
        raise AssertionError("planner must not run for an empty query")

    class P:
        stream = boom

    async def go():
        async with Liberdex(planner=P(), speculate=False) as lx:
            return await lx.search("   ")

    resp = asyncio.run(go())
    assert resp.results == [] and resp.stats["error"] == "empty query"


# ------------------------------------------------------- rare-term coverage
def test_rare_coverage_isolates_the_distinguishing_token():
    docs = [
        _doc("https://d.example/functions", "the function jsonb_path_query_first "
             "returns the first item. postgres syntax reference " * 5),
        _doc("https://d.example/datatype", "the jsonpath type in postgres. "
             "syntax for json path expressions " * 5),
        _doc("https://d.example/other", "postgres syntax overview " * 20),
    ]
    cov = Ranker()._rare_coverage("postgres jsonb_path_query_first syntax", docs)
    assert cov[0] == pytest.approx(1.0)
    assert cov[1] == 0.0 and cov[2] == 0.0


def test_rare_coverage_stays_silent_on_ordinary_words():
    # Every term is common English shared by most candidates; echoing the
    # user's phrasing must not be treated as evidence.
    docs = [_doc(f"https://s{i}.example/p", "the tallest mountain in japan " * 20)
            for i in range(3)]
    docs.append(_doc("https://s9.example/p", "mount fuji is the highest peak " * 20))
    cov = Ranker()._rare_coverage("what is the tallest mountain in japan", docs)
    assert not cov.any()


# ------------------------------------------------------------------- CJK
def test_cjk_text_segments_into_character_bigrams():
    got = tokens("創建索引")
    assert "創建" in got and "索引" in got and "創" in got


def test_cjk_segmentation_keeps_latin_runs_intact():
    got = tokens("如何在 postgres 中创建索引")
    assert "postgres" in got
    assert all(len(t) <= 2 for t in got if t != "postgres")


def test_latin_tokenisation_is_unchanged():
    assert tokens("HTTP/2 head-of-line blocking") == tokens("HTTP/2 head-of-line blocking")
    assert "http" in tokens("HTTP/2 blocking")


def test_non_latin_languages_stop_trusting_the_english_reranker():
    assert weights_for("code", "en").ce > weights_for("code", "zh").ce
    assert weights_for("code", "zh").emb > weights_for("code", "en").emb
    # Latin-script European languages keep the English profile.
    assert weights_for("code", "de").ce == weights_for("code", "en").ce


# --------------------------------------------------- passage selection
def test_select_passages_reaches_deep_into_a_long_document():
    filler = "unrelated prose about other topics. " * 4000   # ~140 KB
    text = "Built-in Types. " + filler + \
        "The dict.get method returns the default when the key is missing. " + filler
    chunks = select_passages(text, ["dict.get", "default", "missing"])
    assert any("dict.get method returns" in c for c in chunks), \
        "the window containing the query terms must be sampled"
    assert any("Built-in Types" in c for c in chunks), "the lede is still kept"


def test_select_passages_falls_back_to_the_lede_without_terms():
    text = "sentence one. " * 3000
    assert select_passages(text, []) == split_passages(text[:12_000], 6)


def test_rank_prefers_the_page_whose_deep_section_answers():
    filler = "general background prose. " * 1200
    good = _doc("https://docs.example.org/stdtypes",
                "Built-in Types. " + filler +
                " dict.get returns the default value if the key is missing. " + filler,
                title="Built-in Types")
    near = _doc("https://blog.example.com/defaultdict",
                "defaultdict factory for missing keys " * 60,
                title="defaultdict tutorial")
    res = Ranker().rank("python dict get default value if key missing",
                        [good, near], Plan(query="q"), top_k=2)
    assert any("stdtypes" in r.url for r in res)


# ------------------------------------------------------------- engine
def test_search_leaves_no_hub_fetch_running_after_it_returns():
    """A task outliving the query tore down libcurl underneath itself."""

    async def go():
        async with Liberdex(planner=None, speculate=False) as lx:
            await lx.search("kafka vs rabbitmq", budget=0.6, expand_hubs=True)
            return [t for t in asyncio.all_tasks() if not t.done()
                    and t is not asyncio.current_task()]

    assert asyncio.run(go()) == []


# --------------------------------------------------------------- snippets
def test_snippet_fills_the_budget_it_is_given():
    """A window sized in words at six characters each hands a caller asking
    for 1500 characters about a thousand."""
    from liberdex.rank import Ranker
    from liberdex.types import Doc

    body = " ".join(
        f"filler{i} word text" if i != 400 else "the marlin weighed 68 kilograms"
        for i in range(900)
    )
    doc = Doc(url="https://x.test/a", final_url="https://x.test/a", status=200, text=body)
    terms = ["marlin", "weighed", "kilograms"]
    for budget in (500, 1500, 3000):
        _parts, snip = Ranker()._passages(terms, "", doc, budget)
        assert budget * 0.85 <= len(snip) <= budget + 20, (budget, len(snip))
        assert "marlin weighed 68" in snip


def test_snippet_widens_a_short_ranking_passage():
    """The passage handed in is a few hundred characters of ranking window. A
    larger budget has to be met from the document, not silently truncated."""
    from liberdex.rank import Ranker
    from liberdex.types import Doc

    body = ("prelude " * 200) + "the summit was reached on 12 May 1996. " + ("coda " * 400)
    passage = "the summit was reached on 12 May 1996."
    doc = Doc(url="https://x.test/b", final_url="https://x.test/b", status=200, text=body)
    _parts, snip = Ranker()._passages(["summit", "reached"], passage, doc, 1500)
    assert len(snip) > 3 * len(passage)
    assert "12 may 1996" in snip.lower()


def test_snippet_windows_split_the_budget_and_keep_document_order():
    """Three windows share one budget, never overlap, and read front to back."""
    from liberdex.rank import WINDOW_JOIN, Ranker
    from liberdex.types import Doc

    filler = " ".join(f"pad{i}" for i in range(600))
    body = (f"{filler} alpha marker one {filler} beta marker two "
            f"{filler} gamma marker three {filler}")
    terms = ["alpha", "beta", "gamma", "marker"]
    doc = Doc(url="https://x.test/c", final_url="https://x.test/c", status=200, text=body)
    _parts, snip = Ranker()._passages(terms, "", doc, 1500, 3)
    parts = snip.strip(". ").split(WINDOW_JOIN)
    assert len(parts) == 3
    assert 1500 * 0.85 <= len(snip) <= 1500 + 40, len(snip)
    # All three facts are visible, and in the order the document states them.
    assert [p for p in ("alpha", "beta", "gamma") if p in snip] == \
        ["alpha", "beta", "gamma"]
    assert snip.index("alpha") < snip.index("beta") < snip.index("gamma")


def test_snippet_one_window_is_contiguous():
    """The default stays a single window, so the HTTP server is unchanged."""
    from liberdex.rank import WINDOW_JOIN, Ranker
    from liberdex.types import Doc

    filler = " ".join(f"pad{i}" for i in range(600))
    body = f"{filler} alpha marker {filler} beta marker {filler}"
    doc = Doc(url="https://x.test/d", final_url="https://x.test/d", status=200, text=body)
    _parts, snip = Ranker()._passages(["alpha", "beta", "marker"], "", doc, 1500, 1)
    assert WINDOW_JOIN not in snip
    assert 1500 * 0.85 <= len(snip) <= 1500 + 20


def test_snippet_falls_back_to_the_seed_without_a_body():
    """A route candidate that never fetched still gets the text it came with."""
    from liberdex.rank import Ranker
    from liberdex.types import Doc

    doc = Doc(url="https://x.test/e", final_url="https://x.test/e", status=200)
    _parts, snip = Ranker()._passages(["capital"], "Canberra is the capital.", doc, 500)
    assert snip == "Canberra is the capital."


# ------------------------------------------------------------------ deep links
def test_extract_keeps_the_article_inside_a_boilerplate_named_wrapper():
    """django wraps its docs in `<div class="container sidebar-right">`.

    The class names where the sidebar goes, not what the div is, and dropping it
    took the whole article with it.
    """
    body = b"The real content of the page is here. " * 40
    html = (b"""<html><head><title>T</title></head><body>
      <div class="container sidebar-right"><main id="main-content">
      <article id="docs-content"><p>""" + body + b"""</p></article></main></div>
      </body></html>""")
    d = Doc(url="https://e.example/a", final_url="https://e.example/a", status=200,
            content_type="text/html", body=html)
    extract(d)
    assert "real content" in d.text and len(d.text) > 1000


def test_extract_keeps_a_utility_class_container_holding_the_page():
    """Tailwind's `overflow-hidden` is not a hidden element, and the container
    carrying the bulk of a page's text is not chrome whatever it is called."""
    body = b"Every word of the answer lives in this block. " * 40
    html = (b"""<html><head><title>T</title></head><body>
      <div class="w-full py-12 overflow-hidden"><section><p>""" + body +
            b"""</p></section></div>
      <div class="footer"><a href="/x">x</a></div></body></html>""")
    d = Doc(url="https://e.example/a", final_url="https://e.example/a", status=200,
            content_type="text/html", body=html)
    extract(d)
    assert "lives in this block" in d.text


def test_extract_still_drops_named_chrome():
    body = b"The real content of the page is here. " * 40
    html = (b"""<html><head><title>T</title></head><body>
      <div class="sidebar"><a href="/1">one</a> a short aside</div>
      <article><p>""" + body + b"""</p></article></body></html>""")
    d = Doc(url="https://e.example/a", final_url="https://e.example/a", status=200,
            content_type="text/html", body=html)
    extract(d)
    assert "short aside" not in d.text


def test_path_fit_reads_a_slug_as_words():
    from liberdex.rank import path_fit
    v = [set(tokens("gdpr right to erasure article 17 conditions"))]
    assert path_fit(v, "/art-17-gdpr/") > 0.3          # one whole token
    assert path_fit(v, "/") == 0.0


def test_path_fit_keeps_whole_identifiers_matchable():
    from liberdex.rank import path_fit
    v = [set(tokens("postgres max_slot_wal_keep_size replication"))]
    assert path_fit(v, "/docs/current/max_slot_wal_keep_size.html") > 0.3


def test_path_fit_prefers_the_page_named_after_the_query():
    """A question-shaped slug covers a query term too, but as one word in
    eleven, and the F1's precision term prices that difference."""
    from liberdex.rank import path_fit
    v = [set(tokens("nginx client_max_body_size default value directive"))]
    reference = path_fit(
        v, "/en/docs/http/ngx_http_core_module.html client_max_body_size")
    question = path_fit(
        v, "/questions/27197856/node-nginx-413-request-entity-too-large-"
           "client_max_body_size")
    assert reference > question


def test_unversioned_collapses_release_segments():
    from liberdex.engine import _unversioned
    a = _unversioned("https://www.db.example/docs/18/runtime-config.html")
    b = _unversioned("https://www.db.example/docs/current/runtime-config.html")
    assert a == b
    assert _unversioned("https://e.example/a/b") != _unversioned("https://e.example/a/c")


def _deep_docs():
    directory = Doc(
        url="https://site.example/docs/", final_url="https://site.example/docs/",
        status=200, title="Documentation", site="site.example",
        text="Documentation index for the project.",
        links=[("https://site.example/docs/configure-pdb/",
                "Specifying a Disruption Budget for your Application", ""),
               ("https://site.example/docs/about/", "About us", "")])
    answer = Doc(
        url="https://site.example/docs/configure-pdb/",
        final_url="https://site.example/docs/configure-pdb/", status=200,
        title="Specifying a Disruption Budget for your Application",
        site="site.example", text="minAvailable and maxUnavailable are set here.",
        links=[("https://site.example/docs/about/", "About us", "")])
    return directory, answer


def test_follow_deep_follows_a_directory_to_the_page_it_names():
    directory, _answer = _deep_docs()
    fired = []

    def dispatch(cand, want_links=False):
        fired.append(cand.url)
        return True
    counts = {"expand": 0}
    plan = Plan(query="q", intent="code", lang="en")
    n = Liberdex._follow_deep(
        Liberdex.__new__(Liberdex),
        "pod disruption budget application", plan, [directory], dispatch, counts)
    assert n == 1 and fired == ["https://site.example/docs/configure-pdb/"]


def test_follow_deep_stays_quiet_on_the_page_that_was_wanted():
    """The page whose own name answers the query outscores its own links, so
    nothing is fetched, which keeps this off the latency budget."""
    _directory, answer = _deep_docs()
    fired = []

    def dispatch(cand, want_links=False):
        fired.append(cand.url)
        return True
    plan = Plan(query="q", intent="code", lang="en")
    n = Liberdex._follow_deep(
        Liberdex.__new__(Liberdex),
        "specifying a disruption budget for your application", plan, [answer],
        dispatch, {"expand": 0})
    assert n == 0 and fired == []


# ------------------------------------------------------ snippet window choice
def test_window_end_reaches_the_full_stop_of_the_sentence_it_cut():
    """A window that ends on "Acanthops bidens is native to" is one word short
    of the answer."""
    from liberdex.rank import _snap

    text = ("Acanthops bidens is a species of mantis. "
            "Acanthops bidens is native to Mexico.[2] References follow here.")
    cut = text.index("Mexico") + 3          # a width that ends inside the answer
    seg, _, _ = _snap(text, 0, cut)
    assert seg.endswith("native to Mexico.[2]"), seg


def test_rare_query_term_locates_the_fact_on_a_page_that_names_the_entity_everywhere():
    """A year table row with the answer beats a reference list that repeats the entity."""
    from liberdex.rank import Ranker
    from liberdex.types import Doc

    lede = "The Romer-Simpson Medal is the highest award of the Society. "
    rows = [f"{y} Person {y}" for y in range(1900, 2023)]
    rows[rows.index("1997 Person 1997")] = "1997 Colin Patterson"
    refs = " ".join(f"Romer-Simpson Medal awarded to Person {i}, Society news, retrieved 2014."
                    for i in range(40))
    body = lede + " ".join(rows) + " " + refs
    doc = Doc(url="https://x.test/f", final_url="https://x.test/f", status=200, text=body)
    _parts, snip = Ranker()._passages(
        ["romer", "simpson", "medal", "1997"], "", doc, 1500, 3)
    assert "Colin Patterson" in snip


def test_mediawiki_chrome_is_dropped_and_the_infobox_is_kept():
    html = b"""<html><head><title>W</title></head><body>
      <div class="mw-body"><div id="mw-content-text"><div class="mw-parser-output">
      <table class="ambox"><tr><td>This article needs more citations. Find sources: news newspapers</td></tr></table>
      <table class="infobox"><tr><th>Died</th><td>November 8, 1995</td></tr></table>
      <p>Sergei Kobozev was a boxer.<sup class="reference">[1]</sup> He was reported missing.</p>
      <h2>Past awards<span class="mw-editsection">[edit]</span></h2>
      <p>1997 Colin Patterson</p>
      <div class="navbox">Faraday Medal Beilby Medal Corday-Morgan Prize Centenary Prize</div>
      <div class="printfooter">Retrieved from "https://en.encyclopedia.example/w/index.php?title=X"</div>
      <div id="catlinks">Categories: Awards Hidden categories: Stubs</div>
      </div></div></div></body></html>"""
    d = Doc(url="https://en.encyclopedia.example/wiki/X", final_url="https://en.encyclopedia.example/wiki/X",
            status=200, content_type="text/html", body=html)
    extract(d)
    assert "November 8, 1995" in d.text and "Colin Patterson" in d.text
    assert "[1]" not in d.text and "[edit]" not in d.text
    for chrome in ("Find sources", "Beilby Medal", "Retrieved from", "Hidden categories"):
        assert chrome not in d.text, chrome


def test_the_answer_model_orders_the_serp_it_read():
    from liberdex.answer import judged, parse_judgement
    from liberdex.types import Result
    pages, body = parse_judgement("PAGES: 3 1 9 3\n\nZverev won.", 5)
    assert pages == [2, 0] and body == "Zverev won."
    rows = [Result(url=f"https://s/{i}", title=str(i), snippet="", site="s", score=1.0)
            for i in range(6)]
    # Read the first four: 2 and 0 listed, 1 and 3 rejected, 4 and 5 unseen.
    assert [r.title for r in judged(rows, pages, read=4)] == ["2", "0", "4", "5"]
    # Too few left: rejected rows fill the SERP back up to the floor.
    assert [r.title for r in judged(rows[:3], [2], read=3)] == ["2", "0", "1"]
    assert parse_judgement("no header here", 5) == ([], "no header here")


# ---------------------------------------------------------------- deadline
def test_a_deadline_moves_and_keeps_its_origin():
    from liberdex.types import Deadline
    d = Deadline(1.0)
    d.extend(0.5)
    assert d.budget == 1.5 and 1.0 < d.remaining <= 1.5
    d.extend(-1.0)
    assert d.budget == 0.5 and d.remaining <= 0.5 and not d.expired()
    d.extend(-1.0)
    assert d.expired() and d.remaining == 0.0
