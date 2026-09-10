"""The wire shape: native vocabulary, the request aliases, and the budget."""
from __future__ import annotations

import datetime

import pytest

from liberdex.api import DEPTH, SearchRequest, freshness_floor, plan_view, reply
from liberdex.budget import apply as budget_fit
from liberdex.budget import count_tokens, truncate
from liberdex.rank import WINDOW_JOIN
from liberdex.types import Candidate, Plan, Result, SearchResponse


def mkresult(i: int = 0, passages=("alpha beta", "gamma delta"), text="") -> Result:
    return Result(
        url=f"https://x.test/{i}", title=f"Title {i}", snippet=WINDOW_JOIN.join(passages),
        site="x.test", score=round(1.0 - i / 10, 3), published="2024-05-01",
        source="route:wikipedia", passages=list(passages),
        passage_scores=[0.6, 0.2][:len(passages)], text=text, status=200,
    )


def mkresponse(n: int = 2, **kw) -> SearchResponse:
    return SearchResponse(
        query="q", intent="reference", results=[mkresult(i) for i in range(n)],
        timings={"total_ms": 1930.0}, stats={"lang": "en", "domains": 1}, **kw,
    )


# ------------------------------------------------------------------- requests
def test_there_are_no_borrowed_field_names():
    # One shape, liberdex's own words. Somebody else's names are not aliases
    # here; they are unknown fields, and the request must not quietly accept
    # and ignore them.
    for alias in ("max_results", "search_depth", "include_raw_content"):
        r = SearchRequest(query="q", **{alias: 1})
        assert not hasattr(r, alias) or getattr(r, alias) is None
        assert r.top_k == 10 and r.depth == "standard" and r.include_text is False


def test_native_names_are_the_real_ones():
    r = SearchRequest(query="q", top_k=3, depth="deep", include_text=True)
    kw = r.engine_kwargs()
    assert kw["top_k"] == 3 and kw["budget"] == 24.0 and kw["keep_text"] is True


def test_markdown_implies_asking_for_the_text():
    """Markdown is a rendering of the page text, so requesting it without
    requesting text would silently return nothing."""
    r = SearchRequest(query="q", format="markdown")
    assert r.include_text is True and r.engine_kwargs()["keep_text"] is True


def test_a_request_needs_a_query_or_a_url_to_seed_from():
    with pytest.raises(ValueError):
        SearchRequest()
    assert SearchRequest(like="https://x.test/a").queries() == []


def test_a_query_list_is_cleaned_and_capped():
    r = SearchRequest(query=["a", "  ", "b", ""] + [f"q{i}" for i in range(10)])
    assert r.queries()[:2] == ["a", "b"] and len(r.queries()) == 8


def test_freshness_becomes_an_absolute_floor():
    r = SearchRequest(query="q", freshness="week")
    expected = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
    assert r.published_after == expected
    assert freshness_floor(None) == ""


def test_an_explicit_after_date_beats_the_freshness_shorthand():
    r = SearchRequest(query="q", freshness="year", published_after="2020-01-01")
    assert r.published_after == "2020-01-01"


def test_intent_reaches_the_engine():
    assert SearchRequest(query="q", intent="news").engine_kwargs()["intent"] == "news"


# ------------------------------------------------------------------ responses
def test_the_native_row_carries_passages_and_their_scores():
    req = SearchRequest(query="q")
    out = reply(mkresponse(), req)
    row = out.results[0]
    assert row.passages == ["alpha beta", "gamma delta"]
    assert row.passage_scores == [0.6, 0.2]
    # Provenance is surfaced, not hidden: no index engine can report it.
    assert row.source == "route:wikipedia"
    assert row.status == 200
    # Not asked for, so not paid for.
    assert row.text is None and row.favicon is None and row.images is None


def test_markdown_text_actually_reaches_the_row():
    resp = mkresponse(1)
    resp.results[0].text = "# Heading\n\n[link](https://x.test)"
    row = reply(resp, SearchRequest(query="q", format="markdown")).results[0]
    assert row.text.startswith("# Heading")


def test_optional_fields_appear_only_when_asked_for():
    resp = mkresponse(1)
    resp.results[0].text = "the whole page"
    resp.results[0].favicon = "https://x.test/f.ico"
    resp.results[0].images = ["https://x.test/i.png"]
    req = SearchRequest(query="q", include_text=True, include_favicon=True,
                        include_images=True)
    row = reply(resp, req).results[0]
    assert row.text == "the whole page"
    assert row.favicon == "https://x.test/f.ico"
    assert row.images == ["https://x.test/i.png"]


def test_the_plan_is_only_rendered_when_it_was_asked_for():
    assert plan_view(None) is None
    plan = Plan(query="q", intent="code", lang="en", hubs=["https://h.test"],
                routes=["github"], route_queries={"github": "tokio"},
                candidates=[Candidate(url="https://d.test/a", prior=0.9)])
    view = plan_view(plan)
    assert view.routes == ["github"] and view.route_queries == {"github": "tokio"}
    assert view.candidates[0]["url"] == "https://d.test/a"


# --------------------------------------------------------------------- budget
def test_a_token_budget_is_actually_honoured():
    rows = [mkresult(i, passages=("word " * 300,)) for i in range(5)]
    kept, dropped = budget_fit(rows, token_budget=400)
    spent = sum(count_tokens(r.title) + count_tokens(r.url)
                + sum(count_tokens(p) for p in r.passages) for r in kept)
    assert spent <= 400 and dropped == 5 - len(kept)


def test_a_budget_too_small_for_one_row_shrinks_it_instead_of_returning_none():
    """Ten rows with no content is not a cheaper answer, it is no answer.
    Neither is an empty list."""
    kept, _ = budget_fit([mkresult(0, passages=("word " * 500,))], token_budget=40)
    assert len(kept) == 1 and kept[0].passages and kept[0].passages[0].endswith("...")


def test_the_per_page_cap_drops_page_text_before_it_drops_passages():
    """Passages are query-selected; the page text is everything."""
    row = mkresult(0, passages=("alpha " * 40,), text="filler " * 400)
    budget_fit([row], token_budget_per_page=80)
    assert row.passages and row.passages[0]
    assert count_tokens(row.text) < 80


def test_passages_per_page_trims_scores_alongside_their_passages():
    row = mkresult(0, passages=("a", "b"))
    budget_fit([row], passages_per_page=1)
    assert row.passages == ["a"] and row.passage_scores == [0.6]


def test_truncate_lands_on_a_word_boundary():
    out = truncate("alpha beta gamma delta epsilon " * 20, 12)
    assert out.endswith("...") and "  " not in out


def test_no_budget_means_no_change():
    rows = [mkresult(i) for i in range(3)]
    kept, dropped = budget_fit(rows, token_budget=0)
    assert kept == rows and dropped == 0


# ------------------------------------------------------------- absolute score
def test_relevance_is_bounded_where_the_fusion_score_is_not():
    """`score` is a weighted sum and runs past 2.5, so a caller thresholding on
    it would treat everything as a perfect match. `relevance` is the absolute
    reading in 0..1 and is the field to threshold on; both ship, named for what
    they are.
    """
    from liberdex.api import SearchReply, SearchResult
    reply_ = SearchReply(
        request_id="r", query="q", intent="informational", lang="en",
        results=[SearchResult(url="https://x.test/a", title="A", site="x.test",
                              score=2.5115, relevance=0.94, passages=["p"])],
        timings={"total_ms": 1000.0}, stats={})
    row = reply_.results[0]
    assert row.score > 1.0
    assert 0.0 <= row.relevance <= 1.0


def test_a_plan_never_reports_the_same_url_twice():
    from liberdex.api import plan_view
    from liberdex.types import Candidate, Plan
    plan = Plan(query="q", candidates=[
        Candidate(url="https://a.test/1", prior=0.9),
        Candidate(url="https://b.test/2", prior=0.8),
        Candidate(url="https://a.test/1", prior=0.4),
    ])
    urls = [c["url"] for c in plan_view(plan).candidates]
    assert urls == ["https://a.test/1", "https://b.test/2"]


def test_extract_refuses_something_that_was_never_a_url():
    """Otherwise a typo is dispatched and comes back as `DNSError`."""
    import pytest
    from pydantic import ValidationError

    from liberdex.api import ExtractRequest
    with pytest.raises(ValidationError, match="not http"):
        ExtractRequest(urls=["not-a-url"])
    assert ExtractRequest(urls=["https://x.test/a"]).urls == ["https://x.test/a"]


def test_an_empty_plan_says_whether_it_is_an_answer_or_a_misconfiguration():
    from liberdex.api import plan_stats
    from liberdex.types import Plan
    assert plan_stats(Plan(query="q")) == {}
    assert plan_stats(Plan(query="q", raw="planner error: 401 unauthorized")) == {
        "planner_error": "401 unauthorized"}


# ----------------------------------------------------------------- rounds
def test_deep_is_the_tier_with_a_second_pass():
    assert DEPTH["deep"].get("refine") is True
    assert not DEPTH["standard"].get("refine") and not DEPTH["fast"].get("refine")


def test_a_page_turn_is_a_new_round_with_the_shown_pages_excluded():
    from liberdex.api import PAGE_EXTRA_BUDGET
    r = SearchRequest(query="q", page=2,
                      exclude_urls=["https://a.test/1", " ", "https://a.test/2"])
    kw = r.engine_kwargs()
    assert kw["page"] == 2
    assert kw["exclude_urls"] == ["https://a.test/1", "https://a.test/2"]
    # The round costs a planner call; the tier's budget grows by that much.
    assert kw["budget"] == DEPTH["standard"]["budget"] + PAGE_EXTRA_BUDGET
    # An explicit budget is the caller's, page or no page.
    assert SearchRequest(query="q", page=3, budget=4.0).engine_kwargs()["budget"] == 4.0
    assert SearchRequest(query="q").engine_kwargs()["page"] == 1


def test_the_reply_says_which_page_it_is():
    from liberdex.types import SearchResponse
    resp = SearchResponse(query="q", intent="reference", results=[])
    assert reply(resp, SearchRequest(query="q", page=2)).page == 2
    assert reply(resp, SearchRequest(query="q")).page == 1


# ----------------------------------------------------------------- a supplied plan
def test_a_supplied_plan_replaces_the_planner_and_keeps_the_guards():
    r = SearchRequest(query="tcp handshake", plan={
        "candidates": [{"url": "https://en.encyclopedia.example/wiki/TCP", "prior": 2.0},
                       "https://www.google.com/search?q=tcp", "http://example.com/x"],
        "routes": {"wikipedia": "tcp handshake", "nonesuch": "x"},
        "hubs": ["https://example.com/"], "intent": "code", "lang": "EN"})
    plan = r.engine_kwargs()["plan"]
    assert [c.url for c in plan.candidates] == ["https://en.encyclopedia.example/wiki/TCP",
                                                "https://example.com/x"]
    assert plan.candidates[0].prior == 1.0 and plan.candidates[1].prior == 0.55
    assert plan.routes == ["wikipedia"] and plan.route_queries == {"wikipedia": "tcp handshake"}
    assert plan.intent == "code" and plan.lang == "en"
    assert plan.as_dict()["routes"] == {"wikipedia": "tcp handshake"}


def test_a_plan_goes_with_one_query():
    with pytest.raises(ValueError):
        SearchRequest(query=["a", "b"], plan={"candidates": []})


def test_no_plan_means_none():
    assert SearchRequest(query="q").engine_kwargs()["plan"] is None


# ------------------------------------------------------------- plan grace
def test_the_tier_waits_for_the_first_url_unless_the_budget_is_the_callers():
    from types import SimpleNamespace

    from liberdex.api import PLAN_GRACE
    kw = SearchRequest(query="q").engine_kwargs()
    assert kw["budget"] == DEPTH["standard"]["budget"]
    assert kw["plan_grace"] == PLAN_GRACE["standard"]
    assert SearchRequest(query="q", depth="fast").engine_kwargs()["plan_grace"] == PLAN_GRACE["fast"]
    # An explicit budget is a wall clock.
    assert SearchRequest(query="q", budget=4.0).engine_kwargs()["plan_grace"] == 0.0
    # A supplied plan runs no planner.
    kw = SearchRequest(query="q", plan={"candidates": [{"url": "https://a.test/"}]}).engine_kwargs()
    assert kw["plan_grace"] == 0.0
    # A whole-reply planner grows the budget to its floor and gets no grace on top.
    kw = SearchRequest(query="q").engine_kwargs(planner=SimpleNamespace(min_budget=15.0))
    assert kw["budget"] == 15.0 and kw["plan_grace"] == 0.0
    # A floor below the tier changes nothing: page 2 has 16s, the floor is 15.
    kw = SearchRequest(query="q", page=2).engine_kwargs(planner=SimpleNamespace(min_budget=15.0))
    assert kw["budget"] == 16.0 and kw["plan_grace"] == PLAN_GRACE["standard"]
    # A streaming planner declares no floor.
    kw = SearchRequest(query="q").engine_kwargs(planner=SimpleNamespace(finish_floor=15.0))
    assert kw["budget"] == DEPTH["standard"]["budget"] and kw["plan_grace"] == PLAN_GRACE["standard"]


def test_a_query_of_spaces_is_no_query():
    with pytest.raises(ValueError):
        SearchRequest(query="   ")
    assert SearchRequest(query="  tcp  ").query == "tcp"
    with pytest.raises(ValueError):
        SearchRequest(like="   ")


def test_a_date_bound_the_engine_cannot_read_is_refused_not_dropped():
    with pytest.raises(ValueError, match="published_after"):
        SearchRequest(query="q", published_after="not-a-date")
    with pytest.raises(ValueError, match="published_before"):
        SearchRequest(query="q", published_before="yesterday")
    assert SearchRequest(query="q", published_after="2024-01-05").published_after == "2024-01-05"
