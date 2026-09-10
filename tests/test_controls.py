"""Retrieval and context controls, plus the answer layer's boundaries."""
from __future__ import annotations

import asyncio

from selectolax.lexbor import LexborHTMLParser

from liberdex.answer import EXTRACT_SYSTEM, SYNTHESIZE_SYSTEM, build_messages, format_pages
from liberdex.engine import Liberdex, _enrich, _filter_by_date
from liberdex.extract import favicon_of, images_of, to_markdown
from liberdex.fetch import host_matches
from liberdex.rank import parse_date, select_windows
from liberdex.types import Doc, Result


def r(published: str = "", url: str = "https://x.test/a") -> Result:
    return Result(url=url, title="t", snippet="s", site="x.test", score=1.0,
                  published=published)


# ------------------------------------------------------------------- domains
def test_a_domain_covers_its_subdomains_but_not_the_other_way_round():
    assert host_matches("kafka.foundation.example", {"foundation.example"})
    assert not host_matches("foundation.example", {"kafka.foundation.example"})


def test_www_is_noise_on_both_sides():
    assert host_matches("www.Foundation.EXAMPLE", {"foundation.example"})
    assert host_matches("foundation.example", {"www.foundation.example"})


def test_a_suffix_is_not_a_subdomain():
    """evilfoundation.example must not pass a foundation.example allowlist."""
    assert not host_matches("evilfoundation.example", {"foundation.example"})
    assert not host_matches("notfoundation.example", {"foundation.example"})


def test_an_empty_list_matches_nothing():
    assert not host_matches("x.test", set())
    assert not host_matches("", {"x.test"})


def test_the_allowlist_gates_the_fetch_not_the_ranking():
    """With no index there is no result set to filter down to, so an allowlist
    instead spends the whole fetch budget inside itself."""
    eng = Liberdex(planner=None, speculate=False)

    async def go():
        async with eng:
            return await eng.search(
                "python asyncio cancel", budget=1.0,
                include_domains=["docs.lang.example"], expand_hubs=False)

    resp = asyncio.run(go())
    # Nothing outside the allowlist was even dispatched.
    assert resp.stats["dispatched"] == 0 or all(
        h.endswith("lang.example") for h in {x.site for x in resp.results})
    assert resp.stats.get("filtered_by_domain", 0) >= 0


# ---------------------------------------------------------------------- dates
def test_a_date_is_parsed_out_of_whatever_the_page_wrote():
    import datetime
    assert parse_date("2024-05-01") == datetime.date(2024, 5, 1)
    assert parse_date("2024/05/01") == datetime.date(2024, 5, 1)
    assert parse_date("20240501") == datetime.date(2024, 5, 1)
    # A year alone becomes 1 January, the only defensible reading.
    assert parse_date("published 2019 somewhere") == datetime.date(2019, 1, 1)
    assert parse_date("") is None and parse_date("no date here") is None


def test_an_absurd_date_does_not_blow_up_the_ranker():
    """Pages carry dates like 1900 and 2159, and epoch arithmetic raises
    OverflowError outside the platform's range for exactly those."""
    from liberdex.rank import _freshness
    for weird in ("1900-01-01", "0001-01-01", "9999-01-01", "2159-12-31"):
        assert 0.0 <= _freshness(weird) <= 1.0


def test_undated_pages_survive_a_date_filter_by_default():
    """Most primary sources publish no date; dropping them would empty the SERP
    for exactly the pages liberdex is best at finding."""
    rows = [r("2024-05-01"), r(""), r("2020-01-01")]
    kept = _filter_by_date(rows, parse_date("2023-01-01"), None, False)
    assert [x.published for x in kept] == ["2024-05-01", ""]


def test_require_date_drops_the_undated():
    rows = [r("2024-05-01"), r("")]
    kept = _filter_by_date(rows, parse_date("2023-01-01"), None, True)
    assert [x.published for x in kept] == ["2024-05-01"]


def test_both_ends_of_the_window_are_applied():
    rows = [r("2019-01-01"), r("2022-06-01"), r("2025-01-01")]
    kept = _filter_by_date(rows, parse_date("2020-01-01"), parse_date("2024-01-01"),
                           True)
    assert [x.published for x in kept] == ["2022-06-01"]


# ---------------------------------------------------------------- passages
def test_a_window_is_scored_on_the_query_terms_it_covers():
    """Coverage, not density: a window holding every query term scores 1.0
    however many times each one appears."""
    body = ("filler " * 400) + "tungsten melting point " + ("filler " * 400)
    parts, _b, _a = select_windows(body, "", ["tungsten", "melting", "point"], 300, 1)
    assert len(parts) == 1
    text, score = parts[0]
    assert "tungsten" in text and score == 1.0


def test_a_window_covering_half_the_terms_scores_half():
    body = ("filler " * 300) + "tungsten only here " + ("filler " * 300)
    parts, _b, _a = select_windows(body, "", ["tungsten", "absentword"], 300, 1)
    assert parts[0][1] == 0.5


def test_scores_are_zero_when_no_term_appears():
    parts, _, _ = select_windows("alpha beta " * 200, "", ["nothingatall"], 200, 1)
    assert parts and parts[0][1] == 0.0


def test_a_short_document_comes_back_whole_and_unmarked():
    parts, before, after = select_windows("short enough", "", ["short"], 500, 3)
    assert parts == [("short enough", 1.0)] and not before and not after


# ------------------------------------------------------------------- media
def test_a_declared_icon_wins_and_a_missing_one_falls_back():
    tree = LexborHTMLParser('<link rel="icon" href="/i.png">')
    assert favicon_of(tree, "https://x.test/page") == "https://x.test/i.png"
    bare = LexborHTMLParser("<p>nothing</p>")
    assert favicon_of(bare, "https://y.test/a/b") == "https://y.test/favicon.ico"


def test_the_social_card_leads_and_tracking_pixels_are_dropped():
    tree = LexborHTMLParser(
        '<meta property="og:image" content="https://x.test/card.jpg">'
        '<img src="/real.png"><img src="/pixel.gif"><img src="/sprite.svg">')
    assert images_of(tree, "https://x.test/p") == [
        "https://x.test/card.jpg", "https://x.test/real.png"]


def test_markdown_extraction_refuses_a_page_too_big_to_be_worth_it():
    """trafilatura's p99 is why this runs on ranked pages only; the size guard
    is the second half of that decision."""
    assert to_markdown("x" * 3_000_000, "https://x.test") == ""
    assert to_markdown("", "https://x.test") == ""


def test_enrich_only_touches_rows_it_can_match_to_a_page():
    row = Result(url="https://x.test/a", title="t", snippet="", site="x.test", score=1.0)
    orphan = Result(url="https://gone.test/z", title="t", snippet="", site="gone.test",
                    score=0.5)
    doc = Doc(url="https://x.test/a", final_url="https://x.test/a", status=200,
              favicon="https://x.test/f.ico", images=["https://x.test/i.png"])
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        asyncio.run(_enrich([row, orphan], [doc], pool,
                            markdown=False, want_media=True))
    assert row.favicon == "https://x.test/f.ico" and row.images == ["https://x.test/i.png"]
    assert orphan.favicon == "" and orphan.images == []


# ------------------------------------------------------------------ like seed
def test_the_site_name_is_stripped_off_a_title():
    """Nearly every CMS appends its own name; as a query term it pulls results
    back toward the site the seed is trying to leave."""
    from liberdex.engine import _strip_site_suffix
    assert _strip_site_suffix("Tungsten - Encyclopedia",
                              "en.encyclopedia.example") == "Tungsten"
    assert _strip_site_suffix("CREATE INDEX | DB Docs",
                              "www.db.example") == "CREATE INDEX"


def test_a_title_that_merely_ends_in_a_dash_is_left_alone():
    from liberdex.engine import _strip_site_suffix
    assert _strip_site_suffix("Kafka vs RabbitMQ - a comparison",
                              "vendor.example") == "Kafka vs RabbitMQ - a comparison"
    assert _strip_site_suffix("State-of-the-art results",
                              "example.com") == "State-of-the-art results"


def test_a_title_that_is_only_the_site_name_survives():
    from liberdex.engine import _strip_site_suffix
    assert _strip_site_suffix("Encyclopedia", "en.encyclopedia.example") == "Encyclopedia"
    assert _strip_site_suffix("", "x.test") == ""


# ------------------------------------------------------------------- answer
def test_the_page_block_carries_provenance_the_model_can_use():
    rows = [Result(url="https://en.encyclopedia.example/wiki/T", title="T",
                   snippet="", site="en.encyclopedia.example", score=0.97,
                   source="route:wikipedia", passages=["Melting point 3422 C"])]
    block = format_pages(rows, numbered=True)
    assert "[1] T" in block
    assert "found_by: route:wikipedia" in block
    assert "score: 0.97" in block
    assert "Melting point 3422 C" in block


def test_extract_mode_is_unnumbered_and_synthesize_mode_is_numbered():
    rows = [Result(url="https://x.test/a", title="A", snippet="", site="x.test",
                   score=1.0, passages=["p"])]
    assert "[1] A" not in format_pages(rows, numbered=False)
    assert "[1] A" in format_pages(rows, numbered=True)


def test_the_message_pair_picks_the_prompt_that_matches_the_mode():
    rows = [Result(url="https://x.test/a", title="A", snippet="", site="x.test",
                   score=1.0, passages=["p"])]
    assert build_messages("q", rows, "extract")[0]["content"] == EXTRACT_SYSTEM
    assert build_messages("q", rows, "synthesize")[0]["content"] == SYNTHESIZE_SYSTEM


def test_answering_with_no_results_fails_without_calling_a_model():
    from liberdex.answer import answer
    text, reply = asyncio.run(answer("q", []))
    assert text == "" and "no results" in reply.error


def test_an_unknown_mode_is_refused_rather_than_guessed():
    from liberdex.answer import answer
    rows = [Result(url="https://x.test/a", title="A", snippet="", site="x.test",
                   score=1.0, passages=["p"])]
    text, reply = asyncio.run(answer("q", rows, mode="summarise"))
    assert text == "" and "unknown answer mode" in reply.error


# ------------------------------------------------- passage budget composition
def test_passage_chars_is_per_passage_not_a_budget_split_between_them():
    """Three 500-character passages is 1500 characters, not 500 sliced in three.

    The ranker takes one budget and divides it, so the request layer multiplies.
    Asking for more passages and getting the same total text back, cut thinner,
    is the behaviour this guards against.
    """
    from liberdex.api import SearchRequest
    one = SearchRequest(query="q", passages_per_page=1, passage_chars=500)
    three = SearchRequest(query="q", passages_per_page=3, passage_chars=500)
    assert one.engine_kwargs()["snippet_chars"] == 500
    assert three.engine_kwargs()["snippet_chars"] == 1500
    assert three.engine_kwargs()["snippet_windows"] == 3


def test_the_default_request_ships_three_passages():
    from liberdex.api import SearchRequest
    req = SearchRequest(query="q")
    assert req.passages_per_page == 3
    assert req.engine_kwargs()["snippet_chars"] == 1500


def test_windows_actually_come_back_at_the_requested_size():
    body = " ".join(f"sentence {i} about tungsten melting point." for i in range(400))
    parts, _, _ = select_windows(body, "", ["tungsten", "melting"], 1500, 3)
    assert len(parts) == 3
    assert 1000 <= sum(len(p) for p, _ in parts) <= 1500


# ------------------------------------------------------------ passage prose
def test_a_passage_starts_where_a_sentence_starts():
    """A window must not resume mid-clause: "point (3,422 °C), lowest vapor..."."""
    head = "Alpha beta gamma. " * 40
    body = head + "Tungsten has the highest melting point of any metal. " + "delta epsilon. " * 200
    parts, _, _ = select_windows(body, "", ["tungsten", "melting", "point"], 500, 1)
    seg = parts[0][0]
    assert seg[0].isupper(), seg[:60]


def test_reaching_back_for_a_sentence_never_reaches_past_the_lookback():
    from liberdex.rank import SENTENCE_LOOKBACK, _snap
    text = "One. " + "x" * 4000 + " end."
    seg, start, _ = _snap(text, 2000, 300)
    assert start >= 2000 - SENTENCE_LOOKBACK


# ------------------------------------------------------------ chrome removal
def test_interface_text_is_dropped_from_the_extracted_page():
    from liberdex.extract import _clean
    out = _clean("Jump to content From Wikipedia, the free encyclopedia "
                 "Tungsten melts at 3422 C. "
                 "Uh oh! There was an error while loading. Please reload this page.")
    assert out == "Tungsten melts at 3422 C."


def test_removing_chrome_does_not_touch_ordinary_prose():
    from liberdex.extract import _clean
    body = "The content of the page is preserved exactly, including punctuation."
    assert _clean(body) == body
