"""The parts that stopped a non-English, document-heavy corpus from working.

Each test here is a defect seen on a real German-language query, not a
hypothetical: compound words silently zeroing every lexical
signal, PDFs arriving as empty pages, a site's own search endpoint going
unread, and a "create this article" link outranking the article.
"""
from __future__ import annotations

from selectolax.lexbor import LexborHTMLParser

from liberdex.extract import _ACTION_LINK, extract, extract_links, search_template
from liberdex.query import content_terms
from liberdex.rank import _f1, path_fit
from liberdex.types import Doc


def _terms(s: str, lang: str = "de") -> set[str]:
    return {t.lower() for t in content_terms(s, lang)}


# ------------------------------------------------------------- decompounding
def test_compound_words_score_above_zero():
    """The defect: a perfect subject match scoring exactly 0.0.

    "Strompreise" and "Strompreisentwicklung" are the same subject and
    share no whole token, so bm25, the title fit, the path fit and the anchor
    fit that decides whether a hub's link is followed all read them as
    unrelated. Below HUB_LINK_FIT (0.22) the link is never followed at all.
    """
    q = _terms("Aktuelle Strompreise Deutschland Entwicklung 2023")
    doc = _terms("Strompreisentwicklung Preisvergleich Haushaltsstrom")
    assert _f1(q, doc) > 0.3


def test_compound_match_still_separates_periods():
    """Partial matching must not blur the constraint the user wrote down.

    A report and its predecessor sit next to each other under near-identical
    anchors; only the quarter and the year tell them apart, and those are short
    tokens that character n-grams must leave alone.
    """
    q = _terms("Luftmessnetzbericht München Q3 2025")
    right = _f1(q, _terms("Luftmessnetz München Report Q3 2025"))
    wrong = _f1(q, _terms("Luftmessnetz München Report Q1 2019"))
    assert right > wrong + 0.2


def test_compound_match_reaches_the_url_path():
    q = _terms("Luftmessnetzbericht München Q3 2025")
    assert path_fit([q], "/berichte/luftmessnetz/muenchen-report-q3-2025") > 0.4


def test_unrelated_text_stays_near_zero():
    """The n-gram pass must not make everything match everything."""
    q = _terms("Strompreise Deutschland")
    assert _f1(q, _terms("Fußballergebnisse Bundesliga Spieltag")) < 0.15


def test_english_morphology_also_benefits():
    q = _terms("postgres create index concurrently", "en")
    assert _f1(q, _terms("Building Indexes Concurrently PostgreSQL", "en")) > 0.4


# ------------------------------------------------------------- site search
def _tree(html: str) -> LexborHTMLParser:
    return LexborHTMLParser(html)


def test_search_template_reads_wordpress_form():
    t = _tree('<form role="search" action="https://a.example/"><input name="s"></form>')
    assert search_template(t, "https://a.example/x") == "https://a.example/?s={}"


def test_search_template_keeps_the_controller_parameters():
    """TYPO3 selects its search controller in the action's own query string;
    dropping those parameters points at an endpoint that does not exist."""
    html = ('<form class="search" method="get" '
            'action="/suche?tx_ind[action]=search"><input name="tx_ind[sword]">'
            '</form>')
    got = search_template(_tree(html), "https://b.example/page")
    assert got.startswith("https://b.example/suche?")
    assert "action%5D=search" in got
    assert got.endswith("sword%5D={}")


def test_search_template_ignores_a_faceted_filter():
    """`search[priceMax]` is a price slider on a catalogue. Its wrapper says
    "search" and its leaf does not, and the leaf is what counts."""
    html = ('<form class="search" action="/produkte">'
            '<input name="search[priceMax]"><input name="search[colour]"></form>')
    assert search_template(_tree(html), "https://c.example/") == ""


def test_search_template_skips_post_forms():
    """A POST search is not addressable as a URL, and issuing one would be
    writing to someone else's site rather than reading from it."""
    html = '<form role="search" method="post" action="/s"><input name="q"></form>'
    assert search_template(_tree(html), "https://d.example/") == ""


def test_search_template_absent_when_there_is_no_form():
    assert search_template(_tree("<p>no form here</p>"), "https://e.example/") == ""


def test_extract_records_the_search_endpoint():
    doc = Doc(url="https://f.example/a", final_url="https://f.example/a",
              status=200, content_type="text/html")
    doc.body = (b'<html><body><form role="search" action="/suche">'
                b'<input name="q"></form></body></html>')
    extract(doc, want_links=True)
    assert doc.search_url == "https://f.example/suche?q={}"


# ------------------------------------------------------------- action links
def test_action_links_are_not_documents():
    """A MediaWiki redlink echoes the query back as its anchor, so it scores a
    perfect anchor match and is a blank edit form."""
    for href in ("/w/index.php?title=Foo&action=edit",
                 "/w/index.php?title=Foo&veaction=edit",
                 "/wiki/Special:Search",
                 "/de/login",
                 "/checkout"):
        assert _ACTION_LINK.search(href), href


def test_ordinary_documents_are_not_mistaken_for_actions():
    for href in ("/berichte/luftmessnetz/muenchen-report-q3-2025",
                 "/suche?q=abc",
                 "/insights/figures/berlin-2026?utm_source=x",
                 "/registered-office-guide",
                 "/blog/how-to-login-securely"):
        assert not _ACTION_LINK.search(href), href


def test_extract_links_drops_the_redlink():
    html = ('<html><body>'
            '<a href="/w/index.php?title=X&action=edit">X</a>'
            '<a href="/wiki/X">X</a></body></html>')
    links = extract_links(_tree(html), "https://de.encyclopedia.example/wiki/Y")
    assert [u for u, _a, _c in links] == ["https://de.encyclopedia.example/wiki/X"]


# ------------------------------------------------------------- pdf documents
def _minimal_pdf(text: str = "Luftqualitaetsbericht Wien 2024") -> bytes:
    """A one-page PDF with a text layer, built by hand so the test needs no
    fixture file and no network."""
    body = f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET".encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(body), body),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    start = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += (b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objs) + 1, start))
    return bytes(out)


def test_pdf_body_is_extracted_not_discarded():
    """A PDF must not arrive as a zero-length page for the ranker to penalise
    as thin and dead: in a document-heavy corpus it is the authoritative copy."""
    doc = Doc(url="https://g.example/bericht.pdf",
              final_url="https://g.example/bericht.pdf",
              status=200, content_type="application/pdf")
    doc.body = _minimal_pdf()
    extract(doc)
    assert "Luftqualitaetsbericht" in doc.text


def test_pdf_without_metadata_title_is_named_by_its_filename():
    doc = Doc(url="https://g.example/wohnungsmarkt-wien-2024.pdf",
              final_url="https://g.example/wohnungsmarkt-wien-2024.pdf",
              status=200, content_type="application/pdf")
    doc.body = _minimal_pdf()
    extract(doc)
    assert doc.title == "wohnungsmarkt wien 2024"


def test_pdf_detected_by_magic_when_the_header_lies():
    """Plenty of servers hand a PDF back as text/html or octet-stream."""
    doc = Doc(url="https://g.example/x", final_url="https://g.example/x",
              status=200, content_type="text/html")
    doc.body = _minimal_pdf()
    extract(doc)
    assert "Luftqualitaetsbericht" in doc.text


def test_scanned_pdf_reports_nothing_rather_than_guessing():
    """No text layer means no text. Saying so is honest; OCR is another
    product, and inventing a title from the filename alone would be a
    result that looks like a document and holds nothing."""
    doc = Doc(url="https://g.example/scan.pdf",
              final_url="https://g.example/scan.pdf",
              status=200, content_type="application/pdf")
    doc.body = b"%PDF-1.4\n" + b"\x00" * 200
    extract(doc)
    assert doc.text == ""


# ------------------------------------------------------------- search atlas
def test_atlas_prefers_the_endpoint_seen_from_most_pages(tmp_path):
    """A section-scoped search box is read off whichever page happened to land;
    the site-wide one is the endpoint most of the site's pages point at."""
    from liberdex.cache import SiteSearchAtlas
    a = SiteSearchAtlas(str(tmp_path / "atlas.sqlite3"))
    a.put("example.org", "https://example.org/section/search?q={}")
    a.put("example.org", "https://example.org/suche?q={}")
    a.put("example.org", "https://example.org/suche?q={}")
    assert a.get("example.org") == "https://example.org/suche?q={}"
    assert a.count() == 1


def test_atlas_refuses_a_template_with_no_slot(tmp_path):
    from liberdex.cache import SiteSearchAtlas
    a = SiteSearchAtlas(str(tmp_path / "atlas.sqlite3"))
    a.put("example.org", "https://example.org/suche")
    assert a.get("example.org") == ""


def test_atlas_expires_a_stale_endpoint(tmp_path):
    """Sites redesign; an endpoint learned once must not be trusted forever."""
    import time as _time

    from liberdex.cache import ATLAS_TTL, SiteSearchAtlas
    a = SiteSearchAtlas(str(tmp_path / "atlas.sqlite3"))
    a.put("example.org", "https://example.org/suche?q={}")
    assert a.get("example.org")
    assert a.get("example.org", now=_time.time() + ATLAS_TTL + 1) == ""


def test_search_template_refuses_a_third_party_index():
    """An embedded third-party site-search widget is not this site's own
    search, and firing at it would mean querying somebody else's index."""
    html = ('<form role="search" action="https://cse.thirdparty.example/search">'
            '<input name="q"></form>')
    assert search_template(_tree(html), "https://www.example.org/a") == ""


def test_search_template_allows_a_search_subdomain():
    html = ('<form role="search" action="https://search.example.org/s">'
            '<input name="q"></form>')
    assert (search_template(_tree(html), "https://www.example.org/a")
            == "https://search.example.org/s?q={}")


# ------------------------------------------------------------- authority
def test_government_authority_is_not_only_anglophone():
    """A registry-operated government domain is evidence in every country, and
    listing only `.gov`/`.gov.uk` scored a national statistics office outside
    the anglosphere the same as a content farm."""
    from liberdex.rank import authority
    for host in ("statistik.gv.at", "www.wien.gv.at", "destatis.bund.de",
                 "insee.gouv.fr", "stat.go.jp", "ec.europa.eu", "www.gov.uk",
                 "gov.uk"):
        assert authority(host) >= 0.10, host
    for host in ("tuwien.ac.at", "www.u-tokyo.ac.jp"):
        assert authority(host) >= 0.07, host


def test_authority_is_not_granted_by_a_lookalike_name():
    from liberdex.rank import authority
    for host in ("notgov.example", "mygov.example.com", "gov-grants.example",
                 "beispielshop.at"):
        assert authority(host) == 0.0, host


def test_atlas_does_not_collapse_sibling_government_hosts(tmp_path):
    """`registrable` is a two-label approximation and maps every `*.gv.at` to
    `gv.at`; keying on it would fire one regional government's query at
    another's search box."""
    from liberdex.cache import SiteSearchAtlas
    from liberdex.engine import _atlas_host
    a = SiteSearchAtlas(str(tmp_path / "atlas.sqlite3"))
    a.put(_atlas_host("https://www.tirol.gv.at/x"), "https://www.tirol.gv.at/suche/?q={}")
    a.put(_atlas_host("https://www.wien.gv.at/y"), "https://www.wien.gv.at/suche?q={}")
    assert a.get("tirol.gv.at") == "https://www.tirol.gv.at/suche/?q={}"
    assert a.get("wien.gv.at") == "https://www.wien.gv.at/suche?q={}"
    assert a.count() == 2


def test_registrable_keeps_registry_operated_second_levels_apart():
    """Collapsing every `*.gv.at` into one host made SERP diversity treat a
    federal government and two regional ones as a single source."""
    from liberdex.fetch import registrable
    for host, want in (
        ("tirol.gv.at", "tirol.gv.at"),
        ("www.wien.gv.at", "wien.gv.at"),
        ("stat.go.jp", "stat.go.jp"),
        ("insee.gouv.fr", "insee.gouv.fr"),
        ("sub.example.co.uk", "example.co.uk"),
        ("bbc.co.uk", "bbc.co.uk"),
        ("a.b.example.com", "example.com"),
        ("kafka.foundation.example", "foundation.example"),
        ("en.encyclopedia.example", "encyclopedia.example"),
    ):
        assert registrable(host) == want, host


def test_wiki_search_pages_are_dropped_in_every_language():
    """`Special:` is only the English spelling; de.wikipedia writes
    `Spezial:Suche`, and its search page reached rank 1."""
    for href in ("/w/index.php?title=Spezial:Suche&search=foerderung",
                 "/wiki/Spezial:Suche", "/wiki/Especial:Buscar",
                 "/wiki/Special:Search"):
        assert _ACTION_LINK.search(href), href
    for href in ("/wiki/Steiermark", "/suche?search=abc"):
        assert not _ACTION_LINK.search(href), href


def test_concurrent_pdf_extraction_does_not_abort_the_process():
    """PDFium is not thread-safe and extraction runs in a thread pool. Without a
    lock this aborts with SIGABRT rather than raising, so the failure is a dead
    worker and not a caught exception."""
    from concurrent.futures import ThreadPoolExecutor
    body = _minimal_pdf()

    def work(_i: int) -> int:
        doc = Doc(url="u", final_url="https://h.example/a.pdf", status=200,
                  content_type="application/pdf")
        doc.body = body
        extract(doc)
        return len(doc.text)

    with ThreadPoolExecutor(max_workers=6) as ex:
        out = list(ex.map(work, range(36)))
    assert all(n > 0 for n in out)


# ------------------------------------------------------------------- Chinese
# Chinese is written without spaces, so a whitespace `content_terms` returns
# the whole question as one word. Every bag-of-words signal then scored the right
# article at zero and `admit` dropped every zh.wikipedia route candidate before
# the fetch (route_dropped == candidates_route on a live query).
def test_chinese_question_splits_into_content_words():
    assert content_terms("量子计算机的工作原理是什么", "zh") == ["量子计算机", "工作原理"]
    assert content_terms("澳大利亚的首都是哪里", "zh") == ["澳大利亚", "首都"]


def test_chinese_keeps_single_character_nouns_but_not_grammar():
    terms = content_terms("猫的寿命有多长", "zh")
    assert "猫" in terms and "寿命" in terms
    assert "在" not in content_terms("如何在 postgres 中创建索引", "zh")
    assert "postgres" in content_terms("如何在 postgres 中创建索引", "zh")


def test_stop_split_does_not_break_content_words():
    """和 is a conjunction and also the middle of 共和国."""
    assert content_terms("中华人民共和国的人口有多少", "zh")[0] == "中华人民共和国"


def test_latin_inside_a_han_run_comes_out_whole():
    assert content_terms("python教程 列表推导式", "zh")[:2] == ["python", "教程"]


def test_chinese_title_fit_is_above_the_route_gate():
    from liberdex.engine import ROUTE_TITLE_FIT, _title_fit
    vocabs = [frozenset(_terms("量子计算机的工作原理是什么", "zh"))]
    assert _title_fit("量子计算机 - 维基百科，自由的百科全书", vocabs) > ROUTE_TITLE_FIT
    assert _title_fit("量子计算机 - 维基百科，自由的百科全书", vocabs) > 0.3
    assert _title_fit("足球比赛结果 英超联赛", vocabs) < ROUTE_TITLE_FIT


def test_chinese_partial_overlap_scores_by_character_bigrams():
    q = _terms("量子计算机的工作原理", "zh")
    assert _f1(q, _terms("量子计算", "zh")) > 0.4
    assert _f1(q, _terms("电子计算机", "zh")) < _f1(q, _terms("量子计算", "zh"))


def test_chinese_query_terms_reach_passage_windows():
    from liberdex.rank import query_terms, select_windows
    terms = query_terms("珠穆朗玛峰的高度是多少", "zh")
    assert "高度" in terms  # two characters, which the Latin >= 3 rule dropped
    body = "无关的段落。" * 100 + "珠穆朗玛峰的高度为8848.86米。" + "另一段无关文字。" * 100
    wins, _, _ = select_windows(body, "", terms, 300)
    assert "8848" in wins[0][0]
    assert wins[0][0].endswith("。")


def test_japanese_particles_split_kana_runs():
    assert "日本" in content_terms("日本で一番高い山は", "ja")


def test_latin_paths_are_untouched():
    assert content_terms("postgres create index concurrently", "en") == \
        ["postgres", "create", "index", "concurrently"]
    from liberdex.rank import _grams
    assert _grams("index") == frozenset({"^inde", "index", "ndex$"})


def test_simplified_and_traditional_titles_dedup_on_canonical():
    """zh.wikipedia serves one article under both spellings of its title."""
    from liberdex.rank import Ranker
    canon = "https://zh.encyclopedia.example/wiki/%E6%9F%8F%E6%9E%97%E5%A2%99%E5%80%92%E5%A1%8C"
    a = Doc(url=canon, final_url=canon, status=200, title="柏林墙倒塌",
            text="柏林墙倒塌发生于1989年11月9日。" * 20, canonical=canon)
    b = Doc(url=canon.replace("%A2%99", "%9C%8D%E7%89%86"),
            final_url=canon.replace("%A2%99", "%9C%8D%E7%89%86"), status=200,
            title="柏林圍牆倒塌", text="柏林圍牆倒塌發生於1989年11月9日。" * 20,
            canonical=canon)
    assert len(Ranker()._dedup([a, b])) == 1


def test_canonical_is_not_trusted_across_sites_or_depths():
    from liberdex.rank import _dedup_key
    d = Doc(url="https://a.example/reports/q3-2025", final_url="https://a.example/reports/q3-2025",
            status=200, canonical="https://a.example/reports")
    assert _dedup_key(d) == d.final_url  # a listing page, not this page
    d.canonical = "https://b.example/reports/q3-2025"
    assert _dedup_key(d) == d.final_url  # another site's claim
    d.canonical = "https://www.a.example/reports/q3-2025-final"
    assert _dedup_key(d) == d.canonical


def test_extract_reads_the_canonical_link():
    from liberdex.extract import extract
    html = ('<html lang="zh"><head><title>t</title>'
            '<link rel="canonical" href="/wiki/A"></head><body><p>'
            + "正文内容。" * 30 + '</p></body></html>')
    doc = Doc(url="https://zh.encyclopedia.example/wiki/B",
              final_url="https://zh.encyclopedia.example/wiki/B",
              status=200, body=html.encode(), content_type="text/html; charset=utf-8")
    extract(doc)
    assert doc.canonical == "https://zh.encyclopedia.example/wiki/A"


# ------------------------------------------------------------------- scripts
def test_thresholds_come_from_the_script_table():
    from liberdex.script import ALPHABETIC, HAN, script_of
    assert script_of("index") is ALPHABETIC and script_of("索引") is HAN
    assert script_of("python教程") is HAN  # any unspaced run makes it one


def test_title_bag_reads_an_unspaced_title_as_words():
    from liberdex.rank import title_bag
    assert title_bag("量子计算机 - 维基百科，自由的百科全书") >= {"量子计算机", "维基百科"}
    assert title_bag("Quantum computing - Wikipedia") == {"quantum", "computing", "wikipedia"}


def test_european_function_words_are_stopped_for_their_language():
    assert content_terms("hvordan virker en varmepumpe", "da") == ["virker", "varmepumpe"]
    assert content_terms("jak funguje tepelné čerpadlo", "cs") == ["funguje", "tepelné", "čerpadlo"]
    assert content_terms("как работает тепловой насос", "ru") == ["работает", "тепловой", "насос"]


def test_accept_language_follows_the_plan():
    from liberdex.fetch import accept_language
    assert accept_language("en") is None and accept_language("") is None
    assert accept_language("de") == {"Accept-Language": "de,en;q=0.5"}
