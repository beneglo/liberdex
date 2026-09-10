"""Recall of last resort, and the constraint that keeps its results honest.

Every test here is a defect seen on a live query rather than a
hypothetical. The query was "münchen luftqualität messbericht", and the SERP it
produced was the German Wikipedia article on the city, the German Wikipedia
article on a treaty signed there, and a consultancy report about forestry. Seven
of the ten hosts the planner named correctly, the ones that publish this report,
returned 404, a JavaScript shell, or nothing at all.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from liberdex.cache import SitemapStore
from liberdex.engine import _NO_ROUTE, ROUTE_TITLE_FIT, _specific, _title_fit
from liberdex.rank import Ranker, fold, path_fit, weights_for
from liberdex.sitemap import locale_of, parse, pick, robots_sitemaps
from liberdex.types import Doc


def _doc(url, title="", text="", description="") -> Doc:
    return Doc(url=url, final_url=url, status=200, title=title, text=text,
               description=description)


# ------------------------------------------------------------------ folding
def test_umlaut_folds_to_the_spelling_urls_use():
    """The defect: `luftqualität` scoring 0.0 against `/luftqualitaet/`.

    German transliterates ü as ue in an address, always. Every German page
    answering a German query is therefore spelled, in its own URL, in the one
    way the query never is.
    """
    assert fold("luftqualität") == "luftqualitaet"
    assert fold("münchen") == "muenchen"
    assert fold("straße") == "strasse"
    assert fold("københavn") == "koebenhavn"
    # A plain accent is dropped rather than spelled out, which is what French
    # and Spanish addresses do.
    assert fold("café") == "cafe"
    # ASCII is returned untouched, and by identity: this runs per token.
    assert fold("report") == "report"


def test_folded_slug_matches_the_query_that_names_it():
    q = [{"luftqualität", "messbericht", "münchen"}]
    hit = path_fit(q, "/muenchen-luftqualitaetsmessbericht-2025/")
    miss = path_fit(q, "/karriere/stellenangebote/team-assistenz/")
    assert hit > 0.0
    assert hit > miss * 3


# ------------------------------------------------------------------ sitemap
def test_parse_reads_both_document_types():
    urlset = (b'<?xml version="1.0"?><urlset><url><loc>https://site.example/x</loc></url>'
              b"<url><loc>https://site.example/y</loc></url></urlset>")
    locs, is_index = parse(urlset)
    assert locs == ["https://site.example/x", "https://site.example/y"]
    assert not is_index
    index = b"<sitemapindex><sitemap><loc>https://site.example/s1.xml</loc></sitemap></sitemapindex>"
    locs, is_index = parse(index)
    assert is_index and locs == ["https://site.example/s1.xml"]


def test_parse_unescapes_the_one_entity_a_sitemap_must_escape():
    body = b"<urlset><url><loc>https://site.example/x?a=1&amp;b=2</loc></url></urlset>"
    assert parse(body)[0] == ["https://site.example/x?a=1&b=2"]


def test_parse_survives_a_javascript_shell():
    """One host answers /sitemap.xml with 370 KB of its application, status 200.

    Finding no <loc> is the correct answer, and it is what lets the reader fall
    through to whatever robots.txt declared instead.
    """
    assert parse(b"<!doctype html><html><body><div id=app></div></html>") == ([], False)


def test_robots_sitemap_lines_from_another_host_are_not_ours():
    body = (b"User-agent: *\nSitemap: https://site.example/sitemap.xml\n"
            b"Sitemap: https://tracker.example.com/s.xml\n")
    assert robots_sitemaps(body, "site.example") == ["https://site.example/sitemap.xml"]


def test_pick_declines_a_catalogue_that_does_not_answer():
    """A host with nothing relevant must contribute nothing.

    Without a floor, `pick` returns a host's least irrelevant page, which is
    how a careers listing and a contact form reached a SERP about a measurement
    report.
    """
    vocabs = [{"luftqualität", "messbericht", "münchen"}]
    careers = ["https://agency.example/karriere/stellenangebote/",
               "https://agency.example/kontakt/", "https://agency.example/impressum/"]
    assert pick(careers, vocabs, 3) == []
    good = careers + ["https://agency.example/muenchen-luftqualitaetsmessbericht-2025/"]
    assert pick(good, vocabs, 3) == [
        "https://agency.example/muenchen-luftqualitaetsmessbericht-2025/"]


def test_sitemap_store_remembers_a_miss_separately_from_a_hit():
    """A host that blocks us is asked once per miss-TTL, not once per query."""
    path = os.path.join(tempfile.mkdtemp(), "sm.sqlite3")
    store = SitemapStore(path)
    assert store.get("never.example") is None          # never asked
    store.put("blocked.example", [])
    assert store.get("blocked.example") == []          # asked, has nothing to give
    store.put("good.example", ["https://good.example/a", "https://good.example/b"])
    assert store.get("good.example") == ["https://good.example/a", "https://good.example/b"]
    assert store.count() == 1                     # misses are not hosts


def test_index_children_keep_their_own_order_unless_one_really_fits():
    """The defect: one index lists twelve child sitemaps, of which `pages.xml`
    holds all 117 articles. Scored against a Dutch air-quality query,
    `construction-numbers.xml` picked up a spurious n-gram hit and was followed
    instead; five URLs came back where there were 117.
    """
    kids = ["https://portal.example/sitemap/pages.xml",
            "https://portal.example/sitemap/construction-numbers.xml",
            "https://portal.example/sitemap/users.xml"]
    vocabs = [{"amsterdam", "luchtkwaliteit", "metingen", "cijfers"}]
    # Nothing here fits convincingly, so nothing jumps the queue.
    assert pick(kids, vocabs, 2, min_fit=0.20, tie_shortest=False) == []
    # A child that genuinely names the subject still wins.
    named = kids + ["https://portal.example/sitemap/luchtkwaliteit-metingen.xml"]
    assert pick(named, vocabs, 1, min_fit=0.20, tie_shortest=False) == [
        "https://portal.example/sitemap/luchtkwaliteit-metingen.xml"]


def test_a_foreign_language_edition_is_not_the_answer():
    """One publisher puts the same report under fourteen locale prefixes and the
    slug fit cannot tell them apart: "taipei-air-quality-report" under /zh-tw/
    scored exactly as well for a German query as the German edition."""
    locs = ["https://publisher.example/zh-tw/insights/taipei-report",
            "https://publisher.example/de-de/luftqualitaet-messbericht"]
    got = pick(locs, [{"messbericht", "luftqualität"}], 3, min_fit=0.06,
               lang="de")
    assert got == ["https://publisher.example/de-de/luftqualitaet-messbericht"]


def test_locale_prefix_is_read_only_where_it_is_one():
    assert locale_of("/zh-tw/insights/taipei") == "zh"
    assert locale_of("/de/markt") == "de"
    # Not a locale: a section that happens to be two letters is rare, but a
    # long first segment is the common case and must not be misread.
    assert locale_of("/insights/report") == ""
    assert locale_of("/") == ""


# --------------------------------------------------- reading the plan's tail
def test_a_specific_address_is_told_from_a_generic_one():
    """The planner is told to fall back to a homepage when it cannot recall a
    page, so a homepage low in its list is an admission and a deep path low in
    its list is a recollection it was merely unsure of. The answer to
    "amsterdam luchtkwaliteit metingen cijfers" was the planner's thirteenth
    line, behind four homepages."""
    for generic in ("https://portal.example/", "https://www.institute.example/insights",
                    "https://www.lab.example/research/",
                    "https://www.example.com/nl-nl/research"):
        assert not _specific(generic)
    for specific in ("https://institute.example/onderzoek/luchtkwaliteit-metingen",
                     "https://lab.example/muenchen-luftqualitaetsmessbericht-2025/",
                     "https://kv.example/docs/latest/commands/scan/",
                     "https://www.db.example/docs/current/datatype-json.html"):
        assert _specific(specific)


# ------------------------------------------------------------ route hygiene
def test_route_result_whose_title_shares_nothing_is_not_fetched():
    """Wikipedia's own search, asked for a Chicago air quality report, offered
    Eric Lefkofsky and the MLS All-Star Game. Both cost a fetch and then lost."""
    vocabs = [frozenset({"chicago", "air", "quality", "report", "ozone",
                         "monitoring"})]
    assert _title_fit("Eric Lefkofsky – Wikipedia", vocabs) < ROUTE_TITLE_FIT
    assert _title_fit("2025 MLS All-Star Game", vocabs) < ROUTE_TITLE_FIT
    assert _title_fit("Chicago Air Quality Report Q2", vocabs) > ROUTE_TITLE_FIT


def test_planner_declining_a_route_is_not_a_search_for_the_word_none():
    """`R github: none` fired GitHub's code search for "none" and put six
    unrelated repositories in a German-language SERP."""
    for word in ("none", "N/A", "keine", "-"):
        assert word.strip().lower() in _NO_ROUTE


# --------------------------------------------------------- aspect coverage
def _aspect(query_vocabs, docs):
    return Ranker()._aspect_coverage(query_vocabs, docs)


def test_aspect_coverage_counts_constraints_not_similarity():
    """The defect passage selection cannot show: a 199 KB article on the city of
    München scored 0.914 from the cross-encoder for a query about its air
    quality measurements, because the window it was shown said "München" and
    nothing downstream could see the rest.
    """
    vocabs = [frozenset({"münchen", "messbericht", "luftqualität"})]
    answer = _doc("https://lab.example/muenchen-luftqualitaetsmessbericht-2025/",
                  title="München: Messbericht zur Luftqualität",
                  text="Der Messbericht zur Luftqualität in München.")
    neighbour = _doc("https://de.encyclopedia.example/wiki/München",
                     title="München",
                     text="München ist die Landeshauptstadt des Freistaats "
                          "Bayern." + " Stadt." * 400)
    got = _aspect(vocabs, [answer, neighbour])
    assert got[0] == 1.0
    assert got[1] <= 0.34
    assert got[0] - got[1] > 0.5


def test_aspect_coverage_scores_each_phrasing_separately():
    """A German page and an English page about the same thing each satisfy one
    phrasing completely and their union not at all."""
    vocabs = [frozenset({"münchen", "messbericht", "luftqualität"}),
              frozenset({"munich", "air", "quality", "report"})]
    english = _doc("https://publisher.example/munich-air-quality-report/",
                   title="Munich Air Quality Report",
                   text="Air quality monitoring in Munich.")
    assert _aspect(vocabs, [english])[0] == 1.0


def test_aspect_coverage_reads_identity_not_a_passing_mention():
    """A page that mentions the subject once, two screens down, is not about it."""
    vocabs = [frozenset({"münchen", "messbericht", "luftqualität"})]
    buried = _doc("https://agency.example/ueber-uns/",
                  title="Über uns",
                  text="Wir sind ein Unternehmen. " * 200
                       + "Messbericht Luftqualität München.")
    assert _aspect(vocabs, [buried])[0] <= 0.34


def test_aspect_coverage_weights_the_constraint_that_separates():
    """The defect: "melting point of tungsten in celsius" is four words and one
    entity. Nearly every candidate says "melting" and "point"; the page that
    does not say "tungsten" is the wrong page however much of the frame it
    covers. Unweighted, the elements data page outranked the tungsten article.
    """
    vocabs = [frozenset({"melting", "point", "tungsten", "celsius"})]
    subject = _doc("https://en.encyclopedia.example/wiki/Tungsten",
                   title="Tungsten - Wikipedia",
                   text="Tungsten is a chemical element. Its melting point is "
                        "3422 degrees Celsius, the highest of all metals.")
    frame = _doc("https://en.encyclopedia.example/wiki/Melting_points_of_the_elements",
                 title="Melting points of the elements (data page)",
                 text="Melting point in celsius and kelvin for hydrogen, "
                      "helium, lithium, beryllium, boron and carbon.")
    got = _aspect(vocabs, [subject, frame])
    assert got[0] > got[1]
    assert got[0] - got[1] > 0.3


def test_aspect_weight_is_live_in_every_intent():
    """Per-intent tables override named weights; a new one silently absent from
    the fusion for `product` would have gone unnoticed."""
    for intent in ("informational", "product", "code", "news", "reference",
                   "academic", "navigational", "local"):
        assert weights_for(intent).aspect > 0.0


def test_aspect_coverage_is_empty_rather_than_wrong_without_a_query():
    assert _aspect([], [_doc("https://agency.example/a", title="A")]).tolist() == [0.0]
    got = _aspect([frozenset({"luftqualität"})], [_doc("https://agency.example/a")])
    assert isinstance(got, np.ndarray) and got[0] == 0.0
