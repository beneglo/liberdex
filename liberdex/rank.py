"""Ranking cascade.

Three stages, each a few milliseconds over a couple of hundred passages:
  1. bm25s lexical over every passage
  2. potion-retrieval-32M static embeddings
  3. FlashRank ms-marco-MiniLM-L-12 on the top 20, scored as pages land
     (see Prescore)
Fused with query-independent priors (planner confidence, host authority,
multi-source consensus, URL and title match) into a final SERP score.
"""
from __future__ import annotations

import datetime
import math
import os
import re
import threading
import time
import unicodedata
from bisect import bisect_left
from dataclasses import dataclass, fields, replace
from functools import lru_cache
from typing import Iterable, Optional
from urllib.parse import unquote, urlsplit

import numpy as np

from .fetch import registrable
from .query import content_bag, content_terms
from .routes import ROUTES
from .script import SENT_END_BARE, SENT_END_SPACED, UNSPACED, script_of, unspaced
from .types import Doc, Plan, Result

_WORD = re.compile(r"[^\W_][\w'+#._-]*", re.UNICODE)

# Snowball stemmers, keyed by ISO 639-1. A language absent here is left
# unstemmed rather than mangled by the English rules.
STEMMER_LANGS: dict[str, str] = {
    "en": "english", "de": "german", "fr": "french", "es": "spanish",
    "pt": "portuguese", "it": "italian", "nl": "dutch", "sv": "swedish",
    "no": "norwegian", "da": "danish", "fi": "finnish", "ru": "russian",
    "hu": "hungarian", "ro": "romanian", "tr": "turkish", "ar": "arabic",
    "el": "greek", "id": "indonesian", "ta": "tamil",
}

# Stopword lists bm25s ships. Anything else gets none.
BM25_STOPWORD_LANGS = frozenset(
    {"en", "de", "fr", "es", "pt", "it", "nl", "sv", "ru"}
)

# Written without spaces; needs character-bigram segmentation, not a stemmer.
CJK_LANGS = frozenset({"zh", "ja"})

# An identifier, not a word: punctuation inside it, or a digit. `dict.get`,
# `max_slot_wal_keep_size`, `HTTP/3`, `s41586-021-03819-2`, `ENOSPC`.
_IDENTIFIER = re.compile(r"[a-z0-9][_./:-][a-z0-9]|\d")
# A bare year looks like an identifier to the rule above and is not one. It is
# a recency constraint and is scored as one, by `date_fit`.
_BARE_NUMBER = re.compile(r"^\d{1,4}$")
_QUERY_YEAR = re.compile(r"\b(19|20)\d{2}\b")
# Quarters and months are the same kind of token: a period, not a name.
_PERIOD = re.compile(
    r"^(q[1-4]|h[12]|fy\d{2,4}|[12]?\d(st|nd|rd|th)?)$", re.I)
_SENT_SPLIT = re.compile(rf"(?<={SENT_END_SPACED})\s+|(?<={SENT_END_BARE})|\n+")

PASSAGE_CHARS = 620
PASSAGE_OVERLAP = 140
MAX_PASSAGES_PER_DOC = 6
# Enough to yield MAX_PASSAGES_PER_DOC windows with overlap, and no more.
PASSAGE_SCAN_CHARS = 12_000
# Windows taken from wherever the query's words are, rather than from the top.
TARGETED_PASSAGES = 3
# Below this, a page's title, path and domain say nothing about the query.
ORPHAN_FIT = 0.02
# How much of a page counts as what the page announces itself to be. Its title,
# description and address, plus the opening: enough to carry a lede and a first
# heading, short enough that a passing mention two screens down is not identity.
ASPECT_HEAD = 800
# Share of a term's character n-grams that must appear in that identity for the
# constraint to count as met. Lower and unrelated German compounds satisfy each
# other; at 1.0 no compound satisfies its own head word.
ASPECT_HIT = 0.55
# How many windows of one document the cross-encoder gets to see.
CE_WINDOWS_PER_DOC = 2
# How long the ranker waits, in total, for scores still being computed for
# pages that landed in the last moments before it ran.
PRESCORE_WAIT = 0.12

# Hosts that are near-always the best available source for their subject.
AUTHORITY: dict[str, float] = {
    "wikipedia.org": 0.16, "docs.python.org": 0.16, "developer.mozilla.org": 0.16,
    "stackoverflow.com": 0.13, "github.com": 0.10, "arxiv.org": 0.12,
    "nih.gov": 0.12, "who.int": 0.12, "nature.com": 0.10, "science.org": 0.10,
    "ieee.org": 0.08, "acm.org": 0.08, "semanticscholar.org": 0.08,
    "britannica.com": 0.10, "reuters.com": 0.10, "apnews.com": 0.10,
    "bbc.com": 0.09, "nytimes.com": 0.08, "theguardian.com": 0.07,
    "wsj.com": 0.07, "ft.com": 0.07, "bloomberg.com": 0.07,
    "python.org": 0.10, "rust-lang.org": 0.10, "go.dev": 0.10,
    "kubernetes.io": 0.09, "postgresql.org": 0.10, "redis.io": 0.08,
    "cloudflare.com": 0.07, "aws.amazon.com": 0.08, "cloud.google.com": 0.08,
    "learn.microsoft.com": 0.08, "docs.oracle.com": 0.07, "npmjs.com": 0.06,
    "pypi.org": 0.07, "crates.io": 0.07, "readthedocs.io": 0.06,
    "news.ycombinator.com": 0.05, "reddit.com": 0.03, "medium.com": -0.02,
    "pinterest.com": -0.15, "quora.com": -0.05, "facebook.com": -0.10,
    "w3schools.com": -0.03, "geeksforgeeks.org": -0.02,
}

# Structural authority: a class of publisher, not a named site. Applied only
# when the curated table has nothing to say about the host.
#
# Every country runs the same convention under a different label, so listing
# only the English-speaking ones would make the rule mean "an American or
# British government source". These are registry-operated second-level domains
# with a published eligibility rule, which is what makes them evidence: you
# cannot buy one. Suffixes are matched whole, so `.gov` does not match
# `x.gov.uk` and each needs its entry.
_TLD_AUTHORITY: tuple[tuple[str, float], ...] = (
    # supranational
    (".europa.eu", 0.10), (".int", 0.10),
    # government
    (".gov", 0.10), (".mil", 0.10),
    (".gov.uk", 0.10), (".gov.au", 0.10), (".govt.nz", 0.10), (".gc.ca", 0.10),
    (".gv.at", 0.10), (".bund.de", 0.10), (".admin.ch", 0.10),
    (".gouv.fr", 0.10), (".fgov.be", 0.10), (".overheid.nl", 0.10),
    (".gob.es", 0.10), (".gob.mx", 0.10), (".gob.ar", 0.10), (".gob.cl", 0.10),
    (".gob.pe", 0.10), (".gov.br", 0.10), (".gov.pt", 0.10), (".gov.it", 0.10),
    (".gov.pl", 0.10), (".gov.gr", 0.10), (".gov.ie", 0.10), (".gov.za", 0.10),
    (".gov.in", 0.10), (".gov.sg", 0.10), (".go.jp", 0.10), (".go.kr", 0.10),
    (".go.id", 0.10), (".go.th", 0.10),
    # academic
    (".edu", 0.07), (".ac.uk", 0.07), (".edu.au", 0.07), (".ac.nz", 0.07),
    (".ac.at", 0.07), (".ac.jp", 0.07), (".ac.kr", 0.07), (".ac.in", 0.07),
    (".ac.za", 0.07), (".ac.il", 0.07), (".edu.cn", 0.07), (".edu.sg", 0.07),
    (".edu.hk", 0.07), (".edu.tw", 0.07), (".edu.br", 0.07), (".edu.mx", 0.07),
    (".edu.pl", 0.07),
)


def authority(host: str) -> float:
    """Curated bonus for the host or any parent of it, else its TLD class.

    Suffix walking lets one entry cover a whole family: `wikipedia.org` scores
    every language edition, `medium.com` still applies its penalty to
    `<publication>.medium.com`.
    """
    parts = host.split(".")
    for i in range(len(parts) - 1):
        v = AUTHORITY.get(".".join(parts[i:]))
        if v is not None:
            return v
    for suffix, v in _TLD_AUTHORITY:
        # `gov.uk` is itself a site, not only a suffix others sit under.
        if host.endswith(suffix) or host == suffix[1:]:
            return v
    return 0.0


SPAM = re.compile(
    r"(coupon|casino|escort|\bporn|viagra|crypto-?giveaway|free-?download|"
    r"torrent|crack|keygen|essay-?writing)", re.I
)

_TRACKING = re.compile(r"^(utm_|fbclid|gclid|msclkid|mc_|ref_?$|source$|_ga)", re.I)


def normalize_url(url: str) -> str:
    """Canonical form for dedup: drop tracking params, fragment, trailing slash."""
    try:
        p = urlsplit(url)
    except ValueError:
        return url
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if p.query:
        keep = [
            kv for kv in p.query.split("&")
            if kv and not _TRACKING.match(kv.split("=", 1)[0])
        ]
        query = "&".join(sorted(keep))
    else:
        query = ""
    # Percent-encoded and literal forms of one path are one page.
    try:
        path = unquote(p.path)
    except Exception:
        path = p.path
    path = path.rstrip("/") or "/"
    scheme = "https"
    port = f":{p.port}" if p.port and p.port not in (80, 443) else ""
    return f"{scheme}://{host}{port}{path}" + (f"?{query}" if query else "")


def _content_word(t: str) -> bool:
    """Long enough to be evidence rather than noise, by the token's script."""
    return len(t) >= script_of(t).min_word


def _dedup_key(d: Doc) -> str:
    """The address a page says it lives at, when that claim is a sibling.

    A wiki serves two spellings of one title as two URLs with one canonical,
    and their bodies differ in every character the spellings disagree on, so
    the prose fingerprint cannot see that they are one article. The canonical
    is trusted only within its own site and at the same path depth: a CMS that
    points every report at its listing page would otherwise fold distinct
    pages into one.
    """
    url = d.final_url or d.url
    if not d.canonical:
        return url
    try:
        c, u = urlsplit(d.canonical), urlsplit(url)
    except ValueError:
        return url
    cpath, upath = c.path.rstrip("/"), u.path.rstrip("/")
    if (not cpath or c.scheme not in ("http", "https")
            or registrable(c.hostname or "") != registrable(u.hostname or "")
            or cpath.count("/") != upath.count("/")):
        return url
    return d.canonical


def _cjk_grams(tok: str) -> list[str]:
    """Unigrams plus character bigrams for a run of Han or kana.

    Chinese and Japanese are written without spaces, so a whitespace
    tokeniser sees one enormous token per phrase and BM25, title overlap and
    anchor matching all go silent at once. Character bigrams are the standard
    answer and need no dictionary or model.
    """
    out: list[str] = []
    run: list[str] = []
    latin: list[str] = []

    def flush_run() -> None:
        if not run:
            return
        out.extend(run)
        out.extend(run[i] + run[i + 1] for i in range(len(run) - 1))
        run.clear()

    def flush_latin() -> None:
        if latin:
            out.append("".join(latin))
            latin.clear()

    for ch in tok:
        if UNSPACED.match(ch):
            flush_latin()
            run.append(ch)
        else:
            flush_run()
            latin.append(ch)
    flush_run()
    flush_latin()
    return out


# In prose `-` and `.` bind a token together (`re-encoding`, `dict.get`) and
# `tokens()` keeps them whole, which is what makes an identifier query match.
# In a URL they are the space character.
_SLUG = re.compile(r"[-_+./:%~]+")
# Path furniture: in half the URLs on the web and in no query.
_PATH_STOP = frozenset({
    "html", "htm", "php", "asp", "aspx", "jsp", "cgi", "shtml", "index",
    "www", "en", "docs", "doc", "page", "id", "amp",
})


def path_terms(path: str) -> tuple[set[str], set[str]]:
    """A URL path as two bags of words: whole slugs, and the words in them.

    Scored separately rather than pooled because the fit is an F1 and pooling
    costs precision: a long slug split into ten pieces makes the one that
    matches the query count for a tenth of the address instead of a fifth.
    Whichever spelling the query is written in wins.
    """
    whole: set[str] = set()
    for t in tokens(path):
        if t in _PATH_STOP:
            continue
        whole.add(t)
        # `name.html` is the identifier plus a file extension.
        base, _, ext = t.rpartition(".")
        if base and ext in _PATH_STOP:
            whole.add(base)
    split = {p for t in whole for p in _SLUG.split(t)
             if len(p) >= 2 and p not in _PATH_STOP}
    return whole, split


def path_fit(vocabs: list[set[str]], path: str) -> float:
    whole, split = path_terms(path)
    return max(best_f1(vocabs, whole), best_f1(vocabs, split))


def title_bag(title: str, latin_min: int = 1) -> set[str]:
    """A title as a bag the query's content vocabulary can be scored against.

    An unspaced title is read into words the same way the query was; scoring
    words against the bigram soup `tokens` makes of it never matches a whole
    word.
    """
    if unspaced(title):
        return content_bag(title)
    return {t for t in tokens(title) if len(t) >= latin_min}


def tokens(s: str) -> list[str]:
    out: list[str] = []
    for t in _WORD.findall(s.lower()):
        if UNSPACED.search(t):
            out.extend(_cjk_grams(t))
        else:
            out.append(t)
    return out


def segmented(s: str) -> str:
    """Whitespace-joined tokens, so a whitespace tokeniser sees CJK words."""
    return " ".join(tokens(s))


@dataclass(slots=True)
class Passage:
    doc_idx: int
    text: str


def split_passages(text: str, limit: int = MAX_PASSAGES_PER_DOC) -> list[str]:
    """Overlapping windows on sentence boundaries; first window carries the lede."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= PASSAGE_CHARS:
        return [text]
    sents = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    out: list[str] = []
    buf: list[str] = []
    size = 0
    for s in sents:
        buf.append(s)
        size += len(s) + 1
        if size >= PASSAGE_CHARS:
            out.append(" ".join(buf))
            if len(out) >= limit:
                return out
            # Overlap: keep trailing sentences up to PASSAGE_OVERLAP chars.
            back, keep = 0, []
            for prev in reversed(buf):
                back += len(prev)
                keep.append(prev)
                if back >= PASSAGE_OVERLAP:
                    break
            buf = list(reversed(keep))
            size = sum(len(x) + 1 for x in buf)
    if buf and len(out) < limit:
        tail = " ".join(buf)
        if not out or tail not in out[-1]:
            out.append(tail)
    return out


def anchor_terms(url: str) -> list[str]:
    """Search terms implied by a URL fragment.

    A deep link is the planner being precise: `stdtypes.html#dict.get` says
    the answer is one section of a very long page. Both spellings are tried,
    since a fragment is sometimes the literal symbol (`#dict.get`) and
    sometimes a slug of the heading (`#DATATYPE-JSONPATH`).
    """
    try:
        frag = urlsplit(url).fragment.strip()
    except ValueError:
        return []
    if not frag or len(frag) > 120:
        return []
    frag = unquote(frag).lower()
    words = [w for w in re.split(r"[-_+.:/]+", frag) if len(w) >= 3]
    out = [frag]
    if len(words) > 1:
        out.append(" ".join(words))
    # A single short word out of a fragment is not an anchor: "dict" would
    # match the table of contents.
    out.extend(w for w in words if len(w) >= 8 and w not in out)
    return [a for a in out if len(a) >= 6][:4]


def pin(results: list[Result]) -> list[Result]:
    """Rows carrying a fixed position take it; the rest keep their order."""
    fixed = [r for r in results if r.debug.get("rank")]
    if not fixed:
        return results
    out = [r for r in results if not r.debug.get("rank")]
    for r in sorted(fixed, key=lambda r: r.debug["rank"]):
        out.insert(min(r.debug["rank"] - 1, len(out)), r)
    return out


def select_passages(
    text: str, terms: list[str], limit: int = MAX_PASSAGES_PER_DOC,
    anchors: Optional[list[str]] = None,
) -> list[str]:
    """The lede, plus the windows where the query's words actually appear.

    The first N windows of a document are fine for an article and wrong for a
    reference page, where the section that answers may be 100 000 characters
    in. Locating the query terms first is a scan of the raw string in C,
    cheaper than the sentence split it feeds.
    """
    anchors = anchors or []
    if len(text) <= PASSAGE_SCAN_CHARS or not (terms or anchors):
        return split_passages(text[:PASSAGE_SCAN_CHARS], limit)
    head = split_passages(text[:PASSAGE_SCAN_CHARS], max(1, limit - TARGETED_PASSAGES))
    low = text.lower()
    # The section the planner pointed at goes in first and unconditionally:
    # it is a stronger statement than term frequency.
    for a in anchors:
        i = low.find(a)
        if i >= 0:
            lo = max(0, i - PASSAGE_OVERLAP)
            head.insert(0, text[lo:lo + PASSAGE_CHARS + PASSAGE_OVERLAP])
            break
    bins: dict[int, set[int]] = {}
    for j, t in enumerate(terms):
        start, found = PASSAGE_SCAN_CHARS, 0
        while found < 400:
            i = low.find(t, start)
            if i < 0:
                break
            bins.setdefault(i // PASSAGE_CHARS, set()).add(j)
            start, found = i + len(t), found + 1
    if not bins:
        return head
    # Prefer windows covering the most distinct query terms, earliest first.
    ranked = sorted(bins.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    for b, hit in ranked[:TARGETED_PASSAGES]:
        if len(hit) < 2 and len(terms) > 1:
            continue
        lo = max(0, b * PASSAGE_CHARS - PASSAGE_OVERLAP)
        head.append(text[lo:lo + PASSAGE_CHARS + 2 * PASSAGE_OVERLAP])
    return head


# What separates two windows of one document in the joined `snippet` string.
# Visible as a gap rather than as prose, so a consumer cannot read across the
# cut and take two halves of different sentences for one.
WINDOW_JOIN = " [...] "
# Below this a window is too short to carry a sentence with its context, so the
# window count is reduced rather than the budget sliced any thinner.
MIN_WINDOW_CHARS = 250


def _occurrences(low: str, terms: list[str], cap: int = 1024) -> list[list[int]]:
    """Sorted start offsets of each term, terms that never appear dropped."""
    out: list[list[int]] = []
    for t in terms:
        pos: list[int] = []
        at = low.find(t)
        while at >= 0 and len(pos) < cap:
            pos.append(at)
            at = low.find(t, at + len(t))
        if pos:
            out.append(pos)
    return out


# How far back a window may reach for the sentence it started inside. Snapping
# outward rather than inward: pulling the start forward to the next sentence
# would drop the text between, and reaching back cannot lose a hit since the
# old window is a subset of the new one.
SENTENCE_LOOKBACK = 80
# How far forward the end may reach for the full stop of the sentence it cut,
# so a window does not stop one word short of the fact it was chosen for. The
# character budget holds because `select_windows` takes the expected extension
# off each window's nominal width before it is placed.
SENTENCE_LOOKAHEAD = 60
# A sentence ends at .!? followed by whitespace, or on a wiki by citation
# markers and then whitespace: "native to Mexico.[2] References".
_SENT_BREAK = re.compile(rf"{SENT_END_SPACED}(?:\[\d+\])*\s|{SENT_END_BARE}")


def _snap(text: str, start: int, width: int) -> tuple[str, int, int]:
    """`width` characters from `start`, on word boundaries, sentence-aligned.

    Both ends are snapped inward to a word, then outward to a sentence when one
    begins shortly before the start or ends shortly after the end, so the window
    reads as prose rather than resuming mid-clause and does not stop one word
    short of the fact it was chosen for.
    """
    end = min(len(text), start + width)
    if start > 0:
        sp = text.find(" ", start, start + 40)
        if sp >= 0:
            start = sp + 1
    if end < len(text):
        sp = text.rfind(" ", start, end)
        if sp > start:
            end = sp
        m = _SENT_BREAK.search(text, end, min(len(text), end + SENTENCE_LOOKAHEAD))
        if m:
            end = m.end()
    if start > 0:
        breaks = list(_SENT_BREAK.finditer(text, max(0, start - SENTENCE_LOOKBACK), start))
        if breaks:
            start = breaks[-1].end()
    return text[start:end].strip(), start, end


# What a reference list, a nav bar or a table of links looks like once the
# markup is gone: citation arrows, "Retrieved", pipes, bracketed edit links.
_LISTY = re.compile(r"↑|\bRetrieved\b|\bArchived\b|\[edit\]|\s\|\s|»|\^\s")
LEDE_BONUS = 0.5


def _prose_factor(window: str) -> float:
    """1.0 for running text, down to 0.4 for a window that is mostly a list."""
    if not window:
        return 1.0
    listy = len(_LISTY.findall(window))
    # Sentences: a full stop followed by a space and a capital or a digit.
    sentences = len(re.findall(r"[.!?]\s+[A-ZÄÖÜ0-9]", window))
    if listy >= 3 and sentences <= 1:
        return 0.4
    if listy >= 3:
        return 0.7
    letters = sum(c.isalpha() for c in window)
    if letters < 0.55 * len(window):
        return 0.7
    return 1.0


def select_windows(body: str, seed: str, terms: list[str], max_chars: int,
                   windows: int = 1) -> tuple[list[tuple[str, float]], bool, bool]:
    """The `windows` densest regions of `body`, sharing one character budget.

    Query terms name the entity, and in a reference article the entity is
    named throughout, so term density alone cannot say where on the page the
    answer sits, and the answer is usually well past the first screen.
    Splitting one budget across several separated windows covers more of the
    page for the same number of tokens handed to the consumer.

    Each window carries the fraction of the query's distinct content terms
    that appear inside it, in 0..1. Coverage, not density: a window scoring
    1.0 contains every term the query supplied, whichever page it came from,
    so a consumer can rank or drop passages on it.

    Returns (windows, truncated_before, truncated_after), the flags saying
    whether text was dropped ahead of the first window and after the last.
    """
    text = re.sub(r"\s+", " ", body or "").strip()
    if not text:
        seed = re.sub(r"\s+", " ", seed or "").strip()[:max_chars]
        return ([(seed, 0.0)] if seed else []), False, False
    if len(text) <= max_chars:
        return [(text, 1.0)], False, False
    n = max(1, min(windows, max_chars // MIN_WINDOW_CHARS))
    # Each window may grow by up to SENTENCE_LOOKAHEAD to finish its last
    # sentence; half of that is the expected growth, and it comes off the
    # nominal width so the shipped total holds at `max_chars`.
    per = max(MIN_WINDOW_CHARS // 2, max_chars // n - SENTENCE_LOOKAHEAD // 2)
    wanted = [t for t in terms if _content_word(t)]
    offs = _occurrences(text.lower(), wanted)
    if not offs:
        seg, _, end = _snap(text, 0, per)
        return [(seg, 0.0)], False, end < len(text)

    # `hits` counts how many of the query's terms occur in a window, not how
    # many times, and the denominator is the query's count, not the page's:
    # normalising by what the page uses would make 1.0 mean "everything this
    # page had", which cannot be compared between two pages.
    total = len(wanted) or 1
    # The window is chosen on a different sum. A query names its entity and
    # then the one detail that picks the fact out, and on the page that
    # answers, the entity is on every line while the detail is on one. So
    # each term weighs what it can tell about where on this page: the inverse
    # log of how often the page uses it. The coverage returned to the
    # consumer stays the plain fraction.
    weight = [1.0 / (1.0 + math.log2(len(pos))) for pos in offs]
    # Score every window start, best first; ties go to the earlier window,
    # which is where a reference page states the plain facts.
    step = max(1, per // 2)
    # The last stride rarely lands on `len(text) - per`, and without that start
    # the final `step` characters of the document are in no window at all.
    starts = list(range(0, len(text) - per + 1, step))
    if starts[-1] != len(text) - per:
        starts.append(len(text) - per)
    # A window that reads as a list of citations, links or navigation is
    # discounted; the opening window, where an article states its subject
    # and its plain fact, gets the tie.
    scored = []
    for i in starts:
        present = [k for k, pos in enumerate(offs)
                   if bisect_left(pos, i + per) > bisect_left(pos, i)]
        score = (sum(weight[k] for k in present) * _prose_factor(text[i:i + per])
                 + (LEDE_BONUS if i == 0 else 0.0))
        scored.append((score, -i, len(present)))
    scored.sort()
    picked: list[tuple[int, int]] = []
    for _, neg_i, hits in reversed(scored):
        if len(picked) >= n:
            break
        i = -neg_i
        # `per` apart, so no two windows can show the same sentence twice.
        if all(abs(i - p) >= per for p, _ in picked):
            picked.append((i, hits))
    picked.sort()

    out: list[tuple[str, float]] = []
    first, last = 0, 0
    for i, hits in picked:
        seg, s, e = _snap(text, i, per)
        if not seg:
            continue
        if not out:
            first = s
        last = max(last, e)
        out.append((seg, round(hits / total, 4)))
    if not out:
        return [(text[:max_chars], 0.0)], False, False
    return out, first > 0, last < len(text)


# Character n-grams, the language-neutral stand-in for a decompounder.
#
# German glues its nouns together, and a token comparison cannot see inside
# the glue: "Strompreise" and "Strompreisentwicklung" are the same subject and
# share no token, so every overlap score in this module would read zero. The
# same defect waits in Dutch, the Scandinavian languages, Finnish and
# Hungarian, and it costs English the morphology of "index" and "indexing".
#
# McNamee & Mayfield (Information Retrieval 7:73-97, 2004) compared n-gram
# tokenisation with dictionary decompounding across eight European languages
# and found n=4 or 5 recovers most of the gain with no lexicon. The width, the
# shortest token that gets grams at all, and whether the ends are anchored are
# properties of the script and live in `script.Script`.
#
# Soft overlap never outranks a literal one: held under 1.0 so that when both
# fire the exact match still wins.
_SOFT = 0.8


# Every language that writes diacritics has a convention for spelling them
# without, and URLs follow it: a German slug is `luftqualitaet`, a Danish one
# `koebenhavn`. Folding both sides to that ASCII form lets a query reach the
# slug of the page answering it. The digraph expansions come first, because
# stripping the diaeresis off `ü` yields `u` and the convention is `ue`;
# whatever is left is a plain accent its own language drops.
_FOLD = str.maketrans({
    "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
    "æ": "ae", "ø": "oe", "å": "aa", "œ": "oe",
})


@lru_cache(maxsize=65536)
def fold(tok: str) -> str:
    """A token as the web would spell it in a URL. ASCII-only input is itself."""
    if tok.isascii():
        return tok
    t = unicodedata.normalize("NFC", tok).translate(_FOLD)
    if t.isascii():
        return t
    return "".join(
        c for c in unicodedata.normalize("NFKD", t)
        if not unicodedata.combining(c)
    )


@lru_cache(maxsize=32768)
def _grams(tok: str) -> frozenset[str]:
    """A token's character n-grams, anchored so prefixes and suffixes count.

    The anchors are what make "preise" match the tail of a compound rather
    than merely appear inside it. Folded first, so two spellings of one word
    share every gram.
    """
    tok = fold(tok)
    sc = script_of(tok)
    if len(tok) < sc.gram_min:
        return frozenset()
    t = f"^{tok}$" if sc.anchored else tok
    n = sc.gram_n
    if len(t) <= n:
        return frozenset((t,))
    return frozenset(t[i:i + n] for i in range(len(t) - n + 1))


def gram_set(bag: Iterable[str]) -> frozenset[str]:
    out: set[str] = set()
    for t in bag:
        out |= _grams(t)
    return frozenset(out)


@lru_cache(maxsize=4096)
def _gram_set_cached(bag: frozenset[str]) -> frozenset[str]:
    return gram_set(bag)


def _grams_of(bag) -> frozenset[str]:
    """Gram set of a bag, cached when the bag is frozen.

    The query side of a comparison is the same few vocabularies for every
    link on a page, so it is built once; the field side is a fresh anchor
    each time.
    """
    if type(bag) is frozenset:
        return _gram_set_cached(bag)
    return gram_set(bag)


def _dice(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    recall, precision = inter / len(a), inter / len(b)
    return recall ** 0.75 * precision ** 0.25


def best_f1(vocabs: list[set[str]], field: set[str]) -> float:
    """Best fit against the query or any single expansion, never their union."""
    if not vocabs or not field:
        return 0.0
    # One gram set for the field, reused across every vocabulary.
    fg = gram_set(field)
    return max((_f1(v, field, fg) for v in vocabs), default=0.0)


def _f1(qset: set[str], field: set[str],
        field_grams: Optional[frozenset[str]] = None) -> float:
    """Overlap as F1, not recall.

    Plain recall rewards length: a page whose title is a whole paragraph
    covers every query term by accident. F1 makes a short exact title win.

    Scored on whole tokens and on character n-grams, and the better of the
    two is taken, so the n-gram pass can only rescue a pair the token pass
    scored too low, never demote one it scored well.
    """
    if not qset or not field:
        return 0.0
    exact = 0.0
    inter = len(qset & field)
    if inter:
        recall = inter / len(qset)
        precision = inter / len(field)
        # Recall-dominant with a precision penalty. Full F1 over-punishes a
        # long but legitimate documentation title.
        exact = recall ** 0.75 * precision ** 0.25
        # The soft score is capped at _SOFT, so past it the n-gram pass cannot
        # change the answer.
        if exact >= _SOFT:
            return exact
    # A compound is a whole query word with more attached
    # ("luftqualitaetsbericht" answers "bericht"), and dice on 5-grams gives
    # that pair less credit than two short exact matches. Whole-word
    # containment counts the compound as a hit at a discount, both ways.
    compound = _compound_f1(qset, field)
    qg = _grams_of(qset)
    if not qg:
        return max(exact, compound)
    fg = field_grams if field_grams is not None else gram_set(field)
    return max(exact, compound, _SOFT * _dice(qg, fg))


# A query word must be long enough (`Script.compound_min`) before being found
# inside a field word is evidence rather than coincidence: "art" is inside
# "smart", "markt" is not inside anything by accident. A field word is a slug
# piece, a narrower vocabulary, so it may be a little shorter. Two words that
# merely share a constituent ("stadtbibliothek", "saatgutbibliothek") are not
# a hit. What a compound hit is worth against an exact one:
_COMPOUND = 0.85


def _compound_f1(qset: set[str], field: set[str]) -> float:
    """Recall-dominant F1 where a word inside a longer word counts as a hit."""
    if not qset or not field:
        return 0.0
    qf = [(t, fold(t)) for t in qset]
    ff = [(t, fold(t)) for t in field]
    q_hit = 0.0
    f_hit: dict[str, float] = {}
    for t, tf in qf:
        best = 0.0
        for u, uf in ff:
            if tf == uf:
                w = 1.0
            elif len(tf) >= script_of(tf).compound_min and len(uf) > len(tf) and tf in uf:
                w = _COMPOUND
            elif len(uf) >= script_of(uf).compound_field_min and len(tf) > len(uf) and uf in tf:
                w = _COMPOUND
            else:
                continue
            best = max(best, w)
            f_hit[u] = max(f_hit.get(u, 0.0), w)
        q_hit += best
    if not q_hit:
        return 0.0
    recall = q_hit / len(qset)
    precision = min(1.0, sum(f_hit.values()) / len(field))
    return recall ** 0.75 * precision ** 0.25


def host_fit(host: str, qset: set[str]) -> float:
    """How much of the site's own name the query accounts for.

    A domain named after the thing being asked about is the strongest
    evidence of a primary source available without a curated list:
    nobelprize.org for a Nobel question, numpy.org for a numpy one. Matching
    is substring because domains run their words together.

    Precision matters more than usual here. Requiring the query terms to
    cover most of the label is what stops "python" from anointing
    python-tutorials-for-beginners.example as an official source.
    """
    label = registrable(host).rsplit(".", 1)[0]
    label = label.replace("-", "").replace(".", "").replace("_", "")
    if len(label) < 3 or not qset:
        return 0.0
    terms = [t for t in qset if len(t) >= 3]
    if not terms:
        return 0.0
    hits = [t for t in terms if t in label]
    if not hits:
        return 0.0
    recall = len(hits) / len(terms)
    precision = min(1.0, sum(len(t) for t in hits) / len(label))
    return recall ** 0.6 * precision ** 0.9


def _minmax(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return a
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


class Models:
    """Lazily loaded, process-wide, thread-safe. Call preload() at startup."""

    _lock = threading.Lock()
    _embedder = None
    _reranker = None
    _stemmers: dict[str, object] = {}

    @classmethod
    def embedder(cls):
        if cls._embedder is None:
            with cls._lock:
                if cls._embedder is None:
                    try:
                        from model2vec import StaticModel
                        cls._embedder = StaticModel.from_pretrained(
                            os.environ.get("LIBERDEX_EMBED_MODEL",
                                           "minishlab/potion-retrieval-32M")
                        )
                    except Exception:
                        cls._embedder = False
        return cls._embedder

    @classmethod
    def reranker(cls):
        if cls._reranker is None:
            with cls._lock:
                if cls._reranker is None:
                    try:
                        from flashrank import Ranker
                        cls._reranker = Ranker(
                            # Twelve layers, not two. The reranker is off the critical
                            # path (see Prescore), so it can afford to read.
                            model_name=os.environ.get(
                                "LIBERDEX_RERANK_MODEL", "ms-marco-MiniLM-L-12-v2"
                            ),
                            cache_dir=os.environ.get(
                                "LIBERDEX_MODEL_DIR",
                                os.path.expanduser("~/.cache/liberdex/flashrank"),
                            ),
                        )
                    except Exception:
                        cls._reranker = False
        return cls._reranker

    @classmethod
    def stemmer(cls, lang: str = "en"):
        lang = (lang or "en").lower()
        name = STEMMER_LANGS.get(lang)
        if name is None:
            return False  # no stemmer beats the wrong stemmer
        cached = cls._stemmers.get(name, None)
        if cached is None:
            with cls._lock:
                cached = cls._stemmers.get(name, None)
                if cached is None:
                    try:
                        import Stemmer
                        cached = Stemmer.Stemmer(name)
                    except Exception:
                        cached = False
                    cls._stemmers[name] = cached
        return cached

    _executor = None

    @classmethod
    def executor(cls):
        """One worker: the cross-encoder is CPU-bound and onnxruntime already
        uses every core, so a second thread would only contend."""
        if cls._executor is None:
            with cls._lock:
                if cls._executor is None:
                    from concurrent.futures import ThreadPoolExecutor
                    cls._executor = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="liberdex-ce")
        return cls._executor

    @classmethod
    def preload(cls) -> None:
        cls.embedder()
        cls.reranker()
        cls.stemmer("en")


# A result has to be about the query, not merely reachable from it. The signals
# behind this are absolute (a cross-encoder probability and a cosine), not
# min-maxed, so the threshold means the same thing on every query and the SERP
# can end early instead of padding itself.
MIN_RELEVANCE = 0.18
# Never fewer than this many rows while candidates remain. There is no floor
# under it: the relevance reading is 0.0 on some pages that do hold the answer,
# so `relevance` is reported on every row and the caller makes that cut.
MIN_RESULTS = 3
# What the cross-encoder reads of a page: the title, then the lede and the
# window where the query's words cluster. Two windows, because a single
# excerpt is a length bias rather than a relevance judgement (see rank()).
CE_LEDE_CHARS = 600
CE_WINDOW_CHARS = 900


def ce_windows(terms: list[str], doc: Doc) -> list[str]:
    """The texts the cross-encoder scores for one page, from the page alone."""
    body = doc.text or ""
    lead = doc.title[:140] + ". " if doc.title else ""
    picked: list[str] = []
    lede = body[:CE_LEDE_CHARS].strip() or (doc.description or "").strip()
    if lede:
        picked.append(lede)
    if body and terms:
        parts, _, _ = select_windows(body, "", terms, CE_LEDE_CHARS, 1)
        for w, _sc in parts:
            w = w.strip()
            if w and w[:80] not in lede:
                picked.append(w)
                break
    if not picked:
        cand = doc.candidate
        fallback = " ".join(x for x in (cand.title if cand else "",
                                        cand.snippet if cand else "") if x)
        if fallback.strip():
            picked.append(fallback.strip())
    return [lead + w[:CE_WINDOW_CHARS] for w in picked[:CE_WINDOWS_PER_DOC]]


def query_terms(query: str, lang: str = "en") -> list[str]:
    """The stopword-filtered content vocabulary, as rank() derives it."""
    out = [t.strip('"').lower() for t in content_terms(query, lang)]
    return [t for t in dict.fromkeys(out) if _content_word(t)][:10]


class Prescore:
    """Cross-encoder scores computed while pages are still landing.

    The fetch wave is seconds of waiting on the network with the CPU idle,
    and a page can be scored the moment it is extracted. Scoring the head of
    the list after the last page arrives would put the reranker on the
    critical path and cap it at a model small enough to score twenty pages in
    a hundred milliseconds. Scored as they land, nearly every score is ready
    when the ranker runs and the model can be one that actually reads.
    """
    def __init__(self, ranker: "Ranker", query: str) -> None:
        self.ranker = ranker
        self.query = query
        self._terms: dict[str, list[str]] = {}
        self._futures: dict[str, object] = {}

    def terms(self, lang: str) -> list[str]:
        lang = lang or "en"
        t = self._terms.get(lang)
        if t is None:
            t = self._terms[lang] = query_terms(self.query, lang)
        return t

    def submit(self, doc: Doc, lang: str = "en") -> None:
        key = doc.final_url or doc.url
        if not doc.ok or key in self._futures or not Models.reranker():
            return
        terms = self.terms(lang)
        try:
            self._futures[key] = Models.executor().submit(
                self.ranker._ce_best, self.query, ce_windows(terms, doc))
        except RuntimeError:
            pass

    def get(self, doc: Doc, wait: float = 0.0) -> Optional[float]:
        """The score if it is in, or arrives within `wait` seconds."""
        fut = self._futures.get(doc.final_url or doc.url)
        if fut is None:
            return None
        try:
            return fut.result(timeout=wait)  # type: ignore[union-attr]
        except Exception:
            return None

    def close(self) -> None:
        for f in self._futures.values():
            f.cancel()  # type: ignore[union-attr]


@dataclass(slots=True)
class Weights:
    """Fusion weights. The defaults are the informational intent's."""

    ce: float = 1.00          # cross-encoder relevance
    lex: float = 0.42         # bm25 over passages
    emb: float = 0.38         # static-embedding cosine
    title: float = 0.30       # query terms in the title
    url: float = 0.18         # query terms in the URL path
    # The site is named after what was asked about. Small but global: the only
    # evidence of a primary source that does not come from a curated list.
    host: float = 0.24
    # Fraction of the query's content words the document contains anywhere. A
    # tiebreaker, not a driver: an unmatched word is sometimes a synonym.
    covered: float = 0.18
    # Share of the query's distinct constraints the page announces itself to
    # be about, read off its title, address and opening rather than off a
    # passage chosen for containing the query. Large, because it is the only
    # signal a passage cannot launder, and a SERP full of plausible neighbours
    # is made of pages matching one constraint emphatically and the rest not
    # at all. Higher starts preferring a verbose page that mentions every
    # constraint to the canonical one about the subject.
    aspect: float = 0.28
    # IDF-weighted coverage of the query's identifier-shaped terms. The largest
    # non-cross-encoder term on purpose: on a long-tail query the literal
    # token is the entire question.
    rare: float = 0.50
    # Planner and route confidence. Good evidence for the site, weaker for
    # which page on it.
    prior: float = 0.34
    authority: float = 1.00   # curated host bonus, already small-valued
    consensus: float = 0.14   # proposed by N independent sources
    fresh: float = 0.00       # set per intent
    thin: float = 0.22        # penalty for an empty or thin page
    dead: float = 0.30        # penalty for a page that did not return 200
    root: float = 0.10        # penalty for a bare homepage on a topical query
    # Penalty when nothing about the page's identity (title, path, domain)
    # shares a word with the query or any expansion.
    orphan: float = 0.35
    # The page belongs to the period the query named. Non-zero only when the
    # query names one, and then it is the constraint the user wrote down.
    date: float = 0.45


INTENT_WEIGHTS: dict[str, dict[str, float]] = {
    "news": {"fresh": 0.45, "authority": 0.8, "ce": 0.9},
    "academic": {"prior": 0.45, "authority": 1.1},
    # On a programming task the project's own documentation is the answer and
    # the encyclopedia entry about the project is not, so the host bonus is
    # turned down to a tie-breaker.
    "code": {"lex": 0.52, "url": 0.24, "authority": 0.55},
    "navigational": {"url": 0.5, "title": 0.45, "ce": 0.7, "prior": 0.5,
                     "host": 0.85},
    "reference": {"authority": 1.2},
    # "best X", "X vs Y", "is X worth it". Neither the encyclopedia entry for
    # the category nor the vendor's marketing answers a buying question, so
    # the host bonus stops being a driver. What decides is whether the page
    # is about the comparison and recent enough for the products to exist.
    "product": {"authority": 0.45, "prior": 0.28, "fresh": 0.20, "ce": 1.15},
}


# ms-marco-MiniLM and potion-retrieval are English models. On Latin-script
# European text they degrade gracefully; on a script their vocabulary does not
# cover, the cross-encoder stops discriminating. There its weight goes to the
# signals that survive translation: term overlap and the static embedding.
NON_LATIN_LANGS = frozenset({
    "zh", "ja", "ko", "ar", "he", "fa", "ur", "ru", "uk", "bg", "sr", "mk",
    "el", "hi", "bn", "ta", "te", "kn", "ml", "th", "lo", "km", "my", "ka",
    "hy", "am", "yi",
})

SCRIPT_WEIGHTS: dict[str, float] = {
    "ce": 0.35, "emb": 0.85, "lex": 0.70, "title": 0.40, "url": 0.24,
}


def weights_for(intent: str, lang: str = "en") -> Weights:
    w = Weights()
    for k, v in INTENT_WEIGHTS.get(intent, {}).items():
        setattr(w, k, v)
    if lang in NON_LATIN_LANGS:
        for k, v in SCRIPT_WEIGHTS.items():
            setattr(w, k, v)
    return w


WEIGHT_FIELDS = frozenset(f.name for f in fields(Weights))


def merge_weights(base: Weights, overrides: Optional[dict]) -> Weights:
    """`base` with named fields replaced. Unknown names are an error, since a
    silently ignored typo would let a caller believe they had reweighted the
    SERP when they had not."""
    if not overrides:
        return base
    unknown = sorted(set(overrides) - WEIGHT_FIELDS)
    if unknown:
        raise ValueError(f"unknown ranking weights: {', '.join(unknown)}; "
                         f"known: {', '.join(sorted(WEIGHT_FIELDS))}")
    out = replace(base)
    for k, v in overrides.items():
        setattr(out, k, float(v))
    return out


def route_fit(source: str, intent: str) -> float:
    """How much to trust a route's own confidence given the resolved intent.

    Speculative routes fire on a heuristic intent guess made before the
    planner has said anything. When the planner disagrees, those candidates
    are still worth keeping, but not at face value.
    """
    if not source.startswith("route:"):
        return 1.0
    route = ROUTES.get(source.split(":", 1)[1])
    if route is None:
        return 1.0
    return 1.0 if intent in route.speculative_for else 0.62


_YEAR = re.compile(r"(19|20)\d{2}")
_DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%Y%m%d")


def parse_date(published: str) -> Optional[datetime.date]:
    """A publication date out of whatever a page put in its metadata.

    Pages date themselves in ISO 8601, in a slash format, or with nothing more
    than a year in a JSON-LD blob. A year alone becomes 1 January, which is the
    only defensible reading and is what the freshness decay already assumed.
    """
    if not published:
        return None
    head = published.strip()[:10]
    for fmt in _DATE_FORMATS:
        try:
            return datetime.datetime.strptime(head, fmt).date()
        except ValueError:
            continue
    m = _YEAR.search(published)
    if m:
        try:
            return datetime.date(int(m.group(0)), 1, 1)
        except ValueError:
            return None
    return None


# Words that ask for the current value without naming a date. On a query that
# carries one, the newest page is the answer and a well-written old one is not.
_RECENCY_WORDS = frozenset(
    "latest current now today newest recent recently upcoming ongoing live "
    "aktuell aktuelle aktuellen neueste heute derniere dernier actuel actual "
    "atual ultimo ultima ultimi attuale reciente".split()
)


def query_period(query: str) -> tuple[frozenset[str], bool]:
    """The years a query asks for, and whether it asks for "the latest".

    Recency is a property of the question, not of its intent: a report for a
    named year and a company's quarterly results both want the newest
    document, and neither is classified as news.
    """
    years = frozenset(m.group(0) for m in _QUERY_YEAR.finditer(query))
    low = query.lower()
    wants = any(w in _RECENCY_WORDS for w in re.findall(r"[a-zà-öø-ÿ]+", low))
    return years, wants


def date_fit(published: str, path: str, title: str,
             years: frozenset[str]) -> float:
    """-1..1: does this page belong to the period the query named?

    A year in the query is a filter, not a lexical term: in the lexical bag
    it matches the copyright footer of every page on the web. Here it is a
    constraint on the document's own date, met by its metadata, its address
    or its title, and failed by a page visibly from an earlier year.
    """
    if not years:
        return 0.0
    d = parse_date(published)
    if d is not None and str(d.year) in years:
        return 1.0
    blob = f"{path} {title}".lower()
    if any(y in blob for y in years):
        return 0.8
    if d is not None:
        # A dated page from before the period asked for is the wrong document,
        # not merely an unmatched one. Undated pages stay neutral: most primary
        # sources publish no date at all.
        newest = max(int(y) for y in years)
        if d.year < newest:
            return -min(1.0, (newest - d.year) / 3.0)
    return 0.0


def _freshness(published: str, now: float = 0.0) -> float:
    """0..1, 1 = today, decaying with a ~2 year half-life.

    Date subtraction, not epoch arithmetic: pages carry dates like 1900 and
    2159, and `mktime` raises OverflowError outside the platform's epoch range
    for exactly those.
    """
    d = parse_date(published)
    if d is None:
        return 0.0
    try:
        ref = datetime.date.fromtimestamp(now or time.time())
    except (OverflowError, OSError, ValueError):
        ref = datetime.date.today()
    age_days = max(0.0, float((ref - d).days))
    return math.exp(-age_days / 730.0)


class Ranker:
    def __init__(self, weights: Optional[Weights] = None) -> None:
        self.base_weights = weights

    # ---------------------------------------------------------------- scoring
    def rank(
        self,
        query: str,
        docs: list[Doc],
        plan: Optional[Plan] = None,
        *,
        top_k: int = 10,
        rerank_depth: int = 20,
        max_per_host: int = 2,
        time_budget: float = 0.30,
        keep_text: bool = False,
        snippet_chars: int = 340,
        snippet_windows: int = 1,
        min_relevance: float = MIN_RELEVANCE,
        weights: Optional[dict] = None,
        prescore: Optional[Prescore] = None,
    ) -> list[Result]:
        t_start = time.perf_counter()
        intent = plan.intent if plan else "informational"
        lang = (plan.lang if plan else "en") or "en"
        # Per-request overrides sit on top of the intent's weights, so a caller
        # who nudges one term keeps every other adjustment the intent makes.
        w = merge_weights(self.base_weights or weights_for(intent, lang), weights)

        docs = self._dedup(docs)
        if not docs:
            return []
        if intent == "navigational":
            # "postgresql docs" should be allowed to return three pages of the
            # postgresql site; host clustering is for topical queries.
            max_per_host = max(max_per_host, 4)

        q_years, wants_recent = query_period(query)
        if q_years or wants_recent:
            # The query said "2026", or "latest". Whatever the intent's table
            # says, on this query an old page is the wrong page.
            w = replace(w, fresh=max(w.fresh, 0.28))
        qtok = tokens(query)
        qset = set(qtok)
        qterms = [t.strip('"').lower() for t in content_terms(query, lang)]
        qterms = [t for t in dict.fromkeys(qterms) if _content_word(t)][:10]
        # Three vocabularies. `qset` is the union, for lexical scoring, where more
        # query-adjacent words is better. `qterms` is the stopword-filtered content
        # vocabulary that picks passages and snippet windows, since an interrogative
        # scores every paragraph equally. `vocabs` keeps the query and each
        # expansion separate, because a title is scored on how completely it is
        # the subject, and pooling three expansions into one twelve-word set makes
        # an exact title indistinguishable from a near miss.
        vocabs = [set(qtok)]
        if plan and plan.expansions:
            for e in plan.expansions[:6]:
                etok = tokens(e)
                qset.update(etok)
                if etok:
                    vocabs.append(set(etok))
        vocabs = [v for v in vocabs if v]
        expansions = list((plan.expansions[:3] if plan else []))
        lex_query = " ".join([query, *expansions])
        for e in expansions:
            for t in content_terms(e, lang):
                t = t.strip('"').lower()
                if len(t) >= 3 and t not in qterms:
                    qterms.append(t)
        qterms = qterms[:16]

        # ---- build the passage pool -------------------------------------
        passages: list[Passage] = []
        anchor_passage: list[str] = ["" for _ in docs]
        for i, d in enumerate(docs):
            body = d.text or d.description or ""
            lead = " ".join(x for x in (d.title, d.description) if x)
            chunks = (select_passages(body, qterms, anchors=anchor_terms(d.url))
                      if body else [])
            if lead:
                chunks.insert(0, (lead + ". " + body[:400]).strip())
            if not chunks:
                cand = d.candidate
                fallback = " ".join(
                    x for x in (d.title, cand.title if cand else "",
                                cand.snippet if cand else "") if x
                ).strip()
                if fallback:
                    chunks = [fallback]
            anchored = anchor_terms(d.url)
            if anchored and chunks and any(a in chunks[0].lower() for a in anchored):
                anchor_passage[i] = chunks[0]
            for c in chunks[:MAX_PASSAGES_PER_DOC + 1]:
                passages.append(Passage(i, c))

        n_docs = len(docs)
        lex = np.zeros(n_docs, dtype=np.float32)
        emb = np.zeros(n_docs, dtype=np.float32)
        best_passage: list[str] = ["" for _ in range(n_docs)]
        lex_passage: list[str] = ["" for _ in range(n_docs)]
        snippet_pool: list[str] = ["" for _ in range(n_docs)]

        if passages:
            # BM25 runs over the query plus the planner's expansions: lexical
            # matching is the one stage that cannot bridge a synonym on its own,
            # and the X line is the vocabulary an authoritative page would print.
            lex, lex_best = self._bm25(lex_query, qtok, passages, n_docs, lang)
            emb, emb_best = self._embed(query, passages, n_docs)
            for i in range(n_docs):
                # The semantically best passage reads better, so the cross-encoder
                # scores it. The snippet window searches both: the answer string is
                # as often in the keyword-dense passage as in the one the embedding
                # likes.
                best_passage[i] = emb_best[i] or lex_best[i]
                lex_passage[i] = lex_best[i]
                pool = [p for p in (anchor_passage[i], best_passage[i],
                                    lex_best[i]) if p]
                seen_pool: list[str] = []
                for p in pool:
                    if p not in seen_pool:
                        seen_pool.append(p)
                snippet_pool[i] = " \u2026 ".join(seen_pool)

        lexn, embn = _minmax(lex), _minmax(emb)
        rare = self._rare_coverage(query, docs, lang, expansions)
        covered = self._coverage(qterms, docs)
        aspect = self._aspect_coverage(vocabs, docs)

        # ---- query-independent features ---------------------------------
        # 0 title, 1 url path, 2 prior, 3 authority, 4 freshness, 5 thin,
        # 6 host name, 7 non-200, 8 bare homepage, 9 identity orphan,
        # 10 period fit
        feats = np.zeros((n_docs, 11), dtype=np.float32)
        gone = np.zeros(n_docs, dtype=bool)
        # One clock reading for the whole SERP: two pages published the same
        # day must not get different freshness because ranking took a second.
        now = time.time()
        for i, d in enumerate(docs):
            cand = d.candidate
            title = (d.title or (cand.title if cand else "")).lower()
            feats[i, 0] = best_f1(vocabs, title_bag(title))
            parts = urlsplit(d.final_url)
            # The fragment is part of the address the planner chose, and often the
            # only part that names the query.
            path = unquote(parts.path + " " + urlsplit(d.url).fragment).lower()
            feats[i, 1] = path_fit(vocabs, path)
            host_full = (parts.hostname or "").lower()
            # The name in the domain: numpy.org over pypi.org/project/numpy.
            feats[i, 6] = host_fit(host_full, qset)
            feats[i, 2] = ((cand.prior * route_fit(cand.source, intent))
                           if cand else 0.5)
            feats[i, 3] = authority(host_full)
            feats[i, 4] = _freshness(d.published or (cand.published if cand else ""), now)
            body_len = len(d.text or "")
            # Thin means the fetch gave us nothing, not that the record is
            # naturally short: a video or an answer body arrives complete from the
            # site's own API.
            feats[i, 5] = (0.0 if (cand is not None and cand.skip_fetch)
                           else (1.0 if body_len < 200 else 0.0))
            # 404/410 means the page is not there, so never a result. 401/403/429
            # is usually a bot wall in front of a perfectly good page, so it is
            # penalised, not excluded; the route's title and snippet still carry it.
            feats[i, 7] = 1.0 if (d.status and d.status != 200) else 0.0
            gone[i] = d.status in (404, 410)
            # A bare homepage answers a navigational query and nothing else.
            feats[i, 8] = (1.0 if intent != "navigational"
                           and path.strip("/") == "" and not parts.query else 0.0)
            # A page whose title, path and domain share nothing with the query or
            # any expansion is about a different subject that happens to discuss
            # the same words. Identity is the check topicality cannot make.
            feats[i, 9] = (1.0 if intent != "navigational" and max(
                feats[i, 0], feats[i, 1], feats[i, 6]) < ORPHAN_FIT else 0.0)
            feats[i, 10] = date_fit(
                d.published or (cand.published if cand else ""),
                parts.path, d.title or "", q_years)

        consensus = np.array(
            [min(3, d.consensus) - 1 for d in docs], dtype=np.float32
        ) / 2.0

        # Static signals are worth a lot when the page is on topic and nothing
        # when it is not, so they are kept apart from relevance and gated below.
        static = (
            w.prior * feats[:, 2] + w.authority * feats[:, 3]
            + w.consensus * consensus
        )
        # Period fit rides with freshness rather than with relevance: on its own
        # it would put any 2026 page above the answer, which is the failure the
        # freshness gate already exists to prevent.
        period = w.fresh * feats[:, 4] + w.date * feats[:, 10]
        relevance = (
            w.lex * lexn + w.emb * embn + w.title * feats[:, 0]
            + w.url * feats[:, 1] + w.host * feats[:, 6] + w.rare * rare
            + w.covered * covered + w.aspect * aspect
        )
        spam = np.zeros(n_docs, dtype=np.float32)
        for i, d in enumerate(docs):
            m = SPAM.search(d.final_url) or SPAM.search(d.title or "")
            # "how to crack a password hash" is a legitimate query; the word
            # only signals spam when the user did not ask for it.
            if m and m.group(0).lower().strip("-") not in qset:
                spam[i] = 1.0
        penalty = (w.thin * feats[:, 5] + w.dead * feats[:, 7]
                   + w.root * feats[:, 8] + w.orphan * feats[:, 9] + spam)
        prelim = relevance + static + period - penalty

        # ---- cross-encoder on the head of the list ----------------------
        ce = np.zeros(n_docs, dtype=np.float32)
        order = np.argsort(-prelim)
        head = [int(i) for i in order[:rerank_depth]]
        spent = time.perf_counter() - t_start
        if head and spent < time_budget:
            # Two windows per document, not one. On a long page the embedding's
            # pick may be the wrong section, and a stub's single passage is its
            # lede, so one window per document is a length bias.
            items: list[tuple[int, str]] = []
            # Scores computed while the page was landing. A page still being
            # scored gets a short wait; the rest of the head is scored here.
            waited = 0.0
            for i in head:
                if prescore is None:
                    break
                left = max(0.0, min(PRESCORE_WAIT - waited, time_budget - spent))
                t_w = time.perf_counter()
                sc = prescore.get(docs[i], wait=left)
                waited += time.perf_counter() - t_w
                if sc is not None:
                    ce[i] = sc
                    continue
                for text in ce_windows(qterms, docs[i]):
                    items.append((i, text))
            for i in (head if prescore is None else []):
                lead = docs[i].title[:140] + ". " if docs[i].title else ""
                windows = [w for w in (anchor_passage[i], best_passage[i],
                                       lex_passage[i], docs[i].description) if w]
                picked: list[str] = []
                for wnd in windows:
                    if wnd not in picked:
                        picked.append(wnd)
                    if len(picked) >= CE_WINDOWS_PER_DOC:
                        break
                for wnd in picked:
                    items.append((i, lead + wnd[:900]))
            for i, sc in self._cross_encode(query, items).items():
                ce[i] = sc

        # The cross-encoder emits a calibrated probability, so it is used as one.
        # Min-maxing would destroy that: when every candidate scores between
        # 0.997 and 1.000, which happens whenever a plan is on topic, rescaling
        # stretches noise across the full range and spends the fusion's largest
        # weight on it. Raw, a tie stays a tie and the other signals decide.
        proxy = np.maximum(lexn, embn)
        reranked = np.zeros(n_docs, dtype=bool)
        reranked[head] = True
        rel_signal = np.where(reranked, ce, proxy)
        gate = 0.30 + 0.70 * rel_signal
        # Freshness is gated with the other query-independent signals: ungated,
        # any page with a recent date outranks the answer on a news query.
        final = (relevance + gate * (static + period)
                 - penalty + w.ce * ce)

        # Absolute, un-normalised agreement: a cross-encoder probability where the
        # cross-encoder ran, a cosine everywhere else. Unlike `final` this means
        # the same thing on every query, so it can carry a fixed floor. Where the
        # cross-encoder ran it is the authority: shared boilerplate scores a
        # respectable cosine against almost any query, so letting the embedding
        # override a cross-encoder zero re-admits the pages the floor removes.
        raw_rel = np.where(reranked, ce, emb)
        # Identity, a query word in the title or the path, likewise speaks only
        # for documents the cross-encoder never reached: a page that merely
        # shares a name with the subject must not be floored back above the bar.
        ident_rel = np.maximum(feats[:, 0], feats[:, 1])
        raw_rel = np.where(reranked, raw_rel, np.maximum(raw_rel, ident_rel))

        # ---- assemble SERP ----------------------------------------------
        # Two passes over the same ordering. The first respects the per-host cap;
        # the second runs only if the SERP came out short, and relaxes the cap
        # rather than reaching below the relevance floor: a third page from a
        # site clearly on topic beats a first page from one that is not.
        results: list[Result] = []
        per_host: dict[str, int] = {}
        order = [int(i) for i in np.argsort(-final)]
        pinned = {i for i, d in enumerate(docs) if d.candidate and d.candidate.rank}
        # A 404 is never an answer however well it scores, unless it is all we
        # have, in which case a dead link still beats a blank page.
        if not gone.all():
            order = [i for i in order if not gone[i] or i in pinned]
        order = sorted(pinned, key=lambda i: docs[i].candidate.rank) + [
            i for i in order if i not in pinned]
        used: set[int] = set()
        for cap in (max_per_host, max_per_host + 3):
            if len(results) >= top_k:
                break
            for i in order:
                if i in used:
                    continue
                d = docs[i]
                host = registrable((urlsplit(d.final_url).hostname or "").lower())
                if i not in pinned:
                    if len(results) >= MIN_RESULTS and raw_rel[i] < min_relevance:
                        continue
                    if per_host.get(host, 0) >= cap:
                        continue
                title = d.title or (d.candidate.title if d.candidate else "") or d.final_url
                parts, snippet = self._passages(qterms, snippet_pool[i], d,
                                                snippet_chars, snippet_windows)
                if not (title or snippet):
                    continue
                used.add(i)
                per_host[host] = per_host.get(host, 0) + 1
                results.append(self._row(
                    d, i, final, feats, lexn, embn, emb, ce, raw_rel, rare, aspect,
                    title, snippet, parts, keep_text,
                ))
                if len(results) >= top_k:
                    break
        # The relaxed second pass appends below the first, so the merged list is
        # only piecewise sorted.
        results.sort(key=lambda r: -r.score)
        return pin(results)

    def _row(self, d, i, final, feats, lexn, embn, emb, ce, raw_rel, rare, aspect,
             title: str, snippet: str, parts: list[tuple[str, float]],
             keep_text: bool) -> "Result":
        host = registrable((urlsplit(d.final_url).hostname or "").lower())
        return Result(
            url=d.final_url,
            title=title[:200],
            snippet=snippet,
            site=d.site or host,
            score=round(float(final[i]), 4),
            relevance=round(float(raw_rel[i]), 4),
            published=(d.published or (d.candidate.published if d.candidate else ""))[:10],
            source=d.candidate.source if d.candidate else "",
            passages=[p for p, _ in parts],
            passage_scores=[sc for _, sc in parts],
            # The whole extracted document, not a slice: extraction already bounds
            # it, and a slice of a reference page drops the section that answers.
            text=(d.text or "") if keep_text else "",
            text_chars=len(d.text or ""),
            status=d.status,
            debug={
                **({"rank": d.candidate.rank} if d.candidate and d.candidate.rank else {}),
                "lex": round(float(lexn[i]), 3), "emb": round(float(embn[i]), 3),
                "emb_raw": round(float(emb[i]), 3),
                "ce": round(float(ce[i]), 3), "rel": round(float(raw_rel[i]), 3),
                "rare": round(float(rare[i]), 3),
                "aspect": round(float(aspect[i]), 3),
                "title": round(float(feats[i, 0]), 3),
                "url": round(float(feats[i, 1]), 3), "host": round(float(feats[i, 6]), 3),
                "prior": round(float(feats[i, 2]), 3),
                "auth": round(float(feats[i, 3]), 3), "fresh": round(float(feats[i, 4]), 3),
                "orphan": round(float(feats[i, 9]), 1),
                "date": round(float(feats[i, 10]), 2),
                "status": d.status, "bytes": d.bytes_in, "fetch_ms": round(d.fetch_ms),
                "textlen": len(d.text or ""), "err": d.error,
            },
        )

    # ------------------------------------------------------------- internals
    @staticmethod
    def _fingerprint(text: str) -> str:
        """Cheap shingle of a page's opening prose.

        Client-rendered sites serve the same shell HTML for every URL, so ten
        different pages extract to one identical paragraph. Same opening, same
        page, to a reader.
        """
        words = _WORD.findall(text.lower())
        if len(words) < 12:
            return ""
        # Paired with a coarse length bucket: two pages of a documentation site
        # often share forty words of preamble and then diverge completely.
        return " ".join(words[:40]) + f"#{len(text) // 512}"

    def _dedup(self, docs: list[Doc]) -> list[Doc]:
        """One row per page. Two addresses are one page when they resolve to
        the same URL, declare the same canonical, or open with the same prose."""
        by_url: dict[str, Doc] = {}
        for d in docs:
            if d.error and not d.body and not d.text:
                # Never fetched. Kept only if a route already handed us a title and
                # a snippet; a bare URL is a row the reader cannot evaluate.
                if not (d.candidate and d.candidate.title and d.candidate.snippet):
                    continue
            key = normalize_url(_dedup_key(d))
            prev = by_url.get(key)
            if prev is None:
                by_url[key] = d
                continue
            merged = prev.consensus + 1
            # Keep whichever copy actually has content.
            winner = d if len(d.text or "") > len(prev.text or "") else prev
            loser = prev if winner is d else d
            if not winner.candidate and loser.candidate:
                winner.candidate = loser.candidate
            winner.consensus = merged
            by_url[key] = winner

        out: list[Doc] = []
        by_text: dict[str, Doc] = {}
        for d in by_url.values():
            fp = self._fingerprint(d.text or "")
            if not fp:
                out.append(d)
                continue
            twin = by_text.get(fp)
            if twin is None:
                by_text[fp] = d
                out.append(d)
                continue
            # Same rendered page under two addresses: keep the longer body, and
            # keep the shorter, more canonical URL when the bodies tie.
            if len(d.text or "") > len(twin.text or "") or (
                len(d.text or "") == len(twin.text or "")
                and len(d.final_url) < len(twin.final_url)
            ):
                out[out.index(twin)] = d
                by_text[fp] = d
        return out

    def _coverage(self, terms: list[str], docs: list[Doc]) -> np.ndarray:
        """Fraction of the query's content words the document contains at all.

        Every other signal here measures similarity, and two pages on the same
        topic have that in abundance; this one says which of them uses the words
        the query used.
        """
        out = np.zeros(len(docs), dtype=np.float32)
        if not terms:
            return out
        for i, d in enumerate(docs):
            blob = ((d.title or "") + " " + (d.text or "")).lower()
            if not blob.strip():
                continue
            out[i] = sum(1 for t in terms if t in blob) / len(terms)
        return out

    def _aspect_coverage(
        self, vocabs: list[frozenset[str]], docs: list[Doc]
    ) -> np.ndarray:
        """How many of the query's distinct constraints the page is about.

        Every other relevance signal reads a passage, and a passage is chosen for
        containing the query's words, so on a long page they all read the one
        window that mentions them and nothing downstream can see that the rest is
        about something else. This one reads what the page says it is (title,
        description, address, opening) rather than what it happens to contain.
        Coverage, not similarity: a page satisfying one of three constraints
        scores a third however emphatically it satisfies that one.

        Scored against the query and each expansion separately, best wins, for
        the reason `best_f1` does it. Matching is by n-gram so that a compound
        satisfies its head word.
        """
        out = np.zeros(len(docs), dtype=np.float32)
        if not vocabs:
            return out
        # Short terms have no gram set, but they are matched exactly and can be
        # the whole constraint ("http", "103").
        vocab_terms = [[t for t in v if len(t) >= 3] for v in vocabs]
        if not any(vocab_terms):
            return out
        idents: list[set[str]] = []
        igrams: list[frozenset[str]] = []
        for d in docs:
            ident = set(tokens(f"{d.title} {d.description}"))
            parts = urlsplit(d.final_url or d.url)
            ident |= path_terms(unquote(parts.path).lower())[1]
            ident |= set(tokens((d.text or "")[:ASPECT_HEAD]))
            idents.append(ident)
            # One gram set for the whole identity and one containment test per
            # term against it; this runs over every candidate.
            igrams.append(gram_set(ident) if ident else frozenset())

        n = len(docs)
        for terms in vocab_terms:
            if not terms:
                continue
            present = np.zeros((n, len(terms)), dtype=np.float32)
            for j, t in enumerate(terms):
                tg = _grams(t)
                for i, ident in enumerate(idents):
                    if not ident:
                        continue
                    if t in ident or (
                            tg and len(tg & igrams[i]) / len(tg) >= ASPECT_HIT):
                        present[i, j] = 1.0
            # Not every constraint constrains equally: nearly every candidate says
            # "melting" and "point", and the page that does not say "tungsten" is
            # the wrong page however much of the rest it covers. Each aspect is
            # weighted by how much it separates these candidates.
            df = present.sum(axis=0)
            idf = np.log((n + 1.0) / (df + 0.5))
            np.clip(idf, 0.0, None, out=idf)
            total = float(idf.sum())
            # Every term in every candidate, or none in any: fall back to plain
            # coverage rather than divide by zero.
            scores = ((present @ idf) / total if total > 1e-6
                      else present.mean(axis=1))
            np.maximum(out, scores, out=out)
        return out

    def _rare_coverage(
        self, query: str, docs: list[Doc], lang: str = "en",
        expansions: Optional[list[str]] = None,
    ) -> np.ndarray:
        """How much of the query's distinctive vocabulary a document contains.

        Every other relevance signal is a similarity, and none of them can say
        "this page contains the literal identifier the user asked about, and
        that one does not". For a long-tail query the whole question is one rare
        token: a function name, an error string, a config parameter.

        Only identifier-shaped tokens count: punctuation inside the word, or a
        digit. Rarity alone is not enough, because an ordinary word that happens
        to be rare among the candidates ("tallest" where the page says
        "highest") has synonyms and identifiers do not. When nothing qualifies
        the feature stays silent.
        """
        n = len(docs)
        out = np.zeros(n, dtype=np.float32)
        terms = [t.lower() for t in dict.fromkeys(content_terms(query, lang))]
        terms = [t.strip('"') for t in terms if len(t.strip('"')) >= 3][:12]
        if not terms or not n:
            return out
        blobs = [
            ((d.title or "") + " " + (d.text or "")).lower() for d in docs
        ]
        present = np.zeros((n, len(terms)), dtype=np.float32)
        for j, t in enumerate(terms):
            # A rare term is trusted literally, so it is matched literally: as a
            # word, not a substring.
            pat = re.compile(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])")
            for i, blob in enumerate(blobs):
                if pat.search(blob):
                    present[i, j] = 1.0
        df = present.sum(axis=0)
        idf = np.log((n + 1.0) / (df + 0.5))
        np.clip(idf, 0.0, None, out=idf)
        # Keep only identifier-shaped terms that separate these documents. A
        # word every candidate contains says nothing.
        for j, t in enumerate(terms):
            if (not _IDENTIFIER.search(t) or _BARE_NUMBER.match(t)
                    or _PERIOD.match(t) or df[j] >= n or df[j] == 0):
                idf[j] = 0.0
        total = float(idf.sum())
        if total <= 1e-6:
            return out
        return (present @ idf) / total

    def _bm25(
        self, query: str, qtok: list[str], passages: list[Passage], n_docs: int,
        lang: str = "en",
    ) -> tuple[np.ndarray, list[str]]:
        scores = np.zeros(n_docs, dtype=np.float32)
        best = ["" for _ in range(n_docs)]
        texts = [p.text for p in passages]
        try:
            # bm25s raises its own logger to DEBUG at import time.
            import logging as _lg

            import bm25s
            _lg.getLogger("bm25s").setLevel(_lg.WARNING)
            # Stemming German with an English stemmer is worse than not stemming,
            # and an English stopword list deletes nothing from a Japanese query.
            stemmer = Models.stemmer(lang) or None
            stopwords = lang if lang in BM25_STOPWORD_LANGS else None
            corpus_texts, qtext = texts, query
            if lang in CJK_LANGS:
                # bm25s splits on whitespace, which this text does not have. Segment
                # into character bigrams first; no stemmer or stopword list applies.
                corpus_texts = [segmented(t) for t in texts]
                qtext = segmented(query)
                stemmer, stopwords = None, None
            corpus = bm25s.tokenize(
                corpus_texts, stopwords=stopwords, stemmer=stemmer, show_progress=False
            )
            idx = bm25s.BM25(k1=1.2, b=0.75)
            idx.index(corpus, show_progress=False)
            qt = bm25s.tokenize(
                [qtext], stopwords=stopwords, stemmer=stemmer, show_progress=False
            )
            k = min(len(texts), 400)
            ids, ss = idx.retrieve(qt, k=k, show_progress=False)
            for rank_i in range(ids.shape[1]):
                pi = int(ids[0, rank_i])
                s = float(ss[0, rank_i])
                di = passages[pi].doc_idx
                if s > scores[di]:
                    scores[di] = s
                    best[di] = texts[pi]
        except Exception:
            qset = set(qtok)
            for p in passages:
                t = set(tokens(p.text))
                s = len(qset & t) / max(1, len(qset))
                if s > scores[p.doc_idx]:
                    scores[p.doc_idx] = s
                    best[p.doc_idx] = p.text
        for i in range(n_docs):
            if not best[i]:
                for p in passages:
                    if p.doc_idx == i:
                        best[i] = p.text
                        break
        return scores, best

    def _embed(
        self, query: str, passages: list[Passage], n_docs: int
    ) -> tuple[np.ndarray, list[str]]:
        scores = np.zeros(n_docs, dtype=np.float32)
        best = ["" for _ in range(n_docs)]
        model = Models.embedder()
        if not model:
            return scores, best
        try:
            texts = [p.text[:1500] for p in passages]
            mat = np.asarray(model.encode(texts), dtype=np.float32)
            qv = np.asarray(model.encode([query]), dtype=np.float32)[0]
            mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
            qv /= np.linalg.norm(qv) + 1e-9
            sims = mat @ qv
            scores[:] = -1.0
            for pi, p in enumerate(passages):
                s = float(sims[pi])
                di = p.doc_idx
                if s > scores[di]:
                    scores[di] = s
                    best[di] = texts[pi]
            np.maximum(scores, 0.0, out=scores)
        except Exception:
            pass
        return scores, best

    def _ce_best(self, query: str, texts: list[str]) -> Optional[float]:
        """The best cross-encoder score over `texts`, or None when it cannot run."""
        out = self._cross_encode(query, [(0, t) for t in texts])
        return out.get(0)

    def _cross_encode(self, query: str, items: list[tuple[int, str]]) -> dict[int, float]:
        """Score every (document, window) pair; a document keeps its best."""
        rk = Models.reranker()
        if not rk or not items:
            return {}
        try:
            from flashrank import RerankRequest
            payload, owner = [], []
            for doc_idx, text in items:
                if not text.strip():
                    continue
                payload.append({"id": len(owner), "text": text[:1400]})
                owner.append(doc_idx)
            if not payload:
                return {}
            out = rk.rerank(RerankRequest(query=query, passages=payload))
            best: dict[int, float] = {}
            for r in out:
                i = owner[int(r["id"])]
                sc = float(r["score"])
                if sc > best.get(i, -1.0):
                    best[i] = sc
            return best
        except Exception:
            return {}

    def _passages(self, terms: list[str], passage: str, doc: Doc,
                  max_chars: int = 340, windows: int = 1
                  ) -> tuple[list[tuple[str, float]], str]:
        """The densest windows of query terms across the whole document.

        The passages the ranker scored are a few windows of a document that may
        have hundreds, so they are only the fallback for a document with no
        extracted body. `terms` is the content vocabulary, not the raw token
        union: scoring a window on "what", "was" and "date" rewards prose over
        the sentence carrying the fact.

        Returns the windows with their scores and the joined single-string form.
        """
        seed = passage or doc.description or ""
        if not seed and doc.candidate:
            seed = doc.candidate.snippet or ""
        body = doc.text or ""
        parts, before, after = select_windows(body, seed, terms, max_chars, windows)
        if not parts:
            return [], ""
        if len(parts) == 1 and not before and not after:
            return parts, parts[0][0]
        joined = WINDOW_JOIN.join(p for p, _ in parts)
        if before:
            joined = "... " + joined
        if after:
            joined = joined + " ..."
        return parts, joined
