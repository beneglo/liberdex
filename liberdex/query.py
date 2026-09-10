"""Query rewriting for keyword APIs, and a cheap language guess.

Site-native search endpoints want keywords, not sentences: a six-word
natural-language query returns zero Wikipedia hits where its three-word head
returns hundreds. So each route gets the query trimmed to as many content terms
as its backend tolerates, but only when the planner did not write that route's
query itself, which it does better.

The language guess exists because the speculative routes fire before the planner
has said anything, and sending a German query to en.wikipedia.org wastes the one
request that was supposed to hide the planner's latency. The planner's own `L`
line supersedes it the moment it arrives.
"""
from __future__ import annotations

import re

from .script import pieces, unspaced

_TOKEN = re.compile(r'"[^"]+"|[^\W_][\w+#._\'-]*', re.UNICODE)

STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "doing", "have", "has", "had", "having", "will",
    "would", "shall", "should", "can", "could", "may", "might", "must",
    "i", "you", "he", "she", "it", "we", "they", "me", "my", "your", "his",
    "her", "its", "our", "their", "this", "that", "these", "those",
    "of", "in", "on", "at", "to", "for", "with", "from", "by", "about", "into",
    "over", "after", "before", "between", "under", "and", "or", "but", "if",
    "then", "than", "so", "as", "not", "no", "there", "here", "when", "where",
    "why", "how", "what", "which", "who", "whom", "whose", "any", "some",
    "get", "got", "use", "used", "using", "make", "made", "want", "need",
    "please", "help", "explain", "tell", "show", "give", "find", "look",
    "best", "good", "way", "ways", "much", "many", "very", "really", "just",
    "s", "t", "re", "ve", "ll", "d", "m",
    # Verbs that carry the question shape rather than the subject.
    "work", "works", "working", "happen", "happens", "mean", "means",
    "called", "call", "know", "understand", "difference", "differences",
}

# Terms that carry intent, not subject matter; keep them out of keyword APIs.
NOISE = {"tutorial", "guide", "example", "examples", "step", "steps", "vs",
         "教程", "指南", "示例", "例子", "步骤", "步驟"}


def content_terms(query: str, lang: str = "en") -> list[str]:
    """The query's content words, in order, quoted phrases kept whole.

    The stop list is English plus the language's own marker words: the
    marker sets are function words by construction, which is what a stop
    list is, and without them a German question keeps "wer" as its subject.
    """
    stop = STOP if lang == "en" else STOP | _MARKERS.get(lang, frozenset())
    out: list[str] = []
    for m in _TOKEN.finditer(query):
        tok = m.group(0)
        if tok.startswith('"'):
            out.append(tok)
            continue
        if unspaced(tok):
            # One Han character can be a whole noun (猫, 车, 水), so the
            # single-letter rule below does not apply to it.
            out.extend(p for p in pieces(tok)
                       if p.lower() not in stop and (len(p) > 1 or unspaced(p)))
            continue
        low = tok.lower()
        if low in stop or len(low) == 1:
            continue
        out.append(tok)
    return out


def content_bag(text: str, lang: str = "en") -> set[str]:
    """`content_terms` as the lower-cased bag every fit compares against."""
    return {t.lower() for t in content_terms(text, lang)}


def keyword_query(query: str, max_terms: int = 0, *, drop_noise: bool = True,
                  lang: str = "en") -> str:
    """Trim to the most informative `max_terms` content words, keeping order."""
    terms = content_terms(query, lang)
    if drop_noise and len(terms) > 2:
        stripped = [t for t in terms if t.lower() not in NOISE]
        if len(stripped) >= 2:
            terms = stripped
    if not terms:
        return query.strip()
    if max_terms and len(terms) > max_terms:
        # The first content term is almost always the subject ("rust", "postgres")
        # and dropping it wrecks the query, so it is pinned; the rest compete on
        # informativeness. Emit in original order so phrase matching still works.
        rest = sorted(range(1, len(terms)),
                      key=lambda i: (-_informativeness(terms[i]), i))[:max_terms - 1]
        terms = [terms[i] for i in sorted([0] + rest)]
    return " ".join(terms)


def _informativeness(tok: str) -> float:
    """Cheap proxy: long, capitalised, punctuated or digit-bearing = specific."""
    s = 0.0
    s += min(len(tok), 14) / 14.0
    if tok[:1].isupper():
        s += 0.35
    if len(tok) >= 2 and tok.isupper():
        s += 0.6  # acronyms (HNSW, TLS, HTTP) are the most specific terms there are
    if any(c.isdigit() for c in tok):
        s += 0.25
    if any(c in "._-+#" for c in tok):
        s += 0.4
    if tok.startswith('"'):
        s += 1.0
    return s


# --------------------------------------------------------------- language
# Unicode ranges that identify a language (or a small family) outright.
_SCRIPTS: tuple[tuple[int, int, str], ...] = (
    (0x0400, 0x04FF, "ru"), (0x0590, 0x05FF, "he"), (0x0600, 0x06FF, "ar"),
    (0x0900, 0x097F, "hi"), (0x0E00, 0x0E7F, "th"), (0x1100, 0x11FF, "ko"),
    (0x3040, 0x30FF, "ja"), (0x3400, 0x4DBF, "zh"), (0x4E00, 0x9FFF, "zh"),
    (0xAC00, 0xD7AF, "ko"), (0x0370, 0x03FF, "el"),
)

# Function words common in one language and rare in the others here. They steer
# the first Wikipedia request only; the planner's L line wins.
_MARKERS: dict[str, frozenset[str]] = {
    "de": frozenset("der die das ist wie was wer wo warum ein eine und nicht "
                    "mit von im den dem des auf fuer für über kann".split()),
    "fr": frozenset("le la les des est comment pourquoi qui quoi une un et "
                    "dans pour avec sur ne pas plus quel quelle".split()),
    "es": frozenset("el la los las es como por que quien donde una un y en "
                    "para con del cual cuales cuanto".split()),
    "pt": frozenset("o a os as é como porque quem onde uma um e em para com "
                    "do da dos das qual quais quanto".split()),
    "it": frozenset("il lo la gli le è come perche perché chi dove una un e "
                    "in per con del della quale quali quanto".split()),
    "nl": frozenset("de het een is hoe wat wie waar waarom en niet met van "
                    "voor op naar welke kan".split()),
    "pl": frozenset("jak co kto gdzie dlaczego jest nie i w na z do czy "
                    "który która które".split()),
    "tr": frozenset("nasıl nedir kim nerede neden ve bir bu ile için değil "
                    "olan daha çok".split()),
    "sv": frozenset("hur vad vem var varför är och inte med av för till den "
                    "det en ett".split()),
    "da": frozenset("hvordan hvad hvem hvor hvorfor er og ikke med af for til "
                    "den det en et kan".split()),
    "no": frozenset("hvordan hva hvem hvor hvorfor er og ikke med av for til "
                    "den det en et kan".split()),
    "fi": frozenset("miten mikä kuka missä miksi on ja ei kanssa varten että "
                    "tai myös se tämä".split()),
    "cs": frozenset("jak co kdo kde proč je není a v na s pro do který která "
                    "které".split()),
    "hu": frozenset("hogyan mi ki hol miért van nem és a az egy hogy vagy "
                    "mint melyik".split()),
    "ro": frozenset("cum ce cine unde de ce este nu și în pe cu pentru la "
                    "care sau un o".split()),
    "ru": frozenset("как что кто где почему это не и в на с для по от или "
                    "какой какая какие".split()),
    "uk": frozenset("як що хто де чому це не і в на з для по від або який "
                    "яка які".split()),
}


def guess_lang(query: str) -> str:
    """Best-effort ISO 639-1 for the query. `en` when nothing else fits."""
    q = query.strip()
    if not q:
        return "en"
    counts: dict[str, int] = {}
    for ch in q:
        cp = ord(ch)
        if cp < 0x0370:
            continue
        for lo, hi, lang in _SCRIPTS:
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
                break
    if counts:
        # Japanese text is mostly Han; a single kana settles it either way.
        if counts.get("ja"):
            return "ja"
        return max(counts.items(), key=lambda kv: kv[1])[0]
    words = set(re.findall(r"[a-zà-öø-ÿğışçöü]+", q.lower()))
    if not words:
        return "en"
    best, best_n = "en", 0
    for lang, markers in _MARKERS.items():
        n = len(words & markers)
        if n > best_n:
            best, best_n = lang, n
    # One shared function word ("die", "a", "in") is noise; two is a signal.
    return best if best_n >= 2 else "en"

