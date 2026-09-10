"""Fit a SERP inside a token budget.

The engine already runs on a wall-clock budget checked at every stage. This is
the same idea applied to the other resource a consumer actually runs out of:
an agent hands the SERP to a model, and a page of extracted text is thousands
of tokens it may not have. Almost every search API leaves this to the caller,
which means the caller discovers the overrun after paying for it.

Exact counts need a tokenizer. `tiktoken` is an optional extra; without it the
count is the four-characters-per-token estimate, which is wrong by a few
percent and never by a factor.
"""
from __future__ import annotations

from typing import Optional

from .types import Result

CHARS_PER_TOKEN = 4
_encoder: Optional[object] = None


def _enc():
    """cl100k_base if tiktoken is installed, else None. Loaded once."""
    global _encoder
    if _encoder is None:
        try:
            import tiktoken
            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _encoder = False
    return _encoder or None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    enc = _enc()
    if enc is None:
        return max(1, len(text) // CHARS_PER_TOKEN)
    try:
        return len(enc.encode(text, disallowed_special=()))
    except Exception:
        return max(1, len(text) // CHARS_PER_TOKEN)


def truncate(text: str, max_tokens: int) -> str:
    """`text` cut to `max_tokens`, on a word boundary, with an ellipsis."""
    if max_tokens <= 0 or not text:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    enc = _enc()
    if enc is not None:
        try:
            text = enc.decode(enc.encode(text, disallowed_special=())[:max_tokens])
        except Exception:
            text = text[:max_tokens * CHARS_PER_TOKEN]
    else:
        text = text[:max_tokens * CHARS_PER_TOKEN]
    cut = text.rfind(" ")
    if cut > len(text) * 0.6:
        text = text[:cut]
    return text.rstrip() + " ..."


def result_tokens(r: Result) -> int:
    return (count_tokens(r.title) + count_tokens(r.url)
            + sum(count_tokens(p) for p in r.passages) + count_tokens(r.text))


def apply(results: list[Result], *, token_budget: int = 0,
          token_budget_per_page: int = 0, passages_per_page: int = 0,
          min_results: int = 1) -> tuple[list[Result], int]:
    """Trim in place to fit, returning the kept rows and how many were dropped.

    Per-page caps are applied first, so a global budget is spent on rows the
    caller already agreed to pay for. When the global budget runs out the
    remaining rows are dropped rather than emptied: ten rows with no content is
    not a cheaper answer, it is no answer, and a short SERP is the behaviour
    the ranker already commits to elsewhere.
    """
    if passages_per_page > 0:
        for r in results:
            if len(r.passages) > passages_per_page:
                r.passages = r.passages[:passages_per_page]
                r.passage_scores = r.passage_scores[:passages_per_page]

    if token_budget_per_page > 0:
        for r in results:
            _fit_page(r, token_budget_per_page)

    if token_budget <= 0:
        return results, 0

    kept: list[Result] = []
    spent = 0
    for r in results:
        cost = result_tokens(r)
        if spent + cost > token_budget:
            if len(kept) >= min_results:
                break
            # Below the floor we shrink the row rather than drop it, so a tiny
            # budget still returns something with content in it.
            _fit_page(r, max(1, token_budget - spent))
            cost = result_tokens(r)
        kept.append(r)
        spent += cost
    return kept, len(results) - len(kept)


def _fit_page(r: Result, max_tokens: int) -> None:
    """Shrink one row's content to `max_tokens`, page text going first.

    Passages are query-selected and the text is the whole page, so under
    pressure the passages are the part worth keeping.
    """
    budget = max_tokens - count_tokens(r.title) - count_tokens(r.url)
    if budget <= 0:
        r.passages, r.passage_scores, r.text = [], [], ""
        return
    used = sum(count_tokens(p) for p in r.passages)
    if used > budget:
        keep: list[str] = []
        scores: list[float] = []
        left = budget
        for p, sc in zip(r.passages, r.passage_scores or [0.0] * len(r.passages)):
            n = count_tokens(p)
            if n <= left:
                keep.append(p)
                scores.append(sc)
                left -= n
            elif not keep:
                keep.append(truncate(p, left))
                scores.append(sc)
                left = 0
            if left <= 0:
                break
        r.passages, r.passage_scores = keep, scores
        used = sum(count_tokens(p) for p in r.passages)
    if r.text:
        r.text = truncate(r.text, max(0, budget - used))
