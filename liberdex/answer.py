"""Answering from a liberdex SERP.

Two modes, named for what they do:

    extract      the answer and nothing else. One fact, checkable, no prose.
    synthesize   an answer composed across the pages, with inline citations.

The prompts lean on what this engine hands a model and a snippet-only engine
cannot: every page carries a calibrated relevance score and a `source` saying
whether it came from the planner's memory, a site-native route, or a hub's
outbound links. That provenance says which page to believe when two disagree,
so the prompt says so.
The answer call is also the SERP's last judgement. The model has just read the
top pages in full, the only stage of the pipeline that reads whole pages rather
than windows of them, and its first line says which of them were about the
question, best first. `judged` orders the SERP by that line.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .llm import Chat, Reply
from .models import Profile, resolve_role
from .rank import pin
from .types import Result

MAX_PAGES = 8
# Enough for a composed answer with citations; short enough that a model which
# starts rambling gets cut off rather than billed for a page of it.
MAX_ANSWER_TOKENS = 900
# What the model reads of each page. The lede first, because that is where an
# encyclopedia or a report states its subject and its headline fact, then the
# windows the ranker selected for the query. A window alone can be a reference
# list while the lede names the answer.
LEAD_CHARS = 1400
TOP_PAGE_CHARS = 3200
PAGE_CHARS = 1400
TOP_PAGES = 4

EXTRACT_SYSTEM = """\
You answer from a set of web pages that have already been retrieved and ranked \
for you. Return the answer itself and nothing around it.

What you are given: pages in rank order, each with its URL, the site it came \
from, a relevance score in 0-1, and how it was found: `llm` means a model \
recalled the URL from memory, `route:<site>` means the site's own search \
returned it, `expand` means it was linked from a hub page. The text under each \
page is its opening followed by the passages selected for this query, not the \
whole page.

Rules:
- First line: `PAGES:` followed by the numbers of the pages that are actually \
about the question, most useful first, e.g. `PAGES: 3 1 5`. Leave out every \
page that is off-topic, a product listing, a navigation page or about \
something else that merely shares a word with the question. Then an empty line.
- Then the answer only. No preamble, no restatement of the question, no \
explanation, no source list, no closing sentence.
- Use only what the pages say. If they do not contain the answer, reply exactly: \
NOT FOUND
- When pages disagree, prefer the primary or official source, meaning the \
organisation, standard, project or publication the question is about, over \
commentary about it. A higher relevance score breaks a tie between sources of \
equal standing; it does not outrank being the primary source.
- Keep the answer's own units, spelling and precision as the page states them.\
"""

SYNTHESIZE_SYSTEM = """\
You answer from a set of web pages that have already been retrieved and ranked \
for you.

What you are given: pages in rank order, each with a number, its URL, the site \
it came from, a relevance score in 0-1, and how it was found: `llm` means a \
model recalled the URL from memory, `route:<site>` means the site's own search \
returned it, `expand` means it was linked from a hub page. The text under each \
page is its opening followed by the passages selected for this query, not the \
whole page.

Rules:
- First line: `PAGES:` followed by the numbers of the pages that are actually \
about the question, most useful first, e.g. `PAGES: 3 1 5`. Leave out every \
page that is off-topic, a product listing, a navigation page or about \
something else that merely shares a word with the question. Then an empty line.
- Answer the question directly in the first sentence, then add only what a \
reader needs to trust or use that answer.
- Cite with the page's number in square brackets, immediately after the claim \
it supports: [2]. Cite every factual claim. Several numbers may support one \
claim: [1][4].
- Use only what the pages say. Never fill a gap from your own knowledge. If the \
pages only partly answer the question, say which part is missing.
- When pages disagree, say so and name which source you are following and why, \
preferring the primary or official source over commentary about it.
- No preamble, no "based on the search results", no closing summary. Plain \
prose, a few sentences; use a short list only if the answer is genuinely a list.\
"""


def page_body(r: Result, chars: int) -> str:
    """The lede of the page, then the query-selected windows not already in it."""
    text = (r.text or "").strip()
    parts: list[str] = []
    if text:
        parts.append(text[:LEAD_CHARS])
    for p in (r.passages or ([r.snippet] if r.snippet else [])):
        p = p.strip()
        if not p:
            continue
        probe = p[:120]
        if any(probe in x for x in parts):
            continue
        parts.append(p)
    body = "\n[...]\n".join(parts)
    if not body and text:
        body = text[:chars]
    return body[:chars]


def format_pages(results: Sequence[Result], *, numbered: bool,
                 max_pages: int = MAX_PAGES) -> str:
    """The SERP as the model sees it: provenance in the header, the page below."""
    blocks: list[str] = []
    for i, r in enumerate(results[:max_pages], 1):
        body = page_body(r, TOP_PAGE_CHARS if i <= TOP_PAGES else PAGE_CHARS)
        head = f"[{i}] " if numbered else ""
        blocks.append(
            f"{head}{r.title or r.url}\n"
            f"url: {r.url}\n"
            f"site: {r.site or '-'}  score: {r.score:.2f}  found_by: {r.source or '-'}\n"
            f"{body.strip()}"
        )
    return "\n\n".join(blocks)


def build_messages(query: str, results: Sequence[Result], mode: str,
                   max_pages: int = MAX_PAGES) -> list[dict[str, str]]:
    numbered = mode == "synthesize"
    system = SYNTHESIZE_SYSTEM if numbered else EXTRACT_SYSTEM
    return [
        {"role": "system", "content": system},
        {"role": "user", "content":
            f"Question: {query}\n\n"
            f"Pages:\n\n{format_pages(results, numbered=numbered, max_pages=max_pages)}\n\n"
            f"Answer:"},
    ]


# Not "none". Answering is off the search deadline so a little thinking is
# affordable, and several endpoints refuse to switch it off at all ("Reasoning
# is mandatory for this endpoint and cannot be disabled"). Unset is worse than
# either: reasoning tokens bill against max_tokens, so an unbounded budget
# truncates the answer itself.
DEFAULT_REASONING_EFFORT = "minimal"


def chat_for(model: Optional[str] = None, *, profile: Optional[Profile] = None,
             timeout: float = 30.0,
             reasoning_effort: Optional[str] = None) -> Chat:
    """The Chat bound to the `answer` role, or to an explicit model override."""
    if profile is None:
        profile, resolved = resolve_role("answer", model)
    else:
        resolved = model or profile.default_model
    effort = (reasoning_effort
              or os.environ.get("LIBERDEX_ANSWER_REASONING_EFFORT")
              or DEFAULT_REASONING_EFFORT)
    return Chat(profile, resolved, temperature=0.0,
                max_tokens=MAX_ANSWER_TOKENS, timeout=timeout,
                reasoning_effort=effort)


STRUCTURED_SYSTEM = """\
You fill in a JSON object from a set of web pages that have already been \
retrieved and ranked for you.

What you are given: pages in rank order, each with a number, its URL, the site \
it came from, a relevance score in 0-1, and how it was found: `llm` means a \
model recalled the URL from memory, `route:<site>` means the site's own search \
returned it, `expand` means it was linked from a hub page. The text under each \
page is the passage selected for this query, not the whole page.

Rules:
- Return only the JSON object the schema describes. No prose around it.
- Fill a field only from what the pages say. Never fill one from your own \
knowledge, and never invent a plausible value to avoid leaving a gap.
- A field the pages do not support is null, or omitted where the schema allows \
it. An empty field is a correct answer; a guessed one is not.
- For every field you did fill, add an entry to `_grounding`: the field's name, \
the page numbers that support it, and your confidence: `high` when a page \
states it outright, `medium` when it follows from what a page says, `low` when \
you are reading between the lines.
- Prefer the primary or official source over commentary about it when pages \
disagree.\
"""

# Per-field grounding beats a flat citation list: a caller can act on the field
# that was well supported and re-check the one that was not.
GROUNDING_SCHEMA = {
    "type": "array",
    "description": "One entry per field you filled in.",
    "items": {
        "type": "object",
        "properties": {
            "field": {"type": "string"},
            "pages": {"type": "array", "items": {"type": "integer"}},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        },
        "required": ["field", "pages", "confidence"],
        "additionalProperties": False,
    },
}


def wrap_schema(schema: dict) -> dict:
    """The caller's schema plus a `_grounding` array, as a response_format."""
    props = dict(schema.get("properties") or {})
    props["_grounding"] = GROUNDING_SCHEMA
    body = {**schema, "type": "object", "properties": props}
    return {"type": "json_schema",
            "json_schema": {"name": "liberdex_answer", "schema": body}}


async def structured(
    query: str,
    results: Sequence[Result],
    schema: dict,
    *,
    model: Optional[str] = None,
    chat: Optional[Chat] = None,
    budget: float = 45.0,
    max_pages: int = MAX_PAGES,
) -> tuple[Optional[dict], Reply]:
    """Fill `schema` from the ranked pages. Returns (object, the raw Reply).

    Providers with native `response_format: json_schema` take it directly. For
    the rest, `liberdex[schema]` installs instructor, which picks the best mode
    the provider has and retries on a validation failure.
    """
    import orjson

    if not results:
        return None, Reply(error="no results to answer from")
    if not isinstance(schema, dict) or not schema.get("properties"):
        return None, Reply(error="output_schema must be an object schema "
                                 "with `properties`")
    own = chat is None
    c = chat or chat_for(model, timeout=budget)
    messages = [
        {"role": "system", "content": STRUCTURED_SYSTEM},
        {"role": "user", "content":
            f"Question: {query}\n\n"
            f"Pages:\n\n{format_pages(results, numbered=True, max_pages=max_pages)}\n\n"
            f"JSON:"},
    ]
    try:
        reply = await c.complete(messages, response_format=wrap_schema(schema),
                                 budget=budget)
    finally:
        if own:
            await c.aclose()
    if not reply.ok:
        return None, reply
    try:
        return orjson.loads(reply.text), reply
    except Exception:
        # Some endpoints wrap the object in a fenced block despite the schema.
        text = reply.text.strip().removeprefix("```json").removeprefix("```")
        text = text.removesuffix("```").strip()
        try:
            return orjson.loads(text), reply
        except Exception as e:
            return None, Reply(error=f"model did not return JSON: {e}",
                               model=reply.model,
                               prompt_tokens=reply.prompt_tokens,
                               completion_tokens=reply.completion_tokens)


_PAGES_LINE = re.compile(r"^\s*PAGES?\s*:\s*([0-9][0-9 ,]*)\s*$", re.I | re.M)


@dataclass(slots=True)
class Answer:
    """What one answer call produced: the text, the raw reply, and the model's
    reading of which pages were about the question, best first, 0-based."""
    text: str = ""
    reply: Reply = field(default_factory=Reply)
    pages: list[int] = field(default_factory=list)


def parse_judgement(text: str, n: int) -> tuple[list[int], str]:
    """Split the `PAGES:` line off the answer. Numbers are 1-based in the
    prompt and 0-based here; unknown or repeated numbers are dropped."""
    m = _PAGES_LINE.search(text)
    if not m:
        return [], text.strip()
    order: list[int] = []
    for tok in re.split(r"[ ,]+", m.group(1).strip()):
        if not tok.isdigit():
            continue
        i = int(tok) - 1
        if 0 <= i < n and i not in order:
            order.append(i)
    body = (text[:m.start()] + text[m.end():]).strip()
    return order, body


def judged(results: Sequence[Result], pages: list[int],
           read: int = MAX_PAGES, keep: int = 3) -> list[Result]:
    """The SERP as the answer model, having read it, says it should be.

    The model read the top `read` pages in full and listed the ones that were
    about the question. That is the one judgement in the pipeline made on
    whole pages rather than on windows of them, so it orders the SERP: the
    listed pages first, in its order, then the pages it never saw, in theirs.
    A page it read and left out was judged off-topic and is dropped, unless
    fewer than `keep` rows would remain, since a thin SERP still beats an empty
    one. `relevance` on every row still says what the ranker thought.
    """
    if not pages:
        return list(results)
    listed = set(pages)
    out = [results[i] for i in pages if i < len(results)]
    unseen = [r for i, r in enumerate(results) if i >= read and i not in listed]
    rejected = [r for i, r in enumerate(results) if i < read and i not in listed]
    out.extend(unseen)
    out.extend(r for r in rejected if r.debug.get("rank"))
    if len(out) < keep:
        out.extend(r for r in rejected if r not in out)
        out = out[:keep]
    return pin(out)


async def respond(
    query: str,
    results: Sequence[Result],
    *,
    mode: str = "extract",
    model: Optional[str] = None,
    chat: Optional[Chat] = None,
    budget: float = 30.0,
    max_pages: int = MAX_PAGES,
) -> Answer:
    """Answer `query` from `results`, and say which pages were about it.

    Never raises. A dead answer model costs the answer, not the search: the
    caller already has results worth returning.
    """
    if not results:
        return Answer(reply=Reply(error="no results to answer from"))
    if mode not in ("extract", "synthesize"):
        return Answer(reply=Reply(error=f"unknown answer mode {mode!r}"))
    own = chat is None
    c = chat or chat_for(model)
    try:
        reply = await c.complete(build_messages(query, results, mode, max_pages),
                                 budget=budget)
    finally:
        if own:
            await c.aclose()
    if not reply.ok:
        return Answer(reply=reply)
    pages, text = parse_judgement(reply.text, min(len(results), max_pages))
    # The extract prompt is told to say this rather than guess; passing it
    # through as if it were an answer would be worse than returning nothing.
    if text.upper().startswith("NOT FOUND"):
        text = ""
    return Answer(text=text, reply=reply, pages=pages)


async def answer(
    query: str,
    results: Sequence[Result],
    *,
    mode: str = "extract",
    model: Optional[str] = None,
    chat: Optional[Chat] = None,
    budget: float = 30.0,
    max_pages: int = MAX_PAGES,
) -> tuple[str, Reply]:
    """`respond`, as (answer_text, the raw Reply)."""
    a = await respond(query, results, mode=mode, model=model, chat=chat,
                      budget=budget, max_pages=max_pages)
    return a.text, a.reply
