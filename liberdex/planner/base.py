"""Planner protocol + incremental line-protocol parser."""
from __future__ import annotations

import re
from typing import AsyncIterator, Optional, Protocol

from ..routes import ROUTES
from ..types import Candidate, Deadline, Findings, Plan

VALID_INTENTS = {
    "navigational", "informational", "code", "academic", "news", "product",
    "local", "reference",
}

_URL_OK = re.compile(r"^https?://[a-z0-9.-]+\.[a-z]{2,24}(/|$|\?|#)", re.I)
_SEARCH_ENGINE = re.compile(
    r"^https?://([a-z0-9-]+\.)*(google|bing|duckduckgo|yandex|baidu|ecosia|"
    r"startpage|brave|searx)\.[a-z.]+/", re.I
)
_LANG = re.compile(r"^[a-z]{2,3}(?:[-_][a-z]{2,4})?$", re.I)
# A model that ignores the format tends to fall back on markdown: fences, list
# bullets, bold tags. Strip that skin rather than dropping the line.
_MD_SKIN = re.compile(r"^(?:[-*+]\s+|\d+[.)]\s+|>\s+|#+\s+)+")
_FENCE = re.compile(r"^\s*(?:```|~~~)")
# "**U** 0.9 https://...": emphasis wrapped around the tag letter itself.
_MD_TAG = re.compile(r"^[*_`]{1,2}\s*([A-Za-z])\s*[*_`]{1,2}\s")


def _clean_url(u: str) -> str:
    u = u.strip().strip('<>"\'`,;')
    u = u.rstrip(".")
    # A trailing ")" is prose punctuation only when the URL has no "(" of its
    # own; a disambiguated wiki title ends in one that belongs to it.
    while u.endswith(")") and u.count("(") < u.count(")"):
        u = u[:-1]
    if u.startswith("http://"):
        u = "https://" + u[7:]
    return u


def valid_url(u: str) -> bool:
    return bool(u) and bool(_URL_OK.match(u)) and not _SEARCH_ENGINE.match(u) and len(u) < 400


class PlanParser:
    """Feed it text deltas; it yields Candidates the moment a U line completes."""

    def __init__(self, query: str) -> None:
        self.plan = Plan(query=query)
        self._buf = ""
        self._seen: set[str] = set()

    def feed(self, delta: str) -> list[Candidate]:
        self._buf += delta
        out: list[Candidate] = []
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            c = self._line(line)
            if c is not None:
                out.append(c)
        return out

    def finish(self) -> list[Candidate]:
        out: list[Candidate] = []
        if self._buf.strip():
            c = self._line(self._buf)
            if c is not None:
                out.append(c)
        self._buf = ""
        self.plan.candidates.extend(out)
        return out

    def _line(self, line: str) -> Optional[Candidate]:
        line = line.strip()
        if not line or len(line) > 2000 or _FENCE.match(line):
            return None
        line = _MD_TAG.sub(r"\1 ", _MD_SKIN.sub("", line).strip()).strip("*`")
        if not line:
            return None
        self.plan.raw += line + "\n"
        tag, _, rest = line.partition(" ")
        tag = tag.upper().rstrip(":")
        rest = rest.strip()

        if tag == "I":
            v = rest.split()[0].lower() if rest else ""
            if v in VALID_INTENTS:
                self.plan.intent = v  # type: ignore[assignment]
            return None

        if tag == "L":
            v = rest.split()[0].strip() if rest else ""
            if _LANG.match(v):
                self.plan.lang = v.split("-")[0].split("_")[0].lower()
            return None

        if tag == "R":
            # "wikipedia: HNSW | reddit: r/rust async trait": the planner
            # writes each site's own search query. Bare names stay valid and
            # fall back to the generic trim.
            for part in rest.split("|"):
                part = part.strip()
                if not part:
                    continue
                name, sep, rq = part.partition(":")
                name = name.strip().lower()
                if name not in ROUTES:
                    # Tolerate a comma-separated bare list on one segment.
                    for alt in re.split(r"[,\s]+", part.lower()):
                        alt = alt.strip()
                        if alt in ROUTES and alt not in self.plan.routes:
                            self.plan.routes.append(alt)
                    continue
                if name not in self.plan.routes:
                    self.plan.routes.append(name)
                rq = rq.strip() if sep else ""
                if rq:
                    self.plan.route_queries[name] = rq[:240]
            return None

        if tag == "X":
            for part in rest.split("|"):
                part = part.strip()
                if 1 < len(part) < 160:
                    self.plan.expansions.append(part)
            return None

        if tag == "H":
            url = _clean_url(rest.split(" :: ")[0].split(" ")[0])
            if valid_url(url) and url not in self._seen:
                self._seen.add(url)
                self.plan.hubs.append(url)
            return None

        if tag == "P":
            m = re.match(r"^(\d{1,2})\s+(\S+)$", rest)
            if not m:
                return None
            url = _clean_url(m.group(2))
            if not valid_url(url) or url in self._seen:
                return None
            self._seen.add(url)
            cand = Candidate(url=url, prior=1.0, source="llm",
                             rank=max(1, min(10, int(m.group(1)))))
            self.plan.candidates.append(cand)
            return cand

        if tag == "U":
            m = re.match(r"^([01](?:\.\d+)?|\.\d+)\s+(\S+)(?:\s*::\s*(.*))?$", rest)
            if m:
                conf, url, why = float(m.group(1)), _clean_url(m.group(2)), (m.group(3) or "")
            else:
                # Tolerate a missing confidence value rather than dropping the URL.
                parts = rest.split(" :: ", 1)
                url = _clean_url(parts[0].split(" ")[0])
                conf, why = 0.55, (parts[1] if len(parts) > 1 else "")
            if not valid_url(url) or url in self._seen:
                return None
            self._seen.add(url)
            cand = Candidate(
                url=url,
                reason=why.strip()[:200],
                prior=max(0.05, min(1.0, conf)),
                source="llm",
            )
            self.plan.candidates.append(cand)
            return cand
        return None


def plan_from_dict(query: str, data: dict) -> Plan:
    """A plan written by someone other than liberdex's own planner.

    The host of an agent CLI is already a strong model the user pays for; when
    it writes the plan itself there is no second LLM call and no seconds to
    first URL. The shape is the line protocol as JSON, in liberdex's words:

        {"candidates": [{"url": ..., "prior": 0.8, "reason": ...}, ...],
         "routes": {"wikipedia": "tcp three-way handshake", ...},
         "hubs": [...], "expansions": [...], "intent": ..., "lang": ...}

    Every guard the parser applies to a model's stream applies here too, so a
    search-engine URL or a 400-character one is dropped rather than fetched and
    a supplied plan can do nothing the planner could not.
    """
    plan = Plan(query=query)
    intent = str(data.get("intent") or "").strip().lower()
    if intent in VALID_INTENTS:
        plan.intent = intent  # type: ignore[assignment]
    lang = str(data.get("lang") or "").strip().lower()
    if _LANG.match(lang):
        plan.lang = lang.replace("_", "-")
    seen: set[str] = set()
    for item in (data.get("candidates") or [])[:40]:
        if isinstance(item, str):
            item = {"url": item}
        if not isinstance(item, dict):
            continue
        url = _clean_url(str(item.get("url") or ""))
        if not valid_url(url) or url in seen:
            continue
        seen.add(url)
        try:
            prior = float(item.get("prior", 0.55))
        except (TypeError, ValueError):
            prior = 0.55
        plan.candidates.append(Candidate(
            url=url, reason=str(item.get("reason") or "")[:200],
            prior=max(0.05, min(1.0, prior)), source="llm", lang=plan.lang,
        ))
    routes = data.get("routes") or {}
    if isinstance(routes, list):
        routes = {str(r): "" for r in routes}
    for name, q in list(routes.items())[:8]:
        name = str(name).strip().lower()
        if name not in ROUTES:
            continue
        plan.routes.append(name)
        q = " ".join(str(q or "").split())[:200]
        if q:
            plan.route_queries[name] = q
    for h in (data.get("hubs") or [])[:6]:
        h = _clean_url(str(h))
        if valid_url(h) and h not in plan.hubs:
            plan.hubs.append(h)
    for x in (data.get("expansions") or [])[:8]:
        x = " ".join(str(x or "").split())[:120]
        if x and x not in plan.expansions:
            plan.expansions.append(x)
    return plan


class SuppliedPlanner:
    """A planner that already has the answer: the plan was handed to it.

    Yields every candidate at once and is done. There is no second pass: the
    caller who wrote the plan is the one who would write it, and it is not on
    this side of the call.
    """
    name = "supplied"

    def __init__(self, plan: Plan) -> None:
        self.plan = plan

    async def stream(
        self, query: str, deadline: Deadline
    ) -> AsyncIterator[tuple[str, object]]:
        for cand in self.plan.candidates:
            yield ("candidate", cand)
        yield ("done", self.plan)


class Planner(Protocol):
    async def stream(
        self, query: str, deadline: Deadline
    ) -> AsyncIterator[tuple[str, object]]:
        """Yield ('candidate', Candidate) / ('meta', Plan) / ('done', Plan)."""
        ...

    async def refine(
        self, query: str, findings: Findings, deadline: Deadline
    ) -> AsyncIterator[tuple[str, object]]:
        """The second pass: the same events, planned over the first round's
        report rather than blind. Optional; the engine checks for it."""
        ...
