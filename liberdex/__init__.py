# liberdex: an index-free web search engine.
# Copyright (C) 2026 Gloria Capital GmbH
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version. See the LICENSE file for the full text.

"""liberdex: an index-free web search engine.

There is no crawl and no inverted index. An LLM names the pages that plausibly
hold the answer from parametric memory alone; liberdex fetches them, extracts
the text, and reranks it into a ranked SERP.
"""
import logging as _logging
import os as _os

# These libraries are chatty on import and on every call; liberdex is a library,
# so it stays quiet unless the host app turns logging back on.
_os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
for _name in ("bm25s", "flashrank", "huggingface_hub", "model2vec", "httpx",
              "trafilatura", "urllib3", "filelock"):
    _logging.getLogger(_name).setLevel(_logging.WARNING)

from .budget import apply as fit_budget
from .engine import Liberdex, guess_intent, search
from .fetch import Fetcher
from .llm import Chat, Reply
from .models import Profile, profiles, resolve_role
from .rank import Models, Ranker
from .types import Candidate, Doc, Plan, Result, SearchResponse

try:
    from importlib.metadata import version as _version
    __version__ = _version("liberdex")
except Exception:  # an un-installed checkout
    __version__ = "0.0.0"

# Where liberdex lives. One place, so an install command, a README link and a
# container tag cannot disagree.
REPO = "beneglo/liberdex"
SITE = "https://liberdex.net"
IMAGE = "ghcr.io/beneglo/liberdex"
__all__ = [
    "Liberdex", "search", "guess_intent", "Fetcher", "Ranker", "Models",
    "Candidate", "Doc", "Plan", "Result", "SearchResponse",
    "Profile", "profiles", "resolve_role", "Chat", "Reply", "fit_budget",
    "__version__", "REPO", "SITE", "IMAGE",
]
