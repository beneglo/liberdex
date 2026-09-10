"""HTML to title, main text, metadata and links.

selectolax (lexbor) parses the page: it is an order of magnitude faster than
lxml, and one tree serves the metadata, link, search-form and main-text
passes. Main text comes from a Readability-style density score over the
stripped tree; trafilatura is optional, for markdown and for callers that
ask for it.
"""
from __future__ import annotations

import re
import threading
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import orjson
from selectolax.lexbor import LexborHTMLParser

from .fetch import registrable
from .types import Doc

_WS = re.compile(r"[ \t\r\f\v]+")
_NL = re.compile(r"\n{3,}")
_CHARSET = re.compile(r"charset=([\w\-]+)", re.I)

DROP_TAGS = (
    "script", "style", "noscript", "svg", "nav", "header", "footer", "aside",
    "form", "iframe", "template", "button", "select",
)

MAIN_SELECTORS = (
    "article", "main", '[role="main"]', "#content", ".post-content",
    ".article-content", ".entry-content", ".markdown-body", "#main-content",
)

_NAV_NAME = re.compile(
    r"(^|[-_ ])(nav|navbar|menu|breadcrumb|sidebar|footer|header|masthead|"
    r"topbar|toolbar|pagination|pager|social|share|cookie|banner|megamenu|"
    r"site-?map|skip-?link|lang(uage)?-?(switch|select|picker)|"
    r"country-?(switch|select|picker)|utility-?nav)([-_ ]|$)", re.I)

_BAD_LINK = re.compile(
    r"^(javascript:|mailto:|tel:|#)|\.(png|jpe?g|gif|svg|webp|ico|css|js|zip|"
    r"tar|gz|mp4|mp3|woff2?|ttf)(\?|$)", re.I
)

# A link that does something rather than pointing at something: edit forms,
# "search for X", sign-in, share. The site writes the query into them verbatim
# (a wiki redlink offering to create the article just asked for is a perfect
# anchor match and a blank edit form), so they read as the best link on the
# page and can never hold an answer.
_ACTION_LINK = re.compile(
    r"[?&](?:action=(?:edit|submit|history|raw|watch|delete|purge|info)"
    r"|veaction=|redlink=1|do=(?:edit|login|register)|replytocom=)"
    # A wiki's own search results, in whatever language it names the namespace:
    # `Special:` is only the English spelling, so the script endpoint plus a
    # query parameter is the rule that holds across editions.
    r"|index\.php\?[^#]*[?&]search="
    r"|/wiki/(?:Special|Spezial|Spécial|Especial|Speciale|Spesial|Specjalna|"
    r"Служебная|Служебна|特別|특수):"
    r"|/(?:login|signin|sign-in|signup|sign-up|register|logout|"
    r"cart|checkout)(?:[/?]|$)", re.I)


def decode(body: bytes, content_type: str = "") -> str:
    if not body:
        return ""
    m = _CHARSET.search(content_type or "")
    if m:
        try:
            return body.decode(m.group(1), "replace")
        except LookupError:
            pass
    # Peek at a meta charset before defaulting to utf-8.
    head = body[:2048]
    m = _CHARSET.search(head.decode("ascii", "ignore"))
    if m and m.group(1).lower() not in ("utf-8", "utf8"):
        try:
            return body.decode(m.group(1), "replace")
        except LookupError:
            pass
    return body.decode("utf-8", "replace")


def _meta(tree: LexborHTMLParser, *names: str) -> str:
    for n in names:
        for attr in ("property", "name", "itemprop"):
            node = tree.css_first(f'meta[{attr}="{n}"]')
            if node:
                v = (node.attributes.get("content") or "").strip()
                if v:
                    return v
    return ""


_JSONLD_DATE_KEYS = ("datePublished", "dateCreated", "uploadDate", "datePosted")


def _jsonld_date(tree: LexborHTMLParser) -> str:
    for node in tree.css('script[type="application/ld+json"]')[:4]:
        raw = node.text(strip=True)
        if not raw or len(raw) > 200_000:
            continue
        try:
            data = orjson.loads(raw)
        except Exception:
            continue
        stack = [data]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                for k in _JSONLD_DATE_KEYS:
                    v = cur.get(k)
                    if isinstance(v, str) and len(v) >= 8:
                        return v[:10]
                stack.extend(cur.values())
            elif isinstance(cur, list):
                stack.extend(cur)
    return ""


# Chrome that survives the class and id filter because it carries no class
# worth matching: body text on a page that happens to be interface. Every entry
# is an exact phrase from a site fetched constantly, and none of them can
# contain an answer. A pattern general enough to swallow a sentence does not
# belong here.
_CHROME = re.compile(
    r"Uh oh! There was an error while loading\.\s*Please reload this page\.|"
    r"Jump to content|Skip to (?:main )?content|"
    r"From Wikipedia, the free encyclopedia|"
    r"You signed (?:in|out) (?:in|from) another tab or window\.[^.]*\.|"
    r"Please enable JavaScript[^.]*\.",
    re.I,
)


def _clean(text: str, limit: int = 250_000) -> str:
    """Collapse whitespace and strip known chrome.

    The limit is generous on purpose: large reference pages put the paragraph
    that answers the query well past any modest cap, and keeping the whole
    document buys exact-term matching over all of it.
    """
    text = _CHROME.sub(" ", text)
    text = _WS.sub(" ", text)
    text = _NL.sub("\n\n", text)
    return text.strip()[:limit]


# The word has to start a class token, because a class token is read left to
# right: `sidebar-right` is a sidebar, `overflow-hidden` is not hidden, and
# `no-print-header` is not a header. A few conventional prefixes (`js-`, `is-`)
# are allowed in front, since they carry no meaning of their own.
_BOILER = re.compile(
    r"(?:^|\s)(?:js-|c-|l-|u-|is-|has-)?"
    r"(nav|navbar|menu|sidebar|side-bar|footer|header|masthead|comment|"
    r"cookie|consent|banner|advert|ad|ads|promo|related|share|social|breadcrumb|"
    r"pagination|paginate|subscribe|newsletter|modal|popup|overlay|toc|"
    r"table-of-contents|skip-link|screen-reader|sr-only|hidden|widget|"
    r"site-info|copyright|legal|disclaimer)(?:[\s_-]|$)",
    re.I,
)

_BLOCK_TAGS = ("article", "main", "section", "div", "td")

# MediaWiki's furniture, by the class names MediaWiki gives it. Wikipedia is
# fetched on most searches and none of its chrome is caught by the name test
# above: a navbox is `navbox`, not `nav`; the category bar is `catlinks`. The
# cost is not cosmetic: a navbox listing every award a society gives repeats
# the query's words more densely than the laureate table with the answer in
# it, and wins the snippet. Each selector is a MediaWiki class or id, exact, so
# nothing outside a wiki matches. The infobox and the reference list are not
# here: both carry facts.
_MEDIAWIKI_CHROME = (
    ".navbox", ".navbox-styles", "#catlinks", ".catlinks", ".printfooter",
    ".mw-jump-link", ".ambox", ".mbox-small", ".asbox", ".dmbox", ".ombox",
    ".hatnote", ".mw-editsection", "sup.reference", ".mw-cite-backlink",
    ".mw-indicators", ".sistersitebox", ".side-box", ".noprint",
    ".mw-authority-control", "#mw-navigation", ".mw-portlet",
    ".vector-header-container", ".vector-page-toolbar", ".mw-empty-elt",
)
_MEDIAWIKI_SEL = ", ".join(_MEDIAWIKI_CHROME)


_ROOTISH = {"html", "body", "head", "#document"}


def _drop(node) -> None:
    """decompose() raises on root-ish nodes and on already-detached ones."""
    if (node.tag or "") in _ROOTISH:
        return
    try:
        node.decompose()
    except Exception:
        pass


# A layout wrapper is named after what it contains, not after what it is:
# `<div class="container sidebar-right">` says where the sidebar goes, not
# that the div is one, and dropping it takes the whole article with it. So
# anything holding a substantial main-content element is a wrapper and is
# kept; the chrome inside it is still dropped on its own account.
_MAIN_SEL = ", ".join(MAIN_SELECTORS)
_WRAPS_MAIN_CHARS = 400
# Chrome is a small share of a page's text; the article is most of it. A named
# node holding this much of the page is a container that happens to carry a
# boilerplate word, not the boilerplate itself.
_BULK_SHARE = 0.25
_BULK_CHARS = 1000


def _text_len(node) -> int:
    try:
        return len(node.text(separator="", strip=False))
    except Exception:
        return 0


def _wraps_main(node) -> bool:
    inner = node.css_first(_MAIN_SEL)
    return inner is not None and _text_len(inner) > _WRAPS_MAIN_CHARS


def _strip_boilerplate(tree: LexborHTMLParser) -> None:
    """Remove interface furniture, and nothing that is holding the page.

    The name test is a substring match over a class attribute, and class
    attributes are full of words that mean something else in context
    (`overflow-hidden`, `field--label-hidden`, `no-print-header`), often on
    the element wrapping the whole document. Two structural checks stand in
    front of the drop: is this element wrapping the main-content element, and
    is it carrying the bulk of the page's text.
    """
    for tag in DROP_TAGS:
        for node in tree.css(tag):
            _drop(node)
    # Only a wiki carries these names, and a wiki carries hundreds of them.
    if tree.css_first(".mw-parser-output, #mw-content-text, .mw-body") is not None:
        for node in tree.css(_MEDIAWIKI_SEL):
            _drop(node)
    body = tree.body
    total = _text_len(body) if body is not None else 0
    for node in tree.css("[class],[id],[role],[aria-hidden]"):
        attrs = node.attributes
        aria = attrs.get("aria-hidden") == "true"
        role = attrs.get("role") or ""
        blob = f"{attrs.get('class') or ''} {attrs.get('id') or ''}"
        named = bool(blob.strip() and _BOILER.search(blob))
        if not (aria or named
                or role in ("navigation", "banner", "contentinfo",
                            "complementary", "search")):
            continue
        if _wraps_main(node):
            continue
        n = _text_len(node)
        if total and n > _BULK_CHARS and n > _BULK_SHARE * total:
            continue
        _drop(node)


# An alphabetic run this long is not a word but two blocks concatenated
# because the source HTML had no whitespace between them.
_RUN_TOGETHER = re.compile(r"[A-Za-z]{46,}")


def _minified(text: str) -> bool:
    """Whether block boundaries vanished, rather than one long identifier.

    One 60-character identifier in a long reference page is not minification;
    real minification produces these runs constantly.
    """
    if len(text) < 2000:
        return bool(_RUN_TOGETHER.search(text))
    return len(_RUN_TOGETHER.findall(text)) > max(3, len(text) // 20_000)


def _node_text(node) -> str:
    """Text as a browser would lay it out, not one line per inline element.

    `separator="\n"` puts a break between every inline child, and Sphinx
    wraps each token of a signature in its own span, so `get(key,
    default=None)` would come out on four lines and never match. Source
    whitespace reproduces what the page reads like: HTML puts real whitespace
    between blocks and none inside a signature.
    """
    text = node.text(separator="", strip=False)
    # Unless the HTML is minified, in which case block boundaries have gone
    # and breaking on children is the better of two bad options.
    if _minified(text):
        return node.text(separator="\n", strip=True)
    return text


# What is left of a block once its anchor text is removed. A menu has almost
# nothing, since its links are its content. A list article has an article's
# worth: the year beside each name, the dates, the lede above the list.
_NAV_PROSE_CHARS = 1200


def _density_score(node) -> tuple[float, str]:
    """Readability-style: reward text, punish link-heavy navigation blocks.

    The link-ratio test on its own throws away a whole class of article.
    Award pages, discographies and "List of ..." articles are lists of linked
    names with the fact beside them, and a wiki links the years too. So the
    ratio disqualifies a block only when what remains after the anchors is
    too thin to be an article; a list article still pays for its links
    through the `1 - link_ratio` factor.
    """
    text = _node_text(node)
    n = len(text)
    if n < 200:
        return (0.0, text)
    link_chars = sum(len(a.text(strip=True)) for a in node.css("a"))
    link_ratio = link_chars / max(1, n)
    if link_ratio > 0.55 and (n - link_chars) < _NAV_PROSE_CHARS:
        return (0.0, text)
    # Paragraph count is the strongest single signal of prose.
    paras = len(node.css("p"))
    score = n * (1.0 - link_ratio) * (1.0 + min(paras, 40) / 20.0)
    return (score, text)


# Elements a browser lays out on their own line. Anything not here (span, a,
# code, em, strong) is inline and must not be separated; see _node_text.
_LINE_TAGS = (
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "div", "section", "article",
    "header", "footer", "main", "aside", "nav", "li", "tr", "td", "th",
    "blockquote", "pre", "figcaption", "dt", "dd", "br", "hr",
)


def _separate_blocks(tree: LexborHTMLParser) -> None:
    """Put a newline after every block element that the source did not.

    A page can be neither minified nor properly spaced: some themes emit
    adjacent headings as one token. Inserting the break in the DOM rather
    than passing `separator="\n"` keeps the distinction that matters: block
    boundaries get whitespace, inline ones still do not.
    """
    body = tree.body
    if body is None:
        return
    for node in body.css(",".join(_LINE_TAGS)):
        try:
            node.insert_after("\n")
        except Exception:
            # Detached or root-ish node; the text is readable without.
            continue


def _selectolax_text(tree: LexborHTMLParser) -> str:
    _strip_boilerplate(tree)
    _separate_blocks(tree)
    for sel in MAIN_SELECTORS:
        node = tree.css_first(sel)
        if node:
            score, text = _density_score(node)
            if score > 0 and len(text) > 400:
                return text
    best_score, best_text = 0.0, ""
    body = tree.body
    if body is None:
        return ""
    seen = 0
    for tag in _BLOCK_TAGS:
        for node in body.css(tag):
            seen += 1
            if seen > 900:  # pathological DOM: stop scanning, use what we have
                break
            score, text = _density_score(node)
            if score > best_score:
                best_score, best_text = score, text
        if seen > 900:
            break
    if best_score <= 0:
        return _node_text(body)
    return best_text


_find_date = None


def _get_htmldate():
    """htmldate's finder, imported on first use. It ships with trafilatura."""
    global _find_date
    if _find_date is None:
        try:
            from htmldate import find_date
            _find_date = find_date
        except Exception:
            _find_date = False
    return _find_date


# Enough of the page to carry a date without paying to scan a whole archive.
_DATE_HTML_LIMIT = 300_000


def _htmldate(html: str, url: str) -> str:
    """A publication date when the page declares none in its metadata.

    Some pages print their date only in the body: a research index's report,
    an agency's release. A query that names a
    period cannot be answered without one. htmldate runs only where the cheap
    path already gave up, so pages that declare a date properly pay nothing.
    """
    fn = _get_htmldate()
    if not fn or len(html) > _DATE_HTML_LIMIT:
        return ""
    try:
        # Metadata plus the obvious in-page patterns. The extensive pass costs
        # several times the latency for a couple of points of accuracy.
        return fn(html, url=url, extensive_search=False) or ""
    except Exception:
        return ""


_trafilatura_extract = None


def _get_trafilatura():
    """trafilatura's extractor, imported on first use: the import is slow."""
    global _trafilatura_extract
    if _trafilatura_extract is None:
        try:
            from trafilatura import extract as tex
            _trafilatura_extract = tex
        except Exception:
            _trafilatura_extract = False
    return _trafilatura_extract


# The DOM element that carries a link's own sentence: the list item, table
# cell, definition or paragraph it sits in. Chakrabarti, Punera & Subramanyam
# (WWW 2002) and Pant & Srinivasan (TKDE 2006) both find the anchor's immediate
# neighbourhood predicts the target far better than the anchor alone, and that
# a window widening past it gets worse. So this stops at the first block
# ancestor.
_CONTEXT_TAGS = frozenset({"li", "td", "th", "dd", "dt", "p", "figcaption",
                           "article", "section", "h1", "h2", "h3", "h4"})
_CONTEXT_CHARS = 220
# A menu is a container whose links are nearly all it contains (Cai et al.,
# SIGIR 2004, block-level link analysis). The ratio of anchor text to text is
# what tells a list of reports from the country picker three divs above it.
_NAV_MIN_LINKS = 6
_NAV_LINK_DENSITY = 0.7
_NAV_ROLES = frozenset({"navigation", "banner", "contentinfo", "search",
                        "menu", "menubar", "tablist"})


def _nav_ancestor(node) -> bool:
    """Is this link inside the site's furniture rather than its content?

    On a CMS the furniture is most of the anchors on a page (the country
    picker, the services menu, the office locations), and a menu label is
    exactly one topic word, so they read best against a query. Dropping them
    before anything is scored stops a search for a report from following an
    organisation's list of its own field offices.
    """
    hops = 0
    cur = node.parent
    while cur is not None and hops < 8:
        tag = (cur.tag or "").lower()
        if tag in ("nav", "header", "footer", "aside"):
            return True
        attrs = cur.attributes
        if (attrs.get("role") or "") in _NAV_ROLES:
            return True
        blob = f"{attrs.get('class') or ''} {attrs.get('id') or ''}"
        if blob.strip() and _NAV_NAME.search(blob):
            return True
        hops += 1
        cur = cur.parent
    return False


def _link_context(node) -> str:
    """The text of the block the link sits in, minus the link's own words."""
    cur = node.parent
    hops = 0
    while cur is not None and hops < 5:
        if (cur.tag or "").lower() in _CONTEXT_TAGS:
            try:
                txt = _WS.sub(" ", cur.text(separator=" ", strip=True))
            except Exception:
                return ""
            if len(txt) > 4:
                return txt[:_CONTEXT_CHARS]
            return ""
        hops += 1
        cur = cur.parent
    return ""


def extract_links(tree: LexborHTMLParser, base: str, limit: int = 300
                  ) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    # Content links first, furniture only if the page has room left. A page
    # that is genuinely all navigation still returns its links.
    deferred: list[tuple[str, str, str]] = []
    dense: set[int] = _dense_link_blocks(tree)
    for a in tree.css("a[href]"):
        href = (a.attributes.get("href") or "").strip()
        if not href or _BAD_LINK.match(href) or _ACTION_LINK.search(href):
            continue
        try:
            url = urljoin(base, href)
        except Exception:
            continue
        if not url.startswith(("http://", "https://")):
            continue
        url = url.split("#", 1)[0]
        if url in seen:
            continue
        seen.add(url)
        anchor = _WS.sub(" ", a.text(strip=True))[:160]
        if _nav_ancestor(a) or id(a.parent) in dense:
            deferred.append((url, anchor, ""))
            continue
        out.append((url, anchor, _link_context(a)))
        if len(out) >= limit:
            return out
    for row in deferred:
        if len(out) >= limit:
            break
        out.append(row)
    return out


def _dense_link_blocks(tree: LexborHTMLParser) -> set[int]:
    """Parents whose children are mostly links: menus, indexes of everything."""
    counts: dict[int, tuple[int, int, object]] = {}
    for a in tree.css("a[href]"):
        par = a.parent
        if par is None:
            continue
        try:
            n = len(a.text(strip=True))
        except Exception:
            n = 0
        c, t, _ = counts.get(id(par), (0, 0, par))
        counts[id(par)] = (c + 1, t + n, par)
    out: set[int] = set()
    for key, (c, anchor_chars, par) in counts.items():
        if c < _NAV_MIN_LINKS:
            continue
        total = _text_len(par)
        if total and anchor_chars / total >= _NAV_LINK_DENSITY:
            out.add(key)
    return out


# A page nearly always declares its icon, but the well-known path is served by
# most sites that do not, so a missing <link> is not a missing favicon.
_ICON_SELECTORS = (
    'link[rel="icon"]', 'link[rel="shortcut icon"]', 'link[rel="apple-touch-icon"]',
    'link[rel="apple-touch-icon-precomposed"]', 'link[rel="mask-icon"]',
)
# Tracking pixels, spacers and sprite sheets all look like images to a parser.
_BAD_IMAGE = re.compile(r"(^data:|/(pixel|spacer|blank|1x1|tracking)[.\-_]|\.svg(\?|$))", re.I)


def favicon_of(tree: LexborHTMLParser, base: str) -> str:
    for sel in _ICON_SELECTORS:
        node = tree.css_first(sel)
        if not node:
            continue
        href = (node.attributes.get("href") or "").strip()
        if not href:
            continue
        try:
            url = urljoin(base, href)
        except Exception:
            continue
        if url.startswith(("http://", "https://", "data:")):
            return url[:600]
    parts = urlsplit(base)
    if parts.scheme and parts.netloc:
        return f"{parts.scheme}://{parts.netloc}/favicon.ico"
    return ""


# The name a site gives the box you type a query into. Ordered: an exact match
# on one of these beats a fuzzy one, and `q` is the near-universal convention.
_QUERY_NAMES = (
    "q", "query", "s", "search", "keyword", "keywords", "term", "terms",
    "suche", "suchbegriff", "sword", "querystr", "search_query", "searchword",
    "text", "wd", "kw", "recherche", "busca", "ricerca", "szukaj",
)
# Fields a search form carries that are not the query: submitting them empty is
# what the site's own form does, so they are kept, but none of them is the box.
_NOT_QUERY = re.compile(
    r"csrf|token|nonce|referrer|__|controller|action|hash|lang|locale|"
    r"page|limit|sort|order|type|category|filter|submit|button", re.I)


def _is_search_form(node) -> bool:
    attrs = node.attributes
    if (attrs.get("role") or "").lower() == "search":
        return True
    blob = " ".join(filter(None, (
        attrs.get("action"), attrs.get("class"), attrs.get("id"),
        attrs.get("name"), attrs.get("data-testid"))))
    if re.search(r"search|suche|recherche|busca|ricerca|szukaj", blob, re.I):
        return True
    return node.css_first('input[type="search"]') is not None


def _query_field(node) -> str:
    """The name of the input that takes the query, or "" if there is none."""
    cands: list[tuple[int, str]] = []
    for inp in node.css("input, textarea"):
        a = inp.attributes
        itype = (a.get("type") or "text").lower()
        if itype in ("hidden", "submit", "button", "image", "checkbox", "radio"):
            continue
        name = (a.get("name") or "").strip()
        if not name or _NOT_QUERY.search(name):
            continue
        low = name.lower()
        # The parameter's own leaf, not the wrapper: frameworks nest the box
        # (`search[sword]`) but so does every faceted filter (`search[priceMax]`
        # is a slider), and only the leaf tells them apart.
        leaf = re.split(r"[\[\]]+", low.rstrip("]"))[-1] or low
        if low in _QUERY_NAMES or leaf in _QUERY_NAMES:
            rank = _QUERY_NAMES.index(low if low in _QUERY_NAMES else leaf)
        elif itype == "search":
            rank = len(_QUERY_NAMES)
        else:
            continue
        cands.append((rank, name))
    if not cands:
        return ""
    return min(cands)[1]


def search_template(tree: LexborHTMLParser, base: str) -> str:
    """The site's own search URL, as a template with `{}` where the query goes.

    When the planner recalls the right host but not the right path, the
    site's search box is the missing index: every publisher runs one over its
    own content, and the form states the endpoint and the parameter name
    exactly, where guessing `/search?q=` is wrong more often than right.

    GET forms only. A POST search cannot be expressed as a URL, and issuing
    one would be writing to someone else's site rather than reading from it.
    """
    page_host = registrable((urlsplit(base).hostname or "").lower())
    if not page_host:
        return ""
    for form in tree.css("form"):
        if not _is_search_form(form):
            continue
        if (form.attributes.get("method") or "get").lower() != "get":
            continue
        field = _query_field(form)
        if not field:
            continue
        action = (form.attributes.get("action") or "").strip() or base
        try:
            url = urljoin(base, action)
        except Exception:
            continue
        if not url.startswith(("http://", "https://")):
            continue
        parts = urlsplit(url)
        # The site's own search only. A form pointing elsewhere is a hosted
        # search widget, somebody else's index. A search subdomain of the same
        # registrable domain still counts.
        if registrable((parts.hostname or "").lower()) != page_host:
            continue
        # The action's own query string selects the search controller on some
        # frameworks, so it is kept and the query field appended to it.
        keep = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                if k != field]
        # The form's other visible fields are scope pickers; their empty value
        # is what the site sends.
        pairs = keep + [(field, "\x00")]
        qs = urlencode(pairs).replace("%00", "{}")
        return urlunsplit((parts.scheme, parts.netloc, parts.path, qs, ""))[:600]
    return ""


def images_of(tree: LexborHTMLParser, base: str, limit: int = 6) -> list[str]:
    """The social card image first, then the largest-looking inline images.

    og:image is the one image a publisher chose to represent the page, so it
    leads regardless of where it sits in the document.
    """
    out: list[str] = []
    seen: set[str] = set()

    def add(href: str) -> None:
        href = (href or "").strip()
        if not href or _BAD_IMAGE.search(href):
            return
        try:
            url = urljoin(base, href)
        except Exception:
            return
        if not url.startswith(("http://", "https://")) or url in seen:
            return
        seen.add(url)
        out.append(url[:600])

    add(_meta(tree, "og:image", "og:image:url", "twitter:image", "twitter:image:src"))
    for img in tree.css("img"):
        if len(out) >= limit:
            break
        attrs = img.attributes
        add(attrs.get("src") or attrs.get("data-src") or "")
    return out[:limit]


# The fetcher never keeps more than this much of a page, and in fast mode
# trafilatura's cost is bounded well inside it.
MARKDOWN_MAX_HTML = 2_000_000


def to_markdown(html: str, url: str = "") -> str:
    """Main content as markdown, or "" if trafilatura cannot or will not.

    Callers run this on the handful of pages that made the SERP, never on
    everything that was fetched, and always off the event loop.
    """
    if not html or len(html) >= MARKDOWN_MAX_HTML:
        return ""
    tex = _get_trafilatura()
    if not tex:
        return ""
    try:
        return (tex(
            html, output_format="markdown", favor_recall=True,
            include_comments=False, include_tables=True,
            include_formatting=True, include_links=True,
            fast=True, url=url or None,
        ) or "").strip()
    except Exception:
        return ""


_PDF_MAGIC = b"%PDF-"
# Enough of a report to rank and to answer from. Ranking reads the first few
# thousand characters of a document, so pages past this contribute nothing.
PDF_MAX_PAGES = 12
# What a converter writes into the Title field when the author set none. A
# filename is a better title than "Microsoft Word - final_v3.docx".
_PDF_BAD_TITLE = re.compile(
    r"^(microsoft word|microsoft powerpoint|microsoft excel|untitled|unbenannt|"
    r"dokument\d*|print|druck|layout|.*\.(docx?|pptx?|indd|qxd|pdf)\s*$|"
    r"[a-f0-9-]{16,}$)", re.I)


# PDFium is not thread-safe and extraction runs in a thread pool: two documents
# parsed at once abort the process with SIGABRT rather than raise. The library
# ships no lock of its own, so every entry into PDFium goes through this one.
# PDFs therefore parse one at a time, which at tens of milliseconds each is far
# cheaper than a crash.
_PDF_LOCK = threading.Lock()


_PDFIUM = None


def _pdfium():
    """pypdfium2, imported on the first PDF and never on a machine that fetches none."""
    global _PDFIUM
    if _PDFIUM is None:
        try:
            import pypdfium2
            _PDFIUM = pypdfium2
        except Exception:
            _PDFIUM = False
    return _PDFIUM or None


# The banner a publisher stamps on page one, above the document's own name:
# a licence grant, a copyright line, a preprint stamp, a confidentiality note.
_PDF_FURNITURE = re.compile(
    r"\b(all rights reserved|copyright|\(c\)\s*\d{4}|©|licen[sc]e|"
    r"permission is (hereby )?granted|hereby grants|creative commons|"
    r"arxiv:\s*\d|doi:|issn|isbn|confidential|embargo|"
    r"provided proper attribution|for internal use|draft -|preprint)\b", re.I)


def _headline(text: str) -> str:
    """The first line of a document that reads as its title.

    Many PDFs carry no Title at all and the title is the first thing printed
    on page one. A title is short, is not a sentence, and is not a lone date
    or page number.
    """
    for raw in text[:1500].split("\n"):
        line = _WS.sub(" ", raw).strip(" \t-—·|")
        if not (12 <= len(line) <= 180):
            continue
        if line.endswith((".", ";", ",")) and len(line) > 90:
            continue  # a paragraph, not a heading
        if _PDF_FURNITURE.search(line):
            continue
        # A line starting mid-sentence is the tail of the banner above it,
        # a licence grant that wrapped. Better the filename than someone
        # else's small print.
        if line[:1].islower():
            continue
        letters = sum(c.isalpha() for c in line)
        if letters / len(line) < 0.5 or len(line.split()) < 2:
            continue
        return line[:300]
    return ""


def _pdf_title(meta: dict, url: str, text: str = "") -> str:
    """The document's name: its own Title, its first heading, or its filename."""
    t = _WS.sub(" ", str(meta.get("Title") or "")).strip()
    if t and not _PDF_BAD_TITLE.match(t) and len(t) > 4:
        return t[:300]
    stem = urlsplit(url).path.rsplit("/", 1)[-1]
    stem = re.sub(r"\.pdf$", "", stem, flags=re.I).replace("%20", " ")
    stem = _WS.sub(" ", re.sub(r"[-_+]+", " ", stem)).strip()
    # A filename that is only an identifier names nothing; the page itself does.
    if not stem or sum(c.isalpha() for c in stem) < max(4, 0.4 * len(stem)):
        return (_headline(text) or stem or t or "")[:300]
    return stem[:300]


def extract_pdf(doc: Doc) -> Doc:
    """Text and title from a PDF, via PDFium.

    For statistics offices, ministries, agencies, standards bodies and
    preprint servers, the PDF is where the authoritative version lives, and
    treating one as an unfetchable binary leaves a hole no ranking can fill.

    Scanned PDFs carry no text layer and come back empty. That is reported as
    is rather than papered over with OCR, which is a different product.
    """
    pdfium = _pdfium()
    if pdfium is None:
        return doc
    parts: list[str] = []
    meta: dict = {}
    with _PDF_LOCK:
        pdf = None
        try:
            pdf = pdfium.PdfDocument(doc.body)
            n = len(pdf)
            for i in range(min(n, PDF_MAX_PAGES)):
                try:
                    parts.append(pdf[i].get_textpage().get_text_range())
                except Exception:
                    continue
            try:
                meta = pdf.get_metadata_dict() or {}
            except Exception:
                meta = {}
        except Exception:
            return doc
        finally:
            if pdf is not None:
                try:
                    pdf.close()
                except Exception:
                    pass
    text = _clean("\n".join(parts))
    if not text:
        return doc
    doc.text = text
    doc.title = doc.title or _pdf_title(meta, doc.final_url, text)
    # A PDF states its own date in metadata far more reliably than a web page
    # states one in its markup; `D:20240411103000+02'00'` is the standard form.
    raw_date = str(meta.get("CreationDate") or meta.get("ModDate") or "")
    m = re.search(r"(\d{4})(\d{2})(\d{2})", raw_date)
    if m:
        doc.published = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return doc


def extract(doc: Doc, *, want_links: bool = False, use_trafilatura: bool = False,
            want_media: bool = False, want_date: bool = False) -> Doc:
    """Populate doc.title/text/description/published/links in place."""
    if not doc.body:
        return doc
    host = urlsplit(doc.final_url).hostname or ""
    doc.site = host[4:] if host.startswith("www.") else host

    ctype = doc.content_type or ""
    if "pdf" in ctype or doc.body[:5] == _PDF_MAGIC:
        return extract_pdf(doc)
    if "json" in ctype:
        doc.text = _clean(decode(doc.body, ctype), 60_000)
        doc.title = doc.title or doc.final_url
        return doc

    html = decode(doc.body, ctype)
    if not html:
        return doc

    try:
        tree = LexborHTMLParser(html)
    except Exception:
        doc.text = _clean(re.sub(r"<[^>]+>", " ", html), 60_000)
        return doc

    title = ""
    tnode = tree.css_first("title")
    if tnode:
        title = _WS.sub(" ", tnode.text(strip=True))
    if not title:
        title = _meta(tree, "og:title", "twitter:title")
    if not title:
        h1 = tree.css_first("h1")
        if h1:
            title = _WS.sub(" ", h1.text(strip=True))
    doc.title = title[:300]

    doc.description = _meta(
        tree, "description", "og:description", "twitter:description"
    )[:600]
    doc.published = (
        _meta(tree, "article:published_time", "datePublished", "og:published_time",
              "date", "pubdate", "article:modified_time")
        or _jsonld_date(tree)
        # The body-text pass costs milliseconds on every page a search
        # fetches, so it is bought only by a query that asked about a period.
        # On the rest no signal in the fusion looks for a date.
        or (_htmldate(html, doc.final_url) if want_date else "")
    )[:32]
    htag = tree.css_first("html")
    if htag:
        doc.lang = (htag.attributes.get("lang") or "")[:8]
    canon = tree.css_first('link[rel="canonical"]')
    if canon:
        href = (canon.attributes.get("href") or "").strip()
        if href:
            doc.canonical = urljoin(doc.final_url, href)[:2048]

    if want_links:
        doc.links = extract_links(tree, doc.final_url)
        # Free here: the tree is already parsed, and the form it holds is the
        # address of the only index this host has.
        doc.search_url = search_template(tree, doc.final_url)
    if want_media:
        doc.favicon = favicon_of(tree, doc.final_url)
        doc.images = images_of(tree, doc.final_url)

    text = _selectolax_text(tree)
    if use_trafilatura and len(html) < 300_000:
        tex = _get_trafilatura()
        if tex:
            try:
                alt = tex(
                    html, output_format="txt", favor_recall=True,
                    include_comments=False, include_tables=True,
                    include_formatting=False, no_fallback=True, url=doc.final_url,
                ) or ""
            except Exception:
                alt = ""
            if len(alt) > 400:
                text = alt

    doc.text = _clean(text)
    if not doc.text and doc.description:
        doc.text = doc.description
    return doc
