"""The search page.

One HTML shell, one stylesheet, one script, served by the same uvicorn
process as the API, off the same origin, with no build step and no request
to anyone else.

It adds routes and touches nothing else. `GET /search` stays JSON, because an
API user who pokes the endpoint in a browser should get the API, so the page
lives at `/` and reads its query from `?q=`. That is also what the OpenSearch
descriptor points a browser's address bar at.

Mount it with `mount(app)`; close it with `aclose()`.
"""
from __future__ import annotations

import os
from xml.sax.saxutils import quoteattr

from fastapi import APIRouter, FastAPI, Query, Request, Response
from fastapi.staticfiles import StaticFiles

from . import icons

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")

# The privacy claim, in a form the browser enforces. Every asset this page needs
# is its own; if the page ever reached for a font, a script or a tracking pixel
# somewhere else, this would stop it rather than merely disapprove.
CSP = ("default-src 'self'; img-src 'self' data:; style-src 'self'; "
       "script-src 'self'; connect-src 'self'; form-action 'self'; "
       "base-uri 'none'; frame-ancestors 'none'")

SHELL_HEADERS = {
    "content-security-policy": CSP,
    # Clicking a result must not tell the site which query surfaced it.
    "referrer-policy": "no-referrer",
    "x-content-type-options": "nosniff",
    "cache-control": "no-cache",
}

router = APIRouter(include_in_schema=False)

_shell: tuple[float, str] = (0.0, "")


def _read_shell() -> str:
    """The shell, re-read when it changes, so editing it needs no restart."""
    global _shell
    path = os.path.join(STATIC, "index.html")
    stamp = os.path.getmtime(path)
    if stamp != _shell[0]:
        with open(path, encoding="utf-8") as fh:
            _shell = (stamp, fh.read())
    return _shell[1]


@router.get("/")
async def home() -> Response:
    return Response(_read_shell(), media_type="text/html; charset=utf-8",
                    headers=SHELL_HEADERS)


@router.get("/opensearch.xml")
async def opensearch(request: Request) -> Response:
    """What a browser reads to offer liberdex as its search engine.

    The template is built from the request rather than a setting, so the same
    file is correct on localhost, on a port, and behind a hostname on the LAN.
    """
    base = str(request.base_url)
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<OpenSearchDescription xmlns="http://a9.com/-/spec/opensearch/1.1/">
  <ShortName>liberdex</ShortName>
  <Description>Index-free web search, running on this machine.</Description>
  <InputEncoding>UTF-8</InputEncoding>
  <Image width="16" height="16" type="image/svg+xml">{base}favicon.svg</Image>
  <Url type="text/html" method="get" template={quoteattr(base + "?q={searchTerms}")}/>
  <moz:SearchForm xmlns:moz="http://www.mozilla.org/2006/browser/search/">{base}</moz:SearchForm>
</OpenSearchDescription>
"""
    return Response(body, media_type="application/opensearchdescription+xml",
                    headers={"cache-control": "public, max-age=3600"})


@router.get("/icon")
async def icon(
    u: str = Query(default="", max_length=icons.MAX_URL),
    site: str = Query(default="", max_length=253),
) -> Response:
    """A result's favicon, fetched by liberdex so the browser never asks for it.

    Always answers with an image: a site whose icon is missing, private, huge or
    not actually an image gets a monogram, because a broken image in a SERP row
    is a worse answer than a letter in a box.
    """
    got = await icons.get(u) if u else None
    if got is None:
        return Response(icons.monogram(site), media_type="image/svg+xml",
                        headers=_icon_headers(cacheable=False))
    body, ctype = got
    return Response(body, media_type=ctype, headers=_icon_headers())


def _icon_headers(*, cacheable: bool = True) -> dict[str, str]:
    return {
        # The bytes come from a third party. Nothing about them is allowed to
        # run, and nothing is allowed to reinterpret their type: an SVG icon
        # is a picture here and only a picture.
        "content-security-policy": "default-src 'none'; sandbox",
        "x-content-type-options": "nosniff",
        "content-disposition": "inline",
        "cache-control": ("public, max-age=86400" if cacheable
                          else "public, max-age=300"),
    }


@router.get("/favicon.svg")
@router.get("/favicon.ico")
async def favicon() -> Response:
    path = os.path.join(STATIC, "favicon.svg")
    with open(path, "rb") as fh:
        return Response(fh.read(), media_type="image/svg+xml",
                        headers={"cache-control": "public, max-age=86400"})


@router.get("/manifest.webmanifest")
async def manifest() -> Response:
    return Response(
        '{"name":"liberdex","short_name":"liberdex","start_url":"/",'
        '"display":"standalone","background_color":"#050D1C",'
        '"theme_color":"#050D1C",'
        '"icons":[{"src":"/favicon.svg","sizes":"any","type":"image/svg+xml"}]}',
        media_type="application/manifest+json",
        headers={"cache-control": "public, max-age=3600"})


def mount(app: FastAPI) -> None:
    """Add the page to an existing liberdex app. Idempotent."""
    if getattr(app.state, "web_mounted", False):
        return
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.include_router(router)
    app.state.web_mounted = True


async def aclose() -> None:
    """Release what the page holds open. Safe to call when nothing was opened."""
    await icons.aclose()


__all__ = ["mount", "aclose", "router", "CSP", "STATIC"]
