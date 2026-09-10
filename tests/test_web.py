"""The search page: its routes, its headers, and the favicon proxy's fencing.

Same stub-engine pattern as test_server.py: no network, no models, no LLM.
"""
from __future__ import annotations

import asyncio
import re
from xml.etree import ElementTree

import pytest
from fastapi.testclient import TestClient

import liberdex.server as srv
from liberdex.web import CSP, STATIC, icons
from test_server import StubEngine


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(srv, "_engine", StubEngine())
    return TestClient(srv.app)


@pytest.fixture(autouse=True)
def _empty_icon_cache():
    icons._cache.clear()
    yield
    icons._cache.clear()


def read(name: str) -> str:
    with open(f"{STATIC}/{name}", encoding="utf-8") as fh:
        return fh.read()


# ------------------------------------------------------------------- the shell
def test_the_page_is_served_from_the_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "<title>liberdex</title>" in r.text


def test_the_shell_carries_the_headers_that_enforce_the_privacy_claim(client):
    h = client.get("/").headers
    assert h["content-security-policy"] == CSP
    # Clicking a result must not tell the site which query surfaced it.
    assert h["referrer-policy"] == "no-referrer"
    assert h["x-content-type-options"] == "nosniff"


def test_the_csp_permits_no_third_party_anything():
    for directive in ("default-src 'self'", "connect-src 'self'",
                      "script-src 'self'", "style-src 'self'"):
        assert directive in CSP
    assert "unsafe-inline" not in CSP


@pytest.mark.parametrize("asset", ["index.html", "app.css", "app.js", "theme.js"])
def test_no_asset_reaches_for_another_origin(asset):
    """The CSP would block it, but a blocked request is still a request made."""
    body = read(asset)
    assert not re.search(r"https?://", body), f"{asset} names an absolute URL"
    assert not re.search(r"""(?:src|href)\s*=\s*["']//""", body)


def test_the_shell_offers_itself_as_a_search_engine():
    shell = read("index.html")
    assert 'rel="search"' in shell and "/opensearch.xml" in shell


# -------------------------------------------------------------------- browser
def test_opensearch_describes_this_host_and_not_a_baked_in_one(client):
    r = client.get("/opensearch.xml", headers={"host": "box.local:9999"})
    assert r.status_code == 200
    root = ElementTree.fromstring(r.text)          # parses, so it is valid XML
    url = root.find("{http://a9.com/-/spec/opensearch/1.1/}Url")
    assert url.get("template") == "http://box.local:9999/?q={searchTerms}"


def test_favicon_and_manifest_are_served(client):
    assert client.get("/favicon.svg").headers["content-type"].startswith("image/svg")
    assert client.get("/favicon.ico").status_code == 200
    body = client.get("/manifest.webmanifest").json()
    assert body["start_url"] == "/" and body["theme_color"] == "#050D1C"


def test_static_assets_are_mounted(client):
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/static/fonts/archivo-var-latin.woff2").status_code == 200


# ------------------------------------------------------------- the icon proxy
@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "ftp://example.com/f.ico",
    "data:image/png;base64,AAAA",
    "gopher://example.com/",
    "",
    "https://" + "a" * 4000,
])
def test_only_http_urls_are_even_considered(url):
    assert icons.target(url) == ""


@pytest.mark.parametrize("host", ["127.0.0.1", "10.0.0.1", "192.168.1.1",
                                  "169.254.169.254", "localhost", "0.0.0.0"])
def test_the_proxy_will_not_reach_into_this_network(host):
    assert asyncio.run(icons.public_host(host)) is False


def test_an_unresolvable_host_is_not_public():
    assert asyncio.run(icons.public_host("no-such-host.invalid")) is False


def test_a_blocked_url_still_answers_with_an_image(client):
    r = client.get("/icon", params={"u": "http://127.0.0.1/f.ico", "site": "x.test"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/svg")
    assert r.headers["content-security-policy"] == "default-src 'none'; sandbox"
    assert r.headers["x-content-type-options"] == "nosniff"


def test_a_missing_icon_falls_back_to_the_sites_initial(client):
    assert b">P<" in client.get("/icon", params={"site": "press.example"}).content
    # A leading www. is not the site's name.
    assert b">G<" in client.get("/icon", params={"site": "www.guide.example"}).content
    assert b">?<" in client.get("/icon", params={"site": ""}).content


def test_the_monogram_never_carries_markup():
    assert b"<script" not in icons.monogram("<script>alert(1)</script>")
    assert icons.monogram('"onload=x').count(b"<text") == 1


def test_a_real_icon_is_served_through_us_and_then_cached(client, monkeypatch):
    calls = []

    async def fake(url):
        calls.append(url)
        return (b"\x89PNG fake", "image/png")

    monkeypatch.setattr(icons, "_fetch", fake)
    url = "https://x.test/f.png"
    for _ in range(3):
        r = client.get("/icon", params={"u": url, "site": "x.test"})
        assert r.content == b"\x89PNG fake"
        assert r.headers["content-type"] == "image/png"
    assert calls == [url], "the second search for the same site refetched its icon"


def test_a_page_that_is_not_an_image_becomes_a_monogram(client, monkeypatch):
    async def fake(url):
        return None                      # what _fetch does for HTML, 404s, 2 MB

    monkeypatch.setattr(icons, "_fetch", fake)
    r = client.get("/icon", params={"u": "https://x.test/f.ico", "site": "x.test"})
    assert r.headers["content-type"].startswith("image/svg")
    assert b">X<" in r.content


def test_a_dead_icon_is_not_refetched_on_every_search(client, monkeypatch):
    calls = []

    async def fake(url):
        calls.append(url)
        return None

    monkeypatch.setattr(icons, "_fetch", fake)
    for _ in range(3):
        client.get("/icon", params={"u": "https://x.test/f.ico", "site": "x.test"})
    assert calls == ["https://x.test/f.ico"]


# ---------------------------------------------------------- nothing else moved
def test_the_api_still_answers_json_at_slash_search(client):
    """The page lives at `/` precisely so this endpoint keeps its meaning."""
    r = client.get("/search", params={"q": "tokio"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["query"] == "tokio"


def test_mounting_twice_is_harmless():
    from liberdex import web
    before = len(srv.app.routes)
    web.mount(srv.app)
    assert len(srv.app.routes) == before
