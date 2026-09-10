"""`liberdex serve`: the search page and the HTTP API, one process.

    liberdex serve                      # http://127.0.0.1:8080
    liberdex serve --open               # and open it
    liberdex serve --host 0.0.0.0 --port 80 --planner-model openrouter/...

Flags become the environment the server reads, so a flag and an env var are
the same setting spelled twice; the flag wins because it was typed last.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import urllib.request
import webbrowser

LOOPBACK = ("127.0.0.1", "localhost", "::1")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="liberdex serve")
    ap.add_argument("--host", default=os.environ.get("LIBERDEX_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("LIBERDEX_PORT", "8080")))
    ap.add_argument("--open", action="store_true", help="open the page once it is up")
    ap.add_argument("-p", "--planner", default=None,
                    choices=["auto", "claude", "openai", "replay", "none"])
    ap.add_argument("--planner-model", default=None, help="profile/model")
    ap.add_argument("--answer-model", default=None, help="profile/model")
    ap.add_argument("--keys", default=None,
                    help="comma-separated API keys callers must present "
                         "(LIBERDEX_KEYS); unset means open")
    ap.add_argument("--cors", default=None,
                    help="comma-separated origins allowed to call the API")
    ap.add_argument("--settings", choices=["on", "off"], default=None,
                    help="the page's settings drawer; default on for a "
                         "loopback bind, off otherwise")
    ap.add_argument("--reload", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    env = {
        "LIBERDEX_PLANNER": args.planner,
        "LIBERDEX_PLANNER_MODEL": args.planner_model,
        "LIBERDEX_ANSWER_MODEL": args.answer_model,
        "LIBERDEX_KEYS": args.keys,
        "LIBERDEX_CORS": args.cors,
        "LIBERDEX_SETTINGS": {"on": "1", "off": "0", None: None}[args.settings],
        "LIBERDEX_HOST": args.host,
    }
    for k, v in env.items():
        if v is not None:
            os.environ[k] = v

    url = f"http://{'localhost' if args.host in LOOPBACK or args.host == '0.0.0.0' else args.host}:{args.port}/"
    if args.host not in LOOPBACK and not os.environ.get("LIBERDEX_KEYS"):
        print(f"liberdex serve: bound to {args.host} with no --keys; anyone who can "
              f"reach this port can search through it", file=sys.stderr)
    if args.open:
        threading.Thread(target=_open_when_up, args=(url,), daemon=True).start()
    print(f"liberdex {url}", file=sys.stderr)

    import uvicorn
    uvicorn.run("liberdex.server:app", host=args.host, port=args.port,
                reload=args.reload, log_level="warning", access_log=False)
    return 0


def _open_when_up(url: str, wait: float = 180.0) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait:
        try:
            with urllib.request.urlopen(url + "health", timeout=2) as r:
                if r.status == 200:
                    webbrowser.open(url)
                    return
        except Exception:
            pass
        time.sleep(0.5)
