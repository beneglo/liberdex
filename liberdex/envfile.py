"""`~/.config/liberdex/env`: the keys the user gave liberdex, and nothing else.

The search page's settings drawer writes provider keys here. `KEY=VALUE`
lines, mode 0600, loaded into the environment at startup for whatever the
shell did not already export. A config file that is checked in
(`models.toml`) never holds a value, only the name of the variable that does.
"""
from __future__ import annotations

import os
from typing import Optional


def path() -> str:
    return os.path.join(os.environ.get("XDG_CONFIG_HOME")
                        or os.path.expanduser("~/.config"), "liberdex", "env")


def read() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        with open(path(), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip().strip('"')
    except OSError:
        pass
    return out


def load() -> None:
    """Into the environment, where the shell has not already spoken."""
    for k, v in read().items():
        if k and k not in os.environ:
            os.environ[k] = v


def write(updates: dict[str, Optional[str]]) -> str:
    """Set (or, with None, remove) keys; every other line is kept as it was.

    The environment of this process is updated too, so a key typed into the
    page is live for the next request without a restart.
    """
    p = path()
    lines: list[str] = []
    try:
        with open(p, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError:
        pass
    kept = [ln for ln in lines
            if not (ln.strip() and "=" in ln
                    and ln.split("=", 1)[0].strip() in updates)]
    for k, v in updates.items():
        if v is None:
            os.environ.pop(k, None)
            continue
        if "\n" in v or "\r" in v or "\n" in k or "=" in k:
            raise ValueError(f"{k}: not a single KEY=VALUE line")
        kept.append(f"{k}={v}")
        os.environ[k] = v
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(kept) + ("\n" if kept else ""))
    os.chmod(p, 0o600)
    return p
