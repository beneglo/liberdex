"""What the search page's settings drawer reads and writes.

Model choices go to `~/.config/liberdex/models.toml`, the file the engine
already reads; key values go to `~/.config/liberdex/env`, never into TOML.
The drawer is served only on a loopback bind unless `LIBERDEX_SETTINGS=1`
says otherwise: it can write API keys to the machine's disk, which is a
thing a page on the open internet must not be able to do.
"""
from __future__ import annotations

import os
import shutil
from typing import Any, Optional

import httpx

from . import __version__, envfile
from .models import auto_planner, configured_spec, profiles, resolve_role

LOOPBACK = ("127.0.0.1", "localhost", "::1", "")
ROLES = ("planner", "answer")


def enabled() -> bool:
    flag = os.environ.get("LIBERDEX_SETTINGS")
    if flag is not None:
        return flag not in ("0", "false", "no", "")
    return os.environ.get("LIBERDEX_HOST", "") in LOOPBACK


def snapshot(planner_error: str = "") -> dict[str, Any]:
    known = profiles()
    envkeys = envfile.read()
    rows = []
    for name, p in sorted(known.items()):
        row: dict[str, Any] = {
            "name": name, "kind": p.kind, "base_url": p.base_url, "local": p.local,
            "default_model": p.default_model,
            "api_key_env": list(p.api_key_env),
            # Presence only. The value never leaves the machine's environment.
            "has_key": bool(p.key()) and not (p.local and not any(
                os.environ.get(k) for k in p.api_key_env)),
            "key_from_file": any(k in envkeys for k in p.api_key_env),
        }
        if p.kind in ("claude_cli", "cli"):
            head = "claude" if p.kind == "claude_cli" else (p.argv[0] if p.argv else "")
            row["binary"] = head
            row["on_path"] = bool(head) and shutil.which(head) is not None
            row["has_key"] = row["on_path"]
        rows.append(row)

    roles: dict[str, Any] = {}
    profile, model, why = auto_planner()
    roles["planner"] = {
        "configured": configured_spec("planner"),
        "resolved": f"{profile.name}/{model}" if profile else "",
        "why": why,
    }
    try:
        ap, am = resolve_role("answer")
        answer_resolved = f"{ap.name}/{am}"
    except Exception as e:
        answer_resolved = ""
        why = str(e)
    roles["answer"] = {"configured": configured_spec("answer"),
                       "resolved": answer_resolved}

    from .commons import DEFAULT_URL as COMMONS_URL
    return {
        "version": __version__,
        "profiles": rows,
        "roles": roles,
        "commons": {
            "url": os.environ.get("LIBERDEX_COMMONS_URL") or COMMONS_URL,
            # Presence only; the key itself never leaves the environment.
            "registered": bool(os.environ.get("LIBERDEX_COMMONS_KEY")),
        },
        "planner_error": planner_error,
        "config": {"models": os.path.join(
            os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"),
            "liberdex", "models.toml"), "env": envfile.path()},
    }


def update(body: dict[str, Any]) -> dict[str, str]:
    """Apply a drawer submission. Returns what was written, by file."""
    from .install import Writer, write_model
    written: dict[str, str] = {}
    w = Writer(dry_run=False)
    for role in ROLES:
        spec = body.get(f"{role}_model")
        if spec is None:
            continue
        spec = str(spec).strip()
        write_model(w, role, spec)
        # The environment outranks the file, so a flag given at `serve` time
        # would otherwise keep winning over what was just typed.
        env_name = {"planner": "LIBERDEX_PLANNER_MODEL", "answer": "LIBERDEX_ANSWER_MODEL"}[role]
        if spec:
            os.environ[env_name] = spec
        else:
            os.environ.pop(env_name, None)
        written[role] = spec
    keys = body.get("keys") or {}
    if keys:
        allowed = {k for p in profiles().values() for k in p.api_key_env}
        updates: dict[str, Optional[str]] = {}
        for name, value in keys.items():
            if name not in allowed:
                raise ValueError(f"{name} is not a key any profile reads")
            value = str(value or "").strip()
            # One key per line is the whole file format; a value carrying a
            # line break would write a second, unasked-for line.
            if any(ch in value for ch in "\r\n\x00"):
                raise ValueError(f"{name} must be a single line")
            updates[name] = value or None
        written["env"] = envfile.write(updates)
    return written


async def list_models(profile_name: str) -> list[str]:
    """What the profile's endpoint says it serves, for the drawer's picker."""
    p = profiles().get(profile_name)
    if p is None or p.kind != "openai" or not p.base_url:
        return []
    key = p.key()
    if not key and not p.local:
        return []
    try:
        async with httpx.AsyncClient(timeout=8.0) as c:
            r = await c.get(f"{p.base_url.rstrip('/')}/models", headers=p.merged_headers())
    except httpx.HTTPError:
        return []
    if r.status_code != 200:
        return []
    try:
        data = r.json().get("data") or []
    except Exception:
        return []
    ids = [str(m.get("id") or "") for m in data if isinstance(m, dict)]
    return sorted(i for i in ids if i)[:2000]
