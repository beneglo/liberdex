"""`liberdex install <host>`: wire liberdex into an agent CLI.

    liberdex install claude-code
    liberdex install codex --model openrouter/google/gemini-3.5-flash-lite
    liberdex install opencode --dry-run
    liberdex install hermes            # provider plugin; --mcp for the server only
    liberdex install openclaw
    liberdex install t3code            # claude-code + codex + opencode, which it runs

Two things per host: the MCP server entry in the host's own config, and the
skill, which is the file that tells the host's model when to reach for liberdex
and how to write a plan. hermes gets a third: liberdex as its own web search
backend, in its process, planned by its model. Every write is printed, nothing
is written twice, and nothing else in the host's config is touched.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Callable, Optional

from . import REPO

HOSTS = ("claude-code", "codex", "opencode", "cursor", "gemini",
         "hermes", "openclaw", "t3code")
SKILL_NAME = "liberdex"
# Until there is a release on PyPI, uvx and pip install from the repository.
GIT_SOURCE = f"git+https://github.com/{REPO}"



def home(*parts: str) -> str:
    return os.path.join(os.path.expanduser("~"), *parts)


def config_home() -> str:
    return os.environ.get("XDG_CONFIG_HOME") or home(".config")


def skill_source() -> str:
    """The bundled skill: inside the wheel, or beside the package in a checkout."""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "skills", SKILL_NAME),
                 os.path.join(os.path.dirname(here), "skills", SKILL_NAME)):
        if os.path.isfile(os.path.join(cand, "SKILL.md")):
            return cand
    raise FileNotFoundError("the liberdex skill is missing from this install")


def server_command() -> list[str]:
    """How the host should start the server: this install's own `liberdex`
    when it has one, `uvx` otherwise, the running interpreter as a last resort."""
    exe = shutil.which("liberdex")
    if exe:
        return [exe, "mcp"]
    if shutil.which("uvx"):
        return ["uvx", "--from", GIT_SOURCE, "liberdex", "mcp"]
    return [sys.executable, "-m", "liberdex.cli", "mcp"]


class Writer:
    """Filesystem writes that say what they do and can be asked not to."""

    def __init__(self, dry_run: bool, *, mcp_only: bool = False) -> None:
        self.dry_run = dry_run
        # hermes: skip the in-process provider, wire the MCP server only.
        self.mcp_only = mcp_only
        self.touched: list[str] = []
        # What to tell the user at the end, set by the host that knows.
        self.done: list[str] = []

    def note(self, text: str) -> None:
        print(f"  {text}")

    def say(self, verb: str, path: str) -> None:
        print(f"  {verb:<9} {path}")

    def write_text(self, path: str, text: str) -> None:
        exists = os.path.exists(path)
        if exists:
            with open(path, encoding="utf-8") as fh:
                if fh.read() == text:
                    self.say("unchanged", path)
                    return
        self.say("update" if exists else "write", path)
        self.touched.append(path)
        if self.dry_run:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)

    def merge_json(self, path: str, key: str, name: str, entry: dict) -> None:
        """`data[key][name] = entry`, everything else in the file kept.
        `key` may be dotted (`mcp.servers`) for a nested section."""
        data: dict = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                raw = fh.read().strip()
            if raw:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise SystemExit(f"{path} is not JSON I can safely edit: {e}")
        section = data
        for part in key.split("."):
            section = section.setdefault(part, {})
        if section.get(name) == entry:
            self.say("unchanged", path)
            return
        section[name] = entry
        self.say("update" if os.path.exists(path) else "write", path)
        self.touched.append(path)
        if self.dry_run:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")

    def copy_skill(self, dest: str) -> None:
        src = skill_source()
        for root, _dirs, files in os.walk(src):
            for f in files:
                sp = os.path.join(root, f)
                dp = os.path.join(dest, os.path.relpath(sp, src))
                with open(sp, encoding="utf-8") as fh:
                    self.write_text(dp, fh.read())

    def run(self, argv: list[str]) -> bool:
        print(f"  run       {' '.join(argv)}")
        if self.dry_run:
            return True
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as e:
            print(f"            failed: {e}")
            return False
        if r.returncode != 0:
            print(f"            exit {r.returncode}: {(r.stderr or r.stdout).strip()[:200]}")
            return False
        return True


# --------------------------------------------------------------------- hosts
def claude_code(w: Writer, cmd: list[str]) -> None:
    # The CLI is the supported way to edit its own config; the file is the
    # fallback for a machine where `claude` is not on PATH yet.
    if not (shutil.which("claude") and
            w.run(["claude", "mcp", "add", "--scope", "user", "--transport", "stdio",
                   SKILL_NAME, "--", *cmd])):
        w.merge_json(home(".claude.json"), "mcpServers", SKILL_NAME,
                     {"type": "stdio", "command": cmd[0], "args": cmd[1:]})
    w.copy_skill(home(".claude", "skills", SKILL_NAME))


def codex(w: Writer, cmd: list[str]) -> None:
    if not (shutil.which("codex") and
            w.run(["codex", "mcp", "add", SKILL_NAME, "--", *cmd])):
        path = home(".codex", "config.toml")
        block = (f'\n[mcp_servers.{SKILL_NAME}]\ncommand = {json.dumps(cmd[0])}\n'
                 f'args = {json.dumps(cmd[1:])}\n')
        existing = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
        if f"[mcp_servers.{SKILL_NAME}]" in existing:
            w.say("unchanged", path)
        else:
            w.write_text(path, existing.rstrip("\n") + "\n" + block if existing else block.lstrip("\n"))
    # Codex reads its own skills directory and the cross-tool one.
    w.copy_skill(home(".codex", "skills", SKILL_NAME))
    w.copy_skill(home(".agents", "skills", SKILL_NAME))


def opencode(w: Writer, cmd: list[str]) -> None:
    w.merge_json(os.path.join(config_home(), "opencode", "opencode.json"), "mcp",
                 SKILL_NAME, {"type": "local", "command": cmd, "enabled": True})
    w.copy_skill(os.path.join(config_home(), "opencode", "skills", SKILL_NAME))


def cursor(w: Writer, cmd: list[str]) -> None:
    w.merge_json(home(".cursor", "mcp.json"), "mcpServers", SKILL_NAME,
                 {"command": cmd[0], "args": cmd[1:]})
    w.copy_skill(home(".cursor", "skills", SKILL_NAME))


def gemini(w: Writer, cmd: list[str]) -> None:
    w.merge_json(home(".gemini", "settings.json"), "mcpServers", SKILL_NAME,
                 {"command": cmd[0], "args": cmd[1:]})
    w.copy_skill(home(".gemini", "skills", SKILL_NAME))


# ------------------------------------------------------------------ hermes
def hermes_home() -> str:
    return os.environ.get("HERMES_HOME") or home(".hermes")


def hermes_python() -> Optional[str]:
    """The interpreter hermes runs on: the venv its installer makes, else the
    one the `hermes` launcher execs (`HERMES_BIN=...`), else the shebang of a
    pip-installed `hermes` script."""
    cand = os.path.join(hermes_home(), "hermes-agent", "venv", "bin", "python")
    if os.path.isfile(cand):
        return cand
    exe = shutil.which("hermes")
    if not exe:
        return None
    try:
        with open(exe, encoding="utf-8", errors="replace") as fh:
            head = fh.read(4096)
    except OSError:
        return None
    m = re.search(r'^HERMES_BIN=["\']?([^"\'\n]+)', head, re.M)
    if m and os.path.isfile(os.path.expanduser(m.group(1))):
        return os.path.expanduser(m.group(1))
    first = head.split("\n", 1)[0]
    if first.startswith("#!") and "python" in first:
        py = first[2:].split()[-1]
        if os.path.isfile(py):
            return py
    return None


def python_version(py: str) -> tuple[int, int]:
    try:
        r = subprocess.run([py, "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
                           capture_output=True, text=True, timeout=20)
        a, b = r.stdout.split()
        return int(a), int(b)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return (0, 0)


def liberdex_spec() -> str:
    """What to install into another interpreter: this checkout when the
    package runs from one, else the repository."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isfile(os.path.join(root, "pyproject.toml")):
        return root
    return GIT_SOURCE


def hermes(w: Writer, cmd: list[str]) -> None:
    skill = os.path.join(hermes_home(), "skills", SKILL_NAME)
    have_cli = bool(shutil.which("hermes"))
    if not have_cli:
        w.copy_skill(skill)
        w.note("`hermes` is not on PATH, so its config was left alone. Once it is, "
               "run `liberdex install hermes` again, or add to ~/.hermes/config.yaml:")
        for line in ("mcp_servers:", f"  {SKILL_NAME}:",
                     f"    command: {json.dumps(cmd[0])}", f"    args: {json.dumps(cmd[1:])}"):
            w.note("  " + line)
        w.done.append("tools are `mcp_liberdex_search` and `mcp_liberdex_extract`.")
        return
    if not w.mcp_only and _hermes_provider(w):
        w.copy_skill(skill)
        w.done.append("`web_search` and `web_extract` are liberdex now; hermes's own "
                      "model writes the plan. `hermes tools` shows the backend.")
        return
    # The MCP server: a process of its own, on this interpreter, any Python.
    w.run(["hermes", "mcp", "add", SKILL_NAME, "--command", cmd[0], "--args", *cmd[1:]])
    w.copy_skill(skill)
    w.done.append("tools are `mcp_liberdex_search` and `mcp_liberdex_extract`; "
                  "`/reload-mcp` in a running session.")


def _hermes_provider(w: Writer) -> bool:
    """liberdex into hermes's own interpreter, enabled, chosen as the backend.
    False, with the reason printed, when any step cannot be done."""
    py = hermes_python()
    if not py:
        w.note("could not find hermes's Python; falling back to the MCP server")
        return False
    ver = python_version(py)
    if ver < (3, 12):
        w.note(f"hermes runs Python {ver[0]}.{ver[1]} and liberdex needs 3.12; "
               "falling back to the MCP server, which runs on its own interpreter")
        return False
    spec = liberdex_spec()
    if shutil.which("uv"):
        ok = w.run(["uv", "pip", "install", "--python", py, spec])
    else:
        ok = w.run([py, "-m", "pip", "install", spec])
    if not ok or not w.run([py, "-c", "import liberdex.hosts.hermes"]):
        w.note("liberdex did not go into hermes's Python; falling back to the MCP server")
        return False
    if not (w.run(["hermes", "plugins", "enable", SKILL_NAME]) and
            w.run(["hermes", "config", "set", "web.backend", SKILL_NAME])):
        w.note("hermes did not take the setting; falling back to the MCP server")
        return False
    return True


# ---------------------------------------------------------------- openclaw
def openclaw(w: Writer, cmd: list[str]) -> None:
    entry = {"command": cmd[0], "args": cmd[1:], "enabled": True}
    path = home(".openclaw", "openclaw.json")
    written = (shutil.which("openclaw") and
               w.run(["openclaw", "config", "set", f"mcp.servers.{SKILL_NAME}",
                      json.dumps(entry)]))
    if not written:
        if _strict_json(path):
            w.merge_json(path, "mcp.servers", SKILL_NAME, entry)
        else:
            w.note(f"{path} is JSON5 with comments, which I will not rewrite. Add:")
            w.note(f'  mcp: {{ servers: {{ {SKILL_NAME}: {json.dumps(entry)} }} }}')
    # Its own skills directory and the cross-tool one it also reads.
    w.copy_skill(home(".openclaw", "skills", SKILL_NAME))
    w.copy_skill(home(".agents", "skills", SKILL_NAME))
    w.done.append("tools are `liberdex` `search` and `extract`; "
                  f"`openclaw mcp doctor {SKILL_NAME} --probe` checks the server.")


def _strict_json(path: str) -> bool:
    """Absent, empty, or plain JSON: something `merge_json` can rewrite whole."""
    if not os.path.exists(path):
        return True
    try:
        with open(path, encoding="utf-8") as fh:
            raw = fh.read().strip()
        if raw:
            json.loads(raw)
        return True
    except (OSError, json.JSONDecodeError):
        return False


# ------------------------------------------------------------------ t3code
def t3code(w: Writer, cmd: list[str]) -> None:
    """T3 Code has no tool config of its own: it runs Claude Code, Codex and
    OpenCode and reads theirs."""
    for name, fn in (("claude-code", claude_code), ("codex", codex), ("opencode", opencode)):
        print(f"  ({name})")
        fn(w, cmd)
    w.done.append(f"the command written is absolute ({cmd[0]}), which T3 Code needs.")
    w.done.append("T3 Code started from the Dock has no shell environment: keys belong "
                  "in ~/.config/liberdex/env, and with a plan from the host none is needed.")


INSTALLERS: dict[str, Callable[[Writer, list[str]], None]] = {
    "claude-code": claude_code, "codex": codex, "opencode": opencode,
    "cursor": cursor, "gemini": gemini,
    "hermes": hermes, "openclaw": openclaw, "t3code": t3code,
}


# ---------------------------------------------------------------------- model
def write_model(w: Writer, role: str, spec: str) -> None:
    """`[models.<role>] model = "<spec>"` into the user config, keeping the rest."""
    path = os.path.join(config_home(), "liberdex", "models.toml")
    existing = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
    import tomllib
    try:
        data = tomllib.loads(existing) if existing else {}
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"{path}: {e}")
    if (data.get("models") or {}).get(role, {}).get("model") == spec:
        w.say("unchanged", path)
        return
    header = f"[models.{role}]"
    lines = existing.splitlines()
    out: list[str] = []
    i = 0
    replaced = False
    while i < len(lines):
        line = lines[i]
        if line.strip() == header:
            out.append(line)
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("["):
                if lines[i].split("=", 1)[0].strip() == "model":
                    out.append(f'model = "{spec}"')
                    replaced = True
                else:
                    out.append(lines[i])
                i += 1
            if not replaced:
                out.append(f'model = "{spec}"')
                replaced = True
            continue
        out.append(line)
        i += 1
    if not replaced:
        if out and out[-1].strip():
            out.append("")
        out += [header, f'model = "{spec}"']
    w.write_text(path, "\n".join(out) + "\n")


# ----------------------------------------------------------------------- main
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="liberdex install")
    ap.add_argument("host", nargs="?", choices=HOSTS)
    ap.add_argument("--model", default=None,
                    help="profile/model for searches that arrive without a plan, "
                         "written to ~/.config/liberdex/models.toml")
    ap.add_argument("--command", default=None,
                    help="how the host starts the server (default: this install's "
                         "liberdex, else uvx from the repository)")
    ap.add_argument("--mcp", action="store_true",
                    help="hermes: the MCP server only, not the in-process provider")
    ap.add_argument("--dry-run", action="store_true", help="print, write nothing")
    args = ap.parse_args(argv)
    if not args.host and not args.model:
        ap.print_help()
        return 2

    w = Writer(args.dry_run, mcp_only=args.mcp)
    if args.model:
        print("planner model")
        write_model(w, "planner", args.model)
    if args.host:
        cmd = args.command.split() if args.command else server_command()
        print(f"{args.host}")
        INSTALLERS[args.host](w, cmd)
        print()
        print(f"done. restart {args.host}; "
              + " ".join(w.done or ["the tools are `search` and `extract`."]))
        print("optional: `liberdex warmup` downloads the ranking models now "
              "(~130 MB) instead of on the first search.")
    if args.dry_run:
        print("\n(dry run: nothing was written)")
    return 0
