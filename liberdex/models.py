"""Model profiles: which endpoint, which credentials, which quirks.

liberdex calls an LLM in two places, and they want different models:

    planner   recall URLs from memory, streamed, on the critical path. Wants
              output throughput. A small local model is a legitimate choice.
    answer    read the ranked pages and state the answer. Not streamed, off
              the search deadline. Wants accuracy.

A `Profile` is one endpoint plus everything liberdex has to know to talk to it.
Nearly every provider speaks OpenAI `/chat/completions`, local runtimes
included, so the transport is shared and the per-provider differences live in
three fields: `headers`, `extra_body`, and `reasoning_style`. That is enough to
absorb OpenRouter's `reasoning` object, Gemini's rejected effort levels and a
vLLM sampler flag without a vendor adapter for each.
Resolution order, loosest to tightest: built-in profile, config file
(`~/.config/liberdex/models.toml`, then `./liberdex.toml`), environment,
explicit argument.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from typing import Any, Optional

CONFIG_NAME = "models.toml"
LOCAL_CONFIG = "liberdex.toml"

# OpenAI-compatible servers reject an empty Authorization header even when they
# do not check it, so local endpoints get a placeholder rather than nothing.
LOCAL_KEY = "local"


@dataclass(slots=True)
class Profile:
    """One LLM endpoint, and the shape of request it wants."""

    name: str
    # "openai" is any OpenAI-compatible /chat/completions endpoint, which is
    # nearly all of them. "claude_cli" shells out to the `claude` binary and
    # needs no key at all.
    kind: str = "openai"
    base_url: str = ""
    # kind == "cli" only: the command that runs one prompt to completion.
    # `{model}` is substituted; the prompt goes to stdin and the reply is read
    # from stdout. No shell, no streaming: the whole reply lands at once, so a
    # planner on one of these is honest at `deep` and cut off at `fast`.
    argv: tuple[str, ...] = ()
    # Env var names, in preference order. Names, never values: a config file
    # that is checked in must not be able to leak a key.
    api_key_env: tuple[str, ...] = ()
    api_key: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    # OpenAI's reasoning models reject `max_tokens` and require
    # `max_completion_tokens`; every local server understands only the former.
    max_tokens_param: str = "max_tokens"
    # How this endpoint spells "think less". See reasoning_payload().
    reasoning_style: str = "none"
    default_model: str = ""
    # Local servers are on a loopback interface and do not need a key; saying so
    # lets `models check` report "unreachable" instead of "no credentials".
    local: bool = False

    def key(self) -> str:
        if self.api_key:
            return self.api_key
        for name in self.api_key_env:
            v = os.environ.get(name)
            if v:
                return v
        return LOCAL_KEY if self.local else ""

    def merged_headers(self, extra: Optional[dict[str, str]] = None) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        key = self.key()
        if key:
            h["Authorization"] = f"Bearer {key}"
        h.update(self.headers)
        h.update(extra or {})
        return h


def reasoning_payload(style: str, effort: Optional[str]) -> dict[str, Any]:
    """The request fragment that sets (or clears) a thinking budget.

    The planner recalls URLs and has nothing to reason about, so every
    reasoning token is latency before the first URL is dispatched. This is the
    one provider difference liberdex normalises itself rather than pushing into
    `extra_body`, because it is the actual latency lever.
    """
    if not effort or style == "none":
        return {}
    if style == "openrouter":
        # OpenRouter takes a top-level object, not `reasoning_effort`, and is
        # the only endpoint that can switch thinking off outright. The trace is
        # never parsed, so do not pay to stream it back.
        if effort == "none":
            return {"reasoning": {"enabled": False}}
        return {"reasoning": {"effort": effort, "exclude": True}}
    if style == "google":
        # Gemini 3 Preview rejects "medium" outright through the OpenAI shim
        # while low and high both work, and 2.5 Pro cannot disable thinking.
        mapped = {"medium": "low", "minimal": "low", "none": "low"}.get(effort, effort)
        return {"reasoning_effort": mapped}
    if style == "openai":
        # OpenAI has no "off"; minimal is the floor.
        return {"reasoning_effort": "minimal" if effort == "none" else effort}
    return {}


def _hosted(name: str, base_url: str, *keys: str, **kw: Any) -> Profile:
    return Profile(name=name, base_url=base_url, api_key_env=tuple(keys), **kw)


# Shipped so the common cases need no config file at all.
BUILTIN: dict[str, Profile] = {p.name: p for p in [
    # --- local runtimes; no key, no cost, no rate limit ---
    _hosted("ollama", "http://localhost:11434/v1", local=True,
       default_model="qwen3:8b"),
    _hosted("lmstudio", "http://localhost:1234/v1", local=True),
    _hosted("llamacpp", "http://localhost:8080/v1", local=True),
    _hosted("vllm", "http://localhost:8000/v1", "VLLM_API_KEY", local=True),

    # --- hosted, OpenAI-compatible ---
    _hosted("openrouter", "https://openrouter.ai/api/v1",
       "OPENROUTER_API_KEY", "OPENROUTER_KEY", "LIBERDEX_API_KEY",
       reasoning_style="openrouter",
       default_model="google/gemini-3.5-flash-lite",
       headers={"HTTP-Referer": "https://liberdex.net", "X-Title": "liberdex"}),
    _hosted("openai", "https://api.openai.com/v1", "OPENAI_API_KEY",
       reasoning_style="openai", max_tokens_param="max_completion_tokens",
       default_model="gpt-5-mini"),
    _hosted("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    _hosted("together", "https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    _hosted("mistral", "https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    _hosted("deepseek", "https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    _hosted("xai", "https://api.x.ai/v1", "XAI_API_KEY"),
    _hosted("cohere", "https://api.cohere.ai/compatibility/v1", "COHERE_API_KEY"),
    _hosted("gemini", "https://generativelanguage.googleapis.com/v1beta/openai",
       "GEMINI_API_KEY", "GOOGLE_API_KEY", reasoning_style="google"),
    # Anthropic's OpenAI-compatible endpoint is documented as a testing shim: no
    # extended thinking, no prompt caching, temperature capped at 1.0. For full
    # Anthropic behaviour use the claude-cli profile instead.
    _hosted("anthropic", "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),

    # --- not HTTP at all: a subscription the machine already has ---
    Profile(name="claude-cli", kind="claude_cli", default_model="opus", local=True),
    # The same idea for the other agent CLIs, through one mechanism: a command
    # that reads a prompt and prints a reply. Claude's keeps its own path
    # because it streams; these do not, and say so in `models check`.
    # No default model on these: an empty model drops the `--model` pair
    # (see cli_argv) and the CLI uses whatever its own config says, which is
    # the one model name guaranteed to exist on that machine.
    Profile(name="codex-cli", kind="cli", local=True,
            argv=("codex", "exec", "--model", "{model}", "--skip-git-repo-check",
                  "--color", "never", "-")),
    # `plan` is read-only mode: no tool approvals to hang on in headless use.
    Profile(name="gemini-cli", kind="cli", local=True,
            argv=("gemini", "--model", "{model}", "--approval-mode", "plan",
                  "--output-format", "text", "--prompt", "")),
    Profile(name="opencode-cli", kind="cli", local=True,
            argv=("opencode", "run", "--model", "{model}", "")),
]}

CLI_KINDS = ("claude_cli", "cli")


def cli_argv(profile: Profile, model: str) -> list[str]:
    """The command for one `kind == "cli"` call, model substituted.

    An empty argument means "the prompt goes here": some CLIs take the prompt
    positionally rather than on stdin (gemini's --prompt). It stays a marker
    here and is filled by the caller, which has the prompt. With no model, the
    `{model}` argument goes, and so does the flag before it.
    """
    out: list[str] = []
    for a in profile.argv:
        if "{model}" in a:
            if model:
                out.append(a.replace("{model}", model))
            elif out and out[-1].startswith("-"):
                out.pop()
            continue
        out.append(a)
    return out


# Where a bare model name with no profile prefix lands, in preference order.
_FALLBACK_ORDER = ("openrouter", "openai", "gemini", "groq", "ollama")

DEFAULT_ROLE_MODEL = {
    "planner": "openrouter/google/gemini-3.5-flash-lite",
    # The answer role deliberately has no default of its own: one configured
    # key should make the whole engine work. It follows the planner unless the
    # user says otherwise.
    "answer": "",
}

ROLE_ENV = {
    "planner": ("LIBERDEX_PLANNER_MODEL",),
    "answer": ("LIBERDEX_ANSWER_MODEL",),
}


def config_paths() -> list[str]:
    """User config first, then a project-local file that overrides it."""
    home = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return [os.path.join(home, "liberdex", CONFIG_NAME),
            os.path.join(os.getcwd(), LOCAL_CONFIG)]


def load_config(paths: Optional[list[str]] = None) -> dict[str, Any]:
    """Merge every config file that exists. Later files win, key by key."""
    out: dict[str, Any] = {"providers": {}, "models": {}}
    for path in (paths if paths is not None else config_paths()):
        try:
            with open(path, "rb") as fh:
                data = tomllib.load(fh)
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            continue
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"{path}: {e}") from e
        for section in ("providers", "models"):
            for name, body in (data.get(section) or {}).items():
                out[section].setdefault(name, {}).update(body or {})
    return out


def profiles(config: Optional[dict[str, Any]] = None) -> dict[str, Profile]:
    """Built-ins, with config-file profiles overriding and extending them."""
    cfg = load_config() if config is None else config
    # Copy the mutable fields too: BUILTIN is process-global and must not
    # be reachable through anything this function hands back.
    out = {name: replace(p, headers=dict(p.headers), extra_body=dict(p.extra_body))
           for name, p in BUILTIN.items()}
    for name, body in (cfg.get("providers") or {}).items():
        base = out.get(name) or Profile(name=name)
        keys = body.get("api_key_env") or base.api_key_env
        if isinstance(keys, str):
            keys = (keys,)
        out[name] = replace(
            base,
            name=name,
            kind=body.get("kind", base.kind),
            base_url=str(body.get("base_url", base.base_url)).rstrip("/"),
            api_key_env=tuple(keys),
            headers={**base.headers, **(body.get("headers") or {})},
            extra_body={**base.extra_body, **(body.get("extra_body") or {})},
            max_tokens_param=body.get("max_tokens_param", base.max_tokens_param),
            reasoning_style=body.get("reasoning_style", base.reasoning_style),
            default_model=body.get("default_model", base.default_model),
            # A command on this machine has no key to ask for.
            local=bool(body.get("local", base.local
                                or body.get("kind", base.kind) in CLI_KINDS)),
            argv=tuple(body.get("argv") or base.argv),
        )
    return out


def split_model(spec: str, known: dict[str, Profile]) -> tuple[str, str]:
    """`"openrouter/anthropic/claude-x"` -> `("openrouter", "anthropic/claude-x")`.

    The model id may itself contain slashes, so only the first segment is
    considered, and only when it names a profile. Anything else is a bare model
    name for whichever provider the environment can actually reach.
    """
    head, _, tail = spec.partition("/")
    if tail and head in known:
        return head, tail
    if spec in known:                      # "ollama" -> that profile's default
        return spec, known[spec].default_model
    for name in _FALLBACK_ORDER:
        p = known.get(name)
        if p is not None and (p.key() or p.local):
            return name, spec
    return "openrouter", spec


def configured_spec(role: str, config: Optional[dict[str, Any]] = None) -> str:
    """What the user asked for in this role, from the environment or the config
    file, or "" when they said nothing and the built-in default applies."""
    cfg = load_config() if config is None else config
    for name in ROLE_ENV.get(role, ()):
        v = os.environ.get(name) or ""
        if v:
            return v
    return str((cfg.get("models") or {}).get(role, {}).get("model") or "")


def auto_planner(config: Optional[dict[str, Any]] = None) -> tuple[Optional[Profile], str, str]:
    """Which planner this machine can run right now, and why.

    Returns (profile, model, reason). The user's own setting wins; failing
    that, the first hosted provider with a key in the environment; failing
    that, the `claude` binary if it is installed. With none of those there is
    nothing to plan with, and (None, "", reason) says so rather than sending
    the default provider a request it will refuse.
    """
    import shutil
    cfg = load_config() if config is None else config
    spec = configured_spec("planner", cfg)
    if spec:
        profile, model = resolve_role("planner", spec, config=cfg)
        return profile, model, "configured"
    known = profiles(cfg)
    for name in _FALLBACK_ORDER:
        p = known.get(name)
        if p is not None and not p.local and p.key() and p.default_model:
            return p, p.default_model, f"{name} key in environment"
    if shutil.which("claude"):
        p = known["claude-cli"]
        return p, p.default_model, "claude binary on PATH"
    return None, "", ("no planner: set LIBERDEX_PLANNER_MODEL, export an API key "
                      "(OPENROUTER_API_KEY, OPENAI_API_KEY, ...), or install the "
                      "claude CLI")


def resolve_role(
    role: str,
    override: Optional[str] = None,
    *,
    config: Optional[dict[str, Any]] = None,
) -> tuple[Profile, str]:
    """Which endpoint and model serve `role`, honouring the precedence chain.

    CLI argument, then environment, then config file, then the built-in
    default. The answer role falls back to the planner's model so that a single
    configured key is enough to run everything.
    """
    cfg = load_config() if config is None else config
    known = profiles(cfg)

    spec = override or ""
    if not spec:
        for name in ROLE_ENV.get(role, ()):
            spec = os.environ.get(name) or ""
            if spec:
                break
    if not spec:
        spec = str((cfg.get("models") or {}).get(role, {}).get("model") or "")
    if not spec:
        spec = DEFAULT_ROLE_MODEL.get(role, "")
    if not spec and role != "planner":
        # Follow whatever the planner turned out to be on this machine (a
        # configured spec, a key in the environment, the claude binary) so that
        # one working planner is a working engine. On the claude CLI the answer
        # takes the faster model; the plan is the recall task, not this.
        profile, model, _why = auto_planner(config=cfg)
        if profile is not None:
            if profile.kind == "claude_cli":
                model = "sonnet"
            return profile, model
        return resolve_role("planner", config=cfg)
    if not spec:
        raise ValueError(f"no model configured for role {role!r}")

    name, model = split_model(spec, known)
    profile = known[name]
    body = (cfg.get("models") or {}).get(role, {})
    if body.get("extra_body"):
        profile = replace(profile,
                          extra_body={**profile.extra_body, **body["extra_body"]})
    return profile, (model or profile.default_model)
