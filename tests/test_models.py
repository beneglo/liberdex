"""Model profiles, role resolution, and the provider quirks they absorb."""
from __future__ import annotations

import pytest

from liberdex.models import (
    BUILTIN,
    Profile,
    load_config,
    profiles,
    reasoning_payload,
    resolve_role,
    split_model,
)

EMPTY = {"providers": {}, "models": {}}


def test_every_builtin_profile_can_say_where_it_talks():
    for name, p in BUILTIN.items():
        if p.kind == "openai":
            assert p.base_url.startswith("http"), name
        assert p.name == name


def test_local_profiles_need_no_key_but_still_send_one():
    """An OpenAI-compatible server rejects an empty Authorization header even
    when it never checks the value."""
    ollama = BUILTIN["ollama"]
    assert ollama.local
    assert ollama.key() == "local"
    assert ollama.merged_headers()["Authorization"] == "Bearer local"


def test_hosted_profile_with_no_key_configured_reports_no_key(monkeypatch):
    for name in BUILTIN["openrouter"].api_key_env:
        monkeypatch.delenv(name, raising=False)
    assert BUILTIN["openrouter"].key() == ""


def test_api_key_env_is_read_in_preference_order(monkeypatch):
    p = Profile(name="x", api_key_env=("FIRST_K", "SECOND_K"))
    monkeypatch.delenv("FIRST_K", raising=False)
    monkeypatch.setenv("SECOND_K", "second")
    assert p.key() == "second"
    monkeypatch.setenv("FIRST_K", "first")
    assert p.key() == "first"


def test_split_model_keeps_slashes_in_the_model_id():
    known = profiles(EMPTY)
    assert split_model("openrouter/anthropic/claude-x", known) == \
        ("openrouter", "anthropic/claude-x")
    assert split_model("ollama/qwen3:8b", known) == ("ollama", "qwen3:8b")


def test_a_bare_profile_name_means_that_profile_s_default_model():
    known = profiles(EMPTY)
    assert split_model("ollama", known) == ("ollama", known["ollama"].default_model)


def test_a_bare_model_name_falls_back_to_a_reachable_provider(monkeypatch):
    """No prefix means "whatever this machine can actually talk to"."""
    for p in BUILTIN.values():
        for k in p.api_key_env:
            monkeypatch.delenv(k, raising=False)
    known = profiles(EMPTY)
    name, model = split_model("gpt-5-mini", known)
    # Every hosted key is unset, so the local runtime wins.
    assert name == "ollama" and model == "gpt-5-mini"


def test_reasoning_is_spelled_differently_per_provider():
    # OpenRouter is the only one that can switch thinking off outright.
    assert reasoning_payload("openrouter", "none") == {"reasoning": {"enabled": False}}
    assert reasoning_payload("openrouter", "low") == \
        {"reasoning": {"effort": "low", "exclude": True}}
    # OpenAI has no "off"; minimal is the floor.
    assert reasoning_payload("openai", "none") == {"reasoning_effort": "minimal"}
    # Gemini 3 rejects "medium" through the OpenAI shim while low and high work.
    assert reasoning_payload("google", "medium") == {"reasoning_effort": "low"}
    # A plain server would 400 on a field it does not know.
    assert reasoning_payload("none", "high") == {}


def test_openai_wants_max_completion_tokens():
    assert BUILTIN["openai"].max_tokens_param == "max_completion_tokens"
    assert BUILTIN["ollama"].max_tokens_param == "max_tokens"


def test_config_file_extends_and_overrides_builtins(tmp_path):
    cfg = tmp_path / "models.toml"
    cfg.write_text(
        '[providers.ollama]\n'
        'base_url = "http://box.lan:11434/v1"\n'
        '[providers.myllm]\n'
        'base_url = "https://llm.internal/v1/"\n'
        'api_key_env = "MYLLM_KEY"\n'
        'max_tokens_param = "max_completion_tokens"\n'
        'headers = { "X-Tenant" = "search" }\n'
        '[models.planner]\n'
        'model = "myllm/small"\n'
    )
    conf = load_config([str(cfg)])
    known = profiles(conf)
    assert known["ollama"].base_url == "http://box.lan:11434/v1"
    # An override touches one field and leaves the rest of the built-in alone.
    assert known["ollama"].local is True
    assert known["myllm"].base_url == "https://llm.internal/v1"   # trailing / gone
    assert known["myllm"].headers["X-Tenant"] == "search"
    profile, model = resolve_role("planner", config=conf)
    assert (profile.name, model) == ("myllm", "small")


def test_overriding_a_builtin_does_not_mutate_it(tmp_path):
    cfg = tmp_path / "models.toml"
    cfg.write_text('[providers.openrouter]\nbase_url = "http://elsewhere/v1"\n')
    profiles(load_config([str(cfg)]))
    assert BUILTIN["openrouter"].base_url == "https://openrouter.ai/api/v1"


def test_precedence_is_argument_then_env_then_file(tmp_path, monkeypatch):
    cfg = tmp_path / "models.toml"
    cfg.write_text('[models.planner]\nmodel = "ollama/from-file"\n')
    conf = load_config([str(cfg)])

    monkeypatch.delenv("LIBERDEX_PLANNER_MODEL", raising=False)
    assert resolve_role("planner", config=conf)[1] == "from-file"

    monkeypatch.setenv("LIBERDEX_PLANNER_MODEL", "ollama/from-env")
    assert resolve_role("planner", config=conf)[1] == "from-env"

    assert resolve_role("planner", "ollama/from-arg", config=conf)[1] == "from-arg"


def test_the_answer_role_follows_the_planner_when_unset(monkeypatch):
    """One configured key should be enough to run the whole engine."""
    monkeypatch.delenv("LIBERDEX_ANSWER_MODEL", raising=False)
    monkeypatch.setenv("LIBERDEX_PLANNER_MODEL", "ollama/qwen3:8b")
    assert resolve_role("answer", config=EMPTY) == resolve_role("planner", config=EMPTY)


def test_a_broken_config_file_says_which_file(tmp_path):
    bad = tmp_path / "models.toml"
    bad.write_text("[providers.x\n")
    with pytest.raises(ValueError, match="models.toml"):
        load_config([str(bad)])


def test_a_missing_config_file_is_not_an_error(tmp_path):
    assert load_config([str(tmp_path / "nope.toml")]) == EMPTY


# ------------------------------------------------------------- the cli kind
def test_a_cli_profile_substitutes_the_model_or_drops_the_flag():
    from liberdex.models import BUILTIN, cli_argv
    codex = BUILTIN["codex-cli"]
    assert cli_argv(codex, "gpt-5.5")[:4] == ["codex", "exec", "--model", "gpt-5.5"]
    assert "--model" not in cli_argv(codex, "")
    assert cli_argv(codex, "")[-1] == "-"
    gemini = BUILTIN["gemini-cli"]
    argv = cli_argv(gemini, "")
    assert argv[-2:] == ["--prompt", ""] and "--model" not in argv
    assert "plan" in argv


def test_a_config_file_can_add_a_cli_provider():
    from liberdex.models import profiles, resolve_role
    cfg = {"providers": {"mine": {"kind": "cli", "argv": ["my-agent", "-m", "{model}", "-"],
                                  "default_model": "big"}},
           "models": {"planner": {"model": "mine"}}}
    p = profiles(cfg)["mine"]
    assert p.kind == "cli" and p.argv == ("my-agent", "-m", "{model}", "-") and p.local
    profile, model = resolve_role("planner", config=cfg)
    assert profile.name == "mine" and model == "big"


def test_the_answer_follows_a_cli_planner_by_name():
    from liberdex.models import BUILTIN
    from liberdex.planner.cli import CLIPlanner
    from liberdex.planner.factory import answer_default
    assert answer_default(CLIPlanner(BUILTIN["codex-cli"])) == "codex-cli"
    assert answer_default(CLIPlanner(BUILTIN["codex-cli"], "gpt-5.5")) == "codex-cli/gpt-5.5"
