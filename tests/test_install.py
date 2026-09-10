"""`liberdex install` against a throwaway home: what it writes, and only that."""
from __future__ import annotations

import json
import os

import pytest

from liberdex import install


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    # No host CLI on PATH: every installer takes its file-editing branch.
    monkeypatch.setattr(install.shutil, "which", lambda name: None)
    return tmp_path


def run(argv):
    return install.main(argv)


@pytest.mark.parametrize("host", install.HOSTS)
def test_every_host_gets_a_server_entry_and_the_skill(fake_home, host, capsys):
    assert run([host, "--command", "uvx liberdex mcp"]) == 0
    out = capsys.readouterr().out
    assert "SKILL.md" in out
    skills = [p for p in out.splitlines() if p.strip().endswith("SKILL.md")]
    assert skills, out
    for line in skills:
        path = line.split()[-1]
        assert os.path.isfile(path)
        assert open(path).read().startswith("---\nname: liberdex\n")


def test_skill_frontmatter_is_valid_yaml():
    # A ": " in the plain description scalar reads as a nested mapping and
    # hosts refuse to load the skill.
    yaml = pytest.importorskip("yaml")
    path = os.path.join(os.path.dirname(__file__), "..", "skills", "liberdex", "SKILL.md")
    head = open(path).read().split("---\n")[1]
    meta = yaml.safe_load(head)
    assert meta["name"] == "liberdex"
    assert isinstance(meta["description"], str)


def test_json_hosts_merge_and_keep_what_was_there(fake_home):
    path = fake_home / ".cursor" / "mcp.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}, "theme": "dark"}))
    run(["cursor", "--command", "uvx liberdex mcp"])
    data = json.loads(path.read_text())
    assert data["theme"] == "dark" and data["mcpServers"]["other"] == {"command": "x"}
    assert data["mcpServers"]["liberdex"] == {"command": "uvx", "args": ["liberdex", "mcp"]}


def test_opencode_uses_its_own_shape(fake_home):
    run(["opencode", "--command", "uvx liberdex mcp"])
    data = json.loads((fake_home / ".config" / "opencode" / "opencode.json").read_text())
    assert data["mcp"]["liberdex"] == {"type": "local", "command": ["uvx", "liberdex", "mcp"],
                                       "enabled": True}


def test_codex_appends_a_toml_block_once(fake_home):
    path = fake_home / ".codex" / "config.toml"
    path.parent.mkdir()
    path.write_text('model = "gpt-5.5"\n')
    run(["codex", "--command", "uvx liberdex mcp"])
    run(["codex", "--command", "uvx liberdex mcp"])
    text = path.read_text()
    assert text.count("[mcp_servers.liberdex]") == 1
    assert text.startswith('model = "gpt-5.5"\n')
    import tomllib
    data = tomllib.loads(text)
    assert data["mcp_servers"]["liberdex"] == {"command": "uvx", "args": ["liberdex", "mcp"]}


def test_a_second_run_changes_nothing(fake_home, capsys):
    run(["gemini", "--command", "uvx liberdex mcp"])
    capsys.readouterr()
    run(["gemini", "--command", "uvx liberdex mcp"])
    out = capsys.readouterr().out
    assert "write" not in out and "update" not in out
    assert "unchanged" in out


def test_dry_run_writes_nothing(fake_home, capsys):
    run(["claude-code", "--dry-run", "--command", "uvx liberdex mcp"])
    out = capsys.readouterr().out
    assert "nothing was written" in out
    assert not (fake_home / ".claude.json").exists()
    assert not (fake_home / ".claude").exists()


def test_model_goes_into_the_config_the_engine_reads(fake_home):
    cfg = fake_home / ".config" / "liberdex" / "models.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('[providers.myllm]\nbase_url = "http://x/v1"\n\n[models.answer]\nmodel = "a/b"\n')
    run(["--model", "openrouter/google/gemini-3.5-flash-lite"])
    import tomllib
    data = tomllib.loads(cfg.read_text())
    assert data["models"]["planner"]["model"] == "openrouter/google/gemini-3.5-flash-lite"
    assert data["models"]["answer"]["model"] == "a/b"
    assert data["providers"]["myllm"]["base_url"] == "http://x/v1"
    run(["--model", "ollama/qwen3:8b"])
    data = tomllib.loads(cfg.read_text())
    assert data["models"]["planner"]["model"] == "ollama/qwen3:8b"
    assert cfg.read_text().count("[models.planner]") == 1


# ------------------------------------------------------------------ hermes
def test_hermes_without_its_cli_copies_the_skill_and_prints_the_snippet(fake_home, capsys):
    run(["hermes", "--command", "/opt/liberdex mcp"])
    out = capsys.readouterr().out
    assert (fake_home / ".hermes" / "skills" / "liberdex" / "SKILL.md").is_file()
    assert "mcp_servers" in out and '"/opt/liberdex"' in out
    assert "mcp_liberdex_search" in out


@pytest.fixture()
def hermes_cli(fake_home, monkeypatch):
    """`hermes` on PATH, every command recorded and pretended to succeed."""
    argvs = []
    monkeypatch.setattr(install.shutil, "which",
                        lambda name: "/usr/bin/" + name if name in ("hermes", "uv") else None)
    monkeypatch.setattr(install.Writer, "run", lambda self, argv: argvs.append(argv) or True)
    return argvs


def test_hermes_provider_goes_into_its_python(fake_home, hermes_cli, monkeypatch):
    monkeypatch.setattr(install, "hermes_python", lambda: "/hermes/venv/bin/python")
    monkeypatch.setattr(install, "python_version", lambda py: (3, 12))
    run(["hermes", "--command", "/opt/liberdex mcp"])
    flat = [" ".join(a) for a in hermes_cli]
    assert any(f.startswith("uv pip install --python /hermes/venv/bin/python ") for f in flat)
    assert "hermes plugins enable liberdex" in flat
    assert "hermes config set web.backend liberdex" in flat
    assert not any("mcp add" in f for f in flat)
    assert (fake_home / ".hermes" / "skills" / "liberdex" / "SKILL.md").is_file()


def test_hermes_on_old_python_gets_the_mcp_server(fake_home, hermes_cli, monkeypatch, capsys):
    monkeypatch.setattr(install, "hermes_python", lambda: "/hermes/venv/bin/python")
    monkeypatch.setattr(install, "python_version", lambda py: (3, 11))
    run(["hermes", "--command", "/opt/liberdex mcp"])
    out = capsys.readouterr().out
    assert "3.11" in out and "falling back" in out
    assert hermes_cli == [["hermes", "mcp", "add", "liberdex", "--command", "/opt/liberdex",
                           "--args", "mcp"]]


def test_hermes_mcp_flag_skips_the_provider(fake_home, hermes_cli, monkeypatch):
    monkeypatch.setattr(install, "hermes_python", lambda: "/hermes/venv/bin/python")
    monkeypatch.setattr(install, "python_version", lambda py: (3, 12))
    run(["hermes", "--mcp", "--command", "/opt/liberdex mcp"])
    assert hermes_cli[0][:3] == ["hermes", "mcp", "add"]
    assert len(hermes_cli) == 1


def test_hermes_python_is_found_in_its_venv(fake_home, monkeypatch):
    py = fake_home / ".hermes" / "hermes-agent" / "venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    monkeypatch.delenv("HERMES_HOME", raising=False)
    assert install.hermes_python() == str(py)


def test_hermes_python_is_read_off_the_launcher(fake_home, monkeypatch):
    venv = fake_home / "venv" / "bin" / "python"
    venv.parent.mkdir(parents=True)
    venv.write_text("")
    launcher = fake_home / "hermes"
    launcher.write_text(f'#!/usr/bin/env bash\nHERMES_BIN="{venv}"\nexec "$HERMES_BIN" "$@"\n')
    monkeypatch.setattr(install.shutil, "which", lambda name: str(launcher) if name == "hermes" else None)
    assert install.hermes_python() == str(venv)


# ---------------------------------------------------------------- openclaw
def test_openclaw_merges_plain_json_and_copies_two_skills(fake_home):
    path = fake_home / ".openclaw" / "openclaw.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"agent": {"model": "x"}, "mcp": {"servers": {"o": {"url": "u"}}}}))
    run(["openclaw", "--command", "/opt/liberdex mcp"])
    data = json.loads(path.read_text())
    assert data["agent"] == {"model": "x"} and data["mcp"]["servers"]["o"] == {"url": "u"}
    assert data["mcp"]["servers"]["liberdex"] == {"command": "/opt/liberdex", "args": ["mcp"],
                                                  "enabled": True}
    assert (fake_home / ".openclaw" / "skills" / "liberdex" / "SKILL.md").is_file()
    assert (fake_home / ".agents" / "skills" / "liberdex" / "SKILL.md").is_file()


def test_openclaw_leaves_a_commented_config_alone(fake_home, capsys):
    path = fake_home / ".openclaw" / "openclaw.json"
    path.parent.mkdir()
    text = '{\n  // my gateway\n  agent: { model: "x" },\n}\n'
    path.write_text(text)
    run(["openclaw", "--command", "/opt/liberdex mcp"])
    out = capsys.readouterr().out
    assert path.read_text() == text
    assert "JSON5" in out and '"command": "/opt/liberdex"' in out


# ------------------------------------------------------------------ t3code
def test_t3code_writes_the_three_hosts_it_runs(fake_home, capsys):
    run(["t3code", "--command", "/opt/liberdex mcp"])
    out = capsys.readouterr().out
    assert json.loads((fake_home / ".claude.json").read_text())["mcpServers"]["liberdex"]
    assert "[mcp_servers.liberdex]" in (fake_home / ".codex" / "config.toml").read_text()
    assert json.loads((fake_home / ".config" / "opencode" / "opencode.json").read_text())["mcp"]["liberdex"]
    assert "absolute" in out and "Dock" in out


@pytest.mark.parametrize("host", ["hermes", "openclaw", "t3code"])
def test_dry_run_writes_nothing_for_the_new_hosts(fake_home, host, capsys):
    run([host, "--dry-run", "--command", "/opt/liberdex mcp"])
    assert "nothing was written" in capsys.readouterr().out
    assert not (fake_home / ".hermes").exists()
    assert not (fake_home / ".openclaw").exists()
    assert not (fake_home / ".claude.json").exists()
