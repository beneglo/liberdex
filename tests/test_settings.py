"""The settings drawer's routes and the files behind them, in a throwaway home."""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

import liberdex.server as srv
from liberdex import envfile, settings
from test_server import StubEngine


class Engine(StubEngine):
    planner = None


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    for k in ("LIBERDEX_PLANNER_MODEL", "LIBERDEX_ANSWER_MODEL",
              "LIBERDEX_SETTINGS", "LIBERDEX_HOST", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    return tmp_path


@pytest.fixture()
def client(home, monkeypatch):
    stub = Engine()
    monkeypatch.setattr(srv, "_engine", stub)
    monkeypatch.setattr(srv, "_build_planner", lambda: None)
    c = TestClient(srv.app)
    c.stub = stub
    return c


def test_the_drawer_is_served_on_loopback_only(client, monkeypatch):
    assert client.get("/settings").status_code == 200
    monkeypatch.setenv("LIBERDEX_HOST", "0.0.0.0")
    assert client.get("/settings").status_code == 404
    assert client.put("/settings", json={}).status_code == 404
    assert client.get("/settings/models", params={"profile": "openrouter"}).status_code == 404
    monkeypatch.setenv("LIBERDEX_SETTINGS", "1")
    assert client.get("/settings").status_code == 200


def test_the_snapshot_says_what_it_knows_and_never_a_key_value(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret")
    body = client.get("/settings").json()
    names = {p["name"] for p in body["profiles"]}
    assert {"openrouter", "ollama", "claude-cli", "codex-cli"} <= names
    orow = next(p for p in body["profiles"] if p["name"] == "openrouter")
    assert orow["has_key"] is True
    assert "sk-or-secret" not in str(body)
    assert body["roles"]["planner"]["resolved"].startswith("openrouter/")


def test_saving_writes_models_toml_and_the_env_file_and_rebuilds(client, home):
    r = client.put("/settings", json={
        "planner_model": "openrouter/google/gemini-3.5-flash-lite",
        "answer_model": "claude-cli/sonnet",
        "keys": {"OPENROUTER_API_KEY": "sk-or-typed"}})
    assert r.status_code == 200, r.text
    body = r.json()
    toml = (home / ".config" / "liberdex" / "models.toml").read_text()
    assert 'model = "openrouter/google/gemini-3.5-flash-lite"' in toml
    assert 'model = "claude-cli/sonnet"' in toml
    assert "sk-or-typed" not in toml
    env = (home / ".config" / "liberdex" / "env").read_text()
    assert "OPENROUTER_API_KEY=sk-or-typed" in env
    assert oct(os.stat(home / ".config" / "liberdex" / "env").st_mode)[-3:] == "600"
    assert body["roles"]["planner"]["resolved"] == "openrouter/google/gemini-3.5-flash-lite"
    assert body["roles"]["answer"]["resolved"] == "claude-cli/sonnet"
    assert "sk-or-typed" not in str(body)


def test_a_key_nobody_reads_is_refused(client):
    r = client.put("/settings", json={"keys": {"EVIL_VAR": "x"}})
    assert r.status_code == 422


def test_the_env_file_keeps_lines_it_was_not_asked_about(home):
    p = envfile.write({"A": "1", "B": "2"})
    envfile.write({"B": None, "C": "3"})
    assert open(p).read() == "A=1\nC=3\n"
    assert os.environ["C"] == "3" and "B" not in os.environ


def test_enabled_reads_the_bind_and_the_flag(monkeypatch):
    monkeypatch.delenv("LIBERDEX_SETTINGS", raising=False)
    monkeypatch.setenv("LIBERDEX_HOST", "127.0.0.1")
    assert settings.enabled() is True
    monkeypatch.setenv("LIBERDEX_HOST", "0.0.0.0")
    assert settings.enabled() is False
    monkeypatch.setenv("LIBERDEX_SETTINGS", "0")
    monkeypatch.setenv("LIBERDEX_HOST", "127.0.0.1")
    assert settings.enabled() is False


def test_a_key_value_with_a_line_break_is_refused(client, home):
    r = client.put("/settings", json={"keys": {"OPENROUTER_API_KEY": "sk\nEVIL=1"}})
    assert r.status_code == 422
    assert not (home / ".config" / "liberdex" / "env").exists()
    with pytest.raises(ValueError):
        envfile.write({"OPENROUTER_API_KEY": "a\nb"})
