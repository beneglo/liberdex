import pytest


@pytest.fixture(autouse=True)
def _no_commons(monkeypatch):
    """Tests never talk to the commons unless they hand an engine a client of
    their own: a lookup would otherwise try the public instance from every
    search a test runs."""
    monkeypatch.setattr("liberdex.engine._default_commons", lambda: None)
