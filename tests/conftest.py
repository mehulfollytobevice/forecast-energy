"""Shared pytest configuration."""

from __future__ import annotations

import pytest

_PREFECT_FLOW_MODULES = frozenset({"test_flows", "test_aggregate_training"})


@pytest.fixture(autouse=True)
def writable_prefect_home(
    request: pytest.FixtureRequest,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Give Prefect's in-process API server a writable SQLite home."""
    module = request.module
    if module is None:
        return
    if module.__name__.rsplit(".", 1)[-1] not in _PREFECT_FLOW_MODULES:
        return
    home = tmp_path / "prefect_home"
    home.mkdir()
    monkeypatch.setenv("PREFECT_HOME", str(home))
