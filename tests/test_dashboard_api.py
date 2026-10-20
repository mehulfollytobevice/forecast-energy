"""Dashboard API against the fixture snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import BACKTEST_FIXTURE, FIXTURE, app

pytestmark = pytest.mark.usefixtures("monkeypatch")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("FORECAST_SNAPSHOT", str(FIXTURE))
    monkeypatch.setenv("BACKTEST_SNAPSHOT", str(BACKTEST_FIXTURE))
    return TestClient(app)


def test_forecast_matches_fixture(client: TestClient) -> None:
    expected = json.loads(FIXTURE.read_text())
    response = client.get("/api/forecast")
    assert response.status_code == 200
    assert response.json() == expected


def test_backtest_matches_fixture(client: TestClient) -> None:
    expected = json.loads(BACKTEST_FIXTURE.read_text())
    response = client.get("/api/backtest")
    assert response.status_code == 200
    payload = response.json()
    assert payload == expected
    assert payload["lead_hours"] == 1
    assert len(payload["nodes"]) == 54
    sample = next(node for node in payload["nodes"] if node["level"] == "pjm")
    assert "metrics" in sample
    assert len(sample["timestamps"]) == len(sample["actuals"]) == len(sample["p50"])


def test_backtest_falls_back_to_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("BACKTEST_SNAPSHOT", raising=False)
    missing = tmp_path / "missing_backtest.json"

    def _no_parquets(**_kwargs: object) -> dict:
        raise FileNotFoundError("no parquets")

    monkeypatch.setattr(
        "pjm_forecasting.backtest_snapshot.default_backtest_snapshot_path",
        lambda: missing,
    )
    monkeypatch.setattr(
        "pjm_forecasting.backtest_snapshot.load_area_frames_from_disk",
        _no_parquets,
    )
    client = TestClient(app)
    response = client.get("/api/backtest")
    assert response.status_code == 200
    assert response.json()["lead_hours"] == 1
    assert response.json() == json.loads(BACKTEST_FIXTURE.read_text())


def test_backtest_merges_real_areas_over_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("BACKTEST_SNAPSHOT", raising=False)
    import pandas as pd

    stamps = ["2026-01-03T00:00:00-05:00", "2026-01-03T01:00:00-05:00"]
    frames = {
        "AECO": pd.DataFrame(
            {
                "timestamp": pd.to_datetime(stamps),
                "actual": [100.0, 110.0],
                "p10": [90.0, 100.0],
                "p50": [101.0, 111.0],
                "p90": [120.0, 130.0],
            }
        )
    }
    monkeypatch.setattr(
        "pjm_forecasting.backtest_snapshot.default_backtest_snapshot_path",
        lambda: tmp_path / "missing.json",
    )
    monkeypatch.setattr(
        "pjm_forecasting.backtest_snapshot.load_area_frames_from_disk",
        lambda **_kwargs: frames,
    )
    client = TestClient(app)
    response = client.get("/api/backtest")
    assert response.status_code == 200
    payload = response.json()
    by_key = {(n["id"], n["level"]): n for n in payload["nodes"]}
    assert by_key[("AECO", "load_area")]["p50"] == [101.0, 111.0]
    # Unevaluated areas still come from the fixture.
    fixture = json.loads(BACKTEST_FIXTURE.read_text())
    fixture_bc = next(n for n in fixture["nodes"] if n["id"] == "BC" and n["level"] == "load_area")
    assert by_key[("BC", "load_area")]["p50"] == fixture_bc["p50"]
    assert len(payload["nodes"]) == len(fixture["nodes"])


def test_health(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ok"}


def test_geo_served(client: TestClient) -> None:
    geo = Path(__file__).resolve().parents[1] / "app" / "geo" / "pjm_regions.geojson"
    response = client.get("/geo/pjm_regions.geojson")
    assert response.status_code == 200
    assert response.json()["type"] == "FeatureCollection"
    assert len(response.json()["features"]) == len(json.loads(geo.read_text())["features"])
