"""Historical forecast-run parsing, resume, and version selection. HTTP is mocked."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import pjm_forecasting.ingest.openmeteo as openmeteo
from pjm_forecasting.backtesting import select_forecast_versions
from pjm_forecasting.ingest.openmeteo import (
    build_versioned_load,
    build_versioned_predictors,
    download_historical_runs,
    frame_from_single_run,
)
from pjm_forecasting.schema import WEATHER_COLUMNS


class _Response:
    def __init__(self, status_code: int, payload: dict | str) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    def json(self):
        if isinstance(self._payload, str):
            raise ValueError("not json")
        return self._payload


def _payload(start: str = "2024-06-01T00:00", periods: int = 6) -> dict:
    times = pd.date_range(start, periods=periods, freq="h")
    hourly = {"time": [stamp.strftime("%Y-%m-%dT%H:%M") for stamp in times]}
    for name in WEATHER_COLUMNS:
        hourly[name] = [1.0 + index for index in range(periods)]
    hourly["shortwave_radiation"] = [-1.0, *([0.0] * (periods - 1))]
    return {"hourly": hourly}


def test_parser_keeps_run_time_and_publication_delay():
    run = pd.Timestamp("2024-06-01T00:00", tz="UTC")
    frame = frame_from_single_run(_payload(), "AECO", run, pd.Timedelta(hours=6))
    assert frame["run_initialized_at"].nunique() == 1
    assert frame["available_at"].iloc[0] == run + pd.Timedelta(hours=6)
    assert frame["shortwave_radiation"].min() == 0
    assert str(frame["timestamp"].dt.tz) == "US/Eastern"
    assert frame["load_area"].iloc[0] == "AECO"


def test_apply_openmeteo_auth_free_tier_unchanged():
    url, params = openmeteo.apply_openmeteo_auth(
        "https://single-runs-api.open-meteo.com/v1/forecast",
        {"run": "2024-06-01T00:00"},
        api_key="",
    )
    assert url == "https://single-runs-api.open-meteo.com/v1/forecast"
    assert "apikey" not in params


def test_apply_openmeteo_auth_adds_key_and_rewrites_customer_hosts():
    single_url, single_params = openmeteo.apply_openmeteo_auth(
        "https://single-runs-api.open-meteo.com/v1/forecast",
        {"run": "2024-06-01T00:00"},
        api_key="secret-key",
    )
    assert single_url == "https://customer-single-runs-api.open-meteo.com/v1/forecast"
    assert single_params["apikey"] == "secret-key"
    assert single_params["run"] == "2024-06-01T00:00"

    archive_url, archive_params = openmeteo.apply_openmeteo_auth(
        "https://archive-api.open-meteo.com/v1/archive",
        {"latitude": 1.0},
        api_key="secret-key",
    )
    assert archive_url == "https://customer-archive-api.open-meteo.com/v1/archive"
    assert archive_params["apikey"] == "secret-key"

    forecast_url, forecast_params = openmeteo.apply_openmeteo_auth(
        "https://api.open-meteo.com/v1/forecast",
        {"latitude": 1.0},
        api_key="secret-key",
    )
    assert forecast_url == "https://customer-api.open-meteo.com/v1/forecast"
    assert forecast_params["apikey"] == "secret-key"


def test_request_single_run_sends_apikey(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []

    def fake_get(url, params=None, timeout=None):
        calls.append({"url": url, "params": dict(params or {})})
        return _Response(200, _payload())

    monkeypatch.setattr(openmeteo.requests, "get", fake_get)
    monkeypatch.setenv("OPEN_METEO_API_KEY", "from-env")
    openmeteo.request_single_run(
        "AECO",
        "https://single-runs-api.open-meteo.com/v1/forecast",
        {"run": "2024-06-01T00:00"},
    )
    assert calls[0]["url"] == "https://customer-single-runs-api.open-meteo.com/v1/forecast"
    assert calls[0]["params"]["apikey"] == "from-env"


def test_download_skips_a_cached_run(tmp_path, monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []

    def fake_get(url, params=None, timeout=None):
        calls.append({"url": url, "params": dict(params or {})})
        return _Response(200, _payload())

    monkeypatch.setattr(openmeteo.requests, "get", fake_get)
    start = pd.Timestamp("2024-06-01T00:00", tz="UTC")
    end = start + pd.Timedelta(hours=6)
    now = start + pd.Timedelta(hours=8)
    common = dict(
        out_dir=tmp_path,
        model="ecmwf_ifs",
        publication_delay=pd.Timedelta(hours=6),
        cadence=pd.Timedelta(hours=6),
        workers=1,
        url="https://single-runs.example/v1/forecast",
        now=now,
    )
    first = download_historical_runs(["AECO"], start, end, **common)
    assert len(calls) == 1
    assert calls[0]["params"]["run"] == "2024-06-01T00:00"
    assert calls[0]["params"]["models"] == "ecmwf_ifs"
    assert first[0].status == "written"

    second = download_historical_runs(["AECO"], start, end, **common)
    assert len(calls) == 1
    assert second[0].status == "skipped"

    built = build_versioned_predictors(
        tmp_path, "AECO", tmp_path / "AECO.parquet", pd.Timedelta(hours=6)
    )
    table = pd.read_parquet(built)
    assert list(table["available_at"].unique()) == [start + pd.Timedelta(hours=6)]


def test_hourly_rate_limit_waits_then_retries(monkeypatch: pytest.MonkeyPatch):
    sleeps: list[float] = []
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Response(
                429,
                {
                    "error": True,
                    "reason": "Hourly API request limit exceeded. Please try again in the next hour.",
                },
            )
        return _Response(200, _payload())

    monkeypatch.setattr(openmeteo.requests, "get", fake_get)
    monkeypatch.setattr(openmeteo.time, "sleep", sleeps.append)
    monkeypatch.setattr(openmeteo, "_seconds_until_next_hour", lambda now=None: 12.0)
    openmeteo._HOURLY_LIMIT_UNTIL = 0.0
    payload = openmeteo.request_single_run(
        "AECO",
        "https://single-runs.example/v1/forecast",
        {"run": "2024-03-14T06:00"},
    )
    assert "hourly" in payload
    assert calls["n"] == 2
    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(12.0, abs=0.25)


def test_build_versioned_predictors_drops_nonfinite_weather(tmp_path):
    run = pd.Timestamp("2024-06-01T00:00", tz="UTC")
    hours = pd.date_range("2024-06-01", periods=4, freq="h", tz="US/Eastern")
    frame = pd.DataFrame(
        {
            "timestamp": hours,
            "timestamp_utc": hours.tz_convert("UTC"),
            "load_area": "AECO",
            "run_initialized_at": [run] * 4,
            "available_at": [run + pd.Timedelta(hours=6)] * 4,
            **{name: [1.0, 2.0, float("nan"), 4.0] for name in WEATHER_COLUMNS},
        }
    )
    raw = tmp_path / "AECO"
    raw.mkdir()
    frame.to_parquet(raw / "20240601T0000Z.parquet", index=False)
    built = build_versioned_predictors(
        tmp_path, "AECO", tmp_path / "AECO.parquet", pd.Timedelta(hours=6)
    )
    table = pd.read_parquet(built)
    assert len(table) == 3
    assert table["temperature_2m"].notna().all()


def test_unavailable_run_is_skipped(tmp_path, monkeypatch: pytest.MonkeyPatch):
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        return _Response(
            400,
            {
                "error": True,
                "reason": "The requested model run is not available. Model: ecmwf_ifs, run: 2024-03-14T18:00Z",
            },
        )

    monkeypatch.setattr(openmeteo.requests, "get", fake_get)
    start = pd.Timestamp("2024-03-14T18:00", tz="UTC")
    end = start + pd.Timedelta(hours=6)
    now = start + pd.Timedelta(hours=8)
    common = dict(
        out_dir=tmp_path,
        model="ecmwf_ifs",
        publication_delay=pd.Timedelta(hours=6),
        cadence=pd.Timedelta(hours=6),
        workers=1,
        url="https://single-runs.example/v1/forecast",
        now=now,
    )
    results = download_historical_runs(["AECO"], start, end, **common)
    assert len(results) == 1
    assert results[0].status == "skipped"
    assert not results[0].path.is_file()
    marker = openmeteo.run_unavailable_marker_path(results[0].path)
    assert marker.is_file()
    assert calls["n"] == 1

    second = download_historical_runs(["AECO"], start, end, **common)
    assert second[0].status == "skipped"
    assert calls["n"] == 1


def test_download_cancels_remaining_work_on_error(tmp_path, monkeypatch: pytest.MonkeyPatch):
    import threading

    started = threading.Event()
    release = threading.Event()
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        run = (params or {}).get("run", "")
        if run == "2024-06-01T00:00":
            return _Response(500, {"error": True, "reason": "boom"})
        started.set()
        assert release.wait(timeout=2), "worker was not cancelled promptly"
        return _Response(200, _payload())

    monkeypatch.setattr(openmeteo.requests, "get", fake_get)
    start = pd.Timestamp("2024-06-01T00:00", tz="UTC")
    end = start + pd.Timedelta(hours=12)
    now = start + pd.Timedelta(hours=14)
    with pytest.raises(RuntimeError, match="boom"):
        download_historical_runs(
            ["AECO"],
            start,
            end,
            out_dir=tmp_path,
            model="ecmwf_ifs",
            publication_delay=pd.Timedelta(hours=6),
            cadence=pd.Timedelta(hours=6),
            workers=2,
            url="https://single-runs.example/v1/forecast",
            now=now,
        )
    release.set()
    assert calls["n"] <= 2


def test_version_selection_hides_a_run_that_is_not_published_yet():
    early = pd.Timestamp("2024-06-01T00:00", tz="UTC")
    late = pd.Timestamp("2024-06-01T06:00", tz="UTC")
    valid = pd.Timestamp("2024-06-01T12:00", tz="US/Eastern")
    rows = []
    for run, temperature in ((early, 1.0), (late, 9.0)):
        rows.append(
            {
                "timestamp": valid,
                "available_at": run + pd.Timedelta(hours=6),
                "run_initialized_at": run,
                "load_area": "AECO",
                "temperature_2m": temperature,
            }
        )
    frame = pd.DataFrame(rows)
    selected = select_forecast_versions(frame, early + pd.Timedelta(hours=6))
    assert len(selected) == 1
    assert selected["temperature_2m"].iloc[0] == pytest.approx(1.0)
    assert select_forecast_versions(frame, early).empty


def test_versioned_load_is_visible_one_hour_later():
    hour = pd.Timestamp("2024-06-01T00:00", tz="US/Eastern")
    frame = pd.DataFrame(
        {
            "timestamp": [hour, hour],
            "load": [10.0, 99.0],
            "load_area": ["AECO", "BC"],
        }
    )
    versioned = build_versioned_load(frame, "AECO", pd.Timedelta(hours=1))
    assert len(versioned) == 1
    assert versioned["load_area"].iloc[0] == "AECO"
    assert versioned["load"].iloc[0] == pytest.approx(10.0)
    assert versioned["available_at"].iloc[0] == hour + pd.Timedelta(hours=1)
