"""Load-area Open-Meteo downloads.

Unit tests mock the HTTP call. Archive and forecast payloads share one parser.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
import requests

import update_zone_weather as zone_weather
from pjm_forecasting.schema import WEATHER_COLUMNS
from pjm_forecasting.utils import EASTERN
from update_zone_weather import (
    DEFAULT_WORKERS,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_FORECAST_URL,
    OUTPUT_COLUMNS,
    archive_date_range,
    cap_workers,
    download_areas,
    eastern_window,
    fetch_area_weather,
    hourly_frame_from_payload,
    main,
    parse_args,
    resolve_areas,
    to_area_weather,
)


class _Response:
    def __init__(self, status_code: int, payload: dict | list | str) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers: dict[str, str] = {}
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    def json(self):
        if isinstance(self._payload, str):
            raise ValueError("not json")
        return self._payload


def _hours(start: str = "2018-01-01T04:00", periods: int = 8) -> dict:
    """UTC hours, one hour before the 6-hour Eastern window."""
    times = pd.date_range(start, periods=periods, freq="h")
    radiation = [-2.0 if index == 1 else 0.0 for index in range(periods)]
    return {
        "hourly": {
            "time": [stamp.strftime("%Y-%m-%dT%H:%M") for stamp in times],
            "temperature_2m": [5.0 + index for index in range(periods)],
            "relative_humidity_2m": [80.0] * periods,
            "wind_speed_10m": [3.5] * periods,
            "shortwave_radiation": radiation,
            "surface_pressure": [1010.0] * periods,
        }
    }


def _patch_get(monkeypatch: pytest.MonkeyPatch, payload: dict | None = None) -> list[dict]:
    calls: list[dict] = []
    body = payload if payload is not None else _hours()

    def fake_get(url, params=None, timeout=None):
        calls.append({"url": url, "params": dict(params or {}), "timeout": timeout})
        return _Response(200, body)

    monkeypatch.setattr(zone_weather.requests, "get", fake_get)
    return calls


def test_resolve_areas_rejects_rto_and_zone_codes():
    assert resolve_areas("all")[0] == "AECO"
    assert "UGI" in resolve_areas("all")
    assert resolve_areas("dom, aeco, AECO") == ["DOM", "AECO"]
    with pytest.raises(ValueError, match="RTO"):
        resolve_areas("AECO,RTO")
    with pytest.raises(ValueError, match="AEP"):
        resolve_areas("AEP")


def test_archive_dates_use_eastern_calendar_days():
    full_day = eastern_window("2018-01-01", "2018-01-01")
    assert full_day[0] == pd.Timestamp("2018-01-01 00:00", tz=EASTERN)
    assert full_day[1] == pd.Timestamp("2018-01-02 00:00", tz=EASTERN)
    assert archive_date_range(*full_day) == ("2018-01-01", "2018-01-02")

    hours = eastern_window("2018-01-01T00:00", "2018-01-01T05:00")
    assert hours[1] - hours[0] == pd.Timedelta(hours=6)
    assert archive_date_range(*hours) == ("2018-01-01", "2018-01-01")
    with pytest.raises(ValueError):
        eastern_window("2018-01-02", "2018-01-01")


def test_either_host_becomes_the_same_weather_columns():
    archive = hourly_frame_from_payload(_hours(), "AECO")
    forecast = hourly_frame_from_payload(_hours(periods=4), "AECO")
    assert list(archive.columns) == ["timestamp", "timestamp_utc", *WEATHER_COLUMNS]
    assert list(forecast.columns) == list(archive.columns)
    assert archive["relative_humidity_2m"].iloc[0] == pytest.approx(80.0)
    assert "relative_humidity" not in archive.columns


def test_to_area_weather_clips_the_window_and_negative_radiation():
    start, end = eastern_window("2018-01-01T00:00", "2018-01-01T05:00")
    raw = hourly_frame_from_payload(_hours(), "AECO")
    weather = to_area_weather(raw, "AECO", start, end)
    assert list(weather.columns) == OUTPUT_COLUMNS
    assert weather["load_area"].tolist() == ["AECO"] * 6
    assert weather["temperature_2m"].tolist() == [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    assert weather["shortwave_radiation"].iloc[0] == pytest.approx(0.0)
    assert weather["shortwave_radiation"].min() >= 0
    assert weather["surface_pressure"].iloc[0] == pytest.approx(1010.0)
    assert str(weather["timestamp_utc"].dt.tz) == "UTC"
    assert weather["timestamp"].dt.hour.tolist() == [0, 1, 2, 3, 4, 5]


def test_workers_default_and_cap():
    assert DEFAULT_WORKERS == 8
    assert cap_workers(8) == 8
    assert cap_workers(32) == 16
    with pytest.raises(ValueError):
        cap_workers(0)


def test_cli_forecast_does_not_require_dates(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="start and end"):
        main([])
    args = parse_args(["--forecast", "--areas", "AECO", "--forecast-hours", "48"])
    assert args.forecast is True
    assert args.forecast_hours == 48

    captured: dict = {}

    def fake_download(areas, start, end, **kwargs):
        captured.update({"areas": areas, "start": start, "end": end, **kwargs})
        return []

    monkeypatch.setattr(zone_weather, "download_areas", fake_download)
    code = main(
        [
            "--forecast",
            "--areas",
            "AECO",
            "--workers",
            "1",
            "--out-dir",
            str(tmp_path),
            "--forecast-hours",
            "24",
        ]
    )
    assert code == 0
    assert captured["areas"] == ["AECO"]
    assert captured["forecast"] is True
    assert captured["forecast_hours"] == 24
    assert captured["out_dir"] == tmp_path


def test_archive_request_matches_the_forecast_variables(monkeypatch):
    # Empty env wins over .env so free-tier URL assertions stay stable.
    monkeypatch.setenv("OPEN_METEO_API_KEY", "")
    calls = _patch_get(monkeypatch)
    weather = fetch_area_weather("AECO", "2018-01-01T00:00", "2018-01-01T05:00")
    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == OPEN_METEO_ARCHIVE_URL
    params = call["params"]
    assert "models" not in params
    assert "apikey" not in params
    assert params["wind_speed_unit"] == "ms"
    assert params["timezone"] == "UTC"
    assert params["temperature_unit"] == "celsius"
    assert params["hourly"] == (
        "temperature_2m,relative_humidity_2m,wind_speed_10m,"
        "shortwave_radiation,surface_pressure"
    )
    assert params["latitude"] == pytest.approx(39.3643)
    assert params["longitude"] == pytest.approx(-74.4229)
    assert len(weather) == 6

    forecast_calls = _patch_get(monkeypatch, _hours(periods=4))
    forecast = fetch_area_weather("aeco", forecast=True, forecast_hours=48)
    assert forecast_calls[0]["url"] == OPEN_METEO_FORECAST_URL
    forecast_params = forecast_calls[0]["params"]
    assert forecast_params["hourly"] == params["hourly"]
    assert forecast_params["timezone"] == params["timezone"]
    assert forecast_params["wind_speed_unit"] == params["wind_speed_unit"]
    assert "models" not in forecast_params
    assert "apikey" not in forecast_params
    assert "start_date" not in forecast_params
    assert forecast_params["forecast_hours"] == 48
    assert len(forecast) == 4


def test_commercial_key_rewrites_archive_and_forecast_hosts(monkeypatch):
    monkeypatch.setenv("OPEN_METEO_API_KEY", "customer-secret")
    calls = _patch_get(monkeypatch)
    fetch_area_weather("AECO", "2018-01-01T00:00", "2018-01-01T05:00")
    assert calls[0]["url"] == "https://customer-archive-api.open-meteo.com/v1/archive"
    assert calls[0]["params"]["apikey"] == "customer-secret"

    forecast_calls = _patch_get(monkeypatch, _hours(periods=4))
    fetch_area_weather("AECO", forecast=True, forecast_hours=48)
    assert forecast_calls[0]["url"] == "https://customer-api.open-meteo.com/v1/forecast"
    assert forecast_calls[0]["params"]["apikey"] == "customer-secret"


def test_api_error_names_the_load_area(monkeypatch):
    def error_get(url, params=None, timeout=None):
        return _Response(400, {"error": True, "reason": "archive does not cover that date"})

    monkeypatch.setattr(zone_weather.requests, "get", error_get)
    with pytest.raises(RuntimeError, match="AECO"):
        fetch_area_weather("AECO", "2018-01-01T00:00", "2018-01-01T05:00")


def test_http_429_retries_then_succeeds(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(zone_weather.time, "sleep", lambda delay: sleeps.append(delay))
    zone_weather._rate_limited_until = 0.0
    state = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        state["n"] += 1
        if state["n"] == 1:
            return _Response(
                429,
                {
                    "error": True,
                    "reason": (
                        "Minutely API request limit exceeded. Please try again in one minute."
                    ),
                },
            )
        return _Response(200, _hours())

    monkeypatch.setattr(zone_weather.requests, "get", fake_get)
    weather = fetch_area_weather("aeco", "2018-01-01T00:00", "2018-01-01T05:00")
    zone_weather._rate_limited_until = 0.0
    assert state["n"] == 2
    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(60.0, abs=0.1)
    assert len(weather) == 6


def test_connection_error_names_the_load_area(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(zone_weather.requests, "get", fake_get)
    with pytest.raises(RuntimeError, match="AECO"):
        fetch_area_weather("AECO", "2018-01-01T00:00", "2018-01-01T05:00")


def test_download_skips_existing_archive_files(monkeypatch, tmp_path):
    start, end = "2018-01-01", "2018-01-01"
    cached = zone_weather.area_output_path(tmp_path, "AECO", start, end)
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"keep")

    def fail_fetch(*args, **kwargs):
        raise AssertionError("cached load area should not be fetched")

    monkeypatch.setattr(zone_weather, "fetch_area_weather", fail_fetch)
    results = download_areas(["AECO"], start, end, out_dir=tmp_path, workers=1)
    assert results[0].status == "skipped"
    assert cached.read_bytes() == b"keep"
