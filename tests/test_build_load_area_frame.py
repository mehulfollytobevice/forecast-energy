"""Cache reads for the joined training table. No network."""

from __future__ import annotations

import pandas as pd
import pytest

import build_load_area_frame as builder
from pjm_forecasting.schema import WEATHER_COLUMNS


def _stamp() -> pd.Timestamp:
    return pd.Timestamp("2024-01-01", tz="US/Eastern")


def _load_row(load: float) -> dict:
    stamp = _stamp()
    return {
        "timestamp": stamp,
        "timestamp_utc": stamp.tz_convert("UTC"),
        "load_area": "AECO",
        "zone": "AE",
        "mkt_region": "MIDATL",
        "load": load,
    }


def _weather_row(temperature: float) -> dict:
    stamp = _stamp()
    row = {
        "timestamp": stamp,
        "timestamp_utc": stamp.tz_convert("UTC"),
        "load_area": "AECO",
        "temperature_2m": temperature,
        "relative_humidity_2m": 80.0,
        "wind_speed_10m": 3.0,
        "shortwave_radiation": 0.0,
        "surface_pressure": 1010.0,
    }
    return row


def test_cache_reads_keep_the_last_duplicate_and_the_latest_forecast(tmp_path):
    load_root = tmp_path / "load"
    weather_root = tmp_path / "weather"
    early = load_root / "load_areas" / "AECO" / "2024-01.parquet"
    late = load_root / "load_areas" / "AECO" / "2024-02.parquet"
    archive = weather_root / "AECO" / "2018_2024.parquet"
    older_forecast = weather_root / "AECO" / "forecast_20240101.parquet"
    latest_forecast = weather_root / "AECO" / "forecast_20240201.parquet"
    for path in (early, late, archive, older_forecast, latest_forecast):
        path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([_load_row(10.0)]).to_parquet(early, index=False)
    pd.DataFrame([_load_row(12.0)]).to_parquet(late, index=False)
    pd.DataFrame([_weather_row(5.0)]).to_parquet(archive, index=False)
    pd.DataFrame([_weather_row(1.0)]).to_parquet(older_forecast, index=False)
    pd.DataFrame([_weather_row(7.0)]).to_parquet(latest_forecast, index=False)

    loaded = builder.read_load_cache(load_root)
    assert len(loaded) == 1
    assert loaded["load"].iloc[0] == pytest.approx(12.0)

    archive_weather = builder.read_archive_weather(weather_root)
    assert len(archive_weather) == 1
    assert archive_weather["temperature_2m"].iloc[0] == pytest.approx(5.0)
    assert set(WEATHER_COLUMNS).issubset(archive_weather.columns)

    forecast_weather = builder.read_forecast_weather(weather_root)
    assert len(forecast_weather) == 1
    assert forecast_weather["temperature_2m"].iloc[0] == pytest.approx(7.0)

    output = tmp_path / "processed" / "load_area_hourly.parquet"
    code = builder.main(
        [
            "--load-root",
            str(load_root),
            "--weather-root",
            str(weather_root),
            "--output",
            str(output),
        ]
    )
    assert code == 0
    written = pd.read_parquet(output)
    assert len(written) == 1
    assert written["load"].iloc[0] == pytest.approx(12.0)
    assert written["temperature_2m"].iloc[0] == pytest.approx(5.0)
