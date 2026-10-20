"""Latest-available forecast-run weather selection."""

from __future__ import annotations

import pandas as pd
import pytest

from pjm_forecasting.ingest.openmeteo import (
    build_forecast_weather_history,
    latest_run_weather,
)
from pjm_forecasting.schema import WEATHER_COLUMNS


def _versioned() -> pd.DataFrame:
    runs = [
        pd.Timestamp("2024-06-01T00:00", tz="UTC"),
        pd.Timestamp("2024-06-01T06:00", tz="UTC"),
    ]
    rows: list[dict] = []
    for run in runs:
        for hour in range(12):
            stamp = pd.Timestamp("2024-06-01T00:00", tz="US/Eastern") + pd.Timedelta(hours=hour)
            rows.append(
                {
                    "timestamp": stamp,
                    "available_at": run + pd.Timedelta(hours=6),
                    "run_initialized_at": run,
                    "load_area": "AECO",
                    **{column: float(hour + run.hour) for column in WEATHER_COLUMNS},
                }
            )
    return pd.DataFrame(rows)


def test_build_forecast_weather_history_picks_latest_published_run():
    history = build_forecast_weather_history(_versioned())
    assert history["load_area"].nunique() == 1
    # 00Z run publishes at 06:00 UTC = 02:00 ET; 06Z at 12:00 UTC = 08:00 ET.
    early = history.loc[history["timestamp"] == pd.Timestamp("2024-06-01T02:00", tz="US/Eastern")]
    late = history.loc[history["timestamp"] == pd.Timestamp("2024-06-01T08:00", tz="US/Eastern")]
    assert not early.empty
    assert not late.empty
    assert float(early["temperature_2m"].iloc[0]) == 2.0  # hour 2 from 00Z run
    assert float(late["temperature_2m"].iloc[0]) == 14.0  # hour 8 from 06Z run


def test_latest_run_weather_returns_one_run():
    weather = latest_run_weather(
        _versioned(),
        as_of=pd.Timestamp("2024-06-01T18:00", tz="UTC"),
    )
    assert weather["timestamp"].nunique() == 12
    assert set(WEATHER_COLUMNS).issubset(weather.columns)


def test_build_forecast_weather_history_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        build_forecast_weather_history(pd.DataFrame())
