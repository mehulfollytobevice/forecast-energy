"""Training table and inference frame. No network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pjm_forecasting.frame import build_inference_frame, build_load_area_frame
from pjm_forecasting.schema import FRAME_COLUMNS, WEATHER_COLUMNS


def _stamp(hour: int) -> pd.Timestamp:
    return pd.Timestamp("2024-01-01", tz="US/Eastern") + pd.Timedelta(hours=hour)


def _load_row(area: str, hour: int, load: float, zone: str = "AE", region: str = "MIDATL") -> dict:
    stamp = _stamp(hour)
    return {
        "timestamp": stamp,
        "timestamp_utc": stamp.tz_convert("UTC"),
        "load_area": area,
        "zone": zone,
        "mkt_region": region,
        "load": load,
    }


def _weather_row(area: str, hour: int, temperature: float) -> dict:
    stamp = _stamp(hour)
    row = {
        "timestamp": stamp,
        "timestamp_utc": stamp.tz_convert("UTC"),
        "load_area": area,
        "temperature_2m": temperature,
        "relative_humidity_2m": 80.0,
        "wind_speed_10m": 3.0,
        "shortwave_radiation": 0.0,
        "surface_pressure": 1010.0,
    }
    return row


def test_join_rejects_duplicate_keys():
    load = pd.DataFrame([_load_row("AECO", 0, 10.0), _load_row("AECO", 0, 11.0)])
    weather = pd.DataFrame([_weather_row("AECO", 0, 1.0)])
    with pytest.raises(ValueError, match="timestamp_utc, load_area"):
        build_load_area_frame(load, weather)


def test_join_keeps_the_overlap_only():
    load = pd.DataFrame([_load_row("AECO", 0, 10.0), _load_row("AECO", 1, 12.0)])
    weather = pd.DataFrame([_weather_row("AECO", 1, 5.0), _weather_row("BC", 1, 6.0)])
    frame = build_load_area_frame(load, weather)
    assert list(frame.columns) == list(FRAME_COLUMNS)
    assert len(frame) == 1
    assert frame.loc[0, "load_area"] == "AECO"
    assert frame.loc[0, "load"] == pytest.approx(12.0)
    assert frame.loc[0, "temperature_2m"] == pytest.approx(5.0)
    assert frame.loc[0, "zone"] == "AE"


def test_inference_nulls_load_only_after_the_origin():
    history = pd.DataFrame(
        [
            {**_load_row("AECO", hour, 10.0 + hour), **{column: 1.0 for column in WEATHER_COLUMNS}}
            for hour in range(3)
        ]
    )
    history["timestamp_utc"] = history["timestamp"].dt.tz_convert("UTC")
    forecast = pd.DataFrame([_weather_row("AECO", hour, 20.0 + hour) for hour in range(1, 5)])
    frame = build_inference_frame(history, forecast, horizon_hours=2, origin=_stamp(2))
    assert frame["timestamp"].dt.hour.tolist() == [0, 1, 2, 3, 4]
    assert frame["load"].iloc[:3].tolist() == [10.0, 11.0, 12.0]
    assert frame["load"].iloc[3:].isna().all()
    assert frame["temperature_2m"].iloc[3] == pytest.approx(23.0)


def test_inference_errors_when_a_horizon_hour_has_no_weather():
    history = pd.DataFrame([{**_load_row("AECO", 0, 10.0), **{c: 1.0 for c in WEATHER_COLUMNS}}])
    history["timestamp_utc"] = history["timestamp"].dt.tz_convert("UTC")
    forecast = pd.DataFrame([_weather_row("AECO", 1, 4.0), _weather_row("AECO", 2, 5.0)])
    forecast.loc[1, "wind_speed_10m"] = np.nan
    with pytest.raises(ValueError, match="missing weather"):
        build_inference_frame(history, forecast, horizon_hours=2, origin=_stamp(0))
