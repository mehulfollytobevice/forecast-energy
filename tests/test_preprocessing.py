"""OpenSTEF frame preparation, including daylight-saving hours."""

from __future__ import annotations

import pandas as pd

from pjm_forecasting.preprocessing import OPENSTEF_COLUMNS, prepare_openstef_frame
from pjm_forecasting.schema import WEATHER_COLUMNS


def _frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for stamp in index:
        row = {
            "timestamp": stamp,
            "timestamp_utc": stamp.tz_convert("UTC"),
            "load_area": "AECO",
            "zone": "AE",
            "mkt_region": "MIDATL",
            "load": 10.0 + stamp.hour,
        }
        for column in WEATHER_COLUMNS:
            row[column] = 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_prepare_drops_identifiers_and_keeps_both_fallback_hours():
    utc = pd.date_range("2024-11-03 04:00", periods=5, freq="h", tz="UTC")
    prepared = prepare_openstef_frame(_frame(utc.tz_convert("US/Eastern")), "aeco")
    assert list(prepared.columns) == list(OPENSTEF_COLUMNS)
    assert prepared.index.name == "timestamp"
    assert str(prepared.index.tz) == "US/Eastern"
    assert prepared.index.is_monotonic_increasing
    assert prepared.index.is_unique
    offsets = {stamp.utcoffset() for stamp in prepared.index}
    assert len(offsets) == 2
    clock = prepared.index.strftime("%Y-%m-%d %H:%M")
    assert list(clock).count("2024-11-03 01:00") == 2


def test_prepare_does_not_invent_the_spring_forward_hour():
    utc = pd.date_range("2024-03-10 05:00", periods=4, freq="h", tz="UTC")
    prepared = prepare_openstef_frame(_frame(utc.tz_convert("US/Eastern")), "AECO")
    assert "2024-03-10 02:00" not in set(prepared.index.strftime("%Y-%m-%d %H:%M"))
    assert prepared["load"].isna().sum() == 0
    assert len(prepared) == 4
