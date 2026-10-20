"""Quality gates. No network."""

from __future__ import annotations

import pandas as pd
import pytest

from pjm_forecasting.quality import (
    QualityError,
    validate_joined,
    validate_load,
    validate_versioned_load,
    validate_versioned_weather,
    validate_weather,
)
from pjm_forecasting.schema import WEATHER_COLUMNS


def _stamp(hour: int) -> pd.Timestamp:
    return pd.Timestamp("2024-01-01", tz="US/Eastern") + pd.Timedelta(hours=hour)


def _load_frame(hours: range | list[int] | None = None) -> pd.DataFrame:
    rows = []
    for hour in hours if hours is not None else range(8):
        stamp = _stamp(hour)
        rows.append(
            {
                "timestamp": stamp,
                "timestamp_utc": stamp.tz_convert("UTC"),
                "load_area": "AECO",
                "zone": "AE",
                "mkt_region": "MIDATL",
                "load": 100.0 + hour,
            }
        )
    return pd.DataFrame(rows)


def _weather_frame(hours: range | list[int] | None = None) -> pd.DataFrame:
    rows = []
    for hour in hours if hours is not None else range(8):
        stamp = _stamp(hour)
        row = {
            "timestamp": stamp,
            "timestamp_utc": stamp.tz_convert("UTC"),
            "load_area": "AECO",
        }
        for column in WEATHER_COLUMNS:
            row[column] = 1.0 + hour
        rows.append(row)
    return pd.DataFrame(rows)


def _joined_frame() -> pd.DataFrame:
    return _load_frame().merge(_weather_frame(), on=["timestamp", "timestamp_utc", "load_area"])


def test_valid_frames_report_coverage_and_pass():
    report = validate_joined(_joined_frame())
    report.raise_if_fatal()
    assert report.stats["coverage"] == pytest.approx(1.0)
    assert report.stats["rows"] == 8
    assert any("coverage=" in line for line in report.log_lines())


def test_missing_column_is_fatal():
    frame = _load_frame().drop(columns=["load"])
    report = validate_load(frame)
    assert any(issue.check == "columns" and issue.fatal for issue in report.issues)
    with pytest.raises(QualityError, match="columns"):
        report.raise_if_fatal()


def test_bad_timestamp_dtype_is_fatal():
    frame = _load_frame()
    frame["timestamp"] = frame["timestamp"].astype(str)
    report = validate_load(frame)
    assert any(issue.check == "dtypes" and issue.fatal for issue in report.issues)


def test_timezone_mismatch_is_fatal():
    frame = _load_frame()
    frame["timestamp"] = frame["timestamp_utc"]
    report = validate_load(frame)
    assert any(issue.check == "timezone" and issue.fatal for issue in report.issues)


def test_unknown_load_area_is_fatal():
    frame = _load_frame()
    frame["load_area"] = "NOPE"
    report = validate_load(frame)
    assert any(issue.check == "catalog" and issue.fatal for issue in report.issues)


def test_duplicate_keys_are_fatal():
    frame = pd.concat([_load_frame(), _load_frame()], ignore_index=True)
    report = validate_load(frame)
    assert any(issue.check == "duplicates" and issue.fatal for issue in report.issues)


def test_nonfinite_load_is_fatal():
    frame = _load_frame()
    frame.loc[0, "load"] = float("nan")
    report = validate_load(frame)
    assert any(issue.check == "load_values" and issue.fatal for issue in report.issues)


def test_negative_load_is_fatal():
    frame = _load_frame()
    frame.loc[0, "load"] = -1.0
    report = validate_load(frame)
    assert any(issue.check == "load_values" and issue.fatal for issue in report.issues)


def test_missing_weather_is_fatal():
    frame = _weather_frame()
    frame.loc[1, "temperature_2m"] = float("nan")
    report = validate_weather(frame)
    assert any(issue.check == "weather_values" and issue.fatal for issue in report.issues)


def test_timestamp_mismatch_is_fatal():
    frame = _joined_frame()
    frame["timestamp_utc"] = frame["timestamp_utc"] + pd.Timedelta(hours=1)
    report = validate_joined(frame)
    assert any(issue.check == "timestamp_equivalence" and issue.fatal for issue in report.issues)


def test_reversed_time_is_fatal():
    frame = _load_frame()
    frame = frame.iloc[::-1].reset_index(drop=True)
    report = validate_load(frame)
    assert any(issue.check == "chronological" and issue.fatal for issue in report.issues)


def test_long_gap_is_fatal_and_short_gap_is_only_a_statistic():
    gappy = _load_frame([0, 1, 2, 12, 13, 14])
    fatal = validate_load(gappy, max_gap=pd.Timedelta(hours=6))
    assert any(issue.check == "chronological" and issue.fatal for issue in fatal.issues)
    assert fatal.stats["missing_hours"] > 0

    short = _load_frame([0, 1, 2, 4, 5, 6])
    report = validate_load(short, max_gap=pd.Timedelta(hours=6))
    report.raise_if_fatal()
    assert report.stats["missing_hours"] > 0
    assert report.stats["coverage"] < 1


def _versioned_weather() -> pd.DataFrame:
    run = pd.Timestamp("2024-06-01T00:00", tz="UTC")
    rows = []
    for hour in range(4):
        stamp = pd.Timestamp("2024-06-01T06:00", tz="UTC") + pd.Timedelta(hours=hour)
        row = {
            "timestamp": stamp.tz_convert("US/Eastern"),
            "available_at": run + pd.Timedelta(hours=6),
            "run_initialized_at": run,
            "load_area": "AECO",
        }
        for column in WEATHER_COLUMNS:
            row[column] = 2.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_forecast_available_before_the_run_is_fatal():
    frame = _versioned_weather()
    frame["available_at"] = frame["run_initialized_at"] - pd.Timedelta(hours=1)
    report = validate_versioned_weather(frame, publication_delay=pd.Timedelta(hours=6))
    assert any(issue.check == "point_in_time" and issue.fatal for issue in report.issues)


def test_versioned_weather_accepts_the_publication_delay():
    report = validate_versioned_weather(
        _versioned_weather(),
        publication_delay=pd.Timedelta(hours=6),
    )
    report.raise_if_fatal()
    assert report.stats["runs"] == 1


def test_load_available_before_the_hour_is_fatal():
    frame = _load_frame()
    versioned = pd.DataFrame(
        {
            "timestamp": frame["timestamp"],
            "available_at": frame["timestamp"] - pd.Timedelta(hours=1),
            "load": frame["load"],
            "load_area": frame["load_area"],
        }
    )
    report = validate_versioned_load(versioned, load_delay=pd.Timedelta(hours=1))
    assert any(issue.check == "point_in_time" and issue.fatal for issue in report.issues)
