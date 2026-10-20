"""Join load-area load and weather into the table OpenSTEF will train on.

The same columns are used for history and for the daily horizon. Horizon
hours keep weather and set ``load`` to null. Lags and calendar fields are
left for OpenSTEF.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pjm_forecasting.areas import BY_CODE
from pjm_forecasting.schema import FRAME_COLUMNS, LOAD_COLUMNS, WEATHER_COLUMNS
from pjm_forecasting.utils import EASTERN, as_eastern, ensure_dir

JOIN_KEYS = ["timestamp_utc", "load_area"]


def _as_utc(series: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(series, utc=True)
    return stamps


def _as_eastern_series(series: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(series)
    if stamps.dt.tz is None:
        return stamps.dt.tz_localize(EASTERN)
    return stamps.dt.tz_convert(EASTERN)


def _reject_duplicate_keys(frame: pd.DataFrame, name: str) -> None:
    if frame.duplicated(subset=JOIN_KEYS).any():
        raise ValueError(f"{name} has duplicate (timestamp_utc, load_area) rows")


def _prepare_load(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in LOAD_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"load frame is missing columns: {missing}")
    work = frame.loc[:, list(LOAD_COLUMNS)].copy()
    work["timestamp_utc"] = _as_utc(work["timestamp_utc"])
    work["timestamp"] = _as_eastern_series(work["timestamp"])
    work["load_area"] = work["load_area"].astype(str).str.strip().str.upper()
    _reject_duplicate_keys(work, "load frame")
    return work


def _prepare_weather(frame: pd.DataFrame) -> pd.DataFrame:
    required = ("timestamp_utc", "load_area", *WEATHER_COLUMNS)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"weather frame is missing columns: {missing}")
    keep = ["timestamp_utc", "load_area", *WEATHER_COLUMNS]
    work = frame.loc[:, keep].copy()
    work["timestamp_utc"] = _as_utc(work["timestamp_utc"])
    work["load_area"] = work["load_area"].astype(str).str.strip().str.upper()
    for column in WEATHER_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    _reject_duplicate_keys(work, "weather frame")
    return work


def build_load_area_frame(load: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Inner-join load and weather on UTC hour and load area.

    Hours missing from either source are left out. Zone and market region
    come from the load rows.
    """
    if load.empty:
        raise ValueError("load table is empty")
    if weather.empty:
        raise ValueError("weather table is empty")
    load_work = _prepare_load(load)
    weather_work = _prepare_weather(weather)
    joined = load_work.merge(weather_work, on=JOIN_KEYS, how="inner")
    if joined.empty:
        raise ValueError("load and weather have no overlapping (timestamp_utc, load_area) hours")
    joined = joined.sort_values(["load_area", "timestamp_utc"]).reset_index(drop=True)
    return joined.loc[:, list(FRAME_COLUMNS)]


def build_inference_frame(
    history: pd.DataFrame,
    forecast_weather: pd.DataFrame,
    horizon_hours: int,
    origin: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """History through the origin, then ``horizon_hours`` with ``load`` null.

    ``origin`` is the last hour that keeps metered load. When it is omitted,
    each load area uses its own last hour with a load value. Every horizon
    hour must have weather.
    """
    if horizon_hours < 1:
        raise ValueError(f"horizon_hours must be >= 1, got {horizon_hours}")
    if history.empty:
        raise ValueError("history table is empty")
    past = _prepare_load(history)
    weather_columns_present = [column for column in WEATHER_COLUMNS if column in history.columns]
    if weather_columns_present:
        extra = history.loc[:, ["timestamp_utc", "load_area", *weather_columns_present]].copy()
        extra["timestamp_utc"] = _as_utc(extra["timestamp_utc"])
        extra["load_area"] = extra["load_area"].astype(str).str.strip().str.upper()
        past = past.drop(columns=weather_columns_present, errors="ignore").merge(
            extra, on=JOIN_KEYS, how="left"
        )
    forecast = _prepare_weather(forecast_weather)
    shared_origin = as_eastern(origin) if origin is not None else None

    parts: list[pd.DataFrame] = []
    for code, hist in past.groupby("load_area", sort=True):
        area = str(code)
        if shared_origin is None:
            known = hist.loc[hist["load"].notna(), "timestamp"]
            if known.empty:
                raise ValueError(f"{area} has no metered load")
            origin_ts = known.max()
        else:
            origin_ts = shared_origin
        horizon_end = origin_ts + pd.Timedelta(hours=horizon_hours)
        origin_utc = origin_ts.tz_convert("UTC")
        horizon_utc = horizon_end.tz_convert("UTC")
        kept = hist.loc[hist["timestamp_utc"] <= origin_utc].copy()
        future = forecast.loc[
            (forecast["load_area"] == area)
            & (forecast["timestamp_utc"] > origin_utc)
            & (forecast["timestamp_utc"] <= horizon_utc)
        ].copy()
        if len(future) != horizon_hours:
            raise ValueError(
                f"expected {horizon_hours} horizon hours after {origin_ts} for {area}, "
                f"found {len(future)}"
            )
        missing_weather = future[list(WEATHER_COLUMNS)].isna().any(axis=1)
        if missing_weather.any():
            raise ValueError(
                f"{int(missing_weather.sum())} horizon hours for {area} are missing weather"
            )
        catalog = BY_CODE[area]
        future["timestamp"] = future["timestamp_utc"].dt.tz_convert(EASTERN)
        future["zone"] = catalog.zone
        future["mkt_region"] = catalog.mkt_region
        future["load"] = np.nan
        for column in WEATHER_COLUMNS:
            if column not in kept.columns:
                kept[column] = np.nan
        parts.append(kept.loc[:, list(FRAME_COLUMNS)])
        parts.append(future.loc[:, list(FRAME_COLUMNS)])

    frame = pd.concat(parts, ignore_index=True)
    frame = frame.sort_values(["load_area", "timestamp_utc"]).reset_index(drop=True)
    _reject_duplicate_keys(frame, "inference frame")
    return frame


def write_frame(frame: pd.DataFrame, path: Path | str) -> Path:
    """Write the load-area frame to parquet."""
    destination = Path(path)
    ensure_dir(destination.parent)
    frame.to_parquet(destination, index=False)
    return destination
