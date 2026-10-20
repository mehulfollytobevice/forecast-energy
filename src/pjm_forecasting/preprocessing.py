"""OpenSTEF input frames.

The index is the US/Eastern hour. Columns are the load target and the numeric
weather predictors. Identifier columns are dropped so OpenSTEF does not treat
them as features. Lags, holidays, and calendar fields are left to OpenSTEF.
Load is not imputed, and hours missing across a daylight-saving transition
are not invented.
"""

from __future__ import annotations

import pandas as pd

from pjm_forecasting.areas import BY_CODE
from pjm_forecasting.schema import WEATHER_COLUMNS
from pjm_forecasting.utils import EASTERN

OPENSTEF_COLUMNS: tuple[str, ...] = ("load", *WEATHER_COLUMNS)


def prepare_openstef_frame(frame: pd.DataFrame, load_area: str) -> pd.DataFrame:
    """Return one area, sorted, with a timezone-aware hourly index."""
    code = load_area.strip().upper()
    if code not in BY_CODE:
        raise ValueError(f"Unknown load area {load_area!r}")
    if "load_area" not in frame.columns or "timestamp" not in frame.columns:
        raise ValueError("frame must include timestamp and load_area")
    labels = frame["load_area"].astype(str).str.strip().str.upper()
    work = frame.loc[labels == code].copy()
    if work.empty:
        raise ValueError(f"{code} has no rows")
    missing = [column for column in OPENSTEF_COLUMNS if column not in work.columns]
    if missing:
        raise ValueError(f"{code} is missing columns: {missing}")

    stamps = pd.to_datetime(work["timestamp"])
    if stamps.dt.tz is None:
        stamps = stamps.dt.tz_localize(EASTERN)
    else:
        stamps = stamps.dt.tz_convert(EASTERN)
    work = work.assign(timestamp=stamps).sort_values("timestamp")
    if work["timestamp"].duplicated().any():
        raise ValueError(f"{code} has duplicate timestamps")

    prepared = work.loc[:, ["timestamp", *OPENSTEF_COLUMNS]].set_index("timestamp")
    prepared.index.name = "timestamp"
    for column in OPENSTEF_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return prepared
