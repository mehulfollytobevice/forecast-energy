"""One JSON snapshot of reconciled forecasts and a short actuals tail.

The hierarchy is ``LOAD_AREAS`` via :func:`forecast_nodes`. Parent actuals
use :func:`rollup_load`. Parent forecast series use :func:`reconcile_node`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pjm_forecasting.areas import BY_CODE
from pjm_forecasting.ingest.pjm import rollup_load
from pjm_forecasting.reconcile import forecast_nodes, reconcile_node
from pjm_forecasting.utils import EASTERN, as_eastern, project_root

_ACTUAL_COLUMNS = ("timestamp", "timestamp_utc", "load_area", "zone", "mkt_region", "load")


def default_snapshot_path() -> Path:
    """Local snapshot the API serves when ``FORECAST_SNAPSHOT`` is unset."""
    return project_root() / "data" / "processed" / "forecast_snapshot.json"


def _as_eastern(series: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(series)
    if stamps.dt.tz is None:
        return stamps.dt.tz_localize(EASTERN)
    return stamps.dt.tz_convert(EASTERN)


def _iso(value: object) -> str:
    return pd.Timestamp(value).isoformat()


def _numbers(values: pd.Series) -> list[float]:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if array.size and not np.isfinite(array).all():
        raise ValueError("snapshot series contains non-finite values")
    return [float(item) for item in array]


def _prepare_actuals(actuals: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in _ACTUAL_COLUMNS if column not in actuals.columns]
    if missing:
        raise ValueError(f"actuals are missing columns: {missing}")
    work = actuals.loc[:, list(_ACTUAL_COLUMNS)].copy()
    work["timestamp"] = _as_eastern(work["timestamp"])
    work["timestamp_utc"] = pd.to_datetime(work["timestamp_utc"], utc=True)
    work["load_area"] = work["load_area"].astype(str).str.strip().str.upper()
    work["load"] = pd.to_numeric(work["load"], errors="coerce")
    unknown = sorted(set(work["load_area"]) - set(BY_CODE))
    if unknown:
        raise ValueError(f"actuals include load areas outside the catalog: {unknown}")
    work["zone"] = work["load_area"].map(lambda code: BY_CODE[code].zone)
    work["mkt_region"] = work["load_area"].map(lambda code: BY_CODE[code].mkt_region)
    return work


def _history_window(actuals: pd.DataFrame, origin: pd.Timestamp, actual_hours: int) -> pd.DataFrame:
    history = actuals.loc[actuals["timestamp"] < origin]
    if history.empty:
        return history.iloc[0:0]
    stamps = pd.DatetimeIndex(pd.unique(history["timestamp"])).sort_values()
    tail = stamps[-actual_hours:]
    return history.loc[history["timestamp"].isin(tail)].copy()


def _node_actuals(
    node_id: str,
    level: str,
    load_areas: tuple[str, ...],
    window: pd.DataFrame,
) -> pd.DataFrame:
    subset = window.loc[window["load_area"].isin(load_areas)]
    empty = pd.DataFrame(columns=["timestamp", "load"])
    if subset.empty:
        return empty
    if level == "load_area":
        rows = subset.sort_values(["timestamp", "timestamp_utc"])
        return rows.loc[:, ["timestamp", "load"]].reset_index(drop=True)
    if level == "zone":
        rolled = rollup_load(subset, "zone")
    elif level == "mkt_region":
        rolled = rollup_load(subset, "mkt_region")
    elif level == "pjm":
        rolled = rollup_load(subset, "pjm")
    else:
        raise ValueError(f"Unknown forecast level {level!r} for {node_id}")
    return rolled.sort_values("timestamp").loc[:, ["timestamp", "load"]].reset_index(drop=True)


def _stamp(value: str | pd.Timestamp | None, *, default_tz: str) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz=default_tz)
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize(default_tz)
    return stamp


def build_forecast_snapshot(
    child_forecasts: pd.DataFrame,
    aggregate_forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    origin: str | pd.Timestamp,
    *,
    generated_at: str | pd.Timestamp | None = None,
    actual_hours: int = 24,
    snapshot_load_areas: frozenset[str] | None = None,
) -> dict[str, object]:
    """Reconciled tree for every catalog node, plus a short actuals tail.

    ``child_forecasts`` is load-area grain with ``load_area``, ``timestamp``,
    ``p10``, ``p50``, and ``p90``. ``aggregate_forecasts`` holds the same
    quantiles for each aggregate id. One-child zones and SOUTH copy the
    load-area series. ``origin`` is the first forecast hour.

    When ``snapshot_load_areas`` is set, only nodes whose leaf load areas are
    contained in that set are included (partial live runs).
    """
    if actual_hours < 1:
        raise ValueError(f"actual_hours must be >= 1, got {actual_hours}")
    origin_ts = as_eastern(origin)
    if any((origin_ts.minute, origin_ts.second, origin_ts.microsecond, origin_ts.nanosecond)):
        raise ValueError("origin must be an hour beginning")
    generated = _stamp(generated_at, default_tz="UTC")
    window = _history_window(_prepare_actuals(actuals), origin_ts, actual_hours)
    nodes: list[dict[str, object]] = []
    for node in forecast_nodes():
        if snapshot_load_areas is not None and not frozenset(node.load_areas) <= snapshot_load_areas:
            continue
        forecast = reconcile_node(
            node.id,
            child_forecasts,
            aggregate_forecasts,
            level=node.level,
        )
        observed = _node_actuals(node.id, node.level, node.load_areas, window)
        timestamps = [_iso(value) for value in forecast["timestamp"]]
        p10 = _numbers(forecast["p10"])
        p50 = _numbers(forecast["p50"])
        p90 = _numbers(forecast["p90"])
        actual_timestamps = [_iso(value) for value in observed["timestamp"]]
        actual_values = _numbers(observed["load"])
        if not (len(timestamps) == len(p10) == len(p50) == len(p90)):
            raise ValueError(f"{node.id} forecast lengths do not match")
        if len(actual_timestamps) != len(actual_values):
            raise ValueError(f"{node.id} actuals lengths do not match")
        nodes.append(
            {
                "id": node.id,
                "level": node.level,
                "parent_id": node.parent_id,
                "name": node.name,
                "timestamps": timestamps,
                "p10": p10,
                "p50": p50,
                "p90": p90,
                "actual_timestamps": actual_timestamps,
                "actuals": actual_values,
            }
        )
    return {
        "generated_at": _iso(generated),
        "origin": _iso(origin_ts),
        "timezone": EASTERN,
        "nodes": nodes,
    }


def write_forecast_snapshot(
    child_forecasts: pd.DataFrame,
    aggregate_forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    origin: str | pd.Timestamp,
    *,
    path: Path | str | None = None,
    generated_at: str | pd.Timestamp | None = None,
    actual_hours: int = 24,
    snapshot_load_areas: frozenset[str] | None = None,
) -> Path:
    """Write :func:`build_forecast_snapshot` to ``path`` or the default file."""
    payload = build_forecast_snapshot(
        child_forecasts,
        aggregate_forecasts,
        actuals,
        origin,
        generated_at=generated_at,
        actual_hours=actual_hours,
        snapshot_load_areas=snapshot_load_areas,
    )
    destination = Path(path) if path is not None else default_snapshot_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return destination
