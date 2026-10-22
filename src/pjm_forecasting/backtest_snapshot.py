"""Materialize a hierarchical backtest snapshot for the dashboard.

Per-area Beam predictions under ``data/processed/backtests/`` are collapsed to
a fixed lead time (default 1h-ahead), joined to metered load, scored, and
rolled up by summing child series at each parent node.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pjm_forecasting.areas import BY_CODE, LOAD_AREAS
from pjm_forecasting.reconcile import forecast_nodes
from pjm_forecasting.settings import PipelineSettings, load_settings
from pjm_forecasting.training import score_forecast
from pjm_forecasting.utils import EASTERN, project_root

if TYPE_CHECKING:
    from pjm_forecasting.storage import ObjectStore

DEFAULT_LEAD_HOURS = 1
BACKTEST_SNAPSHOT_KEY = "snapshots/backtest/latest.json"


def default_backtest_snapshot_path() -> Path:
    """Local snapshot the API serves when ``BACKTEST_SNAPSHOT`` is unset."""
    return project_root() / "data" / "processed" / "backtest_snapshot.json"


def default_backtest_fixture_path() -> Path:
    """Committed dashboard fixture used to fill areas without real backtests."""
    return project_root() / "app" / "fixtures" / "backtest_snapshot.json"


def area_frames_from_snapshot(payload: dict[str, object]) -> dict[str, pd.DataFrame]:
    """Extract collapsed load-area frames from a snapshot JSON payload."""
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("snapshot is missing nodes")
    frames: dict[str, pd.DataFrame] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if node.get("level") != "load_area":
            continue
        code = node.get("id")
        timestamps = node.get("timestamps")
        actuals = node.get("actuals")
        p10 = node.get("p10")
        p50 = node.get("p50")
        p90 = node.get("p90")
        if not isinstance(code, str) or not timestamps:
            continue
        if not (
            isinstance(timestamps, list)
            and isinstance(actuals, list)
            and isinstance(p10, list)
            and isinstance(p50, list)
            and isinstance(p90, list)
        ):
            raise ValueError(f"load area {code!r} has malformed series")
        n = len(timestamps)
        if not (len(actuals) == len(p10) == len(p50) == len(p90) == n):
            raise ValueError(f"load area {code!r} series lengths do not match")
        frames[code] = pd.DataFrame(
            {
                "timestamp": _as_eastern_index(pd.Series(timestamps)),
                "actual": pd.to_numeric(pd.Series(actuals), errors="coerce"),
                "p10": pd.to_numeric(pd.Series(p10), errors="coerce"),
                "p50": pd.to_numeric(pd.Series(p50), errors="coerce"),
                "p90": pd.to_numeric(pd.Series(p90), errors="coerce"),
            }
        )
    return frames


def merge_area_frames(
    primary: dict[str, pd.DataFrame],
    fallback: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Prefer ``primary`` frames; fill missing load areas from ``fallback``."""
    return {**fallback, **primary}


def _iso(value: object) -> str:
    return pd.Timestamp(value).isoformat()


def _numbers(values: pd.Series) -> list[float]:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if array.size and not np.isfinite(array).all():
        raise ValueError("backtest series contains non-finite values")
    return [float(item) for item in array]


def _as_eastern_index(stamps: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    parsed = pd.DatetimeIndex(pd.to_datetime(stamps))
    if parsed.tz is None:
        return parsed.tz_localize(EASTERN)
    return parsed.tz_convert(EASTERN)


def select_lead_ahead(frame: pd.DataFrame, lead_hours: int = DEFAULT_LEAD_HOURS) -> pd.DataFrame:
    """Keep rows whose whole lead hours equal ``lead_hours`` (one per timestamp)."""
    if lead_hours < 0:
        raise ValueError(f"lead_hours must be >= 0, got {lead_hours}")
    if frame.empty:
        return frame.iloc[0:0].copy()
    required = {"timestamp", "available_at", "quantile_P50"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"predictions are missing columns: {sorted(missing)}")
    work = frame.copy()
    stamps = _as_eastern_index(work["timestamp"])
    available = pd.to_datetime(work["available_at"])
    if available.dt.tz is None:
        available = available.dt.tz_localize("UTC")
    else:
        available = available.dt.tz_convert("UTC")
    stamp_utc = stamps.tz_convert("UTC")
    lead = (stamp_utc - pd.DatetimeIndex(available)).total_seconds() / 3600.0
    work = work.assign(
        timestamp=stamps,
        available_at=available.to_numpy(),
        _lead_hours=np.floor(lead).astype(int),
    )
    selected = work.loc[work["_lead_hours"] == lead_hours].copy()
    if selected.empty:
        return selected.drop(columns=["_lead_hours"])
    selected = selected.sort_values(["timestamp", "available_at"])
    return selected.groupby("timestamp", as_index=False).tail(1).drop(columns=["_lead_hours"])


def _load_actuals(path: Path) -> pd.Series:
    """Metered load indexed by US/Eastern hour beginning."""
    frame = pd.read_parquet(path)
    if "timestamp" not in frame.columns or "load" not in frame.columns:
        raise ValueError(f"{path} must include timestamp and load")
    stamps = _as_eastern_index(frame["timestamp"])
    load = pd.to_numeric(frame["load"], errors="coerce")
    series = pd.Series(load.to_numpy(), index=stamps, name="load")
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series


def collapse_area_backtest(
    predictions: pd.DataFrame,
    actual: pd.Series,
    *,
    lead_hours: int = DEFAULT_LEAD_HOURS,
) -> pd.DataFrame:
    """One row per hour: actual + quantiles at the chosen lead time."""
    selected = select_lead_ahead(predictions, lead_hours=lead_hours)
    if selected.empty:
        return pd.DataFrame(columns=["timestamp", "actual", "p10", "p50", "p90"])
    stamps = _as_eastern_index(selected["timestamp"])
    observed = actual.copy()
    observed.index = _as_eastern_index(observed.index)
    rows = pd.DataFrame(
        {
            "timestamp": stamps,
            "actual": observed.reindex(stamps).to_numpy(),
            "p50": pd.to_numeric(selected["quantile_P50"], errors="coerce").to_numpy(),
        }
    )
    if "quantile_P10" in selected.columns:
        rows["p10"] = pd.to_numeric(selected["quantile_P10"], errors="coerce").to_numpy()
    else:
        rows["p10"] = np.nan
    if "quantile_P90" in selected.columns:
        rows["p90"] = pd.to_numeric(selected["quantile_P90"], errors="coerce").to_numpy()
    else:
        rows["p90"] = np.nan
    rows = rows.dropna(subset=["actual", "p50"]).sort_values("timestamp").reset_index(drop=True)
    return rows


def _metrics_for_frame(frame: pd.DataFrame) -> dict[str, float] | None:
    if frame.empty:
        return None
    forecast = pd.DataFrame(
        {
            "quantile_P50": frame["p50"].to_numpy(),
            "quantile_P10": frame["p10"].to_numpy(),
            "quantile_P90": frame["p90"].to_numpy(),
        },
        index=pd.DatetimeIndex(frame["timestamp"]),
    )
    actual = pd.Series(frame["actual"].to_numpy(), index=forecast.index)
    try:
        return score_forecast(actual, forecast)
    except ValueError:
        return None


def _sum_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["timestamp", "actual", "p10", "p50", "p90"])
    aligned = None
    for frame in frames:
        piece = frame.set_index("timestamp")[["actual", "p10", "p50", "p90"]]
        aligned = piece if aligned is None else aligned.add(piece, fill_value=np.nan)
    assert aligned is not None
    # Require every child to contribute so parent hours stay comparable.
    complete = aligned.dropna(how="any")
    return complete.reset_index()


def _node_payload(node_id: str, level: str, parent_id: str | None, name: str, frame: pd.DataFrame) -> dict:
    metrics = _metrics_for_frame(frame)
    payload: dict[str, object] = {
        "id": node_id,
        "level": level,
        "parent_id": parent_id,
        "name": name,
        "timestamps": [_iso(value) for value in frame["timestamp"]],
        "actuals": _numbers(frame["actual"]),
        "p10": _numbers(frame["p10"]),
        "p50": _numbers(frame["p50"]),
        "p90": _numbers(frame["p90"]),
    }
    if metrics is not None:
        payload["metrics"] = {key: float(value) for key, value in metrics.items()}
    return payload


def build_backtest_snapshot(
    area_frames: dict[str, pd.DataFrame],
    *,
    lead_hours: int = DEFAULT_LEAD_HOURS,
    generated_at: str | pd.Timestamp | None = None,
) -> dict[str, object]:
    """Roll load-area collapsed series into the full forecast hierarchy."""
    if not area_frames:
        raise ValueError("area_frames is empty")
    unknown = sorted(set(area_frames) - set(BY_CODE))
    if unknown:
        raise ValueError(f"unknown load areas: {unknown}")
    for code, frame in area_frames.items():
        required = {"timestamp", "actual", "p10", "p50", "p90"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{code} frame is missing columns: {sorted(missing)}")

    generated = pd.Timestamp(generated_at) if generated_at is not None else pd.Timestamp.now(tz="UTC")
    if generated.tzinfo is None:
        generated = generated.tz_localize("UTC")

    leaves: dict[str, pd.DataFrame] = {
        code: frame.sort_values("timestamp").reset_index(drop=True) for code, frame in area_frames.items()
    }
    nodes: list[dict[str, object]] = []
    window_start: pd.Timestamp | None = None
    window_end: pd.Timestamp | None = None

    for node in forecast_nodes():
        child_frames = [leaves[code] for code in node.load_areas if code in leaves]
        if not child_frames:
            frame = pd.DataFrame(columns=["timestamp", "actual", "p10", "p50", "p90"])
        elif node.level == "load_area":
            frame = child_frames[0]
        else:
            frame = _sum_frames(child_frames)
        nodes.append(_node_payload(node.id, node.level, node.parent_id, node.name, frame))
        if not frame.empty:
            start = pd.Timestamp(frame["timestamp"].iloc[0])
            end = pd.Timestamp(frame["timestamp"].iloc[-1])
            window_start = start if window_start is None else min(window_start, start)
            window_end = end if window_end is None else max(window_end, end)

    if window_start is None or window_end is None:
        raise ValueError("no overlapping backtest hours to snapshot")

    return {
        "generated_at": _iso(generated),
        "timezone": EASTERN,
        "start": _iso(window_start),
        "end": _iso(window_end),
        "lead_hours": int(lead_hours),
        "nodes": nodes,
    }


def load_area_frames_from_disk(
    *,
    settings: PipelineSettings | None = None,
    lead_hours: int = DEFAULT_LEAD_HOURS,
    areas: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Read backtest + versioned load parquet and collapse each area.

    Areas without a predictions parquet are skipped. Raises ``FileNotFoundError``
    when none of the requested areas have predictions on disk.
    """
    settings = settings or load_settings()
    backtest_dir = settings.processed_dir / "backtests"
    load_dir = settings.versioned_load_dir()
    codes = [code.strip().upper() for code in areas] if areas else [area.code for area in LOAD_AREAS]
    frames: dict[str, pd.DataFrame] = {}
    for code in codes:
        if code not in BY_CODE:
            raise ValueError(f"Unknown load area {code!r}")
        pred_path = backtest_dir / f"{code}.parquet"
        load_path = load_dir / f"{code}.parquet"
        if not pred_path.is_file():
            continue
        if not load_path.is_file():
            raise FileNotFoundError(f"missing versioned load for {code}: {load_path}")
        predictions = pd.read_parquet(pred_path)
        if "timestamp" not in predictions.columns and predictions.index.name == "timestamp":
            predictions = predictions.reset_index()
        frames[code] = collapse_area_backtest(
            predictions,
            _load_actuals(load_path),
            lead_hours=lead_hours,
        )
    if not frames:
        raise FileNotFoundError(f"no backtest parquet files under {backtest_dir}")
    return frames


def _load_fixture_payload(path: Path | str | None = None) -> dict[str, object]:
    fixture = Path(path) if path is not None else default_backtest_fixture_path()
    if not fixture.is_file():
        raise FileNotFoundError(f"backtest fixture not found: {fixture}")
    payload = json.loads(fixture.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"backtest fixture is not an object: {fixture}")
    return payload


def resolve_backtest_snapshot(
    *,
    settings: PipelineSettings | None = None,
    lead_hours: int = DEFAULT_LEAD_HOURS,
    generated_at: str | pd.Timestamp | None = None,
    areas: list[str] | None = None,
    fixture_path: Path | str | None = None,
    processed_path: Path | str | None = None,
    area_frames: dict[str, pd.DataFrame] | None = None,
) -> dict[str, object]:
    """Build a full hierarchy: real area frames where present, fixture elsewhere.

    Real frames come from ``area_frames``, else backtest parquets on disk, else
    load-area nodes in an existing processed snapshot. Missing load areas are
    filled from the dashboard fixture so the map stays complete while backtests
    are still rolling out.
    """
    fixture_payload = _load_fixture_payload(fixture_path)
    fixture_frames = area_frames_from_snapshot(fixture_payload)

    real_frames = area_frames
    if real_frames is None:
        try:
            real_frames = load_area_frames_from_disk(
                settings=settings,
                lead_hours=lead_hours,
                areas=areas,
            )
        except FileNotFoundError:
            real_frames = {}
        if not real_frames:
            processed = Path(processed_path) if processed_path is not None else default_backtest_snapshot_path()
            if processed.is_file():
                processed_payload = json.loads(processed.read_text())
                if isinstance(processed_payload, dict):
                    real_frames = area_frames_from_snapshot(processed_payload)

    if not real_frames:
        return fixture_payload

    merged = merge_area_frames(real_frames, fixture_frames)
    return build_backtest_snapshot(
        merged,
        lead_hours=lead_hours,
        generated_at=generated_at,
    )


def write_backtest_snapshot(
    area_frames: dict[str, pd.DataFrame] | None = None,
    *,
    path: Path | str | None = None,
    settings: PipelineSettings | None = None,
    lead_hours: int = DEFAULT_LEAD_HOURS,
    generated_at: str | pd.Timestamp | None = None,
    areas: list[str] | None = None,
    fill_missing_from_fixture: bool = True,
    fixture_path: Path | str | None = None,
) -> Path:
    """Build and write the dashboard backtest snapshot JSON.

    When ``fill_missing_from_fixture`` is true (default), load areas without
    real backtest frames are taken from the committed fixture so parents and
    the map stay populated during a partial rollout.
    """
    if fill_missing_from_fixture:
        payload = resolve_backtest_snapshot(
            settings=settings,
            lead_hours=lead_hours,
            generated_at=generated_at,
            areas=areas,
            fixture_path=fixture_path,
            area_frames=area_frames,
        )
    else:
        frames = area_frames if area_frames is not None else load_area_frames_from_disk(
            settings=settings,
            lead_hours=lead_hours,
            areas=areas,
        )
        payload = build_backtest_snapshot(
            frames,
            lead_hours=lead_hours,
            generated_at=generated_at,
        )
    destination = Path(path) if path is not None else default_backtest_snapshot_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return destination


def publish_backtest_snapshot(
    area_frames: dict[str, pd.DataFrame] | None = None,
    *,
    store: ObjectStore | None = None,
    path: Path | str | None = None,
    settings: PipelineSettings | None = None,
    lead_hours: int = DEFAULT_LEAD_HOURS,
    generated_at: str | pd.Timestamp | None = None,
    areas: list[str] | None = None,
    fill_missing_from_fixture: bool = True,
    fixture_path: Path | str | None = None,
) -> dict[str, object]:
    """Write the dashboard snapshot locally and upload it to object storage.

    Object-store keys match the live forecast layout: ``snapshots/backtest/latest.json``
    plus a timestamped archive copy.
    """
    from pjm_forecasting.storage import ObjectStore as Store

    destination = write_backtest_snapshot(
        area_frames,
        path=path,
        settings=settings,
        lead_hours=lead_hours,
        generated_at=generated_at,
        areas=areas,
        fill_missing_from_fixture=fill_missing_from_fixture,
        fixture_path=fixture_path,
    )
    payload = json.loads(destination.read_text())
    object_store = store if store is not None else Store()
    object_store.write_json(BACKTEST_SNAPSHOT_KEY, payload)
    generated = payload.get("generated_at")
    archive_key: str | None = None
    if isinstance(generated, str) and generated:
        stamp = pd.Timestamp(generated)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        token = stamp.tz_convert("UTC").strftime("%Y%m%dT%H%MZ")
        archive_key = f"snapshots/backtest/{token}.json"
        object_store.write_json(archive_key, payload)
    return {
        "path": str(destination),
        "key": BACKTEST_SNAPSHOT_KEY,
        "archive_key": archive_key,
        "generated_at": generated,
        "start": payload.get("start"),
        "end": payload.get("end"),
    }
