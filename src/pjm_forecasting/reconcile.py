"""Parent forecasts and the weather used to train them.

Displayed P50 at a zone, market region, or PJM is the sum of load-area P50s.
P10 and P90 come from a model of that total, shifted onto the sum, where the
total is more than one load area. Weather for that model is a load-weighted
average of the child cities. Weights are historical mean load, fixed before
the forecast horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from pjm_forecasting.areas import BY_CODE, LOAD_AREAS, LoadArea
from pjm_forecasting.ingest.pjm import RollupLevel, rollup_load
from pjm_forecasting.schema import WEATHER_COLUMNS
from pjm_forecasting.utils import EASTERN, as_eastern

Level = Literal["pjm", "mkt_region", "zone", "load_area"]
_JOINED_COLUMNS = (
    "timestamp",
    "timestamp_utc",
    "load_area",
    "zone",
    "mkt_region",
    "load",
    *WEATHER_COLUMNS,
)


@dataclass(frozen=True)
class ForecastNode:
    """One node in the snapshot tree. ``load_areas`` are the leaves under it."""

    id: str
    level: Level
    parent_id: str | None
    name: str
    load_areas: tuple[str, ...]


def _build_forecast_nodes() -> tuple[ForecastNode, ...]:
    by_region: dict[str, dict[str, list[str]]] = {}
    region_order: list[str] = []
    zone_order: dict[str, list[str]] = {}
    for area in LOAD_AREAS:
        if area.mkt_region not in by_region:
            by_region[area.mkt_region] = {}
            region_order.append(area.mkt_region)
            zone_order[area.mkt_region] = []
        if area.zone not in by_region[area.mkt_region]:
            by_region[area.mkt_region][area.zone] = []
            zone_order[area.mkt_region].append(area.zone)
        by_region[area.mkt_region][area.zone].append(area.code)

    nodes: list[ForecastNode] = [
        ForecastNode("PJM", "pjm", None, "PJM", tuple(area.code for area in LOAD_AREAS)),
    ]
    for region in region_order:
        region_areas = tuple(
            code for zone in zone_order[region] for code in by_region[region][zone]
        )
        nodes.append(ForecastNode(region, "mkt_region", "PJM", region, region_areas))
        for zone in zone_order[region]:
            zone_areas = tuple(by_region[region][zone])
            nodes.append(ForecastNode(zone, "zone", region, zone, zone_areas))
            for code in zone_areas:
                nodes.append(ForecastNode(code, "load_area", zone, code, (code,)))
    seen: set[tuple[str, str]] = set()
    for node in nodes:
        key = (node.level, node.id)
        if key in seen:
            raise RuntimeError(f"duplicate forecast node {node.level} {node.id}")
        seen.add(key)
    return tuple(nodes)


def _aggregate_targets(nodes: tuple[ForecastNode, ...]) -> tuple[ForecastNode, ...]:
    zones = tuple(node for node in nodes if node.level == "zone" and len(node.load_areas) > 1)
    regions = tuple(
        node for node in nodes if node.level == "mkt_region" and len(node.load_areas) > 1
    )
    pjm = tuple(node for node in nodes if node.level == "pjm")
    return zones + regions + pjm


_FORECAST_NODES = _build_forecast_nodes()
_AGGREGATE_TARGETS = _aggregate_targets(_FORECAST_NODES)
_AGGREGATE_IDS = {node.id: node for node in _AGGREGATE_TARGETS}


def forecast_nodes() -> tuple[ForecastNode, ...]:
    """PJM, market regions, zones, and load areas, in drill-down order."""
    return _FORECAST_NODES


def aggregate_targets() -> tuple[ForecastNode, ...]:
    """Totals that are not a single load-area series.

    Multi-area zones, then market regions with more than one load area, then
    full PJM. One-company zones and SOUTH are omitted.
    """
    return _AGGREGATE_TARGETS


def resolve_aggregates(spec: str) -> list[str]:
    """``all`` (or blank) selects every aggregate total, in catalog order."""
    text = spec.strip()
    known = [node.id for node in _AGGREGATE_TARGETS]
    if not text or text.lower() == "all":
        return list(known)
    selected: list[str] = []
    seen: set[str] = set()
    label = ", ".join(known)
    allowed = set(known)
    for part in text.split(","):
        code = part.strip().upper()
        if not code or code in seen:
            continue
        if code not in allowed:
            raise ValueError(f"Unknown aggregate {code!r}. Known aggregates: {label}")
        seen.add(code)
        selected.append(code)
    if not selected:
        raise ValueError("No aggregates selected")
    return selected


def aggregates_for_areas(area_codes: list[str] | tuple[str, ...]) -> list[str]:
    """Aggregate totals whose every child load area appears in ``area_codes``."""
    allowed = frozenset(area_codes)
    return [node.id for node in _AGGREGATE_TARGETS if frozenset(node.load_areas) <= allowed]


def model_area_for_aggregate(aggregate_id: str) -> LoadArea:
    """Synthetic area so an aggregate total can use the load-area training path."""
    node = _require_aggregate(aggregate_id)
    children = [BY_CODE[code] for code in node.load_areas]
    lat = sum(area.lat for area in children) / len(children)
    lon = sum(area.lon for area in children) / len(children)
    if node.level == "zone":
        zone = node.id
        region = children[0].mkt_region
    elif node.level == "mkt_region":
        zone = node.id
        region = node.id
    else:
        zone = "PJM"
        region = "PJM"
    return LoadArea(node.id, zone, region, f"{node.id} total", lat, lon)


def resolve_model_area(code: str) -> LoadArea:
    """Load area, or an aggregate total trained on the same OpenSTEF path."""
    key = code.strip().upper()
    if key in BY_CODE:
        return BY_CODE[key]
    if key in _AGGREGATE_IDS:
        return model_area_for_aggregate(key)
    raise ValueError(f"Unknown load area {code!r}")


def _require_aggregate(aggregate_id: str) -> ForecastNode:
    key = aggregate_id.strip().upper()
    node = _AGGREGATE_IDS.get(key)
    if node is None:
        known = ", ".join(item.id for item in _AGGREGATE_TARGETS)
        raise ValueError(f"Unknown aggregate {aggregate_id!r}. Known aggregates: {known}")
    return node


def _find_node(node_id: str, level: str | None) -> ForecastNode:
    key = node_id.strip().upper()
    matches = [
        node
        for node in _FORECAST_NODES
        if node.id == key and (level is None or node.level == level)
    ]
    if not matches:
        raise ValueError(f"Unknown forecast node {node_id!r}")
    if level is not None:
        if len(matches) != 1:
            raise ValueError(f"Forecast node {node_id!r} level {level!r} is ambiguous")
        return matches[0]
    leaves = {node.load_areas for node in matches}
    if len(leaves) > 1:
        raise ValueError(f"Forecast node {node_id!r} needs a level")
    return matches[0]


def _as_eastern(series: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(series)
    if stamps.dt.tz is None:
        return stamps.dt.tz_localize(EASTERN)
    return stamps.dt.tz_convert(EASTERN)


def _prepare_joined(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in _JOINED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing columns: {missing}")
    work = frame.loc[:, list(_JOINED_COLUMNS)].copy()
    work["timestamp"] = _as_eastern(work["timestamp"])
    work["timestamp_utc"] = pd.to_datetime(work["timestamp_utc"], utc=True)
    work["load_area"] = work["load_area"].astype(str).str.strip().str.upper()
    work["zone"] = work["zone"].astype(str).str.strip().str.upper()
    work["mkt_region"] = work["mkt_region"].astype(str).str.strip().str.upper()
    work["load"] = pd.to_numeric(work["load"], errors="coerce")
    for column in WEATHER_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    return work


def _children_frame(frame: pd.DataFrame, node: ForecastNode) -> pd.DataFrame:
    work = _prepare_joined(frame)
    work = work.loc[work["load_area"].isin(node.load_areas)].copy()
    present = set(work["load_area"])
    missing = [code for code in node.load_areas if code not in present]
    if missing:
        raise ValueError(f"{node.id} is missing load areas: {missing}")
    work["zone"] = work["load_area"].map(lambda code: BY_CODE[code].zone)
    work["mkt_region"] = work["load_area"].map(lambda code: BY_CODE[code].mkt_region)
    if work.duplicated(subset=["timestamp_utc", "load_area"]).any():
        raise ValueError(f"{node.id} has duplicate UTC hours")
    return work


def _rollup_level(node: ForecastNode) -> RollupLevel:
    if node.level == "zone":
        return "zone"
    if node.level == "mkt_region":
        return "mkt_region"
    if node.level == "pjm":
        return "pjm"
    raise ValueError(f"{node.id} is not an aggregate total")


def historical_load_weights(
    frame: pd.DataFrame,
    load_areas: tuple[str, ...] | list[str],
    *,
    before: pd.Timestamp | None = None,
) -> pd.Series:
    """Mean load per child. Hours at or after ``before`` are ignored.

    ``before`` is the first hour of the forecast horizon. Null load is ignored,
    so a horizon that has already blanked load does not move the weights.
    """
    work = frame.loc[frame["load_area"].isin(load_areas)].copy()
    work["timestamp"] = _as_eastern(work["timestamp"])
    work["load"] = pd.to_numeric(work["load"], errors="coerce")
    known = work.loc[work["load"].notna()]
    if before is not None:
        known = known.loc[known["timestamp"] < as_eastern(before)]
    means = known.groupby("load_area", sort=False)["load"].mean()
    missing = [code for code in load_areas if code not in means.index]
    if missing:
        raise ValueError(f"no historical load for {missing}")
    weights = means.reindex(list(load_areas)).astype(float)
    if float(weights.sum()) <= 0:
        raise ValueError("historical load weights sum to zero")
    return weights


def load_weighted_weather(frame: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Load-weighted mean of ``WEATHER_COLUMNS`` at each UTC hour.

    Weights are fixed scalars. A child missing from an hour is left out of
    that hour and the remaining weights are renormalized.
    """
    work = frame.loc[:, ["timestamp_utc", "load_area", *WEATHER_COLUMNS]].copy()
    work["weight"] = work["load_area"].map(weights).astype(float)
    stamps = work["timestamp_utc"].drop_duplicates().sort_values()
    out = stamps.to_frame(name="timestamp_utc").reset_index(drop=True)
    for column in WEATHER_COLUMNS:
        values = pd.to_numeric(work[column], errors="coerce")
        mask = values.notna() & work["weight"].notna() & work["weight"].gt(0)
        selected = work.loc[mask, ["timestamp_utc", "weight"]].copy()
        selected["product"] = values.loc[mask].to_numpy(dtype=float) * selected["weight"].to_numpy()
        grouped = selected.groupby("timestamp_utc", sort=True)
        averaged = (grouped["product"].sum() / grouped["weight"].sum()).rename(column)
        out = out.merge(averaged, left_on="timestamp_utc", right_index=True, how="left")
    return out


def aggregate_training_frame(
    frame: pd.DataFrame,
    aggregate_id: str,
    *,
    before: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Rolled-up load and load-weighted weather for one aggregate total.

    Load uses :func:`rollup_load`. Hours whose load is entirely null stay in
    the frame with null load so a horizon can keep its weather. Weights do
    not use those hours, and they do not use hours at or after ``before``.
    """
    node = _require_aggregate(aggregate_id)
    children = _children_frame(frame, node)
    weights = historical_load_weights(children, node.load_areas, before=before)
    weather = load_weighted_weather(children, weights)
    rolled = rollup_load(children, _rollup_level(node))
    if rolled["timestamp_utc"].duplicated().any():
        raise ValueError(f"{node.id} rollup produced duplicate hours")
    merged = weather.merge(rolled.loc[:, ["timestamp_utc", "load"]], on="timestamp_utc", how="left")
    merged["timestamp"] = merged["timestamp_utc"].dt.tz_convert(EASTERN)
    columns = ["timestamp", "timestamp_utc", "load", *WEATHER_COLUMNS]
    return merged.loc[:, columns].sort_values("timestamp_utc").reset_index(drop=True)


def prepare_aggregate_frame(
    frame: pd.DataFrame,
    aggregate_id: str,
    *,
    before: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """OpenSTEF frame for one aggregate: eastern index, load, and weather."""
    long = aggregate_training_frame(frame, aggregate_id, before=before)
    if long["timestamp"].duplicated().any():
        raise ValueError(f"{aggregate_id} has duplicate timestamps")
    prepared = long.loc[:, ["timestamp", "load", *WEATHER_COLUMNS]].set_index("timestamp")
    prepared.index.name = "timestamp"
    for column in ("load", *WEATHER_COLUMNS):
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return prepared.sort_index()


def _clamp_quantiles(
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.asarray(p50, dtype=float)
    low = np.asarray(p10, dtype=float)
    high = np.asarray(p90, dtype=float)
    floor = np.maximum(median, 0.0)
    low = np.minimum(np.maximum(low, 0.0), floor)
    high = np.maximum(high, floor)
    return low, median, high


def _quantile_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if "p10" not in frame.columns and "quantile_P10" in frame.columns:
        rename["quantile_P10"] = "p10"
    if "p50" not in frame.columns and "quantile_P50" in frame.columns:
        rename["quantile_P50"] = "p50"
    if "p90" not in frame.columns and "quantile_P90" in frame.columns:
        rename["quantile_P90"] = "p90"
    work = frame.rename(columns=rename)
    required = ("timestamp", "p10", "p50", "p90")
    missing = [column for column in required if column not in work.columns]
    if missing:
        raise ValueError(f"forecast is missing columns: {missing}")
    work = work.copy()
    work["timestamp"] = _as_eastern(work["timestamp"])
    for column in ("p10", "p50", "p90"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    return work


def _child_forecasts(frame: pd.DataFrame, node: ForecastNode) -> pd.DataFrame:
    if "load_area" not in frame.columns:
        raise ValueError("forecast is missing columns: ['load_area']")
    work = _quantile_columns(frame)
    work["load_area"] = work["load_area"].astype(str).str.strip().str.upper()
    missing = [code for code in node.load_areas if code not in set(work["load_area"])]
    if missing:
        raise ValueError(f"{node.id} is missing forecasts for {missing}")
    leaves = work.loc[work["load_area"].isin(node.load_areas)].copy()
    if leaves.duplicated(subset=["load_area", "timestamp"]).any():
        raise ValueError(f"{node.id} forecast has duplicate timestamps")
    if leaves[["p10", "p50", "p90"]].isna().any().any():
        raise ValueError(f"{node.id} forecast has missing quantiles")
    stamps: pd.DatetimeIndex | None = None
    for _, part in leaves.groupby("load_area", sort=True):
        current = pd.DatetimeIndex(part["timestamp"].sort_values())
        if stamps is None:
            stamps = current
        elif not stamps.equals(current):
            raise ValueError(f"{node.id} load-area forecasts do not share timestamps")
    return leaves


def _aggregate_slice(frame: pd.DataFrame | None, node_id: str) -> pd.DataFrame | None:
    if frame is None:
        return None
    source = frame.reset_index(drop=True)
    if "id" in source.columns:
        labels = source["id"].astype(str).str.strip().str.upper()
        source = source.loc[labels == node_id].copy()
        if source.empty:
            return None
    work = _quantile_columns(source)
    if work.duplicated("timestamp").any():
        raise ValueError(f"{node_id} aggregate forecast has duplicate timestamps")
    return work


def reconcile_node(
    node_id: str,
    child_forecasts: pd.DataFrame,
    aggregate_forecast: pd.DataFrame | None = None,
    *,
    level: str | None = None,
) -> pd.DataFrame:
    """Reconcile one catalog node to P10, P50, and P90.

    A single load area is returned unchanged. Every other node sums load-area
    P50s. When that node has more than one load area, the aggregate model's
    band is shifted onto the sum and clamped so ``0 <= P10 <= P50 <= P90``.
    """
    node = _find_node(node_id, level)
    leaves = _child_forecasts(child_forecasts, node)
    columns = ["timestamp", "p10", "p50", "p90"]
    if len(node.load_areas) == 1:
        code = node.load_areas[0]
        one = leaves.loc[leaves["load_area"] == code, columns].sort_values("timestamp")
        return one.reset_index(drop=True)

    aggregate = _aggregate_slice(aggregate_forecast, node.id)
    if aggregate is None:
        raise ValueError(f"{node.id} needs an aggregate forecast")
    summed = (
        leaves.groupby("timestamp", as_index=False, sort=True)["p50"]
        .sum(min_count=1)
        .rename(columns={"p50": "summed_p50"})
    )
    merged = summed.merge(
        aggregate.loc[:, columns],
        on="timestamp",
        how="left",
        validate="one_to_one",
    )
    if merged[["summed_p50", "p10", "p50", "p90"]].isna().any().any():
        raise ValueError(f"{node.id} aggregate forecast does not cover the child timestamps")
    p50 = merged["summed_p50"].to_numpy(dtype=float)
    p10 = p50 + (merged["p10"].to_numpy(dtype=float) - merged["p50"].to_numpy(dtype=float))
    p90 = p50 + (merged["p90"].to_numpy(dtype=float) - merged["p50"].to_numpy(dtype=float))
    p10, p50, p90 = _clamp_quantiles(p10, p50, p90)
    result = merged.loc[:, ["timestamp"]].reset_index(drop=True)
    result["p10"] = p10
    result["p50"] = p50
    result["p90"] = p90
    return result
