"""Near-real-time PJM load to fill the metered-load lag.

``hrl_load_metered`` publishes about 10 days late. OpenSTEF needs the last
14 days of load for lags, so this module pulls ``inst_load`` (5-minute) or
``hrl_load_prelim`` (daily), maps zone/region totals down to load areas using
trailing metered shares, and merges so metered always wins.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import requests

from pjm_forecasting.areas import BY_CODE, LOAD_AREAS, CATALOG_CODES
from pjm_forecasting.ingest.pjm import PJM_BASE_URL, PAGE_SIZE, PJMClient
from pjm_forecasting.schema import LOAD_COLUMNS
from pjm_forecasting.utils import EASTERN, ensure_dir

logger = logging.getLogger(__name__)

INST_LOAD_FEED = "inst_load"
PRELIM_LOAD_FEED = "hrl_load_prelim"

# Catalog zone -> proxy feed label(s) tried in order.
ZONE_PROXY_ALIASES: dict[str, tuple[str, ...]] = {
    "AE": ("AE", "AE/MIDATL"),
    "AEP": ("AEP",),
    "AP": ("AP",),
    "ATSI": ("ATSI", "FE"),
    "BC": ("BC", "BG&E", "BG&E/MIDATL", "BGE"),
    "CE": ("CE", "COMED", "NI"),
    "DAY": ("DAY", "DAYTON"),
    "DEOK": ("DEOK",),
    "DOM": ("DOM", "DOMINION"),
    "DPL": ("DPL", "DP&L", "DP&L/MIDATL"),
    "DUQ": ("DUQ", "DUQUESNE"),
    "EKPC": ("EKPC",),
    "JC": ("JC", "JCP&L", "JCP&L/MIDATL"),
    "ME": ("ME", "METED", "METED/MIDATL"),
    "OVEC": ("OVEC",),
    "PE": ("PE", "PECO", "PECO/MIDATL"),
    "PEP": ("PEP", "PEPCO", "PEPCO/MIDATL"),
    "PL": ("PL", "PPL", "PPL/MIDATL"),
    "PN": ("PN", "PENELEC", "PENELEC/MIDATL"),
    "PS": ("PS", "PSEG", "PSE&G", "PSE&G/MIDATL"),
    "RECO": ("RECO", "RECO/MIDATL"),
}

# Zones that share one proxy total and must be split by metered share.
SPLIT_ZONES: dict[str, tuple[str, ...]] = {
    "AE": ("AECO", "VMEU"),
    "AEP": ("AEPAPT", "AEPIMP", "AEPKPT", "AEPOPT"),
    "ATSI": ("OE", "PAPWR"),
    "DPL": ("DPLCO", "EASTON"),
    "PEP": ("PEPCO", "SMECO"),
    "PL": ("PLCO", "UGI"),
}

MIDATL_AREAS = tuple(area.code for area in LOAD_AREAS if area.mkt_region == "MIDATL")
PRELIM_AREAS = frozenset({"AEP", "AP", "ATSI", "DAY", "DEOK", "DOM", "DUQ", "EKPC", "MIDATL", "NI"})


def _labels(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(column).strip().lower() for column in out.columns]
    return out


def hour_of_day_shares(
    metered: pd.DataFrame,
    *,
    lookback_days: int = 28,
) -> pd.DataFrame:
    """Share of each load area inside its zone (and MIDATL) by hour-of-day."""
    work = metered.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    if work["timestamp"].dt.tz is None:
        work["timestamp"] = work["timestamp"].dt.tz_localize(EASTERN)
    else:
        work["timestamp"] = work["timestamp"].dt.tz_convert(EASTERN)
    work["load_area"] = _labels(work["load_area"])
    work["load"] = pd.to_numeric(work["load"], errors="coerce")
    cutoff = work["timestamp"].max() - pd.Timedelta(days=lookback_days)
    work = work.loc[work["timestamp"] >= cutoff].dropna(subset=["load"])
    if work.empty:
        raise ValueError("no metered load in the lookback window for shares")
    work["zone"] = work["load_area"].map(lambda code: BY_CODE[code].zone)
    work["mkt_region"] = work["load_area"].map(lambda code: BY_CODE[code].mkt_region)
    work["hour"] = work["timestamp"].dt.hour
    rows: list[dict[str, Any]] = []
    for zone, areas in SPLIT_ZONES.items():
        piece = work.loc[work["load_area"].isin(areas)]
        if piece.empty:
            continue
        totals = piece.groupby(["timestamp", "hour"], as_index=False)["load"].sum()
        totals = totals.rename(columns={"load": "zone_load"})
        merged = piece.merge(totals, on=["timestamp", "hour"], how="left")
        merged = merged.loc[merged["zone_load"] > 0]
        shares = (
            merged.assign(share=merged["load"] / merged["zone_load"])
            .groupby(["load_area", "hour"], as_index=False)["share"]
            .mean()
        )
        for _, row in shares.iterrows():
            rows.append(
                {
                    "group": zone,
                    "load_area": row["load_area"],
                    "hour": int(row["hour"]),
                    "share": float(row["share"]),
                }
            )
    midatl = work.loc[work["load_area"].isin(MIDATL_AREAS)]
    if not midatl.empty:
        totals = midatl.groupby(["timestamp", "hour"], as_index=False)["load"].sum()
        totals = totals.rename(columns={"load": "region_load"})
        merged = midatl.merge(totals, on=["timestamp", "hour"], how="left")
        merged = merged.loc[merged["region_load"] > 0]
        shares = (
            merged.assign(share=merged["load"] / merged["region_load"])
            .groupby(["load_area", "hour"], as_index=False)["share"]
            .mean()
        )
        for _, row in shares.iterrows():
            rows.append(
                {
                    "group": "MIDATL",
                    "load_area": row["load_area"],
                    "hour": int(row["hour"]),
                    "share": float(row["share"]),
                }
            )
    if not rows:
        raise ValueError("could not compute load-area shares from metered load")
    return pd.DataFrame(rows)


def calibration_ratios(
    metered: pd.DataFrame,
    proxy_zone: pd.DataFrame,
    *,
    lookback_days: int = 28,
) -> dict[str, float]:
    """Median metered/proxy ratio per proxy group where both overlap."""
    meter = metered.copy()
    meter["timestamp"] = pd.to_datetime(meter["timestamp"])
    if meter["timestamp"].dt.tz is None:
        meter["timestamp"] = meter["timestamp"].dt.tz_localize(EASTERN)
    else:
        meter["timestamp"] = meter["timestamp"].dt.tz_convert(EASTERN)
    meter["load_area"] = _labels(meter["load_area"])
    meter["zone"] = meter["load_area"].map(lambda code: BY_CODE[code].zone)
    meter["mkt_region"] = meter["load_area"].map(lambda code: BY_CODE[code].mkt_region)
    cutoff = meter["timestamp"].max() - pd.Timedelta(days=lookback_days)
    meter = meter.loc[meter["timestamp"] >= cutoff]
    zone_metered = (
        meter.groupby(["timestamp", "zone"], as_index=False)["load"].sum().rename(columns={"load": "metered"})
    )
    midatl_metered = (
        meter.loc[meter["mkt_region"] == "MIDATL"]
        .groupby("timestamp", as_index=False)["load"]
        .sum()
        .rename(columns={"load": "metered"})
    )
    midatl_metered["zone"] = "MIDATL"
    zone_metered = pd.concat([zone_metered, midatl_metered], ignore_index=True)

    proxy = proxy_zone.copy()
    proxy["timestamp"] = pd.to_datetime(proxy["timestamp"])
    if proxy["timestamp"].dt.tz is None:
        proxy["timestamp"] = proxy["timestamp"].dt.tz_localize(EASTERN)
    else:
        proxy["timestamp"] = proxy["timestamp"].dt.tz_convert(EASTERN)
    proxy["zone"] = _labels(proxy["zone"])
    proxy["proxy"] = pd.to_numeric(proxy["load"], errors="coerce")
    joined = zone_metered.merge(proxy.loc[:, ["timestamp", "zone", "proxy"]], on=["timestamp", "zone"])
    joined = joined.loc[(joined["proxy"] > 0) & joined["metered"].notna()]
    ratios: dict[str, float] = {}
    for zone, piece in joined.groupby("zone"):
        ratios[str(zone)] = float(np.median(piece["metered"] / piece["proxy"]))
    return ratios


def _dedupe_proxy_areas(stacked: pd.DataFrame) -> pd.DataFrame:
    """Prefer zone-grain proxy over MIDATL when both cover the same hour.

    ``inst_load`` often returns both company zones (AE, PE, ...) and the MIDATL
    region total. Expanding both and summing double-counts every Mid-Atlantic
    load area (~2×). Keep the finer zone estimate; use MIDATL only as fallback.
    """
    keys = ["timestamp", "timestamp_utc", "load_area", "zone", "mkt_region"]
    work = stacked.copy()
    work["_priority"] = np.where(work["_proxy_group"].astype(str) == "MIDATL", 1, 0)
    work = work.sort_values(keys + ["_priority"])
    return (
        work.drop_duplicates(subset=keys, keep="first")
        .drop(columns=["_proxy_group", "_priority"])
        .loc[:, list(LOAD_COLUMNS)]
        .sort_values(["load_area", "timestamp_utc"])
        .reset_index(drop=True)
    )


def split_proxy_to_areas(
    proxy_zone: pd.DataFrame,
    shares: pd.DataFrame,
    ratios: dict[str, float],
) -> pd.DataFrame:
    """Expand zone/region proxy totals into load-area rows.

    Zone totals win over MIDATL for the same ``(timestamp, load_area)`` so
    region and company feeds are not summed.
    """
    proxy = proxy_zone.copy()
    proxy["timestamp"] = pd.to_datetime(proxy["timestamp"])
    if proxy["timestamp"].dt.tz is None:
        proxy["timestamp"] = proxy["timestamp"].dt.tz_localize(EASTERN)
    else:
        proxy["timestamp"] = proxy["timestamp"].dt.tz_convert(EASTERN)
    proxy["zone"] = _labels(proxy["zone"])
    proxy["load"] = pd.to_numeric(proxy["load"], errors="coerce")
    proxy["hour"] = proxy["timestamp"].dt.hour
    rows: list[pd.DataFrame] = []
    for zone, piece in proxy.groupby("zone"):
        group = str(zone)
        scale = float(ratios.get(group, 1.0))
        calibrated = piece.copy()
        calibrated["load"] = calibrated["load"] * scale
        if group == "MIDATL":
            areas = MIDATL_AREAS
        elif group in SPLIT_ZONES:
            areas = SPLIT_ZONES[group]
        else:
            # One-company zone: map zone code to the matching load area.
            areas = tuple(area.code for area in LOAD_AREAS if area.zone == group)
            if not areas and group in BY_CODE:
                areas = (group,)
            if len(areas) == 1:
                out = calibrated.copy()
                out["load_area"] = areas[0]
                out["zone"] = BY_CODE[areas[0]].zone
                out["mkt_region"] = BY_CODE[areas[0]].mkt_region
                out["timestamp_utc"] = out["timestamp"].dt.tz_convert("UTC")
                out["_proxy_group"] = group
                rows.append(out.loc[:, [*LOAD_COLUMNS, "_proxy_group"]])
                continue
        share_piece = shares.loc[shares["group"] == group]
        if share_piece.empty:
            # Equal split fallback.
            equal = 1.0 / max(len(areas), 1)
            share_piece = pd.DataFrame(
                [
                    {"group": group, "load_area": code, "hour": hour, "share": equal}
                    for code in areas
                    for hour in range(24)
                ]
            )
        expanded = calibrated.merge(
            share_piece.loc[:, ["load_area", "hour", "share"]],
            on="hour",
            how="left",
        )
        expanded["load"] = expanded["load"] * expanded["share"].fillna(0.0)
        expanded["zone"] = expanded["load_area"].map(lambda code: BY_CODE[code].zone)
        expanded["mkt_region"] = expanded["load_area"].map(lambda code: BY_CODE[code].mkt_region)
        expanded["timestamp_utc"] = expanded["timestamp"].dt.tz_convert("UTC")
        expanded["_proxy_group"] = group
        rows.append(expanded.loc[:, [*LOAD_COLUMNS, "_proxy_group"]])
    if not rows:
        return pd.DataFrame(columns=list(LOAD_COLUMNS))
    stacked = pd.concat(rows, ignore_index=True)
    return _dedupe_proxy_areas(stacked)


def merge_recent_load(metered: pd.DataFrame, proxy: pd.DataFrame) -> pd.DataFrame:
    """Metered wins wherever it exists; proxy fills the rest."""
    if metered.empty and proxy.empty:
        raise ValueError("both metered and proxy load are empty")
    meter = metered.copy() if not metered.empty else pd.DataFrame(columns=list(LOAD_COLUMNS))
    prox = proxy.copy() if not proxy.empty else pd.DataFrame(columns=list(LOAD_COLUMNS))
    for frame in (meter, prox):
        if frame.empty:
            continue
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        if frame["timestamp"].dt.tz is None:
            frame["timestamp"] = frame["timestamp"].dt.tz_localize(EASTERN)
        else:
            frame["timestamp"] = frame["timestamp"].dt.tz_convert(EASTERN)
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
        frame["load_area"] = _labels(frame["load_area"])
        frame["load"] = pd.to_numeric(frame["load"], errors="coerce")
    meter = meter.dropna(subset=["load"])
    prox = prox.dropna(subset=["load"])
    meter = meter.assign(load_source="metered")
    prox = prox.assign(load_source="proxy")
    if meter.empty:
        return prox.loc[:, [*LOAD_COLUMNS, "load_source"]].reset_index(drop=True)
    if prox.empty:
        return meter.loc[:, [*LOAD_COLUMNS, "load_source"]].reset_index(drop=True)
    keys = ["timestamp_utc", "load_area"]
    meter_keys = meter.loc[:, keys].drop_duplicates()
    prox = prox.merge(meter_keys.assign(_has_metered=True), on=keys, how="left")
    prox = prox.loc[prox["_has_metered"].isna()].drop(columns=["_has_metered"])
    stacked = pd.concat(
        [meter.loc[:, [*LOAD_COLUMNS, "load_source"]], prox.loc[:, [*LOAD_COLUMNS, "load_source"]]],
        ignore_index=True,
    )
    return stacked.sort_values(["load_area", "timestamp_utc"]).reset_index(drop=True)


def _fetch_feed(
    client: PJMClient,
    feed: str,
    fields: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    time_field: str = "datetime_beginning_ept",
) -> pd.DataFrame:
    if not client.api_key:
        raise RuntimeError("PJM_API_KEY is not set")
    headers = {"Ocp-Apim-Subscription-Key": client.api_key}
    items: list[dict[str, Any]] = []
    start_row = 1
    start_naive = start.tz_convert(EASTERN).tz_localize(None) if start.tzinfo else start
    end_naive = end.tz_convert(EASTERN).tz_localize(None) if end.tzinfo else end
    while True:
        params = {
            "startRow": start_row,
            "rowCount": PAGE_SIZE,
            time_field: (
                f"{start_naive.strftime('%Y-%m-%d %H:%M')}to{end_naive.strftime('%Y-%m-%d %H:%M')}"
            ),
            "fields": fields,
        }
        payload = client._get(  # noqa: SLF001 — shared Data Miner client
            f"{PJM_BASE_URL}/{feed}",
            params=params,
            headers=headers,
        )
        batch = payload.get("items") or payload.get("Items") or []
        if not batch:
            break
        items.extend(batch)
        total = int(payload.get("totalRows") or payload.get("TotalRows") or 0)
        start_row += len(batch)
        if total and start_row > total:
            break
        if len(batch) < PAGE_SIZE:
            break
    if not items:
        return pd.DataFrame()
    return pd.DataFrame(items)


def fetch_inst_load_hourly(
    client: PJMClient,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Average 5-minute ``inst_load`` to hourly zone/region totals."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize(EASTERN)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize(EASTERN)
    raw = _fetch_feed(
        client,
        INST_LOAD_FEED,
        "datetime_beginning_utc,datetime_beginning_ept,area,instantaneous_load",
        start_ts,
        end_ts,
    )
    if raw.empty:
        return pd.DataFrame(columns=["timestamp", "zone", "load"])
    work = _normalize(raw)
    area_col = "area" if "area" in work.columns else "load_area"
    work["timestamp_utc"] = pd.to_datetime(work["datetime_beginning_utc"], utc=True)
    work["load"] = pd.to_numeric(work["instantaneous_load"], errors="coerce")
    work["area"] = work[area_col].astype(str).str.strip()
    work["hour"] = work["timestamp_utc"].dt.floor("h")
    hourly = (
        work.groupby(["hour", "area"], as_index=False)["load"]
        .mean()
        .rename(columns={"hour": "timestamp_utc", "area": "zone_raw"})
    )
    mapped = _map_proxy_labels(hourly)
    mapped["timestamp"] = mapped["timestamp_utc"].dt.tz_convert(EASTERN)
    return mapped.loc[:, ["timestamp", "zone", "load"]]


def fetch_prelim_load(
    client: PJMClient,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Daily ``hrl_load_prelim`` zone/region totals."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize(EASTERN)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize(EASTERN)
    raw = _fetch_feed(
        client,
        PRELIM_LOAD_FEED,
        "datetime_beginning_utc,datetime_beginning_ept,load_area,prelim_load_avg_hourly",
        start_ts,
        end_ts,
    )
    if raw.empty:
        return pd.DataFrame(columns=["timestamp", "zone", "load"])
    work = _normalize(raw)
    work["timestamp_utc"] = pd.to_datetime(work["datetime_beginning_utc"], utc=True)
    work["load"] = pd.to_numeric(work["prelim_load_avg_hourly"], errors="coerce")
    work["zone_raw"] = work["load_area"].astype(str).str.strip()
    mapped = _map_proxy_labels(work.loc[:, ["timestamp_utc", "zone_raw", "load"]])
    mapped["timestamp"] = mapped["timestamp_utc"].dt.tz_convert(EASTERN)
    return mapped.loc[:, ["timestamp", "zone", "load"]]


def _map_proxy_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Map feed area labels onto catalog zones / MIDATL."""
    alias_to_zone: dict[str, str] = {}
    for zone, aliases in ZONE_PROXY_ALIASES.items():
        for alias in aliases:
            alias_to_zone[alias.upper()] = zone
    alias_to_zone["MIDATL"] = "MIDATL"
    alias_to_zone["PJM MID ATLANTIC REGION"] = "MIDATL"
    alias_to_zone["NI"] = "CE"
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        label = str(row["zone_raw"]).strip().upper()
        zone = alias_to_zone.get(label)
        if zone is None:
            # Try partial contains.
            for alias, mapped in alias_to_zone.items():
                if alias in label or label in alias:
                    zone = mapped
                    break
        if zone is None:
            continue
        rows.append(
            {
                "timestamp_utc": row["timestamp_utc"],
                "zone": zone,
                "load": float(row["load"]) if pd.notna(row["load"]) else np.nan,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["timestamp_utc", "zone", "load"])
    return (
        pd.DataFrame(rows)
        .groupby(["timestamp_utc", "zone"], as_index=False)["load"]
        .mean()
    )


def fetch_recent_proxy(
    client: PJMClient,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    feed: Literal["auto", "inst_load", "hrl_load_prelim"] = "auto",
) -> tuple[pd.DataFrame, str]:
    """Fetch a zone-level proxy, trying inst_load then prelim when ``auto``."""
    if feed in ("auto", "inst_load"):
        try:
            inst = fetch_inst_load_hourly(client, start, end)
            zones = set(inst["zone"]) if not inst.empty else set()
            # Prefer inst_load when it returns more than region-only labels.
            if zones - {"MIDATL", "WEST", "SOUTH", "PJM"}:
                return inst, "inst_load"
            if feed == "inst_load":
                return inst, "inst_load"
            logger.info("inst_load lacked zone grain; falling back to hrl_load_prelim")
        except (RuntimeError, requests.RequestException) as exc:
            if feed == "inst_load":
                raise
            logger.warning("inst_load failed (%s); falling back to hrl_load_prelim", exc)
    prelim = fetch_prelim_load(client, start, end)
    return prelim, "hrl_load_prelim"


def build_recent_load_areas(
    metered: pd.DataFrame,
    proxy_zone: pd.DataFrame,
    *,
    lookback_days: int = 28,
) -> pd.DataFrame:
    """Calibrate, split, and merge proxy load into load-area grain."""
    shares = hour_of_day_shares(metered, lookback_days=lookback_days)
    ratios = calibration_ratios(metered, proxy_zone, lookback_days=lookback_days)
    areas = split_proxy_to_areas(proxy_zone, shares, ratios)
    return merge_recent_load(metered, areas)


def archive_proxy(frame: pd.DataFrame, path: Path | str) -> Path:
    destination = Path(path)
    ensure_dir(destination.parent)
    frame.to_parquet(destination, index=False)
    return destination


def prediction_history_gaps(
    frame: pd.DataFrame,
    *,
    areas: list[str],
    origin: pd.Timestamp,
    history_days: int,
    max_gap_hours: int,
) -> dict[str, int]:
    """Return ``{area: longest_gap_hours}`` for areas exceeding ``max_gap_hours``.

    Empty history for an area is reported as ``history_days * 24`` hours.
    """
    start = origin - pd.Timedelta(days=history_days)
    work = frame.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    if work["timestamp"].dt.tz is None:
        work["timestamp"] = work["timestamp"].dt.tz_localize(EASTERN)
    else:
        work["timestamp"] = work["timestamp"].dt.tz_convert(EASTERN)
    work["load_area"] = _labels(work["load_area"])
    failures: dict[str, int] = {}
    for code in areas:
        piece = work.loc[
            (work["load_area"] == code)
            & (work["timestamp"] >= start)
            & (work["timestamp"] < origin)
            & work["load"].notna()
        ].sort_values("timestamp")
        if piece.empty:
            # Sentinel: no rows at all in the history window.
            failures[code] = -1
            continue
        stamps = pd.DatetimeIndex(piece["timestamp"])
        expected = pd.date_range(start, origin, freq="h", inclusive="left")
        missing = expected.difference(stamps)
        if len(missing) == 0:
            continue
        # Longest consecutive missing block.
        gap_steps = missing.to_series().diff().dt.total_seconds().fillna(3600) / 3600.0
        run = 1
        longest = 1
        for step in gap_steps.iloc[1:]:
            if step == 1:
                run += 1
                longest = max(longest, run)
            else:
                run = 1
        if longest > max_gap_hours:
            failures[code] = int(longest)
    return failures


def assert_prediction_history(
    frame: pd.DataFrame,
    *,
    areas: list[str],
    origin: pd.Timestamp,
    history_days: int,
    max_gap_hours: int,
) -> None:
    """Fail when any area has a gap longer than ``max_gap_hours`` in history."""
    failures = prediction_history_gaps(
        frame,
        areas=areas,
        origin=origin,
        history_days=history_days,
        max_gap_hours=max_gap_hours,
    )
    if not failures:
        return
    code, longest = next(iter(failures.items()))
    if longest < 0:
        raise ValueError(f"{code} has no load in the {history_days}-day prediction history")
    raise ValueError(
        f"{code} prediction history has a {longest}h gap "
        f"(max allowed {max_gap_hours}h)"
    )


# Silence unused-import lint for CATALOG_CODES when only used in docs/tests.
_ = CATALOG_CODES
