"""PJM's own load forecast (``load_frcstd_7_day``) as a dashboard benchmark.

The feed keeps only the latest vintage: one hourly 7-day forecast per PJM
forecast area, re-issued several times a day. Fetching it while the live
job runs gives the forecast PJM had published at that moment, which is the
fair comparison for our snapshot.

PJM forecasts zones and regions, not every load area. ``PPL/MIDATL`` covers
the whole PL zone (PLCO + UGI) while ``UGI/MIDATL`` is UGI alone. Split-zone
load areas other than UGI, and OVEC, have no PJM benchmark.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pjm_forecasting.ingest.pjm import PJMClient
from pjm_forecasting.ingest.pjm_recent import _fetch_feed
from pjm_forecasting.utils import EASTERN, ensure_dir

PJM_FORECAST_FEED = "load_frcstd_7_day"
PJM_FORECAST_FIELDS = (
    "evaluated_at_datetime_utc,forecast_datetime_beginning_utc,forecast_area,forecast_load_mw"
)

# PJM forecast_area -> snapshot node ids. One-company zones share an id with
# their load area, so a single entry covers both nodes.
FORECAST_AREA_NODES: dict[str, tuple[str, ...]] = {
    "RTO_COMBINED": ("PJM",),
    "MID_ATLANTIC_REGION": ("MIDATL",),
    "WESTERN_REGION": ("WEST",),
    "SOUTHERN_REGION": ("SOUTH",),
    "AE/MIDATL": ("AE",),
    "BG&E/MIDATL": ("BC",),
    "DP&L/MIDATL": ("DPL",),
    "JCP&L/MIDATL": ("JC",),
    "METED/MIDATL": ("ME",),
    "PECO/MIDATL": ("PE",),
    "PEPCO/MIDATL": ("PEP",),
    "PPL/MIDATL": ("PL",),
    "UGI/MIDATL": ("UGI",),
    "PENELEC/MIDATL": ("PN",),
    "PSE&G/MIDATL": ("PS",),
    "RECO/MIDATL": ("RECO",),
    "AEP": ("AEP",),
    "AP": ("AP",),
    "ATSI": ("ATSI",),
    "COMED": ("CE",),
    "DAYTON": ("DAY",),
    "DEOK": ("DEOK",),
    "DUQUESNE": ("DUQ",),
    "EKPC": ("EKPC",),
    "DOMINION": ("DOM",),
}

BENCHMARK_COLUMNS = ("node_id", "timestamp_utc", "load", "evaluated_at")


def parse_pjm_forecast(raw: pd.DataFrame) -> pd.DataFrame:
    """Map raw feed rows to ``node_id`` grain, keeping the latest vintage."""
    if raw.empty:
        return pd.DataFrame(columns=list(BENCHMARK_COLUMNS))
    work = raw.copy()
    work.columns = [str(column).strip().lower() for column in work.columns]
    work["forecast_area"] = work["forecast_area"].astype(str).str.strip().str.upper()
    work["evaluated_at"] = pd.to_datetime(work["evaluated_at_datetime_utc"], utc=True)
    work["timestamp_utc"] = pd.to_datetime(work["forecast_datetime_beginning_utc"], utc=True)
    work["load"] = pd.to_numeric(work["forecast_load_mw"], errors="coerce")
    work = work.dropna(subset=["load"])
    work = work.sort_values("evaluated_at").drop_duplicates(
        subset=["forecast_area", "timestamp_utc"], keep="last"
    )
    rows: list[pd.DataFrame] = []
    for area, node_ids in FORECAST_AREA_NODES.items():
        piece = work.loc[work["forecast_area"] == area]
        if piece.empty:
            continue
        for node_id in node_ids:
            rows.append(piece.assign(node_id=node_id).loc[:, list(BENCHMARK_COLUMNS)])
    if not rows:
        return pd.DataFrame(columns=list(BENCHMARK_COLUMNS))
    return pd.concat(rows, ignore_index=True).sort_values(["node_id", "timestamp_utc"]).reset_index(
        drop=True
    )


def fetch_pjm_forecast(
    client: PJMClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(raw, parsed)`` PJM forecast rows for hours in ``[start, end]``."""
    start_ts = start.tz_localize(EASTERN) if start.tzinfo is None else start
    end_ts = end.tz_localize(EASTERN) if end.tzinfo is None else end
    raw = _fetch_feed(
        client,
        PJM_FORECAST_FEED,
        PJM_FORECAST_FIELDS,
        start_ts,
        end_ts,
        time_field="forecast_datetime_beginning_ept",
    )
    return raw, parse_pjm_forecast(raw)


def attach_benchmark(payload: dict[str, Any], benchmark: pd.DataFrame) -> dict[str, Any]:
    """Add ``benchmark`` (aligned to ``timestamps``; ``None`` where missing) to nodes."""
    if benchmark.empty:
        return payload
    lookup = {
        (str(row.node_id), pd.Timestamp(row.timestamp_utc)): float(row.load)
        for row in benchmark.itertuples(index=False)
    }
    for node in payload["nodes"]:
        node_id = str(node["id"])
        values = [
            lookup.get((node_id, pd.Timestamp(stamp).tz_convert("UTC")))
            for stamp in node["timestamps"]
        ]
        if any(value is not None for value in values):
            node["benchmark"] = values
    payload["benchmark"] = {
        "source": "PJM",
        "feed": PJM_FORECAST_FEED,
        "issued_at": pd.Timestamp(benchmark["evaluated_at"].max()).isoformat(),
    }
    return payload


def archive_pjm_forecast(raw: pd.DataFrame, raw_dir: Path) -> Path | None:
    """Keep each fetched vintage so the benchmark can be scored later."""
    if raw.empty:
        return None
    issued = pd.to_datetime(raw["evaluated_at_datetime_utc"], utc=True).max()
    destination = raw_dir / "pjm_forecast" / f"{PJM_FORECAST_FEED}_{issued.strftime('%Y%m%dT%H%MZ')}.parquet"
    ensure_dir(destination.parent)
    raw.to_parquet(destination, index=False)
    return destination
