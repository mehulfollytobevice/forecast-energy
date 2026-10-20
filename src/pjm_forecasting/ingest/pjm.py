"""PJM Data Miner client for hourly metered load at load-area grain.

``hrl_load_metered`` is one row per distribution company per hour, plus an
``RTO`` total. This module keeps the company rows and can sum them to a
transmission zone, a market region, or the whole interconnection.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from pjm_forecasting.areas import BY_CODE, CATALOG_CODES
from pjm_forecasting.schema import LOAD_COLUMNS
from pjm_forecasting.utils import EASTERN, month_range, project_root

logger = logging.getLogger(__name__)

PJM_BASE_URL = "https://api.pjm.com/api/v1"
METERED_LOAD_FEED = "hrl_load_metered"
METERED_LOAD_FIELDS = (
    "datetime_beginning_utc,datetime_beginning_ept,mkt_region,zone,load_area,mw,is_verified"
)
PAGE_SIZE = 50_000
# Six calls in a minute succeed; the seventh is rejected with HTTP 429.
RATE_LIMIT_CALLS = 6
RATE_LIMIT_WINDOW_S = 60.0
ROLLUP_LABELS = frozenset({"RTO"})
RollupLevel = Literal["zone", "mkt_region", "pjm"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _labels(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def empty_load_frame() -> pd.DataFrame:
    """Hourly load-area schema with timezone-aware timestamp columns."""
    return pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns, US/Eastern]"),
            "timestamp_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "load_area": pd.Series(dtype="string"),
            "zone": pd.Series(dtype="string"),
            "mkt_region": pd.Series(dtype="string"),
            "load": pd.Series(dtype="float64"),
        }
    )


def _duplicate_total_mask(group: pd.DataFrame) -> pd.Series:
    """True on a row that repeats the sum of the other areas in this zone-hour.

    Two catalog areas that happen to match each other are both kept. A code
    outside the catalog that mirrors the only other area is dropped, and so is
    any row that equals two or more siblings (a zone total published beside
    its companies).
    """
    if len(group) < 2:
        return pd.Series(False, index=group.index)
    others = group["mw"].sum() - group["mw"]
    matches = np.isclose(group["mw"].to_numpy(), others.to_numpy(), rtol=1e-4, atol=0.05)
    if len(group) >= 3:
        return pd.Series(matches, index=group.index)
    outside = ~group["load_area"].isin(CATALOG_CODES)
    return pd.Series(matches & outside.to_numpy(), index=group.index)


def _drop_duplicate_totals(work: pd.DataFrame) -> pd.DataFrame:
    if work.empty:
        return work
    drop: list[Any] = []
    grouped = work.groupby(["timestamp_utc", "zone"], sort=False)
    for _, group in grouped:
        mask = _duplicate_total_mask(group)
        drop.extend(group.index[mask.to_numpy()].tolist())
    if not drop:
        return work
    return work.drop(index=drop)


def to_load_area_hours(df: pd.DataFrame) -> pd.DataFrame:
    """One row per load area per UTC hour.

    ``RTO`` totals are removed. A row that duplicates the other areas in its
    zone-hour is removed. Zone and market region come from the catalog, so
    UGI is PL / MIDATL even when the feed says OTHER. Any remaining load area
    that is not in the catalog raises.
    """
    if df.empty:
        return empty_load_frame()

    work = _normalize_columns(df)
    required = {"zone", "load_area", "mw", "datetime_beginning_utc"}
    missing = required - set(work.columns)
    if missing:
        raise ValueError(f"metered load is missing columns: {sorted(missing)}")

    work = work.assign(
        zone=_labels(work["zone"]),
        load_area=_labels(work["load_area"]),
        mw=pd.to_numeric(work["mw"], errors="coerce"),
        timestamp_utc=pd.to_datetime(work["datetime_beginning_utc"], utc=True),
    ).reset_index(drop=True)
    if work["mw"].isna().any() or work["timestamp_utc"].isna().any():
        raise ValueError("hrl_load_metered contains non-numeric mw or timestamps")

    rollup = work["zone"].isin(ROLLUP_LABELS) | work["load_area"].isin(ROLLUP_LABELS)
    work = work.loc[~rollup].copy()
    if work.empty:
        return empty_load_frame()

    work = _drop_duplicate_totals(work)
    unknown = sorted(set(work["load_area"]) - CATALOG_CODES)
    if unknown:
        raise ValueError(
            "hrl_load_metered returned load areas outside the catalog: "
            f"{unknown}. Refusing to drop them."
        )

    work["zone"] = work["load_area"].map(lambda code: BY_CODE[code].zone)
    work["mkt_region"] = work["load_area"].map(lambda code: BY_CODE[code].mkt_region)
    work = work.rename(columns={"mw": "load"})
    if work.duplicated(subset=["timestamp_utc", "load_area"]).any():
        raise ValueError("duplicate UTC hours for a load area after filtering metered load")

    work["timestamp"] = work["timestamp_utc"].dt.tz_convert(EASTERN)
    work = work.sort_values(["load_area", "timestamp_utc"]).reset_index(drop=True)
    return work.loc[:, list(LOAD_COLUMNS)]


def rollup_load(frame: pd.DataFrame, level: RollupLevel) -> pd.DataFrame:
    """Sum load-area ``load`` to a zone, a market region, or full PJM.

    Full PJM is the sum of the rows in ``frame``, not PJM's published ``RTO``
    series. Weather is not aggregated.
    """
    if level not in ("zone", "mkt_region", "pjm"):
        raise ValueError(f"Unknown rollup level {level!r}")
    if frame.empty:
        columns = ["timestamp", "timestamp_utc", "load"]
        if level == "zone":
            columns = ["timestamp", "timestamp_utc", "zone", "mkt_region", "load"]
        elif level == "mkt_region":
            columns = ["timestamp", "timestamp_utc", "mkt_region", "load"]
        return pd.DataFrame(columns=columns)

    work = frame.dropna(subset=["load"]).copy()
    if level == "zone":
        keys = ["timestamp_utc", "zone", "mkt_region"]
    elif level == "mkt_region":
        keys = ["timestamp_utc", "mkt_region"]
    else:
        keys = ["timestamp_utc"]
    grouped = (
        work.groupby(keys, as_index=False, sort=True)["load"]
        .sum()
        .reset_index(drop=True)
    )
    grouped["timestamp"] = grouped["timestamp_utc"].dt.tz_convert(EASTERN)
    if level == "zone":
        order = ["timestamp", "timestamp_utc", "zone", "mkt_region", "load"]
    elif level == "mkt_region":
        order = ["timestamp", "timestamp_utc", "mkt_region", "load"]
    else:
        order = ["timestamp", "timestamp_utc", "load"]
    return grouped.loc[:, order]


def assert_rollups_match(frame: pd.DataFrame) -> None:
    """Zone sums equal each market region, and the three regions equal full PJM."""
    if frame.empty:
        return
    zones = rollup_load(frame, "zone")
    regions = rollup_load(frame, "mkt_region")
    pjm = rollup_load(frame, "pjm")
    from_zones = (
        zones.groupby(["timestamp_utc", "mkt_region"], as_index=False)["load"]
        .sum()
        .set_index(["timestamp_utc", "mkt_region"])["load"]
        .sort_index()
    )
    region_load = regions.set_index(["timestamp_utc", "mkt_region"])["load"].sort_index()
    if not from_zones.index.equals(region_load.index) or not np.allclose(
        from_zones.to_numpy(), region_load.to_numpy()
    ):
        raise ValueError("zone totals do not sum to the market regions")
    from_regions = regions.groupby("timestamp_utc")["load"].sum().sort_index()
    pjm_load = pjm.set_index("timestamp_utc")["load"].sort_index()
    if not from_regions.index.equals(pjm_load.index) or not np.allclose(
        from_regions.to_numpy(), pjm_load.to_numpy()
    ):
        raise ValueError("market-region totals do not sum to full PJM")


class PJMClient:
    """Thin wrapper around Data Miner ``hrl_load_metered``."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        session: requests.Session | None = None,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("PJM_API_KEY")
        self.session = session or requests.Session()
        self._recent_calls: list[float] = []

    @classmethod
    def from_config(cls, config: dict | None = None, api_key: str | None = None) -> PJMClient:
        del config
        return cls(api_key=api_key)

    def fetch_raw_metered_load(self, start: str, end: str) -> pd.DataFrame:
        """Return load-area rows for each month overlapping ``[start, end]``."""
        frames = [self.fetch_month(month) for month in month_range(start, end)]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def fetch_month(self, month: pd.Timestamp) -> pd.DataFrame:
        """Download one calendar month of every market region, including OTHER."""
        start = pd.Timestamp(month)
        if start.tzinfo is not None:
            start = start.tz_convert(EASTERN).tz_localize(None)
        start = start.normalize().replace(day=1)
        end = (start + pd.offsets.MonthBegin(1)) - pd.Timedelta(hours=1)
        return self._fetch_range(start, end)

    def _fetch_range(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if not self.api_key:
            raise RuntimeError(
                "PJM_API_KEY is not set. Copy .env.example to .env and add the Data Miner key."
            )
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        items: list[dict[str, Any]] = []
        start_row = 1
        while True:
            params = {
                "startRow": start_row,
                "rowCount": PAGE_SIZE,
                "datetime_beginning_ept": (
                    f"{start.strftime('%Y-%m-%d %H:%M')}to{end.strftime('%Y-%m-%d %H:%M')}"
                ),
                "fields": METERED_LOAD_FIELDS,
            }
            payload = self._get(
                f"{PJM_BASE_URL}/{METERED_LOAD_FEED}",
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
        return _normalize_columns(pd.DataFrame(items))

    def _pace(self) -> None:
        """Wait so this key stays within Data Miner's calls-per-minute cap."""
        now = time.monotonic()
        self._recent_calls = [t for t in self._recent_calls if now - t < RATE_LIMIT_WINDOW_S]
        wait = 0.0
        if len(self._recent_calls) >= RATE_LIMIT_CALLS:
            wait = RATE_LIMIT_WINDOW_S - (now - self._recent_calls[0]) + 0.25
            if wait > 0:
                time.sleep(wait)
        self._recent_calls.append(time.monotonic())

    def _get(self, url: str, params: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        last_error: Exception | None = None
        attempt = 0
        rate_hits = 0
        while attempt < 5 and rate_hits < 6:
            self._pace()
            try:
                response = self.session.get(url, params=params, headers=headers, timeout=60)
                if response.status_code == 429:
                    rate_hits += 1
                    time.sleep(RATE_LIMIT_WINDOW_S)
                    continue
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                attempt += 1
                time.sleep(2**attempt)
        raise RuntimeError(f"PJM Data Miner request failed after retries: {url}") from last_error


def default_load_root() -> Path:
    """Parent of ``load_areas/{LOAD_AREA}/YYYY-MM.parquet``."""
    return project_root() / "data" / "raw" / "pjm"
