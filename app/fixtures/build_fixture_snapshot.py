"""Write ``forecast_snapshot.json`` for local dashboard dev.

Run from the repo root::

    PYTHONPATH=src python app/fixtures/build_fixture_snapshot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pjm_forecasting.areas import BY_CODE, LOAD_AREAS
from pjm_forecasting.reconcile import aggregate_targets
from pjm_forecasting.snapshot import build_forecast_snapshot

OUT = Path(__file__).with_name("forecast_snapshot.json")
HOURS = 48
ACTUAL_HOURS = 24


def _base_mw(code: str) -> float:
    return 400.0 + (hash(code) % 2800)


def _build() -> dict[str, object]:
    origin = pd.Timestamp("2025-06-01T12:00:00", tz="US/Eastern")
    generated = pd.Timestamp("2025-06-01T11:30:00Z")
    forecast_index = pd.date_range(origin, periods=HOURS, freq="h", tz="US/Eastern")
    actual_index = pd.date_range(
        origin - pd.Timedelta(hours=ACTUAL_HOURS),
        periods=ACTUAL_HOURS,
        freq="h",
        tz="US/Eastern",
    )

    forecast_rows: list[dict[str, object]] = []
    actual_rows: list[dict[str, object]] = []
    for area in LOAD_AREAS:
        base = _base_mw(area.code)
        for hour, stamp in enumerate(forecast_index):
            drift = 80.0 * np.sin(hour / 6.0 + hash(area.code) % 7)
            mid = base + drift
            forecast_rows.append(
                {
                    "load_area": area.code,
                    "timestamp": stamp,
                    "p10": mid - 120.0,
                    "p50": mid,
                    "p90": mid + 180.0,
                }
            )
        for hour, stamp in enumerate(actual_index):
            load = base * 0.95 + 60.0 * np.sin(hour / 5.0)
            actual_rows.append(
                {
                    "timestamp": stamp,
                    "timestamp_utc": stamp.tz_convert("UTC"),
                    "load_area": area.code,
                    "zone": area.zone,
                    "mkt_region": area.mkt_region,
                    "load": load,
                }
            )

    child = pd.DataFrame(forecast_rows)
    aggregate_rows: list[dict[str, object]] = []
    for node in aggregate_targets():
        leaves = child.loc[child["load_area"].isin(node.load_areas)]
        totals = leaves.groupby("timestamp", sort=True)["p50"].sum()
        for stamp, total in totals.items():
            aggregate_rows.append(
                {
                    "id": node.id,
                    "timestamp": stamp,
                    "p10": float(total) * 0.94,
                    "p50": float(total),
                    "p90": float(total) * 1.08,
                }
            )

    payload = build_forecast_snapshot(
        child,
        pd.DataFrame(aggregate_rows),
        pd.DataFrame(actual_rows),
        origin,
        generated_at=generated,
        actual_hours=ACTUAL_HOURS,
    )
    _assert_contract(payload)
    return payload


def _assert_contract(payload: dict[str, object]) -> None:
    nodes = payload["nodes"]
    assert isinstance(nodes, list)
    by_id = {(n["id"], n["level"]): n for n in nodes}
    areas = [n for n in nodes if n["level"] == "load_area"]
    for node in nodes:
        n = len(node["timestamps"])
        assert len(node["p10"]) == len(node["p50"]) == len(node["p90"]) == n == HOURS
        for i in range(n):
            assert node["p10"][i] <= node["p50"][i] <= node["p90"][i]
    pjm = by_id[("PJM", "pjm")]
    for i in range(HOURS):
        total = sum(a["p50"][i] for a in areas)
        assert abs(pjm["p50"][i] - total) < 1e-6


if __name__ == "__main__":
    data = _build()
    OUT.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT} ({len(data['nodes'])} nodes, {HOURS} forecast hours)")
