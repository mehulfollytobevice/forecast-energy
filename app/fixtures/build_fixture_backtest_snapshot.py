"""Write ``backtest_snapshot.json`` for local dashboard dev.

Run from the repo root::

    PYTHONPATH=src python app/fixtures/build_fixture_backtest_snapshot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pjm_forecasting.areas import LOAD_AREAS
from pjm_forecasting.backtest_snapshot import build_backtest_snapshot

OUT = Path(__file__).with_name("backtest_snapshot.json")
HOURS = 7 * 24  # one week (compact fixture; real snapshots use the full window)
LEAD_HOURS = 1


def _base_mw(code: str) -> float:
    return 400.0 + (hash(code) % 2800)


def _build() -> dict[str, object]:
    start = pd.Timestamp("2026-01-01T00:00:00", tz="US/Eastern")
    generated = pd.Timestamp("2026-01-15T12:00:00Z")
    index = pd.date_range(start, periods=HOURS, freq="h", tz="US/Eastern")
    area_frames: dict[str, pd.DataFrame] = {}
    for area in LOAD_AREAS:
        base = _base_mw(area.code)
        phase = hash(area.code) % 7
        rows: list[dict[str, float | pd.Timestamp]] = []
        for hour, stamp in enumerate(index):
            actual = base + 90.0 * np.sin(hour / 8.0 + phase) + 15.0 * np.sin(hour / 3.0)
            err = 25.0 * np.sin(hour / 11.0 + phase) + (hash(f"{area.code}:{hour}") % 21 - 10)
            p50 = actual + err
            width = 80.0 + 0.02 * base
            rows.append(
                {
                    "timestamp": stamp,
                    "actual": float(actual),
                    "p10": float(p50 - width),
                    "p50": float(p50),
                    "p90": float(p50 + width * 1.2),
                }
            )
        area_frames[area.code] = pd.DataFrame(rows)
    payload = build_backtest_snapshot(
        area_frames,
        lead_hours=LEAD_HOURS,
        generated_at=generated,
    )
    _assert_contract(payload)
    return payload


def _assert_contract(payload: dict[str, object]) -> None:
    nodes = payload["nodes"]
    assert isinstance(nodes, list)
    by_id = {(n["id"], n["level"]): n for n in nodes}
    areas = [n for n in nodes if n["level"] == "load_area"]
    assert len(areas) == len(LOAD_AREAS)
    for node in nodes:
        n = len(node["timestamps"])
        assert n == HOURS
        assert len(node["actuals"]) == len(node["p10"]) == len(node["p50"]) == len(node["p90"]) == n
        assert "metrics" in node
        assert node["metrics"]["n"] == float(HOURS)
    pjm = by_id[("PJM", "pjm")]
    for i in range(HOURS):
        total = sum(a["p50"][i] for a in areas)
        assert abs(pjm["p50"][i] - total) < 1e-5


if __name__ == "__main__":
    data = _build()
    # Compact JSON keeps the committed fixture small enough for git.
    OUT.write_text(json.dumps(data, allow_nan=False, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({len(data['nodes'])} nodes, {HOURS} hours)")
