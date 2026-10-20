"""One-shot fixture: reconciled 48h forecast tree and a short actuals tail."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from pjm_forecasting.areas import LOAD_AREAS
from pjm_forecasting.reconcile import aggregate_targets
from pjm_forecasting.snapshot import build_forecast_snapshot

OUT = Path(__file__).resolve().parent / "forecast_snapshot.json"
ORIGIN = pd.Timestamp("2024-06-01T00:00:00", tz="US/Eastern")
GENERATED = pd.Timestamp("2024-06-01T12:00:00Z")
FORECAST_HOURS = 48
ACTUAL_HOURS = 24


def _diurnal(stamp: pd.Timestamp, base: float) -> tuple[float, float, float]:
    hour = stamp.hour + stamp.minute / 60.0
    shape = 1.0 + 0.18 * math.sin((hour - 8.0) * math.pi / 12.0)
    p50 = max(base * shape, 1.0)
    spread = max(p50 * 0.08, 0.5)
    p10 = max(p50 - spread, 0.0)
    p90 = p50 + spread
    return p10, p50, p90


def main() -> None:
    forecast_index = pd.date_range(ORIGIN, periods=FORECAST_HOURS, freq="h", tz="US/Eastern")
    actual_index = pd.date_range(
        ORIGIN - pd.Timedelta(hours=ACTUAL_HOURS),
        periods=ACTUAL_HOURS,
        freq="h",
        tz="US/Eastern",
    )
    forecast_rows: list[dict[str, object]] = []
    actual_rows: list[dict[str, object]] = []
    for index, area in enumerate(LOAD_AREAS):
        base = 8.0 + (index % 7) * 1.25 + area.lat * 0.05
        for stamp in forecast_index:
            p10, p50, p90 = _diurnal(stamp, base)
            forecast_rows.append(
                {
                    "load_area": area.code,
                    "timestamp": stamp,
                    "p10": p10,
                    "p50": p50,
                    "p90": p90,
                }
            )
        for stamp in actual_index:
            _, p50, _ = _diurnal(stamp, base * 0.97)
            actual_rows.append(
                {
                    "timestamp": stamp,
                    "timestamp_utc": stamp.tz_convert("UTC"),
                    "load_area": area.code,
                    "zone": area.zone,
                    "mkt_region": area.mkt_region,
                    "load": p50,
                }
            )

    child_forecasts = pd.DataFrame(forecast_rows)
    aggregate_rows: list[dict[str, object]] = []
    for node in aggregate_targets():
        subset = child_forecasts.loc[child_forecasts["load_area"].isin(node.load_areas)]
        summed = (
            subset.groupby("timestamp", as_index=False)["p50"]
            .sum(min_count=1)
            .rename(columns={"p50": "p50"})
        )
        for stamp in forecast_index:
            row = summed.loc[summed["timestamp"] == stamp]
            total = float(row["p50"].iloc[0])
            spread = max(total * 0.06, 0.5)
            aggregate_rows.append(
                {
                    "id": node.id,
                    "timestamp": stamp,
                    "p10": max(total - spread, 0.0),
                    "p50": total,
                    "p90": total + spread,
                }
            )

    payload = build_forecast_snapshot(
        child_forecasts,
        pd.DataFrame(aggregate_rows),
        pd.DataFrame(actual_rows),
        ORIGIN,
        generated_at=GENERATED,
        actual_hours=ACTUAL_HOURS,
    )
    OUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
