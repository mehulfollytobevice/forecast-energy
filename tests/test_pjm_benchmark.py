"""PJM load_frcstd_7_day parsing and snapshot alignment."""

from __future__ import annotations

import pandas as pd

from pjm_forecasting.ingest.pjm_benchmark import attach_benchmark, parse_pjm_forecast


def _raw() -> pd.DataFrame:
    rows = []
    for evaluated, offset in (("2026-10-19T03:17:09", 0.0), ("2026-10-19T09:17:09", 5.0)):
        for area in ("DOMINION", "RTO_COMBINED", "PPL/MIDATL", "UNKNOWN_AREA"):
            for hour in range(3):
                stamp = pd.Timestamp("2026-10-19T04:00", tz="UTC") + pd.Timedelta(hours=hour)
                rows.append(
                    {
                        "evaluated_at_datetime_utc": evaluated,
                        "forecast_datetime_beginning_utc": stamp.strftime("%Y-%m-%dT%H:%M:%S"),
                        "forecast_area": area,
                        "forecast_load_mw": 1000 + hour + offset,
                    }
                )
    return pd.DataFrame(rows)


def test_parse_keeps_latest_vintage_and_maps_nodes():
    parsed = parse_pjm_forecast(_raw())
    assert set(parsed["node_id"]) == {"DOM", "PJM", "PL"}
    dom = parsed.loc[parsed["node_id"] == "DOM"]
    assert len(dom) == 3
    assert dom["load"].tolist() == [1005.0, 1006.0, 1007.0]


def test_attach_benchmark_aligns_to_node_timestamps():
    parsed = parse_pjm_forecast(_raw())
    stamps = pd.date_range("2026-10-19T00:00", periods=4, freq="h", tz="US/Eastern")
    payload = {
        "nodes": [
            {"id": "DOM", "timestamps": [stamp.isoformat() for stamp in stamps]},
            {"id": "OVEC", "timestamps": [stamp.isoformat() for stamp in stamps]},
        ]
    }
    attach_benchmark(payload, parsed)
    assert payload["nodes"][0]["benchmark"] == [1005.0, 1006.0, 1007.0, None]
    assert "benchmark" not in payload["nodes"][1]
    assert payload["benchmark"]["source"] == "PJM"
    assert payload["benchmark"]["issued_at"].startswith("2026-10-19T09:17:09")
