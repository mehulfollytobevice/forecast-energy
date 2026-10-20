"""Parent P10/P50/P90 shift. No test fits a model or calls the network."""

from __future__ import annotations

import pandas as pd
import pytest

from pjm_forecasting.areas import LOAD_AREAS
from pjm_forecasting.reconcile import reconcile_node

STAMPS = pd.date_range("2024-06-01", periods=2, freq="h", tz="US/Eastern")


def _forecasts(specs: dict[str, list[tuple[float, float, float]]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for code, hours in specs.items():
        for stamp, (low, mid, high) in zip(STAMPS, hours, strict=True):
            rows.append(
                {
                    "load_area": code,
                    "timestamp": stamp,
                    "p10": low,
                    "p50": mid,
                    "p90": high,
                }
            )
    return pd.DataFrame(rows)


def _aggregate(p10: list[float], p50: list[float], p90: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": STAMPS, "p10": p10, "p50": p50, "p90": p90})


def _codes(level: str, value: str) -> list[str]:
    if level == "zone":
        return [area.code for area in LOAD_AREAS if area.zone == value]
    if level == "mkt_region":
        return [area.code for area in LOAD_AREAS if area.mkt_region == value]
    return [area.code for area in LOAD_AREAS]


def test_multi_area_zone_shifts_the_aggregate_band():
    children = _codes("zone", "AE")
    assert set(children) == {"AECO", "VMEU"}
    forecasts = _forecasts(
        {
            "AECO": [(1.0, 10.0, 11.0), (1.0, 4.0, 8.0)],
            "VMEU": [(1.0, 20.0, 21.0), (1.0, 6.0, 9.0)],
        }
    )
    aggregate = _aggregate([30.0, 0.0], [40.0, 50.0], [60.0, 55.0])
    result = reconcile_node("AE", forecasts, aggregate, level="zone")

    assert result["p50"].tolist() == pytest.approx([30.0, 10.0])
    assert result["p10"].iloc[0] == pytest.approx(20.0)
    assert result["p90"].iloc[0] == pytest.approx(50.0)
    assert result["p10"].iloc[0] != pytest.approx(2.0)
    raw_low = 10.0 + (0.0 - 50.0)
    assert raw_low < 0
    assert result["p10"].iloc[1] == pytest.approx(0.0)
    assert result["p90"].iloc[1] == pytest.approx(15.0)
    assert result["p10"].iloc[1] <= result["p50"].iloc[1] <= result["p90"].iloc[1]


def test_market_region_shifts_the_aggregate_band():
    children = _codes("mkt_region", "MIDATL")
    assert len(children) > 1
    forecasts = _forecasts({code: [(1.0, 3.0, 9.0), (1.0, 4.0, 9.0)] for code in children})
    aggregate = _aggregate([10.0, 0.0], [20.0, 1000.0], [40.0, 1000.0])
    result = reconcile_node("MIDATL", forecasts, aggregate, level="mkt_region")

    count = len(children)
    assert result["p50"].tolist() == pytest.approx([3.0 * count, 4.0 * count])
    assert result["p10"].iloc[0] == pytest.approx(3.0 * count + (10.0 - 20.0))
    assert result["p90"].iloc[0] == pytest.approx(3.0 * count + (40.0 - 20.0))
    assert result["p10"].iloc[0] != pytest.approx(float(count))
    raw_low = 4.0 * count + (0.0 - 1000.0)
    assert raw_low < 0
    assert result["p10"].iloc[1] == pytest.approx(0.0)
    assert result["p90"].iloc[1] == pytest.approx(4.0 * count)
    assert result["p10"].iloc[1] <= result["p50"].iloc[1] <= result["p90"].iloc[1]


def test_pjm_shifts_the_aggregate_band():
    children = _codes("pjm", "PJM")
    forecasts = _forecasts({code: [(2.0, 2.0, 4.0), (0.5, 1.0, 2.0)] for code in children})
    aggregate = _aggregate([20.0, 0.0], [30.0, 500.0], [50.0, 510.0])
    result = reconcile_node("PJM", forecasts, aggregate, level="pjm")

    count = len(children)
    assert result["p50"].tolist() == pytest.approx([2.0 * count, 1.0 * count])
    assert result["p10"].iloc[0] == pytest.approx(2.0 * count + (20.0 - 30.0))
    assert result["p90"].iloc[0] == pytest.approx(2.0 * count + (50.0 - 30.0))
    assert result["p10"].iloc[0] != pytest.approx(2.0 * count)
    raw_low = 1.0 * count + (0.0 - 500.0)
    assert raw_low < 0
    assert result["p10"].iloc[1] == pytest.approx(0.0)
    assert result["p90"].iloc[1] == pytest.approx(1.0 * count + (510.0 - 500.0))
    assert 0 <= result["p10"].iloc[1] <= result["p50"].iloc[1] <= result["p90"].iloc[1]


def test_single_child_zone_keeps_the_load_area_forecast():
    forecasts = _forecasts({"DOM": [(5.0, 4.0, 4.0), (5.0, 4.0, 4.0)]})
    zone = reconcile_node("DOM", forecasts, level="zone")
    area = reconcile_node("DOM", forecasts, level="load_area")
    south = reconcile_node("SOUTH", forecasts, level="mkt_region")
    assert zone["p10"].tolist() == pytest.approx([5.0, 5.0])
    assert zone["p50"].tolist() == pytest.approx([4.0, 4.0])
    assert zone["p90"].tolist() == pytest.approx([4.0, 4.0])
    zone_band = zone[["p10", "p50", "p90"]].to_numpy()
    assert area[["p10", "p50", "p90"]].to_numpy() == pytest.approx(zone_band)
    assert south[["p10", "p50", "p90"]].to_numpy() == pytest.approx(zone_band)


def test_multi_area_zone_requires_an_aggregate_forecast():
    forecasts = _forecasts(
        {
            "AECO": [(1.0, 10.0, 11.0), (1.0, 4.0, 8.0)],
            "VMEU": [(1.0, 20.0, 21.0), (1.0, 6.0, 9.0)],
        }
    )
    with pytest.raises(ValueError, match="AE"):
        reconcile_node("AE", forecasts, level="zone")
