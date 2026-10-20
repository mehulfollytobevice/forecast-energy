"""Recent-load proxy split, calibration, and metered-wins merge."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pjm_forecasting.ingest.pjm_recent import (
    build_recent_load_areas,
    calibration_ratios,
    hour_of_day_shares,
    merge_recent_load,
    split_proxy_to_areas,
)


def _metered() -> pd.DataFrame:
    stamps = pd.date_range("2026-09-01", periods=24 * 30, freq="h", tz="US/Eastern")
    rows = []
    for stamp in stamps:
        for code, zone, share in (
            ("AECO", "AE", 0.7),
            ("VMEU", "AE", 0.3),
            ("PE", "PE", 1.0),
        ):
            rows.append(
                {
                    "timestamp": stamp,
                    "timestamp_utc": stamp.tz_convert("UTC"),
                    "load_area": code,
                    "zone": zone,
                    "mkt_region": "MIDATL",
                    "load": 100.0 * share + stamp.hour,
                }
            )
    return pd.DataFrame(rows)


def _proxy_zone() -> pd.DataFrame:
    stamps = pd.date_range("2026-09-20", periods=24 * 10, freq="h", tz="US/Eastern")
    rows = []
    for stamp in stamps:
        rows.append({"timestamp": stamp, "zone": "AE", "load": 100.0 + stamp.hour})
        rows.append({"timestamp": stamp, "zone": "PE", "load": 100.0 + stamp.hour})
    return pd.DataFrame(rows)


def test_hour_of_day_shares_sum_to_one_inside_split_zone():
    shares = hour_of_day_shares(_metered(), lookback_days=28)
    ae = shares.loc[shares["group"] == "AE"]
    totals = ae.groupby("hour")["share"].sum()
    assert np.allclose(totals, 1.0, atol=1e-6)


def test_calibration_ratios_near_one_when_proxy_matches():
    ratios = calibration_ratios(_metered(), _proxy_zone(), lookback_days=28)
    assert "AE" in ratios
    assert 0.5 < ratios["AE"] < 1.5


def test_split_proxy_to_areas_creates_load_areas():
    shares = hour_of_day_shares(_metered(), lookback_days=28)
    ratios = calibration_ratios(_metered(), _proxy_zone(), lookback_days=28)
    areas = split_proxy_to_areas(_proxy_zone(), shares, ratios)
    assert set(areas["load_area"]) >= {"AECO", "VMEU", "PE"}


def test_merge_recent_load_prefers_metered():
    metered = _metered().iloc[:48].copy()
    metered.loc[metered["load_area"] == "AECO", "load"] = 999.0
    proxy = metered.copy()
    proxy["load"] = 1.0
    # Drop the last day of metered so proxy can fill.
    cutoff = metered["timestamp"].max() - pd.Timedelta(hours=12)
    metered_partial = metered.loc[metered["timestamp"] <= cutoff]
    merged = merge_recent_load(metered_partial, proxy)
    aeco = merged.loc[merged["load_area"] == "AECO"].sort_values("timestamp")
    assert (aeco.loc[aeco["load_source"] == "metered", "load"] == 999.0).all()
    assert (aeco.loc[aeco["load_source"] == "proxy", "load"] == 1.0).all()


def test_build_recent_load_areas_end_to_end():
    merged = build_recent_load_areas(_metered(), _proxy_zone(), lookback_days=28)
    assert {"AECO", "VMEU", "PE"}.issubset(set(merged["load_area"]))
    assert "load_source" in merged.columns


def test_split_prefers_zone_over_midatl_not_sum():
    """AE + MIDATL must not double-count AECO (the live ~2× cliff bug)."""
    metered = _metered()
    stamps = pd.date_range("2026-09-20", periods=24, freq="h", tz="US/Eastern")
    proxy_rows = []
    for stamp in stamps:
        ae = 100.0 + stamp.hour
        pe = 100.0 + stamp.hour
        proxy_rows.extend(
            [
                {"timestamp": stamp, "zone": "AE", "load": ae},
                {"timestamp": stamp, "zone": "PE", "load": pe},
                # Region total ≈ AE + PE; summing with zone would ~2× AECO/PE.
                {"timestamp": stamp, "zone": "MIDATL", "load": ae + pe},
            ]
        )
    proxy = pd.DataFrame(proxy_rows)
    shares = hour_of_day_shares(metered, lookback_days=28)
    ratios = {"AE": 1.0, "PE": 1.0, "MIDATL": 1.0}
    areas = split_proxy_to_areas(proxy, shares, ratios)
    aeco = areas.loc[areas["load_area"] == "AECO"].sort_values("timestamp")
    pe = areas.loc[areas["load_area"] == "PE"].sort_values("timestamp")
    ae_share = (
        shares.loc[(shares["group"] == "AE") & (shares["load_area"] == "AECO")]
        .set_index("hour")["share"]
    )
    expected_aeco = aeco["timestamp"].dt.hour.map(lambda hour: ae_share.loc[hour] * (100.0 + hour))
    assert np.allclose(aeco["load"].to_numpy(), expected_aeco.to_numpy(), rtol=1e-6)
    # Doubling would put AECO near AE+MIDATL*share ≈ 2× zone-only.
    assert float(aeco["load"].mean()) < 1.25 * float(expected_aeco.mean())
    assert np.allclose(pe["load"].to_numpy(), (100.0 + pe["timestamp"].dt.hour).to_numpy(), rtol=1e-6)


def test_split_uses_midatl_when_zone_missing():
    """MIDATL remains a fallback for areas with no zone-level proxy row."""
    metered = _metered()
    stamps = pd.date_range("2026-09-20", periods=3, freq="h", tz="US/Eastern")
    # Only region grain — no AE/PE zone totals.
    proxy = pd.DataFrame(
        [{"timestamp": stamp, "zone": "MIDATL", "load": 200.0 + stamp.hour} for stamp in stamps]
    )
    shares = hour_of_day_shares(metered, lookback_days=28)
    areas = split_proxy_to_areas(proxy, shares, {"MIDATL": 1.0})
    assert {"AECO", "VMEU", "PE"}.issubset(set(areas["load_area"]))
    # Shares within MIDATL still sum to the region total each hour.
    for stamp, piece in areas.groupby("timestamp"):
        midatl_areas = piece.loc[piece["load_area"].isin(["AECO", "VMEU", "PE"])]
        expected = 200.0 + stamp.hour
        assert abs(float(midatl_areas["load"].sum()) - expected) < 1e-6
