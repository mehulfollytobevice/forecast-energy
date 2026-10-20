"""Load-area aggregation and rollups. No test calls Data Miner."""

from __future__ import annotations

import pandas as pd
import pytest

from pjm_forecasting.areas import BY_CODE, LOAD_AREAS
from pjm_forecasting.ingest.pjm import assert_rollups_match, rollup_load, to_load_area_hours


def _row(
    utc: str,
    zone: str,
    load_area: str,
    mw: float,
    *,
    mkt_region: str = "MIDATL",
) -> dict:
    return {
        "datetime_beginning_utc": utc,
        "datetime_beginning_ept": utc,
        "mkt_region": mkt_region,
        "zone": zone,
        "load_area": load_area,
        "mw": mw,
        "is_verified": True,
    }


def test_load_areas_stay_separate_and_rto_is_excluded():
    raw = pd.DataFrame(
        [
            _row("2023-11-05 05:00:00", "DPL", "DPLCO", 5.0),
            _row("2023-11-05 05:00:00", "DPL", "EASTON", 1.0),
            _row("2023-11-05 05:00:00", "RTO", "RTO", 9999.0, mkt_region="RTO"),
        ]
    )
    hourly = to_load_area_hours(raw)
    assert set(hourly["load_area"]) == {"DPLCO", "EASTON"}
    assert "RTO" not in hourly["load_area"].tolist()
    zones = rollup_load(hourly, "zone")
    dpl = zones.loc[zones["zone"] == "DPL", "load"]
    assert dpl.iloc[0] == pytest.approx(6.0)
    assert zones.loc[zones["zone"] == "DPL", "mkt_region"].iloc[0] == "MIDATL"


def test_duplicate_zone_total_is_rejected():
    raw = pd.DataFrame(
        [
            _row("2023-11-05 05:00:00", "AE", "AECO", 10.0),
            _row("2023-11-05 05:00:00", "AE", "AE", 10.0),
            _row("2023-11-05 06:00:00", "AEP", "AEPAPT", 1.0, mkt_region="WEST"),
            _row("2023-11-05 06:00:00", "AEP", "AEPIMP", 2.0, mkt_region="WEST"),
            _row("2023-11-05 06:00:00", "AEP", "AEP", 3.0, mkt_region="WEST"),
        ]
    )
    hourly = to_load_area_hours(raw)
    assert hourly["load_area"].tolist() == ["AECO", "AEPAPT", "AEPIMP"]
    assert hourly["load"].tolist() == [10.0, 1.0, 2.0]


def test_ugi_rolls_into_pl_and_midatl():
    raw = pd.DataFrame(
        [
            _row("2023-11-05 05:00:00", "OTHER", "UGI", 4.0, mkt_region="OTHER"),
            _row("2023-11-05 05:00:00", "PL", "PLCO", 6.0),
        ]
    )
    hourly = to_load_area_hours(raw)
    ugi = hourly.loc[hourly["load_area"] == "UGI"].iloc[0]
    assert ugi["zone"] == "PL"
    assert ugi["mkt_region"] == "MIDATL"
    assert BY_CODE["UGI"].zone == "PL"
    zones = rollup_load(hourly, "zone")
    assert zones.loc[zones["zone"] == "PL", "load"].iloc[0] == pytest.approx(10.0)
    regions = rollup_load(hourly, "mkt_region")
    assert regions.loc[regions["mkt_region"] == "MIDATL", "load"].iloc[0] == pytest.approx(10.0)


def test_zone_and_region_sums_equal_full_pjm():
    raw = pd.DataFrame(
        [
            _row("2023-11-05 05:00:00", "AE", "AECO", 10.0),
            _row("2023-11-05 05:00:00", "BC", "BC", 20.0),
            _row("2023-11-05 05:00:00", "AEP", "AEPOPT", 7.0, mkt_region="WEST"),
            _row("2023-11-05 05:00:00", "DOM", "DOM", 5.0, mkt_region="SOUTH"),
            _row("2023-11-05 05:00:00", "RTO", "RTO", 9999.0, mkt_region="RTO"),
        ]
    )
    hourly = to_load_area_hours(raw)
    assert_rollups_match(hourly)
    regions = rollup_load(hourly, "mkt_region").set_index("mkt_region")["load"]
    assert regions["MIDATL"] == pytest.approx(30.0)
    assert regions["WEST"] == pytest.approx(7.0)
    assert regions["SOUTH"] == pytest.approx(5.0)
    assert rollup_load(hourly, "pjm")["load"].iloc[0] == pytest.approx(42.0)


def test_fall_back_keeps_both_utc_hours():
    raw = pd.DataFrame(
        [
            _row("2023-11-05 05:00:00", "AE", "AECO", 10.0),
            _row("2023-11-05 06:00:00", "AE", "AECO", 12.0),
        ]
    )
    hourly = to_load_area_hours(raw)
    assert len(hourly) == 2
    assert hourly["timestamp_utc"].nunique() == 2
    eastern_hours = hourly["timestamp"].dt.tz_convert("US/Eastern").dt.hour.tolist()
    assert eastern_hours == [1, 1]
    assert hourly["load"].tolist() == [10.0, 12.0]


def test_unknown_load_area_fails_closed():
    raw = pd.DataFrame([_row("2023-11-05 05:00:00", "AE", "ZZ", 1.0)])
    with pytest.raises(ValueError, match="ZZ"):
        to_load_area_hours(raw)


def test_catalog_covers_split_zones_once():
    assert len(LOAD_AREAS) == len({area.code for area in LOAD_AREAS})
    by_zone: dict[str, list[str]] = {}
    for area in LOAD_AREAS:
        by_zone.setdefault(area.zone, []).append(area.code)
    assert by_zone["AEP"] == ["AEPAPT", "AEPIMP", "AEPKPT", "AEPOPT"]
    assert by_zone["ATSI"] == ["OE", "PAPWR"]
    assert by_zone["DPL"] == ["DPLCO", "EASTON"]
    assert by_zone["PEP"] == ["PEPCO", "SMECO"]
    assert by_zone["PL"] == ["PLCO", "UGI"]
    assert "AE" not in {area.code for area in LOAD_AREAS}
    assert by_zone["AE"] == ["AECO", "VMEU"]
    assert "RTO" not in by_zone
    counts: dict[str, int] = {}
    for area in LOAD_AREAS:
        counts[area.mkt_region] = counts.get(area.mkt_region, 0) + 1
    assert counts == {"MIDATL": 15, "WEST": 13, "SOUTH": 1}
