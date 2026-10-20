"""PJM load-area cache updates. No test calls Data Miner."""

from __future__ import annotations

import pandas as pd
import pytest

import update_zone_load as zone_load
from pjm_forecasting.ingest.pjm import PJMClient
from update_zone_load import (
    DEFAULT_REFRESH_DAYS,
    area_month_path,
    main,
    month_needs_refresh,
    parse_args,
    plan_months,
    resolve_areas,
    update_load_areas,
)


def _raw_row(area: str, zone: str, region: str, mw: float) -> dict:
    return {
        "datetime_beginning_utc": "2026-10-01T04:00:00Z",
        "datetime_beginning_ept": "2026-10-01T00:00:00",
        "mkt_region": region,
        "zone": zone,
        "load_area": area,
        "mw": mw,
        "is_verified": True,
    }


def test_plan_skips_settled_months_and_refreshes_the_revision_window(tmp_path):
    areas = ["AECO", "BC"]
    for name in (
        "2018-01.parquet",
        "2025-12.parquet",
        "2026-06.parquet",
        "2026-07.parquet",
        "2026-09.parquet",
    ):
        for area in areas:
            path = area_month_path(tmp_path, area, name.removesuffix(".parquet"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"cached")

    plans = plan_months(
        tmp_path,
        "2018-01-01",
        "2026-10-11",
        areas=areas,
        refresh_days=DEFAULT_REFRESH_DAYS,
        today="2026-10-11",
    )
    by_month = {plan.month: plan.status for plan in plans}
    assert len(plans) == 106
    assert by_month["2018-01"] == "skipped"
    assert by_month["2025-12"] == "skipped"
    assert by_month["2026-06"] == "skipped"
    assert by_month["2026-07"] == "refresh"
    assert by_month["2026-08"] == "missing"
    assert by_month["2026-09"] == "refresh"
    assert by_month["2026-10"] == "missing"
    assert plans[0].path.name == "load_areas"

    assert (
        month_needs_refresh(pd.Timestamp("2026-06-01"), today="2026-10-11", refresh_days=90)
        is False
    )
    assert (
        month_needs_refresh(pd.Timestamp("2026-07-01"), today="2026-10-11", refresh_days=90)
        is True
    )
    gaps_only = plan_months(
        tmp_path,
        "2026-06-01",
        "2026-10-11",
        areas=areas,
        refresh_days=0,
        today="2026-10-11",
    )
    assert [plan.status for plan in gaps_only] == [
        "skipped",
        "skipped",
        "missing",
        "skipped",
        "missing",
    ]


def test_a_missing_area_file_makes_the_month_missing(tmp_path):
    path = area_month_path(tmp_path, "AECO", "2026-09")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"cached")
    plans = plan_months(
        tmp_path,
        "2026-09-01",
        "2026-09-30",
        areas=["AECO", "BC"],
        refresh_days=0,
        today="2026-10-11",
    )
    assert plans[0].status == "missing"


def test_dry_run_does_not_open_a_client_or_create_folders(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise AssertionError("PJMClient should not be constructed on a dry run")

    monkeypatch.setattr(zone_load, "PJMClient", boom)
    results = update_load_areas(
        ["AECO", "BC"],
        "2026-09-01",
        "2026-10-11",
        cache_root=tmp_path,
        refresh_days=90,
        dry_run=True,
        today="2026-10-11",
    )
    assert results
    assert {result.status for result in results} == {"missing"}
    assert not any(tmp_path.iterdir())


def test_download_writes_one_file_per_load_area(monkeypatch, tmp_path):
    for area in ("AECO", "BC"):
        settled = area_month_path(tmp_path, area, "2026-09")
        settled.parent.mkdir(parents=True, exist_ok=True)
        settled.write_bytes(b"keep-sept")
    calls: list[str] = []

    def fake_fetch(client: PJMClient, month: pd.Timestamp) -> pd.DataFrame:
        label = f"{pd.Timestamp(month):%Y-%m}"
        calls.append(label)
        return pd.DataFrame(
            [
                _raw_row("AECO", "AE", "MIDATL", 10.0),
                _raw_row("BC", "BC", "MIDATL", 20.0),
                _raw_row("RTO", "RTO", "RTO", 9999.0),
            ]
        )

    monkeypatch.setattr(zone_load, "fetch_month", fake_fetch)
    results = update_load_areas(
        ["AECO", "BC"],
        "2026-09-01",
        "2026-10-11",
        cache_root=tmp_path,
        refresh_days=0,
        dry_run=False,
        today="2026-10-11",
    )
    assert calls == ["2026-10"]
    written = {result.month: result.status for result in results}
    assert written["2026-09"] == "skipped"
    assert written["2026-10"] == "written"
    assert area_month_path(tmp_path, "AECO", "2026-09").read_bytes() == b"keep-sept"
    october = pd.read_parquet(area_month_path(tmp_path, "AECO", "2026-10"))
    assert october["load_area"].tolist() == ["AECO"]
    assert october["load"].tolist() == [10.0]
    assert october["zone"].tolist() == ["AE"]
    bc = pd.read_parquet(area_month_path(tmp_path, "BC", "2026-10"))
    assert bc["load"].tolist() == [20.0]
    assert not (tmp_path / "load_areas" / "RTO").exists()


def test_cli_defaults_and_wires_dry_run(monkeypatch, tmp_path):
    args = parse_args(["--dry-run", "--areas", "aeco"])
    assert args.refresh_days == DEFAULT_REFRESH_DAYS
    assert args.start is None
    assert resolve_areas(args.areas) == ["AECO"]

    captured: dict = {}

    def fake_update(areas, start, end, *, cache_root, refresh_days, dry_run, today=None):
        captured.update(
            {
                "areas": areas,
                "start": start,
                "end": end,
                "cache_root": cache_root,
                "refresh_days": refresh_days,
                "dry_run": dry_run,
            }
        )
        return []

    monkeypatch.setattr(zone_load, "update_load_areas", fake_update)
    monkeypatch.setattr(zone_load, "eastern_today", lambda now=None: "2026-10-11")
    code = main(["--dry-run", "--cache-root", str(tmp_path), "--refresh-days", "0"])
    assert code == 0
    assert captured["areas"][0] == "AECO"
    assert "UGI" in captured["areas"]
    assert "RTO" not in captured["areas"]
    assert captured["start"] == "2018-01-01"
    assert captured["end"] == "2026-10-11"
    assert captured["refresh_days"] == 0
    assert captured["dry_run"] is True
    assert captured["cache_root"] == tmp_path

    with pytest.raises(SystemExit, match="RTO"):
        main(["--areas", "RTO"])
