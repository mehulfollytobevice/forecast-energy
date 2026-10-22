"""Backtest snapshot collapse and parent rollups. No Beam and no network."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pjm_forecasting.backtest_snapshot import (
    BACKTEST_SNAPSHOT_KEY,
    area_frames_from_snapshot,
    build_backtest_snapshot,
    collapse_area_backtest,
    default_backtest_fixture_path,
    default_backtest_snapshot_path,
    merge_area_frames,
    publish_backtest_snapshot,
    resolve_backtest_snapshot,
    select_lead_ahead,
    write_backtest_snapshot,
)
from pjm_forecasting.storage import ObjectStore, StorageConfig
from pjm_forecasting.utils import project_root


def test_default_backtest_snapshot_path() -> None:
    path = project_root() / "data" / "processed" / "backtest_snapshot.json"
    assert default_backtest_snapshot_path() == path


def test_select_lead_ahead_keeps_one_row_per_hour() -> None:
    stamps = pd.date_range("2026-01-01", periods=3, freq="h", tz="US/Eastern")
    rows = []
    for stamp in stamps:
        for lead in (1, 2, 24):
            rows.append(
                {
                    "timestamp": stamp,
                    "available_at": stamp - pd.Timedelta(hours=lead),
                    "quantile_P50": float(lead),
                    "quantile_P10": float(lead) - 1,
                    "quantile_P90": float(lead) + 1,
                }
            )
    selected = select_lead_ahead(pd.DataFrame(rows), lead_hours=1)
    assert len(selected) == 3
    assert (selected["quantile_P50"] == 1.0).all()


def test_collapse_and_parent_sum(tmp_path) -> None:
    stamps = pd.date_range("2026-01-01", periods=4, freq="h", tz="US/Eastern")
    predictions = pd.DataFrame(
        {
            "timestamp": list(stamps) + list(stamps),
            "available_at": list(stamps - pd.Timedelta(hours=1))
            + list(stamps - pd.Timedelta(hours=2)),
            "quantile_P10": [9.0] * 4 + [8.0] * 4,
            "quantile_P50": [10.0] * 4 + [20.0] * 4,
            "quantile_P90": [11.0] * 4 + [22.0] * 4,
        }
    )
    actual = pd.Series([10.5, 10.0, 9.5, 10.0], index=stamps, name="load")
    aeco = collapse_area_backtest(predictions, actual, lead_hours=1)
    assert len(aeco) == 4
    assert list(aeco["p50"]) == [10.0, 10.0, 10.0, 10.0]

    vmeu = aeco.copy()
    vmeu["actual"] = aeco["actual"] + 1.0
    vmeu["p10"] = aeco["p10"] + 1.0
    vmeu["p50"] = aeco["p50"] + 1.0
    vmeu["p90"] = aeco["p90"] + 1.0

    # Fill every catalog load area so parent sums stay defined.
    from pjm_forecasting.areas import LOAD_AREAS

    frames = {area.code: aeco.copy() for area in LOAD_AREAS}
    frames["AECO"] = aeco
    frames["VMEU"] = vmeu

    payload = build_backtest_snapshot(frames, lead_hours=1, generated_at="2026-01-02T00:00:00Z")
    assert payload["lead_hours"] == 1
    assert payload["timezone"] == "US/Eastern"
    by_key = {(n["id"], n["level"]): n for n in payload["nodes"]}
    ae = by_key[("AE", "zone")]
    assert ae["p50"] == [21.0, 21.0, 21.0, 21.0]
    assert ae["actuals"] == pytest.approx([22.0, 21.0, 20.0, 21.0])
    assert "metrics" in ae
    assert ae["metrics"]["n"] == 4.0

    destination = write_backtest_snapshot(
        frames,
        path=tmp_path / "backtest_snapshot.json",
        fill_missing_from_fixture=False,
    )
    assert destination.is_file()


def test_resolve_prefers_real_frames_and_fills_fixture(tmp_path: Path) -> None:
    fixture = json.loads(default_backtest_fixture_path().read_text())
    fixture_frames = area_frames_from_snapshot(fixture)
    assert "AECO" in fixture_frames and "BC" in fixture_frames

    stamps = pd.date_range("2026-01-03", periods=2, freq="h", tz="US/Eastern")
    real = pd.DataFrame(
        {
            "timestamp": stamps,
            "actual": [50.0, 55.0],
            "p10": [40.0, 45.0],
            "p50": [51.0, 56.0],
            "p90": [60.0, 65.0],
        }
    )
    merged = merge_area_frames({"AECO": real}, fixture_frames)
    assert list(merged["AECO"]["p50"]) == [51.0, 56.0]
    assert list(merged["BC"]["p50"]) == list(fixture_frames["BC"]["p50"])

    payload = resolve_backtest_snapshot(
        area_frames={"AECO": real},
        fixture_path=default_backtest_fixture_path(),
        processed_path=tmp_path / "missing.json",
        generated_at="2026-01-04T00:00:00Z",
    )
    by_key = {(n["id"], n["level"]): n for n in payload["nodes"]}
    assert by_key[("AECO", "load_area")]["p50"] == [51.0, 56.0]
    assert by_key[("BC", "load_area")]["p50"] == list(fixture_frames["BC"]["p50"])
    assert len(payload["nodes"]) == len(fixture["nodes"])


def test_publish_backtest_snapshot_uploads_latest(tmp_path: Path) -> None:
    stamps = pd.date_range("2026-01-03", periods=2, freq="h", tz="US/Eastern")
    real = pd.DataFrame(
        {
            "timestamp": stamps,
            "actual": [50.0, 55.0],
            "p10": [40.0, 45.0],
            "p50": [51.0, 56.0],
            "p90": [60.0, 65.0],
        }
    )
    store = ObjectStore(
        StorageConfig(
            bucket="",
            endpoint_url=None,
            access_key=None,
            secret_key=None,
            region="us-east-1",
            local_root=tmp_path / "bucket",
        )
    )
    result = publish_backtest_snapshot(
        {"AECO": real},
        store=store,
        path=tmp_path / "backtest_snapshot.json",
        generated_at="2026-01-04T12:00:00Z",
        fill_missing_from_fixture=True,
    )
    assert result["key"] == BACKTEST_SNAPSHOT_KEY
    assert result["archive_key"] == "snapshots/backtest/20260104T1200Z.json"
    latest = store.read_json(BACKTEST_SNAPSHOT_KEY)
    assert latest["lead_hours"] == 1
    archive = store.read_json(str(result["archive_key"]))
    by_key = {(n["id"], n["level"]): n for n in latest["nodes"]}
    assert by_key[("AECO", "load_area")]["p50"] == [51.0, 56.0]
    assert archive["generated_at"] == latest["generated_at"]
