"""Smoke test for live snapshot publish with a stub predictor."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pjm_forecasting.snapshot import build_forecast_snapshot
from pjm_forecasting.storage import ObjectStore, StorageConfig


def test_publish_snapshot_to_local_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from pjm_forecasting.flows import live as live_mod
    from pjm_forecasting.flows.live import publish_live_snapshot
    from pjm_forecasting.settings import load_settings

    origin = pd.Timestamp("2026-06-01T12:00", tz="US/Eastern")
    child = _all_area_forecasts(origin)
    aggregates = _all_aggregate_forecasts(origin)
    store = ObjectStore(
        StorageConfig(
            bucket="",
            endpoint_url=None,
            access_key=None,
            secret_key=None,
            region="us-east-1",
            local_root=tmp_path,
        )
    )
    settings = load_settings()
    monkeypatch.setattr(live_mod, "default_snapshot_path", lambda: tmp_path / "forecast_snapshot.json")
    monkeypatch.setattr(live_mod, "attach_pjm_benchmark", lambda *args, **kwargs: None)
    result = publish_live_snapshot(
        child,
        aggregates,
        _all_actuals(origin),
        origin,
        weather_run="20260601T0600Z",
        model_version="test",
        load_through=origin.isoformat(),
        model_sources={"AECO": "forecast_runs"},
        store=store,
        settings=settings,
    )
    assert Path(result["path"]).is_file()
    latest = store.read_json("snapshots/forecast/latest.json")
    assert latest["weather_run"] == "20260601T0600Z"
    assert latest["model_version"] == "test"
    assert len(latest["nodes"]) > 0


def test_already_published_idempotent(tmp_path: Path):
    from pjm_forecasting.flows.live import already_published

    store = ObjectStore(
        StorageConfig(
            bucket="",
            endpoint_url=None,
            access_key=None,
            secret_key=None,
            region="us-east-1",
            local_root=tmp_path,
        )
    )
    store.write_json(
        "snapshots/forecast/latest.json",
        {"weather_run": "r1", "load_through": "t1"},
    )
    assert already_published(store, weather_run="r1", load_through="t1")
    assert not already_published(store, weather_run="r2", load_through="t1")


def test_already_published_missing_latest(tmp_path: Path):
    from pjm_forecasting.flows.live import already_published

    store = ObjectStore(
        StorageConfig(
            bucket="",
            endpoint_url=None,
            access_key=None,
            secret_key=None,
            region="us-east-1",
            local_root=tmp_path,
        )
    )
    assert not already_published(store, weather_run="r1", load_through="t1")


def test_build_forecast_snapshot_smoke():
    origin = pd.Timestamp("2026-06-01T12:00", tz="US/Eastern")
    payload = build_forecast_snapshot(
        _all_area_forecasts(origin),
        _all_aggregate_forecasts(origin),
        _all_actuals(origin),
        origin,
    )
    assert payload["origin"]
    assert len(payload["nodes"]) >= 50


def _all_area_forecasts(origin: pd.Timestamp) -> pd.DataFrame:
    from pjm_forecasting.areas import LOAD_AREAS

    stamps = pd.date_range(origin, periods=48, freq="h")
    rows = []
    for area in LOAD_AREAS:
        for stamp in stamps:
            rows.append(
                {
                    "load_area": area.code,
                    "timestamp": stamp,
                    "p10": 10.0,
                    "p50": 12.0,
                    "p90": 14.0,
                }
            )
    return pd.DataFrame(rows)


def _all_aggregate_forecasts(origin: pd.Timestamp) -> pd.DataFrame:
    from pjm_forecasting.reconcile import aggregate_targets

    stamps = pd.date_range(origin, periods=48, freq="h")
    rows = []
    for node in aggregate_targets():
        for stamp in stamps:
            rows.append(
                {
                    "id": node.id,
                    "timestamp": stamp,
                    "p10": 100.0,
                    "p50": 120.0,
                    "p90": 140.0,
                }
            )
    return pd.DataFrame(rows)


def _all_actuals(origin: pd.Timestamp) -> pd.DataFrame:
    from pjm_forecasting.areas import LOAD_AREAS

    stamps = pd.date_range(origin - pd.Timedelta(hours=24), periods=24, freq="h")
    rows = []
    for stamp in stamps:
        for area in LOAD_AREAS:
            rows.append(
                {
                    "timestamp": stamp,
                    "timestamp_utc": stamp.tz_convert("UTC"),
                    "load_area": area.code,
                    "zone": area.zone,
                    "mkt_region": area.mkt_region,
                    "load": 11.0,
                }
            )
    return pd.DataFrame(rows)
