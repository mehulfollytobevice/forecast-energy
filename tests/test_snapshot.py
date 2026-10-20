"""Snapshot JSON from in-memory forecasts. No MLflow and no network."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pjm_forecasting.areas import LOAD_AREAS
from pjm_forecasting.ingest.pjm import rollup_load
from pjm_forecasting.reconcile import aggregate_targets
from pjm_forecasting.snapshot import (
    build_forecast_snapshot,
    default_snapshot_path,
    write_forecast_snapshot,
)
from pjm_forecasting.utils import project_root


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    origin = pd.Timestamp("2024-06-02", tz="US/Eastern")
    actual_index = pd.date_range(
        origin - pd.Timedelta(hours=24),
        periods=24,
        freq="h",
        tz="US/Eastern",
    )
    forecast_index = pd.date_range(origin, periods=3, freq="h", tz="US/Eastern")
    forecast_rows: list[dict[str, object]] = []
    actual_rows: list[dict[str, object]] = []
    for area in LOAD_AREAS:
        if area.code == "AECO":
            mid = 12.0
            metered = 2.0
        elif area.code == "VMEU":
            mid = 8.0
            metered = 5.0
        else:
            mid = 10.0
            metered = 1.0
        for stamp in forecast_index:
            forecast_rows.append(
                {
                    "load_area": area.code,
                    "timestamp": stamp,
                    "p10": mid - 5.0,
                    "p50": mid,
                    "p90": mid + 5.0,
                }
            )
        for stamp in actual_index:
            actual_rows.append(
                {
                    "timestamp": stamp,
                    "timestamp_utc": stamp.tz_convert("UTC"),
                    "load_area": area.code,
                    "zone": area.zone,
                    "mkt_region": area.mkt_region,
                    "load": metered,
                }
            )
    aggregate_rows: list[dict[str, object]] = []
    for node in aggregate_targets():
        low, median, high = (0.0, 1000.0, 1000.0) if node.id == "AE" else (100.0, 200.0, 250.0)
        for stamp in forecast_index:
            aggregate_rows.append(
                {
                    "id": node.id,
                    "timestamp": stamp,
                    "p10": low,
                    "p50": median,
                    "p90": high,
                }
            )
    return (
        pd.DataFrame(forecast_rows),
        pd.DataFrame(aggregate_rows),
        pd.DataFrame(actual_rows),
        origin,
    )


def _node(payload: dict, node_id: str, level: str) -> dict:
    matches = [
        node
        for node in payload["nodes"]
        if node["id"] == node_id and node["level"] == level
    ]
    assert len(matches) == 1
    return matches[0]


def test_default_snapshot_path():
    path = project_root() / "data" / "processed" / "forecast_snapshot.json"
    assert default_snapshot_path() == path


def test_snapshot_reconciles_parents_and_one_child_zones(tmp_path):
    forecasts, aggregates, actuals, origin = _inputs()
    generated = pd.Timestamp("2024-06-02T15:00:00Z")
    payload = build_forecast_snapshot(
        forecasts,
        aggregates,
        actuals,
        origin,
        generated_at=generated,
    )
    zones = {area.zone for area in LOAD_AREAS}
    regions = {area.mkt_region for area in LOAD_AREAS}
    assert set(payload) == {"generated_at", "origin", "timezone", "nodes"}
    assert payload["generated_at"] == generated.isoformat()
    assert payload["origin"] == origin.isoformat()
    assert payload["timezone"] == "US/Eastern"
    assert len(payload["nodes"]) == 1 + len(regions) + len(zones) + len(LOAD_AREAS)

    pjm = _node(payload, "PJM", "pjm")
    areas = [node for node in payload["nodes"] if node["level"] == "load_area"]
    assert pjm["parent_id"] is None
    assert pjm["name"] == "PJM"
    for index in range(len(pjm["p50"])):
        assert pjm["p50"][index] == pytest.approx(sum(node["p50"][index] for node in areas))
    assert len(pjm["timestamps"]) == len(pjm["p10"]) == len(pjm["p50"]) == len(pjm["p90"]) == 3
    assert pjm["timestamps"][0] == payload["origin"]

    ae = _node(payload, "AE", "zone")
    aeco = _node(payload, "AECO", "load_area")
    vmeu = _node(payload, "VMEU", "load_area")
    for index in range(len(ae["p50"])):
        assert ae["p50"][index] == pytest.approx(aeco["p50"][index] + vmeu["p50"][index])
        assert ae["p10"][index] == pytest.approx(0.0)
        assert ae["p90"][index] == pytest.approx(ae["p50"][index])
        assert ae["p10"][index] != pytest.approx(aeco["p10"][index] + vmeu["p10"][index])
    assert ae["parent_id"] == "MIDATL"
    assert aeco["parent_id"] == "AE"
    assert _node(payload, "MIDATL", "mkt_region")["parent_id"] == "PJM"

    dom_zone = _node(payload, "DOM", "zone")
    dom_area = _node(payload, "DOM", "load_area")
    assert dom_zone["parent_id"] == "SOUTH"
    assert dom_area["parent_id"] == "DOM"
    assert _node(payload, "SOUTH", "mkt_region")["parent_id"] == "PJM"
    assert dom_zone["p10"] == pytest.approx(dom_area["p10"])
    assert dom_zone["p50"] == pytest.approx(dom_area["p50"])
    assert dom_zone["p90"] == pytest.approx(dom_area["p90"])
    assert dom_zone["name"] == "DOM"
    assert dom_area["name"] == "DOM"

    assert len(ae["actuals"]) == 24
    assert ae["actual_timestamps"][-1] < payload["origin"]
    last_actual = pd.Timestamp(ae["actual_timestamps"][-1])
    assert last_actual + pd.Timedelta(hours=1) == origin
    rolled = rollup_load(actuals, "zone")
    ae_actual = rolled.loc[rolled["zone"] == "AE", "load"]
    assert ae["actuals"] == pytest.approx(ae_actual.tolist())
    pjm_actual = rollup_load(actuals, "pjm")["load"]
    assert pjm["actuals"] == pytest.approx(pjm_actual.tolist())
    assert dom_zone["actuals"] == pytest.approx(dom_area["actuals"])

    path = write_forecast_snapshot(
        forecasts,
        aggregates,
        actuals,
        origin,
        path=tmp_path / "forecast_snapshot.json",
        generated_at=generated,
    )
    saved = json.loads(path.read_text())
    assert _node(saved, "PJM", "pjm")["p50"] == pytest.approx(pjm["p50"])
    assert saved["nodes"][0]["id"] == "PJM"

    kept = aggregates.loc[aggregates["id"] != "AE"]
    with pytest.raises(ValueError, match="AE"):
        build_forecast_snapshot(forecasts, kept, actuals, origin, generated_at=generated)
