"""Aggregate training frames and the flow that fits them. No OpenSTEF fit."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from pjm_forecasting.areas import BY_CODE, LOAD_AREAS
from pjm_forecasting.ingest.pjm import rollup_load
from pjm_forecasting.reconcile import (
    aggregate_targets,
    aggregate_training_frame,
    prepare_aggregate_frame,
    aggregates_for_areas,
    resolve_aggregates,
)
from pjm_forecasting.schema import WEATHER_COLUMNS


def _frame(
    codes: list[str],
    loads: dict[str, list[float]],
    temps: dict[str, list[float]],
    stamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for code in codes:
        area = BY_CODE[code]
        for i, stamp in enumerate(stamps):
            row: dict[str, object] = {
                "timestamp": stamp,
                "timestamp_utc": stamp.tz_convert("UTC"),
                "load_area": code,
                "zone": area.zone,
                "mkt_region": area.mkt_region,
                "load": loads[code][i],
            }
            for column in WEATHER_COLUMNS:
                row[column] = temps[code][i]
            rows.append(row)
    return pd.DataFrame(rows)


def test_aggregate_targets_are_the_split_totals():
    nodes = aggregate_targets()
    assert [node.id for node in nodes] == [
        "AE",
        "DPL",
        "PEP",
        "PL",
        "AEP",
        "ATSI",
        "MIDATL",
        "WEST",
        "PJM",
    ]
    assert all(len(node.load_areas) > 1 for node in nodes)
    aep = next(node for node in nodes if node.id == "AEP")
    assert aep.load_areas == ("AEPAPT", "AEPIMP", "AEPKPT", "AEPOPT")
    single_zones = {
        area.zone
        for area in LOAD_AREAS
        if sum(item.zone == area.zone for item in LOAD_AREAS) == 1
    }
    assert single_zones.isdisjoint({node.id for node in nodes})
    assert "SOUTH" not in {node.id for node in nodes}
    assert resolve_aggregates("pjm, AE") == ["PJM", "AE"]
    assert aggregates_for_areas(["AECO"]) == []
    assert aggregates_for_areas(["AECO", "VMEU"]) == ["AE"]
    with pytest.raises(ValueError, match="SOUTH"):
        resolve_aggregates("SOUTH")


def test_zone_weather_weights_ignore_the_forecast_horizon():
    stamps = pd.date_range("2024-06-01", periods=3, freq="h", tz="US/Eastern")
    codes = ["AECO", "VMEU"]
    loads = {"AECO": [10.0, 10.0, 5000.0], "VMEU": [30.0, 30.0, 9000.0]}
    temps = {"AECO": [8.0, 8.0, 0.0], "VMEU": [8.0, 8.0, 100.0]}
    frame = _frame(codes, loads, temps, stamps)
    result = aggregate_training_frame(frame, "AE", before=stamps[2])

    horizon = result.loc[result["timestamp"] == stamps[2]].iloc[0]
    assert horizon["load"] == pytest.approx(14000.0)
    for column in WEATHER_COLUMNS:
        assert horizon[column] == pytest.approx(75.0)
    assert str(result["timestamp"].dt.tz) == "US/Eastern"

    rolled = rollup_load(frame, "zone")
    rolled = rolled.loc[rolled["zone"] == "AE", ["timestamp", "load"]]
    merged = result.merge(rolled, on="timestamp", suffixes=("_agg", "_rollup"))
    assert merged["load_agg"].tolist() == pytest.approx(merged["load_rollup"].tolist())

    blank = loads | {"AECO": [10.0, 10.0, np.nan], "VMEU": [30.0, 30.0, np.nan]}
    blanked = aggregate_training_frame(_frame(codes, blank, temps, stamps), "AE")
    future = blanked.loc[blanked["timestamp"] == stamps[2]].iloc[0]
    assert pd.isna(future["load"])
    assert future["temperature_2m"] == pytest.approx(75.0)


def test_region_and_pjm_frames_match_rollup_and_weights():
    stamps = pd.date_range("2024-06-01", periods=2, freq="h", tz="US/Eastern")
    codes = [area.code for area in LOAD_AREAS]
    loads = {code: [float(i + 1), float(i + 1)] for i, code in enumerate(codes)}
    temps = {code: [float((i + 1) * 10), float((i + 1) * 10)] for i, code in enumerate(codes)}
    frame = _frame(codes, loads, temps, stamps)

    def _expected(selected: list[str]) -> tuple[float, float]:
        weight = sum(loads[code][0] for code in selected)
        weather = sum(loads[code][0] * temps[code][0] for code in selected) / weight
        return weight, weather

    mid = [area.code for area in LOAD_AREAS if area.mkt_region == "MIDATL"]
    region = aggregate_training_frame(frame, "MIDATL")
    region_load, region_temp = _expected(mid)
    assert region["load"].tolist() == pytest.approx([region_load, region_load])
    assert region["temperature_2m"].tolist() == pytest.approx([region_temp, region_temp])
    rolled_region = rollup_load(frame, "mkt_region")
    mid_load = rolled_region.loc[rolled_region["mkt_region"] == "MIDATL", "load"]
    assert region["load"].tolist() == pytest.approx(mid_load.tolist())

    total = aggregate_training_frame(frame, "PJM")
    pjm_load, pjm_temp = _expected(codes)
    assert total["load"].tolist() == pytest.approx([pjm_load, pjm_load])
    assert total["temperature_2m"].tolist() == pytest.approx([pjm_temp, pjm_temp])
    assert total["load"].tolist() == pytest.approx(rollup_load(frame, "pjm")["load"].tolist())

    prepared = prepare_aggregate_frame(frame, "PJM")
    assert list(prepared.columns) == ["load", *WEATHER_COLUMNS]
    assert prepared["load"].tolist() == pytest.approx(total["load"].tolist())
    with pytest.raises(ValueError, match="SOUTH"):
        aggregate_training_frame(frame, "SOUTH")


def test_prepare_writes_an_openstef_frame(tmp_path):
    from pjm_forecasting.flows.aggregates import execute_prepare_aggregates

    stamps = pd.date_range("2024-06-01", periods=3, freq="h", tz="US/Eastern")
    codes = ["AECO", "VMEU"]
    loads = {"AECO": [10.0, 10.0, np.nan], "VMEU": [30.0, 30.0, np.nan]}
    temps = {"AECO": [0.0, 0.0, 0.0], "VMEU": [100.0, 100.0, 100.0]}
    frame = _frame(codes, loads, temps, stamps)
    source = tmp_path / "joined.parquet"
    frame.to_parquet(source)
    written = execute_prepare_aggregates(
        {"frame_path": str(source)},
        ["AE"],
        str(tmp_path / "out"),
    )
    assert written["areas"] == ["AE"]
    saved = pd.read_parquet(written["frame_paths"][0])
    expected = prepare_aggregate_frame(frame, "AE")
    assert saved["load"].tolist()[:2] == pytest.approx(expected["load"].tolist()[:2])
    assert "temperature_2m" in saved.columns
    assert pd.isna(saved["load"].iloc[-1])


@pytest.mark.openstef
def test_aggregate_model_uses_the_load_area_horizon():
    from pjm_forecasting.reconcile import model_area_for_aggregate
    from pjm_forecasting.settings import load_settings
    from pjm_forecasting.training import model_id_for_area, workflow_config_for_area

    settings = load_settings()
    area = model_area_for_aggregate("MIDATL")
    config = workflow_config_for_area(area, settings, None, run_name="final", reuse=False)
    assert model_id_for_area("aeco") == "pjm-AECO"
    assert model_id_for_area("midatl") == "pjm-MIDATL"
    assert model_id_for_area("PJM") == "pjm-PJM"
    assert config.model_id == "pjm-MIDATL"
    assert [float(quantile) for quantile in config.quantiles] == [0.1, 0.5, 0.9]
    assert config.horizons[0].value == timedelta(hours=48)
    with pytest.raises(ValueError, match="ZZ"):
        model_id_for_area("ZZ")


def test_aggregate_flow_reuses_the_load_area_fit_path():
    from pjm_forecasting.flows import aggregates as aggregate_flow
    from pjm_forecasting.flows import training as training_flow

    expected = [node.id for node in aggregate_targets()]
    calls: list[str] = []

    def quality(frame_path: str, config_path: str | None) -> dict:
        calls.append("quality")
        return {"frame_path": frame_path, "config_path": config_path}

    def prepare(gate: dict, aggregates: list[str], out_dir: str) -> dict:
        calls.append("prepare")
        assert aggregates == expected
        return {
            "areas": aggregates,
            "frame_paths": [f"{out_dir}/{area}/frame.parquet" for area in aggregates],
        }

    def split(prepared: dict, config_path: str | None) -> dict:
        calls.append("split")
        areas = prepared["areas"]
        return {
            "areas": areas,
            "train_paths": [f"{area}/train.parquet" for area in areas],
            "validation_paths": [f"{area}/validation.parquet" for area in areas],
            "test_paths": [f"{area}/test.parquet" for area in areas],
        }

    def fit(area: str, train_path: str, validation_path: str, config_path: str | None) -> dict:
        calls.append(f"fit:{area}")
        return {
            "load_area": area,
            "train_path": train_path,
            "validation_path": validation_path,
            "model_id": f"pjm-{area}",
            "metrics": {"mae": 1.0},
        }

    def refit(fitted: dict, config_path: str | None) -> dict:
        calls.append(f"refit:{fitted['load_area']}")
        return {**fitted, "run_name": "final", "run_id": "run-1"}

    def persist(refit_result: dict, config_path: str | None) -> dict:
        calls.append(f"persist:{refit_result['load_area']}")
        return refit_result

    gate = quality("frame.parquet", None)
    prepared = prepare(gate, expected, "out")
    splits = split(prepared, None)
    fitted = [
        fit(area, train, validation, None)
        for area, train, validation in zip(
            splits["areas"],
            splits["train_paths"],
            splits["validation_paths"],
            strict=True,
        )
    ]
    refitted = [refit(item, None) for item in fitted]
    saved = [persist(item, None) for item in refitted]

    assert calls[:3] == ["quality", "prepare", "split"]
    fitted_calls = [call.split(":", 1)[1] for call in calls if call.startswith("fit:")]
    assert fitted_calls == expected
    assert [item["load_area"] for item in saved] == expected
    assert saved[0]["run_id"] == "run-1"
    assert aggregate_flow.train_aggregate_areas.__name__ == "train_aggregate_areas"
    assert training_flow.execute_fit is not aggregate_flow.execute_prepare_aggregates
