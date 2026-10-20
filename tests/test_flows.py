"""Prefect tasks run in dependency order, and a failed gate stops the rest."""

from __future__ import annotations

import pytest

from pjm_forecasting.flows import backtesting as backtest_flow
from pjm_forecasting.flows import training as training_flow
from pjm_forecasting.quality import QualityError


def test_training_tasks_follow_the_quality_gate(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    def quality(frame_path: str, config_path: str | None) -> dict:
        calls.append("quality")
        return {"frame_path": frame_path, "stats": {"coverage": 1.0}, "config_path": config_path}

    def prepare(gate: dict, areas: list[str], out_dir: str) -> dict:
        calls.append("prepare")
        assert gate["frame_path"] == "frame.parquet"
        paths = [f"{out_dir}/{area}/frame.parquet" for area in areas]
        return {"areas": areas, "frame_paths": paths}

    def split(
        prepared: dict,
        config_path: str | None,
        split_mode: str = "chronological",
    ) -> dict:
        calls.append("split")
        assert split_mode == "chronological"
        return {
            "areas": prepared["areas"],
            "train_paths": ["train.parquet"],
            "validation_paths": ["validation.parquet"],
            "test_paths": ["test.parquet"],
        }

    def fit(area: str, train_path: str, validation_path: str, config_path: str | None) -> dict:
        calls.append("fit")
        return {
            "load_area": area,
            "train_path": train_path,
            "validation_path": validation_path,
            "model_id": "pjm-AECO",
            "metrics": {"mae": 1.0},
        }

    def refit(fitted: dict, config_path: str | None) -> dict:
        calls.append("refit")
        assert calls[-2] == "fit"
        return {**fitted, "run_name": "final", "run_id": "run-1"}

    def persist(refit_result: dict, config_path: str | None) -> dict:
        calls.append("persist")
        return refit_result

    monkeypatch.setattr(training_flow, "execute_quality_gate", quality)
    monkeypatch.setattr(training_flow, "execute_prepare", prepare)
    monkeypatch.setattr(training_flow, "execute_split", split)
    monkeypatch.setattr(training_flow, "execute_fit", fit)
    monkeypatch.setattr(training_flow, "execute_refit", refit)
    monkeypatch.setattr(training_flow, "execute_persist", persist)

    saved = training_flow.train_load_areas(areas="AECO", frame_path="frame.parquet")
    assert calls == ["quality", "prepare", "split", "fit", "refit", "persist"]
    assert saved[0]["load_area"] == "AECO"
    assert saved[0]["run_id"] == "run-1"


def test_failed_quality_gate_blocks_training(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    def quality(frame_path: str, config_path: str | None) -> dict:
        calls.append("quality")
        raise QualityError("duplicates: duplicate rows")

    def prepare(gate: dict, areas: list[str], out_dir: str) -> dict:
        calls.append("prepare")
        return {"areas": areas, "frame_paths": []}

    monkeypatch.setattr(training_flow, "execute_quality_gate", quality)
    monkeypatch.setattr(training_flow, "execute_prepare", prepare)
    with pytest.raises(Exception, match="duplicates"):
        training_flow.train_load_areas(areas="AECO", frame_path="frame.parquet")
    assert calls == ["quality"]


def test_failed_version_gate_blocks_beam(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    def download(
        areas: list[str],
        config_path: str | None,
        *,
        refresh: bool = False,
        skip_download: bool = False,
    ) -> dict:
        calls.append("download")
        assert skip_download is False
        return {"raw_dir": "raw"}

    def build(
        download_result: dict,
        areas: list[str],
        frame_path: str,
        config_path: str | None,
    ) -> dict:
        calls.append("build")
        return {
            "areas": areas,
            "predictor_paths": ["p.parquet"],
            "load_paths": ["l.parquet"],
        }

    def gate(versioned: dict, config_path: str | None) -> dict:
        calls.append("gate")
        raise QualityError("point_in_time: available_at is earlier than run_initialized_at")

    def beam(area: str, load_path: str, predictor_path: str, config_path: str | None) -> dict:
        calls.append("beam")
        return {"load_area": area}

    monkeypatch.setattr(backtest_flow, "execute_download", download)
    monkeypatch.setattr(backtest_flow, "execute_versioned_tables", build)
    monkeypatch.setattr(backtest_flow, "execute_version_gate", gate)
    monkeypatch.setattr(backtest_flow, "execute_backtest", beam)
    with pytest.raises(Exception, match="point_in_time"):
        backtest_flow.backtest_load_areas(areas="AECO", frame_path="frame.parquet")
    assert calls == ["download", "build", "gate"]


def test_skip_download_uses_cache_without_openmeteo(monkeypatch: pytest.MonkeyPatch, tmp_path):
    calls: list[str] = []

    def boom(*args, **kwargs):
        raise AssertionError("download_historical_runs should not run")

    def build(
        download_result: dict,
        areas: list[str],
        frame_path: str,
        config_path: str | None,
    ) -> dict:
        calls.append("build")
        assert download_result["raw_dir"]
        return {
            "areas": areas,
            "predictor_paths": ["p.parquet"],
            "load_paths": ["l.parquet"],
        }

    def gate(versioned: dict, config_path: str | None) -> dict:
        calls.append("gate")
        return versioned

    def beam(area: str, load_path: str, predictor_path: str, config_path: str | None) -> dict:
        calls.append("beam")
        return {"load_area": area, "predictions_path": "out.parquet"}

    monkeypatch.setattr(backtest_flow, "download_historical_runs", boom)
    monkeypatch.setattr(backtest_flow, "execute_versioned_tables", build)
    monkeypatch.setattr(backtest_flow, "execute_version_gate", gate)
    monkeypatch.setattr(backtest_flow, "execute_backtest", beam)

    saved = backtest_flow.backtest_load_areas(
        areas="AECO",
        frame_path="frame.parquet",
        skip_download=True,
    )
    assert calls == ["build", "gate", "beam"]
    assert saved[0]["load_area"] == "AECO"
