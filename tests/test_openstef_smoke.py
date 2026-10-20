"""One-area fit, MLflow reload, and a short Beam interval.

Marked so the default offline suite does not fit a model.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from pjm_forecasting.areas import BY_CODE
from pjm_forecasting.backtesting import run_area_backtest
from pjm_forecasting.ingest.openmeteo import build_versioned_load
from pjm_forecasting.preprocessing import prepare_openstef_frame
from pjm_forecasting.quality import validate_joined
from pjm_forecasting.schema import FRAME_COLUMNS, WEATHER_COLUMNS
from pjm_forecasting.settings import load_settings
from pjm_forecasting.splits import split_chronological
from pjm_forecasting.training import load_area_model, predict_quantiles, refit_area, train_area

pytestmark = pytest.mark.openstef


def _hourly_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=24 * 40, freq="h", tz="US/Eastern")
    hours = np.arange(len(index))
    load = 800 + 120 * np.sin(2 * np.pi * hours / 24) + 30 * np.sin(2 * np.pi * hours / 168)
    area = BY_CODE["AECO"]
    frame = pd.DataFrame(
        {
            "timestamp": index,
            "timestamp_utc": index.tz_convert("UTC"),
            "load_area": "AECO",
            "zone": area.zone,
            "mkt_region": area.mkt_region,
            "load": load,
        }
    )
    for column in WEATHER_COLUMNS:
        frame[column] = 5.0 + 3.0 * np.sin(2 * np.pi * hours / 24)
    return frame.loc[:, list(FRAME_COLUMNS)]


def _settings(tmp_path):
    base = load_settings()
    return replace(
        base,
        train_end=pd.Timestamp("2024-01-22", tz="US/Eastern"),
        validation_end=pd.Timestamp("2024-01-25", tz="US/Eastern"),
        n_estimators=50,
        max_day_lags=7,
        predict_history=timedelta(days=7),
        cutoff_history=timedelta(days=7),
        validation_stride=timedelta(hours=48),
        training_context=timedelta(days=16),
        predict_context=timedelta(days=7),
        train_interval=timedelta(days=30),
        predict_interval=timedelta(hours=24),
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_location=tmp_path / "artifacts",
        local_artifacts_path=tmp_path / "staging",
        processed_dir=tmp_path / "processed",
        raw_dir=tmp_path / "raw",
        backtest_start=pd.Timestamp("2024-01-28", tz="US/Eastern"),
        backtest_end=pd.Timestamp("2024-01-30", tz="US/Eastern"),
    )


def test_quality_split_fit_reload_and_short_beam(tmp_path):
    frame = _hourly_frame()
    report = validate_joined(frame, max_gap=pd.Timedelta(hours=6))
    report.raise_if_fatal()
    prepared = prepare_openstef_frame(frame, "AECO")
    settings = _settings(tmp_path)
    split = split_chronological(prepared, settings.train_end, settings.validation_end)
    assert not split.train.empty and not split.validation.empty and not split.test.empty

    trained = train_area("AECO", split.train, split.validation, settings=settings)
    assert trained["run_name"] == "validation"
    assert trained["metrics"]["n"] > 0
    assert trained["metrics"]["mae"] >= 0

    final = refit_area("AECO", split.train, split.validation, settings=settings)
    assert final["run_name"] == "final"
    history = pd.concat([split.train, split.validation, split.test]).sort_index()
    origin = split.test.index.min()
    workflow = load_area_model("AECO", settings=settings)
    forecast = predict_quantiles("AECO", history, origin, workflow=workflow, settings=settings)
    assert {"quantile_P10", "quantile_P50", "quantile_P90"}.issubset(forecast.columns)
    assert len(forecast) > 0

    load = build_versioned_load(frame, "AECO", settings.load_delay)
    predictors = _predictors(frame, settings.publication_delay)
    backtest = run_area_backtest(
        "AECO",
        load,
        predictors,
        settings=settings,
        start=settings.backtest_start,
        end=settings.backtest_end,
        show_progress=False,
        cache_dir=str(tmp_path / "beam"),
    )
    predictions = backtest["predictions"]
    assert isinstance(predictions, pd.DataFrame)
    assert not predictions.empty
    assert "quantile_P50" in predictions.columns
    assert not backtest["lead_time_errors"].empty


def _predictors(frame: pd.DataFrame, delay: pd.Timedelta) -> pd.DataFrame:
    """A few published runs covering the January smoke window."""
    pieces = []
    start = pd.Timestamp("2024-01-20T00:00", tz="UTC")
    for step in range(0, 24 * 12, 12):
        run = start + pd.Timedelta(hours=step)
        window = frame.loc[
            (frame["timestamp_utc"] >= run)
            & (frame["timestamp_utc"] < run + pd.Timedelta(hours=48))
        ]
        piece = window.loc[:, ["timestamp", "load_area", *WEATHER_COLUMNS]].copy()
        piece["run_initialized_at"] = run
        piece["available_at"] = run + delay
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)
