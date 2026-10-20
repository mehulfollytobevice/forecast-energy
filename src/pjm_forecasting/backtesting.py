"""Point-in-time OpenSTEF Beam backtests.

Beam retrains on its own schedule from versioned load and versioned forecast
runs. It does not reload the final MLflow model. Metered load is hidden until
``load_delay`` after the hour, and a forecast run is hidden until
``publication_delay`` after its initialisation.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from openstef_beam.backtesting import BacktestConfig, BacktestPipeline
from openstef_beam.backtesting.backtest_forecaster import BacktestForecasterConfig
from openstef_beam.benchmarking.baselines.openstef4 import OpenSTEF4BacktestForecaster
from openstef_beam.evaluation import EvaluationConfig, EvaluationPipeline, Window
from openstef_beam.evaluation.metric_providers import RMAEProvider
from openstef_core.datasets import VersionedTimeSeriesDataset
from openstef_core.types import Q
from openstef_models.presets import create_forecasting_workflow

from pjm_forecasting.areas import BY_CODE
from pjm_forecasting.schema import WEATHER_COLUMNS
from pjm_forecasting.settings import PipelineSettings, load_settings
from pjm_forecasting.training import score_forecast, workflow_config_for_area
from pjm_forecasting.utils import EASTERN, ensure_dir

ASSUMPTIONS = (
    "Archive weather from 2018 trains the stored model. Beam does not use that "
    "model. It retrains on metered load and ECMWF IFS HRES single runs. A run "
    "initialized at run_initialized_at is treated as published "
    "publication_delay later. Metered load for an hour is treated as known "
    "load_availability_delay after the hour begins. The default backtest starts "
    "in 2026, after the 2024-03-14 start of the IFS HRES single-run archive."
)


def select_forecast_versions(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Latest issued forecast for each valid hour that was already published."""
    if "available_at" not in frame.columns or "timestamp" not in frame.columns:
        raise ValueError("forecast versions need timestamp and available_at")
    moment = pd.Timestamp(as_of)
    available = pd.to_datetime(frame["available_at"])
    if available.dt.tz is None:
        raise ValueError("available_at must be timezone-aware")
    if moment.tzinfo is None:
        moment = moment.tz_localize(available.dt.tz)
    else:
        moment = moment.tz_convert(available.dt.tz)
    visible = frame.loc[available <= moment].copy()
    if visible.empty:
        return visible
    visible = visible.sort_values(["timestamp", "available_at"])
    return visible.groupby("timestamp", as_index=False).tail(1)


def _utc_index(stamps: pd.Series) -> pd.DatetimeIndex:
    parsed = pd.to_datetime(stamps)
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(EASTERN)
    return pd.DatetimeIndex(parsed.dt.tz_convert("UTC"), name="timestamp")


def to_versioned_dataset(
    frame: pd.DataFrame,
    value_columns: list[str],
    sample_interval: timedelta,
) -> VersionedTimeSeriesDataset:
    """Build a versioned dataset. Identifier columns are not passed through."""
    required = ["timestamp", "available_at", *value_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"versioned frame is missing columns: {missing}")
    index = _utc_index(frame["timestamp"])
    available = pd.to_datetime(frame["available_at"], utc=True)
    data = pd.DataFrame(index=index)
    data["available_at"] = pd.Series(available.to_numpy(), index=index, dtype="datetime64[ns, UTC]")
    for column in value_columns:
        data[column] = pd.to_numeric(frame[column], errors="coerce").to_numpy()
    data.index.name = "timestamp"
    return VersionedTimeSeriesDataset.from_dataframe(data, sample_interval)


def error_by_lead_time(predictions: pd.DataFrame, actual: pd.Series) -> pd.DataFrame:
    """Mean absolute P50 error grouped by whole lead hours."""
    if predictions.empty:
        return pd.DataFrame(columns=["lead_hours", "mae", "n"])
    frame = predictions.copy()
    stamps = pd.DatetimeIndex(frame.index)
    if stamps.tz is None:
        stamps = stamps.tz_localize("UTC")
    available = pd.to_datetime(frame["available_at"])
    if available.dt.tz is None:
        available = available.dt.tz_localize("UTC")
    delta = stamps.tz_convert("UTC") - pd.DatetimeIndex(available.dt.tz_convert("UTC"))
    lead = pd.Series(delta, index=frame.index).dt.total_seconds() / 3600.0
    frame["lead_hours"] = np.floor(lead).astype(int)
    observed = actual.copy()
    observed.index = pd.DatetimeIndex(observed.index)
    if observed.index.tz is None:
        observed.index = observed.index.tz_localize("UTC")
    else:
        observed.index = observed.index.tz_convert("UTC")
    frame["actual"] = observed.reindex(stamps.tz_convert("UTC")).to_numpy()
    frame = frame.dropna(subset=["actual", "quantile_P50"])
    if frame.empty:
        return pd.DataFrame(columns=["lead_hours", "mae", "n"])
    frame["abs_error"] = (frame["actual"] - frame["quantile_P50"]).abs()
    grouped = frame.groupby("lead_hours", sort=True)["abs_error"]
    return grouped.agg(mae="mean", n="size").reset_index()


def _beam_metrics(
    predictions: VersionedTimeSeriesDataset,
    ground_truth: VersionedTimeSeriesDataset,
    settings: PipelineSettings,
    window: timedelta,
) -> dict[str, float]:
    quantiles = [Q(value) for value in settings.quantiles]
    pipeline = EvaluationPipeline(
        config=EvaluationConfig(
            windows=[Window(lag=timedelta(0), size=window)],
            lead_times=[],
            available_ats=[],
        ),
        quantiles=quantiles,
        window_metric_providers=[RMAEProvider(quantiles=[Q(0.5)])],
        global_metric_providers=[RMAEProvider(quantiles=[Q(0.5)])],
    )
    report = pipeline.run(
        predictions=predictions,
        ground_truth=ground_truth,
        target_column="load",
    )
    metrics: dict[str, float] = {}
    for subset in report.subset_reports:
        for metric in subset.metrics:
            table = metric.to_dataframe()
            if "rMAE" not in table.columns:
                continue
            value = table["rMAE"].dropna()
            if not value.empty:
                metrics["rmae"] = float(value.iloc[0])
    return metrics


def run_area_backtest(
    load_area: str,
    load: pd.DataFrame,
    predictors: pd.DataFrame,
    *,
    settings: PipelineSettings | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    show_progress: bool = False,
    cache_dir: str | None = None,
) -> dict[str, object]:
    """Run one load area through BacktestPipeline and score the predictions."""
    settings = settings or load_settings()
    area = BY_CODE[load_area.strip().upper()]
    window_start = pd.Timestamp(start or settings.backtest_start)
    if window_start.tzinfo is None:
        window_start = window_start.tz_localize(EASTERN)
    window_end = pd.Timestamp(end) if end is not None else settings.backtest_end
    if window_end is None:
        window_end = pd.to_datetime(predictors["timestamp"]).max()
    window_end = pd.Timestamp(window_end)
    if window_end.tzinfo is None:
        window_end = window_end.tz_localize(EASTERN)
    if window_end <= window_start:
        raise ValueError("backtest end must be after start")

    ground_truth = to_versioned_dataset(load, ["load"], settings.sample_interval)
    features = to_versioned_dataset(predictors, list(WEATHER_COLUMNS), settings.sample_interval)
    config = workflow_config_for_area(
        area,
        settings,
        storage=None,
        run_name=None,
        reuse=False,
    )
    workflow = create_forecasting_workflow(config)
    forecaster = OpenSTEF4BacktestForecaster(
        config=BacktestForecasterConfig(
            requires_training=True,
            predict_sample_interval=settings.prediction_sample_interval,
            predict_length=settings.horizon_timedelta,
            predict_min_length=settings.sample_interval,
            predict_context_length=settings.predict_context,
            predict_context_min_coverage=settings.min_coverage,
            training_context_length=settings.training_context,
            training_context_min_coverage=settings.min_coverage,
        ),
        workflow_template=workflow,
        cache_dir=ensure_dir(
            cache_dir or (settings.processed_dir.parent / "interim" / "beam" / area.code)
        ),
    )
    pipeline = BacktestPipeline(
        config=BacktestConfig(
            prediction_sample_interval=settings.prediction_sample_interval,
            predict_interval=settings.predict_interval,
            train_interval=settings.train_interval,
            align_time=settings.align_time,
        ),
        forecaster=forecaster,
    )
    predictions = pipeline.run(
        ground_truth=ground_truth,
        predictors=features,
        start=window_start.tz_convert("UTC").to_pydatetime(),
        end=window_end.tz_convert("UTC").to_pydatetime(),
        show_progress=show_progress,
    )
    table = predictions.data.copy()
    table.index.name = "timestamp"
    actual = load.set_index("timestamp")["load"]
    actual.index = _utc_index(pd.Series(actual.index))
    if "quantile_P50" in table.columns and not table.empty:
        metrics = score_forecast(actual, table)
    else:
        needed_start = window_start - settings.training_context
        predictor_min = pd.to_datetime(predictors["timestamp"]).min()
        predictor_max = pd.to_datetime(predictors["timestamp"]).max()
        message = (
            "Beam produced no forecasts, so there are no metrics to score. "
            f"Each fit looks back {int(settings.training_context.days)} days from "
            f"{window_start.date().isoformat()}, so predictors must start by "
            f"{needed_start.date().isoformat()}. This predictor file only spans "
            f"{predictor_min} to {predictor_max}. Weather in that training window is "
            "almost entirely missing, so OpenSTEF skips the fit and every prediction."
        )
        raise ValueError(message)
    window = window_end - window_start
    try:
        metrics.update(_beam_metrics(predictions, ground_truth, settings, window))
    except (ValueError, KeyError):
        pass
    return {
        "load_area": area.code,
        "predictions": table,
        "metrics": metrics,
        "lead_time_errors": error_by_lead_time(table, actual),
        "start": window_start.isoformat(),
        "end": window_end.isoformat(),
        "assumptions": ASSUMPTIONS,
    }
