"""Typed view of the training, storage, and backtest sections in the YAML config."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from pjm_forecasting.schema import WEATHER_COLUMNS
from pjm_forecasting.utils import EASTERN, as_eastern, load_config, project_root

_DEFAULT_RUN_URL = "https://single-runs-api.open-meteo.com/v1/forecast"


def _hours(value: object, default: float) -> timedelta:
    return timedelta(hours=float(default if value is None else value))


def _days(value: object, default: float) -> timedelta:
    return timedelta(days=float(default if value is None else value))


def _absolute(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else project_root() / raw


def _eastern_bound(value: str) -> pd.Timestamp:
    return as_eastern(value)


def _utc_bound(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


@dataclass(frozen=True)
class PipelineSettings:
    """Boundaries and paths for one training or backtest run."""

    timezone: str
    history_start: str
    prediction_length: int
    quantiles: tuple[float, ...]
    model_type: str
    country_code: str
    horizon: str
    sample_interval: timedelta
    train_end: pd.Timestamp
    validation_end: pd.Timestamp
    max_gap: timedelta
    load_min: float
    tracking_uri: str
    artifact_location: Path
    local_artifacts_path: Path
    forecast_url: str
    forecast_model: str
    archive_start: pd.Timestamp
    cadence: timedelta
    publication_delay: timedelta
    load_delay: timedelta
    run_workers: int
    backtest_start: pd.Timestamp
    backtest_end: pd.Timestamp | None
    predict_interval: timedelta
    train_interval: timedelta
    prediction_sample_interval: timedelta
    training_context: timedelta
    predict_context: timedelta
    min_coverage: float
    align_time: time
    predict_history: timedelta
    cutoff_history: timedelta
    max_day_lags: int
    n_estimators: int
    validation_stride: timedelta
    processed_dir: Path
    raw_dir: Path
    forecast_train_start: pd.Timestamp
    forecast_validation_days: int
    max_validation_mae_ratio: float
    weather_source: str
    schedule_et: tuple[str, ...]
    wait_for_run_minutes: int
    poll_seconds: int
    snapshot_ttl_seconds: int
    stale_after_hours: int
    models_dir: Path
    recent_load_lookback_days: int
    recent_load_feed: str
    weather_columns: tuple[str, ...] = WEATHER_COLUMNS

    @property
    def horizon_timedelta(self) -> timedelta:
        return timedelta(hours=self.prediction_length)

    def forecast_run_dir(self) -> Path:
        return self.raw_dir / "openmeteo" / "forecast_runs"

    def versioned_predictor_dir(self) -> Path:
        return self.processed_dir / "forecast_runs"

    def versioned_load_dir(self) -> Path:
        return self.processed_dir / "versioned_load"

    def openstef_dir(self) -> Path:
        return self.processed_dir / "openstef"

    def forecast_frame_path(self) -> Path:
        return self.processed_dir / "load_area_hourly_fc.parquet"


def _tracking_uri(raw: str) -> str:
    override = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if override:
        return override
    if raw.startswith("sqlite:///") and not raw.startswith("sqlite:////"):
        relative = raw.removeprefix("sqlite:///")
        path = Path(relative)
        if not path.is_absolute():
            path = project_root() / path
        return f"sqlite:///{path}"
    return raw


def _align_time(value: str) -> time:
    parsed = time.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=datetime.fromisoformat("2000-01-01T00:00:00+00:00").tzinfo)
    return parsed


def load_settings(path: Path | str | None = None) -> PipelineSettings:
    """Load pipeline settings, applying ``MLFLOW_TRACKING_URI`` when it is set."""
    load_dotenv(project_root() / ".env")
    config = load_config(path)
    quality = config.get("quality") or {}
    splits = config.get("splits") or {}
    model = config.get("model") or {}
    storage = config.get("storage") or {}
    runs = config.get("forecast_runs") or {}
    backtest = config.get("backtest") or {}
    forecast_training = config.get("forecast_training") or {}
    live = config.get("live") or {}
    data = config["data"]
    end = backtest.get("end")
    quantiles = tuple(float(item) for item in model.get("quantiles", [0.1, 0.5, 0.9]))
    schedule = tuple(str(item) for item in live.get("schedule_et", ["05:15", "11:15", "17:15", "23:15"]))
    return PipelineSettings(
        timezone=str(data.get("timezone", EASTERN)),
        history_start=str(data["history_start"]),
        prediction_length=int(config["horizon"]["prediction_length"]),
        quantiles=quantiles,
        model_type=str(model.get("type", "xgboost")),
        country_code=str(model.get("country_code", "US")),
        horizon=str(model.get("horizon", "PT48H")),
        sample_interval=_hours(model.get("sample_interval_hours"), 1),
        train_end=_eastern_bound(str(splits.get("train_end", "2025-01-01"))),
        validation_end=_eastern_bound(str(splits.get("validation_end", "2026-01-01"))),
        max_gap=_hours(quality.get("max_gap_hours"), 6),
        load_min=float(quality.get("load_min", 0)),
        tracking_uri=_tracking_uri(str(storage.get("tracking_uri", "sqlite:///mlflow.db"))),
        artifact_location=_absolute(str(storage.get("artifact_location", "mlflow/artifacts"))),
        local_artifacts_path=_absolute(str(storage.get("local_artifacts_path", "mlflow/staging"))),
        forecast_url=str(runs.get("url", _DEFAULT_RUN_URL)),
        forecast_model=str(runs.get("model", "ecmwf_ifs")),
        archive_start=_utc_bound(str(runs.get("archive_start", "2024-03-14T00:00:00Z"))),
        cadence=_hours(runs.get("cadence_hours"), 6),
        publication_delay=_hours(runs.get("publication_delay_hours"), 6),
        load_delay=_hours(runs.get("load_availability_delay_hours"), 1),
        run_workers=int(runs.get("workers", 4)),
        backtest_start=_eastern_bound(str(backtest.get("start", "2026-01-01"))),
        backtest_end=None if end in (None, "") else _eastern_bound(str(end)),
        predict_interval=_hours(backtest.get("predict_interval_hours"), 6),
        train_interval=_days(backtest.get("train_interval_days"), 7),
        prediction_sample_interval=_hours(backtest.get("prediction_sample_interval_hours"), 1),
        training_context=_days(backtest.get("training_context_days"), 90),
        predict_context=_days(backtest.get("predict_context_days"), 14),
        min_coverage=float(backtest.get("min_coverage", 0.5)),
        align_time=_align_time(str(backtest.get("align_time", "00:00+00"))),
        predict_history=_days(model.get("predict_history_days"), 14),
        cutoff_history=_days(model.get("cutoff_history_days"), 14),
        max_day_lags=int(model.get("max_day_lags", 14)),
        n_estimators=int(model.get("n_estimators", 100)),
        validation_stride=_hours(model.get("validation_stride_hours"), 24),
        processed_dir=_absolute(str(data["processed_dir"])),
        raw_dir=_absolute(str(data["raw_dir"])),
        forecast_train_start=_eastern_bound(
            str(forecast_training.get("train_start", "2024-03-15"))
        ),
        forecast_validation_days=int(forecast_training.get("validation_days", 90)),
        max_validation_mae_ratio=float(forecast_training.get("max_validation_mae_ratio", 1.25)),
        weather_source=str(forecast_training.get("weather_source", "forecast_runs")),
        schedule_et=schedule,
        wait_for_run_minutes=int(live.get("wait_for_run_minutes", 45)),
        poll_seconds=int(live.get("poll_seconds", 60)),
        snapshot_ttl_seconds=int(live.get("snapshot_ttl_seconds", 30)),
        stale_after_hours=int(live.get("stale_after_hours", 12)),
        models_dir=_absolute(str(live.get("models_dir", "data/models"))),
        recent_load_lookback_days=int(live.get("recent_load_lookback_days", 28)),
        recent_load_feed=str(live.get("recent_load_feed", "auto")),
    )
