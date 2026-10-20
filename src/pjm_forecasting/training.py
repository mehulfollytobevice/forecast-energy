"""Per-area OpenSTEF fit, validation, refit, and local MLflow reload.

Archive weather is the training source. Point-in-time forecast runs are a
separate backtest path. A final model is the refit on train plus validation
and is stored under the MLflow run name ``final``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.types import LeadTime, Q
from openstef_models.integrations.mlflow import MLFlowStorage
from openstef_models.models.forecasting.xgboost_forecaster import XGBoostHyperParams
from openstef_models.presets import ForecastingWorkflowConfig, create_forecasting_workflow
from openstef_models.presets.forecasting_workflow import LocationConfig
from openstef_models.utils.data_split import DataSplitter
from openstef_models.workflows.custom_forecasting_workflow import CustomForecastingWorkflow
from pydantic_extra_types.coordinate import Coordinate, Latitude, Longitude
from pydantic_extra_types.country import CountryAlpha2

from pjm_forecasting.areas import LoadArea
from pjm_forecasting.reconcile import resolve_model_area
from pjm_forecasting.settings import PipelineSettings, load_settings

FINAL_RUN_NAME = "final"
VALIDATION_RUN_NAME = "validation"


def model_id_for_area(load_area: str) -> str:
    """MLflow experiment name for one load area or aggregate total."""
    return f"pjm-{resolve_model_area(load_area).code}"


def mlflow_storage(settings: PipelineSettings) -> MLFlowStorage:
    """Local SQLite tracking store plus a local artifact directory."""
    settings.local_artifacts_path.mkdir(parents=True, exist_ok=True)
    settings.artifact_location.mkdir(parents=True, exist_ok=True)
    return MLFlowStorage(
        tracking_uri=settings.tracking_uri,
        local_artifacts_path=settings.local_artifacts_path,
        artifact_location=settings.artifact_location.resolve().as_uri(),
    )


def workflow_config_for_area(
    area: LoadArea,
    settings: PipelineSettings,
    storage: MLFlowStorage | None,
    *,
    run_name: str | None,
    reuse: bool,
) -> ForecastingWorkflowConfig:
    """OpenSTEF config for one load area, including its city coordinate."""
    quantiles = [Q(value) for value in settings.quantiles]
    hyperparams: dict[str, object] = {}
    if settings.model_type == "xgboost":
        hyperparams["xgboost_hyperparams"] = XGBoostHyperParams(n_estimators=settings.n_estimators)
    return ForecastingWorkflowConfig(
        model_id=model_id_for_area(area.code),
        run_name=run_name,
        model=settings.model_type,
        sample_interval=settings.sample_interval,
        horizons=[LeadTime.from_string(settings.horizon)],
        quantiles=quantiles,
        location=LocationConfig(
            name=area.code,
            description=area.city,
            coordinate=Coordinate(
                latitude=Latitude(Decimal(str(area.lat))),
                longitude=Longitude(Decimal(str(area.lon))),
            ),
            country_code=CountryAlpha2(settings.country_code),
        ),
        target_column="load",
        temperature_column="temperature_2m",
        relative_humidity_column="relative_humidity_2m",
        wind_speed_column="wind_speed_10m",
        radiation_column="shortwave_radiation",
        pressure_column="surface_pressure",
        predict_history=settings.predict_history,
        cutoff_history=settings.cutoff_history,
        max_day_lags=settings.max_day_lags,
        data_splitter=DataSplitter(val_fraction=0.0, test_fraction=0.0),
        mlflow_storage=storage,
        model_reuse_enable=reuse,
        model_selection_enable=False,
        verbosity=0,
        tags={"stage": run_name or "train", "load_area": area.code, "city": area.city},
        experiment_tags={"project": "pjm-forecasting"},
        **hyperparams,
    )


def _dataset(frame: pd.DataFrame, sample_interval: timedelta) -> TimeSeriesDataset:
    return TimeSeriesDataset(data=frame, sample_interval=sample_interval)


def _pinball(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    error = actual - predicted
    loss = np.maximum(quantile * error, (quantile - 1.0) * error)
    return float(np.mean(loss))


def score_forecast(actual: pd.Series, forecast: pd.DataFrame) -> dict[str, float]:
    """Point and quantile scores on rows that have both an actual and a P50."""
    frame = forecast.copy()
    if "quantile_P50" not in frame.columns:
        raise ValueError("forecast is missing quantile_P50")
    observed = actual.reindex(frame.index)
    mask = observed.notna() & frame["quantile_P50"].notna()
    observed = observed.loc[mask].astype(float)
    used = frame.loc[mask]
    if used.empty:
        raise ValueError("forecast and actuals do not overlap")
    point = used["quantile_P50"].astype(float)
    residual = observed.to_numpy() - point.to_numpy()
    scores = {
        "n": float(len(used)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
    }
    for quantile in (0.1, 0.5, 0.9):
        column = f"quantile_P{int(round(quantile * 100))}"
        if column not in used.columns:
            continue
        scores[f"pinball_p{int(round(quantile * 100))}"] = _pinball(
            observed.to_numpy(),
            used[column].astype(float).to_numpy(),
            quantile,
        )
    if {"quantile_P10", "quantile_P90"}.issubset(used.columns):
        inside = (observed >= used["quantile_P10"]) & (observed <= used["quantile_P90"])
        scores["coverage_p10_p90"] = float(inside.mean())
    return scores


def _prediction_frame(
    history: pd.DataFrame,
    origin: pd.Timestamp,
    horizon: timedelta,
    predict_history: timedelta,
) -> pd.DataFrame:
    start = origin - predict_history
    end = origin + horizon - pd.Timedelta(hours=1)
    window = history.loc[(history.index >= start) & (history.index <= end)].copy()
    if window.empty or origin not in window.index:
        raise ValueError(f"no prediction rows at {origin.isoformat()}")
    window.loc[window.index >= origin, "load"] = np.nan
    return window


def rolling_forecast(
    workflow: CustomForecastingWorkflow,
    history: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    settings: PipelineSettings,
) -> pd.DataFrame:
    """Forecast every stride from ``start`` while a full horizon is still in ``history``."""
    pieces: list[pd.DataFrame] = []
    origin = pd.Timestamp(start)
    last_origin = pd.Timestamp(end) - settings.horizon_timedelta + settings.sample_interval
    while origin <= last_origin:
        window = _prediction_frame(
            history,
            origin,
            settings.horizon_timedelta,
            settings.predict_history,
        )
        dataset = _dataset(window, settings.sample_interval)
        forecast = workflow.predict(dataset, forecast_start=origin.to_pydatetime())
        piece = forecast.data.copy()
        piece["origin"] = origin
        pieces.append(piece)
        origin = origin + settings.validation_stride
    if not pieces:
        raise ValueError("validation window is shorter than the forecast horizon")
    combined = pd.concat(pieces)
    combined.index.name = "timestamp"
    return combined


def _run_id(storage: MLFlowStorage, model_id: str, run_name: str) -> str:
    run = storage.search_run(model_id=model_id, run_name=run_name)
    if run is None:
        raise RuntimeError(f"MLflow run {run_name!r} was not stored for {model_id}")
    return str(run.info.run_id)


def train_area(
    load_area: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    settings: PipelineSettings | None = None,
    storage: MLFlowStorage | None = None,
) -> dict[str, object]:
    """Fit on train, then score a rolling forecast over validation."""
    settings = settings or load_settings()
    area = resolve_model_area(load_area)
    storage = storage or mlflow_storage(settings)
    config = workflow_config_for_area(
        area,
        settings,
        storage,
        run_name=VALIDATION_RUN_NAME,
        reuse=False,
    )
    workflow = create_forecasting_workflow(config)
    workflow.fit(
        data=_dataset(train, settings.sample_interval),
        data_val=_dataset(validation, settings.sample_interval),
    )
    history = pd.concat([train, validation]).sort_index()
    forecast = rolling_forecast(
        workflow,
        history,
        start=validation.index.min(),
        end=validation.index.max() + settings.sample_interval,
        settings=settings,
    )
    metrics = score_forecast(validation["load"], forecast)
    return {
        "load_area": area.code,
        "model_id": config.model_id,
        "run_id": _run_id(storage, config.model_id, VALIDATION_RUN_NAME),
        "run_name": VALIDATION_RUN_NAME,
        "metrics": metrics,
    }


def refit_area(
    load_area: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    settings: PipelineSettings | None = None,
    storage: MLFlowStorage | None = None,
) -> dict[str, object]:
    """Refit on train plus validation and store the run named ``final``."""
    settings = settings or load_settings()
    area = resolve_model_area(load_area)
    storage = storage or mlflow_storage(settings)
    config = workflow_config_for_area(
        area,
        settings,
        storage,
        run_name=FINAL_RUN_NAME,
        reuse=False,
    )
    workflow = create_forecasting_workflow(config)
    combined = pd.concat([train, validation]).sort_index()
    workflow.fit(data=_dataset(combined, settings.sample_interval))
    model_id = config.model_id
    return {
        "load_area": area.code,
        "model_id": model_id,
        "run_id": _run_id(storage, model_id, FINAL_RUN_NAME),
        "run_name": FINAL_RUN_NAME,
    }


def load_area_model(
    load_area: str,
    *,
    settings: PipelineSettings | None = None,
    storage: MLFlowStorage | None = None,
) -> CustomForecastingWorkflow:
    """Workflow that reloads the latest ``final`` run on the next predict call."""
    settings = settings or load_settings()
    area = resolve_model_area(load_area)
    storage = storage or mlflow_storage(settings)
    config = workflow_config_for_area(
        area,
        settings,
        storage,
        run_name=FINAL_RUN_NAME,
        reuse=True,
    )
    return create_forecasting_workflow(config)


def predict_quantiles(
    load_area: str,
    history: pd.DataFrame,
    origin: str | pd.Timestamp | datetime,
    *,
    workflow: CustomForecastingWorkflow | None = None,
    settings: PipelineSettings | None = None,
) -> pd.DataFrame:
    """P10/P50/P90 forecast from ``origin`` using a reloaded or supplied model."""
    settings = settings or load_settings()
    stamp = pd.Timestamp(origin)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(history.index.tz)
    else:
        stamp = stamp.tz_convert(history.index.tz)
    model = workflow or load_area_model(load_area, settings=settings)
    window = _prediction_frame(history, stamp, settings.horizon_timedelta, settings.predict_history)
    forecast = model.predict(
        _dataset(window, settings.sample_interval),
        forecast_start=stamp.to_pydatetime(),
    )
    frame = forecast.data.copy()
    frame.index.name = "timestamp"
    frame["origin"] = stamp
    return frame
