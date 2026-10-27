"""Weekly retrain on forecast-run weather, band calibration, and publish."""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")

import pandas as pd
from prefect import flow, get_run_logger, task

from pjm_forecasting.areas import resolve_areas
from pjm_forecasting.band_calibration import (
    apply_band_scales,
    conformal_scales,
    coverage_report,
    write_band_scale,
)
from pjm_forecasting.flows.aggregates import train_aggregate_areas
from pjm_forecasting.flows.training import train_load_areas
from pjm_forecasting.models_bundle import (
    package_bundle,
    publish_archive_fallback,
    publish_bundle,
)
from pjm_forecasting.preprocessing import prepare_openstef_frame
from pjm_forecasting.settings import load_settings
from pjm_forecasting.storage import ObjectStore
from pjm_forecasting.training import load_area_model, predict_quantiles, score_forecast
from pjm_forecasting.utils import EASTERN, ensure_dir

FORECAST_RUNS_PREFIX = "raw/forecast_runs"
LOAD_AREAS_PREFIX = "raw/pjm/load_areas"


def _log(message: str) -> None:
    try:
        get_run_logger().info(message)
    except Exception:
        print(message, flush=True)


def hydrate_training_inputs(settings, store: ObjectStore) -> dict[str, int]:
    """Pull bootstrap caches from object storage when the local dirs are empty."""
    pred = settings.versioned_predictor_dir()
    load_root = settings.raw_dir / "pjm" / "load_areas"
    counts = {"forecast_runs": 0, "load_areas": 0}
    if store.is_local:
        return counts
    pred_n = len(list(pred.glob("*.parquet"))) if pred.is_dir() else 0
    if pred_n == 0:
        counts["forecast_runs"] = store.download_prefix(FORECAST_RUNS_PREFIX, pred)
        _log(f"hydrated {counts['forecast_runs']} forecast-run tables from {FORECAST_RUNS_PREFIX}")
    load_n = len(list(load_root.glob("*/*.parquet"))) if load_root.is_dir() else 0
    if load_n == 0:
        counts["load_areas"] = store.download_prefix(LOAD_AREAS_PREFIX, load_root)
        _log(f"hydrated {counts['load_areas']} load-area files from {LOAD_AREAS_PREFIX}")
    return counts


def build_forecast_training_frame(config_path: str | None = None):
    """Join metered load with latest-available forecast-run weather."""
    from build_load_area_frame import read_forecast_run_weather, read_load_cache
    from pjm_forecasting.frame import build_load_area_frame, write_frame

    settings = load_settings(config_path)
    load = read_load_cache(settings.raw_dir / "pjm")
    weather = read_forecast_run_weather(settings.versioned_predictor_dir())
    frame = build_load_area_frame(load, weather)
    path = settings.forecast_frame_path()
    write_frame(frame, path)
    _log(f"wrote forecast training frame {len(frame)} rows -> {path}")
    return path


def _replay_validation(
    area: str,
    frame: pd.DataFrame,
    settings,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Rolling forecast over the validation window using as-issued weather."""
    prepared = prepare_openstef_frame(frame, area)
    end = prepared.index.max() + pd.Timedelta(hours=1)
    start = end - pd.Timedelta(days=settings.forecast_validation_days)
    workflow = load_area_model(area, settings=settings)
    # Use rolling_forecast from training when weather in frame is already
    # latest-available; that matches short-lead training. For band scales we
    # also need live-like leads, so rebuild each origin's weather from runs.
    predictor_path = settings.versioned_predictor_dir() / f"{area}.parquet"
    if predictor_path.is_file():
        versioned = pd.read_parquet(predictor_path)
    else:
        versioned = None
    pieces: list[pd.DataFrame] = []
    origin = start
    last_origin = end - settings.horizon_timedelta
    while origin <= last_origin:
        history = prepared.copy()
        if versioned is not None:
            from pjm_forecasting.backtesting import select_forecast_versions

            as_of = origin.tz_convert("UTC") if origin.tzinfo else origin.tz_localize(EASTERN).tz_convert("UTC")
            issued = select_forecast_versions(versioned, as_of)
            if not issued.empty:
                issued = issued.set_index(
                    pd.to_datetime(issued["timestamp"]).dt.tz_convert(history.index.tz)
                    if pd.to_datetime(issued["timestamp"]).dt.tz is not None
                    else pd.to_datetime(issued["timestamp"]).dt.tz_localize(EASTERN)
                )
                for column in settings.weather_columns:
                    if column in issued.columns:
                        overlap = history.index.intersection(issued.index)
                        history.loc[overlap, column] = issued.loc[overlap, column].to_numpy()
        try:
            forecast = predict_quantiles(
                area,
                history,
                origin,
                workflow=workflow,
                settings=settings,
            )
            pieces.append(forecast)
        except Exception as exc:  # noqa: BLE001 — keep replaying other origins
            _log(f"{area} skip origin {origin}: {exc}")
        origin = origin + settings.validation_stride
    if not pieces:
        raise ValueError(f"{area} produced no validation forecasts for band calibration")
    combined = pd.concat(pieces)
    combined.index.name = "timestamp"
    metrics = score_forecast(prepared["load"], combined)
    return combined, metrics


def calibrate_bands(
    areas: list[str],
    frame_path: str,
    config_path: str | None,
) -> dict[str, Any]:
    """Compute and write per-area lead-hour band scales."""
    settings = load_settings(config_path)
    frame = pd.read_parquet(frame_path)
    payload: dict[str, Any] = {"areas": {}, "coverage_before": {}, "coverage_after": {}}
    for code in areas:
        forecast, metrics = _replay_validation(code, frame, settings)
        prepared = prepare_openstef_frame(frame, code)
        before = coverage_report(prepared["load"], forecast)
        scales = conformal_scales(
            prepared["load"],
            forecast,
            horizon=settings.prediction_length,
        )
        widened = apply_band_scales(forecast, scales)
        after = coverage_report(prepared["load"], widened)
        payload["areas"][code] = scales
        payload["coverage_before"][code] = before
        payload["coverage_after"][code] = after
        payload.setdefault("metrics", {})[code] = metrics
        _log(
            f"{code} band coverage before={before.get('coverage_p10_p90')} "
            f"after={after.get('coverage_p10_p90')} mae={metrics.get('mae')}"
        )
    path = settings.models_dir / "band_scale.json"
    write_band_scale(path, payload)
    return payload


def passes_publish_gate(
    new_metrics: dict[str, dict[str, float]],
    *,
    store: ObjectStore,
    settings,
) -> bool:
    """Reject publish when any area MAE regresses beyond the configured ratio."""
    try:
        latest = store.read_json("models/latest.json")
    except FileNotFoundError:
        return True
    previous = (latest.get("metrics") or {}).get("areas") or {}
    if not previous:
        return True
    ratio = settings.max_validation_mae_ratio
    for area, scores in new_metrics.items():
        old = previous.get(area) or {}
        old_mae = old.get("mae")
        new_mae = scores.get("mae")
        if old_mae is None or new_mae is None:
            continue
        if float(old_mae) > 0 and float(new_mae) > float(old_mae) * ratio:
            _log(
                f"publish gate failed for {area}: "
                f"new mae {new_mae} > {ratio} x previous {old_mae}"
            )
            return False
    return True


@task(name="build-forecast-frame")
def build_frame_task(config_path: str | None) -> str:
    return str(build_forecast_training_frame(config_path))


@task(name="calibrate-bands")
def calibrate_task(areas: list[str], frame_path: str, config_path: str | None) -> dict[str, Any]:
    return calibrate_bands(areas, frame_path, config_path)


@flow(name="retrain-forecast-models")
def retrain_forecast_models(
    config_path: str | None = None,
    areas: str = "all",
    aggregates: str = "all",
    *,
    skip_archive_fallback: bool = False,
    publish: bool = True,
) -> dict[str, Any]:
    """Build forecast-run frame, train, calibrate bands, and publish a bundle."""
    settings = load_settings(config_path)
    store = ObjectStore()
    ensure_dir(settings.models_dir)
    hydrate_training_inputs(settings, store)
    if not skip_archive_fallback:
        try:
            publish_archive_fallback(settings=settings, store=store)
            _log("published archive-fallback bundle")
        except Exception as exc:  # noqa: BLE001
            _log(f"archive-fallback skipped: {exc}")

    frame_path = build_frame_task(config_path)
    selected = resolve_areas(areas)
    area_results = train_load_areas(
        config_path=config_path,
        areas=areas,
        frame_path=frame_path,
        split_mode="rolling",
    )
    agg_results = train_aggregate_areas(
        config_path=config_path,
        aggregates=aggregates,
        frame_path=frame_path,
        split_mode="rolling",
    )
    band_payload = calibrate_task(selected, frame_path, config_path)
    metrics = {
        "areas": {
            item["load_area"]: item.get("metrics") or band_payload.get("metrics", {}).get(item["load_area"], {})
            for item in area_results
        },
        "aggregates": {item["load_area"]: item.get("metrics") or {} for item in agg_results},
        "coverage_before": band_payload.get("coverage_before", {}),
        "coverage_after": band_payload.get("coverage_after", {}),
    }
    result: dict[str, Any] = {
        "frame_path": frame_path,
        "areas": area_results,
        "aggregates": agg_results,
        "band_scale": band_payload,
        "metrics": metrics,
        "published": False,
    }
    if not publish:
        return result
    if not passes_publish_gate(metrics["areas"], store=store, settings=settings):
        result["published"] = False
        result["publish_blocked"] = True
        return result
    archive = package_bundle(settings=settings, band_scale=band_payload)
    latest = publish_bundle(archive, store=store, metrics=metrics)
    result["published"] = True
    result["latest"] = latest
    _log(f"published model bundle {latest['version']}")
    return result
