"""Prefect backtest flow.

Historical runs are downloaded only when the cache does not already contain
them, unless ``skip_download`` is set. Versioned tables are checked before any
Beam task runs. With ``backtest.end`` null, Beam stops at the latest cached
predictor hour.
"""

from __future__ import annotations

import pandas as pd
from prefect import flow, get_run_logger, task, unmapped
from prefect.task_runners import ThreadPoolTaskRunner

from pjm_forecasting.areas import resolve_areas
from pjm_forecasting.backtesting import run_area_backtest
from pjm_forecasting.ingest.openmeteo import (
    build_versioned_load,
    build_versioned_predictors,
    download_historical_runs,
)
from pjm_forecasting.quality import validate_versioned_load, validate_versioned_weather
from pjm_forecasting.settings import load_settings
from pjm_forecasting.utils import ensure_dir


def _log(message: str) -> None:
    try:
        get_run_logger().info(message)
    except Exception:
        return


def execute_download(
    areas: list[str],
    config_path: str | None,
    *,
    refresh: bool = False,
    skip_download: bool = False,
) -> dict[str, str]:
    """Download missing single runs and return the raw cache directory.

    When ``skip_download`` is true, the existing
    ``data/raw/openmeteo/forecast_runs`` cache is used as-is (no Open-Meteo
    calls). Beam then runs through the latest hour present in that cache when
    ``backtest.end`` is null.
    """
    settings = load_settings(config_path)
    raw_dir = str(settings.forecast_run_dir())
    if skip_download:
        _log(f"skip_download=True; using cached forecast runs under {raw_dir}")
        return {"raw_dir": raw_dir}
    end = settings.backtest_end or pd.Timestamp.now(tz="UTC")
    download_historical_runs(
        areas,
        settings.archive_start,
        end.tz_convert("UTC"),
        out_dir=settings.forecast_run_dir(),
        model=settings.forecast_model,
        publication_delay=settings.publication_delay,
        cadence=settings.cadence,
        workers=settings.run_workers,
        refresh=refresh,
        url=settings.forecast_url,
    )
    return {"raw_dir": raw_dir}


def execute_versioned_tables(
    download: dict[str, str],
    areas: list[str],
    frame_path: str,
    config_path: str | None,
) -> dict[str, list[str]]:
    """Build per-area versioned load and predictor files."""
    settings = load_settings(config_path)
    load = pd.read_parquet(frame_path)
    history_start = settings.backtest_start - settings.training_context - settings.predict_history
    stamps = pd.to_datetime(load["timestamp"])
    if getattr(stamps.dt, "tz", None) is None:
        stamps = stamps.dt.tz_localize("US/Eastern")
    load = load.loc[stamps >= history_start].copy()
    predictor_paths: list[str] = []
    load_paths: list[str] = []
    for code in areas:
        predictor = build_versioned_predictors(
            download["raw_dir"],
            code,
            settings.versioned_predictor_dir() / f"{code}.parquet",
            settings.publication_delay,
        )
        load_path = settings.versioned_load_dir() / f"{code}.parquet"
        build_versioned_load(load, code, settings.load_delay, load_path)
        predictor_paths.append(str(predictor))
        load_paths.append(str(load_path))
    return {"areas": areas, "predictor_paths": predictor_paths, "load_paths": load_paths}


def execute_version_gate(
    versioned: dict[str, list[str]],
    config_path: str | None,
) -> dict[str, list[str]]:
    """Reject the backtest when any versioned table fails a fatal check."""
    settings = load_settings(config_path)
    for code, predictor_path, load_path in zip(
        versioned["areas"],
        versioned["predictor_paths"],
        versioned["load_paths"],
        strict=True,
    ):
        weather_report = validate_versioned_weather(
            pd.read_parquet(predictor_path),
            publication_delay=settings.publication_delay,
            max_gap=settings.max_gap,
        )
        load_report = validate_versioned_load(
            pd.read_parquet(load_path),
            load_delay=settings.load_delay,
            load_min=settings.load_min,
            max_gap=settings.max_gap,
        )
        for line in weather_report.log_lines() + load_report.log_lines():
            _log(f"{code} {line}")
        weather_report.raise_if_fatal()
        load_report.raise_if_fatal()
    return versioned


def execute_backtest(
    area: str,
    load_path: str,
    predictor_path: str,
    config_path: str | None,
) -> dict[str, object]:
    """Run Beam for one area and write its predictions."""
    settings = load_settings(config_path)
    result = run_area_backtest(
        area,
        pd.read_parquet(load_path),
        pd.read_parquet(predictor_path),
        settings=settings,
        show_progress=False,
    )
    destination = ensure_dir(settings.processed_dir / "backtests") / f"{area}.parquet"
    predictions = result["predictions"]
    assert isinstance(predictions, pd.DataFrame)
    output = predictions.copy()
    output.index.name = "timestamp"
    output.reset_index().to_parquet(destination, index=False)
    return {
        "load_area": area,
        "predictions_path": str(destination),
        "metrics": result["metrics"],
        "start": result["start"],
        "end": result["end"],
    }


@task(name="download-forecast-runs")
def download_runs(
    areas: list[str],
    config_path: str | None,
    refresh: bool,
    skip_download: bool = False,
) -> dict[str, str]:
    return execute_download(
        areas, config_path, refresh=refresh, skip_download=skip_download
    )


@task(name="build-versioned-tables")
def build_tables(
    download: dict[str, str],
    areas: list[str],
    frame_path: str,
    config_path: str | None,
) -> dict[str, list[str]]:
    return execute_versioned_tables(download, areas, frame_path, config_path)


@task(name="versioned-quality-gate")
def version_gate(versioned: dict[str, list[str]], config_path: str | None) -> dict[str, list[str]]:
    return execute_version_gate(versioned, config_path)


@task(name="beam-backtest")
def beam_area(
    area: str,
    load_path: str,
    predictor_path: str,
    config_path: str | None,
) -> dict[str, object]:
    return execute_backtest(area, load_path, predictor_path, config_path)


@flow(name="backtest-load-areas", task_runner=ThreadPoolTaskRunner(max_workers=1))
def backtest_load_areas(
    config_path: str | None = None,
    areas: str = "all",
    frame_path: str | None = None,
    refresh: bool = False,
    skip_download: bool = False,
) -> list[dict[str, object]]:
    """Download runs (unless skipped), validate versions, and backtest each area."""
    settings = load_settings(config_path)
    selected = resolve_areas(areas)
    source = frame_path or str(settings.processed_dir / "load_area_hourly.parquet")
    downloaded = download_runs(selected, config_path, refresh, skip_download)
    versioned = build_tables(downloaded, selected, source, config_path)
    checked = version_gate(versioned, config_path)
    saved = beam_area.map(
        checked["areas"],
        checked["load_paths"],
        checked["predictor_paths"],
        unmapped(config_path),
    )
    return [item.result() for item in saved]
