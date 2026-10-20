"""Prefect training flow.

Tasks pass paths and small result dictionaries. A fatal quality report raises
before frame preparation, splits, or any per-area fit.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from prefect import flow, get_run_logger, task, unmapped
from prefect.task_runners import ThreadPoolTaskRunner

from pjm_forecasting.areas import resolve_areas
from pjm_forecasting.preprocessing import prepare_openstef_frame
from pjm_forecasting.quality import validate_joined
from pjm_forecasting.settings import PipelineSettings, load_settings
from pjm_forecasting.splits import split_chronological, split_rolling
from pjm_forecasting.training import refit_area, train_area
from pjm_forecasting.utils import ensure_dir


def _log(message: str) -> None:
    try:
        get_run_logger().info(message)
    except Exception:
        return


def _write_indexed(frame: pd.DataFrame, path: Path) -> str:
    ensure_dir(path.parent)
    output = frame.copy()
    output.index.name = "timestamp"
    output.reset_index().to_parquet(path, index=False)
    return str(path)


def _read_indexed(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    stamps = pd.to_datetime(frame["timestamp"])
    if stamps.dt.tz is None:
        stamps = stamps.dt.tz_localize("US/Eastern")
    frame = frame.assign(timestamp=stamps)
    return frame.set_index("timestamp").sort_index()


def execute_quality_gate(frame_path: str, config_path: str | None) -> dict[str, object]:
    """Validate the joined frame and stop the flow when a fatal check fails."""
    settings = load_settings(config_path)
    frame = pd.read_parquet(frame_path)
    report = validate_joined(frame, max_gap=settings.max_gap, load_min=settings.load_min)
    for line in report.log_lines():
        _log(line)
    report.raise_if_fatal()
    return {"frame_path": frame_path, "stats": report.stats, "config_path": config_path}


def execute_prepare(
    gate: dict[str, object],
    areas: list[str],
    out_dir: str,
) -> dict[str, list[str]]:
    """Write one OpenSTEF frame per load area."""
    frame = pd.read_parquet(str(gate["frame_path"]))
    root = Path(out_dir)
    codes: list[str] = []
    paths: list[str] = []
    for code in areas:
        prepared = prepare_openstef_frame(frame, code)
        path = _write_indexed(prepared, root / code / "frame.parquet")
        codes.append(code)
        paths.append(path)
    return {"areas": codes, "frame_paths": paths}


def execute_split(
    prepared: dict[str, list[str]],
    config_path: str | None,
    *,
    split_mode: str = "chronological",
) -> dict[str, list[str]]:
    """Write train, validation, and test files (chronological or rolling)."""
    settings = load_settings(config_path)
    areas: list[str] = []
    train_paths: list[str] = []
    validation_paths: list[str] = []
    test_paths: list[str] = []
    for code, path in zip(prepared["areas"], prepared["frame_paths"], strict=True):
        frame = _read_indexed(path)
        if split_mode == "rolling":
            split = split_rolling(
                frame,
                validation_days=settings.forecast_validation_days,
                train_start=settings.forecast_train_start,
            )
        else:
            split = split_chronological(frame, settings.train_end, settings.validation_end)
        folder = Path(path).parent
        areas.append(code)
        train_paths.append(_write_indexed(split.train, folder / "train.parquet"))
        validation_paths.append(_write_indexed(split.validation, folder / "validation.parquet"))
        test_paths.append(_write_indexed(split.test, folder / "test.parquet"))
    return {
        "areas": areas,
        "train_paths": train_paths,
        "validation_paths": validation_paths,
        "test_paths": test_paths,
        "split_mode": split_mode,
    }


def execute_fit(
    area: str,
    train_path: str,
    validation_path: str,
    config_path: str | None,
) -> dict[str, object]:
    """Fit one area and score the validation window."""
    settings = load_settings(config_path)
    result = train_area(
        area,
        _read_indexed(train_path),
        _read_indexed(validation_path),
        settings=settings,
    )
    result["train_path"] = train_path
    result["validation_path"] = validation_path
    _log(f"{area} validation {result['metrics']}")
    return result


def execute_refit(fitted: dict[str, object], config_path: str | None) -> dict[str, object]:
    """Refit one area on train plus validation."""
    settings = load_settings(config_path)
    result = refit_area(
        str(fitted["load_area"]),
        _read_indexed(str(fitted["train_path"])),
        _read_indexed(str(fitted["validation_path"])),
        settings=settings,
    )
    result["metrics"] = fitted.get("metrics", {})
    return result


def execute_persist(refit: dict[str, object], config_path: str | None) -> dict[str, object]:
    """Confirm the final run is in the local MLflow store."""
    from pjm_forecasting.training import mlflow_storage

    settings = load_settings(config_path)
    storage = mlflow_storage(settings)
    run = storage.search_run(model_id=str(refit["model_id"]), run_name="final")
    if run is None:
        raise RuntimeError(f"final model for {refit['load_area']} is not in MLflow")
    saved = dict(refit)
    saved["run_id"] = str(run.info.run_id)
    saved["tracking_uri"] = settings.tracking_uri
    _log(f"saved {saved['load_area']} run {saved['run_id']}")
    return saved


@task(name="quality-gate")
def quality_gate(frame_path: str, config_path: str | None) -> dict[str, object]:
    return execute_quality_gate(frame_path, config_path)


@task(name="prepare-frames")
def prepare_frames(gate: dict[str, object], areas: list[str], out_dir: str) -> dict[str, list[str]]:
    return execute_prepare(gate, areas, out_dir)


@task(name="split-frames")
def split_frames(
    prepared: dict[str, list[str]],
    config_path: str | None,
    split_mode: str = "chronological",
) -> dict[str, list[str]]:
    return execute_split(prepared, config_path, split_mode=split_mode)


@task(name="fit-area")
def fit_area(
    area: str,
    train_path: str,
    validation_path: str,
    config_path: str | None,
) -> dict[str, object]:
    return execute_fit(area, train_path, validation_path, config_path)


@task(name="refit-area")
def refit_area_task(fitted: dict[str, object], config_path: str | None) -> dict[str, object]:
    return execute_refit(fitted, config_path)


@task(name="persist-model")
def persist_area(refit: dict[str, object], config_path: str | None) -> dict[str, object]:
    return execute_persist(refit, config_path)


def _default_frame(settings: PipelineSettings) -> str:
    return str(settings.processed_dir / "load_area_hourly.parquet")


@flow(name="train-load-areas", task_runner=ThreadPoolTaskRunner(max_workers=1))
def train_load_areas(
    config_path: str | None = None,
    areas: str = "all",
    frame_path: str | None = None,
    out_dir: str | None = None,
    split_mode: str = "chronological",
) -> list[dict[str, object]]:
    """Quality-check, split, fit, refit, and store one model per load area."""
    settings = load_settings(config_path)
    selected = resolve_areas(areas)
    if frame_path is None and split_mode == "rolling":
        source = str(settings.forecast_frame_path())
    else:
        source = frame_path or _default_frame(settings)
    destination = out_dir or str(settings.openstef_dir())
    gate = quality_gate(source, config_path)
    prepared = prepare_frames(gate, selected, destination)
    splits = split_frames(prepared, config_path, split_mode)
    fitted = fit_area.map(
        splits["areas"],
        splits["train_paths"],
        splits["validation_paths"],
        unmapped(config_path),
    )
    refit = refit_area_task.map(fitted, unmapped(config_path))
    saved = persist_area.map(refit, unmapped(config_path))
    results = [item.result() for item in saved]
    for item in results:
        item["weather_source"] = settings.weather_source
        item["split_mode"] = split_mode
    return results
