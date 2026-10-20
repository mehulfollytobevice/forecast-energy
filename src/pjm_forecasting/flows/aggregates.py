"""Prefect flow for the aggregate totals.

The source frame stays load-area grain. Each total is rolled up, then fit
with the same tasks as ``train_load_areas``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from prefect import flow, task, unmapped
from prefect.task_runners import ThreadPoolTaskRunner

from pjm_forecasting.flows.training import (
    _write_indexed,
    fit_area,
    persist_area,
    quality_gate,
    refit_area_task,
    split_frames,
)
from pjm_forecasting.reconcile import prepare_aggregate_frame, resolve_aggregates
from pjm_forecasting.settings import PipelineSettings, load_settings


def execute_prepare_aggregates(
    gate: dict[str, object],
    aggregates: list[str],
    out_dir: str,
) -> dict[str, list[str]]:
    """Write one OpenSTEF frame per aggregate total."""
    frame = pd.read_parquet(str(gate["frame_path"]))
    root = Path(out_dir)
    codes: list[str] = []
    paths: list[str] = []
    for code in aggregates:
        prepared = prepare_aggregate_frame(frame, code)
        path = _write_indexed(prepared, root / code / "frame.parquet")
        codes.append(code)
        paths.append(path)
    return {"areas": codes, "frame_paths": paths}


@task(name="prepare-aggregate-frames")
def prepare_aggregate_frames(
    gate: dict[str, object],
    aggregates: list[str],
    out_dir: str,
) -> dict[str, list[str]]:
    return execute_prepare_aggregates(gate, aggregates, out_dir)


def _default_frame(settings: PipelineSettings) -> str:
    return str(settings.processed_dir / "load_area_hourly.parquet")


@flow(name="train-aggregate-areas", task_runner=ThreadPoolTaskRunner(max_workers=1))
def train_aggregate_areas(
    config_path: str | None = None,
    aggregates: str = "all",
    frame_path: str | None = None,
    out_dir: str | None = None,
    split_mode: str = "chronological",
) -> list[dict[str, object]]:
    """Quality-check the load-area frame, then fit one model per aggregate total."""
    settings = load_settings(config_path)
    selected = resolve_aggregates(aggregates)
    if frame_path is None and split_mode == "rolling":
        source = str(settings.forecast_frame_path())
    else:
        source = frame_path or _default_frame(settings)
    destination = out_dir or str(settings.openstef_dir() / "aggregates")
    gate = quality_gate(source, config_path)
    prepared = prepare_aggregate_frames(gate, selected, destination)
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
