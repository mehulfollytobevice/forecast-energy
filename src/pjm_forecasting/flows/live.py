"""Live 48-hour forecast flow: refresh data, predict, publish snapshot."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")

import pandas as pd
from prefect import flow, get_run_logger, task

from pjm_forecasting.areas import LOAD_AREAS, resolve_areas
from pjm_forecasting.band_calibration import apply_band_scales
from pjm_forecasting.frame import build_inference_frame
from pjm_forecasting.ingest.openmeteo import (
    build_versioned_predictors,
    download_historical_runs,
    latest_run_weather,
)
from pjm_forecasting.ingest.pjm import PJMClient, default_load_root
from pjm_forecasting.ingest.pjm_benchmark import (
    archive_pjm_forecast,
    attach_benchmark,
    fetch_pjm_forecast,
)
from pjm_forecasting.ingest.pjm_recent import (
    archive_proxy,
    build_recent_load_areas,
    fetch_recent_proxy,
    prediction_history_gaps,
)
from pjm_forecasting.models_bundle import (
    ARCHIVE_FALLBACK_KEY,
    fetch_bundle,
    load_band_scale,
)
from pjm_forecasting.reconcile import aggregates_for_areas, forecast_nodes
from pjm_forecasting.schedule import expected_ecmwf_run, next_origin, should_run
from pjm_forecasting.settings import PipelineSettings, load_settings
from pjm_forecasting.snapshot import build_forecast_snapshot, default_snapshot_path, write_forecast_snapshot
from pjm_forecasting.storage import ObjectStore
from pjm_forecasting.training import load_area_model, predict_quantiles
from pjm_forecasting.utils import EASTERN, ensure_dir


def _log(message: str) -> None:
    try:
        get_run_logger().info(message)
    except Exception:
        print(message, flush=True)


def _rename_quantiles(forecast: pd.DataFrame) -> pd.DataFrame:
    out = forecast.reset_index()
    if "timestamp" not in out.columns and out.index.name == "timestamp":
        out = forecast.reset_index()
    rename = {
        "quantile_P10": "p10",
        "quantile_P50": "p50",
        "quantile_P90": "p90",
    }
    out = out.rename(columns=rename)
    return out


def wait_for_ecmwf_run(
    settings: PipelineSettings,
    *,
    skip_download: bool = False,
) -> pd.Timestamp:
    """Poll until the expected ECMWF run is cached (or timeout)."""
    expected = expected_ecmwf_run(
        publication_delay_hours=int(settings.publication_delay.total_seconds() // 3600),
        cadence_hours=int(settings.cadence.total_seconds() // 3600),
    )
    if skip_download:
        return expected
    deadline = time.time() + settings.wait_for_run_minutes * 60
    areas = resolve_areas("all")[:1]
    while True:
        results = download_historical_runs(
            areas,
            expected,
            expected + pd.Timedelta(hours=1),
            out_dir=settings.forecast_run_dir(),
            model=settings.forecast_model,
            publication_delay=settings.publication_delay,
            cadence=settings.cadence,
            workers=1,
            refresh=False,
            url=settings.forecast_url,
        )
        written = [item for item in results if item.status in ("written", "skipped") and item.rows]
        if written or any(item.status == "skipped" for item in results):
            # skipped with an existing file counts as available
            marker_ok = any(
                (settings.forecast_run_dir() / areas[0] / f"{expected.strftime('%Y%m%dT%H%MZ')}.parquet").is_file()
                for _ in [0]
            )
            if marker_ok or written:
                return expected
        if time.time() >= deadline:
            _log(f"timed out waiting for ECMWF run {expected.isoformat()}; continuing")
            return expected
        _log(f"waiting for ECMWF run {expected.isoformat()}")
        time.sleep(settings.poll_seconds)


def download_latest_runs(
    areas: list[str],
    settings: PipelineSettings,
    *,
    skip_download: bool = False,
) -> dict[str, str]:
    expected = wait_for_ecmwf_run(settings, skip_download=skip_download)
    if not skip_download:
        start = expected - settings.cadence
        download_historical_runs(
            areas,
            start,
            expected + pd.Timedelta(hours=1),
            out_dir=settings.forecast_run_dir(),
            model=settings.forecast_model,
            publication_delay=settings.publication_delay,
            cadence=settings.cadence,
            workers=settings.run_workers,
            refresh=False,
            url=settings.forecast_url,
        )
    for code in areas:
        build_versioned_predictors(
            settings.forecast_run_dir(),
            code,
            settings.versioned_predictor_dir() / f"{code}.parquet",
            settings.publication_delay,
        )
    return {"raw_dir": str(settings.forecast_run_dir()), "run": expected.isoformat()}


def refresh_merged_load(
    settings: PipelineSettings,
    *,
    skip_download: bool = False,
) -> pd.DataFrame:
    from build_load_area_frame import read_load_cache

    from update_zone_load import update_load_areas

    cache_root = default_load_root()
    if not skip_download:
        today = pd.Timestamp.now(tz=EASTERN)
        start = (today - pd.DateOffset(months=1)).strftime("%Y-%m-01")
        end = today.strftime("%Y-%m-%d")
        update_load_areas(
            resolve_areas("all"),
            start,
            end,
            cache_root=cache_root,
            refresh_days=14,
        )
    metered = read_load_cache(cache_root)
    if skip_download:
        # Still try proxy when metered is stale; may no-op without API key.
        try:
            client = PJMClient()
        except Exception:
            return metered.assign(load_source="metered")
    else:
        client = PJMClient()
    end = pd.Timestamp.now(tz=EASTERN)
    start = end - pd.Timedelta(days=max(settings.recent_load_lookback_days, 16))
    try:
        proxy_zone, feed = fetch_recent_proxy(
            client,
            start,
            end,
            feed=settings.recent_load_feed if settings.recent_load_feed in ("auto", "inst_load", "hrl_load_prelim") else "auto",  # type: ignore[arg-type]
        )
        archive_proxy(
            proxy_zone,
            settings.raw_dir / "pjm_recent" / f"{feed}_{end.strftime('%Y%m%dT%H%MZ')}.parquet",
        )
        merged = build_recent_load_areas(
            metered,
            proxy_zone,
            lookback_days=settings.recent_load_lookback_days,
        )
        _log(f"merged load via {feed}; rows={len(merged)}")
        return merged
    except Exception as exc:  # noqa: BLE001
        _log(f"recent-load proxy failed ({exc}); using metered only")
        return metered.assign(load_source="metered")


def build_live_inference(
    areas: list[str],
    load: pd.DataFrame,
    settings: PipelineSettings,
    origin: pd.Timestamp,
) -> tuple[pd.DataFrame, str]:
    weather_parts: list[pd.DataFrame] = []
    run_label = ""
    for code in areas:
        path = settings.versioned_predictor_dir() / f"{code}.parquet"
        versioned = pd.read_parquet(path)
        weather = latest_run_weather(versioned, as_of=pd.Timestamp.now(tz="UTC"))
        weather_parts.append(weather)
        runs = pd.to_datetime(versioned["run_initialized_at"], utc=True)
        available = pd.to_datetime(versioned["available_at"], utc=True)
        visible = runs.loc[available <= pd.Timestamp.now(tz="UTC")]
        if not visible.empty:
            run_label = visible.max().strftime("%Y%m%dT%H%MZ")
    forecast_weather = pd.concat(weather_parts, ignore_index=True)
    # History: load with weather filled from latest-available for past hours.
    history_weather_parts = []
    for code in areas:
        path = settings.versioned_predictor_dir() / f"{code}.parquet"
        from pjm_forecasting.ingest.openmeteo import build_forecast_weather_history

        history_weather_parts.append(build_forecast_weather_history(pd.read_parquet(path)))
    history_weather = pd.concat(history_weather_parts, ignore_index=True)
    load_work = load.loc[load["load_area"].isin(areas)].copy()
    # Attach zone/mkt_region if missing.
    if "zone" not in load_work.columns:
        from pjm_forecasting.areas import BY_CODE

        load_work["zone"] = load_work["load_area"].map(lambda c: BY_CODE[c].zone)
        load_work["mkt_region"] = load_work["load_area"].map(lambda c: BY_CODE[c].mkt_region)
    from pjm_forecasting.frame import build_load_area_frame

    history = build_load_area_frame(load_work.drop(columns=["load_source"], errors="ignore"), history_weather)
    # Origin is the last hour that keeps load; inference starts the next hour.
    origin_keep = origin - pd.Timedelta(hours=1)
    inference = build_inference_frame(
        history,
        forecast_weather,
        settings.prediction_length,
        origin=origin_keep,
    )
    return inference, run_label


def _predict_one(
    code: str,
    history: pd.DataFrame,
    origin: pd.Timestamp,
    settings: PipelineSettings,
    band_scale: dict[str, Any],
    *,
    use_fallback: bool = False,
) -> tuple[pd.DataFrame, str]:
    source = "archive-fallback" if use_fallback else "forecast_runs"
    try:
        if use_fallback:
            # Fallback bundle already extracted into the same MLflow paths when requested.
            pass
        workflow = load_area_model(code, settings=settings)
        forecast = predict_quantiles(
            code,
            history,
            origin,
            workflow=workflow,
            settings=settings,
        )
        scales = (band_scale.get("areas") or {}).get(code) if band_scale else None
        if scales:
            forecast = apply_band_scales(forecast, scales)
        return forecast, source
    except Exception as exc:
        if use_fallback:
            raise
        _log(f"{code} forecast_runs model failed ({exc}); trying archive-fallback")
        raise RuntimeError(str(exc)) from exc


def predict_all(
    areas: list[str],
    aggregates: list[str],
    inference: pd.DataFrame,
    origin: pd.Timestamp,
    settings: PipelineSettings,
    *,
    store: ObjectStore,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    from pjm_forecasting.preprocessing import prepare_openstef_frame
    from pjm_forecasting.reconcile import prepare_aggregate_frame

    band_scale = load_band_scale(settings)
    child_rows: list[pd.DataFrame] = []
    agg_rows: list[pd.DataFrame] = []
    sources: dict[str, str] = {}

    def _history_for(code: str, level: str) -> pd.DataFrame:
        if level == "load_area":
            return prepare_openstef_frame(inference, code)
        return prepare_aggregate_frame(inference, code)

    # Ensure primary bundle is loaded.
    try:
        fetch_bundle(store=store, settings=settings)
    except FileNotFoundError:
        _log("no published forecast_runs bundle; will rely on local MLflow / fallback")

    fallback_extracted = False
    for code in areas:
        history = _history_for(code, "load_area")
        try:
            forecast, source = _predict_one(code, history, origin, settings, band_scale)
        except Exception:
            if not fallback_extracted:
                try:
                    fetch_bundle(store=store, settings=settings, key=ARCHIVE_FALLBACK_KEY)
                    fallback_extracted = True
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(f"{code} failed and archive-fallback missing: {exc}") from exc
            forecast, source = _predict_one(
                code, history, origin, settings, band_scale, use_fallback=True
            )
        piece = _rename_quantiles(forecast)
        piece["load_area"] = code
        child_rows.append(piece.loc[:, ["load_area", "timestamp", "p10", "p50", "p90"]])
        sources[code] = source

    for code in aggregates:
        history = _history_for(code, "aggregate")
        try:
            forecast, source = _predict_one(code, history, origin, settings, band_scale)
        except Exception:
            if not fallback_extracted:
                try:
                    fetch_bundle(store=store, settings=settings, key=ARCHIVE_FALLBACK_KEY)
                    fallback_extracted = True
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(f"{code} failed and archive-fallback missing: {exc}") from exc
            forecast, source = _predict_one(
                code, history, origin, settings, band_scale, use_fallback=True
            )
        piece = _rename_quantiles(forecast)
        piece["id"] = code
        agg_rows.append(piece.loc[:, ["id", "timestamp", "p10", "p50", "p90"]])
        sources[code] = source

    children = pd.concat(child_rows, ignore_index=True) if child_rows else pd.DataFrame()
    aggs = pd.concat(agg_rows, ignore_index=True) if agg_rows else pd.DataFrame()
    return children, aggs, sources


def attach_pjm_benchmark(
    payload: dict[str, Any],
    origin: pd.Timestamp,
    settings: PipelineSettings,
) -> None:
    """Add PJM's latest published forecast to ``payload``; never fails the run."""
    try:
        client = PJMClient()
        end = origin + pd.Timedelta(hours=settings.prediction_length)
        raw, benchmark = fetch_pjm_forecast(client, origin, end)
        archive_pjm_forecast(raw, settings.raw_dir)
        attach_benchmark(payload, benchmark)
        _log(f"attached PJM benchmark issued {payload.get('benchmark', {}).get('issued_at', 'n/a')}")
    except Exception as exc:  # noqa: BLE001
        _log(f"PJM benchmark unavailable ({exc}); publishing without it")


def publish_live_snapshot(
    child_forecasts: pd.DataFrame,
    aggregate_forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    origin: pd.Timestamp,
    *,
    weather_run: str,
    model_version: str,
    load_through: str,
    model_sources: dict[str, str],
    store: ObjectStore,
    settings: PipelineSettings,
    snapshot_load_areas: frozenset[str] | None = None,
) -> dict[str, Any]:
    actuals_work = actuals.drop(columns=["load_source"], errors="ignore")
    payload = build_forecast_snapshot(
        child_forecasts,
        aggregate_forecasts,
        actuals_work,
        origin,
        generated_at=datetime.now(tz=timezone.utc),
        snapshot_load_areas=snapshot_load_areas,
    )
    payload["model_version"] = model_version
    payload["weather_run"] = weather_run
    payload["load_through"] = load_through
    payload["model_sources"] = model_sources
    # Tag nodes with model_source when id matches.
    for node in payload["nodes"]:
        node_id = str(node["id"])
        if node_id in model_sources:
            node["model_source"] = model_sources[node_id]
    attach_pjm_benchmark(payload, origin, settings)

    local_path = write_forecast_snapshot(
        child_forecasts,
        aggregate_forecasts,
        actuals_work,
        origin,
        path=default_snapshot_path(),
        generated_at=payload["generated_at"],
        snapshot_load_areas=snapshot_load_areas,
    )
    # Rewrite with extra metadata.
    import json

    local_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    origin_token = origin.tz_convert("UTC").strftime("%Y%m%dT%H%MZ")
    store.write_json("snapshots/forecast/latest.json", payload)
    store.write_json(f"snapshots/forecast/{origin_token}.json", payload)
    # Keep FORECAST_SNAPSHOT path in sync for the API.
    store.upload_file(local_path, "snapshots/forecast/latest.json")
    return {"path": str(local_path), "origin": origin.isoformat(), "payload": payload}


def already_published(
    store: ObjectStore,
    *,
    weather_run: str,
    load_through: str,
) -> bool:
    try:
        latest = store.read_json("snapshots/forecast/latest.json")
    except FileNotFoundError:
        return False
    return (
        str(latest.get("weather_run", "")) == weather_run
        and str(latest.get("load_through", "")) == load_through
    )


@task(name="download-latest-runs")
def download_runs_task(
    areas: list[str],
    config_path: str | None,
    skip_download: bool,
) -> dict[str, str]:
    return download_latest_runs(areas, load_settings(config_path), skip_download=skip_download)


@task(name="refresh-merged-load")
def refresh_load_task(config_path: str | None, skip_download: bool) -> str:
    settings = load_settings(config_path)
    merged = refresh_merged_load(settings, skip_download=skip_download)
    path = settings.processed_dir / "load_area_merged.parquet"
    ensure_dir(path.parent)
    merged.to_parquet(path, index=False)
    return str(path)


@flow(name="forecast-live")
def forecast_live(
    config_path: str | None = None,
    areas: str = "all",
    *,
    force: bool = False,
    skip_download: bool = False,
    skip_schedule: bool = False,
) -> dict[str, Any]:
    """End-to-end live forecast for the dashboard snapshot."""
    settings = load_settings(config_path)
    store = ObjectStore()
    if not skip_schedule and not force and not should_run(settings.schedule_et):
        _log("outside schedule_et window; exiting")
        return {"skipped": True, "reason": "schedule"}

    selected = resolve_areas(areas)
    downloaded = download_runs_task(selected, config_path, skip_download)
    load_path = refresh_load_task(config_path, skip_download)
    load = pd.read_parquet(load_path)
    origin = next_origin()
    max_gap_hours = int(settings.max_gap.total_seconds() // 3600)
    history_days = int(settings.predict_history.days)
    history_failures = prediction_history_gaps(
        load,
        areas=selected,
        origin=origin,
        history_days=history_days,
        max_gap_hours=max_gap_hours,
    )
    if history_failures:
        _log(
            "skipping areas with prediction-history gaps "
            f"(max {max_gap_hours}h): {history_failures}"
        )
        selected = [code for code in selected if code not in history_failures]
        if not selected:
            raise ValueError(
                "all requested areas failed prediction-history checks: "
                f"{history_failures}"
            )
    aggregates = aggregates_for_areas(selected)
    load_through = (
        pd.to_datetime(load.loc[load["load"].notna(), "timestamp"]).max().isoformat()
    )
    weather_run = downloaded.get("run", "")
    if not force and already_published(store, weather_run=weather_run, load_through=load_through):
        _log("snapshot already published for this weather run and load watermark")
        return {"skipped": True, "reason": "idempotent", "weather_run": weather_run}

    inference, run_label = build_live_inference(selected, load, settings, origin)
    weather_run = run_label or weather_run
    children, aggs, sources = predict_all(
        selected,
        aggregates,
        inference,
        origin,
        settings,
        store=store,
    )
    try:
        latest = store.read_json("models/latest.json")
        model_version = str(latest.get("version", "local"))
    except FileNotFoundError:
        model_version = "local"

    catalog_areas = frozenset(area.code for area in LOAD_AREAS)
    snapshot_areas = frozenset(selected) if frozenset(selected) != catalog_areas else None
    published = publish_live_snapshot(
        children,
        aggs,
        load,
        origin,
        weather_run=weather_run,
        model_version=model_version,
        load_through=load_through,
        model_sources=sources,
        store=store,
        settings=settings,
        snapshot_load_areas=snapshot_areas,
    )
    _log(f"published live snapshot origin={origin.isoformat()} run={weather_run}")
    return {
        "skipped": False,
        "origin": origin.isoformat(),
        "weather_run": weather_run,
        "model_version": model_version,
        "path": published["path"],
        "model_sources": sources,
        "nodes": len(forecast_nodes()),
    }
