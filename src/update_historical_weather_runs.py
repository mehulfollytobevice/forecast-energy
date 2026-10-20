"""Download archived Open-Meteo single runs for each PJM load area.

Files land under ``data/raw/openmeteo/forecast_runs/{LOAD_AREA}/``. A run that
is already cached is skipped. The processed point-in-time table is written to
``data/processed/forecast_runs/{LOAD_AREA}.parquet``.

ECMWF IFS HRES runs start on 2024-03-14. Set ``OPEN_METEO_API_KEY`` in
``.env`` for the commercial customer API; leave it unset for the free tier.

One load area, one week::

    python src/update_historical_weather_runs.py \\
        --start 2026-01-01 --end 2026-01-08 --areas AECO --workers 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pjm_forecasting.areas import resolve_areas
from pjm_forecasting.ingest.openmeteo import (
    build_versioned_predictors,
    cap_run_workers,
    download_historical_runs,
)
from pjm_forecasting.settings import load_settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI for ``python src/update_historical_weather_runs.py``."""
    parser = argparse.ArgumentParser(
        description=(
            "Download ECMWF IFS HRES single runs from Open-Meteo and build "
            "point-in-time predictor tables. Uses OPEN_METEO_API_KEY when set."
        )
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Inclusive UTC start (default: archive start).",
    )
    parser.add_argument("--end", default=None, help="Exclusive UTC end (default: now).")
    parser.add_argument("--areas", default="all", help="all, or comma-separated load-area codes.")
    parser.add_argument("--workers", type=int, default=None, help="Parallel downloads (max 8).")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download runs that are already cached.",
    )
    parser.add_argument("--config", default=None, help="YAML config path.")
    parser.add_argument("--out-dir", default=None, help="Raw run directory.")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Do not write the processed per-area predictor parquet.",
    )
    return parser.parse_args(argv)


def _bound(value: str | None, *, default: pd.Timestamp) -> pd.Timestamp:
    if value is None:
        return default
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def main(argv: list[str] | None = None) -> int:
    """Download missing runs and, unless skipped, build the versioned tables."""
    args = parse_args(argv)
    settings = load_settings(args.config)
    try:
        areas = resolve_areas(args.areas)
        workers = cap_run_workers(
            args.workers if args.workers is not None else settings.run_workers
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    start = _bound(args.start, default=settings.archive_start)
    end = _bound(args.end, default=pd.Timestamp.now(tz="UTC"))
    out_dir = Path(args.out_dir) if args.out_dir else settings.forecast_run_dir()
    results = download_historical_runs(
        areas,
        start,
        end,
        out_dir=out_dir,
        model=settings.forecast_model,
        publication_delay=settings.publication_delay,
        cadence=settings.cadence,
        workers=workers,
        refresh=args.refresh,
        url=settings.forecast_url,
    )
    written = sum(result.status == "written" for result in results)
    skipped = sum(result.status == "skipped" for result in results)
    print(f"done: {written} written, {skipped} skipped", flush=True)
    if args.skip_build:
        return 0
    for code in areas:
        path = build_versioned_predictors(
            out_dir,
            code,
            settings.versioned_predictor_dir() / f"{code}.parquet",
            settings.publication_delay,
        )
        print(f"versioned {code} -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
