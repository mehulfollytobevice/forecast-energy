"""Join cached load-area files into the training table or the inference frame.

Training rows::

    python src/build_load_area_frame.py

Inference rows, after ``update_zone_weather.py --forecast``::

    python src/build_load_area_frame.py --inference
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pjm_forecasting.frame import build_inference_frame, build_load_area_frame, write_frame
from pjm_forecasting.ingest.openmeteo import build_forecast_weather_history
from pjm_forecasting.settings import load_settings
from pjm_forecasting.utils import load_config, project_root

JOIN_KEYS = ["timestamp_utc", "load_area"]


def _read_many(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    stacked = pd.concat(frames, ignore_index=True)
    if set(JOIN_KEYS).issubset(stacked.columns):
        stacked = stacked.drop_duplicates(subset=JOIN_KEYS, keep="last")
    return stacked


def read_load_cache(cache_root: Path) -> pd.DataFrame:
    """Read every monthly load-area parquet under ``cache_root/load_areas``."""
    paths = sorted((cache_root / "load_areas").glob("*/*.parquet"))
    return _read_many(paths)


def read_archive_weather(weather_root: Path) -> pd.DataFrame:
    """Read archive weather, skipping live ``forecast_`` files."""
    paths = [
        path
        for path in sorted(weather_root.glob("*/*.parquet"))
        if not path.name.startswith("forecast_")
    ]
    return _read_many(paths)


def read_forecast_weather(forecast_root: Path) -> pd.DataFrame:
    """Read the latest ``forecast_*.parquet`` in each load-area folder."""
    paths: list[Path] = []
    if not forecast_root.is_dir():
        return pd.DataFrame()
    for area_dir in sorted(path for path in forecast_root.iterdir() if path.is_dir()):
        files = sorted(area_dir.glob("forecast_*.parquet"))
        if files:
            paths.append(files[-1])
    return _read_many(paths)


def read_forecast_run_weather(processed_runs_dir: Path) -> pd.DataFrame:
    """Latest-available-run weather from processed versioned forecast runs."""
    paths = sorted(processed_runs_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no forecast-run tables under {processed_runs_dir}")
    frames = [build_forecast_weather_history(pd.read_parquet(path)) for path in paths]
    return pd.concat(frames, ignore_index=True)


def default_horizon_hours() -> int:
    hours = load_config().get("horizon", {}).get("prediction_length", 48)
    return int(hours)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join load-area load and Open-Meteo weather. "
            "--inference appends the forecast horizon with load set to null."
        )
    )
    parser.add_argument(
        "--load-root",
        default=None,
        help="Parent of load_areas/ (default data/raw/pjm).",
    )
    parser.add_argument(
        "--weather-root",
        default=None,
        help="Archive weather root (default data/raw/openmeteo/load_areas).",
    )
    parser.add_argument(
        "--forecast-root",
        default=None,
        help="Forecast weather root (default data/raw/openmeteo/load_areas).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Parquet path (default data/processed/load_area_hourly.parquet).",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Build the horizon frame instead of the training table.",
    )
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=None,
        help="Forecast hours to keep after the last metered hour.",
    )
    parser.add_argument(
        "--weather-source",
        choices=("archive", "forecast_runs"),
        default="archive",
        help="Training weather: Open-Meteo archive (default) or ECMWF single-run history.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = project_root()
    load_root = Path(args.load_root) if args.load_root else root / "data" / "raw" / "pjm"
    weather_root = (
        Path(args.weather_root)
        if args.weather_root
        else root / "data" / "raw" / "openmeteo" / "load_areas"
    )
    forecast_root = (
        Path(args.forecast_root)
        if args.forecast_root
        else root / "data" / "raw" / "openmeteo" / "load_areas"
    )
    settings = load_settings()
    weather_source = args.weather_source
    if args.inference:
        output = (
            Path(args.output)
            if args.output
            else root / "data" / "processed" / "load_area_inference.parquet"
        )
        horizon = args.horizon_hours
        if horizon is None:
            horizon = default_horizon_hours()
        history = build_load_area_frame(
            read_load_cache(load_root), read_archive_weather(weather_root)
        )
        forecast = read_forecast_weather(forecast_root)
        if forecast.empty:
            raise SystemExit(f"No forecast weather files under {forecast_root}")
        frame = build_inference_frame(history, forecast, horizon)
    else:
        if weather_source == "forecast_runs":
            output = (
                Path(args.output)
                if args.output
                else settings.forecast_frame_path()
            )
            weather = read_forecast_run_weather(settings.versioned_predictor_dir())
        else:
            output = (
                Path(args.output)
                if args.output
                else root / "data" / "processed" / "load_area_hourly.parquet"
            )
            weather = read_archive_weather(weather_root)
        frame = build_load_area_frame(read_load_cache(load_root), weather)
    write_frame(frame, output)
    print(f"wrote {len(frame)} rows -> {output} (weather_source={weather_source})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
