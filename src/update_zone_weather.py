"""Download hourly Open-Meteo weather for each PJM load area.

Archive backfill and the live forecast use the same hourly variables, units,
and ``timezone=UTC``. Saved ``timestamp`` values are US/Eastern. Files land under
``data/raw/openmeteo/load_areas/{LOAD_AREA}/``. ``--forecast`` writes
``forecast_{start}_{end}.parquet`` in that same folder.

The archive lags real time by about five days. Keep ``--end`` inside the
published archive.

Full history::

    python src/update_zone_weather.py --start 2018-01-01 --end 2024-12-31

One load area, a few Eastern hours::

    python src/update_zone_weather.py \\
        --start 2018-01-01T00:00 --end 2018-01-01T05:00 --areas AECO --workers 1

The next 48 hours from the forecast API, written beside the archive files::

    python src/update_zone_weather.py --forecast --areas AECO
"""

from __future__ import annotations

import argparse
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import requests

from pjm_forecasting.areas import BY_CODE, LoadArea, resolve_areas
from pjm_forecasting.ingest.openmeteo import apply_openmeteo_auth
from pjm_forecasting.schema import WEATHER_COLUMNS
from pjm_forecasting.utils import EASTERN, ensure_dir, load_config, project_root

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_TIMEZONE = "UTC"
HOURLY_FIELDS = ",".join(WEATHER_COLUMNS)
_RATE_LIMIT_ATTEMPTS = 30
_RATE_LIMIT_WAIT_S = 60.0
_REQUEST_TIMEOUT_S = 180
_rate_limit_lock = threading.Lock()
_rate_limited_until = 0.0

DEFAULT_WORKERS = 8
MAX_WORKERS = 16
DEFAULT_FORECAST_HOURS = 48
OUTPUT_COLUMNS = ["timestamp", "timestamp_utc", "load_area", *WEATHER_COLUMNS]


@dataclass(frozen=True)
class AreaResult:
    """One load-area file written or left untouched."""

    load_area: str
    path: Path
    status: Literal["written", "skipped"]
    rows: int | None = None


def default_out_dir() -> Path:
    """Archive parquet root, one folder per load area."""
    return project_root() / "data" / "raw" / "openmeteo" / "load_areas"


def default_forecast_dir() -> Path:
    """Live-forecast files live next to the archive, named ``forecast_*.parquet``."""
    return default_out_dir()


def default_forecast_hours() -> int:
    """``horizon.prediction_length`` from config, otherwise 48."""
    try:
        hours = load_config().get("horizon", {}).get("prediction_length")
    except (OSError, TypeError, ValueError):
        hours = None
    if not hours:
        return DEFAULT_FORECAST_HOURS
    return int(hours)


def cap_workers(workers: int) -> int:
    """Clamp a worker count to 1–16. The default is 8."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f"workers must be an integer >= 1, got {workers!r}")
    return min(workers, MAX_WORKERS)


def _is_date_only(value: str) -> bool:
    text = str(value).strip()
    return "T" not in text and ":" not in text and " " not in text


def _parse_eastern(value: str, *, is_end: bool) -> pd.Timestamp:
    text = str(value).strip()
    if not text:
        raise ValueError("start and end are required")
    stamp = pd.Timestamp(text)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(EASTERN)
    else:
        stamp = stamp.tz_convert(EASTERN)
    if not is_end:
        return stamp
    if _is_date_only(text):
        return stamp + pd.Timedelta(days=1)
    return stamp + pd.Timedelta(hours=1)


def eastern_window(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return ``[start, end)`` in US/Eastern.

    A date-only ``end`` includes that whole US/Eastern calendar day. A timestamp
    ``end`` includes that hour.
    """
    start_ts = _parse_eastern(start, is_end=False)
    end_exclusive = _parse_eastern(end, is_end=True)
    if end_exclusive <= start_ts:
        raise ValueError(f"end {end!r} must be on or after start {start!r}")
    return start_ts, end_exclusive


def _date_token(value: str) -> str:
    text = str(value).strip()
    stamp = pd.Timestamp(text)
    if _is_date_only(text):
        return f"{stamp:%Y%m%d}"
    return f"{stamp:%Y%m%dT%H%M}"


def area_output_path(out_dir: Path | str, load_area: str, start: str, end: str) -> Path:
    """``{out_dir}/{LOAD_AREA}/{start}_{end}.parquet``."""
    return Path(out_dir) / load_area / f"{_date_token(start)}_{_date_token(end)}.parquet"


def archive_date_range(start: pd.Timestamp, end_exclusive: pd.Timestamp) -> tuple[str, str]:
    """Inclusive UTC calendar dates covering the Eastern window ``[start, end)``."""
    if end_exclusive <= start:
        raise ValueError("end must be after start")
    start_utc = start.tz_convert("UTC")
    last_hour = (end_exclusive - pd.Timedelta(hours=1)).tz_convert("UTC")
    return f"{start_utc:%Y-%m-%d}", f"{last_hour:%Y-%m-%d}"


def _common_params(area: LoadArea) -> dict[str, Any]:
    return {
        "latitude": area.lat,
        "longitude": area.lon,
        "hourly": HOURLY_FIELDS,
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        "timezone": OPEN_METEO_TIMEZONE,
    }


def archive_params(
    area: LoadArea, start: pd.Timestamp, end_exclusive: pd.Timestamp
) -> dict[str, Any]:
    """Query string for the historical archive. The same variables as the forecast."""
    start_date, end_date = archive_date_range(start, end_exclusive)
    params = _common_params(area)
    params["start_date"] = start_date
    params["end_date"] = end_date
    return params


def forecast_params(area: LoadArea, forecast_hours: int, past_days: int) -> dict[str, Any]:
    """Query string for the live forecast horizon."""
    if forecast_hours < 1:
        raise ValueError(f"forecast_hours must be >= 1, got {forecast_hours}")
    if past_days < 0:
        raise ValueError(f"past_days must be >= 0, got {past_days}")
    params = _common_params(area)
    params["forecast_hours"] = int(forecast_hours)
    if past_days:
        params["past_days"] = int(past_days)
    return params


def _response_payload(load_area: str, response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Open-Meteo returned a non-JSON response for load area {load_area} "
            f"(HTTP {response.status_code})"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Open-Meteo returned an unexpected payload for load area {load_area} "
            f"(HTTP {response.status_code})"
        )
    if response.status_code != 200 or payload.get("error"):
        reason = payload.get("reason") or str(payload)[:300]
        raise RuntimeError(
            f"Open-Meteo request failed for load area {load_area} "
            f"(HTTP {response.status_code}): {reason}"
        )
    return payload


def _rate_limit_wait_seconds(response_text: str) -> float:
    """How long to pause after HTTP 429. The archive API asks for one minute."""
    text = response_text.lower()
    minutes = re.search(r"(\d+)\s*minute", text)
    if minutes:
        return float(minutes.group(1)) * 60.0
    if "minute" in text:
        return 60.0
    seconds = re.search(r"(\d+)\s*second", text)
    if seconds:
        return float(seconds.group(1))
    return _RATE_LIMIT_WAIT_S


def _wait_out_rate_limit() -> None:
    """Sleep if another worker was told to wait out the minute cap."""
    with _rate_limit_lock:
        delay = _rate_limited_until - time.monotonic()
    if delay > 0:
        time.sleep(delay)


def _pause_for_rate_limit(load_area: str, response_text: str, attempt: int) -> None:
    """Hold every worker until the minutely cap can accept another request."""
    global _rate_limited_until
    wait_s = _rate_limit_wait_seconds(response_text)
    with _rate_limit_lock:
        until = time.monotonic() + wait_s
        if until > _rate_limited_until:
            _rate_limited_until = until
        delay = max(0.0, _rate_limited_until - time.monotonic())
    print(
        f"rate limited {load_area}; waiting {delay:.0f}s (attempt {attempt + 1})",
        flush=True,
    )


def request_open_meteo(load_area: str, url: str, params: dict[str, Any]) -> dict[str, Any]:
    """GET one load area. HTTP 429 waits out the minutely cap, then retries."""
    url, params = apply_openmeteo_auth(url, params)
    response: requests.Response | None = None
    for attempt in range(_RATE_LIMIT_ATTEMPTS):
        _wait_out_rate_limit()
        try:
            response = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT_S)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Open-Meteo request failed for load area {load_area}: {exc}"
            ) from exc
        if response.status_code != 429:
            return _response_payload(load_area, response)
        if attempt + 1 == _RATE_LIMIT_ATTEMPTS:
            break
        _pause_for_rate_limit(load_area, response.text[:300], attempt)
    detail = ""
    if response is not None:
        detail = f" HTTP {response.status_code}: {response.text[:300]}"
    raise RuntimeError(
        f"Open-Meteo rate limited load area {load_area} after {_RATE_LIMIT_ATTEMPTS} attempts."
        f"{detail}"
    )


def _localize_times(times: list[str]) -> pd.Series:
    """Treat Open-Meteo timestamps as UTC instants and return US/Eastern.

    UTC keeps both fall-back hours. A local clock label such as
    ``2018-11-04T01:00`` appears only once and cannot be assigned to a fold.
    """
    stamps = pd.to_datetime(pd.Series(times))
    if stamps.dt.tz is None:
        return stamps.dt.tz_localize("UTC").dt.tz_convert(EASTERN)
    return stamps.dt.tz_convert(EASTERN)


def hourly_frame_from_payload(payload: dict[str, Any], load_area: str) -> pd.DataFrame:
    """Map one archive or forecast payload onto the shared weather columns."""
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise RuntimeError(f"Open-Meteo returned no hourly data for load area {load_area}")
    times = hourly.get("time")
    if not isinstance(times, list) or not times:
        raise RuntimeError(f"Open-Meteo returned no hourly data for load area {load_area}")
    missing = [name for name in WEATHER_COLUMNS if name not in hourly]
    if missing:
        raise RuntimeError(
            f"Open-Meteo response for load area {load_area} is missing fields: {missing}"
        )
    n = len(times)
    local = _localize_times(times)
    data: dict[str, Any] = {
        "timestamp": local,
        "timestamp_utc": local.dt.tz_convert("UTC"),
    }
    for name in WEATHER_COLUMNS:
        values = hourly[name]
        if not isinstance(values, list) or len(values) != n:
            raise RuntimeError(
                f"Open-Meteo hourly {name} for load area {load_area} does not match the time axis"
            )
        data[name] = pd.to_numeric(pd.Series(values), errors="coerce")
    return pd.DataFrame(data)


def to_area_weather(
    raw: pd.DataFrame,
    load_area: str,
    start: pd.Timestamp | None = None,
    end_exclusive: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Keep the requested window and write the shared weather schema.

    ``shortwave_radiation`` negatives are clipped to 0. A missing ``start``
    keeps every hour in ``raw``, which is how a forecast payload is stored.
    """
    missing = [col for col in ("timestamp_utc", *WEATHER_COLUMNS) if col not in raw.columns]
    if missing:
        raise ValueError(f"Open-Meteo extract for {load_area} is missing fields: {missing}")
    work = raw.copy()
    work["timestamp_utc"] = pd.to_datetime(work["timestamp_utc"], utc=True)
    for column in WEATHER_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["shortwave_radiation"] = work["shortwave_radiation"].clip(lower=0)
    if start is not None and end_exclusive is not None:
        start_utc = start.tz_convert("UTC")
        end_utc = end_exclusive.tz_convert("UTC")
        mask = (work["timestamp_utc"] >= start_utc) & (work["timestamp_utc"] < end_utc)
        work = work.loc[mask]
    if work.empty:
        raise RuntimeError(
            f"Open-Meteo returned no hours for load area {load_area} in the requested window"
        )
    work["load_area"] = load_area
    work["timestamp"] = work["timestamp_utc"].dt.tz_convert(EASTERN)
    work = (
        work.drop_duplicates(subset=["timestamp_utc"])
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )
    return work.loc[:, OUTPUT_COLUMNS]


def fetch_area_weather(
    load_area: str,
    start: str | None = None,
    end: str | None = None,
    *,
    forecast: bool = False,
    forecast_hours: int = DEFAULT_FORECAST_HOURS,
    past_days: int = 0,
) -> pd.DataFrame:
    """Read one load area from the archive, or from the live forecast API."""
    code = load_area.strip().upper()
    if code not in BY_CODE:
        raise ValueError(f"Unknown load area {load_area!r}")
    area = BY_CODE[code]
    if forecast:
        params = forecast_params(area, forecast_hours, past_days)
        url = OPEN_METEO_FORECAST_URL
        window = None
    else:
        if start is None or end is None:
            raise ValueError("start and end are required for an archive download")
        start_ts, end_exclusive = eastern_window(start, end)
        params = archive_params(area, start_ts, end_exclusive)
        url = OPEN_METEO_ARCHIVE_URL
        window = (start_ts, end_exclusive)
    print(
        f"fetch {code} ({area.city}) lat={area.lat:.4f} lon={area.lon:.4f} "
        f"{'forecast' if forecast else 'archive'}",
        flush=True,
    )
    payload = request_open_meteo(code, url, params)
    raw = hourly_frame_from_payload(payload, code)
    if window is None:
        return to_area_weather(raw, code)
    return to_area_weather(raw, code, window[0], window[1])


def _download_one(task: dict[str, Any]) -> AreaResult:
    code = task["load_area"]
    path = Path(task["path"])
    started = time.perf_counter()
    try:
        frame = fetch_area_weather(
            code,
            task.get("start"),
            task.get("end"),
            forecast=bool(task.get("forecast")),
            forecast_hours=int(task.get("forecast_hours") or DEFAULT_FORECAST_HOURS),
            past_days=int(task.get("past_days") or 0),
        )
    except Exception as exc:
        raise RuntimeError(f"Open-Meteo download failed for load area {code}: {exc}") from exc
    if task.get("forecast"):
        start_token = f"{frame['timestamp'].iloc[0]:%Y%m%dT%H%M}"
        end_token = f"{frame['timestamp'].iloc[-1]:%Y%m%dT%H%M}"
        path = path.parent / f"forecast_{start_token}_{end_token}.parquet"
    ensure_dir(path.parent)
    frame.to_parquet(path, index=False)
    elapsed = time.perf_counter() - started
    print(f"wrote {code} {len(frame)} hours in {elapsed:.1f}s -> {path}", flush=True)
    return AreaResult(load_area=code, path=path, status="written", rows=len(frame))


def download_areas(
    areas: list[str],
    start: str | None,
    end: str | None,
    *,
    out_dir: Path | str,
    workers: int = DEFAULT_WORKERS,
    refresh: bool = False,
    forecast: bool = False,
    forecast_hours: int = DEFAULT_FORECAST_HOURS,
    past_days: int = 0,
) -> list[AreaResult]:
    """Fetch load areas in parallel, skipping archive files that already exist."""
    codes = resolve_areas(",".join(areas))
    workers = cap_workers(workers)
    out_dir = Path(out_dir)
    source, _ = apply_openmeteo_auth(
        OPEN_METEO_FORECAST_URL if forecast else OPEN_METEO_ARCHIVE_URL,
        {},
    )
    print(f"source {source}", flush=True)
    if not forecast:
        if start is None or end is None:
            raise ValueError("start and end are required for an archive download")
        start_ts, end_exclusive = eastern_window(start, end)
        print(
            f"window {start_ts.isoformat()} to {end_exclusive.isoformat()} "
            f"(end exclusive) workers={workers}",
            flush=True,
        )

    by_area: dict[str, AreaResult] = {}
    pending: list[dict[str, Any]] = []
    for code in codes:
        if forecast:
            path = out_dir / code / "forecast.parquet"
        else:
            path = area_output_path(out_dir, code, start or "", end or "")
        if path.is_file() and not refresh and not forecast:
            print(f"skip {code} cached at {path}", flush=True)
            by_area[code] = AreaResult(load_area=code, path=path, status="skipped")
            continue
        pending.append(
            {
                "load_area": code,
                "path": str(path),
                "start": start,
                "end": end,
                "forecast": forecast,
                "forecast_hours": forecast_hours,
                "past_days": past_days,
            }
        )

    if pending:
        pool_size = min(workers, len(pending))
        with ThreadPoolExecutor(max_workers=pool_size) as pool:
            futures = [pool.submit(_download_one, task) for task in pending]
            for future in as_completed(futures):
                result = future.result()
                by_area[result.load_area] = result

    ordered = [by_area[code] for code in codes]
    written = sum(result.status == "written" for result in ordered)
    skipped = sum(result.status == "skipped" for result in ordered)
    print(f"done: {written} written, {skipped} skipped", flush=True)
    return ordered


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI for ``python src/update_zone_weather.py``."""
    parser = argparse.ArgumentParser(
        description=(
            "Download hourly Open-Meteo weather for each PJM load area. "
            "The archive and the live forecast use the same variables. "
            "No API key. --start and --end are required unless --forecast is set."
        ),
        epilog=(
            "examples:\n"
            "  python src/update_zone_weather.py --start 2018-01-01 --end 2024-12-31\n"
            "  python src/update_zone_weather.py --start 2018-01-01T00:00 "
            "--end 2018-01-01T05:00 --areas AECO --workers 1\n"
            "  python src/update_zone_weather.py --forecast\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Inclusive start (YYYY-MM-DD or YYYY-MM-DDTHH:MM, US/Eastern).",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Inclusive end date (YYYY-MM-DD) or hour (YYYY-MM-DDTHH:MM), US/Eastern.",
    )
    parser.add_argument(
        "--areas",
        default="all",
        help="all (default) or a comma-separated list of load-area codes.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel HTTP requests (default {DEFAULT_WORKERS}, max {MAX_WORKERS}).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default data/raw/openmeteo/load_areas).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download even if that load area's archive parquet already exists.",
    )
    parser.add_argument(
        "--forecast",
        action="store_true",
        help="Pull the live forecast horizon instead of the historical archive.",
    )
    parser.add_argument(
        "--forecast-hours",
        type=int,
        default=None,
        help=f"Hours to request from the forecast API (default {DEFAULT_FORECAST_HOURS}).",
    )
    parser.add_argument(
        "--past-days",
        type=int,
        default=0,
        help="Recent days to include with --forecast so the horizon file overlaps history.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Parse args and download. Returns a process status code."""
    args = parse_args(argv)
    try:
        areas = resolve_areas(args.areas)
        workers = cap_workers(args.workers)
        if not args.forecast and (not args.start or not args.end):
            raise ValueError("start and end are required unless --forecast is set")
        forecast_hours = (
            args.forecast_hours if args.forecast_hours is not None else default_forecast_hours()
        )
        if args.forecast and forecast_hours < 1:
            raise ValueError(f"forecast_hours must be >= 1, got {forecast_hours}")
        if args.past_days < 0:
            raise ValueError(f"past_days must be >= 0, got {args.past_days}")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if workers != args.workers:
        print(f"capping --workers from {args.workers} to {workers}", flush=True)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.forecast:
        out_dir = default_forecast_dir()
    else:
        out_dir = default_out_dir()
    print(
        f"areas={len(areas)} forecast={args.forecast} "
        f"workers={workers} refresh={args.refresh} out={out_dir}",
        flush=True,
    )
    download_areas(
        areas,
        args.start,
        args.end,
        out_dir=out_dir,
        workers=workers,
        refresh=args.refresh,
        forecast=args.forecast,
        forecast_hours=forecast_hours,
        past_days=args.past_days,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
