"""Open-Meteo Single Runs downloads for leakage-safe weather.

Each file is one ECMWF IFS HRES initialisation for one load area. Existing
files are left in place. ``available_at`` is the run initialisation plus the
configured publication delay, which is the earliest time that run may be used.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse, urlunparse

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from pjm_forecasting.areas import BY_CODE, LoadArea, resolve_areas
from pjm_forecasting.schema import WEATHER_COLUMNS
from pjm_forecasting.utils import EASTERN, ensure_dir, project_root


SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
OPEN_METEO_TIMEZONE = "UTC"
# Free host -> commercial customer host (same path). Single Runs / archive have
# dedicated customer subdomains; the main forecast uses customer-api.
_CUSTOMER_HOST_REWRITE = {
    "api.open-meteo.com": "customer-api.open-meteo.com",
    "archive-api.open-meteo.com": "customer-archive-api.open-meteo.com",
    "single-runs-api.open-meteo.com": "customer-single-runs-api.open-meteo.com",
}
HOURLY_FIELDS = ",".join(WEATHER_COLUMNS)
MAX_WORKERS = 8
_RATE_LIMIT_ATTEMPTS = 6
_RATE_LIMIT_WAIT_S = 60.0
_REQUEST_TIMEOUT_S = 180
_HOURLY_LIMIT_LOCK = threading.Lock()
_HOURLY_LIMIT_UNTIL = 0.0


def openmeteo_api_key() -> str | None:
    """Commercial key from ``OPEN_METEO_API_KEY``, or ``None`` for the free tier."""
    load_dotenv(project_root() / ".env")
    key = os.getenv("OPEN_METEO_API_KEY", "").strip()
    return key or None


def apply_openmeteo_auth(
    url: str,
    params: dict[str, Any],
    *,
    api_key: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Attach ``apikey`` and rewrite free hosts to customer hosts when a key is set.

    Free-tier hosts stay unchanged when no key is present. With a commercial
    key, requests go to the matching ``customer-*-api.open-meteo.com`` host.
    Single Runs and Historical Archive need Professional or Enterprise.
    """
    key = openmeteo_api_key() if api_key is None else api_key.strip() or None
    out_params = dict(params)
    if not key:
        return url, out_params
    out_params["apikey"] = key
    parsed = urlparse(url)
    customer_host = _CUSTOMER_HOST_REWRITE.get(parsed.hostname or "")
    if customer_host:
        url = urlunparse(parsed._replace(netloc=customer_host))
    return url, out_params

RAW_COLUMNS = [
    "timestamp",
    "timestamp_utc",
    "load_area",
    *WEATHER_COLUMNS,
    "run_initialized_at",
    "available_at",
]
VERSIONED_COLUMNS = [
    "timestamp",
    "available_at",
    "run_initialized_at",
    "load_area",
    *WEATHER_COLUMNS,
]


@dataclass(frozen=True)
class RunResult:
    """One cached model run."""

    load_area: str
    run_initialized_at: pd.Timestamp
    path: Path
    status: Literal["written", "skipped"]
    rows: int | None = None


def cap_run_workers(workers: int) -> int:
    """Clamp a worker count to 1–8."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f"workers must be an integer >= 1, got {workers!r}")
    return min(workers, MAX_WORKERS)


def forecast_run_times(
    start: pd.Timestamp,
    end: pd.Timestamp,
    cadence: pd.Timedelta,
) -> list[pd.Timestamp]:
    """UTC initialisation times on ``[start, end)``."""
    start_utc = pd.Timestamp(start).tz_convert("UTC")
    end_utc = pd.Timestamp(end).tz_convert("UTC")
    if end_utc <= start_utc:
        raise ValueError("end must be after start")
    if cadence <= pd.Timedelta(0):
        raise ValueError("cadence must be positive")
    times = pd.date_range(start_utc, end_utc, freq=cadence, inclusive="left")
    return list(times)


def run_token(run_initialized_at: pd.Timestamp) -> str:
    """File token for one UTC initialisation."""
    stamp = pd.Timestamp(run_initialized_at).tz_convert("UTC")
    return f"{stamp:%Y%m%dT%H%MZ}"


def run_output_path(out_dir: Path | str, load_area: str, run_initialized_at: pd.Timestamp) -> Path:
    """``{out_dir}/{LOAD_AREA}/{YYYYMMDDTHHMMZ}.parquet``."""
    return Path(out_dir) / load_area / f"{run_token(run_initialized_at)}.parquet"


def run_unavailable_marker_path(path: Path | str) -> Path:
    """Sidecar marker when Open-Meteo has no archive for this run."""
    path = Path(path)
    return path.with_suffix(path.suffix + ".unavailable")


def _seconds_until_next_hour(now: float | None = None) -> float:
    """Seconds until the next UTC hour boundary, plus a short buffer."""
    clock = time.time() if now is None else now
    return max(5.0, 3600.0 - (clock % 3600.0) + 5.0)


def _wait_for_hourly_quota(load_area: str) -> None:
    """Block all workers until the shared hourly quota window resets."""
    global _HOURLY_LIMIT_UNTIL
    with _HOURLY_LIMIT_LOCK:
        wait_until = max(_HOURLY_LIMIT_UNTIL, time.time() + _seconds_until_next_hour())
        _HOURLY_LIMIT_UNTIL = wait_until
        wait_s = max(0.0, wait_until - time.time())
    if wait_s > 0:
        print(
            f"Open-Meteo hourly limit hit ({load_area}); "
            f"sleeping {wait_s:.0f}s then resuming...",
            flush=True,
        )
        time.sleep(wait_s)


def single_run_params(
    area: LoadArea,
    run_initialized_at: pd.Timestamp,
    model: str,
) -> dict[str, Any]:
    """Query string for one archived model run."""
    stamp = pd.Timestamp(run_initialized_at).tz_convert("UTC")
    return {
        "latitude": area.lat,
        "longitude": area.lon,
        "hourly": HOURLY_FIELDS,
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        "timezone": OPEN_METEO_TIMEZONE,
        "models": model,
        "run": f"{stamp:%Y-%m-%dT%H:%M}",
    }


class RunNotAvailableError(RuntimeError):
    """Open-Meteo has no archive file for this model initialisation."""


def _response_reason(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:300]
    if isinstance(payload, dict):
        return str(payload.get("reason") or payload)[:300]
    return str(payload)[:300]


def _is_hourly_rate_limit(detail: str) -> bool:
    lower = detail.lower()
    return "hourly" in lower or "next hour" in lower


def _is_run_unavailable(status_code: int, reason: str) -> bool:
    return status_code == 400 and "not available" in reason.lower()


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    # #region agent log
    import json

    try:
        payload = {
            "sessionId": "bbf3e1",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
            "runId": "pre-fix",
        }
        with open(
            "/Users/mehuljain/Documents/Projects/forecast-energy/.cursor/debug-bbf3e1.log",
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(json.dumps(payload) + "\n")
    except Exception:
        return
    # #endregion


def _redact_url(url: str) -> str:
    # #region agent log
    from urllib.parse import parse_qsl, urlsplit, urlunsplit

    parts = urlsplit(url)
    query = "&".join(
        f"{key}=REDACTED" if key.lower() == "apikey" else f"{key}={value}"
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
    )
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))
    # #endregion


def _response_payload(load_area: str, response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        # #region agent log
        raw = response.content or b""
        headers = {key.lower(): value for key, value in response.headers.items()}
        _debug_log(
            "A-E",
            "openmeteo.py:_response_payload",
            "non-json open-meteo response",
            {
                "load_area": load_area,
                "status": response.status_code,
                "final_url": _redact_url(response.url),
                "content_type": headers.get("content-type"),
                "content_encoding": headers.get("content-encoding"),
                "content_length_header": headers.get("content-length"),
                "server": headers.get("server"),
                "body_bytes": len(raw),
                "text_len": len(response.text or ""),
                "body_prefix": raw[:180].decode("utf-8", errors="replace"),
                "redirects": [
                    {"status": item.status_code, "url": _redact_url(item.url)}
                    for item in response.history
                ],
            },
        )
        # #endregion
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
        if _is_run_unavailable(response.status_code, str(reason)):
            raise RunNotAvailableError(
                f"Open-Meteo request failed for load area {load_area} "
                f"(HTTP {response.status_code}): {reason}"
            )
        raise RuntimeError(
            f"Open-Meteo request failed for load area {load_area} "
            f"(HTTP {response.status_code}): {reason}"
        )
    return payload


def request_single_run(
    load_area: str,
    url: str,
    params: dict[str, Any],
    *,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """GET one run.

    Transient connection errors and short 429s retry. An hourly API cap waits
    until the next hour and retries the same request.
    """
    getter = session.get if session is not None else requests.get
    url, params = apply_openmeteo_auth(url, params)
    response: requests.Response | None = None
    last_connection_error: requests.RequestException | None = None
    attempt = 0
    while attempt < _RATE_LIMIT_ATTEMPTS:
        try:
            response = getter(url, params=params, timeout=_REQUEST_TIMEOUT_S)
            last_connection_error = None
        except requests.RequestException as exc:
            last_connection_error = exc
            attempt += 1
            if attempt >= _RATE_LIMIT_ATTEMPTS:
                break
            time.sleep(min(5.0 * attempt, 30.0))
            continue
        if response.status_code != 429:
            return _response_payload(load_area, response)
        detail = _response_reason(response)
        if _is_hourly_rate_limit(detail):
            _wait_for_hourly_quota(load_area)
            continue
        attempt += 1
        if attempt >= _RATE_LIMIT_ATTEMPTS:
            break
        time.sleep(_RATE_LIMIT_WAIT_S)
    if last_connection_error is not None:
        raise RuntimeError(
            f"Open-Meteo request failed for load area {load_area} after "
            f"{_RATE_LIMIT_ATTEMPTS} connection attempts: {last_connection_error}"
        ) from last_connection_error
    detail = ""
    if response is not None:
        detail = f" HTTP {response.status_code}: {response.text[:300]}"
    raise RuntimeError(
        f"Open-Meteo rate limited load area {load_area} after "
        f"{_RATE_LIMIT_ATTEMPTS} attempts.{detail}"
    )


def frame_from_single_run(
    payload: dict[str, Any],
    load_area: str,
    run_initialized_at: pd.Timestamp,
    publication_delay: pd.Timedelta,
) -> pd.DataFrame:
    """Map one Single Runs payload onto the cached run schema."""
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
    stamps = pd.to_datetime(pd.Series(times))
    if stamps.dt.tz is None:
        utc = stamps.dt.tz_localize("UTC")
    else:
        utc = stamps.dt.tz_convert("UTC")
    initialized = pd.Timestamp(run_initialized_at).tz_convert("UTC")
    available = initialized + publication_delay
    data: dict[str, Any] = {
        "timestamp": utc.dt.tz_convert(EASTERN),
        "timestamp_utc": utc,
        "load_area": load_area,
        "run_initialized_at": pd.Series([initialized] * n),
        "available_at": pd.Series([available] * n),
    }
    for name in WEATHER_COLUMNS:
        values = hourly[name]
        if not isinstance(values, list) or len(values) != n:
            raise RuntimeError(
                f"Open-Meteo hourly {name} for load area {load_area} does not match the time axis"
            )
        data[name] = pd.to_numeric(pd.Series(values), errors="coerce")
    frame = pd.DataFrame(data)
    frame["shortwave_radiation"] = frame["shortwave_radiation"].clip(lower=0)
    frame = frame.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    return frame.loc[:, RAW_COLUMNS].reset_index(drop=True)


def download_historical_runs(
    areas: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    out_dir: Path | str,
    model: str,
    publication_delay: pd.Timedelta,
    cadence: pd.Timedelta,
    workers: int = 4,
    refresh: bool = False,
    url: str = SINGLE_RUNS_URL,
    now: pd.Timestamp | None = None,
) -> list[RunResult]:
    """Download missing runs. A run is requested only after its publication time."""
    codes = resolve_areas(",".join(areas))
    workers = cap_run_workers(workers)
    out_dir = Path(out_dir)
    clock = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    runs = [
        run for run in forecast_run_times(start, end, cadence) if run + publication_delay <= clock
    ]
    pending: list[tuple[str, pd.Timestamp, Path]] = []
    results: dict[tuple[str, pd.Timestamp], RunResult] = {}
    for code in codes:
        for run in runs:
            path = run_output_path(out_dir, code, run)
            marker = run_unavailable_marker_path(path)
            if not refresh and (path.is_file() or marker.is_file()):
                results[(code, run)] = RunResult(
                    load_area=code,
                    run_initialized_at=run,
                    path=path,
                    status="skipped",
                )
                continue
            pending.append((code, run, path))


    def _one(task: tuple[str, pd.Timestamp, Path]) -> RunResult:
        code, run, path = task
        area = BY_CODE[code]
        params = single_run_params(area, run, model)
        try:
            payload = request_single_run(code, url, params)
        except RunNotAvailableError as exc:
            marker = run_unavailable_marker_path(path)
            ensure_dir(path.parent)
            marker.write_text(str(exc)[:500] + "\n", encoding="utf-8")
            return RunResult(
                load_area=code,
                run_initialized_at=run,
                path=path,
                status="skipped",
            )
        frame = frame_from_single_run(payload, code, run, publication_delay)
        ensure_dir(path.parent)
        frame.to_parquet(path, index=False)
        return RunResult(
            load_area=code,
            run_initialized_at=run,
            path=path,
            status="written",
            rows=len(frame),
        )

    if pending:
        max_workers = min(workers, len(pending))
        pool = ThreadPoolExecutor(max_workers=max_workers)
        futures = [pool.submit(_one, task) for task in pending]
        shutdown_mode = "wait"
        try:
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception:
                    shutdown_mode = "cancel"
                    for pending_future in futures:
                        pending_future.cancel()
                    raise
                results[(result.load_area, result.run_initialized_at)] = result
        finally:
            pool.shutdown(wait=shutdown_mode == "wait", cancel_futures=True)

    ordered: list[RunResult] = []
    for code in codes:
        for run in runs:
            ordered.append(results[(code, run)])
    return ordered


def build_versioned_predictors(
    raw_dir: Path | str,
    load_area: str,
    out_path: Path | str,
    publication_delay: pd.Timedelta,
) -> Path:
    """Stack cached runs into one point-in-time predictor table.

    ``available_at`` is recomputed from ``run_initialized_at`` so the processed
    file follows the delay in the current config.
    """
    code = load_area.strip().upper()
    if code not in BY_CODE:
        raise ValueError(f"Unknown load area {load_area!r}")
    files = sorted(Path(raw_dir).glob(f"{code}/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no forecast runs cached for {code} under {raw_dir}")
    frames = [pd.read_parquet(path) for path in files]
    stacked = pd.concat(frames, ignore_index=True)
    stacked["timestamp"] = pd.to_datetime(stacked["timestamp"])
    if stacked["timestamp"].dt.tz is None:
        stacked["timestamp"] = stacked["timestamp"].dt.tz_localize(EASTERN)
    else:
        stacked["timestamp"] = stacked["timestamp"].dt.tz_convert(EASTERN)
    stacked["run_initialized_at"] = pd.to_datetime(stacked["run_initialized_at"], utc=True)
    stacked["available_at"] = stacked["run_initialized_at"] + publication_delay
    stacked["load_area"] = code
    for column in WEATHER_COLUMNS:
        stacked[column] = pd.to_numeric(stacked[column], errors="coerce")
    # Open-Meteo single runs often leave the last ~day of the horizon null.
    # Those hours are unusable as predictors; drop them before the quality gate.
    weather_arr = stacked.loc[:, list(WEATHER_COLUMNS)].to_numpy(dtype=float)
    stacked = stacked.loc[np.isfinite(weather_arr).all(axis=1)]
    stacked = (
        stacked.drop_duplicates(subset=["timestamp", "run_initialized_at", "load_area"])
        .sort_values(["run_initialized_at", "timestamp"])
        .reset_index(drop=True)
    )
    destination = Path(out_path)
    ensure_dir(destination.parent)
    stacked.loc[:, VERSIONED_COLUMNS].to_parquet(destination, index=False)
    return destination


def build_forecast_weather_history(
    versioned: pd.DataFrame,
    *,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """One weather row per valid hour from the latest run already published.

    For each ``timestamp``, keeps the newest ``available_at`` that is at or
    before that hour (or ``as_of`` when given). Output columns match the
    archive weather table so :func:`build_load_area_frame` can join it.
    """
    if versioned.empty:
        raise ValueError("versioned predictors are empty")
    required = ["timestamp", "available_at", "load_area", *WEATHER_COLUMNS]
    missing = [column for column in required if column not in versioned.columns]
    if missing:
        raise ValueError(f"versioned predictors missing columns: {missing}")
    work = versioned.copy()
    stamps = pd.to_datetime(work["timestamp"])
    if stamps.dt.tz is None:
        stamps = stamps.dt.tz_localize(EASTERN)
    else:
        stamps = stamps.dt.tz_convert(EASTERN)
    work["timestamp"] = stamps
    available = pd.to_datetime(work["available_at"], utc=True)
    work["available_at"] = available
    work["load_area"] = work["load_area"].astype(str).str.strip().str.upper()
    if as_of is not None:
        moment = pd.Timestamp(as_of)
        if moment.tzinfo is None:
            moment = moment.tz_localize("UTC")
        else:
            moment = moment.tz_convert("UTC")
        work = work.loc[available <= moment].copy()
    else:
        # Per valid hour: only runs published by that hour may contribute.
        stamp_utc = work["timestamp"].dt.tz_convert("UTC")
        work = work.loc[available <= stamp_utc].copy()
    if work.empty:
        raise ValueError("no forecast runs were published for the requested hours")
    work = work.sort_values(["load_area", "timestamp", "available_at"])
    latest = work.groupby(["load_area", "timestamp"], as_index=False, sort=False).tail(1)
    latest["timestamp_utc"] = latest["timestamp"].dt.tz_convert("UTC")
    for column in WEATHER_COLUMNS:
        latest[column] = pd.to_numeric(latest[column], errors="coerce")
    keep = ["timestamp", "timestamp_utc", "load_area", *WEATHER_COLUMNS]
    return latest.loc[:, keep].sort_values(["load_area", "timestamp_utc"]).reset_index(drop=True)


def latest_run_weather(
    versioned: pd.DataFrame,
    *,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Weather from the single most recent published run (live horizon)."""
    required = ["timestamp", "available_at", "run_initialized_at", "load_area", *WEATHER_COLUMNS]
    missing = [column for column in required if column not in versioned.columns]
    if missing:
        raise ValueError(f"versioned predictors missing columns: {missing}")
    work = versioned.copy()
    available = pd.to_datetime(work["available_at"], utc=True)
    work["available_at"] = available
    moment = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    if moment.tzinfo is None:
        moment = moment.tz_localize("UTC")
    else:
        moment = moment.tz_convert("UTC")
    visible = work.loc[available <= moment].copy()
    if visible.empty:
        raise ValueError("no forecast run is published yet")
    run_times = pd.to_datetime(visible["run_initialized_at"], utc=True)
    latest_run = run_times.max()
    piece = visible.loc[run_times == latest_run].copy()
    stamps = pd.to_datetime(piece["timestamp"])
    if stamps.dt.tz is None:
        stamps = stamps.dt.tz_localize(EASTERN)
    else:
        stamps = stamps.dt.tz_convert(EASTERN)
    piece["timestamp"] = stamps
    piece["timestamp_utc"] = stamps.dt.tz_convert("UTC")
    piece["load_area"] = piece["load_area"].astype(str).str.strip().str.upper()
    for column in WEATHER_COLUMNS:
        piece[column] = pd.to_numeric(piece[column], errors="coerce")
    keep = ["timestamp", "timestamp_utc", "load_area", *WEATHER_COLUMNS]
    return piece.loc[:, keep].sort_values(["load_area", "timestamp_utc"]).reset_index(drop=True)


def build_versioned_load(
    frame: pd.DataFrame,
    load_area: str,
    load_delay: pd.Timedelta,
    out_path: Path | str | None = None,
) -> pd.DataFrame:
    """Metered load becomes visible ``load_delay`` after the hour begins."""
    code = load_area.strip().upper()
    if code not in BY_CODE:
        raise ValueError(f"Unknown load area {load_area!r}")
    if "timestamp" not in frame.columns or "load" not in frame.columns:
        raise ValueError("load frame must include timestamp and load")
    if "load_area" in frame.columns:
        labels = frame["load_area"].astype(str).str.strip().str.upper()
        work = frame.loc[labels == code].copy()
    else:
        work = frame.copy()
    if work.empty:
        raise ValueError(f"{code} has no load rows")
    work = work.reset_index(drop=True)
    stamps = pd.to_datetime(work["timestamp"])
    if stamps.dt.tz is None:
        stamps = stamps.dt.tz_localize(EASTERN)
    else:
        stamps = stamps.dt.tz_convert(EASTERN)
    versioned = pd.DataFrame(
        {
            "timestamp": stamps,
            "available_at": stamps + load_delay,
            "load": pd.to_numeric(work["load"], errors="coerce"),
            "load_area": code,
        }
    )
    versioned = versioned.sort_values("timestamp").reset_index(drop=True)
    if out_path is not None:
        destination = Path(out_path)
        ensure_dir(destination.parent)
        versioned.to_parquet(destination, index=False)
    return versioned
