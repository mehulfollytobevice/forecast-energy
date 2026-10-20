"""FastAPI entry: forecast JSON and the built dashboard."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pjm_forecasting.backtest_snapshot import resolve_backtest_snapshot
from pjm_forecasting.settings import load_settings
from pjm_forecasting.snapshot import default_snapshot_path
from pjm_forecasting.storage import ObjectStore, load_storage_config
from pjm_forecasting.utils import project_root

load_dotenv()

ROOT = project_root()
APP_DIR = ROOT / "app"
FIXTURE = APP_DIR / "fixtures" / "forecast_snapshot.json"
BACKTEST_FIXTURE = APP_DIR / "fixtures" / "backtest_snapshot.json"
GEO_DIR = APP_DIR / "geo"
WEB_DIST = APP_DIR / "web" / "dist"

app = FastAPI(title="PJM forecast dashboard", version="0.1.0")

_FORECAST_CACHE: dict[str, Any] = {"payload": None, "fetched_at": 0.0}


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    return ordered


def _forecast_candidates() -> list[Path]:
    env = os.environ.get("FORECAST_SNAPSHOT", "").strip()
    paths: list[Path] = []
    if env:
        paths.append(Path(env))
    # Local object-store layout (Phase 1).
    paths.append(ROOT / "data" / "snapshots" / "forecast" / "latest.json")
    paths.append(default_snapshot_path())
    paths.append(FIXTURE)
    return _unique_paths(paths)


def _load_json(candidates: list[Path], *, kind: str) -> dict:
    for path in candidates:
        if path.is_file():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=500, detail=f"Invalid JSON in {path}") from exc
    tried = ", ".join(str(path) for path in candidates)
    raise HTTPException(status_code=503, detail=f"No {kind} snapshot found. Tried: {tried}")


def _ttl_seconds() -> int:
    try:
        return int(load_settings().snapshot_ttl_seconds)
    except Exception:
        return 30


def _load_forecast_snapshot() -> dict:
    """Prefer object store (bucket or local data/), then file candidates, then fixture."""
    ttl = _ttl_seconds()
    now = time.time()
    cached = _FORECAST_CACHE.get("payload")
    fetched_at = float(_FORECAST_CACHE.get("fetched_at") or 0.0)
    if cached is not None and now - fetched_at < ttl:
        return cached

    store = ObjectStore(load_storage_config())
    try:
        payload = store.read_json("snapshots/forecast/latest.json")
        _FORECAST_CACHE["payload"] = payload
        _FORECAST_CACHE["fetched_at"] = now
        return payload
    except FileNotFoundError:
        pass
    except Exception:
        # Fall through to local files / fixture.
        pass

    payload = _load_json(_forecast_candidates(), kind="forecast")
    _FORECAST_CACHE["payload"] = payload
    _FORECAST_CACHE["fetched_at"] = now
    return payload


@app.get("/api/forecast")
def get_forecast() -> dict:
    """Reconciled hierarchy the live map reads."""
    return _load_forecast_snapshot()


@app.get("/api/backtest")
def get_backtest() -> dict:
    """Hierarchical backtest: actuals vs as-issued forecast over the test window.

    When ``BACKTEST_SNAPSHOT`` is set, that file is served as-is (tests and
    explicit overrides). Otherwise real backtest parquets / processed snapshot
    load areas win, and any remaining areas are filled from the fixture.
    """
    env = os.environ.get("BACKTEST_SNAPSHOT", "").strip()
    if env:
        return _load_json([Path(env)], kind="backtest")
    try:
        return resolve_backtest_snapshot(fixture_path=BACKTEST_FIXTURE)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Invalid backtest snapshot: {exc}") from exc


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if GEO_DIR.is_dir():
    app.mount("/geo", StaticFiles(directory=GEO_DIR), name="geo")

if WEB_DIST.is_dir():
    assets = WEB_DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIST / "index.html")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str) -> FileResponse:
        if full_path.startswith("api/") or full_path.startswith("geo/"):
            raise HTTPException(status_code=404)
        candidate = WEB_DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(WEB_DIST / "index.html")
