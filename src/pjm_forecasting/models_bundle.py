"""Publish and fetch MLflow model bundles plus band-scale metadata."""

from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pjm_forecasting.settings import PipelineSettings, load_settings
from pjm_forecasting.storage import ObjectStore
from pjm_forecasting.utils import ensure_dir, project_root

LATEST_KEY = "models/latest.json"
ARCHIVE_FALLBACK_KEY = "models/archive-fallback.tar.gz"
BAND_SCALE_NAME = "band_scale.json"


def _sqlite_path(tracking_uri: str) -> Path | None:
    if not tracking_uri.startswith("sqlite:///"):
        return None
    raw = tracking_uri.removeprefix("sqlite:///")
    path = Path(raw)
    if not path.is_absolute():
        path = project_root() / path
    return path


def package_bundle(
    *,
    settings: PipelineSettings | None = None,
    band_scale: dict[str, Any] | None = None,
    version: str | None = None,
    out_dir: Path | str | None = None,
) -> Path:
    """Tar the local MLflow DB, artifacts, and optional band_scale.json."""
    settings = settings or load_settings()
    stamp = version or datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination_dir = Path(out_dir) if out_dir is not None else settings.models_dir
    ensure_dir(destination_dir)
    archive_path = destination_dir / f"{stamp}.tar.gz"
    db_path = _sqlite_path(settings.tracking_uri)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        if db_path is not None and db_path.is_file():
            shutil.copy2(db_path, root / "mlflow.db")
        if settings.artifact_location.is_dir():
            shutil.copytree(settings.artifact_location, root / "artifacts", dirs_exist_ok=True)
        if settings.local_artifacts_path.is_dir():
            shutil.copytree(
                settings.local_artifacts_path,
                root / "staging",
                dirs_exist_ok=True,
            )
        if band_scale is not None:
            (root / BAND_SCALE_NAME).write_text(json.dumps(band_scale, indent=2) + "\n")
        meta = {
            "version": stamp,
            "weather_source": settings.weather_source,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        (root / "bundle_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        with tarfile.open(archive_path, "w:gz") as tar:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    tar.add(path, arcname=path.relative_to(root).as_posix())
    return archive_path


def publish_bundle(
    archive_path: Path | str,
    *,
    store: ObjectStore | None = None,
    version: str | None = None,
    metrics: dict[str, Any] | None = None,
    key_prefix: str = "models",
) -> dict[str, Any]:
    """Upload a bundle and point ``models/latest.json`` at it."""
    store = store or ObjectStore()
    path = Path(archive_path)
    stamp = version or path.name.removesuffix(".tar.gz")
    key = f"{key_prefix.rstrip('/')}/{path.name}"
    store.upload_file(path, key)
    payload = {
        "version": stamp,
        "key": key,
        "published_at": datetime.now(tz=timezone.utc).isoformat(),
        "metrics": metrics or {},
    }
    store.write_json(f"{key_prefix.rstrip('/')}/latest.json", payload)
    # Also keep a local copy under data/models for Phase 1.
    local_latest = store.local_path(f"{key_prefix.rstrip('/')}/latest.json")
    ensure_dir(local_latest.parent)
    if not store.is_local:
        local_latest.write_text(json.dumps(payload, indent=2) + "\n")
        shutil.copy2(path, local_latest.parent / path.name)
    return payload


def fetch_bundle(
    *,
    store: ObjectStore | None = None,
    settings: PipelineSettings | None = None,
    key: str | None = None,
    latest_key: str = LATEST_KEY,
    extract_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Download and extract the published bundle into local MLflow paths."""
    store = store or ObjectStore()
    settings = settings or load_settings()
    if key is None:
        latest = store.read_json(latest_key)
        key = str(latest["key"])
        version = str(latest.get("version", Path(key).name))
    else:
        latest = {"key": key, "version": Path(key).name}
        version = str(latest["version"])
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / Path(key).name
        store.download_file(key, archive)
        extract_root = Path(extract_dir) if extract_dir is not None else Path(tmp) / "extract"
        ensure_dir(extract_root)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(extract_root)
        db_src = extract_root / "mlflow.db"
        db_dst = _sqlite_path(settings.tracking_uri)
        if db_src.is_file() and db_dst is not None:
            ensure_dir(db_dst.parent)
            shutil.copy2(db_src, db_dst)
        artifacts_src = extract_root / "artifacts"
        if artifacts_src.is_dir():
            if settings.artifact_location.exists():
                shutil.rmtree(settings.artifact_location)
            shutil.copytree(artifacts_src, settings.artifact_location)
        staging_src = extract_root / "staging"
        if staging_src.is_dir():
            if settings.local_artifacts_path.exists():
                shutil.rmtree(settings.local_artifacts_path)
            shutil.copytree(staging_src, settings.local_artifacts_path)
        band_src = extract_root / BAND_SCALE_NAME
        band_dst = settings.models_dir / BAND_SCALE_NAME
        if band_src.is_file():
            ensure_dir(band_dst.parent)
            shutil.copy2(band_src, band_dst)
        meta_path = extract_root / "bundle_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    return {
        "version": version,
        "key": key,
        "band_scale_path": str(settings.models_dir / BAND_SCALE_NAME),
        "meta": meta,
        "latest": latest,
    }


def publish_archive_fallback(
    *,
    settings: PipelineSettings | None = None,
    store: ObjectStore | None = None,
) -> dict[str, Any]:
    """Bundle the current MLflow store once as the archive-trained fallback."""
    settings = settings or load_settings()
    store = store or ObjectStore()
    archive = package_bundle(settings=settings, version="archive-fallback")
    renamed = settings.models_dir / "archive-fallback.tar.gz"
    if archive != renamed:
        shutil.move(str(archive), str(renamed))
    store.upload_file(renamed, ARCHIVE_FALLBACK_KEY)
    payload = {
        "version": "archive-fallback",
        "key": ARCHIVE_FALLBACK_KEY,
        "published_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    store.write_json("models/archive-fallback.json", payload)
    return payload


def load_band_scale(settings: PipelineSettings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    path = settings.models_dir / BAND_SCALE_NAME
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def tracking_host(tracking_uri: str) -> str:
    parsed = urlparse(tracking_uri)
    return parsed.path or tracking_uri
