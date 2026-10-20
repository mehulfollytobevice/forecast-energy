"""Shared utilities (paths, config loading, logging helpers)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

EASTERN = "US/Eastern"


def project_root() -> Path:
    """Return repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def load_config(path: Path | str | None = None) -> dict:
    """Load a YAML experiment config, resolving relative paths from the repo root."""
    cfg_path = Path(path) if path is not None else project_root() / "configs/default.yaml"
    if not cfg_path.is_file():
        candidate = project_root() / cfg_path
        if candidate.is_file():
            cfg_path = candidate
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def resolve_data_dir(config: dict, key: str) -> Path:
    """Return an absolute data directory from ``config['data'][key]``."""
    raw = Path(config["data"][key])
    return raw if raw.is_absolute() else project_root() / raw


def ensure_dir(path: Path | str) -> Path:
    """Create ``path`` (and parents) if needed and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def month_range(start: str | pd.Timestamp, end: str | pd.Timestamp) -> list[pd.Timestamp]:
    """First-of-month timestamps covering the closed interval ``[start, end]``."""
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    months = pd.date_range(start_ts.replace(day=1), end_ts.replace(day=1), freq="MS")
    return list(months)


def as_eastern(ts: Any) -> pd.Timestamp:
    """Parse ``ts`` to a timezone-aware US/Eastern timestamp."""
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        return stamp.tz_localize(EASTERN)
    return stamp.tz_convert(EASTERN)
