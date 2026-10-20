"""Per-lead-hour P10/P90 widening from validation residuals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pjm_forecasting.utils import ensure_dir


def lead_hours(forecast: pd.DataFrame, origin_column: str = "origin") -> pd.Series:
    """Whole lead hours from ``origin`` to each forecast timestamp."""
    if forecast.empty:
        return pd.Series(dtype=int)
    stamps = pd.DatetimeIndex(forecast.index)
    if stamps.tz is None:
        stamps = stamps.tz_localize("UTC")
    else:
        stamps = stamps.tz_convert("UTC")
    origins = pd.to_datetime(forecast[origin_column])
    if origins.dt.tz is None:
        origins = origins.dt.tz_localize(stamps.tz)
    else:
        origins = origins.dt.tz_convert(stamps.tz)
    delta = stamps - pd.DatetimeIndex(origins)
    return pd.Series(np.floor(delta.total_seconds() / 3600.0).astype(int), index=forecast.index)


def conformal_scales(
    actual: pd.Series,
    forecast: pd.DataFrame,
    *,
    horizon: int = 48,
    alpha: float = 0.1,
    min_samples: int = 5,
) -> dict[str, list[float]]:
    """Scale factors that widen P10/P90 away from P50 by lead hour.

    For each lead hour ``h``, let ``r_lo = (P50 - actual) / max(P50 - P10, eps)``
    on rows where actual < P50, and ``r_hi = (actual - P50) / max(P90 - P50, eps)``
    where actual > P50. The ``(1 - alpha)`` quantile of each residual becomes
    the scale for that tail. Scales are floored at 1.0 (never shrink) and
    smoothed with a 3-point moving average across lead hours.
    """
    if "quantile_P50" not in forecast.columns:
        raise ValueError("forecast is missing quantile_P50")
    has_bands = {"quantile_P10", "quantile_P90"}.issubset(forecast.columns)
    if not has_bands:
        return {
            "p10": [1.0] * horizon,
            "p90": [1.0] * horizon,
            "n": [0.0] * horizon,
        }
    work = forecast.copy()
    work["lead_hours"] = lead_hours(work)
    observed = actual.reindex(work.index)
    work["actual"] = observed.to_numpy()
    work = work.dropna(subset=["actual", "quantile_P50", "quantile_P10", "quantile_P90"])
    eps = 1e-6
    p10_scales = np.ones(horizon, dtype=float)
    p90_scales = np.ones(horizon, dtype=float)
    counts = np.zeros(horizon, dtype=float)
    for lead in range(1, horizon + 1):
        piece = work.loc[work["lead_hours"] == lead]
        counts[lead - 1] = float(len(piece))
        if len(piece) < min_samples:
            continue
        p50 = piece["quantile_P50"].astype(float)
        p10 = piece["quantile_P10"].astype(float)
        p90 = piece["quantile_P90"].astype(float)
        y = piece["actual"].astype(float)
        lo_width = (p50 - p10).clip(lower=eps)
        hi_width = (p90 - p50).clip(lower=eps)
        lo_resid = ((p50 - y) / lo_width).loc[y <= p50]
        hi_resid = ((y - p50) / hi_width).loc[y >= p50]
        if len(lo_resid) >= min_samples:
            p10_scales[lead - 1] = max(1.0, float(np.quantile(lo_resid, 1.0 - alpha)))
        if len(hi_resid) >= min_samples:
            p90_scales[lead - 1] = max(1.0, float(np.quantile(hi_resid, 1.0 - alpha)))
    p10_scales = _smooth(p10_scales)
    p90_scales = _smooth(p90_scales)
    return {
        "p10": [float(value) for value in p10_scales],
        "p90": [float(value) for value in p90_scales],
        "n": [float(value) for value in counts],
    }


def _smooth(values: np.ndarray) -> np.ndarray:
    if values.size < 3:
        return values
    padded = np.concatenate([[values[0]], values, [values[-1]]])
    kernel = np.array([0.25, 0.5, 0.25])
    smoothed = np.convolve(padded, kernel, mode="valid")
    return np.maximum(smoothed, 1.0)


def apply_band_scales(
    forecast: pd.DataFrame,
    scales: dict[str, list[float]],
    *,
    origin_column: str = "origin",
) -> pd.DataFrame:
    """Widen ``quantile_P10`` / ``quantile_P90`` using per-lead scales."""
    if forecast.empty:
        return forecast.copy()
    needed = {"quantile_P10", "quantile_P50", "quantile_P90"}
    if not needed.issubset(forecast.columns):
        return forecast.copy()
    p10_scales = scales.get("p10") or []
    p90_scales = scales.get("p90") or []
    if not p10_scales or not p90_scales:
        return forecast.copy()
    out = forecast.copy()
    leads = lead_hours(out, origin_column=origin_column)
    p50 = out["quantile_P50"].astype(float)
    p10 = out["quantile_P10"].astype(float)
    p90 = out["quantile_P90"].astype(float)
    new_p10 = p10.to_numpy(copy=True)
    new_p90 = p90.to_numpy(copy=True)
    for index, lead in enumerate(leads.to_numpy()):
        if lead < 1:
            continue
        slot = min(int(lead), len(p10_scales)) - 1
        lo = float(p10_scales[slot])
        hi = float(p90_scales[slot])
        center = float(p50.iloc[index])
        new_p10[index] = center - lo * (center - float(p10.iloc[index]))
        new_p90[index] = center + hi * (float(p90.iloc[index]) - center)
    out["quantile_P10"] = new_p10
    out["quantile_P90"] = new_p90
    return out


def coverage_report(
    actual: pd.Series,
    forecast: pd.DataFrame,
) -> dict[str, float]:
    """Share of actuals inside the P10–P90 band."""
    if forecast.empty or not {"quantile_P10", "quantile_P90"}.issubset(forecast.columns):
        return {"n": 0.0, "coverage_p10_p90": float("nan")}
    observed = actual.reindex(forecast.index)
    mask = observed.notna() & forecast["quantile_P10"].notna() & forecast["quantile_P90"].notna()
    if not mask.any():
        return {"n": 0.0, "coverage_p10_p90": float("nan")}
    y = observed.loc[mask]
    inside = (y >= forecast.loc[mask, "quantile_P10"]) & (y <= forecast.loc[mask, "quantile_P90"])
    return {"n": float(mask.sum()), "coverage_p10_p90": float(inside.mean())}


def write_band_scale(path: Path | str, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    ensure_dir(destination.parent)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return destination


def read_band_scale(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
