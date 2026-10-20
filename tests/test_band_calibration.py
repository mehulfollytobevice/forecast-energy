"""Band widening factors and application."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pjm_forecasting.band_calibration import (
    apply_band_scales,
    conformal_scales,
    coverage_report,
)


def _forecast(n: int = 48) -> tuple[pd.Series, pd.DataFrame]:
    origin = pd.Timestamp("2026-01-01T00:00", tz="US/Eastern")
    index = pd.date_range(origin, periods=n, freq="h")
    mid = np.linspace(100, 140, n)
    actual = pd.Series(mid + 8.0, index=index)  # sits above a narrow upper band
    frame = pd.DataFrame(
        {
            "quantile_P10": mid - 3.0,
            "quantile_P50": mid,
            "quantile_P90": mid + 3.0,
            "origin": origin,
        },
        index=index,
    )
    return actual, frame


def test_conformal_scales_are_at_least_one():
    actual, forecast = _forecast()
    scales = conformal_scales(actual, forecast, horizon=48, min_samples=3)
    assert len(scales["p10"]) == 48
    assert all(value >= 1.0 for value in scales["p10"])
    assert all(value >= 1.0 for value in scales["p90"])


def test_apply_band_scales_widens_without_moving_p50():
    actual, forecast = _forecast()
    scales = {"p10": [2.0] * 48, "p90": [2.0] * 48, "n": [1.0] * 48}
    widened = apply_band_scales(forecast, scales)
    assert np.allclose(widened["quantile_P50"], forecast["quantile_P50"])
    assert (widened["quantile_P10"] <= forecast["quantile_P10"]).all()
    assert (widened["quantile_P90"] >= forecast["quantile_P90"]).all()


def test_coverage_report_counts_inside_band():
    actual, forecast = _forecast()
    before = coverage_report(actual, forecast)
    scales = conformal_scales(actual, forecast, horizon=48, min_samples=3)
    after = coverage_report(actual, apply_band_scales(forecast, scales))
    assert before["n"] > 0
    assert after["coverage_p10_p90"] >= before["coverage_p10_p90"] - 1e-9
