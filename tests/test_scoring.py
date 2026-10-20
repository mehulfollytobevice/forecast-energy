"""Forecast scores. No model is fit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pjm_forecasting.backtesting import error_by_lead_time
from pjm_forecasting.training import score_forecast


def test_score_forecast_uses_the_overlap_and_rejects_an_empty_one():
    index = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    actual = pd.Series([10.0, 20.0, np.nan], index=index)
    forecast = pd.DataFrame(
        {
            "quantile_P10": [8.0, 18.0, 28.0],
            "quantile_P50": [12.0, 18.0, 30.0],
            "quantile_P90": [14.0, 22.0, 32.0],
        },
        index=index,
    )
    scores = score_forecast(actual, forecast)
    assert scores["n"] == pytest.approx(2.0)
    assert scores["mae"] == pytest.approx(2.0)
    assert scores["rmse"] == pytest.approx(2.0)
    assert scores["pinball_p10"] == pytest.approx(0.2)
    assert scores["pinball_p50"] == pytest.approx(1.0)
    assert scores["pinball_p90"] == pytest.approx(0.3)
    assert scores["coverage_p10_p90"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="do not overlap"):
        score_forecast(pd.Series([np.nan, np.nan, np.nan], index=index), forecast)


def test_error_by_lead_time_groups_absolute_p50_error_and_drops_missing_actuals():
    issued = pd.Timestamp("2024-01-01T00:00", tz="UTC")
    stamps = pd.DatetimeIndex(
        [
            issued + pd.Timedelta(hours=1),
            issued + pd.Timedelta(hours=1.5),
            issued + pd.Timedelta(hours=3),
            issued + pd.Timedelta(hours=5),
        ]
    )
    predictions = pd.DataFrame(
        {
            "available_at": issued,
            "quantile_P50": [12.0, 16.0, 30.0, 40.0],
        },
        index=stamps,
    )
    actual = pd.Series([10.0, 20.0, 30.0], index=stamps[:3])
    grouped = error_by_lead_time(predictions, actual)
    assert grouped["lead_hours"].tolist() == [1, 3]
    assert grouped["mae"].tolist() == pytest.approx([3.0, 0.0])
    assert grouped["n"].tolist() == [2, 1]
