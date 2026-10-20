"""Rolling train/validation split for forecast-run training."""

from __future__ import annotations

import pandas as pd
import pytest

from pjm_forecasting.splits import SplitError, split_rolling


def _frame(days: int = 120) -> pd.DataFrame:
    index = pd.date_range("2024-03-15", periods=24 * days, freq="h", tz="US/Eastern")
    return pd.DataFrame({"load": range(len(index))}, index=index)


def test_split_rolling_validation_is_trailing_window():
    frame = _frame()
    split = split_rolling(frame, validation_days=10, train_start="2024-03-15")
    assert split.validation.index.min() >= split.train_end
    assert (split.validation.index.max() - split.validation.index.min()) >= pd.Timedelta(days=9)
    assert split.train.index.max() < split.validation.index.min()


def test_split_rolling_rejects_too_little_history():
    frame = _frame(days=5)
    with pytest.raises(SplitError):
        split_rolling(frame, validation_days=30)
