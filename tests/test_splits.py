"""Chronological split guards."""

from __future__ import annotations

import pandas as pd
import pytest

from pjm_forecasting.splits import SplitError, split_chronological


def _frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=24 * 10, freq="h", tz="US/Eastern")
    return pd.DataFrame({"load": range(len(index))}, index=index)


def test_split_is_half_open_and_non_overlapping():
    frame = _frame()
    split = split_chronological(frame, "2024-01-04", "2024-01-08")
    assert split.train.index.max() < split.train_end
    assert split.validation.index.min() >= split.train_end
    assert split.validation.index.max() < split.validation_end
    assert split.test.index.min() >= split.validation_end
    assert split.train.index.intersection(split.validation.index).empty
    assert split.validation.index.intersection(split.test.index).empty
    assert split.train.index.max() < split.validation.index.min()
    assert split.validation.index.max() < split.test.index.min()


def test_split_rejects_an_empty_window():
    frame = _frame()
    with pytest.raises(SplitError, match="non-empty"):
        split_chronological(frame, "2023-01-01", "2023-06-01")


def test_split_rejects_a_boundary_that_would_look_ahead():
    frame = _frame()
    with pytest.raises(SplitError, match="earlier"):
        split_chronological(frame, "2024-01-08", "2024-01-04")
