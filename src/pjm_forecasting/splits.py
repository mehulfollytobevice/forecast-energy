"""Leakage-safe chronological train, validation, and test windows."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pjm_forecasting.utils import as_eastern


class SplitError(ValueError):
    """A chronological split is empty, overlapping, or out of order."""


@dataclass(frozen=True)
class ChronologicalSplit:
    """Half-open windows: ``[start, train_end)``, then validation, then test."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    validation_end: pd.Timestamp


def split_chronological(
    frame: pd.DataFrame,
    train_end: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
) -> ChronologicalSplit:
    """Split on the frame index. Later windows are not visible to earlier ones."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise SplitError("frame index must be a DatetimeIndex")
    if frame.index.tz is None:
        raise SplitError("frame index must be timezone-aware")
    train_bound = as_eastern(train_end)
    validation_bound = as_eastern(validation_end)
    if train_bound >= validation_bound:
        raise SplitError("train_end must be earlier than validation_end")

    index = frame.index
    train = frame.loc[index < train_bound]
    validation = frame.loc[(index >= train_bound) & (index < validation_bound)]
    test = frame.loc[index >= validation_bound]
    if train.empty or validation.empty or test.empty:
        raise SplitError("train, validation, and test windows must all be non-empty")
    if not train.index.is_monotonic_increasing:
        raise SplitError("train index is not chronological")
    if not validation.index.is_monotonic_increasing:
        raise SplitError("validation index is not chronological")
    if not test.index.is_monotonic_increasing:
        raise SplitError("test index is not chronological")
    if train.index.max() >= validation.index.min() or validation.index.max() >= test.index.min():
        raise SplitError("split windows overlap")
    if (
        len(train.index.intersection(validation.index))
        or len(train.index.intersection(test.index))
        or len(validation.index.intersection(test.index))
    ):
        raise SplitError("split windows overlap")
    return ChronologicalSplit(
        train=train,
        validation=validation,
        test=test,
        train_end=train_bound,
        validation_end=validation_bound,
    )


def split_rolling(
    frame: pd.DataFrame,
    *,
    validation_days: int,
    train_start: str | pd.Timestamp | None = None,
    last_load: str | pd.Timestamp | None = None,
) -> ChronologicalSplit:
    """Rolling train/validation split anchored on the last load hour.

    Train is ``[train_start, validation_start)``, validation is the trailing
    ``validation_days`` before ``last_load`` (inclusive of the last hour in
    the frame that is before ``last_load + 1h``). The test window is empty-
    but-non-fatal for callers that only need train and validation: it holds
    any rows at or after ``last_load + 1h``, or a one-row copy of the last
    validation hour when there are none (so existing empty-window guards pass).
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise SplitError("frame index must be a DatetimeIndex")
    if frame.index.tz is None:
        raise SplitError("frame index must be timezone-aware")
    if validation_days < 1:
        raise SplitError("validation_days must be >= 1")

    work = frame.sort_index()
    if last_load is None:
        end = work.index.max() + pd.Timedelta(hours=1)
    else:
        end = as_eastern(last_load) + pd.Timedelta(hours=1)
    validation_bound = end
    train_bound = end - pd.Timedelta(days=int(validation_days))
    if train_start is not None:
        start = as_eastern(train_start)
        work = work.loc[work.index >= start]
    if work.empty:
        raise SplitError("frame is empty after applying train_start")
    if train_bound <= work.index.min():
        raise SplitError("validation window leaves no training rows")

    train = work.loc[work.index < train_bound]
    validation = work.loc[(work.index >= train_bound) & (work.index < validation_bound)]
    test = work.loc[work.index >= validation_bound]
    if train.empty or validation.empty:
        raise SplitError("train and validation windows must both be non-empty")
    if test.empty:
        # ChronologicalSplit historically requires a non-empty test; pad with
        # the last validation hour so callers that ignore test still work.
        test = validation.iloc[-1:].copy()
    return ChronologicalSplit(
        train=train,
        validation=validation,
        test=test,
        train_end=train_bound,
        validation_end=validation_bound,
    )
