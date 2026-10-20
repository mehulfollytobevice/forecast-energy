"""PJM-aligned Eastern-time schedule gate for the live forecast job."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from pjm_forecasting.utils import EASTERN


def normalize_schedule(times: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Accept ``HH:MM`` strings and return a sorted unique tuple."""
    parsed: list[str] = []
    seen: set[str] = set()
    for item in times:
        text = str(item).strip()
        hour, minute = text.split(":")
        stamp = f"{int(hour):02d}:{int(minute):02d}"
        if stamp in seen:
            continue
        seen.add(stamp)
        parsed.append(stamp)
    return tuple(sorted(parsed))


def current_et_clock(now: datetime | pd.Timestamp | None = None) -> str:
    """Current Eastern ``HH:MM``."""
    stamp = pd.Timestamp.now(tz=EASTERN) if now is None else pd.Timestamp(now)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(EASTERN)
    else:
        stamp = stamp.tz_convert(EASTERN)
    return f"{stamp.hour:02d}:{stamp.minute:02d}"


def should_run(
    schedule_et: list[str] | tuple[str, ...],
    now: datetime | pd.Timestamp | None = None,
    *,
    window_minutes: int = 14,
) -> bool:
    """True when ``now`` falls in the same Eastern hour slot as a schedule entry.

    Cron fires at :15 every hour. A schedule entry of ``05:15`` matches any
    minute in ``[05:15, 05:15 + window_minutes)`` Eastern so a slow start still
    counts. Daylight saving is handled by converting ``now`` to Eastern.
    """
    stamp = pd.Timestamp.now(tz=EASTERN) if now is None else pd.Timestamp(now)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(EASTERN)
    else:
        stamp = stamp.tz_convert(EASTERN)
    for entry in normalize_schedule(schedule_et):
        hour, minute = (int(part) for part in entry.split(":"))
        start = stamp.replace(hour=hour, minute=minute, second=0, microsecond=0)
        end = start + pd.Timedelta(minutes=window_minutes)
        if start <= stamp < end:
            return True
    return False


def next_origin(now: datetime | pd.Timestamp | None = None) -> pd.Timestamp:
    """Next whole Eastern hour at or after ``now``."""
    stamp = pd.Timestamp.now(tz=EASTERN) if now is None else pd.Timestamp(now)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(EASTERN)
    else:
        stamp = stamp.tz_convert(EASTERN)
    floored = stamp.floor("h")
    if floored < stamp:
        return floored + pd.Timedelta(hours=1)
    return floored


def expected_ecmwf_run(
    now: datetime | pd.Timestamp | None = None,
    *,
    publication_delay_hours: int = 6,
    cadence_hours: int = 6,
) -> pd.Timestamp:
    """Newest ECMWF init that should already be published at ``now``."""
    stamp = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    available_deadline = stamp - pd.Timedelta(hours=publication_delay_hours)
    hour = (available_deadline.hour // cadence_hours) * cadence_hours
    return available_deadline.replace(hour=hour, minute=0, second=0, microsecond=0)
