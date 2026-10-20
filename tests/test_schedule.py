"""Eastern schedule gate, including a DST transition."""

from __future__ import annotations

import pandas as pd

from pjm_forecasting.schedule import (
    current_et_clock,
    expected_ecmwf_run,
    next_origin,
    normalize_schedule,
    should_run,
)


def test_normalize_schedule_sorts_and_dedupes():
    assert normalize_schedule(["17:15", "05:15", "05:15"]) == ("05:15", "17:15")


def test_should_run_matches_eastern_slot():
    now = pd.Timestamp("2026-03-10T05:20", tz="US/Eastern")
    assert should_run(["05:15", "11:15"], now)
    assert not should_run(["05:15", "11:15"], pd.Timestamp("2026-03-10T06:20", tz="US/Eastern"))


def test_should_run_handles_spring_forward():
    # 2026-03-08 02:00 ET does not exist; 03:00 ET is UTC 07:00 after spring forward.
    before = pd.Timestamp("2026-03-08T01:15", tz="US/Eastern")
    after = pd.Timestamp("2026-03-08T03:15", tz="US/Eastern")
    assert should_run(["01:15"], before)
    assert should_run(["03:15"], after)
    assert current_et_clock(after) == "03:15"


def test_next_origin_floors_to_next_hour():
    now = pd.Timestamp("2026-06-01T10:17", tz="US/Eastern")
    assert next_origin(now) == pd.Timestamp("2026-06-01T11:00", tz="US/Eastern")


def test_expected_ecmwf_run_respects_publication_delay():
    now = pd.Timestamp("2026-06-01T13:00", tz="UTC")
    run = expected_ecmwf_run(now, publication_delay_hours=6, cadence_hours=6)
    assert run == pd.Timestamp("2026-06-01T06:00", tz="UTC")
