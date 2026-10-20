"""Fatal checks and coverage statistics for load, weather, and forecast versions.

Coverage fractions are always reported. A failed fatal check raises
``QualityError`` and is what a Prefect quality gate uses to stop the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pjm_forecasting.areas import CATALOG_CODES
from pjm_forecasting.schema import FRAME_COLUMNS, LOAD_COLUMNS, WEATHER_COLUMNS
from pjm_forecasting.utils import EASTERN

VERSIONED_WEATHER_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "available_at",
    "run_initialized_at",
    "load_area",
    *WEATHER_COLUMNS,
)
VERSIONED_LOAD_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "available_at",
    "load",
    "load_area",
)
DEFAULT_MAX_GAP = pd.Timedelta(hours=6)
RUN_MAX_GAP = pd.Timedelta(hours=48)


class QualityError(ValueError):
    """One or more fatal quality checks failed."""


@dataclass(frozen=True)
class QualityIssue:
    """A single check result."""

    check: str
    fatal: bool
    message: str

    def __str__(self) -> str:
        return f"{self.check}: {self.message}"


@dataclass
class QualityReport:
    """Structured result of one dataset validation."""

    dataset: str
    issues: list[QualityIssue] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    @property
    def fatal(self) -> bool:
        return any(issue.fatal for issue in self.issues)

    def raise_if_fatal(self) -> None:
        if not self.fatal:
            return
        messages = "; ".join(str(issue) for issue in self.issues if issue.fatal)
        raise QualityError(messages)

    def log_lines(self) -> list[str]:
        lines = [f"{self.dataset} {key}={value}" for key, value in self.stats.items()]
        lines.extend(str(issue) for issue in self.issues)
        return lines


def _issue(check: str, message: str, *, fatal: bool = True) -> QualityIssue:
    return QualityIssue(check=check, fatal=fatal, message=message)


def _missing(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> list[str]:
    return [column for column in columns if column not in frame.columns]


def _is_datetime(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series)


def _tz_label(series: pd.Series) -> str | None:
    if not _is_datetime(series):
        return None
    tz = series.dt.tz
    if tz is None:
        return None
    return str(getattr(tz, "zone", None) or tz)


def _is_eastern(series: pd.Series) -> bool:
    label = _tz_label(series)
    return label in {EASTERN, "America/New_York"}


def _is_utc(series: pd.Series) -> bool:
    label = _tz_label(series)
    return label in {"UTC", "tzutc()"}


def _require_columns(issues: list[QualityIssue], frame: pd.DataFrame, columns: list[str]) -> bool:
    missing = _missing(frame, columns)
    if missing:
        issues.append(_issue("columns", f"missing columns: {missing}"))
        return False
    return True


def _catalog_issue(frame: pd.DataFrame) -> QualityIssue | None:
    labels = frame["load_area"].astype(str).str.strip().str.upper()
    unknown = sorted(set(labels) - CATALOG_CODES)
    if unknown:
        return _issue("catalog", f"load areas outside the catalog: {unknown}")
    if labels.isna().any() or (labels == "").any():
        return _issue("catalog", "load_area has blank values")
    return None


def _duplicate_issue(frame: pd.DataFrame, keys: list[str]) -> QualityIssue | None:
    if frame.duplicated(subset=keys).any():
        return _issue("duplicates", f"duplicate rows for {keys}")
    return None


def _coverage_stats(stamps: pd.Series) -> dict[str, object]:
    utc = pd.to_datetime(stamps, utc=True).dropna().sort_values()
    if utc.empty:
        return {"rows": 0, "coverage": 0.0, "longest_gap_hours": None, "missing_hours": 0}
    gaps = utc.diff().dropna()
    longest = gaps.max() if not gaps.empty else pd.Timedelta(0)
    span_hours = (utc.iloc[-1] - utc.iloc[0]) / pd.Timedelta(hours=1)
    expected = int(span_hours) + 1
    present = int(utc.nunique())
    missing = max(expected - present, 0)
    coverage = present / expected if expected else 1.0
    return {
        "rows": int(len(utc)),
        "start": utc.iloc[0].isoformat(),
        "end": utc.iloc[-1].isoformat(),
        "coverage": round(float(coverage), 6),
        "longest_gap_hours": round(longest.total_seconds() / 3600.0, 3),
        "missing_hours": missing,
    }


def _chronological_issues(
    frame: pd.DataFrame,
    stamp_column: str,
    *,
    group_column: str | None,
    max_gap: pd.Timedelta,
) -> tuple[list[QualityIssue], dict[str, object]]:
    issues: list[QualityIssue] = []
    groups = frame.groupby(group_column, sort=False) if group_column else [(None, frame)]
    longest = pd.Timedelta(0)
    rows = 0
    missing = 0
    starts: list[str] = []
    ends: list[str] = []
    coverages: list[float] = []
    for name, group in groups:
        stamps = pd.to_datetime(group[stamp_column], utc=True)
        if stamps.isna().any():
            issues.append(_issue("dtypes", f"{stamp_column} has missing timestamps"))
            continue
        ordered = stamps.reset_index(drop=True)
        if not ordered.is_monotonic_increasing:
            label = f" for {name}" if name is not None else ""
            issues.append(
                _issue("chronological", f"{stamp_column} is not monotonic increasing{label}")
            )
        diffs = ordered.diff().dropna()
        if not diffs.empty and (diffs <= pd.Timedelta(0)).any():
            issues.append(_issue("chronological", f"{stamp_column} repeats or goes backwards"))
        if not diffs.empty and (diffs > max_gap).any():
            hours = max_gap.total_seconds() / 3600.0
            issues.append(
                _issue(
                    "chronological",
                    f"gap exceeds {hours:g} hours" + (f" for {name}" if name is not None else ""),
                )
            )
        stats = _coverage_stats(stamps)
        rows += int(stats["rows"])
        missing += int(stats["missing_hours"])
        coverages.append(float(stats["coverage"]))
        if stats.get("start"):
            starts.append(str(stats["start"]))
            ends.append(str(stats["end"]))
        gap_hours = stats["longest_gap_hours"]
        if gap_hours is not None:
            longest = max(longest, pd.Timedelta(hours=float(gap_hours)))
    summary = {
        "rows": rows,
        "coverage": round(float(np.mean(coverages)), 6) if coverages else 0.0,
        "longest_gap_hours": round(longest.total_seconds() / 3600.0, 3),
        "missing_hours": missing,
        "start": min(starts) if starts else None,
        "end": max(ends) if ends else None,
    }
    return issues, summary


def _timezone_pair(frame: pd.DataFrame, issues: list[QualityIssue]) -> None:
    local = frame["timestamp"]
    utc = frame["timestamp_utc"]
    if not _is_datetime(local) or not _is_datetime(utc):
        issues.append(_issue("dtypes", "timestamp columns must be datetimes"))
        return
    if not _is_eastern(local) or not _is_utc(utc):
        issues.append(
            _issue(
                "timezone",
                "timestamp must be US/Eastern and timestamp_utc must be UTC",
            )
        )
        return
    local_utc = local.dt.tz_convert("UTC")
    if not local_utc.equals(utc):
        issues.append(_issue("timestamp_equivalence", "timestamp does not match timestamp_utc"))


def _aware_column(
    frame: pd.DataFrame,
    column: str,
    *,
    utc: bool,
    issues: list[QualityIssue],
) -> None:
    series = frame[column]
    if not _is_datetime(series):
        issues.append(_issue("dtypes", f"{column} must be a datetime"))
        return
    if series.dt.tz is None:
        issues.append(_issue("timezone", f"{column} must be timezone-aware"))
        return
    if utc and not _is_utc(series):
        issues.append(_issue("timezone", f"{column} must be UTC"))
    if not utc and not _is_eastern(series):
        issues.append(_issue("timezone", f"{column} must be US/Eastern"))


def _load_value_issue(frame: pd.DataFrame, load_min: float) -> QualityIssue | None:
    values = pd.to_numeric(frame["load"], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        return _issue("load_values", "load must be finite")
    if (values < load_min).any():
        return _issue("load_values", f"load must be >= {load_min:g}")
    return None


def _weather_value_issue(frame: pd.DataFrame) -> QualityIssue | None:
    for column in WEATHER_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            return _issue("weather_values", f"{column} must be finite")
    return None


def validate_load(
    frame: pd.DataFrame,
    *,
    max_gap: pd.Timedelta = DEFAULT_MAX_GAP,
    load_min: float = 0.0,
) -> QualityReport:
    """Validate a raw load-area table."""
    issues: list[QualityIssue] = []
    if frame.empty:
        issues.append(_issue("columns", "load table is empty"))
        return QualityReport("load", issues, {"rows": 0, "coverage": 0.0})
    if not _require_columns(issues, frame, list(LOAD_COLUMNS)):
        return QualityReport("load", issues, {"rows": int(len(frame))})
    _timezone_pair(frame, issues)
    catalog = _catalog_issue(frame)
    if catalog:
        issues.append(catalog)
    duplicate = _duplicate_issue(frame, ["timestamp_utc", "load_area"])
    if duplicate:
        issues.append(duplicate)
    load_issue = _load_value_issue(frame, load_min)
    if load_issue:
        issues.append(load_issue)
    chronological, stats = _chronological_issues(
        frame, "timestamp_utc", group_column="load_area", max_gap=max_gap
    )
    issues.extend(chronological)
    return QualityReport("load", issues, stats)


def validate_weather(
    frame: pd.DataFrame,
    *,
    max_gap: pd.Timedelta = DEFAULT_MAX_GAP,
) -> QualityReport:
    """Validate archive or live weather at load-area grain."""
    required = ["timestamp", "timestamp_utc", "load_area", *WEATHER_COLUMNS]
    issues: list[QualityIssue] = []
    if frame.empty:
        issues.append(_issue("columns", "weather table is empty"))
        return QualityReport("weather", issues, {"rows": 0, "coverage": 0.0})
    if not _require_columns(issues, frame, required):
        return QualityReport("weather", issues, {"rows": int(len(frame))})
    _timezone_pair(frame, issues)
    catalog = _catalog_issue(frame)
    if catalog:
        issues.append(catalog)
    duplicate = _duplicate_issue(frame, ["timestamp_utc", "load_area"])
    if duplicate:
        issues.append(duplicate)
    weather_issue = _weather_value_issue(frame)
    if weather_issue:
        issues.append(weather_issue)
    chronological, stats = _chronological_issues(
        frame, "timestamp_utc", group_column="load_area", max_gap=max_gap
    )
    issues.extend(chronological)
    return QualityReport("weather", issues, stats)


def validate_joined(
    frame: pd.DataFrame,
    *,
    max_gap: pd.Timedelta = DEFAULT_MAX_GAP,
    load_min: float = 0.0,
) -> QualityReport:
    """Validate the joined training table."""
    issues: list[QualityIssue] = []
    if frame.empty:
        issues.append(_issue("columns", "joined table is empty"))
        return QualityReport("joined", issues, {"rows": 0, "coverage": 0.0})
    if not _require_columns(issues, frame, list(FRAME_COLUMNS)):
        return QualityReport("joined", issues, {"rows": int(len(frame))})
    _timezone_pair(frame, issues)
    catalog = _catalog_issue(frame)
    if catalog:
        issues.append(catalog)
    duplicate = _duplicate_issue(frame, ["timestamp_utc", "load_area"])
    if duplicate:
        issues.append(duplicate)
    load_issue = _load_value_issue(frame, load_min)
    if load_issue:
        issues.append(load_issue)
    weather_issue = _weather_value_issue(frame)
    if weather_issue:
        issues.append(weather_issue)
    chronological, stats = _chronological_issues(
        frame, "timestamp_utc", group_column="load_area", max_gap=max_gap
    )
    issues.extend(chronological)
    return QualityReport("joined", issues, stats)


def validate_versioned_weather(
    frame: pd.DataFrame,
    *,
    publication_delay: pd.Timedelta,
    max_gap: pd.Timedelta = RUN_MAX_GAP,
) -> QualityReport:
    """Validate point-in-time forecast runs.

    ``available_at`` must be ``run_initialized_at`` plus ``publication_delay``.
    Gap checks are per run, because consecutive valid hours inside one run are
    hourly while successive runs overlap.
    """
    issues: list[QualityIssue] = []
    if frame.empty:
        issues.append(_issue("columns", "versioned weather table is empty"))
        return QualityReport("versioned_weather", issues, {"rows": 0, "coverage": 0.0})
    if not _require_columns(issues, frame, list(VERSIONED_WEATHER_COLUMNS)):
        return QualityReport("versioned_weather", issues, {"rows": int(len(frame))})
    _aware_column(frame, "timestamp", utc=False, issues=issues)
    _aware_column(frame, "available_at", utc=True, issues=issues)
    _aware_column(frame, "run_initialized_at", utc=True, issues=issues)
    catalog = _catalog_issue(frame)
    if catalog:
        issues.append(catalog)
    duplicate = _duplicate_issue(frame, ["timestamp", "run_initialized_at", "load_area"])
    if duplicate:
        issues.append(duplicate)
    weather_issue = _weather_value_issue(frame)
    if weather_issue:
        issues.append(weather_issue)
    if _is_datetime(frame["available_at"]) and _is_datetime(frame["run_initialized_at"]):
        expected = frame["run_initialized_at"] + publication_delay
        if (frame["available_at"] < frame["run_initialized_at"]).any():
            issues.append(
                _issue("point_in_time", "available_at is earlier than run_initialized_at")
            )
        elif not frame["available_at"].equals(expected):
            issues.append(
                _issue(
                    "point_in_time",
                    "available_at must equal run_initialized_at plus the publication delay",
                )
            )
    run_key = frame["run_initialized_at"].astype(str) + "|" + frame["load_area"].astype(str)
    grouped = frame.assign(_run=run_key)
    chronological, stats = _chronological_issues(
        grouped, "timestamp", group_column="_run", max_gap=max_gap
    )
    issues.extend(chronological)
    stats["runs"] = int(frame["run_initialized_at"].nunique())
    return QualityReport("versioned_weather", issues, stats)


def validate_versioned_load(
    frame: pd.DataFrame,
    *,
    load_delay: pd.Timedelta,
    load_min: float = 0.0,
    max_gap: pd.Timedelta = DEFAULT_MAX_GAP,
) -> QualityReport:
    """Validate metered load with an availability delay."""
    issues: list[QualityIssue] = []
    if frame.empty:
        issues.append(_issue("columns", "versioned load table is empty"))
        return QualityReport("versioned_load", issues, {"rows": 0, "coverage": 0.0})
    if not _require_columns(issues, frame, list(VERSIONED_LOAD_COLUMNS)):
        return QualityReport("versioned_load", issues, {"rows": int(len(frame))})
    _aware_column(frame, "timestamp", utc=False, issues=issues)
    _aware_column(frame, "available_at", utc=False, issues=issues)
    catalog = _catalog_issue(frame)
    if catalog:
        issues.append(catalog)
    duplicate = _duplicate_issue(frame, ["timestamp", "load_area"])
    if duplicate:
        issues.append(duplicate)
    load_issue = _load_value_issue(frame, load_min)
    if load_issue:
        issues.append(load_issue)
    if _is_datetime(frame["timestamp"]) and _is_datetime(frame["available_at"]):
        if (frame["available_at"] < frame["timestamp"]).any():
            issues.append(_issue("point_in_time", "load is available before its hour begins"))
        expected = frame["timestamp"] + load_delay
        if frame["available_at"].dt.tz is not None and not frame["available_at"].equals(expected):
            issues.append(
                _issue(
                    "point_in_time",
                    "available_at must equal timestamp plus the load availability delay",
                )
            )
    chronological, stats = _chronological_issues(
        frame, "timestamp", group_column="load_area", max_gap=max_gap
    )
    issues.extend(chronological)
    return QualityReport("versioned_load", issues, stats)
