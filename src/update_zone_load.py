"""Fill PJM hourly metered-load caches, one file per load area.

Each month of ``hrl_load_metered`` is fetched once, then split into
``data/raw/pjm/load_areas/{LOAD_AREA}/YYYY-MM.parquet``. ``RTO`` totals are
not stored. Months inside the revision window are fetched again.

Requires ``PJM_API_KEY`` in ``.env``. The default span is config
``data.history_start`` (2018-01-01) through today, US/Eastern.

Gap-fill every load area, and refresh the recent revision window::

    python src/update_zone_load.py

Show the plan without calling Data Miner::

    python src/update_zone_load.py --dry-run

One load area, gaps only::

    python src/update_zone_load.py --areas AECO --refresh-days 0
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from pjm_forecasting.areas import resolve_areas
from pjm_forecasting.ingest.pjm import (
    PJMClient,
    assert_rollups_match,
    empty_load_frame,
    to_load_area_hours,
)
from pjm_forecasting.utils import EASTERN, ensure_dir, load_config, month_range, project_root

DEFAULT_REFRESH_DAYS = 90
DEFAULT_HISTORY_START = "2018-01-01"


@dataclass(frozen=True)
class MonthPlan:
    """One calendar month, and whether this run should fetch it."""

    month: str
    path: Path
    status: Literal["missing", "refresh", "skipped"]


@dataclass(frozen=True)
class MonthResult:
    """One month written, refreshed, skipped, or only planned."""

    month: str
    path: Path
    status: Literal["written", "refreshed", "skipped", "missing", "refresh"]
    rows: int | None = None


def default_cache_root() -> Path:
    """Parent of ``load_areas/``. Monthly files live under each load area."""
    return project_root() / "data" / "raw" / "pjm"


def default_history_start() -> str:
    """``data.history_start`` from the default config, otherwise 2018-01-01."""
    try:
        start = load_config().get("data", {}).get("history_start")
    except (OSError, TypeError, ValueError):
        start = None
    if not start:
        return DEFAULT_HISTORY_START
    return f"{pd.Timestamp(start):%Y-%m-%d}"


def eastern_today(now: pd.Timestamp | str | None = None) -> str:
    """Calendar date in US/Eastern as ``YYYY-MM-DD``."""
    stamp = pd.Timestamp.now(tz=EASTERN) if now is None else pd.Timestamp(now)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(EASTERN)
    else:
        stamp = stamp.tz_convert(EASTERN)
    return f"{stamp:%Y-%m-%d}"


def area_month_path(cache_root: Path | str, area: str, month: str) -> Path:
    """``{cache_root}/load_areas/{AREA}/YYYY-MM.parquet``."""
    return Path(cache_root) / "load_areas" / area / f"{month}.parquet"


def parse_refresh_days(value: int) -> int:
    """Non-negative day count. Zero fills missing months and refreshes nothing."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"refresh_days must be an integer >= 0, got {value!r}")
    return value


def parse_history_window(start: str, end: str) -> tuple[str, str]:
    """Inclusive ``YYYY-MM-DD`` bounds. ``end`` may be a timestamp; the date is kept."""
    try:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"could not parse start {start!r} or end {end!r}") from exc
    if pd.isna(start_ts) or pd.isna(end_ts):
        raise ValueError(f"could not parse start {start!r} or end {end!r}")
    if end_ts.normalize() < start_ts.normalize():
        raise ValueError(f"end {end!r} must be on or after start {start!r}")
    return f"{start_ts:%Y-%m-%d}", f"{end_ts:%Y-%m-%d}"


def _naive_day(value: pd.Timestamp | str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(EASTERN).tz_localize(None)
    return stamp.normalize()


def month_needs_refresh(
    month: pd.Timestamp,
    *,
    today: pd.Timestamp | str,
    refresh_days: int,
) -> bool:
    """True when this month overlaps the trailing revision window.

    Cache files are monthly, so the whole month is refreshed when any day of
    it falls in ``[today - refresh_days, today]``. Zero days refreshes nothing.
    """
    if refresh_days <= 0:
        return False
    today_day = _naive_day(today)
    window_start = today_day - pd.Timedelta(days=int(refresh_days))
    month_start = pd.Timestamp(month).normalize().replace(day=1)
    if getattr(month_start, "tzinfo", None) is not None:
        month_start = month_start.tz_convert(EASTERN).tz_localize(None)
    month_end = (month_start + pd.offsets.MonthBegin(1)) - pd.Timedelta(days=1)
    return bool(month_end >= window_start and month_start <= today_day)


def _month_complete(cache_root: Path, areas: list[str], month: str) -> bool:
    return all(area_month_path(cache_root, area, month).is_file() for area in areas)


def plan_months(
    cache_root: Path | str,
    start: str,
    end: str,
    *,
    areas: list[str],
    refresh_days: int,
    today: pd.Timestamp | str,
) -> list[MonthPlan]:
    """Decide which months to skip, fill, or refresh. Does not download."""
    cache_root = Path(cache_root)
    start_s, end_s = parse_history_window(start, end)
    refresh_days = parse_refresh_days(refresh_days)
    plans: list[MonthPlan] = []
    for month in month_range(start_s, end_s):
        label = f"{month:%Y-%m}"
        path = cache_root / "load_areas"
        if not _month_complete(cache_root, areas, label):
            status: Literal["missing", "refresh", "skipped"] = "missing"
        elif month_needs_refresh(month, today=today, refresh_days=refresh_days):
            status = "refresh"
        else:
            status = "skipped"
        plans.append(MonthPlan(month=label, path=path, status=status))
    return plans


def _month_span(months: list[str]) -> str:
    if not months:
        return "none"
    if len(months) <= 4:
        return ", ".join(months)
    return f"{months[0]} .. {months[-1]} ({len(months)} months)"


def _print_plan(areas: list[str], plans: list[MonthPlan]) -> None:
    grouped = {
        status: [plan.month for plan in plans if plan.status == status]
        for status in ("missing", "refresh", "skipped")
    }
    print(f"load areas={len(areas)} cache=load_areas/", flush=True)
    print(f"  missing {_month_span(grouped['missing'])}", flush=True)
    print(f"  refresh {_month_span(grouped['refresh'])}", flush=True)
    print(f"  skipped {len(grouped['skipped'])}", flush=True)


def fetch_month(client: PJMClient, month: pd.Timestamp) -> pd.DataFrame:
    """Download one calendar month. Callers must not run months in parallel."""
    return client.fetch_month(month)


def _write_area_month(
    hours: pd.DataFrame,
    cache_root: Path,
    area: str,
    month: str,
) -> int:
    part = hours.loc[hours["load_area"] == area].copy()
    if part.empty:
        part = empty_load_frame()
    path = area_month_path(cache_root, area, month)
    ensure_dir(path.parent)
    part.to_parquet(path, index=False)
    return len(part)


def update_load_areas(
    areas: list[str],
    start: str,
    end: str,
    *,
    cache_root: Path | str,
    refresh_days: int = DEFAULT_REFRESH_DAYS,
    dry_run: bool = False,
    today: pd.Timestamp | str | None = None,
) -> list[MonthResult]:
    """Fill missing months and refresh the recent window, one month at a time."""
    codes = resolve_areas(",".join(areas) if areas else "all")
    start_s, end_s = parse_history_window(start, end)
    refresh_days = parse_refresh_days(refresh_days)
    today_value = eastern_today() if today is None else today
    cache_root = Path(cache_root)
    print(
        f"window {start_s} to {end_s} (inclusive) refresh_days={refresh_days} "
        f"dry_run={dry_run}",
        flush=True,
    )
    plans = plan_months(
        cache_root,
        start_s,
        end_s,
        areas=codes,
        refresh_days=refresh_days,
        today=today_value,
    )
    _print_plan(codes, plans)
    if dry_run:
        results = [
            MonthResult(month=plan.month, path=plan.path, status=plan.status) for plan in plans
        ]
        _print_summary(results, dry_run=True)
        return results

    client: PJMClient | None = None
    results: list[MonthResult] = []
    for plan in plans:
        if plan.status == "skipped":
            results.append(MonthResult(month=plan.month, path=plan.path, status="skipped"))
            continue
        if client is None:
            client = PJMClient()
        print(f"fetch {plan.month}", flush=True)
        started = time.perf_counter()
        raw = fetch_month(client, pd.Timestamp(f"{plan.month}-01"))
        hours = to_load_area_hours(raw)
        assert_rollups_match(hours)
        written = 0
        for code in codes:
            written += _write_area_month(hours, cache_root, code, plan.month)
        elapsed = time.perf_counter() - started
        status: Literal["written", "refreshed"] = (
            "written" if plan.status == "missing" else "refreshed"
        )
        print(
            f"wrote {plan.month} {written} rows across {len(codes)} load areas "
            f"in {elapsed:.1f}s",
            flush=True,
        )
        results.append(
            MonthResult(month=plan.month, path=plan.path, status=status, rows=written)
        )
    _print_summary(results, dry_run=False)
    return results


def _print_summary(results: list[MonthResult], *, dry_run: bool) -> None:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    if dry_run:
        print(
            "dry-run: "
            f"{counts.get('missing', 0)} missing, "
            f"{counts.get('refresh', 0)} refresh, "
            f"{counts.get('skipped', 0)} skipped",
            flush=True,
        )
        return
    print(
        "done: "
        f"{counts.get('written', 0)} written, "
        f"{counts.get('refreshed', 0)} refreshed, "
        f"{counts.get('skipped', 0)} skipped",
        flush=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI for ``python src/update_zone_load.py``."""
    parser = argparse.ArgumentParser(
        description=(
            "Update PJM hourly metered load for each load area. "
            "Files land in data/raw/pjm/load_areas/{LOAD_AREA}/YYYY-MM.parquet. "
            "Months are fetched one at a time."
        ),
        epilog=(
            "examples:\n"
            "  python src/update_zone_load.py\n"
            "  python src/update_zone_load.py --dry-run\n"
            "  python src/update_zone_load.py --areas AECO,UGI --refresh-days 0\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start",
        default=None,
        help=f"Inclusive start date (default {DEFAULT_HISTORY_START}, from config).",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Inclusive end date YYYY-MM-DD (default today, US/Eastern).",
    )
    parser.add_argument(
        "--areas",
        default="all",
        help="all (default) or a comma-separated list of load-area codes.",
    )
    parser.add_argument(
        "--refresh-days",
        type=int,
        default=DEFAULT_REFRESH_DAYS,
        help=(
            "Re-download cached months that overlap the last N days "
            f"(default {DEFAULT_REFRESH_DAYS}). 0 fills gaps only."
        ),
    )
    parser.add_argument(
        "--cache-root",
        default=None,
        help="Parent of load_areas/ (default data/raw/pjm).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print missing, refresh, and skipped months without calling Data Miner.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Parse args and update caches. Returns a process status code."""
    args = parse_args(argv)
    try:
        areas = resolve_areas(args.areas)
        refresh_days = parse_refresh_days(args.refresh_days)
        start = args.start or default_history_start()
        end = args.end or eastern_today()
        start, end = parse_history_window(start, end)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    cache_root = Path(args.cache_root) if args.cache_root else default_cache_root()
    print(
        f"areas={len(areas)} start={start} end={end} "
        f"refresh_days={refresh_days} dry_run={args.dry_run} cache={cache_root}",
        flush=True,
    )
    update_load_areas(
        areas,
        start,
        end,
        cache_root=cache_root,
        refresh_days=refresh_days,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
