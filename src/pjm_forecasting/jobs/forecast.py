"""Run the live forecast job once, on a loop, or immediately with ``--now``."""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")

from pjm_forecasting.flows.live import forecast_live
from pjm_forecasting.schedule import should_run
from pjm_forecasting.settings import load_settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live PJM load forecast job")
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--areas", default="all", help="Comma-separated load areas")
    parser.add_argument(
        "--now",
        action="store_true",
        help="Run immediately, ignoring schedule_et",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Stay running and check the schedule every 15 minutes",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse cached weather runs and load files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Publish even when the watermark is unchanged",
    )
    return parser.parse_args(argv)


def _run_once(args: argparse.Namespace) -> int:
    result = forecast_live(
        config_path=args.config,
        areas=args.areas,
        force=args.force or args.now,
        skip_download=args.skip_download,
        skip_schedule=args.now or args.force,
    )
    if result.get("skipped"):
        print(f"skipped: {result.get('reason')}", flush=True)
        return 0
    print(
        f"published origin={result.get('origin')} "
        f"weather_run={result.get('weather_run')} "
        f"path={result.get('path')}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.loop:
        settings = load_settings(args.config)
        print(f"looping; schedule_et={list(settings.schedule_et)}", flush=True)
        while True:
            if args.now or should_run(settings.schedule_et):
                try:
                    _run_once(args)
                except Exception as exc:  # noqa: BLE001
                    print(f"forecast failed: {exc}", file=sys.stderr, flush=True)
                # Avoid double-firing inside the same window.
                time.sleep(15 * 60)
            else:
                time.sleep(60)
    try:
        return _run_once(args)
    except Exception as exc:  # noqa: BLE001
        print(f"forecast failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
