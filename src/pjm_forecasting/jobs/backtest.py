"""Build the dashboard backtest snapshot and publish it to object storage."""

from __future__ import annotations

import argparse
import sys

from pjm_forecasting.backtest_snapshot import publish_backtest_snapshot, write_backtest_snapshot


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write (and optionally publish) the dashboard backtest snapshot"
    )
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument(
        "--areas",
        default=None,
        help="Comma-separated load areas (default: every area with a parquet on disk)",
    )
    parser.add_argument(
        "--lead-hours",
        type=int,
        default=1,
        help="Lead time to collapse Beam predictions to (default: 1)",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Write data/processed/backtest_snapshot.json without uploading",
    )
    parser.add_argument(
        "--no-fixture-fill",
        action="store_true",
        help="Do not fill missing load areas from the committed fixture",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    areas = [item.strip().upper() for item in args.areas.split(",") if item.strip()] if args.areas else None
    kwargs = {
        "settings": None,
        "lead_hours": args.lead_hours,
        "areas": areas,
        "fill_missing_from_fixture": not args.no_fixture_fill,
    }
    if args.config:
        from pjm_forecasting.settings import load_settings

        kwargs["settings"] = load_settings(args.config)
    try:
        if args.no_publish:
            path = write_backtest_snapshot(**kwargs)
            print(f"wrote {path}", flush=True)
            return 0
        result = publish_backtest_snapshot(**kwargs)
        print(
            f"published key={result.get('key')} "
            f"start={result.get('start')} end={result.get('end')} "
            f"path={result.get('path')}",
            flush=True,
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"backtest snapshot failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
