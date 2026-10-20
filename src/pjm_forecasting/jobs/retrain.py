"""Weekly (or on-demand) retrain on forecast-run weather and publish models."""

from __future__ import annotations

import argparse
import sys

from pjm_forecasting.flows.retrain import retrain_forecast_models
from pjm_forecasting.models_bundle import publish_archive_fallback
from pjm_forecasting.settings import load_settings
from pjm_forecasting.storage import ObjectStore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain forecast-run weather models")
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--areas", default="all", help="Comma-separated load areas")
    parser.add_argument("--aggregates", default="all", help="Comma-separated aggregates")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Upload local caches and the archive-fallback bundle, then retrain",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Train and calibrate without updating models/latest.json",
    )
    parser.add_argument(
        "--skip-archive-fallback",
        action="store_true",
        help="Do not package the archive-trained fallback",
    )
    return parser.parse_args(argv)


def _bootstrap_upload(config_path: str | None) -> None:
    settings = load_settings(config_path)
    store = ObjectStore()
    print("bootstrap: syncing processed forecast_runs and load caches", flush=True)
    if settings.versioned_predictor_dir().is_dir():
        n = store.sync_prefix(settings.versioned_predictor_dir(), "raw/forecast_runs")
        print(f"uploaded {n} forecast-run tables", flush=True)
    load_root = settings.raw_dir / "pjm" / "load_areas"
    if load_root.is_dir():
        n = store.sync_prefix(load_root, "raw/pjm/load_areas")
        print(f"uploaded {n} load-area files", flush=True)
    try:
        publish_archive_fallback(settings=settings, store=store)
        print("published archive-fallback", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"archive-fallback skipped: {exc}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.bootstrap:
            _bootstrap_upload(args.config)
        result = retrain_forecast_models(
            config_path=args.config,
            areas=args.areas,
            aggregates=args.aggregates,
            skip_archive_fallback=args.skip_archive_fallback or args.bootstrap,
            publish=not args.no_publish,
        )
        if result.get("publish_blocked"):
            print("retrain finished but publish gate blocked the new bundle", flush=True)
            return 2
        print(
            f"retrain done published={result.get('published')} "
            f"frame={result.get('frame_path')}",
            flush=True,
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"retrain failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
