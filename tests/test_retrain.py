"""Retrain hydrates empty local caches from object storage."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pjm_forecasting.flows.retrain import hydrate_training_inputs


class _FakeStore:
    is_local = False

    def download_prefix(self, key_prefix: str, local_dir: Path | str) -> int:
        dest = Path(local_dir)
        dest.mkdir(parents=True, exist_ok=True)
        if key_prefix == "raw/forecast_runs":
            (dest / "AECO.parquet").write_bytes(b"weather")
            return 1
        month = dest / "AECO" / "2018-01.parquet"
        month.parent.mkdir(parents=True, exist_ok=True)
        month.write_bytes(b"load")
        return 1


def test_hydrate_downloads_when_local_caches_are_empty(tmp_path: Path):
    settings = SimpleNamespace(
        versioned_predictor_dir=lambda: tmp_path / "processed" / "forecast_runs",
        raw_dir=tmp_path / "raw",
    )
    counts = hydrate_training_inputs(settings, _FakeStore())
    assert counts == {"forecast_runs": 1, "load_areas": 1}
    assert (tmp_path / "processed" / "forecast_runs" / "AECO.parquet").is_file()
    assert (tmp_path / "raw" / "pjm" / "load_areas" / "AECO" / "2018-01.parquet").is_file()


def test_hydrate_skips_when_local_caches_exist(tmp_path: Path):
    pred = tmp_path / "processed" / "forecast_runs"
    load = tmp_path / "raw" / "pjm" / "load_areas" / "AECO"
    pred.mkdir(parents=True)
    load.mkdir(parents=True)
    (pred / "AECO.parquet").write_bytes(b"weather")
    (load / "2018-01.parquet").write_bytes(b"load")
    settings = SimpleNamespace(
        versioned_predictor_dir=lambda: pred,
        raw_dir=tmp_path / "raw",
    )
    counts = hydrate_training_inputs(settings, _FakeStore())
    assert counts == {"forecast_runs": 0, "load_areas": 0}
