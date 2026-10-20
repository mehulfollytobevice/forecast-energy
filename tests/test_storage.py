"""Local object-store fallback."""

from __future__ import annotations

from pathlib import Path

from pjm_forecasting.storage import ObjectStore, StorageConfig


def test_local_write_read_json(tmp_path: Path):
    store = ObjectStore(
        StorageConfig(
            bucket="",
            endpoint_url=None,
            access_key=None,
            secret_key=None,
            region="us-east-1",
            local_root=tmp_path,
        )
    )
    store.write_json("snapshots/forecast/latest.json", {"ok": True, "n": 1})
    assert store.exists("snapshots/forecast/latest.json")
    payload = store.read_json("snapshots/forecast/latest.json")
    assert payload == {"ok": True, "n": 1}


def test_local_upload_download_file(tmp_path: Path):
    store = ObjectStore(
        StorageConfig(
            bucket="",
            endpoint_url=None,
            access_key=None,
            secret_key=None,
            region="us-east-1",
            local_root=tmp_path / "bucket",
        )
    )
    source = tmp_path / "model.bin"
    source.write_bytes(b"abc")
    store.upload_file(source, "models/v1.bin")
    dest = tmp_path / "out.bin"
    store.download_file("models/v1.bin", dest)
    assert dest.read_bytes() == b"abc"
