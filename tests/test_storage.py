"""Local object-store fallback."""

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_read_json_maps_s3_missing_key(tmp_path: Path):
    class NoSuchKey(Exception):
        pass

    class FakeS3:
        def get_object(self, Bucket: str, Key: str):
            raise NoSuchKey("The specified key does not exist.")

    store = ObjectStore(
        StorageConfig(
            bucket="snapshots",
            endpoint_url="https://example.invalid",
            access_key="test",
            secret_key="test",
            region="us-east-1",
            local_root=tmp_path,
        )
    )
    store._client = FakeS3()
    with pytest.raises(FileNotFoundError):
        store.read_json("snapshots/forecast/latest.json")


def test_download_retries_canceled_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class FakeConfig:
        connect_timeout = 60
        read_timeout = 300

    class FakeS3:
        def __init__(self) -> None:
            self.calls = 0
            self.meta = type("meta", (), {"config": FakeConfig()})()

        def head_object(self, Bucket: str, Key: str):
            return {"ContentLength": 3}

        def download_file(self, bucket: str, key: str, dest: str, Config=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError(
                    "An error occurred (RequestCanceled) when calling the GetObject "
                    "operation: Request is canceled."
                )
            Path(dest).write_bytes(b"abc")

    monkeypatch.setattr("pjm_forecasting.storage.time.sleep", lambda _s: None)
    fake = FakeS3()
    store = ObjectStore(
        StorageConfig(
            bucket="snapshots",
            endpoint_url="https://example.invalid",
            access_key="test",
            secret_key="test",
            region="us-east-1",
            local_root=tmp_path,
        )
    )
    store._client = fake
    dest = tmp_path / "bundle.tar.gz"
    store.download_file("models/latest.tar.gz", dest)
    assert dest.read_bytes() == b"abc"
    assert fake.calls == 2
