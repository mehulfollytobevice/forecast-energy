"""Local or S3-compatible object storage for models, caches, and snapshots."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from pjm_forecasting.utils import ensure_dir, project_root

_CONNECT_TIMEOUT_S = 60
_READ_TIMEOUT_S = 300
_DOWNLOAD_ATTEMPTS = 3


def _is_missing_key(exc: BaseException) -> bool:
    """True when S3 (or a compatible store) says the object key is absent."""
    if type(exc).__name__ in {"NoSuchKey", "NotFound"}:
        return True
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    code = str((response.get("Error") or {}).get("Code") or "")
    if code in {"NoSuchKey", "NotFound", "404"}:
        return True
    return (response.get("ResponseMetadata") or {}).get("HTTPStatusCode") == 404


def _is_canceled_request(exc: BaseException) -> bool:
    """True when an S3 transfer aborted a GetObject (timeout or canceled sibling part)."""
    name = type(exc).__name__
    if name in {"RequestCanceled", "ReadTimeoutError", "ConnectTimeoutError", "EndpointConnectionError"}:
        return True
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = str((response.get("Error") or {}).get("Code") or "")
        if code in {"RequestCanceled", "RequestTimeout"}:
            return True
    text = str(exc).lower()
    return "request is canceled" in text or "read timeout" in text


@dataclass(frozen=True)
class StorageConfig:
    """Bucket credentials. Empty ``bucket`` means local filesystem mode."""

    bucket: str
    endpoint_url: str | None
    access_key: str | None
    secret_key: str | None
    region: str
    local_root: Path


def load_storage_config(*, local_root: Path | str | None = None) -> StorageConfig:
    load_dotenv(project_root() / ".env")
    root = Path(local_root) if local_root is not None else project_root() / "data"
    return StorageConfig(
        bucket=os.environ.get("S3_BUCKET", "").strip(),
        endpoint_url=os.environ.get("S3_ENDPOINT_URL", "").strip() or None,
        access_key=os.environ.get("AWS_ACCESS_KEY_ID", "").strip() or None,
        secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip() or None,
        region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1").strip() or "us-east-1",
        local_root=Path(root),
    )


class ObjectStore:
    """Thin wrapper: local ``data/`` when no bucket is configured, else S3."""

    def __init__(self, config: StorageConfig | None = None) -> None:
        self.config = config or load_storage_config()
        self._client: Any = None

    @property
    def is_local(self) -> bool:
        return not self.config.bucket

    def _s3(self) -> Any:
        if self._client is None:
            import boto3
            from botocore.config import Config

            kwargs: dict[str, Any] = {
                "region_name": self.config.region,
                "config": Config(
                    connect_timeout=_CONNECT_TIMEOUT_S,
                    read_timeout=_READ_TIMEOUT_S,
                    retries={"max_attempts": 8, "mode": "standard"},
                ),
            }
            if self.config.endpoint_url:
                kwargs["endpoint_url"] = self.config.endpoint_url
            if self.config.access_key and self.config.secret_key:
                kwargs["aws_access_key_id"] = self.config.access_key
                kwargs["aws_secret_access_key"] = self.config.secret_key
            self._client = boto3.client("s3", **kwargs)
        return self._client

    def local_path(self, key: str) -> Path:
        return self.config.local_root / key

    def upload_file(self, local: Path | str, key: str) -> str:
        source = Path(local)
        if not source.is_file():
            raise FileNotFoundError(source)
        if self.is_local:
            destination = self.local_path(key)
            ensure_dir(destination.parent)
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            return str(destination)
        self._s3().upload_file(str(source), self.config.bucket, key)
        return f"s3://{self.config.bucket}/{key}"

    def download_file(self, key: str, local: Path | str) -> Path:
        destination = Path(local)
        ensure_dir(destination.parent)
        if self.is_local:
            source = self.local_path(key)
            if not source.is_file():
                raise FileNotFoundError(f"missing local object {key}")
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            return destination
        from boto3.s3.transfer import TransferConfig

        transfer = TransferConfig(
            multipart_threshold=64 * 1024 * 1024,
            multipart_chunksize=64 * 1024 * 1024,
            max_concurrency=2,
            num_download_attempts=8,
        )
        last_exc: Exception | None = None
        for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
            try:
                self._s3().download_file(
                    self.config.bucket,
                    key,
                    str(destination),
                    Config=transfer,
                )
                return destination
            except Exception as exc:
                last_exc = exc
                if not _is_canceled_request(exc) or attempt >= _DOWNLOAD_ATTEMPTS:
                    raise
                time.sleep(2 ** attempt)
        raise last_exc  # pragma: no cover

    def exists(self, key: str) -> bool:
        if self.is_local:
            return self.local_path(key).is_file()
        try:
            self._s3().head_object(Bucket=self.config.bucket, Key=key)
            return True
        except Exception:
            return False

    def read_json(self, key: str) -> dict[str, Any]:
        if self.is_local:
            path = self.local_path(key)
            if not path.is_file():
                raise FileNotFoundError(key)
            return json.loads(path.read_text())
        try:
            response = self._s3().get_object(Bucket=self.config.bucket, Key=key)
        except Exception as exc:
            if _is_missing_key(exc):
                raise FileNotFoundError(key) from exc
            raise
        body = response["Body"].read()
        return json.loads(body)

    def write_json(self, key: str, payload: dict[str, Any]) -> str:
        text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
        if self.is_local:
            path = self.local_path(key)
            ensure_dir(path.parent)
            path.write_text(text)
            return str(path)
        self._s3().put_object(
            Bucket=self.config.bucket,
            Key=key,
            Body=text.encode("utf-8"),
            ContentType="application/json",
        )
        return f"s3://{self.config.bucket}/{key}"

    def sync_prefix(self, local_dir: Path | str, key_prefix: str) -> int:
        """Upload every file under ``local_dir`` into ``key_prefix``."""
        root = Path(local_dir)
        if not root.is_dir():
            raise FileNotFoundError(root)
        count = 0
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            key = f"{key_prefix.rstrip('/')}/{relative}"
            self.upload_file(path, key)
            count += 1
        return count

    def download_prefix(self, key_prefix: str, local_dir: Path | str) -> int:
        """Download every object under ``key_prefix`` into ``local_dir``."""
        dest_root = Path(local_dir)
        ensure_dir(dest_root)
        prefix = key_prefix.rstrip("/") + "/"
        if self.is_local:
            source_root = self.local_path(key_prefix)
            if not source_root.is_dir():
                return 0
            count = 0
            for path in sorted(source_root.rglob("*")):
                if not path.is_file():
                    continue
                dest = dest_root / path.relative_to(source_root)
                ensure_dir(dest.parent)
                if path.resolve() != dest.resolve():
                    shutil.copy2(path, dest)
                count += 1
            return count
        count = 0
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self.config.bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            response = self._s3().list_objects_v2(**kwargs)
            for item in response.get("Contents") or []:
                key = str(item.get("Key") or "")
                if not key or key.endswith("/"):
                    continue
                relative = key[len(prefix) :]
                if not relative:
                    continue
                self.download_file(key, dest_root / relative)
                count += 1
            if not response.get("IsTruncated"):
                break
            token = response.get("NextContinuationToken")
            if not token:
                break
        return count
