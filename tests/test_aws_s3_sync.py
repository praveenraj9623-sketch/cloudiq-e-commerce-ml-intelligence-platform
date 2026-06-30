"""Tests for the optional raw Olist S3 sync utility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aws.sync_raw_to_s3 import (
    DEFAULT_S3_PREFIX,
    EXPECTED_OLIST_FILES,
    MANIFEST_KEY,
    AwsSyncConfig,
    S3SyncError,
    find_raw_source_dir,
    normalize_prefix,
    sync_raw_to_s3,
)


class FakeS3Error(Exception):
    """Tiny AWS-style exception test double."""

    def __init__(self, code: str, message: str = "sensitive provider detail") -> None:
        """Create a fake exception with a botocore-like response shape."""
        self.response = {"Error": {"Code": code, "Message": message}}
        super().__init__(message)


class FakeS3Client:
    """Fake S3 client for sync tests."""

    def __init__(
        self,
        remote: dict[str, int] | None = None,
        list_error: Exception | None = None,
    ) -> None:
        """Store fake remote objects and optional list failure."""
        self.remote = remote or {}
        self.list_error = list_error
        self.uploads: list[tuple[str, str, str]] = []
        self.puts: list[dict] = []

    def list_objects_v2(self, **kwargs) -> dict:
        """Return fake remote objects under the requested prefix."""
        if self.list_error:
            raise self.list_error
        prefix = kwargs["Prefix"]
        contents = [
            {"Key": key, "Size": size}
            for key, size in sorted(self.remote.items())
            if key.startswith(prefix)
        ]
        return {"Contents": contents, "IsTruncated": False}

    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:
        """Record an upload call."""
        self.uploads.append((Filename, Bucket, Key))

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:
        """Record a manifest put call."""
        self.puts.append(
            {
                "Bucket": Bucket,
                "Key": Key,
                "Body": Body,
                "ContentType": ContentType,
            }
        )


def _write_raw_repo(tmp_path: Path) -> Path:
    """Create a tiny repo-like raw directory with all expected Olist files."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    for filename in EXPECTED_OLIST_FILES:
        (raw_dir / filename).write_text(f"{filename}\n1\n", encoding="utf-8")
    return raw_dir


def _config() -> AwsSyncConfig:
    """Return the expected test config."""
    return AwsSyncConfig(
        bucket="marketplace-lakehouse-demo-9623",
        region="ap-south-1",
        prefix=DEFAULT_S3_PREFIX,
    )


def test_find_raw_source_dir_detects_existing_olist_csv_directory(tmp_path: Path) -> None:
    """Raw source detection verifies the expected local Olist file set."""
    raw_dir = _write_raw_repo(tmp_path)

    assert find_raw_source_dir(tmp_path) == raw_dir


def test_normal_sync_uploads_only_missing_flat_raw_keys(tmp_path: Path) -> None:
    """Existing source CSV objects are skipped and missing files use raw/<file> keys."""
    _write_raw_repo(tmp_path)
    existing_key = "raw/olist_customers_dataset.csv"
    client = FakeS3Client(remote={existing_key: 10})

    summary = sync_raw_to_s3(_config(), client, tmp_path)

    uploaded_keys = {upload[2] for upload in client.uploads}
    assert summary.discovered == 9
    assert summary.skipped == 1
    assert summary.uploaded == 8
    assert summary.failed == 0
    assert existing_key not in uploaded_keys
    assert all(key.startswith("raw/") for key in uploaded_keys)
    assert not any(key.startswith("raw/olist/") for key in uploaded_keys)


def test_overwrite_uploads_existing_source_objects(tmp_path: Path) -> None:
    """The explicit overwrite flag permits replacing existing source CSV objects."""
    _write_raw_repo(tmp_path)
    remote = {f"raw/{filename}": 1 for filename in EXPECTED_OLIST_FILES}
    client = FakeS3Client(remote=remote)

    summary = sync_raw_to_s3(_config(), client, tmp_path, overwrite=True)

    assert summary.skipped == 0
    assert summary.uploaded == 9
    assert len(client.uploads) == 9


def test_publish_manifest_uploads_only_manifest(tmp_path: Path) -> None:
    """Manifest publishing proves PutObject access without source CSV uploads."""
    _write_raw_repo(tmp_path)
    client = FakeS3Client()

    summary = sync_raw_to_s3(_config(), client, tmp_path, publish_manifest=True)

    assert summary.discovered == 9
    assert summary.skipped == 9
    assert summary.uploaded == 1
    assert summary.manifest_published
    assert client.uploads == []
    assert len(client.puts) == 1
    assert client.puts[0]["Key"] == MANIFEST_KEY
    manifest = json.loads(client.puts[0]["Body"].decode("utf-8"))
    assert manifest["file_count"] == 9
    assert {"filename", "byte_size", "sha256"} <= set(manifest["files"][0])


def test_dry_run_prints_plan_without_uploading(tmp_path: Path, capsys) -> None:
    """Dry-run mode prints planned keys and performs no S3 writes."""
    _write_raw_repo(tmp_path)
    client = FakeS3Client()

    summary = sync_raw_to_s3(_config(), client, tmp_path, dry_run=True)

    captured = capsys.readouterr()
    assert "s3://marketplace-lakehouse-demo-9623/raw/olist_orders_dataset.csv" in captured.out
    assert summary.uploaded == 0
    assert client.uploads == []
    assert client.puts == []


def test_rejects_nested_raw_olist_prefix() -> None:
    """The utility refuses to create the obsolete raw/olist/ prefix."""
    with pytest.raises(S3SyncError):
        normalize_prefix("raw/olist/")


def test_list_error_is_user_safe(tmp_path: Path) -> None:
    """S3 list errors are wrapped without provider internals."""
    _write_raw_repo(tmp_path)
    client = FakeS3Client(list_error=FakeS3Error("AccessDenied", "secret detail"))

    with pytest.raises(S3SyncError) as excinfo:
        sync_raw_to_s3(_config(), client, tmp_path)

    message = str(excinfo.value)
    assert "Access denied" in message
    assert "secret detail" not in message
