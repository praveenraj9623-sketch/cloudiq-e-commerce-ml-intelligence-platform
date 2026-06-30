"""Tests for the optional AWS S3 verifier."""

from __future__ import annotations

import pytest

from aws.verify_s3 import (
    S3Config,
    S3VerificationError,
    describe_s3_error,
    resolve_s3_config,
    verify_s3_access,
)


class FakeS3Error(Exception):
    """Tiny AWS-style exception test double."""

    def __init__(
        self,
        code: str,
        message: str = "sensitive internal detail",
        bucket_region: str | None = None,
    ) -> None:
        """Create a fake exception with a botocore-like response shape."""
        headers = {}
        if bucket_region:
            headers["x-amz-bucket-region"] = bucket_region
        self.response = {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPHeaders": headers},
        }
        super().__init__(message)


class FakeS3Client:
    """Fake S3 client that returns or raises from ``list_objects_v2``."""

    def __init__(self, response: dict | None = None, error: Exception | None = None) -> None:
        """Store the fake response or exception."""
        self.response = response or {}
        self.error = error
        self.called_bucket: str | None = None

    def list_objects_v2(self, *, Bucket: str) -> dict:
        """Return the fake response without making a network call."""
        self.called_bucket = Bucket
        if self.error:
            raise self.error
        return self.response


def test_resolve_s3_config_requires_bucket_and_region() -> None:
    """Required S3 environment variables are validated explicitly."""
    with pytest.raises(S3VerificationError) as excinfo:
        resolve_s3_config({"AWS_S3_BUCKET": "", "AWS_DEFAULT_REGION": ""})

    message = str(excinfo.value)
    assert "AWS_S3_BUCKET" in message
    assert "AWS_DEFAULT_REGION" in message


def test_verify_s3_access_returns_safe_object_summary() -> None:
    """Successful verification returns only bucket, region, keys, and sizes."""
    client = FakeS3Client(
        {
            "KeyCount": 2,
            "Contents": [
                {"Key": "raw/olist_orders_dataset.csv", "Size": 123},
                {"Key": "raw/olist_order_items_dataset.csv", "Size": 456},
            ],
        }
    )

    result = verify_s3_access(
        S3Config(bucket="marketplace-lakehouse-demo-9623", region="ap-south-1"),
        client=client,
    )

    assert client.called_bucket == "marketplace-lakehouse-demo-9623"
    assert result.object_count == 2
    assert result.objects[0].key == "raw/olist_orders_dataset.csv"
    assert result.objects[1].size == 456


def test_access_denied_error_does_not_expose_exception_internals() -> None:
    """AccessDenied is converted to a clear, secret-free message."""
    message = describe_s3_error(
        FakeS3Error("AccessDenied", "top-secret provider detail"),
        "marketplace-lakehouse-demo-9623",
        "ap-south-1",
    )

    assert "Access denied" in message
    assert "top-secret" not in message


def test_wrong_region_error_mentions_configured_and_expected_regions() -> None:
    """Wrong-region errors surface the configured and hinted regions."""
    message = describe_s3_error(
        FakeS3Error("PermanentRedirect", bucket_region="us-east-1"),
        "marketplace-lakehouse-demo-9623",
        "ap-south-1",
    )

    assert "ap-south-1" in message
    assert "us-east-1" in message


def test_verify_s3_access_wraps_missing_bucket_error() -> None:
    """Missing-bucket errors are raised as user-safe verifier errors."""
    client = FakeS3Client(error=FakeS3Error("NoSuchBucket"))

    with pytest.raises(S3VerificationError) as excinfo:
        verify_s3_access(
            S3Config(bucket="missing-bucket", region="ap-south-1"),
            client=client,
        )

    assert "was not found" in str(excinfo.value)
