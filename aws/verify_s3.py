"""Read-only AWS S3 access verifier for CloudIQ.

The script loads local environment variables, uses boto3's default credential
resolution chain, and calls ``list_objects_v2`` against the configured bucket.
It never prints credentials or mutates the bucket.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from dotenv import load_dotenv

logger = logging.getLogger("cloudiq.aws.verify_s3")

REQUIRED_ENV_VARS = ("AWS_S3_BUCKET", "AWS_DEFAULT_REGION")


class S3ClientProtocol(Protocol):
    """Small protocol for the S3 method used by this verifier."""

    def list_objects_v2(self, *, Bucket: str) -> dict[str, Any]:
        """List objects in an S3 bucket."""


@dataclass(frozen=True)
class S3Config:
    """Validated S3 environment configuration."""

    bucket: str
    region: str


@dataclass(frozen=True)
class S3ObjectSummary:
    """Public object metadata safe to print."""

    key: str
    size: int


@dataclass(frozen=True)
class S3VerificationResult:
    """Result of a read-only S3 verification call."""

    bucket: str
    region: str
    object_count: int
    objects: list[S3ObjectSummary]


class S3VerificationError(RuntimeError):
    """User-safe S3 verification failure."""


def resolve_s3_config(env: Mapping[str, str | None]) -> S3Config:
    """Validate required AWS S3 environment variables."""
    missing = [
        name
        for name in REQUIRED_ENV_VARS
        if not str(env.get(name) or "").strip()
    ]
    if missing:
        raise S3VerificationError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Set them in your local .env file or shell session."
        )

    return S3Config(
        bucket=str(env["AWS_S3_BUCKET"]).strip(),
        region=str(env["AWS_DEFAULT_REGION"]).strip(),
    )


def load_s3_config() -> S3Config:
    """Load ``.env`` values and return validated S3 config."""
    load_dotenv()
    return resolve_s3_config(os.environ)


def _error_code(exc: BaseException) -> str:
    """Extract a safe AWS-style error code without exposing details."""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error", {})
        if isinstance(error, dict) and error.get("Code"):
            return str(error["Code"])
    return type(exc).__name__


def _expected_region(exc: BaseException) -> str | None:
    """Extract bucket-region hints when AWS provides them."""
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    metadata = response.get("ResponseMetadata", {})
    if not isinstance(metadata, dict):
        return None
    headers = metadata.get("HTTPHeaders", {})
    if not isinstance(headers, dict):
        return None
    for key in ("x-amz-bucket-region", "X-Amz-Bucket-Region"):
        value = headers.get(key)
        if value:
            return str(value)
    return None


def describe_s3_error(exc: BaseException, bucket: str, region: str) -> str:
    """Convert S3/botocore exceptions into user-safe error messages."""
    code = _error_code(exc)
    logger.debug(
        "S3 verification failed: category=%s bucket=%s region=%s",
        code,
        bucket,
        region,
    )

    if code in {"NoCredentialsError", "PartialCredentialsError"}:
        return (
            "AWS credentials were not found or are incomplete. Configure them "
            "with environment variables, an AWS profile, or another boto3 "
            "default credential provider."
        )
    if code in {"AccessDenied", "AllAccessDisabled", "InvalidAccessKeyId", "SignatureDoesNotMatch"}:
        return (
            f"Access denied while listing bucket '{bucket}'. Check that the "
            "active AWS identity has the least-privilege S3 policy documented "
            "for this project."
        )
    if code in {"NoSuchBucket", "404"}:
        return f"S3 bucket '{bucket}' was not found. Check AWS_S3_BUCKET."
    if code in {
        "PermanentRedirect",
        "AuthorizationHeaderMalformed",
        "IllegalLocationConstraintException",
        "301",
    }:
        expected = _expected_region(exc)
        suffix = f" Expected bucket region appears to be '{expected}'." if expected else ""
        return (
            f"Bucket '{bucket}' is not accessible from configured region "
            f"'{region}'. Check AWS_DEFAULT_REGION.{suffix}"
        )
    return (
        f"Unable to verify S3 access for bucket '{bucket}' in region '{region}'. "
        f"Error category: {code}."
    )


def verify_s3_access(
    config: S3Config,
    client: S3ClientProtocol | None = None,
) -> S3VerificationResult:
    """Verify read access to the configured S3 bucket."""
    if client is None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise S3VerificationError(
                "boto3 is not installed. Run `pip install -r requirements.txt`."
            ) from exc
        client = boto3.client("s3", region_name=config.region)

    try:
        response = client.list_objects_v2(Bucket=config.bucket)
    except Exception as exc:
        raise S3VerificationError(
            describe_s3_error(exc, config.bucket, config.region)
        ) from None

    contents = response.get("Contents", []) or []
    objects = [
        S3ObjectSummary(
            key=str(item.get("Key", "")),
            size=int(item.get("Size", 0) or 0),
        )
        for item in contents
    ]
    return S3VerificationResult(
        bucket=config.bucket,
        region=config.region,
        object_count=int(response.get("KeyCount", len(objects)) or 0),
        objects=objects,
    )


def print_result(result: S3VerificationResult) -> None:
    """Print a concise, secret-free verification summary."""
    print("S3 verification succeeded")
    print(f"Bucket: {result.bucket}")
    print(f"Configured region: {result.region}")
    print(f"Object count: {result.object_count}")
    if not result.objects:
        print("Objects: none returned by list_objects_v2")
        return
    print("Objects:")
    for item in result.objects:
        print(f"- {item.key} ({item.size} bytes)")


def main() -> int:
    """Run the read-only S3 verifier."""
    try:
        config = load_s3_config()
        result = verify_s3_access(config)
    except S3VerificationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
