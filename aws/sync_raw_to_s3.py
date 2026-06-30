"""Safely sync local Olist raw CSV files to the optional CloudIQ S3 bucket.

The utility is intentionally small and conservative:

- discovers the repository's existing local raw CSV directory;
- compares local CSV names against objects under the flat ``raw/`` S3 prefix;
- uploads only missing files unless ``--overwrite`` is supplied;
- never deletes remote objects;
- can publish only a manifest to prove PutObject access without touching
  source CSV objects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol

from dotenv import load_dotenv

EXPECTED_OLIST_FILES = {
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
}
DEFAULT_S3_PREFIX = "raw/"
MANIFEST_KEY = "raw/_manifests/olist_source_manifest.json"
MANIFEST_PATH = Path("reports") / "olist_source_manifest.json"
REQUIRED_ENV_VARS = ("AWS_S3_BUCKET", "AWS_DEFAULT_REGION")


class S3SyncError(RuntimeError):
    """User-safe S3 sync failure."""


class S3ClientProtocol(Protocol):
    """S3 operations used by the sync utility."""

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        """List objects in an S3 prefix."""

    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:
        """Upload a local file to S3."""

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str) -> Any:
        """Write an object body to S3."""


@dataclass(frozen=True)
class AwsSyncConfig:
    """Validated AWS settings for the sync utility."""

    bucket: str
    region: str
    prefix: str


@dataclass(frozen=True)
class LocalCsv:
    """Local CSV file metadata."""

    filename: str
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True)
class ManifestEntry:
    """Manifest record for one source file."""

    filename: str
    byte_size: int
    sha256: str


@dataclass(frozen=True)
class SyncSummary:
    """Summary counts for a sync run."""

    discovered: int
    skipped: int
    uploaded: int
    failed: int
    manifest_path: str
    manifest_published: bool


def repo_root() -> Path:
    """Return the repository root from this script location."""
    return Path(__file__).resolve().parents[1]


def normalize_prefix(prefix: str | None) -> str:
    """Normalize and validate the S3 raw prefix for this project."""
    value = (prefix or DEFAULT_S3_PREFIX).strip().lstrip("/")
    if value and not value.endswith("/"):
        value = f"{value}/"
    if value != DEFAULT_S3_PREFIX:
        raise S3SyncError(
            "AWS_S3_PREFIX must be 'raw/' for this project. "
            "The existing bucket uses flat keys such as raw/olist_customers_dataset.csv."
        )
    return value


def resolve_sync_config(env: Mapping[str, str | None]) -> AwsSyncConfig:
    """Validate required AWS environment variables."""
    missing = [
        name
        for name in REQUIRED_ENV_VARS
        if not str(env.get(name) or "").strip()
    ]
    if missing:
        raise S3SyncError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Set them in local .env or another boto3 credential provider."
        )
    return AwsSyncConfig(
        bucket=str(env["AWS_S3_BUCKET"]).strip(),
        region=str(env["AWS_DEFAULT_REGION"]).strip(),
        prefix=normalize_prefix(env.get("AWS_S3_PREFIX")),
    )


def load_sync_config() -> AwsSyncConfig:
    """Load ``.env`` and return validated AWS sync configuration."""
    load_dotenv()
    return resolve_sync_config(os.environ)


def find_raw_source_dir(root: Path) -> Path:
    """Find the local directory containing the expected Olist raw CSV set."""
    candidates = [
        path
        for path in [root / "data" / "raw", *root.glob("**/raw")]
        if path.is_dir() and ".venv" not in path.parts
    ]
    for candidate in candidates:
        names = {item.name for item in candidate.glob("*.csv")}
        if EXPECTED_OLIST_FILES.issubset(names):
            return candidate
    raise S3SyncError(
        "Could not find the local Olist raw CSV directory containing all 9 expected files."
    )


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_local_csvs(raw_dir: Path) -> list[LocalCsv]:
    """Discover expected local Olist CSV files with size and checksum metadata."""
    files: list[LocalCsv] = []
    missing = sorted(EXPECTED_OLIST_FILES - {item.name for item in raw_dir.glob("*.csv")})
    if missing:
        raise S3SyncError(f"Missing expected local raw CSV files: {', '.join(missing)}")

    for filename in sorted(EXPECTED_OLIST_FILES):
        path = raw_dir / filename
        files.append(
            LocalCsv(
                filename=filename,
                path=path,
                size=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )
    return files


def build_manifest(files: list[LocalCsv], generated_at: str | None = None) -> dict[str, Any]:
    """Build a reproducible manifest payload for local raw CSV files."""
    timestamp = generated_at or datetime.now(timezone.utc).isoformat()
    entries = [
        asdict(
            ManifestEntry(
                filename=item.filename,
                byte_size=item.size,
                sha256=item.sha256,
            )
        )
        for item in files
    ]
    return {
        "dataset": "olist",
        "generated_at_utc": timestamp,
        "file_count": len(entries),
        "files": entries,
    }


def write_manifest(root: Path, manifest: dict[str, Any]) -> Path:
    """Write the local manifest and return its absolute path."""
    manifest_path = root / MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path


def _error_code(exc: BaseException) -> str:
    """Extract a safe AWS-style error code."""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error", {})
        if isinstance(error, dict) and error.get("Code"):
            return str(error["Code"])
    return type(exc).__name__


def describe_s3_error(exc: BaseException, bucket: str, region: str) -> str:
    """Return a user-safe AWS error message without raw exception internals."""
    code = _error_code(exc)
    if code in {"NoCredentialsError", "PartialCredentialsError"}:
        return (
            "AWS credentials were not found or are incomplete. Use a local .env, "
            "AWS profile, or another boto3 default credential provider."
        )
    if code in {"AccessDenied", "AllAccessDisabled", "InvalidAccessKeyId", "SignatureDoesNotMatch"}:
        return f"Access denied for bucket '{bucket}'. Check the least-privilege IAM policy."
    if code in {"NoSuchBucket", "404"}:
        return f"S3 bucket '{bucket}' was not found."
    if code in {
        "PermanentRedirect",
        "AuthorizationHeaderMalformed",
        "IllegalLocationConstraintException",
        "301",
    }:
        return f"Bucket '{bucket}' is not accessible from configured region '{region}'."
    return f"S3 operation failed for bucket '{bucket}' in region '{region}'. Error category: {code}."


def create_s3_client(config: AwsSyncConfig) -> S3ClientProtocol:
    """Create a boto3 S3 client using default credential resolution."""
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise S3SyncError("boto3 is not installed. Run `pip install -r requirements.txt`.") from exc
    return boto3.client("s3", region_name=config.region)


def list_remote_objects(
    client: S3ClientProtocol,
    config: AwsSyncConfig,
) -> dict[str, int]:
    """List remote objects under the configured prefix as key -> size."""
    objects: dict[str, int] = {}
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {
            "Bucket": config.bucket,
            "Prefix": config.prefix,
        }
        if token:
            kwargs["ContinuationToken"] = token
        try:
            response = client.list_objects_v2(**kwargs)
        except Exception as exc:
            raise S3SyncError(describe_s3_error(exc, config.bucket, config.region)) from None

        for item in response.get("Contents", []) or []:
            key = str(item.get("Key", ""))
            objects[key] = int(item.get("Size", 0) or 0)
        if not response.get("IsTruncated"):
            return objects
        token = str(response.get("NextContinuationToken", ""))


def source_key(config: AwsSyncConfig, filename: str) -> str:
    """Return the flat S3 source object key for a local filename."""
    return f"{config.prefix}{filename}"


def print_plan(
    files: list[LocalCsv],
    config: AwsSyncConfig,
    remote: Mapping[str, int],
    overwrite: bool,
    publish_manifest: bool,
    manifest_size: int,
) -> None:
    """Print planned object keys and sizes without exposing credentials."""
    print(f"Bucket: {config.bucket}")
    print(f"Region: {config.region}")
    print(f"Prefix: {config.prefix}")
    print("Planned source CSV objects:")
    for item in files:
        key = source_key(config, item.filename)
        if key in remote and not overwrite:
            action = "skip existing"
        elif key in remote and overwrite:
            action = "overwrite"
        else:
            action = "upload missing"
        print(f"- s3://{config.bucket}/{key} ({item.size} bytes) [{action}]")
    if publish_manifest:
        print(
            f"Planned manifest object: s3://{config.bucket}/{MANIFEST_KEY} "
            f"({manifest_size} bytes) [upload manifest only]"
        )


def sync_raw_to_s3(
    config: AwsSyncConfig,
    client: S3ClientProtocol,
    root: Path,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
    publish_manifest: bool = False,
) -> SyncSummary:
    """Sync missing local Olist raw CSVs or publish only the manifest."""
    raw_dir = find_raw_source_dir(root)
    files = discover_local_csvs(raw_dir)
    manifest = build_manifest(files)
    manifest_path = write_manifest(root, manifest)
    manifest_bytes = manifest_path.read_bytes()
    remote = list_remote_objects(client, config)

    if dry_run:
        print_plan(
            files,
            config,
            remote,
            overwrite=overwrite,
            publish_manifest=publish_manifest,
            manifest_size=len(manifest_bytes),
        )
        skipped = sum(1 for item in files if source_key(config, item.filename) in remote)
        return SyncSummary(
            discovered=len(files),
            skipped=skipped,
            uploaded=0,
            failed=0,
            manifest_path=str(manifest_path),
            manifest_published=False,
        )

    uploaded = 0
    failed = 0
    skipped = 0

    if publish_manifest:
        try:
            client.put_object(
                Bucket=config.bucket,
                Key=MANIFEST_KEY,
                Body=manifest_bytes,
                ContentType="application/json",
            )
            uploaded = 1
        except Exception as exc:
            failed = 1
            raise S3SyncError(describe_s3_error(exc, config.bucket, config.region)) from None
        return SyncSummary(
            discovered=len(files),
            skipped=len(files),
            uploaded=uploaded,
            failed=failed,
            manifest_path=str(manifest_path),
            manifest_published=True,
        )

    for item in files:
        key = source_key(config, item.filename)
        if key in remote and not overwrite:
            skipped += 1
            continue
        try:
            client.upload_file(str(item.path), config.bucket, key)
            uploaded += 1
        except Exception:
            failed += 1

    return SyncSummary(
        discovered=len(files),
        skipped=skipped,
        uploaded=uploaded,
        failed=failed,
        manifest_path=str(manifest_path),
        manifest_published=False,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Safely sync local Olist raw CSVs to optional AWS S3 storage."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned object keys and sizes without uploading anything.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing source CSV objects. Defaults to skipping existing objects.",
    )
    parser.add_argument(
        "--publish-manifest",
        action="store_true",
        help=(
            "Upload only reports/olist_source_manifest.json to "
            "raw/_manifests/olist_source_manifest.json."
        ),
    )
    return parser.parse_args(argv)


def print_summary(summary: SyncSummary) -> None:
    """Print concise summary counts."""
    print("S3 raw sync summary")
    print(f"Discovered: {summary.discovered}")
    print(f"Skipped: {summary.skipped}")
    print(f"Uploaded: {summary.uploaded}")
    print(f"Failed: {summary.failed}")
    print(f"Manifest: {summary.manifest_path}")
    print(f"Manifest published: {'YES' if summary.manifest_published else 'NO'}")


def main(argv: list[str] | None = None) -> int:
    """Run the S3 raw sync utility."""
    args = parse_args(argv)
    try:
        config = load_sync_config()
        client = create_s3_client(config)
        summary = sync_raw_to_s3(
            config,
            client,
            repo_root(),
            dry_run=bool(args.dry_run),
            overwrite=bool(args.overwrite),
            publish_manifest=bool(args.publish_manifest),
        )
    except S3SyncError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print_summary(summary)
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
