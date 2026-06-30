"""Local AWS S3 manifest status helpers for the Streamlit dashboard."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Any

DEFAULT_S3_PREFIX = "raw/"
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "reports" / "olist_source_manifest.json"
AWS_S3_REMEDIATION_COMMAND = "python aws/sync_raw_to_s3.py --publish-manifest"
AWS_S3_EXPLANATORY_CAPTION = (
    "Raw Olist source files are stored in a private Amazon S3 landing zone. "
    "This dashboard reads local manifest evidence only; it does not query S3 "
    "during page rendering. Databricks Free Edition processing is validated "
    "separately through Unity Catalog managed tables and is not live-connected "
    "to this S3 bucket."
)


@dataclass(frozen=True)
class S3ManifestFile:
    """Safe display fields for one manifest source file."""

    file_name: str
    s3_key: str
    byte_size: int
    checksum_present: bool


@dataclass(frozen=True)
class S3ManifestStatus:
    """Dashboard-ready local S3 manifest status."""

    storage_layer: str
    source_files_verified: str
    manifest_status: str
    validation_mode: str
    generated_at_utc: str
    generated_at_display: str
    files: list[S3ManifestFile]
    remediation_command: str
    error_message: str

    def to_dict(self) -> dict[str, Any]:
        """Return a Streamlit-cache-friendly dictionary."""
        payload = asdict(self)
        payload["files"] = [asdict(item) for item in self.files]
        return payload


def _empty_status(manifest_status: str, error_message: str = "") -> S3ManifestStatus:
    """Return a neutral status for missing or invalid local evidence."""
    return S3ManifestStatus(
        storage_layer="Amazon S3",
        source_files_verified="Not verified locally yet",
        manifest_status=manifest_status,
        validation_mode="Local Boto3 verification",
        generated_at_utc="",
        generated_at_display="",
        files=[],
        remediation_command=AWS_S3_REMEDIATION_COMMAND,
        error_message=error_message,
    )


def _safe_int(value: Any) -> int:
    """Convert manifest byte sizes defensively."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def compact_utc_timestamp(value: Any) -> str:
    """Render manifest timestamps as compact UTC strings for the dashboard."""
    raw_value = str(value or "").strip()
    if not raw_value:
        return ""
    normalized = raw_value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return raw_value[:32].rstrip(" -:TZ.")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


def parse_s3_manifest(payload: dict[str, Any]) -> S3ManifestStatus:
    """Parse a local Olist S3 manifest into display-safe status fields."""
    files_payload = payload.get("files", [])
    if not isinstance(files_payload, list):
        return _empty_status("Invalid manifest", "Manifest files field is not a list.")

    rows: list[S3ManifestFile] = []
    for item in files_payload:
        if not isinstance(item, dict):
            continue
        file_name = str(item.get("filename") or item.get("file_name") or "").strip()
        if not file_name:
            continue
        s3_key = str(item.get("s3_key") or item.get("key") or f"{DEFAULT_S3_PREFIX}{file_name}")
        rows.append(
            S3ManifestFile(
                file_name=file_name,
                s3_key=s3_key,
                byte_size=_safe_int(item.get("byte_size") or item.get("size")),
                checksum_present=bool(str(item.get("sha256") or "").strip()),
            )
        )

    if not rows:
        return _empty_status("Invalid manifest", "Manifest contains no source file rows.")

    source_file_count = _safe_int(payload.get("file_count")) or len(rows)
    generated_at_utc = str(payload.get("generated_at_utc") or payload.get("generated_at") or "")
    return S3ManifestStatus(
        storage_layer="Amazon S3",
        source_files_verified=str(source_file_count),
        manifest_status="Published",
        validation_mode="Local Boto3 verification",
        generated_at_utc=generated_at_utc,
        generated_at_display=compact_utc_timestamp(generated_at_utc),
        files=rows,
        remediation_command=AWS_S3_REMEDIATION_COMMAND,
        error_message="",
    )


def load_s3_manifest_status(path: Path = DEFAULT_MANIFEST_PATH) -> S3ManifestStatus:
    """Load local S3 manifest evidence without touching AWS or environment vars."""
    if not path.exists():
        return _empty_status("Not verified locally yet")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError:
        return _empty_status("Invalid manifest", "Manifest JSON could not be parsed.")

    if not isinstance(payload, dict):
        return _empty_status("Invalid manifest", "Manifest root must be a JSON object.")
    return parse_s3_manifest(payload)
