"""Tests for local AWS S3 manifest dashboard status helpers."""

from __future__ import annotations

import json
from pathlib import Path

from src.ui.components import (
    PRODUCT_TRANSLATION_FALLBACK_NOTE,
    methodology_notes_without_duplicate_fallback,
)
from src.ui.aws_s3_status import (
    AWS_S3_REMEDIATION_COMMAND,
    load_s3_manifest_status,
    parse_s3_manifest,
)


def test_valid_manifest_parsing_infers_flat_s3_keys() -> None:
    """Valid manifests become dashboard-safe file rows without full checksums."""
    status = parse_s3_manifest(
        {
            "generated_at_utc": "2026-06-30T12:00:00+00:00",
            "file_count": 2,
            "files": [
                {
                    "filename": "olist_customers_dataset.csv",
                    "byte_size": 123,
                    "sha256": "abc123",
                },
                {
                    "filename": "olist_orders_dataset.csv",
                    "byte_size": 456,
                    "sha256": "",
                },
            ],
        }
    )

    assert status.storage_layer == "Amazon S3"
    assert status.source_files_verified == "2"
    assert status.manifest_status == "Published"
    assert status.validation_mode == "Local Boto3 verification"
    assert status.generated_at_utc == "2026-06-30T12:00:00+00:00"
    assert status.generated_at_display == "2026-06-30 12:00 UTC"
    assert status.files[0].s3_key == "raw/olist_customers_dataset.csv"
    assert status.files[0].checksum_present is True
    assert status.files[1].checksum_present is False


def test_missing_manifest_behavior_is_neutral(tmp_path: Path) -> None:
    """Missing manifest does not crash and gives the remediation command."""
    status = load_s3_manifest_status(tmp_path / "missing.json")

    assert status.manifest_status == "Not verified locally yet"
    assert status.source_files_verified == "Not verified locally yet"
    assert status.files == []
    assert status.remediation_command == AWS_S3_REMEDIATION_COMMAND


def test_malformed_json_behavior_is_neutral(tmp_path: Path) -> None:
    """Malformed JSON is reported without raising to Streamlit."""
    path = tmp_path / "olist_source_manifest.json"
    path.write_text("{not-valid-json", encoding="utf-8")

    status = load_s3_manifest_status(path)

    assert status.manifest_status == "Invalid manifest"
    assert status.source_files_verified == "Not verified locally yet"
    assert "could not be parsed" in status.error_message


def test_streamlit_ui_modules_have_no_boto3_or_dotenv_dependency() -> None:
    """Streamlit UI code stays local-only and does not import cloud clients."""
    root = Path(__file__).resolve().parents[1]
    ui_files = [root / "streamlit_app.py", *sorted((root / "src" / "ui").glob("*.py"))]
    source = "\n".join(path.read_text(encoding="utf-8") for path in ui_files)

    assert "boto3" not in source
    assert "dotenv" not in source
    assert "client(" not in source


def test_duplicate_methodology_note_is_not_rendered_twice() -> None:
    """Generated notes exclude the static fallback card copy."""
    generated_notes = [
        "Reviews Bronze ingestion uses multiline-safe CSV parsing.",
        "13 product rows across 2 categories are retained using untranslated__ fallback labels.",
        "Demand features predict target_units for forecast_month.",
    ]

    rendered_notes = [
        *methodology_notes_without_duplicate_fallback(generated_notes),
        PRODUCT_TRANSLATION_FALLBACK_NOTE,
    ]

    assert rendered_notes.count(PRODUCT_TRANSLATION_FALLBACK_NOTE) == 1


def test_streamlit_css_hardens_dark_select_and_hero_contrast() -> None:
    """Dashboard CSS includes explicit dark-theme text and select controls."""
    root = Path(__file__).resolve().parents[1]
    styles = (root / "src" / "ui" / "styles.py").read_text(encoding="utf-8")
    config = (root / ".streamlit" / "config.toml").read_text(encoding="utf-8")

    assert "--cloudiq-text-primary: #f4f8fb" in styles
    assert "--cloudiq-control-bg-selected" in styles
    assert "--cloudiq-dropdown-bg" in styles
    assert ".cloudiq-hero h1" in styles
    assert "color: var(--cloudiq-text-primary) !important" in styles
    assert 'div[data-testid="stSelectbox"] label' in styles
    assert '[data-baseweb="select"]' in styles
    assert '[role="listbox"]' in styles
    assert '[role="option"][aria-selected="true"]' in styles
    assert ".aws-generated-at" in styles
    assert ".aws-evidence-caption" in styles
    assert 'base = "dark"' in config
    assert 'textColor = "#f4f8fb"' in config


def test_invalid_timestamp_renders_safely() -> None:
    """Unexpected timestamps are compacted rather than raising."""
    status = parse_s3_manifest(
        {
            "generated_at_utc": "not-a-real-timestamp-with-extra-detail",
            "file_count": 1,
            "files": [
                {
                    "filename": "olist_products_dataset.csv",
                    "byte_size": 789,
                    "sha256": "abc",
                }
            ],
        }
    )

    assert status.generated_at_display == "not-a-real-timestamp-with-extra"


def test_to_dict_hides_full_checksum_values() -> None:
    """Display dictionaries contain checksum status rather than checksum values."""
    status = parse_s3_manifest(
        {
            "files": [
                {
                    "filename": "olist_products_dataset.csv",
                    "byte_size": 789,
                    "sha256": "full-checksum-should-not-appear",
                }
            ]
        }
    ).to_dict()

    payload = json.dumps(status)
    assert "checksum_present" in payload
    assert "full-checksum-should-not-appear" not in payload
