"""Audit local Olist raw CSV files before Bronze ingestion.

The audit is intentionally pandas-only. It does not modify files in
``data/raw`` and does not start Spark or the CloudIQ pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"
MARKDOWN_REPORT = REPORTS_DIR / "olist_data_audit.md"
JSON_REPORT = REPORTS_DIR / "olist_data_audit.json"

EXPECTED_FILES: dict[str, str] = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

EXPECTED_SCHEMAS: dict[str, list[str]] = {
    "customers": [
        "customer_id",
        "customer_unique_id",
        "customer_zip_code_prefix",
        "customer_city",
        "customer_state",
    ],
    "geolocation": [
        "geolocation_zip_code_prefix",
        "geolocation_lat",
        "geolocation_lng",
        "geolocation_city",
        "geolocation_state",
    ],
    "order_items": [
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
        "shipping_limit_date",
        "price",
        "freight_value",
    ],
    "payments": [
        "order_id",
        "payment_sequential",
        "payment_type",
        "payment_installments",
        "payment_value",
    ],
    "reviews": [
        "review_id",
        "order_id",
        "review_score",
        "review_comment_title",
        "review_comment_message",
        "review_creation_date",
        "review_answer_timestamp",
    ],
    "orders": [
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "products": [
        "product_id",
        "product_category_name",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ],
    "sellers": [
        "seller_id",
        "seller_zip_code_prefix",
        "seller_city",
        "seller_state",
    ],
    "category_translation": [
        "product_category_name",
        "product_category_name_english",
    ],
}

ORDER_TIMESTAMP_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

DIFF_REVIEW = {
    "src/processing/bronze.py": (
        "Removed the unused TimestampType import. Bronze behavior is "
        "unchanged; the edit was a lint-only cleanup required by Ruff."
    ),
    "tests/test_demand_features.py": (
        "Added a tiny _MockExpr with alias() and patched F.max to return it. "
        "The production function calls F.max(...).alias('m'), so the "
        "pure-Python test double now matches that public behavior without "
        "starting Spark."
    ),
}


def _detect_encoding(path: Path) -> str:
    """Return utf-8 when possible, otherwise latin-1."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            while handle.read(1024 * 1024):
                pass
        return "utf-8"
    except UnicodeDecodeError:
        LOGGER.debug("Falling back to latin-1 for %s", path)
        return "latin-1"


def _read_csv(path: Path, encoding: str) -> pd.DataFrame:
    """Read a CSV with stable string dtypes for audit checks."""
    return pd.read_csv(path, encoding=encoding, dtype=str, low_memory=False)


def _missing_reference_count(
    child: pd.DataFrame,
    child_col: str,
    parent: pd.DataFrame,
    parent_col: str,
) -> int:
    """Count non-null child rows whose reference is absent from the parent."""
    parent_values = set(parent[parent_col].dropna().unique())
    child_values = child[child_col].dropna()
    return int((~child_values.isin(parent_values)).sum())


def _negative_count(df: pd.DataFrame, column: str) -> int:
    """Count negative numeric values in a string-backed DataFrame column."""
    values = pd.to_numeric(df[column], errors="coerce")
    return int((values < 0).sum())


def _invalid_timestamp_count(df: pd.DataFrame, column: str) -> int:
    """Count non-null values that pandas cannot parse as timestamps."""
    non_null = df[column].notna()
    parsed = pd.to_datetime(df[column], errors="coerce")
    return int((non_null & parsed.isna()).sum())


def _duplicate_primary_id_count(df: pd.DataFrame, column: str) -> int:
    """Count repeated non-null primary IDs beyond the first occurrence."""
    non_null = df[df[column].notna()]
    return int(non_null.duplicated(subset=[column]).sum())


def _file_summary(name: str, path: Path, df: pd.DataFrame, encoding: str) -> dict:
    """Build the per-file audit summary."""
    return {
        "name": name,
        "filename": path.name,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
        "encoding": encoding,
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "null_counts": {
            column: int(value) for column, value in df.isna().sum().items()
        },
        "exact_duplicate_rows": int(df.duplicated().sum()),
    }


def _schema_results(dataframes: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Validate exact column order for every loaded expected file."""
    results: dict[str, dict] = {}
    for name, expected in EXPECTED_SCHEMAS.items():
        if name not in dataframes:
            results[name] = {
                "valid": False,
                "expected": expected,
                "actual": [],
                "error": "missing file",
            }
            continue

        actual = list(dataframes[name].columns)
        results[name] = {
            "valid": actual == expected,
            "expected": expected,
            "actual": actual,
            "error": None if actual == expected else "schema mismatch",
        }
    return results


def _relationship_checks(dataframes: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Run core operational cross-file reference checks."""
    checks = {
        "orders.customer_id not found in customers.customer_id": (
            "orders",
            "customer_id",
            "customers",
            "customer_id",
        ),
        "order_items.order_id not found in orders.order_id": (
            "order_items",
            "order_id",
            "orders",
            "order_id",
        ),
        "order_items.product_id not found in products.product_id": (
            "order_items",
            "product_id",
            "products",
            "product_id",
        ),
        "order_items.seller_id not found in sellers.seller_id": (
            "order_items",
            "seller_id",
            "sellers",
            "seller_id",
        ),
        "payments.order_id not found in orders.order_id": (
            "payments",
            "order_id",
            "orders",
            "order_id",
        ),
        "reviews.order_id not found in orders.order_id": (
            "reviews",
            "order_id",
            "orders",
            "order_id",
        ),
    }

    results: dict[str, int] = {}
    for label, (child_name, child_col, parent_name, parent_col) in checks.items():
        if child_name not in dataframes or parent_name not in dataframes:
            results[label] = -1
            continue
        results[label] = _missing_reference_count(
            dataframes[child_name],
            child_col,
            dataframes[parent_name],
            parent_col,
        )
    return results


def _translation_lookup_gap(
    dataframes: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Report optional category-translation lookup gaps."""
    if "products" not in dataframes or "category_translation" not in dataframes:
        return {
            "unmatched_product_row_count": None,
            "unmatched_distinct_category_count": None,
            "unmatched_categories": [],
        }

    products = dataframes["products"]
    translation = dataframes["category_translation"]
    translated_categories = set(
        translation["product_category_name"].dropna().unique()
    )
    product_categories = products["product_category_name"].dropna()
    unmatched = product_categories[
        ~product_categories.isin(translated_categories)
    ]

    return {
        "unmatched_product_row_count": int(len(unmatched)),
        "unmatched_distinct_category_count": int(unmatched.nunique()),
        "unmatched_categories": sorted(str(value) for value in unmatched.unique()),
    }


def _sanity_checks(dataframes: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Run local data sanity checks requested for the raw files."""
    required = [
        "customers",
        "geolocation",
        "order_items",
        "payments",
        "reviews",
        "orders",
        "products",
        "sellers",
    ]
    missing = [name for name in required if name not in dataframes]
    if missing:
        return {
            "skipped_due_to_missing_files": missing,
            "negative_counts": {
                "order_items.price": None,
                "order_items.freight_value": None,
                "payments.payment_value": None,
            },
            "review_score_outside_1_5_count": None,
            "invalid_order_timestamp_parse_counts": {
                column: None for column in ORDER_TIMESTAMP_COLUMNS
            },
            "delivered_customer_before_purchase_count": None,
            "earliest_order_purchase_timestamp": None,
            "latest_order_purchase_timestamp": None,
            "distinct_order_status_values": [],
            "duplicate_primary_ids": {
                "orders.order_id": None,
                "customers.customer_id": None,
                "products.product_id": None,
                "sellers.seller_id": None,
            },
            "geolocation_duplicate_zip_code_rows": None,
        }

    order_items = dataframes["order_items"]
    payments = dataframes["payments"]
    reviews = dataframes["reviews"]
    orders = dataframes["orders"]
    geolocation = dataframes["geolocation"]

    score = pd.to_numeric(reviews["review_score"], errors="coerce")
    invalid_scores = reviews["review_score"].notna() & (
        score.isna() | ~score.between(1, 5)
    )

    timestamp_parse = {
        column: _invalid_timestamp_count(orders, column)
        for column in ORDER_TIMESTAMP_COLUMNS
    }
    purchase_dates = pd.to_datetime(
        orders["order_purchase_timestamp"],
        errors="coerce",
    )
    delivered_dates = pd.to_datetime(
        orders["order_delivered_customer_date"],
        errors="coerce",
    )
    delivered_before_purchase = (
        delivered_dates.notna()
        & purchase_dates.notna()
        & (delivered_dates < purchase_dates)
    )
    valid_purchase_dates = purchase_dates.dropna()

    return {
        "negative_counts": {
            "order_items.price": _negative_count(order_items, "price"),
            "order_items.freight_value": _negative_count(
                order_items,
                "freight_value",
            ),
            "payments.payment_value": _negative_count(
                payments,
                "payment_value",
            ),
        },
        "review_score_outside_1_5_count": int(invalid_scores.sum()),
        "invalid_order_timestamp_parse_counts": timestamp_parse,
        "delivered_customer_before_purchase_count": int(
            delivered_before_purchase.sum()
        ),
        "earliest_order_purchase_timestamp": (
            valid_purchase_dates.min().isoformat()
            if not valid_purchase_dates.empty
            else None
        ),
        "latest_order_purchase_timestamp": (
            valid_purchase_dates.max().isoformat()
            if not valid_purchase_dates.empty
            else None
        ),
        "distinct_order_status_values": sorted(
            str(value) for value in orders["order_status"].dropna().unique()
        ),
        "duplicate_primary_ids": {
            "orders.order_id": _duplicate_primary_id_count(
                orders,
                "order_id",
            ),
            "customers.customer_id": _duplicate_primary_id_count(
                dataframes["customers"],
                "customer_id",
            ),
            "products.product_id": _duplicate_primary_id_count(
                dataframes["products"],
                "product_id",
            ),
            "sellers.seller_id": _duplicate_primary_id_count(
                dataframes["sellers"],
                "seller_id",
            ),
        },
        "geolocation_duplicate_zip_code_rows": int(
            geolocation.duplicated(
                subset=["geolocation_zip_code_prefix"],
            ).sum()
        ),
    }


def _build_warnings(
    file_summaries: dict[str, dict],
    unexpected_csv_files: list[str],
    sanity: dict[str, Any],
    translation_lookup_gap: dict[str, Any],
) -> list[str]:
    """Build non-fatal warnings and informational notes."""
    warnings: list[str] = []
    if unexpected_csv_files:
        warnings.append(
            "Unexpected CSV files are present in data/raw: "
            + ", ".join(unexpected_csv_files)
        )

    for summary in file_summaries.values():
        if summary["exact_duplicate_rows"] > 0:
            warnings.append(
                f"{summary['filename']} has "
                f"{summary['exact_duplicate_rows']} exact duplicate rows."
            )

    geo_zip_duplicates = sanity["geolocation_duplicate_zip_code_rows"]
    if isinstance(geo_zip_duplicates, int) and geo_zip_duplicates > 0:
        warnings.append(
            "Informational: geolocation has "
            f"{geo_zip_duplicates} duplicate zip-code rows; this is not a "
            "Bronze ingestion failure."
        )
    unmatched_rows = translation_lookup_gap["unmatched_product_row_count"]
    unmatched_distinct = translation_lookup_gap[
        "unmatched_distinct_category_count"
    ]
    if isinstance(unmatched_rows, int) and unmatched_rows > 0:
        warnings.append(
            "Optional lookup warning: "
            f"{unmatched_rows} product rows across {unmatched_distinct} "
            "distinct product_category_name values have no English "
            "translation."
        )
    return warnings


def _fatal_failures(
    missing_files: list[str],
    schema_results: dict[str, dict],
    relationship_results: dict[str, int],
    sanity: dict[str, Any],
) -> list[str]:
    """Return failures that make Bronze ingestion unsafe."""
    failures: list[str] = []
    if missing_files:
        failures.append("missing expected files")
    if any(not result["valid"] for result in schema_results.values()):
        failures.append("schema mismatch")
    if any(count != 0 for count in relationship_results.values()):
        failures.append("relationship mismatch")
    if any(
        isinstance(count, int) and count != 0
        for count in sanity["negative_counts"].values()
    ):
        failures.append("negative monetary values")
    score_count = sanity["review_score_outside_1_5_count"]
    if isinstance(score_count, int) and score_count != 0:
        failures.append("review scores outside 1-5")
    if any(
        isinstance(count, int) and count != 0
        for count in sanity["invalid_order_timestamp_parse_counts"].values()
    ):
        failures.append("invalid order timestamps")
    delivered_count = sanity["delivered_customer_before_purchase_count"]
    if isinstance(delivered_count, int) and delivered_count != 0:
        failures.append("delivered date before purchase date")
    if any(
        isinstance(count, int) and count != 0
        for count in sanity["duplicate_primary_ids"].values()
    ):
        failures.append("duplicate primary IDs")
    return failures


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Return a simple GitHub-flavored Markdown table."""
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _render_markdown(report: dict[str, Any]) -> str:
    """Render the audit report as Markdown."""
    file_rows = [
        [
            summary["filename"],
            summary["size_mb"],
            summary["encoding"],
            summary["row_count"],
            "YES" if report["schema_results"][name]["valid"] else "NO",
            summary["exact_duplicate_rows"],
        ]
        for name, summary in report["files"].items()
    ]
    schema_rows = [
        [
            name,
            "PASS" if result["valid"] else "FAIL",
            ", ".join(result["actual"]),
        ]
        for name, result in report["schema_results"].items()
    ]
    relationship_rows = [
        [name, count]
        for name, count in report["relationship_results"].items()
    ]

    lines = [
        "# Olist Data Audit",
        "",
        f"Executive verdict: {report['executive_verdict']}",
        "",
        (
            "Safe to run python run_pipeline.py --layer bronze: "
            f"{report['safe_to_run_bronze']}"
        ),
        "",
        "## File-level summary",
        "",
        _markdown_table(
            [
                "File",
                "Size MB",
                "Encoding",
                "Rows",
                "Schema OK",
                "Exact duplicate rows",
            ],
            file_rows,
        ),
        "",
        "Unexpected CSV files: "
        + (
            ", ".join(report["unexpected_csv_files"])
            if report["unexpected_csv_files"]
            else "none"
        ),
        "",
        "Missing expected files: "
        + (
            ", ".join(report["missing_files"])
            if report["missing_files"]
            else "none"
        ),
        "",
        "## Column and null counts",
        "",
    ]

    for summary in report["files"].values():
        null_rows = [
            [column, count]
            for column, count in summary["null_counts"].items()
        ]
        lines.extend(
            [
                f"### {summary['filename']}",
                "",
                "Columns: " + ", ".join(summary["columns"]),
                "",
                _markdown_table(["Column", "Null count"], null_rows),
                "",
            ]
        )

    lines.extend(
        [
            "## Schema validation result",
            "",
            _markdown_table(["Dataset", "Result", "Actual columns"], schema_rows),
            "",
            "## Relationship validation result",
            "",
            _markdown_table(
                ["Check", "Missing-reference count"],
                relationship_rows,
            ),
            "",
            "## Optional lookup-enrichment warnings",
            "",
            (
                "Unmatched product-category translation row count: "
                f"{report['translation_lookup_gap']['unmatched_product_row_count']}"
            ),
            "",
            (
                "Unmatched distinct product-category count: "
                f"{report['translation_lookup_gap']['unmatched_distinct_category_count']}"
            ),
            "",
            "Unmatched categories: "
            + (
                ", ".join(
                    report["translation_lookup_gap"]["unmatched_categories"]
                )
                if report["translation_lookup_gap"]["unmatched_categories"]
                else "none"
            ),
            "",
            "## Data sanity checks",
            "",
            "```json",
            json.dumps(report["sanity_checks"], indent=2),
            "```",
            "",
            "## Data-quality warnings",
            "",
        ]
    )
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Review of existing Bronze and demand-test diffs",
            "",
        ]
    )
    lines.extend(
        f"- `{path}`: {explanation}"
        for path, explanation in report["diff_review"].items()
    )
    lines.extend(
        [
            "",
            (
                "Safe to run python run_pipeline.py --layer bronze: "
                f"{report['safe_to_run_bronze']}"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_audit() -> dict[str, Any]:
    """Run the full Olist raw data audit and write both reports."""
    expected_filenames = set(EXPECTED_FILES.values())
    actual_csv_files = sorted(path.name for path in RAW_DIR.glob("*.csv"))
    missing_files = sorted(expected_filenames - set(actual_csv_files))
    unexpected_csv_files = sorted(set(actual_csv_files) - expected_filenames)

    dataframes: dict[str, pd.DataFrame] = {}
    file_summaries: dict[str, dict] = {}
    for name, filename in EXPECTED_FILES.items():
        path = RAW_DIR / filename
        if not path.exists():
            continue
        encoding = _detect_encoding(path)
        df = _read_csv(path, encoding)
        dataframes[name] = df
        file_summaries[name] = _file_summary(name, path, df, encoding)

    schema_results = _schema_results(dataframes)
    relationship_results = _relationship_checks(dataframes)
    translation_lookup_gap = _translation_lookup_gap(dataframes)
    sanity = _sanity_checks(dataframes)
    warnings = _build_warnings(
        file_summaries,
        unexpected_csv_files,
        sanity,
        translation_lookup_gap,
    )
    failures = _fatal_failures(
        missing_files,
        schema_results,
        relationship_results,
        sanity,
    )

    verdict = "NOT READY"
    if not failures and warnings:
        verdict = "READY WITH WARNINGS"
    elif not failures:
        verdict = "READY"

    report = {
        "executive_verdict": verdict,
        "safe_to_run_bronze": "YES" if not failures else "NO",
        "missing_files": missing_files,
        "unexpected_csv_files": unexpected_csv_files,
        "files": file_summaries,
        "schema_results": schema_results,
        "relationship_results": relationship_results,
        "translation_lookup_gap": translation_lookup_gap,
        "sanity_checks": sanity,
        "warnings": warnings,
        "fatal_failures": failures,
        "diff_review": DIFF_REVIEW,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    MARKDOWN_REPORT.write_text(_render_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    """Run the audit and print a concise terminal summary."""
    report = run_audit()
    failed_schema = [
        name
        for name, result in report["schema_results"].items()
        if not result["valid"]
    ]

    print(f"Executive verdict: {report['executive_verdict']}")
    print("Row counts:")
    for name, summary in report["files"].items():
        print(f"- {summary['filename']}: {summary['row_count']}")
    print(
        "Failed schema checks: "
        + (", ".join(failed_schema) if failed_schema else "none")
    )
    print("Relationship mismatch counts:")
    for name, count in report["relationship_results"].items():
        print(f"- {name}: {count}")
    print(
        "Unmatched product-category translation rows: "
        f"{report['translation_lookup_gap']['unmatched_product_row_count']}"
    )
    print(
        "Unmatched product-category translation distinct categories: "
        f"{report['translation_lookup_gap']['unmatched_distinct_category_count']}"
    )
    print(f"Bronze ingestion safe: {report['safe_to_run_bronze']}")
    print("Existing file change explanations:")
    for path, explanation in report["diff_review"].items():
        print(f"- {path}: {explanation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
