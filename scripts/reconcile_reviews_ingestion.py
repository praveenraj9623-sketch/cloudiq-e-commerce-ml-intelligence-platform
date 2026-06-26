"""Diagnose Olist reviews CSV parsing counts with pandas and Spark."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import ConfigLoader  # noqa: E402
from src.utils.spark_session import get_spark_session  # noqa: E402

REVIEWS_PATH = ROOT / "data" / "raw" / "olist_order_reviews_dataset.csv"


def _spark_count_current_options(
    spark: "SparkSession",
    path: Path,
) -> int:
    """Count rows using the legacy Bronze reviews CSV reader options."""
    return int(
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv(str(path))
        .count()
    )


def _spark_count_multiline_safe(
    spark: "SparkSession",
    path: Path,
) -> int:
    """Count rows using strict multiline-safe Spark CSV reader options."""
    return int(
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .option("multiLine", "true")
        .option("quote", '"')
        .option("escape", '"')
        .option("mode", "FAILFAST")
        .csv(str(path))
        .count()
    )


def reconcile_reviews() -> dict[str, int | bool]:
    """Return pandas, legacy Spark, and multiline-safe Spark row counts."""
    pandas_count = int(pd.read_csv(REVIEWS_PATH, dtype=str).shape[0])
    spark = None
    try:
        config = ConfigLoader(str(ROOT / "config.yaml"), env_path=str(ROOT / ".env"))
        spark = get_spark_session(
            config,
            app_name="CloudIQ-Reconcile-Reviews-Ingestion",
        )
        current_count = _spark_count_current_options(spark, REVIEWS_PATH)
        multiline_safe_count = _spark_count_multiline_safe(spark, REVIEWS_PATH)
    finally:
        if spark is not None:
            spark.stop()

    return {
        "pandas_raw_count": pandas_count,
        "spark_current_options_count": current_count,
        "spark_multiline_safe_count": multiline_safe_count,
        "multiline_parsing_fixes_discrepancy": (
            multiline_safe_count == pandas_count
            and current_count != pandas_count
        ),
    }


def main() -> int:
    """Print the reconciliation counts and conclusion."""
    results = reconcile_reviews()
    print(f"Pandas raw reviews count: {results['pandas_raw_count']}")
    print(
        "Spark current-options reviews count: "
        f"{results['spark_current_options_count']}"
    )
    print(
        "Spark multiline-safe reviews count: "
        f"{results['spark_multiline_safe_count']}"
    )
    if results["multiline_parsing_fixes_discrepancy"]:
        print(
            "Conclusion: multiline-safe parsing fixes the discrepancy; "
            "quoted review comments containing newlines were being split into "
            "extra rows by the legacy Spark reader."
        )
    else:
        print("Conclusion: multiline-safe parsing did not explain the discrepancy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
