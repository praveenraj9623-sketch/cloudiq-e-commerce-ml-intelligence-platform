"""Validate local Silver Delta outputs after running the Silver layer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyspark.sql import functions as F

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import DataFrame, SparkSession

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import ConfigLoader  # noqa: E402
from src.utils.spark_session import get_spark_session  # noqa: E402

SILVER_TABLES = [
    "orders",
    "order_items",
    "master_orders",
    "customer_profile",
    "product_demand",
    "seller_performance",
]


def _read_delta(spark: "SparkSession", path: Path) -> "DataFrame":
    """Read a Delta table from ``path``."""
    return spark.read.format("delta").load(str(path))


def _count(df: "DataFrame") -> int:
    """Return a DataFrame count as a Python int."""
    return int(df.count())


def validate_silver() -> dict[str, Any]:
    """Run Silver validation checks and return structured results."""
    config = ConfigLoader(str(ROOT / "config.yaml"), env_path=str(ROOT / ".env"))
    spark = get_spark_session(config, app_name="CloudIQ-Validate-Silver")
    try:
        silver_path = config.get_path("paths.silver", create=False)
        bronze_path = config.get_path("paths.bronze", create=False)
        tables = {
            table: _read_delta(spark, silver_path / table)
            for table in SILVER_TABLES
        }
        row_counts = {table: _count(df) for table, df in tables.items()}

        checks: dict[str, bool] = {}
        orders = tables["orders"]
        master = tables["master_orders"]
        customer_profile = tables["customer_profile"]
        product_demand = tables["product_demand"]
        seller_performance = tables["seller_performance"]

        checks["silver_orders_99441_rows"] = row_counts["orders"] == 99_441
        checks["silver_orders_unique_order_id"] = (
            orders.select("order_id").distinct().count() == 99_441
        )
        checks["master_orders_99441_rows"] = (
            row_counts["master_orders"] == 99_441
        )
        checks["master_orders_unique_order_id"] = (
            master.select("order_id").distinct().count() == 99_441
        )
        checks["customer_profile_no_duplicate_ids"] = (
            customer_profile.select("customer_unique_id").distinct().count()
            == row_counts["customer_profile"]
        )
        checks["product_demand_category_not_null"] = (
            product_demand.filter(F.col("category_name_english").isNull()).count()
            == 0
        )

        valid_orders = orders.filter(
            (F.col("order_status") == "delivered")
            & F.col("order_purchase_timestamp").isNotNull()
        ).select("order_id")
        expected_untranslated = (
            tables["order_items"]
            .join(valid_orders, "order_id", "inner")
            .join(
                _read_delta(spark, bronze_path / "products").select(
                    "product_id",
                    "product_category_name",
                ),
                "product_id",
                "left",
            )
            .join(
                _read_delta(spark, bronze_path / "category_translation"),
                "product_category_name",
                "left",
            )
            .filter(
                F.col("product_category_name").isNotNull()
                & F.col("product_category_name_english").isNull()
            )
            .select(
                F.concat(
                    F.lit("untranslated__"),
                    F.col("product_category_name"),
                ).alias("category_name_english")
            )
            .distinct()
        )
        actual_untranslated = (
            product_demand.filter(
                F.col("category_name_english").startswith("untranslated__")
            )
            .select("category_name_english")
            .distinct()
        )
        missing_untranslated = expected_untranslated.join(
            actual_untranslated,
            "category_name_english",
            "left_anti",
        )
        checks["unmatched_categories_preserved"] = (
            missing_untranslated.count() == 0
        )
        checks["seller_revenue_non_negative"] = (
            seller_performance.filter(
                F.col("total_revenue").isNull() | (F.col("total_revenue") < 0)
            ).count()
            == 0
        )

        return {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "row_counts": row_counts,
            "checks": checks,
        }
    finally:
        spark.stop()


def main() -> int:
    """Print validation results and return a process exit code."""
    result = validate_silver()
    print(f"Silver validation: {result['status']}")
    print("Silver row counts:")
    for table, count in result["row_counts"].items():
        print(f"- {table}: {count}")
    print("Checks:")
    for name, passed in result["checks"].items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
