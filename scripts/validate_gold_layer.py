"""Validate local Gold Delta outputs after running the Gold layer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyspark.sql import Window
from pyspark.sql import functions as F

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import DataFrame, SparkSession

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import ConfigLoader  # noqa: E402
from src.utils.spark_session import get_spark_session  # noqa: E402

GOLD_TABLES = [
    "demand_features",
    "churn_features",
    "rfm_segments",
    "bi_revenue",
]


def _read_delta(spark: "SparkSession", path: Path) -> "DataFrame":
    """Read a Delta table from ``path``."""
    return spark.read.format("delta").load(str(path))


def _count(df: "DataFrame") -> int:
    """Return a DataFrame count as a Python int."""
    return int(df.count())


def _expected_demand_features(product_demand: "DataFrame") -> "DataFrame":
    """Recompute leakage-safe demand features from Silver demand."""
    win = Window.partitionBy("category_name_english").orderBy("order_year_month")
    roll_win = win.rowsBetween(-2, 0)
    return (
        product_demand.select(
            "category_name_english",
            "order_year_month",
            "monthly_units",
        )
        .withColumn("feature_cutoff_month", F.col("order_year_month"))
        .withColumn(
            "_feature_month_date",
            F.to_date(F.concat(F.col("order_year_month"), F.lit("-01"))),
        )
        .withColumn(
            "forecast_month",
            F.date_format(F.add_months("_feature_month_date", 1), "yyyy-MM"),
        )
        .withColumn(
            "expected_target_units",
            F.lead("monthly_units", 1).over(win),
        )
        .withColumn("expected_lag_4", F.lag("monthly_units", 3).over(win))
        .withColumn(
            "expected_rolling_mean_3",
            F.avg("monthly_units").over(roll_win),
        )
        .withColumn("expected_lag_1", F.col("monthly_units"))
        .withColumn("expected_lag_2", F.lag("monthly_units", 1).over(win))
        .filter(F.col("expected_lag_4").isNotNull())
        .filter(F.col("expected_target_units").isNotNull())
        .drop("_feature_month_date", "order_year_month")
    )


def _validate_demand_features(
    actual: "DataFrame",
    product_demand: "DataFrame",
) -> dict[str, bool]:
    """Validate demand target and prior-only rolling features."""
    expected = _expected_demand_features(product_demand)
    compared = actual.join(
        expected,
        ["category_name_english", "forecast_month"],
        "full",
    )
    mismatches = compared.filter(
        F.col("target_units").isNull()
        | F.col("lag_1").isNull()
        | F.col("lag_2").isNull()
        | F.col("lag_4").isNull()
        | F.col("rolling_mean_3").isNull()
        | F.col("expected_target_units").isNull()
        | F.col("expected_lag_1").isNull()
        | F.col("expected_lag_2").isNull()
        | F.col("expected_lag_4").isNull()
        | F.col("expected_rolling_mean_3").isNull()
        | (F.abs(F.col("target_units") - F.col("expected_target_units")) > 1e-9)
        | (F.abs(F.col("lag_1") - F.col("expected_lag_1")) > 1e-9)
        | (F.abs(F.col("lag_2") - F.col("expected_lag_2")) > 1e-9)
        | (F.abs(F.col("lag_4") - F.col("expected_lag_4")) > 1e-9)
        | (F.abs(F.col("rolling_mean_3") - F.col("expected_rolling_mean_3")) > 1e-9)
    )
    return {
        "demand_required_fields_not_null": actual.filter(
            F.col("target_units").isNull()
            | F.col("lag_1").isNull()
            | F.col("lag_2").isNull()
            | F.col("lag_4").isNull()
            | F.col("rolling_mean_3").isNull()
        ).count()
        == 0,
        "demand_prior_only_features_match_recompute": mismatches.count() == 0,
        "demand_row_count_matches_recompute": actual.count() == expected.count(),
    }


def validate_gold() -> dict[str, Any]:
    """Run Gold validation checks and return structured results."""
    config = ConfigLoader(str(ROOT / "config.yaml"), env_path=str(ROOT / ".env"))
    spark = get_spark_session(config, app_name="CloudIQ-Validate-Gold")
    try:
        gold_path = config.get_path("paths.gold", create=False)
        silver_path = config.get_path("paths.silver", create=False)
        tables = {
            table: _read_delta(spark, gold_path / table)
            for table in GOLD_TABLES
        }
        row_counts = {table: _count(df) for table, df in tables.items()}

        checks: dict[str, bool] = {}
        checks.update(
            _validate_demand_features(
                tables["demand_features"],
                _read_delta(spark, silver_path / "product_demand"),
            )
        )

        churn = tables["churn_features"]
        churn_columns = set(churn.columns)
        date_leaks = churn.filter(
            (F.to_date("last_purchase_timestamp") > F.col("snapshot_date"))
            | (F.to_date("last_review_creation_date") > F.col("snapshot_date"))
            | (F.to_date("last_delivery_date") > F.col("snapshot_date"))
        )
        checks["churn_has_snapshot_date"] = "snapshot_date" in churn_columns
        checks["churn_has_both_target_classes"] = (
            churn.select("is_churned").distinct().count() == 2
            if row_counts["churn_features"] > 0
            else False
        )
        checks["churn_feature_dates_not_after_snapshot"] = date_leaks.count() == 0

        rfm = tables["rfm_segments"]
        checks["rfm_scores_between_1_and_5"] = (
            rfm.filter(
                ~F.col("recency_score").between(1, 5)
                | ~F.col("frequency_score").between(1, 5)
                | ~F.col("monetary_score").between(1, 5)
            ).count()
            == 0
        )
        checks["rfm_total_between_3_and_15"] = (
            rfm.filter(~F.col("rfm_total").between(3, 15)).count() == 0
        )
        checks["rfm_segment_label_not_null"] = (
            rfm.filter(F.col("segment_label").isNull()).count() == 0
        )

        bi = tables["bi_revenue"]
        checks["bi_revenue_non_negative"] = (
            bi.filter(
                (F.col("total_revenue") < 0)
                | (F.col("avg_order_value") < 0)
                | F.col("total_revenue").isNull()
            ).count()
            == 0
        )
        checks["bi_has_late_delivery_rate"] = (
            "late_delivery_rate" in bi.columns and "return_rate" not in bi.columns
        )

        churn_summary = churn.agg(
            F.min("snapshot_date").alias("min_snapshot"),
            F.max("snapshot_date").alias("max_snapshot"),
            F.countDistinct("snapshot_date").alias("snapshot_count"),
            F.countDistinct("customer_unique_id").alias("unique_customers"),
            F.avg("is_churned").alias("churn_rate"),
        ).first()
        rfm_distribution = {
            row["segment_label"]: int(row["count"])
            for row in rfm.groupBy("segment_label").count().collect()
        }

        return {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "row_counts": row_counts,
            "checks": checks,
            "churn_summary": {
                "min_snapshot": str(churn_summary["min_snapshot"]),
                "max_snapshot": str(churn_summary["max_snapshot"]),
                "snapshot_count": int(churn_summary["snapshot_count"]),
                "unique_customers": int(churn_summary["unique_customers"]),
                "churn_rate": float(churn_summary["churn_rate"]),
            },
            "rfm_distribution": rfm_distribution,
        }
    finally:
        spark.stop()


def main() -> int:
    """Print validation results and return a process exit code."""
    result = validate_gold()
    print(f"Gold validation: {result['status']}")
    print("Gold row counts:")
    for table, count in result["row_counts"].items():
        print(f"- {table}: {count}")
    print("Checks:")
    for name, passed in result["checks"].items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    summary = result["churn_summary"]
    print(
        "Churn summary: "
        f"{summary['min_snapshot']}..{summary['max_snapshot']}, "
        f"snapshots={summary['snapshot_count']}, "
        f"unique_customers={summary['unique_customers']}, "
        f"churn_rate={summary['churn_rate']:.4f}"
    )
    print("RFM segment distribution:")
    for segment, count in sorted(result["rfm_distribution"].items()):
        print(f"- {segment}: {count}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
