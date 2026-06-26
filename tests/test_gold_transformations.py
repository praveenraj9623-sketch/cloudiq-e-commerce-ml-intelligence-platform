"""Focused Gold transformation tests using tiny synthetic data."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from src.processing.gold import GoldLayer
from src.utils.config import ConfigLoader
from src.utils.spark_session import get_spark_session


def _write_config(tmp_path: Path) -> ConfigLoader:
    """Create a minimal Gold test config."""
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"
    bronze_dir = tmp_path / "bronze"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "spark:",
                '  app_name: "CloudIQ-Test-Gold"',
                '  driver_memory: "1g"',
                '  executor_memory: "1g"',
                "  shuffle_partitions: 1",
                "paths:",
                f'  silver: "{silver_dir.as_posix()}"',
                f'  gold: "{gold_dir.as_posix()}"',
                f'  bronze: "{bronze_dir.as_posix()}"',
            ]
        ),
        encoding="utf-8",
    )
    return ConfigLoader(str(config_path), env_path="nonexistent.env")


def _write_delta(spark: Any, rows: list[tuple], schema: StructType, path: Path) -> None:
    """Write rows to a Delta path."""
    spark.createDataFrame(rows, schema=schema).write.format("delta").mode(
        "overwrite"
    ).save(str(path))


def test_demand_features_use_prior_rows_and_next_month_target(
    tmp_path: Path,
) -> None:
    """Demand rolling windows exclude current month and target the next month."""
    config = _write_config(tmp_path)
    spark = get_spark_session(config, app_name="CloudIQ-Test-Gold-Demand")
    silver_dir = tmp_path / "silver"
    schema = StructType(
        [
            StructField("category_name_english", StringType(), False),
            StructField("order_year_month", StringType(), False),
            StructField("monthly_units", IntegerType(), False),
            StructField("monthly_revenue", DoubleType(), False),
            StructField("avg_price", DoubleType(), False),
        ]
    )
    try:
        _write_delta(
            spark,
            [
                ("cat", "2018-01", 10, 100.0, 10.0),
                ("cat", "2018-02", 20, 200.0, 10.0),
                ("cat", "2018-03", 30, 300.0, 10.0),
                ("cat", "2018-04", 40, 400.0, 10.0),
                ("cat", "2018-05", 50, 500.0, 10.0),
                ("cat", "2018-06", 60, 600.0, 10.0),
            ],
            schema,
            silver_dir / "product_demand",
        )
        rows = GoldLayer(spark, config).build_demand_forecast_features().collect()
        assert len(rows) == 1
        row = rows[0]
        assert row["order_year_month"] == "2018-05"
        assert row["monthly_units"] == 50
        assert row["target_next_month"] == 60
        assert row["lag_1"] == 40
        assert row["lag_2"] == 30
        assert row["lag_4"] == 10
        assert row["rolling_mean_3"] == 30
    finally:
        spark.stop()


def test_rfm_score_direction_rewards_recent_frequent_high_value(
    tmp_path: Path,
) -> None:
    """Lower recency and higher frequency/monetary values receive higher scores."""
    config = _write_config(tmp_path)
    spark = get_spark_session(config, app_name="CloudIQ-Test-Gold-RFM")
    silver_dir = tmp_path / "silver"
    schema = StructType(
        [
            StructField("customer_unique_id", StringType(), False),
            StructField("first_purchase", TimestampType(), True),
            StructField("last_purchase", TimestampType(), True),
            StructField("total_orders", IntegerType(), False),
            StructField("total_revenue", DoubleType(), False),
            StructField("avg_order_value", DoubleType(), False),
            StructField("avg_review_score", DoubleType(), True),
            StructField("late_delivery_rate", DoubleType(), True),
            StructField("recency_days", IntegerType(), False),
            StructField("customer_age_days", IntegerType(), False),
            StructField("preferred_payment", StringType(), True),
            StructField("customer_state", StringType(), True),
        ]
    )
    try:
        _write_delta(
            spark,
            [
                ("c1", None, None, 1, 100.0, 100.0, 5.0, 0.0, 100, 1, "cc", "SP"),
                ("c2", None, None, 2, 200.0, 100.0, 5.0, 0.0, 80, 1, "cc", "SP"),
                ("c3", None, None, 3, 300.0, 100.0, 5.0, 0.0, 60, 1, "cc", "SP"),
                ("c4", None, None, 4, 400.0, 100.0, 5.0, 0.0, 40, 1, "cc", "SP"),
                ("c5", None, None, 5, 500.0, 100.0, 5.0, 0.0, 20, 1, "cc", "SP"),
            ],
            schema,
            silver_dir / "customer_profile",
        )
        by_customer = {
            row["customer_unique_id"]: row
            for row in GoldLayer(spark, config).build_rfm_segments().collect()
        }
        assert by_customer["c5"]["recency_score"] == 5
        assert by_customer["c5"]["frequency_score"] == 5
        assert by_customer["c5"]["monetary_score"] == 5
        assert by_customer["c5"]["segment_label"] == "Champion"
        assert by_customer["c1"]["recency_score"] == 1
        assert by_customer["c1"]["frequency_score"] == 1
        assert by_customer["c1"]["monetary_score"] == 1
        assert by_customer["c1"]["segment_label"] == "Lost"
    finally:
        spark.stop()


def test_churn_snapshot_label_boundary_and_feature_cutoff(
    tmp_path: Path,
) -> None:
    """Purchases on snapshot+90 days prevent churn; later purchases do not."""
    config = _write_config(tmp_path)
    spark = get_spark_session(config, app_name="CloudIQ-Test-Gold-Churn")
    order_schema = StructType(
        [
            StructField("order_id", StringType(), False),
            StructField("customer_unique_id", StringType(), False),
            StructField("order_purchase_timestamp", TimestampType(), False),
            StructField("order_revenue", DoubleType(), False),
            StructField("order_delivered_customer_date", TimestampType(), True),
            StructField("is_late", IntegerType(), True),
        ]
    )
    review_schema = StructType(
        [
            StructField("order_id", StringType(), False),
            StructField("review_score", IntegerType(), True),
            StructField("review_creation_date", TimestampType(), True),
        ]
    )
    try:
        orders = spark.createDataFrame(
            [
                ("o1", "c1", datetime(2020, 1, 1), 100.0, datetime(2020, 1, 2), 1),
                ("o2", "c1", datetime(2020, 3, 31), 200.0, datetime(2020, 4, 1), 0),
                ("o3", "c2", datetime(2020, 1, 1), 50.0, datetime(2020, 1, 1), 1),
                ("o4", "c3", datetime(2020, 1, 1), 75.0, datetime(2020, 1, 1), 0),
                ("o5", "c3", datetime(2020, 4, 1), 75.0, datetime(2020, 4, 2), 0),
            ],
            order_schema,
        )
        reviews = spark.createDataFrame(
            [
                ("o1", 5, datetime(2020, 1, 2)),
                ("o3", 4, datetime(2020, 1, 1)),
            ],
            review_schema,
        )
        rows = {
            row["customer_unique_id"]: row
            for row in GoldLayer(spark, config)
            ._build_churn_for_snapshots(orders, reviews, ["2020-01-01"])
            .collect()
        }
        assert rows["c1"]["is_churned"] == 0
        assert rows["c1"]["total_orders"] == 1
        assert rows["c1"]["avg_review_score"] == 0
        assert rows["c1"]["late_delivery_rate"] == 0
        assert rows["c2"]["is_churned"] == 1
        assert rows["c2"]["avg_review_score"] == 4
        assert rows["c2"]["late_delivery_rate"] == 1
        assert rows["c3"]["is_churned"] == 1
    finally:
        spark.stop()
