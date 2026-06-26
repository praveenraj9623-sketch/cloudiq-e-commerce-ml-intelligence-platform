"""Focused Silver transformation tests using tiny Delta fixtures."""

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

from src.processing.silver import SilverLayer
from src.utils.config import ConfigLoader
from src.utils.spark_session import get_spark_session


def _write_config(tmp_path: Path) -> ConfigLoader:
    """Create a minimal Silver test config."""
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "spark:",
                '  app_name: "CloudIQ-Test-Silver"',
                '  driver_memory: "1g"',
                '  executor_memory: "1g"',
                "  shuffle_partitions: 1",
                "paths:",
                f'  bronze: "{bronze_dir.as_posix()}"',
                f'  silver: "{silver_dir.as_posix()}"',
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


def test_clean_orders_keeps_undelivered_late_metrics_null(
    tmp_path: Path,
) -> None:
    """Undelivered orders are not classified as on time."""
    config = _write_config(tmp_path)
    spark = get_spark_session(config, app_name="CloudIQ-Test-Clean-Orders")
    bronze_dir = tmp_path / "bronze"
    schema = StructType(
        [
            StructField("order_id", StringType(), False),
            StructField("customer_id", StringType(), False),
            StructField("order_status", StringType(), False),
            StructField("order_purchase_timestamp", TimestampType(), True),
            StructField("order_approved_at", TimestampType(), True),
            StructField("order_delivered_carrier_date", TimestampType(), True),
            StructField("order_delivered_customer_date", TimestampType(), True),
            StructField("order_estimated_delivery_date", TimestampType(), True),
            StructField("_source", StringType(), True),
        ]
    )
    rows = [
        (
            "o1",
            "c1",
            "delivered",
            datetime(2018, 1, 1),
            None,
            datetime(2018, 1, 2),
            datetime(2018, 1, 4),
            datetime(2018, 1, 5),
            "test",
        ),
        (
            "o2",
            "c2",
            "canceled",
            datetime(2018, 1, 1),
            None,
            None,
            None,
            datetime(2018, 1, 5),
            "test",
        ),
    ]
    try:
        _write_delta(spark, rows, schema, bronze_dir / "orders")
        result = {
            row["order_id"]: row
            for row in SilverLayer(spark, config).clean_orders().collect()
        }
        assert result["o1"]["is_late"] == 0
        assert result["o1"]["delivery_delay_days"] == 0
        assert result["o1"]["delivery_days"] == 3
        assert result["o1"]["order_approved_at"] == result["o1"][
            "order_purchase_timestamp"
        ]
        assert result["o2"]["is_late"] is None
        assert result["o2"]["delivery_delay_days"] is None
        assert result["o2"]["delivery_days"] is None
    finally:
        spark.stop()


def test_product_demand_uses_fulfilled_orders_and_category_fallbacks(
    tmp_path: Path,
) -> None:
    """Demand excludes unfulfilled orders and preserves category fallbacks."""
    config = _write_config(tmp_path)
    spark = get_spark_session(config, app_name="CloudIQ-Test-Product-Demand")
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"

    orders_schema = StructType(
        [
            StructField("order_id", StringType(), False),
            StructField("order_status", StringType(), False),
            StructField("order_purchase_timestamp", TimestampType(), True),
            StructField("order_year_month", StringType(), True),
        ]
    )
    items_schema = StructType(
        [
            StructField("order_id", StringType(), False),
            StructField("order_item_id", IntegerType(), False),
            StructField("product_id", StringType(), False),
            StructField("seller_id", StringType(), False),
            StructField("price", DoubleType(), False),
            StructField("freight_value", DoubleType(), False),
        ]
    )
    products_schema = StructType(
        [
            StructField("product_id", StringType(), False),
            StructField("product_category_name", StringType(), True),
        ]
    )
    translation_schema = StructType(
        [
            StructField("product_category_name", StringType(), False),
            StructField("product_category_name_english", StringType(), False),
        ]
    )
    try:
        _write_delta(
            spark,
            [
                ("o1", "delivered", datetime(2018, 1, 1), "2018-01"),
                ("o2", "delivered", datetime(2018, 1, 2), "2018-01"),
                ("o3", "delivered", datetime(2018, 1, 3), "2018-01"),
                ("o4", "canceled", datetime(2018, 1, 4), "2018-01"),
                ("o5", "delivered", None, None),
            ],
            orders_schema,
            silver_dir / "orders",
        )
        _write_delta(
            spark,
            [
                ("o1", 1, "p1", "s1", 10.0, 1.0),
                ("o2", 1, "p2", "s1", 20.0, 2.0),
                ("o3", 1, "p3", "s1", 30.0, 3.0),
                ("o4", 1, "p2", "s1", 40.0, 4.0),
                ("o5", 1, "p2", "s1", 50.0, 5.0),
            ],
            items_schema,
            silver_dir / "order_items",
        )
        _write_delta(
            spark,
            [("p1", "beleza"), ("p2", "sem_traducao"), ("p3", None)],
            products_schema,
            bronze_dir / "products",
        )
        _write_delta(
            spark,
            [("beleza", "beauty")],
            translation_schema,
            bronze_dir / "category_translation",
        )

        rows = {
            row["category_name_english"]: row["monthly_units"]
            for row in SilverLayer(spark, config).build_product_demand().collect()
        }
        assert rows == {
            "beauty": 1,
            "untranslated__sem_traducao": 1,
            "unknown": 1,
        }
    finally:
        spark.stop()
