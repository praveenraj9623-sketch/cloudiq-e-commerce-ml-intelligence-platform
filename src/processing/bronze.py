"""Bronze ingestion layer for CloudIQ.

Reads the nine raw Olist CSV files declared in ``config.olist_files`` and
writes each as a Delta table under ``paths.bronze``. Required CSVs that are
missing cause a clear :class:`FileNotFoundError` (Phase 1 gate).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import DoubleType, IntegerType

from src.utils.config import ConfigLoader
from src.utils.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import SparkSession

_TS_FMT = "yyyy-MM-dd HH:mm:ss"


class BronzeLayer:
    """Ingest raw Olist CSV files into Bronze Delta tables."""

    def __init__(self, spark: "SparkSession", config: ConfigLoader) -> None:
        self.spark = spark
        self.config = config
        self.logger = get_logger("processing.bronze")
        self.raw_path = config.get_path("paths.raw", create=False)
        self.bronze_path = config.get_path("paths.bronze")

    def _csv_path(self, key: str) -> Path:
        """Return the raw CSV path for an ``olist_files`` key, asserting it exists."""
        filename = self.config.get(f"olist_files.{key}")
        if filename is None:
            raise KeyError(f"olist_files.{key} is not configured")
        path = self.raw_path / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Required Olist CSV missing for '{key}': {path}. "
                "Populate data/raw/ before running the bronze layer."
            )
        return path

    def _read_csv(self, key: str) -> DataFrame:
        """Read a raw CSV with header and schema inference."""
        path = self._csv_path(key)
        self.logger.info("Reading raw CSV: {}", path)
        return (
            self.spark.read.option("header", "true")
            .option("inferSchema", "true")
            .csv(str(path))
        )

    def _add_metadata(self, df: DataFrame, source: str) -> DataFrame:
        """Append ingestion metadata columns to a DataFrame."""
        data_cols = [c for c in df.columns if not c.startswith("_")]
        return (
            df.withColumn("_source", F.lit(source))
            .withColumn("_ingested_at", F.current_timestamp())
            .withColumn("_batch_id", F.lit(str(uuid4())))
            .withColumn(
                "_row_hash",
                F.md5(
                    F.concat_ws(
                        "|", *[F.col(c).cast("string") for c in data_cols]
                    )
                ),
            )
        )

    def _write_delta(
        self,
        df: DataFrame,
        table: str,
        partition_cols: Optional[list[str]] = None,
    ) -> dict:
        """Write a DataFrame as Delta and return a status dict with row count."""
        target = self.bronze_path / table
        try:
            writer = df.write.format("delta").mode("overwrite")
            if partition_cols:
                writer = writer.partitionBy(*partition_cols)
            writer.save(str(target))
            rows = (
                self.spark.read.format("delta").load(str(target)).count()
            )
            self.logger.info("Wrote bronze/{} ({} rows)", table, rows)
            return {"table": table, "rows": rows, "status": "SUCCESS"}
        except Exception as exc:  # noqa: BLE001 - report and continue
            self.logger.opt(exception=True).error(
                "Failed writing bronze/{}", table
            )
            return {"table": table, "status": "FAILED", "error": str(exc)}

    def ingest_orders(self) -> dict:
        """Ingest orders, casting five timestamp columns."""
        df = self._read_csv("orders")
        ts_cols = [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
        for col in ts_cols:
            df = df.withColumn(col, F.to_timestamp(F.col(col), _TS_FMT))
        df = self._add_metadata(df, "olist_orders")
        return self._write_delta(df, "orders", partition_cols=["order_status"])

    def ingest_customers(self) -> dict:
        """Ingest customers."""
        df = self._read_csv("customers")
        df = self._add_metadata(df, "olist_customers")
        return self._write_delta(df, "customers")

    def ingest_order_items(self) -> dict:
        """Ingest order items, casting numeric and timestamp columns."""
        df = self._read_csv("order_items")
        df = (
            df.withColumn(
                "shipping_limit_date",
                F.to_timestamp(F.col("shipping_limit_date"), _TS_FMT),
            )
            .withColumn("price", F.col("price").cast(DoubleType()))
            .withColumn(
                "freight_value", F.col("freight_value").cast(DoubleType())
            )
            .withColumn("shipping_year", F.year(F.col("shipping_limit_date")))
        )
        df = self._add_metadata(df, "olist_order_items")
        return self._write_delta(
            df, "order_items", partition_cols=["shipping_year"]
        )

    def ingest_products(self) -> dict:
        """Ingest products, casting weight and dimension columns."""
        df = self._read_csv("products")
        numeric = [
            "product_name_lenght",
            "product_description_lenght",
            "product_photos_qty",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ]
        for col in numeric:
            if col in df.columns:
                df = df.withColumn(col, F.col(col).cast(DoubleType()))
        df = self._add_metadata(df, "olist_products")
        return self._write_delta(df, "products")

    def ingest_reviews(self) -> dict:
        """Ingest reviews, casting score and timestamp columns."""
        path = self._csv_path("reviews")
        self.logger.info("Reading raw reviews CSV: {}", path)
        df = (
            self.spark.read.option("header", "true")
            .option("inferSchema", "true")
            .option("multiLine", "true")
            .option("quote", '"')
            .option("escape", '"')
            .option("mode", "FAILFAST")
            .csv(str(path))
        )
        df = (
            df.withColumn(
                "review_score", F.col("review_score").cast(IntegerType())
            )
            .withColumn(
                "review_creation_date",
                F.to_timestamp(F.col("review_creation_date"), _TS_FMT),
            )
            .withColumn(
                "review_answer_timestamp",
                F.to_timestamp(F.col("review_answer_timestamp"), _TS_FMT),
            )
        )
        df = self._add_metadata(df, "olist_reviews")
        return self._write_delta(
            df, "reviews", partition_cols=["review_score"]
        )

    def ingest_payments(self) -> dict:
        """Ingest payments, casting installment and value columns."""
        df = self._read_csv("payments")
        df = df.withColumn(
            "payment_installments",
            F.col("payment_installments").cast(IntegerType()),
        ).withColumn(
            "payment_value", F.col("payment_value").cast(DoubleType())
        )
        df = self._add_metadata(df, "olist_payments")
        return self._write_delta(
            df, "payments", partition_cols=["payment_type"]
        )

    def ingest_sellers(self) -> dict:
        """Ingest sellers."""
        df = self._read_csv("sellers")
        df = self._add_metadata(df, "olist_sellers")
        return self._write_delta(df, "sellers")

    def ingest_geolocation(self) -> dict:
        """Ingest geolocation, casting latitude and longitude."""
        df = self._read_csv("geolocation")
        df = df.withColumn(
            "geolocation_lat", F.col("geolocation_lat").cast(DoubleType())
        ).withColumn(
            "geolocation_lng", F.col("geolocation_lng").cast(DoubleType())
        )
        df = self._add_metadata(df, "olist_geo")
        return self._write_delta(df, "geolocation")

    def ingest_category_translation(self) -> dict:
        """Ingest the product category name translation table."""
        df = self._read_csv("category_translation")
        df = self._add_metadata(df, "olist_categories")
        return self._write_delta(df, "category_translation")

    def run_pipeline(self) -> dict:
        """Run all nine ingest methods in order and return a results dict."""
        start = time.time()
        steps = [
            ("orders", self.ingest_orders),
            ("customers", self.ingest_customers),
            ("order_items", self.ingest_order_items),
            ("products", self.ingest_products),
            ("reviews", self.ingest_reviews),
            ("payments", self.ingest_payments),
            ("sellers", self.ingest_sellers),
            ("geolocation", self.ingest_geolocation),
            ("category_translation", self.ingest_category_translation),
        ]
        results: dict = {}
        for name, fn in steps:
            try:
                results[name] = fn()
            except Exception as exc:  # noqa: BLE001 - report and continue
                self.logger.opt(exception=True).error(
                    "Bronze step failed: {}", name
                )
                results[name] = {
                    "table": name,
                    "status": "FAILED",
                    "error": str(exc),
                }
        results["total_duration_seconds"] = round(time.time() - start, 2)
        return results
