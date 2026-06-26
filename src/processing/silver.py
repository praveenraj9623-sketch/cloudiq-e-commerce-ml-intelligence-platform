"""Silver cleaning and join layer for CloudIQ.

Cleans orders and order items, builds a deduplicated ``master_orders`` fact
table without join fan-out, and derives customer, product-demand, and seller
performance tables. Seller revenue and review/late metrics follow Corrections
3 and 4: revenue and order counts come directly from ``order_items``, and
review/late rates use a distinct ``seller_id`` + ``order_id`` mapping.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.utils.config import ConfigLoader
from src.utils.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import SparkSession

# Dataset ends 2018-10-17; used for recency in the customer profile only.
_DATASET_END = "2018-10-17"


class SilverLayer:
    """Build cleaned and conformed Silver Delta tables."""

    def __init__(self, spark: "SparkSession", config: ConfigLoader) -> None:
        self.spark = spark
        self.config = config
        self.logger = get_logger("processing.silver")
        self.bronze_path = config.get_path("paths.bronze", create=False)
        self.silver_path = config.get_path("paths.silver")

    def _read_bronze(self, table: str) -> DataFrame:
        """Read a Bronze Delta table, dropping ingestion metadata columns."""
        df = self.spark.read.format("delta").load(
            f"{self.bronze_path}/{table}"
        )
        meta = [c for c in df.columns if c.startswith("_")]
        return df.drop(*meta) if meta else df

    def _read_silver(self, table: str) -> DataFrame:
        """Read a Silver Delta table."""
        return self.spark.read.format("delta").load(
            f"{self.silver_path}/{table}"
        )

    def clean_orders(self) -> DataFrame:
        """Clean orders and derive calendar and delivery fields."""
        df = self._read_bronze("orders")
        df = (
            df.withColumn(
                "is_canceled",
                F.when(F.col("order_status") == "canceled", 1).otherwise(0),
            )
            .withColumn("order_year", F.year("order_purchase_timestamp"))
            .withColumn("order_month", F.month("order_purchase_timestamp"))
            .withColumn(
                "order_year_month",
                F.date_format("order_purchase_timestamp", "yyyy-MM"),
            )
            .withColumn(
                "delivery_days",
                F.datediff(
                    "order_delivered_customer_date",
                    "order_purchase_timestamp",
                ),
            )
            .withColumn(
                "is_late",
                F.when(
                    F.col("order_delivered_customer_date")
                    > F.col("order_estimated_delivery_date"),
                    1,
                ).otherwise(0),
            )
            .withColumn(
                "delivery_delay_days",
                F.when(
                    F.col("is_late") == 1,
                    F.datediff(
                        "order_delivered_customer_date",
                        "order_estimated_delivery_date",
                    ),
                ).otherwise(0),
            )
            .withColumn(
                "order_approved_at",
                F.coalesce(
                    F.col("order_approved_at"),
                    F.col("order_purchase_timestamp"),
                ),
            )
        )
        target = f"{self.silver_path}/orders"
        df.write.format("delta").mode("overwrite").partitionBy(
            "order_year_month"
        ).save(target)
        self.logger.info("Wrote silver/orders")
        return df

    def clean_order_items(self) -> DataFrame:
        """Clean order items and derive value and freight ratio fields."""
        df = self._read_bronze("order_items")
        df = (
            df.withColumn(
                "total_item_value",
                F.col("price") + F.col("freight_value"),
            )
            .filter(F.col("price") > 0)
            .withColumn(
                "freight_ratio",
                F.when(
                    F.col("total_item_value") > 0,
                    F.col("freight_value") / F.col("total_item_value"),
                ).otherwise(0.0),
            )
        )
        df.write.format("delta").mode("overwrite").save(
            f"{self.silver_path}/order_items"
        )
        self.logger.info("Wrote silver/order_items")
        return df

    def build_master_orders(self) -> DataFrame:
        """Join orders with per-order aggregates without row fan-out."""
        base = self._read_silver("orders")
        input_rows = base.count()

        items_agg = (
            self._read_bronze("order_items")
            .groupBy("order_id")
            .agg(
                F.sum("price").alias("order_revenue"),
                F.sum("freight_value").alias("freight_total"),
                F.count("*").alias("item_count"),
            )
        )
        pay_agg = (
            self._read_bronze("payments")
            .groupBy("order_id")
            .agg(
                F.sum("payment_value").alias("payment_total"),
                F.first("payment_type").alias("primary_payment_type"),
            )
        )
        rev_agg = (
            self._read_bronze("reviews")
            .groupBy("order_id")
            .agg(
                F.avg(F.col("review_score").cast("double")).alias(
                    "avg_review_score"
                )
            )
        )
        customers = self._read_bronze("customers").select(
            "customer_id",
            "customer_unique_id",
            "customer_city",
            "customer_state",
        )

        master = (
            base.join(items_agg, "order_id", "left")
            .join(pay_agg, "order_id", "left")
            .join(rev_agg, "order_id", "left")
            .join(customers, "customer_id", "left")
        )
        master = master.withColumn(
            "revenue_per_item",
            F.when(
                F.col("item_count") > 0,
                F.col("order_revenue") / F.col("item_count"),
            ).otherwise(0.0),
        )
        master = master.fillna(
            0,
            [
                "order_revenue",
                "freight_total",
                "item_count",
                "payment_total",
                "avg_review_score",
                "revenue_per_item",
                "delivery_days",
                "is_late",
                "delivery_delay_days",
            ],
        ).fillna(
            "unknown",
            ["primary_payment_type", "customer_city", "customer_state"],
        )
        master = master.dropDuplicates(["order_id"])

        output_rows = master.count()
        self.logger.info(
            "master_orders input={} output={}", input_rows, output_rows
        )
        master.write.format("delta").mode("overwrite").partitionBy(
            "order_year_month"
        ).save(f"{self.silver_path}/master_orders")
        return master

    def build_customer_profile(self) -> DataFrame:
        """Aggregate master orders into a per-customer profile."""
        master = self._read_silver("master_orders")
        profile = master.groupBy("customer_unique_id").agg(
            F.min("order_purchase_timestamp").alias("first_purchase"),
            F.max("order_purchase_timestamp").alias("last_purchase"),
            F.countDistinct("order_id").alias("total_orders"),
            F.sum("order_revenue").alias("total_revenue"),
            F.avg("order_revenue").alias("avg_order_value"),
            F.avg("avg_review_score").alias("avg_review_score"),
            F.avg("is_late").alias("late_delivery_rate"),
            F.datediff(
                F.lit(_DATASET_END), F.max("order_purchase_timestamp")
            ).alias("recency_days"),
            F.datediff(
                F.max("order_purchase_timestamp"),
                F.min("order_purchase_timestamp"),
            ).alias("customer_age_days"),
            F.first("primary_payment_type").alias("preferred_payment"),
            F.first("customer_state").alias("customer_state"),
        )
        profile.write.format("delta").mode("overwrite").save(
            f"{self.silver_path}/customer_profile"
        )
        self.logger.info("Wrote silver/customer_profile")
        return profile

    def build_product_demand(self) -> DataFrame:
        """Build monthly product demand by English category name."""
        items = self._read_bronze("order_items")
        orders = self._read_silver("orders").select(
            "order_id", "order_year_month"
        )
        products = self._read_bronze("products").select(
            "product_id", "product_category_name"
        )
        translation = self._read_bronze("category_translation")

        joined = (
            items.join(orders, "order_id", "inner")
            .join(products, "product_id", "left")
            .join(translation, "product_category_name", "left")
            .withColumn(
                "category_name_english",
                F.coalesce(
                    F.col("product_category_name_english"),
                    F.col("product_category_name"),
                    F.lit("unknown"),
                ),
            )
        )
        demand = joined.groupBy(
            "category_name_english", "order_year_month"
        ).agg(
            F.count("order_item_id").alias("monthly_units"),
            F.sum("price").alias("monthly_revenue"),
            F.avg("price").alias("avg_price"),
        )
        win = Window.partitionBy("category_name_english").orderBy(
            "order_year_month"
        )
        demand = demand.withColumn(
            "prev_month", F.lag("monthly_units", 1).over(win)
        ).withColumn(
            "mom_growth",
            F.when(
                F.col("prev_month") > 0,
                (F.col("monthly_units") - F.col("prev_month"))
                / F.col("prev_month")
                * 100,
            ).otherwise(0.0),
        )
        demand.write.format("delta").mode("overwrite").partitionBy(
            "order_year_month"
        ).save(f"{self.silver_path}/product_demand")
        self.logger.info("Wrote silver/product_demand")
        return demand

    def build_seller_performance(self) -> DataFrame:
        """Build seller performance (Corrections 3 and 4).

        Revenue and order counts are aggregated directly from ``order_items``
        (never via ``master_orders``) to avoid item-level revenue duplication.
        Review and late-delivery rates use a distinct ``seller_id`` +
        ``order_id`` mapping so each order contributes one value per seller
        regardless of item count.
        """
        items = self._read_bronze("order_items")
        sellers = self._read_bronze("sellers").select(
            "seller_id", "seller_state"
        )
        master = self._read_silver("master_orders")

        # Correction 3: revenue and distinct order count from order_items.
        seller_agg = items.groupBy("seller_id").agg(
            F.countDistinct("order_id").alias("total_orders"),
            F.sum("price").alias("total_revenue_excl_freight"),
            F.sum("freight_value").alias("total_freight"),
            F.sum(F.col("price") + F.col("freight_value")).alias(
                "total_revenue_incl_freight"
            ),
        )

        # Correction 4: distinct seller-order mapping before joining master.
        seller_order_map = items.select(
            "seller_id", "order_id"
        ).dropDuplicates()
        seller_metrics = (
            seller_order_map.join(master, on="order_id", how="left")
            .groupBy("seller_id")
            .agg(
                F.avg("avg_review_score").alias("avg_review"),
                F.avg("is_late").alias("late_rate"),
            )
        )

        perf = (
            seller_agg.join(seller_metrics, "seller_id", "left")
            .join(sellers, "seller_id", "left")
            .withColumn(
                "seller_state",
                F.coalesce(F.col("seller_state"), F.lit("unknown")),
            )
            .withColumn(
                # Tier uses product revenue only (excl. freight) per C3.
                "performance_tier",
                F.when(
                    F.col("total_revenue_excl_freight") > 10000, "high"
                )
                .when(
                    F.col("total_revenue_excl_freight") >= 1000, "medium"
                )
                .otherwise("low"),
            )
        )
        perf.write.format("delta").mode("overwrite").partitionBy(
            "seller_state"
        ).save(f"{self.silver_path}/seller_performance")
        self.logger.info("Wrote silver/seller_performance")
        return perf

    def run_pipeline(self) -> dict:
        """Run all Silver build steps in dependency order."""
        start = time.time()
        steps = [
            ("orders", self.clean_orders),
            ("order_items", self.clean_order_items),
            ("master_orders", self.build_master_orders),
            ("customer_profile", self.build_customer_profile),
            ("product_demand", self.build_product_demand),
            ("seller_performance", self.build_seller_performance),
        ]
        results: dict = {}
        for name, fn in steps:
            step_start = time.time()
            try:
                df = fn()
                results[name] = {
                    "table": name,
                    "rows": df.count(),
                    "status": "SUCCESS",
                    "duration_s": round(time.time() - step_start, 2),
                }
            except Exception as exc:  # noqa: BLE001 - report and continue
                self.logger.opt(exception=True).error(
                    "Silver step failed: {}", name
                )
                results[name] = {
                    "table": name,
                    "status": "FAILED",
                    "error": str(exc),
                }
        results["total_s"] = round(time.time() - start, 2)
        return results
