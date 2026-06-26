"""Gold feature engineering layer for CloudIQ.

Produces churn snapshot features, demand history and lag features, canonical
RFM segments, and BI revenue. The architecture corrections are authoritative:

- Churn (C1): sourced from ``silver/master_orders`` only, using temporal
  snapshots. Eligible customers have a delivered order with
  ``order_delivered_customer_date <= T``. All features are snapshot-time
  knowable; the label is derived from delivered orders in ``(T, T+90d]``.
  ``recency_days`` and ``log_recency`` are excluded.
- Demand (C16/C18): training cutoff derived from
  ``max(order_purchase_timestamp)`` resolves to 2018-09; October 2018 is
  excluded. A continuous monthly grid per category from its first observed
  month through 2018-09 is zero-filled. ``rolling_mean_3`` uses
  ``rowsBetween(-3, -1)`` (prior months only). ``gold/demand_history`` is the
  full series; ``gold/demand_features`` is the lag table after dropping rows
  with insufficient history.
- Segmentation (C5/C14): rule-based ``segment_label`` is canonical; no KMeans
  output is produced in this phase.
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

# Demand history, features, training, and evaluation all end here (C16).
_DEMAND_END_MONTH = "2018-09"
# Churn snapshot window: first day of each month, 2017-01 .. 2018-07 (C1).
_CHURN_SNAPSHOTS = [
    f"{year}-{month:02d}-01"
    for year in (2017, 2018)
    for month in range(1, 13)
    if not (year == 2017 and month == 0)
    if (year, month) <= (2018, 7)
]
_CHURN_HORIZON_DAYS = 90


def derive_demand_cutoff_month(orders: DataFrame) -> str:
    """Derive the demand training cutoff month from an orders DataFrame (C16).

    The cutoff is the last *complete* month before the dataset's final partial
    month. For Olist the maximum purchase timestamp is in October 2018, which
    is incomplete, so the cutoff resolves to ``2018-09``.

    Args:
        orders: A DataFrame containing ``order_purchase_timestamp``.

    Returns:
        The cutoff month as a ``yyyy-MM`` string.
    """
    max_ts = orders.select(
        F.max("order_purchase_timestamp").alias("m")
    ).first()["m"]
    # The final calendar month present in the data is partial; exclude it by
    # stepping back one month from the max-timestamp month.
    max_month = max_ts.replace(day=1)
    if max_month.month == 1:
        cutoff_year, cutoff_month = max_month.year - 1, 12
    else:
        cutoff_year, cutoff_month = max_month.year, max_month.month - 1
    return f"{cutoff_year}-{cutoff_month:02d}"


class GoldLayer:
    """Build Gold feature tables from Silver Delta tables."""

    def __init__(self, spark: "SparkSession", config: ConfigLoader) -> None:
        self.spark = spark
        self.config = config
        self.logger = get_logger("processing.gold")
        self.silver_path = config.get_path("paths.silver", create=False)
        self.gold_path = config.get_path("paths.gold")

    def _read_silver(self, table: str) -> DataFrame:
        """Read a Silver Delta table."""
        return self.spark.read.format("delta").load(
            f"{self.silver_path}/{table}"
        )

    def _read_bronze(self, table: str) -> DataFrame:
        """Read a Bronze Delta table, dropping ingestion metadata columns."""
        bronze = self.config.get_path("paths.bronze", create=False)
        df = self.spark.read.format("delta").load(f"{bronze}/{table}")
        meta = [c for c in df.columns if c.startswith("_")]
        return df.drop(*meta) if meta else df

    def build_churn_features(self) -> DataFrame:
        """Build temporal churn snapshot features (Correction 1).

        Each ``(customer_unique_id, snapshot_month)`` is one row. All features
        use only delivered-by-T information; the label uses delivered orders in
        the 90-day window after T. ``recency_days`` and ``log_recency`` are
        never produced.
        """
        master = self._read_silver("master_orders")
        reviews = self._read_bronze("reviews").select(
            "order_id", "review_score", "review_creation_date"
        )
        master = master.join(
            reviews.groupBy("order_id").agg(
                F.min("review_creation_date").alias("review_creation_date")
            ),
            "order_id",
            "left",
        )

        delivered = master.filter(
            (F.col("order_status") == "delivered")
            & F.col("order_delivered_customer_date").isNotNull()
        )

        snapshots: list[DataFrame] = []
        for snap in _CHURN_SNAPSHOTS:
            t = F.to_date(F.lit(snap))
            snapshot_month = snap[:7]

            # Orders delivered to the customer at or before T.
            past = delivered.filter(
                F.to_date(F.col("order_delivered_customer_date")) <= t
            )
            features = past.groupBy("customer_unique_id").agg(
                F.countDistinct("order_id").alias("total_orders_at_T"),
                F.sum("order_revenue").alias("total_revenue_at_T"),
                F.avg("order_revenue").alias("avg_order_value_at_T"),
                F.avg("is_late").alias("late_delivery_rate_at_T"),
                F.datediff(
                    t, F.max("order_delivered_customer_date")
                ).alias("days_since_last_order_at_T"),
                F.datediff(
                    t, F.min("order_delivered_customer_date")
                ).alias("customer_age_days_at_T"),
            )

            # Review features: only reviews created at or before T.
            review_feat = (
                past.filter(
                    F.to_date(F.col("review_creation_date")) <= t
                )
                .groupBy("customer_unique_id")
                .agg(
                    F.avg("avg_review_score").alias("avg_review_score_at_T")
                )
            )

            # Future delivered orders in (T, T+90d] determine the label.
            future = (
                delivered.filter(
                    (
                        F.to_date(F.col("order_delivered_customer_date"))
                        > t
                    )
                    & (
                        F.to_date(F.col("order_delivered_customer_date"))
                        <= F.date_add(t, _CHURN_HORIZON_DAYS)
                    )
                )
                .select("customer_unique_id")
                .distinct()
                .withColumn("_repeat", F.lit(1))
            )

            snap_df = (
                features.join(review_feat, "customer_unique_id", "left")
                .join(future, "customer_unique_id", "left")
                .withColumn("snapshot_month", F.lit(snapshot_month))
                .withColumn(
                    "is_churned",
                    F.when(F.col("_repeat").isNull(), 1).otherwise(0),
                )
                .drop("_repeat")
            )
            snapshots.append(snap_df)

        churn = snapshots[0]
        for extra in snapshots[1:]:
            churn = churn.unionByName(extra)

        churn = (
            churn.withColumn(
                "purchase_frequency_score_at_T",
                F.col("total_orders_at_T")
                / (F.col("customer_age_days_at_T") + F.lit(1))
                * F.lit(30),
            )
            .withColumn(
                "revenue_per_order_norm_at_T",
                F.col("total_revenue_at_T")
                / (F.col("total_orders_at_T") + F.lit(1)),
            )
            .withColumn(
                "high_late_delivery_flag_at_T",
                F.when(
                    F.col("late_delivery_rate_at_T") > 0.3, 1
                ).otherwise(0),
            )
        )
        numeric = [
            "total_orders_at_T",
            "total_revenue_at_T",
            "avg_order_value_at_T",
            "avg_review_score_at_T",
            "late_delivery_rate_at_T",
            "purchase_frequency_score_at_T",
            "revenue_per_order_norm_at_T",
            "high_late_delivery_flag_at_T",
            "days_since_last_order_at_T",
            "customer_age_days_at_T",
        ]
        churn = churn.fillna(0, numeric)
        churn.write.format("delta").mode("overwrite").partitionBy(
            "snapshot_month"
        ).save(f"{self.gold_path}/churn_features")
        self.logger.info("Wrote gold/churn_features")
        return churn

    def _build_demand_history(self) -> DataFrame:
        """Build the continuous zero-filled monthly grid per category (C16)."""
        orders = self._read_silver("orders")
        cutoff = derive_demand_cutoff_month(orders)
        self.logger.info("Demand training cutoff month: {}", cutoff)

        demand = self._read_silver("product_demand").filter(
            F.col("order_year_month") <= cutoff
        )
        base = demand.select(
            "category_name_english",
            "order_year_month",
            "monthly_units",
            "monthly_revenue",
            "avg_price",
        )

        # First observed month per category and a month index helper.
        first_month = base.groupBy("category_name_english").agg(
            F.min("order_year_month").alias("first_month")
        )

        def _month_index(col: "F.Column") -> "F.Column":
            year = F.substring(col, 1, 4).cast("int")
            month = F.substring(col, 6, 2).cast("int")
            return year * F.lit(12) + month

        cutoff_idx = (
            int(cutoff[:4]) * 12 + int(cutoff[5:7])
        )
        first_month = first_month.withColumn(
            "first_idx", _month_index(F.col("first_month"))
        )
        # Generate the continuous index range [first_idx, cutoff_idx].
        grid = first_month.withColumn(
            "idx",
            F.explode(
                F.sequence(F.col("first_idx"), F.lit(cutoff_idx))
            ),
        )
        grid = grid.withColumn(
            "order_year_month",
            F.concat_ws(
                "-",
                F.format_string(
                    "%04d", ((F.col("idx") - 1) / F.lit(12)).cast("int")
                ),
                F.format_string(
                    "%02d",
                    (((F.col("idx") - 1) % F.lit(12)) + 1).cast("int"),
                ),
            ),
        ).select("category_name_english", "order_year_month")

        history = (
            grid.join(
                base,
                ["category_name_english", "order_year_month"],
                "left",
            )
            .fillna(0, ["monthly_units", "monthly_revenue", "avg_price"])
        )
        return history

    def build_demand_history(self) -> DataFrame:
        """Write ``gold/demand_history`` (full continuous series, C16/C18)."""
        history = self._build_demand_history()
        history.write.format("delta").mode("overwrite").partitionBy(
            "category_name_english"
        ).save(f"{self.gold_path}/demand_history")
        self.logger.info("Wrote gold/demand_history")
        return history

    def build_demand_forecast_features(self) -> DataFrame:
        """Build ``gold/demand_features`` lag table (Correction 16).

        Lags and ``rolling_mean_3`` are computed over the continuous monthly
        grid. ``rolling_mean_3`` uses ``rowsBetween(-3, -1)`` so the current
        month's value is never included. Rows lacking ``lag_4`` history are
        dropped, leaving fewer rows than ``demand_history``.
        """
        history = self._build_demand_history()
        win = Window.partitionBy("category_name_english").orderBy(
            "order_year_month"
        )
        roll_win = win.rowsBetween(-3, -1)
        features = (
            history.withColumn("lag_1", F.lag("monthly_units", 1).over(win))
            .withColumn("lag_2", F.lag("monthly_units", 2).over(win))
            .withColumn("lag_4", F.lag("monthly_units", 4).over(win))
            .withColumn(
                "rolling_mean_3",
                F.avg("monthly_units").over(roll_win),
            )
            .withColumn(
                "month_num",
                F.month(F.to_date(F.col("order_year_month"), "yyyy-MM")),
            )
            .withColumn(
                "is_q4",
                F.when(F.col("month_num") >= 10, 1).otherwise(0),
            )
            .filter(F.col("lag_4").isNotNull())
        )
        features.write.format("delta").mode("overwrite").partitionBy(
            "category_name_english"
        ).save(f"{self.gold_path}/demand_features")
        self.logger.info("Wrote gold/demand_features")
        return features

    def build_rfm_segments(self) -> DataFrame:
        """Build canonical rule-based RFM segments (Corrections 5 and 14)."""
        profile = self._read_silver("customer_profile")
        rec_win = Window.orderBy(F.col("recency_days").desc())
        freq_win = Window.orderBy(F.col("total_orders").asc())
        mon_win = Window.orderBy(F.col("total_revenue").asc())
        rfm = (
            profile.withColumn(
                "recency_score", F.ntile(5).over(rec_win)
            )
            .withColumn("frequency_score", F.ntile(5).over(freq_win))
            .withColumn("monetary_score", F.ntile(5).over(mon_win))
        )
        rfm = rfm.withColumn(
            "rfm_total",
            F.col("recency_score")
            + F.col("frequency_score")
            + F.col("monetary_score"),
        ).withColumn(
            "segment_label",
            F.when(F.col("rfm_total") >= 13, "Champion")
            .when(F.col("rfm_total") >= 10, "Loyal")
            .when(F.col("rfm_total") >= 7, "Potential")
            .when(F.col("rfm_total") >= 4, "At Risk")
            .otherwise("Lost"),
        )
        rfm.write.format("delta").mode("overwrite").save(
            f"{self.gold_path}/rfm_segments"
        )
        self.logger.info("Wrote gold/rfm_segments")
        return rfm

    def build_bi_revenue(self) -> DataFrame:
        """Build the BI revenue summary table."""
        master = self._read_silver("master_orders")
        bi = master.groupBy(
            "order_year",
            "order_month",
            "order_year_month",
            "customer_state",
            "primary_payment_type",
        ).agg(
            F.count("order_id").alias("total_orders"),
            F.sum("order_revenue").alias("total_revenue"),
            F.avg("order_revenue").alias("avg_order_value"),
            F.avg("is_late").alias("return_rate"),
        )
        bi.write.format("delta").mode("overwrite").partitionBy(
            "order_year"
        ).save(f"{self.gold_path}/bi_revenue")
        self.logger.info("Wrote gold/bi_revenue")
        return bi

    def run_pipeline(self) -> dict:
        """Run all Gold build steps in order."""
        start = time.time()
        steps = [
            ("churn_features", self.build_churn_features),
            ("demand_history", self.build_demand_history),
            ("demand_features", self.build_demand_forecast_features),
            ("rfm_segments", self.build_rfm_segments),
            ("bi_revenue", self.build_bi_revenue),
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
                    "Gold step failed: {}", name
                )
                results[name] = {
                    "table": name,
                    "status": "FAILED",
                    "error": str(exc),
                }
        results["total_s"] = round(time.time() - start, 2)
        return results
