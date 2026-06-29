"""Gold feature engineering layer for CloudIQ.

Builds leakage-safe Gold tables:

- ``demand_features`` predicts next month's unit demand from prior-month lags.
- ``churn_features`` is a snapshot training table whose features use only data
  known at or before each snapshot date.
- ``rfm_segments`` applies directionally correct RFM scoring.
- ``bi_revenue`` exposes revenue and late-delivery metrics for reporting.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
import time
from typing import TYPE_CHECKING

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.utils.config import ConfigLoader
from src.utils.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import SparkSession

_CHURN_PRIOR_HISTORY_DAYS = 90
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


def _as_date(value: date | datetime) -> date:
    """Return a ``date`` for a date or datetime value."""
    if isinstance(value, datetime):
        return value.date()
    return value


def _first_month_start_on_or_after(value: date) -> date:
    """Return the first day of the current or next month on/after ``value``."""
    if value.day == 1:
        return value
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def _next_month_start(value: date) -> date:
    """Return the first day of the month after ``value``."""
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def derive_churn_snapshot_dates(
    min_purchase_ts: date | datetime,
    max_purchase_ts: date | datetime,
    prior_history_days: int = _CHURN_PRIOR_HISTORY_DAYS,
    horizon_days: int = _CHURN_HORIZON_DAYS,
) -> list[str]:
    """Derive monthly churn snapshot dates with prior and future coverage."""
    first_allowed = _as_date(min_purchase_ts) + timedelta(days=prior_history_days)
    last_allowed = _as_date(max_purchase_ts) - timedelta(days=horizon_days)
    current = _first_month_start_on_or_after(first_allowed)
    snapshots: list[str] = []
    while current <= last_allowed:
        snapshots.append(current.isoformat())
        current = _next_month_start(current)
    return snapshots


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
        """Build a leakage-safe snapshot churn training table.

        Features are derived from completed, non-cancelled purchases on or
        before ``snapshot_date``. The label is whether the customer has no
        completed purchase in the following 90 days.
        """
        master = self._read_silver("master_orders")
        reviews = self._read_bronze("reviews").select(
            "order_id", "review_score", "review_creation_date"
        )
        completed = self._completed_churn_orders(master)
        bounds = completed.agg(
            F.min("order_purchase_timestamp").alias("min_purchase"),
            F.max("order_purchase_timestamp").alias("max_purchase"),
        ).first()
        if bounds["min_purchase"] is None or bounds["max_purchase"] is None:
            raise ValueError("Cannot build churn features without purchases")

        snapshot_dates = derive_churn_snapshot_dates(
            bounds["min_purchase"],
            bounds["max_purchase"],
        )
        if not snapshot_dates:
            raise ValueError("No churn snapshots have sufficient date coverage")

        churn = self._build_churn_for_snapshots(completed, reviews, snapshot_dates)
        churn.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).partitionBy("snapshot_date").save(f"{self.gold_path}/churn_features")

        written = self.spark.read.format("delta").load(
            f"{self.gold_path}/churn_features"
        )
        row_count = written.count()
        unique_customers = written.select("customer_unique_id").distinct().count()
        summary = written.agg(
            F.min("snapshot_date").alias("min_snapshot"),
            F.max("snapshot_date").alias("max_snapshot"),
            F.avg("is_churned").alias("churn_rate"),
        ).first()
        self.logger.info(
            "Wrote gold/churn_features snapshots={} rows={} "
            "unique_customers={} churn_rate={} range={}..{}",
            len(snapshot_dates),
            row_count,
            unique_customers,
            round(float(summary["churn_rate"]), 4),
            summary["min_snapshot"],
            summary["max_snapshot"],
        )
        return written

    def _completed_churn_orders(self, master: DataFrame) -> DataFrame:
        """Return completed, non-cancelled orders eligible for churn features."""
        return master.filter(
            (F.col("order_status") == "delivered")
            & F.col("order_purchase_timestamp").isNotNull()
            & F.col("customer_unique_id").isNotNull()
        )

    def _build_churn_for_snapshots(
        self,
        completed_orders: DataFrame,
        reviews: DataFrame,
        snapshot_dates: list[str],
    ) -> DataFrame:
        """Build churn rows for explicit snapshot dates."""
        snapshots = self.spark.createDataFrame(
            [(snapshot,) for snapshot in snapshot_dates],
            ["snapshot_date"],
        ).withColumn("snapshot_date", F.to_date("snapshot_date"))
        orders = completed_orders.withColumn(
            "_purchase_date",
            F.to_date("order_purchase_timestamp"),
        ).select(
            "order_id",
            "customer_unique_id",
            "order_purchase_timestamp",
            "_purchase_date",
            "order_revenue",
            "order_delivered_customer_date",
            "is_late",
        )

        past = orders.crossJoin(snapshots).filter(
            F.col("_purchase_date") <= F.col("snapshot_date")
        )
        features = past.groupBy("snapshot_date", "customer_unique_id").agg(
            F.countDistinct("order_id").alias("total_orders"),
            F.sum("order_revenue").alias("total_revenue"),
            F.avg("order_revenue").alias("avg_order_value"),
            F.min("order_purchase_timestamp").alias("first_purchase_timestamp"),
            F.max("order_purchase_timestamp").alias("last_purchase_timestamp"),
            F.datediff(
                F.col("snapshot_date"),
                F.max("order_purchase_timestamp"),
            ).alias("recency_days"),
            F.datediff(
                F.max("order_purchase_timestamp"),
                F.min("order_purchase_timestamp"),
            ).alias("customer_age_days"),
        )

        review_features = (
            past.select("snapshot_date", "order_id", "customer_unique_id")
            .join(reviews, "order_id", "left")
            .filter(
                F.col("review_creation_date").isNull()
                | (F.to_date("review_creation_date") <= F.col("snapshot_date"))
            )
            .groupBy("snapshot_date", "customer_unique_id")
            .agg(
                F.avg(F.col("review_score").cast("double")).alias(
                    "avg_review_score"
                ),
                F.max("review_creation_date").alias("last_review_creation_date"),
            )
        )

        delivery_features = (
            past.filter(
                F.to_date("order_delivered_customer_date")
                <= F.col("snapshot_date")
            )
            .groupBy("snapshot_date", "customer_unique_id")
            .agg(
                F.avg("is_late").alias("late_delivery_rate"),
                F.max("order_delivered_customer_date").alias(
                    "last_delivery_date"
                ),
            )
        )

        future = (
            orders.crossJoin(snapshots)
            .filter(F.col("_purchase_date") > F.col("snapshot_date"))
            .filter(
                F.col("_purchase_date")
                <= F.date_add(F.col("snapshot_date"), _CHURN_HORIZON_DAYS)
            )
            .select("snapshot_date", "customer_unique_id")
            .distinct()
            .withColumn("_has_future_purchase", F.lit(1))
        )

        churn = (
            features.join(
                review_features,
                ["snapshot_date", "customer_unique_id"],
                "left",
            )
            .join(
                delivery_features,
                ["snapshot_date", "customer_unique_id"],
                "left",
            )
            .join(future, ["snapshot_date", "customer_unique_id"], "left")
            .withColumn(
                "is_churned",
                F.when(F.col("_has_future_purchase").isNull(), 1).otherwise(0),
            )
            .drop("_has_future_purchase")
            .fillna(0, ["avg_review_score", "late_delivery_rate"])
            .withColumn(
                "purchase_frequency_30d",
                F.col("total_orders")
                / (F.col("customer_age_days") + F.lit(1))
                * F.lit(30),
            )
            .withColumn("log_recency", F.log1p(F.col("recency_days")))
            .withColumn("log_total_revenue", F.log1p(F.col("total_revenue")))
            .withColumn(
                "revenue_per_order",
                F.col("total_revenue") / F.col("total_orders"),
            )
        )
        return churn

    def build_demand_history(self) -> DataFrame:
        """Write observed monthly demand history for compatibility."""
        history = self._read_silver("product_demand").select(
            "category_name_english",
            "order_year_month",
            "monthly_units",
            "monthly_revenue",
            "avg_price",
        )
        history.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).partitionBy("category_name_english").save(
            f"{self.gold_path}/demand_history"
        )
        self.logger.info("Wrote gold/demand_history")
        return history

    def build_demand_forecast_features(self) -> DataFrame:
        """Build leakage-safe monthly demand forecasting features.

        ``forecast_month`` is the month being predicted and ``target_units`` is
        that month's observed demand. Each feature is available by the end of
        the prior month: ``lag_1`` is the prior month's units and
        ``rolling_mean_3`` averages forecast_month -3 through -1.
        """
        demand = self._read_silver("product_demand").select(
            "category_name_english",
            "order_year_month",
            "monthly_units",
            "monthly_revenue",
            "avg_price",
        )
        win = Window.partitionBy("category_name_english").orderBy(
            "order_year_month"
        )
        roll_win = win.rowsBetween(-2, 0)
        features = (
            demand.withColumn(
                "feature_cutoff_month",
                F.col("order_year_month"),
            )
            .withColumn(
                "_feature_month_date",
                F.to_date(F.concat(F.col("order_year_month"), F.lit("-01"))),
            )
            .withColumn(
                "forecast_month",
                F.date_format(F.add_months("_feature_month_date", 1), "yyyy-MM"),
            )
            .withColumn(
                "target_units",
                F.lead("monthly_units", 1).over(win),
            )
            .withColumn("lag_1", F.col("monthly_units"))
            .withColumn("lag_2", F.lag("monthly_units", 1).over(win))
            .withColumn("lag_4", F.lag("monthly_units", 3).over(win))
            .withColumn(
                "rolling_mean_3",
                F.avg("monthly_units").over(roll_win),
            )
            .withColumn(
                "month_num",
                F.month(F.to_date(F.concat(F.col("forecast_month"), F.lit("-01")))),
            )
            .withColumn(
                "is_q4",
                F.when(F.col("month_num") >= 10, 1).otherwise(0),
            )
            .filter(F.col("lag_4").isNotNull())
            .filter(F.col("target_units").isNotNull())
            .drop("_feature_month_date", "order_year_month")
        )
        features.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).partitionBy("category_name_english").save(
            f"{self.gold_path}/demand_features"
        )
        self.logger.info("Wrote gold/demand_features")
        return features

    def build_rfm_segments(self) -> DataFrame:
        """Build canonical rule-based RFM segments with correct directions."""
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
        rfm.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).save(
            f"{self.gold_path}/rfm_segments"
        )
        distribution = {
            row["segment_label"]: row["count"]
            for row in rfm.groupBy("segment_label").count().collect()
        }
        self.logger.info("Wrote gold/rfm_segments distribution={}", distribution)
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
            F.avg("is_late").alias("late_delivery_rate"),
        )
        bi.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).partitionBy("order_year").save(f"{self.gold_path}/bi_revenue")
        self.logger.info("Wrote gold/bi_revenue")
        return bi

    def run_pipeline(self) -> dict:
        """Run independent Gold build steps and return structured results."""
        start = time.time()
        steps = [
            ("demand_features", self.build_demand_forecast_features),
            ("churn_features", self.build_churn_features),
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
