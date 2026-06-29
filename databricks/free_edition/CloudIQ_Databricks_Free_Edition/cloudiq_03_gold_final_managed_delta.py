# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = "workspace"
SILVER_SCHEMA = "cloudiq_silver"
GOLD_SCHEMA = "cloudiq_gold"

spark.sql(f"""
CREATE SCHEMA IF NOT EXISTS {CATALOG}.{GOLD_SCHEMA}
COMMENT 'CloudIQ Gold layer: ML-ready forecasting, churn, RFM, and BI analytics'
""")

print(f"PASS: Gold schema ready: {CATALOG}.{GOLD_SCHEMA}")

# COMMAND ----------

def read_silver(table_name: str):
    return spark.table(f"{CATALOG}.{SILVER_SCHEMA}.{table_name}")


def write_gold_table(df, table_name: str):
    full_table_name = f"{CATALOG}.{GOLD_SCHEMA}.{table_name}"

    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(full_table_name)
    )

    return spark.table(full_table_name)

# COMMAND ----------

# Forecast contract:
# forecast_month = month being predicted
# target_units = observed units in forecast_month
# lag features use only data available by the end of the prior month.
#
# A category-month spine is created so lag_1 means prior CALENDAR month,
# not simply the prior available row for a category.

product_demand = (
    read_silver("product_demand")
    .filter(F.col("category_name_english").isNotNull())
    .withColumn(
        "forecast_month",
        F.to_date(
            F.concat_ws("-", F.col("order_year_month"), F.lit("01"))
        ),
    )
)

master_orders = read_silver("master_orders")

last_observed_month = (
    product_demand
    .agg(F.max("forecast_month").alias("last_observed_month"))
    .first()["last_observed_month"]
)

if last_observed_month is None:
    raise ValueError("No observed product-demand month was found.")

category_bounds = (
    product_demand
    .groupBy("category_name_english")
    .agg(
        F.min("forecast_month").alias("first_observed_month"),
    )
)

category_month_spine = (
    category_bounds
    .withColumn(
        "forecast_month",
        F.explode(
            F.sequence(
                F.col("first_observed_month"),
                F.lit(last_observed_month),
                F.expr("INTERVAL 1 MONTH"),
            )
        ),
    )
    .select("category_name_english", "forecast_month")
)

observed_monthly_demand = (
    product_demand
    .select(
        "category_name_english",
        "forecast_month",
        F.col("monthly_units").cast("double").alias("target_units"),
        F.col("monthly_revenue").cast("double").alias("monthly_revenue"),
        F.col("avg_price").cast("double").alias("avg_price"),
    )
)

monthly_demand = (
    category_month_spine
    .join(
        observed_monthly_demand,
        ["category_name_english", "forecast_month"],
        "left",
    )
    .withColumn(
        "target_units",
        F.coalesce(F.col("target_units"), F.lit(0.0)),
    )
    .withColumn(
        "monthly_revenue",
        F.coalesce(F.col("monthly_revenue"), F.lit(0.0)),
    )
    .withColumn(
        "avg_price",
        F.coalesce(F.col("avg_price"), F.lit(0.0)),
    )
    .withColumn(
        "order_year_month",
        F.date_format(F.col("forecast_month"), "yyyy-MM"),
    )
)

# Coverage is based on delivered orders because product demand uses delivered orders.
coverage_base = (
    master_orders
    .filter(
        (F.col("order_status") == "delivered")
        & F.col("order_purchase_timestamp").isNotNull()
    )
    .groupBy("order_year_month")
    .agg(
        F.countDistinct("order_id").alias("order_count"),
    )
    .withColumn(
        "period_date",
        F.to_date(
            F.concat_ws("-", F.col("order_year_month"), F.lit("01"))
        ),
    )
)

coverage_limits = coverage_base.agg(
    F.min("period_date").alias("first_period"),
    F.max("period_date").alias("last_period"),
).first()

first_period = coverage_limits["first_period"]
last_period = coverage_limits["last_period"]

coverage_flags = (
    coverage_base
    .withColumn(
        "is_partial_period",
        (F.col("period_date") == F.lit(first_period))
        | (F.col("period_date") == F.lit(last_period)),
    )
    .withColumn(
        "is_low_volume_period",
        F.col("order_count") < F.lit(100),
    )
    .select(
        "order_year_month",
        "order_count",
        "is_partial_period",
        "is_low_volume_period",
    )
)

demand_window = (
    Window
    .partitionBy("category_name_english")
    .orderBy("forecast_month")
)

demand_features = (
    monthly_demand
    .join(coverage_flags, "order_year_month", "left")
    .withColumn(
        "order_count",
        F.coalesce(F.col("order_count"), F.lit(0)),
    )
    .withColumn(
        "is_partial_period",
        F.coalesce(F.col("is_partial_period"), F.lit(True)),
    )
    .withColumn(
        "is_low_volume_period",
        F.coalesce(F.col("is_low_volume_period"), F.lit(True)),
    )
    .withColumn(
        "target_period_comparable",
        ~(
            F.col("is_partial_period")
            | F.col("is_low_volume_period")
        ),
    )
    .withColumn(
        "lag_1_source_month",
        F.lag("forecast_month", 1).over(demand_window),
    )
    .withColumn(
        "lag_2_source_month",
        F.lag("forecast_month", 2).over(demand_window),
    )
    .withColumn(
        "lag_4_source_month",
        F.lag("forecast_month", 4).over(demand_window),
    )
    .withColumn(
        "lag_1",
        F.lag("target_units", 1).over(demand_window),
    )
    .withColumn(
        "lag_2",
        F.lag("target_units", 2).over(demand_window),
    )
    .withColumn(
        "lag_4",
        F.lag("target_units", 4).over(demand_window),
    )
    .withColumn(
        "rolling_mean_3",
        F.avg("target_units").over(
            demand_window.rowsBetween(-3, -1)
        ),
    )
    .withColumn(
        "month_num",
        F.month("forecast_month"),
    )
    .withColumn(
        "is_q4",
        F.when(F.col("month_num") >= 10, F.lit(1))
        .otherwise(F.lit(0)),
    )
    .filter(
        F.col("lag_1").isNotNull()
        & F.col("lag_2").isNotNull()
        & F.col("lag_4").isNotNull()
        & F.col("rolling_mean_3").isNotNull()
    )
    .select(
        "category_name_english",
        "forecast_month",
        "order_year_month",
        "target_units",
        "lag_1",
        "lag_2",
        "lag_4",
        "rolling_mean_3",
        "lag_1_source_month",
        "lag_2_source_month",
        "lag_4_source_month",
        "month_num",
        "is_q4",
        "order_count",
        "is_partial_period",
        "is_low_volume_period",
        "target_period_comparable",
    )
)

gold_demand_features = write_gold_table(
    demand_features,
    "demand_features",
)

print(
    "PASS: Gold demand_features written. "
    f"Rows: {gold_demand_features.count()}"
)

# COMMAND ----------

# Churn contract:
# Feature history includes purchases on snapshot_date.
# The future observation window starts strictly AFTER snapshot_date.
#
# Study window is kept aligned with the locally validated project:
# 2017-01-01 through 2018-05-01.

master_orders = read_silver("master_orders")

valid_orders = (
    master_orders
    .filter(
        (F.col("order_status") == "delivered")
        & F.col("order_purchase_timestamp").isNotNull()
        & F.col("customer_unique_id").isNotNull()
    )
    .select(
        "customer_unique_id",
        "order_id",
        "order_purchase_timestamp",
        "order_revenue",
        "avg_review_score",
        "is_late",
    )
    .withColumn(
        "purchase_date",
        F.to_date(F.col("order_purchase_timestamp")),
    )
)

snapshots = spark.sql("""
SELECT explode(
    sequence(
        to_date('2017-01-01'),
        to_date('2018-05-01'),
        interval 1 month
    )
) AS snapshot_date
""")

historical_orders = (
    valid_orders
    .crossJoin(snapshots)
    .filter(
        F.col("purchase_date") <= F.col("snapshot_date")
    )
)

snapshot_features = (
    historical_orders
    .groupBy("customer_unique_id", "snapshot_date")
    .agg(
        F.min("order_purchase_timestamp").alias("first_purchase"),
        F.max("order_purchase_timestamp").alias("last_purchase"),
        F.countDistinct("order_id").alias("total_orders"),
        F.sum("order_revenue").alias("total_revenue"),
        F.avg("order_revenue").alias("avg_order_value"),
        F.avg("avg_review_score").alias("avg_review_score"),
        F.avg("is_late").alias("late_delivery_rate"),
    )
)

future_purchases = (
    valid_orders
    .crossJoin(snapshots)
    .filter(
        (F.col("purchase_date") > F.col("snapshot_date"))
        & (
            F.col("purchase_date")
            <= F.date_add(F.col("snapshot_date"), 90)
        )
    )
    .select("customer_unique_id", "snapshot_date")
    .distinct()
    .withColumn("has_future_purchase", F.lit(1))
)

churn_features = (
    snapshot_features
    .join(
        future_purchases,
        ["customer_unique_id", "snapshot_date"],
        "left",
    )
    .withColumn(
        "is_churned",
        F.when(F.col("has_future_purchase").isNull(), F.lit(1))
        .otherwise(F.lit(0)),
    )
    .drop("has_future_purchase")
    .withColumn(
        "recency_days",
        F.datediff(
            F.col("snapshot_date"),
            F.to_date("last_purchase"),
        ),
    )
    .withColumn(
        "customer_age_days",
        F.datediff(
            F.to_date("last_purchase"),
            F.to_date("first_purchase"),
        ),
    )
    .withColumn(
        "purchase_frequency_score",
        F.col("total_orders")
        / (F.col("customer_age_days") + F.lit(1))
        * F.lit(30),
    )
    .withColumn(
        "revenue_per_order_norm",
        F.col("total_revenue")
        / (F.col("total_orders") + F.lit(1)),
    )
    .withColumn(
        "high_late_delivery_flag",
        F.when(F.col("late_delivery_rate") > 0.3, F.lit(1))
        .otherwise(F.lit(0)),
    )
    .withColumn(
        "log_total_revenue",
        F.log(F.col("total_revenue") + F.lit(1.0)),
    )
    .withColumn(
        "log_recency",
        F.log(F.col("recency_days") + F.lit(1.0)),
    )
    .fillna(
        0,
        subset=[
            "total_revenue",
            "avg_order_value",
            "avg_review_score",
            "late_delivery_rate",
            "recency_days",
            "customer_age_days",
            "purchase_frequency_score",
            "revenue_per_order_norm",
            "log_total_revenue",
            "log_recency",
        ],
    )
)

gold_churn_features = write_gold_table(
    churn_features,
    "churn_features",
)

print(
    "PASS: Gold churn_features written. "
    f"Rows: {gold_churn_features.count()}"
)

# COMMAND ----------

# The Olist dataset has limited repeat purchasing:
# most customers placed exactly one order.
#
# This implementation avoids arbitrary ntile tie-splitting
# for frequency. Frequency scoring is explicit and reproducible.

customer_profile = read_silver("customer_profile")

recency_window = Window.orderBy(F.col("recency_days").desc())
monetary_window = Window.orderBy(F.col("total_revenue").asc())

rfm_base = (
    customer_profile
    .withColumn(
        "_recency_percent_rank",
        F.percent_rank().over(recency_window),
    )
    .withColumn(
        "_monetary_percent_rank",
        F.percent_rank().over(monetary_window),
    )
)

rfm_segments = (
    rfm_base
    .withColumn(
        "recency_score",
        F.least(
            F.lit(5),
            (
                F.floor(F.col("_recency_percent_rank") * F.lit(5))
                + F.lit(1)
            ).cast("int"),
        ),
    )
    .withColumn(
        "frequency_score",
        F.when(F.col("total_orders") >= 3, F.lit(5))
        .when(F.col("total_orders") == 2, F.lit(4))
        .otherwise(F.lit(1)),
    )
    .withColumn(
        "monetary_score",
        F.least(
            F.lit(5),
            (
                F.floor(F.col("_monetary_percent_rank") * F.lit(5))
                + F.lit(1)
            ).cast("int"),
        ),
    )
    .withColumn(
        "repeat_customer_flag",
        F.when(F.col("total_orders") >= 2, F.lit(1))
        .otherwise(F.lit(0)),
    )
    .withColumn(
        "rfm_total",
        F.col("recency_score")
        + F.col("frequency_score")
        + F.col("monetary_score"),
    )
    .withColumn(
        "segment_label",
        F.when(
            (F.col("recency_score") >= 4)
            & (F.col("frequency_score") >= 4)
            & (F.col("monetary_score") >= 4),
            F.lit("Champion"),
        )
        .when(
            (F.col("recency_score") >= 3)
            & (F.col("monetary_score") >= 3),
            F.lit("Loyal"),
        )
        .when(
            F.col("recency_score") >= 3,
            F.lit("Potential"),
        )
        .when(
            (F.col("recency_score") <= 2)
            & (F.col("monetary_score") >= 3),
            F.lit("At Risk"),
        )
        .otherwise(F.lit("Lost")),
    )
    .drop("_recency_percent_rank", "_monetary_percent_rank")
)

gold_rfm_segments = write_gold_table(
    rfm_segments,
    "rfm_segments",
)

print(
    "PASS: Gold rfm_segments written. "
    f"Rows: {gold_rfm_segments.count()}"
)

display(
    gold_rfm_segments
    .groupBy("segment_label")
    .agg(
        F.count("*").alias("customers"),
        F.sum("repeat_customer_flag").alias("repeat_customers"),
        F.round(F.avg("total_orders"), 4).alias("avg_total_orders"),
        F.round(F.avg("total_revenue"), 2).alias("avg_total_revenue"),
        F.round(F.avg("recency_days"), 2).alias("avg_recency_days"),
    )
    .orderBy(F.desc("customers"))
)

# COMMAND ----------

# Definitions:
# Item Merchandise Value = item prices only, excludes freight.
# Late-delivery rate = late delivered orders / all delivered orders
# with valid actual and estimated delivery dates.

master_orders = read_silver("master_orders")

bi_base = (
    master_orders
    .withColumn(
        "_valid_delivery",
        F.when(
            (F.col("order_status") == "delivered")
            & F.col("order_delivered_customer_date").isNotNull()
            & F.col("order_estimated_delivery_date").isNotNull(),
            F.lit(1),
        ).otherwise(F.lit(0)),
    )
    .withColumn(
        "_late_delivery",
        F.when(
            (F.col("_valid_delivery") == 1)
            & (F.col("is_late") == 1),
            F.lit(1),
        ).otherwise(F.lit(0)),
    )
)

bi_revenue = (
    bi_base
    .groupBy(
        "order_year",
        "order_month",
        "order_year_month",
        "customer_state",
        "primary_payment_type",
    )
    .agg(
        F.countDistinct("order_id").alias("total_orders"),
        F.sum("order_revenue").alias(
            "item_merchandise_value_ex_freight"
        ),
        F.sum("_valid_delivery").alias("delivered_order_count"),
        F.sum("_late_delivery").alias("late_delivery_count"),
    )
    .withColumn(
        "avg_item_value_per_order",
        F.when(
            F.col("total_orders") > 0,
            F.col("item_merchandise_value_ex_freight")
            / F.col("total_orders"),
        ).otherwise(F.lit(0.0)),
    )
    .withColumn(
        "late_delivery_rate",
        F.when(
            F.col("delivered_order_count") > 0,
            F.col("late_delivery_count")
            / F.col("delivered_order_count"),
        ).otherwise(F.lit(None).cast("double")),
    )
)

gold_bi_revenue = write_gold_table(
    bi_revenue,
    "bi_revenue",
)

print(
    "PASS: Gold bi_revenue written. "
    f"Rows: {gold_bi_revenue.count()}"
)

# COMMAND ----------

demand = spark.table(f"{CATALOG}.{GOLD_SCHEMA}.demand_features")
churn = spark.table(f"{CATALOG}.{GOLD_SCHEMA}.churn_features")
rfm = spark.table(f"{CATALOG}.{GOLD_SCHEMA}.rfm_segments")
bi = spark.table(f"{CATALOG}.{GOLD_SCHEMA}.bi_revenue")

customer_profile = read_silver("customer_profile")
master_orders = read_silver("master_orders")

# Demand checks
assert demand.count() > 0, "demand_features is empty."

assert demand.filter(
    F.col("target_units").isNull()
    | F.col("lag_1").isNull()
    | F.col("lag_2").isNull()
    | F.col("lag_4").isNull()
    | F.col("rolling_mean_3").isNull()
).count() == 0, "Demand features contain missing required values."

assert demand.filter(
    F.months_between(
        F.col("forecast_month"),
        F.col("lag_1_source_month"),
    ) != 1
).count() == 0, "lag_1 is not a true prior-calendar-month feature."

assert demand.filter(
    F.months_between(
        F.col("forecast_month"),
        F.col("lag_2_source_month"),
    ) != 2
).count() == 0, "lag_2 is not a true two-month feature."

assert demand.filter(
    F.months_between(
        F.col("forecast_month"),
        F.col("lag_4_source_month"),
    ) != 4
).count() == 0, "lag_4 is not a true four-month feature."

assert demand.filter(
    F.col("target_period_comparable").isNull()
).count() == 0, "Demand comparability flags are missing."

# Churn checks
assert churn.count() == 453_617, (
    "Churn snapshot count differs from the approved "
    "history-through-snapshot-day contract."
)

assert churn.select("is_churned").distinct().count() == 2, (
    "Churn table must contain both label classes."
)

# RFM checks
assert rfm.count() == customer_profile.count(), (
    "RFM customer count does not match Silver customer_profile."
)

assert rfm.filter(
    (F.col("recency_score") < 1)
    | (F.col("recency_score") > 5)
    | (F.col("frequency_score") < 1)
    | (F.col("frequency_score") > 5)
    | (F.col("monetary_score") < 1)
    | (F.col("monetary_score") > 5)
    | F.col("segment_label").isNull()
).count() == 0, "RFM score or segment validation failed."

# BI checks
expected_bi_groups = (
    master_orders
    .select(
        "order_year",
        "order_month",
        "order_year_month",
        "customer_state",
        "primary_payment_type",
    )
    .distinct()
    .count()
)

assert bi.count() == expected_bi_groups, (
    "BI revenue row count does not match distinct Silver business keys."
)

assert bi.filter(
    (F.col("late_delivery_rate") < 0)
    | (F.col("late_delivery_rate") > 1)
).count() == 0, "Late-delivery rate is outside 0–1."

assert bi.filter(
    (F.col("item_merchandise_value_ex_freight") < 0)
    | (F.col("total_orders") < 0)
).count() == 0, "BI revenue contains invalid negative values."

validation_rows = [
    {
        "table_name": "demand_features",
        "actual_rows": demand.count(),
        "validation_status": "PASS",
    },
    {
        "table_name": "churn_features",
        "actual_rows": churn.count(),
        "validation_status": "PASS",
    },
    {
        "table_name": "rfm_segments",
        "actual_rows": rfm.count(),
        "validation_status": "PASS",
    },
    {
        "table_name": "bi_revenue",
        "actual_rows": bi.count(),
        "validation_status": "PASS",
    },
]

gold_validation_df = (
    spark.createDataFrame(validation_rows)
    .orderBy("table_name")
)

display(gold_validation_df)

print("PASS: Final Gold validation completed.")

display(
    rfm
    .groupBy("segment_label")
    .agg(
        F.count("*").alias("customers"),
        F.sum("repeat_customer_flag").alias("repeat_customers"),
        F.round(F.avg("total_orders"), 4).alias("avg_total_orders"),
        F.round(F.avg("total_revenue"), 2).alias("avg_total_revenue"),
        F.round(F.avg("recency_days"), 2).alias("avg_recency_days"),
    )
    .orderBy(F.desc("customers"))
)

display(
    demand
    .groupBy("target_period_comparable")
    .agg(
        F.count("*").alias("feature_rows"),
        F.countDistinct("forecast_month").alias("forecast_months"),
        F.countDistinct("category_name_english").alias("categories"),
    )
    .orderBy("target_period_comparable")
)

# COMMAND ----------

display(
    spark.sql(f"""
    SHOW TABLES IN {CATALOG}.{GOLD_SCHEMA}
    """)
)

# COMMAND ----------

