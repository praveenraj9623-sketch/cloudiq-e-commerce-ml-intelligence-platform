# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = "workspace"
BRONZE_SCHEMA = "cloudiq_bronze"
SILVER_SCHEMA = "cloudiq_silver"

spark.sql(f"""
CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SILVER_SCHEMA}
COMMENT 'CloudIQ Silver layer: cleaned and conformed Olist business marts'
""")

print(f"PASS: Silver schema ready: {CATALOG}.{SILVER_SCHEMA}")

# COMMAND ----------

def read_bronze(table_name: str):
    df = spark.table(f"{CATALOG}.{BRONZE_SCHEMA}.{table_name}")
    metadata_columns = [column for column in df.columns if column.startswith("_")]
    return df.drop(*metadata_columns)


def write_silver_table(df, table_name: str, partition_cols=None):
    full_name = f"{CATALOG}.{SILVER_SCHEMA}.{table_name}"

    writer = (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
    )

    if partition_cols:
        writer = writer.partitionBy(*partition_cols)

    writer.saveAsTable(full_name)
    return spark.table(full_name)

# COMMAND ----------

orders = read_bronze("orders")

clean_orders = (
    orders
    .withColumn(
        "order_approved_at",
        F.coalesce(F.col("order_approved_at"), F.col("order_purchase_timestamp"))
    )
    .withColumn(
        "is_canceled",
        F.when(F.col("order_status") == "canceled", F.lit(1)).otherwise(F.lit(0))
    )
    .withColumn("order_year", F.year("order_purchase_timestamp"))
    .withColumn("order_month", F.month("order_purchase_timestamp"))
    .withColumn(
        "order_year_month",
        F.date_format("order_purchase_timestamp", "yyyy-MM")
    )
    .withColumn(
        "delivery_days",
        F.when(
            F.col("order_delivered_customer_date").isNotNull()
            & F.col("order_purchase_timestamp").isNotNull(),
            F.datediff(
                F.col("order_delivered_customer_date"),
                F.col("order_purchase_timestamp"),
            ),
        )
    )
    .withColumn(
        "is_late",
        F.when(
            F.col("order_delivered_customer_date").isNotNull()
            & F.col("order_estimated_delivery_date").isNotNull()
            & (F.col("order_status") == "delivered"),
            F.when(
                F.col("order_delivered_customer_date")
                > F.col("order_estimated_delivery_date"),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
    )
    .withColumn(
        "delivery_delay_days",
        F.when(
            F.col("is_late") == 1,
            F.datediff(
                F.col("order_delivered_customer_date"),
                F.col("order_estimated_delivery_date"),
            ),
        ).when(F.col("is_late") == 0, F.lit(0))
    )
)

silver_orders = write_silver_table(
    clean_orders,
    "orders",
    partition_cols=["order_year_month"],
)

order_items = read_bronze("order_items")

clean_order_items = (
    order_items
    .filter(F.col("price") > 0)
    .withColumn("total_item_value", F.col("price") + F.col("freight_value"))
    .withColumn(
        "freight_ratio",
        F.when(
            F.col("total_item_value") > 0,
            F.col("freight_value") / F.col("total_item_value"),
        ).otherwise(F.lit(0.0)),
    )
)

silver_order_items = write_silver_table(clean_order_items, "order_items")

print("PASS: Silver orders and order_items written.")

# COMMAND ----------

    items_agg = (
    silver_order_items
    .groupBy("order_id")
    .agg(
        F.sum("price").alias("order_revenue"),
        F.sum("freight_value").alias("freight_total"),
        F.count("order_item_id").alias("item_count"),
    )
)

payments = read_bronze("payments")

payment_totals = (
    payments
    .groupBy("order_id")
    .agg(F.sum("payment_value").alias("payment_total"))
)

payment_window = (
    Window.partitionBy("order_id")
    .orderBy(
        F.col("payment_value").desc(),
        F.col("payment_sequential").asc(),
        F.col("payment_type").asc(),
    )
)

primary_payment = (
    payments
    .withColumn("_payment_rank", F.row_number().over(payment_window))
    .filter(F.col("_payment_rank") == 1)
    .select("order_id", F.col("payment_type").alias("primary_payment_type"))
)

payments_agg = payment_totals.join(primary_payment, "order_id", "left")

reviews_agg = (
    read_bronze("reviews")
    .groupBy("order_id")
    .agg(F.avg(F.col("review_score").cast("double")).alias("avg_review_score"))
)

customers = (
    read_bronze("customers")
    .select(
        "customer_id",
        "customer_unique_id",
        "customer_city",
        "customer_state",
    )
)

master_orders = (
    silver_orders.alias("o")
    .join(items_agg.alias("i"), "order_id", "left")
    .join(payments_agg.alias("p"), "order_id", "left")
    .join(reviews_agg.alias("r"), "order_id", "left")
    .join(customers.alias("c"), "customer_id", "left")
    .withColumn("order_revenue", F.coalesce(F.col("order_revenue"), F.lit(0.0)))
    .withColumn("freight_total", F.coalesce(F.col("freight_total"), F.lit(0.0)))
    .withColumn("item_count", F.coalesce(F.col("item_count"), F.lit(0)))
    .withColumn("payment_total", F.coalesce(F.col("payment_total"), F.lit(0.0)))
    .withColumn("avg_review_score", F.coalesce(F.col("avg_review_score"), F.lit(0.0)))
    .withColumn(
        "revenue_per_item",
        F.when(
            F.col("item_count") > 0,
            F.col("order_revenue") / F.col("item_count"),
        ).otherwise(F.lit(0.0)),
    )
    .withColumn(
        "primary_payment_type",
        F.coalesce(F.col("primary_payment_type"), F.lit("unknown")),
    )
    .withColumn(
        "customer_city",
        F.coalesce(F.col("customer_city"), F.lit("unknown")),
    )
    .withColumn(
        "customer_state",
        F.coalesce(F.col("customer_state"), F.lit("unknown")),
    )
    .dropDuplicates(["order_id"])
)

silver_master_orders = write_silver_table(
    master_orders,
    "master_orders",
    partition_cols=["order_year_month"],
)

print("PASS: Silver master_orders written.")

# COMMAND ----------

reference_date = F.lit("2018-10-17").cast("date")

customer_profile = (
    silver_master_orders
    .groupBy("customer_unique_id")
    .agg(
        F.min("order_purchase_timestamp").alias("first_purchase"),
        F.max("order_purchase_timestamp").alias("last_purchase"),
        F.countDistinct("order_id").alias("total_orders"),
        F.sum("order_revenue").alias("total_revenue"),
        F.avg("order_revenue").alias("avg_order_value"),
        F.avg("avg_review_score").alias("avg_review_score"),
        F.avg("is_late").alias("late_delivery_rate"),
        F.min_by("primary_payment_type", "order_purchase_timestamp").alias(
            "preferred_payment"
        ),
        F.min_by("customer_state", "order_purchase_timestamp").alias(
            "customer_state"
        ),
    )
    .withColumn(
        "recency_days",
        F.datediff(reference_date, F.to_date("last_purchase")),
    )
    .withColumn(
        "customer_age_days",
        F.datediff(F.to_date("last_purchase"), F.to_date("first_purchase")),
    )
    .withColumn(
        "purchase_frequency_score",
        F.col("total_orders") / (F.col("customer_age_days") + F.lit(1)) * F.lit(30),
    )
)

silver_customer_profile = write_silver_table(
    customer_profile,
    "customer_profile",
)

products = read_bronze("products")
translations = read_bronze("category_translation")

delivered_orders = (
    silver_orders
    .filter(
        (F.col("order_status") == "delivered")
        & F.col("order_purchase_timestamp").isNotNull()
    )
    .select("order_id", "order_year_month")
)

category_name = F.coalesce(
    F.col("t.product_category_name_english"),
    F.when(
        F.col("p.product_category_name").isNotNull(),
        F.concat(
            F.lit("untranslated__"),
            F.col("p.product_category_name"),
        ),
    ),
    F.lit("unknown"),
)

product_demand_base = (
    silver_order_items.alias("i")
    .join(delivered_orders.alias("o"), "order_id", "inner")
    .join(products.alias("p"), "product_id", "left")
    .join(
        translations.alias("t"),
        F.col("p.product_category_name") == F.col("t.product_category_name"),
        "left",
    )
    .select(
        category_name.alias("category_name_english"),
        F.col("o.order_year_month"),
        F.col("i.order_item_id"),
        F.col("i.price"),
    )
)

product_demand_window = (
    Window.partitionBy("category_name_english")
    .orderBy("order_year_month")
)

product_demand = (
    product_demand_base
    .groupBy("category_name_english", "order_year_month")
    .agg(
        F.count("order_item_id").alias("monthly_units"),
        F.sum("price").alias("monthly_revenue"),
        F.avg("price").alias("avg_price"),
    )
    .withColumn("prev_month_units", F.lag("monthly_units", 1).over(product_demand_window))
    .withColumn(
        "mom_growth_pct",
        F.when(
            F.col("prev_month_units") > 0,
            (
                (F.col("monthly_units") - F.col("prev_month_units"))
                / F.col("prev_month_units")
            )
            * F.lit(100.0),
        ).otherwise(F.lit(0.0)),
    )
)

silver_product_demand = write_silver_table(
    product_demand,
    "product_demand",
)

sellers = read_bronze("sellers")

seller_order_values = (
    silver_order_items.alias("i")
    .join(sellers.alias("s"), "seller_id", "left")
    .join(
        silver_master_orders.select(
            "order_id",
            "avg_review_score",
            "is_late",
        ).alias("m"),
        "order_id",
        "left",
    )
    .groupBy("seller_id", "seller_state", "order_id")
    .agg(
        F.sum("total_item_value").alias("seller_attributed_order_value"),
        F.max("avg_review_score").alias("avg_review_score"),
        F.max("is_late").alias("is_late"),
    )
)

seller_performance = (
    seller_order_values
    .groupBy("seller_id", "seller_state")
    .agg(
        F.countDistinct("order_id").alias("total_orders"),
        F.sum("seller_attributed_order_value").alias("total_revenue"),
        F.avg("avg_review_score").alias("avg_review"),
        F.avg("is_late").alias("late_rate"),
    )
    .withColumn(
        "performance_tier",
        F.when(F.col("total_revenue") > 10_000, F.lit("high"))
        .when(F.col("total_revenue") >= 1_000, F.lit("medium"))
        .otherwise(F.lit("low")),
    )
)

silver_seller_performance = write_silver_table(
    seller_performance,
    "seller_performance",
)

print("PASS: Silver customer, demand, and seller marts written.")

# COMMAND ----------

silver_checks = [
    ("orders", 99_441),
    ("order_items", 112_650),
    ("master_orders", 99_441),
    ("customer_profile", 96_096),
    ("product_demand", 1_273),
    ("seller_performance", 3_095),
]

results = []

for table_name, expected_rows in silver_checks:
    actual_rows = spark.table(
        f"{CATALOG}.{SILVER_SCHEMA}.{table_name}"
    ).count()

    results.append({
        "table_name": table_name,
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "status": "PASS" if actual_rows == expected_rows else "CHECK",
    })

silver_results_df = spark.createDataFrame(results).orderBy("table_name")
display(silver_results_df)

master = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.master_orders")

assert master.count() == master.select("order_id").distinct().count(), (
    "master_orders contains duplicate order_id values."
)

assert (
    spark.table(f"{CATALOG}.{SILVER_SCHEMA}.product_demand")
    .filter(F.col("category_name_english").isNull())
    .count()
    == 0
), "product_demand has null category names."

print("PASS: Silver validation completed.")

# COMMAND ----------

