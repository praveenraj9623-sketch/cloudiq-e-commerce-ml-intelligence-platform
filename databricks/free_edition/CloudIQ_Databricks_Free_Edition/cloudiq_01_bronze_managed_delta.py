# Databricks notebook source
from pyspark.sql import functions as F
from uuid import uuid4

CATALOG = "workspace"
RAW_SCHEMA = "default"
BRONZE_SCHEMA = "cloudiq_bronze"
RAW_PATH = f"/Volumes/{CATALOG}/{RAW_SCHEMA}/cloudiq_raw"
BATCH_ID = str(uuid4())

spark.sql(f"""
CREATE SCHEMA IF NOT EXISTS {CATALOG}.{BRONZE_SCHEMA}
COMMENT 'CloudIQ Bronze layer: validated raw Olist records stored as managed Delta tables'
""")

print(f"PASS: Bronze schema ready: {CATALOG}.{BRONZE_SCHEMA}")

# COMMAND ----------

def read_olist_csv(file_name: str, multiline: bool = False):
    reader = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
    )

    if multiline:
        reader = (
            reader
            .option("multiLine", "true")
            .option("quote", '"')
            .option("escape", '"')
            .option("mode", "FAILFAST")
        )

    return reader.csv(f"{RAW_PATH}/{file_name}")


def add_bronze_metadata(df, source_file: str):
    source_columns = df.columns
    hash_inputs = [
        F.coalesce(F.col(column).cast("string"), F.lit("<NULL>"))
        for column in source_columns
    ]

    return (
        df.withColumn("_source_file", F.lit(source_file))
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_batch_id", F.lit(BATCH_ID))
        .withColumn("_row_hash", F.md5(F.concat_ws("||", *hash_inputs)))
    )


def write_bronze_table(df, table_name: str) -> int:
    full_name = f"{CATALOG}.{BRONZE_SCHEMA}.{table_name}"

    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(full_name)
    )

    return spark.table(full_name).count()

# COMMAND ----------

# 1. Orders
orders = read_olist_csv("olist_orders_dataset.csv")
order_timestamp_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

for column in order_timestamp_columns:
    orders = orders.withColumn(
        column,
        F.to_timestamp(F.col(column), "yyyy-MM-dd HH:mm:ss")
    )

# 2. Customers
customers = (
    read_olist_csv("olist_customers_dataset.csv")
    .withColumn("customer_zip_code_prefix", F.col("customer_zip_code_prefix").cast("int"))
)

# 3. Order items
order_items = (
    read_olist_csv("olist_order_items_dataset.csv")
    .withColumn(
        "shipping_limit_date",
        F.to_timestamp(F.col("shipping_limit_date"), "yyyy-MM-dd HH:mm:ss")
    )
    .withColumn("order_item_id", F.col("order_item_id").cast("int"))
    .withColumn("price", F.col("price").cast("double"))
    .withColumn("freight_value", F.col("freight_value").cast("double"))
    .withColumn("shipping_year", F.year("shipping_limit_date"))
)

# 4. Products
products = read_olist_csv("olist_products_dataset.csv")

for column in [
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
]:
    products = products.withColumn(column, F.col(column).cast("int"))

for column in [
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
]:
    products = products.withColumn(column, F.col(column).cast("double"))

# 5. Reviews — multiline parsing is mandatory
reviews = (
    read_olist_csv("olist_order_reviews_dataset.csv", multiline=True)
    .withColumn("review_score", F.col("review_score").cast("int"))
    .withColumn(
        "review_creation_date",
        F.to_timestamp(F.col("review_creation_date"), "yyyy-MM-dd HH:mm:ss")
    )
    .withColumn(
        "review_answer_timestamp",
        F.to_timestamp(F.col("review_answer_timestamp"), "yyyy-MM-dd HH:mm:ss")
    )
)

# 6. Payments
payments = (
    read_olist_csv("olist_order_payments_dataset.csv")
    .withColumn("payment_sequential", F.col("payment_sequential").cast("int"))
    .withColumn("payment_installments", F.col("payment_installments").cast("int"))
    .withColumn("payment_value", F.col("payment_value").cast("double"))
)

# 7. Sellers
sellers = (
    read_olist_csv("olist_sellers_dataset.csv")
    .withColumn("seller_zip_code_prefix", F.col("seller_zip_code_prefix").cast("int"))
)

# 8. Geolocation
geolocation = (
    read_olist_csv("olist_geolocation_dataset.csv")
    .withColumn(
        "geolocation_zip_code_prefix",
        F.col("geolocation_zip_code_prefix").cast("int")
    )
    .withColumn("geolocation_lat", F.col("geolocation_lat").cast("double"))
    .withColumn("geolocation_lng", F.col("geolocation_lng").cast("double"))
)

# 9. Portuguese-to-English product-category translation
category_translation = read_olist_csv(
    "product_category_name_translation.csv"
)

bronze_inputs = [
    ("orders", orders, "olist_orders_dataset.csv", 99_441),
    ("customers", customers, "olist_customers_dataset.csv", 99_441),
    ("order_items", order_items, "olist_order_items_dataset.csv", 112_650),
    ("products", products, "olist_products_dataset.csv", 32_951),
    ("reviews", reviews, "olist_order_reviews_dataset.csv", 99_224),
    ("payments", payments, "olist_order_payments_dataset.csv", 103_886),
    ("sellers", sellers, "olist_sellers_dataset.csv", 3_095),
    ("geolocation", geolocation, "olist_geolocation_dataset.csv", 1_000_163),
    ("category_translation", category_translation, "product_category_name_translation.csv", 71),
]

results = []

for table_name, dataframe, source_file, expected_rows in bronze_inputs:
    bronze_df = add_bronze_metadata(dataframe, source_file)
    actual_rows = write_bronze_table(bronze_df, table_name)

    results.append({
        "table_name": table_name,
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "status": "PASS" if actual_rows == expected_rows else "FAIL",
    })

print("Bronze ingestion complete.")

# COMMAND ----------

bronze_results_df = (
    spark.createDataFrame(results)
    .orderBy("table_name")
)

display(bronze_results_df)

failed_tables = bronze_results_df.filter(F.col("status") == "FAIL").count()

assert failed_tables == 0, (
    f"Bronze reconciliation failed: {failed_tables} table(s) "
    "do not match audited raw row counts."
)

print("PASS: All 9 managed Bronze Delta tables match audited raw source counts.")

display(
    spark.sql(f"""
    SHOW TABLES IN {CATALOG}.{BRONZE_SCHEMA}
    """)
)

# COMMAND ----------

