# Databricks notebook source
CATALOG = "workspace"
SCHEMA = "default"
VOLUME = "cloudiq_raw"

spark.sql(f"""
CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}
COMMENT 'CloudIQ raw Olist CSV source files'
""")

print(f"PASS: Volume ready at /Volumes/{CATALOG}/{SCHEMA}/{VOLUME}")

# COMMAND ----------

RAW_PATH = "/Volumes/workspace/default/cloudiq_raw"

files = dbutils.fs.ls(RAW_PATH)

for file in files:
    print(file.name, file.size)

# COMMAND ----------

from pyspark.sql import functions as F

RAW_PATH = "/Volumes/workspace/default/cloudiq_raw"

expected_rows = {
    "olist_customers_dataset.csv": 99_441,
    "olist_geolocation_dataset.csv": 1_000_163,
    "olist_order_items_dataset.csv": 112_650,
    "olist_order_payments_dataset.csv": 103_886,
    "olist_order_reviews_dataset.csv": 99_224,
    "olist_orders_dataset.csv": 99_441,
    "olist_products_dataset.csv": 32_951,
    "olist_sellers_dataset.csv": 3_095,
    "product_category_name_translation.csv": 71,
}

results = []

for file_name, expected_count in expected_rows.items():
    reader = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
    )

    # Reviews contain quoted multiline text; these options are mandatory.
    if file_name == "olist_order_reviews_dataset.csv":
        reader = (
            reader
            .option("multiLine", "true")
            .option("quote", '"')
            .option("escape", '"')
            .option("mode", "FAILFAST")
        )

    actual_count = reader.csv(f"{RAW_PATH}/{file_name}").count()

    results.append({
        "file_name": file_name,
        "expected_rows": expected_count,
        "actual_rows": actual_count,
        "status": "PASS" if actual_count == expected_count else "FAIL",
    })

verification_df = spark.createDataFrame(results).orderBy("file_name")

display(verification_df)

failed_count = verification_df.filter(F.col("status") == "FAIL").count()

if failed_count == 0:
    print("PASS: All 9 CloudIQ Olist CSV files match the validated local row counts.")
else:
    raise ValueError(f"FAILED: {failed_count} file(s) have unexpected row counts.")

# COMMAND ----------

z   