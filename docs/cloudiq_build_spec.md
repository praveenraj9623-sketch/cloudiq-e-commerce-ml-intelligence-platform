# CloudIQ — E-Commerce ML Intelligence Platform | Codex Prompts
# Rule: Paste ONE phase per fresh Codex session. Do not combine phases.
# Dataset: Brazilian E-Commerce Public Dataset (Olist)
#   Download: kaggle datasets download -d olistbr/brazilian-ecommerce --unzip -p data/raw/
#   OR: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
#   9 CSV files, ~100K orders, 2016-2018
# ─────────────────────────────────────────────────────────────────────

━━━ PHASE C1 | requirements.txt ━━━

Python 3.11. Write requirements.txt with these exact versions, grouped by section header comments:

# Data Pipeline
pyspark==3.5.0
delta-spark==3.2.0
pandas==2.2.0
numpy==1.26.4
scipy==1.12.0

# ML Models
scikit-learn==1.4.0
xgboost==2.0.3
mlflow==2.11.0
shap==0.44.0

# API
fastapi==0.110.0
uvicorn==0.27.1
pydantic==2.6.3
httpx==0.27.0
python-multipart==0.0.9

# Azure
azure-storage-blob==12.19.0
azure-identity==1.15.0
azure-mgmt-datafactory==4.0.0

# Frontend
streamlit==1.31.0
plotly==5.19.0

# Utilities
loguru==0.7.2
pyyaml==6.0.1
python-dotenv==1.0.1
tqdm==4.66.2
rich==13.7.1

# Testing
pytest==7.4.4
pytest-asyncio==0.23.5
coverage==7.4.3
ruff==0.3.0


━━━ PHASE C2 | config.yaml ━━━

Write config.yaml for CloudIQ E-Commerce ML Intelligence Platform (Python 3.11, PySpark 3.5).
All secrets use ${ENV_VAR} or ${ENV_VAR:-default} syntax.

Sections:

project:
  name: "CloudIQ"
  version: "1.0.0"
  environment: "local"

azure:
  storage_account: "${AZURE_STORAGE_ACCOUNT}"
  storage_key: "${AZURE_STORAGE_KEY}"
  connection_string: "${AZURE_STORAGE_CONNECTION_STRING}"
  containers: {raw:"cloudiq-raw", bronze:"cloudiq-bronze", silver:"cloudiq-silver", gold:"cloudiq-gold"}
  databricks: {host:"${DATABRICKS_HOST}", token:"${DATABRICKS_TOKEN}", cluster_id:"${DATABRICKS_CLUSTER_ID}"}
  adf_factory_name: "${ADF_FACTORY_NAME}"

spark:
  app_name: "CloudIQ-Pipeline"
  driver_memory: "4g"
  executor_memory: "4g"
  shuffle_partitions: 8

mlflow:
  tracking_uri: "${MLFLOW_TRACKING_URI:-mlruns}"
  experiment_name: "cloudiq_ecommerce"

models:
  demand_forecast:
    target: "weekly_orders"
    horizon: 4
    xgb_params: {n_estimators:300, max_depth:6, learning_rate:0.05, subsample:0.8, colsample_bytree:0.8}
    lag_features: [1, 2, 4, 8]
  churn:
    churn_days_threshold: 90
    target: "is_churned"
    xgb_params: {n_estimators:200, max_depth:5, learning_rate:0.1, scale_pos_weight:3}
  segmentation:
    n_clusters: 5
    random_state: 42

api:
  host: "0.0.0.0"
  port: 8000

paths:
  raw: "data/raw"
  bronze: "data/bronze"
  silver: "data/silver"
  gold: "data/gold"
  models: "models"
  reports: "reports"
  logs: "logs"

olist_files:
  orders: "olist_orders_dataset.csv"
  customers: "olist_customers_dataset.csv"
  order_items: "olist_order_items_dataset.csv"
  products: "olist_products_dataset.csv"
  reviews: "olist_order_reviews_dataset.csv"
  payments: "olist_order_payments_dataset.csv"
  sellers: "olist_sellers_dataset.csv"
  geolocation: "olist_geolocation_dataset.csv"
  category_translation: "product_category_name_translation.csv"


━━━ PHASE C3 | .env.example + src/utils/config.py + src/utils/logger.py ━━━

PROJECT: CloudIQ | Python 3.11

FILE 1: .env.example
Placeholders: AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_KEY, AZURE_STORAGE_CONNECTION_STRING, DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_CLUSTER_ID, ADF_FACTORY_NAME, MLFLOW_TRACKING_URI
Comment at top: "# Minimum for local run: none required. All Azure values optional."

FILE 2: src/utils/config.py
Class ConfigLoader:
- __init__(config_path="config.yaml", env_path=".env"): load dotenv, load YAML, resolve ${VAR} and ${VAR:-default} recursively
- get(key_path: str, default=None): dot-notation e.g. "models.churn.churn_days_threshold"
- get_path(key_path: str, create: bool = True) -> Path: returns Path, creates dir if create=True
- validate() -> bool: find unresolved ${...} → raise ValueError listing them
- _resolve_placeholders(obj): recursive resolver for str/dict/list
- _resolve_string(value: str) -> str: regex replace ${VAR:-default} and ${VAR} patterns
Full type hints, docstrings, loguru logging via get_logger("utils.config").

FILE 3: src/utils/logger.py
Function get_logger(name: str) → loguru logger with .bind(name=name)
- Console: INFO+, colorized, "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {extra[name]} | {message}"
- File: "logs/cloudiq_{time:YYYY-MM-DD}.log", DEBUG+, rotation="1 day", retention="14 days", compression="zip"
- Global _configured flag to prevent duplicate sinks
Decorator log_exceptions(func): catches, logs with traceback, re-raises.
No placeholder code.


━━━ PHASE C4 | src/utils/spark_session.py ━━━

PROJECT: CloudIQ | Python 3.11 | pyspark==3.5.0 | delta-spark==3.2.0

Already exists: src/utils/config.py (ConfigLoader), src/utils/logger.py (get_logger)

Create: src/utils/spark_session.py

Function: get_spark_session(config: ConfigLoader, app_name: str = None) -> SparkSession
- app_name = app_name or config.get("spark.app_name", "CloudIQ")
- Build SparkSession.builder with:
  .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
  .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
  .config("spark.driver.memory", config.get("spark.driver_memory", "4g"))
  .config("spark.executor.memory", config.get("spark.executor_memory", "4g"))
  .config("spark.sql.shuffle.partitions", str(config.get("spark.shuffle_partitions", 8)))
  .config("spark.sql.adaptive.enabled", "true")
  .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
- Log: "SparkSession started — app={app_name}, version={spark.version}"
- Return SparkSession
Full type hints, docstring. No placeholder code.


━━━ PHASE C5 | src/processing/bronze.py — Part 1: class + orders + customers ━━━

PROJECT: CloudIQ | pyspark==3.5.0 | delta-spark==3.2.0

Already exists: src/utils/config.py, src/utils/logger.py, src/utils/spark_session.py

Dataset column schemas (all StringType initially, cast in method):
olist_orders_dataset.csv: order_id, customer_id, order_status, order_purchase_timestamp, order_approved_at, order_delivered_carrier_date, order_delivered_customer_date, order_estimated_delivery_date
olist_customers_dataset.csv: customer_id, customer_unique_id, customer_zip_code_prefix, customer_city, customer_state

Create: src/processing/bronze.py

Class BronzeLayer:

__init__(self, spark: SparkSession, config: ConfigLoader):
- self.spark = spark, self.config = config, self.logger = get_logger("processing.bronze")
- self.raw_path = config.get_path("paths.raw", create=False)
- self.bronze_path = config.get_path("paths.bronze")

_add_metadata(self, df: DataFrame, source: str) -> DataFrame:
- Add: _source=F.lit(source), _ingested_at=F.current_timestamp(), _batch_id=F.lit(str(uuid4())), _row_hash=F.md5(F.concat_ws("|", *[F.col(c).cast("string") for c in df.columns if not c.startswith("_")]))

_write_delta(self, df: DataFrame, table: str, partition_cols: list = None) -> dict:
- Write Delta to {bronze_path}/{table}/, mode="overwrite", partitionBy if partition_cols
- Read back, count rows
- Return {table, rows: int, status: "SUCCESS"} or {table, status: "FAILED", error: str}

ingest_orders(self) -> dict:
- Read CSV from raw_path/olist_orders_dataset.csv (header=true, inferSchema=true)
- Cast all 5 timestamp columns to TimestampType using F.to_timestamp(F.col(c), "yyyy-MM-dd HH:mm:ss")
- _add_metadata(df, "olist_orders")
- _write_delta(df, "orders", partition_cols=["order_status"])

ingest_customers(self) -> dict:
- Read CSV, _add_metadata(df, "olist_customers"), _write_delta(df, "customers")

No placeholder code.


━━━ PHASE C6 | src/processing/bronze.py — Part 2: remaining 7 tables + run_pipeline ━━━

PROJECT: CloudIQ | pyspark==3.5.0 | delta-spark==3.2.0

Already exists: BronzeLayer with __init__, _add_metadata, _write_delta, ingest_orders, ingest_customers

Add these methods to BronzeLayer in src/processing/bronze.py:

olist_order_items_dataset.csv columns: order_id, order_item_id, product_id, seller_id, shipping_limit_date, price, freight_value

ingest_order_items(self) -> dict:
- Read CSV, cast: shipping_limit_date to TimestampType, price and freight_value to DoubleType
- Add derived col: shipping_year = F.year(F.col("shipping_limit_date"))
- _add_metadata(df, "olist_order_items"), _write_delta(df, "order_items", partition_cols=["shipping_year"])

ingest_products(self) -> dict: (columns: product_id, product_category_name, product_name_lenght, product_description_lenght, product_photos_qty, product_weight_g, product_length_cm, product_height_cm, product_width_cm)
- Read CSV, cast weight/dimension columns to DoubleType, _add_metadata(df, "olist_products"), _write_delta(df, "products")

ingest_reviews(self) -> dict: (columns: review_id, order_id, review_score, review_comment_title, review_comment_message, review_creation_date, review_answer_timestamp)
- Read CSV, cast review_score to IntegerType, timestamps to TimestampType
- _add_metadata(df, "olist_reviews"), _write_delta(df, "reviews", partition_cols=["review_score"])

ingest_payments(self) -> dict: (columns: order_id, payment_sequential, payment_type, payment_installments, payment_value)
- Read CSV, cast payment_installments to IntegerType, payment_value to DoubleType
- _add_metadata(df, "olist_payments"), _write_delta(df, "payments", partition_cols=["payment_type"])

ingest_sellers(self) -> dict: (columns: seller_id, seller_zip_code_prefix, seller_city, seller_state)
- Read CSV, _add_metadata(df, "olist_sellers"), _write_delta(df, "sellers")

ingest_geolocation(self) -> dict: (columns: geolocation_zip_code_prefix, geolocation_lat, geolocation_lng, geolocation_city, geolocation_state)
- Read CSV, cast lat and lng to DoubleType, _add_metadata(df, "olist_geo"), _write_delta(df, "geolocation")

ingest_category_translation(self) -> dict: (columns: product_category_name, product_category_name_english)
- Read CSV, _add_metadata(df, "olist_categories"), _write_delta(df, "category_translation")

run_pipeline(self) -> dict:
- Run all 9 ingest methods in order: orders, customers, order_items, products, reviews, payments, sellers, geolocation, category_translation
- On any exception: log error, set status="FAILED", continue with remaining
- Return dict with all 9 results + total_duration_seconds
- Print rich table: table | rows | status for each

No placeholder code.


━━━ PHASE C7 | src/processing/silver.py — clean_orders + clean_order_items ━━━

PROJECT: CloudIQ | pyspark==3.5.0 | delta-spark==3.2.0

Bronze tables in data/bronze/:
- orders: (order_id, customer_id, order_status, 5 timestamps + 4 _metadata cols)
- order_items: (order_id, order_item_id, product_id, seller_id, shipping_limit_date, price, freight_value, shipping_year, + 4 _metadata cols)

Create: src/processing/silver.py

Class SilverLayer:

__init__(self, spark, config):
- self.spark, self.config, self.logger = spark, config, get_logger("processing.silver")
- self.bronze_path = config.get_path("paths.bronze", create=False)
- self.silver_path = config.get_path("paths.silver")

_read_bronze(self, table: str) -> DataFrame:
- spark.read.format("delta").load(f"{bronze_path}/{table}/")
- Drop all columns starting with "_" (metadata)

clean_orders(self) -> DataFrame:
- _read_bronze("orders")
- Add: is_canceled = F.when(F.col("order_status")=="canceled", 1).otherwise(0)
- Add: order_year = F.year("order_purchase_timestamp")
- Add: order_month = F.month("order_purchase_timestamp")
- Add: order_year_month = F.date_format("order_purchase_timestamp", "yyyy-MM")
- Add: delivery_days = F.datediff("order_delivered_customer_date","order_purchase_timestamp")
- Add: is_late = F.when(F.col("order_delivered_customer_date") > F.col("order_estimated_delivery_date"), 1).otherwise(0)
- Add: delivery_delay_days = F.when(F.col("is_late")==1, F.datediff("order_delivered_customer_date","order_estimated_delivery_date")).otherwise(0)
- Fill null order_approved_at with order_purchase_timestamp
- Write to silver/orders/ Delta, mode="overwrite", partitionBy("order_year_month")

clean_order_items(self) -> DataFrame:
- _read_bronze("order_items")
- Add: total_item_value = F.col("price") + F.col("freight_value")
- Filter: price > 0
- Add: freight_ratio = F.when(F.col("total_item_value")>0, F.col("freight_value")/F.col("total_item_value")).otherwise(0.0)
- Write to silver/order_items/ Delta, mode="overwrite"

No placeholder code.


━━━ PHASE C8 | src/processing/silver.py — build_master_orders ━━━

PROJECT: CloudIQ | pyspark==3.5.0 | delta-spark==3.2.0

Already exists: SilverLayer with __init__, _read_bronze, clean_orders, clean_order_items
Silver tables: silver/orders, silver/order_items

Add to class SilverLayer in src/processing/silver.py:

build_master_orders(self) -> DataFrame:
This method joins 6 sources into one master fact table.

Step 1: Read silver/orders as base (already cleaned, has all new columns)

Step 2: aggregate order_items: _read_bronze("order_items")
→ group by order_id: sum(price) as order_revenue, sum(freight_value) as freight_total, count(*) as item_count

Step 3: aggregate payments: _read_bronze("payments")
→ group by order_id: sum(payment_value) as payment_total, first(payment_type) as primary_payment_type

Step 4: aggregate reviews: _read_bronze("reviews")
→ group by order_id: avg(review_score.cast("double")) as avg_review_score

Step 5: read customers: _read_bronze("customers")
→ select customer_id, customer_unique_id, customer_city, customer_state

Step 6: JOIN base orders LEFT JOIN each aggregation on order_id, LEFT JOIN customers on customer_id

Step 7: Post-join:
- Add: revenue_per_item = F.when(F.col("item_count")>0, F.col("order_revenue")/F.col("item_count")).otherwise(0.0)
- Fill numeric nulls: fill(0, ["order_revenue","freight_total","item_count","payment_total","avg_review_score","revenue_per_item","delivery_days","is_late","delivery_delay_days"])
- Fill string nulls: fill("unknown", ["primary_payment_type","customer_city","customer_state"])
- Deduplicate on order_id (dropDuplicates(["order_id"]))

Write to silver/master_orders/ Delta, mode="overwrite", partitionBy("order_year_month")
Return DataFrame. Log: input rows (from orders), output rows (master).
No placeholder code.


━━━ PHASE C9 | src/processing/silver.py — profiles + demand + seller + run ━━━

PROJECT: CloudIQ | pyspark==3.5.0 | delta-spark==3.2.0

Already exists: SilverLayer with build_master_orders, silver/master_orders Delta table

Add to class SilverLayer in src/processing/silver.py:

build_customer_profile(self) -> DataFrame:
- Read silver/master_orders
- Group by customer_unique_id:
  first_purchase = min("order_purchase_timestamp")
  last_purchase = max("order_purchase_timestamp")
  total_orders = countDistinct("order_id")
  total_revenue = sum("order_revenue")
  avg_order_value = avg("order_revenue")
  avg_review_score = avg("avg_review_score")
  late_delivery_rate = avg("is_late")
  recency_days = F.datediff(F.lit("2018-10-17"), F.max("order_purchase_timestamp"))  [dataset ends Oct 2018]
  customer_age_days = F.datediff(F.max("order_purchase_timestamp"), F.min("order_purchase_timestamp"))
  preferred_payment = first("primary_payment_type")
  customer_state = first("customer_state")
- Write to silver/customer_profile/ Delta, mode="overwrite"

build_product_demand(self) -> DataFrame:
- Join: _read_bronze("order_items") + silver/orders (for timestamps) + _read_bronze("products") + _read_bronze("category_translation") on product_category_name
- Group by (category_name_english, order_year_month):
  monthly_units = count("order_item_id"), monthly_revenue = sum("price"), avg_price = avg("price")
- Window: prev_month = LAG("monthly_units",1) OVER (PARTITION BY category_name_english ORDER BY order_year_month)
- Add: mom_growth = F.when(F.col("prev_month")>0, (F.col("monthly_units")-F.col("prev_month"))/F.col("prev_month")*100).otherwise(0.0)
- Write to silver/product_demand/ Delta, mode="overwrite", partitionBy("order_year_month")

build_seller_performance(self) -> DataFrame:
- Join: _read_bronze("order_items") + _read_bronze("sellers") + silver/master_orders
- Group by (seller_id, seller_state):
  total_orders = count("order_id"), total_revenue = sum("order_revenue"), avg_review = avg("avg_review_score"), late_rate = avg("is_late")
- Add: performance_tier = CASE total_revenue > 10000 → "high", 1000-10000 → "medium", else "low"
- Write to silver/seller_performance/ Delta, mode="overwrite", partitionBy("seller_state")

run_pipeline(self) -> dict:
- Run: clean_orders, clean_order_items, build_master_orders, build_customer_profile, build_product_demand, build_seller_performance (in that order)
- Return {results per table, total_s: float}
- On exception per step: log error, status="FAILED", continue

No placeholder code.


━━━ PHASE C10 | src/processing/gold.py — churn_features + demand_features ━━━

PROJECT: CloudIQ | pyspark==3.5.0 | delta-spark==3.2.0

Silver tables: silver/customer_profile, silver/product_demand

Create: src/processing/gold.py

Class GoldLayer:

__init__(self, spark, config):
- self.spark, self.config, self.logger = spark, config, get_logger("processing.gold")
- self.silver_path = config.get_path("paths.silver", create=False)
- self.gold_path = config.get_path("paths.gold")

_read_silver(self, table: str) -> DataFrame:
- spark.read.format("delta").load(f"{silver_path}/{table}/")

build_churn_features(self) -> DataFrame:
- Read silver/customer_profile
- Define churn: is_churned = F.when(F.col("recency_days") > 90, 1).otherwise(0)
- Feature engineering:
  log_total_revenue = F.log1p("total_revenue")
  log_recency = F.log1p("recency_days")
  purchase_frequency_score = F.col("total_orders") / (F.col("customer_age_days") + 1) * 30
  revenue_per_order_norm = F.col("total_revenue") / (F.col("total_orders") + 1)
  high_late_delivery_flag = F.when(F.col("late_delivery_rate") > 0.3, 1).otherwise(0)
- Fill all null numerics with 0
- Write to gold/churn_features/ Delta, mode="overwrite"
- Return DataFrame

build_demand_forecast_features(self) -> DataFrame:
- Read silver/product_demand
- Window spec: PARTITION BY category_name_english ORDER BY order_year_month
- Add lag features: lag_1=LAG("monthly_units",1), lag_2=LAG("monthly_units",2), lag_4=LAG("monthly_units",4)
- Add rolling: rolling_mean_3 = AVG("monthly_units") OVER (same partition, ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
- Add: month_num = F.month(F.to_date(F.col("order_year_month"), "yyyy-MM"))
- Add: is_q4 = F.when(F.col("month_num") >= 10, 1).otherwise(0)
- Drop rows where lag_4 is null (insufficient history)
- Write to gold/demand_features/ Delta, mode="overwrite", partitionBy("category_name_english")
- Return DataFrame

No placeholder code.


━━━ PHASE C11 | src/processing/gold.py — rfm_segments + bi_summary + run ━━━

PROJECT: CloudIQ | pyspark==3.5.0 | delta-spark==3.2.0

Already exists: GoldLayer with __init__, _read_silver, build_churn_features, build_demand_forecast_features

Add to class GoldLayer in src/processing/gold.py:

build_rfm_segments(self) -> DataFrame:
- Read silver/customer_profile
- Window spec: ORDER BY recency_days DESC → recency_score = F.ntile(5).over(window) [desc: lower recency = better]
- Window spec: ORDER BY total_orders ASC → frequency_score = F.ntile(5).over(window) [asc: higher freq = better]
- Window spec: ORDER BY total_revenue ASC → monetary_score = F.ntile(5).over(window)
- rfm_total = recency_score + frequency_score + monetary_score
- segment_label = CASE rfm_total >= 13 → "Champion", >= 10 → "Loyal", >= 7 → "Potential", >= 4 → "At Risk", else → "Lost"
- Write to gold/rfm_segments/ Delta, mode="overwrite"

build_bi_revenue(self) -> DataFrame:
- Read silver/master_orders
- Group by (order_year, order_month, order_year_month, customer_state, primary_payment_type):
  total_orders=count("order_id"), total_revenue=sum("order_revenue"), avg_order_value=avg("order_revenue"), return_rate=avg("is_late")
- Write to gold/bi_revenue/ Delta, mode="overwrite", partitionBy("order_year")

run_pipeline(self) -> dict:
- Run: build_churn_features, build_demand_forecast_features, build_rfm_segments, build_bi_revenue (in order)
- Return {table: {rows, status, duration_s} for each, total_s: float}
- On exception per step: log error, status="FAILED", continue

Also create standalone: run_gold.py at project root
- CLI: --config default="config.yaml"
- Load config, spark, run GoldLayer.run_pipeline()
- Print rich table: table | rows | status | duration
- spark.stop() in finally

No placeholder code.


━━━ PHASE C12 | run_pipeline.py (full pipeline runner) ━━━

PROJECT: CloudIQ | Python 3.11

Already exists: src/processing/bronze.py, silver.py, gold.py (all complete)

Create: run_pipeline.py at project root

import argparse, time, sys
from rich.console import Console
from rich.table import Table

CLI arguments:
--layer: choices=["bronze","silver","gold","all"], default="all"
--config: default="config.yaml"
--verbose: store_true flag

main():
1. Load ConfigLoader(args.config)
2. get_spark_session(config)
3. Based on --layer:
   - "bronze": run BronzeLayer.run_pipeline()
   - "silver": run SilverLayer.run_pipeline()
   - "gold": run GoldLayer.run_pipeline()
   - "all": run bronze → silver → gold in sequence
4. Collect all results
5. Build rich.Table: Layer | Table | Rows | Status | Duration(s)
6. Print table with Console()
7. If any FAILED: print "Pipeline completed with errors" and sys.exit(1)
8. Else: print "Pipeline completed successfully"

Always: spark.stop() in finally block.
if __name__ == "__main__": main()
No placeholder code.


━━━ PHASE C13 | src/models/demand_forecast.py ━━━

PROJECT: CloudIQ | Python 3.11 | xgboost==2.0.3 | mlflow==2.11.0 | pandas==2.2.0

Gold table gold/demand_features columns: category_name_english, order_year_month, monthly_units (target), lag_1, lag_2, lag_4, rolling_mean_3, month_num, is_q4

Create: src/models/demand_forecast.py

Class DemandForecastModel:

__init__(self, spark, config):
- mlflow.set_tracking_uri/experiment from config
- self.feature_cols = ["lag_1","lag_2","lag_4","rolling_mean_3","month_num","is_q4"]
- self.target = config.get("models.demand_forecast.target","weekly_orders")
- self.logger = get_logger("models.demand_forecast")

load_features(self) -> tuple[pd.DataFrame, pd.Series]:
- Read gold/demand_features Delta → .toPandas()
- Sort by category_name_english, order_year_month
- Drop rows where any feature_col is null
- X = df[self.feature_cols], y = df["monthly_units"]
- Return X, y

time_series_split(self, X, y, n_splits=3) -> list[tuple]:
- Sort by index (already sorted by year_month)
- Each split: train = rows 0 to split_point, test = next N rows
- split_point increments by len//n_splits each fold
- Return [(X_train, X_test, y_train, y_test) for each fold]

train(self, X_train, y_train, X_val, y_val, params: dict = None) -> tuple:
- params from config or use defaults
- XGBRegressor(**params, eval_metric="rmse", early_stopping_rounds=50)
- fit with eval_set=[(X_val, y_val)], verbose=False
- val_rmse = sqrt(mean_squared_error(y_val, model.predict(X_val)))
- Return (model, val_rmse)

evaluate(self, model, X_test, y_test) -> dict:
- rmse, mae, r2, mape = compute all metrics
- mape: mean(abs((y_test - y_pred) / (y_test + 1e-8))) * 100
- Return dict

run_with_mlflow(self) -> str:
- mlflow.start_run(run_name="demand_xgb_v1")
- Load features, time_series_split(n_splits=3)
- For each fold: train, evaluate, log fold metrics with step=fold_idx
- Train final model on full data
- mlflow.log_params(params), mlflow.log_metrics(avg metrics)
- mlflow.xgboost.log_model(model, "demand_model")
- mlflow.register_model(uri, "cloudiq_demand_forecast")
- Return run_id

No placeholder code.


━━━ PHASE C14 | src/models/churn.py ━━━

PROJECT: CloudIQ | Python 3.11 | xgboost==2.0.3 | mlflow==2.11.0 | shap==0.44.0

Gold table gold/churn_features: customer_unique_id, is_churned (target), log_total_revenue, log_recency, purchase_frequency_score, revenue_per_order_norm, high_late_delivery_flag, avg_review_score, late_delivery_rate, total_orders

Create: src/models/churn.py

Class ChurnModel:

__init__(self, spark, config):
- mlflow setup from config
- self.feature_cols = ["log_total_revenue","log_recency","purchase_frequency_score","revenue_per_order_norm","high_late_delivery_flag","avg_review_score","late_delivery_rate","total_orders"]
- self.logger = get_logger("models.churn")

load_features(self) -> tuple[pd.DataFrame, pd.Series]:
- Read gold/churn_features Delta → .toPandas()
- X = df[self.feature_cols], y = df["is_churned"]
- Log class distribution: churned % vs not-churned %

train(self) -> str:
- mlflow.start_run(run_name="churn_xgb_v1")
- StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- For each fold:
  XGBClassifier(params from config, use_label_encoder=False, eval_metric="auc")
  Metrics: roc_auc, f1, precision, recall
  mlflow.log_metrics with step=fold_idx
- Train final model on full data
- SHAP: explainer=shap.TreeExplainer(final_model), values=explainer.shap_values(X[:500])
- Save shap summary plot as reports/shap_churn.png
- mlflow.log_artifact("reports/shap_churn.png")
- mlflow.xgboost.log_model(final_model, "churn_model")
- mlflow.register_model(uri, "cloudiq_churn_predictor")
- Return run_id

predict(self, customer_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
- Load Production model from MLflow: mlflow.xgboost.load_model("models:/cloudiq_churn_predictor/Production")
- proba = model.predict_proba(customer_df[self.feature_cols])[:,1]
- prediction = (proba >= threshold).astype(int)
- risk_tier = pd.cut(proba, bins=[0,0.3,0.6,1.0], labels=["low","medium","high"])
- Return DataFrame: churn_probability, churn_prediction, risk_tier

No placeholder code.


━━━ PHASE C15 | src/models/segmentation.py ━━━

PROJECT: CloudIQ | Python 3.11 | scikit-learn==1.4.0 | mlflow==2.11.0

Gold table gold/rfm_segments: customer_unique_id, recency_score, frequency_score, monetary_score, rfm_total, segment_label

Create: src/models/segmentation.py

Class CustomerSegmentation:

__init__(self, spark, config):
- self.n_clusters = config.get("models.segmentation.n_clusters", 5)
- self.random_state = config.get("models.segmentation.random_state", 42)
- self.logger = get_logger("models.segmentation")
- mlflow setup from config

load_features(self) -> pd.DataFrame:
- Read gold/rfm_segments Delta → .toPandas()
- feature_cols = ["recency_score","frequency_score","monetary_score"]
- StandardScaler fit_transform on feature_cols
- Return scaled DataFrame

train(self) -> str:
- mlflow.start_run(run_name="kmeans_segmentation_v1")
- Elbow method: try n_clusters 2 to 8, compute inertia for each
- Save elbow plot as reports/kmeans_elbow.png, mlflow.log_artifact
- Silhouette scores: compute for each k, pick best k
- Train final KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
- silhouette = silhouette_score(X_scaled, labels)
- mlflow.log_metric("silhouette_score", silhouette), mlflow.log_param("n_clusters", n_clusters)
- pickle.dump model to models/kmeans_segmentation.pkl
- mlflow.log_artifact("models/kmeans_segmentation.pkl")
- Return run_id

get_cluster_profiles(self, df_labeled: pd.DataFrame) -> pd.DataFrame:
- Group by cluster_label: compute mean of each RFM score
- Map cluster to human name based on highest scoring dimension
- Return profile DataFrame: cluster | name | avg_recency | avg_frequency | avg_monetary | n_customers

No placeholder code.


━━━ PHASE C16 | src/serving/api.py ━━━

PROJECT: CloudIQ | Python 3.11 | fastapi==0.110.0 | pydantic==2.6.3 | mlflow==2.11.0

Already exists: all 3 models trained, registered in MLflow under "cloudiq_demand_forecast", "cloudiq_churn_predictor"

Create: src/serving/api.py

PYDANTIC MODELS:

class DemandRequest(BaseModel):
  category: str = Field(example="health_beauty")
  horizon_months: int = Field(default=4, ge=1, le=12)

class DemandForecast(BaseModel):
  year_month: str
  predicted_units: float
  lower_bound: float
  upper_bound: float

class DemandResponse(BaseModel):
  category: str
  forecasts: list[DemandForecast]
  model_version: str
  latency_ms: float

class ChurnRequest(BaseModel):
  customer_id: str
  total_orders: int = Field(ge=0)
  total_revenue: float = Field(ge=0)
  recency_days: int = Field(ge=0)
  avg_review_score: float = Field(ge=0, le=5)
  late_delivery_rate: float = Field(ge=0, le=1)

class ChurnResponse(BaseModel):
  customer_id: str
  churn_probability: float
  churn_prediction: int
  risk_tier: str
  model_version: str
  latency_ms: float

class SegmentRequest(BaseModel):
  customer_id: str
  recency_score: int = Field(ge=1, le=5)
  frequency_score: int = Field(ge=1, le=5)
  monetary_score: int = Field(ge=1, le=5)

class SegmentResponse(BaseModel):
  customer_id: str
  segment: str
  rfm_total: int
  recommendations: list[str]

APP:
app = FastAPI(title="CloudIQ API", version="1.0.0")
Add CORSMiddleware(allow_origins=["*"])
Add request timing middleware: log each request method/path/status/latency_ms

Singletons: demand_model, churn_model, loaded at startup via @app.on_event("startup")

ENDPOINTS:

GET /health → {status:"healthy", models:{demand:"loaded", churn:"loaded"}, timestamp:str}
GET / → {message:"CloudIQ API v1", docs:"/docs"}

POST /predict/demand:
- Build lag features from request (use median values if no history) — document this assumption in response
- Predict using demand_model, generate horizon_months forecasts
- Each forecast: lower_bound = prediction * 0.85, upper_bound = prediction * 1.15
- Return DemandResponse

POST /predict/churn:
- Build feature dict from ChurnRequest fields
- compute: log_total_revenue=log1p(total_revenue), log_recency=log1p(recency_days), purchase_frequency_score=total_orders/(max(recency_days,1)/30), revenue_per_order_norm=total_revenue/(total_orders+1), high_late_delivery_flag=1 if late_delivery_rate>0.3 else 0
- Predict proba, return ChurnResponse

POST /predict/segment:
- rfm_total = recency_score + frequency_score + monetary_score
- Apply same CASE logic as GoldLayer: Champion/Loyal/Potential/At Risk/Lost
- recommendations = [3 business action strings based on segment]
- Return SegmentResponse

GET /models → {demand:{name, version, stage}, churn:{name, version, stage}} from MLflow registry

No placeholder code.


━━━ PHASE C17 | tests/test_api.py ━━━

PROJECT: CloudIQ | Python 3.11 | pytest==7.4.4 | pytest-asyncio | httpx

Already exists: src/serving/api.py with all endpoints

Create: tests/test_api.py

10 async tests using httpx.AsyncClient(app=app, base_url="http://test"):
(Use pytest fixture to mock MLflow model loading so tests run without real models)

1. test_health_returns_200: GET /health → status 200, body["status"] == "healthy"
2. test_root_endpoint: GET / → 200, body has "message" key
3. test_demand_valid_request: POST /predict/demand {"category":"health_beauty","horizon_months":4} → 200, len(forecasts)==4, each has year_month/predicted_units/lower_bound/upper_bound
4. test_demand_invalid_horizon: POST /predict/demand {"category":"test","horizon_months":0} → 422
5. test_demand_horizon_too_large: horizon_months=15 → 422
6. test_churn_valid_request: POST /predict/churn with all valid fields → 200, churn_probability between 0 and 1, prediction in [0,1], risk_tier in ["low","medium","high"]
7. test_churn_invalid_review_score: avg_review_score=6.0 → 422
8. test_segment_valid_request: POST /predict/segment {"customer_id":"c1","recency_score":5,"frequency_score":4,"monetary_score":5} → 200, segment=="Champion", len(recommendations)==3
9. test_segment_invalid_score: recency_score=6 → 422
10. test_models_endpoint: GET /models → 200, body has "demand" and "churn" keys

Create: tests/conftest.py
Fixtures:
- mock_demand_model: MagicMock that returns predictions [10.0, 12.0, 11.0, 13.0] on predict()
- mock_churn_model: MagicMock returning predict_proba([[0.3, 0.7]])
- Override app dependencies to inject mocks
No placeholder code.


━━━ PHASE C18 | Dockerfile + docker-compose.yml + .gitignore ━━━

PROJECT: CloudIQ | Python 3.11

Create: Dockerfile
Multi-stage build:
Builder: python:3.11-slim, install build-essential, pip install requirements.txt to /install
Final: python:3.11-slim, copy from builder, WORKDIR /app, copy src/ config.yaml, mkdir -p data/bronze data/silver data/gold models reports logs, EXPOSE 8000, ENV PYTHONPATH=/app PYTHONUNBUFFERED=1, HEALTHCHECK curl -f http://localhost:8000/health || exit 1, CMD ["uvicorn","src.serving.api:app","--host","0.0.0.0","--port","8000","--workers","2"]

Create: docker-compose.yml
Services:
- api: build: ., ports 8000:8000, env_file .env, volumes: ./data:/app/data, ./models:/app/models, ./reports:/app/reports, ./logs:/app/logs, healthcheck: curl /health
- mlflow: ghcr.io/mlflow/mlflow:v2.11.0, command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlflow-artifacts, port 5000:5000, volume mlflow_data:/mlflow-artifacts
volumes: mlflow_data

Create: .gitignore
Python standard + data/raw/ + data/bronze/ + data/silver/ + data/gold/ + *.parquet + *.delta + mlruns/ + models/ + .env + spark-warehouse/ + __pycache__/ + .pytest_cache/ + .mypy_cache/ + logs/ + reports/


━━━ PHASE C19 | .github/workflows/ci_cd.yml ━━━

PROJECT: CloudIQ | Python 3.11 | GitHub Actions

Create: .github/workflows/ci_cd.yml

name: "CloudIQ CI/CD"
triggers: push to main and dev, pull_request to main

jobs:

lint:
  ubuntu-latest
  steps: checkout → python 3.11 → pip install ruff==0.3.0 → ruff check src/ tests/ --max-line-length 100

test:
  needs: lint
  ubuntu-latest
  steps: checkout → python 3.11 → pip install -r requirements.txt → set envs (MLFLOW_TRACKING_URI=mlruns) → pytest tests/ -v --tb=short --cov=src --cov-report=xml → upload coverage artifact

build-docker:
  needs: test
  only on push to main
  steps: checkout → login ghcr.io (GITHUB_TOKEN) → docker build -t ghcr.io/${{github.repository}}/cloudiq-api:${{github.sha}} . → docker push → tag as latest → push latest

deploy-summary:
  needs: build-docker
  only on main push
  steps: echo deployment summary to GitHub Step Summary showing: image tag, build time, API URL

Secrets needed: none beyond GITHUB_TOKEN (auto-provided)


━━━ PHASE C20 | azure/arm_template.json + workflows/databricks_workflow.json ━━━

PROJECT: CloudIQ

Create: azure/arm_template.json
ARM template deploying:
- Azure Data Lake Storage Gen2 account (Standard_LRS, StorageV2, hierarchicalNamespace=true), with 4 blob containers: cloudiq-raw, cloudiq-bronze, cloudiq-silver, cloudiq-gold
- Azure Container Apps environment (consumption plan)
- Azure Container App "cloudiq-api" (1 CPU, 2Gi memory, image ghcr.io/OWNER/cloudiq-api:latest, ingress external HTTPS, min_replicas=0, max_replicas=3)
- Azure Databricks Workspace (Standard tier)
Parameters: location (default eastus), projectName (default cloudiq), resourceGroupName
Outputs: apiUrl (container app FQDN), storageAccountName, databricksWorkspaceUrl

Create: workflows/databricks_workflow.json
Databricks Workflow definition (Databricks REST API 2.1 format):
{
  "name": "cloudiq_daily_pipeline",
  "schedule": {"quartz_cron_expression": "0 0 6 * * ?", "timezone_id": "Asia/Kolkata"},
  "max_concurrent_runs": 1,
  "email_notifications": {"on_failure": ["REPLACE_WITH_EMAIL"]},
  "tasks": [
    {"task_key": "bronze_ingestion", "notebook_task": {"notebook_path": "/Repos/cloudiq/notebooks/01_bronze"}, "new_cluster": {"spark_version": "14.3.x-scala2.12", "node_type_id": "Standard_DS3_v2", "num_workers": 2}},
    {"task_key": "silver_transform", "depends_on": [{"task_key": "bronze_ingestion"}], "notebook_task": {"notebook_path": "/Repos/cloudiq/notebooks/02_silver"}},
    {"task_key": "gold_layer", "depends_on": [{"task_key": "silver_transform"}], "notebook_task": {"notebook_path": "/Repos/cloudiq/notebooks/03_gold"}},
    {"task_key": "model_training", "depends_on": [{"task_key": "gold_layer"}], "python_wheel_task": {"package_name": "cloudiq", "entry_point": "run_training"}}
  ]
}


━━━ PHASE C21 | README.md ━━━

PROJECT: CloudIQ

Create: README.md with all sections complete (no placeholder text):

1. Title: "CloudIQ — E-Commerce ML Intelligence Platform" + CI/CD badge, Python/FastAPI/Azure/Databricks badges

2. One-paragraph description: end-to-end e-commerce intelligence platform using Azure cloud infrastructure, Databricks-compatible PySpark pipeline with Medallion Architecture, three ML models, and FastAPI prediction serving.

3. Architecture (ASCII diagram): Olist CSV → Azure Blob/local → Bronze Delta → Silver Delta → Gold Delta → [Demand Forecast + Churn Prediction + Segmentation] → FastAPI → Power BI

4. Tech Stack table (columns: Layer | Technology | Version):
At minimum 14 rows covering: Data ingestion, Medallion Pipeline, Data Storage, Cloud Platform, Data Warehouse, ML Framework, Experiment Tracking, Model Serving, Visualization, CI/CD, Containerization, Orchestration

5. Dataset table: File | Rows | Key Columns | Purpose (one row per Olist file, 9 rows)

6. ML Models table: Model | Algorithm | Target | Key Metric | MLflow Name

7. API Endpoints table: Method | Endpoint | Input | Output for all 5 endpoints

8. Results table (fill with placeholder metrics you update after training):
Demand Forecast MAPE: X.XX% | Churn AUC: X.XX | Churn F1: X.XX | Segmentation Silhouette: X.XX

9. Quick Start (local — 6 numbered steps): clone → install → kaggle download → run_pipeline.py → python build models → uvicorn

10. Azure Deployment: 3 steps using ARM template

11. Power BI Connection: "Open Power BI Desktop → Get Data → Delta Lake → point to data/gold/bi_revenue/ → build Revenue Trend, Demand Forecast, Customer Segments, Seller KPIs pages"

12. Project Structure: folder tree

13. License: MIT