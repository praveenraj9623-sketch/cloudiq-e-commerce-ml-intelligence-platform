# CloudIQ — E-Commerce ML Intelligence Platform

End-to-end e-commerce intelligence platform built on a local PySpark + Delta
Lake medallion pipeline (Bronze → Silver → Gold) over the Brazilian
E-Commerce (Olist) dataset. This repository currently implements the
fast-track **Phase 0–3 MVP**: foundation utilities, bronze ingestion, silver
cleaning/joins, and gold feature engineering.

## Quick Start (Windows PowerShell)

### 1. Verify Java 17

PySpark and Delta Lake require Java 17. Confirm the version and that
`JAVA_HOME` points at a Java 17 installation:

```powershell
java -version
# Expect: openjdk version "17.x.x" ...

$env:JAVA_HOME
# Expect: a path to a Java 17 JDK, e.g. C:\Program Files\Eclipse Adoptium\jdk-17...
```

If `JAVA_HOME` is unset, set it for the current session:

```powershell
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.11.9-hotspot"
```

### 2. Create and activate a virtual environment (Python 3.11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Provide the dataset

Place the 9 Olist CSV files under `data/raw/` (already present locally; this
directory is never committed). Expected files:

```
data/raw/olist_orders_dataset.csv
data/raw/olist_customers_dataset.csv
data/raw/olist_order_items_dataset.csv
data/raw/olist_products_dataset.csv
data/raw/olist_order_reviews_dataset.csv
data/raw/olist_order_payments_dataset.csv
data/raw/olist_sellers_dataset.csv
data/raw/olist_geolocation_dataset.csv
data/raw/product_category_name_translation.csv
```

### 5. Run the pipeline

```powershell
python run_pipeline.py --layer bronze
python run_pipeline.py --layer silver
python run_pipeline.py --layer gold
python run_pipeline.py --layer all
```

## Quality Checks

```powershell
ruff check src/ tests/
pytest tests/ -v
```

Unit tests run without a Spark JVM or Delta JAR download; a full Spark run is
only exercised when `data/raw/` is populated.

For the full local Spark test suite on this Windows machine, use the JVM thread
mitigation documented in `docs/dashboard_runbook.md`:

```powershell
$env:JAVA_TOOL_OPTIONS="-XX:ActiveProcessorCount=2 -XX:CICompilerCount=2 -XX:TieredStopAtLevel=1 -Xss512k"
python -m pytest tests -q
Remove-Item Env:\JAVA_TOOL_OPTIONS
```

## Run the Dashboard

Export compact local dashboard marts, then launch Streamlit:

```powershell
python scripts/export_dashboard_data.py
streamlit run streamlit_app.py
```

The dashboard reads only local CSV/JSON marts from `data/dashboard/`, so Spark
does not start on every page refresh. Demand forecasting is shown honestly:
the `naive_lag_1` prior-month baseline beat XGBoost on chronological
validation, so XGBoost is not the selected champion model.

## Pipeline Layers

| Layer | Module | Output |
|---|---|---|
| Bronze | `src/processing/bronze.py` | `data/bronze/*` — 9 Olist Delta tables |
| Silver | `src/processing/silver.py` | `master_orders`, `customer_profile`, `product_demand`, `seller_performance` |
| Gold | `src/processing/gold.py` | `churn_features`, `demand_history`, `demand_features`, `rfm_segments`, `bi_revenue` |

## Key Design Decisions (Architecture Corrections)

- **Demand target** is `monthly_units` (C2). Demand history, features, and the
  training/evaluation window all end at **2018-09**; the partial October 2018
  month is excluded (C16).
- **`rolling_mean_3`** uses `rowsBetween(-3, -1)` — the three preceding months
  only, never the current month (C16).
- **Churn features** are temporal snapshots from `silver/master_orders` with a
  future 90-day repeat-purchase label; `recency_days` and `log_recency` are
  excluded (C1).
- **Seller revenue** uses `sum(price)` and `countDistinct(order_id)` directly
  from `order_items`; review and late-rate metrics use a distinct
  `seller_id` + `order_id` mapping (C3, C4).
- **`segment_label`** from `gold/rfm_segments` is the canonical segmentation
  output; no KMeans output is produced in the Gold phase (C5, C14).
- **Config validation** is local-first: `validate(strict=False)` warns on
  unresolved optional placeholders; `validate(strict=True)` raises (C7).
- **Spark** uses `local[*]` with `configure_spark_with_delta_pip`; no session
  is started at import time (C12).

## Project Structure

```
.
├─ config.yaml
├─ requirements.txt
├─ run_pipeline.py
├─ src/
│  ├─ processing/   bronze.py, silver.py, gold.py
│  └─ utils/        config.py, logger.py, spark_session.py
└─ tests/           test_config.py, test_demand_features.py, test_pipeline_smoke.py
```
