# Architecture Corrections — CloudIQ

Corrections to `docs/cloudiq_build_spec.md`. Apply each correction before
implementing the affected phase. Do not implement any phase that contradicts
a correction listed here.

Total corrections: 18

---

## Correction 1 — Churn: Temporal Snapshot Design from silver/master_orders

### Problem
Phase C10 (`build_churn_features`) reads `silver/customer_profile` — a
single end-of-dataset aggregate — and defines the label as:

    is_churned = F.when(F.col("recency_days") > 90, 1).otherwise(0)

`recency_days` is then passed as the predictor `log_recency =
F.log1p("recency_days")`. This is direct target leakage. A single static
snapshot provides no temporal separation between what the model knows and
what it predicts.

### Required Fix

1. **Source table: `silver/master_orders` only.** Do not use
   `silver/customer_profile` as the feature source for churn training.

2. **Snapshot dates T:** the first day of each month from 2017-01 through
   2018-07. Use only dates where a complete 90-day future observation
   window exists within the dataset (dataset ends 2018-10-17; last valid
   T is 2018-07-01, giving a window through 2018-09-29).

3. **Eligible customers at T:** include only customers who have at least
   one order with `order_status = 'delivered'` AND
   `order_delivered_customer_date <= T`. Do not use `'shipped'` as a
   completed customer purchase; a shipped order has not yet been received
   by the customer. Do not use `order_purchase_timestamp <= T` alone —
   an order purchased before T may not have been delivered by T and its
   outcome would not be knowable at T.

4. **Snapshot-time data availability for all features.** Every feature
   must be computed using only information that was knowable at T:
   - **Order-based features** (`total_orders_at_T`, `total_revenue_at_T`,
     `avg_order_value_at_T`, `late_delivery_rate_at_T`, etc.): filter
     `silver/master_orders` to `order_status = 'delivered'` AND
     `order_delivered_customer_date <= T`.
   - **Review-score features** (`avg_review_score_at_T`): join to
     `bronze/reviews` and filter to `review_creation_date <= T`. Do not
     use reviews created after T, even if the underlying order was
     delivered before T.
   - **`customer_age_days_at_T`**: compute as
     `datediff(T, min(order_delivered_customer_date))` over the
     delivered-by-T slice. Do not use the purchase timestamp or delivery
     date of orders not yet delivered at T.
   - **`days_since_last_order_at_T`**: `datediff(T, max(order_delivered_customer_date))`
     over the delivered-by-T slice.
   - Do not use any final order status, final delivery outcome, or review
     that was not knowable at T.

5. **Approved feature columns for model training:**
   `total_orders_at_T`, `total_revenue_at_T`, `avg_order_value_at_T`,
   `avg_review_score_at_T`, `late_delivery_rate_at_T`,
   `purchase_frequency_score_at_T`, `revenue_per_order_norm_at_T`,
   `high_late_delivery_flag_at_T`, `days_since_last_order_at_T`,
   `customer_age_days_at_T`.

6. **Label `is_churned` from future behaviour after T:** a customer is
   labelled churned if they have no `delivered` order with
   `order_delivered_customer_date` in the 90-day window
   `(T, T + 90 days]`. Computed from a separate forward-looking slice of
   `silver/master_orders`. Never derived from the same recency field used
   as a predictor.

7. **Exclude `recency_days` and `log_recency` entirely** from the feature
   matrix.

8. **Each `(customer_unique_id, snapshot_month)` pair is one training row.**

> **Note — Repeat-purchase risk framing:** Olist has a high proportion of
> one-time buyers. If exploratory analysis shows the majority of customers
> have only one delivered order, frame the model as a **repeat-purchase
> risk model** rather than a traditional churn model. Document the chosen
> framing in the MLflow run description.

---

## Correction 2 — Demand Forecast Target: `weekly_orders` → `monthly_units`

### Problem
`config.yaml` sets `models.demand_forecast.target: "weekly_orders"`. The
Silver pipeline produces `monthly_units = count("order_item_id")` grouped
by `(category_name_english, order_year_month)`. The column `weekly_orders`
does not exist anywhere in the pipeline.

### Required Fix

| Location | Change |
|---|---|
| `config.yaml` | `models.demand_forecast.target: "monthly_units"` |
| `DemandForecastModel.__init__` | Default: `config.get("models.demand_forecast.target", "monthly_units")` |
| `DemandForecastModel.load_features` | `y = df["monthly_units"]` |
| API `/predict/demand` response | Field name `predicted_units` (already correct in spec) |
| `README.md` ML Models table | Target column: `monthly_units` |
| All docstrings and comments | Replace every reference to `weekly_orders` with `monthly_units` |

---

## Correction 3 — Seller Performance: Revenue Duplication and Order Count

### Problem
Phase C9 `build_seller_performance` joins `order_items` to `master_orders`
and sums `master_orders.order_revenue`. Because `order_revenue` is already
a per-order aggregate of `price` from the items join in
`build_master_orders`, re-joining to `order_items` re-expands rows to item
level, counting revenue once per item rather than once per order.
`count("order_id")` on item-level rows counts items, not distinct orders.

### Required Fix

Aggregate revenue directly from `order_items`, never through `master_orders`:

    seller_agg = order_items.groupBy("seller_id").agg(
        F.countDistinct("order_id").alias("total_orders"),
        F.sum("price").alias("total_revenue_excl_freight"),
        F.sum("freight_value").alias("total_freight"),
        F.sum(F.col("price") + F.col("freight_value")).alias("total_revenue_incl_freight"),
    )

Use `total_revenue_excl_freight` (product revenue only) for
`performance_tier` classification. Expose both columns in the output table
and document the choice explicitly in code and schema.

---

## Correction 4 — Seller Review and Late-Rate Aggregation

### Problem
Phase C9 `build_seller_performance` joins `order_items` directly to
`master_orders` to compute `avg_review_score` and `late_rate`. A seller
with multiple items in one order appears multiple times in the join result,
causing that order's review score and late-delivery flag to be counted once
per item rather than once per order.

### Required Fix

Create a distinct seller-to-order mapping before joining to `master_orders`:

    seller_order_map = (
        order_items
        .select("seller_id", "order_id")
        .dropDuplicates()
    )

    seller_metrics = (
        seller_order_map
        .join(master_orders, on="order_id", how="left")
        .groupBy("seller_id")
        .agg(
            F.avg("avg_review_score").alias("avg_review"),
            F.avg("is_late").alias("late_rate"),
        )
    )

Join `seller_metrics` to `seller_agg` (from Correction 3) on `seller_id`
to produce the final `seller_performance` table. Each order contributes
exactly one review score and one late-delivery flag per seller, regardless
of item count.

---

## Correction 5 — Segmentation: RFM Rule-Based as Canonical; KMeans Exploratory Only

### Problem
Phase C11 produces rule-based RFM labels (`Champion`, `Loyal`, etc.) in
`gold/rfm_segments`. Phase C15 trains KMeans on the same RFM scores and
produces a second, separate set of cluster labels. The two outputs are
never reconciled and will contradict each other in the API and Streamlit
demo.

### Required Fix

- **`segment_label` from `gold/rfm_segments` is the sole customer-facing
  segmentation output** — used in the API (`/predict/segment`), the
  Streamlit demo, and all reporting.
- **KMeans training and any exploratory cluster labels or reports are
  created only in Phase 5 (ML Models).** Do not produce
  `gold/kmeans_exploration` in Phase 3 (Gold Feature Engineering). RFM
  quintile scoring is produced in Phase 3; KMeans is a separate model
  trained in Phase 5.
- `CustomerSegmentation.train()` must include a docstring stating:
  *"Exploratory analysis only. The production segmentation label is
  `segment_label` from `gold/rfm_segments`.*"
- Do not add a `/predict/kmeans_segment` endpoint.

---

## Correction 6 — CI/CD: Modify Existing .gitlab-ci.yml; Preserve Secret Detection

### Problem
Phase C19 creates `.github/workflows/ci_cd.yml` using GitHub Actions
syntax. This project is hosted on GitLab; that file will never execute.

### Required Fix

- **Do not create a `.github/` directory.**
- **Modify the existing `.gitlab-ci.yml`** at the project root. The file
  currently defines stages `[test, secret-detection]`, sets
  `SECRET_DETECTION_ENABLED`, and includes
  `Security/Secret-Detection.gitlab-ci.yml`. All of this must be
  preserved exactly.
- Prepend `lint`, `test`, `build`, and `deploy` to the stages list. The
  final `stages:` list must be:
  `[lint, test, build, deploy, secret-detection]`.
- Use GitLab CI predefined variables throughout:

  | GitHub Actions | GitLab CI equivalent |
  |---|---|
  | `GITHUB_TOKEN` | `CI_JOB_TOKEN` |
  | `github.sha` | `CI_COMMIT_SHA` |
  | `github.repository` | `CI_PROJECT_PATH` |
  | `ghcr.io/${{ github.repository }}` | `$CI_REGISTRY_IMAGE` |

- Registry login: `docker login -u $CI_REGISTRY_USER -p $CI_JOB_TOKEN $CI_REGISTRY`
- Image tag: `$CI_REGISTRY_IMAGE/cloudiq-api:$CI_COMMIT_SHA`
- `build` job runs only on pushes to `main`:
  `rules: [{if: '$CI_COMMIT_BRANCH == "main"'}]`

---

## Correction 7 — Config Validation: Local-First; Azure/Databricks Optional

### Problem
Phase C3 `ConfigLoader.validate()` raises `ValueError` for all unresolved
`${...}` placeholders. All Azure and Databricks variables have no
`:-default` fallback, so `validate()` always raises on a local run without
those env vars set.

### Required Fix

    def validate(self, strict: bool = False) -> bool: ...

| Tier | Variables | `strict=False` | `strict=True` |
|---|---|---|---|
| Required | `MLFLOW_TRACKING_URI` (has `:-mlruns` default; always resolves) | Pass | Pass |
| Optional | All `azure.*` and `azure.databricks.*` keys | Log `WARNING` per unresolved placeholder; return `True` | Raise `ValueError` listing all unresolved placeholders |

- `validate()` (i.e., `strict=False`) is the default for all local runs.
- `validate(strict=True)` is called only in cloud deployment contexts
  (GitLab CI `deploy` stage, Databricks bootstrap script).
- Preserve the `.env.example` comment: *"# Minimum for local run: none
  required. All Azure values optional."*

---

## Correction 8 — Streamlit Demo: Lightweight Artefacts Only at Runtime

### Problem
The build spec does not define what data sources the Streamlit demo reads
at runtime. A demo that requires PySpark, Delta Lake, Azure credentials,
or an MLflow server cannot run locally or deploy to Streamlit Community
Cloud.

### Required Fix

**Permitted runtime sources in `demo_streamlit/`:**
- CSV, Parquet, and JSON files in `demo_streamlit/data/`
- Lightweight model files (e.g., `*.joblib`) in `demo_streamlit/models/`
  — only if the demo performs interactive inference

**Forbidden runtime imports in any file under `demo_streamlit/`:**
- `pyspark` or any `pyspark.*` submodule
- `delta` or `deltalake`
- `azure-storage-blob`, `azure-identity`, `azure-mgmt-*`
- `databricks-sdk` or `databricks-connect`
- `mlflow` (including `mlflow.pyfunc`, `mlflow.xgboost`, etc.)

**Dependency isolation — `demo_streamlit/requirements.txt`:**
List only `streamlit`, `pandas`, `plotly`, `numpy`. If the demo performs
interactive inference using an exported XGBoost joblib model, also include
a compatible version of `xgboost`. Do not claim `scikit-learn` alone can
load an XGBoost joblib model; `scikit-learn` does not provide
`XGBClassifier` or `XGBRegressor`. If no interactive inference is needed,
use precomputed churn outputs exported as CSV instead.

**Export phase (Phase 6):** `scripts/export_demo_assets.py` reads Gold
Delta tables locally and writes lightweight files to `demo_streamlit/data/`
and optionally `demo_streamlit/models/`. This script runs offline before
the demo; it is not called at demo runtime.

**Asset handling in this repository:** Generated demo assets in
`demo_streamlit/data/` and `demo_streamlit/models/` are gitignored in this
engineering repository. The `.gitignore` is updated in Phase 6 to ignore
`demo_streamlit/data/*` and `demo_streamlit/models/*` while preserving the
`.gitkeep` placeholder files that track the empty directories.

**Deployment to Streamlit Community Cloud:** Copy only the approved compact
artefacts — CSV, JSON, Parquet, and optional joblib files — into the
separate public GitHub repository **`cloudiq-streamlit-demo`** and commit
them there. The Streamlit Community Cloud app connects to that repository,
not to this GitLab repository. The `cloudiq-streamlit-demo` repository
contains only Streamlit application code and pre-exported data files; it
contains no PySpark, Delta, Azure, or MLflow code.

---

## Correction 9 — Model Evaluation Must Be Time-Aware

### Problem
**Demand (Phase C13):** `time_series_split` increments `split_point` by
`len // n_splits` on a dataframe sorted by `(category_name_english,
order_year_month)`. This mixes months across categories, allowing the model
to see future data during training.

**Churn (Phase C14):** `StratifiedKFold(n_splits=5, shuffle=True)` randomly
shuffles snapshot rows, destroying temporal ordering.

### Required Fix

**Demand — global chronological split by `order_year_month`:**
- Derive the sorted list of unique `order_year_month` values across all
  categories.
- Use `sklearn.model_selection.TimeSeriesSplit` on that sorted month list
  to determine fold boundaries.
- For each fold, filter rows by month membership. Every validation and
  test month must be strictly later than every training month across all
  categories simultaneously.
- Do not split by row index on a dataframe sorted by `(category, month)`.

**Churn — chronological snapshot splits:**
- Sort snapshot rows by `snapshot_month`.
- Use a walk-forward split: train on months 1–N, validate on month N+1,
  advance by one month per fold.
- Do not use `StratifiedKFold` with `shuffle=True` on snapshot data.
- Class stratification via oversampling (e.g., SMOTE) may be applied
  within the training split of each fold only, never across the
  train/validation boundary.

---

## Correction 10 — Demand Inference: Category History and Recursive Forecasting

### Problem
Phase C16 `POST /predict/demand` builds lag features using "median values
if no history." Using global median lags as normal production behaviour
produces forecasts not grounded in the requested category's actual demand
pattern.

### Required Fix

**At inference time, for the requested category:**
1. Look up the category's actual monthly history from `gold/demand_history`
   loaded at API startup (see Correction 18).
2. Extract the latest real `monthly_units` values as a four-month rolling
   history queue: `[month_T-3, month_T-2, month_T-1, month_T]` (oldest
   to newest), where T is the last month in `gold/demand_history`
   (2018-09).
3. For `horizon_months = 1`, derive features from the queue and predict
   directly.
4. For `horizon_months > 1`, forecast **recursively** using the following
   exact procedure for each future step H:
   - Derive features from the current four-value queue:
     - `lag_1` = queue[-1] (most recent value)
     - `lag_2` = queue[-2]
     - `lag_4` = queue[-4] (oldest value in the queue)
     - `rolling_mean_3` = mean of queue[-3], queue[-2], queue[-1]
     - `month_num` = calendar month number of the target month
     - `is_q4` = 1 if `month_num >= 10` else 0
   - Predict `monthly_units` for step H.
   - Append the prediction to the queue and remove the oldest value,
     keeping the queue length exactly 4.
   - Advance the target month by one calendar month.
   - Repeat for H+1.
   - The first predicted month is 2018-10 (October 2018).
5. **Insufficient history:** if the category has fewer than 4 months of
   history in `gold/demand_history`, return HTTP 422:
   `{"error": "insufficient_history", "category": "<name>",
   "available_months": <n>, "required_months": 4}`.
   Do not fall back to median values silently.
6. **Unknown category:** if the category does not appear in
   `gold/demand_history`, return HTTP 404:
   `{"error": "category_not_found", "category": "<name>"}`.
7. **Streamlit alternative:** the Streamlit demo may display precomputed
   offline forecasts exported as CSV rather than calling the API
   recursively at runtime.

---

## Correction 11 — Streamlit Import Validation: Python AST Check, Not grep

### Problem
Using `grep -r "pyspark\|delta\|azure\|mlflow\|databricks" demo_streamlit/`
produces false failures on comments, docstrings, and documentation strings
that mention those names without importing them.

### Required Fix

Replace the grep gate with `scripts/check_demo_imports.py`, a Python
AST-based import validator:

1. Walk all `.py` files under `demo_streamlit/` using `pathlib`.
2. Parse each file with `ast.parse()`.
3. Walk the AST and collect all `ast.Import` and `ast.ImportFrom` nodes.
4. Check the top-level module name of each import against the forbidden
   set: `{"pyspark", "delta", "deltalake", "azure", "databricks", "mlflow"}`.
5. Report each violation as `<file>:<line>: forbidden import '<module>'`.
6. Exit with code 1 if any violations are found; exit 0 otherwise.

This script is called in the Phase 6 gate and in the GitLab CI `lint`
stage. It must not flag comments or string literals that mention forbidden
package names.

---

## Correction 12 — Delta Lake Local Runtime: Java 17 and delta-spark Builder Setup

### Problem
Phase C4 specifies `get_spark_session` using standard `SparkSession.builder`
with Delta SQL extension and catalog configs. On a local machine without
Java installed or with an incompatible JVM version, Spark will fail to
start. Additionally, `delta-spark` requires its JAR to be resolved at
session creation time; using only the SQL extension config without
`configure_spark_with_delta_pip` causes `ClassNotFoundException` for Delta
classes at runtime.

### Required Fix

**Phase 0 prerequisite — Java 17:**
- Phase 0 must document Java 17 as a required local dependency.
- The Foundation gate must include: `java -version` exits 0 and reports
  version 17.x; `JAVA_HOME` is set and points to a Java 17 installation.
- Document this in `README.md` Quick Start step 1.

**`get_spark_session` builder setup:**
Use `delta-spark`'s `configure_spark_with_delta_pip` to ensure the Delta
JAR is resolved automatically, in addition to the SQL extension and catalog
configs:

    from delta import configure_spark_with_delta_pip

    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", ...)
        .config("spark.executor.memory", ...)
        .config("spark.sql.shuffle.partitions", ...)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()

Without `configure_spark_with_delta_pip`, Delta writes fail with
`ClassNotFoundException: io.delta.sql.DeltaSparkSessionExtension` or
similar errors because the Delta JAR is not on the classpath.

---

## Correction 13 — Pipeline Runner Introduction Order and pytest-cov

### Problem
Phase C12 introduces `run_pipeline.py` as a standalone deliverable after
all three pipeline layers are complete. However, Phase 1 gate already
requires `python run_pipeline.py --layer bronze`, which means the runner
must exist before Phase 2 and Phase 3 are built. Additionally, Phase 9
uses `pytest --cov`, but `pytest-cov` is not listed in `requirements.txt`.

### Required Fix

**`run_pipeline.py` introduction order:**
- **Phase 1:** introduce `run_pipeline.py` with `--layer bronze` support
  only. The `--layer silver`, `--layer gold`, and `--layer all` paths
  raise `NotImplementedError` with a clear message.
- **Phase 2:** expand `run_pipeline.py` to add `--layer silver` support.
- **Phase 3:** expand `run_pipeline.py` to add `--layer gold` and
  `--layer all` support.
- Do not require `python run_pipeline.py --layer bronze` in Phase 1 gate
  before the runner file exists.

**`pytest-cov`:**
Add `pytest-cov` to `requirements.txt` in the `# Testing` section
alongside `pytest==7.4.4`. The Phase 9 CI command
`pytest --cov=src --cov-report=xml` will fail without it.

---

## Correction 14 — KMeans Phase Consistency

### Problem
An earlier draft stated that KMeans output goes to `gold/kmeans_exploration`
as part of the Gold pipeline. This implies Phase 3 produces KMeans output,
which contradicts the intent that KMeans is a model trained in Phase 5.

### Required Fix

- **Phase 3 (Gold Feature Engineering)** produces only:
  `gold/churn_features`, `gold/demand_features`, `gold/demand_history`,
  `gold/rfm_segments`, and `gold/bi_revenue`. It does not produce any
  KMeans output or `gold/kmeans_exploration`.
- **Phase 5 (ML Models)** trains `CustomerSegmentation` (KMeans) and
  writes exploratory cluster labels to `gold/kmeans_exploration` as a
  side effect of `CustomerSegmentation.train()`.
- Phase 3 gate assertions must not check for `gold/kmeans_exploration`.

---

## Correction 15 — Docker Healthcheck: curl Must Be Installed or Use Python

### Problem
Phase C18 specifies:

    HEALTHCHECK curl -f http://localhost:8000/health || exit 1

The final stage of the multi-stage Dockerfile is based on
`python:3.11-slim`, which does not include `curl` by default. The
healthcheck will always fail with `curl: command not found`.

### Required Fix

Choose one of the following two approaches and apply it consistently in
both `Dockerfile` and `docker-compose.yml`:

**Option A — Install curl explicitly in the final stage:**

    RUN apt-get update && apt-get install -y --no-install-recommends curl \
        && rm -rf /var/lib/apt/lists/*
    HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
        CMD curl -f http://localhost:8000/health || exit 1

**Option B — Python-based healthcheck (no curl dependency):**

    HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
        CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
        || exit 1

Do not reference `curl` in the `HEALTHCHECK` instruction unless Option A
is chosen and `curl` is explicitly installed in the same Dockerfile stage.

---

## Correction 16 — Demand Feature Leakage and Missing-Month Handling

### Problem
Phase C10 `build_demand_forecast_features` computes lag features directly
from the raw `silver/product_demand` table without first ensuring a
continuous monthly series per category. Missing months are silently
omitted, causing lag_1 to reference a month two or more periods back.
Additionally, the rolling mean window is not explicitly bounded to exclude
the current month's target value.

### Required Fix

**Training cutoff:**
Derive `demand_training_cutoff_month` from `max(order_purchase_timestamp)`
in `silver/orders` or `silver/master_orders` before monthly demand
aggregation. For the Olist dataset this resolves to **2018-09** (September
2018). October 2018 is excluded because the dataset ends partway through
that month and its unit counts are incomplete.

**Continuous monthly grid:**
Build each category's calendar from its **first observed month** through
**2018-09 inclusive**. Do not extend the grid into October 2018 or beyond.
Do not create artificial months before a category first appears. Fill
missing months **inside that active timeline** with `monthly_units = 0`.

**Lag features:**
Compute `lag_1`, `lag_2`, and `lag_4` from the continuous monthly grid.
Drop rows only when the required prior-month history is unavailable (fewer
than 4 months since the category's first observed month).

**Rolling mean:**
Compute `rolling_mean_3` using `rowsBetween(-3, -1)` — the three preceding
months only. Never include the current month's `monthly_units` in
`rolling_mean_3`.

**Separate outputs:**
- `gold/demand_history`: the complete continuous monthly grid per category
  (all months, including zero-filled, from first observed month through
  2018-09), written before any lag-row dropping. Used as the API
  category-history source at inference time.
- `gold/demand_features`: the lag-feature table after dropping rows with
  insufficient history. Used for model training only.

**Demand history, lag features, training, validation, and evaluation all
end at 2018-09.** API recursive forecasting starts from 2018-10 as the
first predicted month.

**Required unit tests in `tests/test_demand_features.py`:**

1. `test_rolling_mean_3_uses_three_prior_months`: construct a known
   continuous monthly series (e.g., `[10, 20, 30, 40, 50]`) and assert
   that `rolling_mean_3` at each position equals the mean of exactly the
   three preceding months (e.g., position 4 → mean(20, 30, 40) = 30.0).

2. `test_partial_final_month_excluded`: construct a small orders-level
   DataFrame with `max(order_purchase_timestamp)` in October 2018, derive
   `demand_training_cutoff_month`, and assert it equals `2018-09` —
   confirming the partial final month is excluded.

---

## Correction 17 — MLflow Model Promotion and API Loading Contract

### Problem
Phase C13 and C16 register models in MLflow and load them via
`models:/cloudiq_churn_predictor/Production`. The `/Production` stage
requires an explicit promotion step that is never defined in the spec.
Without a database-backed MLflow server, model registration and aliases
are not supported at all.

### Required Fix

**Backend requirement:**
Model Registry and alias workflows require a database-backed MLflow
tracking server. A file-store `mlruns/` backend does not support model
registration or aliases.
- **Local setup:** run `mlflow server --backend-store-uri sqlite:///mlflow.db
  --default-artifact-root ./mlruns`. Set `config.yaml`
  `mlflow.tracking_uri` to the server URL (e.g., `http://127.0.0.1:5000`).
- **Docker setup:** the `api` service uses the MLflow service URL as its
  tracking URI; the `mlflow` service uses its own SQLite backend store.
  Both are defined in `docker-compose.yml` — document the dependency
  explicitly so the `api` service does not start before `mlflow` is ready.

**Alias assignment:**
For each named model, assign the `champion` alias only to the selected
best registered version after measured validation results are confirmed.
Do not auto-assign `champion` to every training run.

**API model loading:**
Load models via aliases:
- `models:/cloudiq_churn_predictor@champion`
- `models:/cloudiq_demand_forecast@champion`

Do not use `/Production` unless a documented promotion step explicitly
assigns that stage.

**Health and models endpoints:**
`/health` and `/models` must report loaded model `name`, `version`, and
`alias` for each model.

**API tests:**
Add tests that mock alias-based model loading for both
`models:/cloudiq_churn_predictor@champion` and
`models:/cloudiq_demand_forecast@champion`.

---

## Correction 18 — Demand Training Metadata Preservation

### Problem
Phase C13 `DemandForecastModel.load_features` returns only `X` and `y`,
discarding `category_name_english` and `order_year_month`. The custom
`time_series_split` then splits by row index on a mixed-category dataframe,
producing temporally invalid folds. At inference time, the API uses
`gold/demand_features` as the category-history source, but that table has
had early rows dropped during lag computation and is therefore an
incomplete series.

### Required Fix

**Feature loading:**
`DemandForecastModel.load_features` must retain `category_name_english`
and `order_year_month` as metadata columns alongside `X` and `y`. Return
a tuple of `(X, y, metadata_df)` or a single DataFrame with all columns
present, so that fold membership can be determined from metadata before
`X` and `y` are separated.

**Fold splitting:**
`TimeSeriesSplit` operates on the sorted list of unique `order_year_month`
values derived from `metadata_df`. For each fold, rows are selected by
filtering `metadata_df` on month membership. `X` and `y` are sliced only
after fold membership is determined from metadata.

**API category-history source:**
At API startup, load `gold/demand_history` (the complete continuous monthly
series produced before lag-row dropping, per Correction 16) as the
category-history source for recursive inference. Do not use
`gold/demand_features` alone for this purpose — it is missing early months
where lag history was unavailable and cannot serve as a complete series for
the four-month rolling queue.
