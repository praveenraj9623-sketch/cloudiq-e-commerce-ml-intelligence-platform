# Implementation Order — CloudIQ

Build one phase at a time. Do not start a phase until its gate passes.
Each phase lists its inputs, deliverables, and a pass/fail gate.

---

## Phase 0 — Foundation

**Inputs:** Repository with bootstrap structure (`AGENTS.md`, `.gitignore`,
`src/` package stubs, `tests/__init__.py`, empty tracked folders).

**Prerequisites (local machine):**
- Java 17 installed and `JAVA_HOME` set to a Java 17 installation.
  Required for PySpark and Delta Lake. Document in `README.md` Quick
  Start step 1 (C12).

**Deliverables:**
- `requirements.txt` — includes `pytest-cov` in the `# Testing` section
  (C13)
- `config.yaml` — `models.demand_forecast.target: "monthly_units"` (C2)
- `.env.example`
- `src/utils/config.py` — `ConfigLoader` with `validate(strict=False)` (C7)
- `src/utils/logger.py` — `get_logger`, `log_exceptions`
- `src/utils/spark_session.py` — `get_spark_session` using
  `configure_spark_with_delta_pip` builder setup (C12)

**Gate — PASS when:**
- `java -version` exits 0 and reports version 17.x.
- `JAVA_HOME` is set and points to a Java 17 installation.
- `ruff check src/ tests/` exits 0.
- `pytest tests/` exits 0 (unit tests for `ConfigLoader.get`,
  `ConfigLoader.validate(strict=False)`, and `get_logger`).
- `ConfigLoader("config.yaml").validate(strict=False)` returns `True`
  with no Azure env vars set.
- `ConfigLoader("config.yaml").validate(strict=True)` raises `ValueError`
  listing all unresolved Azure and Databricks placeholders.

---

## Phase 1 — Bronze Ingestion

**Inputs:** `data/raw/` populated with the 9 Olist CSV files.

**Deliverables:**
- `src/processing/bronze.py` (`BronzeLayer`, all 9 ingest methods,
  `run_pipeline`)
- `run_pipeline.py` — introduced here with `--layer bronze` support only;
  `--layer silver`, `--layer gold`, and `--layer all` raise
  `NotImplementedError` (C13)

**Gate — PASS when:**
- `python run_pipeline.py --layer bronze` exits 0.
- All 9 Delta tables exist under `data/bronze/`.
- Row counts match expected Olist dataset sizes (documented in test
  assertions).
- `ruff` and `pytest` pass.

---

## Phase 2 — Silver Cleaning and Joins

**Inputs:** `data/bronze/` (all 9 Delta tables from Phase 1).

**Deliverables:**
- `src/processing/silver.py` (`SilverLayer`: `clean_orders`,
  `clean_order_items`, `build_master_orders`, `build_customer_profile`,
  `build_product_demand`, `build_seller_performance`, `run_pipeline`)
- `run_pipeline.py` expanded to add `--layer silver` support (C13)

**Corrections applied:**
- `build_seller_performance` uses `countDistinct("order_id")` and
  `sum("price")` directly from `order_items` for revenue (C3).
- `build_seller_performance` creates a distinct `seller_order_map` via
  `.select("seller_id", "order_id").dropDuplicates()` before joining to
  `master_orders` for `avg_review_score` and `late_rate` (C4).

**Gate — PASS when:**
- `python run_pipeline.py --layer silver` exits 0.
- `silver/master_orders` row count equals `silver/orders` row count
  (no fan-out from joins — verified by assertion).
- `silver/seller_performance.total_revenue_excl_freight` matches an
  independent `sum(price)` computed directly from `bronze/order_items`
  (verified by assertion).
- `ruff` and `pytest` pass.

---

## Phase 3 — Gold Feature Engineering

**Inputs:** `data/silver/` (all Silver Delta tables from Phase 2).

**Deliverables:**
- `src/processing/gold.py` (`GoldLayer`: `build_churn_features`,
  `build_demand_forecast_features`, `build_rfm_segments`,
  `build_bi_revenue`, `run_pipeline`)
- `run_gold.py`
- `run_pipeline.py` expanded to add `--layer gold` and `--layer all`
  support (C13)

**Corrections applied:**
- `build_churn_features` reads `silver/master_orders` directly, not
  `silver/customer_profile`. Eligible customers must have at least one
  order with `order_status = 'delivered'` AND
  `order_delivered_customer_date <= T`. Order-based features use only
  delivered-by-T rows; review-score features use only reviews with
  `review_creation_date <= T`; `customer_age_days_at_T` and
  `days_since_last_order_at_T` are computed from delivered-by-T rows.
  `is_churned` is derived from future delivered orders after T.
  `recency_days` and `log_recency` are excluded from the feature matrix
  (C1).
- `build_demand_forecast_features` derives `demand_training_cutoff_month`
  from `max(order_purchase_timestamp)` in `silver/orders` or
  `silver/master_orders` before aggregation, resolving to 2018-09.
  Builds a continuous monthly grid per category from first observed month
  through 2018-09; fills missing months with `monthly_units = 0`; computes
  `rolling_mean_3` using `rowsBetween(-3, -1)`. Writes `gold/demand_history`
  (complete series before lag-row dropping) and `gold/demand_features`
  (lag-feature table after dropping rows with insufficient history) (C16).
- Target column is `monthly_units` (C2).
- `build_rfm_segments` produces the canonical `segment_label` in
  `gold/rfm_segments`. No KMeans output is produced in this phase (C14).
- `build_bi_revenue` produces `gold/bi_revenue`.

**Gate — PASS when:**
- `python run_pipeline.py --layer gold` exits 0.
- `gold/churn_features` contains no column named `recency_days` or
  `log_recency` (verified by assertion).
- `gold/churn_features` contains a `snapshot_month` column with at least
  5 distinct values (verified by assertion).
- `gold/demand_features` contains column `monthly_units` (verified by
  assertion).
- `gold/demand_history` exists and contains all category×month
  combinations from each category's first observed month through 2018-09,
  including zero-filled months (verified by assertion) (C16, C18).
- `gold/demand_features` row count is less than `gold/demand_history` row
  count, confirming early lag-unavailable rows were dropped (C18).
- Neither `gold/demand_history` nor `gold/demand_features` contains any
  row with `order_year_month >= '2018-10'` (verified by assertion) (C16).
- `gold/kmeans_exploration` does **not** exist (verified by assertion) (C14).
- `pytest tests/test_demand_features.py::test_rolling_mean_3_uses_three_prior_months`
  passes — deterministic series test confirming `rolling_mean_3` equals
  the mean of exactly the three preceding months (C16).
- `pytest tests/test_demand_features.py::test_partial_final_month_excluded`
  passes — constructs a small orders-level DataFrame with
  `max(order_purchase_timestamp)` in October 2018, derives
  `demand_training_cutoff_month`, and asserts it equals `2018-09` (C16).
- `ruff` and `pytest` pass.

---

## Phase 4 — Data Validation

**Inputs:** All Bronze, Silver, and Gold Delta tables.

**Deliverables:**
- `scripts/validate_pipeline.py` — row-count checks, null-rate checks,
  schema assertions, and join-integrity checks for all layers.
- Corresponding test assertions in `tests/`.

**Gate — PASS when:**
- `python scripts/validate_pipeline.py` exits 0 with all checks green.
- No Silver or Gold table exceeds defined null-rate thresholds for key
  columns.
- `ruff` and `pytest` pass.

---

## Phase 5 — ML Models

**Inputs:** `data/gold/` (churn_features, demand_features, rfm_segments).

**Prerequisites:**
- MLflow tracking server running with a SQLite backend:
  `mlflow server --backend-store-uri sqlite:///mlflow.db
  --default-artifact-root ./mlruns`
- `config.yaml` `mlflow.tracking_uri` points to the server URL
  (e.g., `http://127.0.0.1:5000`) (C17).

**Deliverables:**
- `src/models/demand_forecast.py` (`DemandForecastModel`)
- `src/models/churn.py` (`ChurnModel`)
- `src/models/segmentation.py` (`CustomerSegmentation` — exploratory only)
- `gold/kmeans_exploration` — written by `CustomerSegmentation.train()`
  as an exploratory side effect (C14)
- MLflow runs written to the SQLite-backed server (not committed).
- `reports/shap_churn.png`, `reports/kmeans_elbow.png` (not committed).

**Corrections applied:**
- `DemandForecastModel.load_features` retains `category_name_english` and
  `order_year_month` as metadata; fold splits are derived from sorted
  unique `order_year_month` metadata values; `X` and `y` are separated
  only after fold membership is determined from metadata (C18).
- `DemandForecastModel` splits by global `order_year_month` chronology
  using `TimeSeriesSplit` on the sorted unique month list; every
  validation/test month is strictly later than every training month
  across all categories. All folds contain only rows with
  `order_year_month <= '2018-09'` (C9, C16).
- `DemandForecastModel` target default is `"monthly_units"` (C2).
- `ChurnModel` uses walk-forward chronological snapshot splits; earlier
  snapshot months train, later months validate/test. `StratifiedKFold`
  with `shuffle=True` is not used (C9).
- `ChurnModel` feature list excludes `log_recency` (C1).
- `CustomerSegmentation.train()` docstring marks output as exploratory;
  writes `gold/kmeans_exploration` (C5, C14).

**Gate — PASS when:**
- All three models train without error.
- Metrics are logged to MLflow and printed to stdout.
- **No metric values are invented.** Only measured results are reported
  (per `AGENTS.md`).
- All demand training and validation folds contain only rows with
  `order_year_month <= '2018-09'` (verified by assertion) (C16).
- `DemandForecastModel.load_features` retains metadata columns;
  fold splits derived from sorted unique `order_year_month` metadata
  values, not row index (verified by test) (C18).
- After training, the `champion` alias is assigned to the selected best
  registered version of each named model in MLflow — only after measured
  validation results are confirmed, not automatically (C17).
- `gold/kmeans_exploration` exists after `CustomerSegmentation.train()`
  runs (C14).
- `ruff` and `pytest` pass.

---

## Phase 6 — Export Lightweight Artefacts

**Inputs:** `data/gold/` Delta tables and trained model files from Phase 5.

**Deliverables:**
- `scripts/export_demo_assets.py` — reads Gold Delta tables locally,
  writes CSV/Parquet/JSON to `demo_streamlit/data/`; optionally exports
  a joblib model to `demo_streamlit/models/`.
- `scripts/check_demo_imports.py` — Python AST-based import validator
  for all `.py` files under `demo_streamlit/` (C11).
- `.gitignore` updated to ignore `demo_streamlit/data/*` and
  `demo_streamlit/models/*` while preserving `.gitkeep` placeholder files
  (C8).
- `demo_streamlit/data/` populated locally (gitignored in this repository).

**Deployment note:** Generated demo assets are gitignored here and never
committed to this repository. For Streamlit Community Cloud deployment,
copy only the approved compact artefacts (CSV, JSON, Parquet, optional
joblib) into the separate public GitHub repository
**`cloudiq-streamlit-demo`** and commit them there (C8).

**Gate — PASS when:**
- `python scripts/export_demo_assets.py` exits 0.
- All expected files exist under `demo_streamlit/data/`.
- `python scripts/check_demo_imports.py` exits 0 — no forbidden imports
  found in any `.py` file under `demo_streamlit/` (C11).
- `ruff` and `pytest` pass.

---

## Phase 7 — Streamlit Demo

**Inputs:** Pre-exported files in `demo_streamlit/data/` from Phase 6.

**Deliverables:**
- `demo_streamlit/app.py`
- `demo_streamlit/pages/` — one file per dashboard page.
- `demo_streamlit/requirements.txt` — `streamlit`, `pandas`, `plotly`,
  `numpy`; add `xgboost==2.0.3` only if the demo performs interactive
  inference using an exported XGBoost joblib model. Do not include
  `scikit-learn` as a substitute for loading XGBoost models. If no
  interactive inference is needed, use precomputed churn outputs exported
  as CSV (C8).

**Corrections applied:**
- Demo reads only CSV, Parquet, JSON, and optional joblib files (C8).
- No PySpark, Delta, Azure SDK, Databricks SDK, or MLflow import
  anywhere under `demo_streamlit/` (C8).

**Gate — PASS when:**
- `streamlit run demo_streamlit/app.py` starts without error.
- All pages render with the pre-exported data.
- `python scripts/check_demo_imports.py` exits 0 (C11).
- `pip install -r demo_streamlit/requirements.txt` completes in under
  60 seconds on a clean Python 3.11 environment.
- `ruff` and `pytest` pass.

---

## Phase 8 — FastAPI and Docker

**Inputs:** Trained models with `champion` aliases assigned in MLflow
(Phase 5); `gold/demand_history` loaded at API startup (Phase 3).

**Deliverables:**
- `src/serving/api.py` (all endpoints)
- `tests/test_api.py` and `tests/conftest.py`
- `Dockerfile` (multi-stage) — healthcheck uses either explicitly
  installed `curl` or a Python-based `urllib.request` check (C15)
- `docker-compose.yml` — healthcheck consistent with `Dockerfile`;
  `api` service depends on `mlflow` service and uses its URL as
  `MLFLOW_TRACKING_URI` (C15, C17)

**Corrections applied:**
- API loads models via `models:/cloudiq_churn_predictor@champion` and
  `models:/cloudiq_demand_forecast@champion` (C17).
- API startup loads `gold/demand_history` (not `gold/demand_features`)
  as the category-history source for recursive inference (C18).
- `POST /predict/demand` uses the requested category's actual four-month
  rolling history queue from `gold/demand_history` and forecasts
  recursively for `horizon_months > 1` using the exact queue-update
  procedure in C10. First predicted month is 2018-10. Returns HTTP 422
  for insufficient history and HTTP 404 for unknown categories (C10).

**Gate — PASS when:**
- `pytest tests/test_api.py` passes all tests.
- API tests mock `models:/cloudiq_churn_predictor@champion` and
  `models:/cloudiq_demand_forecast@champion` and pass (C17).
- `GET /health` response includes `name`, `version`, and `alias` for
  each loaded model (C17).
- `GET /models` response includes `name`, `version`, and `alias` for
  each model (C17).
- API startup loads `gold/demand_history` as the category-history source
  (verified by test) (C18).
- `docker compose up` starts both `api` and `mlflow` services.
- `GET http://localhost:8000/health` returns `{"status": "healthy"}`.
- `POST /predict/demand` with a category having < 4 months history
  returns HTTP 422 (verified by test) (C10).
- `POST /predict/demand` with an unknown category returns HTTP 404
  (verified by test) (C10).
- `ruff` and `pytest` pass.

---

## Phase 9 — GitLab CI

**Inputs:** All source files from Phases 0–8.

**Deliverables:**
- The existing `.gitlab-ci.yml` is **modified** (not replaced) to add
  `lint`, `test`, `build`, and `deploy` stages while preserving the
  existing `secret-detection` stage, the `SECRET_DETECTION_ENABLED`
  variable, and the `Security/Secret-Detection.gitlab-ci.yml` template
  include. The final `stages:` list must be:
  `[lint, test, build, deploy, secret-detection]` (C6).

**Corrections applied:**
- Uses GitLab CI syntax and `$CI_REGISTRY_IMAGE` throughout (C6).
- No `.github/` directory is created (C6).
- Existing Secret Detection configuration is preserved exactly (C6).
- `lint` stage includes `python scripts/check_demo_imports.py` (C11).
- `test` stage uses `pytest --cov=src --cov-report=xml`; `pytest-cov` is
  present in `requirements.txt` (C13).

**Gate — PASS when:**
- Pipeline runs green on GitLab for a push to `main`.
- `lint` stage: `ruff` exits 0 and `check_demo_imports.py` exits 0.
- `test` stage: `pytest` exits 0; coverage report uploaded as artifact.
- `build` stage: Docker image pushed to GitLab Container Registry.
- `secret-detection` stage: still present and passing.
- No GitHub Actions files exist anywhere in the repository.

---

## Phase 10 — Azure and Databricks (Optional)

**Inputs:** Working local pipeline and Docker image from Phases 1–9.

**Deliverables:**
- `azure/arm_template.json`
- `workflows/databricks_workflow.json`
- Deployment scripts calling `ConfigLoader.validate(strict=True)`.

**Corrections applied:**
- `ConfigLoader.validate(strict=True)` is called here, not in local
  phases (C7).
- ARM template references the GitLab Container Registry image, not GHCR
  (C6).

**Gate — PASS when:**
- `az deployment group create` completes without error.
- Databricks workflow runs all four tasks successfully.
- `ConfigLoader.validate(strict=True)` passes with all Azure env vars set.
