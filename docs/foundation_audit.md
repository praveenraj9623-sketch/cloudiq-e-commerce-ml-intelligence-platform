# CloudIQ Foundation Audit

Date: 2026-06-26

## Current project structure

```text
.
|-- .gitlab-ci.yml
|-- AGENTS.md
|-- README.md
|-- config.yaml
|-- data/
|   |-- raw/
|   |-- bronze/
|   |-- silver/
|   `-- gold/
|-- docs/
|-- logs/
|-- models/
|-- reports/
|-- requirements.txt
|-- run_pipeline.py
|-- scripts/
|   `-- verify_foundation.py
|-- src/
|   |-- __init__.py
|   |-- processing/
|   |   |-- __init__.py
|   |   |-- bronze.py
|   |   |-- silver.py
|   |   `-- gold.py
|   `-- utils/
|       |-- __init__.py
|       |-- config.py
|       |-- logger.py
|       `-- spark_session.py
`-- tests/
    |-- __init__.py
    |-- conftest.py
    |-- test_config.py
    |-- test_demand_features.py
    `-- test_pipeline_smoke.py
```

## Files already implemented

- `config.yaml`: local project, Spark, path, model, and Olist file settings.
- `requirements.txt`: pinned local development dependencies, including the required core pins.
- `run_pipeline.py`: CLI runner for bronze, silver, gold, or all local layers.
- `src/utils/config.py`: `ConfigLoader` with `${VAR}` and `${VAR:-default}` resolution, dot notation access, validation, and `get_path()`.
- `src/utils/logger.py`: Loguru logger factory with console and rotating file sinks guarded against duplicate configuration.
- `src/utils/spark_session.py`: Spark session factory with Delta extension, Delta catalog, adaptive execution, Windows Hadoop home detection, and memory/shuffle settings from `config.yaml`.
- `src/processing/bronze.py`: raw CSV to Bronze Delta ingestion.
- `src/processing/silver.py`: Silver cleaning, joins, customer/product/seller tables.
- `src/processing/gold.py`: Gold churn, demand, RFM, and BI feature tables.
- `tests/conftest.py`: adds the repository root to `sys.path`.
- `tests/test_config.py`: ConfigLoader unit coverage.
- `tests/test_pipeline_smoke.py`: import and public surface smoke tests that do not start Spark.
- `tests/test_demand_features.py`: pure-Python demand feature semantics tests.

## Missing files or directories found

Before this audit, the following required local scaffold directories were missing:

- `scripts/`
- `reports/`
- `models/`

They were created. The required package markers already existed:

- `src/__init__.py`
- `src/utils/__init__.py`
- `src/processing/__init__.py`
- `tests/__init__.py`

The required local data/log directories already existed:

- `logs/`
- `data/raw/`
- `data/bronze/`
- `data/silver/`
- `data/gold/`

## Broken imports or package issues

- No broken imports were found during inspection of the local package layout.
- `scripts/verify_foundation.py` inserts the repository root into `sys.path` so `python scripts/verify_foundation.py` can import `src.*` modules when executed from the repository root.
- Active Python is 3.11.9.
- `requirements.txt` pins the requested core dependency versions.
- The active environment has version drift from those pins:
  - `pyspark` is installed as 3.5.8, but `requirements.txt` pins 3.5.0.
  - `rich` is installed as 13.9.4, but `requirements.txt` pins 13.7.1.

## Delta and Spark configuration risks

- Delta session creation depends on `configure_spark_with_delta_pip(builder).getOrCreate()`, which resolves the Delta Lake JAR when Spark starts. A missing Java runtime, blocked Maven access, or empty Ivy cache can break first-run local startup.
- On Windows, local Spark needs `winutils.exe` and the matching native Hadoop DLLs. `src/utils/spark_session.py` now detects existing `C:/hadoop` or `C:/tools/hadoop` installs, sets `HADOOP_HOME`/`hadoop.home.dir`, prepends `bin` to `PATH`, and passes `java.library.path` before Spark starts.
- Local generated Delta folders under `data/` must stay uncommitted. The smoke output path `data/_smoke_delta/` is now ignored explicitly.
- Existing generated local Delta tables under `data/bronze/`, `data/silver/`, and `data/gold/` are ignored by `.gitignore`.
- The Spark builder keeps the required Delta SQL extension and catalog settings:
  - `spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension`
  - `spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog`
- The Spark builder keeps adaptive execution enabled and uses `config.yaml` for driver memory, executor memory, and shuffle partitions.
- Runtime package drift should be reconciled with `requirements.txt` before relying on environment-specific behavior.
- Spark emits non-fatal Windows cleanup warnings while deleting temporary copied JAR files after `spark.stop()`. The process still exits successfully and the smoke row-count check passes.

## Validation results

- `python -m ruff check .`: passed.
- `python -m pytest tests -q`: passed, 16 tests.
- `python scripts/verify_foundation.py`: passed, wrote and read 2 rows from `data/_smoke_delta/`.

## Exact changes made

- Created the missing local scaffold directories: `scripts/`, `reports/`, and `models/`.
- Added `scripts/verify_foundation.py` to load `config.yaml`, start Spark, write a two-row Delta table to `data/_smoke_delta/`, read it back, check the row count, print `PASS` or `FAIL`, and always stop Spark when it was started.
- Updated `.gitignore` to ignore `data/_smoke_delta/` and root or nested `*.log` files.
- Updated `tests/test_config.py` with focused placeholder coverage for embedded `${VAR:-default}` resolution and made the default-resolution test independent of any ambient `MLFLOW_TRACKING_URI`.
- Updated `tests/conftest.py` with a repo-local `tmp_path` fixture because the default Windows pytest temp root was inaccessible in this environment.
- Updated `tests/test_demand_features.py` so its Spark-column mock supports the production `.alias("m")` call.
- Removed an unused `TimestampType` import from `src/processing/bronze.py` so Ruff passes without changing Bronze behavior.
- Updated `src/utils/spark_session.py` with Windows-only Hadoop home and native library detection for existing local `winutils.exe`/`hadoop.dll` installs, preserving the Delta builder and `configure_spark_with_delta_pip(builder).getOrCreate()` path.
- Added this audit document at `docs/foundation_audit.md`.
