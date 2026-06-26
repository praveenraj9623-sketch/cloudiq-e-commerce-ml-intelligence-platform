"""Export compact local dashboard marts from validated Delta outputs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import ConfigLoader  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.spark_session import get_spark_session  # noqa: E402

JAVA_TOOL_OPTIONS = (
    "-XX:ActiveProcessorCount=2 "
    "-XX:CICompilerCount=2 "
    "-XX:TieredStopAtLevel=1 "
    "-Xss512k"
)

BRONZE_TABLES = [
    "customers",
    "geolocation",
    "order_items",
    "payments",
    "reviews",
    "orders",
    "products",
    "sellers",
    "category_translation",
]
SILVER_TABLES = [
    "orders",
    "order_items",
    "master_orders",
    "customer_profile",
    "product_demand",
    "seller_performance",
]
GOLD_TABLES = [
    "demand_features",
    "churn_features",
    "rfm_segments",
    "bi_revenue",
]

logger = get_logger("scripts.export_dashboard_data")


def safe_divide(numerator: float | int | None, denominator: float | int | None) -> float:
    """Return a finite ratio, using ``0.0`` when the denominator is empty."""
    if denominator in (None, 0):
        return 0.0
    return float(numerator or 0.0) / float(denominator)


def _json_default(value: Any) -> Any:
    """Serialize Spark, pandas, and numeric scalar values for JSON output."""
    if hasattr(value, "asDict"):
        return value.asDict(recursive=True)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> int:
    """Write a JSON payload and return its logical row count."""
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return 1


def write_csv(path: Path, frame: pd.DataFrame) -> int:
    """Write a CSV file and return its row count."""
    frame.to_csv(path, index=False)
    return int(len(frame))


def summarize_overview_kpis(
    total_orders: int,
    total_revenue: float,
    late_delivery_rate: float | None,
    first_month: str | None,
    last_month: str | None,
) -> dict[str, Any]:
    """Create dashboard-level KPI values from aggregate facts."""
    return {
        "total_orders": int(total_orders),
        "total_revenue": float(total_revenue),
        "avg_order_value": safe_divide(total_revenue, total_orders),
        "late_delivery_rate": float(late_delivery_rate or 0.0),
        "first_order_month": first_month,
        "last_order_month": last_month,
        "data_scope": "Historical Olist marketplace orders, 2016-2018",
    }


def _active_delta_adds(delta_path: Path) -> list[dict[str, Any]]:
    """Return active Delta add actions by replaying JSON logs."""
    log_dir = delta_path / "_delta_log"
    if not log_dir.exists():
        raise FileNotFoundError(f"Delta log not found: {log_dir}")

    active: dict[str, dict[str, Any]] = {}
    for log_file in sorted(log_dir.glob("*.json")):
        with log_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                action = json.loads(line)
                if "add" in action:
                    add = action["add"]
                    active[add["path"]] = add
                elif "remove" in action:
                    active.pop(action["remove"]["path"], None)
    return [active[path] for path in sorted(active)]


def _read_delta_pandas(delta_path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read active files from a local Delta table into pandas."""
    import pyarrow.parquet as pq

    frames: list[pd.DataFrame] = []
    for add in _active_delta_adds(delta_path):
        partition_values = add.get("partitionValues", {})
        file_columns = (
            [column for column in columns if column not in partition_values]
            if columns is not None
            else None
        )
        parquet_path = delta_path / Path(unquote(add["path"]))
        frame = pq.ParquetFile(parquet_path).read(columns=file_columns).to_pandas()
        for key, value in partition_values.items():
            if columns is None or key in columns:
                frame[key] = value
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=columns)
    result = pd.concat(frames, ignore_index=True)
    return result[columns] if columns is not None else result


def _load_json_if_present(path: Path) -> dict[str, Any]:
    """Read a JSON file when present."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_csv_if_present(source: Path, target: Path) -> int:
    """Copy a CSV report into the dashboard mart directory."""
    if not source.exists():
        pd.DataFrame().to_csv(target, index=False)
        return 0
    frame = pd.read_csv(source)
    frame.to_csv(target, index=False)
    return int(len(frame))


def _copy_metrics_if_present(source: Path, target: Path) -> int:
    """Copy model metrics JSON into the dashboard mart directory."""
    payload = _load_json_if_present(source)
    if not payload:
        payload = {"status": "missing", "source": str(source)}
    return write_json(target, payload)


def _delta_row_count(delta_path: Path) -> int:
    """Return the active row count for a Delta table using file stats."""
    import pyarrow.parquet as pq

    total = 0
    for add in _active_delta_adds(delta_path):
        stats = json.loads(add.get("stats", "{}"))
        if "numRecords" in stats:
            total += int(stats["numRecords"])
        else:
            parquet_path = delta_path / Path(unquote(add["path"]))
            total += int(pq.ParquetFile(parquet_path).metadata.num_rows)
    return total


def _count_tables(base_path: Path, tables: list[str]) -> dict[str, int]:
    """Count local Delta tables that exist under a layer path."""
    counts: dict[str, int] = {}
    for table in tables:
        table_path = base_path / table
        if table_path.exists():
            counts[table] = _delta_row_count(table_path)
    return counts


def _start_and_stop_spark(config: ConfigLoader) -> None:
    """Verify one local Spark session can start, then stop it immediately."""
    spark: SparkSession | None = None
    try:
        config._config["spark"]["driver_memory"] = "768m"  # noqa: SLF001
        config._config["spark"]["executor_memory"] = "768m"  # noqa: SLF001
        config._config["spark"]["shuffle_partitions"] = 1  # noqa: SLF001
        spark = get_spark_session(config, app_name="CloudIQ-Dashboard-Export")
        logger.info("Spark session verified for dashboard export.")
    finally:
        if spark is not None:
            spark.stop()


def export_dashboard_data() -> dict[str, int]:
    """Export all compact dashboard marts and return file row counts."""
    os.environ.setdefault("JAVA_TOOL_OPTIONS", JAVA_TOOL_OPTIONS)
    config = ConfigLoader(str(ROOT / "config.yaml"), env_path=str(ROOT / ".env"))
    dashboard_dir = ROOT / "data" / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    created: dict[str, int] = {}
    gold_path = config.get_path("paths.gold", create=False)
    silver_path = config.get_path("paths.silver", create=False)
    bronze_path = config.get_path("paths.bronze", create=False)
    reports_path = config.get_path("paths.reports", create=False)

    bi_revenue = _read_delta_pandas(gold_path / "bi_revenue")
    rfm_segments = _read_delta_pandas(gold_path / "rfm_segments")
    demand_features = _read_delta_pandas(gold_path / "demand_features")
    seller_performance = _read_delta_pandas(silver_path / "seller_performance")
    master_orders = _read_delta_pandas(
        silver_path / "master_orders",
        [
            "order_id",
            "order_revenue",
            "is_late",
            "order_year_month",
            "order_status",
            "delivery_days",
            "delivery_delay_days",
        ],
    )

    overview = summarize_overview_kpis(
        total_orders=int(master_orders["order_id"].nunique()),
        total_revenue=float(master_orders["order_revenue"].sum()),
        late_delivery_rate=float(master_orders["is_late"].mean()),
        first_month=str(master_orders["order_year_month"].min()),
        last_month=str(master_orders["order_year_month"].max()),
    )
    created["overview_kpis.json"] = write_json(
        dashboard_dir / "overview_kpis.json",
        overview,
    )

    monthly = (
        bi_revenue.groupby("order_year_month", as_index=False)
        .agg(total_orders=("total_orders", "sum"), total_revenue=("total_revenue", "sum"))
        .sort_values("order_year_month")
    )
    monthly["avg_order_value"] = monthly.apply(
        lambda row: safe_divide(row["total_revenue"], row["total_orders"]),
        axis=1,
    )
    created["monthly_revenue.csv"] = write_csv(
        dashboard_dir / "monthly_revenue.csv",
        monthly,
    )

    state = (
        bi_revenue.groupby("customer_state", as_index=False)
        .agg(total_orders=("total_orders", "sum"), total_revenue=("total_revenue", "sum"))
        .sort_values("total_revenue", ascending=False)
    )
    state["avg_order_value"] = state.apply(
        lambda row: safe_divide(row["total_revenue"], row["total_orders"]),
        axis=1,
    )
    created["state_revenue.csv"] = write_csv(
        dashboard_dir / "state_revenue.csv",
        state,
    )

    payment = (
        bi_revenue.groupby("primary_payment_type", as_index=False)
        .agg(total_orders=("total_orders", "sum"), total_revenue=("total_revenue", "sum"))
        .sort_values("total_orders", ascending=False)
    )
    payment["order_share"] = payment["total_orders"] / payment["total_orders"].sum()
    created["payment_mix.csv"] = write_csv(
        dashboard_dir / "payment_mix.csv",
        payment,
    )

    delivery = (
        master_orders.groupby("order_year_month", as_index=False)
        .agg(
            total_orders=("order_id", "nunique"),
            late_delivery_rate=("is_late", "mean"),
            avg_delivery_days=("delivery_days", "mean"),
            avg_delivery_delay_days=("delivery_delay_days", "mean"),
        )
        .sort_values("order_year_month")
    )
    delivered = (
        master_orders.assign(
            delivered_order_flag=master_orders["order_status"].eq("delivered").astype(int)
        )
        .groupby("order_year_month", as_index=False)
        .agg(delivered_orders=("delivered_order_flag", "sum"))
    )
    delivery = delivery.merge(delivered, on="order_year_month", how="left")
    created["delivery_performance.csv"] = write_csv(
        dashboard_dir / "delivery_performance.csv",
        delivery,
    )

    rfm_distribution = (
        rfm_segments.groupby("segment_label", as_index=False)
        .size()
        .rename(columns={"size": "customers"})
        .sort_values("customers", ascending=False)
    )
    created["rfm_segment_distribution.csv"] = write_csv(
        dashboard_dir / "rfm_segment_distribution.csv",
        rfm_distribution,
    )

    rfm_profiles = (
        rfm_segments.groupby("segment_label", as_index=False)
        .agg(
            customers=("customer_unique_id", "count"),
            avg_recency_days=("recency_days", "mean"),
            avg_total_orders=("total_orders", "mean"),
            avg_total_revenue=("total_revenue", "mean"),
            avg_order_value=("avg_order_value", "mean"),
            avg_review_score=("avg_review_score", "mean"),
            late_delivery_rate=("late_delivery_rate", "mean"),
            avg_rfm_total=("rfm_total", "mean"),
        )
        .sort_values("avg_rfm_total", ascending=False)
    )
    created["rfm_segment_profiles.csv"] = write_csv(
        dashboard_dir / "rfm_segment_profiles.csv",
        rfm_profiles,
    )

    sellers = seller_performance[
        [
            "seller_id",
            "seller_state",
            "total_orders",
            "total_revenue",
            "avg_review",
            "late_rate",
            "performance_tier",
        ]
    ].sort_values("total_revenue", ascending=False)
    created["seller_performance.csv"] = write_csv(
        dashboard_dir / "seller_performance.csv",
        sellers,
    )

    created["demand_validation_predictions.csv"] = _copy_csv_if_present(
        reports_path / "demand_forecast_validation_predictions.csv",
        dashboard_dir / "demand_validation_predictions.csv",
    )
    created["demand_model_metrics.json"] = _copy_metrics_if_present(
        reports_path / "demand_forecast_metrics.json",
        dashboard_dir / "demand_model_metrics.json",
    )

    audit = _load_json_if_present(reports_path / "olist_data_audit.json")
    data_quality = {
        "source_audit": {
            "executive_verdict": audit.get("executive_verdict"),
            "safe_to_run_bronze": audit.get("safe_to_run_bronze"),
            "warnings": audit.get("warnings", []),
            "translation_lookup_gap": audit.get("translation_lookup_gap", {}),
            "raw_file_row_counts": {
                name: details.get("row_count")
                for name, details in audit.get("files", {}).items()
            },
        },
        "pipeline_validation": {
            "status": "validated local Bronze/Silver/Gold outputs",
            "bronze_row_counts": _count_tables(bronze_path, BRONZE_TABLES),
            "silver_row_counts": _count_tables(silver_path, SILVER_TABLES),
            "gold_row_counts": _count_tables(gold_path, GOLD_TABLES),
        },
        "demand_feature_rows": int(len(demand_features)),
        "methodology_notes": [
            "Reviews Bronze ingestion uses multiline-safe CSV parsing.",
            "Unmatched product category translations are preserved with untranslated__ fallbacks.",
            "Demand features use prior-only lags and target the next month's observed units.",
            "Churn classification is not trained because snapshot inactivity is 99.25%.",
        ],
    }
    created["data_quality_summary.json"] = write_json(
        dashboard_dir / "data_quality_summary.json",
        data_quality,
    )

    # Keep a tiny manifest for humans without requiring Streamlit to infer files.
    created["export_manifest.json"] = write_json(
        dashboard_dir / "export_manifest.json",
        {"files": created},
    )

    _start_and_stop_spark(config)

    return created


def main() -> int:
    """Run dashboard export and print file row counts."""
    created = export_dashboard_data()
    print("Dashboard export complete:")
    for filename, rows in created.items():
        print(f"- {filename}: {rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
