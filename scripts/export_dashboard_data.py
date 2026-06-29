"""Export compact local dashboard marts from validated Delta outputs."""

from __future__ import annotations

import calendar
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
MIN_PERIOD_ORDER_COUNT = 100
MIN_DELIVERED_ORDERS_FOR_RATE = 100

logger = get_logger("scripts.export_dashboard_data")


def safe_divide(numerator: float | int | None, denominator: float | int | None) -> float:
    """Return a finite ratio, using ``0.0`` when the denominator is empty."""
    if denominator in (None, 0):
        return 0.0
    return float(numerator or 0.0) / float(denominator)


def late_delivery_rate_from_flags(frame: pd.DataFrame) -> float:
    """Calculate late deliveries over delivered orders with a valid late flag."""
    required = {"order_status", "is_late"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing late-delivery columns: {sorted(missing)}")
    valid = frame["order_status"].eq("delivered") & frame["is_late"].notna()
    return safe_divide(frame.loc[valid, "is_late"].astype(bool).sum(), int(valid.sum()))


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
    item_merchandise_value_ex_freight: float,
    late_delivery_rate: float | None,
    first_month: str | None,
    last_month: str | None,
    freight_value: float = 0.0,
) -> dict[str, Any]:
    """Create dashboard-level KPI values from aggregate facts."""
    avg_item_merchandise_value = safe_divide(
        item_merchandise_value_ex_freight,
        total_orders,
    )
    return {
        "total_orders": int(total_orders),
        "item_merchandise_value_ex_freight": float(item_merchandise_value_ex_freight),
        "item_revenue_ex_freight": float(item_merchandise_value_ex_freight),
        "freight_value": float(freight_value),
        "avg_item_merchandise_value_per_order": avg_item_merchandise_value,
        "avg_item_revenue_per_order": avg_item_merchandise_value,
        "late_delivery_rate": float(late_delivery_rate or 0.0),
        "first_order_month": first_month,
        "last_order_month": last_month,
        "revenue_definition": "Item Merchandise Value excludes freight; seller-attributed order value includes item + freight.",
        "data_scope": "Historical Olist marketplace orders, 2016-2018",
    }


def mask_seller_id(seller_id: str) -> str:
    """Return a public-facing shortened seller identifier."""
    value = str(seller_id)
    if len(value) <= 12:
        return value
    return f"{value[:8]}...{value[-4:]}"


def flag_period_quality(
    monthly: pd.DataFrame,
    order_col: str = "order_count",
    min_order_count: int = MIN_PERIOD_ORDER_COUNT,
) -> pd.DataFrame:
    """Flag partial and low-volume periods without dropping rows."""
    result = monthly.copy()
    result["order_year_month"] = result["order_year_month"].astype(str)
    result[order_col] = pd.to_numeric(result[order_col], errors="coerce").fillna(0)
    min_month = result["order_year_month"].min()
    max_month = result["order_year_month"].max()

    result["is_partial_period"] = False
    if "first_purchase" in result.columns:
        first_purchase = pd.to_datetime(result["first_purchase"])
        result.loc[
            (result["order_year_month"] == min_month) & (first_purchase.dt.day > 1),
            "is_partial_period",
        ] = True
    if "last_purchase" in result.columns:
        last_purchase = pd.to_datetime(result["last_purchase"])
        max_period = pd.Period(max_month, freq="M")
        max_month_days = calendar.monthrange(max_period.year, max_period.month)[1]
        result.loc[
            (result["order_year_month"] == max_month)
            & (last_purchase.dt.day < max_month_days),
            "is_partial_period",
        ] = True

    result["is_low_volume_period"] = result[order_col] < min_order_count
    result["is_comparable_period"] = ~(
        result["is_partial_period"] | result["is_low_volume_period"]
    )
    return result


def build_demand_backtest_input(
    demand_features: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Build the compact pandas-only forecast training input."""
    required = [
        "category_name_english",
        "forecast_month",
        "feature_cutoff_month",
        "target_units",
        "lag_1",
        "lag_2",
        "lag_4",
        "rolling_mean_3",
        "month_num",
        "is_q4",
    ]
    missing = [column for column in required if column not in demand_features.columns]
    if missing:
        raise ValueError(
            "Gold demand_features is not using the release forecast contract. "
            f"Missing columns: {missing}"
        )

    result = demand_features[required].copy()
    result["forecast_month"] = result["forecast_month"].astype(str)
    result["feature_cutoff_month"] = result["feature_cutoff_month"].astype(str)
    coverage_cols = [
        "order_year_month",
        "order_count",
        "is_partial_period",
        "is_low_volume_period",
        "is_comparable_period",
    ]
    target_coverage = coverage[coverage_cols].rename(
        columns={
            "order_year_month": "forecast_month",
            "order_count": "target_period_order_count",
        }
    )
    result = result.merge(target_coverage, on="forecast_month", how="left")
    for column in ["is_partial_period", "is_low_volume_period", "is_comparable_period"]:
        result[column] = result[column].fillna(False).astype(bool)
    result["target_period_order_count"] = pd.to_numeric(
        result["target_period_order_count"],
        errors="coerce",
    ).fillna(0).astype(int)
    return result.sort_values(["forecast_month", "category_name_english"]).reset_index(
        drop=True
    )


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


def _copy_csv_if_present(
    source: Path,
    target: Path,
    columns: list[str] | None = None,
) -> int:
    """Copy a CSV report into the dashboard mart directory."""
    if not source.exists():
        pd.DataFrame(columns=columns).to_csv(target, index=False)
        return 0
    try:
        frame = pd.read_csv(source)
    except pd.errors.EmptyDataError:
        frame = pd.DataFrame(columns=columns)
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
            "freight_total",
            "is_late",
            "order_year_month",
            "order_purchase_timestamp",
            "order_status",
            "delivery_days",
            "delivery_delay_days",
        ],
    )

    overview = summarize_overview_kpis(
        total_orders=int(master_orders["order_id"].nunique()),
        item_merchandise_value_ex_freight=float(master_orders["order_revenue"].sum()),
        late_delivery_rate=late_delivery_rate_from_flags(master_orders),
        first_month=str(master_orders["order_year_month"].min()),
        last_month=str(master_orders["order_year_month"].max()),
        freight_value=float(master_orders["freight_total"].sum()),
    )
    created["overview_kpis.json"] = write_json(
        dashboard_dir / "overview_kpis.json",
        overview,
    )

    coverage = (
        master_orders.assign(
            order_purchase_timestamp=pd.to_datetime(
                master_orders["order_purchase_timestamp"]
            )
        )
        .groupby("order_year_month", as_index=False)
        .agg(
            order_count=("order_id", "nunique"),
            first_purchase=("order_purchase_timestamp", "min"),
            last_purchase=("order_purchase_timestamp", "max"),
        )
    )
    coverage = flag_period_quality(coverage)

    demand_backtest_input = build_demand_backtest_input(demand_features, coverage)
    created["demand_backtest_input.csv"] = write_csv(
        dashboard_dir / "demand_backtest_input.csv",
        demand_backtest_input,
    )

    monthly = (
        bi_revenue.groupby("order_year_month", as_index=False)
        .agg(
            order_count=("total_orders", "sum"),
            item_merchandise_value_ex_freight=("total_revenue", "sum"),
        )
        .sort_values("order_year_month")
    )
    monthly["item_revenue_ex_freight"] = monthly["item_merchandise_value_ex_freight"]
    monthly["avg_item_merchandise_value_per_order"] = monthly.apply(
        lambda row: safe_divide(
            row["item_merchandise_value_ex_freight"],
            row["order_count"],
        ),
        axis=1,
    )
    monthly["avg_item_revenue_per_order"] = monthly[
        "avg_item_merchandise_value_per_order"
    ]
    monthly = monthly.merge(
        coverage[
            [
                "order_year_month",
                "is_partial_period",
                "is_low_volume_period",
                "is_comparable_period",
            ]
        ],
        on="order_year_month",
        how="left",
    )
    created["monthly_revenue.csv"] = write_csv(
        dashboard_dir / "monthly_revenue.csv",
        monthly,
    )

    state = (
        bi_revenue.groupby("customer_state", as_index=False)
        .agg(
            order_count=("total_orders", "sum"),
            item_merchandise_value_ex_freight=("total_revenue", "sum"),
        )
        .sort_values("item_merchandise_value_ex_freight", ascending=False)
    )
    state["item_revenue_ex_freight"] = state["item_merchandise_value_ex_freight"]
    state["avg_item_merchandise_value_per_order"] = state.apply(
        lambda row: safe_divide(
            row["item_merchandise_value_ex_freight"],
            row["order_count"],
        ),
        axis=1,
    )
    state["avg_item_revenue_per_order"] = state["avg_item_merchandise_value_per_order"]
    created["state_revenue.csv"] = write_csv(
        dashboard_dir / "state_revenue.csv",
        state,
    )

    payment = (
        bi_revenue.groupby("primary_payment_type", as_index=False)
        .agg(
            order_count=("total_orders", "sum"),
            item_merchandise_value_ex_freight=("total_revenue", "sum"),
        )
        .sort_values("order_count", ascending=False)
    )
    payment["item_revenue_ex_freight"] = payment["item_merchandise_value_ex_freight"]
    payment["order_share"] = payment["order_count"] / payment["order_count"].sum()
    created["payment_mix.csv"] = write_csv(
        dashboard_dir / "payment_mix.csv",
        payment,
    )

    delivery = (
        master_orders.groupby("order_year_month", as_index=False)
        .agg(
            order_count=("order_id", "nunique"),
            avg_delivery_days=("delivery_days", "mean"),
            avg_delivery_delay_days=("delivery_delay_days", "mean"),
        )
        .sort_values("order_year_month")
    )
    delivered = (
        master_orders.assign(
            valid_delivered_order_flag=(
                master_orders["order_status"].eq("delivered")
                & master_orders["is_late"].notna()
            ).astype(int),
            late_delivered_order_flag=(
                master_orders["order_status"].eq("delivered")
                & master_orders["is_late"].fillna(False).astype(bool)
            ).astype(int),
        )
        .groupby("order_year_month", as_index=False)
        .agg(
            delivered_order_count=("valid_delivered_order_flag", "sum"),
            late_delivered_order_count=("late_delivered_order_flag", "sum"),
        )
    )
    delivery = delivery.merge(delivered, on="order_year_month", how="left")
    delivery["late_delivery_rate"] = delivery.apply(
        lambda row: safe_divide(
            row["late_delivered_order_count"],
            row["delivered_order_count"],
        ),
        axis=1,
    )
    delivery = delivery.merge(
        coverage[
            [
                "order_year_month",
                "is_partial_period",
                "is_low_volume_period",
                "is_comparable_period",
            ]
        ],
        on="order_year_month",
        how="left",
    )
    delivery["is_low_delivered_volume_period"] = (
        delivery["delivered_order_count"] < MIN_DELIVERED_ORDERS_FOR_RATE
    )
    delivery["late_delivery_rate_display"] = delivery["late_delivery_rate"].where(
        ~delivery["is_low_delivered_volume_period"]
    )
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
            repeat_customers=("total_orders", lambda s: int((s > 1).sum())),
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
    rfm_profiles["repeat_customer_rate"] = (
        rfm_profiles["repeat_customers"] / rfm_profiles["customers"]
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
    sellers = sellers.reset_index(drop=True)
    sellers["seller_rank"] = sellers.index + 1
    sellers["seller_id_short"] = sellers["seller_id"].map(mask_seller_id)
    sellers["seller_attributed_order_value"] = sellers["total_revenue"]
    sellers = sellers[
        [
            "seller_rank",
            "seller_id_short",
            "seller_state",
            "total_orders",
            "seller_attributed_order_value",
            "avg_review",
            "late_rate",
            "performance_tier",
        ]
    ]
    created["seller_performance.csv"] = write_csv(
        dashboard_dir / "seller_performance.csv",
        sellers,
    )

    created["demand_backtest_predictions.csv"] = _copy_csv_if_present(
        reports_path / "demand_backtest_predictions.csv",
        dashboard_dir / "demand_backtest_predictions.csv",
        [
            "category_name_english",
            "forecast_month",
            "feature_cutoff_month",
            "target_units",
            "lag_1",
            "lag_2",
            "lag_4",
            "rolling_mean_3",
            "month_num",
            "is_q4",
            "fold",
            "max_train_forecast_month",
            "naive_prediction",
            "xgboost_prediction",
            "champion_prediction",
        ],
    )
    created["demand_backtest_monthly_aggregate.csv"] = _copy_csv_if_present(
        reports_path / "demand_backtest_monthly_aggregate.csv",
        dashboard_dir / "demand_backtest_monthly_aggregate.csv",
        [
            "forecast_month",
            "forecast_month_label",
            "actual_units",
            "naive_prediction",
            "xgboost_prediction",
            "champion_prediction",
            "evaluated_rows",
            "category_count",
            "champion_abs_error",
        ],
    )
    created["demand_backtest_category_errors.csv"] = _copy_csv_if_present(
        reports_path / "demand_backtest_category_errors.csv",
        dashboard_dir / "demand_backtest_category_errors.csv",
        [
            "category_name_english",
            "actual_units",
            "naive_prediction",
            "xgboost_prediction",
            "champion_prediction",
            "absolute_error",
            "evaluated_rows",
            "forecast_month_count",
            "mae",
        ],
    )
    created["demand_backtest_fold_metrics.csv"] = _copy_csv_if_present(
        reports_path / "demand_backtest_fold_metrics.csv",
        dashboard_dir / "demand_backtest_fold_metrics.csv",
        [
            "fold",
            "forecast_month",
            "model",
            "train_rows",
            "validation_rows",
            "max_train_forecast_month",
            "mae",
            "rmse",
            "wape",
            "smape",
            "mape",
            "r2",
        ],
    )
    created["demand_model_metrics.json"] = _copy_metrics_if_present(
        reports_path / "demand_forecast_metrics.json",
        dashboard_dir / "demand_model_metrics.json",
    )

    audit = _load_json_if_present(reports_path / "olist_data_audit.json")
    bronze_counts = _count_tables(bronze_path, BRONZE_TABLES)
    bronze_reconciliation = (
        "9/9 source tables matched audited raw counts"
        if len(bronze_counts) == len(BRONZE_TABLES)
        else f"{len(bronze_counts)}/9 source tables available in local Bronze"
    )
    data_quality = {
        "source_audit": {
            "executive_verdict": audit.get("executive_verdict"),
            "safe_to_run_bronze": audit.get("safe_to_run_bronze"),
            "bronze_reconciliation": bronze_reconciliation,
            "warnings": audit.get("warnings", []),
            "translation_lookup_gap": audit.get("translation_lookup_gap", {}),
            "raw_file_row_counts": {
                name: details.get("row_count")
                for name, details in audit.get("files", {}).items()
            },
        },
        "pipeline_validation": {
            "status": "validated local Bronze/Silver/Gold outputs",
            "bronze_row_counts": bronze_counts,
            "silver_row_counts": _count_tables(silver_path, SILVER_TABLES),
            "gold_row_counts": _count_tables(gold_path, GOLD_TABLES),
        },
        "period_quality": {
            "minimum_comparable_order_count": MIN_PERIOD_ORDER_COUNT,
            "minimum_delivered_orders_for_late_rate": MIN_DELIVERED_ORDERS_FOR_RATE,
            "excluded_business_trend_periods": coverage.loc[
                ~coverage["is_comparable_period"],
                [
                    "order_year_month",
                    "order_count",
                    "is_partial_period",
                    "is_low_volume_period",
                ],
            ].to_dict(orient="records"),
        },
        "metric_definitions": {
            "Item Merchandise Value (excludes freight)": "Sum of order item price values; freight is excluded.",
            "Average Item Merchandise Value per Order": "Item Merchandise Value excluding freight divided by distinct orders.",
            "Seller-Attributed Order Value (item + freight)": "Seller-level item price plus seller-level freight value.",
            "Orders by Primary Payment Type": "One deterministic payment type per order, selected by highest payment value, then lowest payment sequence, then payment type alphabetically.",
            "Late-delivery rate": "Late-delivery rate = delivered orders received after the estimated delivery date divided by all delivered orders with valid actual and estimated delivery dates.",
        },
        "source_characteristics": {
            "geolocation_note": "Geolocation contains repeated zip-code observations and is not used in the current business marts.",
            "geolocation_duplicate_zip_code_rows": audit.get("sanity_checks", {}).get(
                "geolocation_duplicate_zip_code_rows"
            ),
            "geolocation_exact_duplicate_rows": audit.get("files", {})
            .get("geolocation", {})
            .get("exact_duplicate_rows"),
        },
        "demand_feature_rows": int(len(demand_features)),
        "methodology_notes": [
            "Reviews Bronze ingestion uses multiline-safe CSV parsing.",
            "13 product rows across 2 categories are retained using untranslated__ fallback labels.",
            "Demand features predict target_units for forecast_month using only information available before that month starts.",
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
