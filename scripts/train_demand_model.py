"""Train a compact leakage-safe demand forecasting model.

The workflow consumes only ``gold/demand_features``. The target is the next
month's observed unit demand, and validation always uses the final two calendar
months in the feature table.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import ConfigLoader  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

TARGET_COLUMN = "target_next_month"
FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_4",
    "rolling_mean_3",
    "month_num",
    "is_q4",
]
EVALUATION_COLUMNS = ["category_name_english", "order_year_month"]
REQUIRED_COLUMNS = [*EVALUATION_COLUMNS, TARGET_COLUMN, *FEATURE_COLUMNS]
VALIDATION_MONTHS = 2
CHURN_LIMITATION = (
    "Churn classification was intentionally not trained because the snapshot "
    "churn table has a 99.25% inactivity rate, making it unsuitable as a "
    "flagship supervised classifier for this dataset."
)
XGB_PARAMS = {
    "n_estimators": 120,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": 1,
}

logger = get_logger("scripts.train_demand_model")


def zero_safe_mape(y_true: Any, y_pred: Any) -> float:
    """Return MAPE with finite behavior for zero actual values.

    Zero targets use a denominator of ``1.0``. This keeps the metric finite
    while still penalizing non-zero forecasts when actual demand is zero.
    """
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.size == 0:
        raise ValueError("Cannot calculate MAPE for an empty target array.")
    denominator = np.where(np.abs(true) < 1e-12, 1.0, np.abs(true))
    return float(np.mean(np.abs(true - pred) / denominator) * 100.0)


def compute_regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute standard validation metrics for demand forecasts."""
    from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(root_mean_squared_error(true, pred)),
        "mape": zero_safe_mape(true, pred),
        "r2": float(r2_score(true, pred)),
    }


def _month_keys(values: pd.Series) -> pd.Series:
    """Normalize month-like values to YYYY-MM strings."""
    periods = pd.PeriodIndex(values.astype(str), freq="M")
    return pd.Series(periods.astype(str), index=values.index)


def chronological_train_validation_split(
    df: pd.DataFrame,
    date_col: str = "order_year_month",
    validation_months: int = VALIDATION_MONTHS,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Split rows by time, reserving the final calendar months for validation."""
    if validation_months < 1:
        raise ValueError("validation_months must be at least 1.")
    if date_col not in df.columns:
        raise KeyError(f"Missing split column: {date_col}")

    work = df.copy()
    work["_month_key"] = _month_keys(work[date_col])
    ordered_months = sorted(work["_month_key"].dropna().unique().tolist())
    if len(ordered_months) <= validation_months:
        raise ValueError(
            "Need more unique months than the validation window for "
            "chronological splitting."
        )

    validation_values = ordered_months[-validation_months:]
    validation_mask = work["_month_key"].isin(validation_values)
    train = df.loc[~validation_mask].copy()
    validation = df.loc[validation_mask].copy()
    if train.empty or validation.empty:
        raise ValueError("Chronological split produced an empty train or validation set.")
    return train, validation, validation_values


def _active_delta_adds(delta_path: Path) -> list[dict[str, Any]]:
    """Return active Delta add actions by replaying JSON transaction logs."""
    log_dir = delta_path / "_delta_log"
    if not log_dir.exists():
        raise FileNotFoundError(f"Delta log not found: {log_dir}")

    active: dict[str, dict[str, Any]] = {}
    log_files = sorted(log_dir.glob("*.json"))
    if not log_files:
        raise FileNotFoundError(f"No Delta JSON logs found in: {log_dir}")

    for log_file in log_files:
        with log_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                action = json.loads(line)
                if "add" in action:
                    add = action["add"]
                    active[add["path"]] = add
                elif "remove" in action:
                    active.pop(action["remove"]["path"], None)

    if not active:
        raise ValueError(f"No active Delta files found for: {delta_path}")
    return [active[path] for path in sorted(active)]


def _read_gold_demand_features(config: ConfigLoader) -> pd.DataFrame:
    """Read Gold demand features into pandas from active Delta log files."""
    import pyarrow.parquet as pq

    gold_path = config.get_path("paths.gold", create=False)
    demand_path = gold_path / "demand_features"
    frames: list[pd.DataFrame] = []
    for add in _active_delta_adds(demand_path):
        parquet_path = demand_path / Path(unquote(add["path"]))
        frame = pq.read_table(parquet_path).to_pandas()
        for key, value in add.get("partitionValues", {}).items():
            frame[key] = value
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    missing = [column for column in REQUIRED_COLUMNS if column not in result.columns]
    if missing:
        raise ValueError(f"Missing required demand feature columns: {missing}")
    return result[REQUIRED_COLUMNS]


def _prepare_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce the demand feature table for modeling."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required demand feature columns: {missing}")

    prepared = df[REQUIRED_COLUMNS].copy()
    prepared["order_year_month"] = _month_keys(prepared["order_year_month"])
    prepared["category_name_english"] = prepared["category_name_english"].astype(str)
    for column in [TARGET_COLUMN, *FEATURE_COLUMNS]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    before_drop = len(prepared)
    prepared = prepared.dropna(subset=[TARGET_COLUMN, *FEATURE_COLUMNS])
    dropped = before_drop - len(prepared)
    if dropped:
        logger.warning("Dropped {} demand rows with null modeling fields.", dropped)
    if prepared.empty:
        raise ValueError("No usable demand feature rows remain after preparation.")

    return prepared.sort_values(["order_year_month", "category_name_english"])


def _train_xgboost(train: pd.DataFrame) -> "XGBRegressor":
    """Fit a modest XGBoost regressor for the small demand feature table."""
    from xgboost import XGBRegressor

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    return model


def _clip_non_negative(values: Any) -> np.ndarray:
    """Clip forecast outputs to the valid non-negative demand range."""
    return np.maximum(np.asarray(values, dtype=float), 0.0)


def _select_champion(
    naive_metrics: dict[str, float],
    xgb_metrics: dict[str, float],
) -> tuple[str, str]:
    """Choose XGBoost only when it improves validation MAE."""
    if xgb_metrics["mae"] < naive_metrics["mae"]:
        return "xgboost", "XGBoost improved validation MAE over the naive baseline."
    return (
        "naive_lag_1",
        "Naive lag_1 retained because XGBoost did not improve validation MAE.",
    )


def _save_model_artifact(
    model_path: Path,
    champion: str,
    xgb_model: "XGBRegressor",
    validation_months: list[str],
) -> None:
    """Persist the selected model artifact with metadata."""
    import joblib

    artifact: dict[str, Any] = {
        "model_type": champion,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "validation_months": validation_months,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_postprocessing": "clip predictions at zero",
    }
    if champion == "xgboost":
        artifact["model"] = xgb_model
    else:
        artifact["prediction_feature"] = "lag_1"
    joblib.dump(artifact, model_path)


def _save_validation_chart(predictions: pd.DataFrame, chart_path: Path) -> None:
    """Save a simple actual-vs-predicted validation chart."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        predictions[TARGET_COLUMN],
        predictions["champion_prediction"],
        alpha=0.7,
        edgecolor="none",
    )
    max_value = float(
        max(
            predictions[TARGET_COLUMN].max(),
            predictions["champion_prediction"].max(),
            1.0,
        )
    )
    ax.plot([0, max_value], [0, max_value], color="black", linewidth=1)
    ax.set_title("Demand Forecast Validation")
    ax.set_xlabel("Actual next-month units")
    ax.set_ylabel("Predicted next-month units")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)


def _write_modeling_notes(notes_path: Path) -> None:
    """Write compact modeling notes for the demand-only training phase."""
    notes_path.write_text(
        "\n".join(
            [
                "# CloudIQ Modeling Notes",
                "",
                "## Demand Forecast",
                "",
                "- Target: `target_next_month`, the next calendar month's observed "
                "unit demand for each `category_name_english` row in "
                "`gold/demand_features`.",
                "- Features: `lag_1`, `lag_2`, `lag_4`, `rolling_mean_3`, "
                "`month_num`, and `is_q4`.",
                "- Validation: chronological holdout using the final two available "
                "calendar months. No random row split is used.",
                "- Leakage statement: Gold demand features use prior-month lags and "
                "prior-only rolling windows; the model is trained to predict the "
                "next month's units from information available at the feature month.",
                "",
                "## Churn Limitation",
                "",
                "- Churn classification is not trained in this phase. The snapshot "
                "churn table has a 99.25% inactivity rate, making it unsuitable as "
                "a flagship supervised classifier for this dataset.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def train_demand_model() -> dict[str, Any]:
    """Train demand baselines, evaluate them, and save local artifacts."""
    config = ConfigLoader(str(ROOT / "config.yaml"), env_path=str(ROOT / ".env"))
    reports_dir = config.get_path("paths.reports")
    models_dir = config.get_path("paths.models")

    raw_features = _read_gold_demand_features(config)
    features = _prepare_demand_features(raw_features)
    train, validation, validation_months = chronological_train_validation_split(
        features,
        validation_months=VALIDATION_MONTHS,
    )

    naive_predictions = _clip_non_negative(validation["lag_1"])
    naive_metrics = compute_regression_metrics(
        validation[TARGET_COLUMN],
        naive_predictions,
    )

    xgb_model = _train_xgboost(train)
    xgb_predictions = _clip_non_negative(xgb_model.predict(validation[FEATURE_COLUMNS]))
    xgb_metrics = compute_regression_metrics(
        validation[TARGET_COLUMN],
        xgb_predictions,
    )

    champion, champion_reason = _select_champion(naive_metrics, xgb_metrics)
    champion_predictions = (
        xgb_predictions if champion == "xgboost" else naive_predictions
    )

    predictions = validation[EVALUATION_COLUMNS + [TARGET_COLUMN]].copy()
    predictions["naive_prediction"] = naive_predictions
    predictions["xgboost_prediction"] = xgb_predictions
    predictions["champion_prediction"] = champion_predictions

    metrics: dict[str, Any] = {
        "champion_model": champion,
        "champion_reason": champion_reason,
        "metrics": {
            "naive_lag_1": naive_metrics,
            "xgboost": xgb_metrics,
        },
        "validation": {
            "months": validation_months,
            "date_range": {
                "start": validation_months[0],
                "end": validation_months[-1],
            },
            "row_count": int(len(validation)),
        },
        "training": {
            "row_count": int(len(train)),
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "source_table": "gold/demand_features",
            "xgboost_params": XGB_PARAMS,
        },
        "churn_limitation": CHURN_LIMITATION,
    }

    model_path = models_dir / "cloudiq_demand_forecast.joblib"
    predictions_path = reports_dir / "demand_forecast_validation_predictions.csv"
    metrics_path = reports_dir / "demand_forecast_metrics.json"
    chart_path = reports_dir / "demand_forecast_validation.png"
    notes_path = reports_dir / "modeling_notes.md"
    saved_files = {
        "model": str(model_path),
        "validation_predictions": str(predictions_path),
        "metrics": str(metrics_path),
        "chart": str(chart_path),
        "modeling_notes": str(notes_path),
    }
    metrics["saved_files"] = saved_files

    _save_model_artifact(model_path, champion, xgb_model, validation_months)
    predictions.to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _save_validation_chart(predictions, chart_path)
    _write_modeling_notes(notes_path)
    logger.info(
        "Demand model training complete: champion={}, validation={}..{}",
        champion,
        validation_months[0],
        validation_months[-1],
    )
    return metrics


def main() -> int:
    """Run the training workflow and print a concise measured summary."""
    result = train_demand_model()
    print(f"Champion model: {result['champion_model']}")
    for model_name, model_metrics in result["metrics"].items():
        print(
            f"{model_name}: "
            f"MAE={model_metrics['mae']:.4f}, "
            f"RMSE={model_metrics['rmse']:.4f}, "
            f"MAPE={model_metrics['mape']:.4f}, "
            f"R2={model_metrics['r2']:.4f}"
        )
    validation = result["validation"]["date_range"]
    print(f"Validation date range: {validation['start']}..{validation['end']}")
    print(result["champion_reason"])
    print(CHURN_LIMITATION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
