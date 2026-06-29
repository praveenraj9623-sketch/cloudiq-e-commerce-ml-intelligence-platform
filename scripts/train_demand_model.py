"""Run leakage-safe rolling-origin demand forecasting backtests.

The workflow uses the compact dashboard export
``data/dashboard/demand_backtest_input.csv``. Each row predicts demand for
``forecast_month`` using only values known through the end of the prior month:
``lag_1`` is the immediately preceding month's observed units, ``lag_2`` is two
months prior, ``lag_4`` is four months prior, and ``rolling_mean_3`` covers the
three months before the forecast month.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import ConfigLoader  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

TARGET_COLUMN = "target_units"
FORECAST_MONTH_COLUMN = "forecast_month"
FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_4",
    "rolling_mean_3",
    "month_num",
    "is_q4",
]
QUALITY_COLUMNS = [
    "target_period_order_count",
    "is_partial_period",
    "is_low_volume_period",
    "is_comparable_period",
]
EVALUATION_COLUMNS = [
    "category_name_english",
    FORECAST_MONTH_COLUMN,
    "feature_cutoff_month",
]
REQUIRED_COLUMNS = [
    "category_name_english",
    FORECAST_MONTH_COLUMN,
    TARGET_COLUMN,
    *FEATURE_COLUMNS,
]
MIN_VALIDATION_MONTHS = 4
MAX_VALIDATION_MONTHS = 6
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
    """Return finite MAPE by using ``1.0`` as denominator for zero actuals."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.size == 0:
        raise ValueError("Cannot calculate MAPE for an empty target array.")
    denominator = np.where(np.abs(true) < 1e-12, 1.0, np.abs(true))
    return float(np.mean(np.abs(true - pred) / denominator) * 100.0)


def zero_safe_wape(y_true: Any, y_pred: Any) -> float:
    """Return weighted absolute percentage error with a finite zero fallback."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    denominator = max(float(np.sum(np.abs(true))), 1.0)
    return float(np.sum(np.abs(true - pred)) / denominator * 100.0)


def zero_safe_smape(y_true: Any, y_pred: Any) -> float:
    """Return symmetric MAPE with finite behavior for zero actual/pred pairs."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(true) + np.abs(pred)
    denominator = np.where(denominator < 1e-12, 1.0, denominator)
    return float(np.mean((2.0 * np.abs(true - pred)) / denominator) * 100.0)


def compute_regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute measured regression metrics for demand backtests."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.size == 0:
        raise ValueError("Cannot calculate metrics for an empty target array.")
    error = true - pred
    sst = float(np.sum((true - np.mean(true)) ** 2))
    sse = float(np.sum(error**2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape": zero_safe_wape(true, pred),
        "smape": zero_safe_smape(true, pred),
        "mape": zero_safe_mape(true, pred),
        "r2": float(1.0 - (sse / sst)) if sst > 1e-12 else 0.0,
    }


def _month_keys(values: pd.Series) -> pd.Series:
    """Normalize month-like values to YYYY-MM strings."""
    periods = pd.PeriodIndex(values.astype(str), freq="M")
    return pd.Series(periods.astype(str), index=values.index)


def _month_labels(values: pd.Series) -> pd.Series:
    """Return human-readable labels like ``Jun 2018`` for monthly periods."""
    periods = pd.PeriodIndex(values.astype(str), freq="M")
    return pd.Series([period.strftime("%b %Y") for period in periods], index=values.index)


def chronological_train_validation_split(
    df: pd.DataFrame,
    date_col: str = FORECAST_MONTH_COLUMN,
    validation_months: int = 2,
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


def prepare_demand_backtest_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce compact demand backtest rows for modeling."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required demand backtest columns: {missing}")

    prepared = df.copy()
    prepared[FORECAST_MONTH_COLUMN] = _month_keys(prepared[FORECAST_MONTH_COLUMN])
    prepared["category_name_english"] = prepared["category_name_english"].astype(str)
    if "feature_cutoff_month" not in prepared.columns:
        prepared["feature_cutoff_month"] = (
            pd.PeriodIndex(prepared[FORECAST_MONTH_COLUMN], freq="M") - 1
        ).astype(str)
    else:
        prepared["feature_cutoff_month"] = _month_keys(prepared["feature_cutoff_month"])

    for column in [TARGET_COLUMN, *FEATURE_COLUMNS]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    for column in ["is_partial_period", "is_low_volume_period"]:
        if column not in prepared.columns:
            prepared[column] = False
        prepared[column] = prepared[column].fillna(False).astype(bool)

    if "is_comparable_period" not in prepared.columns:
        prepared["is_comparable_period"] = ~(
            prepared["is_partial_period"] | prepared["is_low_volume_period"]
        )
    prepared["is_comparable_period"] = prepared["is_comparable_period"].fillna(False).astype(bool)
    if "target_period_order_count" not in prepared.columns:
        prepared["target_period_order_count"] = np.nan

    before_drop = len(prepared)
    prepared = prepared.dropna(subset=[TARGET_COLUMN, *FEATURE_COLUMNS])
    dropped = before_drop - len(prepared)
    if dropped:
        logger.warning("Dropped {} demand rows with null modeling fields.", dropped)
    if prepared.empty:
        raise ValueError("No usable demand feature rows remain after preparation.")

    return prepared.sort_values([FORECAST_MONTH_COLUMN, "category_name_english"]).reset_index(
        drop=True
    )


def select_rolling_validation_months(
    features: pd.DataFrame,
    min_months: int = MIN_VALIDATION_MONTHS,
    max_months: int = MAX_VALIDATION_MONTHS,
) -> list[str]:
    """Select final eligible target months for expanding-window validation."""
    if FORECAST_MONTH_COLUMN not in features.columns:
        raise KeyError(f"Missing validation month column: {FORECAST_MONTH_COLUMN}")

    work = features.copy()
    work[FORECAST_MONTH_COLUMN] = _month_keys(work[FORECAST_MONTH_COLUMN])
    if "is_comparable_period" in work.columns:
        work = work[work["is_comparable_period"].astype(bool)]
    eligible = sorted(work[FORECAST_MONTH_COLUMN].dropna().unique().tolist())
    if len(eligible) < min_months:
        if len(eligible) < 2:
            raise ValueError("Not enough eligible forecast months for rolling backtest.")
        logger.warning(
            "Only {} eligible validation months found; requested at least {}.",
            len(eligible),
            min_months,
        )
    return eligible[-max_months:]


def _train_xgboost(train: pd.DataFrame) -> "XGBRegressor":
    """Fit a modest XGBoost regressor for the small demand feature table."""
    from xgboost import XGBRegressor

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    return model


def _clip_non_negative(values: Any) -> np.ndarray:
    """Clip forecast outputs to the valid non-negative demand range."""
    return np.maximum(np.asarray(values, dtype=float), 0.0)


def run_rolling_backtest(
    features: pd.DataFrame,
    validation_months: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run expanding-window folds and return predictions plus fold metrics."""
    predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []

    for fold_idx, forecast_month in enumerate(validation_months, start=1):
        train = features[features[FORECAST_MONTH_COLUMN] < forecast_month]
        validation = features[features[FORECAST_MONTH_COLUMN] == forecast_month]
        if train.empty or validation.empty:
            raise ValueError(f"Fold {forecast_month} has empty train or validation data.")

        xgb_model = _train_xgboost(train)
        output_columns = [
            column
            for column in [*EVALUATION_COLUMNS, *QUALITY_COLUMNS, *FEATURE_COLUMNS, TARGET_COLUMN]
            if column in validation.columns
        ]
        fold = validation[output_columns].copy()
        fold["fold"] = fold_idx
        fold["max_train_forecast_month"] = str(train[FORECAST_MONTH_COLUMN].max())
        fold["naive_prediction"] = _clip_non_negative(validation["lag_1"])
        fold["xgboost_prediction"] = _clip_non_negative(
            xgb_model.predict(validation[FEATURE_COLUMNS])
        )
        predictions.append(fold)

        for model_name, column in [
            ("naive_lag_1", "naive_prediction"),
            ("xgboost", "xgboost_prediction"),
        ]:
            metrics = compute_regression_metrics(fold[TARGET_COLUMN], fold[column])
            fold_metrics.append(
                {
                    "fold": fold_idx,
                    FORECAST_MONTH_COLUMN: forecast_month,
                    "model": model_name,
                    "train_rows": int(len(train)),
                    "validation_rows": int(len(validation)),
                    "max_train_forecast_month": str(train[FORECAST_MONTH_COLUMN].max()),
                    **metrics,
                }
            )

    return pd.concat(predictions, ignore_index=True), pd.DataFrame(fold_metrics)


def _aggregate_metrics(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute pooled metrics across all validation folds."""
    return {
        "naive_lag_1": compute_regression_metrics(
            predictions[TARGET_COLUMN],
            predictions["naive_prediction"],
        ),
        "xgboost": compute_regression_metrics(
            predictions[TARGET_COLUMN],
            predictions["xgboost_prediction"],
        ),
    }


def _select_champion(aggregate_metrics: dict[str, dict[str, float]]) -> tuple[str, str]:
    """Select champion by pooled WAPE and MAE, with RMSE as a tie breaker."""
    naive = aggregate_metrics["naive_lag_1"]
    xgb = aggregate_metrics["xgboost"]
    xgb_key = (xgb["wape"], xgb["mae"], xgb["rmse"])
    naive_key = (naive["wape"], naive["mae"], naive["rmse"])
    if xgb["mae"] < naive["mae"] and xgb_key < naive_key:
        return "xgboost", "XGBoost improved pooled chronological WAPE, MAE, and RMSE."
    return (
        "naive_lag_1",
        "Naive lag-1 benchmark retained because XGBoost did not improve the pooled "
        "chronological WAPE/MAE selection criterion.",
    )


def _monthly_aggregate(predictions: pd.DataFrame, champion: str) -> pd.DataFrame:
    """Build monthly aggregate actual-vs-predicted rows from the OOF sample."""
    champion_column = "xgboost_prediction" if champion == "xgboost" else "naive_prediction"
    monthly = (
        predictions.assign(champion_prediction=predictions[champion_column])
        .groupby(FORECAST_MONTH_COLUMN, as_index=False)
        .agg(
            actual_units=(TARGET_COLUMN, "sum"),
            naive_prediction=("naive_prediction", "sum"),
            xgboost_prediction=("xgboost_prediction", "sum"),
            champion_prediction=("champion_prediction", "sum"),
            evaluated_rows=("category_name_english", "size"),
            category_count=("category_name_english", "nunique"),
        )
        .sort_values(FORECAST_MONTH_COLUMN)
    )
    monthly["forecast_month_label"] = _month_labels(monthly[FORECAST_MONTH_COLUMN])
    monthly["champion_abs_error"] = (
        monthly["actual_units"] - monthly["champion_prediction"]
    ).abs()
    return monthly[
        [
            FORECAST_MONTH_COLUMN,
            "forecast_month_label",
            "actual_units",
            "naive_prediction",
            "xgboost_prediction",
            "champion_prediction",
            "evaluated_rows",
            "category_count",
            "champion_abs_error",
        ]
    ]


def _category_errors(predictions: pd.DataFrame, champion: str) -> pd.DataFrame:
    """Build category-level OOF error totals."""
    champion_column = "xgboost_prediction" if champion == "xgboost" else "naive_prediction"
    work = predictions.assign(
        champion_prediction=predictions[champion_column],
        champion_abs_error=(predictions[TARGET_COLUMN] - predictions[champion_column]).abs(),
    )
    grouped = (
        work.groupby("category_name_english", as_index=False)
        .agg(
            actual_units=(TARGET_COLUMN, "sum"),
            naive_prediction=("naive_prediction", "sum"),
            xgboost_prediction=("xgboost_prediction", "sum"),
            champion_prediction=("champion_prediction", "sum"),
            absolute_error=("champion_abs_error", "sum"),
            evaluated_rows=(FORECAST_MONTH_COLUMN, "size"),
            forecast_month_count=(FORECAST_MONTH_COLUMN, "nunique"),
        )
        .sort_values(["absolute_error", "actual_units"], ascending=[False, False])
    )
    grouped["mae"] = grouped["absolute_error"] / grouped["evaluated_rows"].clip(lower=1)
    return grouped.reset_index(drop=True)


def _save_model_artifact(
    model_path: Path,
    champion: str,
    features: pd.DataFrame,
    validation_months: list[str],
) -> None:
    """Persist the selected model artifact with metadata."""
    import joblib

    artifact: dict[str, Any] = {
        "model_type": champion,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "forecast_month_contract": "Features are known through forecast_month - 1.",
        "validation_forecast_months": validation_months,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_postprocessing": "clip predictions at zero",
        "backtest_type": "rolling_origin_out_of_fold",
    }
    if champion == "xgboost":
        artifact["model"] = _train_xgboost(features)
    else:
        artifact["prediction_feature"] = "lag_1"
    joblib.dump(artifact, model_path)


def _save_validation_chart(monthly: pd.DataFrame, chart_path: Path) -> None:
    """Save a simple actual-vs-predicted monthly OOF chart."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(
        monthly["forecast_month_label"],
        monthly["actual_units"],
        marker="o",
        label="Actual units",
    )
    ax.plot(
        monthly["forecast_month_label"],
        monthly["champion_prediction"],
        marker="o",
        label="Champion prediction",
    )
    ax.set_title("Demand Forecast Backtest")
    ax.set_xlabel("Forecast month")
    ax.set_ylabel("Units")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)


def _write_model_card(path: Path, metrics: dict[str, Any]) -> None:
    """Write a compact local model card with measured backtest results."""
    naive = metrics["metrics"]["naive_lag_1"]
    xgb = metrics["metrics"]["xgboost"]
    validation = metrics["validation"]
    path.write_text(
        "\n".join(
            [
                "# Demand Forecast Model Card",
                "",
                "## Target Contract",
                "",
                "- `forecast_month` is the month being predicted.",
                "- `target_units` is the observed unit demand in `forecast_month`.",
                "- Feature availability cutoff is the end of the prior month.",
                "- `lag_1`, `lag_2`, `lag_4`, and `rolling_mean_3` use only months before `forecast_month`.",
                "",
                "## Validation",
                "",
                "- Method: expanding-window rolling-origin validation.",
                f"- Forecast months: {validation['date_range']['start']} through {validation['date_range']['end']}.",
                f"- Evaluated rows: {validation['row_count']}.",
                "- Partial and low-volume target months are excluded from evaluation.",
                "",
                "## Results",
                "",
                f"- Champion: `{metrics['champion_model']}`.",
                f"- Reason: {metrics['champion_reason']}",
                f"- Naive MAE/RMSE/WAPE/sMAPE/MAPE/R2: {naive['mae']:.4f} / {naive['rmse']:.4f} / {naive['wape']:.4f} / {naive['smape']:.4f} / {naive['mape']:.4f} / {naive['r2']:.4f}.",
                f"- XGBoost MAE/RMSE/WAPE/sMAPE/MAPE/R2: {xgb['mae']:.4f} / {xgb['rmse']:.4f} / {xgb['wape']:.4f} / {xgb['smape']:.4f} / {xgb['mape']:.4f} / {xgb['r2']:.4f}.",
                "",
                "## Chart Interpretation",
                "",
                "- The old crossing-line chart came from ambiguous target-month naming and a saved fallback artifact. The corrected chart aggregates the exact out-of-fold validation rows by `forecast_month`.",
                "- The naive line is intentionally one month behind actual demand because the naive forecast equals `lag_1`.",
                "",
                "## Churn Limitation",
                "",
                f"- {CHURN_LIMITATION}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_modeling_notes(notes_path: Path) -> None:
    """Write compact modeling notes for the demand-only training phase."""
    notes_path.write_text(
        "\n".join(
            [
                "# CloudIQ Modeling Notes",
                "",
                "## Demand Forecast Backtest",
                "",
                "- Target: `target_units`, observed unit demand in `forecast_month` "
                "for each `category_name_english`.",
                "- Validation: expanding-window chronological validation over the "
                "final eligible forecast months.",
                "- Leakage statement: all features are available by the end of the "
                "month before `forecast_month`; no feature uses the forecast month "
                "or later.",
                "- Headline metrics prioritize WAPE and MAE. MAPE is retained only "
                "as a diagnostic because low-volume categories can distort it.",
                "",
                "## Churn Limitation",
                "",
                f"- {CHURN_LIMITATION}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _load_backtest_input() -> pd.DataFrame:
    """Load the compact pandas-only demand backtest input CSV."""
    input_path = ROOT / "data" / "dashboard" / "demand_backtest_input.csv"
    if not input_path.exists() or input_path.stat().st_size == 0:
        raise FileNotFoundError(
            "Demand backtest input is missing. Run "
            "`python scripts/export_dashboard_data.py` before training."
        )
    return pd.read_csv(input_path)


def train_demand_model() -> dict[str, Any]:
    """Run rolling backtests, select champion, and save local artifacts."""
    config = ConfigLoader(str(ROOT / "config.yaml"), env_path=str(ROOT / ".env"))
    reports_dir = config.get_path("paths.reports")
    models_dir = config.get_path("paths.models")

    raw_features = _load_backtest_input()
    features = prepare_demand_backtest_input(raw_features)
    comparable_features = features[features["is_comparable_period"]].copy()
    if comparable_features.empty:
        raise ValueError("No comparable demand rows remain for rolling validation.")

    validation_months = select_rolling_validation_months(comparable_features)
    predictions, fold_metrics = run_rolling_backtest(
        comparable_features,
        validation_months,
    )
    aggregate_metrics = _aggregate_metrics(predictions)
    champion, champion_reason = _select_champion(aggregate_metrics)
    champion_column = "xgboost_prediction" if champion == "xgboost" else "naive_prediction"
    predictions["champion_prediction"] = predictions[champion_column]
    monthly = _monthly_aggregate(predictions, champion)
    category_errors = _category_errors(predictions, champion)

    metrics: dict[str, Any] = {
        "champion_model": champion,
        "champion_display_name": (
            "Naive Lag-1 Benchmark" if champion == "naive_lag_1" else "XGBoost"
        ),
        "champion_reason": champion_reason,
        "selection_basis": "pooled out-of-fold validation prioritized by WAPE and MAE, with RMSE secondary",
        "forecast_contract": {
            "forecast_month": "month being predicted",
            "target_units": "observed units in forecast_month",
            "feature_availability_cutoff": "end of prior month",
            "lag_1": "observed units in forecast_month - 1",
            "lag_2": "observed units in forecast_month - 2",
            "lag_4": "observed units in forecast_month - 4",
            "rolling_mean_3": "mean observed units from forecast_month - 3 through forecast_month - 1",
            "naive_baseline": "prediction equals lag_1",
        },
        "metrics": aggregate_metrics,
        "validation": {
            "forecast_months": validation_months,
            "date_range": {
                "start": validation_months[0],
                "end": validation_months[-1],
            },
            "row_count": int(len(predictions)),
            "fold_count": int(len(validation_months)),
        },
        "training": {
            "row_count": int(len(comparable_features)),
            "all_input_rows": int(len(features)),
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "source_file": "data/dashboard/demand_backtest_input.csv",
            "xgboost_params": XGB_PARAMS,
        },
        "metric_definitions": {
            "mae": "Mean absolute category-month unit error.",
            "rmse": "Root mean squared category-month unit error.",
            "wape": "Sum absolute error divided by sum actual units.",
            "smape": "Zero-safe symmetric absolute percentage error.",
            "mape": "Zero-safe category-month MAPE; diagnostic only for low-volume data.",
            "r2": "Secondary diagnostic only.",
        },
        "old_chart_crossing_explanation": (
            "The old crossing chart used ambiguous target-month naming and a fallback "
            "validation artifact. The corrected chart aggregates the exact out-of-fold "
            "rows by forecast_month; the naive line naturally trails changes because "
            "it equals lag_1."
        ),
        "churn_limitation": CHURN_LIMITATION,
    }

    model_path = models_dir / "cloudiq_demand_forecast.joblib"
    predictions_path = reports_dir / "demand_backtest_predictions.csv"
    fold_metrics_path = reports_dir / "demand_backtest_fold_metrics.csv"
    monthly_path = reports_dir / "demand_backtest_monthly_aggregate.csv"
    category_errors_path = reports_dir / "demand_backtest_category_errors.csv"
    metrics_path = reports_dir / "demand_forecast_metrics.json"
    chart_path = reports_dir / "demand_forecast_validation.png"
    model_card_path = reports_dir / "demand_forecast_model_card.md"
    notes_path = reports_dir / "modeling_notes.md"
    legacy_predictions_path = reports_dir / "demand_forecast_validation_predictions.csv"
    saved_files = {
        "model": str(model_path),
        "predictions": str(predictions_path),
        "fold_metrics": str(fold_metrics_path),
        "monthly_aggregate": str(monthly_path),
        "category_errors": str(category_errors_path),
        "metrics": str(metrics_path),
        "chart": str(chart_path),
        "model_card": str(model_card_path),
        "modeling_notes": str(notes_path),
    }
    metrics["saved_files"] = saved_files

    _save_model_artifact(model_path, champion, comparable_features, validation_months)
    predictions.to_csv(predictions_path, index=False)
    predictions.to_csv(legacy_predictions_path, index=False)
    fold_metrics.to_csv(fold_metrics_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    category_errors.to_csv(category_errors_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    _save_validation_chart(monthly, chart_path)
    _write_model_card(model_card_path, metrics)
    _write_modeling_notes(notes_path)
    logger.info(
        "Demand backtest complete: champion={}, validation={}..{}",
        champion,
        validation_months[0],
        validation_months[-1],
    )
    return metrics


def main() -> int:
    """Run the backtest workflow and print a concise measured summary."""
    result = train_demand_model()
    print(f"Champion model: {result['champion_model']}")
    for model_name, model_metrics in result["metrics"].items():
        print(
            f"{model_name}: "
            f"MAE={model_metrics['mae']:.4f}, "
            f"RMSE={model_metrics['rmse']:.4f}, "
            f"WAPE={model_metrics['wape']:.4f}, "
            f"sMAPE={model_metrics['smape']:.4f}, "
            f"MAPE={model_metrics['mape']:.4f}, "
            f"R2={model_metrics['r2']:.4f}"
        )
    validation = result["validation"]["date_range"]
    print(f"Validation forecast range: {validation['start']}..{validation['end']}")
    print(f"Evaluated rows: {result['validation']['row_count']}")
    print(result["champion_reason"])
    print(CHURN_LIMITATION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
