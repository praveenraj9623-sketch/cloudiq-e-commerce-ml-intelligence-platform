"""Audit demand backtest artifacts for forecast-month contract correctness."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FORECAST_MONTH = "forecast_month"
TARGET = "target_units"
TOLERANCE = 1e-7


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV artifact and fail clearly when it is missing or empty."""
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required demand backtest artifact missing: {path}")
    return pd.read_csv(path)


def _month_label(month: str) -> str:
    """Return a chart-safe month label such as ``Jun 2018``."""
    return pd.Period(str(month), freq="M").strftime("%b %Y")


def _compare_numeric_series(left: pd.Series, right: pd.Series) -> bool:
    """Compare numeric values with a small floating-point tolerance."""
    return bool(np.allclose(left.astype(float), right.astype(float), atol=TOLERANCE))


def recompute_monthly_aggregate(predictions: pd.DataFrame) -> pd.DataFrame:
    """Recompute the monthly chart sample from prediction rows."""
    monthly = (
        predictions.groupby(FORECAST_MONTH, as_index=False)
        .agg(
            actual_units=(TARGET, "sum"),
            naive_prediction=("naive_prediction", "sum"),
            xgboost_prediction=("xgboost_prediction", "sum"),
            champion_prediction=("champion_prediction", "sum"),
            evaluated_rows=("category_name_english", "size"),
            category_count=("category_name_english", "nunique"),
        )
        .sort_values(FORECAST_MONTH)
        .reset_index(drop=True)
    )
    monthly["forecast_month_label"] = monthly[FORECAST_MONTH].map(_month_label)
    monthly["champion_abs_error"] = (
        monthly["actual_units"] - monthly["champion_prediction"]
    ).abs()
    return monthly[
        [
            FORECAST_MONTH,
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


def recompute_category_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    """Recompute category-level errors from prediction rows."""
    work = predictions.assign(
        champion_abs_error=(predictions[TARGET] - predictions["champion_prediction"]).abs()
    )
    result = (
        work.groupby("category_name_english", as_index=False)
        .agg(
            actual_units=(TARGET, "sum"),
            naive_prediction=("naive_prediction", "sum"),
            xgboost_prediction=("xgboost_prediction", "sum"),
            champion_prediction=("champion_prediction", "sum"),
            absolute_error=("champion_abs_error", "sum"),
            evaluated_rows=(FORECAST_MONTH, "size"),
            forecast_month_count=(FORECAST_MONTH, "nunique"),
        )
        .sort_values(["absolute_error", "actual_units"], ascending=[False, False])
        .reset_index(drop=True)
    )
    result["mae"] = result["absolute_error"] / result["evaluated_rows"].clip(lower=1)
    return result


def _add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: str,
) -> None:
    """Append a standard audit check record."""
    checks.append(
        {
            "name": name,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        }
    )


def audit_demand_backtest() -> dict[str, Any]:
    """Validate local demand backtest artifacts and write audit reports."""
    reports_dir = ROOT / "reports"
    dashboard_dir = ROOT / "data" / "dashboard"
    predictions = _read_csv(reports_dir / "demand_backtest_predictions.csv")
    monthly = _read_csv(reports_dir / "demand_backtest_monthly_aggregate.csv")
    category_errors = _read_csv(reports_dir / "demand_backtest_category_errors.csv")
    fold_metrics = _read_csv(reports_dir / "demand_backtest_fold_metrics.csv")
    backtest_input = _read_csv(dashboard_dir / "demand_backtest_input.csv")
    metrics = json.loads(
        (reports_dir / "demand_forecast_metrics.json").read_text(encoding="utf-8")
    )

    checks: list[dict[str, Any]] = []
    required_prediction_columns = {
        "category_name_english",
        FORECAST_MONTH,
        "feature_cutoff_month",
        TARGET,
        "lag_1",
        "lag_2",
        "lag_4",
        "rolling_mean_3",
        "naive_prediction",
        "xgboost_prediction",
        "champion_prediction",
        "max_train_forecast_month",
    }
    missing = required_prediction_columns - set(predictions.columns)
    _add_check(
        checks,
        "prediction_columns",
        not missing,
        "All required prediction columns are present." if not missing else f"Missing: {sorted(missing)}",
    )

    parseable_months = True
    try:
        prediction_months = pd.PeriodIndex(predictions[FORECAST_MONTH].astype(str), freq="M")
    except (ValueError, TypeError):
        parseable_months = False
        prediction_months = pd.PeriodIndex([], freq="M")
    _add_check(
        checks,
        "one_forecast_month_per_prediction",
        parseable_months and predictions[FORECAST_MONTH].notna().all(),
        "Every prediction row has one parseable forecast_month.",
    )

    if not missing:
        _add_check(
            checks,
            "naive_equals_lag_1",
            _compare_numeric_series(predictions["naive_prediction"], predictions["lag_1"]),
            "Naive baseline prediction equals lag_1 for every row.",
        )
        train_months = pd.PeriodIndex(
            predictions["max_train_forecast_month"].astype(str),
            freq="M",
        )
        _add_check(
            checks,
            "no_train_row_reaches_validation_month",
            bool((train_months < prediction_months).all()),
            "Every fold trains only on months before its forecast_month.",
        )

    fold_periods_ok = True
    if {"max_train_forecast_month", FORECAST_MONTH}.issubset(fold_metrics.columns):
        fold_periods_ok = bool(
            (
                pd.PeriodIndex(fold_metrics["max_train_forecast_month"].astype(str), freq="M")
                < pd.PeriodIndex(fold_metrics[FORECAST_MONTH].astype(str), freq="M")
            ).all()
        )
    _add_check(
        checks,
        "fold_metric_train_cutoff",
        fold_periods_ok,
        "Fold metric records respect the same train-before-validation cutoff.",
    )

    target_flags = {
        "target_period_order_count",
        "is_partial_period",
        "is_low_volume_period",
        "is_comparable_period",
    }
    missing_flags = target_flags - set(backtest_input.columns)
    _add_check(
        checks,
        "target_period_comparability_flags",
        not missing_flags,
        "Backtest input includes target-period comparability flags."
        if not missing_flags
        else f"Missing flags: {sorted(missing_flags)}",
    )

    expected_monthly = recompute_monthly_aggregate(predictions)
    monthly_sorted = monthly[expected_monthly.columns].sort_values(FORECAST_MONTH).reset_index(
        drop=True
    )
    monthly_equal = len(monthly_sorted) == len(expected_monthly)
    if monthly_equal:
        for column in [
            "actual_units",
            "naive_prediction",
            "xgboost_prediction",
            "champion_prediction",
            "champion_abs_error",
        ]:
            monthly_equal = monthly_equal and _compare_numeric_series(
                monthly_sorted[column],
                expected_monthly[column],
            )
        for column in ["forecast_month", "forecast_month_label", "evaluated_rows", "category_count"]:
            monthly_equal = monthly_equal and monthly_sorted[column].astype(str).equals(
                expected_monthly[column].astype(str)
            )
    _add_check(
        checks,
        "monthly_chart_uses_identical_oof_sample",
        monthly_equal,
        "Monthly aggregate matches the exact prediction rows used for metrics.",
    )

    expected_category = recompute_category_errors(predictions)
    category_sorted = category_errors[expected_category.columns].reset_index(drop=True)
    category_equal = len(category_sorted) == len(expected_category)
    if category_equal:
        category_equal = category_sorted["category_name_english"].equals(
            expected_category["category_name_english"]
        )
        for column in [
            "actual_units",
            "naive_prediction",
            "xgboost_prediction",
            "champion_prediction",
            "absolute_error",
            "mae",
        ]:
            category_equal = category_equal and _compare_numeric_series(
                category_sorted[column],
                expected_category[column],
            )
        for column in ["evaluated_rows", "forecast_month_count"]:
            category_equal = category_equal and category_sorted[column].astype(str).equals(
                expected_category[column].astype(str)
            )
    _add_check(
        checks,
        "category_error_rows_exact",
        category_equal,
        "Category-level error rows exactly match recomputation from predictions.",
    )

    label_pattern = re.compile(r"^[A-Z][a-z]{2} \d{4}$")
    _add_check(
        checks,
        "monthly_labels_are_categorical_months",
        bool(monthly["forecast_month_label"].astype(str).map(label_pattern.match).all()),
        "Monthly chart labels use categorical strings such as Jun 2018.",
    )

    _add_check(
        checks,
        "old_crossing_line_explanation_present",
        bool(metrics.get("old_chart_crossing_explanation")),
        "Metrics JSON documents why the old crossing-line chart was misleading.",
    )

    verdict = "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL"
    payload: dict[str, Any] = {
        "verdict": verdict,
        "checks": checks,
        "prediction_rows": int(len(predictions)),
        "forecast_months": sorted(predictions[FORECAST_MONTH].astype(str).unique().tolist()),
        "champion_model": metrics.get("champion_model"),
        "old_chart_crossing_explanation": metrics.get("old_chart_crossing_explanation"),
    }

    (reports_dir / "demand_backtest_audit.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    report_lines = [
        "# Demand Backtest Audit",
        "",
        f"Verdict: **{verdict}**",
        "",
        "## Checks",
        "",
    ]
    for check in checks:
        report_lines.append(f"- {check['status']} - {check['name']}: {check['detail']}")
    report_lines.extend(
        [
            "",
            "## Chart Crossing Explanation",
            "",
            str(payload["old_chart_crossing_explanation"]),
            "",
        ]
    )
    (reports_dir / "demand_backtest_audit.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    """Run the demand backtest audit."""
    result = audit_demand_backtest()
    print(f"Demand backtest audit verdict: {result['verdict']}")
    print(f"Prediction rows: {result['prediction_rows']}")
    print("Forecast months: " + ", ".join(result["forecast_months"]))
    if result["verdict"] != "PASS":
        failed = [check["name"] for check in result["checks"] if check["status"] != "PASS"]
        print("Failed checks: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
