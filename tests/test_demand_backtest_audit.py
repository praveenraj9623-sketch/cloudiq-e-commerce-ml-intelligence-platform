"""Tests for demand backtest audit recomputation helpers."""

from __future__ import annotations

import pandas as pd

from scripts.audit_demand_backtest import (
    recompute_category_errors,
    recompute_monthly_aggregate,
)


def test_monthly_aggregate_uses_same_prediction_sample() -> None:
    """Monthly chart aggregates the exact out-of-fold prediction rows."""
    predictions = pd.DataFrame(
        {
            "category_name_english": ["a", "b", "a"],
            "forecast_month": ["2018-06", "2018-06", "2018-07"],
            "target_units": [10, 5, 8],
            "naive_prediction": [9, 4, 7],
            "xgboost_prediction": [11, 3, 9],
            "champion_prediction": [9, 4, 7],
        }
    )

    result = recompute_monthly_aggregate(predictions)

    june = result[result["forecast_month"] == "2018-06"].iloc[0]
    assert june["forecast_month_label"] == "Jun 2018"
    assert june["actual_units"] == 15
    assert june["champion_prediction"] == 13
    assert june["evaluated_rows"] == 2


def test_category_errors_are_exact_from_predictions() -> None:
    """Category error rows are recomputed from prediction rows."""
    predictions = pd.DataFrame(
        {
            "category_name_english": ["a", "a", "b"],
            "forecast_month": ["2018-06", "2018-07", "2018-06"],
            "target_units": [10, 8, 5],
            "naive_prediction": [9, 7, 4],
            "xgboost_prediction": [11, 9, 3],
            "champion_prediction": [9, 7, 4],
        }
    )

    result = recompute_category_errors(predictions)

    row_a = result[result["category_name_english"] == "a"].iloc[0]
    assert row_a["actual_units"] == 18
    assert row_a["absolute_error"] == 2
    assert row_a["forecast_month_count"] == 2
