"""Tests for demand-model training helpers."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.train_demand_model import (
    chronological_train_validation_split,
    prepare_demand_backtest_input,
    run_rolling_backtest,
    select_rolling_validation_months,
    zero_safe_mape,
)


class _DummyXgbModel:
    """Small test double that avoids importing XGBoost."""

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Return a deterministic prediction."""
        return frame["lag_1"]


def _sample_features() -> pd.DataFrame:
    """Create synthetic forecast-month demand rows."""
    rows = []
    for month_idx, month in enumerate(
        ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"],
        start=1,
    ):
        rows.append(
            {
                "category_name_english": "a",
                "forecast_month": month,
                "feature_cutoff_month": str(pd.Period(month, freq="M") - 1),
                "target_units": month_idx + 10,
                "lag_1": month_idx,
                "lag_2": month_idx - 1,
                "lag_4": month_idx - 3,
                "rolling_mean_3": float(month_idx - 1),
                "month_num": month_idx,
                "is_q4": 0,
                "target_period_order_count": 200,
                "is_partial_period": False,
                "is_low_volume_period": False,
                "is_comparable_period": True,
            }
        )
    return pd.DataFrame(rows)


def test_zero_safe_mape_handles_zero_targets() -> None:
    """Zero actuals produce finite percentage errors."""
    result = zero_safe_mape([0, 10], [2, 5])

    assert math.isfinite(result)
    assert result == 125.0


def test_zero_safe_mape_zero_prediction_for_zero_target_is_no_error() -> None:
    """A zero forecast for zero actual demand contributes no MAPE error."""
    assert zero_safe_mape([0, 10], [0, 5]) == 25.0


def test_chronological_split_uses_final_two_forecast_months_for_validation() -> None:
    """Validation rows come only from the final two forecast months."""
    df = _sample_features()

    train, validation, validation_months = chronological_train_validation_split(df)

    assert validation_months == ["2024-05", "2024-06"]
    assert set(train["forecast_month"]) == {"2024-01", "2024-02", "2024-03", "2024-04"}
    assert set(validation["forecast_month"]) == {"2024-05", "2024-06"}
    assert len(validation) == 2


def test_prepare_backtest_input_preserves_forecast_cutoff_contract() -> None:
    """Feature cutoff stays before the forecast month after preparation."""
    result = prepare_demand_backtest_input(_sample_features())

    forecast_months = pd.PeriodIndex(result["forecast_month"], freq="M")
    cutoff_months = pd.PeriodIndex(result["feature_cutoff_month"], freq="M")

    assert bool((cutoff_months < forecast_months).all())
    assert result["target_units"].notna().all()
    assert result["is_comparable_period"].all()


def test_rolling_validation_months_skip_flagged_target_periods() -> None:
    """Rolling validation excludes partial and low-volume target months."""
    features = _sample_features()
    features.loc[features["forecast_month"] == "2024-06", "is_partial_period"] = True
    features.loc[features["forecast_month"] == "2024-06", "is_comparable_period"] = False

    months = select_rolling_validation_months(features, min_months=2, max_months=4)

    assert months == ["2024-02", "2024-03", "2024-04", "2024-05"]
    assert "2024-06" not in months


def test_rolling_backtest_uses_prior_months_and_naive_equals_lag_1() -> None:
    """Each fold trains before the validation month and naive uses lag_1."""
    features = prepare_demand_backtest_input(_sample_features())
    validation_months = ["2024-05", "2024-06"]

    with patch("scripts.train_demand_model._train_xgboost", return_value=_DummyXgbModel()):
        predictions, fold_metrics = run_rolling_backtest(features, validation_months)

    assert np.allclose(predictions["naive_prediction"], predictions["lag_1"])
    assert bool(
        (
            pd.PeriodIndex(predictions["max_train_forecast_month"], freq="M")
            < pd.PeriodIndex(predictions["forecast_month"], freq="M")
        ).all()
    )
    assert set(fold_metrics["forecast_month"]) == {"2024-05", "2024-06"}
