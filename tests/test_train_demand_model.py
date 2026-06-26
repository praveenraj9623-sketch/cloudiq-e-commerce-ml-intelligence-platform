"""Tests for demand-model training helpers."""

from __future__ import annotations

import math

import pandas as pd

from scripts.train_demand_model import (
    chronological_train_validation_split,
    zero_safe_mape,
)


def test_zero_safe_mape_handles_zero_targets() -> None:
    """Zero actuals produce finite percentage errors."""
    result = zero_safe_mape([0, 10], [2, 5])

    assert math.isfinite(result)
    assert result == 125.0


def test_zero_safe_mape_zero_prediction_for_zero_target_is_no_error() -> None:
    """A zero forecast for zero actual demand contributes no MAPE error."""
    assert zero_safe_mape([0, 10], [0, 5]) == 25.0


def test_chronological_split_uses_final_two_months_for_validation() -> None:
    """Validation rows come only from the final two calendar months."""
    df = pd.DataFrame(
        {
            "order_year_month": [
                "2024-01",
                "2024-02",
                "2024-03",
                "2024-04",
                "2024-05",
                "2024-05",
            ],
            "category_name_english": ["a", "a", "a", "a", "a", "b"],
            "target_next_month": [2, 3, 4, 5, 6, 7],
            "lag_1": [1, 2, 3, 4, 5, 6],
            "lag_2": [1, 1, 2, 3, 4, 5],
            "lag_4": [1, 1, 1, 1, 2, 3],
            "rolling_mean_3": [1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
            "month_num": [1, 2, 3, 4, 5, 5],
            "is_q4": [0, 0, 0, 0, 0, 0],
        }
    )

    train, validation, validation_months = chronological_train_validation_split(df)

    assert validation_months == ["2024-04", "2024-05"]
    assert set(train["order_year_month"]) == {"2024-01", "2024-02", "2024-03"}
    assert set(validation["order_year_month"]) == {"2024-04", "2024-05"}
    assert len(validation) == 3
