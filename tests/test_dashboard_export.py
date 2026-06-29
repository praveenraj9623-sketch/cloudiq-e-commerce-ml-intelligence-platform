"""Pure tests for dashboard export helper logic."""

from __future__ import annotations

import pandas as pd

from scripts.export_dashboard_data import (
    build_demand_backtest_input,
    flag_period_quality,
    late_delivery_rate_from_flags,
    mask_seller_id,
    safe_divide,
    summarize_overview_kpis,
)
from src.ui.components import (
    late_delivery_rate_definition,
    payment_method_label,
    rfm_segment_description,
    seller_monetary_label,
)


def test_safe_divide_returns_zero_for_empty_denominator() -> None:
    """Dashboard ratios stay finite when a denominator is unavailable."""
    assert safe_divide(10, 0) == 0.0
    assert safe_divide(10, None) == 0.0


def test_summarize_overview_kpis_uses_actual_aggregates() -> None:
    """Overview values derive from provided aggregate facts."""
    result = summarize_overview_kpis(
        total_orders=4,
        item_merchandise_value_ex_freight=200.0,
        late_delivery_rate=0.25,
        first_month="2017-01",
        last_month="2018-07",
        freight_value=20.0,
    )

    assert result["total_orders"] == 4
    assert result["item_merchandise_value_ex_freight"] == 200.0
    assert result["freight_value"] == 20.0
    assert result["avg_item_merchandise_value_per_order"] == 50.0
    assert result["late_delivery_rate"] == 0.25
    assert result["first_order_month"] == "2017-01"
    assert result["last_order_month"] == "2018-07"


def test_flag_period_quality_marks_partial_and_low_volume_months() -> None:
    """Partial and low-volume periods are flagged but retained."""
    result = flag_period_quality(
        pd.DataFrame(
            {
                "order_year_month": ["2018-08", "2018-09", "2018-10"],
                "order_count": [500, 16, 4],
                "first_purchase": pd.to_datetime(
                    ["2018-08-01", "2018-09-01", "2018-10-01"]
                ),
                "last_purchase": pd.to_datetime(
                    ["2018-08-31", "2018-09-30", "2018-10-17"]
                ),
            }
        ),
        min_order_count=100,
    )

    assert bool(result.loc[0, "is_comparable_period"])
    assert bool(result.loc[1, "is_low_volume_period"])
    assert bool(result.loc[2, "is_partial_period"])
    assert len(result) == 3


def test_mask_seller_id_hides_middle_characters() -> None:
    """Public seller identifiers do not expose the full raw ID."""
    assert mask_seller_id("1234567890abcdef") == "12345678...cdef"


def test_rfm_copy_avoids_unproven_loyalty_claims() -> None:
    """RFM helper copy stays relative and descriptive."""
    text = rfm_segment_description("Loyal")
    assert "Higher relative RFM score" in text
    assert "strong repeat" not in text.lower()


def test_late_delivery_rate_uses_delivered_valid_denominator() -> None:
    """Late rate excludes undelivered rows and missing late flags."""
    frame = pd.DataFrame(
        {
            "order_status": ["delivered", "delivered", "canceled", "delivered"],
            "is_late": [True, False, True, None],
        }
    )

    assert late_delivery_rate_from_flags(frame) == 0.5


def test_build_demand_backtest_input_adds_target_period_flags() -> None:
    """Dashboard export keeps forecast-month comparability flags."""
    demand_features = pd.DataFrame(
        {
            "category_name_english": ["books"],
            "forecast_month": ["2018-06"],
            "feature_cutoff_month": ["2018-05"],
            "target_units": [10],
            "lag_1": [8],
            "lag_2": [7],
            "lag_4": [5],
            "rolling_mean_3": [6.5],
            "month_num": [6],
            "is_q4": [0],
        }
    )
    coverage = pd.DataFrame(
        {
            "order_year_month": ["2018-06"],
            "order_count": [200],
            "is_partial_period": [False],
            "is_low_volume_period": [False],
            "is_comparable_period": [True],
        }
    )

    result = build_demand_backtest_input(demand_features, coverage)

    assert result.loc[0, "target_period_order_count"] == 200
    assert bool(result.loc[0, "is_comparable_period"])


def test_dashboard_metric_labels_are_precise() -> None:
    """Terminology helpers return the release-gate wording."""
    assert seller_monetary_label() == "Seller-Attributed Order Value (item + freight)"
    assert payment_method_label() == "Orders by Primary Payment Type"
    assert (
        late_delivery_rate_definition()
        == "Late-delivery rate = delivered orders received after the estimated delivery date divided by all delivered orders with valid actual and estimated delivery dates."
    )
