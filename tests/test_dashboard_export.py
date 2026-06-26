"""Pure tests for dashboard export helper logic."""

from __future__ import annotations

from scripts.export_dashboard_data import safe_divide, summarize_overview_kpis


def test_safe_divide_returns_zero_for_empty_denominator() -> None:
    """Dashboard ratios stay finite when a denominator is unavailable."""
    assert safe_divide(10, 0) == 0.0
    assert safe_divide(10, None) == 0.0


def test_summarize_overview_kpis_uses_actual_aggregates() -> None:
    """Overview values derive from provided aggregate facts."""
    result = summarize_overview_kpis(
        total_orders=4,
        total_revenue=200.0,
        late_delivery_rate=0.25,
        first_month="2017-01",
        last_month="2018-07",
    )

    assert result["total_orders"] == 4
    assert result["total_revenue"] == 200.0
    assert result["avg_order_value"] == 50.0
    assert result["late_delivery_rate"] == 0.25
    assert result["first_order_month"] == "2017-01"
    assert result["last_order_month"] == "2018-07"
