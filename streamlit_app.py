"""Local Streamlit portfolio dashboard for CloudIQ."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ui import (
    PRODUCT_TRANSLATION_FALLBACK_NOTE,
    apply_glass_theme,
    glass_card,
    info_callout,
    methodology_notes_without_duplicate_fallback,
    metric_card,
    payment_method_label,
    plotly_glass_layout,
    rfm_segment_description,
    section_header,
    seller_monetary_label,
    status_badge,
)
from src.ui.aws_s3_status import (
    AWS_S3_EXPLANATORY_CAPTION,
    load_s3_manifest_status,
)

ROOT = Path(__file__).resolve().parent
DASHBOARD_DIR = ROOT / "data" / "dashboard"

st.set_page_config(
    page_title="CloudIQ - E-Commerce ML Intelligence Platform",
    page_icon="CIQ",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def read_csv(name: str) -> pd.DataFrame:
    """Read a dashboard CSV mart."""
    try:
        return pd.read_csv(DASHBOARD_DIR / name)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def read_json(name: str) -> dict[str, Any]:
    """Read a dashboard JSON mart."""
    return json.loads((DASHBOARD_DIR / name).read_text(encoding="utf-8"))


@st.cache_data(ttl=60, show_spinner=False)
def read_aws_s3_manifest_status() -> dict[str, Any]:
    """Read local S3 manifest evidence for dashboard display."""
    return load_s3_manifest_status().to_dict()


def money(value: float | int | None) -> str:
    """Format a revenue value."""
    amount = float(value or 0.0)
    if abs(amount) >= 1_000_000:
        return f"R$ {amount / 1_000_000:.2f}M"
    if abs(amount) >= 1_000:
        return f"R$ {amount / 1_000:.1f}K"
    return f"R$ {amount:,.0f}"


def number(value: float | int | None) -> str:
    """Format a count."""
    return f"{float(value or 0):,.0f}"


def percent(value: float | int | None) -> str:
    """Format a ratio as a percent."""
    return f"{float(value or 0.0) * 100:.1f}%"


def money_full(value: float | int | None) -> str:
    """Format a full Brazilian Real value."""
    return f"R$ {float(value or 0.0):,.2f}"


def metric_or_na(value: float | int | None, suffix: str = "", decimals: int = 2) -> str:
    """Format a metric value, preserving missingness as unavailable."""
    if value is None or pd.isna(value):
        return "Not available"
    return f"{float(value):.{decimals}f}{suffix}"


def style_fig(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Apply the CloudIQ Plotly glass theme."""
    fig.update_layout(**plotly_glass_layout(title))
    fig.update_traces(marker_line_width=0)
    return fig


def render_hero(compact: bool = False) -> None:
    """Render the dashboard hero."""
    hero_class = "cloudiq-hero compact" if compact else "cloudiq-hero"
    st.markdown(
        f"""
        <div class="{hero_class}">
          <h1>CloudIQ - E-Commerce ML Intelligence Platform</h1>
          <div class="cloudiq-subtitle">
            Historical Olist marketplace analysis | 2016-2018 | Local validated pipeline
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    status_badge("Historical data | Validated local pipeline | Not live production data")


def require_dashboard_data() -> bool:
    """Show setup instructions when dashboard marts are missing."""
    required = [
        "overview_kpis.json",
        "monthly_revenue.csv",
        "state_revenue.csv",
        "payment_mix.csv",
        "delivery_performance.csv",
        "rfm_segment_distribution.csv",
        "rfm_segment_profiles.csv",
        "seller_performance.csv",
        "demand_backtest_input.csv",
        "demand_backtest_predictions.csv",
        "demand_backtest_monthly_aggregate.csv",
        "demand_backtest_category_errors.csv",
        "demand_backtest_fold_metrics.csv",
        "demand_model_metrics.json",
        "data_quality_summary.json",
    ]
    missing = [name for name in required if not (DASHBOARD_DIR / name).exists()]
    if not missing:
        return True
    info_callout(
        "Dashboard data marts are missing. Run the local export command before opening the dashboard."
    )
    st.code("python scripts/export_dashboard_data.py\nstreamlit run streamlit_app.py")
    st.caption("Missing files: " + ", ".join(missing))
    return False


def overview_page() -> None:
    """Render the executive overview."""
    kpis = read_json("overview_kpis.json")
    monthly = read_csv("monthly_revenue.csv")
    state = read_csv("state_revenue.csv")
    payments = read_csv("payment_mix.csv")
    comparable_monthly = monthly[monthly["is_comparable_period"].astype(bool)]

    cols = st.columns(4)
    with cols[0]:
        metric_card("Total Orders", number(kpis["total_orders"]), "Silver master orders")
    with cols[1]:
        metric_card(
            "Item Merchandise Value (excludes freight)",
            money(kpis["item_merchandise_value_ex_freight"]),
            "Sum of item prices",
        )
    with cols[2]:
        metric_card(
            "Average Item Merchandise Value per Order",
            money(kpis["avg_item_merchandise_value_per_order"]),
            "Item value / orders",
        )
    with cols[3]:
        metric_card("Late Delivery Rate", percent(kpis["late_delivery_rate"]), "Delivered orders with late flag")

    section_header(
        "Item Revenue Trend",
        "Comparable monthly periods only; low-volume and partial periods are listed below.",
    )
    fig = px.line(
        comparable_monthly,
        x="order_year_month",
        y="item_merchandise_value_ex_freight",
        markers=True,
        labels={
            "order_year_month": "Month",
            "item_merchandise_value_ex_freight": "Item merchandise value (R$)",
        },
    )
    fig.add_bar(
        x=comparable_monthly["order_year_month"],
        y=comparable_monthly["order_count"],
        name="Orders",
        yaxis="y2",
        opacity=0.32,
        marker_color="#96f2c2",
    )
    fig.update_layout(
        yaxis2={
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "title": "Orders",
        }
    )
    fig.update_traces(
        hovertemplate="%{x}<br>R$ %{y:,.2f}<extra></extra>",
        selector={"type": "scatter"},
    )
    st.plotly_chart(style_fig(fig, "Monthly Item Revenue and Orders"), use_container_width=True)
    excluded = monthly[~monthly["is_comparable_period"].astype(bool)]
    with st.expander("Excluded partial or low-volume periods", expanded=False):
        st.dataframe(
            excluded[
                [
                    "order_year_month",
                    "order_count",
                    "is_partial_period",
                    "is_low_volume_period",
                    "item_merchandise_value_ex_freight",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    left, right = st.columns([1.15, 0.85])
    with left:
        section_header("Item Revenue by Customer State")
        state_fig = px.bar(
            state.head(15),
            x="customer_state",
            y="item_merchandise_value_ex_freight",
            color="item_merchandise_value_ex_freight",
            color_continuous_scale=["#7bd3ff", "#96f2c2"],
            labels={
                "customer_state": "State",
                "item_merchandise_value_ex_freight": "Item merchandise value (R$)",
            },
        )
        st.plotly_chart(
            style_fig(state_fig, "Top States by Item Merchandise Value"),
            use_container_width=True,
        )
    with right:
        section_header(payment_method_label())
        payment_fig = px.pie(
            payments,
            values="order_count",
            names="primary_payment_type",
            hole=0.58,
        )
        payment_fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(style_fig(payment_fig, payment_method_label()), use_container_width=True)

    section_header("What This Project Demonstrates")
    glass_card(
        "Local PySpark and Delta medallion processing, raw Olist audit checks, "
        "multiline-safe review ingestion, Silver conformed marts, leakage-safe "
        "Gold demand features, RFM segmentation, and chronological demand-model "
        "selection where the naive prior-month baseline beat XGBoost."
    )


def demand_page() -> None:
    """Render demand forecasting backtest results."""
    metrics = read_json("demand_model_metrics.json")
    predictions = read_csv("demand_backtest_predictions.csv")
    monthly = read_csv("demand_backtest_monthly_aggregate.csv")
    category_errors = read_csv("demand_backtest_category_errors.csv")
    folds = read_csv("demand_backtest_fold_metrics.csv")
    champion = metrics.get("champion_model", "naive_lag_1")
    champion_display = metrics.get("champion_display_name", "Naive Lag-1 Benchmark")
    metric_map = metrics.get("metrics", {})
    champion_metrics = metric_map.get(champion, {})
    validation = metrics.get("validation", {}).get("date_range", {})
    validation_meta = metrics.get("validation", {})

    section_header(
        "Demand Forecast Backtest",
        "Out-of-fold rolling validation using forecast_month targets and prior-month features.",
    )
    glass_card(
        (
            "<div class='metric-label'>Champion</div>"
            f"<div class='metric-value'>{champion_display}</div>"
            f"<div class='metric-caption'>{metrics.get('champion_reason', '')}</div>"
        )
    )

    cols = st.columns(4)
    with cols[0]:
        metric_card("MAE", metric_or_na(champion_metrics.get("mae")), f"Champion: {champion}")
    with cols[1]:
        metric_card("WAPE", metric_or_na(champion_metrics.get("wape"), "%", 1), "Primary pooled metric")
    with cols[2]:
        metric_card("RMSE", metric_or_na(champion_metrics.get("rmse")), "Secondary metric")
    with cols[3]:
        metric_card(
            "Evaluation Rows",
            number(validation_meta.get("row_count")),
            f"{validation_meta.get('fold_count', 0)} folds",
        )

    with st.expander("Naive vs XGBoost metrics", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {"model": model_name, **model_metrics}
                    for model_name, model_metrics in metric_map.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("WAPE and MAE drive model selection. RMSE, sMAPE, MAPE, and R2 are diagnostic.")

    section_header(
        "Monthly Aggregate Actual vs Predicted",
        "The chart uses only the same out-of-fold rows included in the reported metrics.",
    )
    fig = go.Figure()
    if not monthly.empty and {"forecast_month_label", "actual_units", "champion_prediction"}.issubset(monthly.columns):
        fig.add_trace(
            go.Scatter(
                x=monthly["forecast_month_label"],
                y=monthly["actual_units"],
                mode="lines+markers",
                name="Actual units",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=monthly["forecast_month_label"],
                y=monthly["champion_prediction"],
                mode="lines+markers",
                name="Champion prediction",
            )
        )
        st.plotly_chart(
            style_fig(fig, "Rolling Backtest: Actual vs Predicted Units"),
            use_container_width=True,
        )
    else:
        info_callout("Demand backtest monthly aggregates are missing. Run the local export and training commands.")

    section_header("Category-Level Error", "Absolute unit error across the same out-of-fold validation sample.")
    if category_errors.empty:
        info_callout("No category-level backtest errors are available yet.")
    else:
        categories = sorted(category_errors["category_name_english"].dropna().unique().tolist())
        selected = st.selectbox("Category", ["All categories", *categories])
        category_error = category_errors.copy()
        if selected != "All categories":
            category_error = category_error[category_error["category_name_english"] == selected]
        top_errors = category_error.sort_values("absolute_error", ascending=False).head(20)
        err_fig = px.bar(
            top_errors,
            x="absolute_error",
            y="category_name_english",
            orientation="h",
            labels={
                "absolute_error": "Absolute unit error",
                "category_name_english": "Category",
            },
        )
        err_fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(style_fig(err_fig, "Top Category Errors"), use_container_width=True)
        st.dataframe(top_errors, use_container_width=True, hide_index=True)

    info_callout(
        "Historical out-of-fold validation: "
        f"{validation.get('start', '')} to {validation.get('end', '')}. "
        "This is not a live forecast."
    )
    if not predictions.empty:
        info_callout("Naive baseline prediction equals lag_1 for every evaluated category-month row.")
    with st.expander("Fold coverage", expanded=False):
        if folds.empty:
            info_callout("Rolling fold metrics are not available yet.")
        else:
            st.dataframe(folds, use_container_width=True, hide_index=True)


def segmentation_page() -> None:
    """Render customer segmentation outputs."""
    distribution = read_csv("rfm_segment_distribution.csv")
    profiles = read_csv("rfm_segment_profiles.csv")

    section_header("RFM Segment Distribution", "Descriptive segmentation from Gold RFM scores.")
    fig = px.bar(
        distribution,
        x="segment_label",
        y="customers",
        color="segment_label",
        color_discrete_sequence=["#7bd3ff", "#96f2c2", "#ffd38a", "#ff9db2", "#b6a7ff"],
        labels={"segment_label": "Segment", "customers": "Customers"},
    )
    st.plotly_chart(style_fig(fig, "Customers by RFM Segment"), use_container_width=True)

    cols = st.columns(5)
    for idx, segment in enumerate(["Champion", "Loyal", "Potential", "At Risk", "Lost"]):
        with cols[idx]:
            metric_card(
                segment,
                number(distribution.loc[distribution["segment_label"] == segment, "customers"].sum()),
                rfm_segment_description(segment),
            )

    section_header(
        "Segment Profiles",
        "Aggregated only; repeat rate is shown because average order counts are close to one.",
    )
    st.dataframe(profiles, use_container_width=True, hide_index=True)
    info_callout(
        "RFM segments are relative historical behavior tiers. In this dataset, repeat purchasing is limited, so segment labels do not imply proven customer loyalty or churn risk."
    )
    info_callout("Segmentation is descriptive RFM analysis and is not a churn classifier.")


def operations_page() -> None:
    """Render operations and seller intelligence."""
    sellers = read_csv("seller_performance.csv")
    delivery = read_csv("delivery_performance.csv")

    states = ["All"] + sorted(sellers["seller_state"].dropna().unique().tolist())
    tiers = ["All"] + sorted(sellers["performance_tier"].dropna().unique().tolist())
    col_a, col_b = st.columns(2)
    with col_a:
        state_filter = st.selectbox("Seller state", states)
    with col_b:
        tier_filter = st.selectbox("Performance tier", tiers)

    filtered = sellers.copy()
    if state_filter != "All":
        filtered = filtered[filtered["seller_state"] == state_filter]
    if tier_filter != "All":
        filtered = filtered[filtered["performance_tier"] == tier_filter]

    section_header(
        "Seller Performance",
        f"{seller_monetary_label()}; full raw seller IDs are not shown.",
    )
    display = filtered.head(250).copy()
    display["total_orders"] = display["total_orders"].map(lambda value: f"{int(value):,}")
    display["seller_attributed_order_value"] = display[
        "seller_attributed_order_value"
    ].map(money_full)
    display["avg_review"] = display["avg_review"].map(
        lambda value: "" if pd.isna(value) else f"{float(value):.2f}"
    )
    display["late_rate"] = display["late_rate"].map(
        lambda value: "" if pd.isna(value) else percent(value)
    )
    display = display.rename(
        columns={
            "seller_rank": "Rank",
            "seller_id_short": "Seller ID",
            "seller_state": "State",
            "total_orders": "Orders",
            "seller_attributed_order_value": seller_monetary_label(),
            "avg_review": "Avg Review",
            "late_rate": "Late Rate",
            "performance_tier": "Tier",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)

    section_header("Late Delivery Performance", "Months below 100 delivered orders are flagged and excluded from the default trend.")
    delivery_plot = delivery[
        (~delivery["is_low_delivered_volume_period"].astype(bool))
        & (delivery["late_delivery_rate_display"].notna())
    ]
    delivery_fig = px.line(
        delivery_plot,
        x="order_year_month",
        y="late_delivery_rate_display",
        markers=True,
        labels={
            "order_year_month": "Month",
            "late_delivery_rate_display": "Late delivery rate",
        },
        custom_data=["delivered_order_count"],
    )
    delivery_fig.update_traces(
        hovertemplate="%{x}<br>Late rate: %{y:.1%}<br>Delivered orders: %{customdata[0]:,}<extra></extra>"
    )
    st.plotly_chart(style_fig(delivery_fig, "Monthly Late-Delivery Rate"), use_container_width=True)
    with st.expander("Flagged low-denominator delivery periods", expanded=False):
        st.dataframe(
            delivery.loc[
                delivery["is_low_delivered_volume_period"].astype(bool),
                [
                    "order_year_month",
                    "order_count",
                    "delivered_order_count",
                    "late_delivery_rate",
                    "is_partial_period",
                    "is_low_volume_period",
                ],
            ],
            use_container_width=True,
            hide_index=True,
        )

    if not filtered.empty:
        top_state = (
            filtered.groupby("seller_state", as_index=False)[
                "seller_attributed_order_value"
            ]
            .sum()
            .sort_values("seller_attributed_order_value", ascending=False)
            .iloc[0]
        )
        top_tier = filtered["performance_tier"].value_counts().idxmax()
        info_callout(
            "For the current filters, the highest seller-attributed value state is "
            f"{top_state['seller_state']} ({money(top_state['seller_attributed_order_value'])}), "
            f"and the most common performance tier is {top_tier}."
        )
    else:
        info_callout("No sellers match the current filters.")


def quality_page() -> None:
    """Render data quality and methodology notes."""
    quality = read_json("data_quality_summary.json")
    audit = quality.get("source_audit", {})
    validation = quality.get("pipeline_validation", {})
    metric_definitions = quality.get("metric_definitions", {})
    source_characteristics = quality.get("source_characteristics", {})

    cols = st.columns(3)
    with cols[0]:
        metric_card("Source Audit Verdict", str(audit.get("executive_verdict", "unknown")), "Raw Olist audit")
    with cols[1]:
        metric_card(
            "Bronze Reconciliation",
            str(audit.get("bronze_reconciliation", "unknown")),
            "Raw-to-Bronze counts",
        )
    with cols[2]:
        metric_card("Demand Feature Rows", number(quality.get("demand_feature_rows")), "Gold demand_features")

    section_header("Pipeline Record Counts", "Counts are read from local Delta outputs during dashboard export.")
    tabs = st.tabs(["Bronze", "Silver", "Gold"])
    for tab, layer in zip(tabs, ["bronze_row_counts", "silver_row_counts", "gold_row_counts"]):
        with tab:
            counts = validation.get(layer, {})
            frame = pd.DataFrame(
                [{"table": table, "rows": rows} for table, rows in counts.items()]
            )
            st.dataframe(frame, use_container_width=True, hide_index=True)

    section_header("Methodology Notes")
    for note in methodology_notes_without_duplicate_fallback(
        quality.get("methodology_notes", [])
    ):
        info_callout(note)

    glass_card(PRODUCT_TRANSLATION_FALLBACK_NOTE)

    s3_status = read_aws_s3_manifest_status()
    section_header("AWS S3 Raw Landing Zone")
    s3_cols = st.columns(4)
    with s3_cols[0]:
        metric_card("Storage Layer", str(s3_status["storage_layer"]), "Optional object storage")
    with s3_cols[1]:
        metric_card(
            "Source Files Verified",
            str(s3_status["source_files_verified"]),
            "Expected 9",
        )
    with s3_cols[2]:
        metric_card("Manifest Status", str(s3_status["manifest_status"]), "Local evidence")
    with s3_cols[3]:
        metric_card("Validation Mode", str(s3_status["validation_mode"]), "No live S3 reads")
    if s3_status["generated_at_display"]:
        st.markdown(
            '<div class="aws-generated-at">'
            f"Manifest generated: {escape(str(s3_status['generated_at_display']))}"
            "</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div class="aws-evidence-caption">{escape(AWS_S3_EXPLANATORY_CAPTION)}</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Manifest Details", expanded=False):
        if s3_status["generated_at_display"]:
            st.caption(f"Generated timestamp: {s3_status['generated_at_display']}")
        if s3_status["files"]:
            st.dataframe(
                pd.DataFrame(s3_status["files"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            info_callout(str(s3_status["error_message"] or "Not verified locally yet."))
            st.code(str(s3_status["remediation_command"]))

    section_header("Metric Definitions")
    for name, definition in metric_definitions.items():
        info_callout(f"{name}: {definition}")

    section_header("Source Characteristics")
    info_callout(str(source_characteristics.get("geolocation_note", "")))
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "characteristic": "geolocation_duplicate_zip_code_rows",
                    "value": source_characteristics.get("geolocation_duplicate_zip_code_rows"),
                },
                {
                    "characteristic": "geolocation_exact_duplicate_rows",
                    "value": source_characteristics.get("geolocation_exact_duplicate_rows"),
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    period_quality = quality.get("period_quality", {})
    section_header("Period Coverage Flags")
    st.caption(
        "Business trend charts default to comparable periods and keep excluded partial/low-volume months available for review."
    )
    st.dataframe(
        pd.DataFrame(period_quality.get("excluded_business_trend_periods", [])),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    """Run the dashboard."""
    apply_glass_theme()
    page = st.sidebar.radio(
        "Dashboard section",
        [
            "Executive Overview",
            "Demand Forecasting",
            "Customer Segmentation",
            "Operations & Seller Intelligence",
            "Data Quality & Methodology",
        ],
    )
    st.sidebar.caption("Historical dataset · validated local extracts")
    render_hero(compact=page != "Executive Overview")
    if not require_dashboard_data():
        return

    if page == "Executive Overview":
        overview_page()
    elif page == "Demand Forecasting":
        demand_page()
    elif page == "Customer Segmentation":
        segmentation_page()
    elif page == "Operations & Seller Intelligence":
        operations_page()
    else:
        quality_page()


if __name__ == "__main__":
    main()
