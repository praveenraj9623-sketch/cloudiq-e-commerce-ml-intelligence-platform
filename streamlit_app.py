"""Local Streamlit portfolio dashboard for CloudIQ."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ui import (
    apply_glass_theme,
    glass_card,
    info_callout,
    metric_card,
    plotly_glass_layout,
    section_header,
    status_badge,
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
    return pd.read_csv(DASHBOARD_DIR / name)


@st.cache_data(show_spinner=False)
def read_json(name: str) -> dict[str, Any]:
    """Read a dashboard JSON mart."""
    return json.loads((DASHBOARD_DIR / name).read_text(encoding="utf-8"))


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


def style_fig(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Apply the CloudIQ Plotly glass theme."""
    fig.update_layout(**plotly_glass_layout(title))
    fig.update_traces(marker_line_width=0)
    return fig


def render_hero() -> None:
    """Render the dashboard hero."""
    st.markdown(
        """
        <div class="cloudiq-hero">
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
        "demand_validation_predictions.csv",
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

    cols = st.columns(4)
    with cols[0]:
        metric_card("Total Orders", number(kpis["total_orders"]), "Silver master orders")
    with cols[1]:
        metric_card("Total Revenue", money(kpis["total_revenue"]), "Order-item revenue")
    with cols[2]:
        metric_card("Average Order Value", money(kpis["avg_order_value"]), "Revenue / orders")
    with cols[3]:
        metric_card("Late Delivery Rate", percent(kpis["late_delivery_rate"]), "Delivered orders with late flag")

    section_header("Revenue Trend", "Monthly order volume and revenue from Gold BI revenue.")
    fig = px.line(
        monthly,
        x="order_year_month",
        y="total_revenue",
        markers=True,
        labels={"order_year_month": "Month", "total_revenue": "Revenue"},
    )
    fig.add_bar(
        x=monthly["order_year_month"],
        y=monthly["total_orders"],
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
    st.plotly_chart(style_fig(fig, "Monthly Revenue and Orders"), use_container_width=True)

    left, right = st.columns([1.15, 0.85])
    with left:
        section_header("Revenue by Customer State")
        state_fig = px.bar(
            state.head(15),
            x="customer_state",
            y="total_revenue",
            color="total_revenue",
            color_continuous_scale=["#7bd3ff", "#96f2c2"],
            labels={"customer_state": "State", "total_revenue": "Revenue"},
        )
        st.plotly_chart(style_fig(state_fig, "Top States by Revenue"), use_container_width=True)
    with right:
        section_header("Payment Mix")
        payment_fig = px.pie(
            payments,
            values="total_orders",
            names="primary_payment_type",
            hole=0.58,
        )
        payment_fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(style_fig(payment_fig, "Orders by Payment Type"), use_container_width=True)

    section_header("What This Project Demonstrates")
    glass_card(
        "Local PySpark and Delta medallion processing, raw Olist audit checks, "
        "multiline-safe review ingestion, Silver conformed marts, leakage-safe "
        "Gold demand features, RFM segmentation, and chronological demand-model "
        "selection where the naive prior-month baseline beat XGBoost."
    )


def demand_page() -> None:
    """Render demand forecasting results."""
    metrics = read_json("demand_model_metrics.json")
    predictions = read_csv("demand_validation_predictions.csv")
    champion = metrics.get("champion_model", "naive_lag_1")
    metric_map = metrics.get("metrics", {})
    naive = metric_map.get("naive_lag_1", {})
    xgb = metric_map.get("xgboost", {})
    validation = metrics.get("validation", {}).get("date_range", {})

    glass_card(
        (
            "<div class='metric-label'>Champion</div>"
            "<div class='metric-value'>Naive Prior-Month Baseline</div>"
            "<div class='metric-caption'>XGBoost was evaluated but not selected because it performed worse on chronological validation.</div>"
        )
    )

    cols = st.columns(4)
    with cols[0]:
        metric_card("MAE", f"{naive.get('mae', 0):.2f}", f"Champion: {champion}")
    with cols[1]:
        metric_card("RMSE", f"{naive.get('rmse', 0):.2f}", "Validation holdout")
    with cols[2]:
        metric_card("MAPE", f"{naive.get('mape', 0):.1f}%", "Zero-safe metric")
    with cols[3]:
        metric_card("R²", f"{naive.get('r2', 0):.3f}", "Chronological split")

    with st.expander("Naive vs XGBoost metrics", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {"model": "naive_lag_1", **naive},
                    {"model": "xgboost", **xgb},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    filtered = predictions.copy()
    if "category_name_english" in filtered.columns:
        categories = sorted(filtered["category_name_english"].dropna().unique().tolist())
        selected = st.selectbox("Category", ["All categories", *categories])
        if selected != "All categories":
            filtered = filtered[filtered["category_name_english"] == selected]

    section_header("Actual vs Predicted", "Validation predictions from the saved demand forecast report.")
    if filtered.empty:
        info_callout("No validation predictions are available for the selected category.")
    else:
        chart_frame = (
            filtered.groupby("order_year_month", as_index=False)[
                ["target_next_month", "champion_prediction"]
            ]
            .sum()
            .sort_values("order_year_month")
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=chart_frame["order_year_month"],
                y=chart_frame["target_next_month"],
                mode="lines+markers",
                name="Actual",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=chart_frame["order_year_month"],
                y=chart_frame["champion_prediction"],
                mode="lines+markers",
                name="Predicted",
            )
        )
        st.plotly_chart(style_fig(fig, "Validation: Actual vs Predicted Units"), use_container_width=True)

    info_callout(
        "Historical validation: "
        f"{validation.get('start', '2018-06')} to {validation.get('end', '2018-07')}. "
        "This is not a live forecast. MAPE can be unstable for low-volume categories."
    )


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

    descriptions = {
        "Champion": "Highest combined recency, frequency, and monetary scores.",
        "Loyal": "Strong repeat and value profile, below Champion threshold.",
        "Potential": "Middle RFM profile with room for engagement.",
        "At Risk": "Lower combined RFM score than active growth segments.",
        "Lost": "Lowest combined RFM profile in the historical data.",
    }
    cols = st.columns(5)
    for idx, (segment, text) in enumerate(descriptions.items()):
        with cols[idx]:
            metric_card(segment, number(distribution.loc[distribution["segment_label"] == segment, "customers"].sum()), text)

    section_header("Segment Profiles", "Aggregated only; no customer IDs are exported.")
    st.dataframe(profiles, use_container_width=True, hide_index=True)
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

    section_header("Seller Performance", "Seller-level revenue, order count, reviews, and late rate.")
    st.dataframe(filtered.head(250), use_container_width=True, hide_index=True)

    section_header("Late Delivery Performance")
    delivery_fig = px.line(
        delivery,
        x="order_year_month",
        y="late_delivery_rate",
        markers=True,
        labels={"order_year_month": "Month", "late_delivery_rate": "Late delivery rate"},
    )
    st.plotly_chart(style_fig(delivery_fig, "Monthly Late-Delivery Rate"), use_container_width=True)

    if not filtered.empty:
        top_state = (
            filtered.groupby("seller_state", as_index=False)["total_revenue"]
            .sum()
            .sort_values("total_revenue", ascending=False)
            .iloc[0]
        )
        top_tier = filtered["performance_tier"].value_counts().idxmax()
        info_callout(
            "For the current filters, the highest seller revenue state is "
            f"{top_state['seller_state']} ({money(top_state['total_revenue'])}), "
            f"and the most common performance tier is {top_tier}."
        )
    else:
        info_callout("No sellers match the current filters.")


def quality_page() -> None:
    """Render data quality and methodology notes."""
    quality = read_json("data_quality_summary.json")
    audit = quality.get("source_audit", {})
    validation = quality.get("pipeline_validation", {})

    cols = st.columns(3)
    with cols[0]:
        metric_card("Source Audit Verdict", str(audit.get("executive_verdict", "unknown")), "Raw Olist audit")
    with cols[1]:
        metric_card("Bronze Safe", str(audit.get("safe_to_run_bronze", "unknown")), "Audit decision")
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
    for note in quality.get("methodology_notes", []):
        info_callout(note)

    gap = audit.get("translation_lookup_gap", {})
    warnings = audit.get("warnings", [])
    glass_card(
        "Unmatched category translation fallback: "
        f"{gap.get('unmatched_product_row_count', 0)} product rows across "
        f"{gap.get('unmatched_distinct_category_count', 0)} categories. "
        "These are warnings, not raw-data failures."
    )
    if warnings:
        section_header("Data Quality Warnings")
        for warning in warnings:
            st.caption(f"- {warning}")


def main() -> None:
    """Run the dashboard."""
    apply_glass_theme()
    render_hero()
    if not require_dashboard_data():
        return

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
    st.sidebar.caption("Local compact marts in data/dashboard/.")

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
