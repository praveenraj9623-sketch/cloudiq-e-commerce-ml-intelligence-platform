"""Reusable HTML component helpers for the CloudIQ Streamlit dashboard."""

from __future__ import annotations

from html import escape

import streamlit as st

PRODUCT_TRANSLATION_FALLBACK_NOTE = (
    "13 product rows across 2 categories retained using untranslated__ fallback labels."
)


def glass_card(body: str, extra_class: str = "") -> None:
    """Render a frosted glass card with trusted, preformatted HTML body."""
    st.markdown(
        f'<div class="glass-card {escape(extra_class)}">{body}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, caption: str = "") -> None:
    """Render a dashboard metric card."""
    caption_html = (
        f'<div class="metric-caption">{escape(caption)}</div>' if caption else ""
    )
    glass_card(
        (
            f'<div class="metric-label">{escape(label)}</div>'
            f'<div class="metric-value">{escape(value)}</div>'
            f"{caption_html}"
        ),
        "metric-card",
    )


def section_header(title: str, caption: str = "") -> None:
    """Render a consistent section heading."""
    caption_html = (
        f'<div class="section-caption">{escape(caption)}</div>' if caption else ""
    )
    st.markdown(
        f'<div class="section-title">{escape(title)}</div>{caption_html}',
        unsafe_allow_html=True,
    )


def status_badge(text: str) -> None:
    """Render a status badge."""
    st.markdown(
        f'<span class="status-badge">{escape(text)}</span>',
        unsafe_allow_html=True,
    )


def info_callout(text: str) -> None:
    """Render a concise glass callout."""
    st.markdown(
        f'<div class="info-callout">{escape(text)}</div>',
        unsafe_allow_html=True,
    )


def rfm_segment_description(segment: str) -> str:
    """Return evidence-conscious copy for an RFM segment label."""
    descriptions = {
        "Champion": "Highest relative RFM score within this historical marketplace dataset.",
        "Loyal": "Higher relative RFM score within this historical marketplace dataset.",
        "Potential": "Middle relative RFM tier based on recency, frequency, and monetary scores.",
        "At Risk": "Lower relative RFM tier within the historical dataset.",
        "Lost": "Lowest relative RFM tier within the historical dataset.",
    }
    return descriptions.get(segment, "Relative historical RFM behavior tier.")


def seller_monetary_label() -> str:
    """Return the precise seller value label used by the dashboard."""
    return "Seller-Attributed Order Value (item + freight)"


def payment_method_label() -> str:
    """Return the deterministic primary-payment chart label."""
    return "Orders by Primary Payment Type"


def late_delivery_rate_definition() -> str:
    """Return the exact dashboard definition for late-delivery rate."""
    return (
        "Late-delivery rate = delivered orders received after the estimated delivery "
        "date divided by all delivered orders with valid actual and estimated delivery dates."
    )


def methodology_notes_without_duplicate_fallback(notes: list[str]) -> list[str]:
    """Return methodology notes without duplicating the fixed fallback note."""
    return [
        str(note)
        for note in notes
        if str(note).strip() != PRODUCT_TRANSLATION_FALLBACK_NOTE
    ]
