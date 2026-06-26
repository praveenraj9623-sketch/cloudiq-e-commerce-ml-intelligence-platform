"""Reusable HTML component helpers for the CloudIQ Streamlit dashboard."""

from __future__ import annotations

from html import escape

import streamlit as st


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
