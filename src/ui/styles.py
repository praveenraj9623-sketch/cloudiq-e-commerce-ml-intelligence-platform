"""Centralized glassmorphism styling for the Streamlit dashboard."""

from __future__ import annotations

from typing import Any

import streamlit as st


def apply_glass_theme() -> None:
    """Inject the CloudIQ liquid-glass dashboard CSS once per render."""
    st.markdown(
        """
        <style>
        :root {
          --cloudiq-bg: #071014;
          --cloudiq-panel: rgba(255,255,255,0.082);
          --cloudiq-panel-strong: rgba(255,255,255,0.13);
          --cloudiq-border: rgba(255,255,255,0.18);
          --cloudiq-text: #f4f8fb;
          --cloudiq-muted: #aebbc4;
          --cloudiq-blue: #7bd3ff;
          --cloudiq-green: #96f2c2;
          --cloudiq-amber: #ffd38a;
          --cloudiq-rose: #ff9db2;
        }

        .stApp {
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Arial, sans-serif;
          color: var(--cloudiq-text);
          background:
            radial-gradient(circle at 16% 6%, rgba(123,211,255,0.18), transparent 27rem),
            radial-gradient(circle at 80% 0%, rgba(150,242,194,0.13), transparent 24rem),
            linear-gradient(145deg, #061015 0%, #0d1820 43%, #11151d 100%);
        }

        .block-container {
          padding-top: 2.2rem;
          padding-bottom: 3rem;
          max-width: 1320px;
        }

        [data-testid="stSidebar"] {
          background: rgba(8, 18, 24, 0.84);
          border-right: 1px solid rgba(255,255,255,0.12);
        }

        [data-testid="stSidebar"] * {
          color: #e9f2f7;
        }

        .cloudiq-hero {
          padding: 1.4rem 1.6rem;
          margin-bottom: 1rem;
          border-radius: 24px;
          border: 1px solid var(--cloudiq-border);
          background: linear-gradient(135deg, rgba(255,255,255,0.14), rgba(255,255,255,0.055));
          box-shadow: 0 22px 70px rgba(0,0,0,0.34);
          backdrop-filter: blur(22px);
        }

        .cloudiq-hero h1 {
          margin: 0;
          letter-spacing: 0;
          font-size: clamp(2rem, 3.2vw, 3.7rem);
          line-height: 1.05;
        }

        .cloudiq-subtitle {
          color: var(--cloudiq-muted);
          margin-top: 0.55rem;
          font-size: 1rem;
        }

        .glass-card {
          padding: 1rem 1.1rem;
          border-radius: 20px;
          border: 1px solid var(--cloudiq-border);
          background: var(--cloudiq-panel);
          box-shadow: 0 14px 46px rgba(0,0,0,0.25);
          backdrop-filter: blur(18px);
          transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
        }

        .glass-card:hover {
          transform: translateY(-1px);
          border-color: rgba(255,255,255,0.28);
          background: var(--cloudiq-panel-strong);
        }

        .metric-card {
          min-height: 128px;
        }

        .metric-label {
          color: var(--cloudiq-muted);
          font-size: 0.82rem;
          text-transform: uppercase;
          letter-spacing: 0;
          margin-bottom: 0.45rem;
        }

        .metric-value {
          color: var(--cloudiq-text);
          font-size: 1.75rem;
          font-weight: 700;
          line-height: 1.16;
        }

        .metric-caption {
          color: var(--cloudiq-muted);
          font-size: 0.85rem;
          margin-top: 0.45rem;
        }

        .status-badge {
          display: inline-flex;
          align-items: center;
          gap: 0.4rem;
          padding: 0.42rem 0.68rem;
          border-radius: 999px;
          border: 1px solid rgba(150,242,194,0.32);
          color: #dcffec;
          background: rgba(150,242,194,0.11);
          font-size: 0.82rem;
          font-weight: 650;
        }

        .section-title {
          margin: 1.4rem 0 0.35rem;
          font-size: 1.25rem;
          line-height: 1.2;
          font-weight: 720;
        }

        .section-caption {
          margin: 0 0 0.9rem;
          color: var(--cloudiq-muted);
          font-size: 0.93rem;
        }

        .info-callout {
          border-left: 3px solid var(--cloudiq-blue);
          padding: 0.88rem 1rem;
          border-radius: 16px;
          background: rgba(123,211,255,0.10);
          color: #ddecf5;
          border-top: 1px solid rgba(255,255,255,0.10);
          border-right: 1px solid rgba(255,255,255,0.10);
          border-bottom: 1px solid rgba(255,255,255,0.10);
        }

        div[data-testid="stDataFrame"] {
          border-radius: 16px;
          overflow: hidden;
          border: 1px solid rgba(255,255,255,0.14);
        }

        .stTabs [data-baseweb="tab-list"] {
          gap: 0.35rem;
        }

        .stTabs [data-baseweb="tab"] {
          border-radius: 999px;
          padding: 0.42rem 0.8rem;
          background: rgba(255,255,255,0.06);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def plotly_glass_layout(title: str | None = None) -> dict[str, Any]:
    """Return a Plotly layout for transparent glass-friendly charts."""
    return {
        "title": {"text": title or "", "font": {"color": "#f4f8fb", "size": 18}},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#e8f0f5", "family": "-apple-system, Segoe UI, Inter, Arial"},
        "legend": {"font": {"color": "#dbe8ef"}},
        "margin": {"l": 20, "r": 20, "t": 54 if title else 20, "b": 34},
        "xaxis": {
            "gridcolor": "rgba(255,255,255,0.09)",
            "zerolinecolor": "rgba(255,255,255,0.12)",
            "linecolor": "rgba(255,255,255,0.16)",
        },
        "yaxis": {
            "gridcolor": "rgba(255,255,255,0.09)",
            "zerolinecolor": "rgba(255,255,255,0.12)",
            "linecolor": "rgba(255,255,255,0.16)",
        },
        "hoverlabel": {
            "bgcolor": "rgba(12, 22, 30, 0.96)",
            "bordercolor": "rgba(255,255,255,0.18)",
            "font": {"color": "#f4f8fb"},
        },
    }
