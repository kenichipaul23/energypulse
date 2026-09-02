"""
EnergyPulse - Theme & UI Helpers
===================================
Pure presentation layer. Nothing in here touches data loading, feature
engineering, or modeling — nothing in models.py / features.py / data_utils.py
changes because of this file.

Design concept: "Substation control room at night."
  - Deep navy/graphite canvas, like monitoring a power grid after hours.
  - One signature accent — "voltage amber" — used for the EnergyPulse brand,
    primary actions, and anything that represents "your data" in general.
  - A cool cyan used only for the forecasting (prediction) line, so actual
    vs. predicted is instantly readable.
  - Three semantic colors (green / blue / red) are reserved ONLY for the
    Off-Peak / Standard / Peak classes, and used identically everywhere
    (KPI cards, charts, badges) so the color always means the same thing.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#0A0E1A",
    "surface": "#121826",
    "surface_raised": "#1A2235",
    "border": "#232B3E",
    "text": "#EDF1F7",
    "text_muted": "#8A94AC",
    "amber": "#FFC857",       # brand / primary accent ("voltage")
    "cyan": "#4FD9D3",        # forecast / predicted-value accent
    "offpeak": "#34D399",     # green
    "standard": "#60A5FA",    # blue
    "peak": "#F87171",        # red
}

CHART_TEMPLATE = "plotly_dark"


def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        /* ---- Canvas ---- */
        [data-testid="stAppViewContainer"] {{
            background: {COLORS['bg']};
        }}
        [data-testid="stHeader"] {{
            background: transparent;
        }}
        .block-container {{
            padding-top: 2rem;
            max-width: 1200px;
        }}

        /* ---- Sidebar ---- */
        [data-testid="stSidebar"] {{
            background: {COLORS['surface']};
            border-right: 1px solid {COLORS['border']};
        }}
        [data-testid="stSidebar"] .block-container {{
            padding-top: 2rem;
        }}

        /* ---- Headings use the display font ---- */
        h1, h2, h3, h4 {{
            font-family: 'Space Grotesk', sans-serif !important;
            color: {COLORS['text']} !important;
            letter-spacing: -0.01em;
        }}
        p, span, label, div {{
            color: {COLORS['text']};
        }}
        .ep-muted {{
            color: {COLORS['text_muted']} !important;
        }}

        /* ---- Tabs: make the active section unmistakable ---- */
        [data-baseweb="tab-list"] {{
            gap: 4px;
            background: {COLORS['surface']};
            padding: 6px;
            border-radius: 12px;
            border: 1px solid {COLORS['border']};
        }}
        [data-baseweb="tab"] {{
            height: 44px;
            border-radius: 8px !important;
            color: {COLORS['text_muted']} !important;
            font-weight: 600;
            font-family: 'Space Grotesk', sans-serif;
            background: transparent !important;
        }}
        [data-baseweb="tab"]:hover {{
            background: {COLORS['surface_raised']} !important;
            color: {COLORS['text']} !important;
        }}
        [data-baseweb="tab"][aria-selected="true"] {{
            background: {COLORS['amber']} !important;
            color: #16130A !important;
        }}
        [data-baseweb="tab-highlight"] {{
            background: transparent !important;
        }}
        [data-baseweb="tab-border"] {{
            display: none;
        }}

        /* ---- Metrics -> KPI cards ---- */
        [data-testid="stMetric"] {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-left: 3px solid {COLORS['amber']};
            border-radius: 10px;
            padding: 14px 16px;
        }}
        [data-testid="stMetricLabel"] {{
            color: {COLORS['text_muted']} !important;
        }}
        [data-testid="stMetricValue"] {{
            color: {COLORS['text']} !important;
            font-family: 'Space Grotesk', sans-serif;
        }}

        /* ---- Callout / explainer boxes ---- */
        .ep-callout {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-left: 3px solid var(--accent, {COLORS['amber']});
            border-radius: 10px;
            padding: 14px 18px;
            margin: 4px 0 20px 0;
            font-size: 0.92rem;
            color: {COLORS['text_muted']};
            line-height: 1.5;
        }}
        .ep-callout b {{ color: {COLORS['text']}; }}

        /* ---- Section label above a chart ---- */
        .ep-section {{
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            font-size: 1.05rem;
            color: {COLORS['text']};
            margin: 28px 0 4px 0;
        }}
        .ep-section-desc {{
            color: {COLORS['text_muted']};
            font-size: 0.88rem;
            margin-bottom: 10px;
        }}

        /* ---- Hero header ---- */
        .ep-hero-badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            padding: 5px 12px;
            border-radius: 100px;
            font-size: 0.8rem;
            color: {COLORS['text_muted']};
            margin-bottom: 14px;
        }}
        .ep-dot {{
            width: 7px; height: 7px; border-radius: 50%;
            background: {COLORS['amber']};
            box-shadow: 0 0 8px {COLORS['amber']};
            display: inline-block;
        }}

        /* ---- Class badges (Off-Peak / Standard / Peak) ---- */
        .ep-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 10px;
            border-radius: 100px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 6px;
        }}

        /* ---- Divider ---- */
        .ep-hr {{
            border: none;
            border-top: 1px solid {COLORS['border']};
            margin: 8px 0 24px 0;
        }}

        /* ---- Sidebar footer note ---- */
        .ep-sidebar-note {{
            background: {COLORS['surface_raised']};
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            padding: 10px 12px;
            font-size: 0.82rem;
            color: {COLORS['text_muted']};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, tagline: str, badge: str):
    st.markdown(
        f"""
        <div class="ep-hero-badge"><span class="ep-dot"></span>{badge}</div>
        <h1 style="margin-bottom:2px;">{title}</h1>
        <p class="ep-muted" style="font-size:1.02rem; margin-top:0;">{tagline}</p>
        <hr class="ep-hr">
        """,
        unsafe_allow_html=True,
    )


def callout(text_html: str, accent: str = None):
    """A left-accented explainer box — used to orient a first-time viewer
    before showing them a chart or a set of metrics."""
    accent = accent or COLORS["amber"]
    st.markdown(
        f"""<div class="ep-callout" style="--accent: {accent};">{text_html}</div>""",
        unsafe_allow_html=True,
    )


def section(title: str, desc: str = ""):
    """A styled sub-heading + one-line plain-language description, used
    above every chart so a new viewer knows what they're looking at."""
    st.markdown(f'<div class="ep-section">{title}</div>', unsafe_allow_html=True)
    if desc:
        st.markdown(f'<div class="ep-section-desc">{desc}</div>', unsafe_allow_html=True)


def badge(text: str, color: str) -> str:
    return (
        f'<span class="ep-badge" style="background:{color}22; color:{color}; '
        f'border:1px solid {color}55;">● {text}</span>'
    )


def style_fig(fig, height=350):
    """Apply the dark control-room chart styling consistently everywhere."""
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=height,
        margin=dict(t=20, b=20, l=10, r=10),
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font=dict(family="Inter, sans-serif", color=COLORS["text_muted"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
        yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
    )
    return fig