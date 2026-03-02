# ============================================================
# STREAMLIT FRONTEND FOR DISCOURSE FEATURE ANALYZER
# ============================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import time

# Configuration
API_URL = "http://127.0.0.1:8000/extract-features"
API_HEALTH_URL = "http://127.0.0.1:8000/health"

st.set_page_config(
    page_title="Discourse Analyzer",
    layout="wide",
    page_icon="◈",
    initial_sidebar_state="expanded"
)

# ============================================================
# DESIGN SYSTEM — Refined Editorial / Scientific Instrument
# ============================================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0D0F14;
    color: #E8E6E1;
    font-family: 'DM Sans', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #111318;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] > div:first-child {
    padding: 2rem 1.25rem;
}

/* ── Main content area ── */
[data-testid="stMain"] {
    background-color: #0D0F14;
}
.block-container {
    padding: 2.5rem 3rem 4rem;
    max-width: 1400px;
}

/* ── Typography ── */
h1 { font-family: 'DM Serif Display', serif !important; }
h2, h3 { font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important; }
code, .mono { font-family: 'DM Mono', monospace !important; }

/* ── Page header ── */
.page-header {
    display: flex;
    align-items: flex-start;
    gap: 1.5rem;
    padding: 2.5rem 3rem;
    margin: -2.5rem -3rem 3rem;
    background: linear-gradient(135deg, #0D0F14 0%, #131720 60%, #0D1220 100%);
    border-bottom: 1px solid rgba(255,255,255,0.07);
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(90,140,220,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.page-header-glyph {
    font-size: 2.8rem;
    line-height: 1;
    color: #5A8CDC;
    font-family: 'DM Serif Display', serif;
    margin-top: 0.15rem;
}
.page-header-text h1 {
    font-size: 2.1rem !important;
    font-weight: 400 !important;
    color: #E8E6E1;
    margin: 0 0 0.35rem !important;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.page-header-text p {
    color: #6B7280;
    font-size: 0.875rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.05em;
    margin: 0;
}

/* ── Status pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    letter-spacing: 0.04em;
    margin-bottom: 0.5rem;
}
.status-online {
    background: rgba(52, 211, 153, 0.12);
    color: #34D399;
    border: 1px solid rgba(52, 211, 153, 0.25);
}
.status-offline {
    background: rgba(239, 68, 68, 0.12);
    color: #EF4444;
    border: 1px solid rgba(239, 68, 68, 0.25);
}
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: currentColor;
    animation: pulse-dot 2s ease infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Section headers ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4B5563;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.06);
}

/* ── Metric cards ── */
.metric-strip {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1px;
    background: rgba(255,255,255,0.06);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 2.5rem;
}
.metric-cell {
    background: #131720;
    padding: 1.25rem 1.5rem;
}
.metric-cell:first-child { border-radius: 12px 0 0 12px; }
.metric-cell:last-child  { border-radius: 0 12px 12px 0; }
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4B5563;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.75rem;
    color: #E8E6E1;
    line-height: 1;
}
.metric-unit {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #6B7280;
    margin-top: 0.25rem;
}

/* ── Feature category chips ── */
.cat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2.5rem;
}
.cat-chip {
    background: #131720;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.cat-chip::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--accent, #5A8CDC);
    border-radius: 10px 0 0 10px;
}
.cat-chip-name {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6B7280;
    margin-bottom: 0.4rem;
}
.cat-chip-count {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #E8E6E1;
    line-height: 1;
}
.cat-chip-unit {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #4B5563;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: #131720;
    border-radius: 10px;
    padding: 0.25rem;
    gap: 0.25rem;
    border: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 2rem;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
    color: #6B7280 !important;
    border-radius: 7px !important;
    padding: 0.5rem 1.1rem !important;
    border: none !important;
    transition: all 0.15s ease !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #1E2430 !important;
    color: #E8E6E1 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
}
[data-testid="stTabs"] [role="tab"]:hover:not([aria-selected="true"]) {
    color: #9CA3AF !important;
    background: rgba(255,255,255,0.04) !important;
}

/* ── Text area ── */
[data-testid="stTextArea"] textarea {
    background: #131720 !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 10px !important;
    color: #E8E6E1 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    line-height: 1.7 !important;
    padding: 1rem 1.25rem !important;
    resize: vertical !important;
    transition: border-color 0.2s !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(90,140,220,0.5) !important;
    box-shadow: 0 0 0 3px rgba(90,140,220,0.07) !important;
    outline: none !important;
}
[data-testid="stTextArea"] textarea::placeholder {
    color: #374151 !important;
}
[data-testid="stTextArea"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #4B5563 !important;
}

/* ── Primary button ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: #5A8CDC !important;
    color: #fff !important;
    border: none !important;
    border-radius: 9px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.06em !important;
    font-weight: 500 !important;
    padding: 0.65rem 2rem !important;
    transition: all 0.2s ease !important;
    text-transform: uppercase !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover:not(:disabled) {
    background: #4A7ACC !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(90,140,220,0.35) !important;
}
[data-testid="stButton"] > button[kind="primary"]:disabled {
    background: #1E2430 !important;
    color: #374151 !important;
    cursor: not-allowed !important;
}

/* ── Secondary button ── */
[data-testid="stButton"] > button[kind="secondary"],
[data-testid="stDownloadButton"] button {
    background: #1A1F2C !important;
    color: #9CA3AF !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.04em !important;
}
[data-testid="stButton"] > button[kind="secondary"]:hover,
[data-testid="stDownloadButton"] button:hover {
    border-color: rgba(255,255,255,0.2) !important;
    color: #E8E6E1 !important;
    background: #1E2430 !important;
}

/* ── Checkboxes ── */
[data-testid="stCheckbox"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #9CA3AF !important;
    letter-spacing: 0.03em !important;
}
[data-testid="stCheckbox"] label:hover { color: #E8E6E1 !important; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #131720 !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 8px !important;
    color: #9CA3AF !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* ── Text input ── */
[data-testid="stTextInput"] input {
    background: #131720 !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 8px !important;
    color: #E8E6E1 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
}
[data-testid="stTextInput"] input::placeholder { color: #374151 !important; }
[data-testid="stTextInput"] input:focus {
    border-color: rgba(90,140,220,0.4) !important;
    box-shadow: 0 0 0 3px rgba(90,140,220,0.07) !important;
}
[data-testid="stTextInput"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #4B5563 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}
[data-testid="stDataFrame"] iframe {
    border-radius: 10px !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 9px !important;
    border: none !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #131720 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    color: #9CA3AF !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #6B7280 !important;
    letter-spacing: 0.05em !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.06) !important;
    margin: 2rem 0 !important;
}

/* ── Sidebar labels ── */
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stSubheader {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #4B5563 !important;
    font-weight: 500 !important;
}

/* ── Success/Error states ── */
[data-testid="stAlert"][data-baseweb="notification"] {
    background: #131720 !important;
}

/* ── Plotly chart containers ── */
[data-testid="stPlotlyChart"] {
    background: transparent !important;
}

/* ── Input label spacing ── */
.stTextArea > label, .stTextInput > label {
    margin-bottom: 0.5rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0D0F14; }
::-webkit-scrollbar-thumb { background: #1E2430; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #2A3344; }

/* ── Footer ── */
.app-footer {
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.app-footer-brand {
    font-family: 'DM Serif Display', serif;
    font-size: 0.9rem;
    color: #374151;
}
.app-footer-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #374151;
    letter-spacing: 0.08em;
}

/* ── Model info chips in sidebar ── */
.model-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.2rem 0.6rem;
    background: rgba(90,140,220,0.1);
    border: 1px solid rgba(90,140,220,0.2);
    border-radius: 5px;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #5A8CDC;
    margin: 0.2rem 0;
}
.model-chip-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #5A8CDC;
}

/* ── Analysis result header ── */
.result-banner {
    background: linear-gradient(135deg, #131720 0%, #0D1220 100%);
    border: 1px solid rgba(90,140,220,0.2);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.result-banner-icon {
    font-size: 2rem;
    line-height: 1;
}
.result-banner-text h3 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.3rem !important;
    color: #E8E6E1 !important;
    margin: 0 0 0.25rem !important;
    font-weight: 400 !important;
}
.result-banner-text p {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #5A8CDC;
    margin: 0;
    letter-spacing: 0.05em;
}

/* ── Sidebar feature list ── */
.feature-list-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #6B7280;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.feature-list-item-dot {
    width: 4px; height: 4px;
    border-radius: 50%;
    background: #5A8CDC;
    flex-shrink: 0;
}

/* Hide streamlit default branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PLOTLY THEME
# ============================================================
PLOTLY_LAYOUT = dict(
    font=dict(family="DM Mono, monospace", color="#9CA3AF", size=11),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=40, b=10),
    title_font=dict(family="DM Sans, sans-serif", color="#E8E6E1", size=14),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.08)",
        tickfont=dict(family="DM Mono, monospace", size=10, color="#6B7280"),
        title_font=dict(family="DM Mono, monospace", size=10, color="#6B7280")
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.08)",
        tickfont=dict(family="DM Mono, monospace", size=10, color="#6B7280"),
        title_font=dict(family="DM Mono, monospace", size=10, color="#6B7280")
    )
)

PALETTE = ["#5A8CDC", "#A855F7", "#10B981", "#F59E0B", "#EF4444", "#06B6D4"]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data(ttl=60)
def check_api_health():
    try:
        response = requests.get(API_HEALTH_URL, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def categorize_features(features_dict):
    categories = defaultdict(dict)
    for key, value in features_dict.items():
        if '.' in key:
            category = key.split('.')[0]
        else:
            if key[:2].isupper() and len(key) > 2:
                category = "cohmetrix"
            else:
                category = "other"
        categories[category][key] = value
    return categories


def apply_plotly_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def plot_feature_bars(category_features, title, color=None, top_n=15):
    if not category_features:
        return None
    sorted_items = sorted(category_features.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    names = [item[0].split('.')[-1] if '.' in item[0] else item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    bar_color = color or PALETTE[0]
    colors = [bar_color if v >= 0 else "#EF4444" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.3f}" for v in values],
        textposition='outside',
        textfont=dict(size=9, color="#6B7280", family="DM Mono")
    ))
    fig.update_layout(
        title=title,
        height=max(350, len(names) * 26),
        showlegend=False,
        **PLOTLY_LAYOUT
    )
    return fig


def plot_readability_metrics(features):
    readability_features = {
        k.replace('formulas.', ''): v
        for k, v in features.items()
        if k.startswith('formulas.')
    }
    if not readability_features:
        return

    col1, col2 = st.columns([3, 2])

    with col1:
        fig1 = go.Figure(data=[go.Bar(
            x=list(readability_features.values()),
            y=list(readability_features.keys()),
            orientation='h',
            marker=dict(
                color=list(readability_features.values()),
                colorscale=[[0, "#EF4444"], [0.5, "#F59E0B"], [1, "#10B981"]],
                line=dict(width=0),
                showscale=True,
                colorbar=dict(
                    tickfont=dict(family="DM Mono", size=9, color="#6B7280"),
                    outlinewidth=0,
                    thickness=10
                )
            ),
            text=[f"{v:.2f}" for v in readability_features.values()],
            textposition='outside',
            textfont=dict(size=9, color="#6B7280", family="DM Mono")
        )])
        fig1.update_layout(title="Readability Formula Scores", height=380, **PLOTLY_LAYOUT)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        if 'flesch_reading_ease' in readability_features:
            fre = readability_features['flesch_reading_ease']
            if fre >= 80: grade, gauge_color = "6th Grade", "#10B981"
            elif fre >= 60: grade, gauge_color = "8th–9th Grade", "#F59E0B"
            elif fre >= 30: grade, gauge_color = "College Level", "#EF4444"
            else: grade, gauge_color = "Graduate Level", "#7C3AED"

            fig2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fre,
                domain={'x': [0, 1], 'y': [0, 1]},
                title=dict(
                    text=f"Flesch Reading Ease<br><span style='font-size:0.75em;color:{gauge_color};font-family:DM Mono'>{grade}</span>",
                    font=dict(family="DM Sans", color="#E8E6E1", size=13)
                ),
                number=dict(font=dict(family="DM Serif Display", color="#E8E6E1", size=42)),
                gauge=dict(
                    axis=dict(
                        range=[0, 100],
                        tickfont=dict(family="DM Mono", size=9, color="#4B5563"),
                        tickcolor="#4B5563",
                        tickwidth=1
                    ),
                    bar=dict(color=gauge_color, thickness=0.25),
                    bgcolor="rgba(255,255,255,0.03)",
                    borderwidth=1,
                    bordercolor="rgba(255,255,255,0.08)",
                    steps=[
                        {'range': [0, 30],  'color': "rgba(239,68,68,0.1)"},
                        {'range': [30, 60], 'color': "rgba(245,158,11,0.1)"},
                        {'range': [60, 80], 'color': "rgba(16,185,129,0.07)"},
                        {'range': [80, 100],'color': "rgba(16,185,129,0.12)"},
                    ],
                    threshold=dict(
                        line=dict(color=gauge_color, width=2),
                        thickness=0.8, value=fre
                    )
                )
            ))
            fig2.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Mono"))
            st.plotly_chart(fig2, use_container_width=True)


def plot_entity_analysis(entity_features):
    if not entity_features:
        return

    col1, col2 = st.columns(2)

    with col1:
        pne = entity_features.get('ent.percent_named_entities', 0)
        pgn = entity_features.get('ent.percent_general_nouns', 0)
        other = max(0, 100 - pne - pgn)
        fig = go.Figure(go.Pie(
            labels=['Named Entities', 'General Nouns', 'Other'],
            values=[pne, pgn, other],
            hole=0.55,
            marker=dict(colors=["#5A8CDC", "#A855F7", "#1E2430"], line=dict(width=0)),
            textfont=dict(family="DM Mono", size=10, color="#9CA3AF"),
            hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>"
        ))
        fig.update_layout(
            title="Entity Composition",
            height=320,
            showlegend=True,
            legend=dict(
                font=dict(family="DM Mono", size=10, color="#6B7280"),
                bgcolor="rgba(0,0,0,0)"
            ),
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ['xaxis', 'yaxis']}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        metrics = {
            'Entity Density': entity_features.get('ent.entity_density', 0),
            'Named Ent.': entity_features.get('ent.named_entity_density', 0),
            'Unique Ent.': entity_features.get('ent.unique_entity_density', 0),
        }
        fig2 = go.Figure(data=[go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            marker=dict(color=PALETTE[:3], line=dict(width=0)),
            text=[f"{v:.3f}" for v in metrics.values()],
            textposition='outside',
            textfont=dict(size=10, color="#6B7280", family="DM Mono")
        )])
        fig2.update_layout(title="Density Metrics", height=320, **PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)


def plot_discourse_structure(coref_features, lex_features, const_features):
    col1, col2 = st.columns(2)

    with col1:
        if coref_features:
            metrics = {
                'Chains': coref_features.get('coref.num_chains', 0),
                'Refs/Chain': coref_features.get('coref.avg_refs_per_chain', 0),
                'Avg Span': coref_features.get('coref.avg_chain_span', 0),
                'Long Spans': coref_features.get('coref.long_span_chains', 0)
            }
            fig = go.Figure(go.Bar(
                x=list(metrics.values()), y=list(metrics.keys()),
                orientation='h',
                marker=dict(color="#5A8CDC", line=dict(width=0)),
                text=[f"{v:.2f}" for v in metrics.values()],
                textposition='outside',
                textfont=dict(size=10, color="#6B7280", family="DM Mono")
            ))
            fig.update_layout(title="Coreference Resolution", height=280, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if lex_features:
            metrics = {
                'Chains': lex_features.get('lex.num_lexical_chains', 0),
                'Avg Length': lex_features.get('lex.avg_chain_length', 0),
                'Avg Span': lex_features.get('lex.avg_chain_span', 0),
                'Long Spans': lex_features.get('lex.long_span_chains', 0)
            }
            fig = go.Figure(go.Bar(
                x=list(metrics.values()), y=list(metrics.keys()),
                orientation='h',
                marker=dict(color="#A855F7", line=dict(width=0)),
                text=[f"{v:.2f}" for v in metrics.values()],
                textposition='outside',
                textfont=dict(size=10, color="#6B7280", family="DM Mono")
            ))
            fig.update_layout(title="Lexical Chains", height=280, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    if const_features:
        metrics = {
            'VP Density': const_features.get('const.vp_density', 0),
            'NP Density': const_features.get('const.np_density', 0),
            'ADJP Density': const_features.get('const.adjp_density', 0),
            'ADVP Density': const_features.get('const.advp_density', 0),
            'SBAR Density': const_features.get('const.sbar_density', 0),
            'Tree Height': const_features.get('const.avg_tree_height', 0)
        }
        fig = go.Figure(data=[go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            marker=dict(
                color=PALETTE[:6],
                line=dict(width=0)
            ),
            text=[f"{v:.3f}" for v in metrics.values()],
            textposition='outside',
            textfont=dict(size=10, color="#6B7280", family="DM Mono")
        )])
        fig.update_layout(title="Constituency Parse Features", height=320, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)


def plot_pos_radar(pos_features):
    if not pos_features:
        return
    categories = ['Nouns', 'Prepositions', 'Adjectives', 'Adverbs', '1st Person', '3rd Person']
    values = [
        pos_features.get('pos.noun_density', 0),
        pos_features.get('pos.preposition_density', 0),
        pos_features.get('pos.adjective_density', 0),
        pos_features.get('pos.adverb_density', 0),
        pos_features.get('pos.first_person_pronoun_density', 0),
        pos_features.get('pos.third_person_pronoun_density', 0)
    ]
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(90,140,220,0.12)',
        line=dict(color="#5A8CDC", width=2),
        marker=dict(color="#5A8CDC", size=6),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(255,255,255,0.02)",
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.25 if max(values) > 0 else 1],
                tickfont=dict(family="DM Mono", size=9, color="#4B5563"),
                gridcolor="rgba(255,255,255,0.06)",
                linecolor="rgba(255,255,255,0.06)",
            ),
            angularaxis=dict(
                tickfont=dict(family="DM Mono", size=10, color="#9CA3AF"),
                gridcolor="rgba(255,255,255,0.06)",
                linecolor="rgba(255,255,255,0.06)",
            )
        ),
        showlegend=False,
        title=dict(text="POS Distribution / 100 words", font=dict(family="DM Sans", color="#E8E6E1", size=13)),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono"),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_entity_grid_heatmap(eg_features):
    if not eg_features:
        return
    roles = ['S', 'O', 'X', '-']
    matrix = np.zeros((4, 4))
    for i, r1 in enumerate(roles):
        for j, r2 in enumerate(roles):
            matrix[i, j] = eg_features.get(f"eg.trans_{r1}_{r2}", 0)

    fig = go.Figure(go.Heatmap(
        z=matrix, x=roles, y=roles,
        colorscale=[[0, "#0D0F14"], [0.5, "#1E3A5F"], [1, "#5A8CDC"]],
        text=[[f"{v:.3f}" for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont=dict(size=11, family="DM Mono", color="#E8E6E1"),
        showscale=True,
        colorbar=dict(
            tickfont=dict(family="DM Mono", size=9, color="#6B7280"),
            outlinewidth=0, thickness=10
        )
    ))
    fig.update_layout(
        title="Entity Grid Role Transitions",
        height=360,
        xaxis=dict(
            title="To Role", side="bottom",
            tickfont=dict(family="DM Mono", size=11, color="#9CA3AF"),
            title_font=dict(family="DM Mono", size=10, color="#6B7280")
        ),
        yaxis=dict(
            title="From Role",
            tickfont=dict(family="DM Mono", size=11, color="#9CA3AF"),
            title_font=dict(family="DM Mono", size=10, color="#6B7280")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono", color="#9CA3AF"),
        margin=dict(l=60, r=20, t=50, b=60),
        title_font=dict(family="DM Sans", color="#E8E6E1", size=13)
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# MAIN APP
# ============================================================

def main():
    health_status = check_api_health()

    # ── Page header ──
    st.markdown("""
    <div class="page-header">
        <div class="page-header-glyph">◈</div>
        <div class="page-header-text">
            <h1>Discourse Feature Analyzer</h1>
            <p>200+ LINGUISTIC FEATURES · COREFERENCE · ENTITY GRID · READABILITY · COH-METRIX</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown('<div class="section-label">System Status</div>', unsafe_allow_html=True)
        if health_status:
            st.markdown("""<div class="status-pill status-online"><span class="status-dot"></span>BACKEND ONLINE</div>""", unsafe_allow_html=True)
            if health_status.get('models_loaded'):
                for model, loaded in health_status['models_loaded'].items():
                    if loaded:
                        st.markdown(f'<div class="model-chip"><span class="model-chip-dot"></span>{model}</div>', unsafe_allow_html=True)
            if health_status.get('cohmetrix_available'):
                st.markdown('<div class="model-chip"><span class="model-chip-dot"></span>coh-metrix</div>', unsafe_allow_html=True)
        else:
            st.markdown("""<div class="status-pill status-offline"><span class="status-dot"></span>BACKEND OFFLINE</div>""", unsafe_allow_html=True)
            st.markdown("""
            <div style="margin-top:0.75rem; padding:0.75rem; background:#131720; border-radius:8px; border:1px solid rgba(255,255,255,0.06);">
            <code style="font-family:DM Mono,monospace;font-size:0.72rem;color:#5A8CDC;">uvicorn app:app --reload</code>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Analysis Options</div>', unsafe_allow_html=True)
        include_lftk = st.checkbox("LFTK Features", value=True, help="Language Feature Toolkit")
        include_cohmetrix = st.checkbox("Coh-Metrix Indices", value=False, help="Requires CLI setup")

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Feature Modules</div>', unsafe_allow_html=True)
        modules = [
            ("Coreference Resolution", "Entity reference chains"),
            ("Entity Grid", "Syntactic role transitions"),
            ("Lexical Chains", "Semantic word relations"),
            ("Constituency Parsing", "Syntactic tree analysis"),
            ("Named Entity Recognition", "Entity classification"),
            ("Part-of-Speech", "Morphosyntactic tagging"),
            ("Readability Formulas", "Text complexity indices"),
        ]
        if include_lftk:
            modules.append(("LFTK", "Extended linguistic metrics"))
        if include_cohmetrix:
            modules.append(("Coh-Metrix", "Cohesion & coherence"))

        for name, desc in modules:
            st.markdown(f"""
            <div class="feature-list-item">
                <span class="feature-list-item-dot"></span>
                <span title="{desc}">{name}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Input section ──
    st.markdown('<div class="section-label">Input Text</div>', unsafe_allow_html=True)

    col_input, col_stats = st.columns([3, 1])

    with col_input:
        text_input = st.text_area(
            "Text to Analyze",
            height=220,
            placeholder="Paste your text here — narratives, essays, articles, or any natural language content will work best with at least 50 words...",
            label_visibility="collapsed"
        )

    with col_stats:
        if text_input:
            words = text_input.split()
            sents = max(1, text_input.count('.') + text_input.count('!') + text_input.count('?'))
            avg_wlen = len(text_input.replace(' ', '')) / len(words) if words else 0
            avg_slen = len(words) / sents if sents else 0
            st.markdown(f"""
            <div style="display:flex;flex-direction:column;gap:0.75rem;height:100%;justify-content:center;">
                <div style="background:#131720;border:1px solid rgba(255,255,255,0.07);border-radius:9px;padding:0.85rem 1rem;">
                    <div class="metric-label">Words</div>
                    <div class="metric-value" style="font-size:1.4rem;">{len(words):,}</div>
                </div>
                <div style="background:#131720;border:1px solid rgba(255,255,255,0.07);border-radius:9px;padding:0.85rem 1rem;">
                    <div class="metric-label">Sentences</div>
                    <div class="metric-value" style="font-size:1.4rem;">{sents:,}</div>
                </div>
                <div style="background:#131720;border:1px solid rgba(255,255,255,0.07);border-radius:9px;padding:0.85rem 1rem;">
                    <div class="metric-label">Avg Sent Length</div>
                    <div class="metric-value" style="font-size:1.4rem;">{avg_slen:.1f}</div>
                    <div class="metric-unit">words/sentence</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="height:100%;display:flex;align-items:center;justify-content:center;">
                <span style="font-family:DM Mono,monospace;font-size:0.72rem;color:#374151;letter-spacing:0.06em;">STATS APPEAR HERE</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Analyze button ──
    st.markdown('<br>', unsafe_allow_html=True)
    _, col_btn, _ = st.columns([2, 1, 2])
    with col_btn:
        analyze_button = st.button(
            "◈  Run Analysis",
            use_container_width=True,
            type="primary",
            disabled=not health_status or not text_input
        )

    # ── Results ──
    if analyze_button and text_input and health_status:
        if len(text_input.split()) < 5:
            st.warning("Text is very short. For meaningful analysis, enter at least 50 words.")

        with st.spinner("Extracting features — this typically takes 20–40 seconds…"):
            t0 = time.time()
            try:
                response = requests.post(
                    API_URL,
                    json={"text": text_input, "include_lftk": include_lftk, "include_cohmetrix": include_cohmetrix},
                    timeout=120
                )

                if response.status_code == 200:
                    result = response.json()
                    features = result.get('features', {})
                    categories = result.get('categories', {})
                    feature_count = result.get('feature_count', 0)
                    elapsed = time.time() - t0

                    # Result banner
                    st.markdown(f"""
                    <div class="result-banner">
                        <div class="result-banner-icon">◈</div>
                        <div class="result-banner-text">
                            <h3>Analysis Complete</h3>
                            <p>{feature_count} features extracted in {elapsed:.1f}s across {len(categories)} modules</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Category chips
                    cat_colors = ["#5A8CDC", "#A855F7", "#10B981", "#F59E0B", "#EF4444", "#06B6D4", "#F472B6", "#84CC16"]
                    cat_items = list(categories.items())
                    n = len(cat_items)
                    cols = st.columns(min(n, 4))
                    for i, (cat, count) in enumerate(cat_items[:8]):
                        with cols[i % min(n, 4)]:
                            accent = cat_colors[i % len(cat_colors)]
                            st.markdown(f"""
                            <div class="cat-chip" style="--accent:{accent};">
                                <div class="cat-chip-name">{cat}</div>
                                <div class="cat-chip-count">{count}</div>
                                <div class="cat-chip-unit">features</div>
                            </div>
                            """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Categorized features
                    categorized = categorize_features(features)

                    tabs = st.tabs([
                        "◎ Readability",
                        "◉ Entity Analysis",
                        "◈ Discourse Structure",
                        "◇ POS & Grammar",
                        "≡ All Features"
                    ])

                    # Tab 0: Readability
                    with tabs[0]:
                        st.markdown('<div class="section-label">Readability & Text Complexity</div>', unsafe_allow_html=True)
                        plot_readability_metrics(features)

                        agg = categorized.get('aggregates', {})
                        if agg:
                            st.markdown('<div class="section-label" style="margin-top:1.5rem;">Aggregate Statistics</div>', unsafe_allow_html=True)
                            c1, c2 = st.columns(2)
                            with c1:
                                agg_df = pd.DataFrame([
                                    {"Metric": k.replace('aggregates.', '').replace('_', ' ').title(), "Value": round(v, 4) if isinstance(v, float) else v}
                                    for k, v in agg.items()
                                ])
                                st.dataframe(agg_df, use_container_width=True, hide_index=True)
                            with c2:
                                numeric_agg = {k.replace('aggregates.', ''): v for k, v in agg.items() if isinstance(v, (int, float)) and v > 0}
                                if numeric_agg:
                                    fig = go.Figure(go.Scatterpolar(
                                        r=list(numeric_agg.values()),
                                        theta=list(numeric_agg.keys()),
                                        fill='toself',
                                        fillcolor='rgba(90,140,220,0.1)',
                                        line=dict(color="#5A8CDC", width=1.5)
                                    ))
                                    fig.update_layout(
                                        polar=dict(
                                            bgcolor="rgba(0,0,0,0)",
                                            radialaxis=dict(tickfont=dict(family="DM Mono", size=8, color="#4B5563"), gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.05)"),
                                            angularaxis=dict(tickfont=dict(family="DM Mono", size=9, color="#9CA3AF"), gridcolor="rgba(255,255,255,0.05)")
                                        ),
                                        showlegend=False,
                                        height=320,
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        margin=dict(l=20, r=20, t=20, b=20)
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                    # Tab 1: Entity
                    with tabs[1]:
                        st.markdown('<div class="section-label">Named Entity Analysis</div>', unsafe_allow_html=True)
                        ent = categorized.get('ent', {})
                        eg = categorized.get('eg', {})

                        c1, c2 = st.columns(2)
                        with c1:
                            if ent:
                                ent_df = pd.DataFrame([
                                    {"Feature": k.replace('ent.', '').replace('_', ' ').title(), "Value": round(v, 4)}
                                    for k, v in ent.items()
                                ])
                                st.dataframe(ent_df, use_container_width=True, hide_index=True)
                        with c2:
                            if eg:
                                plot_entity_grid_heatmap(eg)

                        if ent:
                            plot_entity_analysis(ent)

                    # Tab 2: Discourse
                    with tabs[2]:
                        st.markdown('<div class="section-label">Discourse Coherence Structure</div>', unsafe_allow_html=True)
                        plot_discourse_structure(
                            categorized.get('coref', {}),
                            categorized.get('lex', {}),
                            categorized.get('const', {})
                        )
                        eg = categorized.get('eg', {})
                        if eg:
                            with st.expander("Entity Grid — Detailed Metrics"):
                                eg_df = pd.DataFrame([
                                    {"Feature": k.replace('eg.', ''), "Value": round(v, 5)}
                                    for k, v in eg.items() if not k.startswith('eg.trans')
                                ])
                                st.dataframe(eg_df, use_container_width=True, hide_index=True)

                    # Tab 3: POS
                    with tabs[3]:
                        st.markdown('<div class="section-label">Part-of-Speech & Grammatical Features</div>', unsafe_allow_html=True)
                        pos = categorized.get('pos', {})
                        c1, c2 = st.columns(2)
                        with c1:
                            if pos:
                                pos_df = pd.DataFrame([
                                    {"Feature": k.replace('pos.', '').replace('_', ' ').title(), "Value": round(v, 4)}
                                    for k, v in pos.items()
                                ])
                                st.dataframe(pos_df, use_container_width=True, hide_index=True)
                        with c2:
                            plot_pos_radar(pos)

                        lftk = categorized.get('lftk', {})
                        if lftk:
                            with st.expander("LFTK — Language Feature Toolkit"):
                                lftk_df = pd.DataFrame([
                                    {"Feature": k.replace('lftk.', ''), "Value": round(v, 5)}
                                    for k, v in lftk.items()
                                ])
                                st.dataframe(lftk_df, use_container_width=True, hide_index=True)

                    # Tab 4: All features
                    with tabs[4]:
                        st.markdown('<div class="section-label">Complete Feature Set</div>', unsafe_allow_html=True)
                        all_df = pd.DataFrame([
                            {
                                "Category": k.split('.')[0] if '.' in k else 'cohmetrix',
                                "Feature": k,
                                "Value": round(v, 6) if isinstance(v, (int, float)) else str(v)
                            }
                            for k, v in features.items()
                        ])

                        fc1, fc2 = st.columns([3, 1])
                        with fc1:
                            search = st.text_input("Search features", "", placeholder="feature name or category…", label_visibility="collapsed")
                        with fc2:
                            cat_opts = ["All"] + sorted(all_df['Category'].unique().tolist())
                            cat_filter = st.selectbox("Category", cat_opts, label_visibility="collapsed")

                        filtered = all_df.copy()
                        if search:
                            filtered = filtered[
                                filtered['Feature'].str.contains(search, case=False) |
                                filtered['Category'].str.contains(search, case=False)
                            ]
                        if cat_filter != "All":
                            filtered = filtered[filtered['Category'] == cat_filter]

                        st.dataframe(filtered, use_container_width=True, hide_index=True, height=520)

                        csv = filtered.to_csv(index=False)
                        st.download_button(
                            "↓ Export CSV",
                            data=csv,
                            file_name=f"discourse_features_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

                    # Coh-Metrix
                    coh = categorized.get('cohmetrix', {})
                    if coh and include_cohmetrix:
                        st.markdown('<br><div class="section-label">Coh-Metrix Analysis</div>', unsafe_allow_html=True)
                        coh_cols = st.columns(3)
                        groups = {
                            'Descriptive': ['DESWC', 'DESSC', 'DESPL', 'DESSL'],
                            'Readability': ['RDFRE', 'RDFKGL', 'RDSMOG'],
                            'Lex. Diversity': ['LDTTRc', 'LDMTLD', 'LDVOCD'],
                        }
                        for idx, (gname, prefixes) in enumerate(groups.items()):
                            with coh_cols[idx]:
                                st.markdown(f'<div class="metric-label">{gname}</div>', unsafe_allow_html=True)
                                gf = {k: v for k, v in coh.items() if any(k.startswith(p) for p in prefixes)}
                                for k, v in gf.items():
                                    st.metric(k.replace('cm.', ''), f"{v:.3f}")

                else:
                    st.error(f"API Error {response.status_code}: {response.text[:200]}")

            except requests.exceptions.Timeout:
                st.error("Request timed out. The text may be too long or the backend is under load.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Ensure the FastAPI server is running on port 8000.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    elif analyze_button and not health_status:
        st.error("Backend is offline. Start the FastAPI server before running analysis.")
    elif analyze_button and not text_input:
        st.warning("Please enter some text to analyze.")

    # ── Footer ──
    st.markdown("""
    <div class="app-footer">
        <span class="app-footer-brand">◈ Discourse Analyzer</span>
        <span class="app-footer-meta">200+ LINGUISTIC FEATURES · BUILT ON SPACY, NLTK & COH-METRIX</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()