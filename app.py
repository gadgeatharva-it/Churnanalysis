"""Streamlit dashboard for ChurnPredict AI."""

from __future__ import annotations

import html
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.append(str(SRC_PATH))

from data_preprocessing import (  # noqa: E402
    DATA_PATH,
    calculate_kpis,
    churn_rate_by,
    clean_dataset,
    dataset_quality_report,
    generate_business_recommendations,
    load_dataset,
)


MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
METRICS_PATH = PROJECT_ROOT / "models" / "model_metrics.json"

PAGE_OPTIONS = [
    "Home",
    "Data Overview",
    "Customer Analytics",
    "Model Performance",
    "Feature Importance",
    "Churn Prediction",
]

# ── New Violet / Teal palette ────────────────────────────────────────────────
BRAND_VIOLET = "#7c3aed"
BRAND_TEAL = "#14b8a6"
BRAND_AMBER = "#f59e0b"
BRAND_ROSE = "#f43f5e"
INK = "#0f172a"
MUTED = "#64748b"
CARD = "#ffffff"
BORDER = "#e0d4f5"
PLOT_TEMPLATE = "plotly_white"
PLOT_COLORS = [BRAND_VIOLET, BRAND_TEAL, BRAND_AMBER, BRAND_ROSE, "#6366f1", "#0ea5e9"]


st.set_page_config(
    page_title="ChurnPredict AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS — deep violet / teal palette + animations
# ═══════════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
<style>
    /* ── Animations ─────────────────────────────────────── */
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(22px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes shimmer {
        0%   { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    @keyframes pulse {
        0%, 100% { opacity: 0.85; }
        50%      { opacity: 1; }
    }

    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.92); }
        to   { opacity: 1; transform: scale(1); }
    }

    @keyframes markerSlide {
        from { opacity: 0; transform: translate(-50%, -50%) scale(0); }
        to   { opacity: 1; transform: translate(-50%, -50%) scale(1); }
    }

    /* ── Root tokens ────────────────────────────────────── */
    :root {
        --bg: #f5f3ff;
        --panel: #ffffff;
        --ink: #0f172a;
        --muted: #64748b;
        --border: #e0d4f5;
        --violet: #7c3aed;
        --teal: #14b8a6;
        --amber: #f59e0b;
        --green: #10b981;
        --red: #ef4444;
        --rose: #f43f5e;
    }

    /* ── App background ──────────────────────────────── */
    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, rgba(124, 58, 237, 0.10), transparent 42rem),
            radial-gradient(ellipse at 90% 80%, rgba(20, 184, 166, 0.07), transparent 38rem),
            linear-gradient(172deg, #f5f3ff 0%, #ecfdf5 50%, #f0f9ff 100%);
        color: var(--ink);
    }

    header[data-testid="stHeader"] {
        background: rgba(245, 243, 255, 0.88);
        border-bottom: 1px solid rgba(224, 212, 245, 0.6);
        backdrop-filter: blur(14px);
    }

    [data-testid="collapsedControl"] {
        display: flex;
        visibility: visible;
        opacity: 1;
        z-index: 999999;
    }

    .block-container {
        max-width: 1520px;
        padding: 4.1rem 2.1rem 2.4rem;
    }

    /* ── Typography ────────────────────────────────────── */
    h1, h2, h3, p, label, span { letter-spacing: 0; }

    h1 {
        font-size: clamp(2rem, 3.1vw, 3rem);
        line-height: 1.08;
        margin-bottom: 0.35rem;
    }

    h2, h3 { color: var(--ink); }

    /* ── Sidebar — deep violet-black ─────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0a1e 0%, #160e2e 100%);
        border-right: 1px solid rgba(124, 58, 237, 0.18);
        box-shadow: 16px 0 44px rgba(15, 10, 30, 0.28);
    }

    [data-testid="stSidebar"] * {
        color: #ddd6fe;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(124, 58, 237, 0.10);
        border: 1px solid rgba(124, 58, 237, 0.18);
        border-radius: 10px;
        padding: 0.5rem 0.7rem;
        margin-bottom: 0.35rem;
        transition: all 220ms cubic-bezier(0.4, 0, 0.2, 1);
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(124, 58, 237, 0.30);
        border-color: rgba(167, 139, 250, 0.5);
        transform: translateX(4px);
        box-shadow: 0 0 16px rgba(124, 58, 237, 0.15);
    }

    /* ── Cards & charts ────────────────────────────────── */
    [data-testid="stMetric"],
    [data-testid="stDataFrame"],
    div[data-testid="stPlotlyChart"] {
        border-radius: 12px;
        animation: fadeSlideUp 0.5s ease-out both;
    }

    div[data-testid="stPlotlyChart"] {
        background: var(--panel);
        border: 1px solid var(--border);
        padding: 0.7rem;
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.06);
    }

    /* ── Hero section ────────────────────────────────── */
    .hero {
        background:
            linear-gradient(135deg, rgba(15, 10, 30, 0.95), rgba(88, 28, 195, 0.88)),
            linear-gradient(45deg, rgba(20, 184, 166, 0.22), rgba(245, 158, 11, 0.14));
        background-size: 200% 200%;
        animation: gradientShift 8s ease infinite;
        border: 1px solid rgba(167, 139, 250, 0.2);
        border-radius: 16px;
        color: #ffffff;
        padding: 1.6rem 1.8rem 1.5rem;
        margin: 0 0 1.2rem;
        box-shadow:
            0 20px 50px rgba(124, 58, 237, 0.18),
            0 0 0 1px rgba(167, 139, 250, 0.08) inset;
        position: relative;
        overflow: hidden;
    }

    .hero::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.06) 50%, transparent 100%);
        background-size: 200% 100%;
        animation: shimmer 4s ease-in-out infinite;
        pointer-events: none;
    }

    .hero h1 { color: #ffffff; margin: 0; position: relative; z-index: 1; }

    .hero p {
        color: #ddd6fe;
        max-width: 920px;
        margin: 0.5rem 0 0;
        font-size: 1.02rem;
        position: relative;
        z-index: 1;
    }

    .eyebrow {
        color: #a78bfa;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
        position: relative;
        z-index: 1;
    }

    /* ── KPI grid ────────────────────────────────────── */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
        gap: 0.85rem;
        margin: 1rem 0 1.2rem;
    }

    .kpi-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        min-height: 108px;
        box-shadow: 0 6px 24px rgba(124, 58, 237, 0.06);
        position: relative;
        overflow: hidden;
        transition: all 280ms cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeSlideUp 0.55s ease-out both;
    }

    .kpi-card:hover {
        transform: translateY(-4px) scale(1.015);
        border-color: #a78bfa;
        box-shadow:
            0 16px 40px rgba(124, 58, 237, 0.12),
            0 0 0 1px rgba(167, 139, 250, 0.15);
    }

    .kpi-card::before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 4px;
        background: linear-gradient(180deg, var(--violet), var(--teal));
        animation: pulse 2.5s ease-in-out infinite;
    }

    .kpi-label {
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.55rem;
    }

    .kpi-value {
        color: var(--ink);
        font-size: clamp(1.5rem, 2vw, 2.1rem);
        font-weight: 850;
        line-height: 1.05;
    }

    .kpi-note {
        color: var(--muted);
        font-size: 0.76rem;
        margin-top: 0.45rem;
    }

    /* ── Panels ──────────────────────────────────────── */
    .panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        margin: 0.65rem 0 1rem;
        box-shadow: 0 6px 24px rgba(124, 58, 237, 0.06);
        animation: fadeSlideUp 0.5s ease-out both;
        transition: all 220ms ease;
    }

    .panel:hover {
        border-color: #c4b5fd;
        box-shadow: 0 12px 32px rgba(124, 58, 237, 0.10);
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid var(--border);
        border-radius: 14px;
        box-shadow: 0 8px 28px rgba(124, 58, 237, 0.05);
        padding: 0.4rem 0.4rem 0.15rem;
    }

    /* ── Inputs ──────────────────────────────────────── */
    [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {
        border-radius: 10px;
        border-color: #d4c4f0;
        min-height: 44px;
    }

    [data-testid="stSlider"] { padding-top: 0.1rem; }

    button[kind="primary"] {
        border-radius: 10px;
        min-height: 44px;
        font-weight: 800;
        background: linear-gradient(135deg, var(--violet), #6d28d9);
        transition: all 200ms ease;
    }

    button[kind="primary"]:hover {
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.3);
        transform: translateY(-1px);
    }

    /* ── Section headings ────────────────────────────── */
    .section-title {
        font-size: 1.08rem;
        font-weight: 850;
        color: var(--ink);
        margin-bottom: 0.25rem;
    }

    .section-copy {
        color: var(--muted);
        margin: 0 0 0.65rem;
        line-height: 1.55;
    }

    /* ── Insight box ─────────────────────────────────── */
    .insight-box {
        border: 1px solid #99f6e4;
        border-left: 4px solid var(--teal);
        background: linear-gradient(135deg, #f0fdfa, #ecfdf5);
        padding: 0.9rem 1.1rem;
        border-radius: 12px;
        margin: 0.5rem 0 1rem;
        color: #134e4a;
        font-weight: 600;
        animation: fadeSlideUp 0.4s ease-out both;
        transition: all 200ms ease;
    }

    .insight-box:hover {
        border-left-width: 6px;
        box-shadow: 0 4px 16px rgba(20, 184, 166, 0.10);
    }

    /* ── Pills ───────────────────────────────────────── */
    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.6rem;
    }

    .pill {
        background: linear-gradient(135deg, #ede9fe, #f5f3ff);
        border: 1px solid #c4b5fd;
        color: #5b21b6;
        border-radius: 999px;
        padding: 0.35rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 750;
        animation: fadeSlideUp 0.4s ease-out both;
        transition: all 200ms ease;
        cursor: default;
    }

    .pill:nth-child(1) { animation-delay: 0.05s; }
    .pill:nth-child(2) { animation-delay: 0.10s; }
    .pill:nth-child(3) { animation-delay: 0.15s; }
    .pill:nth-child(4) { animation-delay: 0.20s; }
    .pill:nth-child(5) { animation-delay: 0.25s; }
    .pill:nth-child(6) { animation-delay: 0.30s; }
    .pill:nth-child(7) { animation-delay: 0.35s; }
    .pill:nth-child(8) { animation-delay: 0.40s; }
    .pill:nth-child(9) { animation-delay: 0.45s; }
    .pill:nth-child(10) { animation-delay: 0.50s; }

    .pill:hover {
        background: linear-gradient(135deg, #ddd6fe, #ede9fe);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.12);
    }

    /* ── Tech pill variant ───────────────────────────── */
    .tech-pill {
        background: linear-gradient(135deg, #ccfbf1, #f0fdfa);
        border: 1px solid #5eead4;
        color: #0f766e;
        border-radius: 999px;
        padding: 0.38rem 0.8rem;
        font-size: 0.78rem;
        font-weight: 750;
        animation: fadeSlideUp 0.4s ease-out both;
        transition: all 200ms ease;
    }

    .tech-pill:hover {
        background: linear-gradient(135deg, #99f6e4, #ccfbf1);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(20, 184, 166, 0.15);
    }

    /* ── Risk badges ─────────────────────────────────── */
    .risk-low,
    .risk-medium,
    .risk-high {
        display: inline-block;
        padding: 0.65rem 0.95rem;
        border-radius: 10px;
        font-weight: 850;
        animation: scaleIn 0.3s ease-out both;
    }

    .risk-low {
        color: #065f46;
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        border: 1px solid #6ee7b7;
    }

    .risk-medium {
        color: #92400e;
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border: 1px solid #fbbf24;
    }

    .risk-high {
        color: #991b1b;
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border: 1px solid #f87171;
    }

    /* ── Probability meter ───────────────────────────── */
    .probability-card {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.25rem;
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.07);
        animation: fadeSlideUp 0.5s ease-out both;
    }

    .probability-title {
        color: var(--ink);
        font-size: 1.08rem;
        font-weight: 850;
        margin-bottom: 0.25rem;
    }

    .probability-value {
        color: var(--ink);
        font-size: clamp(2.4rem, 5vw, 4.4rem);
        font-weight: 900;
        line-height: 1;
        margin: 0.85rem 0 0.75rem;
        background: linear-gradient(135deg, var(--violet), var(--teal));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .probability-meta {
        color: var(--muted);
        font-size: 0.9rem;
        margin-bottom: 0.9rem;
    }

    .risk-meter {
        position: relative;
        height: 18px;
        border-radius: 999px;
        background: linear-gradient(90deg, #10b981 0%, #10b981 35%, #f59e0b 35%, #f59e0b 65%, #ef4444 65%, #ef4444 100%);
        box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.08);
        margin: 1.1rem 0 0.75rem;
    }

    .risk-marker {
        position: absolute;
        top: 50%;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: var(--ink);
        border: 4px solid #ffffff;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.28);
        animation: markerSlide 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) both;
        animation-delay: 0.3s;
    }

    .meter-labels {
        display: flex;
        justify-content: space-between;
        color: var(--muted);
        font-size: 0.8rem;
        font-weight: 700;
    }

    .risk-badge-row {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin-top: 1rem;
    }

    /* ── Methodology panel ───────────────────────────── */
    .method-panel {
        background: linear-gradient(135deg, #faf5ff 0%, #f5f3ff 100%);
        border: 1px solid #ddd6fe;
        border-radius: 14px;
        padding: 1.2rem 1.3rem;
        margin: 0.8rem 0 1rem;
        animation: fadeSlideUp 0.5s ease-out both;
    }

    .method-panel h4 {
        color: var(--violet);
        margin: 0 0 0.5rem;
        font-size: 1rem;
    }

    .method-panel p {
        color: var(--muted);
        margin: 0;
        line-height: 1.6;
        font-size: 0.92rem;
    }

    /* ── Stat highlight ──────────────────────────────── */
    .stat-highlight {
        background: linear-gradient(135deg, #ede9fe, #f5f3ff);
        border: 1px solid #c4b5fd;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        text-align: center;
        animation: fadeSlideUp 0.5s ease-out both;
        transition: all 220ms ease;
    }

    .stat-highlight:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(124, 58, 237, 0.12);
    }

    .stat-highlight .stat-number {
        font-size: 2rem;
        font-weight: 900;
        background: linear-gradient(135deg, var(--violet), var(--teal));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
    }

    .stat-highlight .stat-desc {
        color: var(--muted);
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }

    /* ── Retention strategy cards ─────────────────────── */
    .strategy-card {
        background: linear-gradient(135deg, #f0fdfa, #ecfdf5);
        border: 1px solid #99f6e4;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin: 0.4rem 0;
        animation: fadeSlideUp 0.4s ease-out both;
        transition: all 200ms ease;
    }

    .strategy-card:hover {
        border-color: #5eead4;
        box-shadow: 0 4px 16px rgba(20, 184, 166, 0.10);
        transform: translateX(4px);
    }

    .strategy-card strong { color: #0f766e; }
    .strategy-card p { color: #115e59; margin: 0.25rem 0 0; font-size: 0.9rem; }

    /* ── Responsive ──────────────────────────────────── */
    @media (max-width: 700px) {
        .block-container { padding: 4rem 1rem 1.5rem; }
        .kpi-grid { grid-template-columns: 1fr; }
    }
</style>
"""


@st.cache_data
def get_data(_data_version: float) -> pd.DataFrame:
    """Load and clean data once per Streamlit session."""
    return clean_dataset(load_dataset(DATA_PATH))


@st.cache_resource
def load_model_artifact() -> dict[str, object] | None:
    """Load the trained model artifact if available."""
    if not MODEL_PATH.exists():
        return None
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


@st.cache_data
def load_metrics(_metrics_version: float | None) -> dict[str, object] | None:
    """Load saved training metrics if available."""
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_resource
def retrain_model_artifact() -> dict[str, object] | None:
    """Rebuild the model when a deployed pickle is incompatible with the runtime."""
    try:
        from train_model import train_and_select_best_model  # noqa: WPS433

        pipeline, metrics = train_and_select_best_model()
        return {
            "pipeline": pipeline,
            "feature_columns": list(pipeline.feature_names_in_),
            "best_model_name": metrics.get("best_model_name", "Retrained model"),
            "feature_importance": metrics.get("feature_importance", []),
        }
    except Exception as error:
        st.error(f"Could not rebuild the model automatically: {error}")
        return None


def score_customer_probability(artifact: dict[str, object], customer: pd.DataFrame) -> float | None:
    """Score a customer and recover from stale scikit-learn pickle artifacts."""
    pipeline = artifact["pipeline"]
    feature_columns = artifact.get("feature_columns")
    if feature_columns:
        customer = customer.reindex(columns=feature_columns)

    try:
        return float(pipeline.predict_proba(customer)[0, 1])
    except (AttributeError, TypeError, ValueError) as error:
        st.warning(
            "The saved model was built with a different runtime, so the app is rebuilding it once from the dataset."
        )
        st.caption(f"Recovered from model scoring error: {error}")
        rebuilt_artifact = retrain_model_artifact()
        if not rebuilt_artifact:
            return None
        rebuilt_pipeline = rebuilt_artifact["pipeline"]
        rebuilt_columns = rebuilt_artifact.get("feature_columns")
        if rebuilt_columns:
            customer = customer.reindex(columns=rebuilt_columns)
        return float(rebuilt_pipeline.predict_proba(customer)[0, 1])


# ═══════════════════════════════════════════════════════════════════════════════
#  Utility / rendering helpers
# ═══════════════════════════════════════════════════════════════════════════════

def format_value(label: str, value: float) -> str:
    """Format KPI values consistently."""
    if "Rate" in label:
        return f"{value:.2f}%"
    if "Charges" in label:
        return f"${value:,.2f}"
    if "Tenure" in label:
        return f"{value:.1f}"
    return f"{int(value):,}"


def render_header(title: str, subtitle: str, eyebrow: str = "Customer intelligence workspace") -> None:
    """Render a custom page header."""
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{html.escape(eyebrow)}</div>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_cards(kpis: dict[str, float], base_kpis: dict[str, float] | None = None) -> None:
    """Render custom KPI cards that work in light and dark browser themes."""
    notes = {
        "Total Customers": "filtered customer base",
        "Total Churned Customers": "customers marked churned",
        "Overall Churn Rate": "share of filtered base",
        "Average Monthly Charges": "mean subscription charge",
        "Average Customer Tenure": "months with company",
    }
    cards = []
    for i, (label, value) in enumerate(kpis.items()):
        note = notes.get(label, "")
        if base_kpis and label in base_kpis and base_kpis[label] != value:
            delta = value - base_kpis[label]
            sign = "+" if delta > 0 else ""
            suffix = " pts vs full data" if "Rate" in label else " vs full data"
            note = f"{sign}{delta:.2f}{suffix}"
        delay = f"animation-delay: {i * 0.08}s;"
        cards.append(
            f"<div class='kpi-card' style='{delay}'>"
            f"<div class='kpi-label'>{html.escape(label)}</div>"
            f"<div class='kpi-value'>{html.escape(format_value(label, value))}</div>"
            f"<div class='kpi-note'>{html.escape(note)}</div>"
            "</div>"
        )
    st.markdown(f"<div class='kpi-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def stat_cards(items: list[tuple[str, str, str]]) -> None:
    """Render custom statistic cards with explicit display values."""
    cards = []
    for i, (label, value, note) in enumerate(items):
        delay = f"animation-delay: {i * 0.08}s;"
        cards.append(
            f"<div class='kpi-card' style='{delay}'>"
            f"<div class='kpi-label'>{html.escape(label)}</div>"
            f"<div class='kpi-value'>{html.escape(value)}</div>"
            f"<div class='kpi-note'>{html.escape(note)}</div>"
            "</div>"
        )
    st.markdown(f"<div class='kpi-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def apply_plot_style(fig: go.Figure, height: int | None = None) -> go.Figure:
    """Apply the shared visual language to Plotly charts."""
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=INK, family="Inter, Segoe UI, Arial, sans-serif"),
        title=dict(font=dict(size=18, color=INK), x=0.02, xanchor="left"),
        margin=dict(t=64, l=30, r=24, b=42),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        coloraxis_colorbar=dict(outlinewidth=0),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#e5e7eb", zeroline=False)
    if height:
        fig.update_layout(height=height)
    return fig


def safe_churn_rate(df: pd.DataFrame) -> float:
    """Return churn rate for a non-empty dataframe."""
    if df.empty:
        return 0.0
    return float(df["Churn"].eq("Yes").mean() * 100)


def bar_churn_rate(df: pd.DataFrame, column: str, title: str) -> go.Figure:
    """Create a churn-rate bar chart for a category."""
    rates = churn_rate_by(df, column)
    fig = px.bar(
        rates,
        x=column,
        y="Churn Rate (%)",
        color="Churn Rate (%)",
        color_continuous_scale=[BRAND_TEAL, BRAND_VIOLET],
        text="Churn Rate (%)",
        title=title,
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_width=0)
    fig.update_layout(showlegend=False)
    return apply_plot_style(fig, height=430)


def insight(text: str) -> None:
    """Render a compact business-insight callout."""
    st.markdown(f"<div class='insight-box'>{html.escape(text)}</div>", unsafe_allow_html=True)


def panel(title: str, body: str) -> None:
    """Render a small content panel."""
    st.markdown(
        f"""
        <div class="panel">
            <div class="section-title">{html.escape(title)}</div>
            <p class="section-copy">{html.escape(body)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_filtered_data(df: pd.DataFrame) -> pd.DataFrame:
    """Build dynamic sidebar filters used by BI pages."""
    with st.sidebar:
        st.divider()
        st.subheader("Segment Filters")
        contract = st.multiselect("Contract", sorted(df["Contract"].unique()), default=sorted(df["Contract"].unique()))
        internet = st.multiselect(
            "Internet Service",
            sorted(df["InternetService"].unique()),
            default=sorted(df["InternetService"].unique()),
        )
        payment = st.multiselect(
            "Payment Method",
            sorted(df["PaymentMethod"].unique()),
            default=sorted(df["PaymentMethod"].unique()),
        )
        tenure_range = st.slider(
            "Tenure Range",
            min_value=int(df["tenure"].min()),
            max_value=int(df["tenure"].max()),
            value=(int(df["tenure"].min()), int(df["tenure"].max())),
        )
        charge_range = st.slider(
            "Monthly Charge Range",
            min_value=float(df["MonthlyCharges"].min()),
            max_value=float(df["MonthlyCharges"].max()),
            value=(float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max())),
        )

    filtered = df[
        df["Contract"].isin(contract)
        & df["InternetService"].isin(internet)
        & df["PaymentMethod"].isin(payment)
        & df["tenure"].between(tenure_range[0], tenure_range[1])
        & df["MonthlyCharges"].between(charge_range[0], charge_range[1])
    ]
    return filtered


def top_segment_sentence(df: pd.DataFrame, column: str) -> str:
    """Create a dynamic insight sentence for the highest churn segment."""
    if df.empty:
        return "No customers match the current filters."
    rates = churn_rate_by(df, column)
    row = rates.iloc[0]
    return (
        f"The highest churn segment for {column} is {row[column]} "
        f"with a churn rate of {row['Churn Rate (%)']:.1f}%."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Home
# ═══════════════════════════════════════════════════════════════════════════════

def home_page(df: pd.DataFrame, artifact: dict[str, object] | None, metrics: dict[str, object] | None) -> None:
    render_header(
        "ChurnPredict AI",
        "A portfolio-ready machine learning and business analytics system for understanding and predicting customer churn risk using advanced classification models.",
    )
    metric_cards(calculate_kpis(df))

    # ── Project Highlights ──────────────────────────────────────────────────
    st.subheader("Project Highlights")
    n_models = len(metrics["model_comparison"]) if metrics else 3
    n_features = len(metrics.get("feature_importance", [])) if metrics else "N/A"
    best_model = artifact["best_model_name"] if artifact else "—"
    best_auc = ""
    if metrics and best_model in metrics.get("model_comparison", {}):
        best_auc = f"{metrics['model_comparison'][best_model]['ROC-AUC Score']:.3f}"
    else:
        best_auc = "—"

    highlights_html = f"""
    <div class="kpi-grid">
        <div class="stat-highlight" style="animation-delay:0s;">
            <div class="stat-number">{n_models}</div>
            <div class="stat-desc">ML Models Trained & Compared</div>
        </div>
        <div class="stat-highlight" style="animation-delay:0.08s;">
            <div class="stat-number">{n_features}</div>
            <div class="stat-desc">Engineered Features Analyzed</div>
        </div>
        <div class="stat-highlight" style="animation-delay:0.16s;">
            <div class="stat-number">{df.shape[0]:,}</div>
            <div class="stat-desc">Customer Records Processed</div>
        </div>
        <div class="stat-highlight" style="animation-delay:0.24s;">
            <div class="stat-number">{html.escape(best_auc)}</div>
            <div class="stat-desc">Best ROC-AUC Score</div>
        </div>
    </div>
    """
    st.markdown(highlights_html, unsafe_allow_html=True)

    # ── About panels ────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        panel(
            "What this project demonstrates",
            "A complete, end-to-end data science workflow: data collection and cleaning, exploratory data analysis, feature engineering, model training and hyperparameter comparison, evaluation with multiple metrics, live prediction scoring, and decision-ready business analytics — all wrapped in a modern interactive dashboard.",
        )
    with c2:
        panel(
            "Dataset",
            f"{df.shape[0]:,} telco customers across {df.shape[1]} fields with a baseline churn rate of {safe_churn_rate(df):.2f}%. The dataset combines 7,043 real IBM Telco records and 4,957 synthetic augmented records for a richer analytical experience.",
        )
    with c3:
        model_desc = f"{best_model} (ROC-AUC {best_auc})" if best_model != "—" else "Train model first"
        panel("Current Model", f"The best-performing classifier is {model_desc}. Models were evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC to select the most balanced performer.")

    # ── Methodology ─────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="method-panel">
            <h4>Methodology — CRISP-DM Framework</h4>
            <p>
                This project follows the <strong>Cross-Industry Standard Process for Data Mining (CRISP-DM)</strong> framework.
                Starting from <em>Business Understanding</em> (customer churn directly impacts recurring revenue),
                through <em>Data Understanding</em> (exploring 21 customer attributes),
                <em>Data Preparation</em> (handling missing values, encoding categoricals, scaling numerics),
                <em>Modeling</em> (training Logistic Regression, Random Forest, and Gradient Boosting classifiers),
                <em>Evaluation</em> (multi-metric comparison with ROC-AUC as the primary ranking criterion),
                to <em>Deployment</em> (this interactive Streamlit dashboard with live scoring).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Workflow ────────────────────────────────────────────────────────────
    st.subheader("End-to-End Workflow")
    st.markdown(
        """
        <div class="pill-row">
            <span class="pill">1. Data Collection</span>
            <span class="pill">2. Data Cleaning</span>
            <span class="pill">3. Exploratory Analysis</span>
            <span class="pill">4. Feature Engineering</span>
            <span class="pill">5. Model Training</span>
            <span class="pill">6. Evaluation & Selection</span>
            <span class="pill">7. Live Prediction</span>
            <span class="pill">8. Business Dashboard</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Technology Stack ────────────────────────────────────────────────────
    st.subheader("Technology Stack")
    st.markdown(
        """
        <div class="pill-row">
            <span class="tech-pill">Python</span>
            <span class="tech-pill">Pandas</span>
            <span class="tech-pill">NumPy</span>
            <span class="tech-pill">Scikit-learn</span>
            <span class="tech-pill">Plotly</span>
            <span class="tech-pill">Streamlit</span>
            <span class="tech-pill">Matplotlib</span>
            <span class="tech-pill">Seaborn</span>
            <span class="tech-pill">Jupyter</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Key Findings ────────────────────────────────────────────────────────
    st.subheader("Key Findings")
    findings = generate_business_recommendations(df)
    for finding in findings:
        insight(finding)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Data Overview
# ═══════════════════════════════════════════════════════════════════════════════

def data_overview_page(df: pd.DataFrame) -> None:
    render_header(
        "Data Overview",
        "Inspect the raw customer base, data quality, column types, missing values, and descriptive statistics. A thorough understanding of the data is the foundation of every reliable model.",
        "Data quality console",
    )
    report = dataset_quality_report(df)

    # ── Extended KPIs ───────────────────────────────────────────────────────
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())
    quality_pct = round((1 - missing_cells / total_cells) * 100, 2) if total_cells else 100.0
    n_numeric = len(df.select_dtypes(include=np.number).columns)
    n_categorical = len(df.select_dtypes(exclude=np.number).columns)

    metric_cards(
        {
            "Total Customers": report["shape"][0],
            "Total Churned Customers": int((df["Churn"] == "Yes").sum()),
            "Overall Churn Rate": safe_churn_rate(df),
            "Average Monthly Charges": float(df["MonthlyCharges"].mean()),
            "Average Customer Tenure": float(df["tenure"].mean()),
        }
    )

    # ── Data quality stats ──────────────────────────────────────────────────
    stat_cards([
        ("Data Quality Score", f"{quality_pct:.1f}%", f"{total_cells - missing_cells:,} of {total_cells:,} cells present"),
        ("Total Columns", str(df.shape[1]), f"{n_numeric} numeric, {n_categorical} categorical"),
        ("Duplicate Rows", str(report["duplicate_rows"]), "after deduplication"),
        ("Missing Cells", str(missing_cells), f"out of {total_cells:,} total cells"),
    ])

    # ── Data provenance ─────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="method-panel">
            <h4>Dataset Provenance</h4>
            <p>
                The project uses the <strong>IBM Telco Customer Churn</strong> dataset as the original source.
                7,043 authentic customer records from the IBM archived GitHub repository are combined with
                4,957 synthetically augmented records generated from realistic distributions and churn-risk rules.
                This produces a 12,000-record working dataset with 21 features covering demographics,
                account information, subscribed services, charges, and the binary churn target.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    preview_rows = st.slider("Preview rows", 5, 50, 15)
    st.dataframe(df.head(preview_rows), use_container_width=True, hide_index=True)

    # ── Data type distribution chart + Missing values chart ─────────────────
    c_chart1, c_chart2 = st.columns(2)

    with c_chart1:
        type_counts = pd.DataFrame({
            "Type": ["Numeric", "Categorical"],
            "Count": [n_numeric, n_categorical],
        })
        fig_types = px.pie(
            type_counts,
            names="Type",
            values="Count",
            hole=0.55,
            color="Type",
            color_discrete_map={"Numeric": BRAND_VIOLET, "Categorical": BRAND_TEAL},
            title="Column Data Type Distribution",
        )
        fig_types.update_traces(textposition="inside", textinfo="percent+label+value")
        st.plotly_chart(apply_plot_style(fig_types, height=400), use_container_width=True)

    with c_chart2:
        missing = df.isna().sum().reset_index()
        missing.columns = ["Column", "Missing"]
        missing = missing[missing["Missing"] > 0].sort_values("Missing", ascending=True)
        if missing.empty:
            st.markdown(
                "<div class='insight-box'>No missing values detected in any column — the dataset is fully complete after cleaning.</div>",
                unsafe_allow_html=True,
            )
        else:
            fig_missing = px.bar(
                missing,
                x="Missing",
                y="Column",
                orientation="h",
                title="Missing Values by Column",
                color="Missing",
                color_continuous_scale=[BRAND_TEAL, BRAND_AMBER],
            )
            fig_missing.update_traces(marker_line_width=0)
            st.plotly_chart(apply_plot_style(fig_missing, height=400), use_container_width=True)

    # ── Column info + descriptive stats ─────────────────────────────────────
    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        st.subheader("Column Information")
        column_info = pd.DataFrame(
            {
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Missing Values": df.isna().sum().values,
                "Unique Values": df.nunique().values,
                "Sample Value": [str(df[col].iloc[0]) if len(df) > 0 else "" for col in df.columns],
            }
        )
        st.dataframe(column_info, hide_index=True, use_container_width=True)
    with c2:
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe(include="all").T, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Customer Analytics
# ═══════════════════════════════════════════════════════════════════════════════

def customer_analytics_page(df: pd.DataFrame, base_df: pd.DataFrame) -> None:
    render_header(
        "Customer Analytics",
        "Explore churn behavior by segment. Sidebar filters update every KPI, chart, and recommendation on this page in real time. Drill into demographics, contracts, services, and billing to uncover actionable retention insights.",
        "Interactive BI dashboard",
    )

    if df.empty:
        st.warning("No customers match the selected filters. Adjust the sidebar filters to continue.")
        return

    metric_cards(calculate_kpis(df), calculate_kpis(base_df))

    # ── Churn distribution + Gender ─────────────────────────────────────────
    churn_counts = df["Churn"].value_counts().reset_index()
    churn_counts.columns = ["Churn", "Customers"]
    fig_churn = px.pie(
        churn_counts,
        names="Churn",
        values="Customers",
        hole=0.55,
        color="Churn",
        color_discrete_map={"No": BRAND_VIOLET, "Yes": BRAND_AMBER},
        title="Overall Customer Churn Distribution",
    )
    fig_churn.update_traces(textposition="inside", textinfo="percent+label")
    fig_churn = apply_plot_style(fig_churn, height=430)

    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_churn, use_container_width=True)
    c2.plotly_chart(bar_churn_rate(df, "gender", "Churn Rate by Gender"), use_container_width=True)
    insight(top_segment_sentence(df, "gender"))

    # ── Contract + Internet ─────────────────────────────────────────────────
    c3, c4 = st.columns(2)
    c3.plotly_chart(bar_churn_rate(df, "Contract", "Churn Rate by Contract Type"), use_container_width=True)
    c4.plotly_chart(bar_churn_rate(df, "InternetService", "Churn Rate by Internet Service"), use_container_width=True)
    insight(top_segment_sentence(df, "Contract"))

    # ── Payment + Tenure ────────────────────────────────────────────────────
    c5, c6 = st.columns(2)
    c5.plotly_chart(bar_churn_rate(df, "PaymentMethod", "Churn Rate by Payment Method"), use_container_width=True)
    tenure_fig = px.histogram(
        df,
        x="tenure",
        color="Churn",
        nbins=36,
        barmode="overlay",
        title="Distribution of Customer Tenure",
        color_discrete_map={"No": BRAND_VIOLET, "Yes": BRAND_AMBER},
    )
    c6.plotly_chart(apply_plot_style(tenure_fig, height=430), use_container_width=True)
    insight(top_segment_sentence(df, "PaymentMethod"))

    # ── Senior Citizen + Paperless Billing ──────────────────────────────────
    st.subheader("Demographic & Billing Deep-Dive")
    c7, c8 = st.columns(2)
    c7.plotly_chart(bar_churn_rate(df, "SeniorCitizen", "Churn Rate by Senior Citizen Status"), use_container_width=True)
    c8.plotly_chart(bar_churn_rate(df, "PaperlessBilling", "Churn Rate by Paperless Billing"), use_container_width=True)
    insight(top_segment_sentence(df, "SeniorCitizen"))
    insight(top_segment_sentence(df, "PaperlessBilling"))

    # ── Monthly Charges vs Churn + Correlation Heatmap ──────────────────────
    st.subheader("Financial & Correlation Analysis")
    c9, c10 = st.columns(2)
    charges_fig = px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        title="Monthly Charges vs Churn Status",
        color_discrete_map={"No": BRAND_VIOLET, "Yes": BRAND_AMBER},
    )
    c9.plotly_chart(apply_plot_style(charges_fig, height=430), use_container_width=True)

    numeric_corr = df[["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]].corr()
    heatmap = px.imshow(
        numeric_corr,
        text_auto=".2f",
        color_continuous_scale="Purples",
        title="Correlation Heatmap for Numerical Features",
    )
    c10.plotly_chart(apply_plot_style(heatmap, height=430), use_container_width=True)

    # ── Services Bundle Analysis ────────────────────────────────────────────
    st.subheader("Services Bundle Analysis")
    service_cols = ["PhoneService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]
    available_services = [c for c in service_cols if c in df.columns]
    if available_services:
        svc_df = df.copy()
        svc_df["NumServices"] = svc_df[available_services].apply(
            lambda row: sum(1 for v in row if v == "Yes"), axis=1
        )
        svc_rates = svc_df.groupby("NumServices")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index(name="Churn Rate (%)")
        svc_rates["NumServices"] = svc_rates["NumServices"].astype(str) + " services"
        fig_svc = px.bar(
            svc_rates,
            x="NumServices",
            y="Churn Rate (%)",
            color="Churn Rate (%)",
            color_continuous_scale=[BRAND_TEAL, BRAND_VIOLET],
            text="Churn Rate (%)",
            title="Churn Rate by Number of Services Subscribed",
        )
        fig_svc.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_width=0)
        fig_svc.update_layout(showlegend=False)
        st.plotly_chart(apply_plot_style(fig_svc, height=430), use_container_width=True)
        insight("Customers with fewer active services tend to churn at higher rates — bundling services can be an effective retention strategy.")

    # ── Charges distribution violin ─────────────────────────────────────────
    st.subheader("Charges Distribution")
    c11, c12 = st.columns(2)
    violin_fig = px.violin(
        df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        box=True,
        title="Monthly Charges Distribution (Violin Plot)",
        color_discrete_map={"No": BRAND_VIOLET, "Yes": BRAND_AMBER},
    )
    c11.plotly_chart(apply_plot_style(violin_fig, height=430), use_container_width=True)

    total_charges_fig = px.histogram(
        df,
        x="TotalCharges",
        color="Churn",
        nbins=40,
        barmode="overlay",
        title="Total Charges Distribution by Churn",
        color_discrete_map={"No": BRAND_VIOLET, "Yes": BRAND_AMBER},
    )
    c12.plotly_chart(apply_plot_style(total_charges_fig, height=430), use_container_width=True)

    # ── Recommendations ─────────────────────────────────────────────────────
    st.subheader("Dynamic Business Recommendations")
    for recommendation in generate_business_recommendations(df):
        insight(recommendation)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Model Performance
# ═══════════════════════════════════════════════════════════════════════════════

def model_performance_page(metrics: dict[str, object] | None) -> None:
    render_header(
        "Model Performance",
        "Compare trained classification models using business-relevant metrics, inspect confusion matrices, ROC curves, and understand the evaluation methodology behind model selection.",
        "ML evaluation studio",
    )
    if not metrics:
        st.warning("Model metrics are not available yet. Run `python src/train_model.py` first.")
        return

    comparison = pd.DataFrame(metrics["model_comparison"]).T
    summary_cols = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC Score"]
    best_model = metrics["best_model_name"]
    st.markdown(
        f"<div class='insight-box'>Best-performing model: <strong>{html.escape(best_model)}</strong> — selected based on ROC-AUC as the primary metric, balanced with F1 Score for class-imbalanced data.</div>",
        unsafe_allow_html=True,
    )

    # ── Performance summary cards ───────────────────────────────────────────
    best = metrics["model_comparison"][best_model]
    stat_cards([
        ("Best Model", best_model, "highest ROC-AUC score"),
        ("Accuracy", f"{best['Accuracy']:.3f}", "overall correct predictions"),
        ("Precision", f"{best['Precision']:.3f}", "quality of churn alerts"),
        ("Recall", f"{best['Recall']:.3f}", "share of churners found"),
        ("F1 Score", f"{best['F1 Score']:.3f}", "precision-recall balance"),
        ("ROC-AUC", f"{best['ROC-AUC Score']:.3f}", "ranking power across thresholds"),
    ])

    # ── Model comparison table ──────────────────────────────────────────────
    st.subheader("Model Comparison Leaderboard")
    ranking = comparison[summary_cols].sort_values("ROC-AUC Score", ascending=False)
    st.dataframe(ranking.style.format("{:.4f}"), use_container_width=True)

    # ── Radar chart comparison ──────────────────────────────────────────────
    st.subheader("Multi-Metric Radar Comparison")
    radar_fig = go.Figure()
    radar_colors = [BRAND_VIOLET, BRAND_TEAL, BRAND_AMBER, BRAND_ROSE]
    for i, (model_name, row) in enumerate(comparison.iterrows()):
        vals = [row[c] for c in summary_cols] + [row[summary_cols[0]]]
        cats = summary_cols + [summary_cols[0]]
        radar_fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=cats,
            fill="toself",
            name=model_name,
            line=dict(color=radar_colors[i % len(radar_colors)], width=2),
            opacity=0.7,
        ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor="#e5e7eb")),
        title="Model Performance Radar Chart",
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", color=INK),
        paper_bgcolor=CARD,
        height=500,
    )
    st.plotly_chart(radar_fig, use_container_width=True)

    # ── Detailed inspection ─────────────────────────────────────────────────
    st.subheader("Detailed Model Inspection")
    selected_model = st.selectbox("Inspect model", comparison.index.tolist(), index=list(comparison.index).index(best_model))
    selected = metrics["model_comparison"][selected_model]

    stat_cards(
        [
            ("Accuracy", f"{selected['Accuracy']:.3f}", "overall correct predictions"),
            ("Precision", f"{selected['Precision']:.3f}", "quality of churn alerts"),
            ("Recall", f"{selected['Recall']:.3f}", "share of churners found"),
            ("F1 Score", f"{selected['F1 Score']:.3f}", "precision and recall balance"),
            ("ROC-AUC", f"{selected['ROC-AUC Score']:.3f}", "ranking power across thresholds"),
        ]
    )

    c1, c2 = st.columns(2)
    confusion = selected["Confusion Matrix"]
    cm_fig = px.imshow(
        confusion,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Customers"),
        x=["Stay", "Churn"],
        y=["Stay", "Churn"],
        color_continuous_scale="Purples",
        title=f"Confusion Matrix — {selected_model}",
    )
    c1.plotly_chart(apply_plot_style(cm_fig, height=430), use_container_width=True)

    roc = selected["ROC Curve"]
    roc_fig = go.Figure()
    roc_fig.add_trace(
        go.Scatter(
            x=roc["fpr"],
            y=roc["tpr"],
            mode="lines",
            name=selected_model,
            line=dict(color=BRAND_VIOLET, width=4),
            fill="tozeroy",
            fillcolor="rgba(124, 58, 237, 0.08)",
        )
    )
    roc_fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random baseline",
            line=dict(color=MUTED, dash="dash"),
        )
    )
    roc_fig.update_layout(title=f"ROC Curve — {selected_model}", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    c2.plotly_chart(apply_plot_style(roc_fig, height=430), use_container_width=True)

    # ── Evaluation methodology ──────────────────────────────────────────────
    st.markdown(
        """
        <div class="method-panel">
            <h4>Evaluation Methodology</h4>
            <p>
                All models are trained on an 80/20 stratified train-test split to preserve the original class balance.
                <strong>ROC-AUC</strong> is the primary ranking metric because it measures discriminative ability across all
                classification thresholds — crucial when the cost of missing a churner (false negative) is high.
                <strong>F1 Score</strong> serves as the secondary metric, balancing precision and recall.
                The <strong>confusion matrix</strong> provides an absolute count of correct and incorrect predictions,
                while the <strong>ROC curve</strong> visualises the trade-off between true positive rate and false positive rate.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════

def feature_importance_page(df: pd.DataFrame, metrics: dict[str, object] | None) -> None:
    render_header(
        "Feature Importance",
        "Review the strongest churn drivers from the selected model, understand feature categories, and convert analytical signals into concrete retention actions.",
        "Retention strategy signals",
    )
    if not metrics or not metrics.get("feature_importance"):
        st.warning("Feature importance is not available yet. Run `python src/train_model.py` first.")
        return
    if df.empty:
        st.warning("No customers match the selected filters. Adjust the sidebar filters to continue.")
        return

    top_n = st.slider("Number of features", 5, 20, 10)
    importance = pd.DataFrame(metrics["feature_importance"]).head(top_n)

    # ── Main importance chart ───────────────────────────────────────────────
    fig = px.bar(
        importance.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Top {top_n} Churn Drivers",
        color="Importance",
        color_continuous_scale=[BRAND_TEAL, BRAND_VIOLET],
    )
    st.plotly_chart(apply_plot_style(fig, height=520), use_container_width=True)

    # ── Feature categories breakdown ────────────────────────────────────────
    st.subheader("Feature Categories")
    demographic_features = {"gender", "SeniorCitizen", "Partner", "Dependents"}
    service_features = {"PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"}
    billing_features = {"Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "tenure"}

    all_features = importance["Feature"].tolist()
    cat_counts = {
        "Demographic": sum(1 for f in all_features if any(d in f for d in demographic_features)),
        "Service": sum(1 for f in all_features if any(s in f for s in service_features)),
        "Billing & Account": sum(1 for f in all_features if any(b in f for b in billing_features)),
    }
    # Anything not matched goes to "Other"
    matched = cat_counts["Demographic"] + cat_counts["Service"] + cat_counts["Billing & Account"]
    if matched < len(all_features):
        cat_counts["Other"] = len(all_features) - matched

    cat_df = pd.DataFrame(list(cat_counts.items()), columns=["Category", "Count"])
    cat_df = cat_df[cat_df["Count"] > 0]
    fig_cat = px.pie(
        cat_df,
        names="Category",
        values="Count",
        hole=0.5,
        color="Category",
        color_discrete_map={
            "Demographic": BRAND_AMBER,
            "Service": BRAND_TEAL,
            "Billing & Account": BRAND_VIOLET,
            "Other": MUTED,
        },
        title=f"Feature Category Breakdown (Top {top_n})",
    )
    fig_cat.update_traces(textposition="inside", textinfo="percent+label")

    c_cat1, c_cat2 = st.columns(2)
    c_cat1.plotly_chart(apply_plot_style(fig_cat, height=400), use_container_width=True)

    # ── Cumulative importance ───────────────────────────────────────────────
    cumulative = importance.copy()
    cumulative = cumulative.sort_values("Importance", ascending=False).reset_index(drop=True)
    cumulative["Cumulative"] = cumulative["Importance"].cumsum() / cumulative["Importance"].sum() * 100
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=cumulative["Feature"],
        y=cumulative["Cumulative"],
        mode="lines+markers",
        name="Cumulative %",
        line=dict(color=BRAND_VIOLET, width=3),
        marker=dict(size=8, color=BRAND_TEAL),
        fill="tozeroy",
        fillcolor="rgba(124, 58, 237, 0.06)",
    ))
    fig_cum.update_layout(
        title="Cumulative Feature Importance",
        xaxis_title="Feature",
        yaxis_title="Cumulative Importance (%)",
    )
    c_cat2.plotly_chart(apply_plot_style(fig_cum, height=400), use_container_width=True)

    # ── Table + recommendations ─────────────────────────────────────────────
    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        st.subheader("Top Factors Detail")
        st.dataframe(importance, hide_index=True, use_container_width=True)
    with c2:
        st.subheader("Business Interpretation")
        for recommendation in generate_business_recommendations(df, importance["Feature"].tolist()):
            insight(recommendation)

    # ── Actionable insights panel ───────────────────────────────────────────
    st.subheader("Actionable Insights")
    action_map = {
        "Contract": "Incentivize annual or two-year contracts with discounts and loyalty rewards.",
        "tenure": "Launch early-life engagement campaigns within the first 12 months.",
        "MonthlyCharges": "Offer value bundles and competitive pricing for high-charge segments.",
        "TotalCharges": "Introduce milestone rewards for long-tenure, high-value customers.",
        "InternetService": "Improve fiber-optic service quality and support response times.",
        "OnlineSecurity": "Promote security add-on bundles as a retention hook.",
        "TechSupport": "Invest in proactive technical support to reduce frustration-driven churn.",
        "PaymentMethod": "Offer automatic payment setup bonuses to reduce electronic check churn.",
        "PaperlessBilling": "Ensure digital billing communications are clear and user-friendly.",
        "SeniorCitizen": "Create senior-friendly plans with simplified pricing and dedicated support.",
    }
    for feature in all_features[:6]:
        base_name = feature.split("_")[0] if "_" in feature else feature
        action = action_map.get(base_name, f"Investigate the impact of {feature} on churn and design targeted interventions.")
        st.markdown(
            f"""<div class="strategy-card"><strong>{html.escape(feature)}</strong><p>{html.escape(action)}</p></div>""",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Churn Prediction
# ═══════════════════════════════════════════════════════════════════════════════

def build_customer_input() -> pd.DataFrame:
    """Collect customer attributes for prediction."""
    with st.container(border=True):
        st.subheader("Customer Profile")
        st.caption("Adjust any value and the churn risk updates automatically below.")
        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure", min_value=0, max_value=72, value=12)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])

        with c2:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            internet_defaults = ["No internet service"] if internet_service == "No" else ["Yes", "No"]
            online_security = st.selectbox("Online Security", internet_defaults)
            online_backup = st.selectbox("Online Backup", internet_defaults)
            device_protection = st.selectbox("Device Protection", internet_defaults)
            tech_support = st.selectbox("Tech Support", internet_defaults)
            multiple_lines = (
                "No phone service"
                if phone_service == "No"
                else st.selectbox("Multiple Lines", ["No", "Yes"])
            )

        with c3:
            streaming_options = ["No internet service"] if internet_service == "No" else ["Yes", "No"]
            streaming_tv = st.selectbox("Streaming TV", streaming_options)
            streaming_movies = st.selectbox("Streaming Movies", streaming_options)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
            monthly_charges = st.slider("Monthly Charges", min_value=18.0, max_value=120.0, value=70.0, step=0.5)

    total_charges = float(tenure) * float(monthly_charges)

    return pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
        ]
    )


def prediction_explanation(row: pd.Series, probability: float) -> list[str]:
    """Create a simple, transparent explanation from entered customer values."""
    factors: list[str] = []
    if row["Contract"] == "Month-to-month":
        factors.append("month-to-month contract (highest churn segment)")
    if row["tenure"] <= 12:
        factors.append(f"short customer tenure ({int(row['tenure'])} months)")
    if row["MonthlyCharges"] >= 80:
        factors.append(f"high monthly charges (${row['MonthlyCharges']:.0f})")
    if row["TechSupport"] == "No":
        factors.append("no technical support subscription")
    if row["OnlineSecurity"] == "No":
        factors.append("no online security add-on")
    if row["PaymentMethod"] == "Electronic check":
        factors.append("electronic-check payment method (friction risk)")
    if row["InternetService"] == "Fiber optic":
        factors.append("fiber-optic internet (higher churn segment)")
    if row["PaperlessBilling"] == "Yes":
        factors.append("paperless billing enabled")

    if not factors:
        factors.append("a more stable customer profile based on the selected inputs")

    direction = "high" if probability >= 0.65 else "moderate" if probability >= 0.35 else "low"
    return [
        f"The model estimates a {direction} churn probability of {probability*100:.1f}% for this profile.",
        "Notable profile factors: " + ", ".join(factors[:5]) + ".",
    ]


def render_probability_meter(probability: float, risk_label: str, risk_class: str) -> None:
    """Render a lightweight churn probability meter."""
    value = probability * 100
    st.markdown(
        f"""
        <div class="probability-card">
            <div class="probability-title">Churn Probability</div>
            <div class="probability-meta">Live score from the trained model pipeline</div>
            <div class="probability-value">{value:.1f}%</div>
            <div class="risk-meter">
                <div class="risk-marker" style="left: {value:.1f}%;"></div>
            </div>
            <div class="meter-labels">
                <span>0%</span><span>Low</span><span>Medium</span><span>High</span><span>100%</span>
            </div>
            <div class="risk-badge-row">
                <span class="{html.escape(risk_class)}">{html.escape(risk_label)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_retention_strategies(risk_label: str, row: pd.Series) -> list[tuple[str, str]]:
    """Return retention strategies based on risk level and customer profile."""
    strategies = []
    if risk_label == "High Risk":
        strategies.append(("Immediate Outreach", "Contact the customer within 48 hours with a personalised retention offer before they make a switching decision."))
        strategies.append(("Contract Upgrade Incentive", "Offer a discounted annual or two-year contract to lock in commitment and reduce month-to-month volatility."))
        if row["TechSupport"] == "No":
            strategies.append(("Free Tech Support Trial", "Provide 3 months of complimentary tech support to increase service stickiness and satisfaction."))
        if row["MonthlyCharges"] >= 80:
            strategies.append(("Value Bundle", "Create a personalised value bundle reducing the effective monthly rate by 15-20% while maintaining service quality."))
    elif risk_label == "Medium Risk":
        strategies.append(("Loyalty Program", "Enrol the customer in a loyalty program with milestone rewards at 6, 12, and 24 months to increase retention."))
        strategies.append(("Service Enhancement", "Offer a complimentary service add-on (security, backup, or tech support) to deepen engagement."))
        if row["Contract"] == "Month-to-month":
            strategies.append(("Annual Contract Bonus", "Offer a one-time bonus or discount for switching from month-to-month to an annual contract."))
    else:
        strategies.append(("Maintain Engagement", "Continue positive engagement with regular check-ins and satisfaction surveys to sustain the healthy relationship."))
        strategies.append(("Upsell Opportunities", "This stable customer may be receptive to premium service tiers or add-ons that increase lifetime value."))
    return strategies[:4]


def prediction_page(artifact: dict[str, object] | None, df: pd.DataFrame | None = None) -> None:
    render_header(
        "Churn Prediction",
        "Score an individual customer profile using the same preprocessing pipeline from model training. Get instant churn probability, risk classification, explanation, and personalised retention strategies.",
        "Live risk scoring",
    )
    if not artifact:
        st.warning("The model has not been trained yet. Run `python src/train_model.py` before making predictions.")
        return

    customer = build_customer_input()
    probability = score_customer_probability(artifact, customer)
    if probability is None:
        return
    prediction = int(probability >= 0.5)

    if probability < 0.35:
        risk_label = "Low Risk"
        risk_class = "risk-low"
    elif probability < 0.65:
        risk_label = "Medium Risk"
        risk_class = "risk-medium"
    else:
        risk_label = "High Risk"
        risk_class = "risk-high"

    st.divider()
    c1, c2 = st.columns([0.9, 1.1])
    with c1:
        render_probability_meter(probability, risk_label, risk_class)
    with c2:
        with st.container(border=True):
            st.subheader("Live Prediction Result")
            stat_cards(
                [
                    (
                        "Prediction",
                        "Likely to Churn" if prediction else "Likely to Stay",
                        "classification threshold: 50%",
                    ),
                    ("Probability", f"{probability * 100:.1f}%", "model-estimated churn risk"),
                    ("Monthly Charge", f"${float(customer.iloc[0]['MonthlyCharges']):.2f}", "entered customer value"),
                    ("Tenure", f"{float(customer.iloc[0]['tenure']):.0f} mo", "entered customer value"),
                    ("Total Charges", f"${float(customer.iloc[0]['TotalCharges']):.2f}", "derived from tenure × charge"),
                ]
            )
            st.markdown(f"<span class='{risk_class}'>{risk_label}</span>", unsafe_allow_html=True)

    # ── Prediction explanation ──────────────────────────────────────────────
    st.subheader("Prediction Explanation")
    for line in prediction_explanation(customer.iloc[0], probability):
        insight(line)

    # ── Retention strategies ────────────────────────────────────────────────
    st.subheader("Recommended Retention Strategies")
    strategies = get_retention_strategies(risk_label, customer.iloc[0])
    for title, desc in strategies:
        st.markdown(
            f"""<div class="strategy-card"><strong>{html.escape(title)}</strong><p>{html.escape(desc)}</p></div>""",
            unsafe_allow_html=True,
        )

    # ── Similar customers from dataset ──────────────────────────────────────
    if df is not None and not df.empty:
        st.subheader("Similar Customers in Dataset")
        st.caption("Customers with matching contract type, internet service, and similar tenure and charges.")
        cust = customer.iloc[0]
        similar = df[
            (df["Contract"] == cust["Contract"])
            & (df["InternetService"] == cust["InternetService"])
            & (df["tenure"].between(max(0, cust["tenure"] - 6), cust["tenure"] + 6))
        ].head(8)
        if similar.empty:
            similar = df.head(5)
        display_cols = ["gender", "SeniorCitizen", "Partner", "tenure", "Contract",
                        "InternetService", "MonthlyCharges", "TotalCharges", "Churn"]
        available_display = [c for c in display_cols if c in similar.columns]
        st.dataframe(similar[available_display].head(5), use_container_width=True, hide_index=True)

    # ── What-if hint ────────────────────────────────────────────────────────
    if probability >= 0.35:
        st.subheader("What-If Analysis Hint")
        hints = []
        if customer.iloc[0]["Contract"] == "Month-to-month":
            hints.append("Switching to a **One year** or **Two year** contract typically reduces churn probability significantly.")
        if customer.iloc[0]["TechSupport"] == "No":
            hints.append("Adding **Tech Support** subscription is associated with lower churn rates.")
        if customer.iloc[0]["OnlineSecurity"] == "No":
            hints.append("Enabling **Online Security** can reduce the predicted risk.")
        if customer.iloc[0]["PaymentMethod"] == "Electronic check":
            hints.append("Switching to **automatic payment** (bank transfer or credit card) correlates with lower churn.")
        if not hints:
            hints.append("Try adjusting contract length or adding service add-ons to see how the risk score changes.")
        for h in hints[:3]:
            st.markdown(f"- {h}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    df = get_data(DATA_PATH.stat().st_mtime)
    artifact = load_model_artifact()
    metrics_version = METRICS_PATH.stat().st_mtime if METRICS_PATH.exists() else None
    metrics = load_metrics(metrics_version)

    st.sidebar.markdown("## ChurnPredict AI")
    st.sidebar.caption("Business analytics + ML scoring")
    page = st.sidebar.radio("Navigation", PAGE_OPTIONS)

    filtered_df = get_filtered_data(df) if page in {"Customer Analytics", "Feature Importance"} else df

    if page == "Home":
        home_page(df, artifact, metrics)
    elif page == "Data Overview":
        data_overview_page(df)
    elif page == "Customer Analytics":
        customer_analytics_page(filtered_df, df)
    elif page == "Model Performance":
        model_performance_page(metrics)
    elif page == "Feature Importance":
        feature_importance_page(filtered_df, metrics)
    elif page == "Churn Prediction":
        prediction_page(artifact, df)


if __name__ == "__main__":
    main()
