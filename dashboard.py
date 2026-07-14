"""
Professional Sales Forecasting Dashboard
=========================================
Streamlit + Plotly interactive dashboard for business stakeholders.

Features:
  - Home with KPIs and quick forecast
  - Batch CSV Prediction with download
  - Single Prediction with auto-filled store info
  - Forecast Analysis with trend charts
  - Business Analytics
  - Model Comparison
  - Inventory Management (stockout & demand alerts)
  - Live API Monitoring
  - FastAPI Integration

Run: streamlit run dashboard.py
"""

import os
import io
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────
# Shared Feature Engineering
# ──────────────────────────────────────────────
from feature_engineering import (
    build_feature_row, build_feature_row_with_store_lookup,
    get_store_info, get_holiday_flags, ALL_FEATURES,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Forecasting Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Professional CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary: #2563eb;
    --primary-light: #3b82f6;
    --primary-dark: #1d4ed8;
    --accent: #f59e0b;
    --success: #10b981;
    --danger: #ef4444;
    --surface: #ffffff;
    --bg: #f1f5f9;
    --text: #1e293b;
    --text-muted: #64748b;
    --border: #e2e8f0;
}

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: var(--bg) !important;
}

/* Sidebar - dark bg, white text */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; padding: 6px 0; }
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    padding: 4px 10px !important;
    margin: 2px 0 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"]:hover {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(255,255,255,0.1) !important;
}

/* Main content - FORCE dark text everywhere */
main:not([data-testid="stSidebar"]) {
    color: #1e293b !important;
}
main .stMarkdown p,
main .stMarkdown span,
main .stMarkdown div,
main .stMarkdown h1,
main .stMarkdown h2,
main .stMarkdown h3,
main .stMarkdown li {
    color: #1e293b !important;
}

/* Cards */
.kpi-card {
    background: var(--surface);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    border: 1px solid var(--border);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
}

.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.kpi-value.success { background: linear-gradient(135deg, #10b981, #059669); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.kpi-value.warning { background: linear-gradient(135deg, #f59e0b, #d97706); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.kpi-value.danger  { background: linear-gradient(135deg, #ef4444, #dc2626); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

.kpi-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}

.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text) !important;
    margin: 28px 0 16px 0;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--primary);
    display: inline-block;
}

.glass-card {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 24px;
    border: 1px solid rgba(255,255,255,0.5);
    box-shadow: 0 4px 20px rgba(0,0,0,0.04);
}

/* Chart container */
.chart-container {
    background: var(--surface);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    border: 1px solid var(--border);
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.45) !important;
}

/* Table */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 30px !important;
    background: var(--surface) !important;
}

/* Fix: st.warning / st.success / st.error / st.info — dark text on light bg */
[data-testid="stAlert"] {
    background: var(--surface) !important;
    color: var(--text) !important;
    border-left-width: 4px !important;
}
[data-testid="stAlert"] p,
[data-testid="stAlert"] div,
[data-testid="stAlert"] span,
[data-testid="stAlert"] a,
[data-testid="stAlert"] strong,
[data-testid="stAlert"] b {
    color: var(--text) !important;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="warning"] {
    background: #fffbeb !important;
    border-left-color: #f59e0b !important;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="success"] {
    background: #f0fdf4 !important;
    border-left-color: #10b981 !important;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="error"] {
    background: #fef2f2 !important;
    border-left-color: #ef4444 !important;
}
/* Extra: catch-all for any alert/warning inner text */
.stAlert, .stAlert p, .stAlert div, .stAlert span {
    color: var(--text) !important;
}

/* Fix: st.metric — dark text */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetric"] label,
[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
}
/* Catch-all: any text inside stMetric */
[data-testid="stMetric"] p,
[data-testid="stMetric"] div,
[data-testid="stMetric"] span {
    color: var(--text) !important;
}
.stMetric, .stMetric p, .stMetric div, .stMetric span, .stMetric label {
    color: var(--text) !important;
}

/* Fix: st.progress — visible bar + text (Streamlit 1.59) */
[data-testid="stProgress"] {
    background: transparent !important;
}
/* Progress bar track */
[data-testid="stProgressBarTrack"] {
    background: #e2e8f0 !important;
    border-radius: 8px !important;
    height: 8px !important;
    overflow: hidden !important;
}
[data-testid="stProgressBarTrack"] > div {
    background: linear-gradient(90deg, var(--primary), var(--primary-light)) !important;
    border-radius: 8px !important;
}
/* Progress text label — target the StreamlitMarkdown container */
[data-testid="stProgress"] [data-testid="stMarkdownContainer"],
[data-testid="stProgress"] [data-testid="stMarkdownContainer"] p,
[data-testid="stProgress"] [data-testid="stMarkdownContainer"] div,
[data-testid="stProgress"] [data-testid="stMarkdownContainer"] span,
[data-testid="stProgress"] > div:first-child,
[data-testid="stProgress"] > div:first-child * {
    color: var(--text) !important;
}

/* Fix: st.spinner / st.info text */
[data-testid="stSpinner"] > div {
    color: var(--text) !important;
}

            
/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
    color: #64748b !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #1e293b !important;
}

/* Sidebar toggle arrow - visible on light bg */
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="stSidebar"] button[aria-label] svg {
    fill: #64748b !important;
    color: #64748b !important;
    stroke: #64748b !important;
}

/* Hide streamlit default elements */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: var(--bg) !important; }

/* Markdown tables & text */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown th, .stMarkdown td,
.stMarkdown blockquote, .stMarkdown strong, .stMarkdown em,
.stMarkdown code { color: #1e293b !important; }
.stMarkdown code { background: #f1f5f9 !important; padding: 2px 6px; border-radius: 4px; }
.stMarkdown th { border-bottom: 2px solid #2563eb !important; }
.stMarkdown td { border-bottom: 1px solid #e2e8f0 !important; }
.stMarkdown blockquote { border-left: 4px solid #f59e0b !important; background: #fffbeb !important; padding: 8px 16px !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Load Model
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgboost_sales_forecasting_model.pkl")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "model_metrics.json")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

# ──────────────────────────────────────────────
# Reference Data
# ──────────────────────────────────────────────
STORES = list(range(1, 55))
FAMILIES = [
    "AUTOMOTIVE", "BABY CARE", "BEAUTY", "BEVERAGES", "BOOKS",
    "BREAD/BAKERY", "CELEBRATION", "CLEANING", "DAIRY", "DELI",
    "EGGS", "FROZEN FOODS", "GROCERY I", "GROCERY II",
    "HARDWARE", "HOME AND KITCHEN I", "HOME AND KITCHEN II",
    "HOME APPLIANCES", "HOUSEHOLD", "LADIESWEAR",
    "LAWN AND GARDEN", "LINGERIE", "LIQUOR,WINE,BEER",
    "MAGAZINES", "MEATS", "PERSONAL CARE", "PET SUPPLIES",
    "PLAYERS AND ELECTRONICS", "POULTRY", "PREPARED FOODS",
    "PRODUCE", "SCHOOL AND OFFICE SUPPLIES", "SEAFOOD",
]

CITIES = ["Quito", "Santo Domingo", "Cuenca", "Latacunga", "Machala",
          "Ambato", "Guayaquil", "Daule", "Babahoyo", "Ibarra",
          "Salinas", "Quevedo", "Esmeraldas", "Libertad", "Playas",
          "Cayambe", "Manta", "Loja", "Riobamba", "Puyo",
          "Santo Domingo de los Tsachilas", "Guaranda"]
STATES = ["Pichincha", "Santo Domingo de los Tsachilas", "Azuay", "Cotopaxi",
          "El Oro", "Tungurahua", "Guayas", "Imbabura", "Manabi",
          "Esmeraldas", "Loja", "Chimborazo", "Pastaza", "Bolivar",
          "Santa Elena", "Canar"]
STORE_TYPES = ["A", "B", "C", "D", "E"]
CLUSTERS = list(range(1, 18))

# ──────────────────────────────────────────────
# Prediction Helpers
# ──────────────────────────────────────────────
def predict_sales(model, store_nbr, family, date, **kwargs):
    features = build_feature_row(store_nbr=store_nbr, family=family, date_str=date, **kwargs)
    return max(model.predict(features)[0], 0)

def predict_range(model, store_nbr, family, start, end, **kwargs):
    dates = pd.date_range(start=start, end=end, freq="D")
    rows = []
    for dt in dates:
        p = predict_sales(model, store_nbr, family, dt.strftime("%Y-%m-%d"), **kwargs)
        rows.append({"date": dt, "predicted_sales": p})
    return pd.DataFrame(rows)

def predict_batch_df(model, df_input):
    """Predict from a pre-built DataFrame of features."""
    preds = model.predict(df_input[ALL_FEATURES])
    return np.maximum(preds, 0)

# ──────────────────────────────────────────────
# Color Palette
# ──────────────────────────────────────────────
COLORS = {
    "primary": "#2563eb",
    "secondary": "#7c3aed",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "info": "#06b6d4",
    "dark": "#1e293b",
    "muted": "#94a3b8",
    "bg": "#f1f5f9",
}

def _colored_card(value, label, color_class=""):
    cls = f" {color_class}" if color_class else ""
    return f'<div class="kpi-card"><div class="kpi-value{cls}">{value}</div><div class="kpi-label">{label}</div></div>'

# ──────────────────────────────────────────────
# Sidebar Navigation
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div style="font-size:2.2rem; font-weight:800; color:#3b82f6;">PredictAI</div>
        <div style="font-size:0.8rem; color:#94a3b8; letter-spacing:2px; text-transform:uppercase;">Sales Forecasting</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔮 Single Prediction", "📁 Batch Prediction",
         "📈 Forecast Analysis", "📊 Business Analytics",
         "📉 Model Comparison", "📦 Inventory Management",
         "📡 Live Monitoring"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("Built with XGBoost + FastAPI + Streamlit")

# ──────────────────────────────────────────────
# Dynamic Inventory Builder (from Sales Data)
# ──────────────────────────────────────────────
_inventory_cache = None  # Cached after first computation


def _find_csv(candidates):
    """Find first existing file from candidate paths."""
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def build_inventory_from_sales():
    """
    Build inventory status dynamically from actual sales movement.
    Uses train.csv to compute stock levels, days until stockout, and status.
    No static inventory.csv needed — everything is predicted from sales data.
    """
    global _inventory_cache
    if _inventory_cache is not None:
        return _inventory_cache

    _base = os.path.dirname(os.path.abspath(__file__))
    train_path = _find_csv([
        os.path.join(_base, "train.csv"),
        os.path.join(_base, "store-sales-time-series-forecasting", "train.csv"),
        "/home/z/my-project/upload/grad_review/Graduation Project - Copy/store-sales-time-series-forecasting/train.csv",
    ])
    stores_path = _find_csv([
        os.path.join(_base, "stores.csv"),
        os.path.join(_base, "store-sales-time-series-forecasting", "stores.csv"),
        "/home/z/my-project/upload/grad_review/Graduation Project - Copy/store-sales-time-series-forecasting/stores.csv",
    ])

    if train_path is None:
        _inventory_cache = "NO_TRAIN"
        return None

    train = pd.read_csv(train_path, parse_dates=["date"],
                        usecols=["date", "store_nbr", "family", "sales"])
    stores_df = pd.read_csv(stores_path) if stores_path else None
    last_date = train["date"].max()
    trend_cutoff = last_date - pd.Timedelta(days=30)

    # ── Aggregate per store+family (fully vectorized) ──
    grp = train.groupby(["store_nbr", "family"])
    df = grp["sales"].agg(
        avg_daily_sales="mean",
        median_daily_sales="median",
        std_daily_sales="std",
        max_daily_sales="max",
        total_sales="sum",
        total_days="count",
    ).reset_index()

    # Zero days count
    zero_mask = train["sales"] == 0
    zero_cnt = zero_mask.groupby([train["store_nbr"], train["family"]]).sum().astype(int).reset_index()
    zero_cnt.columns = ["store_nbr", "family", "zero_days"]
    df = df.merge(zero_cnt, on=["store_nbr", "family"], how="left").fillna(0)

    # Recent 30-day average (for trend-aware predictions)
    recent = train[train["date"] >= trend_cutoff]
    rec_avg = recent.groupby(["store_nbr", "family"])["sales"].mean().reset_index()
    rec_avg.columns = ["store_nbr", "family", "recent_30d_avg"]
    df = df.merge(rec_avg, on=["store_nbr", "family"], how="left")
    df["recent_30d_avg"] = df["recent_30d_avg"].fillna(df["avg_daily_sales"])

    # Sales trend: recent vs overall
    df["sales_trend"] = np.where(
        df["avg_daily_sales"] > 0,
        df["recent_30d_avg"] / df["avg_daily_sales"],
        1.0
    )

    # ── Stock Level Simulation (analytical, vectorized) ──
    # max_capacity: enough for ~14 days at peak with 1.5x buffer
    df["max_capacity"] = np.maximum(
        df["max_daily_sales"] * 14 * 1.5,
        df["avg_daily_sales"] * 14 * 2.5
    ).clip(lower=50).astype(int)

    # How many times stock was depleted over the full period
    df["_n_restocks"] = (df["total_sales"] / df["max_capacity"]).astype(int)
    # Sales since last restock
    df["_remaining"] = df["total_sales"] - (df["_n_restocks"] * df["max_capacity"])
    # Current stock = capacity - what's been sold since last restock
    df["current_stock"] = np.maximum(df["max_capacity"] - df["_remaining"], 0).astype(int)
    df["stock_pct"] = (df["current_stock"] / df["max_capacity"] * 100).round(1)

    # Days until stockout (uses recent velocity — trend-aware)
    df["_velocity"] = df["recent_30d_avg"].clip(lower=0.01)
    df["days_until_stockout"] = (df["current_stock"] / df["_velocity"]).clip(upper=999.0).round(1)

    # Restock cycle: how many days max_capacity lasts at avg sales
    df["restock_cycle_days"] = (df["max_capacity"] / df["avg_daily_sales"].clip(lower=0.01)).astype(int).clip(lower=1)
    # Days since last restock
    df["days_since_restock"] = (df["_remaining"] / df["avg_daily_sales"].clip(lower=0.01)).astype(int).clip(lower=0)
    df["lead_time_days"] = 3
    # Reorder level: cover lead time + 20% buffer
    df["reorder_level"] = np.maximum((df["_velocity"] * 3 * 1.2).astype(int), 10)
    df["last_restock_date"] = (last_date - pd.to_timedelta(df["days_since_restock"], unit="D")).dt.strftime("%Y-%m-%d")

    # ── Stock Status (data-driven from days until stockout) ──
    conditions = [
        (df["current_stock"] <= 0) | (df["days_until_stockout"] <= 0),
        df["days_until_stockout"] <= 7,
        df["days_until_stockout"] <= 30,
    ]
    df["stock_status"] = np.select(conditions, ["OUT OF STOCK", "CRITICAL", "LOW"], default="SAFE")

    # Cleanup temp columns, round numerics
    df.drop(columns=["_n_restocks", "_remaining", "_velocity"], inplace=True)
    for c in ["avg_daily_sales", "median_daily_sales", "std_daily_sales", "sales_trend", "recent_30d_avg"]:
        df[c] = df[c].round(2)

    # Merge store metadata
    if stores_df is not None:
        df = df.merge(
            stores_df[["store_nbr", "city", "state", "type", "cluster"]],
            on="store_nbr", how="left"
        )

    col_order = ["store_nbr", "family", "city", "state", "type", "cluster",
                  "current_stock", "max_capacity", "stock_pct", "reorder_level",
                  "restock_cycle_days", "lead_time_days", "days_since_restock",
                  "last_restock_date", "avg_daily_sales", "median_daily_sales",
                  "std_daily_sales", "days_until_stockout", "stock_status",
                  "sales_trend", "recent_30d_avg"]
    df = df[[c for c in col_order if c in df.columns]]

    _inventory_cache = df
    return df


# ──────────────────────────────────────────────
# Seasonal Data Loaders (Traffic + Promotions)
# ──────────────────────────────────────────────
_seasonal_transactions_cache = {}
_seasonal_promos_cache = {}
_seasonal_loaded = False


def _load_seasonal_data():
    """Load and pre-compute seasonal averages from historical data.
    Computes per-store monthly avg transactions and per-family monthly avg onpromotion.
    These are used to make realistic seasonal predictions in the Inventory module.
    """
    global _seasonal_transactions_cache, _seasonal_promos_cache, _seasonal_loaded
    if _seasonal_loaded:
        return

    _base = os.path.dirname(os.path.abspath(__file__))
    _tx_path = os.path.join(_base, "transactions.csv")
    if not os.path.exists(_tx_path):
        _tx_path = os.path.join(_base, "store-sales-time-series-forecasting", "transactions.csv")

    _tr_path = os.path.join(_base, "train.csv")
    if not os.path.exists(_tr_path):
        _tr_path = os.path.join(_base, "store-sales-time-series-forecasting", "train.csv")

    # 1. Transactions: avg per store per month
    if os.path.exists(_tx_path):
        tx = pd.read_csv(_tx_path, parse_dates=["date"])
        tx["month"] = tx["date"].dt.month
        _seasonal_transactions_cache = (
            tx.groupby(["store_nbr", "month"])["transactions"]
            .mean()
            .to_dict()
        )

    # 2. Promotions: avg per family per month
    if os.path.exists(_tr_path):
        tr = pd.read_csv(_tr_path, parse_dates=["date"])
        tr["month"] = tr["date"].dt.month
        _seasonal_promos_cache = (
            tr.groupby(["family", "month"])["onpromotion"]
            .mean()
            .to_dict()
        )

    _seasonal_loaded = True


def get_seasonal_transactions(store_nbr: int, date: pd.Timestamp) -> float:
    """Get historical avg transactions for this store in this month."""
    key = (store_nbr, date.month)
    return _seasonal_transactions_cache.get(key, 0.0)


def get_seasonal_promotions(family: str, date: pd.Timestamp) -> float:
    """Get historical avg onpromotion for this family in this month."""
    key = (family, date.month)
    return _seasonal_promos_cache.get(key, 0.0)

# ══════════════════════════════════════════════
# PAGE 1: HOME
# ══════════════════════════════════════════════
if page == "🏠 Home":
    # Header
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <h1 style="font-size:2rem; font-weight:800; color:#1e293b; margin:0;">
            Sales Forecasting Platform
        </h1>
        <p style="color:#64748b; font-size:1rem; margin:4px 0 0 0;">
            Real-time demand prediction and business intelligence for retail operations
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Status Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(_colored_card("✅ Active" if model else "❌ Missing", "Model Status", "success" if model else "danger"), unsafe_allow_html=True)
    with c2: st.markdown(_colored_card("XGBoost", "Algorithm"), unsafe_allow_html=True)
    with c3: st.markdown(_colored_card("54", "Stores"), unsafe_allow_html=True)
    with c4: st.markdown(_colored_card("33", "Product Families"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Quick Forecast Widget
    st.markdown('<div class="section-title">Quick Forecast</div>', unsafe_allow_html=True)

    qc1, qc2, qc3, qc4 = st.columns([1, 1.5, 1.5, 1])
    with qc1:
        qs = st.selectbox("Store", STORES, key="qs_store")
    with qc2:
        qf = st.selectbox("Family", FAMILIES, index=12, key="qs_family")
    with qc3:
        qd = st.date_input("Date", datetime(2017, 8, 15), key="qs_date")
    with qc4:
        qp = st.number_input("Promotions", 0, 100, 0, key="qs_promo")
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run_quick = st.button("⚡ Forecast", type="primary", use_container_width=True)

    if run_quick and model:
        with st.spinner(""):
            pred = predict_sales(model, qs, qf, qd.strftime("%Y-%m-%d"), onpromotion=qp,
                                 **get_store_info(qs))
        hc1, hc2, hc3 = st.columns(3)
        with hc1:
            st.markdown(f"""
            <div class="kpi-card" style="text-align:center; border-left:4px solid #2563eb;">
                <div class="kpi-value">{pred:,.0f}</div>
                <div class="kpi-label">Predicted Sales</div>
            </div>""", unsafe_allow_html=True)
        with hc2:
            si = get_store_info(qs)
            st.markdown(f"""
            <div class="kpi-card" style="text-align:center; border-left:4px solid #10b981;">
                <div class="kpi-value success">{si['city']}</div>
                <div class="kpi-label">Store Location</div>
            </div>""", unsafe_allow_html=True)
        with hc3:
            dt = pd.Timestamp(qd)
            st.markdown(f"""
            <div class="kpi-card" style="text-align:center; border-left:4px solid #f59e0b;">
                <div class="kpi-value warning">{dt.strftime('%A')}</div>
                <div class="kpi-label">Day of Week</div>
            </div>""", unsafe_allow_html=True)
    elif run_quick and not model:
        st.error("Model not loaded. Place `xgboost_sales_forecasting_model.pkl` in the same folder.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 3: Platform Features
    st.markdown('<div class="section-title">Platform Capabilities</div>', unsafe_allow_html=True)
    fc1, fc2, fc3, fc4 = st.columns(4)

    features = [
        ("🔮", "Single Prediction", "Get instant sales forecast for any store-product-date combination"),
        ("📁", "Batch Prediction", "Upload CSV files for bulk forecasting with downloadable results"),
        ("📈", "Forecast Analysis", "Visualize sales trends, seasonal patterns, and demand projections"),
        ("📡", "Live Monitoring", "Track model performance, detect drift, and ensure prediction quality"),
    ]
    for col, (icon, title, desc) in zip([fc1, fc2, fc3, fc4], features):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
                <div style="font-size:2.5rem;">{icon}</div>
                <div style="font-weight:700; font-size:1.05rem; margin:8px 0 4px 0;">{title}</div>
                <div style="color:#64748b; font-size:0.85rem; line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture
    st.markdown('<div class="section-title">System Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <div style="display:flex; gap:20px; flex-wrap:wrap; justify-content:center; text-align:center;">
            <div style="flex:1; min-width:140px; padding:16px; border-radius:12px; background:#eff6ff;">
                <div style="font-size:1.8rem;">🧠</div>
                <div style="font-weight:700; color:#1e40af;">XGBoost Model</div>
                <div style="color:#64748b; font-size:0.8rem;">Trained Pipeline</div>
            </div>
            <div style="flex:0 0 40px; display:flex; align-items:center; color:#94a3b8; font-size:1.5rem;">→</div>
            <div style="flex:1; min-width:140px; padding:16px; border-radius:12px; background:#f0fdf4;">
                <div style="font-size:1.8rem;">⚡</div>
                <div style="font-weight:700; color:#166534;">FastAPI</div>
                <div style="color:#64748b; font-size:0.8rem;">REST API Service</div>
            </div>
            <div style="flex:0 0 40px; display:flex; align-items:center; color:#94a3b8; font-size:1.5rem;">→</div>
            <div style="flex:1; min-width:140px; padding:16px; border-radius:12px; background:#fef3c7;">
                <div style="font-size:1.8rem;">📊</div>
                <div style="font-weight:700; color:#92400e;">Streamlit</div>
                <div style="color:#64748b; font-size:0.8rem;">Interactive Dashboard</div>
            </div>
            <div style="flex:0 0 40px; display:flex; align-items:center; color:#94a3b8; font-size:1.5rem;">→</div>
            <div style="flex:1; min-width:140px; padding:16px; border-radius:12px; background:#fce7f3;">
                <div style="font-size:1.8rem;">🏢</div>
                <div style="font-weight:700; color:#9d174d;">Business Users</div>
                <div style="color:#64748b; font-size:0.8rem;">Decision Making</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 2: SINGLE PREDICTION
# ══════════════════════════════════════════════
elif page == "🔮 Single Prediction":
    st.markdown("""
    <h1 style="font-size:2rem; font-weight:800; color:#1e293b; margin:0 0 4px 0;">
        🔮 Single Prediction
    </h1>
    <p style="color:#64748b;">Get a sales forecast for a specific store, product, and date</p>
    """, unsafe_allow_html=True)

    if not model:
        st.error("Model not found! Place `xgboost_sales_forecasting_model.pkl` in the same directory.")
        st.stop()

    tab1, tab2 = st.tabs(["📋 Manual Input", "🔄 Auto (Store Lookup)"])

    # Tab 1: Manual
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title">Product & Date</div>', unsafe_allow_html=True)
            store_nbr = st.selectbox("Store Number", STORES, key="sp_store")
            family = st.selectbox("Product Family", FAMILIES, index=12, key="sp_family")
            pred_date = st.date_input("Prediction Date", datetime(2017, 8, 15), key="sp_date")
            onpromotion = st.number_input("Products on Promotion", 0, 500, 0, key="sp_promo")

        with c2:
            st.markdown('<div class="section-title">Store & External Factors</div>', unsafe_allow_html=True)
            city = st.selectbox("City", CITIES, key="sp_city")
            state = st.selectbox("State", STATES, key="sp_state")
            store_type = st.selectbox("Store Type", STORE_TYPES, key="sp_type")
            cluster = st.selectbox("Cluster", CLUSTERS, key="sp_cluster")
            oil = st.number_input("Oil Price (dcoilwtico)", 0.0, 150.0, 46.54, key="sp_oil")
            trans = st.number_input("Transactions", 0.0, 10000.0, 2100.0, step=50.0, key="sp_trans")

        if st.button("🔮 Predict Sales", type="primary", use_container_width=True):
            with st.spinner("Computing prediction..."):
                try:
                    pred = predict_sales(model, store_nbr, family, pred_date.strftime("%Y-%m-%d"),
                                         onpromotion=onpromotion, oil_price=oil, transactions=trans,
                                         city=city, state=state, store_type=store_type, cluster=cluster)
                    st.success("Prediction complete!")

                    # Result cards
                    r1, r2, r3, r4 = st.columns(4)
                    with r1:
                        st.markdown(f'<div class="kpi-card" style="text-align:center; border-left:4px solid #2563eb;"><div class="kpi-value">{pred:,.2f}</div><div class="kpi-label">Predicted Sales</div></div>', unsafe_allow_html=True)
                    with r2:
                        st.markdown(f'<div class="kpi-card" style="text-align:center; border-left:4px solid #10b981;"><div class="kpi-value success">{family}</div><div class="kpi-label">Product Family</div></div>', unsafe_allow_html=True)
                    with r3:
                        st.markdown(f'<div class="kpi-card" style="text-align:center; border-left:4px solid #f59e0b;"><div class="kpi-value warning">Store #{store_nbr}</div><div class="kpi-label">Store</div></div>', unsafe_allow_html=True)
                    with r4:
                        dt = pd.Timestamp(pred_date)
                        st.markdown(f'<div class="kpi-card" style="text-align:center; border-left:4px solid #7c3aed;"><div class="kpi-value" style="background:linear-gradient(135deg,#7c3aed,#6d28d9);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{dt.strftime("%b %d, %Y")}</div><div class="kpi-label">Date</div></div>', unsafe_allow_html=True)

                    # Context
                    is_hol, is_nat, is_reg, is_loc, is_evt = get_holiday_flags(pred_date.strftime("%Y-%m-%d"), city=city, state=state)
                    ctx1, ctx2, ctx3, ctx4 = st.columns(4)
                    with ctx1: st.info(f"📅 **{dt.strftime('%A')}**")
                    with ctx2: st.info(f"🏷️ Promo: {'Yes ('+str(onpromotion)+')' if onpromotion > 0 else 'No'}")
                    with ctx3: st.info(f"🎉 Holiday: {'Yes' if is_hol else 'No'}")
                    with ctx4: st.info(f"🎯 Event: {'Yes' if is_evt else 'No'}")

                except Exception as e:
                    st.error(f"Error: {e}")

    # Tab 2: Auto with store lookup
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title">Select Store & Product</div>', unsafe_allow_html=True)
            auto_store = st.selectbox("Store", STORES, key="auto_store")
            auto_family = st.selectbox("Family", FAMILIES, index=12, key="auto_family")
            auto_date = st.date_input("Date", datetime(2017, 8, 15), key="auto_date")
            auto_promo = st.number_input("Promotions", 0, 500, 0, key="auto_promo")

        with c2:
            si = get_store_info(auto_store)
            st.markdown('<div class="section-title">Auto-Filled Store Info</div>', unsafe_allow_html=True)
            st.info(f"**City:** {si['city']}\n\n**State:** {si['state']}\n\n**Type:** {si['store_type']}\n\n**Cluster:** {si['cluster']}")
            auto_oil = st.number_input("Oil Price", 0.0, 150.0, 46.54, key="auto_oil")
            auto_trans = st.number_input("Transactions", 0.0, 10000.0, 2100.0, step=50.0, key="auto_trans")

        if st.button("⚡ Auto Predict", type="primary", use_container_width=True):
            with st.spinner(""):
                pred = predict_sales(model, auto_store, auto_family, auto_date.strftime("%Y-%m-%d"),
                                     onpromotion=auto_promo, oil_price=auto_oil, transactions=auto_trans,
                                     **si)
            st.markdown(f'<div class="kpi-card" style="text-align:center; border-left:4px solid #2563eb;"><div class="kpi-value">{pred:,.2f}</div><div class="kpi-label">Predicted Sales — Store #{auto_store} | {auto_family}</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 3: BATCH PREDICTION
# ══════════════════════════════════════════════
elif page == "📁 Batch Prediction":
    st.markdown("""
    <h1 style="font-size:2rem; font-weight:800; color:#1e293b; margin:0 0 4px 0;">
        📁 Batch Prediction
    </h1>
    <p style="color:#64748b;">Upload a CSV file with prediction requests, or generate a template</p>
    """, unsafe_allow_html=True)

    if not model:
        st.error("Model not found!")
        st.stop()

    bt1, bt2 = st.tabs(["📤 Upload CSV", "📝 Template & Format"])

    with bt1:
        uploaded = st.file_uploader("Upload CSV file", type=["csv"], help="CSV with columns: store_nbr, family, date, onpromotion, oil_price, transactions")

        # Save uploaded data to session_state so it survives button reruns
        if uploaded is not None:
            try:
                st.session_state["batch_df"] = pd.read_csv(uploaded)
                st.session_state["batch_file_name"] = uploaded.name
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.session_state.pop("batch_df", None)

        if "batch_df" in st.session_state and st.session_state["batch_df"] is not None:
            df_up = st.session_state["batch_df"]
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(df_up):,}</div><div class="kpi-label">Rows Uploaded</div></div>', unsafe_allow_html=True)

            with st.expander("Preview Uploaded Data", expanded=True):
                st.dataframe(df_up.head(10), use_container_width=True)

            required = {"store_nbr", "family", "date"}
            has_all = required.issubset(df_up.columns)
            if not has_all:
                missing = required - set(df_up.columns)
                st.error(f"Missing required columns: {missing}")
                st.stop()

            if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                try:
                    # Check if file has an 'id' column (e.g., Kaggle test set)
                    has_id = "id" in df_up.columns

                    # Build features for ALL rows at once, then predict once
                    progress_bar = st.progress(0, text="Building features...")
                    all_features_list = []
                    total = len(df_up)

                    for idx, row in df_up.iterrows():
                        feat = build_feature_row_with_store_lookup(
                            store_nbr=int(row["store_nbr"]),
                            family=str(row["family"]).strip(),
                            date_str=str(row["date"])[:10],
                            onpromotion=int(row.get("onpromotion", 0)),
                            oil_price=float(row.get("oil_price", 0)),
                            transactions=float(row.get("transactions", 0)),
                        )
                        all_features_list.append(feat)

                        # Update progress every 500 rows
                        if (idx + 1) % 500 == 0 or (idx + 1) == total:
                            progress_bar.progress(
                                min((idx + 1) / total, 1.0),
                                text=f"Building features... {idx + 1:,} / {total:,}"
                            )

                    progress_bar.progress(1.0, text="Running model prediction...")

                    # Single predict call for ALL rows at once (much faster!)
                    all_features = pd.concat(all_features_list, ignore_index=True)
                    predictions = model.predict(all_features)
                    predictions = np.maximum(predictions, 0)

                    progress_bar.empty()

                    # Build results DataFrame
                    results = []
                    for i, (_, row) in enumerate(df_up.iterrows()):
                        r = {
                            "store_nbr": int(row["store_nbr"]),
                            "family": str(row["family"]).strip(),
                            "date": str(row["date"])[:10],
                            "onpromotion": int(row.get("onpromotion", 0)),
                            "predicted_sales": round(float(predictions[i]), 2),
                        }
                        if has_id:
                            r["id"] = int(row["id"])
                        results.append(r)

                    df_results = pd.DataFrame(results)

                    # Reorder columns: id first if present
                    if has_id:
                        df_results = df_results[["id", "store_nbr", "family", "date", "onpromotion", "predicted_sales"]]

                    st.success(f"Predictions complete for {len(df_results):,} rows!")

                    # Summary
                    s1, s2, s3, s4 = st.columns(4)
                    with s1: st.metric("Total Predicted Sales", f"{df_results['predicted_sales'].sum():,.2f}")
                    with s2: st.metric("Average", f"{df_results['predicted_sales'].mean():,.2f}")
                    with s3: st.metric("Max", f"{df_results['predicted_sales'].max():,.2f}")
                    with s4: st.metric("Min", f"{df_results['predicted_sales'].min():,.2f}")

                    st.dataframe(df_results, use_container_width=True)

                    csv_out = df_results.to_csv(index=False)
                    st.download_button("📥 Download Predictions CSV", csv_out,
                                      "batch_predictions.csv", "text/csv")

                except Exception as e:
                    st.error(f"Error during prediction: {e}")

            # Button to clear uploaded data
            if st.button("🗑️ Clear Uploaded File"):
                st.session_state.pop("batch_df", None)
                st.session_state.pop("batch_file_name", None)
                st.rerun()

    with bt2:
        st.markdown("""
        ### Required CSV Format

        Your CSV must have these **columns** (other columns like `id` will be kept in the output):

        | Column | Type | Required | Description |
        |--------|------|----------|-------------|
        | `store_nbr` | int | **Yes** | Store number (1-54) |
        | `family` | string | **Yes** | Product family name |
        | `date` | string | **Yes** | Date in YYYY-MM-DD format |
        | `onpromotion` | int | No (default 0) | Number of items on promotion |
        | `oil_price` | float | No (default 0) | Oil price |
        | `transactions` | float | No (default 0) | Store transactions |
        | `id` | int | No | Row ID (kept in output if present) |

        > **Note:** `city`, `state`, `type`, and `cluster` are auto-filled from `stores.csv`.
        > The Kaggle `test.csv` format (`id, date, store_nbr, family, onpromotion`) is fully supported.
        """)

        # Generate template
        template = pd.DataFrame({
            "store_nbr": [1, 1, 2, 3],
            "family": ["GROCERY I", "BEVERAGES", "PRODUCE", "DAIRY"],
            "date": ["2017-08-15", "2017-08-15", "2017-08-16", "2017-08-16"],
            "onpromotion": [5, 0, 3, 10],
            "oil_price": [46.54, 46.54, 47.0, 47.0],
            "transactions": [2100.0, 2100.0, 1800.0, 1500.0],
        })
        st.download_button("📥 Download Template CSV", template.to_csv(index=False),
                          "batch_template.csv", "text/csv")


# ══════════════════════════════════════════════
# PAGE 4: FORECAST ANALYSIS
# ══════════════════════════════════════════════
elif page == "📈 Forecast Analysis":
    st.markdown("""
    <h1 style="font-size:2rem; font-weight:800; color:#1e293b; margin:0 0 4px 0;">
        📈 Forecast Analysis
    </h1>
    <p style="color:#64748b;">Visualize sales trends, seasonal patterns, and demand projections</p>
    """, unsafe_allow_html=True)

    if not model:
        st.error("Model not found!")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Selection</div>', unsafe_allow_html=True)
        fa_store = st.selectbox("Store", STORES, key="fa_store")
        fa_family = st.selectbox("Family", FAMILIES, index=12, key="fa_family")
    with c2:
        st.markdown('<div class="section-title">Date Range</div>', unsafe_allow_html=True)
        fa_start = st.date_input("Start", datetime(2017, 8, 1), key="fa_start")
        fa_end = st.date_input("End", datetime(2017, 8, 31), key="fa_end")
    fa_promo = st.slider("Promotions", 0, 100, 0, key="fa_promo")

    if fa_start < fa_end:
        if st.button("📈 Generate Forecast", type="primary", use_container_width=True):
            si = get_store_info(fa_store)
            with st.spinner("Generating..."):
                df_f = predict_range(model, fa_store, fa_family,
                                     fa_start.strftime("%Y-%m-%d"),
                                     fa_end.strftime("%Y-%m-%d"),
                                     onpromotion=fa_promo, **si)

                # KPIs
                k1, k2, k3, k4 = st.columns(4)
                with k1: st.metric("Daily Average", f"{df_f['predicted_sales'].mean():,.2f}")
                with k2: st.metric("Peak Day", f"{df_f['predicted_sales'].max():,.2f}")
                with k3: st.metric("Lowest Day", f"{df_f['predicted_sales'].min():,.2f}")
                with k4: st.metric("Total Period", f"{df_f['predicted_sales'].sum():,.2f}")

                # Main trend chart
                st.markdown('<div class="section-title">Sales Forecast Trend</div>', unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_f["date"], y=df_f["predicted_sales"],
                    mode="lines+markers", name="Predicted Sales",
                    line=dict(color=COLORS["primary"], width=2.5),
                    marker=dict(size=5, color=COLORS["warning"]),
                    fill="tozeroy",
                    fillcolor="rgba(37,99,235,0.08)",
                ))
                fig.update_layout(
                    title=f"Forecast: Store #{fa_store} | {fa_family}",
                    template="plotly_white", hovermode="x unified",
                    yaxis_title="Sales", height=420,
                    margin=dict(t=50, b=30, l=60, r=30),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Day of week + Weekend vs Weekday
                ch1, ch2 = st.columns(2)

                with ch1:
                    st.markdown('<div class="section-title">Day-of-Week Pattern</div>', unsafe_allow_html=True)
                    df_f["day_name"] = df_f["date"].dt.day_name()
                    dow = df_f.groupby("day_name")["predicted_sales"].mean()
                    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                    dow = dow.reindex(day_order)
                    fig2 = px.bar(dow.reset_index(), x="day_name", y="predicted_sales",
                                 color="predicted_sales", color_continuous_scale="Blues",
                                 title="Avg Sales by Day of Week", template="plotly_white")
                    fig2.update_layout(height=350)
                    st.plotly_chart(fig2, use_container_width=True)

                with ch2:
                    st.markdown('<div class="section-title">Weekend vs Weekday</div>', unsafe_allow_html=True)
                    df_f["is_weekend"] = df_f["date"].dt.dayofweek.isin([5,6]).astype(int)
                    ww = df_f.groupby("is_weekend")["predicted_sales"].mean()
                    ww.index = ww.index.map({0: "Weekday", 1: "Weekend"})
                    ww.index.name = "is_weekend"
                    fig3 = px.bar(ww.reset_index(), x="is_weekend", y="predicted_sales",
                                 color="is_weekend",
                                 color_discrete_map={"Weekday": COLORS["primary"], "Weekend": COLORS["warning"]},
                                 title="Avg Sales: Weekday vs Weekend", template="plotly_white")
                    fig3.update_layout(height=350, showlegend=False)
                    st.plotly_chart(fig3, use_container_width=True)

                # Download
                csv = df_f[["date","predicted_sales"]].to_csv(index=False)
                st.download_button("📥 Download Forecast CSV", csv,
                                  f"forecast_store{fa_store}_{fa_family.replace('/','_')}.csv", "text/csv")
    else:
        st.warning("Start date must be before end date.")


# ══════════════════════════════════════════════
# PAGE 5: BUSINESS ANALYTICS
# ══════════════════════════════════════════════
elif page == "📊 Business Analytics":
    st.markdown("""
    <h1 style="font-size:2rem; font-weight:800; color:#1e293b; margin:0 0 4px 0;">
        📊 Business Analytics
    </h1>
    <p style="color:#64748b;">Compare forecasts across stores and product families for strategic insights</p>
    """, unsafe_allow_html=True)

    if not model:
        st.error("Model not found!")
        st.stop()

    at1, at2 = st.tabs(["🏪 Store Comparison", "🛒 Family Comparison"])

    with at1:
        st.markdown('<div class="section-title">Multi-Store Forecast Comparison</div>', unsafe_allow_html=True)
        sc1, sc2 = st.columns(2)
        with sc1:
            sel_stores = st.multiselect("Select Stores (max 10)", STORES, default=[1,2,3,4,5], key="ba_stores")
            ba_family = st.selectbox("Family", FAMILIES, index=12, key="ba_family")
        with sc2:
            ba_date = st.date_input("Target Date", datetime(2017, 8, 15), key="ba_date")
            ba_promo = st.slider("Promotions", 0, 100, 0, key="ba_promo")

        if st.button("📊 Compare Stores", type="primary", use_container_width=True):
            if not sel_stores:
                st.warning("Select at least one store.")
            else:
                with st.spinner(""):
                    results = []
                    for s in sel_stores:
                        si = get_store_info(s)
                        p = predict_sales(model, s, ba_family, ba_date.strftime("%Y-%m-%d"), onpromotion=ba_promo, **si)
                        results.append({"store_nbr": s, "city": si["city"], "type": si["store_type"],
                                       "cluster": si["cluster"], "predicted_sales": p})
                    df_comp = pd.DataFrame(results).sort_values("predicted_sales", ascending=False)

                    fig = px.bar(df_comp, x="store_nbr", y="predicted_sales",
                                color="predicted_sales", color_continuous_scale="Blues",
                                text="predicted_sales", title=f"Store Comparison — {ba_family}",
                                template="plotly_white")
                    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside", textfont_color="#1e293b")
                    fig.update_layout(height=420)
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(df_comp.style.format({"predicted_sales": "{:,.2f}"}), use_container_width=True)

    with at2:
        st.markdown('<div class="section-title">Product Family Forecast Comparison</div>', unsafe_allow_html=True)
        fc1, fc2 = st.columns(2)
        with fc1:
            bf_store = st.selectbox("Store", STORES, key="bf_store")
            sel_families = st.multiselect("Families", FAMILIES,
                                         default=["GROCERY I","BEVERAGES","PRODUCE","DAIRY","DELI"], key="bf_fams")
        with fc2:
            bf_date = st.date_input("Date", datetime(2017, 8, 15), key="bf_date")
            bf_promo = st.slider("Promotions", 0, 100, 0, key="bf_promo")

        if st.button("📊 Compare Families", type="primary", use_container_width=True):
            if not sel_families:
                st.warning("Select at least one family.")
            else:
                with st.spinner(""):
                    si = get_store_info(bf_store)
                    results = []
                    for f in sel_families:
                        p = predict_sales(model, bf_store, f, bf_date.strftime("%Y-%m-%d"), onpromotion=bf_promo, **si)
                        results.append({"family": f, "predicted_sales": p})
                    df_comp = pd.DataFrame(results).sort_values("predicted_sales", ascending=False)

                    fig = px.bar(df_comp, x="family", y="predicted_sales",
                                color="predicted_sales", color_continuous_scale="Viridis",
                                text="predicted_sales", title=f"Family Comparison — Store #{bf_store}",
                                template="plotly_white")
                    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside", textfont_color="#1e293b")
                    fig.update_xaxes(tickangle=45)
                    fig.update_layout(height=420)
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(df_comp.style.format({"predicted_sales": "{:,.2f}"}), use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 6: MODEL COMPARISON
# ══════════════════════════════════════════════
elif page == "📉 Model Comparison":
    st.markdown("""
    <h1 style="font-size:2rem; font-weight:800; color:#1e293b; margin:0 0 4px 0;">
        📉 Model Comparison
    </h1>
    <p style="color:#64748b;">Evaluation metrics and performance analysis across all trained models</p>
    """, unsafe_allow_html=True)

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            comp_data = json.load(f)
        df_comp = pd.DataFrame(comp_data).sort_values("RMSE")
        st.success("Metrics loaded from `model_metrics.json`")
    else:
        st.warning("`model_metrics.json` not found. Add this cell to your notebook after model_comparison:")
        st.code("import json\nmetrics = model_comparison.to_dict(orient='records')\nwith open('model_metrics.json','w') as f:\n    json.dump(metrics, f, indent=2)")
        st.info("Showing placeholder below.")
        df_comp = pd.DataFrame({
            "Model": ["Linear Regression", "Random Forest", "XGBoost", "SARIMA"],
            "MAE": [0,0,0,0], "MSE": [0,0,0,0], "RMSE": [0,0,0,0], "R2 Score": [0,0,0,0],
        })

    # Table
    st.markdown('<div class="section-title">Performance Summary</div>', unsafe_allow_html=True)
    st.dataframe(
        df_comp.style.format({"MAE":"{:,.2f}","MSE":"{:,.2f}","RMSE":"{:,.2f}","R2 Score":"{:.4f}"})
        .background_gradient(subset=["RMSE"], cmap="RdYlGn_r"),
        use_container_width=True, hide_index=True,
    )

    # Charts side by side
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown('<div class="section-title">RMSE Comparison</div>', unsafe_allow_html=True)
        fig_rmse = px.bar(df_comp, x="Model", y="RMSE", color="RMSE",
                         color_continuous_scale="RdYlGn_r", template="plotly_white")
        fig_rmse.update_layout(height=380)
        st.plotly_chart(fig_rmse, use_container_width=True)
    with ch2:
        st.markdown('<div class="section-title">R-squared Comparison</div>', unsafe_allow_html=True)
        fig_r2 = px.bar(df_comp, x="Model", y="R2 Score", color="R2 Score",
                       color_continuous_scale="RdYlGn", template="plotly_white")
        fig_r2.update_layout(height=380)
        st.plotly_chart(fig_r2, use_container_width=True)

    # Best model card
    if os.path.exists(METRICS_PATH):
        best = df_comp.iloc[0]
        st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid #10b981;">
            <div style="font-size:1.3rem; font-weight:700; color:#166534; margin-bottom:8px;">
                🏆 Best Model: {best['Model']}
            </div>
            <div style="display:flex; gap:30px; flex-wrap:wrap;">
                <div><span style="color:#64748b;">RMSE:</span> <strong>{best['RMSE']:.2f}</strong></div>
                <div><span style="color:#64748b;">MAE:</span> <strong>{best['MAE']:.2f}</strong></div>
                <div><span style="color:#64748b;">R2:</span> <strong>{best['R2 Score']:.4f}</strong></div>
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════


# ══════════════════════════════════════════════
# PAGE 7: INVENTORY MANAGEMENT
# ══════════════════════════════════════════════
elif page == "📦 Inventory Management":
    st.markdown("""
    <h1 style="font-size:2rem; font-weight:800; color:#1e293b; margin:0 0 4px 0;">
        📦 Inventory Management
    </h1>
    <p style="color:#64748b;">Stock status predicted from actual sales movement</p>
    """, unsafe_allow_html=True)


    # Build inventory dynamically from sales
    with st.spinner("Analyzing sales movement to predict stock status..."):
        df_inv = build_inventory_from_sales()

    if df_inv is None or (isinstance(df_inv, str) and df_inv == "NO_TRAIN"):
        st.error("Could not find **train.csv**. Make sure your sales data is available.")
    else:
        # ── KPI Cards ──
        total_skus = len(df_inv)
        out_of_stock = len(df_inv[df_inv["stock_status"] == "OUT OF STOCK"])
        critical = len(df_inv[df_inv["stock_status"] == "CRITICAL"])
        low_stock = len(df_inv[df_inv["stock_status"] == "LOW"])
        safe = len(df_inv[df_inv["stock_status"] == "SAFE"])
        avg_trend = df_inv["sales_trend"].mean() if "sales_trend" in df_inv.columns else 1.0

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            st.markdown(_colored_card(f"{total_skus:,}", "Total SKUs"), unsafe_allow_html=True)
        with k2:
            st.markdown(_colored_card(f"{out_of_stock:,}", "Out of Stock", "danger"), unsafe_allow_html=True)
        with k3:
            st.markdown(_colored_card(f"{critical:,}", "Critical", "warning"), unsafe_allow_html=True)
        with k4:
            st.markdown(_colored_card(f"{low_stock:,}", "Low Stock", "warning"), unsafe_allow_html=True)
        with k5:
            st.markdown(_colored_card(f"{safe:,}", "Safe", "success"), unsafe_allow_html=True)
        with k6:
            trend_clr = "danger" if avg_trend > 1.15 else ("success" if avg_trend < 0.9 else "default")
            st.markdown(_colored_card(f"{avg_trend:.2f}x", "Avg Sales Trend", trend_clr), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Filters ──
        f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
        with f1:
            selected_store = st.selectbox("Filter by Store", [0] + STORES, format_func=lambda x: "All Stores" if x == 0 else f"Store {x}", key="inv_store")
        with f2:
            all_families = sorted(df_inv["family"].unique().tolist())
            selected_family = st.selectbox("Filter by Family", ["All"] + all_families, key="inv_family")
        with f3:
            status_options = ["All", "SAFE", "LOW", "CRITICAL", "OUT OF STOCK"]
            selected_status = st.selectbox("Filter by Status", status_options, key="inv_status")
        with f4:
            trend_options = ["All", "Surging (>1.1x)", "Stable (0.9-1.1x)", "Declining (<0.9x)"]
            selected_trend = st.selectbox("Filter by Trend", trend_options, key="inv_trend")

        # Apply filters
        df_filtered = df_inv.copy()
        if selected_store != 0:
            df_filtered = df_filtered[df_filtered["store_nbr"] == selected_store]
        if selected_family != "All":
            df_filtered = df_filtered[df_filtered["family"] == selected_family]
        if selected_status != "All":
            df_filtered = df_filtered[df_filtered["stock_status"] == selected_status]
        if "sales_trend" in df_filtered.columns:
            if selected_trend == "Surging (>1.1x)":
                df_filtered = df_filtered[df_filtered["sales_trend"] > 1.1]
            elif selected_trend == "Declining (<0.9x)":
                df_filtered = df_filtered[df_filtered["sales_trend"] < 0.9]
            elif selected_trend == "Stable (0.9-1.1x)":
                df_filtered = df_filtered[(df_filtered["sales_trend"] >= 0.9) & (df_filtered["sales_trend"] <= 1.1)]

        st.markdown(f'<div class="kpi-card" style="margin-bottom:16px;"><div class="kpi-value">{len(df_filtered):,}</div><div class="kpi-label">Filtered Results</div></div>', unsafe_allow_html=True)

        # ── Tabs ──
        inv_tab1, inv_tab2, inv_tab3 = st.tabs(["📋 Stock Status Table", "⚠️ Stockout & Demand Alerts", "📊 Inventory Charts"])

        with inv_tab1:
            # Color-code the status column for display
            def _status_color(status):
                colors = {
                    "SAFE": "#10b981",
                    "LOW": "#f59e0b",
                    "CRITICAL": "#ef4444",
                    "OUT OF STOCK": "#dc2626",
                }
                return colors.get(status, "#64748b")

            if len(df_filtered) > 0:
                # Show top 200 rows with sales trend
                _display_cols = [
                    "store_nbr", "family", "city", "current_stock",
                    "stock_pct", "days_until_stockout", "avg_daily_sales",
                    "sales_trend", "stock_status",
                ]
                # Only include columns that exist
                _display_cols = [c for c in _display_cols if c in df_filtered.columns]
                display_df = df_filtered[_display_cols].head(200).copy()
                display_df.columns = ["Store", "Family", "City", "Current Stock",
                                      "Stock %", "Days to Stockout", "Avg Daily Sales",
                                      "Sales Trend", "Status"]

                st.dataframe(display_df, use_container_width=True, height=500)
                if len(df_filtered) > 200:
                    st.info(f"Showing 200 of {len(df_filtered):,} rows. Use filters to narrow down.")
            else:
                st.info("No items match the selected filters.")

        with inv_tab2:
            from datetime import timedelta as td

            # ── Date & Holiday Selection ──
            st.markdown('<div class="section-title">Select Forecast Period</div>', unsafe_allow_html=True)

            # Quick holiday presets (month-day only, year picked by user)
            holiday_presets = {
                "Custom Date": None,
                "Christmas (Dec 16-25)": ("12-16", 10),
                "Christmas Week (Dec 20-27)": ("12-20", 8),
                "New Year (Dec 25 - Jan 1)": ("12-25", 8),
                "Holy Week (Apr 10-16)": ("04-10", 7),
                "Black Friday (Nov 20-27)": ("11-20", 8),
                "Dia del Trabajo (Apr 28 - May 3)": ("04-28", 6),
                "Independencia Guayaquil (Oct 5-12)": ("10-05", 8),
                "Carnival (Feb 25 - Mar 1)": ("02-25", 5),
                "Mother's Day (May 5-12)": ("05-05", 8),
                "Back to School (Aug 10-20)": ("08-10", 11),
                "Cyber Monday (Nov 24-30)": ("11-24", 7),
            }

            yr1, yr2, yr3 = st.columns([1, 2, 2])
            with yr1:
                selected_year = st.number_input("Year", min_value=2012, max_value=2030, value=2025, step=1, key="inv_year")
            with yr2:
                selected_preset = st.selectbox("Quick Holiday Select", list(holiday_presets.keys()), key="inv_preset")
            with yr3:
                preset_val = holiday_presets[selected_preset]
                if preset_val is not None:
                    mm_dd = preset_val[0]
                    default_date = pd.Timestamp(f"{selected_year}-{mm_dd}")
                    default_days = preset_val[1]
                    st.info(f"{selected_year}-{mm_dd} for {preset_val[1]} days")
                else:
                    default_date = pd.Timestamp(f"{selected_year}-01-01")
                    default_days = 7

                forecast_days = st.number_input("Forecast Days", 1, 60, value=default_days, key="inv_days_num")

            base_date = st.date_input("Start Date", value=default_date.date(), key="inv_base_date")
            base_date = pd.Timestamp(base_date)
            end_date = base_date + td(days=forecast_days)

            st.markdown(f"<p style='color:#64748b; font-size:0.9rem;'>Forecasting from <strong>{base_date.strftime('%Y-%m-%d')}</strong> to <strong>{end_date.strftime('%Y-%m-%d')}</strong> ({forecast_days} days)</p>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Thresholds ──
            tc1, tc2 = st.columns(2)
            with tc1:
                surge_threshold = st.selectbox("Demand Surge Threshold", [1.2, 1.3, 1.5, 2.0],
                                               format_func=lambda x: f"{int((x-1)*100)}% increase",
                                               index=0, key="inv_surge")
            with tc2:
                prestockout_pct = st.slider("Pre-Stockout Alert (%)", 5, 50, 25, 5,
                                            help="Alert when stock drops below this % of capacity")

            if model is not None:
                # ── Upcoming Holidays & Events ──
                st.markdown('<div class="section-title">Holidays & Seasonal Events in This Period</div>', unsafe_allow_html=True)

                # Load holidays — use get_holiday_flags (supports any year via recurring patterns)
                _forecast_dates_hol = pd.date_range(start=base_date, end=end_date, freq="D")
                _hol_rows = []
                # Get unique cities/states from inventory for regional/local context
                _inv_cities = df_filtered["city"].unique().tolist() if "city" in df_filtered.columns else []
                _inv_states = df_filtered["state"].unique().tolist() if "state" in df_filtered.columns else []
                for _d in _forecast_dates_hol:
                    _ds = _d.strftime("%Y-%m-%d")
                    # Check with Quito/Pichincha as default (national always applies)
                    _h, _n, _r, _l, _e = get_holiday_flags(_ds, city="Quito", state="Pichincha")
                    if _h == 1 or _e == 1:
                        _types = []
                        if _n == 1:
                            _types.append("National Holiday")
                        if _r == 1:
                            _types.append("Regional Holiday")
                        if _l == 1:
                            _types.append("Local Holiday")
                        if _e == 1 and _h == 0:
                            _types.append("Event")
                        _hol_rows.append({
                            "Date": _ds,
                            "Type": ", ".join(_types),
                            "Locale": "National" if _n else ("Regional" if _r else ("Local" if _l else "Event")),
                        })
                if _hol_rows:
                    _hol_display = pd.DataFrame(_hol_rows)
                    st.dataframe(_hol_display, use_container_width=True, height=min(len(_hol_display) * 35 + 40, 300))
                else:
                    st.info(f"No holidays/events found between {base_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}.")

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Seasonal Demand Forecast (ML-Powered) ──
                st.markdown('<div class="section-title">Seasonal Demand & Stockout Forecast</div>', unsafe_allow_html=True)
                st.caption(f"ML model predicts demand for the next **{forecast_days} days** (including holidays). "
                           f"Compares seasonal demand vs normal demand. Alerts when stock drops below **{prestockout_pct}%**.")

                # Get items with stock > 0
                active_items = df_filtered[
                    (df_filtered["stock_status"] != "OUT OF STOCK") &
                    (df_filtered["avg_daily_sales"] > 0)
                ].copy()

                if len(active_items) == 0:
                    st.info("No active items to forecast.")
                else:
                    # Build a calendar of the forecast period with holiday info
                    forecast_dates = pd.date_range(start=base_date, end=end_date, freq="D")
                    date_info = []
                    for d in forecast_dates:
                        is_hol, is_nat, is_reg, is_loc, is_evt = get_holiday_flags(
                            d.strftime("%Y-%m-%d"), city="", state=""
                        )
                        date_info.append({
                            "date": d,
                            "is_holiday": is_hol,
                            "is_event": is_evt,
                        })
                    calendar_df = pd.DataFrame(date_info)
                    holiday_days = calendar_df["is_holiday"].sum()
                    event_days = calendar_df["is_event"].sum()
                    normal_days = len(calendar_df) - holiday_days - event_days

                    c_h1, c_h2, c_h3 = st.columns(3)
                    with c_h1:
                        st.markdown(_colored_card(f"{normal_days}", "Normal Days"), unsafe_allow_html=True)
                    with c_h2:
                        st.markdown(_colored_card(f"{holiday_days}", "Holidays", "warning"), unsafe_allow_html=True)
                    with c_h3:
                        st.markdown(_colored_card(f"{event_days}", "Events", "warning"), unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ── Load Seasonal Historical Data ──
                    _load_seasonal_data()

                    st.info("**Season-Aware Prediction:** Using historical traffic (transactions) and promotion patterns for the selected period.")
                    st.markdown("<br>", unsafe_allow_html=True)

                    # ── Run ML Predictions ──
                    progress_inv = st.progress(0, text="Analyzing seasonal demand...")
                    results = []

                    for idx, (_, row) in enumerate(active_items.iterrows()):
                        store = int(row["store_nbr"])
                        family = row["family"]
                        stock = float(row["current_stock"])
                        max_cap = float(row.get("max_capacity", stock * 2))
                        avg_sales = float(row["avg_daily_sales"])

                        total_pred = 0.0
                        holiday_pred = 0.0
                        normal_pred = 0.0
                        season_tx = 0.0
                        season_promo = 0.0

                        for d in forecast_dates:
                            # Get seasonal traffic & promos for this store+family+month
                            season_tx = get_seasonal_transactions(store, d)
                            season_promo = get_seasonal_promotions(family, d)

                            try:
                                feat = build_feature_row_with_store_lookup(
                                    store_nbr=store, family=family,
                                    date_str=d.strftime("%Y-%m-%d"),
                                    onpromotion=int(season_promo),
                                    transactions=season_tx,
                                )
                                pred = max(float(model.predict(feat)[0]), 0)
                            except Exception:
                                pred = avg_sales

                            total_pred += pred
                            d_info = calendar_df[calendar_df["date"] == d].iloc[0]
                            if d_info["is_holiday"] or d_info["is_event"]:
                                holiday_pred += pred
                            else:
                                normal_pred += pred

                        projected_stock = stock - total_pred
                        stock_pct_now = (stock / max_cap * 100) if max_cap > 0 else 100
                        stock_pct_after = (max(projected_stock, 0) / max_cap * 100) if max_cap > 0 else 0

                        # Daily averages for comparison
                        holiday_daily = (holiday_pred / max(holiday_days, 1)) if holiday_days > 0 else 0
                        normal_daily = (normal_pred / max(normal_days, 1)) if normal_days > 0 else avg_sales

                        # Surge ratio: how much higher is demand on holidays vs normal days
                        surge_ratio = holiday_daily / normal_daily if normal_daily > 0 else 1.0

                        # Pre-stockout alert: stock will drop below threshold %
                        prestockout_alert = stock_pct_after < prestockout_pct

                        results.append({
                            "store_nbr": store,
                            "family": family,
                            "city": row["city"],
                            "current_stock": int(stock),
                            "stock_pct_now": round(stock_pct_now, 1),
                            "normal_daily_demand": round(normal_daily, 1),
                            "holiday_daily_demand": round(holiday_daily, 1),
                            "avg_seasonal_traffic": round(season_tx, 0) if season_tx > 0 else 0,
                            "avg_seasonal_promos": round(season_promo, 1) if season_promo > 0 else 0,
                            "surge_ratio": round(surge_ratio, 2),
                            f"demand_surge_{int((surge_threshold-1)*100)}pct": "YES" if surge_ratio >= surge_threshold else "NO",
                            "total_predicted_sales": round(total_pred, 1),
                            "projected_stock": round(max(projected_stock, 0), 1),
                            "stock_pct_after": round(stock_pct_after, 1),
                            "will_stockout": "YES" if projected_stock <= 0 else "NO",
                            f"pre_stockout_alert (<{prestockout_pct}%)": "YES" if prestockout_alert else "NO",
                        })

                        if (idx + 1) % 50 == 0 or (idx + 1) == len(active_items):
                            progress_inv.progress(
                                min((idx + 1) / len(active_items), 1.0),
                                text=f"Analyzing... {idx+1:,} / {len(active_items):,}"
                            )

                    progress_inv.empty()

                    df_res = pd.DataFrame(results)

                    # ── Summary KPIs ──
                    surge_col = f"demand_surge_{int((surge_threshold-1)*100)}pct"
                    ps_col = f"pre_stockout_alert (<{prestockout_pct}%)"
                    n_stockout = len(df_res[df_res["will_stockout"] == "YES"])
                    n_surge = len(df_res[df_res[surge_col] == "YES"])
                    n_prestockout = len(df_res[df_res[ps_col] == "YES"])

                    k1, k2, k3, k4 = st.columns(4)
                    with k1:
                        st.markdown(_colored_card(f"{n_stockout:,}", "Will Stockout", "danger"), unsafe_allow_html=True)
                    with k2:
                        clr = "warning" if n_surge > 0 else "success"
                        st.markdown(_colored_card(f"{n_surge:,}", f"Seasonal Surge (>={int((surge_threshold-1)*100)}%)", clr), unsafe_allow_html=True)
                    with k3:
                        clr2 = "warning" if n_prestockout > 0 else "success"
                        st.markdown(_colored_card(f"{n_prestockout:,}", f"Pre-Stockout (<{prestockout_pct}%)", clr2), unsafe_allow_html=True)
                    with k4:
                        avg_surge = df_res["surge_ratio"].mean()
                        st.markdown(_colored_card(f"{avg_surge:.2f}x", "Avg Surge Ratio"), unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ── Seasonal Demand Surge Table ──
                    surged = df_res[df_res[surge_col] == "YES"].sort_values("surge_ratio", ascending=False)
                    if len(surged) > 0:
                        st.error(f"**{len(surged)} items** show seasonal demand surge of >={int((surge_threshold-1)*100)}% on holidays/events!")
                        st.caption("These products sell significantly more during the upcoming holiday period. Prioritize restocking.")

                        display_surge = surged[[
                            "store_nbr", "family", "city",
                            "current_stock", "stock_pct_now",
                            "normal_daily_demand", "holiday_daily_demand",
                            "avg_seasonal_traffic", "avg_seasonal_promos",
                            "surge_ratio", "projected_stock", "stock_pct_after",
                        ]].head(50).copy()
                        display_surge.columns = [
                            "Store", "Family", "City",
                            "Current Stock", "Stock %",
                            "Normal Day Avg", "Holiday Day Avg",
                            "Avg Traffic (store)", "Avg Promos (family)",
                            "Surge Ratio", "Projected Stock", "After %"
                        ]
                        st.dataframe(display_surge, use_container_width=True)
                    else:
                        st.success(f"No seasonal demand surges detected at the {int((surge_threshold-1)*100)}% threshold.")

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ── Pre-Stockout Alerts (25% threshold) ──
                    pre_alert = df_res[df_res[ps_col] == "YES"].sort_values("stock_pct_after")
                    if len(pre_alert) > 0:
                        st.warning(f"**{len(pre_alert)} items** will drop below **{prestockout_pct}%** capacity in {forecast_days} days. Restock recommended!")
                        st.caption(f"Even if they don't fully stock out, their stock will be critically low — leaving no buffer for unexpected demand.")

                        display_pre = pre_alert[[
                            "store_nbr", "family", "city",
                            "current_stock", "stock_pct_now",
                            "total_predicted_sales", "projected_stock",
                            "stock_pct_after", "will_stockout",
                        ]].head(50).copy()
                        display_pre.columns = [
                            "Store", "Family", "City",
                            "Current Stock", "Stock % Now",
                            f"{forecast_days}d Demand", f"Projected Stock",
                            "Stock % After", "Will Stockout"
                        ]
                        st.dataframe(display_pre, use_container_width=True)
                    else:
                        st.success(f"All items are expected to stay above {prestockout_pct}% capacity.")

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ── Full Stockout Predictions ──
                    stockout_items = df_res[df_res["will_stockout"] == "YES"].sort_values("projected_stock")
                    if len(stockout_items) > 0:
                        st.error(f"**{len(stockout_items)} items** will completely run out of stock!")
                        with st.expander("View Stockout Items", expanded=True):
                            st.dataframe(
                                stockout_items[["store_nbr", "family", "city",
                                               "current_stock", "total_predicted_sales",
                                               "projected_stock", "surge_ratio"]].head(50),
                                use_container_width=True,
                            )

                    # ── Download Full Report ──
                    with st.expander("Download Full Forecast Report"):
                        dl_cols = {
                            "store_nbr": "store_nbr",
                            "family": "family",
                            "city": "city",
                            "current_stock": "current_stock",
                            "stock_pct_now": "stock_pct_now_percent",
                            "normal_daily_demand": "normal_daily_demand_avg",
                            "holiday_daily_demand": "holiday_daily_demand_avg",
                            "surge_ratio": "surge_ratio",
                            surge_col: f"seasonal_surge_{int((surge_threshold-1)*100)}pct",
                            "total_predicted_sales": f"total_predicted_{forecast_days}d_sales",
                            "projected_stock": f"projected_stock_after_{forecast_days}d",
                            "stock_pct_after": f"stock_pct_after_{forecast_days}d",
                            "will_stockout": "will_stockout",
                            ps_col: f"pre_stockout_alert_below_{prestockout_pct}pct",
                        }
                        df_dl = df_res.rename(columns=dl_cols)
                        csv_dl = df_dl.to_csv(index=False)
                        st.download_button(
                            f"Download {forecast_days}-Day Seasonal Forecast",
                            csv_dl,
                            f"seasonal_inventory_forecast_{forecast_days}d.csv",
                            "text/csv",
                        )

            else:
                st.warning("Model not loaded. Cannot run seasonal forecasts.")

        with inv_tab3:
            # ── Chart 1: Stock Status Distribution ──
            st.markdown('<div class="section-title">Stock Status Distribution</div>', unsafe_allow_html=True)
            status_counts = df_filtered["stock_status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            color_map = {"SAFE": "#10b981", "LOW": "#f59e0b", "CRITICAL": "#ef4444", "OUT OF STOCK": "#dc2626"}
            status_counts["Color"] = status_counts["Status"].map(color_map)

            fig_pie = px.pie(status_counts, values="Count", names="Status",
                             color="Status", color_discrete_map=color_map,
                             hole=0.5)
            fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(fig_pie, use_container_width=True)

            # ── Chart 2: Top 15 Families by Low/Critical Stock ──
            with c2:
                problem_items = df_filtered[df_filtered["stock_status"].isin(["CRITICAL", "LOW", "OUT OF STOCK"])]
                if len(problem_items) > 0:
                    family_alerts = problem_items.groupby("family").size().sort_values(ascending=True).tail(15)
                    fig_bar = px.bar(
                        x=family_alerts.values, y=family_alerts.index,
                        orientation="h",
                        color=family_alerts.values,
                        color_continuous_scale="Reds",
                    )
                    fig_bar.update_layout(
                        xaxis_title="Number of Items at Risk",
                        yaxis_title="",
                        margin=dict(t=20, b=20, l=20, r=20),
                        showlegend=False,
                        height=400,
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.success("No items at risk!")

            # ── Chart 3: Stock by Store ──
            st.markdown('<div class="section-title">Stock Status by Store</div>', unsafe_allow_html=True)
            if selected_store == 0:
                store_status = df_filtered.groupby(["store_nbr", "stock_status"]).size().reset_index(name="count")
                fig_store = px.bar(
                    store_status, x="store_nbr", y="count", color="stock_status",
                    color_discrete_map=color_map,
                )
                fig_store.update_layout(
                    xaxis_title="Store Number", yaxis_title="Number of SKUs",
                    margin=dict(t=20, b=20, l=20, r=20),
                )
                st.plotly_chart(fig_store, use_container_width=True)

            # ── Chart 4: Days Until Stockout Distribution ──
            st.markdown('<div class="section-title">Days Until Stockout Distribution</div>', unsafe_allow_html=True)
            non_zero = df_filtered[(df_filtered["days_until_stockout"] > 0) & (df_filtered["days_until_stockout"] < 999)]
            if len(non_zero) > 0:
                fig_hist = px.histogram(
                    non_zero, x="days_until_stockout", nbins=30,
                    color_discrete_sequence=["#3b82f6"],
                )
                fig_hist.add_vline(x=7, line_dash="dash", line_color="#ef4444",
                                   annotation_text="7-day threshold")
                fig_hist.update_layout(
                    xaxis_title="Days Until Stockout", yaxis_title="Number of SKUs",
                    margin=dict(t=20, b=20, l=20, r=20),
                )
                st.plotly_chart(fig_hist, use_container_width=True)



# PAGE 8: LIVE MONITORING
# ══════════════════════════════════════════════
elif page == "📡 Live Monitoring":
    st.markdown("""
    <h1 style="font-size:2rem; font-weight:800; color:#1e293b; margin:0 0 4px 0;">
        📡 Live Monitoring
    </h1>
    <p style="color:#64748b;">Model health, API status, and system diagnostics</p>
    """, unsafe_allow_html=True)

    # Model Health
    st.markdown('<div class="section-title">Model Health</div>', unsafe_allow_html=True)
    mh1, mh2, mh3, mh4 = st.columns(4)
    with mh1:
        status = "Healthy" if model else "Missing"
        clr = "success" if model else "danger"
        st.markdown(_colored_card(status, "Model Status", clr), unsafe_allow_html=True)
    with mh2:
        st.markdown(_colored_card("20", "Features"), unsafe_allow_html=True)
    with mh3:
        has_metrics = os.path.exists(METRICS_PATH)
        st.markdown(_colored_card("✅" if has_metrics else "⚠️", "Metrics File", "success" if has_metrics else "warning"), unsafe_allow_html=True)
    with mh4:
        has_holidays = os.path.exists(os.path.join(os.path.dirname(__file__), "holidays_events.csv"))
        st.markdown(_colored_card("✅" if has_holidays else "⚠️", "Holidays Data", "success" if has_holidays else "warning"), unsafe_allow_html=True)

    # API Connection Test
    st.markdown('<div class="section-title">FastAPI Connection</div>', unsafe_allow_html=True)
    api_url = st.text_input("API URL", "http://localhost:8000", key="mon_api")

    if st.button("🔗 Test Connection", type="primary"):
        try:
            import urllib.request
            start = time.time()
            resp = urllib.request.urlopen(f"{api_url}/health", timeout=5)
            elapsed = (time.time() - start) * 1000
            data = json.loads(resp.read().decode())
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(_colored_card("✅ Connected", "Status", "success"), unsafe_allow_html=True)
            with c2:
                st.markdown(_colored_card(f"{elapsed:.0f} ms", "Response Time"), unsafe_allow_html=True)
            with c3:
                ml = data.get("model_loaded", False)
                st.markdown(_colored_card("✅" if ml else "❌", "API Model", "success" if ml else "danger"), unsafe_allow_html=True)

            # Test predict endpoint
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Testing /predict endpoint...**")
            import urllib.request
            test_body = json.dumps({
                "store_nbr": 1, "family": "GROCERY I", "date": "2017-08-15",
                "onpromotion": 0, "oil_price": 46.54, "transactions": 2100.0,
                "city": "Quito", "state": "Pichincha", "store_type": "D", "cluster": 1,
            }).encode()
            req = urllib.request.Request(f"{api_url}/predict", data=test_body,
                                        headers={"Content-Type": "application/json"})
            resp2 = urllib.request.urlopen(req, timeout=10)
            pred_data = json.loads(resp2.read().decode())
            st.success(f"API Prediction: **{pred_data['predicted_sales']:,.2f}** sales")

        except Exception as e:
            st.error(f"Connection failed: {e}")
            st.info("Make sure the API is running: `uvicorn app:app --host 0.0.0.0 --port 8000`")

    # File Status
    st.markdown('<div class="section-title">File System Status</div>', unsafe_allow_html=True)
    files_check = {
        "xgboost_sales_forecasting_model.pkl": "Trained Model",
        "model_metrics.json": "Model Metrics",
        "store-sales-time-series-forecasting/holidays_events.csv": "Holidays Data",
        "store-sales-time-series-forecasting/stores.csv": "Stores Data",
        "store-sales-time-series-forecasting/train.csv": "Sales Data (train.csv)",
        "app.py": "FastAPI Server",
        "feature_engineering.py": "Feature Engine",
    }
    for fname, label in files_check.items():
        fpath = os.path.join(os.path.dirname(__file__), fname)
        exists = os.path.exists(fpath)
        size = f"({os.path.getsize(fpath)/1024:.1f} KB)" if exists else "(not found)"
        st.markdown(f"{'✅' if exists else '❌'} **{label}** — `{fname}` {size}")

    # System Info
    st.markdown('<div class="section-title">System Information</div>', unsafe_allow_html=True)
    si1, si2, si3 = st.columns(3)
    with si1: st.info(f"**Python:** {sys.version.split()[0]}")
    with si2: st.info(f"**Pandas:** {pd.__version__}")
    with si3: st.info(f"**NumPy:** {np.__version__}")