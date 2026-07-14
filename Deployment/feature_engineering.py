"""
Shared Feature Engineering Module
===================================
Used by both app.py (API) and dashboard.py (Streamlit).
Loads and processes holidays_events.csv exactly like the training notebook.

This ensures predictions match the model's training data format.
"""

import os
import calendar as cal_mod
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ──────────────────────────────────────────────
# Path Configuration
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "store-sales-time-series-forecasting")
PARENT_DIR = os.path.dirname(BASE_DIR)

# Look for CSVs in multiple locations
_hol_candidates = [
    os.path.join(BASE_DIR, "holidays_events.csv"),
    os.path.join(DATASET_DIR, "holidays_events.csv"),
    os.path.join(PARENT_DIR, "holidays_events.csv"),
    os.path.join(PARENT_DIR, "store-sales-time-series-forecasting", "holidays_events.csv"),
    os.path.join(PARENT_DIR, "..", "store-sales-time-series-forecasting", "holidays_events.csv"),
    "/home/z/my-project/upload/grad_review/Graduation Project - Copy/store-sales-time-series-forecasting/holidays_events.csv",
]
HOLIDAYS_CSV = None
for _hc in _hol_candidates:
    if os.path.exists(_hc):
        HOLIDAYS_CSV = _hc
        break

_sto_candidates = [
    os.path.join(BASE_DIR, "stores.csv"),
    os.path.join(DATASET_DIR, "stores.csv"),
    os.path.join(PARENT_DIR, "stores.csv"),
    os.path.join(PARENT_DIR, "store-sales-time-series-forecasting", "stores.csv"),
    os.path.join(PARENT_DIR, "..", "store-sales-time-series-forecasting", "stores.csv"),
    "/home/z/my-project/upload/grad_review/Graduation Project - Copy/store-sales-time-series-forecasting/stores.csv",
]
STORES_CSV = None
for _sc in _sto_candidates:
    if os.path.exists(_sc):
        STORES_CSV = _sc
        break

# Expected feature columns (same as notebook Cell 121)
NUMERIC_FEATURES = [
    "store_nbr", "onpromotion", "dcoilwtico", "transactions",
    "year", "month", "day", "dayofweek", "weekofyear",
    "is_weekend", "is_holiday", "is_national_holiday",
    "is_regional_holiday", "is_local_holiday", "is_event", "cluster",
]

CATEGORICAL_FEATURES = ["family", "city", "state", "type"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ──────────────────────────────────────────────
# Holiday Data (loaded once, cached)
# ──────────────────────────────────────────────

_holiday_lookup = None  # Will be populated on first call
_recurring_patterns = None  # MM-DD -> holiday info for year-agnostic matching


def _get_black_friday(year):
    """Get Black Friday (4th Friday of November) for any year."""
    c = cal_mod.Calendar(firstweekday=cal_mod.MONDAY)
    fridays = [d for d in c.itermonthdates(year, 11) if d.weekday() == 4 and d.month == 11]
    if len(fridays) >= 4:
        return fridays[3]
    return None


def _get_cyber_monday(year):
    """Get Cyber Monday (Monday after Black Friday)."""
    bf = _get_black_friday(year)
    if bf:
        return bf + timedelta(days=3)
    return None


def _load_holiday_data():
    """
    Load and process holidays_events.csv exactly like the notebook (Cell 65).
    Returns a dict with pre-computed holiday info keyed by date string.
    """
    global _holiday_lookup

    if _holiday_lookup is not None:
        return _holiday_lookup

    if HOLIDAYS_CSV is None or not os.path.exists(HOLIDAYS_CSV):
        _holiday_lookup = {"_loaded": False}
        return _holiday_lookup

    try:
        holidays_events = pd.read_csv(HOLIDAYS_CSV, parse_dates=["date"])
    except Exception:
        _holiday_lookup = {"_loaded": False}
        return _holiday_lookup

    # === Same logic as notebook Cell 65 ===

    hol = holidays_events.copy()

    # Keep only non-transferred rows
    hol = hol[hol["transferred"] == False].copy()

    # Split holidays and events
    hol_holiday = hol[hol["type"] == "Holiday"].copy()
    hol_event = hol[hol["type"] == "Event"].copy()

    # National holidays
    national = hol_holiday[hol_holiday["locale"] == "National"][["date"]].drop_duplicates()

    # Regional holidays (mapped to state)
    regional = hol_holiday[hol_holiday["locale"] == "Regional"][["date", "locale_name"]].drop_duplicates()
    regional = regional.rename(columns={"locale_name": "state"})

    # Local holidays (mapped to city)
    local = hol_holiday[hol_holiday["locale"] == "Local"][["date", "locale_name"]].drop_duplicates()
    local = local.rename(columns={"locale_name": "city"})

    # Events
    events = hol_event[["date"]].drop_duplicates()

    # Build lookup: date_str -> {is_national, is_regional, is_event, regional_states, local_cities}
    lookup = {"_loaded": True}

    # National holidays
    for _, row in national.iterrows():
        ds = row["date"].strftime("%Y-%m-%d")
        if ds not in lookup:
            lookup[ds] = {"is_national": 1, "is_regional": 0, "is_local": 0, "is_event": 0,
                          "regional_states": set(), "local_cities": set()}
        else:
            lookup[ds]["is_national"] = 1

    # Regional holidays
    for _, row in regional.iterrows():
        ds = row["date"].strftime("%Y-%m-%d")
        state = row["state"]
        if ds not in lookup:
            lookup[ds] = {"is_national": 0, "is_regional": 0, "is_local": 0, "is_event": 0,
                          "regional_states": set(), "local_cities": set()}
        lookup[ds]["is_regional"] = 1
        lookup[ds]["regional_states"].add(state)

    # Local holidays
    for _, row in local.iterrows():
        ds = row["date"].strftime("%Y-%m-%d")
        city = row["city"]
        if ds not in lookup:
            lookup[ds] = {"is_national": 0, "is_regional": 0, "is_local": 0, "is_event": 0,
                          "regional_states": set(), "local_cities": set()}
        lookup[ds]["is_local"] = 1
        lookup[ds]["local_cities"].add(city)

    # Events
    for _, row in events.iterrows():
        ds = row["date"].strftime("%Y-%m-%d")
        if ds not in lookup:
            lookup[ds] = {"is_national": 0, "is_regional": 0, "is_local": 0, "is_event": 0,
                          "regional_states": set(), "local_cities": set()}
        lookup[ds]["is_event"] = 1

    # Convert sets to lists for JSON compatibility
    for key in lookup:
        if isinstance(lookup[key], dict):
            lookup[key]["regional_states"] = list(lookup[key]["regional_states"])
            lookup[key]["local_cities"] = list(lookup[key]["local_cities"])

    _holiday_lookup = lookup

    # === Build recurring holiday patterns (MM-DD based) ===
    # This allows holiday detection for ANY year, not just 2012-2017.
    # We scan all historical years and aggregate by (month, day).
    # Only truly recurring holidays/events are kept (not one-time events
    # like World Cup 2014 or Terremoto Manabi 2016).
    global _recurring_patterns
    _recurring_patterns = {}

    # --- National recurring holidays ---
    # These appear as type==Holiday, locale==National on the same date
    # in multiple years. Collect unique (month,day) pairs.
    national_dates = hol_holiday[hol_holiday["locale"] == "National"][["date", "description"]].copy()
    national_dates["md"] = national_dates["date"].dt.strftime("%m-%d")

    # One-time events to EXCLUDE from recurring patterns
    _one_time_descriptions = [
        "Terremoto", "Mundial", "Inauguracion", "Copa America",
        "Recupero",  # work-day recoveries are year-specific
    ]

    for md_key, grp in national_dates.groupby("md"):
        descs = grp["description"].unique()
        # Skip one-time events
        is_one_time = any(any(ot in str(d) for ot in _one_time_descriptions) for d in descs)
        if not is_one_time and len(grp) >= 1:
            _recurring_patterns[md_key] = {
                "is_national": 1, "is_regional": 0, "is_local": 0, "is_event": 0,
                "regional_states": [], "local_cities": [],
                "description": descs[0] if len(descs) > 0 else ""
            }

    # --- Regional recurring holidays ---
    regional_dates = hol_holiday[hol_holiday["locale"] == "Regional"][["date", "locale_name"]].copy()
    regional_dates = regional_dates.rename(columns={"locale_name": "state"})
    regional_dates["md"] = regional_dates["date"].dt.strftime("%m-%d")

    for md_key, grp in regional_dates.groupby("md"):
        states = list(grp["state"].unique())
        if md_key not in _recurring_patterns:
            _recurring_patterns[md_key] = {
                "is_national": 0, "is_regional": 0, "is_local": 0, "is_event": 0,
                "regional_states": [], "local_cities": [],
                "description": ""
            }
        _recurring_patterns[md_key]["is_regional"] = 1
        for s in states:
            if s not in _recurring_patterns[md_key]["regional_states"]:
                _recurring_patterns[md_key]["regional_states"].append(s)

    # --- Local recurring holidays ---
    local_dates = hol_holiday[hol_holiday["locale"] == "Local"][["date", "locale_name"]].copy()
    local_dates = local_dates.rename(columns={"locale_name": "city"})
    local_dates["md"] = local_dates["date"].dt.strftime("%m-%d")

    for md_key, grp in local_dates.groupby("md"):
        cities = list(grp["city"].unique())
        if md_key not in _recurring_patterns:
            _recurring_patterns[md_key] = {
                "is_national": 0, "is_regional": 0, "is_local": 0, "is_event": 0,
                "regional_states": [], "local_cities": [],
                "description": ""
            }
        _recurring_patterns[md_key]["is_local"] = 1
        for c in cities:
            if c not in _recurring_patterns[md_key]["local_cities"]:
                _recurring_patterns[md_key]["local_cities"].append(c)

    # --- Recurring Events: Black Friday & Cyber Monday ---
    # These are commercial events that recur every year.
    # Extract from historical data to validate, then compute dynamically.
    event_df = hol_event[["date", "description"]].copy()
    event_df["md"] = event_df["date"].dt.strftime("%m-%d")

    # Find Black Friday and Cyber Monday entries
    bf_entries = event_df[event_df["description"].str.contains("Black Friday", na=False)]
    cm_entries = event_df[event_df["description"].str.contains("Cyber Monday", na=False)]

    if len(bf_entries) > 0:
        # Black Friday is the 4th Friday of November - compute dynamically
        # We'll handle it specially in get_holiday_flags via _get_black_friday()
        pass

    if len(cm_entries) > 0:
        # Cyber Monday is the Monday after Black Friday
        # Handled via _get_cyber_monday()
        pass

    return lookup


def get_holiday_flags(date_str: str, city: str = "", state: str = ""):
    """
    Get holiday flags for a specific date, city, and state.
    Matches the logic from notebook Cell 65 exactly for dates in 2012-2017.
    For dates outside that range, uses recurring holiday patterns derived
    from the historical data so that e.g. Dec 25 is always Navidad.
    """
    lookup = _load_holiday_data()

    if not lookup.get("_loaded", False):
        return (0, 0, 0, 0, 0)

    # 1) Try exact date match (covers 2012-2017)
    info = lookup.get(date_str)
    if info is not None:
        is_national = info["is_national"]
        is_regional = 1 if state in info["regional_states"] else 0
        is_local = 1 if city in info["local_cities"] else 0
        is_event = info["is_event"]
        is_holiday = 1 if (is_national == 1 or is_regional == 1 or is_local == 1) else 0
        return (is_holiday, is_national, is_regional, is_local, is_event)

    # 2) Date not in historical data — use recurring patterns
    # Parse the date
    try:
        dt = pd.Timestamp(date_str)
    except Exception:
        return (0, 0, 0, 0, 0)

    md_key = dt.strftime("%m-%d")
    year = dt.year

    # Check if it's Black Friday or Cyber Monday (dynamic, year-specific)
    is_bf = False
    is_cm = False
    if _recurring_patterns is not None:
        bf = _get_black_friday(year)
        cm = _get_cyber_monday(year)
        dt_date = dt.date() if hasattr(dt, 'date') else dt
        bf_date = bf if bf is None else (bf.date() if hasattr(bf, 'date') else bf)
        cm_date = cm if cm is None else (cm.date() if hasattr(cm, 'date') else cm)
        if bf_date and dt_date == bf_date:
            is_bf = True
        if cm_date and dt_date == cm_date:
            is_cm = True

    # Look up recurring pattern for this month-day
    pattern = _recurring_patterns.get(md_key) if _recurring_patterns else None

    if pattern is None and not is_bf and not is_cm:
        # Not a recurring holiday date
        return (0, 0, 0, 0, 0)

    if pattern is not None:
        is_national = pattern["is_national"]
        is_regional = 1 if state in pattern["regional_states"] else 0
        is_local = 1 if city in pattern["local_cities"] else 0
        is_event = pattern["is_event"]
    else:
        is_national = 0
        is_regional = 0
        is_local = 0
        is_event = 0

    # Mark Black Friday / Cyber Monday as events
    if is_bf or is_cm:
        is_event = 1

    is_holiday = 1 if (is_national == 1 or is_regional == 1 or is_local == 1) else 0

    return (is_holiday, is_national, is_regional, is_local, is_event)


# ──────────────────────────────────────────────
# Feature Builder
# ──────────────────────────────────────────────

def build_feature_row(
    store_nbr: int,
    family: str,
    date_str: str,
    onpromotion: int = 0,
    oil_price: float = 0.0,
    transactions: float = 0.0,
    city: str = "Quito",
    state: str = "Pichincha",
    store_type: str = "D",
    cluster: int = 1,
) -> pd.DataFrame:
    """
    Build a single feature row matching the model's expected input.
    """
    dt = pd.Timestamp(date_str)

    is_holiday, is_national, is_regional, is_local, is_event = get_holiday_flags(
        date_str, city=city, state=state
    )

    row = {
        "store_nbr": store_nbr,
        "onpromotion": onpromotion,
        "dcoilwtico": oil_price,
        "transactions": transactions,
        "year": dt.year,
        "month": dt.month,
        "day": dt.day,
        "dayofweek": dt.dayofweek,
        "weekofyear": int(dt.isocalendar()[1]),
        "is_weekend": 1 if dt.dayofweek in [5, 6] else 0,
        "is_holiday": is_holiday,
        "is_national_holiday": is_national,
        "is_regional_holiday": is_regional,
        "is_local_holiday": is_local,
        "is_event": is_event,
        "cluster": cluster,
        "family": family,
        "city": city,
        "state": state,
        "type": store_type,
    }

    return pd.DataFrame([row])[ALL_FEATURES]


# ──────────────────────────────────────────────
# Store Metadata Helper
# ──────────────────────────────────────────────

_store_lookup = None


def get_store_info(store_nbr: int) -> dict:
    """
    Look up store metadata (city, state, store_type, cluster) from stores.csv.
    Falls back to defaults if file not available.
    """
    global _store_lookup

    if _store_lookup is None:
        _store_lookup = {}
        if os.path.exists(STORES_CSV):
            try:
                stores = pd.read_csv(STORES_CSV)
                for _, row in stores.iterrows():
                    _store_lookup[int(row["store_nbr"])] = {
                        "city": row["city"],
                        "state": row["state"],
                        "store_type": row["type"],     # CSV column "type" -> key "store_type"
                        "cluster": int(row["cluster"]),
                    }
            except Exception:
                pass

    return _store_lookup.get(store_nbr, {
        "city": "Quito",
        "state": "Pichincha",
        "store_type": "D",     # key must be "store_type" not "type"
        "cluster": 1,
    })


def build_feature_row_with_store_lookup(
    store_nbr: int,
    family: str,
    date_str: str,
    onpromotion: int = 0,
    oil_price: float = 0.0,
    transactions: float = 0.0,
) -> pd.DataFrame:
    """
    Build feature row using stores.csv to auto-fill city, state, type, cluster.
    """
    store_info = get_store_info(store_nbr)

    return build_feature_row(
        store_nbr=store_nbr,
        family=family,
        date_str=date_str,
        onpromotion=onpromotion,
        oil_price=oil_price,
        transactions=transactions,
        city=store_info["city"],
        state=store_info["state"],
        store_type=store_info["store_type"],    # key is "store_type"
        cluster=store_info["cluster"],
    )