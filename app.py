"""
FastAPI Application for Sales Forecasting API
================================================
Provides RESTful endpoints for real-time and batch sales predictions
using the deployed XGBoost model.

Run: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging

from feature_engineering import (
    build_feature_row, build_feature_row_with_store_lookup,
    ALL_FEATURES, get_store_info,
)

# ──────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# FastAPI Instance
# ──────────────────────────────────────────────
app = FastAPI(
    title="Sales Forecasting API",
    description="Real-time and batch sales forecasting using XGBoost",
    version="1.0.0",
)

# Enable CORS for dashboard communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Load Model
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgboost_sales_forecasting_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully.")
except FileNotFoundError:
    logger.warning(f"Model file not found at {MODEL_PATH}. Running without model.")
    model = None

# ──────────────────────────────────────────────
# Pydantic Schemas
# ──────────────────────────────────────────────

class ForecastRequest(BaseModel):
    """Single forecast request."""
    store_nbr: int = Field(..., ge=1, le=54, description="Store number (1-54)")
    family: str = Field(..., description="Product family (e.g., 'GROCERY I', 'BEVERAGES')")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    onpromotion: int = Field(default=0, ge=0, description="Number of products on promotion")
    oil_price: Optional[float] = Field(default=None, description="Oil price (dcoilwtico)")
    transactions: Optional[float] = Field(default=None, description="Store transactions")
    city: str = Field(default="Quito", description="Store city")
    state: str = Field(default="Pichincha", description="Store state")
    store_type: str = Field(default="D", description="Store type (A-E)")
    cluster: int = Field(default=1, ge=1, le=17, description="Store cluster (1-17)")

    class Config:
        json_schema_extra = {
            "example": {
                "store_nbr": 1,
                "family": "GROCERY I",
                "date": "2017-08-15",
                "onpromotion": 5,
                "oil_price": 46.54,
                "transactions": 2100.0,
                "city": "Quito",
                "state": "Pichincha",
                "store_type": "D",
                "cluster": 1,
            }
        }


class BatchForecastRequest(BaseModel):
    """Batch forecast request — multiple items at once."""
    items: List[ForecastRequest] = Field(..., min_length=1, max_length=1000)


class ForecastResponse(BaseModel):
    """Single forecast response."""
    store_nbr: int
    family: str
    date: str
    predicted_sales: float
    confidence: str = "high"


class BatchForecastResponse(BaseModel):
    """Batch forecast response."""
    forecasts: List[ForecastResponse]
    total_items: int
    model_used: str = "XGBoost"


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    """Model metadata response."""
    model_type: str
    features: List[str]
    target: str
    version: str


# ──────────────────────────────────────────────
# Feature Engineering (using shared module)
# ──────────────────────────────────────────────

def build_features(request: ForecastRequest) -> pd.DataFrame:
    """
    Convert a ForecastRequest into the feature DataFrame the model expects.
    Uses the shared feature_engineering module which loads real holidays_events.csv.
    """
    return build_feature_row(
        store_nbr=request.store_nbr,
        family=request.family,
        date_str=request.date,
        onpromotion=request.onpromotion,
        oil_price=request.oil_price if request.oil_price is not None else 0.0,
        transactions=request.transactions if request.transactions is not None else 0.0,
        city=request.city,
        state=request.state,
        store_type=request.store_type,
        cluster=request.cluster,
    )


# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
async def root():
    """Root health check."""
    return HealthResponse(
        status="healthy",
        model_loaded=(model is not None),
        version="1.0.0",
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=(model is not None),
        version="1.0.0",
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Return model metadata and expected features."""
    return ModelInfoResponse(
        model_type="XGBoost Regressor (Pipeline with ColumnTransformer)",
        features=ALL_FEATURES,
        target="sales",
        version="1.0.0",
    )


@app.post("/predict", response_model=ForecastResponse)
async def predict_single(request: ForecastRequest):
    """
    Predict sales for a single store-product-date combination.

    Example request body:
    ```json
    {
        "store_nbr": 1,
        "family": "GROCERY I",
        "date": "2017-08-15",
        "onpromotion": 5,
        "oil_price": 46.54,
        "transactions": 2100.0,
        "city": "Quito",
        "state": "Pichincha",
        "store_type": "D",
        "cluster": 1
    }
    ```
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Cannot make predictions.")

    try:
        features_df = build_features(request)
        prediction = model.predict(features_df)[0]
        prediction = max(prediction, 0)  # Sales cannot be negative

        logger.info(
            f"Prediction: store={request.store_nbr}, "
            f"family={request.family}, date={request.date} -> {prediction:.2f}"
        )

        return ForecastResponse(
            store_nbr=request.store_nbr,
            family=request.family,
            date=request.date,
            predicted_sales=round(float(prediction), 4),
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchForecastResponse)
async def predict_batch(request: BatchForecastRequest):
    """
    Predict sales for multiple store-product-date combinations at once.

    Useful for batch forecasting — send up to 1000 items in a single request.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Cannot make predictions.")

    try:
        all_features = pd.concat(
            [build_features(item) for item in request.items],
            ignore_index=True,
        )

        predictions = model.predict(all_features)
        predictions = np.maximum(predictions, 0)

        forecasts = []
        for i, item in enumerate(request.items):
            forecasts.append(
                ForecastResponse(
                    store_nbr=item.store_nbr,
                    family=item.family,
                    date=item.date,
                    predicted_sales=round(float(predictions[i]), 4),
                )
            )

        logger.info(f"Batch prediction: {len(forecasts)} items processed.")

        return BatchForecastResponse(
            forecasts=forecasts,
            total_items=len(forecasts),
            model_used="XGBoost",
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/predict/range")
async def predict_date_range(
    store_nbr: int = Query(..., ge=1, le=54),
    family: str = Query(..., description="Product family"),
    start_date: str = Query(..., description="Start date YYYY-MM-DD"),
    end_date: str = Query(..., description="End date YYYY-MM-DD"),
    onpromotion: int = Query(default=0, ge=0),
    city: str = Query(default="Quito"),
    state: str = Query(default="Pichincha"),
    store_type: str = Query(default="D"),
    cluster: int = Query(default=1, ge=1, le=17),
):
    """
    Predict sales for a date range (e.g., next 30 days).

    Example: /predict/range?store_nbr=1&family=GROCERY%20I&start_date=2017-08-01&end_date=2017-08-31
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        if start > end:
            raise HTTPException(status_code=400, detail="start_date must be before end_date.")

        date_range = pd.date_range(start=start, end=end, freq="D")
        requests_list = []

        for dt in date_range:
            req = ForecastRequest(
                store_nbr=store_nbr,
                family=family,
                date=dt.strftime("%Y-%m-%d"),
                onpromotion=onpromotion,
                city=city,
                state=state,
                store_type=store_type,
                cluster=cluster,
            )
            requests_list.append(req)

        batch_req = BatchForecastRequest(items=requests_list)
        result = await predict_batch(batch_req)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Date range prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/range/auto")
async def predict_range_auto(
    store_nbr: int = Query(..., ge=1, le=54),
    family: str = Query(..., description="Product family"),
    start_date: str = Query(..., description="Start date YYYY-MM-DD"),
    end_date: str = Query(..., description="End date YYYY-MM-DD"),
    onpromotion: int = Query(default=0, ge=0),
):
    """
    Predict sales for a date range using stores.csv to auto-fill city/state/type/cluster.
    You only need store_nbr, family, and dates.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        if start > end:
            raise HTTPException(status_code=400, detail="start_date must be before end_date.")

        date_range = pd.date_range(start=start, end=end, freq="D")
        all_features = pd.concat(
            [build_feature_row_with_store_lookup(store_nbr, family, dt.strftime("%Y-%m-%d"), onpromotion=onpromotion)
             for dt in date_range],
            ignore_index=True,
        )
        predictions = np.maximum(model.predict(all_features), 0)

        forecasts = [ForecastResponse(
            store_nbr=store_nbr, family=family,
            date=dt.strftime("%Y-%m-%d"),
            predicted_sales=round(float(predictions[i]), 4),
        ) for i, dt in enumerate(date_range)]

        return BatchForecastResponse(forecasts=forecasts, total_items=len(forecasts), model_used="XGBoost")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Auto range prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



# ──────────────────────────────────────────────
# Inventory Endpoints
# ──────────────────────────────────────────────
INVENTORY_CSV = os.path.join(os.path.dirname(__file__), "inventory.csv")

def _load_inventory_df():
    """Load inventory.csv or return None."""
    if os.path.exists(INVENTORY_CSV):
        return pd.read_csv(INVENTORY_CSV)
    return None


class InventoryItem(BaseModel):
    store_nbr: int
    family: str
    city: str
    current_stock: int
    reorder_level: int
    days_until_stockout: float
    avg_daily_sales: float
    stock_status: str


class InventoryResponse(BaseModel):
    items: List[InventoryItem]
    total: int
    filters: dict = {}


@app.get("/inventory/status", response_model=InventoryResponse)
async def inventory_status(
    store_nbr: Optional[int] = Query(None, ge=1, le=54),
    status: Optional[str] = Query(None, description="SAFE, LOW, CRITICAL, or OUT OF STOCK"),
):
    """
    Get current inventory status for all or filtered items.

    Examples:
    - /inventory/status
    - /inventory/status?store_nbr=1
    - /inventory/status?status=CRITICAL
    - /inventory/status?store_nbr=1&status=LOW
    """
    df = _load_inventory_df()
    if df is None:
        raise HTTPException(status_code=404, detail="inventory.csv not found. Run inventory_simulation.py first.")

    if store_nbr is not None:
        df = df[df["store_nbr"] == store_nbr]
    if status is not None:
        df = df[df["stock_status"] == status.upper()]

    items = [
        InventoryItem(
            store_nbr=int(r["store_nbr"]),
            family=str(r["family"]),
            city=str(r["city"]),
            current_stock=int(r["current_stock"]),
            reorder_level=int(r["reorder_level"]),
            days_until_stockout=float(r["days_until_stockout"]),
            avg_daily_sales=float(r["avg_daily_sales"]),
            stock_status=str(r["stock_status"]),
        )
        for _, r in df.iterrows()
    ]

    return InventoryResponse(
        items=items,
        total=len(items),
        filters={"store_nbr": store_nbr, "status": status},
    )


class StockoutAlert(BaseModel):
    store_nbr: int
    family: str
    city: str
    current_stock: int
    predicted_7d_sales: float
    projected_stock_7d: float
    will_stockout: bool
    days_until_stockout: float


class StockoutAlertsResponse(BaseModel):
    alerts: List[StockoutAlert]
    total_alerts: int
    total_items_checked: int


@app.get("/inventory/stockout-alerts", response_model=StockoutAlertsResponse)
async def stockout_alerts(
    store_nbr: Optional[int] = Query(None, ge=1, le=54),
):
    """
    Predict which items will run out of stock in the next 7 days.
    Uses the ML model to forecast demand and compares with current stock.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    df = _load_inventory_df()
    if df is None:
        raise HTTPException(status_code=404, detail="inventory.csv not found.")

    if store_nbr is not None:
        df = df[df["store_nbr"] == store_nbr]

    # Only check items that still have stock
    df = df[df["current_stock"] > 0]

    alerts = []
    from datetime import timedelta as td
    base_date = pd.Timestamp("2017-08-16")

    for _, row in df.iterrows():
        store = int(row["store_nbr"])
        family = str(row["family"])
        stock = int(row["current_stock"])

        total_pred_sales = 0.0
        for day_offset in range(7):
            d = base_date + td(days=day_offset)
            try:
                feat = build_feature_row_with_store_lookup(
                    store_nbr=store, family=family,
                    date_str=d.strftime("%Y-%m-%d"),
                    onpromotion=0,
                )
                pred = max(float(model.predict(feat)[0]), 0)
            except Exception:
                pred = float(row["avg_daily_sales"])
            total_pred_sales += pred

        projected = stock - total_pred_sales
        days_to_zero = stock / row["avg_daily_sales"] if row["avg_daily_sales"] > 0 else 999

        alerts.append(StockoutAlert(
            store_nbr=store,
            family=family,
            city=str(row["city"]),
            current_stock=stock,
            predicted_7d_sales=round(total_pred_sales, 2),
            projected_stock_7d=round(max(projected, 0), 2),
            will_stockout=(projected <= 0),
            days_until_stockout=round(days_to_zero, 1),
        ))

    return StockoutAlertsResponse(
        alerts=alerts,
        total_alerts=len([a for a in alerts if a.will_stockout]),
        total_items_checked=len(alerts),
    )


@app.get("/inventory/demand-alerts")
async def demand_alerts(
    store_nbr: Optional[int] = Query(None, ge=1, le=54),
    min_ratio: float = Query(1.0, description="Minimum demand/stock ratio to flag"),
):
    """
    Get products with high demand relative to their current stock.
    demand_ratio = (avg_daily_sales * 7) / current_stock
    Ratio > 1.0 means 7-day demand exceeds current stock.
    """
    df = _load_inventory_df()
    if df is None:
        raise HTTPException(status_code=404, detail="inventory.csv not found.")

    if store_nbr is not None:
        df = df[df["store_nbr"] == store_nbr]

    # Filter: has sales and not already out of stock
    df = df[(df["avg_daily_sales"] > 0) & (df["current_stock"] > 0)].copy()

    # Calculate demand ratio
    df["demand_ratio"] = (df["avg_daily_sales"] * 7) / df["current_stock"]
    high_demand = df[df["demand_ratio"] >= min_ratio].sort_values("demand_ratio", ascending=False)

    result = high_demand[[
        "store_nbr", "family", "city", "current_stock",
        "avg_daily_sales", "demand_ratio", "stock_status",
    ]].to_dict(orient="records")

    return {
        "high_demand_items": result,
        "total": len(result),
        "min_ratio_threshold": min_ratio,
    }


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)