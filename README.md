# Store Sales Forecasting 

End-to-end machine learning project for forecasting retail store sales using **XGBoost** and **Random Forest**, with a full **MLOps pipeline** covering deployment, monitoring, and automated retraining.

Built on the [Kaggle Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition dataset (Corporacion Favorita, Ecuador).

---

## Project Highlights

- **4 models compared**: XGBoost, Tuned Random Forest, Linear Regression, SARIMA
- **Best model**: XGBoost Regressor — **R2 = 0.94**, RMSE = 331.6
- **Interactive Streamlit dashboard** with batch prediction, trend analysis, and model comparison
- **REST API** (FastAPI) with single, batch, and date-range prediction endpoints
- **MLOps**: automated monitoring, data drift detection (KS Test), and scheduled retraining
- **Experiment tracking** with MLflow

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit Dashboard                │
│   (Single/Batch Prediction, Analytics, Monitoring)    │
└─────────────┬───────────────────────────┬───────────┘
              │                           │
      ┌───────▼───────┐          ┌───────▼───────┐
      │   FastAPI      │          │  MLflow UI    │
      │   (REST API)   │          │  (Experiments)│
      └───────┬───────┘          └───────────────┘
              │
      ┌───────▼───────┐     ┌───────────────┐
      │  XGBoost Model │     │  monitoring.py │
      │  (.pkl)        │◄────│  (Drift + Eval)│
      └───────┬───────┘     └───────┬───────┘
              │                     │
      ┌───────▼───────┐     ┌───────▼───────┐
      │   retrain.py   │────►│  New Model     │
      │  (Auto-retrain)│     │  (if better)   │
      └───────────────┘     └───────────────┘
```

---

## Project Structure

```
sales-forecasting/
├── notebook/
│   ├── grad_project_code.ipynb   
├── deployment/
│   ├── app.py                    ← FastAPI
│   ├── dashboard.py              ← Streamlit
│   ├── feature_engineering.py
│   ├── monitoring.py
│   ├── retrain.py
│   ├── requirements.txt
│   ├── xgboost_sales_forecasting_model.pkl
├── docs/
│   ├── Presentation.pptx
│   └── Final_Report.docx
└── README.md
```

---

## Model Performance

| Model | RMSE | MAE | R2 Score |
|-------|------|-----|----------|
| **XGBoost Regressor** | **331.61** | **96.36** | **0.9407** |
| Tuned Random Forest | 411.50 | 144.15 | 0.9086 |
| Linear Regression | 840.96 | 318.22 | 0.6184 |
| SARIMA | 131,228.40 | 86,067.66 | 0.5056 |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Ahmed-Omara1/-Sales-Forecasting-and-Demand-Prediction.git
cd store-sales-forecasting
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open at `http://localhost:8501`.

### 4. Run the API (optional)

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at `http://localhost:8000/docs`.

---

## Features

### Streamlit Dashboard
- **Home**: KPIs, model performance overview, quick forecast
- **Single Prediction**: Predict sales for one store-product-date combination
- **Batch Prediction**: Upload CSV (supports Kaggle test.csv format) with progress bar and downloadable results
- **Forecast Analysis**: Trend charts, seasonal patterns, date-range forecasting
- **Business Analytics**: Sales by store, family, and region
- **Model Comparison**: Side-by-side model metrics and performance charts
- **API Monitoring**: Live API health check and endpoint testing

### FastAPI Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | API health check |
| `GET` | `/model/info` | Model metadata & expected features |
| `POST` | `/predict` | Single prediction |
| `POST` | `/predict/batch` | Batch prediction (up to 1000 items) |
| `GET` | `/predict/range` | Date range prediction |
| `GET` | `/predict/range/auto` | Date range with auto store lookup |

### MLOps Pipeline
- **Monitoring**: MAE, RMSE, R2 evaluation with JSON logging
- **Data Drift Detection**: Kolmogorov-Smirnov test (p-value < 0.05 threshold)
- **Retraining Triggers**:
  - Scheduled: every 90 days
  - Performance degradation: RMSE increases by > 10%
  - New data available: 30+ new days of data
- **Safety**: new model deployed only if it outperforms the current one (with backup)

---

## Feature Engineering

Features used for prediction (20 total):

**Numeric (16):** `store_nbr`, `onpromotion`, `dcoilwtico`, `transactions`, `year`, `month`, `day`, `dayofweek`, `weekofyear`, `is_weekend`, `is_holiday`, `is_national_holiday`, `is_regional_holiday`, `is_local_holiday`, `is_event`, `cluster`

**Categorical (4):** `family`, `city`, `state`, `type`

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python |
| ML Models | XGBoost, Random Forest, Linear Regression, SARIMA |
| Dashboard | Streamlit + Plotly |
| API | FastAPI + Uvicorn + Pydantic |
| Experiment Tracking | MLflow |
| Data | Pandas, NumPy, Scipy |
| Hyperparameter Tuning | RandomizedSearchCV + TimeSeriesSplit |

---

## Dataset

[Kaggle Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

> Forecast sales for thousands of product families sold at Favorita stores located in Ecuador.
