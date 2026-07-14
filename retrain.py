"""
Model Retraining Strategy Script
=================================
Automated retraining pipeline for the Sales Forecasting model.

This addresses Milestone 4 Task 4:
  - Develop a strategy for periodically retraining the model
  - Based on new data, seasonal patterns, or changing external factors
  - Validate performance before deployment
  - Version management with MLflow
"""

import os
import sys
import json
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import datetime

import mlflow
import mlflow.xgboost

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgboost_sales_forecasting_model.pkl")
RETRAINING_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "retraining_config.json")

# Retraining triggers
MIN_NEW_DATA_DAYS = 30        # Minimum new data required for retraining
PERFORMANCE_DEGRADATION = 0.10  # Retrain if RMSE increases by 10%
SCHEDULED_INTERVAL_DAYS = 90    # Retrain every 90 days regardless

# Model hyperparameters (same as original)
XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}

# Expected features
NUMERIC_FEATURES = [
    "store_nbr", "onpromotion", "dcoilwtico", "transactions",
    "year", "month", "day", "dayofweek", "weekofyear",
    "is_weekend", "is_holiday", "is_national_holiday",
    "is_regional_holiday", "is_local_holiday", "is_event", "cluster",
]

CATEGORICAL_FEATURES = ["family", "city", "state", "type"]

TARGET = "sales"


# ──────────────────────────────────────────────
# Retraining Configuration Manager
# ──────────────────────────────────────────────

def load_retraining_config() -> dict:
    """Load retraining configuration or return defaults."""
    if os.path.exists(RETRAINING_CONFIG_PATH):
        with open(RETRAINING_CONFIG_PATH, "r") as f:
            return json.load(f)
    return {
        "last_retrained": None,
        "last_rmse": None,
        "last_r2": None,
        "retraining_count": 0,
        "retraining_history": [],
        "model_versions": [],
    }


def save_retraining_config(config: dict):
    """Save retraining configuration."""
    with open(RETRAINING_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2, default=str)
    logger.info("Retraining config saved.")


# ──────────────────────────────────────────────
# Retraining Trigger Check
# ──────────────────────────────────────────────

def should_retrain(config: dict, new_rmse: float = None) -> tuple:
    """
    Determine if retraining is needed based on multiple criteria.

    Returns (should_retrain: bool, reason: str)
    """
    reasons = []

    # 1. Check scheduled interval
    if config.get("last_retrained"):
        last_retrained = datetime.fromisoformat(config["last_retrained"])
        days_since = (datetime.now() - last_retrained).days
        if days_since >= SCHEDULED_INTERVAL_DAYS:
            reasons.append(f"Scheduled: {days_since} days since last retraining (threshold: {SCHEDULED_INTERVAL_DAYS})")

    # 2. Check performance degradation
    if new_rmse is not None and config.get("last_rmse"):
        degradation = (new_rmse - config["last_rmse"]) / config["last_rmse"]
        if degradation > PERFORMANCE_DEGRADATION:
            reasons.append(
                f"Performance degradation: RMSE increased by {degradation:.1%} "
                f"(current: {new_rmse:.2f}, previous: {config['last_rmse']:.2f})"
            )

    # 3. First time (never retrained with new data)
    if config.get("last_retrained") is None:
        reasons.append("Initial retraining with new data available.")

    should = len(reasons) > 0
    reason_str = " | ".join(reasons) if reasons else "No retraining needed."
    return should, reason_str


# ──────────────────────────────────────────────
# Model Retraining Pipeline
# ──────────────────────────────────────────────

def retrain_model(df: pd.DataFrame, features: list = None, target: str = None) -> dict:
    """
    Full retraining pipeline:

    1. Validate new data quality
    2. Build preprocessor pipeline
    3. Train XGBoost with time-series cross-validation
    4. Evaluate on held-out test set
    5. Compare with current model
    6. Save if improved
    7. Log to MLflow

    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset (new + historical) with all features.
    features : list, optional
        Feature column names. Uses defaults if None.
    target : str, optional
        Target column name. Uses 'sales' if None.

    Returns:
    --------
    dict with retraining results and decisions.
    """
    if features is None:
        features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    if target is None:
        target = TARGET

    config = load_retraining_config()

    logger.info("=" * 50)
    logger.info("Starting Model Retraining Pipeline")
    logger.info("=" * 50)

    # Step 1: Validate data
    logger.info("[Step 1] Validating data quality...")
    df = df.dropna(subset=[target])
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in data: {missing_features}")

    X = df[features]
    y = df[target]

    # Time-based split (80/20)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"  Training samples: {len(X_train):,}")
    logger.info(f"  Test samples: {len(X_test):,}")

    # Step 2: Build pipeline
    logger.info("[Step 2] Building preprocessing pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(**XGB_PARAMS)),
    ])

    # Step 3: Cross-validation
    logger.info("[Step 3] Running time-series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    cv_rmse = -cv_scores
    logger.info(f"  CV RMSE: {cv_rmse.mean():.2f} (+/- {cv_rmse.std():.2f})")

    # Step 4: Train and evaluate
    logger.info("[Step 4] Training final model...")
    pipeline.fit(X_train, y_train)
    y_pred = np.maximum(pipeline.predict(X_test), 0)

    new_mae = mean_absolute_error(y_test, y_pred)
    new_mse = mean_squared_error(y_test, y_pred)
    new_rmse = np.sqrt(new_mse)
    new_r2 = r2_score(y_test, y_pred)

    logger.info(f"  MAE:  {new_mae:.4f}")
    logger.info(f"  RMSE: {new_rmse:.4f}")
    logger.info(f"  R2:   {new_r2:.4f}")

    # Step 5: Compare with current model
    logger.info("[Step 5] Comparing with current model...")
    deployed = False
    comparison_reason = ""

    if config.get("last_rmse") is None:
        deployed = True
        comparison_reason = "No previous model version found. Deploying first retrained model."
    elif new_rmse < config["last_rmse"]:
        improvement = (config["last_rmse"] - new_rmse) / config["last_rmse"] * 100
        deployed = True
        comparison_reason = f"New model improved RMSE by {improvement:.2f}%"
    else:
        degradation = (new_rmse - config["last_rmse"]) / config["last_rmse"] * 100
        deployed = False
        comparison_reason = f"New model degraded RMSE by {degradation:.2f}%. Keeping current model."

    logger.info(f"  Decision: {comparison_reason}")

    # Step 6: Save model if improved
    if deployed:
        logger.info("[Step 6] Saving new model...")
        # Backup old model
        if os.path.exists(MODEL_PATH):
            backup_path = MODEL_PATH.replace(".pkl", f"_backup_v{config['retraining_count']}.pkl")
            joblib.dump(joblib.load(MODEL_PATH), backup_path)
            logger.info(f"  Old model backed up to: {backup_path}")

        # Save new model
        joblib.dump(pipeline, MODEL_PATH)
        logger.info(f"  New model saved to: {MODEL_PATH}")

        # Update config
        config["last_retrained"] = datetime.now().isoformat()
        config["last_rmse"] = new_rmse
        config["last_r2"] = new_r2
        config["retraining_count"] += 1
        config["model_versions"].append({
            "version": config["retraining_count"],
            "timestamp": datetime.now().isoformat(),
            "rmse": new_rmse,
            "r2": new_r2,
            "training_samples": len(X_train),
        })
    else:
        logger.info("[Step 6] Skipping model save (no improvement).")

    # Log to retraining history
    config["retraining_history"].append({
        "timestamp": datetime.now().isoformat(),
        "cv_rmse_mean": float(cv_rmse.mean()),
        "cv_rmse_std": float(cv_rmse.std()),
        "test_mae": float(new_mae),
        "test_rmse": float(new_rmse),
        "test_r2": float(new_r2),
        "deployed": deployed,
        "reason": comparison_reason,
    })

    save_retraining_config(config)

    # Step 7: Log to MLflow
    logger.info("[Step 7] Logging to MLflow...")
    try:
        mlflow.set_experiment("Sales_Forecasting_Retraining")
        with mlflow.start_run(run_name=f"retrain_v{config['retraining_count']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("retraining_version", config["retraining_count"])
            mlflow.log_param("training_samples", len(X_train))
            mlflow.log_param("deployed", deployed)
            mlflow.log_metric("cv_rmse_mean", cv_rmse.mean())
            mlflow.log_metric("cv_rmse_std", cv_rmse.std())
            mlflow.log_metric("test_mae", new_mae)
            mlflow.log_metric("test_rmse", new_rmse)
            mlflow.log_metric("test_r2", new_r2)

            if deployed:
                mlflow.xgboost.log_model(
                    pipeline.named_steps["model"],
                    artifact_path="xgboost_model",
                )
    except Exception as e:
        logger.warning(f"MLflow logging failed: {str(e)}")

    # Return results
    results = {
        "deployed": deployed,
        "reason": comparison_reason,
        "metrics": {
            "MAE": new_mae,
            "MSE": new_mse,
            "RMSE": new_rmse,
            "R2": new_r2,
        },
        "cv_rmse": {
            "mean": float(cv_rmse.mean()),
            "std": float(cv_rmse.std()),
        },
        "previous_rmse": config.get("last_rmse") if not deployed else config.get("last_rmse"),
        "retraining_count": config["retraining_count"],
        "timestamp": datetime.now().isoformat(),
    }

    logger.info("=" * 50)
    logger.info(f"Retraining Complete: {'DEPLOYED' if deployed else 'NOT DEPLOYED'}")
    logger.info("=" * 50)

    return results


# ──────────────────────────────────────────────
# Example Usage
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Model Retraining Strategy - Sales Forecasting")
    print("=" * 60)

    config = load_retraining_config()
    print(f"\n[1] Retraining History: {config['retraining_count']} retraining(s) performed.")

    print("\n[2] Retraining Triggers:")
    print(f"    - Scheduled interval:  every {SCHEDULED_INTERVAL_DAYS} days")
    print(f"    - Performance drop:     RMSE increase > {PERFORMANCE_DEGRADATION:.0%}")
    print(f"    - Minimum new data:     {MIN_NEW_DATA_DAYS} days")

    print("\n[3] Model Hyperparameters:")
    for k, v in XGB_PARAMS.items():
        print(f"    {k}: {v}")

    print("\n[4] Example usage in notebook:")
    print("-" * 40)
    print("""
from retrain import retrain_model, should_retrain, load_retraining_config

# Step 1: Check if retraining is needed
config = load_retraining_config()
needs_retrain, reason = should_retrain(config, new_rmse=550.0)
print(f"Retrain needed: {needs_retrain}")
print(f"Reason: {reason}")

# Step 2: Run retraining pipeline
results = retrain_model(
    df=full_dataset,           # Your combined historical + new data
    features=NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    target="sales",
)

# Step 3: Check results
print(f"Deployed: {results['deployed']}")
print(f"RMSE: {results['metrics']['RMSE']:.4f}")
print(f"R2: {results['metrics']['R2']:.4f}")
    """)