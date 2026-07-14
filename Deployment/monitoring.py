"""
Model Performance Monitoring Script
====================================
Tracks forecast accuracy over time, detects data drift,
and sets up alert mechanisms for model degradation.

This addresses Milestone 4 Task 3:
  - Set up model performance monitoring
  - Track forecast accuracy and detect model drift
  - Establish alert mechanisms
"""

import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats

import mlflow
import mlflow.xgboost

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgboost_sales_forecasting_model.pkl")
MONITORING_LOG_PATH = os.path.join(os.path.dirname(__file__), "monitoring_log.json")
ALERT_THRESHOLD_RMSE = 600.0  # Alert if RMSE exceeds this
ALERT_THRESHOLD_R2 = 0.65     # Alert if R2 drops below this
DRIFT_P_VALUE_THRESHOLD = 0.05  # KS-test threshold for data drift


# ──────────────────────────────────────────────
# Monitoring Logger
# ──────────────────────────────────────────────

def load_monitoring_log() -> list:
    """Load existing monitoring log or return empty list."""
    if os.path.exists(MONITORING_LOG_PATH):
        with open(MONITORING_LOG_PATH, "r") as f:
            return json.load(f)
    return []


def save_monitoring_log(log_data: list):
    """Save monitoring log to JSON file."""
    with open(MONITORING_LOG_PATH, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
    logger.info(f"Monitoring log saved ({len(log_data)} entries).")


def log_evaluation(timestamp: str, metrics: dict, drift_results: dict = None):
    """
    Log a new evaluation entry.

    Parameters:
    -----------
    timestamp : str
        ISO format timestamp of the evaluation.
    metrics : dict
        Dictionary with keys: MAE, MSE, RMSE, R2, sample_size.
    drift_results : dict, optional
        Dictionary with feature drift test results.
    """
    log_data = load_monitoring_log()

    entry = {
        "timestamp": timestamp,
        "metrics": metrics,
        "drift_detected": False,
        "alert_triggered": False,
        "drift_details": drift_results,
    }

    # Check for alerts
    if metrics.get("RMSE", 0) > ALERT_THRESHOLD_RMSE:
        entry["alert_triggered"] = True
        entry["alert_reason"] = f"RMSE ({metrics['RMSE']:.2f}) exceeds threshold ({ALERT_THRESHOLD_RMSE})"
        logger.warning(f"ALERT: {entry['alert_reason']}")

    if metrics.get("R2", 1) < ALERT_THRESHOLD_R2:
        entry["alert_triggered"] = True
        reason = f"R2 ({metrics['R2']:.4f}) below threshold ({ALERT_THRESHOLD_R2})"
        entry["alert_reason"] = entry.get("alert_reason", "") + " | " + reason
        logger.warning(f"ALERT: {reason}")

    # Check for data drift
    if drift_results:
        for feature, result in drift_results.items():
            if result.get("p_value", 1) < DRIFT_P_VALUE_THRESHOLD:
                entry["drift_detected"] = True
                logger.warning(
                    f"DRIFT: Feature '{feature}' shows significant distribution change "
                    f"(p-value={result['p_value']:.4f})"
                )
                break

    log_data.append(entry)
    save_monitoring_log(log_data)

    return entry


# ──────────────────────────────────────────────
# Performance Evaluation
# ──────────────────────────────────────────────

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate the model on test data and return metrics.

    Returns dict with: MAE, MSE, RMSE, R2, sample_size, timestamp
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": round(float(mae), 4),
        "MSE": round(float(mse), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
        "sample_size": len(y_test),
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(f"Model Evaluation: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    return metrics


# ──────────────────────────────────────────────
# Data Drift Detection
# ──────────────────────────────────────────────

def detect_data_drift(
    reference_data: pd.DataFrame,
    new_data: pd.DataFrame,
    numeric_features: list,
    method: str = "ks",
) -> dict:
    """
    Detect distribution shift between reference (training) and new (production) data.

    Uses Kolmogorov-Smirnov test for numeric features.

    Returns dict with feature-level drift results.
    """
    drift_results = {}

    for feature in numeric_features:
        if feature not in reference_data.columns or feature not in new_data.columns:
            continue

        ref_values = reference_data[feature].dropna().values
        new_values = new_data[feature].dropna().values

        if len(ref_values) < 10 or len(new_values) < 10:
            continue

        # Kolmogorov-Smirnov test
        if method == "ks":
            stat, p_value = stats.ks_2samp(ref_values, new_values)
            drift_results[feature] = {
                "ks_statistic": round(float(stat), 4),
                "p_value": round(float(p_value), 4),
                "drift_detected": p_value < DRIFT_P_VALUE_THRESHOLD,
                "ref_mean": round(float(ref_values.mean()), 4),
                "new_mean": round(float(new_values.mean()), 4),
                "ref_std": round(float(ref_values.std()), 4),
                "new_std": round(float(new_values.std()), 4),
            }

    return drift_results


# ──────────────────────────────────────────────
# Monitoring Dashboard Report
# ──────────────────────────────────────────────

def generate_monitoring_report() -> dict:
    """
    Generate a summary report from the monitoring log.
    """
    log_data = load_monitoring_log()

    if not log_data:
        return {"status": "No monitoring data available yet."}

    report = {
        "total_evaluations": len(log_data),
        "latest_evaluation": log_data[-1],
        "alerts_triggered": sum(1 for e in log_data if e.get("alert_triggered", False)),
        "drift_detected_count": sum(1 for e in log_data if e.get("drift_detected", False)),
        "performance_trend": {
            "rmse_history": [e["metrics"]["RMSE"] for e in log_data],
            "r2_history": [e["metrics"]["R2"] for e in log_data],
            "timestamps": [e["timestamp"] for e in log_data],
        },
        "retraining_recommended": False,
    }

    # Check if performance is degrading
    if len(log_data) >= 3:
        recent_rmse = [e["metrics"]["RMSE"] for e in log_data[-3:]]
        if all(recent_rmse[i] > recent_rmse[i - 1] for i in range(1, len(recent_rmse))):
            report["retraining_recommended"] = True
            report["degradation_reason"] = "RMSE has been increasing for 3 consecutive evaluations."

    if log_data[-1]["metrics"]["RMSE"] > ALERT_THRESHOLD_RMSE:
        report["retraining_recommended"] = True
        report["degradation_reason"] = report.get("degradation_reason", "") + \
            f" Latest RMSE ({log_data[-1]['metrics']['RMSE']:.2f}) exceeds threshold."

    return report


def log_to_mlflow(metrics: dict, model_path: str = MODEL_PATH):
    """
    Log monitoring metrics to MLflow for experiment tracking.
    """

    try:
        mlflow.set_experiment("Sales_Forecasting_Monitoring")

        with mlflow.start_run(
            run_name=f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ):

            # Log only numeric metrics
            for key, value in metrics.items():

                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value))

            # Log timestamp as a parameter
            if "timestamp" in metrics:
                mlflow.log_param("timestamp", metrics["timestamp"])

            mlflow.log_param("evaluation_type", "monitoring")

            logger.info("Monitoring metrics logged to MLflow.")

    except Exception as e:
        logger.error(f"Failed to log to MLflow: {str(e)}")


# ──────────────────────────────────────────────
# Example Usage (for notebook integration)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Model Monitoring System - Sales Forecasting")
    print("=" * 60)

    # Example: Load model and run a quick evaluation
    print("\n[1] Loading model...")
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Model not found at {MODEL_PATH}.")
        print("This script is designed to be imported and used within your notebook.")
        print("\nExample usage in notebook:")
        print("-" * 40)
        print("""
from monitoring import evaluate_model, detect_data_drift, log_evaluation, generate_monitoring_report, log_to_mlflow

# Step 1: Evaluate on new data
metrics = evaluate_model(model, X_new, y_new)

# Step 2: Detect data drift
drift_results = detect_data_drift(X_train, X_new, numeric_features)

# Step 3: Log results
entry = log_evaluation(
    timestamp=datetime.now().isoformat(),
    metrics=metrics,
    drift_results=drift_results,
)

# Step 4: Log to MLflow
log_to_mlflow(metrics)

# Step 5: Generate report
report = generate_monitoring_report()
print(json.dumps(report, indent=2))
        """)
        sys.exit(0)

    print("\n[2] Monitoring System Ready")
    print(f"    Alert RMSE threshold: {ALERT_THRESHOLD_RMSE}")
    print(f"    Alert R2 threshold:   {ALERT_THRESHOLD_R2}")
    print(f"    Drift p-value threshold: {DRIFT_P_VALUE_THRESHOLD}")

    print("\n[3] Available functions:")
    print("    - evaluate_model(model, X_test, y_test)")
    print("    - detect_data_drift(reference_data, new_data, numeric_features)")
    print("    - log_evaluation(timestamp, metrics, drift_results)")
    print("    - generate_monitoring_report()")
    print("    - log_to_mlflow(metrics)")

    report = generate_monitoring_report()
    print(f"\n[4] Current Status: {report.get('status', 'Active')}")