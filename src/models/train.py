import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "../../data/CTA_ridership.csv")
MODEL_PATH = os.path.join(BASE_DIR, "../../model.pkl")

# ---------------------------------------------------------------------------
# Daytype encoding — must match src/api/main.py exactly
# U = Sunday/Holiday → 0, W = Weekday → 1, A = Saturday → 2
# ---------------------------------------------------------------------------

DAYTYPE_MAP = {"U": 0, "W": 1, "A": 2}
FEATURES    = ["station_id", "year", "month", "day", "weekday", "daytype"]

# ---------------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------------

print("Loading data...")
df = pd.read_csv(DATA_PATH)

df["rides"] = pd.to_numeric(
    df["rides"].astype(str).str.replace(",", "", regex=False)
)
df["date"]    = pd.to_datetime(df["date"])
df["year"]    = df["date"].dt.year
df["month"]   = df["date"].dt.month
df["day"]     = df["date"].dt.day
df["weekday"] = df["date"].dt.dayofweek
df["daytype"] = df["daytype"].map(DAYTYPE_MAP)

# ---------------------------------------------------------------------------
# CI mode — cap rows to avoid GitHub Actions memory limit.
# Set CI_SAMPLE_SIZE env var to enable (e.g. 200000).
# Full dataset is used when running locally.
# ---------------------------------------------------------------------------

sample_size = os.environ.get("CI_SAMPLE_SIZE")
if sample_size:
    n = int(sample_size)
    print(f"CI mode: sampling {n:,} rows from {len(df):,} total")
    df = df.sample(n=n, random_state=42)
else:
    print(f"Full dataset: {len(df):,} rows")

X = df[FEATURES]
y = df["rides"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")

# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Urban Transit Intelligence")

# ---------------------------------------------------------------------------
# Model 1 — Random Forest with RandomizedSearchCV
# ---------------------------------------------------------------------------

print("\n[1/2] Random Forest — RandomizedSearchCV (n_iter=8, cv=3)...")

RF_PARAM_GRID = {
    "n_estimators":     [100, 150, 200],
    "max_depth":        [20, 25, 30],
    "min_samples_leaf": [1, 2],
    "max_features":     ["sqrt", 0.5],
}

rf_search = RandomizedSearchCV(
    estimator           = RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions = RF_PARAM_GRID,
    n_iter              = 8,
    cv                  = 3,
    scoring             = "r2",
    random_state        = 42,
    n_jobs              = 1,
    verbose             = 1,
)
rf_search.fit(X_train, y_train)

# Log all RF candidates
rf_results = rf_search.cv_results_
for i in range(len(rf_results["params"])):
    with mlflow.start_run(run_name=f"rf_candidate_{i+1}"):
        mlflow.log_param("model", "RandomForest")
        mlflow.log_params(rf_results["params"][i])
        mlflow.log_metric("cv_r2_mean", rf_results["mean_test_score"][i])
        mlflow.log_metric("cv_r2_std",  rf_results["std_test_score"][i])

# Evaluate best RF
rf_best  = rf_search.best_estimator_
rf_preds = rf_best.predict(X_test)
rf_mae   = mean_absolute_error(y_test, rf_preds)
rf_rmse  = mean_squared_error(y_test, rf_preds) ** 0.5
rf_r2    = r2_score(y_test, rf_preds)

with mlflow.start_run(run_name="rf_best"):
    mlflow.log_param("model", "RandomForest")
    mlflow.log_params(rf_search.best_params_)
    mlflow.log_metric("test_mae",  rf_mae)
    mlflow.log_metric("test_rmse", rf_rmse)
    mlflow.log_metric("test_r2",   rf_r2)
    mlflow.sklearn.log_model(rf_best, artifact_path="model")

print(f"RF best  — R²={rf_r2:.4f}  MAE={rf_mae:.2f}  RMSE={rf_rmse:.2f}")
print(f"RF params: {rf_search.best_params_}")

# ---------------------------------------------------------------------------
# Model 2 — XGBoost with RandomizedSearchCV
# ---------------------------------------------------------------------------

print("\n[2/2] XGBoost — RandomizedSearchCV (n_iter=8, cv=3)...")

XGB_PARAM_GRID = {
    "n_estimators":  [200, 300, 400],
    "max_depth":     [6, 8, 10],
    "learning_rate": [0.05, 0.1, 0.15],
    "subsample":     [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
}

xgb_search = RandomizedSearchCV(
    estimator           = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
    param_distributions = XGB_PARAM_GRID,
    n_iter              = 8,
    cv                  = 3,
    scoring             = "r2",
    random_state        = 42,
    n_jobs              = 1,
    verbose             = 1,
)
xgb_search.fit(X_train, y_train)

# Log all XGB candidates
xgb_results = xgb_search.cv_results_
for i in range(len(xgb_results["params"])):
    with mlflow.start_run(run_name=f"xgb_candidate_{i+1}"):
        mlflow.log_param("model", "XGBoost")
        mlflow.log_params(xgb_results["params"][i])
        mlflow.log_metric("cv_r2_mean", xgb_results["mean_test_score"][i])
        mlflow.log_metric("cv_r2_std",  xgb_results["std_test_score"][i])

# Evaluate best XGB
xgb_best  = xgb_search.best_estimator_
xgb_preds = xgb_best.predict(X_test)
xgb_mae   = mean_absolute_error(y_test, xgb_preds)
xgb_rmse  = mean_squared_error(y_test, xgb_preds) ** 0.5
xgb_r2    = r2_score(y_test, xgb_preds)

with mlflow.start_run(run_name="xgb_best"):
    mlflow.log_param("model", "XGBoost")
    mlflow.log_params(xgb_search.best_params_)
    mlflow.log_metric("test_mae",  xgb_mae)
    mlflow.log_metric("test_rmse", xgb_rmse)
    mlflow.log_metric("test_r2",   xgb_r2)
    mlflow.sklearn.log_model(xgb_best, artifact_path="model")

print(f"XGB best — R²={xgb_r2:.4f}  MAE={xgb_mae:.2f}  RMSE={xgb_rmse:.2f}")
print(f"XGB params: {xgb_search.best_params_}")

# ---------------------------------------------------------------------------
# Select and save the best overall model
# ---------------------------------------------------------------------------

print("\n" + "="*55)
print("Model Comparison (test set):")
print(f"  Random Forest : R²={rf_r2:.4f}  MAE={rf_mae:.2f}  RMSE={rf_rmse:.2f}")
print(f"  XGBoost       : R²={xgb_r2:.4f}  MAE={xgb_mae:.2f}  RMSE={xgb_rmse:.2f}")

if xgb_r2 >= rf_r2:
    best_model      = xgb_best
    best_model_name = "XGBoost"
    best_r2, best_mae, best_rmse = xgb_r2, xgb_mae, xgb_rmse
    best_params = xgb_search.best_params_
else:
    best_model      = rf_best
    best_model_name = "RandomForest"
    best_r2, best_mae, best_rmse = rf_r2, rf_mae, rf_rmse
    best_params = rf_search.best_params_

print(f"\nWinner: {best_model_name}")
print(f"  R²   : {best_r2:.4f}")
print(f"  MAE  : {best_mae:.2f}")
print(f"  RMSE : {best_rmse:.2f}")
print(f"  Params: {best_params}")
print("="*55)

joblib.dump(best_model, MODEL_PATH)
print(f"\nSaved {best_model_name} to {MODEL_PATH}")
