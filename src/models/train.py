import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
# Hyperparameter search
# ---------------------------------------------------------------------------

PARAM_GRID = {
    "n_estimators":     [100, 150, 200],
    "max_depth":        [20, 25, 30],
    "min_samples_leaf": [1, 2],
    "max_features":     ["sqrt", 0.5],
}

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Urban Transit Intelligence")

print("\nRunning RandomizedSearchCV (n_iter=8, cv=3)...")

base_model = RandomForestRegressor(random_state=42, n_jobs=-1)

search = RandomizedSearchCV(
    estimator           = base_model,
    param_distributions = PARAM_GRID,
    n_iter              = 8,
    cv                  = 3,
    scoring             = "r2",
    random_state        = 42,
    n_jobs              = 1,   # outer loop sequential to control memory
    verbose             = 1,
)
search.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# Log every candidate run to MLflow
# ---------------------------------------------------------------------------

print("\nLogging all candidate runs to MLflow...")
results = search.cv_results_

for i in range(len(results["params"])):
    with mlflow.start_run(run_name=f"rf_candidate_{i+1}"):
        mlflow.log_params(results["params"][i])
        mlflow.log_metric("cv_r2_mean", results["mean_test_score"][i])
        mlflow.log_metric("cv_r2_std",  results["std_test_score"][i])

# ---------------------------------------------------------------------------
# Evaluate best model on held-out test set
# ---------------------------------------------------------------------------

best_model  = search.best_estimator_
predictions = best_model.predict(X_test)

mae  = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2   = r2_score(y_test, predictions)

# ---------------------------------------------------------------------------
# Log best model run to MLflow
# ---------------------------------------------------------------------------

with mlflow.start_run(run_name="rf_best"):
    mlflow.log_params(search.best_params_)
    mlflow.log_metric("test_mae",  mae)
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_r2",   r2)
    mlflow.sklearn.log_model(best_model, artifact_path="model")

# ---------------------------------------------------------------------------
# Save best model to disk
# ---------------------------------------------------------------------------

joblib.dump(best_model, MODEL_PATH)

print("\n" + "="*50)
print("Best hyperparameters:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")
print(f"\nTest set results:")
print(f"  MAE  : {mae:.2f}")
print(f"  RMSE : {rmse:.2f}")
print(f"  R2   : {r2:.4f}")
print("="*50)
