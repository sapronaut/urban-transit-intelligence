import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../../data/CTA_ridership.csv")
MODEL_PATH = os.path.join(BASE_DIR, "../../model.pkl")

# ---------------------------------------------------------------------------
# Daytype encoding — must match src/api/main.py exactly
# U = Sunday/Holiday → 0, W = Weekday → 1, A = Saturday → 2
# ---------------------------------------------------------------------------

DAYTYPE_MAP = {"U": 0, "W": 1, "A": 2}

# ---------------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["rides"] = pd.to_numeric(
    df["rides"].astype(str).str.replace(",", "", regex=False)
)

df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["weekday"] = df["date"].dt.dayofweek

df["daytype"] = df["daytype"].map(DAYTYPE_MAP)

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

FEATURES = ["station_id", "year", "month", "day", "weekday", "daytype"]

X = df[FEATURES]
y = df["rides"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

N_ESTIMATORS = 100

model = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae  = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2   = r2_score(y_test, predictions)

# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Urban Transit Intelligence")

with mlflow.start_run(run_name="random_forest"):
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, artifact_path="model")

# ---------------------------------------------------------------------------
# Save model
# ---------------------------------------------------------------------------

joblib.dump(model, MODEL_PATH)

print("\nTraining Complete")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")
