import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/cta_ridership.csv")

df["rides"] = df["rides"].astype(str).str.replace(",", "", regex=False)
df["rides"] = pd.to_numeric(df["rides"])

df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["weekday"] = df["date"].dt.dayofweek

df["daytype"] = df["daytype"].astype("category").cat.codes

features = [
    "station_id",
    "year",
    "month",
    "day",
    "weekday",
    "daytype"
]

X = df[features]
y = df["rides"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=20,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Urban Transit Intelligence")

with mlflow.start_run(run_name="random_forest"):

    mlflow.log_param("model", "RandomForest")

    mlflow.log_param("n_estimators", 100)

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    # mlflow.sklearn.log_model(
    #     model,
    #     artifact_path="model"
    # )

joblib.dump(model, "model.pkl")

print("\nTraining Complete")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")