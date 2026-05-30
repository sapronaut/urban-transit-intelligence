import mlflow

mlflow.set_tracking_uri("file:./mlruns")

mlflow.set_experiment("Urban Transit Intelligence")

with mlflow.start_run(run_name="baseline_run"):
    mlflow.log_param("model", "baseline")
    mlflow.log_metric("mae", 120)
    mlflow.log_metric("rmse", 150)

print("Run logged successfully")