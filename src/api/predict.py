from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "Urban Transit Prediction API"}

@app.get("/predict")
def predict(
    station_id: int,
    year: int,
    month: int,
    day: int,
    weekday: int,
    daytype: int
):
    data = pd.DataFrame([{
        "station_id": station_id,
        "year": year,
        "month": month,
        "day": day,
        "weekday": weekday,
        "daytype": daytype
    }])

    prediction = model.predict(data)[0]

    return {
        "predicted_rides": int(prediction)
    }