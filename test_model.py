import joblib
import pandas as pd

model = joblib.load("model.pkl")

data = pd.DataFrame([{
    "station_id": 40350,
    "year": 2026,
    "month": 6,
    "day": 1,
    "weekday": 0,
    "daytype": 0
}])

prediction = model.predict(data)

print("Predicted Riders:", int(prediction[0]))