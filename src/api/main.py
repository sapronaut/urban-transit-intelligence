from fastapi import FastAPI
import pandas as pd
import requests

app = FastAPI()

df = pd.read_csv("data/cta_ridership.csv")

df["rides"] = df["rides"].astype(str).str.replace(",", "", regex=False)
df["rides"] = pd.to_numeric(df["rides"])

@app.get("/")
def home():
    return {"project": "Urban Transit Intelligence System"}

@app.get("/stats")
def stats():
    return {
        "total_rows": len(df),
        "average_rides": float(df["rides"].mean()),
        "max_rides": int(df["rides"].max()),
        "min_rides": int(df["rides"].min())
    }
@app.get("/weather")
def weather():

    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=41.8781"
        "&longitude=-87.6298"
        "&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )

    response = requests.get(url)

    data = response.json()

    current = data["current"]

    return {
        "temperature": current["temperature_2m"],
        "humidity": current["relative_humidity_2m"],
        "wind_speed": current["wind_speed_10m"]
    }