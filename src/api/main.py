import os
import joblib
import requests
import pandas as pd
from fastapi import FastAPI, HTTPException

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Urban Transit Intelligence API",
    description="Ridership stats, weather, and ML-based ridership predictions for CTA stations.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Load data and model once at startup (relative to this file's location)
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../../data/CTA_ridership.csv")
MODEL_PATH = os.path.join(BASE_DIR, "../../model.pkl")

df = pd.read_csv(DATA_PATH)
df["rides"] = pd.to_numeric(
    df["rides"].astype(str).str.replace(",", "", regex=False)
)

model = joblib.load(MODEL_PATH)

# ---------------------------------------------------------------------------
# Daytype encoding — must match src/models/train.py exactly
# U = Sunday/Holiday → 0, W = Weekday → 1, A = Saturday → 2
# ---------------------------------------------------------------------------

DAYTYPE_MAP = {"U": 0, "W": 1, "A": 2}
DAYTYPE_LABELS = {v: k for k, v in DAYTYPE_MAP.items()}

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    return {"project": "Urban Transit Intelligence System"}


@app.get("/stats")
def stats():
    """Summary statistics for the full CTA ridership dataset."""
    return {
        "total_rows": len(df),
        "average_rides": round(float(df["rides"].mean()), 2),
        "max_rides": int(df["rides"].max()),
        "min_rides": int(df["rides"].min()),
        "stations": int(df["station_id"].nunique()),
    }


@app.get("/weather")
def weather():
    """Current Chicago weather from Open-Meteo (no API key required)."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=41.8781"
        "&longitude=-87.6298"
        "&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Weather API error: {e}")

    current = response.json()["current"]
    return {
        "temperature_c": current["temperature_2m"],
        "humidity_pct": current["relative_humidity_2m"],
        "wind_speed_kmh": current["wind_speed_10m"],
    }


@app.get("/predict")
def predict(
    station_id: int,
    year: int,
    month: int,
    day: int,
    weekday: int,
    daytype: str = "W",
):
    """
    Predict daily ridership for a CTA station.

    - **station_id**: CTA station ID (e.g. 40350 for UIC-Halsted)
    - **year** / **month** / **day**: date components
    - **weekday**: day of week — 0 = Monday … 6 = Sunday
    - **daytype**: `W` = Weekday, `A` = Saturday, `U` = Sunday/Holiday
    """
    daytype_upper = daytype.upper()
    if daytype_upper not in DAYTYPE_MAP:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid daytype '{daytype}'. Must be one of: W, A, U",
        )

    features = pd.DataFrame([{
        "station_id": station_id,
        "year": year,
        "month": month,
        "day": day,
        "weekday": weekday,
        "daytype": DAYTYPE_MAP[daytype_upper],
    }])

    prediction = model.predict(features)[0]

    return {
        "station_id": station_id,
        "date": f"{year}-{month:02d}-{day:02d}",
        "daytype": daytype_upper,
        "predicted_rides": int(prediction),
    }
