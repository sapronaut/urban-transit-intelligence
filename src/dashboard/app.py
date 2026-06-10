import os
import streamlit as st
import pandas as pd
import joblib
import requests

st.set_page_config(page_title="Urban Transit Intelligence", layout="wide")
st.title("🌐 Urban Transit Intelligence System")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "../../data/CTA_ridership.csv")
MODEL_PATH = os.path.join(BASE_DIR, "../../model.pkl")

# ---------------------------------------------------------------------------
# Daytype encoding — must match src/api/main.py and src/models/train.py
# ---------------------------------------------------------------------------

DAYTYPE_MAP = {"W": 1, "A": 2, "U": 0}
DAYTYPE_OPTIONS = {
    "Weekday (W)": "W",
    "Saturday (A)": "A",
    "Sunday / Holiday (U)": "U",
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["rides"] = pd.to_numeric(
        df["rides"].astype(str).str.replace(",", "", regex=False)
    )
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# ---------------------------------------------------------------------------
# Dataset Overview
# ---------------------------------------------------------------------------

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Records", f"{len(df):,}")
col2.metric("Stations", df["station_id"].nunique())
col3.metric("Total Riders", f"{int(df['rides'].sum()):,}")

st.dataframe(df.head())

# ---------------------------------------------------------------------------
# Top 10 Stations
# ---------------------------------------------------------------------------

st.header("Top 10 Stations")

top_stations = (
    df.groupby("stationname")["rides"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(top_stations)

# ---------------------------------------------------------------------------
# Ridership Trend
# ---------------------------------------------------------------------------

st.header("Ridership Trend")

daily = df.groupby("date")["rides"].sum().reset_index()
st.line_chart(daily.set_index("date"))

# ---------------------------------------------------------------------------
# Weather — calls Open-Meteo directly (works on Streamlit Cloud)
# ---------------------------------------------------------------------------

st.header("Weather (Chicago)")

try:
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=41.8781"
        "&longitude=-87.6298"
        "&current=temperature_2m,relative_humidity_2m,wind_speed_10m",
        timeout=5,
    )
    response.raise_for_status()
    current = response.json()["current"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Temperature", f"{current['temperature_2m']} °C")
    col2.metric("Humidity",    f"{current['relative_humidity_2m']} %")
    col3.metric("Wind Speed",  f"{current['wind_speed_10m']} km/h")

except Exception as e:
    st.warning(f"Could not fetch weather data: {e}")

# ---------------------------------------------------------------------------
# Ridership Prediction
# ---------------------------------------------------------------------------

st.header("Ridership Prediction")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

col1, col2 = st.columns(2)

with col1:
    station_id = st.number_input("Station ID", value=40350, step=1)
    year       = st.number_input("Year",  value=2026, min_value=2001, max_value=2030, step=1)
    month      = st.number_input("Month", value=6,    min_value=1,    max_value=12,   step=1)

with col2:
    day        = st.number_input("Day",     value=1,  min_value=1, max_value=31, step=1)
    weekday    = st.number_input("Weekday (0=Mon … 6=Sun)", value=0, min_value=0, max_value=6, step=1)
    daytype_label = st.selectbox("Day Type", list(DAYTYPE_OPTIONS.keys()))

if st.button("Predict Riders", use_container_width=True):
    daytype_code = DAYTYPE_MAP[DAYTYPE_OPTIONS[daytype_label]]

    pred_df = pd.DataFrame([{
        "station_id": station_id,
        "year":       year,
        "month":      month,
        "day":        day,
        "weekday":    weekday,
        "daytype":    daytype_code,
    }])

    prediction = model.predict(pred_df)[0]
    st.success(f"Predicted Riders: {int(prediction):,}")
