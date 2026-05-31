import streamlit as st
import pandas as pd
import joblib
import requests
import matplotlib.pyplot as plt

st.set_page_config(page_title="Urban Transit Intelligence", layout="wide")

st.title("🚆 Urban Transit Intelligence System")

import os

# Change this line:
df = pd.read_csv("data/cta_ridership.csv")

# To this:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "../../data/cta_ridership.csv"))

df["rides"] = df["rides"].astype(str).str.replace(",", "")
df["rides"] = pd.to_numeric(df["rides"])

df["date"] = pd.to_datetime(df["date"])

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Records", len(df))

with col2:
    st.metric("Stations", df["station_id"].nunique())

with col3:
    st.metric("Total Riders", f"{int(df['rides'].sum()):,}")

st.dataframe(df.head())

st.header("Top 10 Stations")

top_stations = (
    df.groupby("stationname")["rides"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(top_stations)

st.header("Ridership Trend")

daily = (
    df.groupby("date")["rides"]
    .sum()
    .reset_index()
)

st.line_chart(
    daily.set_index("date")
)

st.header("Weather")

try:
    weather = requests.get(
        "http://127.0.0.1:8000/weather"
    ).json()

    col1, col2, col3 = st.columns(3)

    col1.metric("Temperature", f"{weather['temperature']} °C")
    col2.metric("Humidity", f"{weather['humidity']} %")
    col3.metric("Wind Speed", f"{weather['wind_speed']} km/h")

except:
    st.warning("Weather API not running")

st.header("Ridership Prediction")

model = joblib.load("model.pkl")

station_id = st.number_input(
    "Station ID",
    value=40350
)

year = st.number_input(
    "Year",
    value=2026
)

month = st.number_input(
    "Month",
    value=6
)

day = st.number_input(
    "Day",
    value=1
)

weekday = st.number_input(
    "Weekday (0-6)",
    value=0
)

daytype = st.number_input(
    "Day Type",
    value=0
)

if st.button("Predict Riders"):

    pred_df = pd.DataFrame([{
        "station_id": station_id,
        "year": year,
        "month": month,
        "day": day,
        "weekday": weekday,
        "daytype": daytype
    }])

    prediction = model.predict(pred_df)[0]

    st.success(
        f"Predicted Riders: {int(prediction):,}"
    )