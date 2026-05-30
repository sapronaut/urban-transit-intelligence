import streamlit as st
import pandas as pd

st.title("Urban Transit Intelligence Dashboard")

df = pd.read_csv("data/cta_ridership.csv")

df["rides"] = df["rides"].astype(str).str.replace(",", "", regex=False)
df["rides"] = pd.to_numeric(df["rides"])

df["date"] = pd.to_datetime(df["date"])

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Ridership Over Time")

daily_ridership = (
    df.groupby("date")["rides"]
    .sum()
    .reset_index()
)

st.line_chart(
    daily_ridership.set_index("date")
)