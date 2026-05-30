from fastapi import FastAPI
import pandas as pd

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