import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/cta_ridership.csv")
df = df.sample(200000, random_state=42)

print("=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)
print(df.head())

df["rides"] = df["rides"].astype(str).str.replace(",", "", regex=False)
df["rides"] = pd.to_numeric(df["rides"])

df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["weekday"] = df["date"].dt.dayofweek

print("\n" + "=" * 50)
print("DATASET INFO")
print("=" * 50)
print(df.info())

print("\n" + "=" * 50)
print("COLUMN NAMES")
print("=" * 50)
print(df.columns)

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

print("\n" + "=" * 50)
print("DATA TYPES")
print("=" * 50)
print(df.dtypes)

print("\n" + "=" * 50)
print("STATISTICAL SUMMARY")
print("=" * 50)
print(df.describe())

print("\n" + "=" * 50)
print("TOP 10 STATIONS BY TOTAL RIDERSHIP")
print("=" * 50)

top_stations = (
    df.groupby("stationname")["rides"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print(top_stations)

plt.figure(figsize=(10, 5))
plt.hist(df["rides"], bins=50)
plt.title("Distribution of Daily Ridership")
plt.xlabel("Number of Riders")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

monthly_ridership = (
    df.groupby("month")["rides"]
    .mean()
)

plt.figure(figsize=(10, 5))
monthly_ridership.plot(kind="line", marker="o")
plt.title("Average Monthly Ridership")
plt.xlabel("Month")
plt.ylabel("Average Riders")
plt.grid(True)
plt.tight_layout()
plt.show()

daytype_ridership = (
    df.groupby("daytype")["rides"]
    .mean()
)

plt.figure(figsize=(8, 5))
daytype_ridership.plot(kind="bar")
plt.title("Average Ridership by Day Type")
plt.xlabel("Day Type")
plt.ylabel("Average Riders")
plt.tight_layout()
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY")