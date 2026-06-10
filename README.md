# 🚆 Urban Transit Intelligence System

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue?logo=mlflow)
![License](https://img.shields.io/badge/License-MIT-yellow)

An end-to-end Machine Learning and Data Engineering project that predicts public transit ridership using historical CTA station data, real-time weather information, and intelligent forecasting models.

---

## 📌 Overview

This project analyzes **Chicago Transit Authority (CTA)** ridership patterns and builds a forecasting system capable of predicting station-level ridership based on temporal and operational features.

| Component | Technology |
|-----------|-----------|
| Dashboard | Streamlit |
| Backend API | FastAPI |
| ML Model | Scikit-Learn (Random Forest) |
| Experiment Tracking | MLflow |
| Weather Data | Open-Meteo API |

---

## ✨ Features

### 📊 Data Analytics
- Ridership trend analysis over time
- Station-level performance comparisons
- Top station identification and ranking
- Historical transit usage visualization

### 🤖 Machine Learning
- Random Forest Regression model
- Station-level ridership forecasting
- Model evaluation with MAE, RMSE, and R²

### 🌤 Weather Integration
- Real-time weather via Open-Meteo API
- Temperature, humidity, and wind speed monitoring

### 📈 MLOps
- MLflow experiment tracking and model versioning
- Metric logging and training run comparison

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| MAE | 184.55 |
| RMSE | 464.27 |
| R² Score | 0.9763 |

> The model explains approximately **97.6% of ridership variance** using temporal and station-based features.

---

## 🗂 Project Structure
```
urban-transit-intelligence/
│
├── data/                       # CTA ridership dataset
│
├── src/
│   ├── api/                    # FastAPI weather & prediction endpoints
│   ├── dashboard/              # Streamlit app
│   │   └── app.py
│   └── models/                 # Model training scripts
│
├── notebooks/                  # Exploratory data analysis
├── mlruns/                     # MLflow experiment logs
├── model.pkl                   # Trained Random Forest model
├── requirements.txt
└── README.md
```
---

## 🛠 Tech Stack

**Languages:** Python 3.10+

**Libraries:** Pandas · NumPy · Scikit-Learn · Streamlit · FastAPI · MLflow · Joblib · Matplotlib · Requests

**APIs:** Open-Meteo Weather API

---

## ▶️ Running Locally

### 1. Clone the Repository
```bash
git clone https://github.com/sapronaut/urban-transit-intelligence.git
cd urban-transit-intelligence
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python src/models/train.py
```

### 5. Start the FastAPI Backend
```bash
uvicorn src.api.main:app --reload
```

### 6. Launch the Streamlit Dashboard
```bash
streamlit run src/dashboard/app.py
```

---

## 📊 Dataset

**Source:** [Chicago Transit Authority (CTA) Ridership Data](https://data.cityofchicago.org/)

| Feature | Description |
|---------|-------------|
| `station_id` | Unique station identifier |
| `stationname` | Name of the CTA station |
| `date` | Date of ridership record |
| `daytype` | Type of day — `W` = Weekday, `A` = Saturday, `U` = Sunday/Holiday |
| `rides` | Daily ridership count |

---

## 🌐 Live Demo

[Streamlit App](https://urban-transit-intelligence-dmbk9qfj8vzt2qpemu4n8s.streamlit.app/)

---

## 📄 License

This project is licensed under the MIT License.
