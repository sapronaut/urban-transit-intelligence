# Urban Transit Intelligence System

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue?logo=mlflow)
![License](https://img.shields.io/badge/License-MIT-yellow)

An end-to-end machine learning and data engineering project that predicts public transit ridership using historical CTA station data, real-time weather, and competing ML models with full experiment tracking.

---

## Overview

This project analyzes Chicago Transit Authority (CTA) ridership patterns across 145+ stations from 2001 to present, and builds a forecasting system capable of predicting station-level daily ridership based on temporal and operational features.

| Component | Technology |
|-----------|------------|
| Dashboard | Streamlit |
| Backend API | FastAPI |
| ML Models | Random Forest · XGBoost |
| Hyperparameter Tuning | RandomizedSearchCV |
| Experiment Tracking | MLflow |
| Weather Data | Open-Meteo API |
| CI | GitHub Actions |

---

## Model Performance

Both models are tuned via RandomizedSearchCV (8 candidates, 3-fold CV) and evaluated on a held-out 20% test set. MLflow logs every candidate run from both searches, and the best overall model is automatically selected and saved.

| Model | R² | MAE | RMSE |
|-------|----|-----|------|
| Random Forest | 0.9471 | 310.24 | 691.18 |
| **XGBoost (winner)** | **0.9505** | **361.90** | **665.70** |

XGBoost is selected as the production model based on R² and RMSE.

**XGBoost best hyperparameters:**

| Parameter | Value |
|-----------|-------|
| n_estimators | 200 |
| max_depth | 8 |
| learning_rate | 0.1 |
| subsample | 0.9 |
| colsample_bytree | 0.9 |

---

## Features

**Data Analytics**
- Ridership trend analysis over time
- Station-level performance comparisons
- Top station identification and ranking
- Historical transit usage visualization

**Machine Learning**
- Random Forest and XGBoost regression with RandomizedSearchCV tuning
- Automatic model selection — best R² wins and is saved to production
- Model evaluation with MAE, RMSE, and R²

**Weather Integration**
- Real-time Chicago weather via Open-Meteo API (no API key required)
- Temperature, humidity, and wind speed displayed on dashboard

**MLOps**
- MLflow experiment tracking with per-candidate run logging for both models
- Model artifact versioning
- GitHub Actions CI — trains both models and runs full test suite on every push

---

## Project Structure

```
urban-transit-intelligence/
│
├── .github/workflows/       # GitHub Actions CI
├── data/                    # CTA ridership dataset (1M+ rows)
├── notebooks/               # Exploratory data analysis
├── src/
│   ├── api/
│   │   └── main.py          # FastAPI — /, /stats, /weather, /predict
│   ├── dashboard/
│   │   └── app.py           # Streamlit dashboard
│   └── models/
│       └── train.py         # RF + XGBoost training, search, auto-selection
├── tests/
│   ├── conftest.py          # Shared pytest fixtures
│   └── test_urban_transit.py # 24 tests across 4 test classes
├── requirements.txt
└── README.md
```

---

## Running Locally

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

### 4. Train the Models
```bash
python src/models/train.py
```

Runs RandomizedSearchCV for both Random Forest and XGBoost, logs all candidate runs to MLflow, compares test-set performance, and saves the winner to `model.pkl`.

### 5. Start the FastAPI Backend
```bash
uvicorn src.api.main:app --reload
```

API docs available at `http://127.0.0.1:8000/docs`

### 6. Launch the Streamlit Dashboard
```bash
streamlit run src/dashboard/app.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/stats` | Dataset summary statistics |
| GET | `/weather` | Live Chicago weather |
| GET | `/predict` | Ridership prediction |

**Example prediction request:**
```
GET /predict?station_id=40350&year=2026&month=6&day=15&weekday=0&daytype=W
```

---

## Dataset

**Source:** [Chicago Transit Authority (CTA) Ridership Data](https://data.cityofchicago.org/)

| Feature | Description |
|---------|-------------|
| `station_id` | Unique station identifier |
| `stationname` | Name of the CTA station |
| `date` | Date of ridership record |
| `daytype` | `W` = Weekday, `A` = Saturday, `U` = Sunday/Holiday |
| `rides` | Daily ridership count |

---

## Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

24 tests covering daytype encoding, model input/output contract, data preprocessing, and all 4 API endpoints including mocked weather failure handling.

---

## Tech Stack

**Languages:** Python 3.10+

**Libraries:** Pandas · NumPy · Scikit-Learn · XGBoost · Streamlit · FastAPI · MLflow · Joblib · Requests

**APIs:** Open-Meteo Weather API

---

## Live Demo

[Streamlit App](https://urban-transit-intelligence-dmbk9qfj8vzt2qpemu4n8s.streamlit.app/)

---

## License

This project is licensed under the MIT License.
