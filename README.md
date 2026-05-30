# 🚆 Urban Transit Intelligence System

An end-to-end Machine Learning and Data Engineering project that predicts public transit ridership using historical CTA station data, weather information, and intelligent forecasting models.

## 📌 Project Overview

This project analyzes Chicago Transit Authority (CTA) ridership patterns and builds a forecasting system capable of predicting station-level ridership based on temporal and operational features.

The system combines:

- Data Analysis & Visualization
- Machine Learning Forecasting
- Weather Data Integration
- Experiment Tracking with MLflow
- FastAPI Backend Services
- Streamlit Dashboard

---

## 🚀 Features

### Data Analytics
- Ridership trend analysis
- Station-level performance analysis
- Top stations identification
- Historical transit usage visualization

### Machine Learning
- Random Forest Regression model
- Ridership forecasting
- Model evaluation using:
  - MAE
  - RMSE
  - R² Score

### Weather Integration
- Real-time weather data using Open-Meteo API
- Temperature monitoring
- Humidity monitoring
- Wind speed monitoring

### Dashboard
- Interactive Streamlit dashboard
- Ridership visualizations
- Prediction interface
- Weather monitoring panel

### MLOps
- MLflow experiment tracking
- Model versioning
- Metric logging
- Training run comparison

---

## 🛠 Tech Stack

### Languages
- Python

### Libraries & Frameworks
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- FastAPI
- MLflow
- Joblib
- Requests
- Matplotlib

### APIs
- Open-Meteo Weather API

---

## 📊 Dataset

Dataset Used:
Chicago Transit Authority (CTA) Ridership Data

Features:
- Station ID
- Station Name
- Date
- Day Type
- Daily Ridership

---

## 📈 Model Performance

| Metric | Value |
|----------|----------|
| MAE | 1065.95 |
| RMSE | 1602.14 |
| R² Score | 0.7183 |

The model explains approximately 71.8% of ridership variance using temporal and station-based features.

---

## 📂 Project Structure

urban-transit-intelligence/

├── data/

├── src/

│ ├── api/

│ ├── dashboard/

│ └── models/

├── notebooks/

├── mlruns/

├── model.pkl

├── requirements.txt

└── README.md

---

## ▶️ Running Locally

### Create Virtual Environment

```bash
python -m venv venv
