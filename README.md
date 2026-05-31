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
| MAE | 1,065.95 |
| RMSE | 1,602.14 |
| R² Score | 0.7183 |

> The model explains approximately **71.8% of ridership variance** using temporal and station-based features.

---

## 🗂 Project Structure
