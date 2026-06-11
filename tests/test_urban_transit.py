"""
Tests for Urban Transit Intelligence.

Run with:
    pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DAYTYPE_MAP = {"U": 0, "W": 1, "A": 2}
FEATURES    = ["station_id", "year", "month", "day", "weekday", "daytype"]


# ===========================================================================
# 1. DAYTYPE ENCODING
# ===========================================================================

class TestDaytypeEncoding:

    def test_all_keys_present(self):
        """U, W, A must all be in the map."""
        assert set(DAYTYPE_MAP.keys()) == {"U", "W", "A"}

    def test_values_are_unique(self):
        """No two day types should share the same integer code."""
        codes = list(DAYTYPE_MAP.values())
        assert len(codes) == len(set(codes))

    def test_values_are_integers(self):
        for code in DAYTYPE_MAP.values():
            assert isinstance(code, int)

    def test_known_mappings(self):
        """Spot-check the exact codes used in training and prediction."""
        assert DAYTYPE_MAP["U"] == 0   # Sunday / Holiday
        assert DAYTYPE_MAP["W"] == 1   # Weekday
        assert DAYTYPE_MAP["A"] == 2   # Saturday

    def test_invalid_daytype_not_in_map(self):
        assert "X" not in DAYTYPE_MAP
        assert ""  not in DAYTYPE_MAP


# ===========================================================================
# 2. MODEL — INPUT / OUTPUT CONTRACT
# ===========================================================================

class TestModelContract:

    def test_prediction_returns_numeric(self, trained_model, sample_df):
        prediction = trained_model.predict(sample_df)[0]
        assert isinstance(prediction, (int, float, np.floating, np.integer))

    def test_prediction_is_positive(self, trained_model, sample_df):
        prediction = trained_model.predict(sample_df)[0]
        assert prediction > 0, "Predicted ridership should be positive"

    def test_prediction_is_plausible(self, trained_model, sample_df):
        """Ridership should be within a realistic CTA range (0 – 50,000)."""
        prediction = trained_model.predict(sample_df)[0]
        assert 0 < prediction < 50_000

    def test_batch_prediction_shape(self, trained_model, tiny_training_df):
        """Predicting on N rows should return N results."""
        X = tiny_training_df[FEATURES]
        preds = trained_model.predict(X)
        assert len(preds) == len(tiny_training_df)

    def test_model_uses_correct_features(self, trained_model):
        """Model should have been trained on exactly the expected feature set."""
        assert list(trained_model.feature_names_in_) == FEATURES

    def test_different_daytypes_produce_different_predictions(self, trained_model, sample_row):
        """Weekday vs Sunday should yield different ridership estimates."""
        rows = []
        for dt_code in DAYTYPE_MAP.values():
            row = {**sample_row, "daytype": dt_code}
            rows.append(row)
        preds = trained_model.predict(pd.DataFrame(rows))
        assert len(set(preds.tolist())) > 1, "Different day types should yield different predictions"

    def test_different_stations_produce_different_predictions(self, trained_model, sample_row):
        """Predictions for different stations should not all be identical."""
        stations = [40350, 41130, 40760]
        rows = [{**sample_row, "station_id": s} for s in stations]
        preds = trained_model.predict(pd.DataFrame(rows))
        assert len(set(preds.tolist())) > 1


# ===========================================================================
# 3. DATA PREPROCESSING
# ===========================================================================

class TestDataPreprocessing:

    def test_rides_column_parses_without_commas(self):
        """Rides values like '1,059' should parse to numeric 1059."""
        raw = pd.Series(["273", "1,059", "36,323"])
        cleaned = pd.to_numeric(raw.str.replace(",", "", regex=False))
        assert cleaned.tolist() == [273, 1059, 36323]

    def test_daytype_map_applied_correctly(self):
        """Applying DAYTYPE_MAP to a Series should produce integer codes."""
        s = pd.Series(["W", "A", "U", "W"])
        result = s.map(DAYTYPE_MAP)
        assert result.tolist() == [1, 2, 0, 1]

    def test_no_nulls_after_daytype_mapping(self):
        """Only valid daytype values (U, W, A) should be in the dataset."""
        s = pd.Series(["W", "A", "U", "W", "A"])
        result = s.map(DAYTYPE_MAP)
        assert result.isna().sum() == 0

    def test_date_feature_extraction(self):
        """Year, month, day, weekday should be extracted correctly."""
        dates = pd.to_datetime(pd.Series(["2024-06-01"]))
        assert dates.dt.year[0]     == 2024
        assert dates.dt.month[0]    == 6
        assert dates.dt.day[0]      == 1
        assert dates.dt.dayofweek[0] == 5  # Saturday

    def test_features_all_numeric_after_prep(self, tiny_training_df):
        """All feature columns must be numeric before model fitting."""
        X = tiny_training_df[FEATURES]
        for col in FEATURES:
            assert pd.api.types.is_numeric_dtype(X[col]), f"{col} is not numeric"


# ===========================================================================
# 4. API ROUTES
# ===========================================================================

class TestAPIRoutes:

    @pytest.fixture(autouse=True)
    def client(self, trained_model, tmp_path):
        """
        Spin up a TestClient with the model and data patched in,
        so tests never touch disk or make real HTTP calls.
        """
        import os, sys
        sys.path.insert(0, str(tmp_path))

        # Write a tiny CSV the API can load
        tiny_csv = tmp_path / "CTA_ridership.csv"
        pd.DataFrame([
            {"station_id": 40350, "stationname": "UIC-Halsted",
             "date": "01/01/2024", "daytype": "W", "rides": "1000"},
        ]).to_csv(tiny_csv, index=False)

        import joblib
        model_path = tmp_path / "model.pkl"
        joblib.dump(trained_model, model_path)

        with patch("src.api.main.DATA_PATH",  str(tiny_csv)), \
             patch("src.api.main.MODEL_PATH", str(model_path)):
            from src.api.main import app
            self.app_client = TestClient(app)

        yield

    def test_home_returns_200(self):
        response = self.app_client.get("/")
        assert response.status_code == 200

    def test_stats_returns_expected_keys(self):
        response = self.app_client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        for key in ["total_rows", "average_rides", "max_rides", "min_rides", "stations"]:
            assert key in data, f"Missing key: {key}"

    def test_predict_valid_request(self):
        response = self.app_client.get(
            "/predict",
            params={
                "station_id": 40350,
                "year": 2026, "month": 6, "day": 1,
                "weekday": 0, "daytype": "W",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "predicted_rides" in data
        assert data["predicted_rides"] > 0

    def test_predict_invalid_daytype(self):
        response = self.app_client.get(
            "/predict",
            params={
                "station_id": 40350,
                "year": 2026, "month": 6, "day": 1,
                "weekday": 0, "daytype": "X",
            },
        )
        assert response.status_code == 422

    def test_predict_response_echoes_station_id(self):
        response = self.app_client.get(
            "/predict",
            params={
                "station_id": 41130,
                "year": 2026, "month": 6, "day": 1,
                "weekday": 1, "daytype": "A",
            },
        )
        assert response.status_code == 200
        assert response.json()["station_id"] == 41130

    @patch("src.api.main.requests.get")
    def test_weather_returns_expected_keys(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "current": {
                    "temperature_2m": 22.5,
                    "relative_humidity_2m": 60,
                    "wind_speed_10m": 15.0,
                }
            },
        )
        mock_get.return_value.raise_for_status = lambda: None
        response = self.app_client.get("/weather")
        assert response.status_code == 200
        data = response.json()
        assert "temperature_c"  in data
        assert "humidity_pct"   in data
        assert "wind_speed_kmh" in data

    @patch("src.api.main.requests.get", side_effect=Exception("timeout"))
    def test_weather_upstream_failure_returns_502(self, mock_get):
        response = self.app_client.get("/weather")
        assert response.status_code == 502
