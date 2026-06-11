"""
Shared pytest fixtures for Urban Transit Intelligence tests.
"""

import os
import pytest
import pandas as pd
from unittest.mock import MagicMock
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------------------------------
# Constants (must match src/api/main.py and src/models/train.py)
# ---------------------------------------------------------------------------

DAYTYPE_MAP = {"U": 0, "W": 1, "A": 2}

FEATURES = ["station_id", "year", "month", "day", "weekday", "daytype"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_row():
    """A single valid prediction input as a dict."""
    return {
        "station_id": 40350,
        "year": 2026,
        "month": 6,
        "day": 1,
        "weekday": 0,
        "daytype": DAYTYPE_MAP["W"],
    }


@pytest.fixture(scope="session")
def sample_df(sample_row):
    """Single-row DataFrame ready for model.predict()."""
    return pd.DataFrame([sample_row])


@pytest.fixture(scope="session")
def tiny_training_df():
    """Minimal synthetic DataFrame that mirrors real CTA data shape."""
    rows = []
    for station in [40350, 41130, 40760]:
        for month in range(1, 4):
            for daytype_code in [0, 1, 2]:
                rows.append({
                    "station_id": station,
                    "year": 2024,
                    "month": month,
                    "day": 15,
                    "weekday": 2,
                    "daytype": daytype_code,
                    "rides": 1000 + station % 100 + month * 50 + daytype_code * 10,
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def trained_model(tiny_training_df):
    """A small RF model trained on synthetic data — fast, no disk I/O."""
    X = tiny_training_df[FEATURES]
    y = tiny_training_df["rides"]
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model
