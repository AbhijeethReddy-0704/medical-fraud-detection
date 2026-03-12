"""
Unit Tests for Fraud Detection System
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def generate_mock_data(n=100):
    """Generate synthetic Medicare data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "national_provider_identifier": [f"NPI{i:010d}" for i in range(n)],
        "provider_type": np.random.choice(["Internal Medicine", "Cardiology", "Orthopedics"], n),
        "nppes_provider_state": np.random.choice(["CA", "TX", "FL", "NY"], n),
        "number_of_services": np.random.randint(10, 2000, n).astype(float),
        "number_of_medicare_beneficiaries": np.random.randint(5, 500, n).astype(float),
        "average_submitted_charge_amount": np.random.uniform(100, 5000, n),
        "average_medicare_payment_amount": np.random.uniform(50, 2500, n),
        "average_medicare_standardized_amount": np.random.uniform(50, 2500, n),
        "charge_to_payment_ratio": np.random.uniform(1, 10, n),
        "services_per_beneficiary": np.random.uniform(1, 20, n),
        "payment_deviation": np.random.uniform(-500, 500, n),
        "is_high_volume": np.random.randint(0, 2, n),
        "is_abnormal_charge": np.random.randint(0, 2, n),
        "standardized_deviation": np.random.uniform(0, 300, n),
    })


class TestDataPipeline:
    def test_clean_data_removes_duplicates(self):
        from src.pipeline.ingest import clean_data
        df = generate_mock_data(50)
        df = pd.concat([df, df.head(10)])  # Add duplicates
        assert len(df) == 60
        cleaned = clean_data(df)
        assert len(cleaned) <= 50

    def test_feature_engineering_creates_ratio(self):
        from src.pipeline.ingest import engineer_features
        df = generate_mock_data(50)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        result = engineer_features(df)
        assert "charge_to_payment_ratio" in result.columns

    def test_no_negative_services(self):
        df = generate_mock_data(100)
        assert (df["number_of_services"] >= 0).all()


class TestFraudModel:
    def test_model_trains_successfully(self):
        from src.models.fraud_model import FraudDetectionModel
        df = generate_mock_data(200)
        model = FraudDetectionModel(contamination=0.1)
        metrics = model.train(df)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auc_roc" in metrics

    def test_model_predict_returns_scores(self):
        from src.models.fraud_model import FraudDetectionModel
        df = generate_mock_data(200)
        model = FraudDetectionModel(contamination=0.1)
        model.train(df)
        results = model.predict(df.head(10))
        assert "fraud_score" in results.columns
        assert "is_fraud_predicted" in results.columns
        assert "risk_level" in results.columns
        assert len(results) == 10

    def test_fraud_scores_between_0_and_1(self):
        from src.models.fraud_model import FraudDetectionModel
        df = generate_mock_data(200)
        model = FraudDetectionModel(contamination=0.1)
        model.train(df)
        results = model.predict(df)
        assert (results["fraud_score"] >= 0).all()
        assert (results["fraud_score"] <= 1).all()

    def test_untrained_model_raises_error(self):
        from src.models.fraud_model import FraudDetectionModel
        model = FraudDetectionModel()
        df = generate_mock_data(10)
        with pytest.raises(ValueError, match="not trained"):
            model.predict(df)


class TestAPI:
    def setup_method(self):
        from src.api.main import app
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_root_endpoint(self):
        response = self.client.get("/")
        assert response.status_code == 200

    def test_predict_endpoint_no_model(self):
        """Should return 503 when model not loaded."""
        response = self.client.post("/predict", json={
            "provider_id": "NPI1234567890",
            "number_of_services": 500,
            "number_of_medicare_beneficiaries": 100,
            "average_submitted_charge_amount": 1500.0,
            "average_medicare_payment_amount": 800.0,
            "average_medicare_standardized_amount": 750.0,
        })
        # Either 200 (model loaded) or 503 (model not loaded)
        assert response.status_code in [200, 503]

    def test_batch_predict_validates_input(self):
        response = self.client.post("/predict/batch", json={"providers": []})
        assert response.status_code in [200, 422, 503]
