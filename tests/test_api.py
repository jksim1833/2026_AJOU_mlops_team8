import numpy as np
from fastapi.testclient import TestClient

from src.api import main


class DummyModel:
    def predict_proba(self, row):
        return np.array([[0.4, 0.6]])


def test_predict_uses_metadata_threshold(monkeypatch):
    metadata = {
        "features": ["feature_a"],
        "target": {"0": "normal", "1": "defect"},
        "decision_threshold": 0.7,
        "model_version": "test-model",
        "data_version": "test-data",
    }
    importance = [{"feature": "feature_a", "importance": 1.0}]
    monkeypatch.setattr(
        main,
        "load_model_bundle",
        lambda: (DummyModel(), metadata, importance),
    )
    client = TestClient(main.app)

    response = client.post("/predict", json={"features": {"feature_a": 12.0}})

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] == 0
    assert payload["label_name"] == "normal"
    assert payload["probability"]["defect"] == 0.6


def test_predict_rejects_missing_features(monkeypatch):
    metadata = {
        "features": ["feature_a", "feature_b"],
        "target": {"0": "normal", "1": "defect"},
        "decision_threshold": 0.5,
        "model_version": "test-model",
        "data_version": "test-data",
    }
    monkeypatch.setattr(
        main,
        "load_model_bundle",
        lambda: (DummyModel(), metadata, []),
    )
    client = TestClient(main.app)

    response = client.post("/predict", json={"features": {"feature_a": 12.0}})

    assert response.status_code == 422
    assert response.json()["detail"]["missing_features"] == ["feature_b"]
