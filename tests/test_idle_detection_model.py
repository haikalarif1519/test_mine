"""Tests for the idle detection model."""

import os
import tempfile

import numpy as np
import pytest

from data_preprocessing import generate_synthetic_training_data
from idle_detection_model import IdleDetectionModel


@pytest.fixture
def trained_model():
    """Create and train a model with synthetic data."""
    X, y = generate_synthetic_training_data(n_samples=500)
    model = IdleDetectionModel()
    model.train(X, y)
    return model


@pytest.fixture
def idle_metrics():
    """Metrics representing an idle system."""
    return {
        "cpu_utilization": 2.0,
        "memory_utilization": 20.0,
        "disk_read": 0.1,
        "disk_write": 0.1,
        "network_received": 50.0,
        "network_sent": 30.0,
        "system_idle_time": 98.0,
    }


@pytest.fixture
def active_metrics():
    """Metrics representing an active system."""
    return {
        "cpu_utilization": 75.0,
        "memory_utilization": 80.0,
        "disk_read": 50.0,
        "disk_write": 40.0,
        "network_received": 100000.0,
        "network_sent": 80000.0,
        "system_idle_time": 10.0,
    }


class TestIdleDetectionModel:
    def test_train_returns_results(self):
        X, y = generate_synthetic_training_data(n_samples=200)
        model = IdleDetectionModel()
        results = model.train(X, y)
        assert "accuracy" in results
        assert "report" in results
        assert results["accuracy"] > 0.8

    def test_predict_returns_binary(self, trained_model):
        X, _ = generate_synthetic_training_data(n_samples=10)
        predictions = trained_model.predict(X)
        assert all(p in (0, 1) for p in predictions)

    def test_predict_proba_returns_probabilities(self, trained_model):
        X, _ = generate_synthetic_training_data(n_samples=10)
        proba = trained_model.predict_proba(X)
        assert proba.shape == (10, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_detects_idle_system(self, trained_model, idle_metrics):
        is_idle, prob = trained_model.is_idle(idle_metrics)
        assert is_idle is True
        assert prob > 0.5

    def test_detects_active_system(self, trained_model, active_metrics):
        is_idle, prob = trained_model.is_idle(active_metrics)
        assert is_idle is False
        assert prob < 0.5

    def test_save_and_load(self, trained_model, idle_metrics):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.joblib")
            trained_model.save(model_path)
            assert os.path.exists(model_path)

            loaded_model = IdleDetectionModel()
            loaded_model.load(model_path)

            # Loaded model should produce same predictions
            is_idle_original, _ = trained_model.is_idle(idle_metrics)
            is_idle_loaded, _ = loaded_model.is_idle(idle_metrics)
            assert is_idle_original == is_idle_loaded

    def test_predict_without_model_raises(self):
        model = IdleDetectionModel()
        with pytest.raises(RuntimeError, match="Model is not loaded"):
            model.predict(np.array([[1, 2, 3, 4, 5, 6, 7]]))

    def test_feature_importance(self, trained_model):
        importance = trained_model.get_feature_importance()
        assert len(importance) == 7
        assert all(v >= 0 for v in importance.values())
