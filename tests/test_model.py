"""
Tests for src/model.py
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import FEATURE_COLUMNS, metrics_to_dataframe
from src.model import IdleDetectionModel


def _make_idle_metrics(**overrides):
    base = {
        "cpu_utilization": 1.0,
        "memory_utilization": 30.0,
        "disk_read_bps": 1024.0,
        "disk_write_bps": 1024.0,
        "network_recv_bps": 512.0,
        "network_sent_bps": 512.0,
        "system_idle_time": 99.0,
    }
    base.update(overrides)
    return base


def _make_active_metrics(**overrides):
    base = {
        "cpu_utilization": 80.0,
        "memory_utilization": 70.0,
        "disk_read_bps": 512_000.0,
        "disk_write_bps": 256_000.0,
        "network_recv_bps": 200_000.0,
        "network_sent_bps": 100_000.0,
        "system_idle_time": 20.0,
    }
    base.update(overrides)
    return base


def _build_training_data(n_idle=30, n_active=30):
    """Build a balanced set of clearly idle / active records."""
    records = [_make_idle_metrics() for _ in range(n_idle)] + [
        _make_active_metrics() for _ in range(n_active)
    ]
    return records


class TestIdleDetectionModelTraining:
    def test_train_returns_accuracy_metrics(self):
        model = IdleDetectionModel()
        metrics_list = _build_training_data()
        result = model.train_from_metrics_list(metrics_list)
        assert "train_accuracy" in result
        assert 0.0 <= result["train_accuracy"] <= 1.0

    def test_train_sets_is_trained_flag(self):
        model = IdleDetectionModel()
        assert not model._is_trained
        model.train_from_metrics_list(_build_training_data())
        assert model._is_trained

    def test_train_raises_on_empty_metrics_list(self):
        model = IdleDetectionModel()
        with pytest.raises(ValueError, match="No valid samples"):
            model.train_from_metrics_list([])

    def test_train_with_all_missing_metrics_raises(self):
        # All records have a missing field → dataframe will be empty
        records = [{"cpu_utilization": None} for _ in range(5)]
        model = IdleDetectionModel()
        with pytest.raises(ValueError):
            model.train_from_metrics_list(records)

    def test_train_includes_test_accuracy_when_enough_samples(self):
        model = IdleDetectionModel()
        result = model.train_from_metrics_list(_build_training_data(30, 30))
        assert "test_accuracy" in result

    def test_train_without_test_split_when_few_samples(self):
        model = IdleDetectionModel()
        # Only 5 samples – no test split
        records = _build_training_data(3, 2)
        result = model.train_from_metrics_list(records)
        assert "train_accuracy" in result
        assert "test_accuracy" not in result


class TestIdleDetectionModelPrediction:
    @pytest.fixture
    def trained_model(self):
        model = IdleDetectionModel()
        model.train_from_metrics_list(_build_training_data())
        return model

    def test_predict_idle_for_idle_metrics(self, trained_model):
        assert trained_model.predict(_make_idle_metrics()) is True

    def test_predict_active_for_active_metrics(self, trained_model):
        assert trained_model.predict(_make_active_metrics()) is False

    def test_predict_returns_false_when_metrics_incomplete(self, trained_model):
        metrics = _make_idle_metrics()
        del metrics["cpu_utilization"]
        assert trained_model.predict(metrics) is False

    def test_predict_falls_back_to_rules_when_untrained(self):
        model = IdleDetectionModel()
        # Untrained → rule-based fallback
        assert model.predict(_make_idle_metrics()) is True
        assert model.predict(_make_active_metrics()) is False


class TestIdleDetectionModelPersistence:
    def test_save_and_load_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name
        try:
            model = IdleDetectionModel(model_path=path)
            model.train_from_metrics_list(_build_training_data())
            model.save()
            assert os.path.exists(path)

            loaded = IdleDetectionModel(model_path=path)
            loaded.load()
            assert loaded._is_trained
            assert loaded.predict(_make_idle_metrics()) is True
        finally:
            os.unlink(path)

    def test_save_raises_when_not_trained(self):
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name
        os.unlink(path)
        try:
            model = IdleDetectionModel(model_path=path)
            with pytest.raises(RuntimeError, match="untrained"):
                model.save()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_load_raises_when_file_missing(self):
        model = IdleDetectionModel(model_path="/tmp/nonexistent_model.joblib")
        with pytest.raises(FileNotFoundError):
            model.load()


class TestRuleBasedFallback:
    def test_idle_metrics_return_true(self):
        assert IdleDetectionModel._rule_based_idle(_make_idle_metrics()) is True

    def test_active_metrics_return_false(self):
        assert IdleDetectionModel._rule_based_idle(_make_active_metrics()) is False

    def test_missing_keys_return_false(self):
        assert IdleDetectionModel._rule_based_idle({}) is False

    def test_none_values_return_false(self):
        metrics = {k: None for k in FEATURE_COLUMNS}
        assert IdleDetectionModel._rule_based_idle(metrics) is False
