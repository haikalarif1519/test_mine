"""Tests for data preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from data_preprocessing import (
    create_feature_dataframe,
    generate_synthetic_training_data,
    label_idle_samples,
    normalize_features,
)


class TestCreateFeatureDataframe:
    def test_creates_dataframe_from_metrics_list(self):
        metrics = [
            {
                "cpu_utilization": 5.0,
                "memory_utilization": 30.0,
                "disk_read": 0.1,
                "disk_write": 0.2,
                "network_received": 100.0,
                "network_sent": 50.0,
                "system_idle_time": 95.0,
            }
        ]
        df = create_feature_dataframe(metrics)
        assert len(df) == 1
        assert list(df.columns) == [
            "cpu_utilization", "memory_utilization", "disk_read",
            "disk_write", "network_received", "network_sent",
            "system_idle_time",
        ]

    def test_fills_missing_columns_with_zero(self):
        metrics = [{"cpu_utilization": 10.0}]
        df = create_feature_dataframe(metrics)
        assert df["memory_utilization"].iloc[0] == 0.0
        assert df["disk_read"].iloc[0] == 0.0

    def test_handles_empty_list(self):
        df = create_feature_dataframe([])
        assert len(df) == 0


class TestLabelIdleSamples:
    def test_labels_idle_system(self):
        df = pd.DataFrame([{
            "cpu_utilization": 2.0,
            "memory_utilization": 20.0,
            "disk_read": 0.1,
            "disk_write": 0.1,
            "network_received": 100.0,
            "network_sent": 50.0,
            "system_idle_time": 98.0,
        }])
        labels = label_idle_samples(df)
        assert labels[0] == 1

    def test_labels_active_system(self):
        df = pd.DataFrame([{
            "cpu_utilization": 80.0,
            "memory_utilization": 70.0,
            "disk_read": 50.0,
            "disk_write": 30.0,
            "network_received": 50000.0,
            "network_sent": 30000.0,
            "system_idle_time": 20.0,
        }])
        labels = label_idle_samples(df)
        assert labels[0] == 0

    def test_high_cpu_makes_system_active(self):
        df = pd.DataFrame([{
            "cpu_utilization": 50.0,
            "memory_utilization": 20.0,
            "disk_read": 0.0,
            "disk_write": 0.0,
            "network_received": 0.0,
            "network_sent": 0.0,
            "system_idle_time": 50.0,
        }])
        labels = label_idle_samples(df)
        assert labels[0] == 0


class TestNormalizeFeatures:
    def test_normalizes_to_zero_one_range(self):
        df = pd.DataFrame({
            "cpu_utilization": [0.0, 50.0, 100.0],
            "memory_utilization": [10.0, 50.0, 90.0],
            "disk_read": [0.0, 5.0, 10.0],
            "disk_write": [0.0, 5.0, 10.0],
            "network_received": [0.0, 500.0, 1000.0],
            "network_sent": [0.0, 500.0, 1000.0],
            "system_idle_time": [0.0, 50.0, 100.0],
        })
        normalized, params = normalize_features(df)
        for col in df.columns:
            assert normalized[col].min() >= 0.0
            assert normalized[col].max() <= 1.0

    def test_constant_column_becomes_zero(self):
        df = pd.DataFrame({
            "cpu_utilization": [5.0, 5.0],
            "memory_utilization": [30.0, 30.0],
            "disk_read": [0.0, 0.0],
            "disk_write": [0.0, 0.0],
            "network_received": [0.0, 0.0],
            "network_sent": [0.0, 0.0],
            "system_idle_time": [95.0, 95.0],
        })
        normalized, _ = normalize_features(df)
        assert (normalized["cpu_utilization"] == 0.0).all()


class TestGenerateSyntheticData:
    def test_generates_correct_number_of_samples(self):
        X, y = generate_synthetic_training_data(n_samples=100)
        assert len(X) == 100
        assert len(y) == 100

    def test_generates_correct_idle_ratio(self):
        X, y = generate_synthetic_training_data(n_samples=1000, idle_ratio=0.6)
        idle_count = np.sum(y == 1)
        assert idle_count == 600

    def test_has_all_feature_columns(self):
        X, _ = generate_synthetic_training_data(n_samples=10)
        from config import FEATURE_COLUMNS
        for col in FEATURE_COLUMNS:
            assert col in X.columns

    def test_is_deterministic(self):
        X1, y1 = generate_synthetic_training_data(n_samples=50)
        X2, y2 = generate_synthetic_training_data(n_samples=50)
        pd.testing.assert_frame_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
