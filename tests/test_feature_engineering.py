"""
Tests for src/feature_engineering.py
"""
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    FEATURE_COLUMNS,
    label_idle,
    metrics_to_dataframe,
    metrics_to_feature_vector,
)


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


class TestMetricsToFeatureVector:
    def test_returns_ndarray_for_valid_metrics(self):
        result = metrics_to_feature_vector(_make_idle_metrics())
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(FEATURE_COLUMNS),)

    def test_returns_none_when_key_missing(self):
        metrics = _make_idle_metrics()
        del metrics["cpu_utilization"]
        assert metrics_to_feature_vector(metrics) is None

    def test_returns_none_when_value_is_none(self):
        metrics = _make_idle_metrics(cpu_utilization=None)
        assert metrics_to_feature_vector(metrics) is None

    def test_returns_none_when_value_is_non_numeric(self):
        metrics = _make_idle_metrics(cpu_utilization="N/A")
        assert metrics_to_feature_vector(metrics) is None

    def test_feature_order_matches_feature_columns(self):
        metrics = {col: float(i) for i, col in enumerate(FEATURE_COLUMNS)}
        result = metrics_to_feature_vector(metrics)
        expected = np.arange(len(FEATURE_COLUMNS), dtype=np.float64)
        np.testing.assert_array_equal(result, expected)


class TestMetricsToDataframe:
    def test_returns_dataframe_with_correct_columns(self):
        records = [_make_idle_metrics(), _make_active_metrics()]
        df = metrics_to_dataframe(records)
        assert list(df.columns) == FEATURE_COLUMNS

    def test_drops_rows_with_missing_values(self):
        records = [_make_idle_metrics(), _make_idle_metrics(cpu_utilization=None)]
        df = metrics_to_dataframe(records)
        assert len(df) == 1

    def test_returns_float64_dtype(self):
        records = [_make_idle_metrics()]
        df = metrics_to_dataframe(records)
        assert all(df.dtypes == np.float64)


class TestLabelIdle:
    def test_idle_sample_labelled_1(self):
        df = metrics_to_dataframe([_make_idle_metrics()])
        labels = label_idle(df)
        assert labels.iloc[0] == 1

    def test_active_sample_labelled_0(self):
        df = metrics_to_dataframe([_make_active_metrics()])
        labels = label_idle(df)
        assert labels.iloc[0] == 0

    def test_custom_thresholds_respected(self):
        # cpu_utilization = 8 is above the default 5 → active,
        # but below a custom threshold of 10 → idle (given all other metrics pass)
        metrics = _make_idle_metrics(cpu_utilization=8.0)
        df = metrics_to_dataframe([metrics])

        default_labels = label_idle(df)
        custom_labels = label_idle(df, cpu_idle_percent=10.0)

        assert default_labels.iloc[0] == 0
        assert custom_labels.iloc[0] == 1

    def test_mixed_samples(self):
        records = [_make_idle_metrics(), _make_active_metrics()]
        df = metrics_to_dataframe(records)
        labels = label_idle(df)
        assert labels.tolist() == [1, 0]
