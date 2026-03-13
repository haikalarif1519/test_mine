"""Tests for data_preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from data_preprocessing import (
    FEATURE_COLUMNS,
    RAW_COLUMNS,
    ENGINEERED_COLUMNS,
    _classify_metric,
    compute_class_weights,
    create_sequences,
    engineer_features,
    fit_scaler,
    generate_simulation_data,
    label_dataframe,
    label_row,
    scale_features,
)
import config


# ── _classify_metric ────────────────────────────────────────────────────────

class TestClassifyMetric:
    """Unit tests for the single-metric classifier."""

    def test_idle_normal(self):
        thresholds = {"idle": 5.0, "low_load": 50.0}
        assert _classify_metric(3.0, thresholds) == "idle"

    def test_low_load_normal(self):
        thresholds = {"idle": 5.0, "low_load": 50.0}
        assert _classify_metric(25.0, thresholds) == "low_load"

    def test_high_load_normal(self):
        thresholds = {"idle": 5.0, "low_load": 50.0}
        assert _classify_metric(80.0, thresholds) == "high_load"

    def test_boundary_idle(self):
        thresholds = {"idle": 5.0, "low_load": 50.0}
        assert _classify_metric(5.0, thresholds) == "idle"

    def test_boundary_low_load(self):
        thresholds = {"idle": 5.0, "low_load": 50.0}
        assert _classify_metric(50.0, thresholds) == "low_load"

    def test_inverted_idle(self):
        """system_idle_time: high value → idle."""
        thresholds = {"idle": 95.0, "low_load": 50.0}
        assert _classify_metric(98.0, thresholds, invert=True) == "idle"

    def test_inverted_low_load(self):
        thresholds = {"idle": 95.0, "low_load": 50.0}
        assert _classify_metric(70.0, thresholds, invert=True) == "low_load"

    def test_inverted_high_load(self):
        thresholds = {"idle": 95.0, "low_load": 50.0}
        assert _classify_metric(30.0, thresholds, invert=True) == "high_load"


# ── label_row ───────────────────────────────────────────────────────────────

class TestLabelRow:
    """The row-level labeller should pick the worst (highest-activity) state."""

    def test_all_idle(self):
        row = {
            "cpu_utilization": 2.0,
            "memory_utilization": 10.0,
            "disk_read": 500.0,
            "disk_write": 500.0,
            "network_received": 2_000.0,
            "network_sent": 2_000.0,
            "system_idle_time": 98.0,
        }
        assert label_row(row) == "idle"

    def test_one_metric_high(self):
        row = {
            "cpu_utilization": 90.0,  # high
            "memory_utilization": 10.0,
            "disk_read": 500.0,
            "disk_write": 500.0,
            "network_received": 2_000.0,
            "network_sent": 2_000.0,
            "system_idle_time": 98.0,
        }
        assert label_row(row) == "high_load"

    def test_missing_values(self):
        row = {
            "cpu_utilization": 2.0,
            "memory_utilization": float("nan"),
            "disk_read": None,
            "disk_write": 500.0,
            "network_received": 2_000.0,
            "network_sent": 2_000.0,
            "system_idle_time": 98.0,
        }
        assert label_row(row) == "idle"


# ── label_dataframe ─────────────────────────────────────────────────────────

class TestLabelDataframe:
    def test_adds_column(self):
        df = generate_simulation_data(n_idle=10, n_low=10, n_high=10)
        engineer_features(df)
        label_dataframe(df)
        assert "state_label" in df.columns
        assert set(df["state_label"].unique()) == {"idle", "low_load", "high_load"}


# ── engineer_features ───────────────────────────────────────────────────────

class TestEngineerFeatures:
    def test_adds_columns(self):
        df = generate_simulation_data(n_idle=10, n_low=10, n_high=10)
        engineer_features(df)
        for col in ENGINEERED_COLUMNS:
            assert col in df.columns

    def test_activity_score_range(self):
        df = generate_simulation_data(n_idle=20, n_low=20, n_high=20)
        engineer_features(df)
        # activity_score = cpu * 0.6 + mem * 0.4, both in [0, 100]
        assert df["activity_score"].min() >= 0.0
        assert df["activity_score"].max() <= 100.0

    def test_io_total_nonnegative(self):
        df = generate_simulation_data(n_idle=10, n_low=10, n_high=10)
        engineer_features(df)
        assert (df["io_total"] >= 0).all()

    def test_is_idle_binary(self):
        df = generate_simulation_data(n_idle=10, n_low=10, n_high=10)
        engineer_features(df)
        assert set(df["is_idle"].unique()).issubset({0.0, 1.0})

    def test_is_idle_threshold(self):
        """Idle samples have system_idle_time > 180s → is_idle should be 1."""
        df = generate_simulation_data(n_idle=50, n_low=0, n_high=0)
        engineer_features(df)
        # All idle samples have system_idle_time in [200, 600]
        assert (df["is_idle"] == 1.0).all()


# ── generate_simulation_data ────────────────────────────────────────────────

class TestGenerateSimulationData:
    def test_row_count(self):
        df = generate_simulation_data(n_idle=50, n_low=30, n_high=20)
        assert len(df) == 100

    def test_columns(self):
        df = generate_simulation_data(n_idle=10, n_low=10, n_high=10)
        for col in RAW_COLUMNS:
            assert col in df.columns

    def test_deterministic(self):
        df1 = generate_simulation_data(seed=0)
        df2 = generate_simulation_data(seed=0)
        pd.testing.assert_frame_equal(df1, df2)

    def test_imbalanced_defaults(self):
        """Default counts should reflect realistic class imbalance."""
        df = generate_simulation_data()
        assert len(df) == 375 + 1875 + 1875


# ── Scaler ──────────────────────────────────────────────────────────────────

class TestScaler:
    def test_fit_and_transform(self):
        df = generate_simulation_data(n_idle=50, n_low=50, n_high=50)
        engineer_features(df)
        scaler = fit_scaler(df)
        df_scaled = scale_features(df, scaler)
        # StandardScaler: mean ≈ 0, std ≈ 1 (not bounded to [0,1])
        for col in FEATURE_COLUMNS:
            assert abs(df_scaled[col].mean()) < 0.5  # roughly centred


# ── compute_class_weights ───────────────────────────────────────────────────

class TestComputeClassWeights:
    def test_balanced_data(self):
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        weights = compute_class_weights(y)
        np.testing.assert_allclose(weights, [1.0, 1.0, 1.0], atol=1e-5)

    def test_imbalanced_data(self):
        y = np.array([0] * 10 + [1] * 50 + [2] * 50)
        weights = compute_class_weights(y)
        # Class 0 (minority) should have the highest weight
        assert weights[0] > weights[1]
        assert weights[0] > weights[2]

    def test_output_shape(self):
        y = np.array([0, 1, 2, 0, 1])
        weights = compute_class_weights(y)
        assert weights.shape == (config.NUM_CLASSES,)


# ── create_sequences ────────────────────────────────────────────────────────

class TestCreateSequences:
    def test_shapes(self):
        df = generate_simulation_data(n_idle=50, n_low=50, n_high=50)
        engineer_features(df)
        label_dataframe(df)
        scaler = fit_scaler(df)
        df_scaled = scale_features(df, scaler)
        X, y = create_sequences(df_scaled, seq_length=5)
        assert X.shape == (len(df_scaled) - 5, 5, len(FEATURE_COLUMNS))
        assert y.shape == (len(df_scaled) - 5,)
        assert set(np.unique(y)).issubset({0, 1, 2})
        assert X.dtype == np.float32
        assert y.dtype == np.int64
